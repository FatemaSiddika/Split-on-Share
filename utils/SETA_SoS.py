import importlib
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union
from transformers import AutoModelForCausalLM 
from collections import defaultdict
import heapq
import os  
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D
import gc
import pandas as pd
from datetime import datetime
import json
import copy

def expert_task_ids(key_name):
    r"""Task ids in an expert key, as ints (imported by main.py).
    'exp_T123' -> [1, 2, 3]   (same as re.findall(r'\d', key) before)
    'exp_T-1-10' -> [1, 10]   (dash-separated, so task 10+ is one id)
    """
    s = str(key_name)
    if '-' in s:
        return [int(x) for x in re.findall(r'\d+', s)]
    return [int(d) for d in re.findall(r'\d', s)]

def _debug_print_split_summary(split_plan, new_expert_master_list, expert_definitions, current_task_id):
    """
    Additive debug function — prints a detailed block-level summary after SoS.
    Call after _perform_split_on_share returns.
    Shows: per-expert block counts, per-layer breakdown, totals for verification.
    """
    import re as _re
    SEP = "=" * 72
    print(f"\n{SEP}")
    print(f"  SoS SPLIT DEBUG SUMMARY  —  Task {current_task_id}")
    print(SEP)

    # ── 1. Expert inventory ───────────────────────────────────────────────────
    print(f"\n  EXPERT INVENTORY  (new_expert_master_list):")
    print(f"  {'Key':<18} {'Type':<10} {'task_id':>8} {'init_from':<14} {'Layers':>7} {'Total blocks':>13}")
    print(f"  {'-'*18} {'-'*10} {'-'*8} {'-'*14} {'-'*7} {'-'*13}")
    for key, defn in sorted(new_expert_master_list.items()):
        attn_layers  = defn.get('attn', {})
        total_blocks = sum(len(v) for v in attn_layers.values())
        n_layers     = len(attn_layers)
        init_from    = defn.get('init_from_key', '?')
        task_id      = defn.get('task_id', '?')
        digits       = _re.findall(r'\d', key)
        etype        = 'shared' if len(digits) > 1 else 'unique'
        print(f"  {key:<18} {etype:<10} {str(task_id):>8} {str(init_from):<14} {n_layers:>7} {total_blocks:>13}")

    # ── 2. Split plan — layer-level detail ────────────────────────────────────
    print(f"\n  SPLIT PLAN  ({len(split_plan)} layers affected):")
    if not split_plan:
        print("  (none — no layers had sufficient overlap)")
    else:
        print(f"  {'Layer':<30} {'Expert':<18} {'Role':<10} {'Blocks':>7}")
        print(f"  {'-'*30} {'-'*18} {'-'*10} {'-'*7}")
        for layer_key in sorted(split_plan.keys(), key=lambda x: (x[0], x[1])):
            for expert_key, einfo in split_plan[layer_key].get('new_experts', {}).items():
                n_blocks = len(einfo.get('positions', []))
                digits   = _re.findall(r'\d', expert_key)
                role     = 'shared' if len(digits) > 1 else 'unique'
                print(f"  {str(layer_key):<30} {expert_key:<18} {role:<10} {n_blocks:>7}")

    # ── 3. Grand totals ───────────────────────────────────────────────────────
    total_shared_blocks = 0
    total_unique_blocks = 0
    for key, defn in new_expert_master_list.items():
        digits  = _re.findall(r'\d', key)
        attn    = defn.get('attn', {})
        blocks  = sum(len(v) for v in attn.values())
        if len(digits) > 1:
            total_shared_blocks += blocks
        else:
            total_unique_blocks += blocks

    prior_blocks = 0
    for key, defn in expert_definitions.items():
        attn   = defn.get('attn', {})
        blocks = sum(len(v) for v in attn.values())
        prior_blocks += blocks

    print(f"\n  BLOCK COUNT TOTALS:")
    print(f"  {'Shared expert blocks (exp_T12, etc.)':<45} {total_shared_blocks:>6}")
    print(f"  {'New unique expert blocks (exp_T2, etc.)':<45} {total_unique_blocks:>6}")
    print(f"  {'Prior frozen expert blocks (exp_T1, etc.)':<45} {prior_blocks:>6}")
    print(f"  {'GRAND TOTAL (new shared + new unique)':<45} {total_shared_blocks + total_unique_blocks:>6}")
    print(SEP + "\n")


class SETAModel(torch.nn.Module):
    def __init__(self, config, model, SETA_config_name=None, 
                 task_id=1, 
                 save_path=None, 
                 previous_SETA_model=None,
                 expert_creation_threshold=5,    
                 pruning_min_blocks=5,      
                 tiny_remainder_threshold=5,
                 pretrained_moe_model=None, 
                 pretrained_definitions=None,
                 block_dimension=256,
                 no_expert_sharing=False):
        
        super().__init__()
        self.__dict__['model'] = None
        self.block_dim=block_dimension
        self.peft_config = config
        self.SETA_config_name = SETA_config_name
        self.current_task_id = task_id
        self.expert_creation_threshold = expert_creation_threshold
        self.pruning_min_blocks = pruning_min_blocks         
        self.tiny_remainder_threshold = tiny_remainder_threshold
        self.no_expert_sharing = no_expert_sharing
        self.expert_definitions = {} 
        self.expert_snapshots = {}
        self._budget_cfg = getattr(model, "config", None)   # [SETAFIX] task-1 T for the budget rule

        if pretrained_moe_model is not None:
            print(f"\n{'='*80}")
            print(f"TASK {self.current_task_id} (Phase 2): Initializing from Saved Split Structure")
            print(f"{'='*80}")
            self.model = pretrained_moe_model
            self.expert_definitions = pretrained_definitions
            if hasattr(self.model, "config"):
                self.model_dim = self.model.config.hidden_size
            else:
                self.model_dim = 4096 # Fallback if config is missing (Llama-7b default)

        # Task 1 setup
        elif self.current_task_id == 1:
            if model is None:
                raise ValueError("A base 'model' must be provided for Task 1")
            self.model_dim = model.config.hidden_size
            pos_T1_mlp, pos_T1_attn = self.load_task_positions(save_path, SETA_config_name)
            self.expert_definitions['exp_T1'] = {
                    'mlp': pos_T1_mlp, 'attn': pos_T1_attn,
                    'init_from_key': 'base', 'task_id': 1, 'key_name': 'exp_T1'
                }
            self.model = self.convert_linear_layer_to_sequential_moe(model, self.expert_definitions)

            if save_path:
                self.generate_expert_grid_csv(save_path)


        # Task 2+ setup
        else:
            if previous_SETA_model is None:
                raise ValueError("previous_SETA_model must be provided for Task 2+")
            
            print(f"TASK {self.current_task_id}: Expanding existing MoE model")
            
            #  Take the model reference
            self.expert_definitions = previous_SETA_model.expert_definitions
            self.model = previous_SETA_model.model 
            self.model_dim = previous_SETA_model.model_dim 
            
            #  Release the wrapper reference immediately to help GC
            del previous_SETA_model
            gc.collect()

            #  Load positions and plan split
            pos_T_new_mlp, pos_T_new_attn = self.load_task_positions(save_path, SETA_config_name)
            split_plan, new_expert_definitions, new_expert_master_list = self._perform_split_on_share(
                self.current_task_id, pos_T_new_mlp, pos_T_new_attn
            )

            _debug_print_split_summary(
                split_plan, new_expert_master_list,
                self.expert_definitions, self.current_task_id
            )
            print(f"Applying 'Split-on-Share' plan to {len(split_plan)} layers...")
            
            #  Iterate and apply splits with AGGRESSIVE GC
            count = 0
            for name, module in self.model.named_modules():
                if isinstance(module, SequentialMoELayer):
                    layer_key = self._get_layer_key_from_name(name)
                    if layer_key and layer_key in split_plan:
                        print(f"  - Splitting layer: {name}")
                        
                        module.apply_split_and_add(
                            split_plan=split_plan[layer_key],
                            new_expert_defns=new_expert_definitions.get(layer_key, {}),
                            base_weight=module.base_weight_ref
                        )
                        count += 1
                        if count % 1 == 0: 
                            torch.cuda.empty_cache()
            
            gc.collect()
            torch.cuda.empty_cache()
            
            self.expert_definitions.update(new_expert_master_list)

            if save_path:
                self.generate_expert_grid_csv(save_path)

        print("Snapshotting weights of trainable experts for regularization...")
        self.expert_snapshots = {}
        trainable_keys = self._get_trainable_expert_keys()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                for key in trainable_keys:
                    if f'.experts.{key}.' in name:
                        self.expert_snapshots[name] = param.data.clone().detach().cpu()
                        break
        
        self.print_trainable_parameters()
        self.forward = self.model.forward
        
    def save_moe_checkpoint(self, save_dir):
            """
            Saves the MoE model. 
            Filters out frozen base model weights to save storage.
            Only saves 'experts', 'gating_network', and trainable parameters.
            """
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            def_path = os.path.join(save_dir, "expert_definitions.json")
            
            def keys_to_str(d):
                return {f"{k[0]}|{k[1]}": v for k, v in d.items()}

            serializable_defs = {}
            for exp_key, data in self.expert_definitions.items():
                serializable_defs[exp_key] = {
                    'task_id': data['task_id'],
                    'key_name': data['key_name'],
                    'init_from_key': data.get('init_from_key', 'base'),
                    'mlp': keys_to_str(data['mlp']),
                    'attn': keys_to_str(data['attn'])
                }
            
            with open(def_path, 'w') as f:
                json.dump(serializable_defs, f, indent=2)
                
            weight_path = os.path.join(save_dir, "pytorch_model.bin")
            
            full_state = self.model.state_dict()
            
            moe_only_state = {
                k: v for k, v in full_state.items() 
                if "experts" in k or "gating_network" in k or "selected_weight" in k
            }
            
            torch.save(moe_only_state, weight_path)
            print(f"Saved optimized MoE checkpoint ({len(moe_only_state)} tensors) to {save_dir}")

    def _get_trainable_expert_keys(self):
        """
        Helper to get all expert keys that will be trained for the current task.
        This is simply all keys that have the current task_id.
        """
        trainable_keys = set()
        
        for key, defn in self.expert_definitions.items():
            if defn.get('task_id') == self.current_task_id:
                # 'key' is the expert key (e.g., '1', 'exp_2', 'exp_3')
                trainable_keys.add(key)
                
        return list(trainable_keys)

    def load_task_positions(self, save_path, SETA_config_name):
        positions_mlp = defaultdict(list)
        positions_attn = defaultdict(list)
        
        if self.peft_config.num_submatrix_attn > 0:
            json_path = os.path.join(save_path, f"selected_submatrix_attention_{SETA_config_name}.json")
            loaded_pos = self.load_submatrix_positions_from_json(json_path)
            
            # Route loaded blocks to correct modules
            for k, v in loaded_pos.items():
                if k[0] in ['gate_proj', 'up_proj', 'down_proj']:
                    positions_mlp[k] = v
                else:
                    positions_attn[k] = v
                    
        return positions_mlp, positions_attn
    
    def _budget_T(self, cfg):
        """[SETAFIX] budget rule: T = SETAFIX_PHI x N_base, N_base = BLOCK_BUDGET_FRACTION
        x all v_proj blocks (same N_base as the trainer's Step C). Returns None when
        the budget rule is not active."""
        if self.no_expert_sharing or os.environ.get("SETAFIX_SPLIT", "seam").lower() == "off" \
                or os.environ.get("SETAFIX_LAYER_GATE", "relative").lower() != "budget":
            return None
        _hd = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
        _v_out = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads) * _hd
        _tot = (_v_out // self.block_dim) * (cfg.hidden_size // self.block_dim) * cfg.num_hidden_layers
        _n_base = max(1, int(float(os.environ.get("BLOCK_BUDGET_FRACTION", "0.12")) * _tot))
        return float(os.environ.get("SETAFIX_PHI", "0.025")) * _n_base

    def _perform_split_on_share(self, new_task_id, new_pos_mlp, new_pos_attn):
        """
        Robust Split Logic: Checks expert_creation_threshold PER LAYER to filter noise.

        When self.no_expert_sharing=True all new task positions are routed directly
        into a fresh unique expert (exp_TN) with no overlap detection.  Prior experts
        are left untouched and frozen.  This is the "no sharing" ablation baseline.
        """
        print(f"  Performing 'Split-on-Share' for Task {new_task_id}...")
        if self.no_expert_sharing:
            print(f"   - [no_expert_sharing=True] Skipping overlap detection. "
                  f"Creating a fully isolated expert for Task {new_task_id}.")
        else:
            print(f"   - Min Overlap per Layer: {self.expert_creation_threshold}")
            print(f"   - Min Remainder Size: {self.tiny_remainder_threshold}")
        
        split_plan = defaultdict(lambda: {'splits': [], 'new_experts': {}}) 
        new_expert_master_list = {} 
        new_expert_definitions_by_layer = defaultdict(dict) 

        # [SETAFIX] direction-aware overlap validation (SETAFIX_SPLIT=off -> original)
        _setafix = None
        if not self.no_expert_sharing and os.environ.get("SETAFIX_SPLIT", "seam").lower() != "off":
            try:
                from utils.setafix_seam_split import SeamSplitValidator
                _setafix = SeamSplitValidator(self, new_task_id)
            except Exception as _e:
                print(f"   - [SETAFIX] validator unavailable ({_e}); using original split")

        # [SETAFIX] per-layer floor. expert_creation_threshold is an ABSOLUTE
        # block count that predates the direction test: it existed to stop
        # splits on positional overlap that might be noise. SEAM already does
        # that job per block, so re-applying an absolute count discards
        # validated blocks purely because they are spread thin across layers
        # (T3: 244 accepted blocks arrive at keep_split as 16, and 16 < 0.15 x
        # 322 kills the split). With the validator on, the floor is relative to
        # how much positional overlap that layer actually had.
        #   SETAFIX_LAYER_GATE  phi | relative | absolute | off
        #     phi      : straight phi, no rounding, no fixed minimum.
        #                m = phi x (new task's blocks in this layer). A layer is
        #                carved only if aligned >= m AND the old expert's leftover
        #                is 0 or >= m; otherwise nothing is carved there (old
        #                expert keeps all, new task keeps its own copy). The
        #                tiny-remainder merge is OFF in this mode — the leftover
        #                rule replaces it, so a shared child never receives
        #                blocks the new task did not select.
        #     relative : max(LAYER_MIN, ceil(LAYER_FRAC x positional overlap))
        #     absolute : expert_creation_threshold (legacy)
        #   SETAFIX_LAYER_FRAC  relative mode only (0.25)
        #   SETAFIX_LAYER_MIN   relative mode only (3)
        _layer_gate = os.environ.get("SETAFIX_LAYER_GATE", "relative").lower()
        _layer_frac = float(os.environ.get("SETAFIX_LAYER_FRAC", "0.25"))
        _layer_min = int(os.environ.get("SETAFIX_LAYER_MIN", "3"))
        _phi = float(os.environ.get("SETAFIX_PHI", "0.025"))
        _phi_mode = (_setafix is not None and _layer_gate == "phi")
        # [SETAFIX] budget: one rule, applied per layer after the direction test.
        #   T = PHI x N_base, N_base = BLOCK_BUDGET_FRACTION x total v_proj blocks
        #   (same N_base as the trainer's Step C). A layer is carved iff
        #   aligned >= T, the old expert's leftover there is 0 or >= T, and the
        #   new task's own blocks left there are 0 or >= T. Layers where the new
        #   task selected < T blocks are dropped at load time. So no expert piece
        #   of any kind is ever created below T in a layer.
        #   No tiny-remainder merge, no whole-expert keep_split check.
        _budget_mode = (_setafix is not None and _layer_gate == "budget")
        _T_budget = None
        if _budget_mode:
            _T_budget = self._budget_T(self.model.config)
            print(f"   - [SETAFIX] layer gate: budget  T = {_T_budget:.2f} blocks per layer "
                  f"(shared >= T, old leftover 0 or >= T, new task's own 0 or >= T)")
        _no_merge = _phi_mode or _budget_mode
        if _phi_mode:
            print(f"   - [SETAFIX] layer gate: phi={_phi} (m = phi x new task's blocks "
                  f"in layer; carve iff aligned >= m and leftover in {{0}} or >= m; "
                  f"tiny-remainder merge off)")
        elif _setafix is not None and _layer_gate != "absolute":
            print(f"   - [SETAFIX] layer gate: {_layer_gate} "
                  f"(frac={_layer_frac} min={_layer_min}) — replaces the "
                  f"absolute floor of {self.expert_creation_threshold}")

        def _phi_carve(n_aligned, n_leftover, n_new_in_layer):
            """phi mode: carve this layer iff both pieces clear m = phi x n_new."""
            m = _phi * n_new_in_layer
            return n_aligned >= m and (n_leftover == 0 or n_leftover >= m)

        def _layer_floor(n_candidates):
            """Minimum validated blocks a layer must keep to join a split.
            n_candidates is the POSITIONAL overlap, before the direction test."""
            if _setafix is None or _layer_gate == "absolute":
                return self.expert_creation_threshold
            if _layer_gate == "off":
                return 1
            return max(_layer_min, math.ceil(_layer_frac * n_candidates))
        
        new_sets_mlp = {key: set(val) for key, val in new_pos_mlp.items()}
        new_sets_attn = {key: set(val) for key, val in new_pos_attn.items()}
        
        experts_to_split = [] 
        remaining_new_pos_mlp = copy.deepcopy(new_pos_mlp)
        remaining_new_pos_attn = copy.deepcopy(new_pos_attn)
        
        def _get_new_expert_names(old_key_name, current_new_task_id):
            nums = re.findall(r'\d', str(old_key_name))
            all_ids = set(nums)
            all_ids.add(str(current_new_task_id))
            sorted_shared = sorted(list(all_ids), key=int)
            shared_name = f"exp_T{''.join(sorted_shared)}"
            sorted_unique = sorted(list(set(nums)), key=int)
            unique_name = f"exp_T{''.join(sorted_unique)}"
            return shared_name, unique_name

        # ── Overlap detection + split execution (skipped when no_expert_sharing=True) ─
        if not self.no_expert_sharing:
            for old_key, old_defn in self.expert_definitions.items():
                overlaps_mlp = defaultdict(list)
                overlaps_attn = defaultdict(list)
                unique_mlp = defaultdict(list)
                unique_attn = defaultdict(list)

                total_overlap_count = 0

                all_layer_keys = set(old_defn['mlp'].keys()) | set(old_defn['attn'].keys()) | \
                                 set(new_pos_mlp.keys()) | set(new_pos_attn.keys())

                sorted_layer_keys = sorted(list(all_layer_keys), key=lambda x: (x[1], x[0]))

                for layer_key in sorted_layer_keys:
                    # MLP LOGIC
                    old_set_mlp = set(old_defn['mlp'].get(layer_key, []))
                    new_set_mlp = new_sets_mlp.get(layer_key, set())
                    intersection_mlp = list(old_set_mlp & new_set_mlp)
                    remainder_mlp = list(old_set_mlp - new_set_mlp)

                    # [SETAFIX] blocks whose directions disagree: old expert keeps
                    # them (remainder); the new task keeps its own copy (not
                    # subtracted from remaining_new_pos -> lands in exp_T{new}).
                    if _setafix is not None and intersection_mlp:
                        intersection_mlp, _rej = _setafix.validate(old_key, old_defn, layer_key, intersection_mlp)
                        remainder_mlp = list(set(remainder_mlp) | set(_rej))

                    #  Layer-Wise Gatekeeper
                    if _phi_mode:
                        _carve = _phi_carve(len(intersection_mlp), len(remainder_mlp), len(new_set_mlp))
                    elif _budget_mode:
                        _own_mlp = len(set(remaining_new_pos_mlp.get(layer_key, [])) - set(intersection_mlp))
                        _carve = len(intersection_mlp) >= _T_budget and \
                                 (len(remainder_mlp) == 0 or len(remainder_mlp) >= _T_budget) and \
                                 (_own_mlp == 0 or _own_mlp >= _T_budget)
                    else:
                        _carve = len(intersection_mlp) >= _layer_floor(len(old_set_mlp & new_set_mlp))
                    if not _carve:
                        remainder_mlp = list(set(remainder_mlp) | set(intersection_mlp))
                        intersection_mlp = []

                    #  Cleanup Tiny Remainders (FIXED: Do NOT drop if no intersection)
                    #  [SETAFIX] skipped in phi mode: the leftover rule above replaces it
                    if not _no_merge and 0 < len(remainder_mlp) < self.tiny_remainder_threshold:
                        if len(intersection_mlp) > 0:
                            intersection_mlp = list(set(intersection_mlp) | set(remainder_mlp))
                            remainder_mlp = []
                        else:
                            pass

                    if intersection_mlp:
                        overlaps_mlp[layer_key] = intersection_mlp
                        total_overlap_count += len(intersection_mlp)
                        remaining_new_pos_mlp[layer_key] = list(set(remaining_new_pos_mlp.get(layer_key, [])) - set(intersection_mlp))

                    if remainder_mlp:
                        unique_mlp[layer_key] = remainder_mlp

                    old_set_attn = set(old_defn['attn'].get(layer_key, []))
                    new_set_attn = new_sets_attn.get(layer_key, set())
                    intersection_attn = list(old_set_attn & new_set_attn)
                    remainder_attn = list(old_set_attn - new_set_attn)

                    # [SETAFIX] same direction test for attention blocks
                    if _setafix is not None and intersection_attn:
                        intersection_attn, _rej = _setafix.validate(old_key, old_defn, layer_key, intersection_attn)
                        remainder_attn = list(set(remainder_attn) | set(_rej))

                    #  Layer-Wise Gatekeeper
                    if _phi_mode:
                        _carve = _phi_carve(len(intersection_attn), len(remainder_attn), len(new_set_attn))
                    elif _budget_mode:
                        _own_attn = len(set(remaining_new_pos_attn.get(layer_key, [])) - set(intersection_attn))
                        _carve = len(intersection_attn) >= _T_budget and \
                                 (len(remainder_attn) == 0 or len(remainder_attn) >= _T_budget) and \
                                 (_own_attn == 0 or _own_attn >= _T_budget)
                    else:
                        _carve = len(intersection_attn) >= _layer_floor(len(old_set_attn & new_set_attn))
                    if not _carve:
                        remainder_attn = list(set(remainder_attn) | set(intersection_attn))
                        intersection_attn = []

                    #  Cleanup Tiny Remainders
                    #  [SETAFIX] skipped in phi mode: the leftover rule above replaces it
                    if not _no_merge and 0 < len(remainder_attn) < self.tiny_remainder_threshold:
                        if len(intersection_attn) > 0:
                            intersection_attn = list(set(intersection_attn) | set(remainder_attn))
                            remainder_attn = []
                        else:
                            pass

                    if intersection_attn:
                        overlaps_attn[layer_key] = intersection_attn
                        total_overlap_count += len(intersection_attn)
                        remaining_new_pos_attn[layer_key] = list(set(remaining_new_pos_attn.get(layer_key, [])) - set(intersection_attn))

                    if remainder_attn:
                        unique_attn[layer_key] = remainder_attn

                # Final Decision Logic
                # [SETAFIX] relative-size rule: a tiny aligned overlap does not
                # justify a new shared expert -> give the overlap back (old expert
                # keeps it, new task keeps its own copy) and leave the expert intact.
                if total_overlap_count > 0 and _setafix is not None and not _budget_mode and \
                        not _setafix.keep_split(old_key, old_defn, total_overlap_count):
                    for _lk, _pos in list(overlaps_mlp.items()):
                        remaining_new_pos_mlp[_lk] = list(set(remaining_new_pos_mlp.get(_lk, [])) | set(_pos))
                    for _lk, _pos in list(overlaps_attn.items()):
                        remaining_new_pos_attn[_lk] = list(set(remaining_new_pos_attn.get(_lk, [])) | set(_pos))
                    total_overlap_count = 0
                if total_overlap_count > 0:
                    print(f"    - Expert '{old_key}': Found valid overlap ({total_overlap_count} blocks). Splitting.")
                    experts_to_split.append((old_key, old_defn, overlaps_mlp, overlaps_attn, unique_mlp, unique_attn))

            #  Execute Splits
            for old_key, old_defn, overlaps_mlp, overlaps_attn, unique_mlp, unique_attn in experts_to_split:

                key_shared, key_unique = _get_new_expert_names(old_key, new_task_id)

                #  Create SHARED Expert Definition
                shared_defn = {
                    'mlp': overlaps_mlp,
                    'attn': overlaps_attn,
                    'task_id': new_task_id,
                    'init_from_key': old_key,
                    'key_name': key_shared
                }
                new_expert_master_list[key_shared] = shared_defn

                # Create UNIQUE Expert Definition
                if any(unique_mlp.values()) or any(unique_attn.values()):
                    unique_defn = {
                        'mlp': unique_mlp,
                        'attn': unique_attn,
                        'task_id': old_defn['task_id'],
                        'init_from_key': old_key,
                        'key_name': key_unique
                    }
                    new_expert_master_list[key_unique] = unique_defn
                else:
                    key_unique = None

                # Update Layer Plans
                all_layer_keys = set(overlaps_mlp.keys()) | set(overlaps_attn.keys()) | \
                                 set(unique_mlp.keys()) | set(unique_attn.keys())

                for layer_key in all_layer_keys:
                    split_info = {
                        'old_key_to_delete': old_key,
                        'new_shared': key_shared,
                        'new_unique': key_unique,
                        'shared_pos': overlaps_mlp.get(layer_key, []) + overlaps_attn.get(layer_key, []),
                        'unique_pos': unique_mlp.get(layer_key, []) + unique_attn.get(layer_key, []) if key_unique else []
                    }
                    split_plan[layer_key]['splits'].append(split_info)

                    if split_info['shared_pos']:
                        split_plan[layer_key]['new_experts'][key_shared] = {'positions': split_info['shared_pos'], 'init_from_key': old_key}

                    if key_unique and split_info['unique_pos']:
                        split_plan[layer_key]['new_experts'][key_unique] = {'positions': split_info['unique_pos'], 'init_from_key': old_key}

                del self.expert_definitions[old_key]

        # ── When no_expert_sharing=True: all prior experts stay frozen as-is.
        # remaining_new_pos_* == all of the new task's positions (nothing was absorbed).

        # Create Final "New Unique" Expert
        key_new_task_unique = f"exp_T{new_task_id}"
        
        new_task_unique_defn = {
            'mlp': remaining_new_pos_mlp, 
            'attn': remaining_new_pos_attn, 
            'task_id': new_task_id, 
            'init_from_key': 'base', 
            'key_name': key_new_task_unique
        }
        new_expert_master_list[key_new_task_unique] = new_task_unique_defn

        all_layer_keys = set(remaining_new_pos_mlp.keys()) | set(remaining_new_pos_attn.keys())
        for layer_key in all_layer_keys:
            pos = remaining_new_pos_mlp.get(layer_key, []) + remaining_new_pos_attn.get(layer_key, [])
            if pos:
                new_expert_definitions_by_layer[layer_key][key_new_task_unique] = {
                    'positions': pos,
                    'init_from_key': 'base'
                }
            
        # [SETAFIX] per-block decisions + resulting expert sizes for inspection
        if _setafix is not None:
            try:
                _after = dict(self.expert_definitions); _after.update(new_expert_master_list)
                _setafix.write_report(_after)
            except Exception as _e:
                print(f"   - [SETAFIX] report skipped ({_e})")

        del experts_to_split
        del remaining_new_pos_mlp
        del remaining_new_pos_attn
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        return split_plan, new_expert_definitions_by_layer, new_expert_master_list
    
    def load_submatrix_positions_from_json_all(self, json_path):
        """Load submatrix selections (positions) from JSON file"""
        print(f"    Loading positions from: {json_path}")
        if not os.path.exists(json_path):
            print(f"     WARNING: File not found: {json_path}. Returning empty.")
            return defaultdict(list)
        try:
            with open(json_path, "r") as f:
                json_data = json.load(f)
        except json.JSONDecodeError:
            print(f"     ERROR: Could not decode JSON from {json_path}. Returning empty.")
            return defaultdict(list)

        loaded_positions = defaultdict(list)
        count = 0
        for key_str, value in json_data.items():
            parts = key_str.rsplit("_layer_", 1)
            if len(parts) != 2: continue
            module_name = parts[0]
            try:
                layer_number = int(parts[1])
            except ValueError: continue

            if isinstance(value, list) and all(isinstance(item, (list, tuple)) and len(item) == 2 for item in value):
                 positions = [tuple(item) for item in value]
                 loaded_positions[(module_name, layer_number)] = positions
                 count += len(positions)
            else:
                 print(f"     WARNING: Invalid format for key {key_str} in {json_path}. Skipping.")

        print(f"     Loaded {len(loaded_positions)} layers, {count} position tuples.")
        return loaded_positions

    def load_submatrix_positions_from_json(self, json_path):
        """
        Load submatrix selections (positions) from JSON file.
        Prunes layers with fewer than 'pruning_min_blocks'.
        """
        print(f"    Loading positions from: {json_path}")
        if not os.path.exists(json_path):
            print(f"     WARNING: File not found: {json_path}. Returning empty.")
            return defaultdict(list)
        try:
            with open(json_path, "r") as f:
                json_data = json.load(f)
        except json.JSONDecodeError:
            print(f"     ERROR: Could not decode JSON from {json_path}. Returning empty.")
            return defaultdict(list)

        loaded_positions = defaultdict(list)
        count = 0
        pruned_layers_count = 0
        # [SETAFIX] budget rule: a layer piece below T is never created
        _min_blocks = self.pruning_min_blocks
        _cfg = getattr(self.model, "config", None) if self.model is not None else self._budget_cfg
        _T = self._budget_T(_cfg) if _cfg is not None else None
        if _T is not None:
            _min_blocks = max(_min_blocks, math.ceil(_T))
            print(f"     [SETAFIX] budget rule: layers with < {_min_blocks} selected blocks are not used (T = {_T:.2f})")
        
        for key_str, value in json_data.items():
            parts = key_str.rsplit("_layer_", 1)
            if len(parts) != 2: continue
            module_name = parts[0]
            try:
                layer_number = int(parts[1])
            except ValueError: continue

            if isinstance(value, list) and all(isinstance(item, (list, tuple)) and len(item) == 2 for item in value):
                 positions = [tuple(item) for item in value]
                 
                 # Use self.pruning_min_block
                 if len(positions) < _min_blocks:
                     pruned_layers_count += 1
                     continue

                 loaded_positions[(module_name, layer_number)] = positions
                 count += len(positions)
            else:
                 print(f"     WARNING: Invalid format for key {key_str} in {json_path}. Skipping.")

        print(f"     Loaded {len(loaded_positions)} layers, {count} position tuples.")
        if pruned_layers_count > 0:
            print(f"     [Pruning] Discarded {pruned_layers_count} layers with < {_min_blocks} blocks.")
            
        return loaded_positions
    
    def convert_linear_layer_to_sequential_moe(self, base_model, expert_definitions):
        """
        Converts nn.Linear layers to SequentialMoELayer.
        Reconstructs the EXACT expert structure (shared & specific) from definitions.
        """
        print("Converting model to SequentialMoELayer...")
        model_copy = copy.deepcopy(base_model)
        converted_count = 0

        # Invert definitions to group by layer: layer_key -> {expert_key: definition}
        layer_expert_map = defaultdict(dict)
        for expert_key, defn in expert_definitions.items():
            # Collect positions from both MLP and Attn
            all_layer_keys = set(defn['mlp'].keys()) | set(defn['attn'].keys())
            
            for layer_key in all_layer_keys:
                module_name, layer_number = layer_key
                positions = []
                if module_name in ['gate_proj', 'up_proj', 'down_proj']:
                    positions = defn['mlp'].get(layer_key, [])
                elif module_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    positions = defn['attn'].get(layer_key, [])
                
                if positions:
                    # This preserves the specific/shared identity
                    layer_expert_map[layer_key][expert_key] = {
                        'positions': positions,
                        'init_from_key': defn.get('init_from_key', 'base')
                    }

        #  Apply to model
        for name, module in model_copy.named_modules():
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent_module = model_copy.get_submodule(parent_name) if parent_name else model_copy

            if isinstance(module, nn.Linear):
                layer_key = self._get_layer_key_from_name(name)
                
                if layer_key and layer_key in layer_expert_map:
                    layer_expert_defns = layer_expert_map[layer_key]
                    
                    print(f"   CREATING MoE Layer for: {name} (Experts: {list(layer_expert_defns.keys())})")
                    
                    # Create the layer with ALL previous experts (Shared + Specific)
                    new_moe_layer = SequentialMoELayer(
                        base_weight=module.weight.data.clone(),
                        layer_expert_definitions=layer_expert_defns,
                        model_dim=self.model_dim,
                        block_dim=self.block_dim,    # [MOD] per-arch block dimension
                    )
                    new_moe_layer = new_moe_layer.to(module.weight.device).to(module.weight.dtype)
                    setattr(parent_module, child_name, new_moe_layer)
                    converted_count += 1
                        
        print(f"\nConverted {converted_count} nn.Linear layers to SequentialMoELayer.")

        if converted_count == 0:
            raise RuntimeError(
                "0 nn.Linear layers converted to SequentialMoELayer — no experts exist, "
                "so training would run with 0 trainable params and eval would measure the "
                "base model. Check N_base and PRUNING_MIN_BLOCKS."
            )

        return model_copy
    
    def freeze_experts_for_training(self, task_id):
        print(f"Freezing experts for training Task {task_id}...")
        self.model.requires_grad_(False) # Freeze Base

        trainable_keys = []
        for key, defn in self.expert_definitions.items():
            if defn['task_id'] == task_id:
                trainable_keys.append(key)

        for name, module in self.model.named_modules():
            if isinstance(module, SequentialMoELayer):
                module.gating_network.requires_grad_(True)
                
                for key, expert in module.experts.items():
                    if key in trainable_keys:
                        expert.requires_grad_(True)
                    else:
                        expert.requires_grad_(False)

    def print_trainable_parameters(self):
        """Prints the number of trainable parameters in the model."""
        trainable_params = 0
        all_param = 0
        for param in self.model.parameters():
            num_params = param.numel()
            if num_params == 0 and hasattr(param, "ds_numel"): num_params = param.ds_numel # DeepSpeed Zero 3
            all_param += num_params
            if param.requires_grad: trainable_params += num_params
        print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.6f}")

    def _get_layer_key_from_name(self, name):
        """ Extracts (module_name, layer_number) from a layer name. """
        pattern = re.compile(r'layers\.(\d+)\.(mlp|self_attn)\.(\w+)')
        match = pattern.search(name)
        if match:
            layer_number = int(match.group(1))
            module_part = match.group(3) #  v_proj, gate_proj
            return (module_part, layer_number)
        return None
    
    # def __getattr__(self, name: str):
    #     """Forward missing attributes to the wrapped module."""
    #     try:
    #         return super().__getattr__(name)  # defer to nn.Module's logic
    #     except AttributeError:
    #         return getattr(self.model, name)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        # Always try nn.Module first — it checks _parameters, _buffers, _modules
        # This correctly resolves self.model which is stored in _modules
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        # Plain python attributes that don't exist yet — raise cleanly
        if name in ('block_dim', 'expert_definitions', 'expert_snapshots',
                    'current_task_id', 'peft_config', 'SETA_config_name',
                    'expert_creation_threshold', 'pruning_min_blocks',
                    'tiny_remainder_threshold', 'model_dim'):
            raise AttributeError(f"'SETAModel' object has no attribute '{name}'")
        # Forward to wrapped model for anything else
        model = self._modules.get('model') or self.__dict__.get('model')
        if model is not None:
            return getattr(model, name)
        raise AttributeError(f"'SETAModel' object has no attribute '{name}'")
        
    def set_gating_logging(self, enable=True):
        """Globally enable or disable gating logging on all MoE layers."""
        count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, SequentialMoELayer):
                if enable:
                    module.enable_logging()
                else:
                    module.disable_logging()
                count += 1
        state = "Enabled" if enable else "Disabled"
        print(f"Gating logging {state} for {count} MoE layers.")

    def get_and_clear_gating_logs(self):
        """Retrieve logs from all layers and clear them to free memory."""
        all_logs = {}
        for name, module in self.model.named_modules():
            if isinstance(module, SequentialMoELayer):
                if hasattr(module, 'gating_logs') and module.gating_logs:
                    all_logs[name] = copy.deepcopy(module.gating_logs)
                    module.gating_logs = [] # Clear buffer
        return all_logs

    def generate_expert_grid_csv(self, save_dir):
        """
        Generates a CSV grid showing block counts per expert per layer.
        Includes summary statistics for Unique vs Shared blocks.
        """
        print(f"\nGenerating Expert Grid CSV in {save_dir}...")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        snapshot_data = []
        
        # Iterate through current definitions to build the data
        for exp_key, defn in self.expert_definitions.items():
            # Determine Type (Shared if multiple digits in name, else Unique)
            #  "1" or "exp_T1" -> Unique, "exp_T12" -> Shared
            nums = re.findall(r'\d', str(defn['key_name']))
            is_shared = len(nums) > 1
            expert_type = 'Shared' if is_shared else 'Unique'
            
            # Combine MLP and Attn keys to get full list of active layers
            all_layers = set(defn['mlp'].keys()) | set(defn['attn'].keys())
            
            # Group by Layer Index (ignoring module type for the grid count)
            layer_counts = defaultdict(int)
            for (mod, l_idx) in all_layers:
                # Count blocks in MLP
                if (mod, l_idx) in defn['mlp']:
                    layer_counts[l_idx] += len(defn['mlp'][(mod, l_idx)])
                # Count blocks in Attn
                if (mod, l_idx) in defn['attn']:
                    layer_counts[l_idx] += len(defn['attn'][(mod, l_idx)])

            for l_idx, count in layer_counts.items():
                if count > 0:
                    snapshot_data.append({
                        'Layer': l_idx,
                        'Expert': defn['key_name'],
                        'Blocks': count,
                        'Type': expert_type
                    })

        if not snapshot_data:
            print("  No expert data found to generate CSV.")
            return

        df = pd.DataFrame(snapshot_data)

        #  Pivot to Grid Format (Rows: Expert, Cols: Layer 0-31)
        grid_df = df.pivot(index='Expert', columns='Layer', values='Blocks')
        # Reindex to ensure all 32 layers are present (assuming Llama-7B/13B standard)
        max_layer = 32 # Default standard
        if not df.empty:
            max_layer = max(df['Layer'].max() + 1, 32)
        
        grid_df = grid_df.reindex(columns=range(max_layer), fill_value=0).fillna(0).astype(int)

        #  Add Type Column
        type_map = df[['Expert', 'Type']].drop_duplicates().set_index('Expert')
        grid_df = grid_df.join(type_map)
        
        # Reorder columns: Type first, then layers
        layer_cols = list(range(max_layer))
        cols = ['Type'] + layer_cols
        grid_df = grid_df[cols]

        # Sort Experts (Logic: Unique T1, T2... then Shared T12...)
        def sort_key(s):
            nums = re.findall(r'\d', str(s))
            if not nums: return (0, 0)
            val = nums[0]
            # Primary sort: Length of number string (1 digit < 2 digits)
            # Secondary sort: The number value itself
            return (len(val), int(val))
        
        sorted_experts = sorted(grid_df.index, key=sort_key)
        grid_df = grid_df.reindex(sorted_experts)

        # Calculate Statistics
        numeric_data = grid_df[layer_cols]
        summary_rows = {
            'TOTAL_BLOCKS': numeric_data.sum(),
            'TOTAL_EXPERTS_ACTIVE': (numeric_data > 0).sum(),
            'SUM_UNIQUE_BLOCKS': grid_df.loc[grid_df['Type'] == 'Unique', layer_cols].sum(),
            'SUM_SHARED_BLOCKS': grid_df.loc[grid_df['Type'] == 'Shared', layer_cols].sum()
        }

        # Calculate degrees of sharing (Shared by 2 tasks, 3 tasks, etc.)
        degrees = defaultdict(list)
        for expert_name in grid_df.index:
            nums = re.findall(r'\d', str(expert_name))
            degree = len(nums)
            if degree > 1: degrees[degree].append(expert_name)
        
        for deg in sorted(degrees.keys()):
            subset_experts = degrees[deg]
            summary_rows[f'SUM_SHARED_BY_{deg}_TASKS'] = grid_df.loc[subset_experts, layer_cols].sum()

        summary_df = pd.DataFrame(summary_rows).T
        summary_df['Type'] = 'SUMMARY'
        
        # Combine and Save
        separator = pd.DataFrame([['---'] * len(cols)], columns=cols, index=['---SEPARATOR---'])
        final_csv_df = pd.concat([grid_df, separator, summary_df])
        
        csv_filename = f"expert_grid_status_task_{self.current_task_id}_5k_all_layers_SoS.csv"
        csv_path = os.path.join(save_dir, csv_filename)
        final_csv_df.to_csv(csv_path)
        print(f"  [CSV] Saved Expert Grid Status to: {csv_path}")


def load_SETA_model_from_disk(load_dir, base_model_class=None, base_model_name_or_path=None, device_map="cpu", block_dim=256):
    print(f"Loading SETAModel MoE model from {load_dir}...")
    
    def_path = os.path.join(load_dir, "expert_definitions.json")
    if not os.path.exists(def_path):
        raise FileNotFoundError(f"Could not find expert_definitions.json in {load_dir}")
        
    with open(def_path, 'r') as f:
        loaded_defs = json.load(f)
    
    expert_definitions = {}
    for exp_key, data in loaded_defs.items():
        
        def process_layer_dict(layer_dict):
            processed = {}
            for k, v in layer_dict.items():
                parts = k.split('|')
                key_tuple = (parts[0], int(parts[1]))
                value_tuples = [tuple(item) for item in v]
                processed[key_tuple] = value_tuples
            return processed

        expert_definitions[exp_key] = {
            'task_id': data['task_id'],
            'key_name': data['key_name'],
            'init_from_key': data.get('init_from_key', 'base'),
            'mlp': process_layer_dict(data['mlp']),
            'attn': process_layer_dict(data['attn'])
        }

    weight_path = os.path.join(load_dir, "pytorch_model.bin")
    print(f"  Loading state dictionary from {weight_path}...")
    state_dict = torch.load(weight_path, map_location="cpu")
    
    print("  Loading base HF model...")
    if base_model_class is None:

        base_model_class = AutoModelForCausalLM

    base_model = base_model_class.from_pretrained(
        base_model_name_or_path, 
        device_map=device_map,
        trust_remote_code=True
    )
    
    if "lm_head.weight" in state_dict:
        saved_vocab_size = state_dict["lm_head.weight"].shape[0]
        current_vocab = base_model.get_input_embeddings().weight.shape[0]
        if saved_vocab_size != current_vocab:
            print(f"  [Auto-Resize] Changing model vocab size from {current_vocab} to {saved_vocab_size}.")
            base_model.resize_token_embeddings(saved_vocab_size)
    
    print("  Reconstructing MoE architecture...")
    helper = SETAModel.__new__(SETAModel) 
    helper.model_dim = base_model.config.hidden_size
    # helper.__dict__['block_dim'] = get_block_dim_for_model(base_model.config)
    helper.block_dim = block_dim
    # This creates the structure and fills 'base_weight_ref' from the HF model
    model_structure = helper.convert_linear_layer_to_sequential_moe(base_model, expert_definitions)
    
    print("  Loading expert weights into model...")
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model.") and "layers" in k: 
             new_state_dict[k] = v
        elif k.startswith("base_model.model."): 
             new_state_dict[k[17:]] = v
        else:
            new_state_dict[k] = v
            
    # keys = model_structure.load_state_dict(new_state_dict, strict=False)

    # Filter out size-mismatched keys to handle expert count differences across phases
    filtered_state_dict = {}
    model_state = model_structure.state_dict()
    for k, v in new_state_dict.items():
        if k in model_state and model_state[k].shape != v.shape:
            print(f"  [LoadWarning] Skipping size mismatch: {k} "
                f"ckpt={v.shape} vs model={model_state[k].shape}")
            continue
        filtered_state_dict[k] = v
    keys = model_structure.load_state_dict(filtered_state_dict, strict=False)
    
    # Verify we only missed base keys
    unexpected = keys.unexpected_keys
    missing = keys.missing_keys
    
    if len(unexpected) > 0:
        print(f"  WARNING: {len(unexpected)} unexpected keys found (e.g., {unexpected[:3]})")
    
    # It is EXPECTED to have missing keys for 'base_weight_ref' now.
    base_missing = [k for k in missing if "base_weight_ref" in k]
    real_missing = [k for k in missing if "base_weight_ref" not in k]
    
    print(f"  Weights loaded successfully.")
    print(f"  - Missing 'base_weight_ref' keys (Expected): {len(base_missing)}")
    if len(real_missing) > 0:
        print(f"  - WARNING: Real missing keys: {len(real_missing)} (e.g., {real_missing[:3]})")

    # [FIX-3 GATE-ORDER VERIFY] gating_network.weight is a single ordered tensor,
    # loaded positionally.  Confirm every layer's ModuleDict is in the same
    # (sorted) order used when the checkpoint was written, and that the row count
    # matches the expert count.  A silent mismatch here permutes the router.
    _order_bad, _shape_bad, _n_moe = 0, 0, 0
    for _name, _mod in model_structure.named_modules():
        if not (hasattr(_mod, "experts") and hasattr(_mod, "gating_network")):
            continue
        _n_moe += 1
        _keys = list(_mod.experts.keys())
        if _keys != sorted(_keys):
            _order_bad += 1
            if _order_bad <= 3:
                print(f"  - [gate-order][ERROR] {_name}: experts not sorted.\n"
                      f"      got   : {_keys}\n      sorted: {sorted(_keys)}")
        if _mod.gating_network.out_features != len(_keys):
            _shape_bad += 1
            if _shape_bad <= 3:
                print(f"  - [gate-order][ERROR] {_name}: gating rows="
                      f"{_mod.gating_network.out_features} vs experts={len(_keys)}")
    if _order_bad or _shape_bad:
        raise RuntimeError(
            f"[gate-order] {_order_bad} layer(s) with unsorted experts and "
            f"{_shape_bad} layer(s) with mismatched gate rows out of {_n_moe}. "
            f"Gate rows would be attached to the wrong experts — aborting.")
    print(f"  - [gate-order] verified sorted expert order + gate row count "
          f"on {_n_moe} MoE layers.")

    return model_structure, expert_definitions

class SequentialMoELayer(nn.Module):
    def __init__(self, base_weight, layer_expert_definitions, model_dim, block_dim):
        super().__init__()
        
        self.experts = nn.ModuleDict()
        self.register_buffer('base_weight_ref', base_weight.clone().detach())
        self.model_dim  = model_dim
        self.block_dim  = block_dim     # [MOD] per-instance, replaces global Block_dimension
        
        # [FIX-3 GATE-ORDER] Insert experts in LEXICOGRAPHIC key order.
        # apply_split_and_add() rebuilds self.experts with sorted(keys) and writes
        # gating_network rows in that same order.  This constructor previously used
        # the *insertion order of expert_definitions.json*, which diverges from
        # sorted() because _perform_split_on_share deletes split parents and
        # re-appends children.  Result: gating_network.weight (a single ordered
        # tensor) was loaded positionally into a differently-ordered ModuleDict,
        # silently permuting every gate row at each Phase1->Phase2 handoff.
        # Sorting here makes BOTH construction paths agree, permanently.
        for key in sorted(layer_expert_definitions.keys()):
            defn = layer_expert_definitions[key]
            positions = defn.get('positions', [])
            if not positions:
                continue
            new_expert = LinearLayer_MatrixSparsity(
                self.base_weight_ref,
                index_list=positions,
                block_dim=self.block_dim,
            )
            self.experts[key] = new_expert

        self.gating_network = nn.Linear(self.model_dim, len(self.experts))

        # [FIX-1 GATE-MODE] 'sigmoid' = independent per-expert gates (default),
        # 'softmax' = original behaviour, kept reachable for ablation.
        # Experts own DISJOINT block sets, so softmax's sum-to-one constraint
        # arbitrates nothing; it only divides every expert's contribution by the
        # expert count, which degrades prior tasks AND starves the new one.
        self.gate_mode = os.environ.get("GATE_MODE", "sigmoid").strip().lower()
        if self.gate_mode not in ("sigmoid", "softmax"):
            self.gate_mode = "sigmoid"
        # Under sigmoid, a freshly created expert should start ~fully on so that
        # it trains at g~=1 (the T1 condition) instead of at 1/n_experts.
        self.gate_open_bias = float(os.environ.get("GATE_OPEN_BIAS", "4.0"))
        if self.gate_mode == "sigmoid":
            with torch.no_grad():
                self.gating_network.bias.data.fill_(self.gate_open_bias)

        self.logging_enabled = False
        self.gating_logs = []
        # Stores gate scores from the most recent forward pass so that
        # SETATrainer.compute_loss can apply the gating-RL penalty without
        # a second forward pass.  Cleared after every loss computation.
        self.last_gate_scores = None   # Tensor [batch, seq, n_experts] or None
        self.last_expert_keys = []     # list[str] parallel to gate score columns
    
    def enable_logging(self):
        self.logging_enabled = True
        self.gating_logs = []

    def disable_logging(self):
        self.logging_enabled = False

    def clear_logs(self):
        self.gating_logs = [] 

    def apply_split_and_add(self, split_plan, new_expert_defns, base_weight):
        old_gate_map = {}
        with torch.no_grad():
             old_expert_keys = list(self.experts.keys())
             if hasattr(self, 'gating_network') and len(old_expert_keys) == self.gating_network.out_features:
                for idx, key in enumerate(old_expert_keys):
                    old_gate_map[key] = {
                        'weight': self.gating_network.weight.data[idx, :].clone(),
                        'bias': self.gating_network.bias.data[idx].clone()
                    }
        splits = split_plan.get('splits', [])
        for split in splits:
            old_key = split['old_key_to_delete']
            if old_key in self.experts:
                old_expert = self.experts.pop(old_key)
                
                key_shared = split['new_shared']
                if split['shared_pos']:
                    new_expert_shared = LinearLayer_MatrixSparsity(
                        base_weight, index_list=split['shared_pos'],
                        block_dim=self.block_dim)           # [MOD]
                    self.copy_weights_from_expert(new_expert_shared, old_expert)
                    self.experts[key_shared] = new_expert_shared.to(self.base_weight_ref.device).to(self.base_weight_ref.dtype)
                    del new_expert_shared
                
                key_unique = split['new_unique']
                if key_unique and split['unique_pos']:
                    new_expert_unique = LinearLayer_MatrixSparsity(
                        base_weight, index_list=split['unique_pos'],
                        block_dim=self.block_dim)           # [MOD]
                    self.copy_weights_from_expert(new_expert_unique, old_expert)
                    self.experts[key_unique] = new_expert_unique.to(self.base_weight_ref.device).to(self.base_weight_ref.dtype)
                    del new_expert_unique
                
                del old_expert

        for new_key, defn in new_expert_defns.items():
            positions = defn.get('positions', [])
            if not positions: continue
            if new_key not in self.experts:
                new_expert = LinearLayer_MatrixSparsity(
                    base_weight, index_list=positions,
                    block_dim=self.block_dim)               # [MOD]
                self.experts[new_key] = new_expert.to(self.base_weight_ref.device).to(self.base_weight_ref.dtype)
                del new_expert

        # 4. Rebuild Gating Network
        sorted_keys = sorted(list(self.experts.keys()))
        sorted_experts = nn.ModuleDict()
        for key in sorted_keys:
            sorted_experts[key] = self.experts[key]
        self.experts = sorted_experts

        new_num_experts = len(self.experts)
        
        # Create linear layer AND move to correct Device/Dtype ---
        self.gating_network = nn.Linear(self.model_dim, new_num_experts)
        self.gating_network = self.gating_network.to(self.base_weight_ref.device).to(self.base_weight_ref.dtype)
        
        # 5. Restore weights
        gate_mode = getattr(self, 'gate_mode', 'sigmoid')
        open_bias = float(getattr(self, 'gate_open_bias', 4.0))
        _uninherited = []
        with torch.no_grad():
            for i, new_key in enumerate(sorted_keys):
                if new_key in old_gate_map:
                    self.gating_network.weight.data[i, :] = old_gate_map[new_key]['weight']
                    self.gating_network.bias.data[i] = old_gate_map[new_key]['bias']
                else:
                    # Inherit from ancestor if possible
                    init_key = None
                    for split in splits:
                        if new_key == split['new_shared'] or new_key == split['new_unique']:
                            init_key = split['old_key_to_delete']
                            break

                    if init_key and init_key in old_gate_map:
                        # [FIX-1] Both children inherit the parent's gate value IN FULL.
                        # Under sigmoid the gates are independent, so splitting one
                        # expert into two must not halve either child's activation.
                        self.gating_network.weight.data[i, :] = old_gate_map[init_key]['weight']
                        self.gating_network.bias.data[i] = old_gate_map[init_key]['bias']

                        # Add noise for unique children to break symmetry
                        # is_unique_child = "exp_T" in new_key and len(re.findall(r'\d', new_key)) == 1
                        # if is_unique_child:
                        #      noise = torch.randn_like(self.gating_network.weight.data[i, :]) * 0.02
                        #      self.gating_network.weight.data[i, :] += noise

                        is_shared_child = "exp_T" in new_key and len(re.findall(r'\d', new_key)) > 1
                        if is_shared_child:
                            noise = torch.randn_like(self.gating_network.weight.data[i, :]) * 0.02
                            self.gating_network.weight.data[i, :] += noise

                    else:
                        # [FIX-1] Brand-new expert (exp_TN) with no ancestor.  Previously
                        # this fell through and kept a fresh nn.Linear row, so the ONE
                        # expert that must learn the new task started with the weakest
                        # gate.  Open it explicitly instead.
                        _uninherited.append(new_key)
                        if gate_mode != 'softmax':
                            self.gating_network.bias.data[i] = open_bias

        if _uninherited:
            print(f"      [gate-init] opened {len(_uninherited)} new gate row(s) "
                  f"(mode={gate_mode}): {_uninherited}")

        # [FIX-3 GUARD] An empty old_gate_map means the whole router was silently
        # re-randomised (previously this failed without any message).
        if splits and not old_gate_map:
            print("      [gate-init][WARNING] old_gate_map was EMPTY — no prior gate "
                  "rows could be inherited for this layer. Router re-initialised.")

        assert self.gating_network.out_features == len(self.experts), (
            f"gating_network.out_features={self.gating_network.out_features} != "
            f"len(experts)={len(self.experts)}")
        assert list(self.experts.keys()) == sorted_keys, \
            "expert ModuleDict order diverged from sorted order after split"

        del old_gate_map

    def forward(self, x):
        output = torch.matmul(x, self.base_weight_ref.t())
        if not self.experts: return output

        # ── Routing-mode override (rebuttal ablation; default 'learned') ──
        mode = getattr(self, 'routing_mode', 'learned')
        if mode == 'none':
            self.last_gate_scores = None
            return output                          # frozen base model only

        gate_logits = self.gating_network(x)

        if mode == 'oracle':
            allowed = getattr(self, 'oracle_allowed_idx', None)
            if allowed:
                mask = torch.full_like(gate_logits, float('-inf'))
                mask[..., allowed] = 0.0
                gate_logits = gate_logits + mask
        elif mode == 'random':
            gate_logits = torch.randn_like(gate_logits)

        # [FIX-1 GATE-MODE] Experts hold DISJOINT block sets, so no two experts ever
        # write to the same weight block.  Softmax therefore arbitrates nothing; its
        # only effect is the constraint sum(g)=1, which divides every expert's
        # contribution by the expert count.  Measured consequence: a task learnable
        # in ~150 steps at 1 expert/layer plateaus 3.7x worse at 2.8 experts/layer.
        # Sigmoid gives independent gates: prior experts keep full delta amplitude
        # and the new expert trains at g~=1 regardless of how many experts exist.
        gate_mode = getattr(self, 'gate_mode', 'sigmoid')
        if gate_mode == 'softmax':
            gate_scores = F.softmax(gate_logits, dim=-1)
        else:
            gate_scores = torch.sigmoid(gate_logits)

        if mode == 'uniform':
            # No learned routing. Under softmax this is an even 1/n split;
            # under sigmoid every expert is simply fully on.
            fill = 1.0 if gate_mode != 'softmax' else 1.0 / gate_scores.shape[-1]
            gate_scores = torch.full_like(gate_scores, fill)
        elif mode == 'topk':
            k = min(getattr(self, 'routing_topk', None)
                    or gate_scores.shape[-1], gate_scores.shape[-1])
            topv, topi = gate_scores.topk(k, dim=-1)
            gate_scores = torch.zeros_like(gate_scores).scatter(
                -1, topi, topv)
            if gate_mode == 'softmax':
                # Renormalise only in the coupled regime; under sigmoid the gates
                # are independent and renormalising would reintroduce the coupling.
                gate_scores = gate_scores / gate_scores.sum(-1, keepdim=True).clamp_min(1e-9)

        # Stored so SETATrainer can read routing without a second forward pass.
        # NOTE: under reentrant gradient checkpointing this tensor is DETACHED
        # (the first forward runs under no_grad), so any auxiliary loss built from
        # it is a constant.  SETATrainer now checks .requires_grad and warns.
        self.last_gate_scores = gate_scores
        expert_keys = list(self.experts.keys())
        self.last_expert_keys = expert_keys

        if self.logging_enabled:
            pass

        total_delta_output = torch.zeros_like(output)

        # for i, key in enumerate(expert_keys):
        #     expert = self.experts[key]
        #     y_delta = expert(x)
        #     g_i = gate_scores[..., i].unsqueeze(-1)
        #     total_delta_output = total_delta_output + (y_delta * g_i)

        for i, key in enumerate(expert_keys):
            expert = self.experts[key]
            g_i = gate_scores[..., i].unsqueeze(-1)
            # [MEM] gate applied inside the expert on [B,S,256] row-blocks, so
            # MulBackward no longer saves a full [B,S,out] tensor per expert.
            total_delta_output = total_delta_output + expert(x, gate=g_i)

        return output + total_delta_output
    
    def copy_weights_from_expert(self, new_expert, old_expert):
        """Safely copies weights using normalized indices."""
        bd = self.block_dim    # [MOD] use per-instance block_dim
        old_pos_to_local_idx = {}
        for local_idx, pos in enumerate(old_expert.index_list):
            try:
                norm_pos = tuple(int(x) for x in pos)
                old_pos_to_local_idx[norm_pos] = local_idx
            except ValueError:
                pass

        with torch.no_grad():
            copied_count = 0
            for i_new, pos_raw in enumerate(new_expert.index_list):
                target_pos = tuple(int(x) for x in pos_raw)
                local_idx_old = old_pos_to_local_idx.get(target_pos)
                
                if local_idx_old is not None:
                    dest_start = i_new * bd
                    dest_end   = (i_new + 1) * bd
                    src_start  = local_idx_old * bd
                    src_end    = (local_idx_old + 1) * bd

                    if dest_end <= new_expert.selected_weight.shape[0] and \
                       src_end <= old_expert.selected_weight.shape[0]:
                        
                        new_expert.selected_weight.data[dest_start:dest_end, :] = \
                            old_expert.selected_weight.data[src_start:src_end, :]
                        copied_count += 1
            
            if copied_count == 0 and len(new_expert.index_list) > 0:
                print(f"      [Warning] Copied 0 blocks to {new_expert}. Random init used.")

    
class LinearLayer_MatrixSparsity(torch.nn.Module):
    def __init__(self,
                 base_weight_ref, 
                 bias=None,
                 index_list=[],
                 block_dim=256): 
        super(LinearLayer_MatrixSparsity, self).__init__()

        self.base_weight_ref = base_weight_ref
        self.bias       = bias 
        self.index_list = index_list
        self.block_dim  = block_dim     # [MOD] per-instance, replaces global Block_dimension

        bd = self.block_dim
        self.selected_weight = torch.empty(
            len(index_list) * bd, bd,
            dtype=self.base_weight_ref.dtype, 
            device=self.base_weight_ref.device,
        )

        with torch.no_grad():
            for i in range(len(index_list)):
                index = index_list[i]
                self.selected_weight[i * bd: (i + 1) * bd, :] = \
                    self.base_weight_ref[      
                        index[0] * bd: (index[0] + 1) * bd,
                        index[1] * bd: (index[1] + 1) * bd
                    ]

        # self.selected_weight = nn.Parameter(self.selected_weight)
        # self.selected_weight.requires_grad = True
        # self.fn = linearZ.apply
        # self.block_dim = block_dim 

        self.selected_weight = nn.Parameter(self.selected_weight)
        self.selected_weight.requires_grad = True
        self.fn = linearZ.apply          # kept: reference impl / external imports
        self.block_dim = block_dim 
        # [MEM] cache block layout grouped by output row-block.
        # index_list is read-only after __init__ (verified: no reassignment anywhere),
        # so this cache can never go stale.
        self._by_row = {}
        _seen = set()
        for _i in range(len(index_list)):
            _r, _c = int(index_list[_i][0]), int(index_list[_i][1])
            assert (_r, _c) not in _seen, \
                f"duplicate block {(_r, _c)} in index_list — block-sparse forward sums " \
                f"duplicates, the old dense path overwrote them"
            _seen.add((_r, _c))
            self._by_row.setdefault(_r, []).append((_i, _c))

    # def forward(self, x):
    #     output_delta = self.fn(x,
    #                            self.selected_weight,
    #                            self.base_weight_ref,
    #                            self.index_list,
    #                            self.block_dim)    # [MOD] pass block_dim to autograd fn
    #     return output_delta

    def forward(self, x, gate=None):
        # [MEM] Block-sparse delta. The old path built a dense [out,in] delta_weight
        # per expert (llama2-7b v_proj: 67.1 MB) and pinned it on ctx.delta_weight,
        # which saved_tensors_hooks cannot drop -> 402 experts x 67.1 MB = 27.0 GB
        # resident through the whole forward. Math is unchanged.
        bd = self.block_dim
        W  = self.base_weight_ref
        if W.device != x.device:          # kept: .to() does not move plain attributes
            W = W.to(x.device)
        out = x.new_zeros(*x.shape[:-1], W.shape[0])
        if not self._by_row:
            return out
        for r, items in self._by_row.items():
            acc = None
            for i, c in items:
                xb = x[..., c * bd:(c + 1) * bd]
                db = self.selected_weight[i * bd:(i + 1) * bd, :] \
                     - W[r * bd:(r + 1) * bd, c * bd:(c + 1) * bd]
                t  = torch.matmul(xb, db.t())
                acc = t if acc is None else acc + t
            if gate is not None:
                acc = acc * gate          # [MEM] gate folded here, not on [B,S,out]
            out[..., r * bd:(r + 1) * bd] = acc
        return out

class linearZ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, selected_weight, base_weight, index_list, block_dim):
        device = input.device
        # Ensure base_weight is on the correct device
        base_weight_on_device = base_weight.to(device)
        
        delta_weight = torch.zeros_like(base_weight_on_device)
        
        for i in range(len(index_list)):
            index = index_list[i]
            row_start, row_end = index[0] * block_dim, (index[0] + 1) * block_dim
            col_start, col_end = index[1] * block_dim, (index[1] + 1) * block_dim
            
            base_weight_block = base_weight_on_device[row_start:row_end, col_start:col_end]
            selected_weight_block = selected_weight[i * block_dim: (i + 1) * block_dim, :]
            
            delta_weight[row_start:row_end, col_start:col_end] = selected_weight_block - base_weight_block

        output = torch.matmul(input, delta_weight.t())

        ctx.save_for_backward(input, selected_weight) 
        ctx.index_list = index_list
        ctx.selected_weight_shape = selected_weight.shape
        ctx.delta_weight = delta_weight
        ctx.block_dim = block_dim    # [MOD] save for backward
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, selected_weight = ctx.saved_tensors
        index_list = ctx.index_list
        selected_weight_shape = ctx.selected_weight_shape
        delta_weight = ctx.delta_weight
        block_dim = ctx.block_dim    # [MOD]

        grad_input = torch.matmul(grad_output, delta_weight)

        if input.dim() == 3 and grad_output.dim() == 3:
             grad_delta_weight = torch.matmul(grad_output.transpose(1, 2), input).sum(0)
        elif input.dim() == 2 and grad_output.dim() == 2:
             grad_delta_weight = torch.matmul(grad_output.t(), input)
        else:
             raise ValueError(f"Unexpected dimensions: input {input.dim()}, grad_output {grad_output.dim()}")

        grad_selected_weight = torch.zeros(selected_weight_shape,
                                           dtype=selected_weight.dtype,
                                           device=selected_weight.device)
        for i in range(len(index_list)):
            index = index_list[i]
            row_start, row_end = index[0] * block_dim, (index[0] + 1) * block_dim
            col_start, col_end = index[1] * block_dim, (index[1] + 1) * block_dim
            
            block_grad = grad_delta_weight[row_start:row_end, col_start:col_end]
            grad_selected_weight[i * block_dim: (i + 1) * block_dim, :] = block_grad

        return grad_input, grad_selected_weight, None, None, None   # [MOD] extra None for block_dim
    
# def select_submatrix_based_on_grads(grads, n=660, selection_strategy = "no_restriction", calculate_strategy = "mean_abs", model = "yahma/llama-13b-hf", dataset_name = None, SETA_config_name = None, save_path = None, arch_profile = None, target_projs = None):    
#     """
#     grad: grad information for each MLP Linear weight matrix
#     n: number of sub-matrix to choose
#     arch_profile: optional dict from get_architecture_profile(). When provided,
#                   block dimensions are derived dynamically from model.config instead
#                   of from the hardcoded model-name if-elif chain below.
#                   Supports any architecture (LLaMA-3, Qwen3, Gemma, Mistral …).
#     """

#     if target_projs is None:
#         target_projs = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj")

#     print(f'n: {n}, selection_strategy: {selection_strategy}, calculate_strategy: {calculate_strategy}, model: {model}, dataset_name: {dataset_name}, SETA_config_name: {SETA_config_name}')

#     if arch_profile is not None:
#         Block_dimension = arch_profile['block_dim']
#         grids           = arch_profile['grids']   # {proj_name: (n_rows, n_cols)}
#         print(f"  [select_submatrix] Using arch_profile: block_dim={Block_dimension} | "
#               f"grids={list(grids.keys())}")

#         block_means = {}
#         for key, grad in grads.items():
#             proj_name = key[0] if isinstance(key, tuple) else None
#             if proj_name not in grids:
#                 continue   # skip projections not in profile (e.g. non-attention layers)
#             n_rows, n_cols = grids[proj_name]
#             expected_out   = n_rows * Block_dimension
#             expected_in    = n_cols * Block_dimension
#             if grad.shape[0] != expected_out or grad.shape[1] != expected_in:
#                 print(f"  [select_submatrix] Shape mismatch {key}: "
#                       f"got {grad.shape}, expected ({expected_out},{expected_in}). Skipping.")
#                 continue
#             try:
#                 reshaped_grad = grad.reshape(n_rows, Block_dimension, n_cols, Block_dimension)
#             except RuntimeError as exc:
#                 print(f"  [select_submatrix] Reshape failed {key}: {exc}. Skipping.")
#                 continue
#             if   calculate_strategy == 'mean_abs': block_means[key] = mean_abs(reshaped_grad)
#             elif calculate_strategy == 'abs_mean': block_means[key] = abs_mean_(reshaped_grad)
#             elif calculate_strategy == 'L1':       block_means[key] = L1_norm(reshaped_grad)
#             elif calculate_strategy == 'L2':       block_means[key] = L2_norm(reshaped_grad)
#             else:                                  block_means[key] = mean_abs(reshaped_grad)

#         # Global top-n via min-heap (v_proj filter still applies in no_restriction mode)
#         top_blocks = []
#         count_not_vproj = 0
#         for key, block_mean in block_means.items():
#             key_str = str(key[0]) if isinstance(key, tuple) else str(key)
#             if selection_strategy != "norm_dist" and not any(p == key_str or p in key_str for p in target_projs):
#                 count_not_vproj += 1
#                 continue

#             for i in range(block_mean.shape[0]):
#                 for j in range(block_mean.shape[1]):
#                     abs_mean = block_mean[i, j].item()
#                     if len(top_blocks) < n:
#                         heapq.heappush(top_blocks, (abs_mean, (key, i, j)))
#                     else:
#                         heapq.heappushpop(top_blocks, (abs_mean, (key, i, j)))

#         top_blocks.sort(reverse=True)
#         ranked_blocks             = defaultdict(list)
#         ranked_blocks_with_ranks  = defaultdict(list)
#         for global_rank, (mean, (info, row, col)) in enumerate(top_blocks):
#             ranked_blocks[info].append((row, col))
#             ranked_blocks_with_ranks[info].append((row, col, global_rank))

#         os.makedirs(save_path, exist_ok=True)
#         pkl_path  = os.path.join(save_path, f"ranked_blocks_attention_submatrix_{SETA_config_name}.pkl")
#         json_path = os.path.join(save_path, f"selected_submatrix_attention_{SETA_config_name}.json")
#         with open(pkl_path, "wb") as f:
#             pickle.dump(ranked_blocks, f)
#         json_data = {f"{k[0]}_layer_{k[1]}": v for k, v in ranked_blocks.items()}
#         with open(json_path, "w") as f:
#             json.dump(json_data, f, indent=2)
#         json_ranks = {f"{k[0]}_layer_{k[1]}": v for k, v in ranked_blocks_with_ranks.items()}
#         with open(json_path.replace('.json', '_with_ranks.json'), "w") as f:
#             json.dump(json_ranks, f, indent=2)
#         print(f"  [select_submatrix] arch_profile path: {len(top_blocks)} blocks selected")
#         print(f"  [select_submatrix] count_not_vproj={count_not_vproj}")
#         gc.collect()
#         if torch.cuda.is_available(): torch.cuda.empty_cache()
#         return ranked_blocks
    
#     # Step 1: Calculate absolute value of mean for all grad tensors in every 256x256 block
#     if (model == "yahma/llama-13b-hf") or (model == "NousResearch/Llama-2-13b-hf"):
#         Block_dimension = 256
#         large_d = 54
#         small_d = 20
#     # elif model == "yahma/llama-7b-hf":
#     elif (model == "NousResearch/Llama-2-7b-hf") or (model == "meta-llama/Llama-2-7b-hf") or (model == "yahma/llama-7b-hf") or (model == "meta-llama/Llama-2-7b-chat-hf"):
#         Block_dimension = 256
#         large_d = 43
#         small_d = 16
#     # elif (model == "NousResearch/Meta-Llama-3-8B") or (model == "meta-llama/Meta-Llama-3-8B"):
#     #     Block_dimension = 256
#     #     large_d = 56
#     #     small_d = 16
#     elif (model == "NousResearch/Meta-Llama-3-8B") or (model == "meta-llama/Meta-Llama-3-8B"):
#         Block_dimension = 128
#         large_d = 112   # 14336 / 128
#         small_d = 32    # 4096  / 128
#     elif (model == "openlm-research/open_llama_3b"):
#         Block_dimension = 256
#         large_d = 27  # 3B model has smaller intermediate size
#         small_d = 13  # Proportionally smaller

#     elif "Qwen" in model: # Covers Qwen/Qwen2.5-7B-Instruct, etc.
#         # Qwen 2.5 7B Architecture
#         # Hidden: 3584 | Intermediate: 18944 | KV Heads: 4
#         Block_dimension = 256
#         large_d = 74   # 18944 / 256
#         small_d = 14   # 3584 / 256

#     # Fallback for local paths — detect model size from path name
#     elif "llama2_7b" in model or "llama-2-7b" in model.lower() or "llama_2_7b" in model.lower():
#         Block_dimension = 256
#         large_d = 43
#         small_d = 16
#     # elif "llama3_8b" in model or "llama-3-8b" in model.lower() or "llama_3_8b" in model.lower():
#     #     Block_dimension = 256
#     #     large_d = 56
#     #     small_d = 16

#     elif "llama3_8b" in model or "llama-3-8b" in model.lower() or "llama_3_8b" in model.lower():
#         Block_dimension = 128
#         large_d = 112   # 14336 / 128
#         small_d = 32    # 4096  / 128

#     # ── Qwen3 local paths (e.g. /path/to/models/qwen3_4b) ─────────────────
#     # block_dim=128 for Qwen3-4B (kv_out=640, 640÷256=2.5 not integer)
#     # block_dim=128 for Qwen3-14B (intermediate=13696, 13696÷256=53.5 not integer)
#     # block_dim=256 for Qwen3-8B  (all dims divisible by 256)
#     elif "qwen3_4b" in model.lower() or "qwen3-4b" in model.lower():
#         Block_dimension = 128
#         large_d = 54    # 6912  / 128
#         small_d = 20    # 2560  / 128
#     # elif "qwen3_8b" in model.lower() or "qwen3-8b" in model.lower():
#     #     Block_dimension = 256
#     #     large_d = 43    # 11008 / 256
#     #     small_d = 16    # 4096  / 256
#     elif "qwen3_8b" in model.lower() or "qwen3-8b" in model.lower():
#         Block_dimension = 128
#         large_d = 86    # 11008 / 128  ← verify with model config
#         small_d = 28    # 3584  / 128
#     elif "qwen3_14b" in model.lower() or "qwen3-14b" in model.lower():
#         Block_dimension = 128
#         large_d = 107   # 13696 / 128
#         small_d = 40    # 5120  / 128
#     else:
#         raise ValueError(f"Unknown model '{model}' in select_submatrix_based_on_grads. "
#                          f"Add model dimensions (Block_dimension, large_d, small_d) for this model.")

#     block_means = {}
#     for key, grad in grads.items():
#         # Reshape the grad tensor into 256x256 blocks
#         if key[0] == 'gate_proj' or key[0] == 'up_proj':
#             # print(key[0], grad.size())
#             print(f"gate_proj and up_proj dimension check:{key[0]}, {grad.size()}")

#             reshaped_grad = grad.reshape(large_d, Block_dimension, small_d, Block_dimension)

#         elif key[0] == 'down_proj':
#             # print(key[0], grad.size())
#             print(f"down_proj dimension check:{key[0]}, {grad.size()}")

#             reshaped_grad = grad.reshape(small_d, Block_dimension, large_d, Block_dimension)

#         elif key[0] == 'q_proj' or key[0] == 'k_proj' or key[0] == 'v_proj':

#             if (key[0] == 'k_proj' or key[0] == 'v_proj'):
#                 # GQA models: kv_out < hidden — compute n_rows from actual grad shape
#                 # instead of relying on hardcoded small_d which was set for q_proj
#                 kv_rows = grad.shape[0] // Block_dimension
#                 reshaped_grad = grad.reshape(kv_rows, Block_dimension, small_d, Block_dimension)
#             else:
#                 # # q_proj and o_proj: square (hidden × hidden)
#                 # reshaped_grad = grad.reshape(small_d, Block_dimension, small_d, Block_dimension)

#                 # q_proj: derive out-dim rows from actual grad shape.
#                 # For GQA models (e.g. Qwen3-4B) q_proj out-dim ≠ hidden,
#                 # so small_d (set from hidden) is wrong for the row dimension.
#                 q_rows = grad.shape[0] // Block_dimension
#                 reshaped_grad = grad.reshape(q_rows, Block_dimension, small_d, Block_dimension)

#     # print("tensor shape:", reshaped_grad.shape)
#         if calculate_strategy == 'mean_abs':
#             block_means[key] = mean_abs(reshaped_grad)
#         elif calculate_strategy == 'abs_mean':
#             block_means[key] = abs_mean_(reshaped_grad)
#         elif calculate_strategy == 'L1':
#             block_means[key] = L1_norm(reshaped_grad)
#         elif calculate_strategy == 'L2':
#             block_means[key] = L2_norm(reshaped_grad)

#     # for each linear layer, select certain number of sub-matrix, normal distributed selection
#     if selection_strategy == "norm_dist":
#         # Step 2: Rank all the blocks in all grad tensors using the abs.mean() value
#         ranked_blocks = defaultdict(list)

#         for key, block_mean in block_means.items():
#             indices = torch.argsort(block_mean.view(-1), descending=True)
#             # print("===================================================")
#             # print("indices", indices)
#             top_indices = indices[:n]
#             for idx in top_indices:
#                 # may need to consider int memory cost in the future
#                 row = idx // block_mean.shape[1]
#                 col = idx % block_mean.shape[1]
#                 ranked_blocks[key].append((row.item(), col.item()))

#         print(f'if ranked_blocks: {ranked_blocks}')
#         del indices
#         del top_indices
#         del key
#         del block_mean
#         # Step 3: Return the selected blocks and their corresponding information

#         return ranked_blocks

#     else:
#         # Step 2: Use a min-heap to maintain top n blocks efficiently
#         top_blocks = []
#         count_not_vproj = 0
#         for key, block_mean in block_means.items():
#             key_str = str(key[0]) if isinstance(key, tuple) else str(key)
#             if not any(p == key_str or p in key_str for p in target_projs):
#                 count_not_vproj += 1
#                 continue
#             for i in range(block_mean.shape[0]):
#                 for j in range(block_mean.shape[1]):
#                     abs_mean = block_mean[i, j].item()
#                     if len(top_blocks) < n:
#                         heapq.heappush(top_blocks, (abs_mean, (key, i, j)))
#                     else:
#                         heapq.heappushpop(top_blocks, (abs_mean, (key, i, j)))

#         top_blocks.sort(reverse=True)
        
#         ranked_blocks = defaultdict(list) 
#         ranked_blocks_with_ranks = defaultdict(list)

#         for global_rank, (mean, (info, row, col)) in enumerate(top_blocks):
#             ranked_blocks[info].append((row, col))
#             ranked_blocks_with_ranks[info].append((row, col, global_rank))

#         os.makedirs(save_path, exist_ok=True)

#         save_path_for_pkl = os.path.join(save_path, f"ranked_blocks_attention_submatrix_{SETA_config_name}.pkl")
#         with open(save_path_for_pkl, "wb") as f:
#             pickle.dump(ranked_blocks, f)
        
#         save_path_for_json = os.path.join(save_path, f"selected_submatrix_attention_{SETA_config_name}.json")
#         json_data = {}
#         for key, value in ranked_blocks.items():
#             key_str = f"{key[0]}_layer_{key[1]}"
#             json_data[key_str] = value

#         with open(save_path_for_json, "w") as f:
#             json.dump(json_data, f, indent=2)
#         print(f"Saved Standard JSON to {save_path_for_json}")

#         save_path_for_json_ranks = os.path.join(save_path, f"selected_submatrix_attention_{SETA_config_name}_with_ranks.json")
#         json_data_ranks = {}
#         for key, value in ranked_blocks_with_ranks.items():
#             key_str = f"{key[0]}_layer_{key[1]}"
#             json_data_ranks[key_str] = value

#         with open(save_path_for_json_ranks, "w") as f:
#             json.dump(json_data_ranks, f, indent=2)
        
#         print(f"Saved Ranked JSON to {save_path_for_json_ranks}")

#         print(f"Saved attention submatrix selections to {save_path}")
#         print(f'else ranked_blocks: {ranked_blocks}')
#         print(f"Count of non-v_proj blocks skipped: {count_not_vproj}")
#         print(f'len of block_means.items(): {len(block_means.items())}')


#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#         return ranked_blocks
            
def select_submatrix_based_on_grads(
    grads, 
    n=660, 
    selection_strategy="no_restriction", 
    calculate_strategy="mean_abs", 
    model="yahma/llama-13b-hf", 
    dataset_name=None, 
    SETA_config_name=None, 
    save_path=None, 
    arch_profile=None, 
    target_projs=None
):    
    """
    grads: grad information for each MLP/Attention Linear weight matrix
    n: number of sub-matrices to choose
    arch_profile: optional dict from get_architecture_profile(). When provided,
                  block dimensions are derived dynamically from model.config instead
                  of from the hardcoded model-name if-elif chain below.
                  Supports any architecture (LLaMA-3, Qwen3, Gemma, Mistral …).
    target_projs: tuple of target projection names to filter against (e.g., ("v_proj",)).
    """
    if target_projs is None:
        target_projs = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj")

    print(f'n: {n}, selection_strategy: {selection_strategy}, calculate_strategy: {calculate_strategy}, model: {model}, dataset_name: {dataset_name}, SETA_config_name: {SETA_config_name}')

    block_means = {}
    skipped_non_target_blocks = 0

    # === PHASE 1: Populate block_means (Reshaping & Mean Calculation) ===
    if arch_profile is not None:
        Block_dimension = arch_profile['block_dim']
        grids           = arch_profile['grids']   # {proj_name: (n_rows, n_cols)}
        print(f"  [select_submatrix] Using arch_profile: block_dim={Block_dimension} | grids={list(grids.keys())}")

        for key, grad in grads.items():
            proj_name = key[0] if isinstance(key, tuple) else None
            if proj_name not in grids:
                continue   # skip projections not in profile (e.g. non-attention layers)
            
            n_rows, n_cols = grids[proj_name]
            expected_out   = n_rows * Block_dimension
            expected_in    = n_cols * Block_dimension
            
            if grad.shape[0] != expected_out or grad.shape[1] != expected_in:
                print(f"  [select_submatrix] Shape mismatch {key}: got {grad.shape}, expected ({expected_out},{expected_in}). Skipping.")
                continue
            try:
                reshaped_grad = grad.reshape(n_rows, Block_dimension, n_cols, Block_dimension)
            except RuntimeError as exc:
                print(f"  [select_submatrix] Reshape failed {key}: {exc}. Skipping.")
                continue

            if   calculate_strategy == 'mean_abs': block_means[key] = mean_abs(reshaped_grad)
            elif calculate_strategy == 'abs_mean': block_means[key] = abs_mean_(reshaped_grad)
            elif calculate_strategy == 'L1':       block_means[key] = L1_norm(reshaped_grad)
            elif calculate_strategy == 'L2':       block_means[key] = L2_norm(reshaped_grad)
            else:                                  block_means[key] = mean_abs(reshaped_grad)
    else:
        # Fallback if-elif chain for hardcoded model dimensions
        if (model == "yahma/llama-13b-hf") or (model == "NousResearch/Llama-2-13b-hf"):
            Block_dimension = 256; large_d = 54; small_d = 20
        elif (model == "NousResearch/Llama-2-7b-hf") or (model == "meta-llama/Llama-2-7b-hf") or (model == "yahma/llama-7b-hf") or (model == "meta-llama/Llama-2-7b-chat-hf"):
            Block_dimension = 256; large_d = 43; small_d = 16
        elif (model == "NousResearch/Meta-Llama-3-8B") or (model == "meta-llama/Meta-Llama-3-8B"):
            Block_dimension = 128; large_d = 112; small_d = 32
        elif (model == "openlm-research/open_llama_3b"):
            Block_dimension = 256; large_d = 27; small_d = 13
        elif "Qwen" in model:
            Block_dimension = 256; large_d = 74; small_d = 14
        elif "llama2_7b" in model or "llama-2-7b" in model.lower() or "llama_2_7b" in model.lower():
            Block_dimension = 256; large_d = 43; small_d = 16
        elif "llama3_8b" in model or "llama-3-8b" in model.lower() or "llama_3_8b" in model.lower():
            Block_dimension = 128; large_d = 112; small_d = 32
        elif "qwen3_4b" in model.lower() or "qwen3-4b" in model.lower():
            Block_dimension = 128; large_d = 54; small_d = 20
        elif "qwen3_8b" in model.lower() or "qwen3-8b" in model.lower():
            Block_dimension = 128; large_d = 86; small_d = 28
        elif "qwen3_14b" in model.lower() or "qwen3-14b" in model.lower():
            Block_dimension = 128; large_d = 107; small_d = 40
        else:
            raise ValueError(f"Unknown model '{model}' in select_submatrix_based_on_grads. "
                             f"Add model dimensions (Block_dimension, large_d, small_d) for this model.")

        for key, grad in grads.items():
            proj_key = key[0] if isinstance(key, tuple) else str(key)
            
            # Reshape grad tensor safely, avoiding UnboundLocalError and silent corruption
            if proj_key == 'gate_proj' or proj_key == 'up_proj':
                reshaped_grad = grad.reshape(large_d, Block_dimension, small_d, Block_dimension)
            elif proj_key == 'down_proj':
                reshaped_grad = grad.reshape(small_d, Block_dimension, large_d, Block_dimension)
            elif proj_key in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
                # Safely derive rows/cols from actual grad shape for GQA & square attention heads
                rows = grad.shape[0] // Block_dimension
                cols = grad.shape[1] // Block_dimension
                reshaped_grad = grad.reshape(rows, Block_dimension, cols, Block_dimension)
            else:
                # Safe dynamic fallback for any unexpected projection (e.g. custom adapters)
                if grad.shape[0] % Block_dimension == 0 and grad.shape[1] % Block_dimension == 0:
                    rows = grad.shape[0] // Block_dimension
                    cols = grad.shape[1] // Block_dimension
                    reshaped_grad = grad.reshape(rows, Block_dimension, cols, Block_dimension)
                else:
                    print(f"  [select_submatrix] Skipping unhandled or non-divisible shape for {key}: {grad.shape}")
                    continue

            if   calculate_strategy == 'mean_abs': block_means[key] = mean_abs(reshaped_grad)
            elif calculate_strategy == 'abs_mean': block_means[key] = abs_mean_(reshaped_grad)
            elif calculate_strategy == 'L1':       block_means[key] = L1_norm(reshaped_grad)
            elif calculate_strategy == 'L2':       block_means[key] = L2_norm(reshaped_grad)
            else:                                  block_means[key] = mean_abs(reshaped_grad)

    # === PHASE 2: Submatrix Selection Strategies ===
    ranked_blocks = defaultdict(list)
    ranked_blocks_with_ranks = defaultdict(list)

    if selection_strategy == "norm_dist":
        # Per-layer normal distributed selection
        for key, block_mean in block_means.items():
            key_str = str(key[0]) if isinstance(key, tuple) else str(key)
            if not any(p == key_str or p in key_str for p in target_projs):
                skipped_non_target_blocks += 1
                continue

            indices = torch.argsort(block_mean.view(-1), descending=True)
            top_indices = indices[:n]
            for idx in top_indices:
                row = (idx // block_mean.shape[1]).item()
                col = (idx % block_mean.shape[1]).item()
                ranked_blocks[key].append((row, col))
        print(f'if ranked_blocks: {ranked_blocks}')
    else:
        # Global top-n via min-heap across all evaluated layers
        top_blocks = []
        for key, block_mean in block_means.items():
            key_str = str(key[0]) if isinstance(key, tuple) else str(key)
            if not any(p == key_str or p in key_str for p in target_projs):
                skipped_non_target_blocks += 1
                continue

            for i in range(block_mean.shape[0]):
                for j in range(block_mean.shape[1]):
                    abs_mean = block_mean[i, j].item()
                    if len(top_blocks) < n:
                        heapq.heappush(top_blocks, (abs_mean, (key, i, j)))
                    else:
                        heapq.heappushpop(top_blocks, (abs_mean, (key, i, j)))

        top_blocks.sort(reverse=True)
        for global_rank, (mean, (info, row, col)) in enumerate(top_blocks):
            ranked_blocks[info].append((row, col))
            ranked_blocks_with_ranks[info].append((row, col, global_rank))

        print(f'else ranked_blocks: {ranked_blocks}')

    # === PHASE 3: Unified Serialization & Cleanup ===
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        
        # Save PKL
        pkl_path = os.path.join(save_path, f"ranked_blocks_attention_submatrix_{SETA_config_name}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(ranked_blocks, f)
        
        # Save Standard JSON
        json_path = os.path.join(save_path, f"selected_submatrix_attention_{SETA_config_name}.json")
        json_data = {f"{k[0]}_layer_{k[1]}": v for k, v in ranked_blocks.items()}
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"Saved Standard JSON to {json_path}")
        
        # Save Ranked JSON (only generated when global min-heap rankings exist)
        if ranked_blocks_with_ranks:
            ranks_path = os.path.join(save_path, f"selected_submatrix_attention_{SETA_config_name}_with_ranks.json")
            json_ranks = {f"{k[0]}_layer_{k[1]}": v for k, v in ranked_blocks_with_ranks.items()}
            with open(ranks_path, "w") as f:
                json.dump(json_ranks, f, indent=2)
            print(f"Saved Ranked JSON to {ranks_path}")
        
        print(f"Saved attention submatrix selections to {save_path}")

    print(f"Count of non-target blocks skipped: {skipped_non_target_blocks}")
    print(f"Total evaluated gradient tensors: {len(block_means)}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return ranked_blocks

def mean_abs(grad_tensor):
    return grad_tensor.mean(dim=(1, 3)).abs()

def abs_mean_(grad_tensor):
    return grad_tensor.abs().mean(dim=(1, 3))

def L1_norm(grad_tensor):
    print(f"use L1 norm as calculation strategy")

    return grad_tensor.abs().sum(dim=(1, 3))

def L2_norm(grad_tensor):
    return torch.sqrt(torch.sum(grad_tensor.abs() ** 2, dim=(1, 3)))

# Please refer to https://github.com/microsoft/DeepSpeedExamples/blob/75df1d7250452bcc7c065797a95c982bc8caab0b/applications/DeepSpeed-Chat/dschat/utils/utils.py#L210
def get_optimizer_sparse_grouped_parameters(
    model,
    weight_decay,
    SETA_lr,
    lora_lr=5e-4,
    no_decay_name_list=[
        "bias", "layer_norm.weight", "layernorm.weight", "norm.weight",
        "ln_f.weight"
    ],
    lora_name_list=["lora_right_weight", "lora_left_weight"],
):

    for name, param in model.named_parameters():
        if (not any(nd in name.lower() for nd in no_decay_name_list)
                and param.requires_grad and not any(nd in name.lower() for nd in lora_name_list)):
            print(f"name0:{name}")
    for n, p in model.named_parameters():
        if (not any(nd in n.lower() for nd in no_decay_name_list)
                and p.requires_grad and any(nd in n.lower() for nd in lora_name_list)):
            print(f"name1:{n}")

    for n, p in model.named_parameters():
        if (any(nd in n.lower() for nd in no_decay_name_list) and p.requires_grad):
            print(f"name2:{n}")



    optimizer_grouped_parameters = [
        {
            "params": #tmp
            [
                p for n, p in model.named_parameters()
                if (not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad and not any(nd in n.lower() for nd in lora_name_list))
            ]
            ,
            "weight_decay":
            weight_decay,
            "lr":
            SETA_lr
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad and any(nd in n.lower() for nd in lora_name_list))
            ],
            "weight_decay":
            weight_decay,
            "lr":
            lora_lr
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n.lower()
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            0.0,
        },
    ]

    non_empty_groups = []
    for group in optimizer_grouped_parameters:
        if group["params"]:
            non_empty_groups.append(group)

    print(f"group parameters: {non_empty_groups}")

    return non_empty_groups
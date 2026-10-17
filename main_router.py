import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

# Force CUDA to initialize with correct device BEFORE any other imports
import torch
if torch.cuda.is_available():
    _ = torch.zeros(1).cuda()  # triggers CUDA init on the correct device
    print(f"[Init] CUDA initialized: {torch.cuda.get_device_name(0)}", flush=True)
else:
    print("[Init] WARNING: CUDA not available, running on CPU", flush=True)

from SETA_config import SETAConfig, SETA_create_and_prepare_model, create_local_datasets_task, create_local_datasets, create_hf_datasets, \
    gradient_collection, gradient_collection_finetune, get_optimizer_grouped_parameters, create_trace_benchmark_datasets
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
import collections
import math
import warnings
from packaging import version
from trl import SFTTrainer
import gc 
from huggingface_hub import login
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_utils import TrainOutput
from transformers.utils import logging
from transformers.trainer_callback import TrainerState
#  1: Removed deleted import -> from transformers.trainer_pt_utils import reissue_pt_warnings
try:
    from transformers.file_utils import WEIGHTS_NAME
except ImportError:
    from transformers.utils import WEIGHTS_NAME  # transformers 5.x
import torch.nn as nn
import torch.nn.functional as F   # [FIX-2] used by SETATrainer.compute_loss
import json
import pickle
import re
import sys
import copy
from dataclasses import dataclass, field
from typing import Optional, List, Union, Dict, Any
from datasets import Dataset
from transformers import HfArgumentParser, TrainingArguments, set_seed, Trainer, get_scheduler
from transformers.trainer import *
import importlib.util
from sklearn.metrics import accuracy_score
from dev_trace_benchmark_eval import evaluate_trace_benchmark
# [ROUTER-EVAL] routing-ablation copy of main.py: imports the router copy of the model file
from SETA_SoS_router import SETAModel, load_SETA_model_from_disk, select_submatrix_based_on_grads, expert_task_ids

try:
    from SETA_analysis_hooks import (
        SETAAnalysisLogger, attach_hooks_to_trainer,
        hook_fft_step, hook_training_loss, hook_lambda_drift,
        hook_gate_routing, hook_expert_population, hook_end_of_task_drift,
    )
    _ANALYSIS_HOOKS_AVAILABLE = True
except ImportError:
    _ANALYSIS_HOOKS_AVAILABLE = False
from torch.optim import AdamW
import tempfile
import shutil
from datetime import datetime
import numpy as np
import heapq
import statistics

os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.get_logger(__name__)

def compute_metrics(eval_pred):
    """
    Compute metrics for reasoning tasks (commonsense, math reasoning) with Llama 2 7B
    This handles causal language modeling where we predict next tokens
    """
    import numpy as np
    from sklearn.metrics import accuracy_score
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  
    predictions_flat = predictions.reshape(-1)
    labels_flat = labels.reshape(-1)
    
    valid_mask = labels_flat != -100
    predictions_valid = predictions_flat[valid_mask]
    labels_valid = labels_flat[valid_mask]
    
    if len(labels_valid) == 0:
        return {"eval_accuracy": 0.0, "num_valid_tokens": 0}
    
    token_accuracy = accuracy_score(labels_valid, predictions_valid)
    return {"eval_accuracy": token_accuracy, "num_valid_tokens": len(labels_valid)}

def is_torch_xla_available():
    spec = importlib.util.find_spec("torch_xla")
    return spec is not None

class PatchedTrainerState(TrainerState):
    def __init__(self, stateful_callbacks=None, **kwargs):
        super().__init__(**kwargs)
        self.stateful_callbacks = stateful_callbacks or []

class ExportableState:
    def save_state(self, save_directory: str): raise NotImplementedError
    def load_state(self, load_directory: str): raise NotImplementedError

def _is_peft_model(model):
    return hasattr(model, "peft_config") or "peft" in str(type(model)).lower()

def cleanup_trainer(trainer):
    print("--- Aggressive Cleanup Initiated ---")
    if trainer is None: return
    if hasattr(trainer, 'optimizer'): del trainer.optimizer
    if hasattr(trainer, 'lr_scheduler'): del trainer.lr_scheduler
    if hasattr(trainer, 'model'): 
        try: trainer.model.to('cpu')
        except: pass
        del trainer.model
    if hasattr(trainer, 'accelerator'):
        try: trainer.accelerator.free_memory()
        except: pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("--- Aggressive Cleanup Finished ---")

@dataclass
class SETAModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model"})
    chat_template_format: Optional[str] = field(default="none")
    use_peft_SETA: Optional[bool] = field(default=False)
    SETA_dropout: float = field(default=0.0)
    SETA_offload: Optional[bool] = field(default=False)
    SETA_zero_stage: int = field(default=0)
    num_submatrix_mlp: int = field(default=0)
    num_submatrix_attn: int = field(default=0)
    selection_strategy: str = field(default="no_restriction")
    calculation_strategy: str = field(default="mean_abs")
    target_modules: Optional[List[str]] = field(default=None)
    SETA_deepspeed: Optional[str] = field(default="configs/SETA_deepspeed_config.yaml")
    SETA_learning_rate: Optional[float] = field(default=1e-4)
    SETA_w_decay: Optional[float] = field(default=0.0)
    expert_creation_threshold: Optional[int] = field(default=5)
    pruning_min_blocks: Optional[int] = field(default=5,
        metadata={"help": "Discard layers with fewer than this many selected blocks after split."})
    tiny_remainder_threshold: Optional[int] = field(default=5,
        metadata={"help": "Absorb remainder layers smaller than this threshold into shared expert."})

    block_dimension: Optional[int] = field(default=256,
    metadata={"help": "Block size for sparse submatrix selection. "
                       "256 for LLaMA-2 7B, 128 for LLaMA-3 8B and Qwen3-4B."})

    # [RL_Lamda] Step B + C options — two mode switches, one fraction
    layer_norm_scoring: bool = field(default=False,
        metadata={"help": "Normalise each layer's block scores before heap."})
    divergence_scoring: bool = field(default=False,
        metadata={"help": "Weight blocks by divergence from prior task gradients."})
    budget_mode: str = field(default="legacy",
        metadata={"help": "'legacy' = fixed num_submatrix_attn | 'adaptive' = data-driven"})
    block_budget_fraction: float = field(default=0.0,
        metadata={"help": "Fraction of total v_proj blocks to select. "
                           "Analogous to LoRA rank. 0.0 = use num_submatrix_attn."})
    # [RL_Lamda] Ablation tag — isolates each config's FFT output directory
    ablation_tag: str = field(default="original_seta",
        metadata={"help": "Ablation config name. Appended to save_path_FFT."})

    # [Ablation] No expert sharing — each task trains its own isolated expert
    no_expert_sharing: bool = field(default=False,
        metadata={"help": "If True, skip Split-on-Share overlap detection. "
                           "Each task creates a completely isolated expert (exp_TN). "
                           "Prior experts are kept frozen and never split. "
                           "Useful as an ablation baseline vs. standard SETA sharing."})

@dataclass
class FFTModelArguments:
    full_ft_steps: int = field(default=100)
    use_reentrant: Optional[bool] = field(default=False)
    fft_offload: Optional[bool] = field(default=False)
    fft_zero_stage: Optional[int] = field(default=2)
    # [RL_Lamda] Step A — two mode strings only, no thresholds
    fft_stopping_mode: str = field(default="fixed",
        metadata={"help": "'fixed' = run full_ft_steps | 'dynamic' = self-calibrated stop"})

@dataclass
class TaskControlArguments:
    current_task_id: int = field(default=1)
    resume_checkpoint_path: Optional[str] = field(default=None)
    manual_output_path: Optional[str] = field(default=None)
    skip_eval_task_list: str = field(default="")
    eval_task_id:int = field(default=1)
    use_model_name: str = field(default="llama2_7b_test")
    run_phase: str = field(default="auto", metadata={"help": "phase0_fft, phase1_split, phase2_train, phase3_eval"})
    eval_current_task_only: bool = field(default=False, metadata={"help": "If True, only evaluate the current task."})

@dataclass
class DataArguments:
    splits: Optional[str] = field(default="train,test")
    eval_set_size: Optional[int] = field(default=120)
    compute_fp32_loss: Optional[bool] = field(default=False)
    add_eot_token: Optional[bool] = field(default=False)
    max_seq_length: Optional[int] = field(default=2048)
    append_concat_token: Optional[bool] = field(default=False)
    add_special_tokens: Optional[bool] = field(default=False)
    packing: Optional[bool] = field(default=False)
    train_sample_ratio: Optional[float] = field(default=1.0)
    regularization_type: Optional[str] = field(default="weight_l2")
    # λ for shared expert L2 regularisation:
    #   total_loss = task_loss + lamda × Σ ||θ_shared - θ_snapshot||²
    # θ_snapshot = shared expert weights at split time (before Task k training)
    # Applies only to shared experts — prior unique experts are frozen.
    # lamda=0.0 disables regularisation entirely.
    # Note: 'lambda' is a Python keyword — field is named 'lamda'.
    lamda: Optional[float] = field(default=0.1,
        metadata={"help": "λ base coefficient for shared expert regularisation. "
                           "0.0 = disabled."})
    # Four ablation modes for Phase 2:
    #   fixed  + gating_rl=False  →  plain L2 with same λ for all shared experts
    #   fixed  + gating_rl=True   →  plain L2  +  gate routing penalty
    #   dynamic + gating_rl=False →  per-expert λ (accumulated knowledge × drift)
    #   dynamic + gating_rl=True  →  per-expert λ  +  gate routing penalty
    lambda_mode: Optional[str] = field(default="fixed",
        metadata={"help": "'fixed' = uniform lamda for all shared experts | "
                           "'dynamic' = per-expert λ_i scaled by n_prior and live drift. "
                           "Ignored when lamda=0.0."})
    gating_rl: Optional[bool] = field(default=False,
        metadata={"help": "If True, penalise gate routing to at-risk shared experts "
                           "proportional to λ_i. Uses only current-task data — "
                           "no prior-task samples required."})
    max_samples: Optional[int] = field(default=None)
    task_list: Optional[str] = field(default=None)
    base_dataset_path: Optional[str] = field(default='data_files/TRACE-Benchmark/LLM-CL-Benchmark_5000/')


# =============================================================================
# [RL_Lamda] Step A/B/C — Parameter-free Phase 0 system
#
# Only user-facing parameter: BLOCK_BUDGET_FRACTION (like LoRA rank)
# All other behaviour is self-calibrated from the gradient data.
# =============================================================================

# ── Step A constants (measurement granularity, not thresholds) ───────────────
_FFT_MIN_STEPS = 15   # always run at least 15 steps
_FFT_WINDOW    = 5    # window for CV estimation


def _fsys_check_stopping(grads, step, fft_state):
    """
    Step A: Contribution-ratio self-calibration.

    ratio(t) = step_contribution(t) / cumulative_norm(t)

    Stop when ratio(t) < CV(recent_ratios), where
    CV = std/mean of the ratio over the last _FFT_WINDOW steps.

    The data determines the stopping point — no threshold needed.
    When the contribution ratio falls within its own noise band,
    the block ranking has stabilised.
    """
    if step < _FFT_MIN_STEPS:
        return False, f"step={step}<min={_FFT_MIN_STEPS}"

    total_sq = sum(
        g.float().norm().item() ** 2
        for key, g in grads.items()
        if isinstance(key, tuple) and any(p in str(key[0]) for p in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"))
    )
    current_norm = total_sq ** 0.5 if total_sq > 0 else 1e-10

    prev_norm     = fft_state.get("cumulative_norm", 0.0)
    step_contrib  = abs(current_norm - prev_norm)
    fft_state["cumulative_norm"] = current_norm

    ratio = step_contrib / (current_norm + 1e-10)
    hist  = fft_state.setdefault("norm_history", [])
    hist.append(ratio)

    if len(hist) < _FFT_WINDOW + 1:
        return False, f"step={step} | warming up ({len(hist)}/{_FFT_WINDOW+1})"

    recent = hist[-_FFT_WINDOW:]
    mu  = statistics.mean(recent)
    sig = statistics.stdev(recent) if len(recent) > 1 else float("inf")
    cv  = sig / (mu + 1e-10)

    # below_cv = ratio < cv
    
    below_cv = ratio <= (mu + sig)

    fft_state["stable_count"] = fft_state.get("stable_count", 0) + 1 if below_cv else 0

    log_entry = {"step": step, "ratio": round(ratio,6), "cv": round(cv,6),
                 "mu": round(mu,6), "stable_count": fft_state["stable_count"]}
    fft_state.setdefault("stepA_log", []).append(log_entry)
    # [Analysis hook A1] — record FFT convergence signal per step
    # hook_fft_step is called by the trainer that owns _fsys_state;
    # the trainer passes itself via fft_state["trainer_ref"] if set.
    if "_trainer_ref" in fft_state and _ANALYSIS_HOOKS_AVAILABLE:
        try:
            hook_fft_step(fft_state["_trainer_ref"], step, ratio, cv,
                          fft_state["stable_count"], mu)
        except Exception:
            pass

    log = f"step={step} | ratio={ratio:.4f} | CV={cv:.4f} | stable={fft_state['stable_count']}"
    
    # if fft_state["stable_count"] >= 30:   # 2 consecutive checks = 10 steps confirmed
    #     return True, log + " → STOP"

    # Task-type-aware threshold: classification tasks stabilise faster
    # generation tasks (MeetingBank, 20Minuten) need more steps
    _dataset_name = fft_state.get("dataset_name", "")
    _gen_tasks    = {"MeetingBank", "20Minuten"}
    _stop_thresh  = 15 if _dataset_name in _gen_tasks else 10
    if fft_state["stable_count"] >= _stop_thresh:
        return True, log + f" → STOP (thresh={_stop_thresh})"
    
    return False, log


# ── Step B — Layer-normalised scoring ────────────────────────────────────────

def _fsys_layer_norm_scoring(grads, arch_profile):
    """
    B1: Standardise each layer's block-score matrix to zero mean / unit std.
    Fixes layer starvation — all layers compete fairly in the global heap.
    No parameters.
    """
    block_dim = arch_profile["block_dim"]
    grids     = arch_profile["grids"]
    out = {}
    layer_score_stats = {}   # layer → {before_mean, before_std, after_mean, after_std}
    for key, grad in grads.items():
        proj  = key[0] if isinstance(key, tuple) else None
        layer = key[1] if isinstance(key, tuple) else 0
        if proj not in grids:
            out[key] = grad; continue
        n_rows, n_cols = grids[proj]
        if grad.shape[0] != n_rows * block_dim or grad.shape[1] != n_cols * block_dim:
            out[key] = grad; continue
        try:
            bm   = grad.reshape(n_rows, block_dim, n_cols, block_dim).abs().mean(dim=(1, 3))
            # [FIX-COL] Standardise per input-column block BEFORE the layer-wide z-score.
            # dL/dW scales with input activation norm, so outlier hidden channels inflate
            # an entire column of bm uniformly. A single layer-wide (mu, sig) cannot remove
            # that. Measured on v_proj C-STANCE: 257/288 (89.2%) of all col-0 blocks taken,
            # 17/36 layers took all 8. unbiased=False so n_rows==1 yields 0, not NaN.
            col_mu  = bm.mean(dim=0, keepdim=True)
            col_sig = bm.std(dim=0, keepdim=True, unbiased=False) + 1e-8
            bm_c    = (bm - col_mu) / col_sig
            mu   = bm_c.mean()
            sig  = bm_c.std() + 1e-8
            bm_n = (bm_c - mu) / sig
            bm_n = bm_n.clamp_min(0.0)
            sf   = (bm_n / (bm.abs() + 1e-10)).repeat_interleave(block_dim, dim=0) \
                                                .repeat_interleave(block_dim, dim=1)
            out[key] = grad.float() * sf
            # Record before/after for metrics
            layer_score_stats[f"{proj}_layer_{layer}"] = {
                "before_mean": round(bm.mean().item(), 6),
                "before_std":  round(bm.std().item(),  6),
                "after_mean":  round(bm_n.mean().item(), 6),
                "after_std":   round(bm_n.std().item(),  6),
            }
        except Exception:
            out[key] = grad
    print(f"  [StepB/layer_norm] Applied to {len(out)} tensors.")
    return out, layer_score_stats


def _fsys_save_gradients(grads, arch_profile, save_path_FFT, config_name):
    """
    Save block-mean gradient tensors after FFT for divergence scoring.
    File: {save_path_FFT}/grad_blockmeans_{config_name}.pt
    """
    block_dim = arch_profile["block_dim"]
    grids     = arch_profile["grids"]
    to_save   = {}
    for key, grad in grads.items():
        proj = key[0] if isinstance(key, tuple) else None
        if proj not in grids or not any(p in str(proj) for p in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj")):
            continue
        n_rows, n_cols = grids[proj]
        if grad.shape[0] != n_rows * block_dim or grad.shape[1] != n_cols * block_dim:
            continue
        try:
            bm = grad.reshape(n_rows, block_dim, n_cols, block_dim).abs().mean(dim=(1, 3))
            to_save[key] = bm.cpu().float()
        except Exception:
            pass
    pt_path = os.path.join(save_path_FFT, f"grad_blockmeans_{config_name}.pt")
    os.makedirs(save_path_FFT, exist_ok=True)
    torch.save(to_save, pt_path)
    print(f"  [StepB/divergence] Saved grad block-means → {pt_path} ({len(to_save)} tensors)")


def _fsys_divergence_scoring(grads, arch_profile, save_path_FFT, task_id, prior_config_names):
    """
    B2: score = α×ĝ + (1-α)×|D_k|
    D_k = G_k - mean(G_{1..k-1})  — computed in gradient space (same units).
    α = 1/(1+log(task_id)) — auto-decays, no parameter.

    Loads prior task gradients from grad_blockmeans_*.pt files saved by
    _fsys_save_gradients(). This is the  for the count-proxy bug.
    """
    if task_id <= 1 or not prior_config_names:
        return grads, {}
    alpha = 1.0 / (1.0 + math.log(max(task_id, 2)))
    print(f"  [StepB/divergence] task_id={task_id} | α={alpha:.3f}")

    prior_tensors: dict = {}
    loaded = 0
    for cfg in prior_config_names:
        pt = os.path.join(save_path_FFT, f"grad_blockmeans_{cfg}.pt")
        if not os.path.exists(pt):
            continue
        try:
            bm_dict = torch.load(pt, map_location="cpu")
            for key, bm in bm_dict.items():
                prior_tensors.setdefault(key, []).append(bm.float())
            loaded += 1
        except Exception as e:
            print(f"  [StepB/divergence] Failed to load {pt}: {e}")

    if loaded == 0:
        print("  [StepB/divergence] No prior gradient files found — skipped.")
        return grads, {}

    block_dim = arch_profile["block_dim"]
    grids     = arch_profile["grids"]
    out = {}
    divergence_stats = {}   # layer_key → mean_divergence
    for key, grad in grads.items():
        proj  = key[0] if isinstance(key, tuple) else None
        layer = key[1] if isinstance(key, tuple) else 0
        if proj not in grids:
            out[key] = grad; continue
        n_rows, n_cols = grids[proj]
        if grad.shape[0] != n_rows * block_dim or grad.shape[1] != n_cols * block_dim:
            out[key] = grad; continue
        try:
            bm         = grad.reshape(n_rows, block_dim, n_cols, block_dim).abs().mean(dim=(1, 3))
            prior_mean = torch.stack(prior_tensors[key]).mean(0) if key in prior_tensors \
                         else torch.zeros_like(bm)

            # ── [FIX-5 UNIT MISMATCH] ────────────────────────────────────────
            # `grads` arriving here has ALREADY been through _fsys_layer_norm_scoring,
            # so `bm` is on a standardised (~unit-variance) scale.  `prior_mean` comes
            # from _fsys_save_gradients, which saved RAW |grad| block-means.  Taking
            # (bm - prior_mean) on incompatible scales made the term degenerate:
            #   prior_mean << bm  ->  D_k ~= bm      -> score ~= bm  (divergence is a NO-OP,
            #                                          so full_system == no_divergence)
            #   prior_mean >> bm  ->  D_k ~= prior_mean -> selection ranked by the PRIOR
            #                                          tasks' gradient map
            # Rank-standardise both operands per layer before differencing so the
            # comparison is scale-free and reflects relative block importance.
            def _std(t):
                mu = t.mean()
                sd = t.std()
                if not torch.isfinite(sd) or sd < 1e-12:
                    return torch.zeros_like(t)
                return (t - mu) / sd

            bm_n    = _std(bm)
            # prior_n = _std(prior_mean) if prior_mean.abs().sum() > 0 else torch.zeros_like(bm_n)
            
            # [FIX-COL] bm arrives column-corrected (see FIX-COL in _fsys_layer_norm_scoring),
            # prior_mean is raw |grad| from disk. Put prior through the same col→layer→abs
            # pipeline before _std, or the difference at D_k compares incompatible spaces.
            def _colmatch(t):
                cmu = t.mean(dim=0, keepdim=True)
                csd = t.std(dim=0, keepdim=True, unbiased=False) + 1e-8
                z   = (t - cmu) / csd
                lmu, lsd = z.mean(), z.std()
                if not torch.isfinite(lsd) or lsd < 1e-12:
                    return torch.zeros_like(t)
                return ((z - lmu) / lsd).clamp_min(0.0)
            prior_n = _std(_colmatch(prior_mean)) if prior_mean.abs().sum() > 0 else torch.zeros_like(bm_n)
            D_k     = (bm_n - prior_n).abs()

            # Combine in the standardised space.  bm_n is zero-mean so the sum can
            # be negative; shift to >= 0 because the block-mean of the returned
            # tensor is what _fsys_adaptive_budget ranks on.
            score = alpha * bm_n + (1.0 - alpha) * D_k
            score = score - score.min()

            # Unchanged from the original: sf rescales each block so that the
            # block-mean of the returned gradient equals `score`.
            sf    = (score / (bm.abs() + 1e-10)).repeat_interleave(block_dim, dim=0) \
                                                  .repeat_interleave(block_dim, dim=1)
            out[key] = grad.float() * sf
            divergence_stats[f"{proj}_layer_{layer}"] = {
                "mean_divergence": round(D_k.mean().item(), 6),
                "max_divergence":  round(D_k.max().item(), 6),
                "alpha":           round(alpha, 4),
                "bm_median":       round(bm.median().item(), 8),
                "prior_median":    round(prior_mean.median().item(), 8),
            }
        except Exception:
            out[key] = grad
    print(f"  [StepB/divergence] Applied ({loaded} prior gradient files).")
    return out, divergence_stats


# ── Step C — Adaptive budget ──────────────────────────────────────────────────

def _fsys_find_json(save_path_FFT, config_name):
    """Find selected_submatrix JSON trying both naming conventions."""
    for suffix in ["_with_ranks.json", ".json"]:
        p = os.path.join(save_path_FFT,
                         f"selected_submatrix_attention_{config_name}{suffix}")
        if os.path.exists(p):
            return p
    return None


def _fsys_load_prior_positions(save_path_FFT, prior_config_names):
    """Flat set of 'layer_key_r_c' for all prior task selected blocks."""
    S = set()
    for cfg in prior_config_names:
        p = os.path.join(save_path_FFT, f"selected_submatrix_attention_{cfg}.json")
        if not os.path.exists(p):
            continue
        try:
            with open(p) as f:
                data = json.load(f)
            for lkey, positions in data.items():
                for pos in positions:
                    S.add(f"{lkey}_{pos[0]}_{pos[1]}")
        except Exception:
            pass
    return S


def _fsys_load_usage_map(save_path_FFT):
    """Load block usage counts from disk."""
    p = os.path.join(save_path_FFT, "block_usage_map.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


def _fsys_save_usage_map(save_path_FFT, usage_map, selected_json):
    """Update and save block usage map after selection."""
    for lkey, positions in selected_json.items():
        if lkey not in usage_map:
            usage_map[lkey] = {}
        for pos in positions:
            k = f"{pos[0]}_{pos[1]}"
            usage_map[lkey][k] = usage_map[lkey].get(k, 0) + 1
    p = os.path.join(save_path_FFT, "block_usage_map.json")
    with open(p, "w") as f:
        json.dump(usage_map, f)


def _fsys_adaptive_budget(grads, arch_profile, usage_map, n_base, S_prior, task_id=1):
    """
    Step C: Fully data-driven budget.

    n_base = BLOCK_BUDGET_FRACTION × total_v_proj_blocks  (set by user like LoRA rank)

    Layer quota: inverse of observed coverage from usage_map.
        quota(l) ∝ 1 / (coverage(l) + 0.1)
        Starved layers get more quota automatically — no beta parameter.

    Stop when the marginal unique rate stabilises across 3 windows of 50 blocks.
        No rho_min floor — data determines the stopping point.
    """
    n_layers = arch_profile["n_layers"]
    grids    = arch_profile["grids"]
    block_dim = arch_profile["block_dim"]

    # Layer coverage from usage_map
    layer_used  = [0] * n_layers
    layer_total = [0] * n_layers
    for proj, (n_rows, n_cols) in grids.items():
        if not any(p in str(proj) for p in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj")):
            continue
        for l in range(n_layers):
            layer_total[l] += n_rows * n_cols
            lkey = f"{proj}_layer_{l}"
            if lkey in usage_map:
                layer_used[l] += len(usage_map[lkey])

    inv_cov   = [1.0 / (layer_used[l] / max(layer_total[l], 1) + 0.1) for l in range(n_layers)]
    total_inv = sum(inv_cov)

    # ── [FIX-6 ADAPTIVE BUDGET NEVER FIRED] ──────────────────────────────────
    # Quotas used to be sized so that sum(quota) ~= n_base (minus int() rounding).
    # The walk below can therefore never pass n_base: every layer hits its cap
    # first, the loop exhausts all_blocks, and `expanded = max(0, total - n_base)`
    # is 0 by construction.  Observed on all 6 tasks of both models: N_k* came out
    # at 1210-1216 against N_base=1228, expanded=0 every time — i.e.
    # budget_mode='adaptive' behaved identically to a fixed budget.
    # Size the quotas against the expansion ceiling instead, and let the
    # unique-rate stability rule (below) decide where to actually stop.
    expansion_cap = float(os.environ.get("BUDGET_EXPANSION_CAP", "1.5"))
    n_ceiling     = int(n_base * expansion_cap)
    quota     = {l: max(1, int(n_ceiling * inv_cov[l] / total_inv)) for l in range(n_layers)}

    deep_q = sum(quota[l] for l in range(2 * n_layers // 3, n_layers))
    print(f"  [StepC] N_base={n_base} | quota_ceiling={n_ceiling} "
          f"(cap x{expansion_cap:g}, sum_quota={sum(quota.values())}) | "
          f"deep_quota={deep_q} | S_prior={len(S_prior)}")

    # Build sorted block list
    all_blocks = []
    for key, grad in grads.items():
        proj  = key[0] if isinstance(key, tuple) else None
        layer = key[1] if isinstance(key, tuple) else 0
        if proj not in grids or not any(p in str(proj) for p in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj")):
            continue
        n_rows, n_cols = grids[proj]
        if grad.shape[0] != n_rows * block_dim or grad.shape[1] != n_cols * block_dim:
            continue
        try:
            bm = grad.reshape(n_rows, block_dim, n_cols, block_dim).abs().mean(dim=(1, 3))
        except RuntimeError:
            continue
        for r in range(n_rows):
            for c in range(n_cols):
                all_blocks.append((bm[r, c].item(), key, r, c, layer))
    all_blocks.sort(reverse=True)

    total_count = unique_count = hard_excluded = 0
    layer_counts: dict = {}
    rate_history = []
    rate_log     = []   # [(total_count, unique_count, rate)] for metrics
    _stop_reason = "blocks_exhausted"   # [FIX-6] overwritten on any break

    for score, key, r, c, layer in all_blocks:
        if layer_counts.get(layer, 0) >= quota.get(layer, 9999):
            continue
        # ── HARD EXCLUSION: skip blocks used by 2+ prior tasks ──────────
        lkey      = f"{key[0]}_layer_{key[1]}"
        bkey      = f"{r}_{c}"
        use_count = usage_map.get(lkey, {}).get(bkey, 0)
        if use_count >= 2:
            hard_excluded += 1
            continue
        # ────────────────────────────────────────────────────────────────
        pos_str   = f"{key[0]}_layer_{key[1]}_{r}_{c}"

        is_unique = pos_str not in S_prior
        total_count  += 1
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
        if is_unique:
            unique_count += 1

        # After N_base, check unique rate stability every 50 blocks.
        # Threshold tightens for later tasks so they keep expanding longer:
        #   T1: threshold=0.050 (stops quickly, block space is fresh)
        #   T3: threshold=0.017 (needs more blocks to stabilise)
        #   T6: threshold=0.008 (must keep walking to find unique blocks)
        stability_threshold = max(0.005, 0.05 / max(task_id, 1))
        if total_count >= n_base and total_count % 50 == 0:
            rate = unique_count / total_count
            rate_history.append(rate)
            rate_log.append((total_count, unique_count, rate))   # for metrics
            if len(rate_history) >= 3:
                change = max(rate_history[-3:]) - min(rate_history[-3:])
                if change < stability_threshold:
                    print(f"  [StepC] Rate stable at {rate:.3f} "
                          f"(Δ={change:.4f} < thr={stability_threshold:.3f}, task_id={task_id}) "
                          f"→ stop at {total_count} blocks")
                    _stop_reason = "rate_stable"
                    break

        # [FIX-6] Bound the walk by the same expansion cap used to size the quotas,
        # so the budget can never balloon beyond the declared ceiling.
        if total_count >= n_ceiling:
            _stop_reason = "hit_ceiling"
            break

    _expanded = max(0, total_count - n_base)
    print(f"  [StepC] N_k*={total_count} | unique={unique_count} "
          f"({unique_count/max(total_count,1)*100:.1f}%) | "
          f"expanded={_expanded} | "
          f"hard_excluded={hard_excluded} | stop={_stop_reason}")
    if _expanded == 0 and expansion_cap > 1.0:
        print(f"  [StepC][NOTE] budget did not expand beyond N_base "
              f"(stop={_stop_reason}); 'adaptive' is behaving like 'fixed' for this task.")

    budget_metrics = {
        "n_base":          n_base,
        "n_final":         total_count,
        "unique_count":    unique_count,
        "unique_pct":      round(unique_count / max(total_count, 1) * 100, 2),
        "expanded":        _expanded,
        "stop_reason":     _stop_reason,
        "rate_history":    [{"total": t, "unique": u, "rate": round(r, 4)}
                            for t, u, r in rate_log],
        "per_layer_quota": {str(l): {
                                "quota":            quota.get(l, 0),
                                "actual":           layer_counts.get(l, 0),
                                "coverage_before":  round(layer_used[l] / max(layer_total[l], 1), 4)
                            } for l in range(n_layers)},
    }
    return total_count, budget_metrics


class SETATrainer(Trainer):
    """
    HuggingFace Trainer extended with four anti-forgetting modes for shared experts:

      lambda_mode='fixed',  gating_rl=False  — uniform L2 on all shared experts
      lambda_mode='fixed',  gating_rl=True   — uniform L2  +  gate routing penalty
      lambda_mode='dynamic', gating_rl=False — per-expert λ_i (accumulated × drift)
      lambda_mode='dynamic', gating_rl=True  — per-expert λ_i  +  gate routing penalty

    Dynamic λ formula (per shared expert i, at training step t):
        λ_i(t) = λ_base × (n_prior_i / (k-1)) × (1 + tanh(drift_ratio_i))
    where:
        n_prior_i   = number of prior task IDs encoded in expert i
        k-1         = total prior tasks  (= task_id - 1)
        drift_ratio = ||W_i(t) - W_snap_i|| / ||W_snap_i||   (live, per step)

    Gate routing penalty (gating_rl=True, current-task data only):
        gate_loss += λ_i × mean(gate_score_to_expert_i)   for each shared expert i
    This discourages routing heavy current-task traffic through at-risk shared
    experts, reducing the gradient magnitude flowing into them.
    """

    def __init__(self, model=None, args=None, expert_snapshots=None,
                 regularization_lambda=0.0, lambda_mode='fixed',
                 gating_rl=False, task_id=1, expert_definitions=None, **kwargs):
        super().__init__(model=model, args=args, **kwargs)
        self.regularization_lambda = regularization_lambda
        self.lambda_mode           = lambda_mode   # 'fixed' | 'dynamic'
        self.gating_rl             = bool(gating_rl)
        self.task_id               = int(task_id)
        self.expert_snapshots      = expert_snapshots or {}
        self.expert_definitions    = expert_definitions or {}

        # Pre-build: expert_key → n_prior_tasks (how many prior tasks live in it)
        # Only populated for dynamic mode to avoid overhead in fixed mode.
        self._n_prior_cache: dict = {}
        if self.lambda_mode == 'dynamic' and self.expert_definitions:
            self._build_n_prior_cache()

        if self.regularization_lambda > 0 and self.expert_snapshots:
            n_shared = len([k for k in (self.expert_definitions or {})
                            if len(expert_task_ids(k)) > 1])
            mode_str = (f"lambda_mode={lambda_mode} | gating_rl={gating_rl} | "
                        f"task_id={task_id} | shared_experts={len(self.expert_snapshots)} | "
                        f"n_shared_expert_keys={n_shared}")
            print(f"SETATrainer: λ={regularization_lambda} | {mode_str}")
        elif self.gating_rl and self.expert_snapshots:
            n_shared = len([k for k in (self.expert_definitions or {})
                            if len(expert_task_ids(k)) > 1])
            print(f"SETATrainer: gating_rl=True (no L2 penalty, lamda=0) | "
                  f"n_shared_expert_keys={n_shared}")
        if self.gating_rl:
            n_shared = len([k for k in (self.expert_definitions or {})
                            if len(expert_task_ids(k)) > 1])
            if n_shared == 0:
                print(f"  [WARNING] gating_rl=True but no shared experts at "
                      f"task_id={task_id} — gate_loss will be 0.0 for this entire "
                      f"task (expected at T1, a bug if seen at T2+)")

        # [Analysis hooks] — initialise logger and snapshot expert population
        if _ANALYSIS_HOOKS_AVAILABLE and args is not None:
            try:
                self._analysis_logger = SETAAnalysisLogger(
                    task_id=task_id,
                    lambda_mode=lambda_mode,
                    gating_rl=gating_rl,
                    save_path=getattr(args, 'output_dir', './seta_analysis_logs'),
                )
                attach_hooks_to_trainer(self)
                if self.expert_definitions:
                    hook_expert_population(self)
            except Exception as _e:
                print(f"  [Analysis hooks] init failed (non-critical): {_e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Helper: build n_prior cache from expert_definitions
    # ─────────────────────────────────────────────────────────────────────────
    def _build_n_prior_cache(self):
        """
        For every expert that is TRAINABLE at the current task (task_id == k),
        count how many PRIOR task IDs are encoded in it.

        Key format: 'exp_T1', 'exp_T12', 'exp_T123', etc.
        Each digit in the suffix is a separate single-digit task ID (tasks 1-6).
        n_prior = number of digits in the key that are NOT the current task.

        Example at task k=3:
          exp_T3   -> digits=['3']           -> n_prior = 0  (new unique, skip regularisation)
          exp_T13  -> digits=['1','3']       -> n_prior = 1
          exp_T123 -> digits=['1','2','3']   -> n_prior = 2

        NOTE: uses r'\d' (single-char match) not r'\d+' so 'exp_T123' parses
        as three separate task IDs ['1','2','3'], not one number ['123'].
        """
        current = self.task_id
        for key, defn in self.expert_definitions.items():
            if defn.get('task_id') != current:
                continue
            key_name = str(defn.get('key_name', key))
            task_ids = expert_task_ids(key_name)  # [ROUTER-FIX] 'exp_T123'->[1,2,3], 'exp_T-1-10'->[1,10]
            n_prior = len([t for t in task_ids if int(t) != current])
            self._n_prior_cache[key] = max(n_prior, 0)

    @staticmethod
    def _extract_expert_key(param_name: str):
        """Extract expert key from a dotted parameter path.

        E.g. 'layers.0.self_attn.v_proj.experts.1_2_3.selected_weight' → '1_2_3'
        Returns None if not an expert parameter.
        """
        m = re.search(r'\.experts\.([^.]+)\.', param_name)
        return m.group(1) if m else None

    def _n_prior_for_key(self, expert_key: str) -> int:
        """Return n_prior for expert_key, falling back to counting key digits.

        Uses r'\d' (single-char) so 'exp_T123' -> ['1','2','3'] -> 3 task IDs,
        not r'\d+' which gives ['123'] -> incorrectly n_prior=1 for T3+ shared experts.
        """
        if expert_key in self._n_prior_cache:
            return self._n_prior_cache[expert_key]
        # Fallback: count individual task digits in the key
        task_ids = expert_task_ids(expert_key)  # [ROUTER-FIX] handles 'exp_T-1-10'
        return max(len([t for t in task_ids if int(t) != self.task_id]), 0)

    def _dynamic_lambda(self, param: torch.Tensor, snap: torch.Tensor,
                        n_prior: int) -> float:
        """
        Compute per-expert dynamic λ_i at the current training step.

            λ_i = λ_base × accum_weight × (1 + tanh(drift_ratio))

        accum_weight = n_prior / (k-1)    — more prior tasks → stronger protection
        drift_ratio  = ||W-W_snap|| / ||W_snap||  — actual live drift from snapshot
        tanh caps the drift factor in (0,1) so total multiplier ∈ (accum, 2×accum).

        At step 0: drift=0 → multiplier=1+tanh(0)=1 → λ_i = λ_base × accum_weight.
        As expert drifts: multiplier grows toward 2 → protection strengthens.
        """
        k_minus_1 = max(self.task_id - 1, 1)
        accum_weight = n_prior / k_minus_1            # ∈ (0, 1]

        with torch.no_grad():
            snap_dev    = snap.to(param.device)
            displacement = (param.data - snap_dev).norm().item()
            snap_norm    = snap_dev.norm().item() + 1e-8
            drift_ratio  = displacement / snap_norm   # ≥ 0

        drift_factor = 1.0 + math.tanh(drift_ratio)  # ∈ (1, 2)
        return float(self.regularization_lambda * accum_weight * drift_factor)

    # ─────────────────────────────────────────────────────────────────────────
    # compute_loss — core of all four modes
    # ─────────────────────────────────────────────────────────────────────────
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)

        # Memory monitor every 50 steps
        if self.state.global_step % 50 == 0:
            import psutil
            cpu_ram = psutil.virtual_memory()
            print(f"\n[Step {self.state.global_step}] Memory Report:")
            print(f"  CPU RAM:  {cpu_ram.used/1024**3:.1f}GB used / "
                  f"{cpu_ram.total/1024**3:.1f}GB total ({cpu_ram.percent}%)")
            if torch.cuda.is_available():
                alloc   = torch.cuda.memory_allocated()  / 1024**3
                reservd = torch.cuda.memory_reserved()   / 1024**3
                total_v = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"  GPU VRAM: {alloc:.1f}GB alloc / "
                      f"{reservd:.1f}GB reserved / {total_v:.1f}GB total")

        task_loss = outputs.get("loss", 0.0)
        device    = (task_loss.device if isinstance(task_loss, torch.Tensor)
                     else next(model.parameters()).device)

        # ── [FIX-2 LOSS NORMALISATION] ───────────────────────────────────────
        # Recompute the task loss ourselves as a plain MEAN over valid (non -100)
        # label tokens.  Rationale: HF's ForCausalLMLoss uses reduction="sum"
        # divided by `num_items_in_batch`, which this override never forwards, and
        # whose value depends on the Trainer version.  When a batch contains
        # samples whose response span was truncated away (see the label audit),
        # that denominator collapses and the reported loss explodes — MeetingBank
        # hit ~6e4 on two different models.  A mean over valid tokens is
        # well-defined, version-independent, and identical to the intended loss
        # whenever the labels are correct.
        _labels = inputs.get("labels", None)
        _logits = getattr(outputs, "logits", None)
        if _labels is not None and _logits is not None:
            # NOTE: no .contiguous() here on purpose — for MeetingBank at
            # max_seq_length=2048 with a 151936-token vocabulary that copy is
            # ~1.2 GB per step.  reshape() handles the non-contiguous slice.
            _shift_logits = _logits[..., :-1, :]
            _shift_labels = _labels[..., 1:]
            _n_valid = int((_shift_labels != -100).sum().item())
            if _n_valid > 0:
                task_loss = F.cross_entropy(
                    _shift_logits.reshape(-1, _shift_logits.size(-1)),
                    _shift_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="mean",
                )
                device = task_loss.device
            else:
                # No supervised token in this batch: contribute exactly zero
                # instead of NaN or a sum/0 blow-up.
                task_loss = (_logits.float().sum() * 0.0)
                device = task_loss.device
                if not getattr(self, "_warned_empty_batch", False):
                    self._warned_empty_batch = True
                    print("  [FIX-2][WARNING] a training batch had ZERO valid label "
                          "tokens (response truncated away). Contributing 0 loss. "
                          "Increase max_seq_length for this task.")

        # One-time sanity bound: per-token CE cannot exceed ln(vocab_size) for a
        # sane forward pass, so anything far above that means the pipeline is
        # broken rather than the model being untrained.
        if (not getattr(self, "_loss_sanity_done", False)
                and isinstance(task_loss, torch.Tensor)
                and self.state.global_step >= 5):
            self._loss_sanity_done = True
            _cfg = getattr(getattr(model, "config", None), "vocab_size", None)
            _lv  = float(task_loss.detach().item())
            if _cfg:
                _bound = 2.0 * math.log(_cfg)
                if (not math.isfinite(_lv)) or _lv > _bound:
                    _msg = (f"[FIX-2] task_loss={_lv:.4g} exceeds 2*ln(vocab_size)="
                            f"{_bound:.2f} at step {self.state.global_step}. The loss or "
                            f"the label masking is broken — refusing to train on a "
                            f"corrupted objective. Set LOSS_SANITY_OFF=1 to bypass.")
                    if os.environ.get("LOSS_SANITY_OFF", "0") == "1":
                        print("  [WARNING] " + _msg)
                    else:
                        raise RuntimeError(_msg)
                else:
                    print(f"  [FIX-2] loss sanity OK: task_loss={_lv:.4f} "
                          f"(bound {_bound:.2f})")
        # ─────────────────────────────────────────────────────────────────────

        # ── Expert L2 regularisation ──────────────────────────────────────────
        # Runs for BOTH fixed and dynamic modes when lamda > 0.
        # expert_key → effective λ_i (used again by gating-RL below).
        reg_loss             = torch.tensor(0.0, device=device)
        shared_expert_lambda = {}   # expert_key → λ_i  (for gating-RL reuse)

        if model.training and self.regularization_lambda > 0 and self.expert_snapshots:
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                clean_name = name.replace("module.", "")
                if clean_name not in self.expert_snapshots:
                    continue

                snap        = self.expert_snapshots[clean_name].to(device)
                expert_key  = self._extract_expert_key(clean_name)

                if self.lambda_mode == 'fixed':
                    lam = self.regularization_lambda
                else:  # dynamic
                    n_prior = self._n_prior_for_key(expert_key) if expert_key else 0
                    if n_prior == 0:
                        # New unique expert — no prior knowledge to protect
                        continue
                    lam = self._dynamic_lambda(param, snap, n_prior)

                reg_loss = reg_loss + lam * torch.sum((param - snap) ** 2)

                # [Analysis hook C1/C2/C3] — record per-expert λ_i and drift
                if _ANALYSIS_HOOKS_AVAILABLE and hasattr(self, "_analysis_logger"):
                    try:
                        hook_lambda_drift(self, expert_key, lam, param, snap,
                                         self.state.global_step)
                    except Exception:
                        pass

                # Track max λ per expert key for gating-RL (multiple params per expert)
                if expert_key and self.gating_rl:
                    shared_expert_lambda[expert_key] = max(
                        shared_expert_lambda.get(expert_key, 0.0), lam)

            if self.state.global_step % 50 == 0:
                mode_tag = f"λ-{self.lambda_mode}"
                print(f"  [{mode_tag}] reg_loss={reg_loss.item():.6f} | "
                      f"tracked_experts={len(shared_expert_lambda)}")

        # ── Gating-RL routing penalty ────────────────────────────────────────
        # Uses current-task data only (already in the forward pass).
        # Penalises gate for routing heavy traffic through at-risk shared experts.
        # λ_i for each shared expert comes from the regularisation block above,
        # so no extra computation is needed here.
        gate_loss = torch.tensor(0.0, device=device)

        if model.training and self.gating_rl:
            # When lamda=0, shared_expert_lambda is empty — build it from n_prior
            if not shared_expert_lambda and self.expert_snapshots:
                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        continue
                    clean_name = name.replace("module.", "")
                    if clean_name not in self.expert_snapshots:
                        continue
                    expert_key = self._extract_expert_key(clean_name)
                    if not expert_key:
                        continue
                    snap   = self.expert_snapshots[clean_name].to(device)
                    n_prior = self._n_prior_for_key(expert_key)
                    if n_prior == 0:
                        continue
                    if self.lambda_mode == 'dynamic':
                        lam = self._dynamic_lambda(param, snap, n_prior)
                    else:
                        k_minus_1    = max(self.task_id - 1, 1)
                        accum_weight = n_prior / k_minus_1
                        lam          = self.regularization_lambda * accum_weight
                    shared_expert_lambda[expert_key] = max(
                        shared_expert_lambda.get(expert_key, 0.0), lam)

            if shared_expert_lambda:
                real_model     = model.module if hasattr(model, "module") else model
                gate_penalties = 0
                for _mname, module in real_model.named_modules():
                    lgs = getattr(module, 'last_gate_scores', None)
                    lek = getattr(module, 'last_expert_keys', None)
                    if lgs is None or not lek:
                        continue
                    # lgs shape: [batch, seq_len, n_experts]  or [batch, n_experts]
                    # We need the last dimension to index experts
                    for col_idx, ekey in enumerate(lek):
                        if ekey not in shared_expert_lambda:
                            continue
                        lam_i = shared_expert_lambda[ekey]
                        # mean routing probability to this expert across batch+seq
                        mean_routing = lgs[..., col_idx].mean()
                        gate_loss    = gate_loss + lam_i * mean_routing
                        gate_penalties += 1
                        # [Analysis hook D1/D2/D4] — record gate routing probability
                        if _ANALYSIS_HOOKS_AVAILABLE and hasattr(self, "_analysis_logger"):
                            try:
                                hook_gate_routing(self, ekey,
                                                  float(mean_routing.item()),
                                                  lam_i,
                                                  self.state.global_step)
                            except Exception:
                                pass

                # [FIX-7] gate_loss is built from SequentialMoELayer.last_gate_scores,
                # which is stashed as a side effect INSIDE the checkpointed forward.
                # With use_reentrant=True that tensor is detached, so this whole term
                # is a constant: it inflates the logged loss and steers nothing.
                # Report the true status once instead of letting it look active.
                if not getattr(self, "_gate_grad_checked", False):
                    self._gate_grad_checked = True
                    _has_grad = bool(getattr(gate_loss, "requires_grad", False)) \
                                and gate_loss.grad_fn is not None
                    if _has_grad:
                        print("  [gating_rl] gradient path OK — penalty is ACTIVE.")
                    else:
                        print("  [gating_rl][WARNING] gate_loss has NO grad_fn "
                              "(gate scores were detached, typically by reentrant "
                              "gradient checkpointing). This term is INERT: it is "
                              "added to the reported loss but produces no gradient. "
                              "Set USE_REENTRANT=False, or disable gating_rl.")
                        self._gate_loss_inert = True

                if self.state.global_step % 50 == 0:
                    _tag = " [INERT]" if getattr(self, "_gate_loss_inert", False) else ""
                    print(f"  [gating_rl] gate_loss={gate_loss.item():.6f}{_tag} | "
                          f"penalised_experts={gate_penalties} | "
                          f"shared_lambda_keys={len(shared_expert_lambda)}")

                # [FIX-7b] Scale check. Under sigmoid every gate sits near 1.0
                # instead of ~1/n_experts, so this penalty is ~n_experts times
                # larger than it was under softmax — AND it now actually has a
                # gradient (FIX-7). Measured on Qwen3 T2 FOMC:
                #     gate_loss = 3.32   vs   task_loss = 1.02
                # i.e. 76% of the objective was the gate penalty, not the task.
                # The penalty's premise (shared experts drift and overwrite prior
                # knowledge) is not supported: measured drift is ~0.1%.
                if (not getattr(self, "_gate_scale_checked", False)
                        and isinstance(task_loss, torch.Tensor)
                        and self.state.global_step >= 5):
                    self._gate_scale_checked = True
                    _tl = float(task_loss.detach().item())
                    _gl = float(gate_loss.detach().item())
                    if _tl > 0 and _gl > _tl:
                        print(f"  [gating_rl][WARNING] gate_loss={_gl:.4f} EXCEEDS "
                              f"task_loss={_tl:.4f} ({100.0*_gl/(_gl+_tl):.0f}% of the "
                              f"objective). The model is optimising the routing "
                              f"penalty rather than the task. Under gate_mode="
                              f"'sigmoid' this term scales with the number of "
                              f"experts. Recommended: GATING_RL=False, or lower "
                              f"--lamda. Measured shared-expert drift is ~0.1%, so "
                              f"there is little for this penalty to protect.")
            else:
                # No shared experts found — gate penalty is silent this task.
                # Expected at T1 (no shared experts yet). A bug if seen at T2+.
                if self.state.global_step % 50 == 0:
                    print(f"  [gating_rl] gate_loss=0.000000 | penalised_experts=0 | "
                          f"shared_lambda_keys=0 | "
                          f"[SILENT — no shared experts at task_id={self.task_id}]")

        # [FIX-7] A detached gate_loss contributes no gradient; including it only
        # corrupts the reported loss (and would trip the FIX-2 sanity bound).
        if getattr(self, "_gate_loss_inert", False):
            total_loss = task_loss + reg_loss
        else:
            total_loss = task_loss + reg_loss + gate_loss

        # [Analysis hook C4] — record loss decomposition per step
        if _ANALYSIS_HOOKS_AVAILABLE and hasattr(self, "_analysis_logger") and model.training:
            try:
                hook_training_loss(self, task_loss, reg_loss, gate_loss, total_loss)
            except Exception:
                pass

        # Memory leak : clear logs and stored gate scores after every step
        real_model = model.module if hasattr(model, "module") else model
        for module in real_model.modules():
            if hasattr(module, "clear_logs"):
                module.clear_logs()
            elif hasattr(module, "gating_logs"):
                module.gating_logs = []
            # Clear stored gate scores so stale tensors don't accumulate
            if hasattr(module, "last_gate_scores"):
                module.last_gate_scores = None

        return (total_loss, outputs) if return_outputs else total_loss

# The following code inspired from https://github.com/HectorHHZ/Sparse_Matrix_Tuning/blob/main/peft/train_smt.py
class FFTtrainer(Trainer):
    #  2: Removed statement from parameter list; moved to after super().__init__()
    def __init__(
            self,
            model=None,
            args: TrainingArguments = None,
            data_collator=None,
            train_dataset=None,
            eval_dataset=None,
            tokenizer=None,
            model_init=None,
            compute_metrics=None,
            callbacks=None,
            optimizers=(None, None),
            num_submatrix=0,
            num_submatrix_attention=0,
            # **kwargs,
    ):
        self.mlp_warmup_grads = {}
        self.attention_warmup_grads = {}
        self.num_submatrix = num_submatrix
        self.num_submatrix_attention = num_submatrix_attention
        self.mlp_grads = {}
        self.attention_grads = {}
        # [RL_Lamda] Step A state — reset each train() call
        self._fsys_stopping_mode = "fixed"   # wired before train()
        self._fsys_state         = {}        # carries norm_history, cumulative_norm, stable_count
        # print(f"args within training class: {args}")

        self.is_fsdp_xla_v2_enabled = False  # Add this line

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            # tokenizer=tokenizer,
            processing_class=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            # **kwargs,
        )
        #  2: Must be set AFTER super().__init__() since self.args is set there
        self.current_gradient_accumulation_steps = self.args.gradient_accumulation_steps

    def train(
            self,
            resume_from_checkpoint: Optional[Union[str, bool]] = None,
            trial: Union["optuna.Trial", Dict[str, Any]] = None,
            ignore_keys_for_eval: Optional[List[str]] = None,
            **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            ignore_keys_for_eval (`List[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments used to hide deprecated arguments
        """

        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args 
        self.neftune_noise_alpha = getattr(args, "neftune_noise_alpha", None)

        self.is_in_train = True

        # Attach NEFTune hooks if necessary
        if self.neftune_noise_alpha is not None:
            self.model = self._activate_neftune(self.model)

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if (args.fp16_full_eval or args.bf16_full_eval) and not args.do_train:
            self._move_model_to_device(self.model, args.device)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)
        self._train_batch_size = self.args.train_batch_size

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None:
            if not is_sagemaker_mp_enabled() and not self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint)
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            if state.train_batch_size is not None:
                self._train_batch_size = state.train_batch_size

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if getattr(self, 'place_model_on_device', True):
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model

        inner_training_loop = find_executable_batch_size(
            self._inner_training_loop, self._train_batch_size, args.auto_find_batch_size
        )
        if args.push_to_hub:
            try:
                # Disable progress bars when uploading models during checkpoints to avoid polluting stdout
                hf_hub_utils.disable_progress_bars()
                return inner_training_loop(
                    args=args,
                    resume_from_checkpoint=resume_from_checkpoint,
                    trial=trial,
                    ignore_keys_for_eval=ignore_keys_for_eval,
                )
            finally:
                hf_hub_utils.enable_progress_bars()
        else:
            return inner_training_loop(
                args=args,
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )

    def _inner_training_loop(
            self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        #  3: Set current_gradient_accumulation_steps at start of loop
        self.current_gradient_accumulation_steps = args.gradient_accumulation_steps
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                # if args.include_tokens_per_second:
                if getattr(args, "include_tokens_per_second", False):

                    num_train_tokens = (
                            self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                # if args.include_tokens_per_second:
                if getattr(args, "include_tokens_per_second", False):

                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            # if args.include_tokens_per_second:
            if getattr(args, "include_tokens_per_second", False):
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_v2_enabled or self.is_fsdp_enabled

        #  5: Guard _created_lr_scheduler with getattr
        if getattr(self, '_created_lr_scheduler', False):
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # self.state = TrainerState(
        #     stateful_callbacks=[
        #         cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
        #     ]
        # )
        self.state = PatchedTrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        # self.state = TrainerState()

        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = dict(args.gradient_checkpointing_kwargs)

            # [FIX-7] Passing an EMPTY dict lets torch fall back to its reentrant
            # default (use_reentrant=True), under which the first forward runs in
            # no_grad and every intermediate tensor is detached.  The --use_reentrant
            # CLI flag was parsed into FFTModelArguments and then never used, so it
            # silently had no effect.  Pin it explicitly here.
            gradient_checkpointing_kwargs.setdefault(
                "use_reentrant",
                str(os.environ.get("USE_REENTRANT", "False")).lower() in ("true", "1", "yes"))
            print(f"  [grad-ckpt] gradient_checkpointing_kwargs={gradient_checkpointing_kwargs}")

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                if hasattr(self, '_fsdp_qlora_plugin_updates'):
                    self._fsdp_qlora_plugin_updates()
                self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if getattr(self, 'use_apex', False):
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        elif self.args.optim in [getattr(OptimizerNames, 'LOMO', None), getattr(OptimizerNames, 'ADALOMO', None)]:
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            try:
                self.compare_trainer_and_checkpoint_args(self.args, self.state)
            except Exception:
                pass
            try:
                self._load_callback_state()
            except Exception:
                pass
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if getattr(self, 'hp_name', None) is not None and getattr(self, '_trial', None) is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if getattr(self, 'hp_search_backend', None) == getattr(HPSearchBackend, 'SIGOPT', None) else trial
            try:
                self.state.trial_params = hp_params(assignments)
            except Exception:
                self.state.trial_params = {}
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # if args.eval_on_start:
        if getattr(args, "eval_on_start", False):
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            # if args.past_index >= 0:
            #     self._past = None
            if getattr(args, 'past_index', -1) >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1

                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To  this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        input_device = inputs[main_input_name].device
                        self.state.num_input_tokens_seen += torch.sum(
                            self.accelerator.gather(
                                torch.tensor(inputs[main_input_name].numel(), device=input_device, dtype=torch.int64)
                            )
                        ).item()
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                # with self.accelerator.accumulate(model):
                #     tr_loss_step = self.training_step(model, inputs)

                if self.accelerator.distributed_type != "DEEPSPEED":
                    with self.accelerator.accumulate(model):
                        tr_loss_step = self.training_step(model, inputs)
                else:
                    tr_loss_step = self.training_step(model, inputs)

                if (
                        args.logging_nan_inf_filter
                        and not is_torch_xla_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    if tr_loss.device != tr_loss_step.device:
                        raise ValueError(
                            f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                        )
                    tr_loss += tr_loss_step

                #  6: Guard floating_point_ops with hasattr
                if hasattr(self, 'floating_point_ops'):
                    self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                        steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                        total_batched_samples % args.gradient_accumulation_steps == 0
                        or
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        is_last_step_and_steps_less_than_grad_acc
                ):
                    #  3: Update current_gradient_accumulation_steps dynamically for last partial batch
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.current_gradient_accumulation_steps = (
                            steps_in_epoch % args.gradient_accumulation_steps or args.gradient_accumulation_steps
                        )
                        # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                        # in accelerate. So, explicitly enable sync gradients to True in that case.
                        self.accelerator.gradient_state._set_sync_gradients(True)
                    else:
                        self.current_gradient_accumulation_steps = args.gradient_accumulation_steps

                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        if is_sagemaker_mp_enabled() and args.fp16:
                            _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif getattr(self, 'use_apex', False):
                            from apex import amp
                            _grad_norm = nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            # Check if any real gradients exist before clipping
                            has_grad = any(p.grad is not None for p in model.parameters())
                            if has_grad:
                                # Sanitize before clipping — fixes nan grad_norm from bf16 overflow
                                for p in model.parameters():
                                    if p.grad is not None:
                                        p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0, posinf=1e4, neginf=-1e4)
                                _grad_norm = self.accelerator.clip_grad_norm_(
                                    model.parameters(),
                                    args.max_grad_norm,
                                )
                            else:
                                _grad_norm = torch.tensor(0.0)

                        if (
                                is_accelerate_available()
                                and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                        ):
                            grad_norm = model.get_global_grad_norm()
                            # In some cases the grad norm may not return a float
                            if hasattr(grad_norm, "item"):
                                grad_norm = grad_norm.item()
                        else:
                            grad_norm = _grad_norm

                        # #  grad_norm display — sanitize BEFORE clip_grad_norm_
                        # for _p in model.parameters():
                        #     if _p.grad is not None:
                        #         _p.grad.data = torch.nan_to_num(
                        #             _p.grad.data, nan=0.0, posinf=1e4, neginf=-1e4)

                        # _grad_norm = self.accelerator.clip_grad_norm_(
                        #     model.parameters(),
                        #     args.max_grad_norm,
                        # )
                                
                        # May need to consider the right position
                        self.mlp_warmup_grads, self.attention_warmup_grads = gradient_collection(model,
                                                                                                 self.mlp_warmup_grads,
                                                                                                 self.attention_warmup_grads,
                                                                                                 self.num_submatrix,
                                                                                                 self.num_submatrix_attention)

        # [RL_Lamda] Step A — dynamic stopping (check EVERY step after min_steps)
        # Must check every step so CV is computed on the full history,
        # not sampled every _FFT_WINDOW steps which would miss the stop point.
                        if (self._fsys_stopping_mode == "dynamic"
                                and self.state.global_step > 0):
                            should_stop, log_str = _fsys_check_stopping(
                                self.attention_warmup_grads,
                                self.state.global_step,
                                self._fsys_state)
                            print(f"  [StepA] {log_str}")
                            if should_stop:
                                print(f"  [StepA] *** STOPPING at step {self.state.global_step} ***")
                                self.control.should_training_stop = True
                        elif self._fsys_stopping_mode == "fixed":
                            if self.state.global_step % 10 == 0:
                                print(f"  [StepA] mode=fixed | step={self.state.global_step} | continuing")

                    self.optimizer.step()

                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run:
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time)
                    
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if is_torch_xla_available():
                        import torch_xla.core.xla_model as xm
                        xm.mark_step()
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    import torch_xla.debug.metrics as met
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        # if args.past_index and hasattr(self, "_past"):
        if getattr(args, 'past_index', -1) > 0 and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                import smdistributed.modelparallel.torch as smp
                smp.barrier()

            try:
                self._load_best_model()
            except Exception:
                pass

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        #  6: Guard store_flos with hasattr
        if hasattr(self, 'store_flos'):
            self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        try:
            run_dir = self._get_output_dir(trial)
            checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)
            # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
            if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
                for checkpoint in checkpoints_sorted:
                    if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                        logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                        shutil.rmtree(checkpoint, ignore_errors=True)
        except Exception:
            pass

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        try:
            self._finish_current_push()
        except Exception:
            pass

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        # for param in self.model.module.parameters():
        for param in self.model.parameters():
        # for param in self.model.parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )
    
def main():
    import warnings
    warnings.filterwarnings("ignore", message=".*max_steps is given.*")
    warnings.filterwarnings("ignore", message=".*copy construct from a tensor.*")
    parser = HfArgumentParser((SETAModelArguments, FFTModelArguments, DataArguments, TrainingArguments, TaskControlArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        SETA_model_args, fft_model_args, data_args, training_args, task_ctrl_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        SETA_model_args, fft_model_args, data_args, training_args, task_ctrl_args = parser.parse_args_into_dataclasses()

    training_args.metric_for_best_model = None
    training_args.load_best_model_at_end = False
    training_args.deepspeed = None  
    set_seed(training_args.seed)
    # SETA_model_args.full_ft_steps = fft_model_args.full_ft_steps

    task_id = task_ctrl_args.current_task_id
    print(f"\n{'='*80}\nSTARTING JOB FOR TASK ID: {task_id}\nPHASE REQUESTED: {task_ctrl_args.run_phase}\n{'='*80}\n")

    if hasattr(data_args, 'task_list') and data_args.task_list:
        data_args.task_list = [name.strip() for name in data_args.task_list.split(',')]
    else:
        data_args.task_list = ['C-STANCE', 'FOMC', 'MeetingBank', 'ScienceQA', 'NumGLUE-cm', '20Minuten', 'MedMCQA']

    current_task_name = data_args.task_list[task_id - 1]
    # Login skipped — using local models with offline mode enabled
    # hf_token = "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"
    # login(hf_token)
    
    dataset_path = current_task_name if os.path.isabs(current_task_name) else os.path.join(data_args.base_dataset_path, current_task_name)
    dataset_name = os.path.basename(current_task_name.rstrip('/'))
    
    print(f"Target Dataset: {dataset_name}\nPath: {dataset_path}")

    print("Initializing Base Model and Tokenizer...")
    base_model, peft_config, tokenizer = SETA_create_and_prepare_model(SETA_model_args, fft_model_args, data_args)
    tokenizer.model_max_length = data_args.max_seq_length or 2048

    print(f"\nLoading data for Task {task_id}...")
    data_args.dataset_name_or_path = dataset_path
    
    train_file_path = os.path.join(dataset_path, "train.json")
    eval_file_path = os.path.join(dataset_path, "eval.json")
    test_file_path = os.path.join(dataset_path, "test.json")

    # Load Datasets
    if os.path.exists(eval_file_path) and os.path.exists(train_file_path):
        train_dataset, eval_dataset, data_collator = create_trace_benchmark_datasets(tokenizer, data_args, training_args, use_second_db=False)
    elif os.path.exists(test_file_path) and os.path.exists(train_file_path):
        train_dataset, eval_dataset, data_collator = create_local_datasets_task(tokenizer, data_args, training_args, apply_chat_template=SETA_model_args.chat_template_format != "none", use_second_db=False)
    else:
        raise ValueError(f"Missing train/test files in {dataset_path}")

    # Add right after train_dataset is created in main():
    sample = train_dataset[0]
    labels = sample['labels']
    labels_tensor = labels.clone().detach() if isinstance(labels, torch.Tensor) else torch.tensor(labels)
    non_masked = (labels_tensor != -100).sum().item()
    print(f"Label check: {non_masked}/{len(labels)} tokens are real labels")

    # ── [FIX-2 LABEL AUDIT] ──────────────────────────────────────────────────
    # ROOT CAUSE this catches: the "### Response: <answer>" span sits at the END
    # of every TRACE prompt.  When max_seq_length truncates a long sample, the
    # response is cut off entirely and the sample is left with ZERO valid label
    # tokens.  The per-batch loss then becomes sum(CE)/tiny_denominator, which is
    # how MeetingBank reached train_loss ~6e4 on BOTH llama2_7b and qwen3_4b
    # (5.69e4 / 6.24e4 — matched across a 4.7x vocabulary difference, i.e. a
    # normalisation artefact, not a numerical explosion).
    # Sample 0 of MeetingBank is already 712 tokens against MAX_SEQ_LENGTH=768.
    _audit_n = min(int(os.environ.get("LABEL_AUDIT_SAMPLES", "512")), len(train_dataset))
    _vocab_size = int(getattr(base_model.config, "vocab_size", 0) or 0)
    _empty, _full, _oov, _at_cap, _valid_counts, _lengths = 0, 0, 0, 0, [], []
    _cap = int(data_args.max_seq_length or 2048)
    for _i in range(_audit_n):
        _s = train_dataset[_i]
        _lb = _s['labels']
        _lb = _lb.clone().detach() if isinstance(_lb, torch.Tensor) else torch.tensor(_lb)
        _n_tok = int(_lb.numel())
        _n_val = int((_lb != -100).sum().item())
        _lengths.append(_n_tok)
        _valid_counts.append(_n_val)
        if _n_val == 0:
            _empty += 1
        elif _n_val == _n_tok:
            _full += 1
        if _n_tok >= _cap:
            _at_cap += 1
        if _vocab_size and _n_val > 0:
            if int(_lb[_lb != -100].max().item()) >= _vocab_size:
                _oov += 1
    _mean_valid = (sum(_valid_counts) / max(len(_valid_counts), 1))
    print(f"[Label audit] n={_audit_n} | max_seq_length={_cap} | "
          f"seq_len mean={sum(_lengths)/max(len(_lengths),1):.0f} max={max(_lengths) if _lengths else 0} | "
          f"valid_labels mean={_mean_valid:.1f}")
    print(f"[Label audit] samples with ZERO valid labels : {_empty} ({100.0*_empty/max(_audit_n,1):.1f}%)")
    print(f"[Label audit] samples at truncation cap      : {_at_cap} ({100.0*_at_cap/max(_audit_n,1):.1f}%)")
    print(f"[Label audit] samples with ALL tokens labeled: {_full}")
    print(f"[Label audit] samples with out-of-vocab label: {_oov}")

    # ── [FIX-2b LEFT-TRUNCATION REPAIR] ──────────────────────────────────────
    # Raising max_seq_length is not sufficient.  Measured on MeetingBank at 2048:
    #   seq_len mean=1333 max=2048 | ZERO valid labels: 192/500 (38.4%)
    #                              | at truncation cap : 198/500 (39.6%)
    # ~40% of transcripts exceed any window we can afford, and because the
    # supervised "### Response:" span sits at the END, right-truncation removes
    # exactly the part we train on.  Keeping the LAST max_seq_length tokens
    # instead preserves the response and drops transcript prefix, which is the
    # correct trade for a summarisation task and costs no extra memory.
    _repair = str(os.environ.get("LEFT_TRUNCATE_REPAIR", "True")).lower() in ("true", "1", "yes")
    if _repair and _empty > 0:
        class _LeftTruncatedDataset(torch.utils.data.Dataset):
            """Wraps a tokenised dataset; for samples whose label span was lost to
            right-truncation, re-truncate from the left so the response survives.
            Samples that already have valid labels are returned untouched."""
            def __init__(self, base, cap, pad_id):
                self.base, self.cap, self.pad_id = base, cap, pad_id
                self.n_repaired = 0
            def __len__(self):
                return len(self.base)
            def __getitem__(self, idx):
                s = dict(self.base[idx])
                lab = s.get('labels', None)
                if lab is None:
                    return s
                if not isinstance(lab, torch.Tensor):
                    lab = torch.tensor(lab)
                if int((lab != -100).sum()) > 0:
                    return s          # already fine
                for k in ('input_ids', 'labels', 'attention_mask'):
                    v = s.get(k, None)
                    if v is None:
                        continue
                    if not isinstance(v, torch.Tensor):
                        v = torch.tensor(v)
                    s[k] = v[-self.cap:]          # keep the TAIL, not the head
                return s

        _pad_id = getattr(tokenizer, "pad_token_id", None)
        train_dataset = _LeftTruncatedDataset(train_dataset, _cap, _pad_id)
        # Re-audit the repaired dataset so the numbers below reflect reality.
        _empty2 = 0
        for _i in range(_audit_n):
            _lb = train_dataset[_i]['labels']
            _lb = _lb if isinstance(_lb, torch.Tensor) else torch.tensor(_lb)
            if int((_lb != -100).sum()) == 0:
                _empty2 += 1
        print(f"[FIX-2b] left-truncation repair applied: zero-label samples "
              f"{_empty} -> {_empty2} of {_audit_n}")
        if _empty2 < _empty:
            _empty = _empty2
        else:
            print("  [WARNING] repair did not help — the response span is longer than "
                  "max_seq_length, or the label mask is built before truncation. "
                  "Increase max_seq_length further.")
    # ─────────────────────────────────────────────────────────────────────────

    _audit_strict = str(os.environ.get("LABEL_AUDIT_STRICT", "True")).lower() in ("true", "1", "yes")
    _empty_frac = _empty / max(_audit_n, 1)
    if _oov > 0:
        raise ValueError(
            f"[Label audit] {_oov} sample(s) contain label ids >= vocab_size "
            f"({_vocab_size}). The loss is undefined — fix tokenisation before training.")
    if _empty_frac > 0.02:
        _msg = (f"[Label audit] {100.0*_empty_frac:.1f}% of '{dataset_name}' samples have NO "
                f"valid label tokens at max_seq_length={_cap}. The response span is being "
                f"truncated away, which inflates the loss by orders of magnitude. "
                f"Raise MAX_SEQ_LENGTH for this task (or truncate the prompt from the LEFT "
                f"so the response is always kept).")
        if _audit_strict:
            raise ValueError(_msg + "  Set LABEL_AUDIT_STRICT=False to override.")
        print("  [WARNING] " + _msg)
    elif _empty > 0:
        print(f"  [WARNING] {_empty} sample(s) have no valid labels; they contribute nothing.")
    if _full > 0:
        print(f"  [WARNING] {_full} sample(s) have EVERY token labeled — the prompt/response "
              f"split failed for those (loss will include instruction tokens).")
    # ─────────────────────────────────────────────────────────────────────────

    # [Split-on-Share layout] FFT_steps_* dirs live under one FFT_steps/ folder.
    # SOS_FFT_ROOT is exported by scripts/setafix_pipeline.slurm; unset -> old behaviour (cwd).
    current_dir = os.environ.get("SOS_FFT_ROOT") or os.getcwd()
    # save_path_FFT = os.path.join(current_dir,f'FFT_steps_RL_Lamda_DA_{task_ctrl_args.use_model_name}_{SETA_model_args.ablation_tag}')

    _fft_mode_tag = (
        f"StepA{1 if fft_model_args.fft_stopping_mode == 'dynamic' else 0}"
        f"B{1 if SETA_model_args.layer_norm_scoring else 0}"
        f"C{1 if SETA_model_args.budget_mode == 'adaptive' else 0}"
        f"_bgt{str(SETA_model_args.block_budget_fraction).replace('.', 'p')}"
    )

    # --- Dynamic prefix based on bash environment variable ---
    proj_mode = os.environ.get("PROJECTION_MODE", "v_proj_only")
    if proj_mode == "v_proj_only":
        fft_prefix = "FFT_steps"
    else:
        fft_prefix = f"FFT_steps_{proj_mode}"
    # ----------------------------------------------------------------

    save_path_FFT = os.path.join(current_dir, f'{fft_prefix}_{task_ctrl_args.use_model_name}_{SETA_model_args.ablation_tag}_{_fft_mode_tag}')


    # [RL_Lamda] Config name: simple task+model — no fixed numbers baked in
    SETA_config_name = f"{dataset_name}_{task_ctrl_args.use_model_name}"
    
    if task_ctrl_args.manual_output_path:
        run_output_dir = task_ctrl_args.manual_output_path
    else:
        run_output_dir = os.path.join(training_args.output_dir, f"task_{task_id}_{dataset_name}")
    
    init_structure_path = os.path.join(run_output_dir, "init_structure_checkpoint")

    phase_requested = task_ctrl_args.run_phase
    should_run_phase0_fft = False
    should_run_phase1_split = False
    should_run_phase2_train = False
    should_run_phase3_eval = False

    if phase_requested == "auto":
        print("WARNING: 'auto' phase detected. Defaulting to sequential execution.")
        should_run_phase2_train = True
        should_run_phase3_eval = True 
    elif phase_requested == "phase0_fft":
        should_run_phase0_fft = True
    elif phase_requested == "phase1_split":
        should_run_phase1_split = True
    elif phase_requested == "phase2_train":
        should_run_phase2_train = True
    elif phase_requested == "phase3_eval":
        should_run_phase3_eval = True

    SETA_model = None

    # =========================================================================
    # [RL_Lamda] Arch profile — used by Step A/B/C (built once from model config)
    # =========================================================================

    # 1. Map PROJECTION_MODE to combinations of Attention and MLP layers
    proj_mode = os.environ.get("PROJECTION_MODE", "all").lower()

    if "v_proj" in proj_mode:
        target_projs = ("v_proj",)
    elif "qkv" in proj_mode and "mlp" in proj_mode:
        target_projs = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj")
    elif "qkv" in proj_mode:
        target_projs = ("q_proj", "k_proj", "v_proj")
    elif "q_proj" in proj_mode:
        target_projs = ("q_proj",)
    elif "mlp" in proj_mode:
        target_projs = ("gate_proj", "up_proj", "down_proj")
    elif "all" in proj_mode or "both" in proj_mode:
        target_projs = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj")
    else:
        target_projs = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj")

    # 2. Accept target_projs in the profiler
    def _build_arch_profile(model_name_or_path, base_model, target_projs):
        """Extract block grid dimensions from model config."""
        config  = base_model.config
        n_layers = config.num_hidden_layers
        block_dim = SETA_model_args.block_dimension

        grids = {}
        for name, mod in base_model.named_modules():
            for proj_key in target_projs:
                if proj_key in name and hasattr(mod, "weight") and proj_key not in grids:
                    out_f, in_f = mod.weight.shape
                    grids[proj_key] = (out_f // block_dim, in_f // block_dim)

        total_blocks = {proj: nr * nc for proj, (nr, nc) in grids.items()}
        return {"n_layers": n_layers, "block_dim": block_dim,
                "grids": grids, "total_blocks_per_proj": total_blocks}

    arch_profile = _build_arch_profile(SETA_model_args.model_name_or_path, base_model, target_projs)


    print(f"  [arch_profile] n_layers={arch_profile['n_layers']} | "
          f"block_dim={arch_profile['block_dim']} | grids={arch_profile['grids']}")

    # =========================================================================
    # [PHASE 0] FFT / GRADIENT COLLECTION
    # =========================================================================
    if should_run_phase0_fft:
        print(f"\n{'='*40}\n[PHASE 0] FFT & STRUCTURE GENERATION\n{'='*40}")
        
        # [RL_Lamda] Skip guard: check whether gradient file already exists in the
        # shared grads store.  The selection JSON lives in the per-variant
        # save_path_FFT and won't exist yet on a first run, so we must check the
        # grads dir here, not the selection dir.
        expected_grad_path = os.path.join(save_path_FFT, f"grad_blockmeans_{SETA_config_name}.pt")
        print(f'expected_grad_path: {expected_grad_path}')
        if os.path.exists(expected_grad_path):
            print(f"Gradient file already exists at: {expected_grad_path}")
            print("Skipping FFT Phase.")
            return
        
        fft_training_args = copy.deepcopy(training_args)
        fft_training_args.max_steps = fft_model_args.full_ft_steps
        fft_training_args.num_train_epochs = 9999  # max_steps takes precedence; suppress warning
        fft_training_args.output_dir = os.path.join(run_output_dir, "FFT_temp")
        
        fft_training_args.bf16 = False
        fft_training_args.fp16 = False
        fft_training_args.gradient_checkpointing = True   # re-enable — saves activation memory
        fft_training_args.per_device_train_batch_size = 1  # reduce from 2 to 1 for long sequences
        fft_training_args.gradient_accumulation_steps = training_args.gradient_accumulation_steps * 2  # compensate

        # fft_training_args.optim = "adamw_bnb_8bit"  # 8-bit optimizer — cuts optimizer memory from 56GB to 14GB

        # 8-bit AdamW, paged to host memory under pressure. Numerically identical
        # to adamw_bnb_8bit — same update rule, same 8-bit states — but the states
        # migrate to CPU when VRAM is tight instead of OOM-ing. Required for 8B:
        # fp32 weights (32.1) + fp32 grads (32.1) + resident states (16.1) = 80.3 GB
        # exceeds the 79.25 GB card before any activation is allocated.
        fft_training_args.optim = "paged_adamw_8bit"


        # ── Task-aware FFT learning rate ─────────────────────────────────────
        # Generation tasks have much longer sequences and unstable loss scales.
        # 20Minuten hit loss=1.49M at step 9 with lr=9.65e-6 — reduce for safety.
        _current_task_name = data_args.task_list[task_ctrl_args.current_task_id - 1]
        _gen_tasks_lr      = {"MeetingBank", "20Minuten"}
        if _current_task_name in _gen_tasks_lr:
            fft_training_args.learning_rate = 2e-6
            print(f"  [FFT LR] Generation task '{_current_task_name}' → lr reduced to 2e-6")
        else:
            print(f"  [FFT LR] Classification task '{_current_task_name}' → lr={training_args.learning_rate:.2e}")
        # ─────────────────────────────────────────────────────────────────────
        
        base_model = base_model.float() 
        print(f"Starting FFT for {fft_model_args.full_ft_steps} steps "
              f"(stopping_mode={fft_model_args.fft_stopping_mode})...")
        fft_trainer = FFTtrainer(
            model=base_model, 
            tokenizer=tokenizer,
            args=fft_training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            num_submatrix=SETA_model_args.num_submatrix_mlp,
            num_submatrix_attention=SETA_model_args.num_submatrix_attn,
        )
        # [RL_Lamda] Step A — wire stopping mode before train()
        fft_trainer._fsys_stopping_mode = fft_model_args.fft_stopping_mode
        # fft_trainer._fsys_state         = {}
        fft_trainer._fsys_state = {"dataset_name": dataset_name}
        # [Analysis hook A1] — give _fsys_check_stopping access to the trainer
        fft_trainer._fsys_state["_trainer_ref"] = fft_trainer
        if fft_model_args.fft_stopping_mode not in ("fixed", "dynamic"):
            raise ValueError(
                f"fft_stopping_mode must be 'fixed' or 'dynamic', "
                f"got: '{fft_model_args.fft_stopping_mode}'. "
                f"Check that the slurm variable FFT_STOPPING_MODE was not corrupted.")
        print(f"  [StepA] mode={fft_model_args.fft_stopping_mode} | "
              f"max_steps={fft_model_args.full_ft_steps} | "
              f"min_steps={_FFT_MIN_STEPS}")

        fft_trainer.train()

        # ── [FIX-2 FFT VALIDITY GATE] ────────────────────────────────────────
        # The block-mean gradients produced here are (a) the block selection for
        # THIS task and (b) saved to disk and averaged into `prior_mean` for every
        # LATER task's divergence scoring.  A diverged FFT therefore poisons the
        # whole remaining stream: MeetingBank's FFT reached train_loss 1.02e10 on
        # llama2_7b, and that tensor was then loaded by tasks 4, 5 and 6.
        # Refuse to persist gradients from a run whose loss is not physically
        # possible for a well-formed language-modelling objective.
        _fft_loss = None
        try:
            for _entry in reversed(fft_trainer.state.log_history):
                if "train_loss" in _entry:
                    _fft_loss = float(_entry["train_loss"]); break
                if _fft_loss is None and "loss" in _entry:
                    _fft_loss = float(_entry["loss"])
        except Exception:
            pass
            
        # _vs = int(getattr(base_model.config, "vocab_size", 0) or 0)
        # if _fft_loss is not None and _vs:
        #     _bound = 2.0 * math.log(_vs)
        #     print(f"  [FFT sanity] train_loss={_fft_loss:.4g} | bound=2*ln({_vs})={_bound:.2f}")

        # _vs = int(getattr(base_model.config, "vocab_size", 0) or 0)
        # if _fft_loss is not None and _vs:
        #     _accum = max(int(getattr(fft_training_args, "gradient_accumulation_steps", 1) or 1), 1)
        #     _fft_loss = _fft_loss / _accum  # custom loop sums micro-batch losses; undo accum inflation
        #     _bound = 2.0 * math.log(_vs)
        #     print(f"  [FFT sanity] train_loss={_fft_loss:.4g} (raw/{_accum}) | bound=2*ln({_vs})={_bound:.2f}")

        _vs = int(getattr(base_model.config, "vocab_size", 0) or 0)
        if _fft_loss is not None and _vs:
            _accum = max(int(getattr(fft_training_args, "gradient_accumulation_steps", 1) or 1), 1)
            _fft_loss = _fft_loss / _accum  # custom loop sums micro-batch losses; undo accum inflation
            _bound = 2.0 * math.log(_vs)
            print(f"  [FFT sanity] train_loss={_fft_loss:.4g} (raw/{_accum}) | bound=2*ln({_vs})={_bound:.2f}")

            if (not math.isfinite(_fft_loss)) or _fft_loss > _bound:
                _m = (f"[FFT sanity] FFT for '{dataset_name}' diverged "
                      f"(train_loss={_fft_loss:.4g} > {_bound:.2f}). Its gradients would "
                      f"corrupt this task's block selection AND every later task's "
                      f"divergence scoring, so they will NOT be saved. Check "
                      f"max_seq_length / label masking for this dataset "
                      f"(see the label audit above). Set LOSS_SANITY_OFF=1 to bypass.")
                if os.environ.get("LOSS_SANITY_OFF", "0") == "1":
                    print("  [WARNING] " + _m)
                else:
                    raise RuntimeError(_m)
        # ─────────────────────────────────────────────────────────────────────

        raw_grads = fft_trainer.attention_warmup_grads

        # [RL_Lamda] Step B — save gradient tensors to the shared grads store
        # BEFORE any scoring (needed for divergence scoring in later tasks)
        os.makedirs(save_path_FFT, exist_ok=True)
        _fsys_save_gradients(raw_grads, arch_profile, save_path_FFT, SETA_config_name)

        # [RL_Lamda] Step B — layer-normalised scoring
        scored_grads = raw_grads
        layer_norm_stats = {}
        if SETA_model_args.layer_norm_scoring:
            scored_grads, layer_norm_stats = _fsys_layer_norm_scoring(scored_grads, arch_profile)

        # [RL_Lamda] Step B — divergence scoring (uses real gradient tensors from grads store)
        divergence_stats = {}
        if SETA_model_args.divergence_scoring and task_id > 1:
            prior_cfgs = [
                f"{data_args.task_list[tid-1]}_{task_ctrl_args.use_model_name}"
                for tid in range(1, task_id)
                if data_args.task_list and tid <= len(data_args.task_list)
            ]
            scored_grads, divergence_stats = _fsys_divergence_scoring(
                scored_grads, arch_profile, save_path_FFT, task_id, prior_cfgs)

        # [RL_Lamda] Step C — compute block budget
        if SETA_model_args.budget_mode == "adaptive":
            # N_base from BLOCK_BUDGET_FRACTION (the one user parameter, like LoRA rank)
            # total_v_proj_blocks = sum(
            #     n_rows * n_cols
            #     for proj, (n_rows, n_cols) in arch_profile["grids"].items()
            #     if "v_proj" in str(proj)
            # ) * arch_profile["n_layers"]
            # n_base = max(1, int(SETA_model_args.block_budget_fraction * total_v_proj_blocks))

            # Budget is ALWAYS 25% × v_proj block count, regardless of projection_mode.
            # projection_mode only decides WHICH projections compete for those slots.
            # total_v_proj_blocks = sum(
            #     n_rows * n_cols
            #     for proj, (n_rows, n_cols) in arch_profile["grids"].items()
            #     if "v_proj" in str(proj)
            # ) * arch_profile["n_layers"]
            # n_base = max(1, int(SETA_model_args.block_budget_fraction * total_v_proj_blocks))

            _cfg = base_model.config
            _head_dim = getattr(_cfg, "head_dim", None) or (_cfg.hidden_size // _cfg.num_attention_heads)
            _v_out = getattr(_cfg, "num_key_value_heads", _cfg.num_attention_heads) * _head_dim
            _bd = arch_profile["block_dim"]
            total_v_proj_blocks = (_v_out // _bd) * (_cfg.hidden_size // _bd) * arch_profile["n_layers"]
            n_base = max(1, int(SETA_model_args.block_budget_fraction * total_v_proj_blocks))
            print(f"  [StepC] v_proj reference blocks={total_v_proj_blocks} | N_base={n_base}")

            prior_cfgs_c = [
                f"{data_args.task_list[tid-1]}_{task_ctrl_args.use_model_name}"
                for tid in range(1, task_id)
                if data_args.task_list and tid <= len(data_args.task_list)
            ]
            # Prior positions come from the shared grads store (written by prior FFT runs)
            S_prior   = _fsys_load_prior_positions(save_path_FFT, prior_cfgs_c)
            # Usage map is per-variant — each RL mode tracks its own block usage
            usage_map = _fsys_load_usage_map(save_path_FFT)
            actual_budget, budget_metrics = _fsys_adaptive_budget(
                scored_grads, arch_profile, usage_map, n_base, S_prior,
                task_id=task_id)
        # else:
        #     # original_seta: use fixed num_submatrix_attn unchanged
        #     actual_budget  = SETA_model_args.num_submatrix_attn
        #     budget_metrics = {"n_base": actual_budget, "n_final": actual_budget,
        #                       "mode": "legacy_fixed"}
        #     print(f"  [Budget] Fixed budget={actual_budget} (original_seta mode)")

        # print("Selecting submatrices based on collected gradients...")


        else:
            # original_seta: use fixed num_submatrix_attn unchanged
            actual_budget  = SETA_model_args.num_submatrix_attn
            budget_metrics = {"n_base": actual_budget, "n_final": actual_budget,
                              "mode": "legacy_fixed"}
            print(f"  [Budget] Fixed budget={actual_budget} (original_seta mode)")

        # [GUARD] A budget below the pruning floor means every selected layer is
        # discarded in Phase 1, producing 0 experts and a base-model eval.
        _pmb = int(os.environ.get("PRUNING_MIN_BLOCKS", 8))
        if actual_budget < _pmb:
            raise RuntimeError(
                f"Budget {actual_budget} < PRUNING_MIN_BLOCKS {_pmb}: every selected "
                f"layer will be pruned in Phase 1. proj_mode={proj_mode}, "
                f"bbf={SETA_model_args.block_budget_fraction}, n_base={n_base}"
            )

        print("Selecting submatrices based on collected gradients...")
        
        selected_submatrix = select_submatrix_based_on_grads(
            scored_grads,
            actual_budget,
            selection_strategy=SETA_model_args.selection_strategy,
            calculate_strategy=SETA_model_args.calculation_strategy,
            model=SETA_model_args.model_name_or_path,
            dataset_name=dataset_name,
            SETA_config_name=SETA_config_name,
            save_path=save_path_FFT,
            arch_profile=arch_profile,
            target_projs=target_projs  # <--- Pass the Attention + MLP tuple here
        )

        # [SETAFIX] keep the selected blocks' gradient TILES (fp16) so the split
        # can compare directions before any training. Pure add-on after selection.
        try:
            from setafix_seam_split import save_grad_tiles
            save_grad_tiles(raw_grads, save_path_FFT, SETA_config_name, arch_profile["block_dim"])
        except Exception as _e:
            print(f"[SETAFIX] grad-tile save skipped: {_e}")

        # [RL_Lamda] Update per-variant usage map after selection
        if SETA_model_args.budget_mode == "adaptive":
            saved_json_path = os.path.join(
                save_path_FFT, f"selected_submatrix_attention_{SETA_config_name}.json")
            if not os.path.exists(saved_json_path):
                saved_json_path = os.path.join(
                    save_path_FFT,
                    f"selected_submatrix_attention_{SETA_config_name}_with_ranks.json")
            if os.path.exists(saved_json_path):
                with open(saved_json_path) as f_json:
                    saved_json = json.load(f_json)
                _fsys_save_usage_map(save_path_FFT, usage_map, saved_json)

        # ── Collect and save per-task selection metrics ────────────────────────
        # Load the saved JSON for per-layer position analysis
        sel_json_path = os.path.join(
            save_path_FFT,
            f"selected_submatrix_attention_{SETA_config_name}_with_ranks.json")
        if not os.path.exists(sel_json_path):
            sel_json_path = os.path.join(
                save_path_FFT,
                f"selected_submatrix_attention_{SETA_config_name}.json")
        selected_json = {}
        if os.path.exists(sel_json_path):
            with open(sel_json_path) as f_json:
                selected_json = json.load(f_json)

        # Per-layer: count, positions, overlap with each prior task
        per_layer_selection = {}
        for lkey, positions in selected_json.items():
            m = re.search(r"_layer_(\d+)$", lkey)
            layer_idx = int(m.group(1)) if m else -1
            rcs = [[p[0], p[1]] for p in positions]   # strip rank if present
            per_layer_selection[lkey] = {
                "layer": layer_idx,
                "count": len(rcs),
                "positions": rcs,
            }

        # Cross-task overlap per layer
        # Prior selection JSONs live in the per-variant save_path_FFT — each RL
        # variant wrote its own selection files for prior tasks.12027892
        cross_task_overlap = {}
        for lkey, info in per_layer_selection.items():
            current_set = {(p[0], p[1]) for p in info["positions"]} 
            overlap_per_task = {}
            for prior_tid in range(1, task_id):
                if not data_args.task_list or prior_tid > len(data_args.task_list):
                    continue
                prior_cfg  = f"{data_args.task_list[prior_tid-1]}_{task_ctrl_args.use_model_name}"
                prior_path = _fsys_find_json(save_path_FFT, prior_cfg)
                if prior_path and os.path.exists(prior_path):
                    try:
                        with open(prior_path) as fp:
                            prior_data = json.load(fp)
                        prior_positions = {(p[0], p[1]) for p in prior_data.get(lkey, [])}
                        overlap = len(current_set & prior_positions)
                        overlap_per_task[f"T{prior_tid}"] = overlap
                    except Exception:
                        pass
            cross_task_overlap[lkey] = {
                "unique": info["count"] - sum(overlap_per_task.values()),
                "overlap_per_task": overlap_per_task,
            }

        # Assemble complete metrics record
        task_metrics = {
            "task_id":            task_id,
            "task_name":          dataset_name,
            "SETA_config_name":   SETA_config_name,
            "ablation_tag":       SETA_model_args.ablation_tag,
            "fft_stopping_mode":  fft_model_args.fft_stopping_mode,
            "actual_fft_steps":   fft_trainer.state.global_step,
            "stepA_log":          fft_trainer._fsys_state.get("stepA_log", []),
            "stepB_layer_norm":   layer_norm_stats,
            "stepB_divergence":   divergence_stats,
            "stepC_budget":       budget_metrics,
            "per_layer_selection": per_layer_selection,
            "cross_task_overlap": cross_task_overlap,
        }

        metrics_path = os.path.join(
            save_path_FFT, f"selection_metrics_{SETA_config_name}.json")
        with open(metrics_path, "w") as f_m:
            json.dump(task_metrics, f_m, indent=2)
        print(f"  [Metrics] Saved → {metrics_path}")
        
        print(f"Structure saved to {save_path_FFT}")
        # Do NOT call cleanup_trainer — moving 7B model to CPU causes OOM → exit -9 → dependency broken
        print("Phase 0 Complete. Exiting.")
        os._exit(0)

    # =========================================================================
    # [PHASE 1] SPLIT / SOS (Structure Only)
    # =========================================================================
    _setafix_structure_only = os.environ.get("SETAFIX_STRUCTURE_ONLY", "0") == "1"
    if should_run_phase1_split and task_id == 1 and _setafix_structure_only:
        # [SETAFIX] structure-only chain needs a task-1 checkpoint without training:
        # build the fresh MoE exactly as Phase 2 would, save it, exit.
        print(f"\n{'='*40}\n[PHASE 1] TASK 1 STRUCTURE (structure-only mode)\n{'='*40}")
        SETA_model = SETAModel(
            SETA_model_args, base_model, SETA_config_name, task_id=1,
            save_path=save_path_FFT, previous_SETA_model=None,
                expert_creation_threshold=SETA_model_args.expert_creation_threshold,
                pruning_min_blocks=SETA_model_args.pruning_min_blocks,
                tiny_remainder_threshold=SETA_model_args.tiny_remainder_threshold,
                block_dimension=SETA_model_args.block_dimension,
                no_expert_sharing=SETA_model_args.no_expert_sharing
        )
        print(f"Saving task-1 structure to: {init_structure_path}")
        SETA_model.save_moe_checkpoint(init_structure_path)
        print("Phase 1 (task 1, structure-only) Complete. Exiting.")
        os._exit(0)

    if should_run_phase1_split and task_id > 1:
        print(f"\n{'='*40}\n[PHASE 1] SPLIT / SOS\n{'='*40}")
        if not task_ctrl_args.resume_checkpoint_path:
            raise ValueError(f"Phase 1 requires --resume_checkpoint_path")
        # [SETAFIX] tell the split validator where its evidence lives
        os.environ.setdefault("SETAFIX_FFT_DIR", save_path_FFT)
        os.environ.setdefault("SETAFIX_CONFIG_NAME", SETA_config_name)
        os.environ.setdefault("SETAFIX_MODEL_NAME", str(task_ctrl_args.use_model_name))
        os.environ.setdefault("SETAFIX_TASK_LIST", ",".join(data_args.task_list or []))
        os.environ.setdefault("SETAFIX_REPORT_DIR", run_output_dir)

        expert_json_path = os.path.join(task_ctrl_args.resume_checkpoint_path, "expert_definitions.json")
        if os.path.exists(expert_json_path):
            print(f"Loading previous SETA model from {task_ctrl_args.resume_checkpoint_path}")
            prev_model, prev_defs = load_SETA_model_from_disk(task_ctrl_args.resume_checkpoint_path, base_model_class=None, base_model_name_or_path=SETA_model_args.model_name_or_path, block_dim=SETA_model_args.block_dimension)
        else:
            print(f"Loading previous Dense base model from {task_ctrl_args.resume_checkpoint_path}")
            prev_model = base_model.__class__.from_pretrained(task_ctrl_args.resume_checkpoint_path, torch_dtype=base_model.dtype)
            prev_defs = None

        class PreviousModelWrapper:
            def __init__(self, model, definitions):
                self.model = model
                self.model_dim = model.config.hidden_size
                self.expert_definitions = definitions

        wrapper = PreviousModelWrapper(prev_model, prev_defs)
        
        SETA_model = SETAModel(
            SETA_model_args, None, SETA_config_name, task_id=task_id,
            save_path=save_path_FFT, previous_SETA_model=wrapper,
                expert_creation_threshold=SETA_model_args.expert_creation_threshold,
                pruning_min_blocks=SETA_model_args.pruning_min_blocks,
                tiny_remainder_threshold=SETA_model_args.tiny_remainder_threshold,
                block_dimension=SETA_model_args.block_dimension,
                no_expert_sharing=SETA_model_args.no_expert_sharing
        )
        
        print(f"Saving split structure to: {init_structure_path}")
        SETA_model.save_moe_checkpoint(init_structure_path)
        print("Phase 1 Complete. Exiting.")
        os._exit(0)

    # =========================================================================
    # [PHASE 2] TRAINING
    # =========================================================================
    if should_run_phase2_train:
        print(f"\n{'='*40}\n[PHASE 2] TRAINING\n{'='*40}")

        if task_id == 1:
            print("Task 1: Creating fresh SETA MoE model")
            # Deep-convert ALL tensors to fp32 including plain tensor attributes
            def convert_to_fp32(module):
                for param in module.parameters():
                    param.data = param.data.float()
                for buf in module.buffers():
                    buf.data = buf.data.float()
                for attr_name in list(vars(module).keys()):
                    attr = getattr(module, attr_name, None)
                    if isinstance(attr, torch.Tensor) and attr.is_floating_point():
                        setattr(module, attr_name, attr.float())
                for child in module.children():
                    convert_to_fp32(child)
            convert_to_fp32(base_model)
            print(f"  [Phase 2] Cast base model to fp32 for stable MoE training")
            SETA_model = SETAModel(
                SETA_model_args, base_model, SETA_config_name, task_id=1,
                save_path=save_path_FFT, previous_SETA_model=None,
                    expert_creation_threshold=SETA_model_args.expert_creation_threshold,
                    pruning_min_blocks=SETA_model_args.pruning_min_blocks,
                    tiny_remainder_threshold=SETA_model_args.tiny_remainder_threshold,
                    block_dimension=SETA_model_args.block_dimension,
                    no_expert_sharing=SETA_model_args.no_expert_sharing
            )
        else:
            if not os.path.exists(os.path.join(init_structure_path, "pytorch_model.bin")):
                 raise FileNotFoundError(f"Split Structure not found at {init_structure_path}. Run Phase 1 first.")
            print(f"Loading structure from {init_structure_path}")
            currnt_model, current_defs = load_SETA_model_from_disk(
                init_structure_path, base_model_class=None,
                base_model_name_or_path=SETA_model_args.model_name_or_path,
                block_dim=SETA_model_args.block_dimension)
            # Deep-convert ALL tensors to fp32 including plain tensor attributes
            # inside SequentialMoELayer that .float() misses (causing float != Half error)
            def convert_to_fp32(module):
                for param in module.parameters():
                    param.data = param.data.float()
                for buf in module.buffers():
                    buf.data = buf.data.float()
                for attr_name in list(vars(module).keys()):
                    attr = getattr(module, attr_name, None)
                    if isinstance(attr, torch.Tensor) and attr.is_floating_point():
                        setattr(module, attr_name, attr.float())
                for child in module.children():
                    convert_to_fp32(child)
            convert_to_fp32(currnt_model)
            print(f"  [Phase 2] Cast loaded model to fp32 for stable MoE training")
            SETA_model = SETAModel(
                SETA_model_args, None, SETA_config_name, task_id=task_id,
                save_path=save_path_FFT, previous_SETA_model=None,
                    expert_creation_threshold=SETA_model_args.expert_creation_threshold,
                    pruning_min_blocks=SETA_model_args.pruning_min_blocks,
                    tiny_remainder_threshold=SETA_model_args.tiny_remainder_threshold,
                    pretrained_moe_model=currnt_model, pretrained_definitions=current_defs,
                    block_dimension=SETA_model_args.block_dimension,
                    no_expert_sharing=SETA_model_args.no_expert_sharing
            )
            del base_model; gc.collect()

        current_model_vocab_size = SETA_model.model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > current_model_vocab_size:
            SETA_model.model.resize_token_embeddings(len(tokenizer))

        SETA_model.freeze_experts_for_training(task_id=task_id)
        # [SETAFIX] training-time protections, both default OFF (SETAFIX_ORTHO / SETAFIX_GATE_PROTECT)
        try:
            from setafix_seam_split import attach_ortho_hooks, freeze_prior_gate_rows
            _setafix_hooks = (attach_ortho_hooks(SETA_model, task_id, run_output_dir),
                              freeze_prior_gate_rows(SETA_model, task_id))
        except Exception as _e:
            print(f"[SETAFIX] training hooks skipped: {_e}")
        SETA_model.print_trainable_parameters()
        if training_args.gradient_checkpointing: SETA_model.model.enable_input_require_grads()

        # [FIX-7] Phase 2 uses the stock HF Trainer, which reads
        # args.gradient_checkpointing_kwargs (None -> reentrant default).  Under
        # reentrant checkpointing the first forward runs in no_grad, so
        # SequentialMoELayer.last_gate_scores is DETACHED and any auxiliary loss
        # built from it (gating_rl) is a constant with no gradient.  Pin the flag.
        if training_args.gradient_checkpointing:
            _gck = dict(training_args.gradient_checkpointing_kwargs or {})
            _gck.setdefault(
                "use_reentrant",
                str(os.environ.get("USE_REENTRANT", "False")).lower() in ("true", "1", "yes"))
            training_args.gradient_checkpointing_kwargs = _gck
            print(f"  [grad-ckpt] Phase-2 gradient_checkpointing_kwargs={_gck}")

        training_args.output_dir = run_output_dir
        trainer = SETATrainer(
            model=SETA_model.model, processing_class=tokenizer, args=training_args,
            train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator,
            compute_metrics=compute_metrics,
            regularization_lambda=data_args.lamda,
            lambda_mode=data_args.lambda_mode,
            gating_rl=data_args.gating_rl,
            task_id=task_id,
            expert_definitions=SETA_model.expert_definitions,
            expert_snapshots=SETA_model.expert_snapshots
        )
        
        # [PROBE] Verify MoE init is numerically identical to base model.
        # _m = trainer.model.eval()
        # with torch.no_grad():
        #     _b = next(iter(trainer.get_train_dataloader()))
        #     _b = {k: v.to(_m.device) for k, v in _b.items() if torch.is_tensor(v)}
        #     _l = _m(**_b).loss.item()
        # print(f"  [PROBE] init loss = {_l:.4f} (base/FFT reference ~1.57)")
        # _m.train()
        
        print(f"Starting SETA MoE Training...")
        trainer.train()

        # [Analysis hook E1/C2/C3] — record final expert drift and flush all logs
        if _ANALYSIS_HOOKS_AVAILABLE and hasattr(trainer, "_analysis_logger"):
            try:
                hook_end_of_task_drift(trainer)
                trainer._analysis_logger.flush_all()
                print(f"  [Analysis hooks] logs flushed to {trainer._analysis_logger.log_dir}")
            except Exception as _e:
                print(f"  [Analysis hooks] flush failed (non-critical): {_e}")

        print(f"Saving to {run_output_dir}...")
        unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
        SETA_model.model = unwrapped_model
        SETA_model.save_moe_checkpoint(run_output_dir)
        
        cleanup_trainer(trainer)
        print("Phase 2 Complete. Exiting to allow separate Eval job.")
        return 

    # =========================================================================
    # [PHASE 3] EVALUATION
    # =========================================================================
    if should_run_phase3_eval:
        print(f"\n{'='*40}\n[PHASE 3] EVALUATION\n{'='*40}")

        print(f"Loading trained model from: {run_output_dir}")
        if not os.path.exists(os.path.join(run_output_dir, "pytorch_model.bin")):
            raise FileNotFoundError(f"Trained model not found at {run_output_dir}. Did Phase 2 finish?")

        currnt_model, current_defs = load_SETA_model_from_disk(
            run_output_dir, base_model_class=None,
            base_model_name_or_path=SETA_model_args.model_name_or_path,
            block_dim=SETA_model_args.block_dimension)
        SETA_model = SETAModel(
            SETA_model_args, None, SETA_config_name, task_id=task_id, save_path=save_path_FFT,
                expert_creation_threshold=SETA_model_args.expert_creation_threshold,
                pruning_min_blocks=SETA_model_args.pruning_min_blocks,
                tiny_remainder_threshold=SETA_model_args.tiny_remainder_threshold,
                pretrained_moe_model=currnt_model, pretrained_definitions=current_defs,
                block_dimension=SETA_model_args.block_dimension,
                no_expert_sharing=SETA_model_args.no_expert_sharing
        )
        if 'base_model' in locals() and base_model is not None:
            del base_model; gc.collect()

        current_vocab = SETA_model.model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > current_vocab:
            SETA_model.model.resize_token_embeddings(len(tokenizer))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        eval_model = SETA_model.model.to(device)

        # Deep-convert ALL tensors to fp32 — including plain tensor attributes inside
        # custom SequentialMoELayer that are not registered as nn.Parameter or buffers
        # and are therefore missed by .parameters(), .buffers(), and .to()
        def force_fp32_all_tensors(module: torch.nn.Module) -> None:
            # 1. Registered parameters and buffers
            for param in module.parameters():
                param.data = param.data.float()
            for buf in module.buffers():
                buf.data = buf.data.float()
            # 2. Plain tensor attributes on the module itself (e.g. expert_weight, weight_fp16)
            for attr_name in list(vars(module).keys()):
                attr = getattr(module, attr_name, None)
                if isinstance(attr, torch.Tensor) and attr.is_floating_point():
                    setattr(module, attr_name, attr.float())
                elif isinstance(attr, (list, tuple)):
                    new_list = []
                    for item in attr:
                        if isinstance(item, torch.Tensor) and item.is_floating_point():
                            new_list.append(item.float())
                        else:
                            new_list.append(item)
                    try:
                        setattr(module, attr_name, type(attr)(new_list))
                    except Exception:
                        pass
                elif isinstance(attr, dict):
                    for k, v in attr.items():
                        if isinstance(v, torch.Tensor) and v.is_floating_point():
                            attr[k] = v.float()
            # 3. Recurse into child modules
            for child in module.children():
                force_fp32_all_tensors(child)

        force_fp32_all_tensors(eval_model)

        # Sanitize NaN/Inf that survive fp16→fp32 conversion
        nan_count = 0
        for name, param in eval_model.named_parameters():
            if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                nan_count += 1
                param.data = torch.nan_to_num(param.data, nan=0.0, posinf=1e4, neginf=-1e4)
        print(f"  [Weight Health] Fixed NaN/Inf in {nan_count} parameter tensors.")
        print(f"  [Eval] Model dtype: {next(eval_model.parameters()).dtype}")
        eval_model.eval()

    
        # Ensure eval_task_id is read as an integer
        # eval_id = int(task_ctrl_args.eval_task_id) if task_ctrl_args.eval_task_id else None

        # EVAL_TASK_IDS (colon- or comma-separated) scores several datasets from
        # the one loaded model. Falls back to the single --eval_task_id.
        _ids_env = os.environ.get("EVAL_TASK_IDS", "").strip()
        if _ids_env:
            eval_ids = [int(x) for x in re.split(r"[,:]", _ids_env) if x.strip()]
        elif task_ctrl_args.eval_task_id:
            eval_ids = [int(task_ctrl_args.eval_task_id)]
        else:
            eval_ids = None
        
        print(f"=====================================================")
        print(f"Model trained up to Task ID: {task_id}")
        
        for i, target_task_name in enumerate(data_args.task_list):
            target_id = i + 1
            
            # If the bash script provided specific eval ids, ONLY evaluate those
            if eval_ids is not None:
                if target_id not in eval_ids:
                    continue  # Skip all other datasets
            
            # Fallback logic just in case you run the script manually without eval_task_id
            else:
                if target_id > task_id: 
                    break 
                if task_ctrl_args.eval_current_task_only and target_id != task_id:
                    continue

            # If it reaches here, it is the correct dataset to evaluate
            print(f"--> Starting Evaluation on Dataset {target_id}: {target_task_name}")

            print(f"--- Evaluating Task {target_id}: {target_task_name} ---")

            # [ROUTER-EVAL] inference-time routing; the trained weights are unchanged.
            #   EVAL_ROUTING_MODE = learned | uniform | oracle | topk   (unset = learned)
            #   EVAL_ROUTING_TOPK = k        (topk mode; setting it alone also selects topk)
            #   EVAL_UNIFORM_VALUE = g       (uniform mode; unset = every gate 1.0 under sigmoid)
            # oracle: only experts whose name holds the evaluated task id keep their
            # learned gate, all others get gate 0; a layer with no such expert runs
            # the base weight only.
            _topk  = os.environ.get("EVAL_ROUTING_TOPK", "").strip()
            _rmode = os.environ.get("EVAL_ROUTING_MODE", "").strip().lower()
            _explicit = bool(_rmode or _topk)
            if not _rmode:
                _rmode = "topk" if _topk else "learned"
            if _rmode not in ("learned", "uniform", "oracle", "topk"):
                raise ValueError(f"EVAL_ROUTING_MODE must be learned|uniform|oracle|topk (got {_rmode!r})")
            if _rmode == "topk" and not _topk:
                raise ValueError("EVAL_ROUTING_MODE=topk needs EVAL_ROUTING_TOPK=<k>")
            _uval = os.environ.get("EVAL_UNIFORM_VALUE", "").strip()
            _n_layers = _n_empty = 0
            for _m in eval_model.modules():
                if hasattr(_m, "gating_network") and hasattr(_m, "experts"):
                    _n_layers += 1
                    _m.routing_mode = _rmode
                    _m.routing_topk = int(_topk) if _rmode == "topk" else None
                    _m.uniform_value = float(_uval) if (_rmode == "uniform" and _uval) else None
                    _m.oracle_allowed_idx = None
                    if _rmode == "oracle":
                        _allowed = [_i for _i, _k in enumerate(_m.experts.keys())
                                    if target_id in expert_task_ids(_k)]
                        if _allowed:
                            _m.oracle_allowed_idx = _allowed
                        else:
                            _m.routing_mode = "none"
                            _n_empty += 1
            if _rmode == "oracle":
                print(f"  [routing] oracle, task {target_id}: {_n_layers} MoE layer(s), "
                      f"{_n_empty} without an expert for this task (base weight only)")
            else:
                print(f"  [routing] {_rmode}"
                      + (f" k={_topk}" if _rmode == "topk" else "")
                      + (f" value={_uval or 'default'}" if _rmode == "uniform" else "")
                      + f" on {_n_layers} MoE layer(s)")

            # Each routing variant writes to its own folder; nothing existing is
            # overwritten. No routing variable set = the original folder name.
            _sfx = os.environ.get("EVAL_OUT_SUFFIX", "").strip()
            if not _sfx and _explicit:
                _sfx = f"_top{_topk}" if _rmode == "topk" else f"_{_rmode}"
            eval_output_dir = os.path.join(run_output_dir, f"eval_task_{target_id}_{target_task_name}{_sfx}")
            print(f"  [routing] output: {eval_output_dir}")

            torch.cuda.empty_cache()


            evaluate_trace_benchmark(
                SETAtrainer=None, model=eval_model, tokenizer=tokenizer,
                data_path=data_args.base_dataset_path, datasets=[target_task_name],
                output_dir=eval_output_dir, max_samples=data_args.max_samples,
                per_device_eval_batch_size=training_args.per_device_eval_batch_size,
                dtype='fp32', use_eval_split=False
            )
            gc.collect(); 
            torch.cuda.empty_cache()

    print("\nRun Complete.")

def save_gradients(gradients, grad_name, SETA_config_name, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    gradient_file = os.path.join(output_dir, grad_name, f"{SETA_config_name }_gradients.pt")
    folder = os.path.join(output_dir, grad_name)
    os.makedirs(folder, exist_ok=True)
    torch.save(gradients, gradient_file)
    print(f'gradient_file: {gradient_file} saved.')

if __name__ == "__main__":
    main()
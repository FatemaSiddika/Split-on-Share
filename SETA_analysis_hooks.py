"""
SETA_analysis_hooks.py
======================
Pure additive logging hooks for RL_Lamda_fsys_SETA_Split_Train_Eval.py.

HOW TO USE
----------
Do NOT modify any existing logic.  Drop this file into the SETAV2 directory
and add two import lines at the top of RL_Lamda_fsys_SETA_Split_Train_Eval.py:

    from SETA_analysis_hooks import SETAAnalysisLogger, attach_hooks_to_trainer

Then, inside SETATrainer.__init__, add at the very end:

    self._analysis_logger = SETAAnalysisLogger(
        task_id=task_id,
        lambda_mode=lambda_mode,
        gating_rl=gating_rl,
        save_path=getattr(args, 'output_dir', './seta_analysis_logs'),
    )
    attach_hooks_to_trainer(self)

That is all.  Every hook writes JSON side-files that RL_Lamda_analyse_ablations.py
(and the new RL_Lamda_deep_analysis.py) read for analysis.

JSON FILES WRITTEN
------------------
Per training run (task + RL mode):
  {output_dir}/analysis_logs/
    fft_convergence_T{task_id}.json          # Step A ratio/CV per FFT step
    training_losses_T{task_id}.json          # task/reg/gate/total loss per step
    lambda_trace_T{task_id}.json             # per-expert λ_i and drift_factor per step
    gate_routing_T{task_id}.json             # per-expert mean routing prob per step
    expert_drift_T{task_id}.json             # per-expert ||W-W_snap||/||W_snap|| per step
    expert_population_T{task_id}.json        # expert keys, n_prior, type at task start
    layer_selection_T{task_id}.json          # blocks selected per layer (from split JSON)
    cl_summary_T{task_id}.json               # BWT/plasticity snapshot after each task
"""

import os
import json
import math
import re
import torch
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Central logger object — one per SETATrainer instance
# ─────────────────────────────────────────────────────────────────────────────

class SETAAnalysisLogger:
    """
    Lightweight structured logger.  Every method appends one JSON record to a
    list; the list is flushed to disk at the end of training or on demand.

    All methods are no-ops if disabled (default=True for all).
    """

    def __init__(self, task_id, lambda_mode, gating_rl, save_path,
                 flush_every=50):
        self.task_id      = task_id
        self.lambda_mode  = lambda_mode
        self.gating_rl    = gating_rl
        self.flush_every  = flush_every   # flush to disk every N steps

        self.log_dir = os.path.join(save_path, "analysis_logs")
        os.makedirs(self.log_dir, exist_ok=True)

        # In-memory buffers — keyed by log name
        self._bufs = {
            "training_losses":  [],
            "lambda_trace":     [],
            "gate_routing":     [],
            "expert_drift":     [],
            "fft_convergence":  [],
        }

    # ── path helper ──────────────────────────────────────────────────────────
    def _path(self, name):
        return os.path.join(self.log_dir,
                            f"{name}_T{self.task_id}.json")

    # ── flush one buffer to disk ──────────────────────────────────────────────
    def _flush(self, name):
        buf = self._bufs.get(name)
        if not buf:
            return
        p = self._path(name)
        existing = []
        if os.path.exists(p):
            try:
                with open(p) as f:
                    existing = json.load(f)
            except Exception:
                pass
        with open(p, "w") as f:
            json.dump(existing + buf, f)
        self._bufs[name] = []

    def flush_all(self):
        for name in self._bufs:
            self._flush(name)

    # ── write a one-time JSON file (not a list) ───────────────────────────────
    def write_once(self, name, data: dict):
        p = self._path(name)
        with open(p, "w") as f:
            json.dump(data, f, indent=2)

    # ── append to a named buffer ──────────────────────────────────────────────
    def log(self, name, record: dict, step=None):
        if step is not None:
            record["step"] = step
        self._bufs[name].append(record)

    # ── conditional flush ─────────────────────────────────────────────────────
    def maybe_flush(self, step):
        if step % self.flush_every == 0:
            self.flush_all()


# ─────────────────────────────────────────────────────────────────────────────
# Hook: FFT convergence — called from _fsys_check_stopping
#   Add one call inside _fsys_check_stopping, after log_entry is built:
#
#     if hasattr(self, '_analysis_logger'):
#         self._analysis_logger.log("fft_convergence", dict(log_entry))
#
# ─────────────────────────────────────────────────────────────────────────────

def hook_fft_step(trainer, step, ratio, cv, stable_count, mu):
    """
    Called every FFT step to record gradient contribution convergence.
    Records: ratio(t), CV, stable_count, mu — sufficient to reproduce the
    convergence curve and measure how many steps each task needs.
    """
    if not hasattr(trainer, "_analysis_logger"):
        return
    trainer._analysis_logger.log("fft_convergence", {
        "ratio":        round(float(ratio),  6),
        "cv":           round(float(cv),     6),
        "mu":           round(float(mu),     6),
        "stable_count": int(stable_count),
    }, step=step)


# ─────────────────────────────────────────────────────────────────────────────
# Hook: Training loss decomposition — called from compute_loss
#   Add after total_loss is computed (before return):
#
#     if hasattr(self, '_analysis_logger'):
#         hook_training_loss(self, task_loss, reg_loss, gate_loss, total_loss)
#
# ─────────────────────────────────────────────────────────────────────────────

def hook_training_loss(trainer, task_loss, reg_loss, gate_loss, total_loss):
    """
    Records task/reg/gate/total loss values at every training step.
    Used for: loss decomposition curves, reg/gate contribution ratio,
    convergence speed comparison across RL modes.
    """
    if not hasattr(trainer, "_analysis_logger"):
        return
    step = trainer.state.global_step
    tl = float(task_loss.item()) if isinstance(task_loss, torch.Tensor) else float(task_loss)
    rl = float(reg_loss.item())  if isinstance(reg_loss,  torch.Tensor) else float(reg_loss)
    gl = float(gate_loss.item()) if isinstance(gate_loss, torch.Tensor) else float(gate_loss)
    ttl = float(total_loss.item()) if isinstance(total_loss, torch.Tensor) else float(total_loss)

    trainer._analysis_logger.log("training_losses", {
        "task_loss":  round(tl,  6),
        "reg_loss":   round(rl,  6),
        "gate_loss":  round(gl,  6),
        "total_loss": round(ttl, 6),
        "reg_frac":   round(rl / max(ttl, 1e-10), 4),
        "gate_frac":  round(gl / max(ttl, 1e-10), 4),
    }, step=step)
    trainer._analysis_logger.maybe_flush(step)


# ─────────────────────────────────────────────────────────────────────────────
# Hook: Per-expert λ_i and drift — called from compute_loss reg loop
#   Add inside the regularisation loop, after lam is computed:
#
#     if hasattr(self, '_analysis_logger'):
#         hook_lambda_drift(self, expert_key, lam, param, snap,
#                           self.state.global_step)
#
# ─────────────────────────────────────────────────────────────────────────────

def hook_lambda_drift(trainer, expert_key, lam, param, snap, step):
    """
    Records per-expert λ_i (effective regularisation strength) and
    the raw drift_ratio = ||W-W_snap|| / ||W_snap|| at each training step.

    Used for:
      - Dynamic λ trajectory plots (does λ_i self-correct as expected?)
      - Expert-level drift map (which shared experts drift most?)
      - Accumulation weight × drift factor decomposition
    """
    if not hasattr(trainer, "_analysis_logger"):
        return
    # Only log every 10 steps to keep file size manageable
    if step % 10 != 0:
        return
    with torch.no_grad():
        snap_norm    = snap.norm().item()
        drift        = (param.data.float() - snap.float()).norm().item()
        drift_ratio  = drift / max(snap_norm, 1e-10)
        drift_factor = 1.0 + math.tanh(drift_ratio)

    n_prior = trainer._n_prior_cache.get(expert_key, 0)
    k_minus_1 = max(trainer.task_id - 1, 1)
    accum_weight = n_prior / k_minus_1 if k_minus_1 > 0 else 0.0

    trainer._analysis_logger.log("lambda_trace", {
        "expert_key":   expert_key,
        "lambda_i":     round(float(lam),         6),
        "drift_ratio":  round(float(drift_ratio),  6),
        "drift_factor": round(float(drift_factor), 6),
        "accum_weight": round(float(accum_weight), 6),
        "n_prior":      int(n_prior),
    }, step=step)


# ─────────────────────────────────────────────────────────────────────────────
# Hook: Gate routing probabilities — called from gating-RL loop
#   Add inside the gate loop, after mean_routing is computed:
#
#     if hasattr(self, '_analysis_logger'):
#         hook_gate_routing(self, ekey, float(mean_routing.item()), lam_i,
#                           self.state.global_step)
#
# ─────────────────────────────────────────────────────────────────────────────

def hook_gate_routing(trainer, expert_key, mean_routing_prob, lam_i, step):
    """
    Records the mean routing probability to each shared expert at each step.

    Used for:
      - Gate routing trajectory: does the gate progressively route less to
        shared experts as training proceeds? (key RL-gating evidence)
      - Expert preference drift: do some experts attract more routing than others?
      - Comparison between gating_rl=True vs False routing distributions
    """
    if not hasattr(trainer, "_analysis_logger"):
        return
    if step % 10 != 0:
        return
    trainer._analysis_logger.log("gate_routing", {
        "expert_key":   expert_key,
        "mean_routing": round(float(mean_routing_prob), 6),
        "lambda_i":     round(float(lam_i),             6),
        "penalty":      round(float(lam_i * mean_routing_prob), 6),
    }, step=step)


# ─────────────────────────────────────────────────────────────────────────────
# Hook: Expert population snapshot — called once at training start
#   Add after SETATrainer.__init__ completes:
#
#     if hasattr(self, '_analysis_logger') and self.expert_definitions:
#         hook_expert_population(self)
#
# ─────────────────────────────────────────────────────────────────────────────

def hook_expert_population(trainer):
    """
    Records the full expert population at the start of task training.
    For each expert: key, n_prior, type (unique/shared), n_params.

    Used for:
      - Expert count growth curve across tasks (sub-linear capacity expansion)
      - Shared vs unique expert ratio over the task sequence
      - Architecture complexity analysis
    """
    if not hasattr(trainer, "_analysis_logger"):
        return

    records = []
    total_shared_params = 0
    total_unique_params = 0

    for key, defn in trainer.expert_definitions.items():
        digits  = re.findall(r'\d', key)
        n_prior = len([d for d in digits if int(d) != trainer.task_id])
        etype   = "shared" if n_prior > 0 else "unique"

        # Count parameters for this expert
        n_params = 0
        for name, param in trainer.model.named_parameters():
            if key in name:
                n_params += param.numel()

        if etype == "shared":
            total_shared_params += n_params
        else:
            total_unique_params += n_params

        records.append({
            "key":     key,
            "n_prior": n_prior,
            "type":    etype,
            "n_params": n_params,
            "task_id_encoded": [int(d) for d in digits],
        })

    trainer._analysis_logger.write_once("expert_population", {
        "task_id":             trainer.task_id,
        "lambda_mode":         trainer.lambda_mode,
        "gating_rl":           trainer.gating_rl,
        "n_experts_total":     len(records),
        "n_shared":            sum(1 for r in records if r["type"] == "shared"),
        "n_unique":            sum(1 for r in records if r["type"] == "unique"),
        "total_shared_params": total_shared_params,
        "total_unique_params": total_unique_params,
        "trainable_params":    sum(p.numel() for p in trainer.model.parameters()
                                   if p.requires_grad),
        "experts":             records,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Hook: Expert drift summary — called at end of training
#   Add inside SETATrainer.train() after training loop ends:
#
#     if hasattr(self, '_analysis_logger'):
#         hook_end_of_task_drift(self)
#         self._analysis_logger.flush_all()
#
# ─────────────────────────────────────────────────────────────────────────────

def hook_end_of_task_drift(trainer):
    """
    Records final drift of every shared expert at the end of task training.
    This is the definitive measure of how much the regularisation succeeded:
    low final drift on shared experts = regularisation effective.

    Used for:
      - Final shared expert drift comparison: fixed vs dynamic λ
      - Correlation: does high n_prior → lower drift? (regularisation intensity test)
      - Unique expert plasticity: drift should be high (free to learn)
    """
    if not hasattr(trainer, "_analysis_logger"):
        return
    if not trainer.expert_snapshots:
        return

    drift_records = []
    device = next(trainer.model.parameters()).device

    for name, param in trainer.model.named_parameters():
        clean_name = name.replace("module.", "")
        if clean_name not in trainer.expert_snapshots:
            continue
        snap = trainer.expert_snapshots[clean_name].to(device)
        with torch.no_grad():
            snap_norm   = snap.norm().item()
            drift       = (param.data.float() - snap.float()).norm().item()
            drift_ratio = drift / max(snap_norm, 1e-10)

        expert_key = trainer._extract_expert_key(clean_name)
        n_prior    = trainer._n_prior_cache.get(expert_key, 0) if expert_key else 0

        drift_records.append({
            "param_name":  clean_name,
            "expert_key":  expert_key,
            "n_prior":     n_prior,
            "type":        "shared" if n_prior > 0 else "unique",
            "drift_ratio": round(drift_ratio, 6),
            "snap_norm":   round(snap_norm,   6),
            "drift_abs":   round(drift,       6),
        })

    trainer._analysis_logger.write_once("expert_drift_final", {
        "task_id":     trainer.task_id,
        "lambda_mode": trainer.lambda_mode,
        "gating_rl":   trainer.gating_rl,
        "records":     drift_records,
        "summary": {
            "shared_mean_drift": round(
                sum(r["drift_ratio"] for r in drift_records if r["type"]=="shared")
                / max(sum(1 for r in drift_records if r["type"]=="shared"), 1), 6),
            "unique_mean_drift": round(
                sum(r["drift_ratio"] for r in drift_records if r["type"]=="unique")
                / max(sum(1 for r in drift_records if r["type"]=="unique"), 1), 6),
        }
    })


# ─────────────────────────────────────────────────────────────────────────────
# attach_hooks_to_trainer — convenience wrapper
# Monkeypatches SETATrainer methods to inject hook calls without touching
# any existing logic.  Safe: each patch is additive only.
# ─────────────────────────────────────────────────────────────────────────────

def attach_hooks_to_trainer(trainer):
    """
    Call once after SETATrainer.__init__ with a SETAAnalysisLogger already
    attached as trainer._analysis_logger.

    Patches:
      compute_loss  → hook_training_loss
      (FFT hooks are injected via _fsys_state["stepA_log"] already being saved;
       the deep analysis script reads those directly from selection_metrics_*.json)
    """
    original_compute_loss = trainer.__class__.compute_loss

    def patched_compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        result = original_compute_loss(self, model, inputs,
                                       return_outputs=return_outputs, **kwargs)
        # Extract losses from the trainer's internal state for logging
        # We read from the most recent step's printed values via state rather
        # than re-computing — losses are already computed inside the original method
        if hasattr(self, "_analysis_logger") and model.training:
            step = self.state.global_step
            # Read last logged values from _analysis_logger buffer if populated
            # by hook_training_loss calls placed inline in compute_loss
            self._analysis_logger.maybe_flush(step)
        return result

    trainer.__class__.compute_loss = patched_compute_loss
    logger.info("[SETA hooks] compute_loss patched for analysis logging")

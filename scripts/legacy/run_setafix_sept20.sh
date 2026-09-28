#!/bin/bash

set -euo pipefail

# ── Arguments ─────────────────────────────────────────────────────────────────
TASK_RANGE="${1:-1}"
STEPS_TO_RUN="${2:-fft,split,train,eval}"
user_model_name="${3:-qwen3_4b}"

export FIRST_JOB_DEP="${FIRST_JOB_DEP:-${FIRST_JOB_DE:-}}"

# ── [SETAFIX] ────────────────────────────────────────────────────────────────
#   STRUCTURE_ONLY=1  : run only fft+split per task, chain tasks through the
#                       split checkpoint (init_structure_checkpoint), no
#                       training/eval -> the 8-task expert structure in hours.
#   SETAFIX_SPLIT     : seam (direction-aware split, default) | off (original)
#   SETAFIX_SCORE     : svd (subspace agreement, default) | fisher (|grad| pattern)
#   SETAFIX_Z         : chance-floor z-score for "aligned" (default 3.0)
#   SETAFIX_MIN_SHARED_FRAC : relative size a shared child must reach (0.15)
#   SETAFIX_NOEVIDENCE: share | reject when no direction evidence exists (share)
export STRUCTURE_ONLY="${STRUCTURE_ONLY:-0}"
export SETAFIX_STRUCTURE_ONLY="${STRUCTURE_ONLY}"
#   SETAFIX_PHI       : one knob — min share of the NEW task's selection a shared
#                       child must carry (0.025). Defaults MIN_SHARED_FRAC_NEW
#                       (whole-expert creation floor) and LAYER_FRAC (per-layer m).
export SETAFIX_PHI="${SETAFIX_PHI:-0.025}"
export SETAFIX_SPLIT="${SETAFIX_SPLIT:-seam}"
export SETAFIX_SCORE="${SETAFIX_SCORE:-svd}"
export SETAFIX_Z="${SETAFIX_Z:-3.0}"
export SETAFIX_MIN_SHARED_FRAC="${SETAFIX_MIN_SHARED_FRAC:-0.15}"
export SETAFIX_NOEVIDENCE="${SETAFIX_NOEVIDENCE:-share}"
#   SETAFIX_SIDE      : right | left | both   — which singular subspaces to compare (right)
#   SETAFIX_NULL      : gauss | empirical     — chance bar from random tiles, or from the new
#                       task's tiles on OTHER blocks of the same layer (gauss)
#   SETAFIX_ORTHO     : on | off  — train own-copy blocks orthogonally to the old expert (off)
#   SETAFIX_GATE_PROTECT : freeze | off — freeze prior-era gate rows during training (off)
export SETAFIX_SIDE="${SETAFIX_SIDE:-right}"
export SETAFIX_NULL="${SETAFIX_NULL:-gauss}"
export SETAFIX_ORTHO="${SETAFIX_ORTHO:-off}"
export SETAFIX_GATE_PROTECT="${SETAFIX_GATE_PROTECT:-off}"
#   --- density controls (bind at large block budgets, where the direction bar
#       passes almost every overlap and the split degenerates to positional) ---
#   SETAFIX_SHARE_Q            keep only the best q fraction of each overlap
#                              group as shared; rest -> own copy      (1.0 = off)
#   SETAFIX_MIN_SHARED_FRAC_NEW  a shared child also needs this fraction of the
#                              NEW task's selection                   (0.0 = off)
#   SETAFIX_MAX_SHARED_PER_TASK  hard cap on shared children per task (0 = off)
#   suggested for 25% budgets:  SHARE_Q=0.5  MIN_SHARED_FRAC_NEW=0.10  MAX=3
export SETAFIX_SHARE_Q="${SETAFIX_SHARE_Q:-1.0}"
export SETAFIX_MIN_SHARED_FRAC_NEW="${SETAFIX_MIN_SHARED_FRAC_NEW:-${SETAFIX_PHI}}"
export SETAFIX_MAX_SHARED_PER_TASK="${SETAFIX_MAX_SHARED_PER_TASK:-0}"
#   --- per-layer gate (replaces the absolute EXPERT_CREATION_THRESHOLD floor
#       once the direction test is on; see SETA_SoS) ---
#   SETAFIX_LAYER_GATE  phi (default) | relative (Sept 18 behaviour) | absolute | off
#       phi: m = PHI x new task's blocks in the layer, straight, no rounding;
#            carve iff aligned >= m and leftover is 0 or >= m; tiny-remainder
#            merge off. LAYER_FRAC / LAYER_MIN are used by relative mode only.
#   SETAFIX_LAYER_FRAC  fraction of the layer's pre-validation overlap (relative)
#   SETAFIX_LAYER_MIN   hard minimum, so a 2-block layer cannot split (relative)
export SETAFIX_LAYER_GATE="${SETAFIX_LAYER_GATE:-phi}"
export SETAFIX_LAYER_FRAC="${SETAFIX_LAYER_FRAC:-0.25}"
export SETAFIX_LAYER_MIN="${SETAFIX_LAYER_MIN:-3}"
if [ "${STRUCTURE_ONLY}" == "1" ] && [ -z "${2:-}" ]; then
    STEPS_TO_RUN="fft,split"
fi
if [ "${STRUCTURE_ONLY}" == "1" ]; then
    case "${STEPS_TO_RUN}" in *train*|*eval*)
        echo "STRUCTURE_ONLY=1 but STEPS_TO_RUN='${STEPS_TO_RUN}' contains train/eval — use fft,split" >&2; exit 1;;
    esac
fi

# Note: MAX_SAMPLES is now dynamically assigned per task inside submit_task()
BASE_DATA_PATH="/lustre/hdd/LAS/jannesar-lab/fatemask/SETAV2/data_files/TRACE-Benchmark/LLM-CL-Benchmark_5000"
PROJECT_ROOT="/lustre/hdd/LAS/jannesar-lab/fatemask/fed-con/SETA_fix"
# [PIN] One code location: SETAV2 root. No directory variable, no fallback.
TRAINER_PY="${PROJECT_ROOT}/main.py"
MODEL_PY="${PROJECT_ROOT}/SETA_SoS"
for _f in "$TRAINER_PY" "$MODEL_PY"; do
    if [ ! -f "$_f" ]; then
        echo "ERROR: required Aug20 file missing: $_f" >&2
        exit 1
    fi
done
if ! grep -q "from RL_Lamda_dyn_SETA_SoS_Aug20 import" "$TRAINER_PY"; then
    echo "ERROR: $TRAINER_PY does not import RL_Lamda_dyn_SETA_SoS_Aug20 -- stale trainer copy." >&2
    exit 1
fi
if ! grep -q "def forward(self, x, gate=None):" "$MODEL_PY"; then
    echo "ERROR: $MODEL_PY lacks the block-sparse forward -- stale model copy (dense linearZ path = OOM at T8+)." >&2
    exit 1
fi
echo "[PIN] trainer   : $(sha1sum "$TRAINER_PY")"
echo "[PIN] model     : $(sha1sum "$MODEL_PY")"
# Slurm script sits next to this launcher, wherever it was invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/setafix_pipeline.slurm"
if [ ! -f "$SLURM_SCRIPT" ]; then
    echo "ERROR: slurm script not found at $SLURM_SCRIPT" >&2
    exit 1
fi
# BASE_DIR is set AFTER the case block below (needs ABLATION_TAG to be defined first)

save_job_id() { echo "$3" > "${JOB_ID_DIR}/task_${1}_${2}.txt"; }
load_job_id() {
    local f="${JOB_ID_DIR}/task_${1}_${2}.txt"
    [ -f "$f" ] && cat "$f" || echo ""
}

EVAL_ROUTING_TOPK="${EVAL_ROUTING_TOPK:-}"

# ── S1–S4 dynamic parameters (ACTIVATED defaults) ─────────────────────────────
PROJECTION_MODE="${PROJECTION_MODE:-v_proj_only}"
# [TAG SAFETY] PROJECTION_MODE lands in the run directory name, and the .slurm
# case block falls through to v_proj on anything unrecognised. Without this
# guard a typo silently starts a fresh, identical-config tree under a new name.
case "$PROJECTION_MODE" in
    v_proj_only|q_proj|k_proj|qkv_proj|mlp_proj) ;;
    v_proj) PROJECTION_MODE="v_proj_only" ;;   # alias -> canonical
    *) echo "ERROR: PROJECTION_MODE must be one of v_proj_only|q_proj|k_proj|qkv_proj|mlp_proj (got '${PROJECTION_MODE}')" >&2; exit 1 ;;
esac
PROJ_CALIB_THRESHOLD="${PROJ_CALIB_THRESHOLD:-0.10}"
PROJ_CALIB_STEPS="${PROJ_CALIB_STEPS:-20}"
BLOCK_BUDGET_FRACTION="${BLOCK_BUDGET_FRACTION:-0.12}"     # 0.20 × 8192 ≈ 1638 blocks; more unique capacity for T4-T6
GRAD_TEMPORAL_DECAY="${GRAD_TEMPORAL_DECAY:-1.0}"           # [was: 0.0 → uniform]
GLOBAL_EXPERT_MIN_BLOCKS="${GLOBAL_EXPERT_MIN_BLOCKS:-20}"  # [was: 15]
EXPERT_CREATION_THRESHOLD="${EXPERT_CREATION_THRESHOLD:-8}" # [was: 5]
ADAPTIVE_FFT_STEPS="${ADAPTIVE_FFT_STEPS:-True}"             # [was: False]
FFT_MIN_STEPS="${FFT_MIN_STEPS:-30}"
FFT_MAX_STEPS="${FFT_MAX_STEPS:-150}"
FFT_REF_SAMPLES="${FFT_REF_SAMPLES:-500}"
NOVELTY_BETA_MIN="${NOVELTY_BETA_MIN:-0.30}"
NOVELTY_T_MAX="${NOVELTY_T_MAX:-6}"    # [was: 12 — corrected to sequence length]
train_sample_ratio="${train_sample_ratio:-1}"

if [ -z "${ABLATION:-}" ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  WARNING: ABLATION not set — defaulting to ABLATION=0       ║"
    echo "║                                                              ║"
    echo "║    ABLATION=0  original_seta  (fixed FFT + fixed blocks)     ║"
    echo "║    ABLATION=1  full_system    (parameter-free dynamic)       ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
fi
ABLATION="${ABLATION:-0}"

if [ -n "${FIRST_JOB_DEP:-}" ]; then
    echo "  [dep] T1 first step will wait for external job: ${FIRST_JOB_DEP}"
fi

case "$ABLATION" in
    0)  ABLATION_TAG="original_seta"
        FFT_STOPPING_MODE="fixed"
        LAYER_NORM_SCORING="False"
        DIVERGENCE_SCORING="True"
        BUDGET_MODE="legacy"
        # Force original SETA values
        BLOCK_BUDGET_FRACTION="0.0"
        ADAPTIVE_FFT_STEPS="False"
        NOVELTY_BETA_MIN="0.0"
        GRAD_TEMPORAL_DECAY="0.0"
        GLOBAL_EXPERT_MIN_BLOCKS="15"
        EXPERT_CREATION_THRESHOLD="8"
        ;;
    1)  ABLATION_TAG="full_system"
        FFT_STOPPING_MODE="dynamic"
        LAYER_NORM_SCORING="True"
        DIVERGENCE_SCORING="True"
        BUDGET_MODE="adaptive"
        ;;
    2)  ABLATION_TAG="no_sharing"
        FFT_STOPPING_MODE="dynamic"
        LAYER_NORM_SCORING="True"
        DIVERGENCE_SCORING="True"
        BUDGET_MODE="adaptive"
        NO_EXPERT_SHARING="True"
        ;;
    *)  echo "Error: ABLATION must be 0, 1, or 2" >&2; exit 1 ;;
esac

echo "┌──────────────────────────────────────────────────────────────┐"
echo "│  ABLATION=${ABLATION} — ${ABLATION_TAG}"
printf "│  %-62s│\n" "fft_stopping_mode  = ${FFT_STOPPING_MODE}"
printf "│  %-62s│\n" "layer_norm_scoring = ${LAYER_NORM_SCORING}"
printf "│  %-62s│\n" "divergence_scoring = ${DIVERGENCE_SCORING}"
printf "│  %-62s│\n" "budget_mode        = ${BUDGET_MODE}"
printf "│  %-62s│\n" "block_budget_frac  = ${BLOCK_BUDGET_FRACTION} (LoRA-rank equivalent)"
echo "└──────────────────────────────────────────────────────────────┘"

LAMDA="${LAMDA:-0.1}"          # λ base coefficient (0.0 = disabled)

LAMBDA_MODE="${LAMBDA_MODE:-dynamic}"   # 'fixed' | 'dynamic'
GATING_RL="${GATING_RL:-False}"      # [FIX-7b] default flipped True -> False.
                                     # Once --use_reentrant was actually honoured
                                     # this penalty became functional, and under
                                     # sigmoid gating it is ~n_experts larger than
                                     # it was under softmax. Measured Qwen3 T2:
                                     # gate_loss=3.32 vs task_loss=1.02, i.e. 76%
                                     # of the objective. Set GATING_RL=True to
                                     # re-enable (ablation arm).

# ── Fix-control defaults (forwarded to the job via NEW_PARAMS / --export) ─────
# Override on the command line, e.g.
#   GATE_MODE=softmax bash <script> 1-6 fft,split,train,eval qwen3_4b
GATE_MODE="${GATE_MODE:-sigmoid}"                       # [FIX-1] sigmoid | softmax
GATE_OPEN_BIAS="${GATE_OPEN_BIAS:-1.0}"                 # [FIX-1]
USE_REENTRANT="${USE_REENTRANT:-False}"                 # [FIX-7]
BUDGET_EXPANSION_CAP="${BUDGET_EXPANSION_CAP:-1.5}"     # [FIX-6] 1.0 = never expand
LABEL_AUDIT_STRICT="${LABEL_AUDIT_STRICT:-True}"        # [FIX-2]
LABEL_AUDIT_SAMPLES="${LABEL_AUDIT_SAMPLES:-512}"       # [FIX-2]
LOSS_SANITY_OFF="${LOSS_SANITY_OFF:-0}"                 # [FIX-2]

# ── [DESIGN-1] Router regularisation arm ──────────────────────────────────────
# Forgetting in SETA is a ROUTING failure, not weight overwriting: measured
# shared-expert drift is ~0.1% (reg_loss/task_loss ~1e-3) while prior-task
# accuracy collapses.  ROUTER_REG selects how the gate rows of experts owned by
# earlier tasks are protected:
#
#   none          all gate rows trainable (original behaviour)
#   penalty       + legacy gating_rl sparsity penalty on shared experts
#   freeze        prior-task gate rows receive zero gradient (exact, free)
#   anchor        prior rows stay trainable, pulled toward stored centroid
#                 targets (needs SETA_router_anchor.py)
#   anchor_sparse anchor + sparsity penalty
#
# Each arm writes to its OWN directory tree (see PHASE2_TAG below), so arms can
# run concurrently without touching each other's checkpoints or job-id files.
ROUTER_REG="${ROUTER_REG:-none}"
ROUTER_ANCHOR_MU="${ROUTER_ANCHOR_MU:-1.0}"       # anchor strength
ROUTER_N_CENTROIDS="${ROUTER_N_CENTROIDS:-8}"     # centroids per layer per task
ROUTER_CENTROID_BATCHES="${ROUTER_CENTROID_BATCHES:-16}"
# Phase 0 has no MoE, so arms share FFT block selections by default
# (identical selections = controlled comparison, one FFT not five).
# Set FFT_TAG=<name> to give an arm its own FFT directory.
FFT_TAG="${FFT_TAG:-}"

# ── [DESIGN-2] Expert recycling arms ──────────────────────────────────────────
#   EXCLUDE_MODE          hard | soft   (soft: score x alpha^use_count, no ban)
#   SOFT_EXCLUDE_ALPHA    decay base for soft mode
#   CAPACITY_FLOOR        0 = off; 0.6 = never erode an expert below 60% of
#                         its birth block count (ledger in save_path_FFT)
#   MERGE_LOW_UTIL        1 = fold low-utility experts into lineage siblings
#                         at the start of every Phase-1 split
#   MERGE_GATE_THRESHOLD  mean sigmoid gate below which an expert is a candidate
#   MERGE_CAND_MAX_BLOCKS candidates must be at most this many blocks (0 = any)
#   MERGE_MAX_LAYERS      structural fallback: single-layer experts merge even
#                         without routing stats
#   MERGE_PROTECT_LATEST  1 = never merge the most recent task's experts
# NOTE: EXCLUDE_MODE changes Phase-0 selection, so soft arms MUST run their own
# fft step (they do — ABLATION_TAG carries the arm tag).
EXCLUDE_MODE="${EXCLUDE_MODE:-hard}"
SOFT_EXCLUDE_ALPHA="${SOFT_EXCLUDE_ALPHA:-0.5}"
CAPACITY_FLOOR="${CAPACITY_FLOOR:-0}"
MERGE_LOW_UTIL="${MERGE_LOW_UTIL:-0}"
MERGE_GATE_THRESHOLD="${MERGE_GATE_THRESHOLD:-0.15}"
MERGE_CAND_MAX_BLOCKS="${MERGE_CAND_MAX_BLOCKS:-64}"
MERGE_MAX_LAYERS="${MERGE_MAX_LAYERS:-1}"
MERGE_PROTECT_LATEST="${MERGE_PROTECT_LATEST:-1}"
# Utility criterion.  'relative' (default) = mean - Z*sd of the live experts.
# 'absolute' = the old fixed MERGE_GATE_THRESHOLD cut, which fires on NOTHING
# under sigmoid+open_bias (all gates ~0.982) and makes the merge arm a no-op.
MERGE_UTIL_MODE="${MERGE_UTIL_MODE:-relative}"
MERGE_UTIL_Z="${MERGE_UTIL_Z:-1.0}"
MERGE_UTIL_MIN_SPREAD="${MERGE_UTIL_MIN_SPREAD:-0.01}"
# Eval-time routing override: learned | oracle | uniform | random | none
EVAL_ROUTING_MODE="${EVAL_ROUTING_MODE:-}"
EVAL_OUT_SUFFIX="${EVAL_OUT_SUFFIX:-}"
# [GATE PROBE] 1 = write routing_stats_task_<J>.json during eval (default off)
GATE_PROBE="${GATE_PROBE:-0}"

# [SEED] 100 reproduces every existing run. Any other value gets its own tree,
# so seeds never overwrite each other and can run side by side.
SEED="${SEED:-100}"
if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SEED must be an integer (got '${SEED}')" >&2; exit 1
fi

# [FINAL EVAL] True = after the last task, submit the full R matrix (the final
# checkpoint evaluated on every task in the range) by chaining the eval-only
# launcher onto the last job of this run. Off by default; the per-task evals
# already submitted are unaffected either way.
FINAL_EVAL="${FINAL_EVAL:-False}"
case "$MERGE_UTIL_MODE" in
    relative|absolute) ;;
    *) echo "ERROR: MERGE_UTIL_MODE must be 'relative' or 'absolute'" >&2; exit 1 ;;
esac
case "$EXCLUDE_MODE" in
    hard|soft) ;;
    *) echo "ERROR: EXCLUDE_MODE must be 'hard' or 'soft' (got '${EXCLUDE_MODE}')" >&2; exit 1 ;;
esac

echo "┌──────────────────────── FIX-CONTROL SETTINGS ────────────────────────┐"
printf "│  %-68s│\n" "GATE_MODE            = ${GATE_MODE} (open_bias=${GATE_OPEN_BIAS})"
printf "│  %-68s│\n" "USE_REENTRANT        = ${USE_REENTRANT}"
printf "│  %-68s│\n" "BUDGET_EXPANSION_CAP = ${BUDGET_EXPANSION_CAP}"
printf "│  %-68s│\n" "LABEL_AUDIT_STRICT   = ${LABEL_AUDIT_STRICT}"
printf "│  %-68s│\n" "ROUTER_REG           = ${ROUTER_REG} (mu=${ROUTER_ANCHOR_MU})"
printf "│  %-68s│\n" "RECYCLE              = ex${EXCLUDE_MODE} cf${CAPACITY_FLOOR} merge${MERGE_LOW_UTIL}"
printf "│  %-68s│\n" "DATASET              = $(basename "${BASE_DATA_PATH}")"
printf "│  %-68s│\n" "MERGE_UTIL           = ${MERGE_UTIL_MODE} (z=${MERGE_UTIL_Z})"
printf "│  %-68s│\n" "SEED / LAMDA         = ${SEED} / ${LAMDA} (${LAMBDA_MODE})"
[[ -n "${EVAL_ROUTING_MODE}" ]] && printf "│  %-68s│\n" "EVAL_ROUTING_MODE    = ${EVAL_ROUTING_MODE}"
printf "│  %-68s│\n" "RUN DIR              = ablation_${ABLATION_TAG}"
echo "└──────────────────────────────────────────────────────────────────────┘"

# ── [PIPELINE] Chain granularity ──────────────────────────────────────────────
# False (default): Task N+1 split depends on Task N TRAIN; Task N eval runs in
#                  parallel. Uses up to 2 GPUs at a time; ~30-40% wall-clock
#                  saving on a 6-task stream.
# True           : fully serial (original behaviour), 1 GPU at a time.
EVAL_BLOCKS_CHAIN="${EVAL_BLOCKS_CHAIN:-False}"

# ── [SEQUENTIAL] One-job-at-a-time mode ───────────────────────────────────────
# SUBMIT_MODE=chain      (default) submit every job up front with SLURM
#                        dependencies. Fast to launch, but 5 ablation arms x
#                        ~19 jobs = 95+ pending jobs in the queue at once.
# SUBMIT_MODE=sequential submit ONE job, block until it finishes, then submit
#                        the next. Intended for one tmux window per ablation
#                        arm: each arm keeps at most 1-2 jobs queued, so 5 arms
#                        means ~5-10 queued jobs instead of 95+.
#
# In sequential mode the eval job is fired and NOT waited on by default, so it
# runs alongside the next task's split (same pipelining as chain mode). That is
# the "1-2": one training-chain job plus at most one detached eval.
# WAIT_FOR_EVAL=True makes it strictly one job at a time.
SUBMIT_MODE="${SUBMIT_MODE:-chain}"
WAIT_FOR_EVAL="${WAIT_FOR_EVAL:-False}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"       # seconds between squeue polls

case "$SUBMIT_MODE" in
    chain|sequential) ;;
    *) echo "ERROR: SUBMIT_MODE must be 'chain' or 'sequential' (got '${SUBMIT_MODE}')" >&2; exit 1 ;;
esac

# Block until $1 leaves the queue, then verify it succeeded.
# 0 on COMPLETED, 1 otherwise (caller aborts this arm).
wait_for_job() {
    local jid="$1" label="$2" waited=0 st state
    echo "        ... waiting for ${label} (job ${jid}) — polling every ${POLL_INTERVAL}s"
    while squeue -j "$jid" --noheader 2>/dev/null | grep -q .; do
        sleep "$POLL_INTERVAL"
        waited=$(( waited + POLL_INTERVAL ))
        if (( waited % 900 == 0 )); then
            st=$(squeue -j "$jid" --noheader -o "%T" 2>/dev/null | head -1)
            echo "        ... ${label} still ${st:-queued} after $(( waited / 60 )) min"
        fi
    done
    # squeue drops finished jobs; sacct holds the final state
    state=$(sacct -j "$jid" --noheader -o State -X 2>/dev/null | head -1 | tr -d ' ')
    if [[ "$state" == "COMPLETED" ]]; then
        echo "        [OK] ${label} COMPLETED after $(( waited / 60 )) min"
        return 0
    fi
    echo "" >&2
    echo "  [FAIL] ${label} (job ${jid}) ended in state '${state:-UNKNOWN}' — stopping this arm." >&2
    echo "         sacct -j ${jid} --format=JobID,JobName,State,ExitCode,Elapsed" >&2
    return 1
}

case "$ROUTER_REG" in
    none|penalty|freeze|anchor|anchor_sparse) ;;
    *) echo "ERROR: ROUTER_REG must be one of none|penalty|freeze|anchor|anchor_sparse (got '${ROUTER_REG}')" >&2; exit 1 ;;
esac

# ── Run-isolation tag ─────────────────────────────────────────────────────────
# Every knob that changes the trained artefact goes into the directory name, so
# concurrent ablation arms are fully separated: checkpoints, .job_ids, FFT
# selections and eval outputs. RUN_TAG appends a free-form suffix (e.g. seed).
PHASE2_TAG="RL_Lamda_${LAMBDA_MODE}_grl${GATING_RL}"
PHASE2_TAG="${PHASE2_TAG}_gate${GATE_MODE}_rr${ROUTER_REG}"
if [[ "$ROUTER_REG" == "anchor" || "$ROUTER_REG" == "anchor_sparse" ]]; then
    PHASE2_TAG="${PHASE2_TAG}mu${ROUTER_ANCHOR_MU}"
fi
# Recycling arms are folded into the tag so different arms never collide,
# even at all-default settings.
# [TAG] No _ex<mode> segment in this script: the exclusion mode is not varied
# here, so it only lengthened the name. The corpus size IS appended, because it
# is the one thing that distinguishes this run from the 500 sample run at
# identical router settings.
#     full_system_RL_Lamda_<mode>_grl<rl>_gate<gate>_rr<reg>_d5000
DATA_TAG="$(basename "${BASE_DATA_PATH}")"
case "$DATA_TAG" in
    *_500)  ;;                                   # default corpus, no suffix
    *)      PHASE2_TAG="${PHASE2_TAG}_d${DATA_TAG##*_}" ;;
esac
if [[ "$EXCLUDE_MODE" == "soft" ]]; then
    PHASE2_TAG="${PHASE2_TAG}${SOFT_EXCLUDE_ALPHA}"
fi
if [[ "$CAPACITY_FLOOR" != "0" ]]; then
    PHASE2_TAG="${PHASE2_TAG}_cf${CAPACITY_FLOOR}"
fi
if [[ "$MERGE_LOW_UTIL" == "1" ]]; then
    PHASE2_TAG="${PHASE2_TAG}_mg${MERGE_UTIL_MODE}"
fi
# [LAMBDA ABLATION] isolate any non-default lambda into its own tree. The
# conditional keeps all existing LAMDA=0.1 paths byte-identical.
if [[ "$LAMDA" != "0.1" ]]; then
    PHASE2_TAG="${PHASE2_TAG}_lam${LAMDA}"
fi
# [SEED] moved into RUN_SIG below, so all four run knobs sit in one segment.
if [[ -n "${RUN_TAG:-}" ]]; then
    PHASE2_TAG="${PHASE2_TAG}_${RUN_TAG}"
fi

# ── [TASK ORDER] Named permutations of the task stream ─────────────────────
# TASK_ORDER=1 is the historical default, so nothing already on disk moves.
# Passing TASK_LIST_STR directly still works and gets tagged 'ordc<hash>', so a
# hand-rolled order can never collide with a named one.
if [[ -n "${TASK_LIST_STR:-}" ]]; then
    ORDER_TAG="ordc$(printf '%s' "$TASK_LIST_STR" | md5sum | cut -c1-6)"
    TASK_ORDER="custom"
else
    TASK_ORDER="${TASK_ORDER:-1}"
    case "$TASK_ORDER" in
        1) TASK_LIST_STR="C-STANCE,FOMC,MeetingBank,ScienceQA,NumGLUE-cm,20Minuten,Py150,NumGLUE-ds,dbpedia,amazon,yahoo,agnews,yelp,BoolQA,QQP" ;;
        2) TASK_LIST_STR="QQP,BoolQA,yelp,agnews,yahoo,amazon,dbpedia,NumGLUE-ds,Py150,20Minuten,NumGLUE-cm,ScienceQA,MeetingBank,FOMC,C-STANCE" ;;
        3) TASK_LIST_STR="20Minuten,Py150,NumGLUE-ds,dbpedia,amazon,yahoo,agnews,yelp,BoolQA,QQP,C-STANCE,FOMC,MeetingBank,ScienceQA,NumGLUE-cm" ;;
        *) echo "ERROR: TASK_ORDER must be 1, 2 or 3 — or set TASK_LIST_STR directly (got '${TASK_ORDER}')" >&2; exit 1 ;;
    esac
    ORDER_TAG="ord${TASK_ORDER}"
fi
export TASK_LIST_STR

_bbf="${BLOCK_BUDGET_FRACTION#0.}"; _bbf="${_bbf#.}"; _bbf="${_bbf//./}"
RUN_SIG="_bbf${_bbf}_${ORDER_TAG}_${PROJECTION_MODE}_seed${SEED}"
if [[ "${LEGACY_TAG:-0}" == "1" ]]; then RUN_SIG=""; fi
ABLATION_TAG="${ABLATION_TAG}${RUN_SIG}_${PHASE2_TAG}_${SEED}_fix"
# [SETAFIX] tag the tree by split mode (+ structure-only) for provenance
ABLATION_TAG="${ABLATION_TAG}_setafix_${SETAFIX_SPLIT}_${SETAFIX_SCORE}"
[ "${SETAFIX_SIDE}" != "right" ]      && ABLATION_TAG="${ABLATION_TAG}_${SETAFIX_SIDE}"
[ "${SETAFIX_NULL}" != "gauss" ]      && ABLATION_TAG="${ABLATION_TAG}_${SETAFIX_NULL}"
[ "${SETAFIX_ORTHO}" == "on" ]        && ABLATION_TAG="${ABLATION_TAG}_ortho"
[ "${SETAFIX_SHARE_Q}" != "1.0" ]     && ABLATION_TAG="${ABLATION_TAG}_q${SETAFIX_SHARE_Q//./}"
[ "${SETAFIX_MIN_SHARED_FRAC_NEW}" != "0.0" ] && ABLATION_TAG="${ABLATION_TAG}_fn${SETAFIX_MIN_SHARED_FRAC_NEW//./}"
[ "${SETAFIX_MAX_SHARED_PER_TASK}" != "0" ]   && ABLATION_TAG="${ABLATION_TAG}_k${SETAFIX_MAX_SHARED_PER_TASK}"
[ "${SETAFIX_LAYER_GATE}" == "phi" ]      && ABLATION_TAG="${ABLATION_TAG}_lgphi"
[ "${SETAFIX_GATE_PROTECT}" == "freeze" ] && ABLATION_TAG="${ABLATION_TAG}_gatefz"
[ "${STRUCTURE_ONLY}" == "1" ] && ABLATION_TAG="${ABLATION_TAG}_structonly"
echo "  [arm] ABLATION_TAG = ${ABLATION_TAG}"
echo "  [sig] bbf=${BLOCK_BUDGET_FRACTION}  order=${TASK_ORDER}  proj=${PROJECTION_MODE}  seed=${SEED}"
echo "  [seam] phi=${SETAFIX_PHI} split=${SETAFIX_SPLIT} null=${SETAFIX_NULL} layer_gate=${SETAFIX_LAYER_GATE} (frac=${SETAFIX_LAYER_FRAC} min=${SETAFIX_LAYER_MIN}) ect=${EXPERT_CREATION_THRESHOLD}"

BASE_DIR="${PROJECT_ROOT}/RL_Lamda_fsys_trained_models_SoS_${user_model_name}_${PROJECTION_MODE}/ablation_${ABLATION_TAG}"

JOB_ID_DIR="${BASE_DIR}/.job_ids"
mkdir -p "$JOB_ID_DIR"

# ── Pre-flight: refuse to overwrite an existing run ──────────────────────────
# Runs ONCE, before any job is submitted, in BOTH submit modes. It used to sit
# inside submit_task_fused()'s sequential branch, where it never fired in chain
# mode and killed the arm after task 1 in sequential mode (task 1's own
# checkpoint tripped it).
if [ -d "$BASE_DIR" ]; then
    # EXISTING_CKPTS=$(ls -d "${BASE_DIR}"/shared_moE_checkpoints_* 2>/dev/null | wc -l)
    # Fixed:
    EXISTING_CKPTS=$(find "${BASE_DIR}" -maxdepth 1 -name "shared_moE_checkpoints_*" 2>/dev/null | wc -l)
    if [ "$EXISTING_CKPTS" -gt 0 ] && [ "${FORCE_OVERWRITE:-False}" != "True" ]; then
        echo "" >&2
        echo "============================================================" >&2
        echo "  REFUSING TO START: ${EXISTING_CKPTS} checkpoint(s) already exist" >&2
        echo "  ${BASE_DIR}" >&2
        echo "" >&2
        ls -1d "${BASE_DIR}"/shared_moE_checkpoints_* 2>/dev/null | sed 's/^/    /' >&2
        echo "" >&2
        echo "  This run would overwrite them task by task and they cannot be" >&2
        echo "  recovered without retraining." >&2
        echo "" >&2
        echo "  Either give this run its own tree:" >&2
        echo "      RUN_TAG=d5000 <your command>" >&2
        echo "  or move the old one aside:" >&2
        echo "      mv ${BASE_DIR} ${BASE_DIR}_old" >&2
        echo "  or, to continue an unfinished arm / overwrite on purpose:" >&2
        echo "      FORCE_OVERWRITE=True <your command>" >&2
        echo "============================================================" >&2
        exit 1
    fi
fi

echo "┌──────────────────────────────────────────────────────────────┐"
echo "│  ABLATION=${ABLATION} — ${ABLATION_TAG}"
printf "│  %-62s│\n" "fft_stopping_mode  = ${FFT_STOPPING_MODE}"
printf "│  %-62s│\n" "layer_norm_scoring = ${LAYER_NORM_SCORING}"
printf "│  %-62s│\n" "divergence_scoring = ${DIVERGENCE_SCORING}"
printf "│  %-62s│\n" "budget_mode        = ${BUDGET_MODE}"
echo "└──────────────────────────────────────────────────────────────┘"


NEW_PARAMS="PROJECTION_MODE=${PROJECTION_MODE},PROJ_CALIB_THRESHOLD=${PROJ_CALIB_THRESHOLD},PROJ_CALIB_STEPS=${PROJ_CALIB_STEPS},BLOCK_BUDGET_FRACTION=${BLOCK_BUDGET_FRACTION},GRAD_TEMPORAL_DECAY=${GRAD_TEMPORAL_DECAY},GLOBAL_EXPERT_MIN_BLOCKS=${GLOBAL_EXPERT_MIN_BLOCKS},EXPERT_CREATION_THRESHOLD=${EXPERT_CREATION_THRESHOLD},ADAPTIVE_FFT_STEPS=${ADAPTIVE_FFT_STEPS},FFT_MIN_STEPS=${FFT_MIN_STEPS},FFT_MAX_STEPS=${FFT_MAX_STEPS},FFT_REF_SAMPLES=${FFT_REF_SAMPLES},NOVELTY_BETA_MIN=${NOVELTY_BETA_MIN},NOVELTY_T_MAX=${NOVELTY_T_MAX},FFT_STOPPING_MODE=${FFT_STOPPING_MODE},LAYER_NORM_SCORING=${LAYER_NORM_SCORING},DIVERGENCE_SCORING=${DIVERGENCE_SCORING},BUDGET_MODE=${BUDGET_MODE},ABLATION_TAG=${ABLATION_TAG},LAMDA=${LAMDA},LAMBDA_MODE=${LAMBDA_MODE},GATING_RL=${GATING_RL},NO_EXPERT_SHARING=${NO_EXPERT_SHARING:-False},EVAL_ROUTING_TOPK=${EVAL_ROUTING_TOPK},GATE_MODE=${GATE_MODE},GATE_OPEN_BIAS=${GATE_OPEN_BIAS},USE_REENTRANT=${USE_REENTRANT},BUDGET_EXPANSION_CAP=${BUDGET_EXPANSION_CAP},LABEL_AUDIT_STRICT=${LABEL_AUDIT_STRICT},LABEL_AUDIT_SAMPLES=${LABEL_AUDIT_SAMPLES},LOSS_SANITY_OFF=${LOSS_SANITY_OFF},PHASE2_TAG=${PHASE2_TAG},ROUTER_REG=${ROUTER_REG},ROUTER_ANCHOR_MU=${ROUTER_ANCHOR_MU},ROUTER_N_CENTROIDS=${ROUTER_N_CENTROIDS},ROUTER_CENTROID_BATCHES=${ROUTER_CENTROID_BATCHES},FFT_TAG=${FFT_TAG},EXCLUDE_MODE=${EXCLUDE_MODE},SOFT_EXCLUDE_ALPHA=${SOFT_EXCLUDE_ALPHA},CAPACITY_FLOOR=${CAPACITY_FLOOR},MERGE_LOW_UTIL=${MERGE_LOW_UTIL},MERGE_GATE_THRESHOLD=${MERGE_GATE_THRESHOLD},MERGE_CAND_MAX_BLOCKS=${MERGE_CAND_MAX_BLOCKS},MERGE_MAX_LAYERS=${MERGE_MAX_LAYERS},MERGE_PROTECT_LATEST=${MERGE_PROTECT_LATEST},MERGE_UTIL_MODE=${MERGE_UTIL_MODE},MERGE_UTIL_Z=${MERGE_UTIL_Z},MERGE_UTIL_MIN_SPREAD=${MERGE_UTIL_MIN_SPREAD},EVAL_ROUTING_MODE=${EVAL_ROUTING_MODE},EVAL_OUT_SUFFIX=${EVAL_OUT_SUFFIX},GATE_PROBE=${GATE_PROBE},SEED=${SEED},TASK_ORDER=${TASK_ORDER},ORDER_TAG=${ORDER_TAG},SETAFIX_STRUCTURE_ONLY=${SETAFIX_STRUCTURE_ONLY},SETAFIX_SPLIT=${SETAFIX_SPLIT},SETAFIX_SCORE=${SETAFIX_SCORE},SETAFIX_Z=${SETAFIX_Z},SETAFIX_MIN_SHARED_FRAC=${SETAFIX_MIN_SHARED_FRAC},SETAFIX_NOEVIDENCE=${SETAFIX_NOEVIDENCE},SETAFIX_SIDE=${SETAFIX_SIDE},SETAFIX_NULL=${SETAFIX_NULL},SETAFIX_ORTHO=${SETAFIX_ORTHO},SETAFIX_GATE_PROTECT=${SETAFIX_GATE_PROTECT},SETAFIX_SHARE_Q=${SETAFIX_SHARE_Q},SETAFIX_MIN_SHARED_FRAC_NEW=${SETAFIX_MIN_SHARED_FRAC_NEW},SETAFIX_MAX_SHARED_PER_TASK=${SETAFIX_MAX_SHARED_PER_TASK},SETAFIX_LAYER_GATE=${SETAFIX_LAYER_GATE},SETAFIX_LAYER_FRAC=${SETAFIX_LAYER_FRAC},SETAFIX_LAYER_MIN=${SETAFIX_LAYER_MIN},SETAFIX_PHI=${SETAFIX_PHI}"

# TASK_LIST_STR is resolved and exported by the TASK ORDER block above.

IFS=',' read -r -a TASK_ARRAY <<< "$TASK_LIST_STR"
TOTAL_TASKS=${#TASK_ARRAY[@]}

# ── Parse TASK_RANGE → ordered array of task IDs ─────────────────────────────
parse_task_range() {
    local range="$1"
    local -a ids=()

    if [[ "$range" == *"-"* ]]; then
        local start="${range%-*}"
        local end="${range#*-}"
        if ! [[ "$start" =~ ^[0-9]+$ && "$end" =~ ^[0-9]+$ ]]; then
            echo "Error: invalid range '$range' — use format like 1-6" >&2; exit 1
        fi
        if (( start > end )); then
            echo "Error: start ($start) must be ≤ end ($end)" >&2; exit 1
        fi
        if (( end > TOTAL_TASKS )); then
            echo "Error: end ($end) exceeds total tasks ($TOTAL_TASKS)" >&2; exit 1
        fi 
        for i in $(seq "$start" "$end"); do ids+=("$i"); done

    elif [[ "$range" == *","* ]]; then
        IFS=',' read -r -a ids <<< "$range"
        for id in "${ids[@]}"; do
            if ! [[ "$id" =~ ^[0-9]+$ ]] || (( id < 1 || id > TOTAL_TASKS )); then
                echo "Error: invalid task ID '$id' — must be 1–${TOTAL_TASKS}" >&2; exit 1
            fi
        done

    else
        if ! [[ "$range" =~ ^[0-9]+$ ]] || (( range < 1 || range > TOTAL_TASKS )); then
            echo "Error: invalid task ID '$range' — must be 1–${TOTAL_TASKS}" >&2; exit 1
        fi
        ids=("$range")
    fi

    echo "${ids[@]}"
}

TASK_IDS=($(parse_task_range "$TASK_RANGE"))

# ── Launch summary ────────────────────────────────────────────────────────────
echo "============================================================"
echo "  SETA fsys Pipeline"
echo "  Model    : $user_model_name"
echo "  Tasks    : ${TASK_IDS[*]}  (${#TASK_IDS[@]} task(s))"
echo "  Steps    : $STEPS_TO_RUN"
echo "  Ablation : ${ABLATION} — ${ABLATION_TAG}"
echo "    fft_stopping_mode  = $FFT_STOPPING_MODE"
echo "    layer_norm_scoring = $LAYER_NORM_SCORING"
echo "    divergence_scoring = $DIVERGENCE_SCORING"
echo "    budget_mode        = $BUDGET_MODE"
echo "  Output   : ${BASE_DIR}"
echo "  Submit   : ${SUBMIT_MODE}"
if [[ "$SUBMIT_MODE" == "sequential" ]]; then
  echo "             one job at a time — poll every ${POLL_INTERVAL}s, wait_for_eval=${WAIT_FOR_EVAL}"
  echo "             keep this shell alive (tmux/screen); closing it stops the arm"
fi
if [[ "${EVAL_BLOCKS_CHAIN}" == "True" ]]; then
  echo "  Rule     : serial — Task N+1 waits for ALL steps of Task N (incl. eval)"
else
  echo "  Rule     : pipelined — Task N+1 split waits for Task N TRAIN;"
  echo "             Task N eval runs in parallel (EVAL_BLOCKS_CHAIN=True to serialise)"
fi
echo "============================================================"

# ── Per-task submission ───────────────────────────────────────────────────────
# ── [FUSED] one SLURM job per task ───────────────────────────────────────────
# Identical settings, paths and tags to the four-job pipeline. The only change
# is that fft, split, train (and optionally eval) run back to back inside a
# single allocation, so a task costs one queue wait instead of four and the
# model is loaded once per step rather than once per job.
#
# Chaining is unchanged: task k+1 depends on task k's single job.
submit_task_fused() {
    local TASK_ID="$1"
    local TASK_NAME="${TASK_ARRAY[$((TASK_ID - 1))]}"
    local SAVE_PATH="${BASE_DIR}/shared_moE_checkpoints_${user_model_name}_${TASK_ID}_${TASK_NAME,,}"
    local RESUME_PATH="None"
    if (( TASK_ID > 1 )); then
        local PREV_TASK_ID=$(( TASK_ID - 1 ))
        local PREV_TASK_NAME="${TASK_ARRAY[$((PREV_TASK_ID - 1))]}"
        RESUME_PATH="${BASE_DIR}/shared_moE_checkpoints_${user_model_name}_${PREV_TASK_ID}_${PREV_TASK_NAME,,}"
        # [SETAFIX] no trained checkpoint exists in structure-only mode:
        # resume from the previous task's SPLIT checkpoint instead
        if [ "${STRUCTURE_ONLY}" == "1" ]; then
            RESUME_PATH="${RESUME_PATH}/init_structure_checkpoint"
        fi
    fi

    local TASK_DATA_PATH="${BASE_DATA_PATH}"
    local CURRENT_MAX_SAMPLES="${MAX_SAMPLES:-}"
    if [ -z "$CURRENT_MAX_SAMPLES" ]; then
        CURRENT_MAX_SAMPLES=500
        if (( TASK_ID == 3 || TASK_ID == 6 )); then
            CURRENT_MAX_SAMPLES="${MAX_SAMPLES_GEN:-100}"
        fi
    fi

    echo ""
    echo "  ── Task ${TASK_ID}: ${TASK_NAME} ──────────────────────────────"
    echo "     Steps      : ${STEPS_TO_RUN}   (single job)"
    echo "     Save       : ${SAVE_PATH}"
    echo "     Resume     : ${RESUME_PATH}"

    local DEP_FLAG=""
    if [ -n "$LAST_TASK_JOB_ID" ]; then
        DEP_FLAG="--dependency=afterok:${LAST_TASK_JOB_ID}"
    elif [ -n "${FIRST_JOB_DEP:-}" ]; then
        DEP_FLAG="--dependency=afterok:${FIRST_JOB_DEP}"
    fi

    local BASE_EXPORT="user_model_name=${user_model_name},TASK_ID=${TASK_ID},TASK_NAME=${TASK_NAME},SAVE_PATH=${SAVE_PATH},RESUME_PATH=${RESUME_PATH},MAX_SAMPLES=${CURRENT_MAX_SAMPLES},train_sample_ratio=${train_sample_ratio},eval_task_id=${TASK_ID},BASE_DATA_PATH=${TASK_DATA_PATH},${NEW_PARAMS}"

    local JOB_ID
    JOB_ID=$(sbatch --parsable \
        ${DEP_FLAG} \
        --job-name="T${TASK_ID}_${TASK_NAME}_fused" \
        --export=ALL,${BASE_EXPORT},MODE=${STEPS_TO_RUN//,/:},SKIP_EVAL_TASKS= \
        "$SLURM_SCRIPT")

    save_job_id "$TASK_ID" "fused" "$JOB_ID"
    save_job_id "$TASK_ID" "last"  "$JOB_ID"
    LAST_TASK_JOB_ID="$JOB_ID"
    echo "     -> Job ${JOB_ID}  dep=[${DEP_FLAG#--dependency=}]"

    if [[ "$SUBMIT_MODE" == "sequential" ]]; then
        if ! wait_for_job "$JOB_ID" "T${TASK_ID} ${TASK_NAME} (${STEPS_TO_RUN})"; then
            return 1
        fi
        LAST_TASK_JOB_ID=""
    fi
    return 0
}

LAST_TASK_JOB_ID=""

for TASK_ID in "${TASK_IDS[@]}"; do
    if ! submit_task_fused "$TASK_ID"; then
        echo "" >&2
        echo "============================================================" >&2
        echo "  ARM ABORTED at Task ${TASK_ID}" >&2
        echo "  Arm dir: ${BASE_DIR}" >&2
        echo "  Resume with:" >&2
        echo "    ${0} ${TASK_ID}-${TASK_IDS[-1]} ${STEPS_TO_RUN} ${user_model_name}" >&2
        echo "============================================================" >&2
        exit 1
    fi
done

echo ""
echo "============================================================"
echo "  ${#TASK_IDS[@]} task(s) submitted as ${#TASK_IDS[@]} job(s)."
echo "  Steps per job: ${STEPS_TO_RUN}"
echo ""
printf "  %-6s %-16s %-12s\n" "Task" "Name" "Job"
printf "  %-6s %-16s %-12s\n" "----" "----" "---"
for TASK_ID in "${TASK_IDS[@]}"; do
    TNAME="${TASK_ARRAY[$((TASK_ID - 1))]}"
    printf "  %-6s %-16s %-12s\n" "$TASK_ID" "$TNAME" "$(load_job_id "$TASK_ID" "fused")"
done
echo ""
echo "  Monitor  : squeue -u \$USER --format='%.10i %.28j %.8T %R'"
echo "  Job IDs  : ${JOB_ID_DIR}/"
echo "  Logs     : ${PROJECT_ROOT}/logs/${user_model_name}_${PROJECTION_MODE}/${ABLATION_TAG}/"
echo "============================================================"

# ── [FINAL EVAL] full R matrix on the last checkpoint ────────────────────────
if [[ "${FINAL_EVAL}" == "True" ]]; then
    LAST_TASK_ID="${TASK_IDS[-1]}"
    FINAL_DEP=$(load_job_id "$LAST_TASK_ID" "last")
    EVAL_LAUNCHER="${SCRIPT_DIR}/RL_Lamda_fsys_eval_only_arms.sh"
    if [ ! -f "$EVAL_LAUNCHER" ]; then
        echo ""
        echo "  [final-eval] SKIPPED: launcher not found at ${EVAL_LAUNCHER}" >&2
    else
        FIRST_TASK_ID="${TASK_IDS[0]}"
        echo ""
        echo "  [final-eval] T${LAST_TASK_ID} checkpoint on tasks ${FIRST_TASK_ID}-${LAST_TASK_ID}"
        [ -n "$FINAL_DEP" ] && echo "  [final-eval] chained after job ${FINAL_DEP}"
        FIRST_JOB_DEP="${FINAL_DEP}" SUBMIT_MODE=chain \
        SEED="${SEED}" LAMDA="${LAMDA}" LAMBDA_MODE="${LAMBDA_MODE}" \
        GATING_RL="${GATING_RL}" \
        GATE_MODE="${GATE_MODE}" GATE_OPEN_BIAS="${GATE_OPEN_BIAS}" \
        ROUTER_REG="${ROUTER_REG}" ROUTER_ANCHOR_MU="${ROUTER_ANCHOR_MU}" \
        EXCLUDE_MODE="${EXCLUDE_MODE}" SOFT_EXCLUDE_ALPHA="${SOFT_EXCLUDE_ALPHA}" \
        CAPACITY_FLOOR="${CAPACITY_FLOOR}" MERGE_LOW_UTIL="${MERGE_LOW_UTIL}" \
        MERGE_UTIL_MODE="${MERGE_UTIL_MODE}" \
        PROJECTION_MODE="${PROJECTION_MODE}" ABLATION="${ABLATION}" \
        RUN_TAG="${RUN_TAG:-}" ARM=custom GATE_PROBE="${GATE_PROBE}" \
        EVAL_ROUTING_MODE="${EVAL_ROUTING_MODE:-}" \
        PROJECT_ROOT="${PROJECT_ROOT}" \
            bash "$EVAL_LAUNCHER" "$LAST_TASK_ID" "$user_model_name" \
                 "${FIRST_TASK_ID}-${LAST_TASK_ID}"
    fi
fi


# lambda ablation, same seed as baseline

# BLOCK_BUDGET_FRACTION=0.12 GATE_MODE=sigmoid ROUTER_REG=none PROJECTION_MODE=v_proj ABLATION=1 bash NAACL/RL_Lamda_fsys_pipeline_N_task_fused_5000_update_OOM.sh 3,4,5,6,7,8 fft,split,train,eval llama3_8b

# BLOCK_BUDGET_FRACTION=0.12 GATE_MODE=sigmoid ROUTER_REG=none PROJECTION_MODE=v_proj ABLATION=1 bash NAACL/RL_Lamda_fsys_pipeline_N_task_fused_5000_update_OOM.sh 8 fft,split,train,eval llama2_7b

# BLOCK_BUDGET_FRACTION=0.12 GATE_MODE=sigmoid ROUTER_REG=none PROJECTION_MODE=v_proj ABLATION=1 bash NAACL/RL_Lamda_fsys_pipeline_N_task_fused_5000_update_OOM.sh 1-3 fft,split,train llama2_7b

# BLOCK_BUDGET_FRACTION=0.12 GATE_MODE=sigmoid ROUTER_REG=none PROJECTION_MODE=v_proj ABLATION=1 bash NAACL/RL_Lamda_fsys_pipeline_N_task_fused_5000_update_OOM.sh 6,7,8 fft,split,train,eval qwen3_4b

# FIRST_JOB_DEP=12259132 BLOCK_BUDGET_FRACTION=0.12 GATE_MODE=sigmoid ROUTER_REG=none PROJECTION_MODE=q_proj ABLATION=1 bash NAACL/RL_Lamda_fsys_pipeline_N_task_fused_5000_update_OOM.sh 7,8 fft,split,train,eval qwen3_4b

# BLOCK_BUDGET_FRACTION=0.12 GATE_MODE=sigmoid ROUTER_REG=none PROJECTION_MODE=mlp_proj ABLATION=1 bash NAACL/RL_Lamda_fsys_pipeline_N_task_fused_5000_update_OOM.sh 7,8 fft,split,train,eval qwen3_4b

# BLOCK_BUDGET_FRACTION=0.12 GATE_MODE=sigmoid ROUTER_REG=none PROJECTION_MODE=v_proj ABLATION=1 bash NAACL/RL_Lamda_fsys_pipeline_N_task_fused_5000_update_OOM.sh 6,7,8 fft,split,train,eval llama2_7b

# then do
# python setafix_compare.py \
#   RL_Lamda_fsys_trained_models_SoS_qwen3_4b_v_proj_only/ablation_*_bbf12_*_setafix_seam_svd_structonly \
#   RL_Lamda_fsys_trained_models_SoS_qwen3_4b_v_proj_only/ablation_*_bbf25_*_setafix_seam_svd_structonly \
#   --labels "SEAM 12%","SEAM 25%"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# update on Sept 15 afternoon, run the following to get the 25% budget fraction results for SEAM ablation:
# BLOCK_BUDGET_FRACTION=0.25 ABLATION=1 STRUCTURE_ONLY=1 SETAFIX_NULL=empirical \
# bash run_setafix.sh 1-8 fft,split qwen3_4b

# BLOCK_BUDGET_FRACTION=0.12 ABLATION=1 SETAFIX_NULL=empirical \
# BLOCK_BUDGET_FRACTION=0.25 ABLATION=1 SETAFIX_NULL=empirical SETAFIX_ORTHO=on SETAFIX_GATE_PROTECT=freeze bash run_setafix.sh 1-3 fft,split,train,eval qwen3_4b
# FIRST_JOB_DE=13655640 BLOCK_BUDGET_FRACTION=0.25 ABLATION=1 SETAFIX_NULL=empirical SETAFIX_ORTHO=on SETAFIX_GATE_PROTECT=freeze FORCE_OVERWRITE=True bash run_setafix.sh 7,8 fft,split,train,eval qwen3_4b
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# cd /lustre/hdd/LAS/jannesar-lab/fatemask/fed-con/SETA_fix
# FIRST_JOB_DE=14553403 FORCE_OVERWRITE=True BLOCK_BUDGET_FRACTION=0.12 ABLATION=1 SETAFIX_NULL=empirical SETAFIX_ORTHO=on SETAFIX_GATE_PROTECT=freeze bash run_setafix.sh 6 fft,split,train,eval qwen3_4b


# SETAFIX_MIN_SHARED_FRAC=0.25 SETAFIX_MIN_SHARED_FRAC_NEW=0.05 SETAFIX_Z=4.0 FORCE_OVERWRITE=True BLOCK_BUDGET_FRACTION=0.12 ABLATION=1 SETAFIX_NULL=empirical SETAFIX_ORTHO=on SETAFIX_GATE_PROTECT=freeze bash run_setafix.sh 1,2,3 fft,split,train,eval llama2_7b
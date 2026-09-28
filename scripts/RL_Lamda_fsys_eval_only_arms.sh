#!/bin/bash

set -euo pipefail

# ── Arguments ─────────────────────────────────────────────────────────────────
MODEL_TASK_RANGE="${1:-1}"
user_model_name="${2:-qwen3_4b}"
EVAL_TASK_RANGE="${3:-}"          # optional — default cumulative 1..MODEL_TASK_ID

# ── Infrastructure ────────────────────────────────────────────────────────────
TRAIN_RATIO=1.0
# [SETAFIX] eval-only launcher for the Split-on-Share tree. Resolves the SAME
# ABLATION_TAG as run_setafix.sh (run signature + phase-2 tag + setafix
# suffixes), so it lands on the checkpoints run_setafix.sh wrote.
# PROJECT_ROOT, DATASET -> BASE_DATA_PATH, task orders, LEARNING_RATE: sos_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/sos_common.sh"
#
# What to evaluate (argument 3 wins over EVAL_SCHEME):
#   bash <this> 8   <model>          final row: checkpoint 8 on tasks 1..8
#   bash <this> 1-8 <model>          full grid: every checkpoint N on tasks 1..N
#   EVAL_SCHEME=diag bash <this> 1-8 <model>
#                                    diagonal: every checkpoint N on task N only
#   bash <this> 8   <model> 3,5      checkpoint 8 on tasks 3 and 5 only
# Then:  python cl_metrics.py ... (the command is printed at the end)
EVAL_SCHEME="${EVAL_SCHEME:-cumulative}"
case "$EVAL_SCHEME" in
    cumulative|diag) ;;
    *) echo "ERROR: EVAL_SCHEME must be cumulative or diag (got '${EVAL_SCHEME}')" >&2; exit 1 ;;
esac
# [PIN] Same canonical code as the training launcher: SETAV2 root, one path,
# no directory variable, no fallback. Eval imports the same model file, so the
# same fail-fast checks apply.
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
# Same slurm as the training launcher (handles MODE=eval as a one-step list),
# resolved next to this script — wherever it was invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/setafix_pipeline.slurm"
if [ ! -f "$SLURM_SCRIPT" ]; then
    echo "ERROR: slurm script not found at $SLURM_SCRIPT" >&2
    exit 1
fi

# ── S1–S4 dynamic parameters (must match the training run) ────────────────────
PROJECTION_MODE="${PROJECTION_MODE:-v_proj_only}"
# [TAG SAFETY] Identical guard to the training launcher. The default here used
# to be 'v_proj' while training defaulted to 'v_proj_only', so an eval with no
# explicit PROJECTION_MODE resolved to a parent directory training never wrote.
sos_canonical_projection   # v_proj -> v_proj_only, vqk_proj -> qkv_proj, ... (sos_common.sh)
PROJ_CALIB_THRESHOLD="${PROJ_CALIB_THRESHOLD:-0.10}"
PROJ_CALIB_STEPS="${PROJ_CALIB_STEPS:-20}"
BLOCK_BUDGET_FRACTION="${BLOCK_BUDGET_FRACTION:-0.12}"   # must match the training launcher
GRAD_TEMPORAL_DECAY="${GRAD_TEMPORAL_DECAY:-1.0}"
GLOBAL_EXPERT_MIN_BLOCKS="${GLOBAL_EXPERT_MIN_BLOCKS:-20}"
EXPERT_CREATION_THRESHOLD="${EXPERT_CREATION_THRESHOLD:-8}"
ADAPTIVE_FFT_STEPS="${ADAPTIVE_FFT_STEPS:-True}"
FFT_MIN_STEPS="${FFT_MIN_STEPS:-30}"
FFT_MAX_STEPS="${FFT_MAX_STEPS:-150}"
FFT_REF_SAMPLES="${FFT_REF_SAMPLES:-500}"
NOVELTY_BETA_MIN="${NOVELTY_BETA_MIN:-0.30}"
NOVELTY_T_MAX="${NOVELTY_T_MAX:-6}"
EVAL_ROUTING_TOPK="${EVAL_ROUTING_TOPK:-}"
train_sample_ratio="${train_sample_ratio:-1}"

# ── Fix-control defaults (must match the training run) ────────────────────────
GATE_MODE="${GATE_MODE:-sigmoid}"
GATE_OPEN_BIAS="${GATE_OPEN_BIAS:-1.0}"   # must match the training launcher
USE_REENTRANT="${USE_REENTRANT:-False}"
BUDGET_EXPANSION_CAP="${BUDGET_EXPANSION_CAP:-1.5}"
LABEL_AUDIT_STRICT="${LABEL_AUDIT_STRICT:-True}"
LABEL_AUDIT_SAMPLES="${LABEL_AUDIT_SAMPLES:-512}"
LOSS_SANITY_OFF="${LOSS_SANITY_OFF:-0}"

# ── [SETAFIX] split knobs — must match the training run (same defaults as
#    run_setafix.sh). Only the ones that reach the directory name matter for
#    resolving the tree, but all are forwarded so eval loads the same code path.
export SETAFIX_STRUCTURE_ONLY="0"
export SETAFIX_PHI="${SETAFIX_PHI:-0.020}"
export SETAFIX_SPLIT="${SETAFIX_SPLIT:-seam}"
export SETAFIX_SCORE="${SETAFIX_SCORE:-svd}"
export SETAFIX_Z="${SETAFIX_Z:-3.0}"
export SETAFIX_MIN_SHARED_FRAC="${SETAFIX_MIN_SHARED_FRAC:-0.15}"
export SETAFIX_NOEVIDENCE="${SETAFIX_NOEVIDENCE:-share}"
export SETAFIX_SIDE="${SETAFIX_SIDE:-right}"
export SETAFIX_NULL="${SETAFIX_NULL:-gauss}"
export SETAFIX_ORTHO="${SETAFIX_ORTHO:-off}"
export SETAFIX_GATE_PROTECT="${SETAFIX_GATE_PROTECT:-off}"
export SETAFIX_SHARE_Q="${SETAFIX_SHARE_Q:-1.0}"
export SETAFIX_MIN_SHARED_FRAC_NEW="${SETAFIX_MIN_SHARED_FRAC_NEW:-${SETAFIX_PHI}}"
export SETAFIX_MAX_SHARED_PER_TASK="${SETAFIX_MAX_SHARED_PER_TASK:-0}"
export SETAFIX_LAYER_GATE="${SETAFIX_LAYER_GATE:-phi}"
export SETAFIX_LAYER_FRAC="${SETAFIX_LAYER_FRAC:-0.25}"
export SETAFIX_LAYER_MIN="${SETAFIX_LAYER_MIN:-3}"
case "$SETAFIX_LAYER_GATE" in
    phi|budget|relative|absolute|off) ;;
    *) echo "ERROR: SETAFIX_LAYER_GATE must be phi|budget|relative|absolute|off (got '${SETAFIX_LAYER_GATE}')" >&2; exit 1 ;;
esac

# ── Router regularisation ─────────────────────────────────────────────
ROUTER_REG="${ROUTER_REG:-none}"
ROUTER_ANCHOR_MU="${ROUTER_ANCHOR_MU:-1.0}"
ROUTER_N_CENTROIDS="${ROUTER_N_CENTROIDS:-8}"
ROUTER_CENTROID_BATCHES="${ROUTER_CENTROID_BATCHES:-16}"
FFT_TAG="${FFT_TAG:-}"

case "$ROUTER_REG" in
    none|penalty|freeze|anchor|anchor_sparse) ;;
    *) echo "ERROR: ROUTER_REG must be one of none|penalty|freeze|anchor|anchor_sparse (got '${ROUTER_REG}')" >&2; exit 1 ;;
esac

# ── [DESIGN-2] Arm presets ────────────────────────────────────────────────────
# The preset only fills in variables you have not set yourself, so
#   ARM=full SOFT_EXCLUDE_ALPHA=0.3 ...
# gives the full system with a different alpha and still resolves to the tree
# that alpha trained in.
ARM="${ARM:-custom}"
case "$ARM" in
    base)   : "${EXCLUDE_MODE:=hard}"; : "${CAPACITY_FLOOR:=0}";   : "${MERGE_LOW_UTIL:=0}" ;;
    soft)   : "${EXCLUDE_MODE:=soft}"; : "${CAPACITY_FLOOR:=0}";   : "${MERGE_LOW_UTIL:=0}" ;;
    floor)  : "${EXCLUDE_MODE:=hard}"; : "${CAPACITY_FLOOR:=0.6}"; : "${MERGE_LOW_UTIL:=0}" ;;
    merge)  : "${EXCLUDE_MODE:=hard}"; : "${CAPACITY_FLOOR:=0}";   : "${MERGE_LOW_UTIL:=1}" ;;
    full)   : "${EXCLUDE_MODE:=soft}"; : "${CAPACITY_FLOOR:=0.6}"; : "${MERGE_LOW_UTIL:=1}" ;;
    custom) ;;
    *) echo "ERROR: ARM must be base|soft|floor|merge|full|custom (got '${ARM}')" >&2; exit 1 ;;
esac

EXCLUDE_MODE="${EXCLUDE_MODE:-hard}"
SOFT_EXCLUDE_ALPHA="${SOFT_EXCLUDE_ALPHA:-0.5}"
CAPACITY_FLOOR="${CAPACITY_FLOOR:-0}"
MERGE_LOW_UTIL="${MERGE_LOW_UTIL:-0}"
MERGE_GATE_THRESHOLD="${MERGE_GATE_THRESHOLD:-0.15}"
MERGE_CAND_MAX_BLOCKS="${MERGE_CAND_MAX_BLOCKS:-64}"
MERGE_MAX_LAYERS="${MERGE_MAX_LAYERS:-1}"
MERGE_PROTECT_LATEST="${MERGE_PROTECT_LATEST:-1}"
MERGE_UTIL_MODE="${MERGE_UTIL_MODE:-relative}"
MERGE_UTIL_Z="${MERGE_UTIL_Z:-1.0}"
MERGE_UTIL_MIN_SPREAD="${MERGE_UTIL_MIN_SPREAD:-0.01}"
MERGE_STATS_GLOB="${MERGE_STATS_GLOB:-}"

case "$EXCLUDE_MODE" in
    hard|soft) ;;
    *) echo "ERROR: EXCLUDE_MODE must be 'hard' or 'soft' (got '${EXCLUDE_MODE}')" >&2; exit 1 ;;
esac

# ── Eval-time routing variant ─────────────────────────────────────────────────
# Blank = whatever the checkpoint learned.  Any other value is an inference-time
# override on the same weights, so the checkpoint is never modified.
EVAL_ROUTING_MODE="${EVAL_ROUTING_MODE:-}"
case "${EVAL_ROUTING_MODE}" in
    ""|learned|oracle|uniform|random|none) ;;
    *) echo "ERROR: EVAL_ROUTING_MODE must be learned|oracle|uniform|random|none (got '${EVAL_ROUTING_MODE}')" >&2; exit 1 ;;
esac

# Auto-suffix the output directory for any non-learned variant, so an oracle
# sweep cannot overwrite the learned-routing numbers already in the tree.
if [ -z "${EVAL_OUT_SUFFIX:-}" ] && [ -n "${EVAL_ROUTING_MODE}" ] \
   && [ "${EVAL_ROUTING_MODE}" != "learned" ]; then
    EVAL_OUT_SUFFIX="_${EVAL_ROUTING_MODE}"
fi
EVAL_OUT_SUFFIX="${EVAL_OUT_SUFFIX:-}"
GATE_PROBE="${GATE_PROBE:-0}"
SEED="${SEED:-100}"

# ── Ablation mode (must match the training run) ───────────────────────────────
ABLATION="${ABLATION:-1}"
case "$ABLATION" in
    0)  ABLATION_TAG="original_seta"
        FFT_STOPPING_MODE="fixed"
        LAYER_NORM_SCORING="False"
        DIVERGENCE_SCORING="True"
        BUDGET_MODE="legacy"
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

# ── Phase 2 mode (must match the training run) ────────────────────────────────
LAMDA="${LAMDA:-0.1}"
LAMBDA_MODE="${LAMBDA_MODE:-dynamic}"
GATING_RL="${GATING_RL:-False}"

# ── Run-isolation tag — IDENTICAL chain to the training pipeline script ───────
PHASE2_TAG="RL_Lamda_${LAMBDA_MODE}_grl${GATING_RL}"
PHASE2_TAG="${PHASE2_TAG}_gate${GATE_MODE}_rr${ROUTER_REG}"
if [[ "$ROUTER_REG" == "anchor" || "$ROUTER_REG" == "anchor_sparse" ]]; then
    PHASE2_TAG="${PHASE2_TAG}mu${ROUTER_ANCHOR_MU}"
fi
# [TAG] Mirrors RL_Lamda_fsys_pipeline_N_task_fused_5000.sh exactly. That script
# drops the _ex<mode> segment and appends the corpus size instead, so this must
# do the same or it resolves to a directory training never wrote to.
DATA_TAG="$(basename "${BASE_DATA_PATH}")"
case "$DATA_TAG" in
    *_500)  ;;
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
if [[ "$LAMDA" != "0.1" ]]; then
    PHASE2_TAG="${PHASE2_TAG}_lam${LAMDA}"
fi
sos_lr_tag   # non-default LEARNING_RATE -> _lr<value> (sos_common.sh)
# [SEED] moved into RUN_SIG below, so all four run knobs sit in one segment.
if [[ -n "${RUN_TAG:-}" ]]; then
    PHASE2_TAG="${PHASE2_TAG}_${RUN_TAG}"
fi

# ── [TASK ORDER] Named permutations — must match the training launcher ─────
# The checkpoint directory name embeds the task name at position N, so the
# order has to be resolved before any path is built.
sos_resolve_task_order   # TASK_ORDER 1-5 or TASK_LIST_STR -> TASK_LIST_STR, ORDER_TAG (sos_common.sh)

# ── [RUN SIGNATURE] Byte-identical to the training launcher ────────────────
#   bbf<n>   BLOCK_BUDGET_FRACTION  (0.25 -> bbf25, 0.12 -> bbf12)
#   ord<n>   task stream order
#   <proj>   PROJECTION_MODE
#   seed<n>  SEED
# LEGACY_TAG=1 reaches checkpoints written before the signature existed.
_bbf="${BLOCK_BUDGET_FRACTION#0.}"; _bbf="${_bbf#.}"; _bbf="${_bbf//./}"
RUN_SIG="_bbf${_bbf}_${ORDER_TAG}_${PROJECTION_MODE}_seed${SEED}"
if [[ "${LEGACY_TAG:-0}" == "1" ]]; then RUN_SIG=""; fi

# ABLATION_TAG="${ABLATION_TAG}${RUN_SIG}_${PHASE2_TAG}_fix"

ABLATION_TAG="${ABLATION_TAG}${RUN_SIG}_${PHASE2_TAG}_${SEED}_fix"
# [SETAFIX] same suffix chain as run_setafix.sh
ABLATION_TAG="${ABLATION_TAG}_setafix_${SETAFIX_SPLIT}_${SETAFIX_SCORE}"
[ "${SETAFIX_SIDE}" != "right" ]      && ABLATION_TAG="${ABLATION_TAG}_${SETAFIX_SIDE}"
[ "${SETAFIX_NULL}" != "gauss" ]      && ABLATION_TAG="${ABLATION_TAG}_${SETAFIX_NULL}"
[ "${SETAFIX_ORTHO}" == "on" ]        && ABLATION_TAG="${ABLATION_TAG}_ortho"
[ "${SETAFIX_SHARE_Q}" != "1.0" ]     && ABLATION_TAG="${ABLATION_TAG}_q${SETAFIX_SHARE_Q//./}"
[ "${SETAFIX_MIN_SHARED_FRAC_NEW}" != "0.0" ] && ABLATION_TAG="${ABLATION_TAG}_fn${SETAFIX_MIN_SHARED_FRAC_NEW//./}"
[ "${SETAFIX_MAX_SHARED_PER_TASK}" != "0" ]   && ABLATION_TAG="${ABLATION_TAG}_k${SETAFIX_MAX_SHARED_PER_TASK}"
[ "${SETAFIX_LAYER_GATE}" == "phi" ]      && ABLATION_TAG="${ABLATION_TAG}_lgphi"
[ "${SETAFIX_LAYER_GATE}" == "budget" ]   && ABLATION_TAG="${ABLATION_TAG}_lgbud${SETAFIX_PHI//./}"
[ "${SETAFIX_GATE_PROTECT}" == "freeze" ] && ABLATION_TAG="${ABLATION_TAG}_gatefz"

MODEL_ROOT="${SOS_MODELS_DIR}/RL_Lamda_fsys_trained_models_SoS_${user_model_name}_${PROJECTION_MODE}"
BASE_DIR="${MODEL_ROOT}/ablation_${ABLATION_TAG}"

# ── [DISCOVERY] LIST_ARMS=1 prints every trained tree and its checkpoints ──
# Cheapest way to find the exact knobs behind a directory when an eval misses.
if [[ "${LIST_ARMS:-0}" == "1" ]]; then
    echo "Trees under ${MODEL_ROOT}:"
    if [ -d "$MODEL_ROOT" ]; then
        for _d in "${MODEL_ROOT}"/ablation_*; do
            [ -d "$_d" ] || continue
            echo ""
            echo "  $(basename "$_d")"
            for _c in "${_d}"/shared_moE_checkpoints_*; do
                [ -d "$_c" ] || continue
                if [ -f "${_c}/expert_definitions.json" ] || [ -f "${_c}/pytorch_model.bin" ]; then
                    echo "      [ok]      $(basename "$_c")"
                else
                    echo "      [partial] $(basename "$_c")"
                fi
            done
        done
    else
        echo "  (model root does not exist)"
    fi
    exit 0
fi

if [ ! -d "$BASE_DIR" ]; then
    echo "ERROR: arm directory not found:" >&2
    echo "       ${BASE_DIR}" >&2
    echo "" >&2
    echo "       Resolved from:" >&2
    echo "         BLOCK_BUDGET_FRACTION = ${BLOCK_BUDGET_FRACTION}  -> bbf${_bbf}" >&2
    echo "         TASK_ORDER            = ${TASK_ORDER}  -> ${ORDER_TAG}" >&2
    echo "         PROJECTION_MODE       = ${PROJECTION_MODE}" >&2
    echo "         SEED                  = ${SEED}" >&2
    echo "         ARM / EXCLUDE / CF / MERGE / LAMDA / RUN_TAG -> ${PHASE2_TAG}" >&2
    echo "         SETAFIX_NULL / ORTHO / GATE_PROTECT -> ${ABLATION_TAG##*_fix}" >&2
    echo "         SETAFIX_LAYER_GATE    = ${SETAFIX_LAYER_GATE}" >&2
    echo "         SETAFIX_PHI           = ${SETAFIX_PHI}  -> lgbud${SETAFIX_PHI//./} / fn${SETAFIX_MIN_SHARED_FRAC_NEW//./}" >&2
    echo "" >&2
    echo "       Every one of these must match the training run. For a checkpoint" >&2
    echo "       written before the run signature existed, add LEGACY_TAG=1." >&2
    echo "" >&2
    echo "       Trees that do exist:" >&2
    ls -1d "${MODEL_ROOT}"/ablation_* 2>/dev/null | while read -r _p; do
        echo "         $(basename "$_p")" >&2
    done
    [ -d "$MODEL_ROOT" ] || echo "         (model root ${MODEL_ROOT} does not exist)" >&2
    echo "" >&2
    echo "       Full inventory: LIST_ARMS=1 bash ${0} ${MODEL_TASK_RANGE} ${user_model_name}" >&2
    exit 1
fi

# ── Submission mode ───────────────────────────────────────────────────────────
SUBMIT_MODE="${SUBMIT_MODE:-chain}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"
STOP_ON_FAIL="${STOP_ON_FAIL:-True}"

case "$SUBMIT_MODE" in
    chain|sequential) ;;
    *) echo "ERROR: SUBMIT_MODE must be 'chain' or 'sequential' (got '${SUBMIT_MODE}')" >&2; exit 1 ;;
esac

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
    state=$(sacct -j "$jid" --noheader -o State -X 2>/dev/null | head -1 | tr -d ' ')
    if [[ "$state" == "COMPLETED" ]]; then
        echo "        [OK] ${label} COMPLETED after $(( waited / 60 )) min"
        return 0
    fi
    echo "" >&2
    echo "  [FAIL] ${label} (job ${jid}) ended in state '${state:-UNKNOWN}'." >&2
    echo "         sacct -j ${jid} --format=JobID,JobName,State,ExitCode,Elapsed" >&2
    return 1
}

# ── NEW_PARAMS export string — same keys the training pipeline forwards ───────
# MERGE_STATS_GLOB is deliberately absent: glob characters break --export.
# It is exported through the environment instead, below.
NEW_PARAMS="PROJECTION_MODE=${PROJECTION_MODE},PROJ_CALIB_THRESHOLD=${PROJ_CALIB_THRESHOLD},PROJ_CALIB_STEPS=${PROJ_CALIB_STEPS},BLOCK_BUDGET_FRACTION=${BLOCK_BUDGET_FRACTION},GRAD_TEMPORAL_DECAY=${GRAD_TEMPORAL_DECAY},GLOBAL_EXPERT_MIN_BLOCKS=${GLOBAL_EXPERT_MIN_BLOCKS},EXPERT_CREATION_THRESHOLD=${EXPERT_CREATION_THRESHOLD},ADAPTIVE_FFT_STEPS=${ADAPTIVE_FFT_STEPS},FFT_MIN_STEPS=${FFT_MIN_STEPS},FFT_MAX_STEPS=${FFT_MAX_STEPS},FFT_REF_SAMPLES=${FFT_REF_SAMPLES},NOVELTY_BETA_MIN=${NOVELTY_BETA_MIN},NOVELTY_T_MAX=${NOVELTY_T_MAX},FFT_STOPPING_MODE=${FFT_STOPPING_MODE},LAYER_NORM_SCORING=${LAYER_NORM_SCORING},DIVERGENCE_SCORING=${DIVERGENCE_SCORING},BUDGET_MODE=${BUDGET_MODE},ABLATION_TAG=${ABLATION_TAG},LAMDA=${LAMDA},LAMBDA_MODE=${LAMBDA_MODE},GATING_RL=${GATING_RL},NO_EXPERT_SHARING=${NO_EXPERT_SHARING:-False},EVAL_ROUTING_TOPK=${EVAL_ROUTING_TOPK},GATE_MODE=${GATE_MODE},GATE_OPEN_BIAS=${GATE_OPEN_BIAS},USE_REENTRANT=${USE_REENTRANT},BUDGET_EXPANSION_CAP=${BUDGET_EXPANSION_CAP},LABEL_AUDIT_STRICT=${LABEL_AUDIT_STRICT},LABEL_AUDIT_SAMPLES=${LABEL_AUDIT_SAMPLES},LOSS_SANITY_OFF=${LOSS_SANITY_OFF},PHASE2_TAG=${PHASE2_TAG},ROUTER_REG=${ROUTER_REG},ROUTER_ANCHOR_MU=${ROUTER_ANCHOR_MU},ROUTER_N_CENTROIDS=${ROUTER_N_CENTROIDS},ROUTER_CENTROID_BATCHES=${ROUTER_CENTROID_BATCHES},FFT_TAG=${FFT_TAG},train_sample_ratio=${train_sample_ratio},EXCLUDE_MODE=${EXCLUDE_MODE},SOFT_EXCLUDE_ALPHA=${SOFT_EXCLUDE_ALPHA},CAPACITY_FLOOR=${CAPACITY_FLOOR},MERGE_LOW_UTIL=${MERGE_LOW_UTIL},MERGE_GATE_THRESHOLD=${MERGE_GATE_THRESHOLD},MERGE_CAND_MAX_BLOCKS=${MERGE_CAND_MAX_BLOCKS},MERGE_MAX_LAYERS=${MERGE_MAX_LAYERS},MERGE_PROTECT_LATEST=${MERGE_PROTECT_LATEST},MERGE_UTIL_MODE=${MERGE_UTIL_MODE},MERGE_UTIL_Z=${MERGE_UTIL_Z},MERGE_UTIL_MIN_SPREAD=${MERGE_UTIL_MIN_SPREAD},EVAL_ROUTING_MODE=${EVAL_ROUTING_MODE},EVAL_OUT_SUFFIX=${EVAL_OUT_SUFFIX},GATE_PROBE=${GATE_PROBE},SEED=${SEED},TASK_ORDER=${TASK_ORDER},ORDER_TAG=${ORDER_TAG},SETAFIX_STRUCTURE_ONLY=${SETAFIX_STRUCTURE_ONLY},SETAFIX_SPLIT=${SETAFIX_SPLIT},SETAFIX_SCORE=${SETAFIX_SCORE},SETAFIX_Z=${SETAFIX_Z},SETAFIX_MIN_SHARED_FRAC=${SETAFIX_MIN_SHARED_FRAC},SETAFIX_NOEVIDENCE=${SETAFIX_NOEVIDENCE},SETAFIX_SIDE=${SETAFIX_SIDE},SETAFIX_NULL=${SETAFIX_NULL},SETAFIX_ORTHO=${SETAFIX_ORTHO},SETAFIX_GATE_PROTECT=${SETAFIX_GATE_PROTECT},SETAFIX_SHARE_Q=${SETAFIX_SHARE_Q},SETAFIX_MIN_SHARED_FRAC_NEW=${SETAFIX_MIN_SHARED_FRAC_NEW},SETAFIX_MAX_SHARED_PER_TASK=${SETAFIX_MAX_SHARED_PER_TASK},SETAFIX_LAYER_GATE=${SETAFIX_LAYER_GATE},SETAFIX_LAYER_FRAC=${SETAFIX_LAYER_FRAC},SETAFIX_LAYER_MIN=${SETAFIX_LAYER_MIN},SETAFIX_PHI=${SETAFIX_PHI},PROJECT_ROOT=${PROJECT_ROOT},SOS_FFT_ROOT=${SOS_FFT_ROOT},LEARNING_RATE=${LEARNING_RATE},GEN_DATA_PATH=${GEN_DATA_PATH}"

export MERGE_STATS_GLOB

# ── Task name list ────────────────────────────────────────────────────────────
# (resolved and exported by the TASK ORDER block above)
IFS=',' read -r -a TASK_ARRAY <<< "$TASK_LIST_STR"
TOTAL_TASKS=${#TASK_ARRAY[@]}

# ── Parse a range spec → ordered array of task IDs ────────────────────────────
parse_task_range() {
    local range="$1"
    local -a ids=()

    if [[ "$range" == *"-"* ]]; then
        local start="${range%-*}" end="${range#*-}"
        if ! [[ "$start" =~ ^[0-9]+$ && "$end" =~ ^[0-9]+$ ]]; then
            echo "Error: invalid range '$range' — use format like 1-8" >&2; exit 1
        fi
        (( start > end )) && { echo "Error: start ($start) must be <= end ($end)" >&2; exit 1; }
        (( end > TOTAL_TASKS )) && { echo "Error: end ($end) exceeds total tasks ($TOTAL_TASKS)" >&2; exit 1; }
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

MODEL_TASK_IDS=($(parse_task_range "$MODEL_TASK_RANGE"))

# ── Job ID persistence (inside the arm tree) ──────────────────────────────────
JOB_ID_DIR="${BASE_DIR}/.job_ids"
mkdir -p "$JOB_ID_DIR"
save_eval_job_id() { echo "$3" > "${JOB_ID_DIR}/task_${1}_eval_on_${2}${EVAL_OUT_SUFFIX}.txt"; }

# ── Banner ────────────────────────────────────────────────────────────────────
ARM_DESC="ex${EXCLUDE_MODE}"
[[ "$EXCLUDE_MODE" == "soft" ]] && ARM_DESC="${ARM_DESC}(a=${SOFT_EXCLUDE_ALPHA})"
[[ "$CAPACITY_FLOOR" != "0" ]] && ARM_DESC="${ARM_DESC} cf=${CAPACITY_FLOOR}"
[[ "$MERGE_LOW_UTIL" == "1" ]] && ARM_DESC="${ARM_DESC} merge=${MERGE_UTIL_MODE}"

echo "============================================================"
echo "  SETA — Eval-Only Launcher"
echo "  Model        : ${user_model_name}"
echo "  Arm          : ${ARM}  →  ${ARM_DESC}"
echo "  Router       : GATE_MODE=${GATE_MODE}  ROUTER_REG=${ROUTER_REG}"
echo "  Ablation     : ${ABLATION} — ${ABLATION_TAG}"
echo "  Signature    : bbf=${BLOCK_BUDGET_FRACTION}  order=${TASK_ORDER}  proj=${PROJECTION_MODE}  seed=${SEED}${LEGACY_TAG:+  LEGACY_TAG=1}"
echo "  SEAM         : split=${SETAFIX_SPLIT} null=${SETAFIX_NULL} ortho=${SETAFIX_ORTHO} gate_protect=${SETAFIX_GATE_PROTECT}"
echo "  Split rule   : layer_gate=${SETAFIX_LAYER_GATE} phi=${SETAFIX_PHI} fn=${SETAFIX_MIN_SHARED_FRAC_NEW}"
echo "  Model tasks  : ${MODEL_TASK_IDS[*]}  (${#MODEL_TASK_IDS[@]} checkpoint(s))"
if [ -n "${EVAL_TASK_RANGE}" ]; then
    echo "  Eval tasks   : ${EVAL_TASK_RANGE}  (explicit)"
elif [[ "$EVAL_SCHEME" == "diag" ]]; then
    echo "  Eval scheme  : diag  (model TN → dataset N)"
else
    echo "  Eval scheme  : cumulative  (model TN → datasets 1..N)"
fi
echo "  Data         : DATASET=${DATASET}  ${BASE_DATA_PATH}"
echo "  LR (tag)     : ${LEARNING_RATE}"
echo "  Routing      : ${EVAL_ROUTING_MODE:-<learned>}${EVAL_ROUTING_TOPK:+  topk=${EVAL_ROUTING_TOPK}}"
[ -n "${EVAL_OUT_SUFFIX}" ] && echo "  Out suffix   : ${EVAL_OUT_SUFFIX}  (learned results left intact)"
echo "  Submit       : ${SUBMIT_MODE}"
[[ "$SUBMIT_MODE" == "sequential" ]] && echo "                 one job at a time — poll every ${POLL_INTERVAL}s"
echo "  Arm dir      : ${BASE_DIR}"
echo "  SLURM script : ${SLURM_SCRIPT}"
echo "============================================================"

[ -n "${FIRST_JOB_DEP:-}" ] && echo "  [global dep] first eval job waits for external job: ${FIRST_JOB_DEP}"

declare -A SUBMITTED_JOBS    # "MODEL:EVAL" → job id

# ── Per-checkpoint submission ─────────────────────────────────────────────────
submit_eval_for_task() {
    local MODEL_TASK_ID="$1"
    local TASK_NAME="${TASK_ARRAY[$((MODEL_TASK_ID - 1))]}"
    local SAVE_PATH="${BASE_DIR}/shared_moE_checkpoints_${user_model_name}_${MODEL_TASK_ID}_${TASK_NAME,,}"

    echo ""
    echo "  ── Model T${MODEL_TASK_ID}: ${TASK_NAME} ──────────────────────────────"
    echo "     Checkpoint : ${SAVE_PATH}"

    # The checkpoint only has to exist NOW for an unchained submission. When a
    # dependency is set the job that writes it is still queued or running, so
    # skipping here would silently submit nothing — which is exactly what a
    # chained eval is for. ALLOW_MISSING_CKPT=True forces the same leniency.
    if [ -z "${FIRST_JOB_DEP:-}" ] && [[ "${ALLOW_MISSING_CKPT:-False}" != "True" ]]; then
        if [ ! -d "$SAVE_PATH" ]; then
            echo "     [SKIP] checkpoint directory not found — skipping Model T${MODEL_TASK_ID}."
            echo "            Checkpoints present in this tree:"
            local _found=0
            for _c in "${BASE_DIR}"/shared_moE_checkpoints_*; do
                [ -d "$_c" ] || continue
                echo "              $(basename "$_c")"
                _found=1
            done
            [ "$_found" -eq 0 ] && echo "              (none — this arm has not trained yet)"
            echo "            (set FIRST_JOB_DEP=<jobid> to queue behind the job"
            echo "             that will create it, or ALLOW_MISSING_CKPT=True)"
            return 0
        fi
        if [ ! -f "${SAVE_PATH}/pytorch_model.bin" ] && [ ! -f "${SAVE_PATH}/expert_definitions.json" ]; then
            echo "     [SKIP] checkpoint incomplete — neither pytorch_model.bin nor"
            echo "            expert_definitions.json is present. Contents:"
            ls -1 "$SAVE_PATH" 2>/dev/null | head -20 | sed 's/^/              /'
            return 0
        fi
        echo "     Weights    : $(ls -1 "${SAVE_PATH}"/*.bin "${SAVE_PATH}"/*.safetensors 2>/dev/null | wc -l) file(s), experts=$([ -f "${SAVE_PATH}/expert_definitions.json" ] && echo yes || echo no)"
    elif [ ! -d "$SAVE_PATH" ]; then
        echo "     [pending] checkpoint does not exist yet; the job below waits"
        echo "               for ${FIRST_JOB_DEP:-<none>} to create it."
    fi

    local BASE_EXPORT="user_model_name=${user_model_name},TASK_ID=${MODEL_TASK_ID},TASK_NAME=${TASK_NAME},SAVE_PATH=${SAVE_PATH},RESUME_PATH=None,TRAIN_RATIO=${TRAIN_RATIO},BASE_DATA_PATH=${BASE_DATA_PATH},MODE=eval,EVAL_ONLY=True,SKIP_EVAL_TASKS=,${NEW_PARAMS}"

    local -a EVAL_IDS=()
    if [ -n "${EVAL_TASK_RANGE}" ]; then
        local _raw
        _raw=$(parse_task_range "${EVAL_TASK_RANGE}")
        read -r -a EVAL_IDS <<< "$_raw"
    elif [[ "$EVAL_SCHEME" == "diag" ]]; then
        EVAL_IDS=("$MODEL_TASK_ID")
    else
        for (( _i=1; _i<=MODEL_TASK_ID; _i++ )); do EVAL_IDS+=("$_i"); done
    fi

    echo "     Eval tasks : ${EVAL_IDS[*]}"

    local LAST_EVAL_JOB_ID=""

    # ── [ONE_JOB] one sbatch for every requested dataset ─────────────────────
    # The model load dominates an eval-only run, so scoring N datasets in N jobs
    # pays that cost N times. EVAL_TASK_IDS makes the python side loop over the
    # set inside a single process. ONE_JOB=False restores one job per dataset.
    if [[ "${ONE_JOB:-True}" == "True" ]]; then
        local IDS_JOINED=""
        local KEPT=()
        for EVAL_ID in "${EVAL_IDS[@]}"; do
            if (( EVAL_ID > MODEL_TASK_ID )) && [[ "${ALLOW_FUTURE_EVAL:-False}" != "True" ]]; then
                echo "     [SKIP] dataset T${EVAL_ID} is ahead of model T${MODEL_TASK_ID}"
                continue
            fi
            KEPT+=("$EVAL_ID")
            IDS_JOINED="${IDS_JOINED:+${IDS_JOINED}:}${EVAL_ID}"
        done
        if [ ${#KEPT[@]} -eq 0 ]; then
            echo "     [SKIP] nothing to evaluate"
            return 0
        fi

        # A SLURM job carries ONE MAX_SAMPLES, so bundling a generation dataset
        # with the classification ones drags every dataset down to the generation
        # cap (25) and destroys their sample size. Group by cap instead: one job
        # per group, each dataset evaluated at the count it should be.
        # MAX_SAMPLES set explicitly overrides the split and gives a single job.
        local -a GRP_STD=() GRP_GEN=()
        for _e in "${KEPT[@]}"; do
            if [ -n "${MAX_SAMPLES:-}" ]; then
                GRP_STD+=("$_e")
            elif sos_is_gen_task "${TASK_ARRAY[$((_e - 1))]}"; then
                GRP_GEN+=("$_e")
            else
                GRP_STD+=("$_e")
            fi
        done

        local DEP_FLAG=""
        [ -n "${FIRST_JOB_DEP:-}" ] && DEP_FLAG="--dependency=afterok:${FIRST_JOB_DEP}"

        local JOB_TAG="${ARM}"
        [ -n "${EVAL_ROUTING_MODE}" ] && JOB_TAG="${JOB_TAG}_${EVAL_ROUTING_MODE}"
        [ -n "${EVAL_ROUTING_TOPK:-}" ] && JOB_TAG="${JOB_TAG}_top${EVAL_ROUTING_TOPK}"

        local _SUBMITTED_ANY=0
        local -a _GROUP_JOBS=()

        submit_group() {
            local _cap="$1"; shift
            local -a _ids=("$@")
            [ ${#_ids[@]} -eq 0 ] && return 0

            local _joined=""
            for _i in "${_ids[@]}"; do
                _joined="${_joined:+${_joined}:}${_i}"
            done

            local _jid
            _jid=$(sbatch --parsable \
                ${DEP_FLAG} \
                --job-name="T${MODEL_TASK_ID}_ev${_ids[0]}-${_ids[-1]}_${JOB_TAG}" \
                --export=ALL,${BASE_EXPORT},eval_task_id=${_ids[0]},EVAL_TASK_IDS=${_joined},MAX_SAMPLES=${_cap} \
                "$SLURM_SCRIPT")

            for _i in "${_ids[@]}"; do
                save_eval_job_id "$MODEL_TASK_ID" "$_i" "$_jid"
                SUBMITTED_JOBS["${MODEL_TASK_ID}:${_i}"]="$_jid"
            done
            _GROUP_JOBS+=("$_jid")
            _SUBMITTED_ANY=1
            echo "     -> Job ${_jid}  [datasets ${_ids[*]} | Samples: ${_cap}]  dep=[${DEP_FLAG#--dependency=}]"
            return 0
        }

        submit_group "${MAX_SAMPLES:-500}" "${GRP_STD[@]}"
        submit_group "${MAX_SAMPLES_GEN:-100}" "${GRP_GEN[@]}"

        if [ "$_SUBMITTED_ANY" -eq 0 ]; then
            echo "     [SKIP] nothing to evaluate"
            return 0
        fi

        if [[ "$SUBMIT_MODE" == "sequential" ]]; then
            for _jid in "${_GROUP_JOBS[@]}"; do
                wait_for_job "$_jid" "T${MODEL_TASK_ID} eval group ${_jid}" || return 1
            done
        fi
        return 0
    fi

    for EVAL_ID in "${EVAL_IDS[@]}"; do
        if (( EVAL_ID > MODEL_TASK_ID )) && [[ "${ALLOW_FUTURE_EVAL:-False}" != "True" ]]; then
            echo "     [SKIP] dataset T${EVAL_ID} is ahead of model T${MODEL_TASK_ID} (ALLOW_FUTURE_EVAL=True to force)"
            continue
        fi

        local EVAL_TASK_NAME="${TASK_ARRAY[$((EVAL_ID - 1))]}"

        # Generation tasks (MeetingBank, 20Minuten) are slow — fewer samples.
        local CURRENT_MAX_SAMPLES="${MAX_SAMPLES:-}"
        if [ -z "$CURRENT_MAX_SAMPLES" ]; then
            CURRENT_MAX_SAMPLES=500
            if sos_is_gen_task "$EVAL_TASK_NAME"; then
                CURRENT_MAX_SAMPLES="${MAX_SAMPLES_GEN:-100}"
            fi
        fi

        local DEP_FLAG=""
        if [[ "$SUBMIT_MODE" == "chain" ]]; then
            if [ -n "$LAST_EVAL_JOB_ID" ]; then
                DEP_FLAG="--dependency=afterok:${LAST_EVAL_JOB_ID}"
            elif [ -n "${FIRST_JOB_DEP:-}" ]; then
                DEP_FLAG="--dependency=afterok:${FIRST_JOB_DEP}"
            fi
        elif [ -z "${_SEQ_STARTED:-}" ] && [ -n "${FIRST_JOB_DEP:-}" ]; then
            DEP_FLAG="--dependency=afterok:${FIRST_JOB_DEP}"
            _SEQ_STARTED=1
        fi

        local JOB_TAG="${ARM}"
        [ -n "${EVAL_ROUTING_MODE}" ] && JOB_TAG="${JOB_TAG}_${EVAL_ROUTING_MODE}"

        local JOB_ID
        JOB_ID=$(sbatch --parsable \
            ${DEP_FLAG} \
            --job-name="T${MODEL_TASK_ID}_ev${EVAL_ID}_${EVAL_TASK_NAME}_${JOB_TAG}" \
            --export=ALL,${BASE_EXPORT},eval_task_id=${EVAL_ID},MAX_SAMPLES=${CURRENT_MAX_SAMPLES} \
            "$SLURM_SCRIPT")

        save_eval_job_id "$MODEL_TASK_ID" "$EVAL_ID" "$JOB_ID"
        SUBMITTED_JOBS["${MODEL_TASK_ID}:${EVAL_ID}"]="$JOB_ID"
        LAST_EVAL_JOB_ID="$JOB_ID"

        echo "     -> Job ${JOB_ID}  [Model T${MODEL_TASK_ID} (${TASK_NAME}) → Dataset T${EVAL_ID}: ${EVAL_TASK_NAME} | Samples: ${CURRENT_MAX_SAMPLES}]  dep=[${DEP_FLAG#--dependency=}]"

        if [[ "$SUBMIT_MODE" == "sequential" ]]; then
            if ! wait_for_job "$JOB_ID" "T${MODEL_TASK_ID} eval on T${EVAL_ID} (${EVAL_TASK_NAME})"; then
                SUBMITTED_JOBS["${MODEL_TASK_ID}:${EVAL_ID}"]="${JOB_ID}!"
                [[ "$STOP_ON_FAIL" == "True" ]] && return 1
                echo "        [continue] STOP_ON_FAIL=False — moving to the next eval"
            fi
            LAST_EVAL_JOB_ID=""
        fi
    done
    return 0
}

for MODEL_TASK_ID in "${MODEL_TASK_IDS[@]}"; do
    if ! submit_eval_for_task "$MODEL_TASK_ID"; then
        echo "" >&2
        echo "============================================================" >&2
        echo "  ARM '${ARM}' (${ARM_DESC}) EVAL ABORTED at Model T${MODEL_TASK_ID}" >&2
        echo "  Arm dir: ${BASE_DIR}" >&2
        echo "  Fix the failure, then resume with:" >&2
        echo "    SUBMIT_MODE=${SUBMIT_MODE} ARM=${ARM} GATE_MODE=${GATE_MODE} ROUTER_REG=${ROUTER_REG} \\" >&2
        echo "      BLOCK_BUDGET_FRACTION=${BLOCK_BUDGET_FRACTION} TASK_ORDER=${TASK_ORDER} SEED=${SEED} \\" >&2
        echo "      ${EVAL_ROUTING_MODE:+EVAL_ROUTING_MODE=${EVAL_ROUTING_MODE} }PROJECTION_MODE=${PROJECTION_MODE} ABLATION=${ABLATION} \\" >&2
        echo "      bash ${0} ${MODEL_TASK_ID}-${MODEL_TASK_IDS[-1]} ${user_model_name} ${EVAL_TASK_RANGE}" >&2
        echo "============================================================" >&2
        exit 1
    fi
done

# ── Summary table ─────────────────────────────────────────────────────────────
TOTAL_JOBS=0
for MODEL_TASK_ID in "${MODEL_TASK_IDS[@]}"; do
    for (( D=1; D<=TOTAL_TASKS; D++ )); do
        [[ -v SUBMITTED_JOBS["${MODEL_TASK_ID}:${D}"] ]] && TOTAL_JOBS=$(( TOTAL_JOBS + 1 ))
    done
done

echo ""
echo "============================================================"
echo "  ${TOTAL_JOBS} eval job(s) across ${#MODEL_TASK_IDS[@]} checkpoint(s)"
echo "  Arm ${ARM} (${ARM_DESC}), routing ${EVAL_ROUTING_MODE:-learned}"
echo ""

printf "  %-6s" "Model"
for (( D=1; D<=TOTAL_TASKS; D++ )); do printf "  %-13s" "D${D}:${TASK_ARRAY[$((D-1))]}"; done
echo ""
printf "  %-6s" "-----"
for (( D=1; D<=TOTAL_TASKS; D++ )); do printf "  %-13s" "-------------"; done
echo ""

for MODEL_TASK_ID in "${MODEL_TASK_IDS[@]}"; do
    printf "  %-6s" "T${MODEL_TASK_ID}"
    for (( D=1; D<=TOTAL_TASKS; D++ )); do
        KEY="${MODEL_TASK_ID}:${D}"
        if [[ -v SUBMITTED_JOBS["$KEY"] ]]; then CELL="${SUBMITTED_JOBS[$KEY]}"; else CELL="-"; fi
        printf "  %-13s" "$CELL"
    done
    echo ""
done

echo ""
echo "  ('!' suffix = job did not end COMPLETED)"
echo "  Job IDs  : ${JOB_ID_DIR}/"
echo "  Results  : ${BASE_DIR}/shared_moE_checkpoints_${user_model_name}_<N>_<task>/eval_task_<J>_<name>${EVAL_OUT_SUFFIX}/"
echo "  Metrics  : python ${PROJECT_ROOT}/cl_metrics.py --base_dir ${BASE_DIR} \\"
echo "               --model ${user_model_name} --task_list \"${TASK_LIST_STR}\" --last ${MODEL_TASK_IDS[-1]} [--grid]"
echo "             (run after the jobs finish; LA/OP/BWT from diagonal + final row)"
echo "  Analyse  : bash ${PROJECT_ROOT}/analysis/run_analysis.sh"
echo "  Monitor  : squeue -u \$USER --format='%.10i %.34j %.8T %R'"
echo "============================================================"



# FIRST_JOB_DEP=16582644 BLOCK_BUDGET_FRACTION=0.12 ABLATION=1 SETAFIX_NULL=empirical SETAFIX_ORTHO=on SETAFIX_GATE_PROTECT=freeze SETAFIX_LAYER_GATE=budget SETAFIX_PHI=0.02 bash RL_Lamda_fsys_eval_only_arms.sh 8 qwen3_4b 1-7

# BLOCK_BUDGET_FRACTION=0.12 ABLATION=1 SETAFIX_NULL=empirical SETAFIX_ORTHO=on SETAFIX_GATE_PROTECT=freeze SETAFIX_LAYER_GATE=budget SETAFIX_PHI=0.015 bash RL_Lamda_fsys_eval_only_arms.sh 8 qwen3_4b 1-7
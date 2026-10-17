#!/bin/bash
# ============================================================================
# run_sos_ablation.sh — Split-on-Share ablations of the SEAM and regularization
# knobs on the TRACE 8-task stream, 500-sample set.
# Self-contained launcher, based on run_setafix_order.sh (not called or sourced).
# Each arm is ONE GPU job (task 1 -> 8, then task 8 on tasks 1-7 + LA/OP/BWT),
# submitted with setafix_pipeline_streams_ablation.slurm.
#   FFT    : fixed 50 steps (FFT_STEPS=50, FFT_MODE=fixed); FFT_MODE=dynamic
#            stops by itself with FFT_STEPS as the ceiling
#
#   Stream : TASK_ORDER=4 = C-STANCE, FOMC, MeetingBank, Py150, ScienceQA,
#            NumGLUE-cm, NumGLUE-ds, 20Minuten (TRACE order, no duplicate task)
#   Data   : DATASET=500 from fatemask/SETAV2/data_files/TRACE-Benchmark
#   Seed   : 100 (SEED="100 42" for both)
#   Base   : Z=3.0  PHI=0.012  SHARE_Q=1.0  LAMDA=0.1 dynamic  merge off
#
# Arms, one knob changed from base each (16 in total):
#   base  : base
#   z     : SETAFIX_Z       1.0 2.0 4.0            SEAM agreement bar
#   phi   : SETAFIX_PHI     0.006 0.025 0.04       SEAM size floor
#   q     : SETAFIX_SHARE_Q 0.5 0.75               SEAM shared fraction kept
#   lam   : LAMDA           0.0 0.05 0.2 0.5       regularizer (0 = off)
#   merge : MERGE_UTIL_Z    0.0 1.0 2.0            low-utility merge ON
# SETAFIX_Z and MERGE_UTIL_Z are not in the run-folder name, so those arms carry
# a RUN_TAG (z20, mz05, ...); the other knobs are tagged by the launcher.
#
# Usage (from the project root; args = task range, steps, model). Submit ONE
# group per command; jobs go to jannesar-lab / nova (128G, 5 days) by
# default. Every job is submitted with no dependency:
#   ABL_GROUPS="base z" bash scripts/run_sos_ablation.sh      # 1st: base + Z (4 jobs)
#   ABL_GROUPS="phi q"  bash scripts/run_sos_ablation.sh      # 2nd: PHI + SHARE_Q (5)
#   ABL_GROUPS="lam"    bash scripts/run_sos_ablation.sh      # 3rd: lambda (4)
#   ABL_GROUPS="merge"  bash scripts/run_sos_ablation.sh      # 4th: merge (3)
#   ARMS="base lam0" bash scripts/run_sos_ablation.sh         # chosen arms by label
#   DRY_RUN=1 ABL_GROUPS="z" bash scripts/run_sos_ablation.sh # list, submit nothing
#   REPORT=1 bash scripts/run_sos_ablation.sh                 # LA / OP / BWT table
# After a timeout rerun the same command with FORCE_OVERWRITE=True: finished
# tasks are skipped.
# ============================================================================

set -euo pipefail

# ── Ablation arms: one GPU job per arm ───────────────────────────────────────
# Without ABL_ARM this part submits the selected arms, each by re-running this
# launcher with ABL_ARM=<label> and the arm's settings; with ABL_ARM set it
# falls through and launches that one arm.
ABL_MANIFEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/logs/ablation_manifest.txt"
ABL_TABLE=(
  "base|base|"
  "z|z10|SETAFIX_Z=1.0 RUN_TAG=z10"
  "z|z20|SETAFIX_Z=2.0 RUN_TAG=z20"
  "z|z40|SETAFIX_Z=4.0 RUN_TAG=z40"
  "phi|phi006|SETAFIX_PHI=0.006"
  "phi|phi025|SETAFIX_PHI=0.025"
  "phi|phi040|SETAFIX_PHI=0.04"
  "q|q050|SETAFIX_SHARE_Q=0.5"
  "q|q075|SETAFIX_SHARE_Q=0.75"
  "lam|lam0|LAMDA=0.0"
  "lam|lam005|LAMDA=0.05"
  "lam|lam02|LAMDA=0.2"
  "lam|lam05|LAMDA=0.5"
  "merge|mz00|MERGE_LOW_UTIL=1 MERGE_UTIL_Z=0.0 RUN_TAG=mz00"
  "merge|mz10|MERGE_LOW_UTIL=1 MERGE_UTIL_Z=1.0 RUN_TAG=mz10"
  "merge|mz20|MERGE_LOW_UTIL=1 MERGE_UTIL_Z=2.0 RUN_TAG=mz20"
)
if [ -z "${ABL_ARM:-}" ]; then
    if [ "${REPORT:-0}" == "1" ]; then
        [ -f "$ABL_MANIFEST" ] || { echo "No arms submitted yet (${ABL_MANIFEST} not found)."; exit 0; }
        printf "%-8s %-10s %-6s %-30s %8s %8s %9s\n" "arm" "model" "seed" "setting" "LA" "OP" "BWT"
        while IFS='|' read -r _l _m _s _o _d; do
            _f="${_d}/cl_metrics_T8.json"
            if [ -f "$_f" ]; then
                python3 -c "import json,sys;r=json.load(open(sys.argv[1]));g=lambda k:'%.4f'%r[k] if r.get(k) is not None else '---';print('%-8s %-10s %-6s %-30s %8s %8s %9s'%(sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5] or '-',g('learning_accuracy'),g('overall_performance'),g('backward_transfer')))" "$_f" "$_l" "$_m" "$_s" "$_o"
            else
                printf "%-8s %-10s %-6s %-30s %s\n" "$_l" "$_m" "$_s" "${_o:--}" "not finished"
            fi
        done < "$ABL_MANIFEST"
        exit 0
    fi
    # One group per command: submitting everything at once is not allowed.
    if [ -z "${ABL_GROUPS:-}" ] && [ -z "${ARMS:-}" ]; then
        echo "Pick a group (or arms) to submit, e.g.  ABL_GROUPS=\"base z\" bash $0" >&2
        echo "  groups: base z phi q lam merge   (labels: ARMS=\"lam0 mz10\")" >&2
        exit 1
    fi
    _groups="${ABL_GROUPS:-}"
    # Lab allocation (fatemask) by default; each can be overridden on the command line.
    export SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-jannesar-lab}"
    export SBATCH_PARTITION="${SBATCH_PARTITION:-nova}"
    export SBATCH_MEM_PER_NODE="${SBATCH_MEM_PER_NODE:-128G}"
    export ONEJOB_TIME="${ONEJOB_TIME:-5-00:00:00}"   # same as setafix_all_tasks_order.slurm
    echo "  [submit] account=${SBATCH_ACCOUNT} partition=${SBATCH_PARTITION} mem=${SBATCH_MEM_PER_NODE} time=${ONEJOB_TIME}"
    _n=0
    for _row in "${ABL_TABLE[@]}"; do
        IFS='|' read -r _g _l _o <<< "$_row"
        if [ -n "${ARMS:-}" ]; then [[ " ${ARMS} " == *" ${_l} "* ]] || continue
        else [[ " ${_groups} " == *" ${_g} "* ]] || continue; fi
        _n=$((_n + 1))
        echo ""
        echo "################ arm ${_l}  (${_o:-base settings}) ################"
        if [ -n "${DRY_RUN:-}" ]; then echo "[dry] ABL_ARM=${_l} ${_o} bash ${BASH_SOURCE[0]} $*"; continue; fi
        # Base values first, then the arm's override, so leftovers in the shell
        # can never leak into an arm.
        _out=$(env -u SETAFIX_MIN_SHARED_FRAC_NEW -u TASK_LIST_STR \
            SETAFIX_Z=3.0 SETAFIX_PHI=0.012 SETAFIX_SHARE_Q=1.0 LAMDA=0.1 LAMBDA_MODE=dynamic \
            MERGE_LOW_UTIL=0 MERGE_UTIL_Z=1.0 RUN_TAG= \
            ABL_ARM="$_l" ABL_OVERRIDES="$_o" $_o \
            bash "${BASH_SOURCE[0]}" "$@" 2>&1) || { echo "$_out"; exit 1; }
        echo "$_out"
    done
    [ "$_n" -gt 0 ] || { echo "ERROR: no arm matches ABL_GROUPS='${_groups}' ARMS='${ARMS:-}'" >&2; exit 1; }
    echo ""
    echo "${_n} arm(s) $([ -n "${DRY_RUN:-}" ] && echo listed || echo submitted).  Results: REPORT=1 bash scripts/run_sos_ablation.sh"
    exit 0
fi

# ── Seeds: one GPU job per seed ──────────────────────────────────────────────
# SEEDS (or SEED) may list several seeds, e.g. SEED="100 42". Default: 100.
# Each seed re-runs this launcher with that one seed, so every seed gets its
# own GPU job and its own output folder (the seed is part of the tag).
SEEDS="${SEEDS:-${SEED:-100}}"
if [[ "$(echo $SEEDS)" == *" "* ]]; then
    for _seed in $SEEDS; do
        echo ""
        echo "################ seed ${_seed} ################"
        SEEDS="$_seed" SEED="$_seed" bash "${BASH_SOURCE[0]}" "$@" || exit $?
    done
    exit 0
fi
export SEED="$(echo $SEEDS)"

# ── Arguments ─────────────────────────────────────────────────────────────────
TASK_RANGE="${1:-all}"
STEPS_TO_RUN="${2:-fft,split,train,eval}"
user_model_name="${3:-qwen3_4b}"

# ── Stream defaults: the settings of the current SoS runs ────────────────────
# Each one can still be overridden in front of the command.
export ABLATION="${ABLATION:-1}"
export BLOCK_BUDGET_FRACTION="${BLOCK_BUDGET_FRACTION:-0.12}"
export SETAFIX_NULL="${SETAFIX_NULL:-empirical}"
export SETAFIX_ORTHO="${SETAFIX_ORTHO:-on}"
export SETAFIX_GATE_PROTECT="${SETAFIX_GATE_PROTECT:-freeze}"
export SETAFIX_LAYER_GATE="${SETAFIX_LAYER_GATE:-budget}"
export SETAFIX_PHI="${SETAFIX_PHI:-0.012}"

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
export SETAFIX_PHI="${SETAFIX_PHI:-0.020}"
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
#       once the direction test is on; see SETA_SoS.py) ---
#   SETAFIX_LAYER_GATE  phi (default) | relative (Sept 18 behaviour) | absolute | off
#       phi: m = PHI x new task's blocks in the layer, straight, no rounding;
#            carve iff aligned >= m and leftover is 0 or >= m; tiny-remainder
#            merge off. LAYER_FRAC / LAYER_MIN are used by relative mode only.
#   SETAFIX_LAYER_FRAC  fraction of the layer's pre-validation overlap (relative)
#   SETAFIX_LAYER_MIN   hard minimum, so a 2-block layer cannot split (relative)
#       budget: T = PHI x N_base (N_base = BLOCK_BUDGET_FRACTION x all v_proj
#            blocks, 691 for qwen3_4b at 0.12; PHI=0.02 -> T = 13.8). Per layer,
#            after the direction test, split only if: shared >= T, old expert's
#            leftover 0 or >= T, new task's own blocks 0 or >= T. Layers where a
#            task selected < T blocks are not used. No expert piece below T is
#            ever created. No remainder merge, no whole-expert check.
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
# [Split-on-Share] PROJECT_ROOT (this folder's parent), DATASET -> BASE_DATA_PATH,
# task orders, LEARNING_RATE and the folder layout come from sos_common.sh.
# ── Shared configuration (paths, datasets, task-order helpers) ──────────────
# Copied from sos_common_order.sh (paths, datasets, task-order helpers).
# Shared configuration, sourced by the launchers in scripts/.
# Only paths, dataset / task-order tables and naming live here. Nothing in this
# file changes what the trainer computes.

# ── Folder layout ─────────────────────────────────────────────────────────────
#   <PROJECT_ROOT>/                   python code (trainer, model, helpers)
#   <PROJECT_ROOT>/scripts/           launchers + slurm scripts (this folder)
#   <PROJECT_ROOT>/analysis/          analysis-only code; its logs in analysis/logs
#   <PROJECT_ROOT>/FFT_steps/         every FFT_steps_* directory
#   <PROJECT_ROOT>/trained_models/    every RL_Lamda_fsys_trained_models_SoS_* directory
#   <PROJECT_ROOT>/logs/              training / eval job logs
SOS_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SOS_SCRIPTS_DIR}/.." && pwd)}"
SOS_MODELS_DIR="${PROJECT_ROOT}/trained_models"
SOS_FFT_ROOT="${PROJECT_ROOT}/FFT_steps"
SOS_LOG_ROOT="${PROJECT_ROOT}/logs"
export PROJECT_ROOT SOS_FFT_ROOT SOS_LOG_ROOT

# ── Datasets ──────────────────────────────────────────────────────────────────
#   DATASET=5000  TRACE, 5000-sample set, 8 tasks            (default = current runs)
#   DATASET=500   500-sample set: the 8 TRACE tasks + 7 more (15 tasks)
# BASE_DATA_PATH, if set, overrides DATASET (same as before).
DATA_ROOT="${DATA_ROOT:-/lustre/hdd/LAS/jannesar-lab/fatemask/SETAV2/data_files/TRACE-Benchmark}"
DATA_PATH_5000="${DATA_PATH_5000:-${DATA_ROOT}/LLM-CL-Benchmark_5000}"
DATA_PATH_500="${DATA_PATH_500:-${DATA_ROOT}/LLM-CL-Benchmark_500}"
if [ -z "${BASE_DATA_PATH:-}" ]; then
    DATASET="${DATASET:-500}"
    case "$DATASET" in
        5000) BASE_DATA_PATH="${DATA_PATH_5000}" ;;
        500)  BASE_DATA_PATH="${DATA_PATH_500}" ;;
        *) echo "ERROR: DATASET must be 500 or 5000 (got '${DATASET}'), or set BASE_DATA_PATH" >&2; exit 1 ;;
    esac
else
    DATASET="custom"
fi
# The generation tasks always read the 500-sample set (existing rule, see
# setafix_pipeline.slurm). Kept here only so the preflight check looks there.
GEN_DATA_PATH="${DATA_PATH_500}"
export BASE_DATA_PATH DATASET

# Generation tasks: fewer eval samples, 500-sample data path. Existing rule;
# it is now matched by task NAME so it follows the task wherever an order puts
# it (for TASK_ORDER=1 these are positions 3 and 6, exactly as before).
sos_is_gen_task() {
    case "$1" in MeetingBank|20Minuten) return 0 ;; *) return 1 ;; esac
}

# ── Task orders ───────────────────────────────────────────────────────────────
#   1-3  15-task lists (unchanged). Run 1-8 of order 1 for the 8 TRACE tasks,
#        1-15 for the 15-task set (DATASET=500).
#   4    8 TRACE tasks in the TRACE-paper order
#   5    8 TRACE tasks, order 1 reversed
#   TASK_LIST_STR=<a,b,c,...> gives any custom order (tagged ordc<hash>).
sos_resolve_task_order() {
    if [[ -n "${TASK_LIST_STR:-}" ]]; then
        ORDER_TAG="ordc$(printf '%s' "$TASK_LIST_STR" | md5sum | cut -c1-6)"
        TASK_ORDER="custom"
    else
        TASK_ORDER="${TASK_ORDER:-1}"
        case "$TASK_ORDER" in
            1) TASK_LIST_STR="C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,Py150,NumGLUE-ds,20Minuten,dbpedia,amazon,yahoo,agnews,yelp,BoolQA,QQP" ;;
            2) TASK_LIST_STR="QQP,BoolQA,yelp,agnews,yahoo,amazon,dbpedia,NumGLUE-ds,Py150,20Minuten,NumGLUE-cm,ScienceQA,MeetingBank,FOMC,C-STANCE" ;;
            3) TASK_LIST_STR="20Minuten,Py150,NumGLUE-ds,dbpedia,amazon,yahoo,agnews,yelp,BoolQA,QQP,C-STANCE,FOMC,MeetingBank,ScienceQA,NumGLUE-cm" ;;
            4) TASK_LIST_STR="C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten" ;;
            5) TASK_LIST_STR="NumGLUE-ds,Py150,20Minuten,NumGLUE-cm,ScienceQA,MeetingBank,FOMC,C-STANCE" ;;
            *) echo "ERROR: TASK_ORDER must be 1-5 — or set TASK_LIST_STR directly (got '${TASK_ORDER}')" >&2; exit 1 ;;
        esac
        ORDER_TAG="ord${TASK_ORDER}"
    fi
    export TASK_LIST_STR
}

# ── Learning rate ─────────────────────────────────────────────────────────────
# LEARNING_RATE -> --learning_rate. The trainer uses it for Phase-2 training and
# for FFT on non-generation tasks (generation tasks keep their fixed 2e-6 FFT
# rate inside the trainer). Default = the value every existing run used.
# A non-default value gets its own tree (_lr<value>), like LAMDA does.
LEARNING_RATE="${LEARNING_RATE:-9.65e-6}"
export LEARNING_RATE
sos_lr_tag() {
    if [[ "$LEARNING_RATE" != "9.65e-6" ]]; then
        PHASE2_TAG="${PHASE2_TAG}_lr${LEARNING_RATE}"
    fi
}

# ── Projection aliases (canonical names are what land in the directory) ──────
sos_canonical_projection() {
    case "$PROJECTION_MODE" in
        v_proj_only|q_proj|k_proj|qkv_proj|mlp_proj) ;;
        v_proj)                    PROJECTION_MODE="v_proj_only" ;;
        vqk_proj|qvk_proj|qkv|vqk) PROJECTION_MODE="qkv_proj" ;;
        mlp)                       PROJECTION_MODE="mlp_proj" ;;
        *) echo "ERROR: PROJECTION_MODE must be one of v_proj_only|q_proj|k_proj|qkv_proj|mlp_proj (got '${PROJECTION_MODE}')" >&2; exit 1 ;;
    esac
}

# ── Preflight: every task of the run must have its data folder ───────────────
# Prints what is actually on disk when something is missing.
sos_check_task_data() {   # args: task names
    local _missing=0 _t _p
    for _t in "$@"; do
        _p="${BASE_DATA_PATH}"
        sos_is_gen_task "$_t" && _p="${GEN_DATA_PATH}"
        if [ ! -f "${_p}/${_t}/train.json" ] && [ ! -f "${_p}/${_t}/test.json" ]; then
            echo "ERROR: no data for task '${_t}' under ${_p}/${_t}" >&2
            _missing=1
        fi
    done
    if [ "$_missing" -eq 1 ]; then
        echo "       Task folders present in ${BASE_DATA_PATH}:" >&2
        ls -1 "${BASE_DATA_PATH}" 2>/dev/null | sed 's/^/         /' >&2 || echo "         (path does not exist)" >&2
        echo "       (DATASET=${DATASET}; the 15-task set needs DATASET=500)" >&2
        exit 1
    fi
}
# ── end of shared configuration ─────────────────────────────────────────────
# [PIN] One code location: SETAV2 root. No directory variable, no fallback.
TRAINER_PY="${PROJECT_ROOT}/main.py"
MODEL_PY="${PROJECT_ROOT}/SETA_SoS.py"
for _f in "$TRAINER_PY" "$MODEL_PY"; do
    if [ ! -f "$_f" ]; then
        echo "ERROR: required Aug20 file missing: $_f" >&2
        exit 1
    fi
done
if ! grep -q "from SETA_SoS import" "$TRAINER_PY"; then
    echo "ERROR: $TRAINER_PY does not import SETA_SoS -- stale trainer copy." >&2
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
SLURM_SCRIPT="${SCRIPT_DIR}/setafix_pipeline_streams_ablation.slurm"
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
sos_canonical_projection   # v_proj -> v_proj_only, vqk_proj -> qkv_proj, ... (shared configuration above)
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
    1)  ABLATION_TAG="order_full_system"
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
# [ABLATION] FFT probe: a FIXED number of steps by default (FFT_STEPS=50), same
# for every arm. FFT_MODE=dynamic keeps the self-calibrated stop with FFT_STEPS
# as its ceiling. Both are tagged (_fftfix50 / _fftdyn50) so runs never mix.
FFT_STOPPING_MODE="${FFT_MODE:-fixed}"
export FFT_STEPS="${FFT_STEPS:-50}"
case "$FFT_STOPPING_MODE" in fixed|dynamic) ;; *) echo "ERROR: FFT_MODE must be fixed or dynamic" >&2; exit 1 ;; esac

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
BUDGET_EXPANSION_CAP="${BUDGET_EXPANSION_CAP:-1.0}"     # [FIX-6] 1.0 = never expand
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
if [[ "$FINAL_EVAL" == "True" ]]; then
    echo "ERROR: FINAL_EVAL=True is not supported here; the job already runs the final row + metrics" >&2; exit 1
fi
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
printf "│  %-68s│\n" "LEARNING_RATE        = ${LEARNING_RATE}"
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
SUBMIT_MODE="onejob"   # one GPU job runs every task of the stream
WAIT_FOR_EVAL="${WAIT_FOR_EVAL:-False}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"       # seconds between squeue polls

# SUBMIT_MODE=onejob     ONE slurm job runs every task in the range back to back
#                        (fft,split,train,eval per task = the diagonal), then the
#                        final checkpoint on all earlier tasks (the final row),
#                        then prints/saves LA, OP, BWT. See setafix_all_tasks.slurm.
case "$SUBMIT_MODE" in
    chain|sequential|onejob) ;;
    *) echo "ERROR: SUBMIT_MODE must be 'chain', 'sequential' or 'onejob' (got '${SUBMIT_MODE}')" >&2; exit 1 ;;
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
sos_lr_tag   # non-default LEARNING_RATE -> _lr<value> (shared configuration above)
# [SEED] moved into RUN_SIG below, so all four run knobs sit in one segment.
if [[ -n "${RUN_TAG:-}" ]]; then
    PHASE2_TAG="${PHASE2_TAG}_${RUN_TAG}"
fi

# ── [TASK ORDER] Named permutations of the task stream ─────────────────────
# TASK_ORDER=1 is the historical default, so nothing already on disk moves.
# Passing TASK_LIST_STR directly still works and gets tagged 'ordc<hash>', so a
# hand-rolled order can never collide with a named one.
# ── [ABLATION] Task stream ──────────────────────────────────────────────────
# TASK_ORDER=4 = the TRACE 8-task order without the duplicated Py150 of order 1.
unset TASK_LIST_STR
export TASK_ORDER="${TASK_ORDER:-4}"
sos_resolve_task_order   # TASK_ORDER 1-5 or TASK_LIST_STR -> TASK_LIST_STR, ORDER_TAG (shared configuration above)

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
[ "${SETAFIX_LAYER_GATE}" == "budget" ]   && ABLATION_TAG="${ABLATION_TAG}_lgbud${SETAFIX_PHI//./}"
[ "${SETAFIX_GATE_PROTECT}" == "freeze" ] && ABLATION_TAG="${ABLATION_TAG}_gatefz"
[ "${STRUCTURE_ONLY}" == "1" ] && ABLATION_TAG="${ABLATION_TAG}_structonly"
ABLATION_TAG="${ABLATION_TAG}_fft${FFT_STOPPING_MODE:0:3}${FFT_STEPS}"   # [ABLATION] FFT probe length
echo "  [arm] ABLATION_TAG = ${ABLATION_TAG}"
echo "  [sig] bbf=${BLOCK_BUDGET_FRACTION}  order=${TASK_ORDER}  proj=${PROJECTION_MODE}  seed=${SEED}"
echo "  [seam] phi=${SETAFIX_PHI} split=${SETAFIX_SPLIT} null=${SETAFIX_NULL} layer_gate=${SETAFIX_LAYER_GATE} (frac=${SETAFIX_LAYER_FRAC} min=${SETAFIX_LAYER_MIN}) ect=${EXPERT_CREATION_THRESHOLD}"

BASE_DIR="${SOS_MODELS_DIR}/RL_Lamda_fsys_trained_models_SoS_${user_model_name}_${PROJECTION_MODE}/ablation_${ABLATION_TAG}"
# [ABLATION] record arm -> run folder for REPORT=1 (one line per arm/model/seed)
mkdir -p "$(dirname "$ABL_MANIFEST")"
{ grep -v "^${ABL_ARM}|${user_model_name}|${SEED}|" "$ABL_MANIFEST" 2>/dev/null || true; \
  echo "${ABL_ARM}|${user_model_name}|${SEED}|${ABL_OVERRIDES:-}|${BASE_DIR}"; } > "${ABL_MANIFEST}.tmp" && mv "${ABL_MANIFEST}.tmp" "$ABL_MANIFEST"

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


NEW_PARAMS="PROJECTION_MODE=${PROJECTION_MODE},PROJ_CALIB_THRESHOLD=${PROJ_CALIB_THRESHOLD},PROJ_CALIB_STEPS=${PROJ_CALIB_STEPS},BLOCK_BUDGET_FRACTION=${BLOCK_BUDGET_FRACTION},GRAD_TEMPORAL_DECAY=${GRAD_TEMPORAL_DECAY},GLOBAL_EXPERT_MIN_BLOCKS=${GLOBAL_EXPERT_MIN_BLOCKS},EXPERT_CREATION_THRESHOLD=${EXPERT_CREATION_THRESHOLD},ADAPTIVE_FFT_STEPS=${ADAPTIVE_FFT_STEPS},FFT_MIN_STEPS=${FFT_MIN_STEPS},FFT_MAX_STEPS=${FFT_MAX_STEPS},FFT_REF_SAMPLES=${FFT_REF_SAMPLES},NOVELTY_BETA_MIN=${NOVELTY_BETA_MIN},NOVELTY_T_MAX=${NOVELTY_T_MAX},FFT_STOPPING_MODE=${FFT_STOPPING_MODE},LAYER_NORM_SCORING=${LAYER_NORM_SCORING},DIVERGENCE_SCORING=${DIVERGENCE_SCORING},BUDGET_MODE=${BUDGET_MODE},ABLATION_TAG=${ABLATION_TAG},LAMDA=${LAMDA},LAMBDA_MODE=${LAMBDA_MODE},GATING_RL=${GATING_RL},NO_EXPERT_SHARING=${NO_EXPERT_SHARING:-False},EVAL_ROUTING_TOPK=${EVAL_ROUTING_TOPK},GATE_MODE=${GATE_MODE},GATE_OPEN_BIAS=${GATE_OPEN_BIAS},USE_REENTRANT=${USE_REENTRANT},BUDGET_EXPANSION_CAP=${BUDGET_EXPANSION_CAP},LABEL_AUDIT_STRICT=${LABEL_AUDIT_STRICT},LABEL_AUDIT_SAMPLES=${LABEL_AUDIT_SAMPLES},LOSS_SANITY_OFF=${LOSS_SANITY_OFF},PHASE2_TAG=${PHASE2_TAG},ROUTER_REG=${ROUTER_REG},ROUTER_ANCHOR_MU=${ROUTER_ANCHOR_MU},ROUTER_N_CENTROIDS=${ROUTER_N_CENTROIDS},ROUTER_CENTROID_BATCHES=${ROUTER_CENTROID_BATCHES},FFT_TAG=${FFT_TAG},EXCLUDE_MODE=${EXCLUDE_MODE},SOFT_EXCLUDE_ALPHA=${SOFT_EXCLUDE_ALPHA},CAPACITY_FLOOR=${CAPACITY_FLOOR},MERGE_LOW_UTIL=${MERGE_LOW_UTIL},MERGE_GATE_THRESHOLD=${MERGE_GATE_THRESHOLD},MERGE_CAND_MAX_BLOCKS=${MERGE_CAND_MAX_BLOCKS},MERGE_MAX_LAYERS=${MERGE_MAX_LAYERS},MERGE_PROTECT_LATEST=${MERGE_PROTECT_LATEST},MERGE_UTIL_MODE=${MERGE_UTIL_MODE},MERGE_UTIL_Z=${MERGE_UTIL_Z},MERGE_UTIL_MIN_SPREAD=${MERGE_UTIL_MIN_SPREAD},EVAL_ROUTING_MODE=${EVAL_ROUTING_MODE},EVAL_OUT_SUFFIX=${EVAL_OUT_SUFFIX},GATE_PROBE=${GATE_PROBE},SEED=${SEED},TASK_ORDER=${TASK_ORDER},ORDER_TAG=${ORDER_TAG},SETAFIX_STRUCTURE_ONLY=${SETAFIX_STRUCTURE_ONLY},SETAFIX_SPLIT=${SETAFIX_SPLIT},SETAFIX_SCORE=${SETAFIX_SCORE},SETAFIX_Z=${SETAFIX_Z},SETAFIX_MIN_SHARED_FRAC=${SETAFIX_MIN_SHARED_FRAC},SETAFIX_NOEVIDENCE=${SETAFIX_NOEVIDENCE},SETAFIX_SIDE=${SETAFIX_SIDE},SETAFIX_NULL=${SETAFIX_NULL},SETAFIX_ORTHO=${SETAFIX_ORTHO},SETAFIX_GATE_PROTECT=${SETAFIX_GATE_PROTECT},SETAFIX_SHARE_Q=${SETAFIX_SHARE_Q},SETAFIX_MIN_SHARED_FRAC_NEW=${SETAFIX_MIN_SHARED_FRAC_NEW},SETAFIX_MAX_SHARED_PER_TASK=${SETAFIX_MAX_SHARED_PER_TASK},SETAFIX_LAYER_GATE=${SETAFIX_LAYER_GATE},SETAFIX_LAYER_FRAC=${SETAFIX_LAYER_FRAC},SETAFIX_LAYER_MIN=${SETAFIX_LAYER_MIN},SETAFIX_PHI=${SETAFIX_PHI},PROJECT_ROOT=${PROJECT_ROOT},SOS_FFT_ROOT=${SOS_FFT_ROOT},LEARNING_RATE=${LEARNING_RATE},GEN_DATA_PATH=${GEN_DATA_PATH}"

# TASK_LIST_STR is resolved and exported by the TASK ORDER block above.

IFS=',' read -r -a TASK_ARRAY <<< "$TASK_LIST_STR"
TOTAL_TASKS=${#TASK_ARRAY[@]}
[[ "$TASK_RANGE" == "all" ]] && TASK_RANGE="1-${TOTAL_TASKS}"

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
_RUN_TASK_NAMES=(); for _i in "${TASK_IDS[@]}"; do _RUN_TASK_NAMES+=("${TASK_ARRAY[$((_i - 1))]}"); done
sos_check_task_data "${_RUN_TASK_NAMES[@]}"

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
        if sos_is_gen_task "$TASK_NAME"; then
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

# ── [ONEJOB] every task of the range in ONE slurm allocation ─────────────────
# Same settings, tags, paths and python command as the per-task jobs: the
# driver (setafix_all_tasks.slurm) runs setafix_pipeline.slurm once per task
# with bash, so nothing about a task's run changes except that it no longer
# waits in the queue. Afterwards it evaluates the final checkpoint on every
# earlier task (final row) and computes LA / OP / BWT with cl_metrics.py.
#   ONEJOB_TIME   walltime for the whole stream (default 7-00:00:00; your
#                 partition limit applies)
#   SKIP_DONE     True (default): a task whose outputs already exist is skipped,
#                 so rerunning the same command after a timeout continues where
#                 it stopped (needs FORCE_OVERWRITE=True to pass the preflight)
#   FINAL_ROW     True (default): final checkpoint on tasks 1..N-1 + metrics
if [[ "$SUBMIT_MODE" == "onejob" ]]; then
    ALL_SLURM="${SLURM_SCRIPT}"   # same file; SOS_STREAM_DRIVER=1 selects its driver part
    [ -f "$ALL_SLURM" ] || { echo "ERROR: $ALL_SLURM not found" >&2; exit 1; }
    _IDS_STR="$(IFS=:; echo "${TASK_IDS[*]}")"
    _DEP_FLAG=""
    [ -n "${FIRST_JOB_DEP:-}" ] && _DEP_FLAG="--dependency=${FIRST_JOB_DEP_TYPE:-afterok}:${FIRST_JOB_DEP}"
    _ONE_EXPORT="user_model_name=${user_model_name},BASE_DIR=${BASE_DIR},TASK_IDS_STR=${_IDS_STR},STEPS=${STEPS_TO_RUN//,/:},STRUCTURE_ONLY=${STRUCTURE_ONLY},MAX_SAMPLES_OVERRIDE=${MAX_SAMPLES:-},MAX_SAMPLES_GEN=${MAX_SAMPLES_GEN:-100},train_sample_ratio=${train_sample_ratio},BASE_DATA_PATH=${BASE_DATA_PATH},SKIP_DONE=${SKIP_DONE:-True},FINAL_ROW=${FINAL_ROW:-True},PIPELINE_SLURM=${SLURM_SCRIPT},SOS_STREAM_DRIVER=1,${NEW_PARAMS}"
    JOB_ID=$(sbatch --parsable ${_DEP_FLAG} \
        ${ONEJOB_TIME:+--time=${ONEJOB_TIME}} \
        --job-name="abl_${ABL_ARM}_T${TASK_IDS[0]}-${TASK_IDS[-1]}_${user_model_name}_s${SEED}" \
        --export=ALL,${_ONE_EXPORT} \
        "$ALL_SLURM")
    save_job_id "${TASK_IDS[0]}-${TASK_IDS[-1]}" "onejob" "$JOB_ID"
    echo ""
    echo "============================================================"
    echo "  ONE job for tasks ${TASK_IDS[*]}: ${JOB_ID}  dep=[${_DEP_FLAG#--dependency=}]"
    echo "  Steps per task : ${STEPS_TO_RUN}"
    echo "  Final row      : ${FINAL_ROW:-True}  (T${TASK_IDS[-1]} on tasks 1-$(( ${TASK_IDS[-1]} - 1 )), then LA/OP/BWT)"
    echo "  Walltime       : ${ONEJOB_TIME:-slurm header}"
    echo "  Metrics        : ${BASE_DIR}/cl_metrics_T${TASK_IDS[-1]}.json"
    echo "  Logs           : ${SOS_LOG_ROOT}/${user_model_name}_${PROJECTION_MODE}/${ABLATION_TAG}/"
    echo "  Monitor        : squeue -u \$USER --format='%.10i %.34j %.8T %R'"
    echo "============================================================"
    exit 0
fi

# cd /lustre/hdd/LAS/jannesar-lab/fatemask/fed-con/Split-on-Share
# run:
# SETAFIX_SIGN=on RUN_TAG=SEAM ABL_GROUPS="base z" bash scripts/run_sos_ablation.sh    # 4 jobs: base, Z 1.0 / 2.0 / 4.0
# SETAFIX_SIGN=on RUN_TAG=SEAM ABL_GROUPS="phi q"  bash scripts/run_sos_ablation.sh    # 5 jobs: PHI 0.006 / 0.025 / 0.04, SHARE_Q 0.5 / 0.75


# need to run:
# run in fatemask: SETAFIX_SIGN=on RUN_TAG=SEAM ABL_GROUPS="lam"    bash scripts/run_sos_ablation.sh    # 4 jobs: λ 0.0 / 0.05 / 0.2 / 0.5
# SETAFIX_SIGN=on RUN_TAG=SEAM ABL_GROUPS="merge"  bash scripts/run_sos_ablation.sh    # 3 jobs: merge Z 0.0 / 1.0 / 2.0

# results collect:
# REPORT=1 bash scripts/run_sos_ablation.sh
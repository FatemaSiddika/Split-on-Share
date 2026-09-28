#!/bin/bash

#SBATCH --account=jannesar-lab
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=nova
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --exclude=nova21-gpu-2,nova21-gpu-10

# setafix_all_tasks.slurm — ONE allocation for a whole task stream.
# Submitted by:  SUBMIT_MODE=onejob bash scripts/run_setafix.sh <range> <steps> <model>
#
# For each task it runs setafix_pipeline.slurm with bash (not sbatch), with the
# exact variables the per-task launcher would have exported, so the python
# command, settings and paths are identical to a per-task job. Each task still
# writes its own log file (T<id>_<task>_fused) under logs/.
#
# Then (FINAL_ROW=True, steps include train and eval):
#   final row   the last checkpoint evaluated on tasks 1..N-1, exactly as
#               RL_Lamda_fsys_eval_only_arms.sh does it (datasets grouped by
#               sample cap, one eval process per group)
#   metrics     cl_metrics.py: LA, OP, BWT from diagonal + final row
#
# Everything arrives through --export from run_setafix.sh.

set -u
PROJECT_ROOT="${PROJECT_ROOT:-/lustre/hdd/LAS/jannesar-lab/fatemask/fed-con/Split-on-Share}"
PIPE="${PIPELINE_SLURM:-${PROJECT_ROOT}/scripts/setafix_pipeline.slurm}"

LOG_DIR="${PROJECT_ROOT}/logs/${user_model_name}_${PROJECTION_MODE}/${ABLATION_TAG}"
mkdir -p "$LOG_DIR"
_ts=$(date +%F_%H-%M-%S)
DRIVER_LOG="${LOG_DIR}/${_ts}_${ABLATION_TAG}_${SLURM_JOB_NAME}_${SLURM_JOB_ID}_driver.txt"
exec > >(tee -a "$DRIVER_LOG") 2>&1

IFS=',' read -r -a TASK_ARRAY <<< "${TASK_LIST_STR}"
IFS=':,' read -r -a IDS <<< "${TASK_IDS_STR}"
IFS=':,' read -r -a STEP_LIST <<< "${STEPS}"
has_step() { local s; for s in "${STEP_LIST[@]}"; do [ "$s" == "$1" ] && return 0; done; return 1; }
is_gen()   { case "$1" in MeetingBank|20Minuten) return 0 ;; *) return 1 ;; esac; }
task_name() { echo "${TASK_ARRAY[$(( $1 - 1 ))]}"; }
save_path() { local n; n="$(task_name "$1")"; echo "${BASE_DIR}/shared_moE_checkpoints_${user_model_name}_${1}_${n,,}"; }

echo "============================================================"
echo "  ONE-JOB STREAM   job ${SLURM_JOB_ID}  host $(hostname)"
echo "  model    : ${user_model_name}   proj=${PROJECTION_MODE}"
echo "  tasks    : ${IDS[*]}"
echo "  steps    : ${STEPS}"
echo "  tree     : ${BASE_DIR}"
echo "  final row: ${FINAL_ROW}   skip_done: ${SKIP_DONE}"
echo "  logs     : ${LOG_DIR}"
echo "============================================================"

_T0=$SECONDS

# ── 1. every task: fft, split, train, eval (eval on its own task = diagonal) ──
for TASK_ID in "${IDS[@]}"; do
    TASK_NAME="$(task_name "$TASK_ID")"
    SAVE_PATH="$(save_path "$TASK_ID")"
    RESUME_PATH="None"
    if (( TASK_ID > 1 )); then
        RESUME_PATH="$(save_path $(( TASK_ID - 1 )))"
        [ "${STRUCTURE_ONLY:-0}" == "1" ] && RESUME_PATH="${RESUME_PATH}/init_structure_checkpoint"
    fi
    CUR_MAX="${MAX_SAMPLES_OVERRIDE:-}"
    if [ -z "$CUR_MAX" ]; then
        CUR_MAX=500
        is_gen "$TASK_NAME" && CUR_MAX="${MAX_SAMPLES_GEN:-100}"
    fi

    # SKIP_DONE: this task's last requested output already exists
    if [[ "${SKIP_DONE}" == "True" ]]; then
        _done=""
        if has_step eval; then
            _done="${SAVE_PATH}/eval_task_${TASK_ID}_${TASK_NAME}/evaluation_summary.json"
        elif has_step train; then
            _done="${SAVE_PATH}/expert_definitions.json"
        elif has_step split; then
            _done="${SAVE_PATH}/init_structure_checkpoint/expert_definitions.json"
        fi
        if [ -n "$_done" ] && [ -f "$_done" ]; then
            echo "[T${TASK_ID} ${TASK_NAME}] already done ($(basename "$_done") exists) — skipped"
            continue
        fi
    fi

    echo ""
    echo "[T${TASK_ID} ${TASK_NAME}] start  (elapsed $(( (SECONDS - _T0) / 60 )) min)"
    echo "     save   : ${SAVE_PATH}"
    echo "     resume : ${RESUME_PATH}"
    _S0=$SECONDS
    TASK_ID="$TASK_ID" TASK_NAME="$TASK_NAME" SAVE_PATH="$SAVE_PATH" RESUME_PATH="$RESUME_PATH" \
    MAX_SAMPLES="$CUR_MAX" eval_task_id="$TASK_ID" MODE="$STEPS" SKIP_EVAL_TASKS="" EVAL_TASK_IDS="" \
    SOS_JOB_LABEL="T${TASK_ID}_${TASK_NAME}_fused" \
        bash "$PIPE" > /dev/null 2>&1
    _rc=$?
    if [ $_rc -ne 0 ]; then
        echo "[T${TASK_ID} ${TASK_NAME}] FAILED (exit ${_rc}) after $(( (SECONDS - _S0) / 60 )) min — stopping the stream."
        echo "     task log: ls -t ${LOG_DIR}/*T${TASK_ID}_${TASK_NAME}_fused_${SLURM_JOB_ID}.*"
        exit $_rc
    fi
    echo "[T${TASK_ID} ${TASK_NAME}] done in $(( (SECONDS - _S0) / 60 )) min"
done

# ── 2. final row: last checkpoint on every earlier task ──────────────────────
LAST="${IDS[-1]}"
if [[ "${FINAL_ROW}" != "True" ]] || ! has_step train || ! has_step eval || (( LAST < 2 )); then
    echo ""
    echo "[final row] skipped (FINAL_ROW=${FINAL_ROW}, steps=${STEPS}, last task=${LAST})"
    exit 0
fi

LAST_NAME="$(task_name "$LAST")"
LAST_PATH="$(save_path "$LAST")"
# Same grouping as RL_Lamda_fsys_eval_only_arms.sh: one eval process per sample
# cap (standard 500, generation MAX_SAMPLES_GEN), MAX_SAMPLES override = one group.
GRP_STD=(); GRP_GEN=()
for (( J=1; J<LAST; J++ )); do
    if [ -n "${MAX_SAMPLES_OVERRIDE:-}" ]; then GRP_STD+=("$J")
    elif is_gen "$(task_name "$J")"; then     GRP_GEN+=("$J")
    else                                      GRP_STD+=("$J"); fi
done

run_final_group() {
    local _cap="$1"; shift
    local -a _ids=("$@")
    [ ${#_ids[@]} -eq 0 ] && return 0
    local _joined; _joined="$(IFS=:; echo "${_ids[*]}")"
    echo ""
    echo "[final row] T${LAST} on tasks ${_ids[*]}  (samples ${_cap})"
    local _S0=$SECONDS
    TASK_ID="$LAST" TASK_NAME="$LAST_NAME" SAVE_PATH="$LAST_PATH" RESUME_PATH="None" \
    MAX_SAMPLES="$_cap" eval_task_id="${_ids[0]}" EVAL_TASK_IDS="$_joined" MODE="eval" SKIP_EVAL_TASKS="" \
    SOS_JOB_LABEL="T${LAST}_ev${_ids[0]}-${_ids[-1]}_final" \
        bash "$PIPE" > /dev/null 2>&1
    local _rc=$?
    if [ $_rc -ne 0 ]; then
        echo "[final row] eval of tasks ${_ids[*]} FAILED (exit ${_rc}) — metrics will list them as missing."
        return 0
    fi
    echo "[final row] done in $(( (SECONDS - _S0) / 60 )) min"
}
run_final_group "${MAX_SAMPLES_OVERRIDE:-500}" "${GRP_STD[@]}"
run_final_group "${MAX_SAMPLES_GEN:-100}"      "${GRP_GEN[@]}"

# ── 3. LA / OP / BWT ─────────────────────────────────────────────────────────
source /lustre/hdd/LAS/jannesar-lab/fatemask/miniconda3_new/etc/profile.d/conda.sh
conda activate /lustre/hdd/LAS/jannesar-lab/fatemask/seta_env
python -u "${PROJECT_ROOT}/cl_metrics.py" \
    --base_dir "${BASE_DIR}" --model "${user_model_name}" \
    --task_list "${TASK_LIST_STR}" --last "${LAST}"

echo ""
echo "[stream] finished in $(( (SECONDS - _T0) / 60 )) min"

#!/bin/bash
##SBATCH --account=<your_account>      # set for your cluster, then remove one '#'
##SBATCH --partition=<your_partition>  # set for your cluster, then remove one '#'
#SBATCH --time=2-23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# setafix_pipeline_streams.slurm — ONE GPU job for a whole task stream.
# Submitted by the launchers in scripts/ (sos_common_order.sh), which export:
#   SOS_RUN_DIR SOS_MODEL SOS_TASKS SOS_FIRST SOS_LAST SOS_STEPS (colon list)
# 1. tasks SOS_FIRST..SOS_LAST in order: fft, split, train, eval on its own task
# 2. final row: the last checkpoint evaluated on tasks 1..LAST-1 (FINAL_ROW=True)
# 3. LA / OP / BWT with utils/cl_metrics.py
# Every other setting comes from utils/run_defaults.py (main.py fills it in).
# SKIP_DONE=True (default): a task whose last requested output exists is skipped.
set -u
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_ROOT"
LOG_DIR="${PROJECT_ROOT}/logs/${SOS_MODEL}/$(basename "$SOS_RUN_DIR")"
mkdir -p "$LOG_DIR"
exec > >(tee -a "${LOG_DIR}/$(date +%F_%H-%M-%S)_${SLURM_JOB_ID}_driver.txt") 2>&1

# ── Environment (exported to the job by sbatch --export=ALL) ─────────────────
#   CUDA_MODULE  module to load, e.g. cuda/12.4.1      (skipped when empty)
#   CONDA_ENV    conda env name or path to activate     (skipped when empty:
#                the environment active at submit time is used)
#   CONDA_SH     conda.sh to source (default: from `conda info --base`)
[ -n "${CUDA_MODULE:-}" ] && module load "${CUDA_MODULE}"
if [ -n "${CONDA_ENV:-}" ]; then
    source "${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
    conda activate "${CONDA_ENV}"
fi

IFS=',' read -r -a TASKS <<< "$SOS_TASKS"
IFS=':,' read -r -a STEPS <<< "$SOS_STEPS"
has_step() { local s; for s in "${STEPS[@]}"; do [ "$s" == "$1" ] && return 0; done; return 1; }
ckpt() { echo "${SOS_RUN_DIR}/shared_moE_checkpoints_${SOS_MODEL}_${1}_${TASKS[$(( $1 - 1 ))],,}"; }
phase_of() { case "$1" in fft) echo phase0_fft ;; split) echo phase1_split ;; train) echo phase2_train ;; eval) echo phase3_eval ;; esac; }
# run <task id> <step> <log name>: one main.py process, its output in its own log
run() {
    python -u "${PROJECT_ROOT}/main.py" --use_model_name "$SOS_MODEL" --task_list "$SOS_TASKS" \
        --current_task_id "$1" --run_phase "$(phase_of "$2")" --run_dir "$SOS_RUN_DIR" \
        >> "${LOG_DIR}/${SLURM_JOB_ID}_$3.txt" 2>&1
}

echo "STREAM job ${SLURM_JOB_ID} on $(hostname): ${SOS_MODEL}, tasks ${SOS_FIRST}-${SOS_LAST}, steps ${SOS_STEPS}"
echo "  run folder: ${SOS_RUN_DIR}"
T0=$SECONDS

# ── 1. every task: fft, split, train, eval (eval on its own task = diagonal) ──
for (( ID=SOS_FIRST; ID<=SOS_LAST; ID++ )); do
    NAME="${TASKS[$(( ID - 1 ))]}"
    if [ "${SKIP_DONE:-True}" == "True" ]; then
        DONE=""
        if has_step eval;    then DONE="$(ckpt "$ID")/eval_task_${ID}_${NAME}/evaluation_summary.json"
        elif has_step train; then DONE="$(ckpt "$ID")/expert_definitions.json"
        elif has_step split; then DONE="$(ckpt "$ID")/init_structure_checkpoint/expert_definitions.json"; fi
        if [ -n "$DONE" ] && [ -f "$DONE" ]; then echo "[T${ID} ${NAME}] already done, skipped"; continue; fi
    fi
    for STEP in "${STEPS[@]}"; do
        S0=$SECONDS
        if ! run "$ID" "$STEP" "T${ID}_${NAME}"; then
            echo "[T${ID} ${NAME}] ${STEP} FAILED — stream stopped. Log: ${LOG_DIR}/${SLURM_JOB_ID}_T${ID}_${NAME}.txt"
            exit 1
        fi
        echo "[T${ID} ${NAME}] ${STEP} done in $(( (SECONDS - S0) / 60 )) min"
    done
done

# ── 2. final row: last checkpoint on every earlier task ──────────────────────
LAST="$SOS_LAST"
if [ "${FINAL_ROW:-True}" != "True" ] || ! has_step train || ! has_step eval || (( LAST < 2 )); then
    echo "[final row] skipped"; exit 0
fi
IDS="$(seq -s: 1 $(( LAST - 1 )))"
if EVAL_TASK_IDS="$IDS" run "$LAST" eval "T${LAST}_final_row"; then
    echo "[final row] T${LAST} on tasks ${IDS//:/,} done"
else
    echo "[final row] FAILED — metrics will list the missing tasks"
fi

# ── 3. LA / OP / BWT ─────────────────────────────────────────────────────────
python -u "${PROJECT_ROOT}/utils/cl_metrics.py" --base_dir "$SOS_RUN_DIR" --model "$SOS_MODEL" \
    --task_list "$SOS_TASKS" --last "$LAST"
echo "[stream] finished in $(( (SECONDS - T0) / 60 )) min"

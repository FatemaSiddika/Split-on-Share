#!/bin/bash
# sos_common_order.sh — shared by the launchers in scripts/.
# Every run setting has its default in utils/run_defaults.py. This file only
# names the run folder and submits ONE GPU job for the whole task stream
# (setafix_pipeline_streams.slurm), which trains task 1, 2, ... in order, then
# evaluates the last checkpoint on the earlier tasks and reports LA / OP / BWT.
#
# Environment options (all optional):
#   SEED, BLOCK_BUDGET_FRACTION, SETAFIX_PHI, SETAFIX_Z, NO_EXPERT_SHARING, ...
#                  run settings, see utils/run_defaults.py
#   RUN_TAG        extra suffix for the run folder name
#   FORCE_OVERWRITE=True   continue / overwrite an existing run folder
#                  (finished tasks are skipped, SKIP_DONE=True by default)
#   ONEJOB_TIME    walltime of the job (default: the slurm header)
#   DEPEND         job id this job waits for (afterok)
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT

# sos_submit <stream name> <task list> <task range | all> <steps> <model>
sos_submit() {
    local stream="$1" tasks="$2" range="$3" steps="$4" model="$5"
    local -a T; IFS=',' read -r -a T <<< "$tasks"
    local n=${#T[@]}
    [ "$range" == "all" ] && range="1-$n"
    [[ "$range" == *-* ]] || range="${range}-${range}"
    local first="${range%-*}" last="${range#*-}"
    if ! [[ "$first" =~ ^[0-9]+$ && "$last" =~ ^[0-9]+$ ]] || (( first < 1 || last > n || first > last )); then
        echo "ERROR: task range '$3' must be a or a-b within 1-${n}" >&2; exit 1
    fi
    local s; for s in ${steps//,/ }; do case "$s" in fft|split|train|eval) ;; *)
        echo "ERROR: steps '$steps' must be a comma list of fft,split,train,eval" >&2; exit 1 ;;
    esac; done
    python3 "${PROJECT_ROOT}/utils/run_defaults.py" --check-data \
        "$(IFS=,; echo "${T[*]:$((first - 1)):$((last - first + 1))}")"

    local seed="${SEED:-100}"
    local name="${stream}_$(python3 "${PROJECT_ROOT}/utils/run_defaults.py" --tag)_seed${seed}${RUN_TAG:+_${RUN_TAG}}"
    local run_dir="${PROJECT_ROOT}/trained_models/${model}/${name}"
    if compgen -G "${run_dir}/shared_moE_checkpoints_*" > /dev/null && [ "${FORCE_OVERWRITE:-False}" != "True" ]; then
        echo "REFUSING TO START: ${run_dir} already has checkpoints." >&2
        echo "  Continue it (finished tasks are skipped): FORCE_OVERWRITE=True <same command>" >&2
        echo "  Or give this run its own folder:          RUN_TAG=<name> <same command>" >&2
        exit 1
    fi
    mkdir -p "$run_dir"

    export SEED="$seed" SOS_RUN_DIR="$run_dir" SOS_MODEL="$model" SOS_TASKS="$tasks" \
           SOS_FIRST="$first" SOS_LAST="$last" SOS_STEPS="${steps//,/:}"
    local job
    job=$(sbatch --parsable ${ONEJOB_TIME:+--time=${ONEJOB_TIME}} ${DEPEND:+--dependency=afterok:${DEPEND}} \
        --job-name="${stream}_T${first}-${last}_${model}_s${seed}" --export=ALL \
        "${PROJECT_ROOT}/scripts/setafix_pipeline_streams.slurm")
    echo "job ${job}: ${stream} tasks ${first}-${last}, ${model}, seed ${seed}"
    echo "  run folder : ${run_dir}"
    echo "  logs       : ${PROJECT_ROOT}/logs/${model}/${name}/"
}

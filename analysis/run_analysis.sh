#!/bin/bash
# run_analysis.sh — analyse/diagnose the split structure (SEAM vs original).
#
#   bash analysis/run_analysis.sh                          # SLURM (GPU), auto-find both structonly trees
#   bash analysis/run_analysis.sh --dep 12517107           # wait for job(s) first: afterok:<ids>
#   bash analysis/run_analysis.sh --dep 12517107:12517066  # several ids, colon separated
#   bash analysis/run_analysis.sh --local                  # run in this shell, no SLURM
#   bash analysis/run_analysis.sh [opts] TREE_A TREE_B ... # explicit trees
#
# Env: MODEL (qwen3_4b), PROJ (v_proj_only), OUT (default <first tree>/setafix_analysis),
#      GPU_TYPE (a100), DEP_KIND (afterok | afterany)
# Logs go to analysis/logs/ (this folder).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # .../Split-on-Share/analysis
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${HERE}/.." && pwd)}"
MODEL="${MODEL:-qwen3_4b}"; PROJ="${PROJ:-v_proj_only}"
CONDA_SH="/lustre/hdd/LAS/jannesar-lab/fatemask/miniconda3_new/etc/profile.d/conda.sh"
CONDA_ENV="/lustre/hdd/LAS/jannesar-lab/fatemask/seta_env"

LOCAL=0; DEP=""; TREES=()
while [ $# -gt 0 ]; do
    case "$1" in
        --local) LOCAL=1 ;;
        --dep)   DEP="$2"; shift ;;
        --dep=*) DEP="${1#--dep=}" ;;
        *)       TREES+=("$1") ;;
    esac
    shift
done
if [ ${#TREES[@]} -eq 0 ]; then
    ROOT="${PROJECT_ROOT}/trained_models/RL_Lamda_fsys_trained_models_SoS_${MODEL}_${PROJ}"
    for m in seam off; do
        for d in "${ROOT}"/ablation_*_setafix_${m}_*_structonly; do [ -d "$d" ] && TREES+=("$d"); done
    done
fi
[ ${#TREES[@]} -eq 0 ] && { echo "no trees found (pass them explicitly)"; exit 1; }
echo "trees:"; printf '   %s\n' "${TREES[@]}"
OUT="${OUT:-${TREES[0]}/setafix_analysis}"

if [ "$LOCAL" == "1" ]; then
    source "${CONDA_SH}"; conda activate "${CONDA_ENV}"
    export PATH="${CONDA_ENV}/bin:$PATH"
    mkdir -p "${HERE}/logs"
    cd "${HERE}" && python -u setafix_analyze.py "${TREES[@]}" --out "${OUT}" \
        2>&1 | tee "${HERE}/logs/local_$(date +%F_%H-%M-%S).txt"
    exit ${PIPESTATUS[0]}
fi

mkdir -p "${HERE}/logs"
case "${GPU_TYPE:-a100}" in a100) GRES="gpu:a100:1";; h200) GRES="gpu:h200:1";; l40s) GRES="gpu:l40s:1";; *) GRES="gpu:1";; esac
DEPFLAG=""; [ -n "$DEP" ] && DEPFLAG="--dependency=${DEP_KIND:-afterok}:${DEP}"

JOB=$(sbatch --parsable ${DEPFLAG} \
      --account="${SLURM_ACCOUNT:-jannesar-lab}" --partition=nova --gres="${GRES}" \
      --nodes=1 --ntasks-per-node=4 --mem=32G --time=01:00:00 --job-name=setafix_analysis \
      --output="${HERE}/logs/%j.txt" --error="${HERE}/logs/%j.txt" \
      --export=ALL,ANALYSIS_TREES="$(IFS=,; echo "${TREES[*]}")",ANALYSIS_OUT="${OUT}",ANALYSIS_DIR="${HERE}" \
      "${HERE}/setafix_analysis.slurm")
echo "submitted job ${JOB}${DEPFLAG:+  (${DEPFLAG#--dependency=})}"
echo "log     -> ${HERE}/logs/${JOB}.txt"
echo "outputs -> ${OUT}"



# bash analysis/run_analysis.sh --dep 12517107               # after the SEAM chain's T8
# bash analysis/run_analysis.sh --dep 12517107:12517066      # after both chains' T8
# DEP_KIND=afterany bash analysis/run_analysis.sh --dep 12517107   # run even if a task failed
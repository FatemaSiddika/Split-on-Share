#!/bin/bash
# run_probe.sh — matched-vs-null agreement probe on saved gradient tiles.
#   bash analysis/run_probe.sh                 # auto-find the 12% and 25% FFT dirs, run locally
#   bash analysis/run_probe.sh FFT_A FFT_B     # explicit dirs
#   SIDE=both bash analysis/run_probe.sh       # compare output subspaces too
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # .../Split-on-Share/analysis
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${HERE}/.." && pwd)}"
CONDA_SH="/lustre/hdd/LAS/jannesar-lab/fatemask/miniconda3_new/etc/profile.d/conda.sh"
CONDA_ENV="/lustre/hdd/LAS/jannesar-lab/fatemask/seta_env"
DIRS=("$@")
if [ ${#DIRS[@]} -eq 0 ]; then
    for d in "${PROJECT_ROOT}"/FFT_steps/FFT_steps_*setafix_seam*structonly*; do [ -d "$d" ] && DIRS+=("$d"); done
fi
[ ${#DIRS[@]} -eq 0 ] && { echo "no FFT_steps_* dirs found — pass them explicitly"; exit 1; }
echo "probing:"; printf '   %s\n' "${DIRS[@]}"
source "${CONDA_SH}"; conda activate "${CONDA_ENV}"; export PATH="${CONDA_ENV}/bin:$PATH"
cd "${HERE}"; mkdir -p "${HERE}/logs"
python -u setafix_probe_null.py "${DIRS[@]}" \
    --labels "${LABELS:-12%,25%}" --side "${SIDE:-right}" \
    --layers "${LAYERS:-6}" --blocks "${BLOCKS:-40}" --null-k "${NULLK:-16}" \
    --out "${OUT:-${HERE}/probe_out}" 2>&1 | tee "${HERE}/logs/probe_$(date +%F_%H-%M-%S).txt"

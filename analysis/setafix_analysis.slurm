#!/bin/bash
# setafix_analysis.slurm — submitted by run_analysis.sh; same env setup as setafix_pipeline.slurm
set -u
echo "[analysis] $(date)  host=$(hostname)"
module load cuda/12.4.1-cz3ljd3
source /lustre/hdd/LAS/jannesar-lab/fatemask/miniconda3_new/etc/profile.d/conda.sh
conda activate /lustre/hdd/LAS/jannesar-lab/fatemask/seta_env
export PATH="/lustre/hdd/LAS/jannesar-lab/fatemask/seta_env/bin:$PATH"
cd "${ANALYSIS_DIR:-/lustre/hdd/LAS/jannesar-lab/fatemask/fed-con/Split-on-Share/analysis}"
IFS=',' read -r -a TREES <<< "${ANALYSIS_TREES}"
echo "[analysis] trees: ${TREES[*]}"
python -u setafix_analyze.py "${TREES[@]}" --out "${ANALYSIS_OUT}"
echo "[analysis] done $(date)"

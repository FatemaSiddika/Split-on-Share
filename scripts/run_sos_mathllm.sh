#!/bin/bash
# run_sos_mathllm.sh — Math stream: ScienceQA -> NumGLUE-cm -> NumGLUE-ds.
# One GPU job per seed. Data: the 5000-sample set.
# Usage:  bash scripts/run_sos_mathllm.sh [range] [steps] [model]
#   range  a or a-b or all (default all)   steps  default fft,split,train,eval
#   model  default qwen3_4b
# Options (environment):
#   SEED   one or more seeds   (default "100 42")
#   BLOCK_BUDGET_FRACTION (default here 0.12), SETAFIX_PHI (default here 0.012),
#   plus the run settings in utils/run_defaults.py and the options in sos_common_order.sh
# Examples:
#   bash scripts/run_sos_mathllm.sh all fft,split,train,eval llama2_7b
#   SEED=100 FORCE_OVERWRITE=True bash scripts/run_sos_mathllm.sh 2-3   # resume from task 2
source "$(dirname "${BASH_SOURCE[0]}")/sos_common_order.sh"

RANGE="${1:-all}"; STEPS="${2:-fft,split,train,eval}"; MODEL="${3:-qwen3_4b}"
export DATASET="${DATASET:-5000}"
export BLOCK_BUDGET_FRACTION="${BLOCK_BUDGET_FRACTION:-0.12}" SETAFIX_PHI="${SETAFIX_PHI:-0.012}"

SEEDS="${SEED:-100 42}"   # read once: each submit exports SEED
for seed in $SEEDS; do
    SEED="$seed" sos_submit "math" "ScienceQA,NumGLUE-cm,NumGLUE-ds" "$RANGE" "$STEPS" "$MODEL"
done

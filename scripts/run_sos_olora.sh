#!/bin/bash
# run_sos_olora.sh — O-LoRA task orders 1-6 (Wang et al., 2023, Table 7).
# Orders 1-3: 4 tasks; orders 4-6: the 15-task long sequences.
# One GPU job per (order, seed). Data: the 500-sample set (all 15 tasks).
# Usage:  bash scripts/run_sos_olora.sh [range] [steps] [model]
#   range  a or a-b or all (default all)   steps  default fft,split,train,eval
#   model  default qwen3_4b
# Examples:
#   OLORA_ORDER="1 2 3 4 5 6" SEED=100 bash scripts/run_sos_olora.sh
#   OLORA_ORDER=5 FORCE_OVERWRITE=True bash scripts/run_sos_olora.sh 3-15   # resume at task 3
source "$(dirname "${BASH_SOURCE[0]}")/sos_common_order.sh"

RANGE="${1:-all}"; STEPS="${2:-fft,split,train,eval}"; MODEL="${3:-qwen3_4b}"
export DATASET="${DATASET:-500}"
export BLOCK_BUDGET_FRACTION="${BLOCK_BUDGET_FRACTION:-0.12}" SETAFIX_PHI="${SETAFIX_PHI:-0.012}"
SEEDS="${SEED:-100 42}"   # read once: each submit exports SEED

for order in ${OLORA_ORDER:-1}; do
    case "$order" in
        1) TASKS="dbpedia,amazon,yahoo,agnews" ;;
        2) TASKS="dbpedia,amazon,agnews,yahoo" ;;
        3) TASKS="yahoo,amazon,agnews,dbpedia" ;;
        4) TASKS="MNLI,CB,WiC,COPA,QQP,BoolQA,RTE,IMDB,yelp,amazon,SST-2,dbpedia,agnews,MultiRC,yahoo" ;;
        5) TASKS="MultiRC,BoolQA,WiC,MNLI,CB,COPA,QQP,RTE,IMDB,SST-2,dbpedia,agnews,yelp,amazon,yahoo" ;;
        6) TASKS="yelp,amazon,MNLI,CB,COPA,QQP,RTE,IMDB,SST-2,dbpedia,agnews,yahoo,MultiRC,BoolQA,WiC" ;;
        *) echo "ERROR: OLORA_ORDER must be 1-6 (got '${order}')" >&2; exit 1 ;;
    esac
    for seed in $SEEDS; do
        SEED="$seed" sos_submit "olora_o${order}" "$TASKS" "$RANGE" "$STEPS" "$MODEL"
    done
done

#!/bin/bash
# run_setafix_order.sh — TRACE task streams, one GPU job per seed.
# Usage:  bash scripts/run_setafix_order.sh [range] [steps] [model]
#   range  a or a-b or all      (default 1-8: the 8 TRACE tasks)
#   steps  comma list of fft,split,train,eval   (default all four)
#   model  qwen3_4b | llama2_7b | llama3_8b | qwen3_8b | qwen3_14b  (default qwen3_4b)
# Examples:
#   bash scripts/run_setafix_order.sh 1-8 fft,split,train,eval qwen3_4b
#   DATASET=500 bash scripts/run_setafix_order.sh 1-15
#   NO_EXPERT_SHARING=True SEED="100 42 7" bash scripts/run_setafix_order.sh 1-8 fft,split,train,eval llama2_7b
source "$(dirname "${BASH_SOURCE[0]}")/sos_common_order.sh"

RANGE="${1:-1-8}"; STEPS="${2:-fft,split,train,eval}"; MODEL="${3:-qwen3_4b}"
export DATASET="${DATASET:-5000}" ONEJOB_TIME="${ONEJOB_TIME:-7-00:00:00}"

if [ -n "${TASK_LIST_STR:-}" ]; then
    TASKS="$TASK_LIST_STR"; STREAM="trace_ordc$(printf '%s' "$TASKS" | md5sum | cut -c1-6)"
else
    case "${TASK_ORDER:-1}" in
        1) TASKS="C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten,dbpedia,amazon,yahoo,agnews,yelp,BoolQA,QQP" ;;
        2) TASKS="QQP,BoolQA,yelp,agnews,yahoo,amazon,dbpedia,NumGLUE-ds,Py150,20Minuten,NumGLUE-cm,ScienceQA,MeetingBank,FOMC,C-STANCE" ;;
        3) TASKS="20Minuten,Py150,NumGLUE-ds,dbpedia,amazon,yahoo,agnews,yelp,BoolQA,QQP,C-STANCE,FOMC,MeetingBank,ScienceQA,NumGLUE-cm" ;;
        4) TASKS="C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten" ;;
        5) TASKS="NumGLUE-ds,Py150,20Minuten,NumGLUE-cm,ScienceQA,MeetingBank,FOMC,C-STANCE" ;;
        *) echo "ERROR: TASK_ORDER must be 1-5, or set TASK_LIST_STR" >&2; exit 1 ;;
    esac
    STREAM="trace_ord${TASK_ORDER:-1}"
fi
[ "$DATASET" == "500" ] && STREAM="${STREAM}_d500"

SEEDS="${SEED:-100}"   # read once: each submit exports SEED
for seed in $SEEDS; do
    SEED="$seed" sos_submit "$STREAM" "$TASKS" "$RANGE" "$STEPS" "$MODEL"
done

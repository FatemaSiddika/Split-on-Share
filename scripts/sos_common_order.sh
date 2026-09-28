#!/bin/bash
# sos_common.sh — shared configuration, sourced by the launchers in scripts/.
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
    DATASET="${DATASET:-5000}"
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

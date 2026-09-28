#!/bin/bash
# collect_results.sh — collect one run's results (matrix, LA / OP / BWT) into
# <run>/results/, finding the run folder from the SAME settings you trained with.
# Read-only on the run; submits nothing. No GPU.
#
#   cd <...>/Split-on-Share
#   <same env as training> bash scripts/collect_results.sh <MODEL> [plain|order]
#
#   MODEL        qwen3_4b | llama2_7b | ...   (3rd argument of run_setafix*.sh)
#   plain|order  which launcher's naming to use: run_setafix.sh (default) or
#                run_setafix_order.sh
#
#   e.g.
#   SUBMIT_MODE=onejob BLOCK_BUDGET_FRACTION=0.12 ABLATION=1 SETAFIX_NULL=empirical SETAFIX_ORTHO=on \
#     SETAFIX_GATE_PROTECT=freeze SETAFIX_LAYER_GATE=budget SETAFIX_PHI=0.012 \
#     bash scripts/collect_results.sh qwen3_4b order
#
# How the folder is found: the launcher's own code up to the line that sets
# BASE_DIR is run with your settings (that part only reads and checks files),
# so the folder name always follows the launcher's current naming rules.
# EVAL_OUT_SUFFIX, if set, selects the eval folders written with that suffix.
set -euo pipefail
case "${1:-}" in -h|--help|"") sed -n '2,21p' "${BASH_SOURCE[0]}"; exit 0 ;; esac
MODEL="$1"
VARIANT="${2:-plain}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
case "$VARIANT" in
    plain) LAUNCHER="${SCRIPT_DIR}/run_setafix.sh" ;;
    order) LAUNCHER="${SCRIPT_DIR}/run_setafix_order.sh" ;;
    *) echo "ERROR: 2nd argument must be plain or order (got '${VARIANT}')" >&2; exit 1 ;;
esac
[ -f "$LAUNCHER" ] || { echo "ERROR: launcher not found: ${LAUNCHER}" >&2; exit 1; }
grep -q '^BASE_DIR=' "$LAUNCHER" || { echo "ERROR: no 'BASE_DIR=' line in ${LAUNCHER}" >&2; exit 1; }

# the launcher's code up to BASE_DIR, next to the launcher so its relative
# 'source sos_common*.sh' works; removed on exit
TMP="${SCRIPT_DIR}/.collect_basedir_$$.sh"
ERR="$(mktemp)"
trap 'rm -f "$TMP" "$ERR"' EXIT
{ sed -n '1,/^BASE_DIR=/p' "$LAUNCHER"; echo 'echo "$BASE_DIR" >&3'; } > "$TMP"
if ! BASE_DIR=$(bash "$TMP" 1 fft,split,train,eval "$MODEL" 3>&1 1>/dev/null 2>"$ERR"); then
    cat "$ERR" >&2
    echo "ERROR: $(basename "$LAUNCHER") refused these settings (above)." >&2
    exit 1
fi
[ -n "$BASE_DIR" ] || { echo "ERROR: could not resolve the run folder" >&2; exit 1; }

echo "launcher : $(basename "$LAUNCHER")   model: ${MODEL}"
echo "run      : ${BASE_DIR}"
if [ ! -d "$BASE_DIR" ]; then
    echo "ERROR: that folder does not exist — the settings differ from the training run." >&2
    echo "       Runs present for ${MODEL}:" >&2
    ls -1d "$(dirname "$BASE_DIR")"/ablation_* 2>/dev/null | sed 's/^/         /' >&2 || true
    exit 1
fi
cd "$PROJECT_ROOT"
exec python3 helpers/collect_results.py "$BASE_DIR" --suffix "${EVAL_OUT_SUFFIX:-}"



# cd /lustre/hdd/LAS/jannesar-lab/fatemask/Split-on-Share

# # order run
# python3 helpers/collect_results.py --model qwen3_4b --launcher order BLOCK_BUDGET_FRACTION=0.12 ABLATION=1 SETAFIX_NULL=empirical SETAFIX_ORTHO=on SETAFIX_GATE_PROTECT=freeze SETAFIX_LAYER_GATE=budget SETAFIX_PHI=0.012

# # plain run
# python3 helpers/collect_results.py --model qwen3_4b --launcher plain BLOCK_BUDGET_FRACTION=0.12 ABLATION=1 SETAFIX_NULL=empirical SETAFIX_ORTHO=on SETAFIX_GATE_PROTECT=freeze SETAFIX_LAYER_GATE=budget SETAFIX_PHI=0.012

# # every run
# python3 helpers/collect_results.py --all trained_models
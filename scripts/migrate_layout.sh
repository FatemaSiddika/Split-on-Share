#!/bin/bash
# migrate_layout.sh — move existing run folders into the Split-on-Share layout.
#
#   bash scripts/migrate_layout.sh                       # dry run, looks in the project root
#   bash scripts/migrate_layout.sh /path/to/fed-con/SETA_fix   # dry run, old SETA_fix folder
#   APPLY=1 bash scripts/migrate_layout.sh [SRC]         # really move
#
# Moves, keeping every name and everything inside unchanged:
#   SRC/FFT_steps_*                     -> <PROJECT_ROOT>/FFT_steps/
#   SRC/RL_Lamda_fsys_trained_models_*/ablation_*
#                                       -> <PROJECT_ROOT>/trained_models/RL_Lamda_fsys_trained_models_*/
#   SRC/logs/<model_proj>/*             -> <PROJECT_ROOT>/logs/<model_proj>/   (only with LOGS=1)
# Never overwrites: an existing target is reported and left alone.
# Do not run it while jobs of those runs are still queued or running.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/sos_common.sh"
SRC="$(cd "${1:-$PROJECT_ROOT}" && pwd)"
APPLY="${APPLY:-0}"

echo "source : ${SRC}"
echo "target : ${PROJECT_ROOT}   ($([ "$APPLY" == "1" ] && echo "MOVING" || echo "dry run — APPLY=1 to move"))"
echo ""

move_into() {   # $1 = source dir, $2 = target parent
    local _src="$1" _dst="$2/$(basename "$1")"
    if [ -e "$_dst" ]; then
        echo "  [exists, skipped] $_dst"
        return 0
    fi
    echo "  $(basename "$_src")  ->  ${2#${PROJECT_ROOT}/}/"
    if [ "$APPLY" == "1" ]; then
        mkdir -p "$2"
        mv "$_src" "$_dst"
    fi
}

_n=0
for d in "${SRC}"/FFT_steps_*; do
    [ -d "$d" ] || continue; move_into "$d" "${SOS_FFT_ROOT}"; _n=$((_n + 1))
done
# run trees move one ablation_* folder at a time, so a model root that already
# exists in the new layout (new runs started there) is merged, not skipped
for r in "${SRC}"/RL_Lamda_fsys_trained_models_*; do
    [ -d "$r" ] || continue
    for d in "$r"/ablation_*; do
        [ -d "$d" ] || continue; move_into "$d" "${SOS_MODELS_DIR}/$(basename "$r")"; _n=$((_n + 1))
    done
done
if [ "${LOGS:-0}" == "1" ] && [ -d "${SRC}/logs" ] && [ "${SRC}/logs" != "${SOS_LOG_ROOT}" ]; then
    for r in "${SRC}"/logs/*; do
        [ -d "$r" ] || continue
        for d in "$r"/*; do
            [ -e "$d" ] || continue; move_into "$d" "${SOS_LOG_ROOT}/$(basename "$r")"; _n=$((_n + 1))
        done
    done
fi
echo ""
echo "${_n} folder(s) $([ "$APPLY" == "1" ] && echo "handled" || echo "would be moved")."

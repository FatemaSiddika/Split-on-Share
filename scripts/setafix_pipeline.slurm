#!/bin/bash

#SBATCH --account=jannesar-lab
#SBATCH --time=1-23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6 
#SBATCH --mem=128G   
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=nova
#SBATCH --output=/dev/null   
#SBATCH --error=/dev/null
#SBATCH --exclude=nova21-gpu-2,nova21-gpu-10

timestamp=$(date +%F_%H-%M-%S)

if [ -z "${PHASE2_TAG:-}" ]; then
    PHASE2_TAG="RL_Lamda_${LAMBDA_MODE:-dynamic}_grl${GATING_RL:-False}_gate${GATE_MODE:-sigmoid}_rr${ROUTER_REG:-none}"
    echo "  [logs] PHASE2_TAG not exported — derived locally: ${PHASE2_TAG}"
fi

# [Split-on-Share] PROJECT_ROOT is exported by the launchers in scripts/.
# The fallback is only used when this file is sbatch-ed by hand.
PROJECT_ROOT="${PROJECT_ROOT:-/lustre/hdd/LAS/jannesar-lab/fatemask/fed-con/Split-on-Share}"
export SOS_FFT_ROOT="${SOS_FFT_ROOT:-${PROJECT_ROOT}/FFT_steps}"   # FFT_steps_* dirs go here (read by the trainer)
mkdir -p "$SOS_FFT_ROOT"
LOG_DIR="${PROJECT_ROOT}/logs/${user_model_name}_${PROJECTION_MODE}/${ABLATION_TAG}"
mkdir -p "$LOG_DIR"

# [FUSED] MODE may be a comma list, e.g. "fft,split,train" or "fft,split,train,eval".
# set_phase_vars maps ONE step name onto exactly the same four variables the
# single-step script set, so the python invocation below is unchanged.
set_phase_vars() {
    local _step="$1"
    if [ "$_step" == "fft" ]; then
        DO_TRAIN="True"; DO_EVAL="False"; RUN_PHASE="phase0_fft"; STEP_TAG="FFT"; EVAL_ONLY="False"
    elif [[ "$_step" == "split" || "$_step" == "sos" ]]; then
        DO_TRAIN="True"; DO_EVAL="False"; RUN_PHASE="phase1_split"; STEP_TAG="SPLIT"; EVAL_ONLY="False"
    elif [ "$_step" == "train" ]; then
        DO_TRAIN="True"; DO_EVAL="False"; RUN_PHASE="phase2_train"; STEP_TAG="TRAIN"; EVAL_ONLY="False"
    elif [ "$_step" == "eval" ]; then
        DO_TRAIN="False"; DO_EVAL="True"; RUN_PHASE="phase3_eval"; STEP_TAG="EVAL"; EVAL_ONLY="True"
    else
        DO_TRAIN="True"; DO_EVAL="False"; RUN_PHASE="auto"; STEP_TAG="AUTO"; EVAL_ONLY="False"
    fi
}

# MODE arrives colon separated: sbatch --export is itself comma separated, so a
# comma inside a value truncates it and every step after the first is silently
# dropped as an unknown variable name. Both separators are accepted here.
IFS=',:' read -r -a FUSED_STEPS <<< "${MODE}"
set_phase_vars "${FUSED_STEPS[0]}"

_job_label="${SOS_JOB_LABEL:-${SLURM_JOB_NAME}}"
out_file="${LOG_DIR}/${timestamp}_${ABLATION_TAG}_${_job_label}_${SLURM_JOB_ID}.txt"
err_file="${LOG_DIR}/${timestamp}_${ABLATION_TAG}_${_job_label}_${SLURM_JOB_ID}.err"

exec > >(tee -a "$out_file")
exec 2> >(tee -a "$err_file" >&2)

echo "STARTING JOB: $SLURM_JOB_NAME ($STEP_TAG)"
echo "Task: $TASK_NAME (ID: $TASK_ID)"
echo "Fused steps: ${MODE}   (one SLURM job for this task)"
echo "============================================================"
echo "  ABLATION CONFIG"
echo "  ablation_tag       : ${ABLATION_TAG}"
echo "  fft_stopping_mode  : ${FFT_STOPPING_MODE}"
echo "  layer_norm_scoring : ${LAYER_NORM_SCORING}"
echo "  divergence_scoring : ${DIVERGENCE_SCORING}"
echo "  budget_mode        : ${BUDGET_MODE}"
echo "  block_budget_frac  : ${BLOCK_BUDGET_FRACTION}"
echo "  task_order         : ${TASK_ORDER:-?}  (${ORDER_TAG:-?})"
echo "  projection_mode    : ${PROJECTION_MODE:-?}"
echo "  seed               : ${SEED:-?}"
echo "============================================================"

module load cuda/12.4.1-cz3ljd3
source /lustre/hdd/LAS/jannesar-lab/fatemask/miniconda3_new/etc/profile.d/conda.sh
conda activate /lustre/hdd/LAS/jannesar-lab/fatemask/seta_env
export PATH="/lustre/hdd/LAS/jannesar-lab/fatemask/seta_env/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Fix-control environment variables ────────────────────────────────────────
# These are read with os.environ.get() inside the Python code.  They must be
# EXPORTED (not just set) or the defaults silently apply.
#
#   GATE_MODE             sigmoid | softmax   [FIX-1] per-expert independent gates.
#                         'softmax' reproduces the original coupled behaviour and
#                         is the ablation arm for the paper.
#   GATE_OPEN_BIAS        initial gate bias for a newly created expert under
#                         sigmoid (4.0 -> sigma(4)=0.982, i.e. effectively on).
#   USE_REENTRANT         False = non-reentrant gradient checkpointing, which keeps
#                         the autograd graph so auxiliary losses built from
#                         intermediate tensors are not silently detached [FIX-7].
#   BUDGET_EXPANSION_CAP  ceiling for Step-C budget growth as a multiple of
#                         N_base [FIX-6].  1.0 == old behaviour (never expands).
#   LABEL_AUDIT_STRICT    True = abort when >2% of samples have no valid label
#                         tokens (the MeetingBank truncation failure) [FIX-2].
#   LABEL_AUDIT_SAMPLES   how many training samples to audit at startup.
#   LOSS_SANITY_OFF       1 = downgrade the "loss > 2*ln(vocab)" abort to a warning.
export GATE_MODE="${GATE_MODE:-sigmoid}"
export GATE_OPEN_BIAS="${GATE_OPEN_BIAS:-4.0}"
export USE_REENTRANT="${USE_REENTRANT:-False}"
export BUDGET_EXPANSION_CAP="${BUDGET_EXPANSION_CAP:-1.0}"
export LABEL_AUDIT_STRICT="${LABEL_AUDIT_STRICT:-True}"
export LABEL_AUDIT_SAMPLES="${LABEL_AUDIT_SAMPLES:-512}"
export LOSS_SANITY_OFF="${LOSS_SANITY_OFF:-0}"

# ── [DESIGN-1] Router regularisation ─────────────────────────────────────────
#   ROUTER_REG  none | penalty | freeze | anchor | anchor_sparse
# Each arm writes to its own directory tree (PHASE2_TAG in the launcher), so
# arms can run side by side without interfering.
export ROUTER_REG="${ROUTER_REG:-none}"
export ROUTER_ANCHOR_MU="${ROUTER_ANCHOR_MU:-1.0}"
export ROUTER_N_CENTROIDS="${ROUTER_N_CENTROIDS:-8}"
export ROUTER_CENTROID_BATCHES="${ROUTER_CENTROID_BATCHES:-16}"
export FFT_TAG="${FFT_TAG:-}"

# ── [DESIGN-2] Expert recycling ──────────────────────────────────────────────
export EXCLUDE_MODE="${EXCLUDE_MODE:-hard}"
export SOFT_EXCLUDE_ALPHA="${SOFT_EXCLUDE_ALPHA:-0.5}"
export CAPACITY_FLOOR="${CAPACITY_FLOOR:-0}"
export MERGE_LOW_UTIL="${MERGE_LOW_UTIL:-0}"
export MERGE_GATE_THRESHOLD="${MERGE_GATE_THRESHOLD:-0.15}"
export MERGE_CAND_MAX_BLOCKS="${MERGE_CAND_MAX_BLOCKS:-64}"
export MERGE_MAX_LAYERS="${MERGE_MAX_LAYERS:-1}"
export MERGE_PROTECT_LATEST="${MERGE_PROTECT_LATEST:-1}"
export MERGE_STATS_GLOB="${MERGE_STATS_GLOB:-}"
export MERGE_UTIL_MODE="${MERGE_UTIL_MODE:-relative}"
export MERGE_UTIL_Z="${MERGE_UTIL_Z:-1.0}"
export MERGE_UTIL_MIN_SPREAD="${MERGE_UTIL_MIN_SPREAD:-0.01}"
export EVAL_ROUTING_MODE="${EVAL_ROUTING_MODE:-}"
export EVAL_OUT_SUFFIX="${EVAL_OUT_SUFFIX:-}"
export GATE_PROBE="${GATE_PROBE:-0}"
export SEED="${SEED:-100}"

echo "------------------------------------------------------------"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "nvcc: $(which nvcc)"
echo "PyTorch CUDA: $(python3 -c 'import torch; print(torch.version.cuda)')"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "------------------------------------------------------------"

cd "${PROJECT_ROOT}"

user_model_name=$user_model_name

if [ "$user_model_name" == "llama3_8b" ]; then
    MODEL_PATH="/lustre/hdd/LAS/jannesar-lab/fatemask/models/llama3_8b"
elif [ "$user_model_name" == "llama2_7b" ]; then
    MODEL_PATH="/lustre/hdd/LAS/jannesar-lab/fatemask/models/llama2_7b"
elif [ "$user_model_name" == "qwen3_4b" ]; then
    MODEL_PATH="/lustre/hdd/LAS/jannesar-lab/fatemask/models/qwen3_4b"
elif [ "$user_model_name" == "qwen3_8b" ]; then
    MODEL_PATH="/lustre/hdd/LAS/jannesar-lab/fatemask/models/qwen3_8b"
elif [ "$user_model_name" == "qwen3_14b" ]; then
    MODEL_PATH="/lustre/hdd/LAS/jannesar-lab/fatemask/models/qwen3_14b"
else
    echo "Error: Unsupported model name '$user_model_name'."; exit 1
fi

echo "User Model Name: $user_model_name"
echo "Base Model Path: $MODEL_PATH"
echo "From slurm eval_task_id: $eval_task_id"


PROJECTION_MODE="${PROJECTION_MODE:-v_proj_only}"
PROJ_CALIB_THRESHOLD="${PROJ_CALIB_THRESHOLD:-0.10}"
PROJ_CALIB_STEPS="${PROJ_CALIB_STEPS:-20}"


case "$PROJECTION_MODE" in
    qkv_proj)   TARGET_MODULES_ARG='["q_proj", "k_proj", "v_proj"]' ;;
    q_proj)     TARGET_MODULES_ARG='["q_proj"]' ;;
    k_proj)     TARGET_MODULES_ARG='["k_proj"]' ;;
    mlp_proj)   TARGET_MODULES_ARG='["gate_proj", "up_proj", "down_proj"]' ;;
    v_proj_only|*) TARGET_MODULES_ARG='["v_proj"]' ;;
esac

case "$PROJECTION_MODE" in
    mlp_proj)      NSM_MLP=1; NSM_ATTN=0 ;;
    q_proj|k_proj|qkv_proj|v_proj_only|*) NSM_MLP=0; NSM_ATTN=1 ;;
esac

BLOCK_BUDGET_FRACTION="${BLOCK_BUDGET_FRACTION:-0.12}"   # keep in sync with the .sh

GRAD_TEMPORAL_DECAY="${GRAD_TEMPORAL_DECAY:-1.0}"

if [ -z "${BLOCK_DIMENSION:-}" ]; then
    case "$user_model_name" in
        llama2_7b)                                    BLOCK_DIMENSION=256 ;;
        llama3_8b|qwen3_4b|qwen3_8b|qwen3_14b)       BLOCK_DIMENSION=128 ;;
        *)                                            BLOCK_DIMENSION=256 ;;
    esac
fi

if [ -z "${EXPERT_CREATION_THRESHOLD:-}" ]; then
    case "$user_model_name" in
        qwen3_4b)   EXPERT_CREATION_THRESHOLD=8 ;;
        *)          EXPERT_CREATION_THRESHOLD=10 ;;
    esac
fi

if [ -z "${PRUNING_MIN_BLOCKS:-}" ]; then
    case "$user_model_name" in
        qwen3_4b)   PRUNING_MIN_BLOCKS=8 ;;
        *)          PRUNING_MIN_BLOCKS=10 ;;
    esac
fi

if [ -z "${TINY_REMAINDER_THRESHOLD:-}" ]; then
    case "$user_model_name" in
        qwen3_4b)   TINY_REMAINDER_THRESHOLD=8 ;;
        *)          TINY_REMAINDER_THRESHOLD=10 ;;
    esac
fi

if [ -z "${GLOBAL_EXPERT_MIN_BLOCKS:-}" ]; then
    case "$user_model_name" in
        qwen3_4b)   GLOBAL_EXPERT_MIN_BLOCKS=8  ;;
        *)          GLOBAL_EXPERT_MIN_BLOCKS=20 ;;
    esac
fi


case "$TASK_NAME" in
    MeetingBank)
        # ── [FIX-2 ROOT CAUSE] ────────────────────────────────────────────────
        # MeetingBank prompts are long meeting transcripts and the supervised
        # "### Response: <summary>" span sits at the END of the sequence.  At the
        # previous MAX_SEQ_LENGTH=768 (sample 0 alone is 712 tokens) the response
        # was truncated away on most samples, leaving ZERO valid label tokens.
        # The batch loss then became sum(CE)/tiny_denominator, which is why
        # Phase-2 train_loss reached 5.69e4 on llama2_7b and 6.24e4 on qwen3_4b —
        # two models with a 4.7x vocabulary difference landing within 10% of each
        # other, i.e. a normalisation artefact, not a numerical blow-up.
        # 20Minuten (646 tokens) fitted under 768 and trained fine, which is
        # exactly why the failure was MeetingBank-specific.
        # Batch is kept at 1 and grad-accum at 8 so the effective batch is
        # unchanged; only the window grows.
        case "$user_model_name" in
            llama3_8b)
                MAX_SEQ_LENGTH=1536
                TRAIN_BATCH=1
                GRAD_ACCUM=8
                ;;
            llama2_7b)
                MAX_SEQ_LENGTH=2048
                TRAIN_BATCH=1
                GRAD_ACCUM=8
                ;;
            qwen3_4b|qwen3_8b)
                MAX_SEQ_LENGTH=2048
                TRAIN_BATCH=1
                GRAD_ACCUM=8
                ;;
            *)
                MAX_SEQ_LENGTH=1536
                TRAIN_BATCH=1
                GRAD_ACCUM=8
                ;;
        esac
        ;;
    20Minuten)
        case "$user_model_name" in
            llama3_8b)
                MAX_SEQ_LENGTH=512
                TRAIN_BATCH=1
                GRAD_ACCUM=8
                ;;
            *)
                MAX_SEQ_LENGTH=768
                TRAIN_BATCH=2
                GRAD_ACCUM=4
                ;;
        esac
        ;;
    *)
        MAX_SEQ_LENGTH=512
        TRAIN_BATCH=2
        GRAD_ACCUM=4
        ;;
esac


ADAPTIVE_FFT_STEPS="${ADAPTIVE_FFT_STEPS:-True}"
FFT_MIN_STEPS="${FFT_MIN_STEPS:-30}"
FFT_MAX_STEPS="${FFT_MAX_STEPS:-150}"
FFT_REF_SAMPLES="${FFT_REF_SAMPLES:-500}"


NOVELTY_BETA_MIN="${NOVELTY_BETA_MIN:-0.30}"
NOVELTY_T_MAX="${NOVELTY_T_MAX:-6}"

# ── Dynamic dataset path based on Task ID ─────────────────────────────────────
EFFECTIVE_BASE_DATA_PATH="${BASE_DATA_PATH:-/lustre/hdd/LAS/jannesar-lab/fatemask/SETAV2/data_files/TRACE-Benchmark/LLM-CL-Benchmark_500}"
# Generation tasks read the 500-sample set. Matched by task NAME (was TASK_ID 3|6,
# which are exactly MeetingBank and 20Minuten in TASK_ORDER=1).
if [[ "$TASK_NAME" == "MeetingBank" || "$TASK_NAME" == "20Minuten" ]]; then
    EFFECTIVE_BASE_DATA_PATH="${GEN_DATA_PATH:-/lustre/hdd/LAS/jannesar-lab/fatemask/SETAV2/data_files/TRACE-Benchmark/LLM-CL-Benchmark_500}"
fi


echo "=== S1–S4 Active Parameter Values ==="
echo "  [S1.2] projection_mode         = $PROJECTION_MODE"
echo "  [S1.2] effective_base_data_path= $EFFECTIVE_BASE_DATA_PATH"
echo "  [S2.1] expert_creation_thresh   = $EXPERT_CREATION_THRESHOLD   [was: 5]"
echo "  [S2.1] pruning_min_blocks       = $PRUNING_MIN_BLOCKS         [was: 5]"
echo "  [S2.1] tiny_remainder_thresh     = $TINY_REMAINDER_THRESHOLD     [was: 5]"
echo "  [S2.1] block_dimension          = $BLOCK_DIMENSION              [was: 256]"
echo "  --- fix-control settings ---"
echo "  [FIX-1] gate_mode               = $GATE_MODE (open_bias=$GATE_OPEN_BIAS)"
echo "  [FIX-2] max_seq_length          = $MAX_SEQ_LENGTH  (label_audit_strict=$LABEL_AUDIT_STRICT)"
echo "  [FIX-6] budget_expansion_cap    = $BUDGET_EXPANSION_CAP"
echo "  [FIX-7] use_reentrant           = $USE_REENTRANT"
echo "  [D-1]   router_reg              = $ROUTER_REG (mu=$ROUTER_ANCHOR_MU)"
echo "  learning_rate                   = ${LEARNING_RATE:-9.65e-6}"
echo "  fft root                        = ${SOS_FFT_ROOT}"

echo "======================================"

run_step() {
  python -u "${PROJECT_ROOT}/main.py" \
    --seed ${SEED} \
    --model_name_or_path $MODEL_PATH \
    --use_model_name ${user_model_name} \
    --task_list "${TASK_LIST_STR}" \
    --chat_template_format None \
    --use_peft_SETA True \
    --SETA_dropout 0.0 \
    --SETA_offload False \
    --SETA_learning_rate 1e-4 \
    --SETA_w_decay 0.0 \
    --num_submatrix_mlp ${NSM_MLP} \
    --num_submatrix_attn ${NSM_ATTN} \
    --selection_strategy no_restriction \
    --calculation_strategy mean_abs \
    --target_modules "${TARGET_MODULES_ARG}" \
    --base_dataset_path "${EFFECTIVE_BASE_DATA_PATH}" \
    --eval_set_size 4 \
    --append_concat_token False \
    --packing False \
    --num_train_epochs 3 \
    --logging_steps 10 \
    --log_level info \
    --logging_strategy steps \
    --eval_steps 999999 \
    --eval_strategy no \
    --save_strategy no \
    --hub_private_repo False \
    --hub_strategy every_save \
    --learning_rate "${LEARNING_RATE:-9.65e-6}" \
    --lr_scheduler_type linear \
    --weight_decay 0.0 \
    --warmup_steps 2 \
    --max_grad_norm 1.0 \
    --per_device_eval_batch_size 1 \
    --per_device_train_batch_size ${TRAIN_BATCH} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --gradient_checkpointing True \
    --use_reentrant False \
    --full_ft_steps 100 \
    --load_best_model_at_end False \
    --metric_for_best_model None \
    --save_steps 999999 \
    --save_total_limit 0 \
    --lamda "${LAMDA:-0.1}" \
    --lambda_mode "${LAMBDA_MODE:-fixed}" \
    --gating_rl "${GATING_RL:-False}" \
    --max_samples "${MAX_SAMPLES:-100}" \
    --train_sample_ratio "${train_sample_ratio:-1.0}" \
    --do_train $DO_TRAIN \
    --do_eval $DO_EVAL \
    --resume_checkpoint_path $RESUME_PATH \
    --current_task_id $TASK_ID \
    --manual_output_path $SAVE_PATH \
    --output_dir $SAVE_PATH \
    --run_phase $RUN_PHASE \
    --eval_current_task_only "${EVAL_ONLY:-False}" \
    --skip_eval_task_list "${SKIP_EVAL_TASKS:-}" \
    --eval_task_id $eval_task_id \
    --expert_creation_threshold "${EXPERT_CREATION_THRESHOLD}" \
    --block_dimension           "${BLOCK_DIMENSION}" \
    --pruning_min_blocks        "${PRUNING_MIN_BLOCKS}" \
    --tiny_remainder_threshold  "${TINY_REMAINDER_THRESHOLD}" \
    --block_budget_fraction "${BLOCK_BUDGET_FRACTION}" \
    --fft_stopping_mode "${FFT_STOPPING_MODE}" \
    --layer_norm_scoring "${LAYER_NORM_SCORING}" \
    --divergence_scoring "${DIVERGENCE_SCORING}" \
    --budget_mode "${BUDGET_MODE}" \
    --ablation_tag "${ABLATION_TAG}" \
    --no_expert_sharing "${NO_EXPERT_SHARING:-False}" \
    --report_to none
}

# ── [FUSED] run every requested step inside this one allocation ──────────────
_FUSED_T0=$SECONDS
for _step in "${FUSED_STEPS[@]}"; do
    _step="$(echo "$_step" | tr -d '[:space:]')"
    [ -z "$_step" ] && continue
    set_phase_vars "$_step"
    _S0=$SECONDS
    echo ""
    echo "############################################################"
    echo "# [FUSED ${STEP_TAG}] task ${TASK_ID} (${TASK_NAME})  phase=${RUN_PHASE}"
    echo "# elapsed in job so far: $(( (SECONDS - _FUSED_T0) / 60 )) min"
    echo "############################################################"

    run_step
    _rc=$?
    if [ $_rc -ne 0 ]; then
        echo "" >&2
        echo "[FUSED] step '${_step}' FAILED with exit code ${_rc}." >&2
        echo "        Remaining steps are skipped so the failure is not masked." >&2
        echo "        Rerun just this task with STEPS=${_step},... once fixed." >&2
        exit $_rc
    fi
    echo "[FUSED ${STEP_TAG}] done in $(( (SECONDS - _S0) / 60 )) min"
done

echo ""
echo "[FUSED] all steps (${MODE}) completed for task ${TASK_ID} "\
     "in $(( (SECONDS - _FUSED_T0) / 60 )) min"

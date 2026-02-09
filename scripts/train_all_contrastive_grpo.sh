#!/bin/bash
# =============================================================================
# Launch All Contrastive GRPO Training Jobs (3 models × 2 datasets = 6 jobs)
#
# Each job uses 1 GPU. With 8 GPUs, all 6 run simultaneously.
#
# Usage:
#   ./scripts/train_all_contrastive_grpo.sh              # Run all 6
#   ./scripts/train_all_contrastive_grpo.sh --dry-run     # Print commands only
#   NUM_CHOICES=8 ./scripts/train_all_contrastive_grpo.sh # 8-way MCQ
# =============================================================================

set -e

# Configurable defaults
NUM_CHOICES="${NUM_CHOICES:-4}"
REWARD_TYPE="${REWARD_TYPE:-mrr}"
QUESTION_TYPE="${QUESTION_TYPE:-all}"
EPOCHS="${EPOCHS:-3}"
SLAKE_PATH="${SLAKE_PATH:-./data/Slake1.0}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-./checkpoints}"
SEED="${SEED:-42}"
DRY_RUN=false

if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# Model definitions
MODELS=(
    "Qwen/Qwen3-VL-8B-Instruct:qwen3vl_8b"
    "OpenGVLab/InternVL3-8B-hf:internvl3_8b"
    "llava-hf/llava-v1.6-mistral-7b-hf:llava_next_7b"
)
DATASETS=("rad_vqa" "slake")

echo "============================================================"
echo "  Contrastive GRPO Training - All Jobs"
echo "============================================================"
echo "  MCQ Choices:    ${NUM_CHOICES}"
echo "  Reward Type:    ${REWARD_TYPE}"
echo "  Question Type:  ${QUESTION_TYPE}"
echo "  Epochs:         ${EPOCHS}"
echo "  Seed:           ${SEED}"
echo "  Checkpoint Dir: ${CHECKPOINT_BASE}"
echo "============================================================"
echo ""

GPU=0
PIDS=()
LOG_DIR="${CHECKPOINT_BASE}/contrastive_logs"
mkdir -p "$LOG_DIR"

for model_entry in "${MODELS[@]}"; do
    MODEL_ID="${model_entry%%:*}"
    MODEL_KEY="${model_entry##*:}"

    for DATASET in "${DATASETS[@]}"; do
        OUTPUT_DIR="${CHECKPOINT_BASE}/contrast_grpo_${MODEL_KEY}_${DATASET}"
        LOG_FILE="${LOG_DIR}/contrast_grpo_${MODEL_KEY}_${DATASET}.log"

        # Dataset-specific args
        EXTRA_ARGS=""
        if [[ "$DATASET" == "slake" ]]; then
            EXTRA_ARGS="--slake_path ${SLAKE_PATH}"
        fi

        CMD="python scripts/train_grpo_contrastive.py \
            --model_id ${MODEL_ID} \
            --dataset ${DATASET} \
            --output_dir ${OUTPUT_DIR} \
            --num_choices ${NUM_CHOICES} \
            --reward_type ${REWARD_TYPE} \
            --question_type ${QUESTION_TYPE} \
            --hard_negatives \
            --epochs ${EPOCHS} \
            --seed ${SEED} \
            --gpu ${GPU} \
            ${EXTRA_ARGS}"

        echo "[GPU ${GPU}] ${MODEL_KEY} × ${DATASET}"
        echo "  → ${OUTPUT_DIR}"

        if $DRY_RUN; then
            echo "  CMD: ${CMD}"
            echo ""
        else
            echo "  Log: ${LOG_FILE}"
            nohup bash -c "${CMD}" > "${LOG_FILE}" 2>&1 &
            PIDS+=($!)
            echo "  PID: ${PIDS[-1]}"
            echo ""
        fi

        GPU=$((GPU + 1))
    done
done

if $DRY_RUN; then
    echo "[DRY RUN] Would launch ${#MODELS[@]} × ${#DATASETS[@]} = $((${#MODELS[@]} * ${#DATASETS[@]})) jobs on GPUs 0-$((GPU - 1))"
    exit 0
fi

echo "============================================================"
echo "  ${#PIDS[@]} jobs launched on GPUs 0-$((GPU - 1))"
echo "  PIDs: ${PIDS[*]}"
echo "============================================================"
echo ""
echo "Monitor:"
echo "  tail -f ${LOG_DIR}/contrast_grpo_*.log"
echo ""
echo "Wait for all:"
echo "  wait ${PIDS[*]}"
echo ""

# Optionally wait
if [[ "$1" == "--wait" ]] || [[ "$2" == "--wait" ]]; then
    echo "Waiting for all jobs to complete..."
    FAILED=0
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            FAILED=$((FAILED + 1))
        fi
    done
    echo ""
    echo "Done. Failed: ${FAILED}/${#PIDS[@]}"
fi
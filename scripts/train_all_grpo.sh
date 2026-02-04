#!/bin/bash
# =============================================================================
# Medical VQA GRPO Training — Multi-Model Parallel Training
# =============================================================================
# Trains Qwen3-VL-8B, InternVL3-8B, and LLaVA-NeXT-7B on RAD-VQA & SLAKE
# using GRPO with verifiable rewards on closed (yes/no) questions.
#
# Runs 6 jobs in parallel on GPUs 0-5 (one job per GPU).
# GPUs 6-7 are left free for evaluation or smoke tests.
#
# Models (3 models × 2 datasets = 6 jobs):
#   - Qwen/Qwen3-VL-8B-Instruct
#   - OpenGVLab/InternVL3-8B-hf
#   - llava-hf/llava-v1.6-mistral-7b-hf
#
# Datasets:
#   - RAD-VQA  (auto-downloaded from HuggingFace)
#   - SLAKE    (local at ./data/Slake1.0)
#
# Standard hyperparameters (consistent with SFT runs):
#   LR=5e-6, LoRA r=64 α=128, epochs=3, effective BS=8
#   num_generations=4, temperature=1.0, beta=0.0, loss=DAPO
#   save_strategy=epoch, save_total_limit=5 (keeps all 3 epoch checkpoints)
#
# Usage:
#   ./scripts/train_all_grpo.sh                    # Run all 6 jobs
#   ./scripts/train_all_grpo.sh --dry-run           # Print commands only
#   SLAKE_PATH=/my/path ./scripts/train_all_grpo.sh # Override SLAKE location
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================

SLAKE_PATH="${SLAKE_PATH:-./data/Slake1.0}"
OUTPUT_BASE="${OUTPUT_BASE:-./checkpoints}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE — commands printed but not executed ==="
fi

# =============================================================================
# Hyperparameters (standard setup, consistent across all models)
# =============================================================================

LEARNING_RATE=5e-6
LORA_R=64
LORA_ALPHA=128
NUM_EPOCHS=3
BATCH_SIZE=1
GRAD_ACCUM=8           # effective BS = 1 × 8 = 8
NUM_GENERATIONS=8
TEMPERATURE=0.8
BETA=0.0
LOSS_TYPE="dapo"
MAX_COMPLETION_LENGTH=128
QUESTION_TYPE="closed"

ACCURACY_WEIGHT=3.0
FORMAT_WEIGHT=1.0

SAVE_STRATEGY="epoch"
SAVE_TOTAL_LIMIT=5
REPORT_TO="none"

# =============================================================================
# Model Registry
# =============================================================================

declare -A MODELS
MODELS["qwen"]="Qwen/Qwen3-VL-8B-Instruct"
MODELS["internvl"]="OpenGVLab/InternVL3-8B-hf"
MODELS["llava"]="llava-hf/llava-v1.6-mistral-7b-hf"

declare -A MODEL_NAMES
MODEL_NAMES["qwen"]="qwen3vl_8b"
MODEL_NAMES["internvl"]="internvl3_8b"
MODEL_NAMES["llava"]="llava_next_7b"

# =============================================================================
# GPU Assignment (6 jobs → GPUs 0-5)
# =============================================================================

declare -A GPU_MAP
# RAD-VQA
GPU_MAP["qwen_rad_vqa"]=0
GPU_MAP["internvl_rad_vqa"]=1
GPU_MAP["llava_rad_vqa"]=2
# SLAKE
GPU_MAP["qwen_slake"]=3
GPU_MAP["internvl_slake"]=4
GPU_MAP["llava_slake"]=5

# =============================================================================
# Helper
# =============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

run_training() {
    local MODEL_KEY=$1
    local DATASET=$2
    local GPU=$3

    local MODEL_ID="${MODELS[$MODEL_KEY]}"
    local MODEL_NAME="${MODEL_NAMES[$MODEL_KEY]}"
    local OUTPUT_DIR="${OUTPUT_BASE}/grpo_${DATASET}_${MODEL_NAME}_${QUESTION_TYPE}_lr${LEARNING_RATE}_r${LORA_R}_${TIMESTAMP}"
    local LOG_FILE="${OUTPUT_BASE}/logs/grpo_${MODEL_NAME}_${DATASET}_${TIMESTAMP}.log"

    mkdir -p "${OUTPUT_BASE}/logs"

    local CMD="python scripts/train_grpo.py \
        --model_id ${MODEL_ID} \
        --output_dir ${OUTPUT_DIR} \
        --dataset ${DATASET} \
        --question_type ${QUESTION_TYPE} \
        --gpu ${GPU} \
        --epochs ${NUM_EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --grad_accum ${GRAD_ACCUM} \
        --learning_rate ${LEARNING_RATE} \
        --num_generations ${NUM_GENERATIONS} \
        --temperature ${TEMPERATURE} \
        --beta ${BETA} \
        --loss_type ${LOSS_TYPE} \
        --max_completion_length ${MAX_COMPLETION_LENGTH} \
        --lora_r ${LORA_R} \
        --lora_alpha ${LORA_ALPHA} \
        --accuracy_weight ${ACCURACY_WEIGHT} \
        --format_weight ${FORMAT_WEIGHT} \
        --save_strategy ${SAVE_STRATEGY} \
        --save_total_limit ${SAVE_TOTAL_LIMIT} \
        --report_to ${REPORT_TO}"

    # Add SLAKE path if needed
    if [ "$DATASET" = "slake" ]; then
        CMD="${CMD} --slake_path ${SLAKE_PATH}"
    fi

    if [ "$DRY_RUN" = true ]; then
        echo ""
        echo "=== ${MODEL_NAME} / ${DATASET} / GPU ${GPU} ==="
        echo "$CMD"
        echo "Output: $OUTPUT_DIR"
        echo "Log:    $LOG_FILE"
    else
        log "Starting: ${MODEL_NAME} / ${DATASET} on GPU ${GPU}"
        log "Output:   ${OUTPUT_DIR}"
        log "Log:      ${LOG_FILE}"

        nohup $CMD > "$LOG_FILE" 2>&1 &
        local PID=$!
        echo "$PID ${MODEL_NAME} ${DATASET} ${GPU}" >> "${OUTPUT_BASE}/logs/grpo_pids_${TIMESTAMP}.txt"
        log "Started with PID: $PID"
    fi
}

# =============================================================================
# Pre-flight Checks
# =============================================================================

echo "============================================================"
echo "Medical VQA GRPO — Multi-Model Parallel Training"
echo "============================================================"
echo "Timestamp:       $TIMESTAMP"
echo "SLAKE Path:      $SLAKE_PATH"
echo "Output Base:     $OUTPUT_BASE"
echo ""
echo "Hyperparameters:"
echo "  Learning Rate:   $LEARNING_RATE"
echo "  LoRA r:          $LORA_R"
echo "  LoRA α:          $LORA_ALPHA"
echo "  Epochs:          $NUM_EPOCHS"
echo "  Effective BS:    $((BATCH_SIZE * GRAD_ACCUM))"
echo "  Num Generations: $NUM_GENERATIONS"
echo "  Temperature:     $TEMPERATURE"
echo "  Beta (KL):       $BETA"
echo "  Loss Type:       $LOSS_TYPE"
echo "  Question Type:   $QUESTION_TYPE"
echo ""
echo "Reward Weights:"
echo "  Accuracy:        $ACCURACY_WEIGHT"
echo "  Format:          $FORMAT_WEIGHT"
echo ""
echo "Models:"
echo "  Qwen:     ${MODELS[qwen]}"
echo "  InternVL: ${MODELS[internvl]}"
echo "  LLaVA:    ${MODELS[llava]}"
echo ""
echo "Datasets:"
echo "  - RAD-VQA (HuggingFace)"
echo "  - SLAKE   (local: $SLAKE_PATH)"
echo "============================================================"

# Check SLAKE path
if [ ! -d "$SLAKE_PATH" ]; then
    echo "ERROR: SLAKE dataset not found at $SLAKE_PATH"
    echo "Set SLAKE_PATH environment variable or place data at ./data/Slake1.0"
    exit 1
fi
if [ ! -f "$SLAKE_PATH/train.json" ]; then
    echo "ERROR: train.json not found in $SLAKE_PATH"
    exit 1
fi

echo ""
echo "SLAKE dataset found. Launching training jobs..."
echo ""

# =============================================================================
# Launch (6 jobs: 3 models × 2 datasets)
# =============================================================================

# RAD-VQA (GPUs 0-2)
for MODEL_KEY in qwen internvl llava; do
    GPU=${GPU_MAP["${MODEL_KEY}_rad_vqa"]}
    run_training "$MODEL_KEY" "rad_vqa" "$GPU"
done

# SLAKE (GPUs 3-5)
for MODEL_KEY in qwen internvl llava; do
    GPU=${GPU_MAP["${MODEL_KEY}_slake"]}
    run_training "$MODEL_KEY" "slake" "$GPU"
done

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "============================================================"
echo "Training Jobs Launched"
echo "============================================================"

if [ "$DRY_RUN" = true ]; then
    echo "Dry run complete. No jobs started."
else
    echo ""
    echo "6 GRPO training jobs running in parallel:"
    echo ""
    echo "  RAD-VQA:"
    echo "    GPU 0: Qwen3-VL-8B"
    echo "    GPU 1: InternVL3-8B"
    echo "    GPU 2: LLaVA-NeXT-7B"
    echo ""
    echo "  SLAKE:"
    echo "    GPU 3: Qwen3-VL-8B"
    echo "    GPU 4: InternVL3-8B"
    echo "    GPU 5: LLaVA-NeXT-7B"
    echo ""
    echo "  Free:  GPUs 6-7 (for eval / smoke tests)"
    echo ""
    echo "Monitor progress:"
    echo "  tail -f ${OUTPUT_BASE}/logs/grpo_*_${TIMESTAMP}.log"
    echo ""
    echo "Check running jobs:"
    echo "  cat ${OUTPUT_BASE}/logs/grpo_pids_${TIMESTAMP}.txt"
    echo ""
    echo "GPU utilization:"
    echo "  watch -n 1 nvidia-smi"
    echo "============================================================"
fi
#!/bin/bash
# =============================================================================
# Medical VQA SFT Training - Multi-Model Parallel Training
# =============================================================================
# Trains Qwen-VL, InternVL, and LLaVA-NeXT models on SLAKE and RAD-VQA
# Runs in parallel across GPUs 0-5
#
# Models (3 models × 2 datasets = 6 jobs):
#   - Qwen/Qwen2-VL-7B-Instruct
#   - OpenGVLab/InternVL3-8B-hf  
#   - llava-hf/llava-v1.6-mistral-7b-hf
#
# Datasets:
#   - SLAKE (requires local path)
#   - RAD-VQA (auto-downloaded from HuggingFace)
#
# Usage:
#   ./train_all_models.sh                      # Run all
#   ./train_all_models.sh --dry-run            # Print commands only
#   SLAKE_PATH=/path/to/Slake1.0 ./train_all_models.sh
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================

# SLAKE dataset path
SLAKE_PATH="./data/Slake1.0"

# Output base directory
OUTPUT_BASE="${OUTPUT_BASE:-./checkpoints}"

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Dry run mode
DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE - Commands will be printed but not executed ==="
fi

# =============================================================================
# Hyperparameters (consistent across all models)
# =============================================================================

LEARNING_RATE=5e-5
LORA_R=64
LORA_ALPHA=128
NUM_EPOCHS=5
WARMUP_RATIO=0.03
MAX_LENGTH=2048
SAVE_STRATEGY="epoch"
SAVE_TOTAL_LIMIT=5
QUESTION_TYPE="all"

# Batch size: effective batch size = 16 for all models
BATCH_SIZE=2
GRAD_ACCUM=8

# =============================================================================
# Model Definitions
# =============================================================================

declare -A MODELS
MODELS["qwen"]="Qwen/Qwen3-VL-8B-Instruct"
MODELS["internvl"]="OpenGVLab/InternVL3-8B-hf"
MODELS["llava"]="llava-hf/llava-v1.6-mistral-7b-hf"

# Short names for output directories
declare -A MODEL_NAMES
MODEL_NAMES["qwen"]="qwen3vl_8b"
MODEL_NAMES["internvl"]="internvl3_8b"
MODEL_NAMES["llava"]="llava_next_7b"

# =============================================================================
# GPU Assignment (6 jobs total: 3 models × 2 datasets)
# =============================================================================

declare -A GPU_MAP
# SLAKE
GPU_MAP["qwen_slake"]=0
GPU_MAP["internvl_slake"]=1
GPU_MAP["llava_slake"]=2
# RAD-VQA
GPU_MAP["qwen_rad_vqa"]=3
GPU_MAP["internvl_rad_vqa"]=4
GPU_MAP["llava_rad_vqa"]=5

# =============================================================================
# Helper Functions
# =============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

run_training() {
    local MODEL_KEY=$1
    local DATASET=$2
    local GPU=$3
    
    local MODEL_ID=${MODELS[$MODEL_KEY]}
    local MODEL_NAME=${MODEL_NAMES[$MODEL_KEY]}
    local OUTPUT_DIR="${OUTPUT_BASE}/${DATASET}_${MODEL_NAME}_${QUESTION_TYPE}_lr${LEARNING_RATE}_r${LORA_R}_${TIMESTAMP}"
    local LOG_FILE="${OUTPUT_BASE}/logs/${MODEL_NAME}_${DATASET}_${TIMESTAMP}.log"
    
    # Create log directory
    mkdir -p "${OUTPUT_BASE}/logs"
    
    # Build command (using arguments supported by current train_sft.py)
    local CMD="python scripts/train_sft.py \
        --model_id ${MODEL_ID} \
        --output_dir ${OUTPUT_DIR} \
        --dataset ${DATASET} \
        --question_type ${QUESTION_TYPE} \
        --split train \
        --epochs ${NUM_EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --grad_accum ${GRAD_ACCUM} \
        --learning_rate ${LEARNING_RATE} \
        --max_length ${MAX_LENGTH} \
        --lora_r ${LORA_R} \
        --lora_alpha ${LORA_ALPHA} \
        --gpu ${GPU}"
    
    if [ "$DRY_RUN" = true ]; then
        echo ""
        echo "=== ${MODEL_NAME} / ${DATASET} / GPU ${GPU} ==="
        echo "$CMD"
        echo "Log: $LOG_FILE"
    else
        log "Starting: ${MODEL_NAME} / ${DATASET} on GPU ${GPU}"
        log "Output: ${OUTPUT_DIR}"
        log "Log: ${LOG_FILE}"
        
        # Run in background with nohup
        nohup $CMD > "$LOG_FILE" 2>&1 &
        local PID=$!
        echo "$PID ${MODEL_NAME} ${DATASET} ${GPU}" >> "${OUTPUT_BASE}/logs/pids_${TIMESTAMP}.txt"
        log "Started with PID: $PID"
    fi
}

# =============================================================================
# Pre-flight Checks
# =============================================================================

echo "============================================================"
echo "Medical VQA Multi-Model Parallel Training"
echo "============================================================"
echo "Timestamp:       $TIMESTAMP"
echo "SLAKE Path:      $SLAKE_PATH"
echo "Output Base:     $OUTPUT_BASE"
echo ""
echo "Hyperparameters:"
echo "  Learning Rate: $LEARNING_RATE"
echo "  LoRA r:        $LORA_R"
echo "  LoRA α:        $LORA_ALPHA"
echo "  Epochs:        $NUM_EPOCHS"
echo "  Effective BS:  $((BATCH_SIZE * GRAD_ACCUM))"
echo "  Question Type: $QUESTION_TYPE"
echo ""
echo "Models:"
echo "  Qwen:     ${MODELS[qwen]}"
echo "  InternVL: ${MODELS[internvl]}"
echo "  LLaVA:    ${MODELS[llava]}"
echo ""
echo "Datasets:"
echo "  - SLAKE (local)"
echo "  - RAD-VQA (HuggingFace)"
echo "============================================================"

# Check SLAKE path
if [ ! -d "$SLAKE_PATH" ]; then
    echo "ERROR: SLAKE dataset not found at $SLAKE_PATH"
    echo "Please set SLAKE_PATH environment variable."
    exit 1
fi

# Check if train.json exists
if [ ! -f "$SLAKE_PATH/train.json" ]; then
    echo "ERROR: train.json not found in $SLAKE_PATH"
    exit 1
fi

echo ""
echo "SLAKE dataset found. Starting training jobs..."
echo ""

# =============================================================================
# Launch Training Jobs
# =============================================================================

# SLAKE training (GPUs 0-2)
for MODEL_KEY in qwen internvl llava; do
    GPU=${GPU_MAP["${MODEL_KEY}_slake"]}
    run_training "$MODEL_KEY" "slake" "$GPU"
done

# RAD-VQA training (GPUs 3-5)
for MODEL_KEY in qwen internvl llava; do
    GPU=${GPU_MAP["${MODEL_KEY}_rad_vqa"]}
    run_training "$MODEL_KEY" "rad_vqa" "$GPU"
done

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "============================================================"
echo "Training Jobs Launched"
echo "============================================================"

if [ "$DRY_RUN" = true ]; then
    echo "Dry run complete. No jobs were started."
else
    echo ""
    echo "6 training jobs started in parallel:"
    echo ""
    echo "  SLAKE Dataset:"
    echo "    GPU 0: Qwen3-VL-8B"
    echo "    GPU 1: InternVL3-8B"
    echo "    GPU 2: LLaVA-NeXT-7B"
    echo ""
    echo "  RAD-VQA Dataset:"
    echo "    GPU 3: Qwen3-VL-8B"
    echo "    GPU 4: InternVL3-8B"
    echo "    GPU 5: LLaVA-NeXT-7B"
    echo ""
    echo "Monitor progress:"
    echo "  ./scripts/monitor_training.sh"
    echo "  tail -f ${OUTPUT_BASE}/logs/*_${TIMESTAMP}.log"
    echo ""
    echo "Check running jobs:"
    echo "  cat ${OUTPUT_BASE}/logs/pids_${TIMESTAMP}.txt"
    echo ""
    echo "GPU utilization:"
    echo "  watch -n 1 nvidia-smi"
    echo ""
    echo "============================================================"
fi
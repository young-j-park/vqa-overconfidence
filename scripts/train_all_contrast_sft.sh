#!/bin/bash
# =============================================================================
# Batch Launcher: Light SFT Adaptation for Contrastive GRPO Models
#
# Runs 1-epoch SFT adaptation on contrastive-GRPO-trained models.
# Currently: InternVL3-8B and Qwen3-VL-8B on RAD-VQA and SLAKE
# (LLaVA excluded — contrastive GRPO not yet fully trained)
#
# Usage:
#   bash scripts/train_all_contrast_sft.sh              # Run all 4 jobs
#   bash scripts/train_all_contrast_sft.sh --dry-run    # Preview commands
#   DRY_RUN=true bash scripts/train_all_contrast_sft.sh # Same as above
# =============================================================================

set -euo pipefail

# ---- Configuration ----
CHECKPOINT_BASE="./checkpoints"
OUTPUT_BASE="./checkpoints"
SLAKE_PATH="./data/Slake1.0"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# Training hyperparameters (conservative for light adaptation)
LEARNING_RATE=1e-5
NUM_EPOCHS=1
BATCH_SIZE=4
GRAD_ACCUM=8
LORA_R=64
LORA_ALPHA=128
QUESTION_TYPE="closed"

# Parse --dry-run flag
DRY_RUN=${DRY_RUN:-false}
for arg in "$@"; do
    if [ "$arg" = "--dry-run" ]; then
        DRY_RUN=true
    fi
done

# ---- Model registry ----
declare -A MODELS
MODELS[internvl]="OpenGVLab/InternVL3-8B-hf"
MODELS[qwen]="Qwen/Qwen3-VL-8B-Instruct"

declare -A MODEL_KEYS
MODEL_KEYS[internvl]="internvl3_8b"
MODEL_KEYS[qwen]="qwen3vl_8b"

# ---- Contrastive GRPO checkpoint mapping ----
# Format: contrast_grpo_{model_key}_{dataset}/final_model
declare -A CONTRAST_ADAPTERS
CONTRAST_ADAPTERS[internvl_rad_vqa]="${CHECKPOINT_BASE}/contrast_grpo_internvl3_8b_rad_vqa/final_model"
CONTRAST_ADAPTERS[internvl_slake]="${CHECKPOINT_BASE}/contrast_grpo_internvl3_8b_slake/final_model"
CONTRAST_ADAPTERS[qwen_rad_vqa]="${CHECKPOINT_BASE}/contrast_grpo_qwen3vl_8b_rad_vqa/final_model"
CONTRAST_ADAPTERS[qwen_slake]="${CHECKPOINT_BASE}/contrast_grpo_qwen3vl_8b_slake/final_model"

# ---- GPU assignment ----
# 4 jobs → assign to 4 GPUs (adjust as needed)
declare -A GPU_MAP
GPU_MAP[internvl_rad_vqa]=0
GPU_MAP[internvl_slake]=1
GPU_MAP[qwen_rad_vqa]=2
GPU_MAP[qwen_slake]=3

# ---- Helper ----
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

run_job() {
    local MODEL_SHORT=$1   # internvl or qwen
    local DATASET=$2       # rad_vqa or slake
    local JOB_KEY="${MODEL_SHORT}_${DATASET}"

    local MODEL_ID="${MODELS[$MODEL_SHORT]}"
    local MODEL_KEY="${MODEL_KEYS[$MODEL_SHORT]}"
    local ADAPTER="${CONTRAST_ADAPTERS[$JOB_KEY]}"
    local GPU="${GPU_MAP[$JOB_KEY]}"
    local OUTPUT_DIR="${OUTPUT_BASE}/contrast_sft_${MODEL_KEY}_${DATASET}"
    local LOG_DIR="${OUTPUT_BASE}/logs"
    local LOG_FILE="${LOG_DIR}/contrast_sft_${MODEL_KEY}_${DATASET}_${TIMESTAMP}.log"

    mkdir -p "${LOG_DIR}"

    # Build command
    local CMD="python scripts/train_contrast_sft_adapt.py \
        --model_id ${MODEL_ID} \
        --contrast_adapter ${ADAPTER} \
        --dataset ${DATASET} \
        --question_type ${QUESTION_TYPE} \
        --output_dir ${OUTPUT_DIR} \
        --epochs ${NUM_EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --grad_accum ${GRAD_ACCUM} \
        --learning_rate ${LEARNING_RATE} \
        --lora_r ${LORA_R} \
        --lora_alpha ${LORA_ALPHA} \
        --gpu ${GPU}"

    # Add SLAKE path if needed
    if [ "$DATASET" = "slake" ]; then
        CMD="${CMD} --slake_path ${SLAKE_PATH}"
    fi

    if [ "$DRY_RUN" = true ]; then
        echo ""
        echo "=== ${MODEL_KEY} / ${DATASET} / GPU ${GPU} ==="
        echo "$CMD"
        echo "Output: $OUTPUT_DIR"
        echo "Log:    $LOG_FILE"
    else
        # Check adapter exists
        if [ ! -d "$ADAPTER" ]; then
            log "WARNING: Adapter not found: $ADAPTER — skipping ${JOB_KEY}"
            return
        fi

        log "Starting: ${MODEL_KEY} / ${DATASET} on GPU ${GPU}"
        log "  Adapter: ${ADAPTER}"
        log "  Output:  ${OUTPUT_DIR}"
        log "  Log:     ${LOG_FILE}"

        nohup $CMD > "$LOG_FILE" 2>&1 &
        local PID=$!
        echo "$PID ${MODEL_KEY} ${DATASET} ${GPU}" >> "${LOG_DIR}/pids_contrast_sft_${TIMESTAMP}.txt"
        log "  Started with PID: $PID"
    fi
}

# ---- Print banner ----
echo "============================================================"
echo "Contrastive GRPO → Light SFT Adaptation"
echo "============================================================"
echo "Timestamp:       $TIMESTAMP"
echo "Checkpoint Base: $CHECKPOINT_BASE"
echo "Output Base:     $OUTPUT_BASE"
echo ""
echo "Adaptation Config:"
echo "  Epochs:        $NUM_EPOCHS"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Effective BS:  $((BATCH_SIZE * GRAD_ACCUM))"
echo "  LoRA r:        $LORA_R"
echo "  LoRA α:        $LORA_ALPHA"
echo "  Question Type: $QUESTION_TYPE"
echo ""
echo "Models:"
echo "  InternVL3-8B: ${MODELS[internvl]}"
echo "  Qwen3-VL-8B:  ${MODELS[qwen]}"
echo ""
echo "Jobs (4 total):"
echo "  GPU ${GPU_MAP[internvl_rad_vqa]}: InternVL3 / RAD-VQA"
echo "  GPU ${GPU_MAP[internvl_slake]}: InternVL3 / SLAKE"
echo "  GPU ${GPU_MAP[qwen_rad_vqa]}: Qwen3-VL  / RAD-VQA"
echo "  GPU ${GPU_MAP[qwen_slake]}: Qwen3-VL  / SLAKE"
if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "  *** DRY RUN — commands will be printed, not executed ***"
fi
echo "============================================================"

# ---- Pre-flight: check SLAKE ----
if [ ! -d "$SLAKE_PATH" ]; then
    echo "WARNING: SLAKE dataset not found at $SLAKE_PATH"
    echo "SLAKE jobs will fail unless the path is corrected."
fi

# ---- Launch all jobs ----
for model in internvl qwen; do
    for dataset in rad_vqa slake; do
        run_job "$model" "$dataset"
    done
done

# ---- Summary ----
if [ "$DRY_RUN" = false ]; then
    echo ""
    echo "============================================================"
    echo "All jobs launched! Monitor with:"
    echo "  tail -f ${OUTPUT_BASE}/logs/contrast_sft_*_${TIMESTAMP}.log"
    echo ""
    echo "PIDs saved to:"
    echo "  ${OUTPUT_BASE}/logs/pids_contrast_sft_${TIMESTAMP}.txt"
    echo "============================================================"
fi

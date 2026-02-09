#!/bin/bash
# =============================================================================
# Augmented Training — Multi-Model Parallel Launcher
# =============================================================================
# Runs augmented SFT and GRPO training for all model × dataset combinations.
# Uses SAME hyperparameters as the baseline scripts (train_all_models.sh and
# train_all_grpo.sh) for fair comparison.
#
# Prerequisites:
#   1. Run scripts/generate_paraphrases.py first to create augmentation cache
#   2. Baseline models already trained (for comparison)
#
# Usage:
#   ./scripts/train_all_augmented.sh                 # Run all
#   ./scripts/train_all_augmented.sh --dry-run        # Preview commands
#   ./scripts/train_all_augmented.sh --sft-only       # SFT only
#   ./scripts/train_all_augmented.sh --grpo-only      # GRPO only
# =============================================================================

set -e

SLAKE_PATH="${SLAKE_PATH:-./data/Slake1.0}"
OUTPUT_BASE="${OUTPUT_BASE:-./checkpoints}"
AUGMENTED_DIR="${AUGMENTED_DIR:-./data/augmented}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NUM_PARAPHRASES="${NUM_PARAPHRASES:-8}"

DRY_RUN=false
SFT_ONLY=false
GRPO_ONLY=false

for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
        --sft-only) SFT_ONLY=true ;;
        --grpo-only) GRPO_ONLY=true ;;
    esac
done

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
# SFT Hyperparameters (SAME as train_all_models.sh)
# =============================================================================
SFT_LR=5e-5
SFT_EPOCHS=5
SFT_BATCH=2
SFT_GRAD_ACCUM=8
SFT_LORA_R=64
SFT_LORA_ALPHA=128
SFT_QUESTION_TYPE="all"

# =============================================================================
# GRPO Hyperparameters (SAME as train_all_grpo.sh)
# =============================================================================
GRPO_LR=5e-6
GRPO_EPOCHS=3
GRPO_BATCH=1
GRPO_GRAD_ACCUM=8
GRPO_NUM_GEN=8
GRPO_TEMP=0.8
GRPO_BETA=0.0
GRPO_LOSS="dapo"
GRPO_MAX_COMP=128
GRPO_QUESTION_TYPE="closed"
GRPO_ACC_WEIGHT=3.0
GRPO_FMT_WEIGHT=1.0

# =============================================================================
# GPU Assignment
# =============================================================================
# 6 jobs at a time: 3 models × 2 datasets
# SFT first (GPUs 0-5), then GRPO (GPUs 0-5)

declare -A GPU_MAP
GPU_MAP["qwen_rad_vqa"]=0
GPU_MAP["internvl_rad_vqa"]=1
GPU_MAP["llava_rad_vqa"]=2
GPU_MAP["qwen_slake"]=3
GPU_MAP["internvl_slake"]=4
GPU_MAP["llava_slake"]=5

# =============================================================================
# Helpers
# =============================================================================

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

run_augmented_sft() {
    local MODEL_KEY=$1 DATASET=$2 GPU=$3
    local MODEL_ID="${MODELS[$MODEL_KEY]}"
    local MODEL_NAME="${MODEL_NAMES[$MODEL_KEY]}"
    local OUTPUT_DIR="${OUTPUT_BASE}/aug_sft_${DATASET}_${MODEL_NAME}_q_only_n${NUM_PARAPHRASES}_${TIMESTAMP}"
    local LOG_FILE="${OUTPUT_BASE}/logs/aug_sft_${MODEL_NAME}_${DATASET}_${TIMESTAMP}.log"

    mkdir -p "${OUTPUT_BASE}/logs"

    local CMD="python scripts/train_sft_augmented.py \
        --model_id ${MODEL_ID} \
        --output_dir ${OUTPUT_DIR} \
        --dataset ${DATASET} \
        --question_type ${SFT_QUESTION_TYPE} \
        --augmented_dir ${AUGMENTED_DIR} \
        --augment_mode q_only \
        --num_paraphrases ${NUM_PARAPHRASES} \
        --epochs ${SFT_EPOCHS} \
        --batch_size ${SFT_BATCH} \
        --grad_accum ${SFT_GRAD_ACCUM} \
        --learning_rate ${SFT_LR} \
        --lora_r ${SFT_LORA_R} \
        --lora_alpha ${SFT_LORA_ALPHA} \
        --gpu ${GPU}"

    [ "$DATASET" = "slake" ] && CMD="$CMD --slake_path ${SLAKE_PATH}"

    if [ "$DRY_RUN" = true ]; then
        echo "=== AUG-SFT: ${MODEL_NAME} / ${DATASET} / GPU ${GPU} ==="
        echo "$CMD"
    else
        log "AUG-SFT: ${MODEL_NAME} / ${DATASET} on GPU ${GPU}"
        nohup $CMD > "$LOG_FILE" 2>&1 &
        echo "$! aug_sft_${MODEL_NAME} ${DATASET} ${GPU}" >> "${OUTPUT_BASE}/logs/aug_pids_${TIMESTAMP}.txt"
        log "  PID: $!"
    fi
}

run_augmented_grpo() {
    local MODEL_KEY=$1 DATASET=$2 GPU=$3
    local MODEL_ID="${MODELS[$MODEL_KEY]}"
    local MODEL_NAME="${MODEL_NAMES[$MODEL_KEY]}"
    local OUTPUT_DIR="${OUTPUT_BASE}/aug_grpo_${DATASET}_${MODEL_NAME}_q_only_n${NUM_PARAPHRASES}_${TIMESTAMP}"
    local LOG_FILE="${OUTPUT_BASE}/logs/aug_grpo_${MODEL_NAME}_${DATASET}_${TIMESTAMP}.log"

    mkdir -p "${OUTPUT_BASE}/logs"

    local CMD="python scripts/train_grpo_augmented.py \
        --model_id ${MODEL_ID} \
        --output_dir ${OUTPUT_DIR} \
        --dataset ${DATASET} \
        --question_type ${GRPO_QUESTION_TYPE} \
        --augmented_dir ${AUGMENTED_DIR} \
        --num_paraphrases ${NUM_PARAPHRASES} \
        --epochs ${GRPO_EPOCHS} \
        --batch_size ${GRPO_BATCH} \
        --grad_accum ${GRPO_GRAD_ACCUM} \
        --learning_rate ${GRPO_LR} \
        --num_generations ${GRPO_NUM_GEN} \
        --temperature ${GRPO_TEMP} \
        --beta ${GRPO_BETA} \
        --loss_type ${GRPO_LOSS} \
        --max_completion_length ${GRPO_MAX_COMP} \
        --accuracy_weight ${GRPO_ACC_WEIGHT} \
        --format_weight ${GRPO_FMT_WEIGHT} \
        --lora_r ${SFT_LORA_R} \
        --lora_alpha ${SFT_LORA_ALPHA} \
        --gpu ${GPU}"

    [ "$DATASET" = "slake" ] && CMD="$CMD --slake_path ${SLAKE_PATH}"

    if [ "$DRY_RUN" = true ]; then
        echo "=== AUG-GRPO: ${MODEL_NAME} / ${DATASET} / GPU ${GPU} ==="
        echo "$CMD"
    else
        log "AUG-GRPO: ${MODEL_NAME} / ${DATASET} on GPU ${GPU}"
        nohup $CMD > "$LOG_FILE" 2>&1 &
        echo "$! aug_grpo_${MODEL_NAME} ${DATASET} ${GPU}" >> "${OUTPUT_BASE}/logs/aug_pids_${TIMESTAMP}.txt"
        log "  PID: $!"
    fi
}

# =============================================================================
# Pre-flight
# =============================================================================

echo "============================================================"
echo "Augmented Training — Multi-Model Launcher"
echo "============================================================"
echo "Timestamp:        $TIMESTAMP"
echo "Paraphrase Dir:   $AUGMENTED_DIR"
echo "Num Paraphrases:  $NUM_PARAPHRASES"
echo "SLAKE Path:       $SLAKE_PATH"
echo "Output Base:      $OUTPUT_BASE"
echo ""

# Check augmentation cache exists
for DS in rad_vqa slake; do
    CACHE="${AUGMENTED_DIR}/${DS}_paraphrases_n${NUM_PARAPHRASES}.jsonl"
    if [ ! -f "$CACHE" ]; then
        echo "ERROR: Paraphrase cache not found: $CACHE"
        echo "Run: python scripts/generate_paraphrases.py --dataset $DS -n $NUM_PARAPHRASES"
        exit 1
    fi
    LINES=$(wc -l < "$CACHE")
    echo "  $DS cache: $LINES entries"
done

echo ""

# =============================================================================
# Launch
# =============================================================================

if [ "$GRPO_ONLY" = false ]; then
    echo "--- Launching Augmented SFT (6 jobs) ---"
    for MODEL_KEY in qwen internvl llava; do
        for DS in rad_vqa slake; do
            GPU=${GPU_MAP["${MODEL_KEY}_${DS}"]}
            run_augmented_sft "$MODEL_KEY" "$DS" "$GPU"
        done
    done

    if [ "$SFT_ONLY" = false ] && [ "$DRY_RUN" = false ]; then
        echo ""
        echo "Waiting for SFT jobs to finish before launching GRPO..."
        wait
        echo "SFT jobs complete."
    fi
fi

if [ "$SFT_ONLY" = false ]; then
    echo ""
    echo "--- Launching Augmented GRPO (6 jobs) ---"
    for MODEL_KEY in qwen internvl llava; do
        for DS in rad_vqa slake; do
            GPU=${GPU_MAP["${MODEL_KEY}_${DS}"]}
            run_augmented_grpo "$MODEL_KEY" "$DS" "$GPU"
        done
    done
fi

echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry run complete."
else
    echo "Jobs launched. Monitor with:"
    echo "  tail -f ${OUTPUT_BASE}/logs/aug_*_${TIMESTAMP}.log"
    echo "  cat ${OUTPUT_BASE}/logs/aug_pids_${TIMESTAMP}.txt"
fi
echo "============================================================"

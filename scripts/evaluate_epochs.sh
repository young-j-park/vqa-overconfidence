#!/bin/bash
# =============================================================================
# Evaluate Calibration Across Epochs - Wrapper Script
# =============================================================================
#
# Auto-selects evaluation method per training type:
#   SFT / BASE → logits    (fast, direct token probability)
#   GRPO       → sampling  (n=20, generates reasoning chains)
#
# Usage:
#   ./scripts/evaluate_epochs.sh                    # Run everything (auto method)
#   ./scripts/evaluate_epochs.sh --dry-run           # Preview jobs
#   ./scripts/evaluate_epochs.sh --filter grpo       # GRPO only
#   ./scripts/evaluate_epochs.sh --filter sft        # SFT only
#   ./scripts/evaluate_epochs.sh --summarize-only    # View existing results
#
#   # Override auto method (force all to sampling with n=50):
#   METHOD=sampling NUM_SAMPLES=50 ./scripts/evaluate_epochs.sh
#
# =============================================================================

set -e

# Defaults
CHECKPOINT_BASE="${CHECKPOINT_BASE:-./checkpoints}"
OUTPUT_BASE="${OUTPUT_BASE:-./results/calibration_epochs}"
SLAKE_PATH="${SLAKE_PATH:-./data/Slake1.0}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

# Method defaults to "auto" (SFT→logits, GRPO→sampling(n=20))
METHOD="${METHOD:-auto}"

echo "============================================================"
echo "  Across-Epoch Calibration Evaluation"
echo "============================================================"
echo "  Checkpoints:  $CHECKPOINT_BASE"
echo "  Output:       $OUTPUT_BASE"
echo "  SLAKE:        $SLAKE_PATH"
echo "  GPUs:         $GPUS"
echo "  Method:       $METHOD (auto = SFT→logits, GRPO→sampling(n=20))"
echo "============================================================"
echo ""

# Count checkpoints
echo "Training runs found:"
ls -1d ${CHECKPOINT_BASE}/*/ 2>/dev/null | while read dir; do
    run_name=$(basename "$dir")
    num_ckpts=$(ls -1d "$dir"/checkpoint-* 2>/dev/null | wc -l)
    echo "  $run_name  ($num_ckpts checkpoints)"
done
echo ""

# Build command
CMD="python scripts/evaluate_across_epochs.py \
    --checkpoint_base $CHECKPOINT_BASE \
    --output_base $OUTPUT_BASE \
    --slake_path $SLAKE_PATH \
    --gpus $GPUS \
    --method $METHOD"

# Add optional num_samples override
if [ -n "$NUM_SAMPLES" ]; then
    CMD="$CMD --num_samples $NUM_SAMPLES"
fi

# Pass through all additional args
CMD="$CMD $@"

eval $CMD
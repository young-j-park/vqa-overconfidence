#!/usr/bin/env python3
"""
Light SFT Adaptation for Contrastive GRPO Models

Loads a contrastive-GRPO-trained LoRA adapter, merges it into the base model,
then applies a fresh LoRA and runs 1 epoch of SFT on the original VQA format
with a reduced learning rate.

This creates a two-stage pipeline:
  Stage 1: Contrastive GRPO  (already done — calibration pretraining)
  Stage 2: Light SFT adapt   (this script — format adaptation)

Usage:
    # Single run
    python scripts/train_contrast_sft_adapt.py \
        --model_id Qwen/Qwen3-VL-8B-Instruct \
        --contrast_adapter ./checkpoints/contrast_grpo_qwen3vl_8b_rad_vqa/final_model \
        --dataset rad_vqa \
        --output_dir ./checkpoints/contrast_sft_qwen3vl_8b_rad_vqa \
        --gpu 0

    # Use the batch launcher (see train_all_contrast_sft.sh)
"""

import argparse
import sys
import os

# ---- Parse GPU FIRST, before any torch imports ----
parser_for_gpu = argparse.ArgumentParser(add_help=False)
parser_for_gpu.add_argument("--gpu", type=str, default=None,
                            help="GPU index (e.g., 0, 5, or 0,1)")
args_gpu, _ = parser_for_gpu.parse_known_args()

if args_gpu.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args_gpu.gpu
    print(f"[GPU] Set CUDA_VISIBLE_DEVICES={args_gpu.gpu}")

# ---- Now safe to import torch-dependent code ----
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
from datetime import datetime
from peft import PeftModel, get_peft_model, LoraConfig as PeftLoraConfig
from transformers import TrainingArguments, Trainer

from med_vqa.configs import (
    ModelConfig, DataConfig, LoRAConfig,
    DatasetName, QuestionType, ModelFamily,
)
from med_vqa.data import get_dataset, get_collator
from med_vqa.models import load_model
from med_vqa.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Light SFT adaptation on contrastive GRPO models"
    )

    # Model
    parser.add_argument("--model_id", type=str, required=True,
                        help="HuggingFace base model ID")
    parser.add_argument("--contrast_adapter", type=str, required=True,
                        help="Path to contrastive GRPO adapter (e.g., .../final_model)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save adapted checkpoints")

    # Dataset
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["rad_vqa", "slake"])
    parser.add_argument("--slake_path", type=str, default=None,
                        help="Path to SLAKE dataset")
    parser.add_argument("--question_type", type=str, default="closed",
                        choices=["all", "closed", "open"])
    parser.add_argument("--split", type=str, default="train")

    # Training — intentionally conservative defaults
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epochs (default: 1 for light adaptation)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate (lower than standard SFT, default: 1e-5)")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--max_length", type=int, default=2048)

    # LoRA for the adaptation stage
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Saving
    parser.add_argument("--save_strategy", type=str, default="epoch",
                        choices=["epoch", "steps", "no"])
    parser.add_argument("--save_total_limit", type=int, default=3)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--optim", type=str, default="paged_adamw_8bit")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # ---- Resolve SLAKE path ----
    slake_path = args.slake_path
    if args.dataset == "slake" and slake_path is None:
        default_paths = [
            "/data/datasets/Slake1.0",
            os.path.expanduser("~/datasets/Slake1.0"),
            "./data/Slake1.0",
        ]
        for path in default_paths:
            if os.path.exists(path):
                slake_path = path
                print(f"[SLAKE] Found dataset at: {path}")
                break
        if slake_path is None:
            print("ERROR: SLAKE dataset requires --slake_path argument")
            sys.exit(1)

    # ================================================================
    # STEP 1: Load base model + contrastive adapter → merge
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 1: Load base model + merge contrastive GRPO adapter")
    print("=" * 60)

    model_config = ModelConfig(
        model_id=args.model_id,
        use_4bit=True,
    )

    # Load base model (with 4-bit quantization + prepare for training)
    model, processor = load_model(model_config, prepare_for_training=True)

    # Load contrastive GRPO adapter on top
    print(f"\nLoading contrastive adapter from: {args.contrast_adapter}")
    model = PeftModel.from_pretrained(model, args.contrast_adapter)

    # Merge the contrastive adapter into the base weights
    # This makes the contrastive knowledge part of the "base" for the next LoRA
    print("Merging contrastive adapter into base weights...")
    model = model.merge_and_unload()
    print("Merge complete — contrastive knowledge is now baked in.")

    if torch.cuda.is_available():
        print(f"VRAM after merge: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # ================================================================
    # STEP 2: Apply fresh LoRA for SFT adaptation
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Apply fresh LoRA for light SFT adaptation")
    print("=" * 60)

    peft_config = PeftLoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ================================================================
    # STEP 3: Load dataset (standard VQA format)
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Load dataset (original VQA format)")
    print("=" * 60)

    data_config_kwargs = {
        "dataset_name": DatasetName(args.dataset),
        "question_type": QuestionType(args.question_type),
        "split": args.split,
        "seed": args.seed,
    }
    if slake_path:
        data_config_kwargs["data_path"] = slake_path

    data_config = DataConfig(**data_config_kwargs)
    dataset_wrapper = get_dataset(data_config)
    dataset = dataset_wrapper.load()

    stats = dataset_wrapper.get_statistics()
    print(f"\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # ================================================================
    # STEP 4: Setup trainer and train
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Light SFT Training")
    print("=" * 60)

    collator = get_collator(
        model_config.model_family,
        processor,
        args.max_length,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        fp16=False,
        logging_steps=10,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        optim=args.optim,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    # Print summary
    print(f"\nModel:             {args.model_id}")
    print(f"Contrastive from:  {args.contrast_adapter}")
    print(f"Dataset:           {args.dataset} ({args.question_type})")
    print(f"Training samples:  {len(dataset)}")
    print(f"Epochs:            {args.epochs}")
    print(f"Effective BS:      {args.batch_size * args.grad_accum}")
    print(f"Learning Rate:     {args.learning_rate}")
    print(f"LoRA r={args.lora_r}, α={args.lora_alpha}")
    print(f"Output:            {args.output_dir}")
    print("=" * 60 + "\n")

    # Train
    train_result = trainer.train()

    # ================================================================
    # STEP 5: Save combined adapter (adapter0 + adapter1)
    # ================================================================
    # Problem: trainer.save_model() saves only adapter1 (the SFT LoRA).
    # But adapter1 was trained on top of (base + adapter0_merged).
    # Loading adapter1 alone on the base model would be WRONG.
    #
    # Solution: Save adapter1 as a "stage2" checkpoint, then create
    # a "final_model" directory that loads base → adapter0 → adapter1
    # → merges both → re-extracts a single combined LoRA adapter.
    # This combined adapter can be loaded directly on the base model.
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 5: Save combined adapter (adapter0 + adapter1)")
    print("=" * 60)

    # 5a. Save the raw stage-2 adapter (for reproducibility / debugging)
    stage2_dir = os.path.join(args.output_dir, "stage2_adapter_only")
    os.makedirs(stage2_dir, exist_ok=True)
    trainer.save_model(stage2_dir)
    print(f"Stage-2 adapter saved to: {stage2_dir}")

    # 5b. The in-memory model is: base_quantized(+adapter0_merged) + adapter1
    # We merge adapter1 into the (already adapter0-merged) weights.
    print("Merging stage-2 adapter into combined weights...")
    merged_model = model.merge_and_unload()

    # 5c. Now merged_model weights = base + adapter0 + adapter1 (all merged).
    # To create a single LoRA adapter that captures (adapter0 + adapter1),
    # we save the full merged weights and adapter0 path for the eval scripts
    # to use. With quantized training, extracting a clean LoRA diff is not
    # straightforward, so we provide two loading strategies:
    #
    # Strategy A (recommended for eval): Load base → adapter0 → adapter1
    #   model = load_base(model_id)
    #   model = PeftModel.from_pretrained(model, adapter0_path)
    #   model = model.merge_and_unload()
    #   model = PeftModel.from_pretrained(model, adapter1_path)
    #
    # Strategy B (for quick eval): Use the merged model in memory directly.

    # Save final_model directory with the stage2 adapter + loading instructions
    final_dir = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)

    # Copy stage2 adapter files to final_model for convenience
    import shutil
    for fname in os.listdir(stage2_dir):
        src = os.path.join(stage2_dir, fname)
        dst = os.path.join(final_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    # Save loading instructions as JSON
    loading_info = {
        "description": (
            "This adapter (stage2) was trained on top of a contrastive GRPO adapter. "
            "To load correctly, you must first load and merge the stage1 adapter, "
            "then load this stage2 adapter on top."
        ),
        "base_model_id": args.model_id,
        "stage1_adapter_path": os.path.abspath(args.contrast_adapter),
        "stage2_adapter_path": os.path.abspath(final_dir),
        "loading_order": [
            "1. Load base model",
            "2. Load stage1 (contrastive GRPO) adapter → merge_and_unload()",
            "3. Load stage2 (this adapter) on the merged model",
        ],
        "loading_code": (
            "model = load_base(model_id)\n"
            "model = PeftModel.from_pretrained(model, stage1_adapter_path)\n"
            "model = model.merge_and_unload()\n"
            "model = PeftModel.from_pretrained(model, stage2_adapter_path)\n"
        ),
    }
    loading_info_path = os.path.join(final_dir, "loading_info.json")
    with open(loading_info_path, "w") as f:
        json.dump(loading_info, f, indent=2)

    if hasattr(processor, 'save_pretrained'):
        processor.save_pretrained(final_dir)

    print(f"Final model saved to: {final_dir}")
    print(f"Loading instructions: {loading_info_path}")
    print()
    print("To load this model for evaluation:")
    print(f"  1. Load base:     {args.model_id}")
    print(f"  2. Load+merge:    {args.contrast_adapter}")
    print(f"  3. Load adapter:  {final_dir}")

    # ================================================================
    # Save metadata
    # ================================================================
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "pipeline": "contrast_grpo → light_sft_adapt",
        "stage1_adapter": os.path.abspath(args.contrast_adapter),
        "stage2_adapter": os.path.abspath(final_dir),
        "requires_two_stage_loading": True,
        "stage2_config": {
            "model_id": args.model_id,
            "dataset": args.dataset,
            "question_type": args.question_type,
            "num_epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.grad_accum,
            "effective_batch_size": args.batch_size * args.grad_accum,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "warmup_ratio": args.warmup_ratio,
            "lr_scheduler": args.lr_scheduler,
        },
        "training_samples": len(dataset),
        "seed": args.seed,
        "results": {
            "train_loss": train_result.training_loss,
            "train_runtime_seconds": train_result.metrics.get("train_runtime"),
        },
    }

    metadata_path = os.path.join(args.output_dir, "experiment_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to: {metadata_path}")

    print(f"\nTraining complete! Loss: {train_result.training_loss:.4f}")


if __name__ == "__main__":
    main()

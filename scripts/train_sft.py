#!/usr/bin/env python3
"""
SFT Training Script for Medical VQA

Supports RAD-VQA and SLAKE datasets with configurable hyperparameters.

Usage Examples:
    # Train on RAD-VQA
    python train_sft.py \
        --model_id Qwen/Qwen2-VL-2B-Instruct \
        --output_dir ./checkpoints/qwen2-2b-radvqa-closed \
        --dataset rad_vqa \
        --question_type closed \
        --epochs 5

    # Train on SLAKE
    python train_sft.py \
        --model_id Qwen/Qwen2-VL-2B-Instruct \
        --output_dir ./checkpoints/qwen2-2b-slake-closed \
        --dataset slake \
        --slake_path /path/to/Slake1.0 \
        --question_type closed \
        --epochs 5

    # Full configuration example
    python train_sft.py \
        --model_id Qwen/Qwen2-VL-2B-Instruct \
        --output_dir ./checkpoints/experiment \
        --dataset slake \
        --slake_path /path/to/Slake1.0 \
        --question_type closed \
        --epochs 5 \
        --batch_size 4 \
        --grad_accum 8 \
        --learning_rate 5e-5 \
        --lora_r 64 \
        --lora_alpha 128 \
        --save_strategy epoch \
        --save_total_limit 5 \
        --gpu 0
"""

import argparse
import sys
import os

# Parse GPU argument FIRST before any torch imports
parser_for_gpu = argparse.ArgumentParser(add_help=False)
parser_for_gpu.add_argument("--gpu", type=str, default=None, help="GPU index (e.g., 0, 5, or 0,1)")
args_gpu, _ = parser_for_gpu.parse_known_args()

if args_gpu.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args_gpu.gpu
    print(f"[GPU] Set CUDA_VISIBLE_DEVICES={args_gpu.gpu}")

# Now import torch-dependent modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from med_vqa.configs import (
    ModelConfig, DataConfig, SFTConfig, LoRAConfig,
    ExperimentConfig, DatasetName, QuestionType,
)
from med_vqa.training import run_sft_training
from med_vqa.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train VLM on Medical VQA with SFT")
    
    # Model arguments
    parser.add_argument("--model_id", type=str, required=True,
                       help="HuggingFace model ID (e.g., Qwen/Qwen2-VL-2B-Instruct)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save checkpoints")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="rad_vqa",
                       choices=["rad_vqa", "slake"],
                       help="Dataset to use for training")
    parser.add_argument("--slake_path", type=str, default=None,
                       help="Path to SLAKE dataset (required if dataset=slake)")
    parser.add_argument("--question_type", type=str, default="all",
                       choices=["all", "closed", "open"],
                       help="Filter by question type")
    parser.add_argument("--subsample_size", type=int, default=None,
                       help="Subsample dataset to this size (None for full dataset)")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split to use")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=8,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate (recommended: 5e-5 to 1e-4 for medical)")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                       help="Warmup ratio")
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
                       help="Learning rate scheduler type")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=64,
                       help="LoRA rank (recommended: 32-64 for domain adaptation)")
    parser.add_argument("--lora_alpha", type=int, default=128,
                       help="LoRA alpha (recommended: 2×r)")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                       help="LoRA dropout")
    
    # Saving arguments
    parser.add_argument("--save_strategy", type=str, default="epoch",
                       choices=["epoch", "steps", "no"],
                       help="When to save checkpoints")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save checkpoint every N steps (if save_strategy=steps)")
    parser.add_argument("--save_total_limit", type=int, default=5,
                       help="Maximum number of checkpoints to keep")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_4bit", action="store_true", default=True,
                       help="Enable 4-bit quantization (default: True)")
    parser.add_argument("--no_4bit", action="store_true",
                       help="Disable 4-bit quantization")
    parser.add_argument("--gpu", type=str, default=None, 
                       help="GPU index (e.g., 0, 5, or 0,1 for multi-GPU)")
    parser.add_argument("--optim", type=str, default="paged_adamw_8bit",
                       help="Optimizer to use")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Validate SLAKE path
    if args.dataset == "slake" and args.slake_path is None:
        # Try to find in default locations
        default_paths = [
            "/data/datasets/Slake1.0",
            os.path.expanduser("~/datasets/Slake1.0"),
            "./data/Slake1.0",
            os.path.expanduser("~/Downloads/Slake1.0"),
        ]
        for path in default_paths:
            if os.path.exists(path):
                args.slake_path = path
                print(f"[SLAKE] Found dataset at: {path}")
                break
        
        if args.slake_path is None:
            print("ERROR: SLAKE dataset requires --slake_path argument")
            print("Checked default locations:")
            for p in default_paths:
                print(f"  - {p}")
            sys.exit(1)
    
    # Build configuration
    model_config = ModelConfig(
        model_id=args.model_id,
        use_4bit=not args.no_4bit,
    )
    
    # Data config with optional SLAKE path
    data_config_kwargs = {
        "dataset_name": DatasetName(args.dataset),
        "question_type": QuestionType(args.question_type),
        "split": args.split,
        "subsample_size": args.subsample_size,
        "seed": args.seed,
    }
    if args.slake_path:
        data_config_kwargs["data_path"] = args.slake_path
    
    data_config = DataConfig(**data_config_kwargs)
    
    lora_config = LoRAConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    
    train_config = SFTConfig(
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        optim=args.optim,
        lora=lora_config,
    )
    
    experiment_config = ExperimentConfig(
        model=model_config,
        data=data_config,
        training=train_config,
        experiment_name=f"sft_{os.path.basename(args.output_dir)}",
        seed=args.seed,
    )
    
    # Print configuration summary
    print("\n" + "=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"Model:           {args.model_id}")
    print(f"Dataset:         {args.dataset}")
    if args.slake_path:
        print(f"SLAKE Path:      {args.slake_path}")
    print(f"Question Type:   {args.question_type}")
    print(f"Subsample:       {args.subsample_size or 'Full dataset'}")
    print(f"Output Dir:      {args.output_dir}")
    print()
    print("Hyperparameters:")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Batch Size:    {args.batch_size}")
    print(f"  Grad Accum:    {args.grad_accum}")
    print(f"  Effective BS:  {args.batch_size * args.grad_accum}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Warmup Ratio:  {args.warmup_ratio}")
    print(f"  LR Scheduler:  {args.lr_scheduler}")
    print()
    print("LoRA Configuration:")
    print(f"  Rank (r):      {args.lora_r}")
    print(f"  Alpha (α):     {args.lora_alpha}")
    print(f"  Dropout:       {args.lora_dropout}")
    print()
    print("Saving:")
    print(f"  Strategy:      {args.save_strategy}")
    print(f"  Keep Last:     {args.save_total_limit}")
    print("=" * 60 + "\n")
    
    # Run training
    results = run_sft_training(experiment_config)
    
    print(f"\nTraining complete! Results: {results}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
SFT Training Script for Medical VQA

Usage Examples:
    # Train Qwen-VL-2B on RAD-VQA closed questions
    python train_sft.py \
        --model_id Qwen/Qwen3-VL-2B-Instruct \
        --output_dir ./checkpoints/qwen3-2b-closed \
        --dataset rad_vqa \
        --question_type closed \
        --epochs 10

    # Train on specific GPU
    python train_sft.py \
        --model_id Qwen/Qwen3-VL-2B-Instruct \
        --output_dir ./checkpoints/qwen3-2b-closed \
        --gpu 5

    # Train with subsampling
    python train_sft.py \
        --model_id Qwen/Qwen3-VL-4B-Instruct \
        --output_dir ./checkpoints/qwen3-4b-subset100 \
        --subsample_size 100 \
        --seed 42
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
                       help="HuggingFace model ID (e.g., Qwen/Qwen3-VL-2B-Instruct)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save checkpoints")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="rad_vqa",
                       choices=["rad_vqa", "slake"],
                       help="Dataset to use for training")
    parser.add_argument("--question_type", type=str, default="all",
                       choices=["all", "closed", "open"],
                       help="Filter by question type")
    parser.add_argument("--subsample_size", type=int, default=None,
                       help="Subsample dataset to this size (None for full dataset)")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split to use")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=2048)
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_4bit", action="store_true", default=True)
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--gpu", type=str, default=None, 
                       help="GPU index (e.g., 0, 5, or 0,1 for multi-GPU)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Build configuration
    model_config = ModelConfig(
        model_id=args.model_id,
        use_4bit=not args.no_4bit,
    )
    
    data_config = DataConfig(
        dataset_name=DatasetName(args.dataset),
        question_type=QuestionType(args.question_type),
        split=args.split,
        subsample_size=args.subsample_size,
        seed=args.seed,
    )
    
    lora_config = LoRAConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    
    train_config = SFTConfig(
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        lora=lora_config,
    )
    
    experiment_config = ExperimentConfig(
        model=model_config,
        data=data_config,
        training=train_config,
        experiment_name=f"sft_{os.path.basename(args.output_dir)}",
        seed=args.seed,
    )
    
    # Run training
    results = run_sft_training(experiment_config)
    
    print(f"\nTraining complete! Results: {results}")


if __name__ == "__main__":
    main()

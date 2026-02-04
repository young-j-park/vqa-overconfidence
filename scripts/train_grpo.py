#!/usr/bin/env python3
"""
GRPO Training Script for Medical VQA (Closed Questions)

Supports: Qwen3-VL-8B, InternVL3-8B, LLaVA-NeXT-7B
Datasets: RAD-VQA, SLAKE

Uses TRL GRPOTrainer with verifiable rewards on closed (yes/no) questions.
The model is prompted with <think>/<answer> tags for structured reasoning.

Usage Examples:
    # Qwen3-VL on RAD-VQA
    python scripts/train_grpo.py \
        --model_id Qwen/Qwen3-VL-8B-Instruct \
        --dataset rad_vqa \
        --output_dir ./checkpoints/grpo_qwen3vl_8b_rad_vqa \
        --gpu 0

    # InternVL3 on SLAKE
    python scripts/train_grpo.py \
        --model_id OpenGVLab/InternVL3-8B-hf \
        --dataset slake \
        --slake_path ./data/Slake1.0 \
        --output_dir ./checkpoints/grpo_internvl3_8b_slake \
        --gpu 1

    # Full configuration
    python scripts/train_grpo.py \
        --model_id Qwen/Qwen3-VL-8B-Instruct \
        --dataset rad_vqa \
        --output_dir ./checkpoints/grpo_experiment \
        --epochs 3 \
        --batch_size 1 \
        --grad_accum 8 \
        --num_generations 8 \
        --learning_rate 5e-6 \
        --lora_r 64 \
        --lora_alpha 128 \
        --save_strategy epoch \
        --gpu 0
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

from med_vqa.configs import (
    ModelConfig, DataConfig, GRPOConfig, LoRAConfig,
    ExperimentConfig, DatasetName, QuestionType,
)
from med_vqa.training import run_grpo_training
from med_vqa.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="GRPO Training for Medical VQA (Closed Questions)"
    )

    # ---- Model ----
    parser.add_argument("--model_id", type=str, required=True,
                        help="HuggingFace model ID")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for checkpoints and logs")

    # ---- Dataset ----
    parser.add_argument("--dataset", type=str, default="rad_vqa",
                        choices=["rad_vqa", "slake"])
    parser.add_argument("--slake_path", type=str, default=None,
                        help="Path to SLAKE dataset (required if dataset=slake)")
    parser.add_argument("--question_type", type=str, default="closed",
                        choices=["closed", "all"],
                        help="Question type filter (default: closed for RLVR)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max training samples (for smoke testing)")

    # ---- GRPO hyperparameters ----
    parser.add_argument("--num_generations", type=int, default=8,
                        help="Generations per prompt (G in GRPO)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature for generation")
    parser.add_argument("--beta", type=float, default=0.0,
                        help="KL penalty (0.0 = no ref model, saves memory)")
    parser.add_argument("--max_completion_length", type=int, default=128,
                        help="Max tokens for generated completion")
    parser.add_argument("--max_prompt_length", type=int, default=None,
                        help="Max prompt length (None = no truncation)")
    parser.add_argument("--loss_type", type=str, default="dapo",
                        choices=["grpo", "dapo", "dr_grpo"],
                        help="GRPO loss formulation")

    # ---- Training ----
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Per-device train batch size")
    parser.add_argument("--grad_accum", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                        help="Learning rate (lower than SFT)")

    # ---- LoRA ----
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # ---- Reward weights ----
    parser.add_argument("--accuracy_weight", type=float, default=3.0)
    parser.add_argument("--format_weight", type=float, default=1.0)

    # ---- Saving ----
    parser.add_argument("--save_strategy", type=str, default="epoch",
                        choices=["epoch", "steps"])
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save every N steps (if save_strategy=steps)")
    parser.add_argument("--save_total_limit", type=int, default=5)

    # ---- Logging ----
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--report_to", type=str, default="none",
                        choices=["tensorboard", "wandb", "none"])

    # ---- Misc ----
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU index (e.g., 0, 5, or 0,1)")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # ---- Resolve SLAKE path ----
    slake_path = args.slake_path
    if args.dataset == "slake" and slake_path is None:
        default = "./data/Slake1.0"
        if os.path.exists(default):
            slake_path = default
            print(f"[SLAKE] Found dataset at: {slake_path}")
        else:
            print("ERROR: SLAKE dataset requires --slake_path argument")
            sys.exit(1)

    # ---- Build configs ----
    model_config = ModelConfig(
        model_id=args.model_id,
        use_4bit=False,  # GRPO needs full precision for generation
    )

    data_config_kwargs = {
        "dataset_name": DatasetName(args.dataset),
        "question_type": QuestionType(args.question_type),
        "subsample_size": args.max_samples,
        "seed": args.seed,
    }
    if slake_path:
        data_config_kwargs["data_path"] = slake_path
    data_config = DataConfig(**data_config_kwargs)

    lora_config = LoRAConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_generations=args.num_generations,
        temperature=args.temperature,
        beta=args.beta,
        loss_type=args.loss_type,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        accuracy_weight=args.accuracy_weight,
        format_weight=args.format_weight,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        report_to=args.report_to,
        lora=lora_config,
    )

    experiment_config = ExperimentConfig(
        model=model_config,
        data=data_config,
        grpo=grpo_config,
        experiment_name=f"grpo_{os.path.basename(args.output_dir)}",
        seed=args.seed,
    )

    # ---- Print summary ----
    num_gpus = len(args.gpu.split(",")) if args.gpu else 1
    print("\n" + "=" * 60)
    print("GRPO Training Configuration")
    print("=" * 60)
    print(f"Model:           {args.model_id}")
    print(f"Dataset:         {args.dataset}")
    print(f"Question Type:   {args.question_type}")
    if slake_path:
        print(f"SLAKE Path:      {slake_path}")
    print(f"Output Dir:      {args.output_dir}")
    print()
    print("GRPO Hyperparameters:")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Batch Size:      {args.batch_size}")
    print(f"  Grad Accum:      {args.grad_accum}")
    print(f"  Effective BS:    {args.batch_size * args.grad_accum * num_gpus}")
    print(f"  Learning Rate:   {args.learning_rate}")
    print(f"  Num Generations: {args.num_generations}")
    print(f"  Temperature:     {args.temperature}")
    print(f"  Beta (KL):       {args.beta}")
    print(f"  Loss Type:       {args.loss_type}")
    print(f"  Max Completion:  {args.max_completion_length}")
    print()
    print("LoRA Configuration:")
    print(f"  Rank (r):        {args.lora_r}")
    print(f"  Alpha (α):       {args.lora_alpha}")
    print(f"  Dropout:         {args.lora_dropout}")
    print()
    print("Reward Weights:")
    print(f"  Accuracy:        {args.accuracy_weight}")
    print(f"  Format:          {args.format_weight}")
    print()
    print("Saving:")
    print(f"  Strategy:        {args.save_strategy}")
    print(f"  Keep Last:       {args.save_total_limit}")
    print("=" * 60 + "\n")

    # ---- Train ----
    results = run_grpo_training(experiment_config)
    print(f"\nTraining complete! Results: {results}")


if __name__ == "__main__":
    main()
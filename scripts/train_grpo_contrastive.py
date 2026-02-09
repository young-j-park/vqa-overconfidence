#!/usr/bin/env python3
"""
Contrastive (SimCLR-style) GRPO Training for Medical VQA

Reformulates VQA as a multiple-choice task: given an image, choose the
correct (Q, A) pair from a set of candidates. Negatives come from different
images.

Compatible with GRPO: the reward is based on whether the model picks the
correct letter (binary, ranking-penalty, or MRR-inspired).

Reward options:
  - binary:  1.0 correct, 0.0 wrong   (simplest)
  - ranking: 1.0 correct, -0.1 wrong   (penalizes wrong picks)
  - mrr:     1/(1+distance) partial credit based on choice proximity

Usage:
    python scripts/train_grpo_contrastive.py \
        --model_id Qwen/Qwen3-VL-8B-Instruct \
        --dataset rad_vqa \
        --output_dir ./checkpoints/contrast_grpo_qwen_rad_vqa \
        --num_choices 4 \
        --reward_type mrr \
        --gpu 0

    # With 8 choices and hard negatives
    python scripts/train_grpo_contrastive.py \
        --model_id OpenGVLab/InternVL3-8B-hf \
        --dataset slake \
        --slake_path ./data/Slake1.0 \
        --output_dir ./checkpoints/contrast_grpo_internvl_slake \
        --num_choices 8 \
        --hard_negatives \
        --reward_type mrr \
        --gpu 1
"""

import argparse
import sys
import os

# Parse GPU FIRST
parser_for_gpu = argparse.ArgumentParser(add_help=False)
parser_for_gpu.add_argument("--gpu", type=str, default=None)
args_gpu, _ = parser_for_gpu.parse_known_args()

if args_gpu.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args_gpu.gpu
    print(f"[GPU] Set CUDA_VISIBLE_DEVICES={args_gpu.gpu}")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datetime import datetime

import torch
from trl import GRPOConfig as TRLGRPOConfig, GRPOTrainer
from peft import LoraConfig

from med_vqa.configs import (
    ModelConfig, DataConfig, DatasetName, QuestionType, ModelFamily,
)
from med_vqa.data import get_dataset
from med_vqa.data.contrastive import (
    build_contrastive_dataset,
    format_contrastive_for_grpo,
    get_contrastive_reward,
    contrastive_accuracy_reward,
)
from med_vqa.training.grpo_trainer import (
    get_lora_target_modules,
    format_reward,
)
from med_vqa.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Contrastive GRPO Training for Medical VQA"
    )

    # Model
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # Dataset
    parser.add_argument("--dataset", type=str, default="rad_vqa",
                        choices=["rad_vqa", "slake"])
    parser.add_argument("--slake_path", type=str, default=None)
    parser.add_argument("--question_type", type=str, default="all",
                        choices=["all", "closed", "open"])

    # Contrastive settings
    parser.add_argument("--num_choices", type=int, default=4,
                        help="Total MCQ choices (1 correct + N-1 negatives)")
    parser.add_argument("--hard_negatives", action="store_true", default=True,
                        help="Sample negatives from same answer_type")
    parser.add_argument("--no_hard_negatives", action="store_false",
                        dest="hard_negatives")
    parser.add_argument("--reward_type", type=str, default="mrr",
                        choices=["binary", "ranking", "mrr"],
                        help="Reward function type")

    # GRPO hyperparameters (SAME defaults as baseline)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--max_completion_length", type=int, default=256,
                        help="Longer than baseline because MCQ reasoning is verbose")
    parser.add_argument("--max_prompt_length", type=int, default=None)
    parser.add_argument("--loss_type", type=str, default="dapo")

    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-6)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Reward weights
    parser.add_argument("--contrastive_weight", type=float, default=3.0,
                        help="Weight for contrastive accuracy reward")
    parser.add_argument("--format_weight", type=float, default=1.0)

    # Saving
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--save_total_limit", type=int, default=5)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--report_to", type=str, default="none")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # ---- Resolve paths ----
    slake_path = args.slake_path
    if args.dataset == "slake" and slake_path is None:
        default = "./data/Slake1.0"
        if os.path.exists(default):
            slake_path = default
        else:
            raise ValueError("--slake_path required for SLAKE")

    model_config = ModelConfig(model_id=args.model_id)
    ds_name = DatasetName.RAD_VQA if args.dataset == "rad_vqa" else DatasetName.SLAKE
    q_type = {
        "all": QuestionType.ALL,
        "closed": QuestionType.CLOSED,
        "open": QuestionType.OPEN,
    }[args.question_type]

    # ---- Load base dataset ----
    print("\n[1/4] Loading base dataset...")
    data_config = DataConfig(
        dataset_name=ds_name,
        question_type=q_type,
        split="train",
        data_path=slake_path,
        seed=args.seed,
    )
    dataset_wrapper = get_dataset(data_config)
    base_dataset = dataset_wrapper.load()
    print(f"  Base samples: {len(base_dataset)}")

    # ---- Build contrastive MCQ dataset ----
    print("\n[2/4] Building contrastive MCQ dataset...")
    contrastive_samples = build_contrastive_dataset(
        base_dataset=base_dataset,
        num_choices=args.num_choices,
        hard_negatives=args.hard_negatives,
        seed=args.seed,
    )

    if args.max_samples and len(contrastive_samples) > args.max_samples:
        contrastive_samples = contrastive_samples[:args.max_samples]

    # Format for GRPO
    dataset = format_contrastive_for_grpo(
        contrastive_samples,
        model_family=model_config.model_family.value,
    )

    # Handle InternVL image resizing
    if model_config.model_family == ModelFamily.INTERNVL:
        print("[InternVL] Resizing images to 448x448")
        from PIL import Image

        def resize_img(example):
            img = example["image"]
            if isinstance(img, Image.Image):
                example["image"] = img.convert("RGB").resize((448, 448))
            return example

        dataset = dataset.map(resize_img)

    print(f"  Contrastive samples: {len(dataset)}")

    # ---- Setup GRPO Trainer ----
    print("\n[3/4] Setting up GRPO trainer...")

    # Get reward function
    contrastive_reward_fn = get_contrastive_reward(args.reward_type)

    target_modules = get_lora_target_modules(model_config.model_family)
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )

    model_init_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if model_config.model_family == ModelFamily.INTERNVL:
        model_init_kwargs["trust_remote_code"] = True
    try:
        import flash_attn
        model_init_kwargs["attn_implementation"] = "flash_attention_2"
    except ImportError:
        pass

    training_args = TRLGRPOConfig(
        output_dir=args.output_dir,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        remove_unused_columns=False,
        temperature=args.temperature,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        beta=args.beta,
        loss_type=args.loss_type,
        gradient_checkpointing=True,
        bf16=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        reward_weights=[args.contrastive_weight, args.format_weight],
        save_strategy=args.save_strategy,
        save_steps=500,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        log_completions=True,
        report_to=[args.report_to] if args.report_to != "none" else [],
        model_init_kwargs=model_init_kwargs,
        seed=args.seed,
        dataloader_num_workers=2,
    )

    trainer = GRPOTrainer(
        model=args.model_id,
        reward_funcs=[contrastive_reward_fn, format_reward],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    # ---- Train ----
    num_gpus = len(args.gpu.split(",")) if args.gpu else 1

    print("\n" + "=" * 60)
    print("Starting Contrastive GRPO Training")
    print("=" * 60)
    print(f"Model:           {args.model_id}")
    print(f"Dataset:         {args.dataset}")
    print(f"Question Type:   {args.question_type}")
    print(f"MCQ Choices:     {args.num_choices}")
    print(f"Hard Negatives:  {args.hard_negatives}")
    print(f"Reward Type:     {args.reward_type}")
    print(f"Samples:         {len(dataset)}")
    print(f"Epochs:          {args.epochs}")
    print(f"Num generations: {args.num_generations}")
    print(f"Effective BS:    {args.batch_size * args.grad_accum * num_gpus}")
    print(f"Learning rate:   {args.learning_rate}")
    print(f"Loss type:       {args.loss_type}")
    print(f"LoRA:            r={args.lora_r}, α={args.lora_alpha}")
    print(f"Max completion:  {args.max_completion_length}")
    print(f"Output:          {args.output_dir}")
    print("=" * 60 + "\n")

    train_result = trainer.train()

    # Save
    final_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_dir)
    print(f"Final model saved to: {final_dir}")

    # ---- Save metadata ----
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "training_type": "grpo_contrastive",
        "model_id": args.model_id,
        "model_family": model_config.model_family.value,
        "dataset": args.dataset,
        "question_type": args.question_type,
        "training_samples": len(dataset),
        "seed": args.seed,
        "contrastive": {
            "num_choices": args.num_choices,
            "hard_negatives": args.hard_negatives,
            "reward_type": args.reward_type,
        },
        "grpo": {
            "num_epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.grad_accum,
            "learning_rate": args.learning_rate,
            "num_generations": args.num_generations,
            "temperature": args.temperature,
            "beta": args.beta,
            "loss_type": args.loss_type,
            "max_completion_length": args.max_completion_length,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "contrastive_weight": args.contrastive_weight,
            "format_weight": args.format_weight,
        },
        "results": {
            "train_loss": train_result.training_loss,
            "train_runtime_seconds": train_result.metrics.get("train_runtime"),
        },
    }

    meta_path = os.path.join(args.output_dir, "experiment_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()

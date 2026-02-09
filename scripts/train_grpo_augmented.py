#!/usr/bin/env python3
"""
Augmented GRPO Training for Medical VQA

Same as train_grpo.py but with question paraphrase augmentation.
Uses identical hyperparameters to the baseline for fair comparison.

Since GRPO uses closed (yes/no) questions only, we only augment questions
(not answers — the answer is always "yes" or "no").

The augmentation happens at dataset construction time: for each sample,
the question in the prompt is randomly replaced with a paraphrased version.
Different seeds produce different augmented versions.

Usage:
    python scripts/train_grpo_augmented.py \
        --model_id Qwen/Qwen3-VL-8B-Instruct \
        --dataset rad_vqa \
        --output_dir ./checkpoints/aug_grpo_qwen3vl_rad_vqa \
        --augmented_dir ./data/augmented \
        --num_paraphrases 8 \
        --gpu 0

    # SLAKE
    python scripts/train_grpo_augmented.py \
        --model_id OpenGVLab/InternVL3-8B-hf \
        --dataset slake \
        --slake_path ./data/Slake1.0 \
        --output_dir ./checkpoints/aug_grpo_internvl3_slake \
        --augmented_dir ./data/augmented \
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
import random
from datetime import datetime
from typing import Dict, List, Optional

import torch
from datasets import Dataset
from PIL import Image
from trl import GRPOConfig as TRLGRPOConfig, GRPOTrainer
from peft import LoraConfig

from med_vqa.configs import (
    ModelConfig, DataConfig, GRPOConfig, LoRAConfig as MedLoRAConfig,
    ExperimentConfig, DatasetName, QuestionType, ModelFamily,
)
from med_vqa.training.grpo_trainer import (
    SYSTEM_PROMPT,
    get_lora_target_modules,
    load_and_format_dataset,
    accuracy_reward,
    format_reward,
)
from med_vqa.data.augmented import (
    load_paraphrase_cache,
    find_paraphrase_cache,
    presample_grpo_augmented,
)
from med_vqa.utils import set_seed


def load_and_format_dataset_augmented(
    dataset_name: str,
    paraphrase_cache: Dict[str, Dict],
    slake_path: Optional[str] = None,
    question_type: str = "closed",
    max_samples: Optional[int] = None,
    model_family: Optional[ModelFamily] = None,
    seed: int = 42,
) -> Dataset:
    """Load, format, and augment dataset for GRPO training.

    Same as load_and_format_dataset but with question paraphrasing.
    """
    # First, load the standard formatted records
    base_dataset = load_and_format_dataset(
        dataset_name=dataset_name,
        slake_path=slake_path,
        question_type=question_type,
        max_samples=max_samples,
        model_family=model_family,
        seed=seed,
    )

    # Convert to list for augmentation
    records = [base_dataset[i] for i in range(len(base_dataset))]

    # Augment: swap questions with paraphrases
    augmented_records = presample_grpo_augmented(
        formatted_records=records,
        paraphrase_cache=paraphrase_cache,
        augment_questions=True,
        seed=seed,
    )

    dataset = Dataset.from_list(augmented_records)
    print(f"[AugGRPO] Augmented {len(dataset)} samples")
    return dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Augmented GRPO Training for Medical VQA"
    )

    # Model
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # Dataset
    parser.add_argument("--dataset", type=str, default="rad_vqa",
                        choices=["rad_vqa", "slake"])
    parser.add_argument("--slake_path", type=str, default=None)
    parser.add_argument("--question_type", type=str, default="closed",
                        choices=["closed", "all"])
    parser.add_argument("--max_samples", type=int, default=None)

    # Augmentation
    parser.add_argument("--augmented_dir", type=str, default="./data/augmented")
    parser.add_argument("--num_paraphrases", type=int, default=8)

    # GRPO hyperparameters (SAME defaults as train_grpo.py / train_all_grpo.sh)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--max_completion_length", type=int, default=128)
    parser.add_argument("--max_prompt_length", type=int, default=None)
    parser.add_argument("--loss_type", type=str, default="dapo",
                        choices=["grpo", "dapo", "dr_grpo"])

    # Training (SAME defaults as baseline)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-6)

    # LoRA (SAME defaults)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Reward weights (SAME defaults)
    parser.add_argument("--accuracy_weight", type=float, default=3.0)
    parser.add_argument("--format_weight", type=float, default=1.0)

    # Saving
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=5)

    # Misc
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default=None)

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
            raise ValueError("--slake_path required for SLAKE dataset")

    model_config = ModelConfig(model_id=args.model_id)

    # ---- Load paraphrase cache ----
    print("\n[1/3] Loading paraphrase cache...")
    cache_path = find_paraphrase_cache(
        args.augmented_dir, args.dataset, args.num_paraphrases
    )
    cache = load_paraphrase_cache(cache_path)

    # ---- Load and augment dataset ----
    print("\n[2/3] Loading and augmenting dataset...")
    dataset = load_and_format_dataset_augmented(
        dataset_name=args.dataset,
        paraphrase_cache=cache,
        slake_path=slake_path,
        question_type=args.question_type,
        max_samples=args.max_samples,
        model_family=model_config.model_family,
        seed=args.seed,
    )

    # ---- Setup GRPO Trainer ----
    print("\n[3/3] Setting up GRPO trainer...")

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
        reward_weights=[args.accuracy_weight, args.format_weight],
        save_strategy=args.save_strategy,
        save_steps=args.save_steps if args.save_strategy == "steps" else 500,
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
        reward_funcs=[accuracy_reward, format_reward],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    # ---- Train ----
    num_gpus = len(args.gpu.split(",")) if args.gpu else 1

    print("\n" + "=" * 60)
    print("Starting Augmented GRPO Training")
    print("=" * 60)
    print(f"Model:           {args.model_id}")
    print(f"Dataset:         {args.dataset}")
    print(f"Question Type:   {args.question_type}")
    print(f"Augmentation:    question paraphrases (n={args.num_paraphrases})")
    print(f"Samples:         {len(dataset)}")
    print(f"Epochs:          {args.epochs}")
    print(f"Num generations: {args.num_generations}")
    print(f"Effective BS:    {args.batch_size * args.grad_accum * num_gpus}")
    print(f"Learning rate:   {args.learning_rate}")
    print(f"Loss type:       {args.loss_type}")
    print(f"LoRA:            r={args.lora_r}, α={args.lora_alpha}")
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
        "training_type": "grpo_augmented",
        "model_id": args.model_id,
        "model_family": model_config.model_family.value,
        "dataset": args.dataset,
        "question_type": args.question_type,
        "training_samples": len(dataset),
        "seed": args.seed,
        "augmentation": {
            "mode": "q_only",
            "num_paraphrases": args.num_paraphrases,
            "augmented_dir": args.augmented_dir,
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
            "accuracy_weight": args.accuracy_weight,
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

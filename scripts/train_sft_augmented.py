#!/usr/bin/env python3
"""
Augmented SFT Training for Medical VQA

Same as train_sft.py but with paraphrase augmentation from cached paraphrases.
Uses identical hyperparameters to the baseline for fair comparison.

The key difference: at each epoch, questions (and optionally answers) are
randomly swapped to a paraphrased version, so the model sees diverse
phrasings across epochs while receiving the same number of gradient steps.

Augmentation modes:
  - q_only:  Swap questions only (for both closed and open)
  - q_and_a: Swap questions + answers (answers only for open-ended, ≥4 words)
  - none:    No augmentation (equivalent to train_sft.py, for sanity check)

Usage:
    python scripts/train_sft_augmented.py \
        --model_id Qwen/Qwen3-VL-8B-Instruct \
        --dataset rad_vqa \
        --output_dir ./checkpoints/aug_sft_qwen3vl_rad_vqa \
        --augmented_dir ./data/augmented \
        --augment_mode q_only \
        --num_paraphrases 8 \
        --gpu 0

    # Same hyperparameters as baseline
    python scripts/train_sft_augmented.py \
        --model_id Qwen/Qwen3-VL-8B-Instruct \
        --dataset rad_vqa \
        --output_dir ./checkpoints/aug_sft_qwen3vl_rad_vqa \
        --augmented_dir ./data/augmented \
        --augment_mode q_and_a \
        --epochs 5 \
        --batch_size 2 \
        --grad_accum 8 \
        --learning_rate 5e-5 \
        --lora_r 64 \
        --lora_alpha 128 \
        --gpu 0
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

from med_vqa.configs import (
    ModelConfig, DataConfig, SFTConfig, LoRAConfig,
    ExperimentConfig, DatasetName, QuestionType,
)
from med_vqa.training import run_sft_training
from med_vqa.training.sft_trainer import VQASFTTrainer
from med_vqa.data import get_dataset
from med_vqa.data.augmented import (
    load_paraphrase_cache,
    find_paraphrase_cache,
    AugmentedVQADataset,
)
from med_vqa.models import load_model
from med_vqa.utils import set_seed

from datetime import datetime
from dataclasses import asdict
import json
import torch
from peft import get_peft_model
from trl import SFTConfig as TRLSFTConfig, SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Augmented SFT Training for Medical VQA"
    )

    # Model
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # Dataset
    parser.add_argument("--dataset", type=str, default="rad_vqa",
                        choices=["rad_vqa", "slake"])
    parser.add_argument("--slake_path", type=str, default="./data/Slake1.0")
    parser.add_argument("--question_type", type=str, default="all",
                        choices=["all", "closed", "open"])
    parser.add_argument("--split", type=str, default="train")

    # Augmentation
    parser.add_argument("--augmented_dir", type=str, default="./data/augmented",
                        help="Directory containing paraphrase JSONL caches")
    parser.add_argument("--augment_mode", type=str, default="q_only",
                        choices=["q_only", "q_and_a", "none"],
                        help="Augmentation mode")
    parser.add_argument("--num_paraphrases", type=int, default=8,
                        help="Number of paraphrases (must match cache)")
    parser.add_argument("--swap_probability", type=float, default=1.0,
                        help="Probability of swapping per sample (1.0=always)")

    # Training (SAME defaults as train_sft.py / train_all_models.sh)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")

    # LoRA (SAME defaults)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Saving
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--save_total_limit", type=int, default=5)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # ---- Resolve dataset ----
    ds_name = DatasetName.RAD_VQA if args.dataset == "rad_vqa" else DatasetName.SLAKE
    slake_path = args.slake_path if args.dataset == "slake" else None

    q_type = {
        "all": QuestionType.ALL,
        "closed": QuestionType.CLOSED,
        "open": QuestionType.OPEN,
    }[args.question_type]

    data_config = DataConfig(
        dataset_name=ds_name,
        question_type=q_type,
        split=args.split,
        data_path=slake_path,
        seed=args.seed,
    )

    # ---- Load base dataset ----
    print("\n[1/4] Loading base dataset...")
    dataset_wrapper = get_dataset(data_config)
    base_dataset = dataset_wrapper.load()

    stats = dataset_wrapper.get_statistics()
    print(f"  Dataset: {stats['name']}, {stats['total_samples']} samples")
    print(f"  Closed: {stats['closed_questions']}, Open: {stats['open_questions']}")

    # ---- Load paraphrase cache & wrap ----
    print("\n[2/4] Setting up augmentation...")
    if args.augment_mode != "none":
        cache_path = find_paraphrase_cache(
            args.augmented_dir, args.dataset, args.num_paraphrases
        )
        cache = load_paraphrase_cache(cache_path)

        augment_q = True
        augment_a = args.augment_mode == "q_and_a"

        dataset = AugmentedVQADataset(
            base_dataset=base_dataset,
            paraphrase_cache=cache,
            augment_questions=augment_q,
            augment_answers=augment_a,
            swap_probability=args.swap_probability,
            seed=args.seed,
        )
        print(f"  Mode: {args.augment_mode}")
        print(f"  Swap probability: {args.swap_probability}")
        print(f"  Cache entries: {len(cache)}")
    else:
        dataset = base_dataset
        print("  Mode: none (baseline)")

    # ---- Load model ----
    print("\n[3/4] Loading model...")
    model_config = ModelConfig(model_id=args.model_id)
    model, processor = load_model(model_config, prepare_for_training=True)

    lora_config = LoRAConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    peft_config = lora_config.to_peft_config()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ---- Setup trainer ----
    print("\n[4/4] Setting up trainer...")

    # Get collator
    from med_vqa.data import get_collator
    collator = get_collator(model_config.model_family, processor, args.max_length)

    num_gpus = len(args.gpu.split(",")) if args.gpu else 1

    training_args = TRLSFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler,
        bf16=True,
        gradient_checkpointing=True,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        logging_steps=1,
        report_to=[],
        remove_unused_columns=False,
        seed=args.seed,
        max_length=args.max_length,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    # ---- Train ----
    print("\n" + "=" * 60)
    print("Starting Augmented SFT Training")
    print("=" * 60)
    print(f"Model:           {args.model_id}")
    print(f"Dataset:         {args.dataset}")
    print(f"Question Type:   {args.question_type}")
    print(f"Augment Mode:    {args.augment_mode}")
    print(f"Samples:         {len(dataset)}")
    print(f"Epochs:          {args.epochs}")
    print(f"Effective BS:    {args.batch_size * args.grad_accum * num_gpus}")
    print(f"Learning Rate:   {args.learning_rate}")
    print(f"LoRA:            r={args.lora_r}, α={args.lora_alpha}")
    print(f"Output:          {args.output_dir}")
    print("=" * 60 + "\n")

    train_result = trainer.train()

    # Save
    final_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_dir)
    print(f"Model saved to: {final_dir}")

    # ---- Save metadata ----
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "training_type": "sft_augmented",
        "model_id": args.model_id,
        "model_family": model_config.model_family.value,
        "dataset": args.dataset,
        "question_type": args.question_type,
        "training_samples": len(dataset),
        "seed": args.seed,
        "augmentation": {
            "mode": args.augment_mode,
            "num_paraphrases": args.num_paraphrases,
            "swap_probability": args.swap_probability,
            "augmented_dir": args.augmented_dir,
        },
        "training": {
            "num_epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.grad_accum,
            "effective_batch_size": args.batch_size * args.grad_accum * num_gpus,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "lr_scheduler": args.lr_scheduler,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
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
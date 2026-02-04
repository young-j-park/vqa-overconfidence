"""
GRPO (Group Relative Policy Optimization) trainer for Medical VQA.

Supports: Qwen3-VL-8B, InternVL3-8B, LLaVA-NeXT-7B
Datasets: RAD-VQA, SLAKE (closed questions only)

Uses TRL's GRPOTrainer with verifiable rewards on closed (yes/no) questions.
The model is prompted with <think>/<answer> tags for structured reasoning.

References:
    - TRL GRPOTrainer: https://huggingface.co/docs/trl/en/grpo_trainer
    - TRL GRPO VLM notebook: huggingface/trl/examples/notebooks/grpo_qwen3_vl.ipynb
"""

import os
import re
import json
import random
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from dataclasses import asdict

import torch
from datasets import Dataset, load_dataset
from PIL import Image
from trl import GRPOConfig as TRLGRPOConfig, GRPOTrainer
from peft import LoraConfig

from ..configs import ExperimentConfig, GRPOConfig, ModelConfig, ModelFamily, DatasetName


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = (
    "You are a helpful medical assistant. "
    "Given the medical image and question, first think about your reasoning, "
    "then provide your final answer. "
    "Output the thinking process in <think> </think> and final answer in "
    "<answer> </answer> tags. "
    "The output format should be as follows:\n"
    "<think> reasoning process here </think>"
    "<answer> answer here </answer>\n"
    "(The answer should be a single word: yes or no. "
    "Do not provide any explanation in the answer tags.)"
)


# =============================================================================
# LoRA Target Modules
# =============================================================================

def get_lora_target_modules(model_family: ModelFamily) -> list:
    """Get LoRA target modules for each model family.

    All three supported families share the same standard transformer
    projection layers, but this is kept as a lookup for future families
    that may differ.
    """
    _TARGETS = {
        ModelFamily.QWEN_VL: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        ModelFamily.INTERNVL: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        ModelFamily.LLAVA: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        ModelFamily.LLAVA_NEXT: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    }
    return _TARGETS.get(model_family, ["q_proj", "v_proj"])


# =============================================================================
# Dataset Loading & Formatting
# =============================================================================

def load_and_format_dataset(
    dataset_name: str,
    slake_path: Optional[str] = None,
    question_type: str = "closed",
    max_samples: Optional[int] = None,
    model_family: Optional[ModelFamily] = None,
    seed: int = 42,
) -> Dataset:
    """Load dataset and format into TRL conversational vision format.

    Output columns:
        prompt       – list[dict]  (conversational chat messages)
        image        – PIL.Image
        ground_truth – str         (for reward function)
    """

    # ------------------------------------------------------------------
    # 1. Load raw data
    # ------------------------------------------------------------------
    if dataset_name == "rad_vqa":
        raw = load_dataset("flaviagiammarino/vqa-rad", split="train")
        if "visual_data" in raw.column_names:
            raw = raw.rename_column("visual_data", "image")

    elif dataset_name == "slake":
        if slake_path is None:
            # Try default location
            default = "./data/Slake1.0"
            if os.path.exists(default):
                slake_path = default
            else:
                raise ValueError(
                    "--slake_path is required for SLAKE dataset "
                    "(or place it at ./data/Slake1.0)"
                )

        slake_json = os.path.join(slake_path, "train.json")
        if not os.path.exists(slake_json):
            raise FileNotFoundError(f"SLAKE train.json not found at {slake_json}")

        with open(slake_json, "r") as f:
            slake_data = json.load(f)

        # English only
        slake_data = [
            item for item in slake_data if item.get("q_lang", "en") == "en"
        ]

        records = []
        for item in slake_data:
            img_path = os.path.join(slake_path, "imgs", item["img_name"])
            if os.path.exists(img_path):
                records.append({
                    "question": item["question"],
                    "answer": str(item["answer"]).strip(),
                    "image": Image.open(img_path).convert("RGB"),
                    "answer_type": item.get("answer_type", "OPEN"),
                })
        raw = Dataset.from_list(records)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # ------------------------------------------------------------------
    # 2. Filter to closed (yes/no) questions
    # ------------------------------------------------------------------
    original_size = len(raw)

    if question_type == "closed":
        def is_closed(example):
            ans = str(example["answer"]).lower().strip()
            return ans in ["yes", "no"]
        raw = raw.filter(is_closed)

    filtered_size = len(raw)
    print(f"Dataset: {dataset_name}  |  "
          f"filter={question_type}  |  "
          f"{original_size} -> {filtered_size}")

    # ------------------------------------------------------------------
    # 3. Format into TRL conversational vision format
    # ------------------------------------------------------------------
    is_internvl = model_family == ModelFamily.INTERNVL

    if is_internvl:
        print("[InternVL] Resizing images to 448x448 "
              "(workaround for dynamic tiling bug, TRL #4061)")

    formatted_records = []
    for i in range(len(raw)):
        example = raw[i]
        question = example.get("question", example.get("Question", ""))
        answer = str(
            example.get("answer", example.get("Answer", ""))
        ).lower().strip()

        prompt = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]

        image = example["image"]
        if is_internvl:
            image = image.convert("RGB").resize((448, 448))

        formatted_records.append({
            "prompt": prompt,
            "image": image,
            "ground_truth": answer,
        })

    # Shuffle deterministically
    rng = random.Random(seed)
    rng.shuffle(formatted_records)

    # Subsample
    if max_samples is not None and max_samples < len(formatted_records):
        formatted_records = formatted_records[:max_samples]
        print(f"[SUBSAMPLE] Truncated to {max_samples} samples")

    dataset = Dataset.from_list(formatted_records)
    print(f"Final training samples: {len(dataset)}")
    return dataset


# =============================================================================
# Reward Functions
# =============================================================================

def accuracy_reward(completions, ground_truth, **kwargs):
    """Binary accuracy reward for closed (yes/no) medical VQA.

    Parses <answer>...</answer> tags from the completion, compares with
    ground truth.  Returns 1.0 for correct, 0.0 for incorrect.
    """
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        if isinstance(completion, list):
            content = completion[0].get("content", "") if completion else ""
        else:
            content = str(completion)

        match = re.search(
            r"<answer>\s*(.*?)\s*</answer>",
            content,
            re.IGNORECASE | re.DOTALL,
        )

        if match:
            predicted = match.group(1).lower().strip()
        else:
            content_lower = content.lower()
            if "yes" in content_lower and "no" not in content_lower:
                predicted = "yes"
            elif "no" in content_lower and "yes" not in content_lower:
                predicted = "no"
            else:
                predicted = ""

        rewards.append(1.0 if predicted == gt.lower().strip() else 0.0)

    return rewards


def format_reward(completions, **kwargs):
    """Format compliance reward.

    Checks for <think>...</think><answer>...</answer> structure.
    Returns 1.0 (both tags), 0.5 (answer only), 0.0 (neither).
    """
    rewards = []
    for completion in completions:
        if isinstance(completion, list):
            content = completion[0].get("content", "") if completion else ""
        else:
            content = str(completion)

        has_think = bool(
            re.search(r"<think>.*?</think>", content, re.DOTALL)
        )
        has_answer = bool(
            re.search(r"<answer>.*?</answer>", content, re.DOTALL)
        )

        if has_think and has_answer:
            rewards.append(1.0)
        elif has_answer:
            rewards.append(0.5)
        else:
            rewards.append(0.0)

    return rewards


# =============================================================================
# VQAGRPOTrainer  (high-level wrapper)
# =============================================================================

class VQAGRPOTrainer:
    """GRPO trainer for Medical VQA tasks.

    Wraps TRL's GRPOTrainer with dataset preparation, reward functions,
    and metadata saving consistent with the rest of the framework.
    """

    def __init__(self, config: ExperimentConfig):
        if config.grpo is None:
            raise ValueError("ExperimentConfig.grpo must be set for GRPO training")

        self.config = config
        self.model_config = config.model
        self.grpo_config = config.grpo
        self.data_config = config.data
        self.trainer: Optional[GRPOTrainer] = None
        self.dataset: Optional[Dataset] = None

    # -----------------------------------------------------------------
    def setup(self) -> None:
        """Prepare dataset, LoRA config, and build TRL GRPOTrainer."""

        print("=" * 60)
        print("Setting up GRPO Training")
        print("=" * 60)

        # --- Dataset ---
        slake_path = getattr(self.data_config, "data_path", None)
        self.dataset = load_and_format_dataset(
            dataset_name=self.data_config.dataset_name.value,
            slake_path=slake_path,
            question_type=self.data_config.question_type.value,
            max_samples=self.data_config.subsample_size,
            model_family=self.model_config.model_family,
            seed=self.config.seed,
        )

        # --- LoRA ---
        peft_config = LoraConfig(
            r=self.grpo_config.lora.r,
            lora_alpha=self.grpo_config.lora.lora_alpha,
            lora_dropout=self.grpo_config.lora.lora_dropout,
            target_modules=get_lora_target_modules(
                self.model_config.model_family
            ),
            bias="none",
            task_type="CAUSAL_LM",
        )

        # --- Model init kwargs ---
        model_init_kwargs = {
            "torch_dtype": torch.bfloat16 if self.grpo_config.bf16 else torch.float16,
            "trust_remote_code": self.model_config.trust_remote_code,
            "device_map": "auto",
        }
        if self.model_config.use_flash_attention:
            try:
                import flash_attn  # noqa
                model_init_kwargs["attn_implementation"] = "flash_attention_2"
                print("Flash Attention 2 enabled")
            except ImportError:
                pass

        # --- GRPOConfig ---
        training_args = TRLGRPOConfig(
            output_dir=self.grpo_config.output_dir,
            # Data
            max_prompt_length=self.grpo_config.max_prompt_length,
            max_completion_length=self.grpo_config.max_completion_length,
            num_generations=self.grpo_config.num_generations,
            remove_unused_columns=False,
            # Generation
            temperature=self.grpo_config.temperature,
            # Training
            num_train_epochs=self.grpo_config.num_epochs,
            per_device_train_batch_size=self.grpo_config.per_device_batch_size,
            gradient_accumulation_steps=self.grpo_config.gradient_accumulation_steps,
            learning_rate=self.grpo_config.learning_rate,
            beta=self.grpo_config.beta,
            loss_type=self.grpo_config.loss_type,
            gradient_checkpointing=self.grpo_config.gradient_checkpointing,
            bf16=self.grpo_config.bf16,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            max_grad_norm=1.0,
            # Reward
            reward_weights=[
                self.grpo_config.accuracy_weight,
                self.grpo_config.format_weight,
            ],
            # Saving
            save_strategy=self.grpo_config.save_strategy,
            save_steps=(
                self.grpo_config.save_steps
                if self.grpo_config.save_strategy == "steps"
                else 500
            ),
            save_total_limit=self.grpo_config.save_total_limit,
            # Logging
            logging_steps=self.grpo_config.logging_steps,
            log_completions=self.grpo_config.log_completions,
            report_to=(
                [self.grpo_config.report_to]
                if self.grpo_config.report_to != "none"
                else []
            ),
            # Model
            model_init_kwargs=model_init_kwargs,
            # Misc
            seed=self.config.seed,
            dataloader_num_workers=2,
        )

        self.trainer = GRPOTrainer(
            model=self.model_config.model_id,
            reward_funcs=[accuracy_reward, format_reward],
            args=training_args,
            train_dataset=self.dataset,
            peft_config=peft_config,
        )

        print(f"Trainer ready  |  samples={len(self.dataset)}")

    # -----------------------------------------------------------------
    def train(self) -> Dict[str, Any]:
        """Run training and return results dict."""
        if self.trainer is None:
            raise RuntimeError("Call setup() before train()")

        print("\n" + "=" * 60)
        print("Starting GRPO Training")
        print("=" * 60)
        print(f"Model:           {self.model_config.model_id}")
        print(f"Dataset:         {self.data_config.dataset_name.value}")
        print(f"Samples:         {len(self.dataset)}")
        print(f"Epochs:          {self.grpo_config.num_epochs}")
        print(f"Num generations: {self.grpo_config.num_generations}")
        print(f"Effective BS:    "
              f"{self.grpo_config.per_device_batch_size * self.grpo_config.gradient_accumulation_steps}")
        print(f"Learning rate:   {self.grpo_config.learning_rate}")
        print(f"Loss type:       {self.grpo_config.loss_type}")
        print(f"LoRA r={self.grpo_config.lora.r}, "
              f"α={self.grpo_config.lora.lora_alpha}")
        print(f"Save strategy:   {self.grpo_config.save_strategy}")
        print(f"Output:          {self.grpo_config.output_dir}")
        print("=" * 60 + "\n")

        train_result = self.trainer.train()

        # Save final model
        final_dir = os.path.join(self.grpo_config.output_dir, "final_model")
        self.trainer.save_model(final_dir)
        print(f"Final model saved to: {final_dir}")

        # Save metadata
        self._save_metadata(train_result)

        return {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime"),
            "output_dir": self.grpo_config.output_dir,
        }

    # -----------------------------------------------------------------
    def _save_metadata(self, train_result) -> None:
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "model_id": self.model_config.model_id,
            "model_family": self.model_config.model_family.value,
            "dataset": self.data_config.dataset_name.value,
            "question_type": self.data_config.question_type.value,
            "training_samples": len(self.dataset),
            "seed": self.config.seed,
            "grpo": {
                "num_epochs": self.grpo_config.num_epochs,
                "batch_size": self.grpo_config.per_device_batch_size,
                "gradient_accumulation": self.grpo_config.gradient_accumulation_steps,
                "learning_rate": self.grpo_config.learning_rate,
                "num_generations": self.grpo_config.num_generations,
                "temperature": self.grpo_config.temperature,
                "beta": self.grpo_config.beta,
                "loss_type": self.grpo_config.loss_type,
                "max_completion_length": self.grpo_config.max_completion_length,
                "lora_r": self.grpo_config.lora.r,
                "lora_alpha": self.grpo_config.lora.lora_alpha,
                "accuracy_weight": self.grpo_config.accuracy_weight,
                "format_weight": self.grpo_config.format_weight,
            },
            "results": {
                "train_loss": train_result.training_loss,
                "train_runtime_seconds": train_result.metrics.get("train_runtime"),
            },
        }
        path = os.path.join(self.grpo_config.output_dir, "experiment_metadata.json")
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        config_path = os.path.join(self.grpo_config.output_dir, "config.json")
        self.config.save(config_path)
        print(f"Metadata saved to: {path}")


# =============================================================================
# Convenience entry-point
# =============================================================================

def run_grpo_training(config: ExperimentConfig) -> Dict[str, Any]:
    """Run GRPO training from an ExperimentConfig.

    Args:
        config: Must have config.grpo populated.

    Returns:
        Training results dictionary.
    """
    trainer = VQAGRPOTrainer(config)
    trainer.setup()
    return trainer.train()

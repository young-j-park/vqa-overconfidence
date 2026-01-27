"""
Supervised Fine-Tuning (SFT) trainer for Medical VQA.

Provides a unified training interface that works with all supported
model families and datasets.
"""

import os
import json
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import asdict

import torch
from transformers import TrainingArguments, Trainer
from peft import get_peft_model
from trl import SFTTrainer, SFTConfig as TRLSFTConfig

from ..configs import ExperimentConfig, SFTConfig, ModelConfig
from ..data import get_dataset, get_collator
from ..models import load_model


class VQASFTTrainer:
    """Unified SFT trainer for Medical VQA tasks."""
    
    def __init__(self, config: ExperimentConfig):
        """Initialize trainer with experiment configuration.
        
        Args:
            config: Full experiment configuration including model, data, and training settings
        """
        self.config = config
        self.model_config = config.model
        self.data_config = config.data
        self.train_config = config.training
        
        if self.train_config is None:
            raise ValueError("Training configuration is required for SFT trainer")
        
        self.model = None
        self.processor = None
        self.dataset = None
        self.trainer = None
    
    def setup(self) -> None:
        """Setup model, dataset, and trainer."""
        print("=" * 60)
        print("Setting up VQA SFT Training")
        print("=" * 60)
        
        # Load and prepare model
        print("\n[1/3] Loading model...")
        self.model, self.processor = load_model(
            self.model_config,
            prepare_for_training=True
        )
        
        # Apply LoRA
        peft_config = self.train_config.lora.to_peft_config()
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        
        # Load and prepare dataset
        print("\n[2/3] Loading dataset...")
        dataset_wrapper = get_dataset(self.data_config)
        self.dataset = dataset_wrapper.load()
        
        # Print dataset statistics
        stats = dataset_wrapper.get_statistics()
        print(f"\nDataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Setup trainer
        print("\n[3/3] Setting up trainer...")
        self._setup_trainer()
        
        print("\n" + "=" * 60)
        print("Setup complete!")
        print("=" * 60)
    
    def _setup_trainer(self) -> None:
        """Configure and create the trainer."""
        
        # Get appropriate collator for model family
        collator = get_collator(
            self.model_config.model_family,
            self.processor,
            self.train_config.max_length
        )
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=self.train_config.output_dir,
            num_train_epochs=self.train_config.num_epochs,
            per_device_train_batch_size=self.train_config.per_device_batch_size,
            gradient_accumulation_steps=self.train_config.gradient_accumulation_steps,
            learning_rate=self.train_config.learning_rate,
            lr_scheduler_type=self.train_config.lr_scheduler_type,
            warmup_ratio=self.train_config.warmup_ratio,
            bf16=self.train_config.bf16,
            fp16=not self.train_config.bf16,
            logging_steps=self.train_config.logging_steps,
            save_strategy=self.train_config.save_strategy,
            save_total_limit=self.train_config.save_total_limit,
            optim=self.train_config.optim,
            gradient_checkpointing=self.train_config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            report_to="none",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            data_collator=collator,
        )
    
    def train(self) -> Dict[str, Any]:
        """Run training and return results."""
        if self.trainer is None:
            raise RuntimeError("Trainer not setup. Call setup() first.")
        
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)
        print(f"Model: {self.model_config.model_id}")
        print(f"Dataset: {self.data_config.dataset_name.value}")
        print(f"Question type: {self.data_config.question_type.value}")
        print(f"Training samples: {len(self.dataset)}")
        print(f"Epochs: {self.train_config.num_epochs}")
        print(f"Output: {self.train_config.output_dir}")
        print("=" * 60 + "\n")
        
        # Train
        train_result = self.trainer.train()
        
        # Save model
        print(f"\nSaving model to {self.train_config.output_dir}...")
        self.trainer.save_model()
        
        if hasattr(self.processor, 'save_pretrained'):
            self.processor.save_pretrained(self.train_config.output_dir)
        
        # Save experiment metadata
        self._save_metadata(train_result)
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        
        return {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime"),
            "output_dir": self.train_config.output_dir,
        }
    
    def _save_metadata(self, train_result) -> None:
        """Save experiment metadata and configuration."""
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "model_id": self.model_config.model_id,
            "model_family": self.model_config.model_family.value,
            "dataset": self.data_config.dataset_name.value,
            "question_type": self.data_config.question_type.value,
            "subsample_size": self.data_config.subsample_size,
            "training_samples": len(self.dataset),
            "seed": self.config.seed,
            "training": {
                "num_epochs": self.train_config.num_epochs,
                "batch_size": self.train_config.per_device_batch_size,
                "gradient_accumulation": self.train_config.gradient_accumulation_steps,
                "effective_batch_size": (
                    self.train_config.per_device_batch_size * 
                    self.train_config.gradient_accumulation_steps
                ),
                "learning_rate": self.train_config.learning_rate,
                "lora_r": self.train_config.lora.r,
                "lora_alpha": self.train_config.lora.lora_alpha,
            },
            "results": {
                "train_loss": train_result.training_loss,
                "train_runtime_seconds": train_result.metrics.get("train_runtime"),
            }
        }
        
        # Save as JSON
        metadata_path = os.path.join(self.train_config.output_dir, "experiment_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Also save full config
        config_path = os.path.join(self.train_config.output_dir, "config.json")
        self.config.save(config_path)
        
        print(f"Metadata saved to: {metadata_path}")


def run_sft_training(config: ExperimentConfig) -> Dict[str, Any]:
    """Convenience function to run SFT training.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Training results dictionary
    """
    trainer = VQASFTTrainer(config)
    trainer.setup()
    return trainer.train()

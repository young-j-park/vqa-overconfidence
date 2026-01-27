"""
Configuration classes for the Medical VQA framework.

Uses dataclasses for type safety and easy serialization.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Literal
from enum import Enum
import json
import os


class ModelFamily(str, Enum):
    """Supported model families."""
    QWEN_VL = "qwen_vl"
    INTERNVL = "internvl"
    LLAVA = "llava"
    LLAVA_NEXT = "llava_next"


class QuestionType(str, Enum):
    """Question type filters."""
    ALL = "all"
    CLOSED = "closed"  # Yes/No questions
    OPEN = "open"      # Free-form answers


class DatasetName(str, Enum):
    """Supported datasets."""
    RAD_VQA = "rad_vqa"
    SLAKE = "slake"
    # Add more as needed


@dataclass
class ModelConfig:
    """Configuration for model loading."""
    model_id: str
    model_family: Optional[ModelFamily] = None  # Auto-detected if None
    use_4bit: bool = True
    use_flash_attention: bool = True
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = True
    
    def __post_init__(self):
        if self.model_family is None:
            self.model_family = self._detect_family()
    
    def _detect_family(self) -> ModelFamily:
        """Auto-detect model family from model_id."""
        model_id_lower = self.model_id.lower()
        
        if "qwen" in model_id_lower and "vl" in model_id_lower:
            return ModelFamily.QWEN_VL
        elif "internvl" in model_id_lower:
            return ModelFamily.INTERNVL
        elif "llava" in model_id_lower:
            if "next" in model_id_lower or "1.6" in model_id_lower:
                return ModelFamily.LLAVA_NEXT
            return ModelFamily.LLAVA
        else:
            raise ValueError(f"Cannot detect model family for: {self.model_id}")


@dataclass
class DataConfig:
    """Configuration for dataset loading and preprocessing."""
    dataset_name: DatasetName
    question_type: QuestionType = QuestionType.ALL
    split: str = "train"
    subsample_size: Optional[int] = None  # None means use full dataset
    seed: int = 42
    
    # Dataset-specific paths (for local datasets)
    data_path: Optional[str] = None
    image_dir: Optional[str] = None


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters."""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: Optional[List[str]] = None  # None means "all-linear"
    
    def to_peft_config(self):
        """Convert to PEFT LoraConfig."""
        from peft import LoraConfig as PeftLoraConfig
        
        return PeftLoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type=self.task_type,
            target_modules=self.target_modules or "all-linear",
        )


@dataclass
class SFTConfig:
    """Configuration for Supervised Fine-Tuning."""
    output_dir: str
    num_epochs: int = 10
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    max_length: int = 2048
    bf16: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    optim: str = "paged_adamw_8bit"
    
    # LoRA config
    lora: LoRAConfig = field(default_factory=LoRAConfig)


@dataclass 
class GRPOConfig:
    """Configuration for GRPO training (placeholder for future implementation)."""
    output_dir: str
    num_epochs: int = 3
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-6
    num_generations: int = 4
    temperature: float = 0.7
    beta: float = 0.1
    max_completion_length: int = 64
    max_prompt_length: int = 2048
    bf16: bool = True
    gradient_checkpointing: bool = True
    
    # LoRA config
    lora: LoRAConfig = field(default_factory=LoRAConfig)


@dataclass
class InferenceConfig:
    """Configuration for model inference."""
    adapter_path: Optional[str] = None  # None means base model
    num_samples: int = 1  # For sampling-based inference
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    batch_size: int = 1


@dataclass
class EvaluationConfig:
    """Configuration for calibration evaluation."""
    output_dir: str
    num_samples: int = 100  # Samples per question for empirical probability
    num_bins: int = 10  # For ECE calculation
    max_examples: Optional[int] = None


@dataclass
class ExperimentConfig:
    """Full experiment configuration combining all sub-configs."""
    model: ModelConfig
    data: DataConfig
    training: Optional[SFTConfig] = None
    inference: Optional[InferenceConfig] = None
    evaluation: Optional[EvaluationConfig] = None
    
    # Experiment metadata
    experiment_name: str = "experiment"
    seed: int = 42
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        
        # Reconstruct nested dataclasses
        model_config = ModelConfig(**data["model"])
        data_config = DataConfig(**data["data"])
        
        training_config = None
        if data.get("training"):
            lora_data = data["training"].pop("lora", {})
            lora_config = LoRAConfig(**lora_data)
            training_config = SFTConfig(**data["training"], lora=lora_config)
        
        inference_config = None
        if data.get("inference"):
            inference_config = InferenceConfig(**data["inference"])
        
        eval_config = None
        if data.get("evaluation"):
            eval_config = EvaluationConfig(**data["evaluation"])
        
        return cls(
            model=model_config,
            data=data_config,
            training=training_config,
            inference=inference_config,
            evaluation=eval_config,
            experiment_name=data.get("experiment_name", "experiment"),
            seed=data.get("seed", 42),
        )


# Convenience function for quick config creation
def create_sft_config(
    model_id: str,
    output_dir: str,
    dataset: str = "rad_vqa",
    question_type: str = "all",
    subsample_size: Optional[int] = None,
    num_epochs: int = 10,
    seed: int = 42,
) -> ExperimentConfig:
    """Create a standard SFT experiment configuration."""
    
    return ExperimentConfig(
        model=ModelConfig(model_id=model_id),
        data=DataConfig(
            dataset_name=DatasetName(dataset),
            question_type=QuestionType(question_type),
            subsample_size=subsample_size,
            seed=seed,
        ),
        training=SFTConfig(output_dir=output_dir, num_epochs=num_epochs),
        experiment_name=f"sft_{os.path.basename(output_dir)}",
        seed=seed,
    )

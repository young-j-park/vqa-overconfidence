"""
Model loading utilities for different VLM families.

Supports:
- Qwen-VL (2B, 4B, 8B)
- InternVL3 (2B, 4B, 8B)
- LLaVA (1.5, NeXT)

Handles:
- 4-bit quantization (QLoRA)
- Full precision loading
- Flash Attention 2
- Adapter loading (PEFT/LoRA)
"""

import torch
from typing import Tuple, Optional, Dict, Any
from transformers import (
    AutoModelForImageTextToText,
    AutoModel,
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
)
from peft import PeftModel, prepare_model_for_kbit_training

from ..configs import ModelConfig, ModelFamily


def get_bnb_config(use_4bit: bool = True, compute_dtype: str = "bfloat16") -> Optional[BitsAndBytesConfig]:
    """Create BitsAndBytes quantization config."""
    if not use_4bit:
        return None
    
    dtype = torch.bfloat16 if compute_dtype == "bfloat16" else torch.float16
    
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_str, torch.bfloat16)


class ModelLoader:
    """Unified model loader for different VLM families."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.processor = None
    
    def load(self, adapter_path: Optional[str] = None) -> Tuple[Any, Any]:
        """Load model and processor.
        
        Args:
            adapter_path: Optional path to LoRA adapter checkpoint
            
        Returns:
            Tuple of (model, processor)
        """
        # Check dependencies
        self._check_dependencies()
        
        # Get quantization config
        bnb_config = get_bnb_config(
            self.config.use_4bit, 
            self.config.torch_dtype
        )
        
        # Load based on model family
        if self.config.model_family == ModelFamily.QWEN_VL:
            self.model, self.processor = self._load_qwen_vl(bnb_config)
        elif self.config.model_family == ModelFamily.INTERNVL:
            self.model, self.processor = self._load_internvl(bnb_config)
        elif self.config.model_family == ModelFamily.LLAVA:
            self.model, self.processor = self._load_llava(bnb_config)
        elif self.config.model_family == ModelFamily.LLAVA_NEXT:
            self.model, self.processor = self._load_llava_next(bnb_config)
        else:
            raise ValueError(f"Unsupported model family: {self.config.model_family}")
        
        # Load adapter if specified
        if adapter_path:
            print(f"Loading LoRA adapter from: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        
        # Ensure pad token
        self._ensure_pad_token()
        
        print(f"Model loaded: {self.config.model_id}")
        print(f"  Family: {self.config.model_family.value}")
        print(f"  4-bit: {self.config.use_4bit}")
        if torch.cuda.is_available():
            print(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        return self.model, self.processor
    
    def prepare_for_training(self) -> None:
        """Prepare model for k-bit training."""
        if self.config.use_4bit and self.model is not None:
            self.model = prepare_model_for_kbit_training(self.model)
            self.model.gradient_checkpointing_enable()
    
    def _check_dependencies(self):
        """Check model-specific dependencies."""
        if self.config.model_family == ModelFamily.INTERNVL:
            try:
                import timm
            except ImportError:
                raise ImportError(
                    "InternVL requires 'timm' package. "
                    "Please run: pip install timm"
                )
    
    def _get_common_kwargs(self, bnb_config: Optional[BitsAndBytesConfig]) -> Dict:
        """Get common model loading kwargs."""
        kwargs = {
            "torch_dtype": get_torch_dtype(self.config.torch_dtype),
            "device_map": self.config.device_map,
            "low_cpu_mem_usage": True,
        }
        
        if bnb_config:
            kwargs["quantization_config"] = bnb_config
        
        if self.config.use_flash_attention:
            kwargs["attn_implementation"] = "flash_attention_2"
        
        return kwargs
    
    def _load_qwen_vl(self, bnb_config) -> Tuple[Any, Any]:
        """Load Qwen-VL model."""
        kwargs = self._get_common_kwargs(bnb_config)
        
        model = AutoModelForImageTextToText.from_pretrained(
            self.config.model_id,
            **kwargs
        )
        processor = AutoProcessor.from_pretrained(self.config.model_id)
        
        return model, processor
    
    def _load_internvl(self, bnb_config) -> Tuple[Any, Any]:
        """Load InternVL3 model (HuggingFace version).
        
        Uses AutoModelForImageTextToText for generation capabilities.
        """
        kwargs = self._get_common_kwargs(bnb_config)
        kwargs["trust_remote_code"] = self.config.trust_remote_code
        
        # InternVL3-hf uses AutoModelForImageTextToText (same as Qwen-VL)
        model = AutoModelForImageTextToText.from_pretrained(
            self.config.model_id,
            **kwargs
        )
        processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code
        )
        
        # Ensure pad token is set
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        
        return model, processor
    
    def _load_llava(self, bnb_config) -> Tuple[Any, Any]:
        """Load LLaVA 1.5 model."""
        kwargs = self._get_common_kwargs(bnb_config)
        
        model = LlavaForConditionalGeneration.from_pretrained(
            self.config.model_id,
            **kwargs
        )
        processor = AutoProcessor.from_pretrained(self.config.model_id)
        
        return model, processor
    
    def _load_llava_next(self, bnb_config) -> Tuple[Any, Any]:
        """Load LLaVA-NeXT (1.6+) model."""
        kwargs = self._get_common_kwargs(bnb_config)
        
        model = LlavaNextForConditionalGeneration.from_pretrained(
            self.config.model_id,
            **kwargs
        )
        processor = AutoProcessor.from_pretrained(self.config.model_id)
        
        return model, processor
    
    def _ensure_pad_token(self):
        """Ensure processor has a pad token set."""
        if self.processor is None:
            return
        
        tokenizer = getattr(self.processor, 'tokenizer', self.processor)
        
        if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token


def load_model(
    config: ModelConfig,
    adapter_path: Optional[str] = None,
    prepare_for_training: bool = False
) -> Tuple[Any, Any]:
    """Convenience function to load model and processor.
    
    Args:
        config: Model configuration
        adapter_path: Optional path to LoRA adapter
        prepare_for_training: Whether to prepare for k-bit training
        
    Returns:
        Tuple of (model, processor)
    """
    loader = ModelLoader(config)
    model, processor = loader.load(adapter_path)
    
    if prepare_for_training:
        loader.prepare_for_training()
    
    return model, processor


def load_base_model_for_grpo(config: ModelConfig) -> Tuple[Any, Any]:
    """Load model without quantization for GRPO training.
    
    GRPO typically requires full precision models.
    """
    # Create a copy of config without quantization
    full_precision_config = ModelConfig(
        model_id=config.model_id,
        model_family=config.model_family,
        use_4bit=False,  # Full precision
        use_flash_attention=config.use_flash_attention,
        torch_dtype=config.torch_dtype,
        device_map=config.device_map,
        trust_remote_code=config.trust_remote_code,
    )
    
    loader = ModelLoader(full_precision_config)
    return loader.load()

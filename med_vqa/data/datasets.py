"""
Dataset loading and preprocessing for Medical VQA.

Provides a unified interface for different VQA datasets with support for:
- Question type filtering (closed/open/all)
- Subsampling with seed control
- Conversion to unified format
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Iterator
from datasets import Dataset, load_dataset
import random

from ..configs import DataConfig, QuestionType, DatasetName


@dataclass
class VQASample:
    """Unified VQA sample format across all datasets."""
    image: Any  # PIL Image
    question: str
    answer: str
    answer_type: str  # "closed" or "open"
    
    # Optional metadata
    question_id: Optional[str] = None
    image_id: Optional[str] = None
    dataset_source: Optional[str] = None
    extra_metadata: Optional[Dict] = None


class BaseVQADataset(ABC):
    """Abstract base class for VQA datasets."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self._raw_dataset: Optional[Dataset] = None
        self._processed_dataset: Optional[Dataset] = None
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name identifier."""
        pass
    
    @abstractmethod
    def _load_raw(self) -> Dataset:
        """Load raw dataset from source."""
        pass
    
    @abstractmethod
    def _determine_answer_type(self, sample: Dict) -> str:
        """Determine if a sample is closed (yes/no) or open."""
        pass
    
    @abstractmethod
    def _to_unified_format(self, sample: Dict, idx: int) -> Dict:
        """Convert dataset-specific format to unified format."""
        pass
    
    def load(self) -> Dataset:
        """Load and process the dataset."""
        # Load raw data
        self._raw_dataset = self._load_raw()
        
        # Convert to unified format (with index for ID generation)
        self._processed_dataset = self._raw_dataset.map(
            lambda sample, idx: self._to_unified_format(sample, idx),
            with_indices=True,
            remove_columns=self._raw_dataset.column_names,
        )
        
        # Filter by question type
        self._processed_dataset = self._filter_by_question_type()
        
        # Shuffle with seed
        self._processed_dataset = self._processed_dataset.shuffle(seed=self.config.seed)
        
        # Subsample if requested
        if self.config.subsample_size is not None:
            self._processed_dataset = self._subsample()
        
        return self._processed_dataset
    
    def _filter_by_question_type(self) -> Dataset:
        """Filter dataset by question type."""
        if self.config.question_type == QuestionType.ALL:
            return self._processed_dataset
        
        target_type = self.config.question_type.value
        
        original_size = len(self._processed_dataset)
        filtered = self._processed_dataset.filter(
            lambda x: x["answer_type"] == target_type
        )
        
        print(f"[{self.name}] Filtered {self.config.question_type.value} questions: "
              f"{len(filtered)}/{original_size} ({len(filtered)/original_size*100:.1f}%)")
        
        return filtered
    
    def _subsample(self) -> Dataset:
        """Subsample dataset with seed control."""
        target_size = min(self.config.subsample_size, len(self._processed_dataset))
        
        # Use deterministic selection after shuffle
        subsampled = self._processed_dataset.select(range(target_size))
        
        print(f"[{self.name}] Subsampled to {target_size} examples (seed={self.config.seed})")
        
        return subsampled
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if self._processed_dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        closed_count = sum(1 for x in self._processed_dataset if x["answer_type"] == "closed")
        open_count = len(self._processed_dataset) - closed_count
        
        return {
            "name": self.name,
            "total_samples": len(self._processed_dataset),
            "closed_questions": closed_count,
            "open_questions": open_count,
            "closed_ratio": closed_count / len(self._processed_dataset) if len(self._processed_dataset) > 0 else 0,
            "question_type_filter": self.config.question_type.value,
            "subsample_size": self.config.subsample_size,
            "seed": self.config.seed,
        }


class RADVQADataset(BaseVQADataset):
    """RAD-VQA (Radiology VQA) dataset."""
    
    HF_DATASET_ID = "flaviagiammarino/vqa-rad"
    
    @property
    def name(self) -> str:
        return "RAD-VQA"
    
    def _load_raw(self) -> Dataset:
        """Load RAD-VQA from HuggingFace."""
        print(f"[{self.name}] Loading from HuggingFace: {self.HF_DATASET_ID}")
        return load_dataset(self.HF_DATASET_ID, split=self.config.split)
    
    def _determine_answer_type(self, sample: Dict) -> str:
        """RAD-VQA: closed if answer is yes/no."""
        answer = str(sample.get("answer", "")).lower().strip()
        return "closed" if answer in ["yes", "no"] else "open"
    
    def _to_unified_format(self, sample: Dict, idx: int) -> Dict:
        """Convert RAD-VQA format to unified format."""
        return {
            "image": sample["image"],
            "question": sample["question"],
            "answer": str(sample["answer"]),
            "answer_type": self._determine_answer_type(sample),
            "question_id": f"rad_vqa_q_{idx}",
            "image_id": f"rad_vqa_img_{idx}",
            "dataset_source": self.name,
        }


class SLAKEDataset(BaseVQADataset):
    """SLAKE (Semantically-Labeled Knowledge-Enhanced) Medical VQA dataset."""
    
    HF_DATASET_ID = "BoKelworworker/SLAKE"
    
    @property
    def name(self) -> str:
        return "SLAKE"
    
    def _load_raw(self) -> Dataset:
        """Load SLAKE from HuggingFace."""
        print(f"[{self.name}] Loading from HuggingFace: {self.HF_DATASET_ID}")
        
        # SLAKE has different splits structure
        # Map our split names to SLAKE's structure
        split_mapping = {
            "train": "train",
            "test": "test", 
            "validation": "validate",
        }
        hf_split = split_mapping.get(self.config.split, self.config.split)
        
        return load_dataset(self.HF_DATASET_ID, split=hf_split)
    
    def _determine_answer_type(self, sample: Dict) -> str:
        """SLAKE: has explicit answer_type field, but we normalize it."""
        # SLAKE uses "CLOSED" and "OPEN" in answer_type field
        answer_type = str(sample.get("answer_type", "")).upper()
        
        if answer_type == "CLOSED":
            return "closed"
        elif answer_type == "OPEN":
            return "open"
        else:
            # Fallback: check answer content
            answer = str(sample.get("answer", "")).lower().strip()
            return "closed" if answer in ["yes", "no"] else "open"
    
    def _to_unified_format(self, sample: Dict, idx: int) -> Dict:
        """Convert SLAKE format to unified format."""
        # Use existing IDs if available, otherwise generate from index
        question_id = sample.get("qid") or f"slake_q_{idx}"
        image_id = sample.get("img_id") or sample.get("img_name") or f"slake_img_{idx}"
        
        return {
            "image": sample["img"],  # SLAKE uses 'img' instead of 'image'
            "question": sample["question"],
            "answer": str(sample["answer"]),
            "answer_type": self._determine_answer_type(sample),
            "question_id": str(question_id),
            "image_id": str(image_id),
            "dataset_source": self.name,
        }


# Dataset Registry
_DATASET_REGISTRY: Dict[DatasetName, type] = {
    DatasetName.RAD_VQA: RADVQADataset,
    DatasetName.SLAKE: SLAKEDataset,
}


def get_dataset(config: DataConfig) -> BaseVQADataset:
    """Factory function to get the appropriate dataset class."""
    if config.dataset_name not in _DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {config.dataset_name}. "
                        f"Available: {list(_DATASET_REGISTRY.keys())}")
    
    dataset_cls = _DATASET_REGISTRY[config.dataset_name]
    return dataset_cls(config)


def register_dataset(name: DatasetName, dataset_cls: type):
    """Register a new dataset class."""
    _DATASET_REGISTRY[name] = dataset_cls


def list_available_datasets() -> List[str]:
    """List all registered datasets."""
    return [d.value for d in _DATASET_REGISTRY.keys()]

"""
Dataset loading and preprocessing for Medical VQA.

Provides a unified interface for different VQA datasets with support for:
- Question type filtering (closed/open/all)
- Subsampling with seed control
- Conversion to unified format
- Local file loading (for datasets like SLAKE)
- Optimized image handling with HuggingFace Image feature
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from datasets import Dataset, Features, Image, Value, load_dataset
import json
import os

from ..configs import DataConfig, QuestionType, DatasetName


# Unified schema for all VQA datasets
VQA_FEATURES = Features({
    "image": Image(),  # HuggingFace optimized image handling
    "question": Value("string"),
    "answer": Value("string"),
    "answer_type": Value("string"),  # "closed" or "open"
    "question_id": Value("string"),
    "image_id": Value("string"),
    "dataset_source": Value("string"),
})


@dataclass
class VQASample:
    """Unified VQA sample format across all datasets."""
    image: Any  # PIL Image or path
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
        """Load raw dataset and return in unified format with VQA_FEATURES."""
        pass
    
    def load(self) -> Dataset:
        """Load and process the dataset."""
        # Load raw data (already in unified format)
        self._processed_dataset = self._load_raw()
        
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
    
    def _determine_answer_type(self, answer: str) -> str:
        """RAD-VQA: closed if answer is yes/no."""
        answer_lower = answer.lower().strip()
        return "closed" if answer_lower in ["yes", "no"] else "open"
    
    def _load_raw(self) -> Dataset:
        """Load RAD-VQA from HuggingFace and convert to unified format."""
        print(f"[{self.name}] Loading from HuggingFace: {self.HF_DATASET_ID}")
        raw_dataset = load_dataset(self.HF_DATASET_ID, split=self.config.split)
        
        print(f"[{self.name}] Converting {len(raw_dataset)} samples to unified format...")
        
        # Build unified samples list
        samples = []
        for idx, sample in enumerate(raw_dataset):
            answer = str(sample["answer"])
            samples.append({
                "image": sample["image"],  # Already PIL Image from HF
                "question": sample["question"],
                "answer": answer,
                "answer_type": self._determine_answer_type(answer),
                "question_id": f"rad_vqa_q_{idx}",
                "image_id": f"rad_vqa_img_{idx}",
                "dataset_source": self.name,
            })
        
        # Create dataset with optimized features
        dataset = Dataset.from_list(samples, features=VQA_FEATURES)
        print(f"[{self.name}] Loaded {len(dataset)} samples")
        
        return dataset


class SLAKEDataset(BaseVQADataset):
    """SLAKE (Semantically-Labeled Knowledge-Enhanced) Medical VQA dataset.
    
    Supports loading from local files (official SLAKE dataset directory).
    
    The official SLAKE dataset structure:
        Slake1.0/
        ├── train.json
        ├── test.json  
        ├── validate.json
        └── imgs/
            ├── xmlab0/source.jpg
            ├── xmlab1/source.jpg
            └── ...
    
    To use local files, set data_path in DataConfig:
        DataConfig(
            dataset_name=DatasetName.SLAKE,
            data_path="/path/to/Slake1.0",
            ...
        )
    """
    
    # Default local paths to check
    DEFAULT_LOCAL_PATHS = [
        "./data/Slake1.0",
    ]
    
    @property
    def name(self) -> str:
        return "SLAKE"
    
    def _find_local_path(self) -> str:
        """Find SLAKE dataset in common locations."""
        # First check config
        if self.config.data_path:
            expanded = os.path.expanduser(self.config.data_path)
            if os.path.exists(expanded):
                return expanded
            raise FileNotFoundError(f"SLAKE data_path not found: {self.config.data_path}")
        
        # Check default locations
        for path in self.DEFAULT_LOCAL_PATHS:
            expanded = os.path.expanduser(path)
            if os.path.exists(expanded):
                print(f"[{self.name}] Found local dataset at: {expanded}")
                return expanded
        
        raise FileNotFoundError(
            f"SLAKE dataset not found. Please set data_path in DataConfig.\n"
            f"Checked locations:\n" + 
            "\n".join(f"  - {p}" for p in self.DEFAULT_LOCAL_PATHS)
        )
    
    def _normalize_answer_type(self, answer_type: str) -> str:
        """Normalize SLAKE answer_type (CLOSED/OPEN -> closed/open)."""
        answer_type_upper = str(answer_type).upper()
        if answer_type_upper == "CLOSED":
            return "closed"
        elif answer_type_upper == "OPEN":
            return "open"
        else:
            return "open"  # Default to open for unknown types
    
    def _load_raw(self) -> Dataset:
        """Load SLAKE from local files with optimized image handling."""
        data_path = self._find_local_path()
        
        # Map split names
        split_mapping = {
            "train": "train.json",
            "test": "test.json",
            "validation": "validate.json",
            "validate": "validate.json",
        }
        
        split_file = split_mapping.get(self.config.split)
        if split_file is None:
            raise ValueError(f"Unknown split: {self.config.split}. "
                           f"Available: {list(split_mapping.keys())}")
        
        json_path = os.path.join(data_path, split_file)
        imgs_dir = os.path.join(data_path, "imgs")
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Split file not found: {json_path}")
        if not os.path.exists(imgs_dir):
            raise FileNotFoundError(f"Images directory not found: {imgs_dir}")
        
        print(f"[{self.name}] Loading from local: {json_path}")
        
        # Load JSON annotations
        with open(json_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)
        
        # Filter to English only
        english_annotations = [a for a in annotations if a.get("q_lang") == "en"]
        print(f"[{self.name}] Filtered to English: {len(english_annotations)}/{len(annotations)}")
        
        # Build samples with image PATHS (not loaded PIL images)
        # HuggingFace Image feature will handle lazy loading
        samples = []
        missing_images = 0
        
        for idx, ann in enumerate(english_annotations):
            img_name = ann.get("img_name", "")
            img_path = os.path.join(imgs_dir, img_name)
            
            if os.path.exists(img_path):
                # Store path - HuggingFace Image feature loads lazily
                qid = ann.get("qid", idx)
                img_id = ann.get("img_id", idx)
                
                samples.append({
                    "image": img_path,  # Path string, not PIL Image!
                    "question": ann["question"],
                    "answer": str(ann["answer"]),
                    "answer_type": self._normalize_answer_type(ann.get("answer_type", "OPEN")),
                    "question_id": f"slake_q_{qid}",
                    "image_id": f"slake_img_{img_id}",
                    "dataset_source": self.name,
                })
            else:
                missing_images += 1
        
        if missing_images > 0:
            print(f"[{self.name}] Warning: {missing_images} images not found")
        
        # Create dataset with optimized HuggingFace Image feature
        # Images will be loaded lazily and efficiently by Arrow
        dataset = Dataset.from_list(samples, features=VQA_FEATURES)
        print(f"[{self.name}] Loaded {len(dataset)} samples (images loaded lazily)")
        
        return dataset


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
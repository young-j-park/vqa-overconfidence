# Medical VQA Training Framework

A modular, extensible framework for training and evaluating Vision-Language Models (VLMs) on Medical Visual Question Answering tasks.

## Features

- **Multiple Model Support**: QwenVL, InternVL, LLaVA (1.5 and NeXT)
- **Multiple Datasets**: RAD-VQA, SLAKE (easily extensible)
- **Training Methods**: SFT (Supervised Fine-Tuning), GRPO (planned)
- **Question Type Filtering**: Train on closed-only, open-only, or mixed questions
- **Dataset Subsampling**: Control training set size with seed for reproducibility
- **Calibration Evaluation**: ECE, MCE, overconfidence metrics
- **Optimized Data Loading**: Lazy image loading with HuggingFace Image feature

## Project Structure

```
med_vqa/
├── configs/          # Configuration dataclasses
│   └── config.py     # ModelConfig, DataConfig, SFTConfig, etc.
├── data/             # Dataset loading and preprocessing
│   ├── datasets.py   # RADVQADataset, SLAKEDataset, registry
│   └── collators.py  # Model-specific data collators
├── models/           # Model loading utilities
│   └── loader.py     # Unified model loader for all families
├── training/         # Training implementations
│   ├── sft_trainer.py   # SFT training
│   └── grpo_trainer.py  # GRPO (placeholder for future)
├── inference/        # Inference utilities
│   └── predictor.py  # Unified inference class
├── evaluation/       # Evaluation metrics
│   └── calibration.py   # ECE, MCE, overconfidence
└── utils/            # Helper utilities
    └── helpers.py    # Seed setting, GPU selection, logging

scripts/
├── quick_test.py         # ⭐ Quick E2E test (start here!)
├── test_framework.py     # Component unit tests
├── train_sft.py          # SFT training CLI
├── run_inference.py      # Inference CLI
├── run_experiments.py    # Batch experiment runner
└── evaluate_calibration.py  # Calibration evaluation CLI
```

## Scripts Overview

| Script | Purpose | GPU Required |
|--------|---------|--------------|
| `quick_test.py` | Full pipeline test with minimal data | Yes |
| `test_framework.py --basic` | Unit tests for configs, datasets, metrics | No |
| `train_sft.py` | Train SFT models | Yes |
| `run_inference.py` | Run inference (base or SFT) | Yes |
| `evaluate_calibration.py` | Compute ECE, MCE, overconfidence | Yes |
| `run_experiments.py` | Batch experiments on multiple GPUs | Yes |

## Installation

```bash
pip install -r requirements.txt

# Optional: Flash Attention (recommended for speed)
pip install flash-attn --no-build-isolation
```

## Dataset Setup

### RAD-VQA

RAD-VQA is automatically downloaded from HuggingFace. No setup required.

```python
DataConfig(
    dataset_name=DatasetName.RAD_VQA,
    split="train",  # or "test"
)
```

### SLAKE

SLAKE requires downloading the dataset manually from the official source.

#### 1. Download SLAKE

Download from the official website: https://www.med-vqa.com/slake/

Or use the Google Drive link provided on the project page.

#### 2. Extract the dataset

```bash
unzip Slake1.0.zip -d /path/to/datasets/
```

Expected structure:
```
Slake1.0/
├── train.json      # 9,835 samples (4,919 English)
├── test.json       # 2,094 samples (1,061 English)
├── validate.json   # 2,099 samples (1,053 English)
├── imgs/
│   ├── xmlab0/
│   │   └── source.jpg
│   ├── xmlab1/
│   │   └── source.jpg
│   └── ... (642 image directories)
└── KG/             # Knowledge graph (optional)
```

#### 3. Configure the path

Option A: Set `data_path` in DataConfig:
```python
DataConfig(
    dataset_name=DatasetName.SLAKE,
    data_path="/path/to/Slake1.0",
    split="train",
)
```

Option B: Place in a default location (auto-detected):
- `/data/datasets/Slake1.0`
- `~/datasets/Slake1.0`
- `./data/Slake1.0`
- `~/Downloads/Slake1.0`

#### SLAKE Dataset Statistics

| Split | Total | English | EN-Closed | EN-Open |
|-------|-------|---------|-----------|---------|
| train | 9,835 | 4,919 | 1,943 | 2,976 |
| test | 2,094 | 1,061 | 416 | 645 |
| validate | 2,099 | 1,053 | 422 | 631 |

**Note**: The framework automatically filters to English-only questions.

## Quick Test (Verify Everything Works)

Run a full end-to-end test with minimal data:

```bash
# Test with RAD-VQA (default) - ~5-10 min
python scripts/quick_test.py --gpu 0

# Test with SLAKE dataset
python scripts/quick_test.py --gpu 0 --dataset slake --slake_path /path/to/Slake1.0

# Test with SLAKE (auto-detect path)
python scripts/quick_test.py --gpu 0 --dataset slake

# Test ALL supported model families - ~20-30 min
python scripts/quick_test.py --test_all_models --gpu 0

# Test specific model families
python scripts/quick_test.py --model_families qwen,internvl --gpu 0

# Fast test: inference only (no training/calibration) - ~2 min per model
python scripts/quick_test.py --inference_only --gpu 0

# Keep outputs for inspection
python scripts/quick_test.py --keep_outputs --output_dir ./test_outputs --gpu 0
```

### Supported Model Families

| Family | Default Test Model | Flag |
|--------|-------------------|------|
| Qwen-VL | `Qwen/Qwen2-VL-2B-Instruct` | `--model_families qwen` |
| InternVL3 | `OpenGVLab/InternVL3-1B-hf` | `--model_families internvl` |
| LLaVA 1.5 | `llava-hf/llava-1.5-7b-hf` | `--model_families llava` |
| LLaVA-NeXT | `llava-hf/llava-v1.6-mistral-7b-hf` | `--model_families llava_next` |

### What the Quick Test Does

| Test | Description | Time/Model |
|------|-------------|------------|
| 1. Base Inference | Load base model, run inference on 5 test samples | ~1-2 min |
| 2. SFT Training | Train LoRA adapter for 1 epoch on 5 samples | ~2-3 min |
| 3. SFT Inference | Load trained adapter, run inference | ~1-2 min |
| 4. Calibration | Sample 10x per question, compute ECE/MCE | ~2-3 min |

## Component Tests (No GPU Required)

For testing individual components without GPU:

```bash
# Run basic tests (imports, configs, datasets, evaluation logic)
python scripts/test_framework.py --basic

# Test specific components
python scripts/test_framework.py --test configs     # Configuration classes
python scripts/test_framework.py --test datasets    # Dataset loading
python scripts/test_framework.py --test evaluation  # Calibration metrics
```

## GPU Selection

All scripts support the `--gpu` argument to specify which GPU(s) to use:

```bash
# Use GPU 0
python scripts/train_sft.py --gpu 0 ...

# Use GPU 5
python scripts/train_sft.py --gpu 5 ...

# Use multiple GPUs (for distributed training)
python scripts/train_sft.py --gpu 0,1,2,3 ...
```

You can also set GPU programmatically:

```python
from med_vqa.utils import set_gpu

# IMPORTANT: Call this BEFORE importing torch!
set_gpu(5)  # Use GPU 5

# Now import other modules
from med_vqa.training import run_sft_training
```

## Quick Start

### 1. SFT Training

```bash
# Train Qwen-VL-2B on RAD-VQA closed questions
python scripts/train_sft.py \
    --model_id Qwen/Qwen3-VL-2B-Instruct \
    --output_dir ./checkpoints/qwen3-2b-radvqa-closed \
    --dataset rad_vqa \
    --question_type closed \
    --epochs 10

# Train on SLAKE dataset
python scripts/train_sft.py \
    --model_id Qwen/Qwen3-VL-2B-Instruct \
    --output_dir ./checkpoints/qwen3-2b-slake-closed \
    --dataset slake \
    --slake_path /path/to/Slake1.0 \
    --question_type closed \
    --epochs 10

# Train on mixed questions with subsampling
python scripts/train_sft.py \
    --model_id Qwen/Qwen3-VL-4B-Instruct \
    --output_dir ./checkpoints/qwen3-4b-all-n500 \
    --dataset rad_vqa \
    --question_type all \
    --subsample_size 500 \
    --seed 42
```

### 2. Inference

```bash
# Base model inference
python scripts/run_inference.py \
    --model_id Qwen/Qwen3-VL-2B-Instruct \
    --dataset rad_vqa \
    --split test \
    --output_path ./results/base_predictions.json

# SFT model inference
python scripts/run_inference.py \
    --model_id Qwen/Qwen3-VL-2B-Instruct \
    --adapter_path ./checkpoints/qwen3-2b-closed \
    --dataset rad_vqa \
    --split test \
    --output_path ./results/sft_predictions.json

# Inference on SLAKE
python scripts/run_inference.py \
    --model_id Qwen/Qwen3-VL-2B-Instruct \
    --dataset slake \
    --slake_path /path/to/Slake1.0 \
    --split test \
    --output_path ./results/slake_predictions.json
```

### 3. Calibration Evaluation

```bash
python scripts/evaluate_calibration.py \
    --model_id Qwen/Qwen3-VL-2B-Instruct \
    --adapter_path ./checkpoints/qwen3-2b-closed \
    --dataset rad_vqa \
    --num_samples 100 \
    --output_dir ./results/calibration/qwen3-2b-sft
```

### 4. Batch Experiments

```bash
# Run multiple experiments on multiple GPUs
python scripts/run_experiments.py \
    --models "Qwen/Qwen3-VL-2B-Instruct,Qwen/Qwen3-VL-4B-Instruct" \
    --question_types "closed,all" \
    --gpus "0,1,2,3" \
    --epochs 10
```

## Programmatic Usage

```python
from med_vqa.configs import (
    ModelConfig, DataConfig, SFTConfig, 
    ExperimentConfig, DatasetName, QuestionType
)
from med_vqa.training import run_sft_training

# Create experiment configuration
config = ExperimentConfig(
    model=ModelConfig(model_id="Qwen/Qwen3-VL-2B-Instruct"),
    data=DataConfig(
        dataset_name=DatasetName.RAD_VQA,
        question_type=QuestionType.CLOSED,
        subsample_size=100,
        seed=42,
    ),
    training=SFTConfig(
        output_dir="./checkpoints/my_experiment",
        num_epochs=10,
    ),
    seed=42,
)

# Run training
results = run_sft_training(config)
```

### Using SLAKE programmatically

```python
from med_vqa.configs import DataConfig, DatasetName, QuestionType
from med_vqa.data import get_dataset

# Load SLAKE dataset
config = DataConfig(
    dataset_name=DatasetName.SLAKE,
    data_path="/path/to/Slake1.0",  # Required for SLAKE
    question_type=QuestionType.CLOSED,
    split="train",
    subsample_size=100,
    seed=42,
)

dataset_wrapper = get_dataset(config)
dataset = dataset_wrapper.load()

# Access samples (images loaded lazily)
sample = dataset[0]
print(sample["question"])  # Text loaded immediately
print(sample["image"])     # Image loaded on-demand
```

## Data Loading Optimization

The framework uses HuggingFace's `Image` feature for optimized data loading:

- **Lazy Loading**: Images are loaded on-demand, not all at once
- **Memory Mapped**: Arrow format enables efficient memory usage
- **Fast Batching**: Optimized for DataLoader and Trainer

```python
# Images stored as paths, loaded lazily
VQA_FEATURES = Features({
    "image": Image(),  # HuggingFace handles lazy loading
    "question": Value("string"),
    "answer": Value("string"),
    "answer_type": Value("string"),
    ...
})
```

## Adding New Datasets

1. Create a new dataset class in `med_vqa/data/datasets.py`:

```python
class MyDataset(BaseVQADataset):
    @property
    def name(self) -> str:
        return "MyDataset"
    
    def _load_raw(self) -> Dataset:
        # Load data and return in unified format
        samples = []
        for item in my_data:
            samples.append({
                "image": item["image_path"],  # Path for lazy loading
                "question": item["question"],
                "answer": item["answer"],
                "answer_type": "closed" if item["answer"].lower() in ["yes", "no"] else "open",
                "question_id": f"my_q_{item['id']}",
                "image_id": f"my_img_{item['id']}",
                "dataset_source": self.name,
            })
        
        return Dataset.from_list(samples, features=VQA_FEATURES)
```

2. Register it in the dataset registry:

```python
from med_vqa.configs import DatasetName
from med_vqa.data import register_dataset

# Add to DatasetName enum in configs/config.py
# MY_DATASET = "my_dataset"

register_dataset(DatasetName.MY_DATASET, MyDataset)
```

## Adding New Models

The framework auto-detects model families from model IDs. To add support for a new model family:

1. Add the family to `ModelFamily` enum in `configs/config.py`
2. Add detection logic in `ModelConfig._detect_family()`
3. Add loading logic in `models/loader.py`
4. Add a collator in `data/collators.py`

## Configuration Options

### ModelConfig
- `model_id`: HuggingFace model ID
- `use_4bit`: Enable 4-bit quantization (default: True)
- `use_flash_attention`: Enable Flash Attention 2 (default: True)

### DataConfig
- `dataset_name`: Dataset to use (rad_vqa, slake)
- `data_path`: Path to local dataset (required for SLAKE)
- `question_type`: Filter questions (all, closed, open)
- `subsample_size`: Limit dataset size (None for full)
- `seed`: Random seed for reproducibility

### SFTConfig
- `num_epochs`: Training epochs
- `per_device_batch_size`: Batch size per GPU
- `gradient_accumulation_steps`: Gradient accumulation
- `learning_rate`: Learning rate
- `lora`: LoRA configuration (r, alpha, dropout)

## Future Work

- [ ] GRPO (Group Relative Policy Optimization) training
- [ ] Multi-GPU distributed training
- [ ] Additional datasets (PathVQA, VQA-Med)
- [ ] Additional models (Phi-Vision, LLaVA-OneVision)
- [ ] Uncertainty quantification beyond calibration

## Citation

If you use this framework, please cite:

```bibtex
@software{med_vqa_framework,
  title = {Medical VQA Training Framework},
  year = {2025},
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
# Calibration Evaluation Scripts

Evaluate calibration of Medical VQA models comparing BASE (pretrained) vs SFT (fine-tuned).

## Quick Start

```bash
# Run all evaluations (both sampling and logit methods)
python scripts/evaluate_base_vs_sft.py --gpus 0,1,2,3,4,5 --force

# View results
python scripts/summarize_calibration.py --results_dir ./results/calibration
```

## Scripts Overview

| Script | Purpose |
|--------|---------|
| `evaluate_calibration.py` | Main evaluation (supports both sampling and logit methods) |
| `evaluate_base_vs_sft.py` | Batch runner for all models |
| `summarize_calibration.py` | Aggregates results into comparison tables |
| `debug_model_outputs.py` | Debug raw model outputs |
| `debug_n_differences.py` | Debug sample count inconsistencies |

## Evaluation Methods

### 1. Sampling-based (`--method sampling`)
- Generates 100 samples per question with temperature=0.7
- Computes empirical P(yes) = yes_count / valid_count
- Confidence = max(P(yes), P(no))
- **Pros**: Captures actual generation behavior
- **Cons**: Slow (100 forward passes per question)

### 2. Logit-based (`--method logits`)
- Single forward pass, extracts logits for "yes"/"no" tokens
- Confidence = softmax(logit_yes, logit_no)
- **Pros**: Fast, consistent
- **Cons**: Doesn't capture sampling behavior

### 3. Both (`--method both`, default)
- Runs both methods and saves results separately

## Model Configurations

| Model | HuggingFace ID |
|-------|----------------|
| Qwen3-VL-8B | `Qwen/Qwen3-VL-8B-Instruct` |
| InternVL3-8B | `OpenGVLab/InternVL3-8B-hf` |
| LLaVA-NeXT-7B | `llava-hf/llava-v1.6-mistral-7b-hf` |

## Datasets

| Dataset | Questions (closed) | Source |
|---------|-------------------|--------|
| RAD-VQA | 251 | `flaviagiammarino/vqa-rad` (HuggingFace) |
| SLAKE | 416 | `./data/Slake1.0` (manual download) |

## Prompting Modes

- **BASE models**: Chain-of-Thought prompting (auto-detected)
  ```
  {question}
  
  Think step by step about this medical image, then provide your final answer.
  You must end your response with exactly one of these formats:
  - "The answer is (yes)" if yes
  - "The answer is (no)" if no
  ```

- **SFT models**: Direct prompting (just the question)

## Key Features

### Never Skip Questions
- All questions are evaluated regardless of parse failures
- If all 100 samples are unparseable → random assignment (50% confidence)
- Ensures consistent N across all models for fair comparison

### Statistics Tracked
| Metric | Description |
|--------|-------------|
| `ece` | Expected Calibration Error (lower = better) |
| `mce` | Maximum Calibration Error |
| `overconfidence` | Sum of (conf - acc) when conf > acc |
| `accuracy` | Fraction correct |
| `mean_confidence` | Average model confidence |
| `unknown_rate` | % unparseable responses |
| `random_assignment_rate` | % questions randomly assigned |
| `avg_valid_response_rate` | Average parseable rate per question |

## Usage Examples

### Single Model Evaluation
```bash
# BASE model with both methods
python scripts/evaluate_calibration.py \
    --model_id Qwen/Qwen3-VL-8B-Instruct \
    --dataset rad_vqa \
    --method both \
    --output_dir ./results/calibration/base_qwen_rad_vqa \
    --gpu 0

# SFT model
python scripts/evaluate_calibration.py \
    --model_id Qwen/Qwen3-VL-8B-Instruct \
    --adapter_path ./checkpoints/rad_vqa_qwen3vl_8b_all_lr5e-5_r64_20260127_233346 \
    --dataset rad_vqa \
    --method both \
    --output_dir ./results/calibration/sft_qwen_rad_vqa \
    --gpu 0
```

### Batch Evaluation
```bash
# All models, all datasets
python scripts/evaluate_base_vs_sft.py --gpus 0,1,2,3,4,5

# Specific models
python scripts/evaluate_base_vs_sft.py --models qwen,internvl --gpus 0,1

# Only logits (faster)
python scripts/evaluate_base_vs_sft.py --method logits --gpus 0,1,2,3,4,5

# Force re-run (ignore existing)
python scripts/evaluate_base_vs_sft.py --gpus 0,1,2,3,4,5 --force

# Dry run
python scripts/evaluate_base_vs_sft.py --dry-run
```

### View Results
```bash
# Text table
python scripts/summarize_calibration.py --results_dir ./results/calibration

# Markdown
python scripts/summarize_calibration.py --results_dir ./results/calibration --format markdown

# CSV
python scripts/summarize_calibration.py --results_dir ./results/calibration --format csv --output results.csv

# Detailed comparison
python scripts/summarize_calibration.py --results_dir ./results/calibration --detailed
```

## Output Structure

```
results/calibration/
├── base_qwen_rad_vqa_20260128_123456/
│   ├── config.json           # Evaluation config
│   ├── summary.txt           # Human-readable summary
│   ├── sampling/
│   │   ├── metrics.json      # Calibration metrics
│   │   └── detailed_results.json  # Per-question results
│   └── logits/
│       ├── metrics.json
│       └── detailed_results.json
├── sft_qwen_rad_vqa_20260128_123456/
│   └── ...
└── ...
```

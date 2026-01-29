#!/usr/bin/env python3
"""
Calibration Evaluation with Chain-of-Thought Prompting

For base models that naturally do CoT, this script:
1. Prompts models to reason step-by-step
2. Asks for final answer in structured format: "The answer is (yes/no)"
3. Parses the structured answer

This allows fair comparison between:
- Base models (CoT + structured answer)
- SFT models (direct yes/no)

Usage:
    # Base model with CoT prompting
    python scripts/evaluate_calibration_cot.py \
        --model_id llava-hf/llava-v1.6-mistral-7b-hf \
        --dataset rad_vqa \
        --output_dir ./results/calibration_cot/llava_base \
        --gpu 0

    # SFT model (will use direct prompting automatically)
    python scripts/evaluate_calibration_cot.py \
        --model_id llava-hf/llava-v1.6-mistral-7b-hf \
        --adapter_path ./checkpoints/rad_vqa_llava_next_7b_... \
        --dataset rad_vqa \
        --output_dir ./results/calibration_cot/llava_sft \
        --gpu 0
"""

import argparse
import sys
import os
import json
import re
from dataclasses import dataclass
from typing import Optional, List, Tuple, Any
from tqdm import tqdm

# Parse GPU first
parser_for_gpu = argparse.ArgumentParser(add_help=False)
parser_for_gpu.add_argument("--gpu", type=str, default=None)
args_gpu, _ = parser_for_gpu.parse_known_args()

if args_gpu.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args_gpu.gpu
    print(f"[GPU] Set CUDA_VISIBLE_DEVICES={args_gpu.gpu}")

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from med_vqa.configs import ModelConfig, DataConfig, DatasetName, QuestionType, ModelFamily
from med_vqa.data import get_dataset
from med_vqa.models import load_model
from med_vqa.utils import set_seed


# Prompt templates
COT_PROMPT_TEMPLATE = """{question}

Think step by step about this medical image, then provide your final answer.
You must end your response with exactly one of these formats:
- "The answer is (yes)" if yes
- "The answer is (no)" if no"""

DIRECT_PROMPT_TEMPLATE = """{question}

Answer with only 'yes' or 'no'."""


@dataclass 
class CalibrationResult:
    question: str
    ground_truth: str
    predicted: str
    confidence: float
    p_yes: float
    p_no: float
    is_correct: bool
    yes_count: int
    no_count: int
    unknown_count: int
    sample_responses: List[str]  # Store actual responses for debugging


def parse_structured_answer(text: str) -> Optional[str]:
    """
    Parse answer from structured format.
    
    Looks for patterns like:
    - "The answer is (yes)" / "The answer is (no)"
    - "the answer is yes" / "the answer is no"
    - "Answer: yes" / "Answer: no"
    - Fallback to simple yes/no detection
    """
    text_lower = text.lower()
    
    # Pattern 1: "The answer is (yes/no)"
    match = re.search(r'the answer is \(?(yes|no)\)?', text_lower)
    if match:
        return match.group(1)
    
    # Pattern 2: "Answer: yes/no"
    match = re.search(r'answer:\s*(yes|no)', text_lower)
    if match:
        return match.group(1)
    
    # Pattern 3: "my answer is yes/no"
    match = re.search(r'my answer is\s*(yes|no)', text_lower)
    if match:
        return match.group(1)
    
    # Pattern 4: "I would say yes/no"
    match = re.search(r'i would say\s*(yes|no)', text_lower)
    if match:
        return match.group(1)
    
    # Pattern 5: "Therefore, yes/no"
    match = re.search(r'therefore,?\s*(yes|no)', text_lower)
    if match:
        return match.group(1)
    
    # Pattern 6: Check last line for yes/no
    lines = text.strip().split('\n')
    last_line = lines[-1].lower().strip()
    if last_line in ['yes', 'no', 'yes.', 'no.']:
        return last_line.rstrip('.')
    
    # Fallback: Simple detection (same as before)
    if text_lower.strip() in ["yes", "yes.", "no", "no."]:
        return text_lower.strip().rstrip('.')
    
    if text_lower.startswith("yes"):
        return "yes"
    if text_lower.startswith("no"):
        return "no"
    
    # Last resort: count occurrences
    yes_count = text_lower.count("yes")
    no_count = text_lower.count("no")
    
    # Only use if one is clearly dominant
    if yes_count > 0 and no_count == 0:
        return "yes"
    if no_count > 0 and yes_count == 0:
        return "no"
    
    return None


class CoTCalibrationEvaluator:
    """Evaluator with CoT support for base models."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        adapter_path: Optional[str] = None,
        use_cot: bool = True,
        num_samples: int = 100,
        temperature: float = 0.7,
    ):
        self.model_config = model_config
        self.adapter_path = adapter_path
        self.use_cot = use_cot and (adapter_path is None)  # Only use CoT for base models
        self.num_samples = num_samples
        self.temperature = temperature
        
        self.model = None
        self.processor = None
    
    def load_model(self):
        print("Loading model...")
        self.model, self.processor = load_model(
            self.model_config,
            adapter_path=self.adapter_path,
            prepare_for_training=False,
        )
        self.model.eval()
        
        mode = "CoT prompting" if self.use_cot else "Direct prompting"
        print(f"Mode: {mode}")
        
        if self.adapter_path:
            print(f"Adapter: {self.adapter_path}")
    
    def _format_prompt(self, image: Any, question: str) -> dict:
        """Format prompt based on mode (CoT or direct)."""
        
        if self.use_cot:
            formatted_question = COT_PROMPT_TEMPLATE.format(question=question)
        else:
            formatted_question = question  # SFT models: just the question
        
        if self.model_config.model_family == ModelFamily.QWEN_VL:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": formatted_question},
                    ],
                }
            ]
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt"
            )
            
        elif self.model_config.model_family == ModelFamily.INTERNVL:
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": formatted_question},
                    ],
                },
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(images=image, text=text, return_tensors="pt")
            
        elif self.model_config.model_family in [ModelFamily.LLAVA, ModelFamily.LLAVA_NEXT]:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": formatted_question},
                    ],
                },
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(text=text, images=image, return_tensors="pt")
        else:
            raise ValueError(f"Unsupported: {self.model_config.model_family}")
        
        return inputs
    
    def _generate(self, inputs: dict, max_new_tokens: int = 256) -> str:
        """Generate response."""
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=self.temperature,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
        
        prompt_len = inputs["input_ids"].shape[1]
        generated = outputs[0, prompt_len:]
        return self.processor.tokenizer.decode(generated, skip_special_tokens=True)
    
    def evaluate_single(self, sample: dict) -> CalibrationResult:
        """Evaluate single question with multiple samples."""
        
        inputs = self._format_prompt(sample["image"], sample["question"])
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Use longer generation for CoT
        max_tokens = 256 if self.use_cot else 64
        
        yes_count = 0
        no_count = 0
        unknown_count = 0
        responses = []
        
        for _ in range(self.num_samples):
            response = self._generate(inputs, max_new_tokens=max_tokens)
            responses.append(response)
            
            parsed = parse_structured_answer(response)
            if parsed == "yes":
                yes_count += 1
            elif parsed == "no":
                no_count += 1
            else:
                unknown_count += 1
        
        # Compute confidence
        valid_count = yes_count + no_count
        if valid_count > 0:
            p_yes = yes_count / valid_count
            p_no = no_count / valid_count
        else:
            p_yes = 0.5
            p_no = 0.5
        
        predicted = "yes" if p_yes >= p_no else "no"
        confidence = max(p_yes, p_no)
        
        gt = sample["answer"].lower().strip()
        is_correct = (predicted == gt)
        
        return CalibrationResult(
            question=sample["question"],
            ground_truth=sample["answer"],
            predicted=predicted,
            confidence=confidence,
            p_yes=p_yes,
            p_no=p_no,
            is_correct=is_correct,
            yes_count=yes_count,
            no_count=no_count,
            unknown_count=unknown_count,
            sample_responses=responses[:3],  # Keep first 3 for debugging
        )
    
    def evaluate_dataset(self, dataset, show_progress: bool = True) -> List[CalibrationResult]:
        """Evaluate entire dataset."""
        results = []
        iterator = tqdm(dataset, desc="Evaluating") if show_progress else dataset
        
        for sample in iterator:
            try:
                result = self.evaluate_single(sample)
                results.append(result)
            except Exception as e:
                print(f"Error: {sample['question'][:50]}... - {e}")
        
        return results


def compute_metrics(results: List[CalibrationResult], num_bins: int = 10) -> dict:
    """Compute calibration metrics."""
    
    # Filter out results with all unknown
    valid_results = [r for r in results if (r.yes_count + r.no_count) > 0]
    
    if not valid_results:
        return {"error": "No valid results"}
    
    confidences = np.array([r.confidence for r in valid_results])
    accuracies = np.array([r.is_correct for r in valid_results])
    
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    
    ece = 0.0
    mce = 0.0
    overconfidence = 0.0
    bin_data = []
    
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = in_bin.sum()
        
        if bin_size > 0:
            bin_acc = accuracies[in_bin].mean()
            bin_conf = confidences[in_bin].mean()
            
            gap = abs(bin_acc - bin_conf)
            ece += (bin_size / len(valid_results)) * gap
            mce = max(mce, gap)
            
            if bin_conf > bin_acc:
                overconfidence += (bin_size / len(valid_results)) * (bin_conf - bin_acc)
            
            bin_data.append({
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "bin_size": int(bin_size),
                "accuracy": float(bin_acc),
                "confidence": float(bin_conf),
                "gap": float(gap),
            })
    
    # Compute unknown rate
    total_samples = sum(r.yes_count + r.no_count + r.unknown_count for r in results)
    total_unknown = sum(r.unknown_count for r in results)
    
    return {
        "ece": float(ece),
        "mce": float(mce),
        "overconfidence": float(overconfidence),
        "accuracy": float(accuracies.mean()),
        "mean_confidence": float(confidences.mean()),
        "num_samples": len(valid_results),
        "num_questions_total": len(results),
        "num_questions_valid": len(valid_results),
        "unknown_rate": float(total_unknown / total_samples) if total_samples > 0 else 0,
        "bin_data": bin_data,
    }


def save_results(results: List[CalibrationResult], metrics: dict, output_dir: str):
    """Save results."""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    detailed = []
    for r in results:
        detailed.append({
            "question": r.question,
            "ground_truth": r.ground_truth,
            "predicted": r.predicted,
            "confidence": r.confidence,
            "p_yes": r.p_yes,
            "p_no": r.p_no,
            "is_correct": r.is_correct,
            "yes_count": r.yes_count,
            "no_count": r.no_count,
            "unknown_count": r.unknown_count,
            "sample_responses": r.sample_responses,
        })
    
    with open(os.path.join(output_dir, "detailed_results.json"), "w") as f:
        json.dump(detailed, f, indent=2)
    
    print(f"Results saved to: {output_dir}")


def print_summary(metrics: dict, use_cot: bool):
    """Print summary."""
    print("\n" + "=" * 60)
    print(f"CALIBRATION RESULTS ({'CoT' if use_cot else 'Direct'})")
    print("=" * 60)
    print(f"Questions Total:  {metrics.get('num_questions_total', 'N/A')}")
    print(f"Questions Valid:  {metrics.get('num_questions_valid', 'N/A')}")
    print(f"Unknown Rate:     {metrics.get('unknown_rate', 0):.2%}")
    print(f"Accuracy:         {metrics['accuracy']:.4f}")
    print(f"Mean Confidence:  {metrics['mean_confidence']:.4f}")
    print(f"ECE:              {metrics['ece']:.4f}")
    print(f"MCE:              {metrics['mce']:.4f}")
    print(f"Overconfidence:   {metrics['overconfidence']:.4f}")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="rad_vqa", choices=["rad_vqa", "slake"])
    parser.add_argument("--slake_path", type=str, default="./data/Slake1.0")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_examples", type=int, default=None)
    
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--num_bins", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    
    # Force CoT even for SFT models (for ablation)
    parser.add_argument("--force_cot", action="store_true",
                       help="Use CoT prompting even for SFT models")
    parser.add_argument("--force_direct", action="store_true",
                       help="Use direct prompting even for base models")
    
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default=None)
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Determine prompting mode
    is_sft = args.adapter_path is not None
    if args.force_cot:
        use_cot = True
    elif args.force_direct:
        use_cot = False
    else:
        use_cot = not is_sft  # CoT for base, direct for SFT
    
    print("=" * 60)
    print("Calibration Evaluation with CoT Support")
    print("=" * 60)
    print(f"Model:    {args.model_id}")
    print(f"Adapter:  {args.adapter_path or 'None (BASE)'}")
    print(f"Mode:     {'Chain-of-Thought' if use_cot else 'Direct'}")
    print(f"Dataset:  {args.dataset}")
    print(f"Samples:  {args.num_samples}")
    print("=" * 60)
    
    # Load dataset
    data_config_kwargs = {
        "dataset_name": DatasetName(args.dataset),
        "question_type": QuestionType.CLOSED,
        "split": args.split,
        "subsample_size": args.max_examples,
        "seed": args.seed,
    }
    if args.dataset == "slake":
        data_config_kwargs["data_path"] = args.slake_path
    
    data_config = DataConfig(**data_config_kwargs)
    dataset = get_dataset(data_config).load()
    print(f"\nLoaded {len(dataset)} closed questions")
    
    # Setup evaluator
    model_config = ModelConfig(model_id=args.model_id)
    evaluator = CoTCalibrationEvaluator(
        model_config=model_config,
        adapter_path=args.adapter_path,
        use_cot=use_cot,
        num_samples=args.num_samples,
        temperature=args.temperature,
    )
    evaluator.load_model()
    
    # Evaluate
    print("\nRunning evaluation...")
    results = evaluator.evaluate_dataset(dataset)
    
    # Compute metrics
    metrics = compute_metrics(results, args.num_bins)
    
    # Print and save
    print_summary(metrics, use_cot)
    save_results(results, metrics, args.output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Calibration Evaluation for Medical VQA Models

Supports two evaluation methods:
1. SAMPLING: Generate 100 samples, compute empirical P(yes/no)
2. LOGITS: Extract token probabilities directly from model logits

Features:
- Chain-of-Thought prompting for base models (auto-detected)
- Direct prompting for SFT models  
- NEVER skips questions - random assignment if all samples unparseable
- Tracks parse rates and random assignment statistics

Usage:
    # Sampling-based evaluation (default)
    python scripts/evaluate_calibration.py \
        --model_id Qwen/Qwen3-VL-8B-Instruct \
        --dataset rad_vqa \
        --method sampling \
        --output_dir ./results/calibration/base_qwen_rad_vqa \
        --gpu 0

    # Logit-based evaluation
    python scripts/evaluate_calibration.py \
        --model_id Qwen/Qwen3-VL-8B-Instruct \
        --adapter_path ./checkpoints/... \
        --dataset rad_vqa \
        --method logits \
        --output_dir ./results/calibration/sft_qwen_rad_vqa \
        --gpu 0

    # Both methods
    python scripts/evaluate_calibration.py \
        --model_id ... --method both ...
"""

import argparse
import sys
import os
import json
import re
import random
from dataclasses import dataclass, field
from typing import Optional, List, Any, Dict
from tqdm import tqdm

# Parse GPU first before importing torch
parser_for_gpu = argparse.ArgumentParser(add_help=False)
parser_for_gpu.add_argument("--gpu", type=str, default=None)
args_gpu, _ = parser_for_gpu.parse_known_args()

if args_gpu.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args_gpu.gpu
    print(f"[GPU] Set CUDA_VISIBLE_DEVICES={args_gpu.gpu}")

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from med_vqa.configs import ModelConfig, DataConfig, DatasetName, QuestionType, ModelFamily
from med_vqa.data import get_dataset
from med_vqa.models import load_model
from med_vqa.utils import set_seed


# =============================================================================
# Prompt Templates
# =============================================================================

COT_PROMPT_TEMPLATE = """{question}

Think step by step about this medical image, then provide your final answer.
You must end your response with exactly one of these formats:
- "The answer is (yes)" if yes
- "The answer is (no)" if no"""

DIRECT_PROMPT_TEMPLATE = "{question}"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CalibrationResult:
    """Result for a single question."""
    question: str
    ground_truth: str
    predicted: str
    confidence: float
    p_yes: float
    p_no: float
    is_correct: bool
    # Sampling-specific
    yes_count: int = 0
    no_count: int = 0
    unknown_count: int = 0
    was_random: bool = False
    valid_response_rate: float = 1.0
    # Logit-specific
    logit_yes: float = 0.0
    logit_no: float = 0.0
    # Method used
    method: str = "sampling"


# =============================================================================
# Parsing Functions
# =============================================================================

def parse_yes_no(text: str, use_cot: bool = False) -> Optional[str]:
    """Parse yes/no from model response."""
    text_lower = text.lower().strip()
    
    if use_cot:
        # Pattern 1: "The answer is (yes/no)"
        match = re.search(r'the answer is \(?(yes|no)\)?', text_lower)
        if match:
            return match.group(1)
        
        # Pattern 2: "Answer: yes/no"
        match = re.search(r'answer:\s*(yes|no)', text_lower)
        if match:
            return match.group(1)
        
        # Pattern 3: "Therefore, yes/no"
        match = re.search(r'therefore,?\s*(yes|no)', text_lower)
        if match:
            return match.group(1)
        
        # Pattern 4: Check last line
        lines = text.strip().split('\n')
        last_line = lines[-1].lower().strip()
        if last_line in ['yes', 'no', 'yes.', 'no.']:
            return last_line.rstrip('.')
    
    # Direct / fallback parsing
    if text_lower in ["yes", "yes.", "yes,", "yes!"]:
        return "yes"
    if text_lower in ["no", "no.", "no,", "no!"]:
        return "no"
    
    if text_lower.startswith("yes"):
        return "yes"
    if text_lower.startswith("no"):
        return "no"
    
    # Contains (only if unambiguous)
    has_yes = "yes" in text_lower
    has_no = "no" in text_lower
    
    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"
    
    return None


# =============================================================================
# Token ID Utilities
# =============================================================================

def get_yes_no_token_ids(tokenizer) -> Dict[str, List[int]]:
    """Get token IDs for yes/no variants."""
    yes_variants = ["yes", "Yes", "YES", " yes", " Yes"]
    no_variants = ["no", "No", "NO", " no", " No"]
    
    yes_ids = []
    no_ids = []
    
    for v in yes_variants:
        tokens = tokenizer.encode(v, add_special_tokens=False)
        if tokens:
            yes_ids.append(tokens[0])
    
    for v in no_variants:
        tokens = tokenizer.encode(v, add_special_tokens=False)
        if tokens:
            no_ids.append(tokens[0])
    
    # Remove duplicates
    yes_ids = list(set(yes_ids))
    no_ids = list(set(no_ids))
    
    return {"yes": yes_ids, "no": no_ids}


# =============================================================================
# Evaluator Class
# =============================================================================

class CalibrationEvaluator:
    """
    Unified evaluator supporting both sampling and logit-based methods.
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        adapter_path: Optional[str] = None,
        method: str = "sampling",  # "sampling", "logits", or "both"
        use_cot: Optional[bool] = None,
        num_samples: int = 100,
        temperature: float = 0.7,
        seed: int = 42,
    ):
        self.model_config = model_config
        self.adapter_path = adapter_path
        self.method = method
        self.num_samples = num_samples
        self.temperature = temperature
        self.seed = seed
        
        # Auto-detect prompting mode
        if use_cot is None:
            self.use_cot = (adapter_path is None)  # CoT for base, direct for SFT
        else:
            self.use_cot = use_cot
        
        self.max_new_tokens = 256 if self.use_cot else 64
        
        self.model = None
        self.processor = None
        self.yes_no_ids = None
        self.rng = random.Random(seed)
    
    def load_model(self):
        """Load model and processor."""
        print("Loading model...")
        self.model, self.processor = load_model(
            self.model_config,
            adapter_path=self.adapter_path,
            prepare_for_training=False,
        )
        self.model.eval()
        
        # Get yes/no token IDs for logit method
        self.yes_no_ids = get_yes_no_token_ids(self.processor.tokenizer)
        print(f"Yes token IDs: {self.yes_no_ids['yes']}")
        print(f"No token IDs: {self.yes_no_ids['no']}")
        
        mode = "Chain-of-Thought" if self.use_cot else "Direct"
        print(f"Prompting mode: {mode}")
        print(f"Evaluation method: {self.method}")
    
    def _format_prompt(self, image: Any, question: str) -> dict:
        """Format input prompt."""
        if self.use_cot:
            formatted_question = COT_PROMPT_TEMPLATE.format(question=question)
        else:
            formatted_question = DIRECT_PROMPT_TEMPLATE.format(question=question)
        
        if self.model_config.model_family == ModelFamily.QWEN_VL:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": formatted_question},
            ]}]
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt"
            )
        elif self.model_config.model_family == ModelFamily.INTERNVL:
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": formatted_question},
            ]}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(images=image, text=text, return_tensors="pt")
        elif self.model_config.model_family in [ModelFamily.LLAVA, ModelFamily.LLAVA_NEXT]:
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": formatted_question},
            ]}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(text=text, images=image, return_tensors="pt")
        else:
            raise ValueError(f"Unsupported: {self.model_config.model_family}")
        
        return inputs
    
    def _generate_single(self, inputs: dict) -> str:
        """Generate a single response (for sampling method)."""
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        generated = outputs[0, prompt_len:]
        return self.processor.tokenizer.decode(generated, skip_special_tokens=True)
    
    def _get_logit_probs(self, inputs: dict) -> tuple:
        """Get yes/no probabilities from logits."""
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits
        
        # Get logits for yes/no tokens
        yes_logits = [logits[tid].item() for tid in self.yes_no_ids["yes"]]
        no_logits = [logits[tid].item() for tid in self.yes_no_ids["no"]]
        
        max_yes_logit = max(yes_logits) if yes_logits else float('-inf')
        max_no_logit = max(no_logits) if no_logits else float('-inf')
        
        # Softmax over yes/no only
        logit_tensor = torch.tensor([max_yes_logit, max_no_logit])
        probs = F.softmax(logit_tensor, dim=0)
        
        p_yes = probs[0].item()
        p_no = probs[1].item()
        
        return p_yes, p_no, max_yes_logit, max_no_logit
    
    def evaluate_single_sampling(self, sample: dict) -> CalibrationResult:
        """Evaluate using sampling method."""
        question = sample["question"]
        ground_truth = sample["answer"].lower().strip()
        
        yes_count = 0
        no_count = 0
        unknown_count = 0
        
        try:
            inputs = self._format_prompt(sample["image"], question)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            for _ in range(self.num_samples):
                try:
                    response = self._generate_single(inputs)
                    parsed = parse_yes_no(response, use_cot=self.use_cot)
                    
                    if parsed == "yes":
                        yes_count += 1
                    elif parsed == "no":
                        no_count += 1
                    else:
                        unknown_count += 1
                except Exception:
                    unknown_count += 1
        except Exception as e:
            print(f"  Warning: Failed '{question[:50]}...': {e}")
            unknown_count = self.num_samples
        
        # Compute statistics
        total = yes_count + no_count + unknown_count
        valid_count = yes_count + no_count
        valid_response_rate = valid_count / total if total > 0 else 0
        
        was_random = False
        if valid_count > 0:
            p_yes = yes_count / valid_count
            p_no = no_count / valid_count
            predicted = "yes" if p_yes >= p_no else "no"
            confidence = max(p_yes, p_no)
        else:
            was_random = True
            predicted = "yes" if self.rng.random() < 0.5 else "no"
            p_yes = 0.5
            p_no = 0.5
            confidence = 0.5
        
        is_correct = (predicted == ground_truth)
        
        return CalibrationResult(
            question=question,
            ground_truth=ground_truth,
            predicted=predicted,
            confidence=confidence,
            p_yes=p_yes,
            p_no=p_no,
            is_correct=is_correct,
            yes_count=yes_count,
            no_count=no_count,
            unknown_count=unknown_count,
            was_random=was_random,
            valid_response_rate=valid_response_rate,
            method="sampling",
        )
    
    def evaluate_single_logits(self, sample: dict) -> CalibrationResult:
        """Evaluate using logit method."""
        question = sample["question"]
        ground_truth = sample["answer"].lower().strip()
        
        try:
            inputs = self._format_prompt(sample["image"], question)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            p_yes, p_no, logit_yes, logit_no = self._get_logit_probs(inputs)
            
            predicted = "yes" if p_yes >= p_no else "no"
            confidence = max(p_yes, p_no)
            is_correct = (predicted == ground_truth)
            
            return CalibrationResult(
                question=question,
                ground_truth=ground_truth,
                predicted=predicted,
                confidence=confidence,
                p_yes=p_yes,
                p_no=p_no,
                is_correct=is_correct,
                logit_yes=logit_yes,
                logit_no=logit_no,
                valid_response_rate=1.0,  # Always valid for logits
                method="logits",
            )
        except Exception as e:
            print(f"  Warning: Logit extraction failed '{question[:50]}...': {e}")
            # Fallback to random
            predicted = "yes" if self.rng.random() < 0.5 else "no"
            return CalibrationResult(
                question=question,
                ground_truth=ground_truth,
                predicted=predicted,
                confidence=0.5,
                p_yes=0.5,
                p_no=0.5,
                is_correct=(predicted == ground_truth),
                was_random=True,
                valid_response_rate=0.0,
                method="logits",
            )
    
    def evaluate_dataset(
        self,
        dataset,
        show_progress: bool = True,
    ) -> Dict[str, List[CalibrationResult]]:
        """
        Evaluate dataset with specified method(s).
        
        Returns dict with keys "sampling" and/or "logits".
        """
        results = {}
        
        methods_to_run = []
        if self.method in ["sampling", "both"]:
            methods_to_run.append("sampling")
        if self.method in ["logits", "both"]:
            methods_to_run.append("logits")
        
        for method in methods_to_run:
            print(f"\nRunning {method} evaluation...")
            method_results = []
            
            iterator = tqdm(dataset, desc=f"Evaluating ({method})") if show_progress else dataset
            
            for sample in iterator:
                if method == "sampling":
                    result = self.evaluate_single_sampling(sample)
                else:
                    result = self.evaluate_single_logits(sample)
                method_results.append(result)
            
            assert len(method_results) == len(dataset)
            results[method] = method_results
        
        return results


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_metrics(results: List[CalibrationResult], num_bins: int = 10) -> dict:
    """Compute calibration metrics."""
    confidences = np.array([r.confidence for r in results])
    accuracies = np.array([r.is_correct for r in results])
    
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
            ece += (bin_size / len(results)) * gap
            mce = max(mce, gap)
            
            if bin_conf > bin_acc:
                overconfidence += (bin_size / len(results)) * (bin_conf - bin_acc)
            
            bin_data.append({
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "bin_size": int(bin_size),
                "accuracy": float(bin_acc),
                "confidence": float(bin_conf),
                "gap": float(gap),
            })
    
    # Response statistics
    total_yes = sum(r.yes_count for r in results)
    total_no = sum(r.no_count for r in results)
    total_unknown = sum(r.unknown_count for r in results)
    total_samples = total_yes + total_no + total_unknown
    
    random_count = sum(1 for r in results if r.was_random)
    avg_valid_rate = np.mean([r.valid_response_rate for r in results])
    
    return {
        "ece": float(ece),
        "mce": float(mce),
        "overconfidence": float(overconfidence),
        "accuracy": float(accuracies.mean()),
        "mean_confidence": float(confidences.mean()),
        "num_questions": len(results),
        "random_assignment_count": random_count,
        "random_assignment_rate": float(random_count / len(results)) if results else 0,
        "avg_valid_response_rate": float(avg_valid_rate),
        "total_yes_responses": int(total_yes),
        "total_no_responses": int(total_no),
        "total_unknown_responses": int(total_unknown),
        "unknown_rate": float(total_unknown / total_samples) if total_samples > 0 else 0,
        "bin_data": bin_data,
    }


# =============================================================================
# I/O Functions
# =============================================================================

def save_results(
    results_dict: Dict[str, List[CalibrationResult]],
    output_dir: str,
    config: dict,
    num_bins: int = 10,
):
    """Save results for all methods."""
    os.makedirs(output_dir, exist_ok=True)
    
    all_metrics = {}
    
    for method, results in results_dict.items():
        metrics = compute_metrics(results, num_bins)
        metrics["method"] = method
        all_metrics[method] = metrics
        
        # Save method-specific files
        method_dir = output_dir if len(results_dict) == 1 else os.path.join(output_dir, method)
        os.makedirs(method_dir, exist_ok=True)
        
        with open(os.path.join(method_dir, "metrics.json"), "w") as f:
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
                "was_random": r.was_random,
                "valid_response_rate": r.valid_response_rate,
                "logit_yes": r.logit_yes,
                "logit_no": r.logit_no,
                "method": r.method,
            })
        
        with open(os.path.join(method_dir, "detailed_results.json"), "w") as f:
            json.dump(detailed, f, indent=2)
    
    # Save config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Save summary
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write("Calibration Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model:           {config.get('model_id', 'N/A')}\n")
        f.write(f"Adapter:         {config.get('adapter_path', 'None (BASE)')}\n")
        f.write(f"Dataset:         {config.get('dataset', 'N/A')}\n")
        f.write(f"Prompting:       {'CoT' if config.get('use_cot') else 'Direct'}\n")
        f.write(f"Method(s):       {config.get('method', 'N/A')}\n")
        f.write("\n")
        
        for method, metrics in all_metrics.items():
            f.write(f"--- {method.upper()} ---\n")
            f.write(f"Questions:       {metrics['num_questions']}\n")
            f.write(f"Accuracy:        {metrics['accuracy']:.4f}\n")
            f.write(f"Mean Confidence: {metrics['mean_confidence']:.4f}\n")
            f.write(f"ECE:             {metrics['ece']:.4f}\n")
            f.write(f"MCE:             {metrics['mce']:.4f}\n")
            f.write(f"Overconfidence:  {metrics['overconfidence']:.4f}\n")
            f.write(f"Unknown Rate:    {metrics['unknown_rate']:.2%}\n")
            f.write(f"Random Assigns:  {metrics['random_assignment_count']} ({metrics['random_assignment_rate']:.2%})\n")
            f.write("\n")
    
    print(f"Results saved to: {output_dir}")
    return all_metrics


def print_summary(all_metrics: dict, use_cot: bool):
    """Print summary."""
    for method, metrics in all_metrics.items():
        print(f"\n{'=' * 60}")
        print(f"RESULTS: {method.upper()} ({'CoT' if use_cot else 'Direct'})")
        print("=" * 60)
        print(f"Questions:        {metrics['num_questions']}")
        print(f"Accuracy:         {metrics['accuracy']:.4f}")
        print(f"Mean Confidence:  {metrics['mean_confidence']:.4f}")
        print(f"ECE:              {metrics['ece']:.4f}")
        print(f"MCE:              {metrics['mce']:.4f}")
        print(f"Overconfidence:   {metrics['overconfidence']:.4f}")
        print("-" * 60)
        print(f"Unknown Rate:     {metrics['unknown_rate']:.2%}")
        print(f"Valid Resp Rate:  {metrics['avg_valid_response_rate']:.2%}")
        print(f"Random Assigns:   {metrics['random_assignment_count']} ({metrics['random_assignment_rate']:.2%})")
        print("=" * 60)


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Calibration evaluation")
    
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    
    parser.add_argument("--dataset", type=str, default="rad_vqa",
                       choices=["rad_vqa", "slake"])
    parser.add_argument("--slake_path", type=str, default="./data/Slake1.0")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_examples", type=int, default=None)
    
    parser.add_argument("--method", type=str, default="both",
                       choices=["sampling", "logits", "both"])
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--num_bins", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    
    parser.add_argument("--force_cot", action="store_true")
    parser.add_argument("--force_direct", action="store_true")
    
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default=None)
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    if args.force_cot:
        use_cot = True
    elif args.force_direct:
        use_cot = False
    else:
        use_cot = None
    
    print("=" * 60)
    print("Calibration Evaluation")
    print("=" * 60)
    print(f"Model:      {args.model_id}")
    print(f"Adapter:    {args.adapter_path or 'None (BASE)'}")
    print(f"Dataset:    {args.dataset}")
    print(f"Method:     {args.method}")
    print(f"Samples/Q:  {args.num_samples}")
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
    evaluator = CalibrationEvaluator(
        model_config=model_config,
        adapter_path=args.adapter_path,
        method=args.method,
        use_cot=use_cot,
        num_samples=args.num_samples,
        temperature=args.temperature,
        seed=args.seed,
    )
    evaluator.load_model()
    
    # Evaluate
    results_dict = evaluator.evaluate_dataset(dataset)
    
    # Save
    config = {
        "model_id": args.model_id,
        "adapter_path": args.adapter_path,
        "dataset": args.dataset,
        "split": args.split,
        "method": args.method,
        "num_samples": args.num_samples,
        "num_bins": args.num_bins,
        "temperature": args.temperature,
        "use_cot": evaluator.use_cot,
        "seed": args.seed,
    }
    
    all_metrics = save_results(results_dict, args.output_dir, config, args.num_bins)
    print_summary(all_metrics, evaluator.use_cot)


if __name__ == "__main__":
    main()

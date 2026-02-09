#!/usr/bin/env python3
"""
Calibration Evaluation for Medical VQA Models

Supports three model types:
  - BASE: base model with Chain-of-Thought prompting
  - SFT:  SFT fine-tuned model with direct prompting
  - GRPO: GRPO-trained model with <think>/<answer> structured prompting

Evaluation methods:
  1. SAMPLING: Generate N samples, compute empirical P(yes/no)
  2. LOGITS:   Extract token probabilities directly from model logits

Features:
  - Auto-detects prompting mode from adapter path (base/sft/grpo)
  - GRPO-aware parsing: extracts answer from <answer>...</answer> tags
  - NEVER skips questions — random assignment if all samples unparseable
  - Tracks parse rates and random assignment statistics

Usage:
    # Base model (auto-detects CoT prompting)
    python scripts/evaluate_calibration.py \
        --model_id Qwen/Qwen3-VL-8B-Instruct \
        --dataset rad_vqa \
        --method sampling \
        --output_dir ./results/calibration/base_qwen_rad_vqa \
        --gpu 0

    # SFT model (auto-detects direct prompting)
    python scripts/evaluate_calibration.py \
        --model_id Qwen/Qwen3-VL-8B-Instruct \
        --adapter_path ./checkpoints/rad_vqa_qwen3vl_8b_all_lr5e-5_r64_... \
        --dataset rad_vqa \
        --output_dir ./results/calibration/sft_qwen_rad_vqa \
        --gpu 0

    # GRPO model (auto-detects <think>/<answer> prompting)
    python scripts/evaluate_calibration.py \
        --model_id Qwen/Qwen3-VL-8B-Instruct \
        --adapter_path ./checkpoints/grpo_rad_vqa_qwen3vl_8b_.../final_model \
        --dataset rad_vqa \
        --output_dir ./results/calibration/grpo_qwen_rad_vqa \
        --gpu 0

    # Force a specific prompting mode
    python scripts/evaluate_calibration.py \
        --model_id ... --adapter_path ... --prompt_mode grpo ...

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
from med_vqa.models.two_stage_loader import smart_load_model
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

GRPO_PROMPT_TEMPLATE = """{question}

Analyze the medical image carefully. Provide your reasoning inside <think> tags, then your final answer (yes or no) inside <answer> tags.

<think>
"""


# =============================================================================
# Prompt Mode Detection
# =============================================================================

def detect_prompt_mode(adapter_path: Optional[str]) -> str:
    if adapter_path is None:
        return "cot"
    path_lower = adapter_path.lower()
    if "contrast_sft" in path_lower:
        return "direct"  # contrast_sft uses standard VQA format
    elif "grpo" in path_lower:
        return "grpo"
    else:
        return "direct"

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
    prompt_mode: str = "direct"


# =============================================================================
# Parsing Functions
# =============================================================================

def parse_yes_no(text: str, prompt_mode: str = "direct") -> Optional[str]:
    """Parse yes/no from model response.

    Parsing priority:
      1. <answer>...</answer> tags  (always checked first)
      2. Mode-specific heuristics (CoT patterns, direct match)
      3. Fallback contains-based matching

    Args:
        text: Model response text
        prompt_mode: One of "cot", "direct", "grpo"

    Returns:
        'yes', 'no', or None if unclear
    """
    text_lower = text.lower().strip()

    # -----------------------------------------------------------------
    # Priority 1: <answer> tags (works for GRPO and any model that
    # happens to output them)
    #
    # IMPORTANT: Use findall and take the LAST match. Some models
    # confuse </think> with </answer>, producing patterns like:
    #   <think>reasoning...</answer><answer>yes</answer>
    # The first <answer>...</answer> span contains reasoning text;
    # the last one contains the actual yes/no answer.
    # -----------------------------------------------------------------
    answer_matches = re.findall(
        r"<answer>\s*(.*?)\s*</answer>", text_lower, re.DOTALL
    )
    if answer_matches:
        # Check matches from last to first
        for inner_raw in reversed(answer_matches):
            inner = inner_raw.strip().rstrip(".")
            if inner in ("yes", "no"):
                return inner
            if "yes" in inner and "no" not in inner:
                return "yes"
            if "no" in inner and "yes" not in inner:
                return "no"
    
    # Priority 1b: Incomplete <answer> tag (truncated before </answer>)
    # E.g., "<answer> Yes" at end of response without closing tag
    incomplete_match = re.search(
        r"<answer>\s*(yes|no)\s*$", text_lower
    )
    if incomplete_match:
        return incomplete_match.group(1)

    # -----------------------------------------------------------------
    # Priority 2: Mode-specific parsing
    # -----------------------------------------------------------------
    if prompt_mode == "cot":
        # Pattern: "The answer is (yes/no)"
        match = re.search(r"the answer is \(?(yes|no)\)?", text_lower)
        if match:
            return match.group(1)

        # Pattern: "Answer: yes/no"
        match = re.search(r"answer:\s*(yes|no)", text_lower)
        if match:
            return match.group(1)

        # Pattern: "Therefore, yes/no"
        match = re.search(r"therefore,?\s*(yes|no)", text_lower)
        if match:
            return match.group(1)

        # Last line check
        lines = text.strip().split("\n")
        last_line = lines[-1].lower().strip()
        if last_line in ["yes", "no", "yes.", "no."]:
            return last_line.rstrip(".")

    elif prompt_mode == "grpo":
        # For GRPO, if <answer> tags failed above, try to find answer
        # after </think> tag
        think_end = text_lower.find("</think>")
        if think_end != -1:
            after_think = text_lower[think_end + len("</think>"):].strip()
            if after_think.startswith("yes") or after_think == "yes":
                return "yes"
            if after_think.startswith("no") or after_think == "no":
                return "no"

    # -----------------------------------------------------------------
    # Priority 3: Fallback — exact / starts-with / contains
    # -----------------------------------------------------------------
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
    Unified evaluator supporting sampling and logit-based methods,
    with auto-detection for base / SFT / GRPO prompting.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        adapter_path: Optional[str] = None,
        method: str = "sampling",  # "sampling", "logits", or "both"
        prompt_mode: Optional[str] = None,  # "cot", "direct", "grpo", or None for auto
        num_samples: int = 100,
        samples_per_batch: int = 10,  # num_return_sequences per forward pass
        temperature: float = 0.7,
        seed: int = 42,
    ):
        self.model_config = model_config
        self.adapter_path = adapter_path
        self.method = method
        self.num_samples = num_samples
        self.samples_per_batch = samples_per_batch
        self.temperature = temperature
        self.seed = seed

        # Auto-detect prompting mode
        if prompt_mode is not None:
            self.prompt_mode = prompt_mode
        else:
            self.prompt_mode = detect_prompt_mode(adapter_path)

        # Set max tokens based on mode
        if self.prompt_mode == "cot":
            self.max_new_tokens = 256
        elif self.prompt_mode == "grpo":
            self.max_new_tokens = 384  # <think> reasoning + <answer> tag
        else:
            self.max_new_tokens = 64

        self.model = None
        self.processor = None
        self.yes_no_ids = None
        self.rng = random.Random(seed)

    def load_model(self):
        """Load model and processor."""
        print("Loading model...")
        self.model, self.processor = smart_load_model(
            self.model_config,
            adapter_path=self.adapter_path,
            prepare_for_training=False,
        )
        self.model.eval()

        # Get yes/no token IDs for logit method
        self.yes_no_ids = get_yes_no_token_ids(self.processor.tokenizer)
        print(f"Yes token IDs: {self.yes_no_ids['yes']}")
        print(f"No token IDs: {self.yes_no_ids['no']}")

        mode_labels = {"cot": "Chain-of-Thought (BASE)", "direct": "Direct (SFT)", "grpo": "Structured <think>/<answer> (GRPO)"}
        print(f"Prompting mode: {mode_labels.get(self.prompt_mode, self.prompt_mode)}")
        print(f"Max new tokens: {self.max_new_tokens}")
        print(f"Evaluation method: {self.method}")
        print(f"Samples per question: {self.num_samples}")
        print(f"Samples per batch (num_return_sequences): {self.samples_per_batch}")

    def _format_prompt(self, image: Any, question: str) -> dict:
        """Format input prompt based on prompt_mode."""
        if self.prompt_mode == "cot":
            formatted_question = COT_PROMPT_TEMPLATE.format(question=question)
        elif self.prompt_mode == "grpo":
            formatted_question = GRPO_PROMPT_TEMPLATE.format(question=question)
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

    def _generate_batch(self, inputs: dict, num_sequences: int) -> List[str]:
        """Generate multiple responses in parallel using num_return_sequences.
        
        Args:
            inputs: Tokenized inputs
            num_sequences: Number of sequences to generate in parallel
            
        Returns:
            List of generated response strings
        """
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=num_sequences,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
        
        prompt_len = inputs["input_ids"].shape[1]
        responses = []
        for i in range(num_sequences):
            generated = outputs[i, prompt_len:]
            text = self.processor.tokenizer.decode(generated, skip_special_tokens=True)
            responses.append(text)
        
        return responses

    def _get_logit_probs(self, inputs: dict) -> tuple:
        """Get yes/no probabilities from logits."""
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits

        # Get logits for yes/no tokens
        yes_logits = [logits[tid].item() for tid in self.yes_no_ids["yes"]]
        no_logits = [logits[tid].item() for tid in self.yes_no_ids["no"]]

        max_yes_logit = max(yes_logits) if yes_logits else float("-inf")
        max_no_logit = max(no_logits) if no_logits else float("-inf")

        # Softmax over yes/no only
        logit_tensor = torch.tensor([max_yes_logit, max_no_logit])
        probs = F.softmax(logit_tensor, dim=0)

        p_yes = probs[0].item()
        p_no = probs[1].item()

        return p_yes, p_no, max_yes_logit, max_no_logit

    def evaluate_single_sampling(self, sample: dict) -> CalibrationResult:
        """Evaluate using sampling method with batched generation."""
        question = sample["question"]
        ground_truth = sample["answer"].lower().strip()

        yes_count = 0
        no_count = 0
        unknown_count = 0

        try:
            inputs = self._format_prompt(sample["image"], question)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate samples in batches
            samples_remaining = self.num_samples
            while samples_remaining > 0:
                batch_size = min(self.samples_per_batch, samples_remaining)
                try:
                    responses = self._generate_batch(inputs, batch_size)
                    for response in responses:
                        parsed = parse_yes_no(response, prompt_mode=self.prompt_mode)
                        if parsed == "yes":
                            yes_count += 1
                        elif parsed == "no":
                            no_count += 1
                        else:
                            unknown_count += 1
                except Exception as e:
                    # If batch fails, count all as unknown
                    print(f"  Batch generation failed (batch_size={batch_size}): {type(e).__name__}: {e}")
                    unknown_count += batch_size
                samples_remaining -= batch_size
                
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

        is_correct = predicted == ground_truth

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
            prompt_mode=self.prompt_mode,
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
            is_correct = predicted == ground_truth

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
                prompt_mode=self.prompt_mode,
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
                prompt_mode=self.prompt_mode,
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
            print(f"\nRunning {method} evaluation ({self.prompt_mode} prompting)...")
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
        metrics["prompt_mode"] = config.get("prompt_mode", "unknown")
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
                "prompt_mode": r.prompt_mode,
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
        f.write(f"Prompt Mode:     {config.get('prompt_mode', 'N/A')}\n")
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


def print_summary(all_metrics: dict, prompt_mode: str):
    """Print summary."""
    mode_labels = {"cot": "CoT (BASE)", "direct": "Direct (SFT)", "grpo": "<think>/<answer> (GRPO)"}
    mode_label = mode_labels.get(prompt_mode, prompt_mode)

    for method, metrics in all_metrics.items():
        print(f"\n{'=' * 60}")
        print(f"RESULTS: {method.upper()} — {mode_label}")
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
    parser = argparse.ArgumentParser(description="Calibration evaluation (BASE / SFT / GRPO)")

    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)

    parser.add_argument("--dataset", type=str, default="rad_vqa",
                        choices=["rad_vqa", "slake"])
    parser.add_argument("--slake_path", type=str, default="./data/Slake1.0")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_examples", type=int, default=None)

    parser.add_argument("--method", type=str, default="both",
                        choices=["sampling", "logits", "both"])
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Total samples per question")
    parser.add_argument("--samples_per_batch", type=int, default=10,
                        help="Samples per forward pass (num_return_sequences). "
                             "Higher = better GPU utilization but more VRAM. "
                             "Try 10-25 for H100.")
    parser.add_argument("--num_bins", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)

    # Prompt mode: auto-detect or force
    parser.add_argument("--prompt_mode", type=str, default=None,
                        choices=["cot", "direct", "grpo"],
                        help="Prompting mode (default: auto-detect from adapter path)")
    # Legacy flags (still supported)
    parser.add_argument("--force_cot", action="store_true")
    parser.add_argument("--force_direct", action="store_true")

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Determine prompt mode
    if args.prompt_mode:
        prompt_mode = args.prompt_mode
    elif args.force_cot:
        prompt_mode = "cot"
    elif args.force_direct:
        prompt_mode = "direct"
    else:
        prompt_mode = None  # auto-detect

    mode_display = prompt_mode or detect_prompt_mode(args.adapter_path)

    print("=" * 60)
    print("Calibration Evaluation")
    print("=" * 60)
    print(f"Model:        {args.model_id}")
    print(f"Adapter:      {args.adapter_path or 'None (BASE)'}")
    print(f"Dataset:      {args.dataset}")
    print(f"Prompt Mode:  {mode_display}")
    print(f"Method:       {args.method}")
    print(f"Samples/Q:    {args.num_samples}")
    print(f"Batch size:   {args.samples_per_batch} (num_return_sequences)")
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
        prompt_mode=prompt_mode,
        num_samples=args.num_samples,
        samples_per_batch=args.samples_per_batch,
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
        "prompt_mode": evaluator.prompt_mode,
        "num_samples": args.num_samples,
        "samples_per_batch": args.samples_per_batch,
        "num_bins": args.num_bins,
        "temperature": args.temperature,
        "seed": args.seed,
    }

    all_metrics = save_results(results_dict, args.output_dir, config, args.num_bins)
    print_summary(all_metrics, evaluator.prompt_mode)


if __name__ == "__main__":
    main()
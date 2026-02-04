#!/usr/bin/env python3
"""
Diagnostic: Save raw LLaVA GRPO generations at different batch sizes.

Runs on 100 randomly sampled closed questions from RAD-VQA.
Tests num_return_sequences = 10, 20, 50 (run on separate GPUs).

Saves ALL raw generation text so we can analyze parse failures offline.

Usage:
    # Run all three in parallel
    python scripts/diagnose_llava_grpo.py --num_sequences 10  --gpu 3 &
    python scripts/diagnose_llava_grpo.py --num_sequences 20  --gpu 4 &
    python scripts/diagnose_llava_grpo.py --num_sequences 50  --gpu 5 &
"""

import argparse
import sys
import os
import json
import re
import time
from typing import Optional, List, Any, Dict
from tqdm import tqdm

# Parse GPU first
parser_for_gpu = argparse.ArgumentParser(add_help=False)
parser_for_gpu.add_argument("--gpu", type=str, default=None)
args_gpu, _ = parser_for_gpu.parse_known_args()

if args_gpu.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args_gpu.gpu
    print(f"[GPU] Set CUDA_VISIBLE_DEVICES={args_gpu.gpu}")

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from med_vqa.configs import ModelConfig, DataConfig, DatasetName, QuestionType, ModelFamily
from med_vqa.data import get_dataset
from med_vqa.models import load_model
from med_vqa.utils import set_seed


# Same prompt template as evaluate_calibration.py
GRPO_PROMPT_TEMPLATE = """{question}

Analyze the medical image carefully. Provide your reasoning inside <think> tags, then your final answer (yes or no) inside <answer> tags.

<think>
"""


def parse_yes_no_grpo(text: str) -> Optional[str]:
    """Parse yes/no from GRPO response (same logic as evaluate_calibration.py)."""
    text_lower = text.lower().strip()

    # Last <answer> tag match
    answer_matches = re.findall(
        r"<answer>\s*(.*?)\s*</answer>", text_lower, re.DOTALL
    )
    if answer_matches:
        for inner_raw in reversed(answer_matches):
            inner = inner_raw.strip().rstrip(".")
            if inner in ("yes", "no"):
                return inner
            if "yes" in inner and "no" not in inner:
                return "yes"
            if "no" in inner and "yes" not in inner:
                return "no"

    # Incomplete tag fallback
    incomplete_match = re.search(r"<answer>\s*(yes|no)\s*$", text_lower)
    if incomplete_match:
        return incomplete_match.group(1)

    # Fallback: direct/contains
    if text_lower in ["yes", "yes.", "no", "no."]:
        return text_lower.rstrip(".")
    if text_lower.startswith("yes"):
        return "yes"
    if text_lower.startswith("no"):
        return "no"

    has_yes = "yes" in text_lower
    has_no = "no" in text_lower
    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"

    return None


def format_prompt(processor, model_family, image, question):
    """Format input prompt for any supported model family."""
    formatted_question = GRPO_PROMPT_TEMPLATE.format(question=question)
    
    if model_family == ModelFamily.QWEN_VL:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": formatted_question},
        ]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        )
    elif model_family == ModelFamily.INTERNVL:
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": formatted_question},
        ]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(images=image, text=text, return_tensors="pt")
    elif model_family in [ModelFamily.LLAVA, ModelFamily.LLAVA_NEXT]:
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": formatted_question},
        ]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=text, images=image, return_tensors="pt")
    else:
        raise ValueError(f"Unsupported model family: {model_family}")
    
    return inputs


def generate_batch(model, processor, inputs, num_sequences, max_new_tokens=384, temperature=0.7):
    """Generate multiple responses, return list of strings."""
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_sequences,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    responses = []
    for i in range(outputs.shape[0]):  # Use actual output count, not num_sequences
        generated = outputs[i, prompt_len:]
        text = processor.tokenizer.decode(generated, skip_special_tokens=True)
        responses.append(text)

    return responses


def main():
    parser = argparse.ArgumentParser(description="Diagnose LLaVA GRPO generation")
    parser.add_argument("--num_sequences", type=int, required=True,
                        help="num_return_sequences per call (10, 20, or 50)")
    parser.add_argument("--num_questions", type=int, default=100,
                        help="Number of questions to evaluate")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Total samples per question")
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_id", type=str,
                        default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--adapter_path", type=str,
                        default="./checkpoints/grpo_rad_vqa_llava_next_7b_closed_lr5e-6_r64_20260202_003625/final_model")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    if args.output_dir is None:
        model_short = args.model_id.split("/")[-1].lower().replace("-", "_")
        args.output_dir = f"./results/diagnose/{model_short}_grpo_nrs{args.num_sequences}"

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"GRPO Generation Diagnostic")
    print(f"  model:                {args.model_id}")
    print(f"  adapter:              {args.adapter_path}")
    print(f"  num_return_sequences: {args.num_sequences}")
    print(f"  num_samples/question: {args.num_samples}")
    print(f"  num_questions:        {args.num_questions}")
    print(f"  batches/question:     {args.num_samples // args.num_sequences}")
    print(f"  output:               {args.output_dir}")
    print("=" * 60)

    # Load dataset
    data_config = DataConfig(
        dataset_name=DatasetName.RAD_VQA,
        question_type=QuestionType.CLOSED,
        split="test",
        subsample_size=args.num_questions,
        seed=args.seed,
    )
    dataset = get_dataset(data_config).load()
    print(f"Loaded {len(dataset)} questions")

    # Load model
    model_config = ModelConfig(model_id=args.model_id)
    model, processor = load_model(
        model_config, adapter_path=args.adapter_path, prepare_for_training=False
    )
    model.eval()
    print("Model loaded")

    # Run evaluation
    all_results = []
    batches_per_question = args.num_samples // args.num_sequences
    total_errors = 0

    for qi, sample in enumerate(tqdm(dataset, desc="Questions")):
        question = sample["question"]
        ground_truth = sample["answer"].lower().strip()

        q_result = {
            "question_idx": qi,
            "question": question,
            "ground_truth": ground_truth,
            "raw_generations": [],
            "parsed": [],
            "errors": [],
            "yes_count": 0,
            "no_count": 0,
            "unknown_count": 0,
        }

        try:
            inputs = format_prompt(processor, model_config.model_family, sample["image"], question)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        except Exception as e:
            q_result["errors"].append(f"Prompt formatting failed: {e}")
            q_result["unknown_count"] = args.num_samples
            all_results.append(q_result)
            continue

        for batch_i in range(batches_per_question):
            try:
                responses = generate_batch(
                    model, processor, inputs, args.num_sequences
                )

                for resp in responses:
                    parsed = parse_yes_no_grpo(resp)
                    q_result["raw_generations"].append(resp)
                    q_result["parsed"].append(parsed)

                    if parsed == "yes":
                        q_result["yes_count"] += 1
                    elif parsed == "no":
                        q_result["no_count"] += 1
                    else:
                        q_result["unknown_count"] += 1

            except Exception as e:
                error_msg = f"Batch {batch_i} failed: {type(e).__name__}: {e}"
                q_result["errors"].append(error_msg)
                q_result["unknown_count"] += args.num_sequences
                total_errors += 1
                print(f"  Q{qi} batch {batch_i}: {error_msg}")

        all_results.append(q_result)

    # Compute summary stats
    total_q = len(all_results)
    total_unknown = sum(r["unknown_count"] for r in all_results)
    total_yes = sum(r["yes_count"] for r in all_results)
    total_no = sum(r["no_count"] for r in all_results)
    total_gen = total_yes + total_no + total_unknown
    random_q = sum(1 for r in all_results if r["yes_count"] + r["no_count"] == 0)

    summary = {
        "num_sequences": args.num_sequences,
        "num_samples": args.num_samples,
        "num_questions": total_q,
        "total_generations": total_gen,
        "total_yes": total_yes,
        "total_no": total_no,
        "total_unknown": total_unknown,
        "unknown_rate": total_unknown / total_gen if total_gen > 0 else 0,
        "random_assignment_questions": random_q,
        "random_rate": random_q / total_q if total_q > 0 else 0,
        "batch_errors": total_errors,
    }

    # Save everything
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(args.output_dir, "detailed_with_generations.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Also save a compact version without full generation text (just first 200 chars)
    compact = []
    for r in all_results:
        compact.append({
            "question_idx": r["question_idx"],
            "question": r["question"],
            "ground_truth": r["ground_truth"],
            "yes_count": r["yes_count"],
            "no_count": r["no_count"],
            "unknown_count": r["unknown_count"],
            "errors": r["errors"],
            "sample_generations": [g[:200] for g in r["raw_generations"][:5]],
            "sample_parsed": r["parsed"][:5],
        })
    with open(os.path.join(args.output_dir, "compact_results.json"), "w") as f:
        json.dump(compact, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print(f"RESULTS (num_return_sequences={args.num_sequences})")
    print("=" * 60)
    print(f"Questions:         {total_q}")
    print(f"Total generations: {total_gen}")
    print(f"  Yes:             {total_yes}")
    print(f"  No:              {total_no}")
    print(f"  Unknown:         {total_unknown} ({summary['unknown_rate']:.1%})")
    print(f"Random questions:  {random_q} ({summary['random_rate']:.1%})")
    print(f"Batch errors:      {total_errors}")
    print(f"\nSaved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
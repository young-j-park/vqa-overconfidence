#!/usr/bin/env python3
"""
Question Paraphrase Generation for Medical VQA Data Augmentation

Generates N paraphrases of each question (and optionally answer for open-ended)
using a local LLM via HuggingFace transformers pipeline.

Default model: openai/gpt-oss-120b (MoE, fits on 1 H100 with MXFP4).
Can use any text-generation model.

Outputs cached JSONL files per dataset, one line per original sample:
{
  "question_id": "rad_vqa_q_0",
  "image_id": "rad_vqa_img_0",
  "original_question": "Is there a fracture?",
  "original_answer": "yes",
  "answer_type": "closed",
  "paraphrased_questions": ["Does the image show a fracture?", ...],
  "paraphrased_answers": null  // or [...] for open-ended
}

Usage:
    # Default: gpt-oss-120b, auto device_map (uses all visible GPUs)
    python scripts/generate_paraphrases.py \
        --dataset rad_vqa \
        --num_paraphrases 8 \
        --output_dir ./data/augmented

    # Use specific GPUs
    CUDA_VISIBLE_DEVICES=0,1 python scripts/generate_paraphrases.py \
        --dataset both -n 8

    # Use a different model
    python scripts/generate_paraphrases.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --dataset rad_vqa -n 8

    # Resume (automatically skips already-cached samples)
    python scripts/generate_paraphrases.py --dataset both

    # Check existing cache
    python scripts/generate_paraphrases.py --summary_only
"""

import argparse
import json
import os
import re
import sys
import time
from typing import List, Dict, Optional, Any

from tqdm import tqdm

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from med_vqa.configs import DataConfig, DatasetName, QuestionType
from med_vqa.data import get_dataset


# =============================================================================
# Prompts
# =============================================================================

QUESTION_PARAPHRASE_PROMPT = """\
Paraphrase the following medical question. Keep the exact same meaning and expected answer. Keep medical terminology accurate. Output only the paraphrased question, nothing else.

Original: {question}
Paraphrase:"""

ANSWER_PARAPHRASE_PROMPT = """\
Paraphrase the following medical answer to the given question. Preserve all medical facts, terminology, and specificity. Output only the paraphrased answer, nothing else.

Question: {question}
Original answer: {answer}
Paraphrase:"""


# =============================================================================
# Model Loading
# =============================================================================

def load_model(model_id: str):
    """Load model and tokenizer with device_map='auto'.

    Returns (model, tokenizer) tuple.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {model_id}")
    print(f"  Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({mem:.0f} GB)")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )

    print(f"Model loaded successfully.")
    return model, tokenizer


# =============================================================================
# Generation
# =============================================================================

def parse_single_output(text: str, original_question: str = "") -> Optional[str]:
    """Parse a single paraphrase from model output.

    For standard instruct models (Qwen, Llama, etc.) the output should
    be a clean single sentence. We just do minimal cleanup.

    Returns None if output is empty, truncated, or garbage.
    """
    text = text.strip()
    if not text:
        return None

    # Remove any XML-like tags (just in case)
    text = re.sub(r"<.*?>", "", text).strip()

    # Take only the first non-empty line
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return None
    result = lines[0]

    # Remove numbering prefix: "1. ", "1) ", etc.
    result = re.sub(r"^\d+[\.\)\:\-\s]+\s*", "", result).strip()
    # Remove surrounding quotes
    result = result.strip("\"'")

    # ---- Quality checks ----
    if not result or len(result) < 5:
        return None
    if len(result) > 500:
        return None

    # Reject truncated: must have 4+ words
    if len(result.split()) < 4:
        return None

    return result


def generate_texts_batch(
    model, tokenizer, prompt: str, num_samples: int,
    max_new_tokens: int = 128, temperature: float = 0.9, top_p: float = 0.95,
) -> List[str]:
    """Generate multiple completions for a single prompt using model.generate().

    Batches the same prompt num_samples times for efficient GPU utilization.
    """
    import torch

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    # Tokenize once, repeat for batch
    inputs = tokenizer([text], return_tensors="pt", padding=True).to(model.device)
    input_len = inputs.input_ids.shape[1]

    # Expand to batch
    input_ids = inputs.input_ids.expand(num_samples, -1)
    attention_mask = inputs.attention_mask.expand(num_samples, -1)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

    results = []
    for i in range(num_samples):
        output_ids = outputs[i][input_len:].tolist()
        content = tokenizer.decode(output_ids, skip_special_tokens=True)
        results.append(content)

    return results


def generate_question_paraphrases(
    model, tokenizer, question: str, n: int, buffer: int = 8,
) -> List[str]:
    """Generate N question paraphrases by batched sampling."""
    prompt = QUESTION_PARAPHRASE_PROMPT.format(question=question)
    total_samples = n + buffer

    raws = generate_texts_batch(model, tokenizer, prompt, total_samples,
                                temperature=1.0)

    results = []
    seen = {question.lower().strip()}

    for raw in raws:
        parsed = parse_single_output(raw)
        if parsed and parsed.lower().strip() not in seen:
            results.append(parsed)
            seen.add(parsed.lower().strip())
        if len(results) >= n:
            break

    return results[:n]


def generate_answer_paraphrases(
    model, tokenizer, question: str, answer: str, n: int, buffer: int = 8,
) -> Optional[List[str]]:
    """Generate N answer paraphrases by batched sampling. Skips short answers."""
    if len(answer.split()) <= 3:
        return None

    prompt = ANSWER_PARAPHRASE_PROMPT.format(question=question, answer=answer)
    total_samples = n + buffer

    raws = generate_texts_batch(model, tokenizer, prompt, total_samples,
                                temperature=1.0)

    results = []
    seen = {answer.lower().strip()}

    for raw in raws:
        parsed = parse_single_output(raw)
        if parsed and parsed.lower().strip() not in seen:
            results.append(parsed)
            seen.add(parsed.lower().strip())
        if len(results) >= n:
            break

    return results[:n] if results else None


# =============================================================================
# Dataset Loading
# =============================================================================

def load_dataset_samples(
    dataset_name: str,
    slake_path: Optional[str] = None,
    split: str = "train",
) -> List[Dict[str, Any]]:
    """Load dataset samples into a list of dicts."""
    ds_name = DatasetName.RAD_VQA if dataset_name == "rad_vqa" else DatasetName.SLAKE

    config = DataConfig(
        dataset_name=ds_name,
        question_type=QuestionType.ALL,
        split=split,
        data_path=slake_path,
    )

    dataset_wrapper = get_dataset(config)
    dataset = dataset_wrapper.load()

    samples = []
    for i in range(len(dataset)):
        s = dataset[i]
        samples.append({
            "question_id": s["question_id"],
            "image_id": s["image_id"],
            "question": s["question"],
            "answer": s["answer"],
            "answer_type": s["answer_type"],
        })

    return samples


# =============================================================================
# Cache I/O
# =============================================================================

def get_cache_path(output_dir: str, dataset_name: str, n: int) -> str:
    return os.path.join(output_dir, f"{dataset_name}_paraphrases_n{n}.jsonl")


def load_existing_cache(cache_path: str) -> Dict[str, Dict]:
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    cache[entry["question_id"]] = entry
    return cache


# =============================================================================
# Processing
# =============================================================================

def process_dataset(
    model,
    tokenizer,
    dataset_name: str,
    samples: List[Dict],
    num_paraphrases: int,
    output_dir: str,
    paraphrase_answers: bool = True,
):
    """Process dataset sequentially (GPU-bound)."""
    cache_path = get_cache_path(output_dir, dataset_name, num_paraphrases)
    existing = load_existing_cache(cache_path)
    todo = [s for s in samples if s["question_id"] not in existing]

    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name}")
    print(f"  Total samples:    {len(samples)}")
    print(f"  Already cached:   {len(existing)}")
    print(f"  Remaining:        {len(todo)}")
    print(f"  Paraphrases/Q:    {num_paraphrases}")
    print(f"  Paraphrase A:     {paraphrase_answers}")
    print(f"  Cache:            {cache_path}")
    print(f"{'='*60}\n")

    if not todo:
        print("All samples already cached. Skipping.")
        return

    os.makedirs(output_dir, exist_ok=True)
    cache_file = open(cache_path, "a")

    completed = 0
    failed = 0
    t0 = time.time()

    for sample in tqdm(todo, desc=dataset_name):
        q_paraphrases = generate_question_paraphrases(
            model, tokenizer, sample["question"], num_paraphrases
        )

        a_paraphrases = None
        if paraphrase_answers and sample["answer_type"] == "open":
            a_paraphrases = generate_answer_paraphrases(
                model, tokenizer, sample["question"], sample["answer"], num_paraphrases
            )

        if q_paraphrases:
            result = {
                "question_id": sample["question_id"],
                "image_id": sample["image_id"],
                "original_question": sample["question"],
                "original_answer": sample["answer"],
                "answer_type": sample["answer_type"],
                "paraphrased_questions": q_paraphrases,
                "paraphrased_answers": a_paraphrases,
            }
            cache_file.write(json.dumps(result) + "\n")
            cache_file.flush()
            completed += 1
        else:
            failed += 1

    cache_file.close()
    elapsed = time.time() - t0
    rate = completed / elapsed * 60 if elapsed > 0 else 0

    print(f"\n{dataset_name} done: {completed} OK, {failed} failed")
    print(f"  Time: {elapsed/60:.1f} min ({rate:.1f} samples/min)")
    print(f"  Total cached: {len(existing) + completed}")


# =============================================================================
# Summary
# =============================================================================

def print_cache_summary(output_dir: str, dataset_name: str, n: int):
    cache_path = get_cache_path(output_dir, dataset_name, n)
    if not os.path.exists(cache_path):
        print(f"  No cache for {dataset_name}")
        return

    cache = load_existing_cache(cache_path)
    total = len(cache)
    closed = sum(1 for v in cache.values() if v["answer_type"] == "closed")

    q_lens = [len(v["paraphrased_questions"]) for v in cache.values()]
    a_count = sum(
        1 for v in cache.values()
        if v.get("paraphrased_answers") and len(v["paraphrased_answers"]) > 0
    )

    print(f"\n  {dataset_name} (n={n}):")
    print(f"    Total:          {total}")
    print(f"    Closed / Open:  {closed} / {total - closed}")
    if q_lens:
        print(f"    Avg Q paras:    {sum(q_lens)/len(q_lens):.1f}")
        print(f"    Min/Max Q:      {min(q_lens)}/{max(q_lens)}")
    print(f"    With A paras:   {a_count}")

    for ex in list(cache.values())[:2]:
        print(f"\n    Example ({ex['answer_type']}):")
        print(f"      Q: {ex['original_question']}")
        print(f"      A: {ex['original_answer']}")
        if ex["paraphrased_questions"]:
            print(f"      Q'[0]: {ex['paraphrased_questions'][0]}")
        if ex.get("paraphrased_answers"):
            print(f"      A'[0]: {ex['paraphrased_answers'][0]}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate question/answer paraphrases for Medical VQA"
    )

    parser.add_argument("--dataset", type=str, default="both",
                        choices=["rad_vqa", "slake", "both"])
    parser.add_argument("--slake_path", type=str, default="./data/Slake1.0")
    parser.add_argument("--split", type=str, default="train")

    parser.add_argument("--num_paraphrases", "-n", type=int, default=8)
    parser.add_argument("--paraphrase_answers", action="store_true", default=True)
    parser.add_argument("--no_paraphrase_answers", action="store_false",
                        dest="paraphrase_answers")

    parser.add_argument("--output_dir", type=str, default="./data/augmented")

    parser.add_argument("--model", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507",
                        help="HuggingFace model ID")

    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--summary_only", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    datasets_to_process = []
    if args.dataset in ["rad_vqa", "both"]:
        datasets_to_process.append("rad_vqa")
    if args.dataset in ["slake", "both"]:
        datasets_to_process.append("slake")

    if args.summary_only:
        print("=" * 60)
        print("Paraphrase Cache Summary")
        print("=" * 60)
        for ds in datasets_to_process:
            print_cache_summary(args.output_dir, ds, args.num_paraphrases)
        return

    # Load model once, reuse for all datasets
    model, tokenizer = load_model(args.model)

    for ds_name in datasets_to_process:
        slake_path = args.slake_path if ds_name == "slake" else None
        samples = load_dataset_samples(ds_name, slake_path, args.split)

        if args.max_samples:
            samples = samples[:args.max_samples]

        process_dataset(
            model=model,
            tokenizer=tokenizer,
            dataset_name=ds_name,
            samples=samples,
            num_paraphrases=args.num_paraphrases,
            output_dir=args.output_dir,
            paraphrase_answers=args.paraphrase_answers,
        )

    # Final summary
    print("\n" + "=" * 60)
    print("Final Summary")
    print("=" * 60)
    for ds in datasets_to_process:
        print_cache_summary(args.output_dir, ds, args.num_paraphrases)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Evaluate Calibration for Contrastive SFT Adapted Models

Dispatches calibration evaluation jobs for contrast_sft models
(contrastive GRPO → light SFT adaptation pipeline).

These models require two-stage adapter loading:
  1. Load base model
  2. Load & merge stage-1 (contrastive GRPO) adapter
  3. Load stage-2 (SFT adaptation) adapter

This is handled automatically by smart_load_model() via loading_info.json.

Usage:
    # Evaluate all contrast_sft models across GPUs
    python scripts/evaluate_contrast_sft.py --gpus 0,1,2,3

    # Dry run
    python scripts/evaluate_contrast_sft.py --gpus 0,1,2,3 --dry-run

    # Force re-evaluation
    python scripts/evaluate_contrast_sft.py --gpus 0,1,2,3 --force

    # Custom output directory
    python scripts/evaluate_contrast_sft.py --gpus 0,1,2,3 \
        --output_base ./results/calibration_contrast_sft
"""

import argparse
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional
from threading import Thread, Lock
from queue import Queue

# =============================================================================
# Configuration
# =============================================================================

MODEL_REGISTRY = {
    "internvl3_8b": {
        "model_id": "OpenGVLab/InternVL3-8B-hf",
        "short_name": "InternVL3-8B",
    },
    "qwen3vl_8b": {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "short_name": "Qwen3-VL-8B",
    },
    # LLaVA excluded — contrastive GRPO not yet fully trained
    # "llava_next_7b": {
    #     "model_id": "llava-hf/llava-v1.6-mistral-7b-hf",
    #     "short_name": "LLaVA-NeXT-7B",
    # },
}

# Expected checkpoint directory names (under checkpoint_base)
# Format: contrast_sft_{model_key}_{dataset}
CONTRAST_SFT_RUNS = {
    ("internvl3_8b", "rad_vqa"): "contrast_sft_internvl3_8b_rad_vqa",
    ("internvl3_8b", "slake"):   "contrast_sft_internvl3_8b_slake",
    ("qwen3vl_8b",   "rad_vqa"): "contrast_sft_qwen3vl_8b_rad_vqa",
    ("qwen3vl_8b",   "slake"):   "contrast_sft_qwen3vl_8b_slake",
}

DATASETS = ["rad_vqa", "slake"]


# =============================================================================
# Helpers
# =============================================================================

def find_adapter_path(checkpoint_base: str, run_name: str) -> Optional[str]:
    """Find the adapter path for a contrast_sft run.

    Looks for:
      1. {run_dir}/final_model/  (preferred — has loading_info.json)
      2. {run_dir}/ itself (if it has adapter files + loading_info.json)
    """
    run_dir = os.path.join(checkpoint_base, run_name)

    if not os.path.isdir(run_dir):
        return None

    # Check final_model subdir first
    final_model_dir = os.path.join(run_dir, "final_model")
    if os.path.isdir(final_model_dir):
        loading_info = os.path.join(final_model_dir, "loading_info.json")
        if os.path.isfile(loading_info):
            return final_model_dir

    # Check run_dir itself
    loading_info = os.path.join(run_dir, "loading_info.json")
    if os.path.isfile(loading_info):
        return run_dir

    # Fallback: check if adapter files exist (even without loading_info)
    for fname in ["adapter_model.safetensors", "adapter_model.bin", "adapter_config.json"]:
        if os.path.isfile(os.path.join(final_model_dir, fname)):
            print(f"  WARNING: {final_model_dir} has adapter files but no loading_info.json")
            print(f"           Two-stage loading may not work correctly!")
            return final_model_dir

    return None


def results_exist(output_dir: str) -> bool:
    """Check if evaluation results already exist."""
    if not os.path.isdir(output_dir):
        return False
    for subpath in [
        "metrics.json",
        "sampling/metrics.json",
        "logits/metrics.json",
    ]:
        if os.path.isfile(os.path.join(output_dir, subpath)):
            return True
    return False


def build_command(
    model_id: str,
    adapter_path: str,
    dataset: str,
    output_dir: str,
    slake_path: str,
    method: str,
    num_samples: int,
    samples_per_batch: int,
    num_bins: int,
    max_examples: Optional[int],
    gpu: int,
    seed: int,
) -> List[str]:
    """Build evaluate_calibration.py command."""
    cmd = [
        sys.executable, "scripts/evaluate_calibration.py",
        "--model_id", model_id,
        "--adapter_path", adapter_path,
        "--dataset", dataset,
        "--split", "test",
        "--method", method,
        "--num_samples", str(num_samples),
        "--samples_per_batch", str(samples_per_batch),
        "--num_bins", str(num_bins),
        "--output_dir", output_dir,
        "--seed", str(seed),
        "--gpu", str(gpu),
        # contrast_sft uses standard VQA format → direct prompting
        "--prompt_mode", "direct",
    ]

    if dataset == "slake":
        cmd.extend(["--slake_path", slake_path])

    if max_examples:
        cmd.extend(["--max_examples", str(max_examples)])

    return cmd


print_lock = Lock()


def run_job(job_name: str, cmd: List[str], dry_run: bool) -> Tuple[str, bool, str]:
    """Run a single evaluation job."""
    if dry_run:
        with print_lock:
            print(f"\n[DRY RUN] {job_name}")
            print(f"  {' '.join(cmd)}")
        return job_name, True, "dry run"

    with print_lock:
        print(f"\n[RUNNING] {job_name}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=21600  # 6h timeout
        )
        if result.returncode == 0:
            with print_lock:
                print(f"[SUCCESS] {job_name}")
            return job_name, True, "success"
        else:
            with print_lock:
                print(f"[FAILED] {job_name}")
                stderr_tail = result.stderr[-500:] if result.stderr else "(no stderr)"
                print(f"  stderr: {stderr_tail}")
            return job_name, False, result.stderr or "unknown error"
    except subprocess.TimeoutExpired:
        with print_lock:
            print(f"[TIMEOUT] {job_name}")
        return job_name, False, "timeout"
    except Exception as e:
        with print_lock:
            print(f"[ERROR] {job_name}: {e}")
        return job_name, False, str(e)


def worker(gpu_id: int, job_queue: Queue, results: list, results_lock: Lock,
           dry_run: bool):
    """Worker thread for a specific GPU."""
    while not job_queue.empty():
        try:
            job_name, cmd = job_queue.get_nowait()
        except Exception:
            break

        # Set GPU in command (replace the --gpu arg value)
        try:
            gpu_idx = cmd.index("--gpu")
            cmd[gpu_idx + 1] = str(gpu_id)
        except ValueError:
            cmd.extend(["--gpu", str(gpu_id)])

        result = run_job(job_name, cmd, dry_run)
        with results_lock:
            results.append(result)
        job_queue.task_done()


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate calibration for contrastive SFT adapted models"
    )
    parser.add_argument("--checkpoint_base", type=str, default="./checkpoints",
                        help="Base directory for model checkpoints")
    parser.add_argument("--output_base", type=str,
                        default="./results/calibration",
                        help="Base directory for evaluation results")
    parser.add_argument("--slake_path", type=str, default="./data/Slake1.0")
    parser.add_argument("--gpus", type=str, required=True,
                        help="Comma-separated GPU indices")
    parser.add_argument("--method", type=str, default="logits",
                        choices=["logits", "sampling", "both"],
                        help="Evaluation method (default: logits for SFT)")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Samples per question (for sampling method)")
    parser.add_argument("--samples_per_batch", type=int, default=10)
    parser.add_argument("--num_bins", type=int, default=10)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true",
                        help="Re-evaluate even if results exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model keys to evaluate "
                             "(default: all available)")
    return parser.parse_args()


def main():
    args = parse_args()

    gpus = [int(g.strip()) for g in args.gpus.split(",")]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Filter models if specified
    if args.models:
        model_keys = [m.strip() for m in args.models.split(",")]
    else:
        model_keys = list(MODEL_REGISTRY.keys())

    # Discover and build jobs
    jobs = []
    skipped = []
    missing = []

    gpu_idx = 0
    for model_key in model_keys:
        if model_key not in MODEL_REGISTRY:
            print(f"Warning: Unknown model '{model_key}', skipping")
            continue

        model_info = MODEL_REGISTRY[model_key]
        model_id = model_info["model_id"]
        short_name = model_info["short_name"]

        for dataset in DATASETS:
            run_key = (model_key, dataset)
            if run_key not in CONTRAST_SFT_RUNS:
                continue

            run_name = CONTRAST_SFT_RUNS[run_key]
            adapter_path = find_adapter_path(args.checkpoint_base, run_name)

            if adapter_path is None:
                missing.append((run_name, model_key, dataset))
                continue

            job_name = f"CONTRAST_SFT_{short_name}_{dataset}"
            output_dir = os.path.join(
                args.output_base,
                f"contrast_sft_{model_key}_{dataset}_{timestamp}",
            )

            if not args.force and results_exist(
                os.path.join(args.output_base, f"contrast_sft_{model_key}_{dataset}")
            ):
                # Check without timestamp too
                skipped.append((job_name, f"contrast_sft_{model_key}_{dataset}"))
                continue

            cmd = build_command(
                model_id=model_id,
                adapter_path=adapter_path,
                dataset=dataset,
                output_dir=output_dir,
                slake_path=args.slake_path,
                method=args.method,
                num_samples=args.num_samples,
                samples_per_batch=args.samples_per_batch,
                num_bins=args.num_bins,
                max_examples=args.max_examples,
                gpu=gpus[gpu_idx % len(gpus)],
                seed=args.seed,
            )

            jobs.append((job_name, cmd))
            gpu_idx += 1

    # Print summary
    print("=" * 70)
    print("CONTRASTIVE SFT — CALIBRATION EVALUATION")
    print("=" * 70)
    print(f"Checkpoint Base: {args.checkpoint_base}")
    print(f"Output Base:     {args.output_base}")
    print(f"Method:          {args.method}")
    print(f"GPUs:            {gpus}")
    print(f"Force:           {args.force}")
    print()

    if missing:
        print(f"Missing checkpoints ({len(missing)}):")
        for run_name, mk, ds in missing:
            print(f"  ✗ {run_name} ({mk} / {ds})")
        print()

    if skipped:
        print(f"Skipped (existing results): {len(skipped)}")
        for name, path in skipped:
            print(f"  ⏭ {name}")
        print()

    print(f"Jobs to run: {len(jobs)}")
    for name, cmd in jobs:
        print(f"  → {name}")
    print("=" * 70)

    if not jobs:
        print("\nNo jobs to run!")
        return

    if getattr(args, 'dry_run', False):
        for name, cmd in jobs:
            run_job(name, cmd, dry_run=True)
        return

    # Dispatch jobs across GPUs using thread pool
    job_queue = Queue()
    for job in jobs:
        job_queue.put(job)

    results = []
    results_lock = Lock()

    threads = []
    for gpu_id in gpus:
        t = Thread(
            target=worker,
            args=(gpu_id, job_queue, results, results_lock, False),
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # Print final summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    successes = sum(1 for _, ok, _ in results if ok)
    failures = sum(1 for _, ok, _ in results if not ok)
    print(f"  Succeeded: {successes}")
    print(f"  Failed:    {failures}")
    if failures:
        print("\nFailed jobs:")
        for name, ok, msg in results:
            if not ok:
                print(f"  ✗ {name}: {msg[:100]}")
    print("=" * 70)


if __name__ == "__main__":
    main()

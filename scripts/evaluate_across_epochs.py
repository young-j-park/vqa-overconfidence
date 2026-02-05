#!/usr/bin/env python3
"""
Evaluate Calibration Across Epochs

Discovers all checkpoint-* subdirectories inside each training run,
then dispatches calibration evaluation jobs across multiple GPUs.

Evaluation strategy (auto-selected per training method):
  - SFT / BASE:  logits method (fast, directly reads yes/no token probabilities)
  - GRPO:        sampling method with n=20 (generates <think>...<answer> chains)

For each (model, dataset, training_method, epoch) combination, runs
evaluate_calibration.py with the appropriate adapter_path.

Also evaluates the BASE model (no adapter) and the FINAL checkpoint
(the training run root directory itself, which contains the last saved model).

Usage:
    # Discover and run everything across 8 GPUs (auto method selection)
    python scripts/evaluate_across_epochs.py --gpus 0,1,2,3,4,5,6,7

    # Dry run to see what would be launched
    python scripts/evaluate_across_epochs.py --gpus 0,1,2,3,4,5,6,7 --dry-run

    # Override: force all jobs to use sampling with n=50
    python scripts/evaluate_across_epochs.py --method sampling --num_samples 50 --gpus 0,1,2,3,4,5,6,7

    # Only GRPO checkpoints
    python scripts/evaluate_across_epochs.py --filter grpo --gpus 0,1,2,3,4,5,6,7

    # Only SFT checkpoints
    python scripts/evaluate_across_epochs.py --filter sft --gpus 0,1,2,3,4,5,6,7

    # Specific models only
    python scripts/evaluate_across_epochs.py --models qwen --gpus 0,1,2,3,4,5,6,7

    # Force re-evaluation (skip existing check)
    python scripts/evaluate_across_epochs.py --gpus 0,1,2,3,4,5,6,7 --force

    # View existing results only
    python scripts/evaluate_across_epochs.py --summarize-only
"""

import argparse
import os
import re
import subprocess
import sys
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from queue import Queue
from threading import Lock, Thread


# =============================================================================
# Model Registry
# =============================================================================

MODEL_REGISTRY = {
    "qwen3vl_8b": {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "short_name": "Qwen3-VL-8B",
        "aliases": ["qwen3vl", "qwen"],
    },
    "internvl3_8b": {
        "model_id": "OpenGVLab/InternVL3-8B-hf",
        "short_name": "InternVL3-8B",
        "aliases": ["internvl3", "internvl"],
    },
    "llava_next_7b": {
        "model_id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "short_name": "LLaVA-NeXT-7B",
        "aliases": ["llava_next", "llava"],
    },
}

DATASET_CONFIGS = {
    "rad_vqa": {
        "dataset_arg": "rad_vqa",
        "extra_args": [],
    },
    "slake": {
        "dataset_arg": "slake",
        "extra_args": ["--slake_path", "./data/Slake1.0"],
    },
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EvalJob:
    """A single evaluation job to dispatch."""
    job_name: str
    model_id: str
    model_key: str
    adapter_path: Optional[str]  # None for BASE model
    dataset: str
    training_method: str  # "base", "sft", "grpo"
    checkpoint_label: str  # "base", "epoch-1", "step-100", "final"
    checkpoint_step: Optional[int]  # numeric step for sorting
    output_dir: str
    gpu: Optional[int] = None
    # Per-job eval settings (determined by training_method)
    eval_method: str = "logits"      # "sampling" or "logits"
    eval_num_samples: int = 20       # only used if eval_method == "sampling"

    @property
    def sort_key(self) -> Tuple:
        """Sort key for ordering jobs."""
        method_order = {"base": 0, "sft": 1, "grpo": 2}
        return (
            self.dataset,
            self.model_key,
            method_order.get(self.training_method, 99),
            self.checkpoint_step or 0,
        )


# Per-training-method evaluation defaults
# SFT/BASE: logits (fast, sufficient for direct yes/no models)
# GRPO: sampling (must generate <think>...<answer> chains and parse)
METHOD_DEFAULTS = {
    "base": {"method": "logits", "num_samples": 1},
    "sft":  {"method": "logits", "num_samples": 1},
    "grpo": {"method": "sampling", "num_samples": 20},
}


# =============================================================================
# Discovery Functions
# =============================================================================

def identify_model_key(dirname: str) -> Optional[str]:
    """Identify which model a checkpoint directory belongs to."""
    dirname_lower = dirname.lower()
    for model_key in MODEL_REGISTRY:
        if model_key in dirname_lower:
            return model_key
    return None


def identify_dataset(dirname: str) -> Optional[str]:
    """Identify which dataset a checkpoint directory targets."""
    dirname_lower = dirname.lower()
    if "slake" in dirname_lower:
        return "slake"
    elif "rad_vqa" in dirname_lower:
        return "rad_vqa"
    return None


def identify_training_method(dirname: str) -> str:
    """Identify training method from directory name."""
    if dirname.lower().startswith("grpo_"):
        return "grpo"
    else:
        return "sft"


def discover_checkpoints(checkpoint_dir: str) -> List[Tuple[str, int]]:
    """
    Discover checkpoint subdirectories inside a training run.

    Returns list of (checkpoint_path, step_number) sorted by step.
    """
    checkpoints = []
    run_path = Path(checkpoint_dir)

    if not run_path.exists():
        return checkpoints

    for item in run_path.iterdir():
        if not item.is_dir():
            continue

        # Match checkpoint-NNN pattern
        match = re.match(r"checkpoint-(\d+)", item.name)
        if match:
            step = int(match.group(1))
            # Verify it contains adapter files
            has_adapter = (
                (item / "adapter_model.safetensors").exists()
                or (item / "adapter_model.bin").exists()
                or (item / "adapter_config.json").exists()
                or (item / "model.safetensors").exists()
            )
            if has_adapter:
                checkpoints.append((str(item), step))

    # Sort by step number
    checkpoints.sort(key=lambda x: x[1])
    return checkpoints


def has_final_adapter(run_dir: str) -> bool:
    """Check if the run directory itself contains a final saved adapter."""
    run_path = Path(run_dir)
    return (
        (run_path / "adapter_model.safetensors").exists()
        or (run_path / "adapter_model.bin").exists()
        or (run_path / "adapter_config.json").exists()
    )


def discover_all_jobs(
    checkpoint_base: str,
    output_base: str,
    training_runs: List[str],
    include_base: bool = True,
    force: bool = False,
) -> List[EvalJob]:
    """
    Discover all evaluation jobs from checkpoint directories.

    For each training run:
      - Evaluates each checkpoint-* subdirectory
      - Evaluates the final model (run root if it has adapter files)
    Optionally also adds BASE model evaluations (no adapter).
    """
    jobs = []
    seen_base = set()  # Track (model_key, dataset) to avoid duplicate base evals

    for run_name in training_runs:
        run_dir = os.path.join(checkpoint_base, run_name)

        if not os.path.isdir(run_dir):
            print(f"  Warning: {run_dir} not found, skipping")
            continue

        model_key = identify_model_key(run_name)
        dataset = identify_dataset(run_name)
        training_method = identify_training_method(run_name)

        if model_key is None or dataset is None:
            print(f"  Warning: Cannot identify model/dataset for {run_name}, skipping")
            continue

        model_info = MODEL_REGISTRY[model_key]
        model_id = model_info["model_id"]

        # --- BASE model evaluation (once per model+dataset) ---
        if include_base:
            base_key = (model_key, dataset)
            if base_key not in seen_base:
                seen_base.add(base_key)
                base_output = os.path.join(
                    output_base, f"base_{model_key}_{dataset}"
                )
                base_defaults = METHOD_DEFAULTS["base"]
                if force or not _results_exist(base_output):
                    jobs.append(EvalJob(
                        job_name=f"BASE_{model_info['short_name']}_{dataset}",
                        model_id=model_id,
                        model_key=model_key,
                        adapter_path=None,
                        dataset=dataset,
                        training_method="base",
                        checkpoint_label="base",
                        checkpoint_step=0,
                        output_dir=base_output,
                        eval_method=base_defaults["method"],
                        eval_num_samples=base_defaults["num_samples"],
                    ))

        # --- Per-checkpoint evaluations ---
        checkpoints = discover_checkpoints(run_dir)
        method_defaults = METHOD_DEFAULTS.get(training_method, METHOD_DEFAULTS["sft"])
        print(f"  {run_name}: found {len(checkpoints)} checkpoints "
              f"(eval: {method_defaults['method']}, "
              f"n={method_defaults['num_samples']})")

        for ckpt_path, step in checkpoints:
            label = f"step-{step}"
            output_dir = os.path.join(
                output_base,
                f"{training_method}_{model_key}_{dataset}",
                label,
            )
            if force or not _results_exist(output_dir):
                jobs.append(EvalJob(
                    job_name=f"{training_method.upper()}_{model_info['short_name']}_{dataset}_{label}",
                    model_id=model_id,
                    model_key=model_key,
                    adapter_path=ckpt_path,
                    dataset=dataset,
                    training_method=training_method,
                    checkpoint_label=label,
                    checkpoint_step=step,
                    output_dir=output_dir,
                    eval_method=method_defaults["method"],
                    eval_num_samples=method_defaults["num_samples"],
                ))

        # --- Final model (run root) ---
        if has_final_adapter(run_dir):
            final_output = os.path.join(
                output_base,
                f"{training_method}_{model_key}_{dataset}",
                "final",
            )
            max_step = max((s for _, s in checkpoints), default=0) + 1
            if force or not _results_exist(final_output):
                jobs.append(EvalJob(
                    job_name=f"{training_method.upper()}_{model_info['short_name']}_{dataset}_final",
                    model_id=model_id,
                    model_key=model_key,
                    adapter_path=run_dir,
                    dataset=dataset,
                    training_method=training_method,
                    checkpoint_label="final",
                    checkpoint_step=max_step,
                    output_dir=final_output,
                    eval_method=method_defaults["method"],
                    eval_num_samples=method_defaults["num_samples"],
                ))

    # Sort for deterministic ordering
    jobs.sort(key=lambda j: j.sort_key)
    return jobs


def _results_exist(output_dir: str) -> bool:
    """Check if results already exist in output directory."""
    if not os.path.exists(output_dir):
        return False
    metrics_file = os.path.join(output_dir, "metrics.json")
    sampling_metrics = os.path.join(output_dir, "sampling", "metrics.json")
    logits_metrics = os.path.join(output_dir, "logits", "metrics.json")
    return (
        os.path.exists(metrics_file)
        or os.path.exists(sampling_metrics)
        or os.path.exists(logits_metrics)
    )


# =============================================================================
# Execution
# =============================================================================

def build_eval_command(job: EvalJob, num_bins: int, temperature: float,
                       seed: int, slake_path: str) -> List[str]:
    """Build the evaluate_calibration.py command for a job.

    Method and num_samples come from the job itself (determined by training_method).
    """
    cmd = [
        sys.executable, "scripts/evaluate_calibration.py",
        "--model_id", job.model_id,
        "--dataset", job.dataset,
        "--split", "test",
        "--method", job.eval_method,
        "--num_samples", str(job.eval_num_samples),
        "--num_bins", str(num_bins),
        "--temperature", str(temperature),
        "--output_dir", job.output_dir,
        "--seed", str(seed),
        "--gpu", str(job.gpu),
    ]

    if job.adapter_path:
        cmd.extend(["--adapter_path", job.adapter_path])

    if job.dataset == "slake":
        cmd.extend(["--slake_path", slake_path])

    return cmd


print_lock = Lock()


def run_job(job: EvalJob, num_bins: int,
            temperature: float, seed: int, slake_path: str,
            dry_run: bool) -> Tuple[str, bool, str]:
    """Run a single evaluation job."""
    cmd = build_eval_command(
        job, num_bins, temperature, seed, slake_path
    )

    method_info = f"{job.eval_method}" + (
        f"(n={job.eval_num_samples})" if job.eval_method == "sampling" else ""
    )

    if dry_run:
        with print_lock:
            print(f"\n[DRY RUN] GPU {job.gpu}: {job.job_name}  [{method_info}]")
            print(f"  {' '.join(cmd)}")
        return job.job_name, True, "dry run"

    with print_lock:
        print(f"\n[STARTING] GPU {job.gpu}: {job.job_name}  [{method_info}]")

    # Create log directory
    log_dir = os.path.join(os.path.dirname(job.output_dir), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{job.checkpoint_label}.log")

    try:
        with open(log_file, "w") as lf:
            result = subprocess.run(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                timeout=21600,  # 6h timeout
            )

        if result.returncode == 0:
            with print_lock:
                print(f"[SUCCESS] GPU {job.gpu}: {job.job_name}")
            return job.job_name, True, "success"
        else:
            with print_lock:
                print(f"[FAILED]  GPU {job.gpu}: {job.job_name} (exit code {result.returncode})")
                print(f"  Log: {log_file}")
            return job.job_name, False, f"exit code {result.returncode}"

    except subprocess.TimeoutExpired:
        with print_lock:
            print(f"[TIMEOUT] GPU {job.gpu}: {job.job_name}")
        return job.job_name, False, "timeout"
    except Exception as e:
        with print_lock:
            print(f"[ERROR]   GPU {job.gpu}: {job.job_name}: {e}")
        return job.job_name, False, str(e)


def assign_gpus_round_robin(jobs: List[EvalJob], gpus: List[int]) -> None:
    """Assign GPUs to jobs using round-robin."""
    for i, job in enumerate(jobs):
        job.gpu = gpus[i % len(gpus)]


def assign_gpus_model_affinity(jobs: List[EvalJob], gpus: List[int]) -> None:
    """
    Assign GPUs with model affinity: keep same model on same GPU
    to maximize KV cache / weight reuse between epochs.

    Groups jobs by (model_key, dataset) and assigns each group a GPU.
    """
    groups = {}
    for job in jobs:
        key = (job.model_key, job.training_method, job.dataset)
        if key not in groups:
            groups[key] = []
        groups[key].append(job)

    gpu_idx = 0
    for key in sorted(groups.keys()):
        assigned_gpu = gpus[gpu_idx % len(gpus)]
        for job in groups[key]:
            job.gpu = assigned_gpu
        gpu_idx += 1


# =============================================================================
# Summary / Aggregation
# =============================================================================

def generate_epoch_summary(output_base: str) -> str:
    """Generate a summary table of results across epochs."""
    lines = []
    lines.append("=" * 130)
    lines.append("ACROSS-EPOCH CALIBRATION RESULTS")
    lines.append("=" * 130)

    header = (
        f"{'Method':<6} {'Model':<18} {'Dataset':<10} {'Checkpoint':<14} "
        f"│ {'N':>5} {'Acc':>7} {'ECE':>7} {'MCE':>7} {'OverConf':>8} "
        f"{'Conf':>7} │ {'Unk%':>6} {'Rand%':>6}"
    )
    lines.append(header)
    lines.append("-" * 130)

    # Walk through results directories
    results_data = []

    for root, dirs, files in os.walk(output_base):
        if "metrics.json" in files:
            metrics_path = os.path.join(root, "metrics.json")
            try:
                with open(metrics_path) as f:
                    metrics = json.load(f)

                # Parse path to extract metadata
                rel_path = os.path.relpath(root, output_base)
                parts = rel_path.split(os.sep)

                results_data.append({
                    "path": rel_path,
                    "metrics": metrics,
                })
            except Exception:
                continue

    if not results_data:
        lines.append("  No results found yet.")
        lines.append("=" * 130)
        return "\n".join(lines)

    # Parse and sort
    parsed = []
    for entry in results_data:
        path = entry["path"]
        m = entry["metrics"]

        # Parse: base_qwen3vl_8b_rad_vqa or sft_qwen3vl_8b_rad_vqa/step-100
        parts = path.replace("\\", "/").split("/")

        # Determine checkpoint label
        if len(parts) >= 2:
            ckpt_label = parts[-1]
            run_part = parts[0]
        else:
            ckpt_label = "final"
            run_part = parts[0]

        # If ckpt_label is "sampling" or "logits", go up one level
        if ckpt_label in ("sampling", "logits"):
            if len(parts) >= 3:
                ckpt_label = parts[-2]
                run_part = parts[0]
            else:
                ckpt_label = "final"

        # Parse run_part
        training_method = "base"
        if run_part.startswith("grpo_"):
            training_method = "grpo"
            rest = run_part[5:]
        elif run_part.startswith("sft_"):
            training_method = "sft"
            rest = run_part[4:]
        elif run_part.startswith("base_"):
            training_method = "base"
            rest = run_part[5:]
        else:
            rest = run_part

        # Identify model and dataset
        model_short = "Unknown"
        dataset = "Unknown"
        for mk, info in MODEL_REGISTRY.items():
            if mk in rest:
                model_short = info["short_name"]
                break
        if "slake" in rest:
            dataset = "SLAKE"
        elif "rad_vqa" in rest:
            dataset = "RAD-VQA"

        # Sort key
        step_num = 0
        step_match = re.search(r"step-(\d+)", ckpt_label)
        if step_match:
            step_num = int(step_match.group(1))
        elif ckpt_label == "final":
            step_num = 999999
        elif ckpt_label == "base":
            step_num = -1

        parsed.append({
            "training_method": training_method.upper(),
            "model": model_short,
            "dataset": dataset,
            "checkpoint": ckpt_label,
            "step_num": step_num,
            "metrics": m,
        })

    # Sort
    method_order = {"BASE": 0, "SFT": 1, "GRPO": 2}
    parsed.sort(key=lambda x: (
        x["dataset"],
        x["model"],
        method_order.get(x["training_method"], 99),
        x["step_num"],
    ))

    prev_group = None
    for p in parsed:
        group = (p["dataset"], p["model"], p["training_method"])
        if prev_group is not None and group != prev_group:
            lines.append("-" * 130)
        prev_group = group

        m = p["metrics"]
        n = m.get("num_questions", m.get("num_samples", "?"))
        acc = m.get("accuracy", 0)
        ece = m.get("ece", 0)
        mce = m.get("mce", 0)
        overconf = m.get("overconfidence", 0)
        conf = m.get("mean_confidence", 0)
        unk = m.get("unknown_rate", 0) * 100
        rand = m.get("random_assignment_rate", 0) * 100

        line = (
            f"{p['training_method']:<6} {p['model']:<18} {p['dataset']:<10} "
            f"{p['checkpoint']:<14} │ {n:>5} {acc:>7.4f} {ece:>7.4f} "
            f"{mce:>7.4f} {overconf:>8.4f} {conf:>7.4f} │ "
            f"{unk:>5.1f}% {rand:>5.1f}%"
        )
        lines.append(line)

    lines.append("=" * 130)
    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate calibration across training epochs/checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Paths
    parser.add_argument("--checkpoint_base", type=str, default="./checkpoints",
                        help="Base directory containing training runs")
    parser.add_argument("--output_base", type=str,
                        default="./results/calibration_epochs",
                        help="Base directory for results")
    parser.add_argument("--slake_path", type=str, default="./data/Slake1.0",
                        help="Path to SLAKE dataset")

    # Filtering
    parser.add_argument("--filter", type=str, default=None,
                        choices=["sft", "grpo"],
                        help="Only evaluate SFT or GRPO checkpoints")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model filters (e.g., qwen,internvl)")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset filters (e.g., rad_vqa,slake)")
    parser.add_argument("--no-base", action="store_true",
                        help="Skip BASE model evaluation")

    # Evaluation params
    parser.add_argument("--method", type=str, default="auto",
                        choices=["auto", "sampling", "logits", "both"],
                        help="Evaluation method. 'auto' (default) uses logits "
                             "for SFT/BASE and sampling for GRPO")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Override samples per question for sampling method "
                             "(default: 20 for GRPO, 1 for logits)")
    parser.add_argument("--num_bins", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)

    # Execution
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7",
                        help="Comma-separated GPU IDs")
    parser.add_argument("--max_parallel", type=int, default=None,
                        help="Max parallel jobs (default: number of GPUs)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--force", action="store_true",
                        help="Re-evaluate even if results exist")
    parser.add_argument("--summarize-only", action="store_true",
                        help="Only print summary of existing results")

    return parser.parse_args()


def main():
    args = parse_args()
    gpus = [int(g.strip()) for g in args.gpus.split(",")]
    max_parallel = args.max_parallel or len(gpus)

    # --- Summary only mode ---
    if args.summarize_only:
        print(generate_epoch_summary(args.output_base))
        return

    # --- Discover training runs ---
    print("=" * 70)
    print("ACROSS-EPOCH CALIBRATION EVALUATION")
    print("=" * 70)

    if not os.path.isdir(args.checkpoint_base):
        print(f"Error: Checkpoint directory not found: {args.checkpoint_base}")
        sys.exit(1)

    all_runs = sorted([
        d for d in os.listdir(args.checkpoint_base)
        if os.path.isdir(os.path.join(args.checkpoint_base, d))
    ])

    # Apply filters
    if args.filter:
        if args.filter == "grpo":
            all_runs = [r for r in all_runs if r.startswith("grpo_")]
        elif args.filter == "sft":
            all_runs = [r for r in all_runs if not r.startswith("grpo_")]

    if args.models:
        model_filters = [m.strip().lower() for m in args.models.split(",")]
        filtered_runs = []
        for run in all_runs:
            for mf in model_filters:
                # Check against aliases
                matched = False
                for mk, info in MODEL_REGISTRY.items():
                    if mf in [mk] + info["aliases"]:
                        if mk in run.lower():
                            matched = True
                            break
                if matched:
                    filtered_runs.append(run)
                    break
        all_runs = filtered_runs

    if args.datasets:
        dataset_filters = [d.strip().lower() for d in args.datasets.split(",")]
        all_runs = [
            r for r in all_runs
            if any(df in r.lower() for df in dataset_filters)
        ]

    print(f"\nCheckpoint base:  {args.checkpoint_base}")
    print(f"Output base:      {args.output_base}")
    print(f"GPUs:             {gpus}")
    print(f"Method:           {args.method} "
          f"(SFT/BASE→logits, GRPO→sampling(n=20))"
          if args.method == "auto" else f"Method: {args.method}")
    if args.num_samples is not None:
        print(f"Num samples:      {args.num_samples} (override)")
    print(f"GPU scheduling:   work-stealing (any free GPU grabs next job)")
    print(f"\nDiscovered training runs ({len(all_runs)}):")
    for r in all_runs:
        print(f"  - {r}")

    # --- Discover all jobs ---
    print("\nDiscovering checkpoints...")
    jobs = discover_all_jobs(
        checkpoint_base=args.checkpoint_base,
        output_base=args.output_base,
        training_runs=all_runs,
        include_base=not args.no_base,
        force=args.force,
    )

    if not jobs:
        print("\nNo jobs to run! (All results may already exist. Use --force to re-run)")
        return

    # --- Apply CLI overrides to per-job eval settings ---
    if args.method != "auto":
        for job in jobs:
            job.eval_method = args.method
    if args.num_samples is not None:
        for job in jobs:
            job.eval_num_samples = args.num_samples

    # --- Print job plan ---
    # (GPUs are assigned dynamically at runtime via work-stealing)
    print(f"\nJobs to run: {len(jobs)}")
    print(f"GPU pool:    {gpus} (work-stealing: any free GPU grabs the next job)")
    print("-" * 70)

    # Group by (training_method, model, dataset) for display
    display_groups = {}
    for job in jobs:
        key = (job.training_method, job.model_key, job.dataset)
        if key not in display_groups:
            display_groups[key] = []
        display_groups[key].append(job)

    for key in sorted(display_groups.keys()):
        method, model, dataset = key
        group_jobs = display_groups[key]
        method_info = group_jobs[0].eval_method + (
            f"(n={group_jobs[0].eval_num_samples})"
            if group_jobs[0].eval_method == "sampling" else ""
        )
        ckpts = [j.checkpoint_label for j in group_jobs]
        print(f"  {method.upper():<5} {model:<18} {dataset:<10} "
              f"[{method_info}]  {len(ckpts)} jobs: {', '.join(ckpts)}")

    print("-" * 70)

    # --- Execute ---
    os.makedirs(args.output_base, exist_ok=True)

    # Save job manifest
    manifest_path = os.path.join(args.output_base, "job_manifest.json")
    manifest = [{
        "job_name": j.job_name,
        "model_id": j.model_id,
        "adapter_path": j.adapter_path,
        "dataset": j.dataset,
        "training_method": j.training_method,
        "checkpoint_label": j.checkpoint_label,
        "output_dir": j.output_dir,
        "gpu": "dynamic",  # assigned at runtime via work-stealing
        "eval_method": j.eval_method,
        "eval_num_samples": j.eval_num_samples,
    } for j in jobs]
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Work-stealing execution: shared job queue + GPU pool.
    # Each worker grabs a free GPU, runs one job, returns the GPU,
    # then grabs the next job.  No GPU sits idle while work remains.
    results = []
    results_lock = Lock()

    if args.dry_run:
        for job in jobs:
            run_job(
                job, args.num_bins,
                args.temperature, args.seed, args.slake_path, dry_run=True
            )
        print(f"\n[DRY RUN] {len(jobs)} jobs would be launched across {len(gpus)} GPUs")

        # Print method breakdown
        logit_jobs = sum(1 for j in jobs if j.eval_method == "logits")
        sampling_jobs = sum(1 for j in jobs if j.eval_method == "sampling")
        print(f"  Logits jobs:   {logit_jobs}")
        print(f"  Sampling jobs: {sampling_jobs}")
        return

    job_queue = Queue()
    for job in jobs:
        job_queue.put(job)

    gpu_pool = Queue()
    for gpu_id in gpus:
        gpu_pool.put(gpu_id)

    def steal_worker():
        """Grab a GPU, run one job, return GPU, repeat until queue empty."""
        while True:
            # Try to get a job (non-blocking check first to allow exit)
            try:
                job = job_queue.get(timeout=1)
            except Exception:
                # Queue empty or timeout — check if truly done
                if job_queue.empty():
                    return
                continue

            # Wait for a free GPU (blocks until one is available)
            gpu_id = gpu_pool.get()
            job.gpu = gpu_id

            try:
                result = run_job(
                    job, args.num_bins,
                    args.temperature, args.seed, args.slake_path, dry_run=False
                )
                with results_lock:
                    results.append(result)
            except Exception as e:
                with print_lock:
                    print(f"[ERROR] GPU {gpu_id}: {job.job_name}: {e}")
                with results_lock:
                    results.append((job.job_name, False, str(e)))
            finally:
                # Return GPU to pool so another worker can use it
                gpu_pool.put(gpu_id)
                job_queue.task_done()

    # Launch N worker threads (one per GPU ensures max parallelism)
    worker_threads = []
    for _ in range(max_parallel):
        t = Thread(target=steal_worker, daemon=True)
        t.start()
        worker_threads.append(t)

    # Wait for all jobs to complete
    job_queue.join()

    # Let workers exit
    for t in worker_threads:
        t.join(timeout=5)

    # --- Final summary ---
    print("\n" + "=" * 70)
    print("EXECUTION SUMMARY")
    print("=" * 70)

    successful = sum(1 for _, ok, _ in results if ok)
    failed = [(name, msg) for name, ok, msg in results if not ok]

    print(f"Total:      {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed:     {len(failed)}")

    if failed:
        print("\nFailed jobs:")
        for name, msg in failed:
            print(f"  ✗ {name}: {msg}")

    # Print results table
    print("\n")
    print(generate_epoch_summary(args.output_base))

    print(f"\nResults saved to: {args.output_base}")
    print(f"Re-run summary:   python {sys.argv[0]} --summarize-only "
          f"--output_base {args.output_base}")


if __name__ == "__main__":
    main()
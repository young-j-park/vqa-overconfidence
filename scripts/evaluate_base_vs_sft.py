#!/usr/bin/env python3
"""
Batch Runner for Calibration Evaluation

Runs calibration evaluation for all models (BASE and SFT) across datasets.

Usage:
    # Run all evaluations (both sampling and logits)
    python scripts/evaluate_base_vs_sft.py --gpus 0,1,2,3,4,5

    # Dry run
    python scripts/evaluate_base_vs_sft.py --dry-run

    # Specific models/datasets
    python scripts/evaluate_base_vs_sft.py --models qwen,internvl --datasets rad_vqa

    # Only logits method (faster)
    python scripts/evaluate_base_vs_sft.py --method logits --gpus 0,1,2,3,4,5

    # Force re-run
    python scripts/evaluate_base_vs_sft.py --gpus 0,1,2,3,4,5 --force
"""

import argparse
import os
import subprocess
from datetime import datetime
from typing import List, Tuple, Optional
from pathlib import Path


# =============================================================================
# Model Configurations
# =============================================================================

MODELS = {
    "qwen": {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "short_name": "Qwen3-VL-8B",
        "checkpoints": {
            "rad_vqa": "rad_vqa_qwen3vl_8b_all_lr5e-5_r64_20260127_233346",
            "slake": "slake_qwen3vl_8b_all_lr5e-5_r64_20260127_233346",
        }
    },
    "internvl": {
        "model_id": "OpenGVLab/InternVL3-8B-hf",
        "short_name": "InternVL3-8B",
        "checkpoints": {
            "rad_vqa": "rad_vqa_internvl3_8b_all_lr5e-5_r64_20260127_233346",
            "slake": "slake_internvl3_8b_all_lr5e-5_r64_20260127_233346",
        }
    },
    "llava": {
        "model_id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "short_name": "LLaVA-NeXT-7B",
        "checkpoints": {
            "rad_vqa": "rad_vqa_llava_next_7b_all_lr5e-5_r64_20260127_233346",
            "slake": "slake_llava_next_7b_all_lr5e-5_r64_20260127_233346",
        }
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Batch calibration evaluation")
    
    # Selection
    parser.add_argument("--models", type=str, default="qwen,internvl,llava")
    parser.add_argument("--datasets", type=str, default="rad_vqa,slake")
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--sft-only", action="store_true")
    
    # Paths
    parser.add_argument("--checkpoint_base", type=str, default="./checkpoints")
    parser.add_argument("--slake_path", type=str, default="./data/Slake1.0")
    parser.add_argument("--output_base", type=str, default="./results/calibration")
    
    # Evaluation
    parser.add_argument("--method", type=str, default="both",
                       choices=["sampling", "logits", "both"])
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--num_bins", type=int, default=10)
    parser.add_argument("--max_examples", type=int, default=None)
    
    # Execution
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()


def find_existing_results(output_base: str, prefix: str) -> Optional[str]:
    """Find existing results."""
    if not os.path.exists(output_base):
        return None
    
    for item in os.listdir(output_base):
        if item.startswith(prefix):
            metrics_file = os.path.join(output_base, item, "metrics.json")
            # Also check for method subdirs
            sampling_metrics = os.path.join(output_base, item, "sampling", "metrics.json")
            logits_metrics = os.path.join(output_base, item, "logits", "metrics.json")
            
            if os.path.exists(metrics_file) or os.path.exists(sampling_metrics) or os.path.exists(logits_metrics):
                return os.path.join(output_base, item)
    return None


def build_command(
    model_id: str,
    adapter_path: Optional[str],
    dataset: str,
    output_dir: str,
    slake_path: str,
    method: str,
    num_samples: int,
    num_bins: int,
    max_examples: Optional[int],
    gpu: int,
    seed: int,
) -> List[str]:
    """Build evaluation command."""
    cmd = [
        "python", "scripts/evaluate_calibration.py",
        "--model_id", model_id,
        "--dataset", dataset,
        "--split", "test",
        "--method", method,
        "--num_samples", str(num_samples),
        "--num_bins", str(num_bins),
        "--output_dir", output_dir,
        "--seed", str(seed),
        "--gpu", str(gpu),
    ]
    
    if adapter_path:
        cmd.extend(["--adapter_path", adapter_path])
    
    if dataset == "slake":
        cmd.extend(["--slake_path", slake_path])
    
    if max_examples:
        cmd.extend(["--max_examples", str(max_examples)])
    
    return cmd


def run_job(cmd: List[str], job_name: str, dry_run: bool) -> Tuple[str, bool, str]:
    """Run evaluation job."""
    if dry_run:
        print(f"\n[DRY RUN] {job_name}")
        print(f"  {' '.join(cmd)}")
        return job_name, True, "dry run"
    
    print(f"\n[RUNNING] {job_name}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=21600)  # 6h timeout
        if result.returncode == 0:
            print(f"[SUCCESS] {job_name}")
            return job_name, True, "success"
        else:
            print(f"[FAILED] {job_name}")
            print(f"  stderr: {result.stderr[:500]}")
            return job_name, False, result.stderr
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {job_name}")
        return job_name, False, "timeout"
    except Exception as e:
        print(f"[ERROR] {job_name}: {e}")
        return job_name, False, str(e)


def main():
    args = parse_args()
    
    model_keys = [m.strip() for m in args.models.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")]
    gpus = [int(g.strip()) for g in args.gpus.split(",")]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    skip_existing = not args.force
    
    jobs = []
    skipped = []
    
    gpu_idx = 0
    for model_key in model_keys:
        if model_key not in MODELS:
            print(f"Warning: Unknown model '{model_key}'")
            continue
        
        model_info = MODELS[model_key]
        model_id = model_info["model_id"]
        short_name = model_info["short_name"]
        
        for dataset in datasets:
            # BASE model
            if not args.sft_only:
                job_name = f"BASE_{short_name}_{dataset}"
                prefix = f"base_{model_key}_{dataset}"
                existing = find_existing_results(args.output_base, prefix)
                
                if skip_existing and existing:
                    skipped.append((job_name, existing))
                else:
                    output_dir = os.path.join(args.output_base, f"{prefix}_{timestamp}")
                    cmd = build_command(
                        model_id, None, dataset, output_dir,
                        args.slake_path, args.method, args.num_samples, args.num_bins,
                        args.max_examples, gpus[gpu_idx % len(gpus)], args.seed
                    )
                    jobs.append((job_name, cmd, gpus[gpu_idx % len(gpus)]))
                    gpu_idx += 1
            
            # SFT model
            if not args.base_only:
                checkpoint_name = model_info["checkpoints"].get(dataset)
                if checkpoint_name:
                    adapter_path = os.path.join(args.checkpoint_base, checkpoint_name)
                    job_name = f"SFT_{short_name}_{dataset}"
                    prefix = f"sft_{model_key}_{dataset}"
                    existing = find_existing_results(args.output_base, prefix)
                    
                    if skip_existing and existing:
                        skipped.append((job_name, existing))
                    elif os.path.exists(adapter_path) or args.dry_run:
                        output_dir = os.path.join(args.output_base, f"{prefix}_{timestamp}")
                        cmd = build_command(
                            model_id, adapter_path, dataset, output_dir,
                            args.slake_path, args.method, args.num_samples, args.num_bins,
                            args.max_examples, gpus[gpu_idx % len(gpus)], args.seed
                        )
                        jobs.append((job_name, cmd, gpus[gpu_idx % len(gpus)]))
                        gpu_idx += 1
                    else:
                        print(f"Warning: Checkpoint not found: {adapter_path}")
    
    # Print summary
    print("=" * 70)
    print("CALIBRATION EVALUATION BATCH")
    print("=" * 70)
    print(f"Output:        {args.output_base}")
    print(f"Method:        {args.method}")
    print(f"Num Samples:   {args.num_samples}")
    print(f"GPUs:          {gpus}")
    print(f"Skip Existing: {skip_existing}")
    print()
    
    if skipped:
        print(f"Skipped (existing): {len(skipped)}")
        for name, path in skipped:
            print(f"  ⏭ {name}")
        print()
    
    print(f"Jobs to run: {len(jobs)}")
    for name, _, gpu in jobs:
        print(f"  GPU {gpu}: {name}")
    print("=" * 70)
    
    if not jobs:
        print("\nNo jobs to run!")
        return
    
    os.makedirs(args.output_base, exist_ok=True)
    
    # Run jobs
    results = []
    if args.sequential or args.dry_run:
        for job_name, cmd, gpu in jobs:
            results.append(run_job(cmd, job_name, args.dry_run))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            futures = {executor.submit(run_job, cmd, name, args.dry_run): name 
                      for name, cmd, _ in jobs}
            for future in as_completed(futures):
                results.append(future.result())
    
    # Final summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    successful = sum(1 for r in results if r[1])
    failed = [r for r in results if not r[1]]
    
    print(f"Successful: {successful}/{len(results)}")
    
    if failed:
        print(f"\nFailed jobs:")
        for name, _, error in failed:
            print(f"  ✗ {name}: {error[:100]}")
    
    print("\n" + "=" * 70)
    print(f"View results: python scripts/summarize_calibration.py --results_dir {args.output_base}")
    print("=" * 70)


if __name__ == "__main__":
    main()

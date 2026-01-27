#!/usr/bin/env python3
"""
Batch Experiment Runner for Medical VQA SFT

Run multiple SFT experiments with different configurations.
Supports parallel execution on multiple GPUs.

Usage:
    # Run all experiments
    python run_experiments.py --config experiments.yaml

    # Or use built-in experiment grid
    python run_experiments.py \
        --models "Qwen/Qwen3-VL-2B-Instruct,Qwen/Qwen3-VL-4B-Instruct" \
        --question_types "closed,all" \
        --gpus "0,1,2,3"
"""

import argparse
import subprocess
import os
from queue import Queue
from threading import Thread
from typing import List, Tuple
import itertools


def parse_args():
    parser = argparse.ArgumentParser(description="Run batch SFT experiments")
    
    parser.add_argument("--models", type=str, 
                       default="Qwen/Qwen3-VL-2B-Instruct,Qwen/Qwen3-VL-4B-Instruct,Qwen/Qwen3-VL-8B-Instruct",
                       help="Comma-separated model IDs")
    parser.add_argument("--question_types", type=str, default="closed,all",
                       help="Comma-separated question types")
    parser.add_argument("--datasets", type=str, default="rad_vqa",
                       help="Comma-separated datasets")
    parser.add_argument("--subsample_sizes", type=str, default="",
                       help="Comma-separated subsample sizes (empty for full dataset)")
    parser.add_argument("--gpus", type=str, default="0",
                       help="Comma-separated GPU IDs")
    parser.add_argument("--output_base", type=str, default="./checkpoints",
                       help="Base directory for outputs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true",
                       help="Print commands without executing")
    
    return parser.parse_args()


def get_model_short_name(model_id: str) -> str:
    """Extract short name from model ID."""
    name = model_id.split("/")[-1]
    name = name.replace("-Instruct", "").replace("-hf", "")
    return name


def generate_experiments(args) -> List[Tuple]:
    """Generate all experiment configurations."""
    models = [m.strip() for m in args.models.split(",")]
    question_types = [q.strip() for q in args.question_types.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")]
    
    if args.subsample_sizes:
        subsample_sizes = [int(s.strip()) if s.strip() else None 
                          for s in args.subsample_sizes.split(",")]
    else:
        subsample_sizes = [None]  # Full dataset
    
    experiments = []
    for model, qtype, dataset, subsample in itertools.product(
        models, question_types, datasets, subsample_sizes
    ):
        experiments.append((model, qtype, dataset, subsample))
    
    return experiments


def build_command(
    model_id: str,
    question_type: str,
    dataset: str,
    subsample_size: int,
    output_base: str,
    epochs: int,
    seed: int,
) -> str:
    """Build training command."""
    model_name = get_model_short_name(model_id)
    
    # Build output directory name
    parts = [model_name, dataset, question_type]
    if subsample_size:
        parts.append(f"n{subsample_size}")
    output_dir = os.path.join(output_base, "_".join(parts))
    
    cmd = (
        f"python scripts/train_sft.py "
        f"--model_id {model_id} "
        f"--output_dir {output_dir} "
        f"--dataset {dataset} "
        f"--question_type {question_type} "
        f"--epochs {epochs} "
        f"--seed {seed}"
    )
    
    if subsample_size:
        cmd += f" --subsample_size {subsample_size}"
    
    return cmd, output_dir


def worker(gpu_id: int, queue: Queue, dry_run: bool):
    """Worker thread for running experiments on a specific GPU."""
    print(f"[GPU {gpu_id}] Worker started")
    
    while not queue.empty():
        try:
            cmd, output_dir = queue.get()
            
            print(f"\n[GPU {gpu_id}] Starting: {output_dir}")
            
            if dry_run:
                print(f"[DRY RUN] CUDA_VISIBLE_DEVICES={gpu_id} {cmd}")
            else:
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                subprocess.run(cmd, shell=True, env=env, check=True)
            
            print(f"[GPU {gpu_id}] Finished: {output_dir}")
            
        except subprocess.CalledProcessError as e:
            print(f"[GPU {gpu_id}] Error: {e}")
        except Exception as e:
            print(f"[GPU {gpu_id}] Unexpected error: {e}")
        finally:
            queue.task_done()


def main():
    args = parse_args()
    
    # Generate experiments
    experiments = generate_experiments(args)
    gpus = [int(g.strip()) for g in args.gpus.split(",")]
    
    print("=" * 60)
    print("Medical VQA SFT Experiment Runner")
    print("=" * 60)
    print(f"Total experiments: {len(experiments)}")
    print(f"GPUs: {gpus}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 60)
    
    # Create experiment queue
    experiment_queue = Queue()
    
    for model, qtype, dataset, subsample in experiments:
        cmd, output_dir = build_command(
            model, qtype, dataset, subsample,
            args.output_base, args.epochs, args.seed
        )
        experiment_queue.put((cmd, output_dir))
        print(f"  Queued: {output_dir}")
    
    print("=" * 60)
    
    # Start workers
    threads = []
    for gpu in gpus:
        t = Thread(target=worker, args=(gpu, experiment_queue, args.dry_run))
        t.start()
        threads.append(t)
    
    # Wait for completion
    for t in threads:
        t.join()
    
    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

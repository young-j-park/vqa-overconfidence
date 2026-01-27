#!/usr/bin/env python3
"""
Calibration Evaluation Script for Medical VQA

Evaluate model calibration (ECE, overconfidence) on closed questions.

Usage:
    # Evaluate base model
    python evaluate_calibration.py \
        --model_id Qwen/Qwen3-VL-2B-Instruct \
        --dataset rad_vqa \
        --output_dir ./results/calibration/qwen3-2b-base

    # Evaluate SFT model on specific GPU
    python evaluate_calibration.py \
        --model_id Qwen/Qwen3-VL-2B-Instruct \
        --adapter_path ./checkpoints/qwen3-2b-closed \
        --dataset rad_vqa \
        --gpu 5 \
        --output_dir ./results/calibration/qwen3-2b-sft
"""

import argparse
import sys
import os

# Parse GPU argument FIRST before any torch imports
parser_for_gpu = argparse.ArgumentParser(add_help=False)
parser_for_gpu.add_argument("--gpu", type=str, default=None, help="GPU index")
args_gpu, _ = parser_for_gpu.parse_known_args()

if args_gpu.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args_gpu.gpu
    print(f"[GPU] Set CUDA_VISIBLE_DEVICES={args_gpu.gpu}")

# Now import torch-dependent modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from med_vqa.configs import (
    ModelConfig, DataConfig, InferenceConfig,
    DatasetName, QuestionType,
)
from med_vqa.data import get_dataset
from med_vqa.inference import VQAInference
from med_vqa.evaluation import CalibrationEvaluator
from med_vqa.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VQA model calibration")
    
    # Model arguments
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="rad_vqa",
                       choices=["rad_vqa", "slake"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_examples", type=int, default=None)
    
    # Evaluation arguments
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Samples per question for empirical probability")
    parser.add_argument("--num_bins", type=int, default=10,
                       help="Number of bins for ECE")
    
    # Output
    parser.add_argument("--output_dir", type=str, required=True)
    
    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default=None,
                       help="GPU index (e.g., 0, 5)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    print("=" * 60)
    print("Medical VQA Calibration Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_id}")
    print(f"Adapter: {args.adapter_path or 'None (base model)'}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples per question: {args.num_samples}")
    print("=" * 60)
    
    # Setup configs
    model_config = ModelConfig(model_id=args.model_id)
    
    data_config = DataConfig(
        dataset_name=DatasetName(args.dataset),
        question_type=QuestionType.CLOSED,  # Calibration on closed questions
        split=args.split,
        subsample_size=args.max_examples,
        seed=args.seed,
    )
    
    inference_config = InferenceConfig(
        adapter_path=args.adapter_path,
        num_samples=args.num_samples,
        temperature=0.7,
        do_sample=True,
    )
    
    # Load dataset
    print("\nLoading dataset...")
    dataset_wrapper = get_dataset(data_config)
    dataset = dataset_wrapper.load()
    print(f"Evaluating on {len(dataset)} closed questions")
    
    # Setup inference
    print("\nLoading model...")
    inference = VQAInference(model_config, inference_config)
    inference.load_model()
    
    # Run inference
    print("\nRunning inference...")
    predictions = inference.predict_dataset(
        dataset,
        num_samples=args.num_samples,
    )
    
    # Evaluate calibration
    print("\nComputing calibration metrics...")
    evaluator = CalibrationEvaluator(num_bins=args.num_bins)
    evaluator.add_predictions(predictions)
    
    # Print and save results
    evaluator.print_summary()
    evaluator.save_results(args.output_dir)
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

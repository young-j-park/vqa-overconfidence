#!/usr/bin/env python3
"""
Inference Script for Medical VQA

Run inference on medical VQA datasets using base or SFT-finetuned models.

Usage Examples:
    # Base model inference
    python run_inference.py \
        --model_id Qwen/Qwen3-VL-2B-Instruct \
        --dataset rad_vqa \
        --split test \
        --output_path ./results/base_predictions.json

    # SFT model inference on specific GPU
    python run_inference.py \
        --model_id Qwen/Qwen3-VL-2B-Instruct \
        --adapter_path ./checkpoints/qwen3-2b-closed \
        --dataset rad_vqa \
        --gpu 5 \
        --output_path ./results/sft_predictions.json

    # Sampling for calibration
    python run_inference.py \
        --model_id Qwen/Qwen3-VL-2B-Instruct \
        --dataset rad_vqa \
        --question_type closed \
        --num_samples 100 \
        --output_path ./results/calibration_samples.json
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
from med_vqa.inference import run_inference
from med_vqa.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on Medical VQA")
    
    # Model arguments
    parser.add_argument("--model_id", type=str, required=True,
                       help="HuggingFace model ID")
    parser.add_argument("--adapter_path", type=str, default=None,
                       help="Path to LoRA adapter (None for base model)")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="rad_vqa",
                       choices=["rad_vqa", "slake"])
    parser.add_argument("--question_type", type=str, default="all",
                       choices=["all", "closed", "open"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_examples", type=int, default=None,
                       help="Limit number of examples")
    
    # Inference arguments
    parser.add_argument("--num_samples", type=int, default=1,
                       help="Number of samples per question (for calibration)")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    
    # Output
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to save predictions JSON")
    
    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_4bit", action="store_true", default=True)
    parser.add_argument("--gpu", type=str, default=None,
                       help="GPU index (e.g., 0, 5)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    model_config = ModelConfig(
        model_id=args.model_id,
        use_4bit=args.use_4bit,
    )
    
    data_config = DataConfig(
        dataset_name=DatasetName(args.dataset),
        question_type=QuestionType(args.question_type),
        split=args.split,
        subsample_size=args.max_examples,
        seed=args.seed,
    )
    
    inference_config = InferenceConfig(
        adapter_path=args.adapter_path,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    
    results = run_inference(
        model_config=model_config,
        data_config=data_config,
        inference_config=inference_config,
        output_path=args.output_path,
    )
    
    print(f"\nInference complete! {len(results)} predictions saved to {args.output_path}")


if __name__ == "__main__":
    main()

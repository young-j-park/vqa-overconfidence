#!/usr/bin/env python3
"""
Debug Script: Inspect Raw Model Outputs

Checks why some models have very low N (few parseable yes/no responses).
Prints raw model outputs to see what the model is actually generating.

Usage:
    python scripts/debug_model_outputs.py \
        --model_id llava-hf/llava-v1.6-mistral-7b-hf \
        --adapter_path ./checkpoints/rad_vqa_llava_next_7b_... \
        --dataset rad_vqa \
        --num_examples 10 \
        --gpu 0
"""

import argparse
import sys
import os

# Parse GPU first
parser_for_gpu = argparse.ArgumentParser(add_help=False)
parser_for_gpu.add_argument("--gpu", type=str, default=None)
args_gpu, _ = parser_for_gpu.parse_known_args()

if args_gpu.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args_gpu.gpu
    print(f"[GPU] Set CUDA_VISIBLE_DEVICES={args_gpu.gpu}")

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from med_vqa.configs import ModelConfig, DataConfig, DatasetName, QuestionType
from med_vqa.data import get_dataset
from med_vqa.models import load_model
from med_vqa.utils import set_seed


def parse_yes_no(text: str):
    """Parse yes/no from response."""
    text_lower = text.lower().strip()
    
    if text_lower in ["yes", "yes.", "yes,", "yes!"]:
        return "yes"
    if text_lower in ["no", "no.", "no,", "no!"]:
        return "no"
    if text_lower.startswith("yes"):
        return "yes"
    if text_lower.startswith("no"):
        return "no"
    if "yes" in text_lower and "no" not in text_lower:
        return "yes"
    if "no" in text_lower and "yes" not in text_lower:
        return "no"
    return None


def format_prompt(processor, model_config, image, question):
    """Format input for the model."""
    from med_vqa.configs import ModelFamily
    
    if model_config.model_family == ModelFamily.QWEN_VL:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        )
        
    elif model_config.model_family == ModelFamily.INTERNVL:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(images=image, text=text, return_tensors="pt")
        
    elif model_config.model_family in [ModelFamily.LLAVA, ModelFamily.LLAVA_NEXT]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=text, images=image, return_tensors="pt")
    else:
        raise ValueError(f"Unsupported: {model_config.model_family}")
    
    return inputs


def generate_response(model, processor, inputs, do_sample=True, temperature=0.7, max_new_tokens=64):
    """Generate a response."""
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            max_new_tokens=max_new_tokens,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0, prompt_len:]
    response = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response


def parse_args():
    parser = argparse.ArgumentParser(description="Debug model outputs")
    
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="rad_vqa",
                       choices=["rad_vqa", "slake"])
    parser.add_argument("--slake_path", type=str, default="./data/Slake1.0")
    parser.add_argument("--num_examples", type=int, default=10,
                       help="Number of examples to inspect")
    parser.add_argument("--num_samples", type=int, default=5,
                       help="Samples per question to show variation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default=None)
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    print("=" * 80)
    print("DEBUG: Model Output Inspection")
    print("=" * 80)
    print(f"Model:   {args.model_id}")
    print(f"Adapter: {args.adapter_path or 'None (BASE)'}")
    print(f"Dataset: {args.dataset}")
    print("=" * 80)
    
    # Load model
    print("\nLoading model...")
    model_config = ModelConfig(model_id=args.model_id)
    model, processor = load_model(
        model_config,
        adapter_path=args.adapter_path,
        prepare_for_training=False,
    )
    model.eval()
    print(f"Model family: {model_config.model_family}")
    
    # Load dataset
    print("\nLoading dataset...")
    data_config_kwargs = {
        "dataset_name": DatasetName(args.dataset),
        "question_type": QuestionType.CLOSED,
        "split": "test",
        "subsample_size": args.num_examples,
        "seed": args.seed,
    }
    if args.dataset == "slake":
        data_config_kwargs["data_path"] = args.slake_path
    
    data_config = DataConfig(**data_config_kwargs)
    dataset = get_dataset(data_config).load()
    print(f"Loaded {len(dataset)} examples")
    
    # Statistics
    total_yes = 0
    total_no = 0
    total_unknown = 0
    
    print("\n" + "=" * 80)
    print("SAMPLE OUTPUTS")
    print("=" * 80)
    
    for i, sample in enumerate(dataset):
        print(f"\n{'─' * 80}")
        print(f"Example {i+1}/{len(dataset)}")
        print(f"{'─' * 80}")
        print(f"Question:     {sample['question']}")
        print(f"Ground Truth: {sample['answer']}")
        print()
        
        # Format input
        inputs = format_prompt(processor, model_config, sample["image"], sample["question"])
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate multiple samples
        print("Generated responses:")
        sample_yes = 0
        sample_no = 0
        sample_unknown = 0
        
        for j in range(args.num_samples):
            # With sampling
            response_sampled = generate_response(
                model, processor, inputs,
                do_sample=True, temperature=0.7, max_new_tokens=64
            )
            parsed = parse_yes_no(response_sampled)
            
            if parsed == "yes":
                sample_yes += 1
                total_yes += 1
                status = "✓ yes"
            elif parsed == "no":
                sample_no += 1
                total_no += 1
                status = "✓ no"
            else:
                sample_unknown += 1
                total_unknown += 1
                status = "✗ UNKNOWN"
            
            # Truncate long responses for display
            display_response = response_sampled[:100]
            if len(response_sampled) > 100:
                display_response += "..."
            
            print(f"  [{j+1}] {status:12} | \"{display_response}\"")
        
        # Also show greedy decoding
        response_greedy = generate_response(
            model, processor, inputs,
            do_sample=False, max_new_tokens=64
        )
        parsed_greedy = parse_yes_no(response_greedy)
        greedy_status = parsed_greedy if parsed_greedy else "UNKNOWN"
        
        display_greedy = response_greedy[:100]
        if len(response_greedy) > 100:
            display_greedy += "..."
        
        print(f"\n  [Greedy]     {greedy_status:12} | \"{display_greedy}\"")
        print(f"\n  Summary: yes={sample_yes}, no={sample_no}, unknown={sample_unknown}")
    
    # Final statistics
    total = total_yes + total_no + total_unknown
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print(f"Total responses:   {total}")
    print(f"Parsed as 'yes':   {total_yes} ({100*total_yes/total:.1f}%)")
    print(f"Parsed as 'no':    {total_no} ({100*total_no/total:.1f}%)")
    print(f"UNPARSEABLE:       {total_unknown} ({100*total_unknown/total:.1f}%)")
    print()
    
    if total_unknown > total * 0.3:
        print("⚠️  WARNING: High rate of unparseable responses!")
        print("   This explains the low N in calibration results.")
        print("   The model is not outputting clean yes/no answers.")
    
    print("=" * 80)


if __name__ == "__main__":
    main()

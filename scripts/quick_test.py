#!/usr/bin/env python3
"""
Quick End-to-End Test Script

Tests the entire pipeline with minimal data (5-10 samples) to verify everything works:
1. Base model inference
2. SFT training (1 epoch, 5 samples)
3. SFT model inference
4. Calibration evaluation

Usage:
    # Test single model (default: Qwen)
    python scripts/quick_test.py --gpu 0

    # Test specific model
    python scripts/quick_test.py --model_id Qwen/Qwen2-VL-2B-Instruct --gpu 0

    # Test ALL supported model families (Qwen, InternVL, LLaVA)
    python scripts/quick_test.py --test_all_models --gpu 0

    # Test specific model families
    python scripts/quick_test.py --model_families qwen,internvl --gpu 0

    # Skip training (only test inference)
    python scripts/quick_test.py --skip_training --gpu 0

    # Keep outputs for inspection
    python scripts/quick_test.py --keep_outputs --output_dir ./test_outputs
"""

import argparse
import sys
import os
import shutil
import tempfile

# Parse GPU argument FIRST before any torch imports
parser_for_gpu = argparse.ArgumentParser(add_help=False)
parser_for_gpu.add_argument("--gpu", type=str, default="0", help="GPU index")
args_gpu, _ = parser_for_gpu.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args_gpu.gpu
print(f"[Setup] CUDA_VISIBLE_DEVICES={args_gpu.gpu}")

# Now import everything else
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from med_vqa.configs import (
    ModelConfig, DataConfig, SFTConfig, LoRAConfig,
    InferenceConfig, ExperimentConfig, DatasetName, QuestionType, ModelFamily
)
from med_vqa.data import get_dataset
from med_vqa.models import load_model
from med_vqa.training import VQASFTTrainer
from med_vqa.inference import VQAInference, VQAPrediction
from med_vqa.evaluation import CalibrationEvaluator
from med_vqa.utils import set_seed


# Default models for each family (smallest versions for quick testing)
DEFAULT_MODELS = {
    "qwen": "Qwen/Qwen2-VL-2B-Instruct",
    "internvl": "OpenGVLab/InternVL3-1B-hf",  # InternVL3 (HF version)
    "llava": "llava-hf/llava-1.5-7b-hf",
    "llava_next": "llava-hf/llava-v1.6-mistral-7b-hf",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Quick E2E Test")
    
    # Model selection
    parser.add_argument("--model_id", type=str, default=None,
                       help="Specific model to test (overrides --model_families)")
    parser.add_argument("--model_families", type=str, default="qwen",
                       help="Comma-separated model families to test: qwen,internvl,llava,llava_next")
    parser.add_argument("--test_all_models", action="store_true",
                       help="Test all supported model families")
    
    # GPU and basic settings
    parser.add_argument("--gpu", type=str, default="0", help="GPU index")
    parser.add_argument("--num_train_samples", type=int, default=5,
                       help="Number of training samples")
    parser.add_argument("--num_test_samples", type=int, default=5,
                       help="Number of test samples")
    parser.add_argument("--num_calibration_samples", type=int, default=10,
                       help="Samples per question for calibration")
    
    # Skip options
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip SFT training test")
    parser.add_argument("--skip_calibration", action="store_true",
                       help="Skip calibration test")
    parser.add_argument("--inference_only", action="store_true",
                       help="Only test base model inference (fastest)")
    
    # Output options
    parser.add_argument("--keep_outputs", action="store_true",
                       help="Keep output files after test")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: temp dir)")
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()


def get_models_to_test(args) -> list:
    """Determine which models to test based on arguments."""
    if args.model_id:
        # Specific model provided
        return [args.model_id]
    
    if args.test_all_models:
        # Test all families
        return list(DEFAULT_MODELS.values())
    
    # Parse model families
    families = [f.strip().lower() for f in args.model_families.split(",")]
    models = []
    for family in families:
        if family in DEFAULT_MODELS:
            models.append(DEFAULT_MODELS[family])
        else:
            print(f"Warning: Unknown model family '{family}', skipping")
    
    return models if models else [DEFAULT_MODELS["qwen"]]


def print_header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_success(msg: str):
    print(f"  ✅ {msg}")


def print_info(msg: str):
    print(f"  ℹ️  {msg}")


def print_error(msg: str):
    print(f"  ❌ {msg}")


def test_base_inference(
    model_config: ModelConfig,
    data_config: DataConfig,
    output_dir: str,
) -> bool:
    """Test 1: Base model inference."""
    print_header("TEST 1: Base Model Inference")
    
    try:
        # Load dataset
        print_info(f"Loading {data_config.subsample_size} test samples...")
        dataset_wrapper = get_dataset(data_config)
        dataset = dataset_wrapper.load()
        print_success(f"Loaded {len(dataset)} samples")
        
        # Setup inference
        inference_config = InferenceConfig(
            adapter_path=None,  # Base model
            num_samples=1,
            max_new_tokens=32,
            temperature=0.7,
        )
        
        print_info("Loading base model...")
        inference = VQAInference(model_config, inference_config)
        inference.load_model()
        print_success("Model loaded")
        
        # Run inference on a few samples
        print_info("Running inference...")
        results = inference.predict_dataset(dataset, num_samples=1, show_progress=True)
        
        # Show sample results
        print_info("Sample predictions:")
        for i, r in enumerate(results[:3]):
            print(f"    Q: {r.question[:50]}...")
            print(f"    GT: {r.ground_truth}")
            print(f"    Pred: {r.predictions[0][:50]}...")
            print()
        
        # Save results
        output_path = os.path.join(output_dir, "base_inference.json")
        inference.save_results(results, output_path)
        print_success(f"Results saved to {output_path}")
        
        # Cleanup model to free memory
        del inference
        torch.cuda.empty_cache()
        
        print_success("Base inference test PASSED")
        return True
        
    except Exception as e:
        print_error(f"Base inference test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sft_training(
    model_config: ModelConfig,
    train_data_config: DataConfig,
    output_dir: str,
    seed: int,
) -> str:
    """Test 2: SFT Training. Returns checkpoint path."""
    print_header("TEST 2: SFT Training (1 epoch)")
    
    try:
        checkpoint_dir = os.path.join(output_dir, "sft_checkpoint")
        
        # Create training config - minimal settings
        train_config = SFTConfig(
            output_dir=checkpoint_dir,
            num_epochs=1,
            per_device_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=2e-4,
            logging_steps=1,
            save_strategy="epoch",
            lora=LoRAConfig(r=8, lora_alpha=16),  # Smaller LoRA for speed
        )
        
        experiment_config = ExperimentConfig(
            model=model_config,
            data=train_data_config,
            training=train_config,
            seed=seed,
        )
        
        print_info(f"Training on {train_data_config.subsample_size} samples for 1 epoch...")
        
        trainer = VQASFTTrainer(experiment_config)
        trainer.setup()
        results = trainer.train()
        
        print_success(f"Training complete! Loss: {results['train_loss']:.4f}")
        print_success(f"Checkpoint saved to {checkpoint_dir}")
        
        # Cleanup
        del trainer
        torch.cuda.empty_cache()
        
        print_success("SFT training test PASSED")
        return checkpoint_dir
        
    except Exception as e:
        print_error(f"SFT training test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_sft_inference(
    model_config: ModelConfig,
    data_config: DataConfig,
    adapter_path: str,
    output_dir: str,
) -> bool:
    """Test 3: SFT model inference."""
    print_header("TEST 3: SFT Model Inference")
    
    try:
        # Load dataset
        print_info(f"Loading {data_config.subsample_size} test samples...")
        dataset_wrapper = get_dataset(data_config)
        dataset = dataset_wrapper.load()
        
        # Setup inference with adapter
        inference_config = InferenceConfig(
            adapter_path=adapter_path,
            num_samples=1,
            max_new_tokens=32,
            temperature=0.7,
        )
        
        print_info(f"Loading SFT model from {adapter_path}...")
        inference = VQAInference(model_config, inference_config)
        inference.load_model()
        print_success("SFT model loaded")
        
        # Run inference
        print_info("Running inference...")
        results = inference.predict_dataset(dataset, num_samples=1, show_progress=True)
        
        # Show sample results
        print_info("Sample predictions (SFT):")
        for i, r in enumerate(results[:3]):
            print(f"    Q: {r.question[:50]}...")
            print(f"    GT: {r.ground_truth}")
            print(f"    Pred: {r.predictions[0][:50]}...")
            print()
        
        # Save results
        output_path = os.path.join(output_dir, "sft_inference.json")
        inference.save_results(results, output_path)
        print_success(f"Results saved to {output_path}")
        
        # Cleanup
        del inference
        torch.cuda.empty_cache()
        
        print_success("SFT inference test PASSED")
        return True
        
    except Exception as e:
        print_error(f"SFT inference test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_calibration(
    model_config: ModelConfig,
    data_config: DataConfig,
    adapter_path: str,
    output_dir: str,
    num_samples: int,
) -> bool:
    """Test 4: Calibration evaluation."""
    print_header("TEST 4: Calibration Evaluation")
    
    try:
        # Load dataset (closed questions only)
        closed_data_config = DataConfig(
            dataset_name=data_config.dataset_name,
            question_type=QuestionType.CLOSED,
            split=data_config.split,
            subsample_size=data_config.subsample_size,
            seed=data_config.seed,
        )
        
        print_info(f"Loading closed questions...")
        dataset_wrapper = get_dataset(closed_data_config)
        dataset = dataset_wrapper.load()
        print_success(f"Loaded {len(dataset)} closed questions")
        
        if len(dataset) == 0:
            print_info("No closed questions in subset, skipping calibration test")
            return True
        
        # Setup inference with sampling
        inference_config = InferenceConfig(
            adapter_path=adapter_path,
            num_samples=num_samples,
            max_new_tokens=16,  # Short for yes/no
            temperature=0.7,
            do_sample=True,
        )
        
        print_info(f"Loading model for calibration ({num_samples} samples/question)...")
        inference = VQAInference(model_config, inference_config)
        inference.load_model()
        
        # Run inference with multiple samples
        print_info("Running sampling-based inference...")
        predictions = inference.predict_dataset(
            dataset, 
            num_samples=num_samples,
            show_progress=True
        )
        
        # Evaluate calibration
        print_info("Computing calibration metrics...")
        evaluator = CalibrationEvaluator(num_bins=5)  # Fewer bins for small data
        evaluator.add_predictions(predictions)
        
        metrics = evaluator.compute_metrics()
        
        print_info("Calibration Results:")
        print(f"    Accuracy: {metrics['accuracy']:.3f}")
        print(f"    Mean Confidence: {metrics['mean_confidence']:.3f}")
        print(f"    ECE: {metrics['ece']:.3f}")
        print(f"    Overconfidence: {metrics['overconfidence']:.3f}")
        
        # Save results
        cal_output_dir = os.path.join(output_dir, "calibration")
        evaluator.save_results(cal_output_dir)
        print_success(f"Calibration results saved to {cal_output_dir}")
        
        # Cleanup
        del inference
        torch.cuda.empty_cache()
        
        print_success("Calibration test PASSED")
        return True
        
    except Exception as e:
        print_error(f"Calibration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_tests_for_model(
    model_id: str,
    args,
    output_dir: str,
) -> dict:
    """Run all tests for a single model."""
    
    print("\n" + "=" * 60)
    print(f"  TESTING MODEL: {model_id}")
    print("=" * 60)
    
    # Setup configs
    model_config = ModelConfig(
        model_id=model_id,
        use_4bit=True,
    )
    
    print(f"  Detected family: {model_config.model_family.value}")
    
    train_data_config = DataConfig(
        dataset_name=DatasetName.RAD_VQA,
        question_type=QuestionType.CLOSED,
        split="train",
        subsample_size=args.num_train_samples,
        seed=args.seed,
    )
    
    test_data_config = DataConfig(
        dataset_name=DatasetName.RAD_VQA,
        question_type=QuestionType.ALL,
        split="test",
        subsample_size=args.num_test_samples,
        seed=args.seed,
    )
    
    # Create model-specific output directory
    model_name = model_id.split("/")[-1].replace("-", "_").lower()
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Track results
    results = {}
    
    # Test 1: Base inference
    results["base_inference"] = test_base_inference(
        model_config, test_data_config, model_output_dir
    )
    
    if args.inference_only:
        return results
    
    # Test 2: SFT training
    checkpoint_path = None
    if not args.skip_training:
        checkpoint_path = test_sft_training(
            model_config, train_data_config, model_output_dir, args.seed
        )
        results["sft_training"] = checkpoint_path is not None
    else:
        print_header("TEST 2: SFT Training (SKIPPED)")
        results["sft_training"] = "skipped"
    
    # Test 3: SFT inference
    if checkpoint_path:
        results["sft_inference"] = test_sft_inference(
            model_config, test_data_config, checkpoint_path, model_output_dir
        )
    else:
        print_header("TEST 3: SFT Inference (SKIPPED - no checkpoint)")
        results["sft_inference"] = "skipped"
    
    # Test 4: Calibration
    if not args.skip_calibration:
        adapter_for_cal = checkpoint_path
        results["calibration"] = test_calibration(
            model_config, test_data_config, adapter_for_cal, 
            model_output_dir, args.num_calibration_samples
        )
    else:
        print_header("TEST 4: Calibration (SKIPPED)")
        results["calibration"] = "skipped"
    
    return results


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Get models to test
    models_to_test = get_models_to_test(args)
    
    print("\n" + "=" * 60)
    print("  MEDICAL VQA FRAMEWORK - QUICK E2E TEST")
    print("=" * 60)
    print(f"Models to test: {len(models_to_test)}")
    for m in models_to_test:
        print(f"  - {m}")
    print(f"GPU: {args.gpu}")
    print(f"Train samples: {args.num_train_samples}")
    print(f"Test samples: {args.num_test_samples}")
    if not args.skip_calibration and not args.inference_only:
        print(f"Calibration samples/question: {args.num_calibration_samples}")
    print("=" * 60)
    
    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = tempfile.mkdtemp(prefix="med_vqa_test_")
    
    print(f"\nOutput directory: {output_dir}")
    
    # Run tests for each model
    all_results = {}
    for model_id in models_to_test:
        try:
            all_results[model_id] = run_tests_for_model(model_id, args, output_dir)
        except Exception as e:
            print_error(f"Model {model_id} failed: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_id] = {"error": str(e)}
        
        # Clear GPU memory between models
        torch.cuda.empty_cache()
    
    # Final Summary
    print("\n" + "=" * 60)
    print("  FINAL TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for model_id, results in all_results.items():
        model_name = model_id.split("/")[-1]
        print(f"\n{model_name}:")
        
        if "error" in results:
            print(f"  ❌ ERROR: {results['error']}")
            all_passed = False
            continue
            
        for test_name, result in results.items():
            if result == "skipped":
                status = "⏭️  SKIPPED"
            elif result:
                status = "✅ PASSED"
            else:
                status = "❌ FAILED"
                all_passed = False
            print(f"  {test_name}: {status}")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("  🎉 ALL TESTS PASSED!")
    else:
        print("  ⚠️  SOME TESTS FAILED")
    print("=" * 60)
    
    print(f"\nOutputs saved to: {output_dir}")
    
    # Cleanup if requested
    if not args.keep_outputs and not args.output_dir:
        print("Cleaning up temp directory...")
        shutil.rmtree(output_dir)
        print("Done!")
    else:
        print("Outputs kept for inspection.")


if __name__ == "__main__":
    main()

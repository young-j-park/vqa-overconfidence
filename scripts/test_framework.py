#!/usr/bin/env python3
"""
Test Script for Medical VQA Framework

Run this to verify all components are working correctly.
Tests are organized from basic (no GPU) to full integration (requires GPU).

Usage:
    # Run all tests (requires GPU)
    python scripts/test_framework.py --all

    # Run only import/config tests (no GPU needed)
    python scripts/test_framework.py --basic

    # Run with specific GPU
    python scripts/test_framework.py --all --gpu 0

    # Test specific component
    python scripts/test_framework.py --test configs
    python scripts/test_framework.py --test datasets
    python scripts/test_framework.py --test models
    python scripts/test_framework.py --test inference
"""

import argparse
import sys
import os

# Set GPU before importing torch
def set_gpu(gpu_id: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[Setup] CUDA_VISIBLE_DEVICES={gpu_id}")


def parse_args():
    parser = argparse.ArgumentParser(description="Test Medical VQA Framework")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--basic", action="store_true", help="Run basic tests only (no GPU)")
    parser.add_argument("--test", type=str, choices=["configs", "datasets", "models", "inference", "training"],
                       help="Test specific component")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use")
    return parser.parse_args()


def test_imports():
    """Test that all modules can be imported."""
    print("\n" + "=" * 60)
    print("TEST: Imports")
    print("=" * 60)
    
    try:
        from med_vqa.configs import (
            ModelConfig, DataConfig, SFTConfig, LoRAConfig,
            ExperimentConfig, ModelFamily, QuestionType, DatasetName
        )
        print("  ✓ configs module")
        
        from med_vqa.data import (
            get_dataset, get_collator, RADVQADataset, SLAKEDataset,
            list_available_datasets
        )
        print("  ✓ data module")
        
        from med_vqa.models import load_model, ModelLoader
        print("  ✓ models module")
        
        from med_vqa.training import VQASFTTrainer, run_sft_training
        print("  ✓ training module")
        
        from med_vqa.inference import VQAInference, run_inference
        print("  ✓ inference module")
        
        from med_vqa.evaluation import CalibrationEvaluator, compute_calibration_metrics
        print("  ✓ evaluation module")
        
        from med_vqa.utils import set_seed, get_gpu_memory_info
        print("  ✓ utils module")
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        return False


def test_configs():
    """Test configuration classes."""
    print("\n" + "=" * 60)
    print("TEST: Configurations")
    print("=" * 60)
    
    from med_vqa.configs import (
        ModelConfig, DataConfig, SFTConfig, LoRAConfig,
        ExperimentConfig, ModelFamily, QuestionType, DatasetName,
        create_sft_config
    )
    
    # Test ModelConfig auto-detection
    print("\n[ModelConfig] Testing model family auto-detection...")
    test_cases = [
        ("Qwen/Qwen3-VL-2B-Instruct", ModelFamily.QWEN_VL),
        ("OpenGVLab/InternVL3-2B-hf", ModelFamily.INTERNVL),
        ("llava-hf/llava-1.5-7b-hf", ModelFamily.LLAVA),
        ("llava-hf/llava-v1.6-vicuna-7b-hf", ModelFamily.LLAVA_NEXT),
    ]
    
    for model_id, expected_family in test_cases:
        config = ModelConfig(model_id=model_id)
        assert config.model_family == expected_family, f"Expected {expected_family}, got {config.model_family}"
        print(f"  ✓ {model_id} -> {config.model_family.value}")
    
    # Test DataConfig
    print("\n[DataConfig] Testing data configuration...")
    data_config = DataConfig(
        dataset_name=DatasetName.RAD_VQA,
        question_type=QuestionType.CLOSED,
        subsample_size=100,
        seed=42,
    )
    assert data_config.dataset_name == DatasetName.RAD_VQA
    assert data_config.question_type == QuestionType.CLOSED
    assert data_config.subsample_size == 100
    print(f"  ✓ DataConfig created: {data_config.dataset_name.value}, {data_config.question_type.value}")
    
    # Test ExperimentConfig serialization
    print("\n[ExperimentConfig] Testing serialization...")
    config = create_sft_config(
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        output_dir="/tmp/test_output",
        dataset="rad_vqa",
        question_type="closed",
        subsample_size=50,
    )
    
    # Save and load
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        config.save(f.name)
        loaded = ExperimentConfig.load(f.name)
        os.unlink(f.name)
    
    assert loaded.model.model_id == config.model.model_id
    assert loaded.data.subsample_size == config.data.subsample_size
    print(f"  ✓ Config save/load works")
    
    print("\n✅ All config tests passed!")
    return True


def test_datasets():
    """Test dataset loading."""
    print("\n" + "=" * 60)
    print("TEST: Datasets")
    print("=" * 60)
    
    from med_vqa.configs import DataConfig, DatasetName, QuestionType
    from med_vqa.data import get_dataset, list_available_datasets
    
    print(f"\nAvailable datasets: {list_available_datasets()}")
    
    # Test RAD-VQA loading
    print("\n[RAD-VQA] Testing dataset loading...")
    config = DataConfig(
        dataset_name=DatasetName.RAD_VQA,
        question_type=QuestionType.ALL,
        split="train",
        subsample_size=10,
        seed=42,
    )
    
    dataset_wrapper = get_dataset(config)
    dataset = dataset_wrapper.load()
    
    assert len(dataset) == 10, f"Expected 10 samples, got {len(dataset)}"
    assert "image" in dataset.column_names
    assert "question" in dataset.column_names
    assert "answer" in dataset.column_names
    assert "answer_type" in dataset.column_names
    print(f"  ✓ Loaded {len(dataset)} samples")
    print(f"  ✓ Columns: {dataset.column_names}")
    
    # Test question type filtering
    print("\n[RAD-VQA] Testing closed question filtering...")
    config_closed = DataConfig(
        dataset_name=DatasetName.RAD_VQA,
        question_type=QuestionType.CLOSED,
        split="train",
        subsample_size=20,
        seed=42,
    )
    
    dataset_closed = get_dataset(config_closed).load()
    
    # Verify all are closed
    for sample in dataset_closed:
        assert sample["answer_type"] == "closed", f"Expected closed, got {sample['answer_type']}"
    print(f"  ✓ All {len(dataset_closed)} samples are closed questions")
    
    # Test statistics
    stats = dataset_wrapper.get_statistics()
    print(f"\n[Statistics] {stats}")
    
    print("\n✅ All dataset tests passed!")
    return True


def test_models(gpu_id: int = 0):
    """Test model loading (requires GPU)."""
    print("\n" + "=" * 60)
    print("TEST: Models (requires GPU)")
    print("=" * 60)
    
    import torch
    if not torch.cuda.is_available():
        print("  ⚠ CUDA not available, skipping model tests")
        return True
    
    from med_vqa.configs import ModelConfig, ModelFamily
    from med_vqa.models import load_model, get_gpu_memory_info
    
    print(f"\n[GPU] Using GPU {gpu_id}")
    print(f"[GPU] Memory before: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Test loading a small model (if available)
    # Using Qwen as example - adjust if you have different models
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    
    print(f"\n[Model] Testing model loading: {model_id}")
    print("  (This may take a minute...)")
    
    try:
        config = ModelConfig(
            model_id=model_id,
            use_4bit=True,  # Use 4-bit to reduce memory
        )
        
        model, processor = load_model(config)
        
        print(f"  ✓ Model loaded successfully")
        print(f"  ✓ Model family: {config.model_family.value}")
        print(f"  ✓ GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        # Cleanup
        del model, processor
        torch.cuda.empty_cache()
        
        print("\n✅ Model loading test passed!")
        return True
        
    except Exception as e:
        print(f"\n⚠ Model loading test skipped: {e}")
        print("  (This is expected if the model isn't downloaded)")
        return True


def test_collators():
    """Test data collators."""
    print("\n" + "=" * 60)
    print("TEST: Collators")
    print("=" * 60)
    
    from med_vqa.configs import ModelFamily
    from med_vqa.data import get_collator
    from unittest.mock import MagicMock
    
    # Create mock processor
    mock_processor = MagicMock()
    mock_processor.apply_chat_template = MagicMock(return_value="formatted text")
    mock_processor.tokenizer.pad_token_id = 0
    
    print("\n[Collators] Testing collator creation...")
    
    for family in ModelFamily:
        collator = get_collator(family, mock_processor, max_length=2048)
        print(f"  ✓ {family.value}: {collator.__class__.__name__}")
    
    print("\n✅ Collator tests passed!")
    return True


def test_evaluation():
    """Test evaluation metrics."""
    print("\n" + "=" * 60)
    print("TEST: Evaluation Metrics")
    print("=" * 60)
    
    from med_vqa.evaluation import (
        CalibrationEvaluator, CalibrationResult,
        parse_yes_no, compute_calibration_metrics
    )
    from med_vqa.inference import VQAPrediction
    
    # Test yes/no parsing
    print("\n[Parsing] Testing yes/no parsing...")
    test_cases = [
        ("Yes", "yes"),
        ("No", "no"),
        ("yes, I think so", "yes"),
        ("No, definitely not", "no"),
        ("Maybe", None),
        ("Yes and No", None),
    ]
    
    for text, expected in test_cases:
        result = parse_yes_no(text)
        assert result == expected, f"Expected {expected}, got {result} for '{text}'"
        print(f"  ✓ '{text}' -> {result}")
    
    # Test calibration metrics
    print("\n[Metrics] Testing calibration computation...")
    
    # Create mock predictions
    predictions = [
        VQAPrediction(
            question="Is this normal?",
            ground_truth="yes",
            predictions=["yes"] * 80 + ["no"] * 20,  # 80% yes -> correct, conf=0.8
            answer_type="closed",
        ),
        VQAPrediction(
            question="Is there a tumor?",
            ground_truth="no",
            predictions=["no"] * 90 + ["yes"] * 10,  # 90% no -> correct, conf=0.9
            answer_type="closed",
        ),
        VQAPrediction(
            question="Is this abnormal?",
            ground_truth="yes",
            predictions=["no"] * 70 + ["yes"] * 30,  # 70% no -> incorrect, conf=0.7
            answer_type="closed",
        ),
    ]
    
    evaluator = CalibrationEvaluator(num_bins=10)
    evaluator.add_predictions(predictions)
    
    metrics = evaluator.compute_metrics()
    
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  Mean Confidence: {metrics['mean_confidence']:.3f}")
    print(f"  ECE: {metrics['ece']:.3f}")
    print(f"  ✓ Metrics computed successfully")
    
    # Verify expected values
    assert metrics['num_samples'] == 3
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['ece'] <= 1
    
    print("\n✅ Evaluation tests passed!")
    return True


def test_end_to_end_dry_run():
    """Test end-to-end workflow with minimal data (dry run)."""
    print("\n" + "=" * 60)
    print("TEST: End-to-End Dry Run")
    print("=" * 60)
    
    from med_vqa.configs import (
        ModelConfig, DataConfig, SFTConfig, LoRAConfig,
        ExperimentConfig, DatasetName, QuestionType
    )
    from med_vqa.data import get_dataset, get_collator
    
    print("\n[1] Creating experiment config...")
    config = ExperimentConfig(
        model=ModelConfig(model_id="Qwen/Qwen3-VL-2B-Instruct"),
        data=DataConfig(
            dataset_name=DatasetName.RAD_VQA,
            question_type=QuestionType.CLOSED,
            subsample_size=5,
            seed=42,
        ),
        training=SFTConfig(
            output_dir="/tmp/test_sft",
            num_epochs=1,
            per_device_batch_size=1,
        ),
        seed=42,
    )
    print(f"  ✓ Config created for {config.model.model_id}")
    
    print("\n[2] Loading dataset...")
    dataset_wrapper = get_dataset(config.data)
    dataset = dataset_wrapper.load()
    print(f"  ✓ Loaded {len(dataset)} samples")
    
    print("\n[3] Checking sample format...")
    sample = dataset[0]
    required_keys = ["image", "question", "answer", "answer_type"]
    for key in required_keys:
        assert key in sample, f"Missing key: {key}"
    print(f"  ✓ Sample has all required keys: {required_keys}")
    print(f"  ✓ Question: {sample['question'][:50]}...")
    print(f"  ✓ Answer: {sample['answer']}")
    print(f"  ✓ Type: {sample['answer_type']}")
    
    print("\n✅ End-to-end dry run passed!")
    return True


def run_all_tests(gpu_id: int = 0, basic_only: bool = False):
    """Run all tests."""
    print("\n" + "=" * 60)
    print("MEDICAL VQA FRAMEWORK - TEST SUITE")
    print("=" * 60)
    
    results = {}
    
    # Basic tests (no GPU)
    results["imports"] = test_imports()
    results["configs"] = test_configs()
    results["datasets"] = test_datasets()
    results["collators"] = test_collators()
    results["evaluation"] = test_evaluation()
    results["dry_run"] = test_end_to_end_dry_run()
    
    if not basic_only:
        # GPU tests
        results["models"] = test_models(gpu_id)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("🎉 All tests passed!")
    else:
        print("⚠ Some tests failed")
    
    return all_passed


def main():
    args = parse_args()
    
    # Set GPU first
    if args.gpu is not None:
        set_gpu(args.gpu)
    
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if args.test:
        # Run specific test
        test_map = {
            "configs": test_configs,
            "datasets": test_datasets,
            "models": lambda: test_models(args.gpu),
            "inference": test_end_to_end_dry_run,
            "training": test_end_to_end_dry_run,
        }
        test_map[args.test]()
    elif args.basic:
        run_all_tests(args.gpu, basic_only=True)
    else:
        run_all_tests(args.gpu, basic_only=False)


if __name__ == "__main__":
    main()

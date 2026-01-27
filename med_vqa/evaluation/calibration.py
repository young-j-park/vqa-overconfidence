"""
Calibration evaluation metrics for Medical VQA models.

Computes:
- ECE (Expected Calibration Error)
- MCE (Maximum Calibration Error)
- Overconfidence
- Accuracy
- Per-bin statistics
"""

import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np

from ..inference import VQAPrediction


@dataclass
class CalibrationResult:
    """Result of calibration evaluation for a single sample."""
    question: str
    ground_truth: str
    predicted: str
    confidence: float
    p_yes: float
    p_no: float
    is_correct: bool
    
    # Raw counts
    yes_count: int
    no_count: int
    unknown_count: int


def parse_yes_no(text: str) -> Optional[str]:
    """Parse yes/no from model response.
    
    Args:
        text: Model response text
        
    Returns:
        'yes', 'no', or None if unclear
    """
    text_lower = text.lower().strip()
    
    # Exact match
    if text_lower in ["yes", "yes.", "yes,", "yes!"]:
        return "yes"
    if text_lower in ["no", "no.", "no,", "no!"]:
        return "no"
    
    # Starts with
    if text_lower.startswith("yes"):
        return "yes"
    if text_lower.startswith("no"):
        return "no"
    
    # Contains (less reliable)
    if "yes" in text_lower and "no" not in text_lower:
        return "yes"
    if "no" in text_lower and "yes" not in text_lower:
        return "no"
    
    return None


def compute_empirical_probability(predictions: List[str]) -> CalibrationResult:
    """Compute empirical probability from multiple samples.
    
    Args:
        predictions: List of model predictions
        
    Returns:
        Tuple of (p_yes, p_no, yes_count, no_count, unknown_count)
    """
    yes_count = 0
    no_count = 0
    unknown_count = 0
    
    for pred in predictions:
        parsed = parse_yes_no(pred)
        if parsed == "yes":
            yes_count += 1
        elif parsed == "no":
            no_count += 1
        else:
            unknown_count += 1
    
    valid_count = yes_count + no_count
    
    if valid_count > 0:
        p_yes = yes_count / valid_count
        p_no = no_count / valid_count
    else:
        p_yes = 0.5
        p_no = 0.5
    
    return p_yes, p_no, yes_count, no_count, unknown_count


def evaluate_calibration_single(
    prediction: VQAPrediction,
) -> CalibrationResult:
    """Evaluate calibration for a single sample.
    
    Args:
        prediction: VQAPrediction with multiple samples
        
    Returns:
        CalibrationResult
    """
    p_yes, p_no, yes_count, no_count, unknown_count = compute_empirical_probability(
        prediction.predictions
    )
    
    # Determine prediction and confidence
    predicted = "yes" if p_yes >= 0.5 else "no"
    confidence = max(p_yes, p_no)
    
    # Check correctness
    gt_lower = prediction.ground_truth.lower().strip()
    is_correct = predicted == gt_lower
    
    return CalibrationResult(
        question=prediction.question,
        ground_truth=prediction.ground_truth,
        predicted=predicted,
        confidence=confidence,
        p_yes=p_yes,
        p_no=p_no,
        is_correct=is_correct,
        yes_count=yes_count,
        no_count=no_count,
        unknown_count=unknown_count,
    )


def compute_calibration_metrics(
    results: List[CalibrationResult],
    num_bins: int = 10,
) -> Dict[str, Any]:
    """Compute ECE, MCE, and overconfidence metrics.
    
    Args:
        results: List of CalibrationResult objects
        num_bins: Number of bins for ECE calculation
        
    Returns:
        Dictionary with metrics and bin data
    """
    confidences = np.array([r.confidence for r in results])
    accuracies = np.array([r.is_correct for r in results])
    
    # Bin boundaries
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    
    ece = 0.0
    mce = 0.0
    overconfidence = 0.0
    bin_data = []
    
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = in_bin.sum()
        
        if bin_size > 0:
            bin_acc = accuracies[in_bin].mean()
            bin_conf = confidences[in_bin].mean()
            
            # ECE contribution
            gap = abs(bin_acc - bin_conf)
            ece += (bin_size / len(results)) * gap
            
            # MCE
            mce = max(mce, gap)
            
            # Overconfidence (only when conf > acc)
            if bin_conf > bin_acc:
                overconfidence += (bin_size / len(results)) * (bin_conf - bin_acc)
            
            bin_data.append({
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "bin_size": int(bin_size),
                "accuracy": float(bin_acc),
                "confidence": float(bin_conf),
                "gap": float(gap),
            })
    
    return {
        "ece": float(ece),
        "mce": float(mce),
        "overconfidence": float(overconfidence),
        "accuracy": float(accuracies.mean()),
        "mean_confidence": float(confidences.mean()),
        "num_samples": len(results),
        "bin_data": bin_data,
    }


class CalibrationEvaluator:
    """Evaluator for model calibration on closed questions."""
    
    def __init__(self, num_bins: int = 10):
        """Initialize evaluator.
        
        Args:
            num_bins: Number of bins for ECE calculation
        """
        self.num_bins = num_bins
        self.results: List[CalibrationResult] = []
    
    def add_prediction(self, prediction: VQAPrediction) -> CalibrationResult:
        """Add a prediction and compute its calibration result.
        
        Args:
            prediction: VQAPrediction object
            
        Returns:
            CalibrationResult
        """
        result = evaluate_calibration_single(prediction)
        self.results.append(result)
        return result
    
    def add_predictions(self, predictions: List[VQAPrediction]) -> None:
        """Add multiple predictions.
        
        Args:
            predictions: List of VQAPrediction objects
        """
        for pred in predictions:
            # Only evaluate closed questions
            if pred.answer_type == "closed":
                self.add_prediction(pred)
    
    def compute_metrics(self) -> Dict[str, Any]:
        """Compute all calibration metrics.
        
        Returns:
            Dictionary with ECE, MCE, accuracy, etc.
        """
        if not self.results:
            raise ValueError("No results to evaluate. Add predictions first.")
        
        return compute_calibration_metrics(self.results, self.num_bins)
    
    def save_results(self, output_dir: str, prefix: str = "") -> None:
        """Save evaluation results to files.
        
        Args:
            output_dir: Directory to save results
            prefix: Optional prefix for filenames
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Compute metrics
        metrics = self.compute_metrics()
        
        # Save metrics summary
        metrics_path = os.path.join(output_dir, f"{prefix}metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Save detailed results
        details_path = os.path.join(output_dir, f"{prefix}detailed_results.json")
        detailed = []
        for r in self.results:
            detailed.append({
                "question": r.question,
                "ground_truth": r.ground_truth,
                "predicted": r.predicted,
                "confidence": r.confidence,
                "p_yes": r.p_yes,
                "p_no": r.p_no,
                "is_correct": r.is_correct,
                "yes_count": r.yes_count,
                "no_count": r.no_count,
                "unknown_count": r.unknown_count,
            })
        
        with open(details_path, "w") as f:
            json.dump(detailed, f, indent=2)
        
        # Save human-readable summary
        summary_path = os.path.join(output_dir, f"{prefix}summary.txt")
        with open(summary_path, "w") as f:
            f.write("Calibration Evaluation Results\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Number of samples: {metrics['num_samples']}\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Mean Confidence: {metrics['mean_confidence']:.4f}\n")
            f.write(f"ECE: {metrics['ece']:.4f}\n")
            f.write(f"MCE: {metrics['mce']:.4f}\n")
            f.write(f"Overconfidence: {metrics['overconfidence']:.4f}\n")
            f.write("\nBin Data:\n")
            f.write("-" * 40 + "\n")
            for bin_info in metrics['bin_data']:
                f.write(f"  [{bin_info['bin_lower']:.1f}-{bin_info['bin_upper']:.1f}]: "
                       f"n={bin_info['bin_size']}, "
                       f"acc={bin_info['accuracy']:.3f}, "
                       f"conf={bin_info['confidence']:.3f}, "
                       f"gap={bin_info['gap']:.3f}\n")
        
        print(f"Results saved to: {output_dir}")
    
    def print_summary(self) -> None:
        """Print evaluation summary to console."""
        metrics = self.compute_metrics()
        
        print("\n" + "=" * 50)
        print("CALIBRATION RESULTS")
        print("=" * 50)
        print(f"Samples: {metrics['num_samples']}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Mean Confidence: {metrics['mean_confidence']:.4f}")
        print(f"ECE: {metrics['ece']:.4f}")
        print(f"MCE: {metrics['mce']:.4f}")
        print(f"Overconfidence: {metrics['overconfidence']:.4f}")
        print("=" * 50)

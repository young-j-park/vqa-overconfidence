#!/usr/bin/env python3
"""
Reliability Diagrams for Calibration Analysis

1. Scatter plot: Confidence vs Accuracy for all models (BASE vs SFT)
2. Per-model reliability diagrams showing calibration bins

Usage:
    python scripts/plot_reliability_diagrams.py --results_dir ./results/calibration --output_dir ./figures
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


# Hardcoded summary data
SUMMARY_DATA = [
    # RAD-VQA
    {"model": "Qwen3-VL-8B", "dataset": "RAD-VQA", "type": "BASE", "acc": 0.4821, "conf": 0.9145, "ece": 0.4324},
    {"model": "Qwen3-VL-8B", "dataset": "RAD-VQA", "type": "SFT", "acc": 0.7490, "conf": 0.8664, "ece": 0.1238},
    {"model": "InternVL3-8B", "dataset": "RAD-VQA", "type": "BASE", "acc": 0.6494, "conf": 0.8023, "ece": 0.1529},
    {"model": "InternVL3-8B", "dataset": "RAD-VQA", "type": "SFT", "acc": 0.7092, "conf": 0.8347, "ece": 0.1306},
    {"model": "LLaVA-NeXT-7B", "dataset": "RAD-VQA", "type": "BASE", "acc": 0.5737, "conf": 0.6136, "ece": 0.0399},
    {"model": "LLaVA-NeXT-7B", "dataset": "RAD-VQA", "type": "SFT", "acc": 0.6853, "conf": 0.8557, "ece": 0.1784},
    # SLAKE
    {"model": "Qwen3-VL-8B", "dataset": "SLAKE", "type": "BASE", "acc": 0.4183, "conf": 0.8873, "ece": 0.4690},
    {"model": "Qwen3-VL-8B", "dataset": "SLAKE", "type": "SFT", "acc": 0.7356, "conf": 0.9237, "ece": 0.1881},
    {"model": "InternVL3-8B", "dataset": "SLAKE", "type": "BASE", "acc": 0.6418, "conf": 0.8823, "ece": 0.2448},
    {"model": "InternVL3-8B", "dataset": "SLAKE", "type": "SFT", "acc": 0.6587, "conf": 0.8224, "ece": 0.1638},
    {"model": "LLaVA-NeXT-7B", "dataset": "SLAKE", "type": "BASE", "acc": 0.4688, "conf": 0.6509, "ece": 0.1822},
    {"model": "LLaVA-NeXT-7B", "dataset": "SLAKE", "type": "SFT", "acc": 0.6635, "conf": 0.7738, "ece": 0.1104},
]

# Colors and markers
DATASET_COLORS = {
    "RAD-VQA": "#E74C3C",    # Red
    "SLAKE": "#3498DB",       # Blue
}

MODEL_MARKERS = {
    "Qwen3-VL-8B": "o",       # Circle
    "InternVL3-8B": "s",      # Square
    "LLaVA-NeXT-7B": "^",     # Triangle
}

TYPE_STYLES = {
    "BASE": {"facecolor": "white", "edgewidth": 2},  # Empty marker
    "SFT": {"facecolor": None, "edgewidth": 1.5},    # Filled marker
}


def plot_confidence_vs_accuracy_scatter(output_path: str):
    """
    Plot 1: Scatter plot of Confidence vs Accuracy
    - X-axis: Mean Confidence
    - Y-axis: Accuracy
    - Color: Dataset
    - Shape: Model
    - Fill: BASE (empty) vs SFT (filled)
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot diagonal (perfect calibration)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect calibration', linewidth=2)
    
    # Plot each point
    for d in SUMMARY_DATA:
        color = DATASET_COLORS[d["dataset"]]
        marker = MODEL_MARKERS[d["model"]]
        
        if d["type"] == "BASE":
            ax.scatter(d["conf"], d["acc"], 
                      c='white', marker=marker, s=250, alpha=0.9,
                      edgecolors=color, linewidths=3)
        else:  # SFT
            ax.scatter(d["conf"], d["acc"], 
                      c=color, marker=marker, s=250, alpha=0.8,
                      edgecolors='black', linewidths=1.5)
    
    # Draw arrows from BASE to SFT for each model-dataset pair
    for model in MODEL_MARKERS.keys():
        for dataset in DATASET_COLORS.keys():
            base_data = next((d for d in SUMMARY_DATA if d["model"] == model and d["dataset"] == dataset and d["type"] == "BASE"), None)
            sft_data = next((d for d in SUMMARY_DATA if d["model"] == model and d["dataset"] == dataset and d["type"] == "SFT"), None)
            
            if base_data and sft_data:
                ax.annotate('', 
                           xy=(sft_data["conf"], sft_data["acc"]),
                           xytext=(base_data["conf"], base_data["acc"]),
                           arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5, lw=1.5))
    
    # Labels and formatting
    ax.set_xlabel('Mean Confidence', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('Confidence vs Accuracy\n(Arrows: BASE → SFT)', fontsize=15, fontweight='bold')
    ax.set_xlim(0.55, 1.0)
    ax.set_ylim(0.35, 0.80)
    ax.grid(True, alpha=0.3)
    
    # Add overconfidence/underconfidence regions
    ax.fill_between([0.55, 1.0], [0.55, 1.0], [0.35, 0.35], alpha=0.1, color='red', label='Overconfident region')
    ax.fill_between([0.55, 1.0], [0.55, 1.0], [1.0, 1.0], alpha=0.1, color='green', label='Underconfident region')
    
    # Legend
    legend_elements = [
        # Datasets (colors)
        mpatches.Patch(color=DATASET_COLORS["RAD-VQA"], label='RAD-VQA'),
        mpatches.Patch(color=DATASET_COLORS["SLAKE"], label='SLAKE'),
        # Models (shapes)
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markeredgecolor='black', markersize=12, label='Qwen3-VL-8B'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
               markeredgecolor='black', markersize=12, label='InternVL3-8B'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', 
               markeredgecolor='black', markersize=12, label='LLaVA-NeXT-7B'),
        # Type (fill)
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white', 
               markeredgecolor='gray', markeredgewidth=2, markersize=12, label='BASE (empty)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markeredgecolor='black', markersize=12, label='SFT (filled)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10, ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    # PDF version
    pdf_path = output_path.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved: {pdf_path}")
    plt.close()


def load_bin_data(results_dir: str):
    """Load bin data from all results for reliability diagrams."""
    bin_data = {}
    
    for item in Path(results_dir).iterdir():
        if not item.is_dir():
            continue
        
        # Try to find metrics file
        metrics_file = item / "metrics.json"
        if not metrics_file.exists():
            metrics_file = item / "logits" / "metrics.json"
        
        if not metrics_file.exists():
            continue
        
        try:
            with open(metrics_file) as f:
                metrics = json.load(f)
            
            name = item.name.lower()
            
            # Parse name
            eval_type = "BASE" if name.startswith("base_") else "SFT"
            
            if "qwen" in name:
                model = "Qwen3-VL-8B"
            elif "internvl" in name:
                model = "InternVL3-8B"
            elif "llava" in name:
                model = "LLaVA-NeXT-7B"
            else:
                continue
            
            if "rad_vqa" in name:
                dataset = "RAD-VQA"
            elif "slake" in name:
                dataset = "SLAKE"
            else:
                continue
            
            key = (model, dataset, eval_type)
            bin_data[key] = {
                "bins": metrics.get("bin_data", []),
                "ece": metrics.get("ece", 0),
                "accuracy": metrics.get("accuracy", 0),
                "mean_confidence": metrics.get("mean_confidence", 0),
            }
        except Exception as e:
            print(f"Error loading {item.name}: {e}")
    
    return bin_data


def plot_reliability_diagrams_grid(bin_data: dict, output_path: str):
    """
    Plot 2: Reliability diagrams for each model-dataset combination
    Grid: rows=datasets, cols=models
    Each cell shows BASE vs SFT comparison
    """
    models = ["Qwen3-VL-8B", "InternVL3-8B", "LLaVA-NeXT-7B"]
    datasets = ["RAD-VQA", "SLAKE"]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            ax = axes[i, j]
            
            # Get BASE and SFT data
            base_key = (model, dataset, "BASE")
            sft_key = (model, dataset, "SFT")
            
            base_data = bin_data.get(base_key, {})
            sft_data = bin_data.get(sft_key, {})
            
            # Plot perfect calibration line
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2, label='Perfect')
            
            # Plot BASE reliability diagram
            if base_data.get("bins"):
                bins = base_data["bins"]
                confs = [b["confidence"] for b in bins]
                accs = [b["accuracy"] for b in bins]
                sizes = [b["bin_size"] for b in bins]
                
                # Normalize sizes for visualization
                max_size = max(sizes) if sizes else 1
                normalized_sizes = [100 + 400 * (s / max_size) for s in sizes]
                
                ax.scatter(confs, accs, s=normalized_sizes, c='white', 
                          edgecolors=DATASET_COLORS[dataset], linewidths=2, 
                          alpha=0.7, label=f'BASE (ECE={base_data["ece"]:.3f})')
                
                # Connect with line
                sorted_idx = np.argsort(confs)
                ax.plot([confs[k] for k in sorted_idx], [accs[k] for k in sorted_idx], 
                       color=DATASET_COLORS[dataset], alpha=0.3, linestyle='--', linewidth=1.5)
            
            # Plot SFT reliability diagram
            if sft_data.get("bins"):
                bins = sft_data["bins"]
                confs = [b["confidence"] for b in bins]
                accs = [b["accuracy"] for b in bins]
                sizes = [b["bin_size"] for b in bins]
                
                max_size = max(sizes) if sizes else 1
                normalized_sizes = [100 + 400 * (s / max_size) for s in sizes]
                
                ax.scatter(confs, accs, s=normalized_sizes, c=DATASET_COLORS[dataset], 
                          edgecolors='black', linewidths=1, 
                          alpha=0.8, label=f'SFT (ECE={sft_data["ece"]:.3f})')
                
                sorted_idx = np.argsort(confs)
                ax.plot([confs[k] for k in sorted_idx], [accs[k] for k in sorted_idx], 
                       color=DATASET_COLORS[dataset], alpha=0.5, linewidth=2)
            
            # Formatting
            ax.set_xlim(0.45, 1.05)
            ax.set_ylim(0.0, 1.05)
            ax.set_xlabel('Confidence', fontsize=11)
            ax.set_ylabel('Accuracy', fontsize=11)
            ax.set_title(f'{model}\n({dataset})', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left', fontsize=9)
            
            # Shade overconfidence region
            ax.fill_between([0, 1], [0, 1], [0, 0], alpha=0.05, color='red')
    
    plt.suptitle('Reliability Diagrams: BASE (empty) vs SFT (filled)\nBubble size ∝ bin sample count', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    pdf_path = output_path.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved: {pdf_path}")
    plt.close()


def plot_calibration_gap_bars(bin_data: dict, output_path: str):
    """
    Plot 3: Bar chart showing calibration gap (confidence - accuracy) per bin
    Shows overconfidence (positive) vs underconfidence (negative)
    """
    models = ["Qwen3-VL-8B", "InternVL3-8B", "LLaVA-NeXT-7B"]
    datasets = ["RAD-VQA", "SLAKE"]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            ax = axes[i, j]
            
            base_key = (model, dataset, "BASE")
            sft_key = (model, dataset, "SFT")
            
            base_data = bin_data.get(base_key, {})
            sft_data = bin_data.get(sft_key, {})
            
            bin_centers = np.arange(0.05, 1.0, 0.1)
            width = 0.035
            
            # Plot BASE gaps
            if base_data.get("bins"):
                bins = base_data["bins"]
                gaps = []
                positions = []
                for b in bins:
                    gap = b["confidence"] - b["accuracy"]  # Positive = overconfident
                    gaps.append(gap)
                    positions.append(b["confidence"])
                
                colors = ['#E74C3C' if g > 0 else '#27AE60' for g in gaps]
                ax.bar([p - width for p in positions], gaps, width=width*1.8, 
                      color='white', edgecolor=colors, linewidth=2,
                      alpha=0.7, label=f'BASE')
            
            # Plot SFT gaps
            if sft_data.get("bins"):
                bins = sft_data["bins"]
                gaps = []
                positions = []
                for b in bins:
                    gap = b["confidence"] - b["accuracy"]
                    gaps.append(gap)
                    positions.append(b["confidence"])
                
                colors = ['#E74C3C' if g > 0 else '#27AE60' for g in gaps]
                ax.bar([p + width for p in positions], gaps, width=width*1.8, 
                      color=colors, edgecolor='black', linewidth=1,
                      alpha=0.8, label=f'SFT')
            
            # Zero line
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            
            # Formatting
            ax.set_xlim(0.45, 1.05)
            ax.set_ylim(-0.3, 0.6)
            ax.set_xlabel('Confidence Bin', fontsize=11)
            ax.set_ylabel('Gap (Conf - Acc)', fontsize=11)
            ax.set_title(f'{model}\n({dataset})', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(loc='upper left', fontsize=9)
            
            # Add text annotations
            ax.text(0.95, 0.95, 'Overconfident ↑', transform=ax.transAxes, 
                   fontsize=8, ha='right', va='top', color='#E74C3C')
            ax.text(0.95, 0.05, 'Underconfident ↓', transform=ax.transAxes, 
                   fontsize=8, ha='right', va='bottom', color='#27AE60')
    
    plt.suptitle('Calibration Gap: Confidence - Accuracy\n(Red=Overconfident, Green=Underconfident)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    pdf_path = output_path.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved: {pdf_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="./results/calibration")
    parser.add_argument("--output_dir", type=str, default="./figures")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot 1: Confidence vs Accuracy scatter
    print("\n[1/3] Creating Confidence vs Accuracy scatter plot...")
    plot_confidence_vs_accuracy_scatter(
        os.path.join(args.output_dir, "confidence_vs_accuracy.png")
    )
    
    # Load bin data for reliability diagrams
    print("\n[2/3] Loading bin data for reliability diagrams...")
    bin_data = load_bin_data(args.results_dir)
    print(f"Loaded data for {len(bin_data)} configurations")
    
    if bin_data:
        # Plot 2: Reliability diagrams grid
        print("\n[3/3] Creating reliability diagrams...")
        plot_reliability_diagrams_grid(
            bin_data, 
            os.path.join(args.output_dir, "reliability_diagrams.png")
        )
        
        # Plot 3: Calibration gap bars
        print("\n[4/4] Creating calibration gap bar charts...")
        plot_calibration_gap_bars(
            bin_data,
            os.path.join(args.output_dir, "calibration_gaps.png")
        )
    else:
        print("No bin data found. Run with --results_dir pointing to calibration results.")
    
    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

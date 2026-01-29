#!/usr/bin/env python3
"""
Generate scatter plot of BASE vs SFT confidence.

Color = Dataset (RAD-VQA: Red, SLAKE: Blue)
Shape = Model (Qwen: Circle, InternVL: Square, LLaVA: Triangle)

Usage:
    python scripts/plot_confidence_correlation.py --output ./figures/confidence_correlation.png
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


# Hardcoded data from evaluation results
DATA = [
    {"model": "Qwen3-VL-8B", "dataset": "RAD-VQA", "base_conf": 0.9145, "sft_conf": 0.8664},
    {"model": "InternVL3-8B", "dataset": "RAD-VQA", "base_conf": 0.8023, "sft_conf": 0.8347},
    {"model": "LLaVA-NeXT-7B", "dataset": "RAD-VQA", "base_conf": 0.6136, "sft_conf": 0.8557},
    {"model": "Qwen3-VL-8B", "dataset": "SLAKE", "base_conf": 0.8873, "sft_conf": 0.9237},
    {"model": "InternVL3-8B", "dataset": "SLAKE", "base_conf": 0.8823, "sft_conf": 0.8224},
    {"model": "LLaVA-NeXT-7B", "dataset": "SLAKE", "base_conf": 0.6509, "sft_conf": 0.7738},
]

# Color = Dataset
DATASET_COLORS = {
    "RAD-VQA": "#E74C3C",    # Red
    "SLAKE": "#3498DB",       # Blue
}

# Shape = Model
MODEL_MARKERS = {
    "Qwen3-VL-8B": "o",       # Circle
    "InternVL3-8B": "s",      # Square
    "LLaVA-NeXT-7B": "^",     # Triangle
}


def create_scatter_plot(output_path: str):
    """Create scatter plot of BASE vs SFT confidence."""
    
    # Compute delta conf
    for d in DATA:
        d["delta_conf"] = d["sft_conf"] - d["base_conf"]
    
    # Setup plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    base_confs = [d["base_conf"] for d in DATA]
    sft_confs = [d["sft_conf"] for d in DATA]
    delta_confs = [d["delta_conf"] for d in DATA]
    
    # =========================================================================
    # Plot 1: BASE Conf vs SFT Conf
    # =========================================================================
    ax1 = axes[0]
    
    for d in DATA:
        color = DATASET_COLORS[d["dataset"]]
        marker = MODEL_MARKERS[d["model"]]
        
        ax1.scatter(d["base_conf"], d["sft_conf"], 
                   c=color, marker=marker, s=200, alpha=0.8,
                   edgecolors='black', linewidths=1.5)
    
    # Diagonal line (y=x)
    ax1.plot([0.4, 1.0], [0.4, 1.0], 'k--', alpha=0.3, label='y=x (no change)')
    
    # Regression line
    z = np.polyfit(base_confs, sft_confs, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0.5, 1.0, 100)
    ax1.plot(x_line, p(x_line), 'gray', linestyle='-', alpha=0.5, linewidth=2, label=f'Trend (slope={z[0]:.2f})')
    
    # Correlation
    corr = np.corrcoef(base_confs, sft_confs)[0, 1]
    ax1.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax1.transAxes, 
            fontsize=14, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax1.set_xlabel('BASE Model Confidence', fontsize=12, fontweight='bold')
    ax1.set_ylabel('SFT Model Confidence', fontsize=12, fontweight='bold')
    ax1.set_title('BASE vs SFT Confidence', fontsize=13, fontweight='bold')
    ax1.set_xlim(0.55, 0.98)
    ax1.set_ylim(0.75, 0.98)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right', fontsize=10)
    
    # =========================================================================
    # Plot 2: BASE Conf vs Δ Conf (Confidence Change)
    # =========================================================================
    ax2 = axes[1]
    
    for d in DATA:
        color = DATASET_COLORS[d["dataset"]]
        marker = MODEL_MARKERS[d["model"]]
        
        ax2.scatter(d["base_conf"], d["delta_conf"], 
                   c=color, marker=marker, s=200, alpha=0.8,
                   edgecolors='black', linewidths=1.5)
    
    # Zero line
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3, label='No change')
    
    # Regression line
    z2 = np.polyfit(base_confs, delta_confs, 1)
    p2 = np.poly1d(z2)
    ax2.plot(x_line, p2(x_line), 'gray', linestyle='-', alpha=0.5, linewidth=2, label=f'Trend (slope={z2[0]:.2f})')
    
    # Correlation
    corr2 = np.corrcoef(base_confs, delta_confs)[0, 1]
    ax2.text(0.05, 0.95, f'r = {corr2:.3f}', transform=ax2.transAxes, 
            fontsize=14, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax2.set_xlabel('BASE Model Confidence', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Confidence Change (SFT - BASE)', fontsize=12, fontweight='bold')
    ax2.set_title('BASE Confidence vs Confidence Change', fontsize=13, fontweight='bold')
    ax2.set_xlim(0.55, 0.98)
    ax2.set_ylim(-0.15, 0.30)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10)
    
    # =========================================================================
    # Shared Legend: Color = Dataset, Shape = Model
    # =========================================================================
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
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, 
               bbox_to_anchor=(0.5, -0.02), fontsize=11)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    # Also save as PDF for paper
    pdf_path = output_path.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved: {pdf_path}")
    
    plt.close()
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"BASE Conf vs SFT Conf correlation:  r = {corr:.3f}")
    print(f"BASE Conf vs Δ Conf correlation:    r = {corr2:.3f}")
    print(f"\nBASE Conf range: [{min(base_confs):.3f}, {max(base_confs):.3f}] (spread: {max(base_confs)-min(base_confs):.3f})")
    print(f"SFT Conf range:  [{min(sft_confs):.3f}, {max(sft_confs):.3f}] (spread: {max(sft_confs)-min(sft_confs):.3f})")
    print(f"\n→ SFT reduces confidence spread by {(max(base_confs)-min(base_confs)) - (max(sft_confs)-min(sft_confs)):.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="./figures/confidence_correlation.png")
    args = parser.parse_args()
    
    create_scatter_plot(args.output)


if __name__ == "__main__":
    main()
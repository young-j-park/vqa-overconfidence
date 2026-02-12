#!/usr/bin/env python3
"""
Temperature Scaling Analysis Across Training Epochs
=====================================================
For each model × dataset × method × epoch:
  - Sweep T from 0.1 to 5.0
  - Compute ECE, overconfidence, accuracy
  - Plot how optimal T* and best achievable ECE evolve over training

Usage:
    python temperature_scaling_analysis.py
    (run from project root where results/ lives)
"""

import json
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.special import softmax
from collections import defaultdict
import matplotlib
matplotlib.rcParams.update({"font.size": 11})

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path("results/calibration_epochs")

# Temperature sweep range
T_VALUES = np.concatenate([
    np.arange(0.1, 1.0, 0.05),
    np.arange(1.0, 3.0, 0.1),
    np.arange(3.0, 5.25, 0.25),
])

N_BINS = 10

# Methods to include (skip contrastive)
METHODS = ["base", "sft", "aug_sft", "grpo"]

# Display names and colors
METHOD_DISPLAY = {
    "base":    "Base",
    "sft":     "SFT",
    "aug_sft": "Aug-SFT",
    "grpo":    "GRPO",
}

METHOD_COLORS = {
    "base":    "#636363",
    "sft":     "#1f77b4",
    "aug_sft": "#ff7f0e",
    "grpo":    "#2ca02c",
}

MODEL_DISPLAY = {
    "internvl3_8b": "InternVL3-8B",
    "llava_next_7b": "LLaVA-NeXT-7B",
    "qwen3vl_8b":   "Qwen3-VL-8B",
}

DATASET_DISPLAY = {
    "slake":   "SLAKE",
    "rad_vqa": "RAD-VQA",
}

# Linestyles for methods in epoch evolution plots
METHOD_LINESTYLES = {
    "base":    ":",
    "sft":     "-",
    "aug_sft": "--",
    "grpo":    "-.",
}


# ============================================================
# DISCOVERY: Auto-detect all model/dataset/epoch combos
# ============================================================

def discover_experiments():
    """
    Scan BASE_DIR and return a structured dict:
    {
        (method, model, dataset): {
            step_number_or_0: filepath,
            ...
        }
    }
    """
    experiments = defaultdict(dict)

    for json_path in BASE_DIR.rglob("detailed_results.json"):
        rel = json_path.relative_to(BASE_DIR)
        parts = rel.parts  # e.g. ('sft_qwen3vl_8b_slake', 'step-308', 'detailed_results.json')

        if len(parts) < 2:
            continue

        folder_name = parts[0]

        # Parse method
        method = None
        rest = folder_name
        for m in ["aug_sft", "sft", "grpo", "base"]:  # order matters: aug_sft before sft
            if folder_name.startswith(m + "_"):
                method = m
                rest = folder_name[len(m) + 1:]
                break

        if method is None:
            continue
        if method not in METHODS:
            continue

        # Parse model and dataset from rest
        # rest is like "internvl3_8b_slake" or "qwen3vl_8b_rad_vqa"
        model = None
        dataset = None
        for ds_key in ["rad_vqa", "slake"]:
            if rest.endswith("_" + ds_key):
                dataset = ds_key
                model = rest[:-(len(ds_key) + 1)]
                break

        if model is None or dataset is None:
            continue
        if model not in MODEL_DISPLAY:
            continue

        # Parse step
        if len(parts) == 2:
            # base models or direct detailed_results.json
            step = 0
        elif len(parts) == 3:
            step_dir = parts[1]
            if step_dir == "final":
                step = 999999  # will be sorted last
            elif step_dir.startswith("step-"):
                try:
                    step = int(step_dir.split("-")[1])
                except ValueError:
                    continue
            else:
                continue
        else:
            continue

        experiments[(method, model, dataset)][step] = str(json_path)

    return experiments


# ============================================================
# CORE FUNCTIONS
# ============================================================

def load_results(filepath):
    """Load detailed_results.json. Returns records with logits."""
    with open(filepath) as f:
        data = json.load(f)

    records = []
    for item in data:
        gt = item["ground_truth"].strip().lower()

        has_logits = (
            item.get("logit_yes") is not None
            and item.get("logit_no") is not None
        )
        has_probs = (
            item.get("p_yes") is not None
            and item.get("p_no") is not None
        )

        if has_logits:
            logit_yes = float(item["logit_yes"])
            logit_no = float(item["logit_no"])
        elif has_probs:
            p_yes = float(item["p_yes"])
            p_no = float(item["p_no"])
            eps = 1e-10
            p_yes = np.clip(p_yes, eps, 1 - eps)
            p_no = np.clip(p_no, eps, 1 - eps)
            logit_yes = np.log(p_yes)
            logit_no = np.log(p_no)
        else:
            continue

        records.append({
            "logit_yes": logit_yes,
            "logit_no": logit_no,
            "ground_truth": gt,
        })

    return records


def apply_temperature(records, T):
    """Apply temperature scaling, return (confidences, correctness)."""
    confidences = []
    correctness = []
    for r in records:
        logits = np.array([r["logit_yes"], r["logit_no"]]) / T
        probs = softmax(logits)
        p_yes, p_no = probs[0], probs[1]
        conf = max(p_yes, p_no)
        pred = "yes" if p_yes >= p_no else "no"
        correct = (pred == r["ground_truth"])
        confidences.append(conf)
        correctness.append(correct)
    return np.array(confidences), np.array(correctness)


def compute_ece(confidences, correctness, n_bins=N_BINS):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(confidences)
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if i == 0:
            mask = (confidences >= lo) & (confidences <= hi)
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc = correctness[mask].mean()
        ece += (n_in_bin / total) * abs(avg_conf - avg_acc)
    return ece


def compute_overconfidence(confidences, correctness, n_bins=N_BINS):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    overconf = 0.0
    total = len(confidences)
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if i == 0:
            mask = (confidences >= lo) & (confidences <= hi)
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc = correctness[mask].mean()
        overconf += (n_in_bin / total) * max(0, avg_conf - avg_acc)
    return overconf


def sweep_temperature(records):
    """Sweep all temperatures, return dict of arrays."""
    eces, overconfs, accs = [], [], []
    for T in T_VALUES:
        confs, corrs = apply_temperature(records, T)
        eces.append(compute_ece(confs, corrs))
        overconfs.append(compute_overconfidence(confs, corrs))
        accs.append(corrs.mean())
    return {
        "ece": np.array(eces),
        "overconf": np.array(overconfs),
        "acc": np.array(accs),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    experiments = discover_experiments()

    print(f"Discovered {len(experiments)} experiment groups:\n")
    for (method, model, dataset), steps in sorted(experiments.items()):
        step_list = sorted(steps.keys())
        print(f"  {METHOD_DISPLAY[method]:>8} / {MODEL_DISPLAY[model]:<16} / {DATASET_DISPLAY[dataset]:<8}  "
              f"→ {len(step_list)} checkpoints: {step_list}")

    # ============================================================
    # Process everything
    # ============================================================
    # all_data[(method, model, dataset)] = {step: {ece: [...], overconf: [...], acc: [...], T_opt_ece, ...}}
    all_data = {}

    for (method, model, dataset), steps in sorted(experiments.items()):
        key = (method, model, dataset)
        all_data[key] = {}

        for step in sorted(steps.keys()):
            filepath = steps[step]
            records = load_results(filepath)
            if len(records) == 0:
                print(f"  [SKIP] {method}/{model}/{dataset} step={step}: no records")
                continue

            sweep = sweep_temperature(records)
            idx_opt_ece = np.argmin(sweep["ece"])
            idx_opt_oc = np.argmin(sweep["overconf"])
            idx_T1 = np.argmin(np.abs(T_VALUES - 1.0))

            all_data[key][step] = {
                **sweep,
                "T_opt_ece": T_VALUES[idx_opt_ece],
                "ece_at_T1": sweep["ece"][idx_T1],
                "ece_at_Topt": sweep["ece"][idx_opt_ece],
                "T_opt_oc": T_VALUES[idx_opt_oc],
                "oc_at_T1": sweep["overconf"][idx_T1],
                "oc_at_Topt": sweep["overconf"][idx_opt_oc],
                "acc_at_T1": sweep["acc"][idx_T1],
                "acc_at_Topt": sweep["acc"][idx_opt_ece],
                "n_samples": len(records),
            }

    print(f"\nProcessed all experiments.\n")

    # ============================================================
    # PLOT 1: Temperature sweep curves — one subplot per (model, dataset)
    #   For each, show ECE vs T for each method at FINAL checkpoint
    # ============================================================
    models = sorted(set(m for (_, m, _) in all_data.keys()))
    datasets = sorted(set(d for (_, _, d) in all_data.keys()))

    fig1, axes1 = plt.subplots(len(datasets), len(models), figsize=(6 * len(models), 5 * len(datasets)))
    if len(datasets) == 1:
        axes1 = axes1[np.newaxis, :]
    if len(models) == 1:
        axes1 = axes1[:, np.newaxis]

    for col, model in enumerate(models):
        for row, dataset in enumerate(datasets):
            ax = axes1[row, col]
            for method in METHODS:
                key = (method, model, dataset)
                if key not in all_data or not all_data[key]:
                    continue
                # Use the last (highest step) checkpoint
                last_step = max(all_data[key].keys())
                data = all_data[key][last_step]
                label = METHOD_DISPLAY[method]
                if method != "base":
                    label += f" (step {last_step})" if last_step != 999999 else " (final)"
                ax.plot(T_VALUES, data["ece"], label=label,
                        color=METHOD_COLORS[method],
                        linestyle=METHOD_LINESTYLES[method],
                        linewidth=2)

            ax.axvline(x=1.0, color="gray", linestyle=":", alpha=0.4, label="T=1.0")
            ax.set_xlabel("Temperature (T)")
            ax.set_ylabel("ECE ↓")
            ax.set_title(f"{MODEL_DISPLAY[model]} — {DATASET_DISPLAY[dataset]}")
            ax.set_xlim(T_VALUES[0], T_VALUES[-1])
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

    plt.suptitle("ECE vs Temperature (Final Checkpoints)", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("temp_scaling_ece_curves.png", dpi=150, bbox_inches="tight")
    plt.savefig("temp_scaling_ece_curves.pdf", bbox_inches="tight")
    print("Saved: temp_scaling_ece_curves.png / .pdf")

    # ============================================================
    # PLOT 2: Epoch evolution of optimal T* and ECE(T*) and ECE(T=1)
    #   Faceted by (model, dataset), lines per method
    # ============================================================
    fig2, axes2 = plt.subplots(3, len(models) * len(datasets),
                                figsize=(5 * len(models) * len(datasets), 12))
    # Rows: T*, ECE(T=1), ECE(T*)
    plot_idx = 0
    col_labels = []
    for model in models:
        for dataset in datasets:
            col_labels.append(f"{MODEL_DISPLAY[model]}\n{DATASET_DISPLAY[dataset]}")

            for method in METHODS:
                key = (method, model, dataset)
                if key not in all_data or not all_data[key]:
                    continue
                if method == "base":
                    # Base has no epochs — draw horizontal line
                    step0 = list(all_data[key].keys())[0]
                    d = all_data[key][step0]
                    for row_idx, metric in enumerate(["T_opt_ece", "ece_at_T1", "ece_at_Topt"]):
                        ax = axes2[row_idx, plot_idx]
                        ax.axhline(y=d[metric], color=METHOD_COLORS[method],
                                   linestyle=":", alpha=0.6, linewidth=1.5,
                                   label=METHOD_DISPLAY[method])
                    continue

                steps_sorted = sorted(all_data[key].keys())
                steps_arr = np.array(steps_sorted)
                T_opts = [all_data[key][s]["T_opt_ece"] for s in steps_sorted]
                ece_T1 = [all_data[key][s]["ece_at_T1"] for s in steps_sorted]
                ece_Topt = [all_data[key][s]["ece_at_Topt"] for s in steps_sorted]

                color = METHOD_COLORS[method]
                ls = METHOD_LINESTYLES[method]
                lbl = METHOD_DISPLAY[method]

                axes2[0, plot_idx].plot(steps_arr, T_opts, color=color, linestyle=ls,
                                         linewidth=2, marker="o", markersize=4, label=lbl)
                axes2[1, plot_idx].plot(steps_arr, ece_T1, color=color, linestyle=ls,
                                         linewidth=2, marker="o", markersize=4, label=lbl)
                axes2[2, plot_idx].plot(steps_arr, ece_Topt, color=color, linestyle=ls,
                                         linewidth=2, marker="o", markersize=4, label=lbl)

            # Labels
            axes2[0, plot_idx].set_title(col_labels[-1], fontsize=10)
            axes2[0, plot_idx].set_ylabel("Optimal T*")
            axes2[0, plot_idx].grid(True, alpha=0.3)
            axes2[0, plot_idx].legend(fontsize=7)
            axes2[0, plot_idx].axhline(y=1.0, color="gray", linestyle=":", alpha=0.3)

            axes2[1, plot_idx].set_ylabel("ECE @ T=1.0")
            axes2[1, plot_idx].grid(True, alpha=0.3)

            axes2[2, plot_idx].set_ylabel("ECE @ T* (best achievable)")
            axes2[2, plot_idx].set_xlabel("Training Step")
            axes2[2, plot_idx].grid(True, alpha=0.3)

            plot_idx += 1

    plt.suptitle("Training Evolution of Temperature Scaling", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("temp_scaling_epoch_evolution.png", dpi=150, bbox_inches="tight")
    plt.savefig("temp_scaling_epoch_evolution.pdf", bbox_inches="tight")
    print("Saved: temp_scaling_epoch_evolution.png / .pdf")

    # ============================================================
    # PLOT 3: "Headroom" plot — ECE(T=1) vs ECE(T*) scatter
    #   Shows how much temperature scaling can help each method
    # ============================================================
    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 8))

    for (method, model, dataset), step_data in sorted(all_data.items()):
        if not step_data:
            continue
        color = METHOD_COLORS.get(method, "#333")
        marker = {"slake": "o", "rad_vqa": "s"}.get(dataset, "^")

        for step, d in step_data.items():
            ax3.scatter(d["ece_at_T1"], d["ece_at_Topt"],
                       color=color, marker=marker, alpha=0.6, s=40)

    # Diagonal (no improvement)
    lim = max(ax3.get_xlim()[1], ax3.get_ylim()[1])
    ax3.plot([0, lim], [0, lim], "k--", alpha=0.3, label="No improvement")
    ax3.set_xlabel("ECE @ T=1.0 (original)")
    ax3.set_ylabel("ECE @ T* (best achievable)")
    ax3.set_title("Temperature Scaling Headroom")
    ax3.grid(True, alpha=0.3)

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = []
    for method in METHODS:
        legend_elements.append(Line2D([0], [0], marker="o", color="w",
                                       markerfacecolor=METHOD_COLORS[method],
                                       markersize=10, label=METHOD_DISPLAY[method]))
    legend_elements.append(Line2D([0], [0], marker="o", color="w",
                                   markerfacecolor="gray", markersize=8, label="SLAKE"))
    legend_elements.append(Line2D([0], [0], marker="s", color="w",
                                   markerfacecolor="gray", markersize=8, label="RAD-VQA"))
    ax3.legend(handles=legend_elements, fontsize=9)

    plt.tight_layout()
    plt.savefig("temp_scaling_headroom.png", dpi=150, bbox_inches="tight")
    plt.savefig("temp_scaling_headroom.pdf", bbox_inches="tight")
    print("Saved: temp_scaling_headroom.png / .pdf")

    # ============================================================
    # TABLE: Summary for final checkpoints
    # ============================================================
    print("\n" + "=" * 120)
    print("FINAL CHECKPOINT SUMMARY — OPTIMAL TEMPERATURE (minimizing ECE)")
    print("=" * 120)
    header = (f"{'Method':<10} {'Model':<16} {'Dataset':<10} {'Step':>8} "
              f"{'T*':>5} {'ECE(1.0)':>9} {'ECE(T*)':>9} {'ΔECE':>9} {'%Δ':>7} "
              f"{'Acc(1.0)':>9} {'Acc(T*)':>9} {'N':>6}")
    print(header)
    print("-" * 120)

    for (method, model, dataset), step_data in sorted(all_data.items()):
        if not step_data:
            continue
        last_step = max(step_data.keys())
        d = step_data[last_step]
        delta = d["ece_at_T1"] - d["ece_at_Topt"]
        pct = (delta / d["ece_at_T1"] * 100) if d["ece_at_T1"] > 0 else 0
        step_str = "final" if last_step == 999999 else str(last_step)
        print(f"{METHOD_DISPLAY[method]:<10} {MODEL_DISPLAY[model]:<16} {DATASET_DISPLAY[dataset]:<10} "
              f"{step_str:>8} {d['T_opt_ece']:>5.2f} {d['ece_at_T1']:>9.4f} {d['ece_at_Topt']:>9.4f} "
              f"{delta:>9.4f} {pct:>6.1f}% {d['acc_at_T1']:>9.4f} {d['acc_at_Topt']:>9.4f} "
              f"{d['n_samples']:>6}")

    # ============================================================
    # TABLE: All epochs
    # ============================================================
    print("\n" + "=" * 120)
    print("ALL EPOCHS — OPTIMAL TEMPERATURE (minimizing ECE)")
    print("=" * 120)
    print(header)
    print("-" * 120)

    for (method, model, dataset), step_data in sorted(all_data.items()):
        for step in sorted(step_data.keys()):
            d = step_data[step]
            delta = d["ece_at_T1"] - d["ece_at_Topt"]
            pct = (delta / d["ece_at_T1"] * 100) if d["ece_at_T1"] > 0 else 0
            step_str = "final" if step == 999999 else str(step)
            print(f"{METHOD_DISPLAY[method]:<10} {MODEL_DISPLAY[model]:<16} {DATASET_DISPLAY[dataset]:<10} "
                  f"{step_str:>8} {d['T_opt_ece']:>5.2f} {d['ece_at_T1']:>9.4f} {d['ece_at_Topt']:>9.4f} "
                  f"{delta:>9.4f} {pct:>6.1f}% {d['acc_at_T1']:>9.4f} {d['acc_at_Topt']:>9.4f} "
                  f"{d['n_samples']:>6}")
        # Blank line between experiment groups
        if step_data:
            print()

    plt.close("all")
    print("Done!")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Summarize Calibration Results

Aggregates calibration results from multiple evaluations into comparison tables.
Supports BASE, SFT, and GRPO model types.

Usage:
    python scripts/summarize_calibration.py --results_dir ./results/calibration
    python scripts/summarize_calibration.py --results_dir ./results/calibration --format markdown
    python scripts/summarize_calibration.py --results_dir ./results/calibration --output summary.csv
    python scripts/summarize_calibration.py --results_dir ./results/calibration --detailed
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


def load_metrics(results_dir: str) -> Dict[str, dict]:
    """Load metrics from all result directories."""
    results = {}

    for item in Path(results_dir).iterdir():
        if not item.is_dir():
            continue

        metrics_file = item / "metrics.json"
        if not metrics_file.exists():
            for subdir in ["sampling", "logits"]:
                sub_metrics = item / subdir / "metrics.json"
                if sub_metrics.exists():
                    metrics_file = sub_metrics
                    break

        if not metrics_file.exists():
            continue

        try:
            with open(metrics_file) as f:
                metrics = json.load(f)

            config_file = item / "config.json"
            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)
                metrics["_config"] = config

            results[item.name] = metrics
        except Exception as e:
            print(f"Warning: Failed to load {item.name}: {e}")

    return results


def parse_result_name(name: str) -> Tuple[str, str, str]:
    """Parse result directory name into (type, model, dataset)."""
    parts = name.lower().split("_")

    if parts[0] in ["base", "sft", "grpo"]:
        eval_type = parts[0].upper()
    else:
        if "grpo" in name.lower():
            eval_type = "GRPO"
        elif any(p in name for p in ["lr5e-5", "lr1e-4"]):
            eval_type = "SFT"
        else:
            eval_type = "BASE"

    if "rad_vqa" in name or "radvqa" in name:
        dataset = "RAD-VQA"
    elif "slake" in name:
        dataset = "SLAKE"
    else:
        dataset = "Unknown"

    if "qwen" in name:
        model = "Qwen3-VL-8B"
    elif "internvl" in name:
        model = "InternVL3-8B"
    elif "llava" in name:
        model = "LLaVA-NeXT-7B"
    else:
        model = "Unknown"

    return eval_type, model, dataset


EVAL_TYPE_ORDER = ["BASE", "SFT", "GRPO"]


def format_table_text(results: Dict[str, dict]) -> str:
    """Format results as text table."""
    organized = defaultdict(lambda: defaultdict(dict))
    for name, metrics in results.items():
        eval_type, model, dataset = parse_result_name(name)
        organized[dataset][model][eval_type] = {"name": name, "metrics": metrics}

    lines = []
    lines.append("=" * 130)
    lines.append("CALIBRATION RESULTS SUMMARY")
    lines.append("=" * 130)

    header = (
        f"{'Dataset':<10} {'Model':<18} │ {'Type':<5} │ "
        f"{'N':>5} {'Acc':>7} {'ECE':>7} {'MCE':>7} {'OverConf':>8} {'Conf':>7} │ "
        f"{'Unknown%':>8} {'Random%':>8}"
    )
    lines.append(header)
    lines.append("-" * 130)

    for dataset in sorted(organized.keys()):
        for model in sorted(organized[dataset].keys()):
            model_results = organized[dataset][model]

            base_ece = None
            base_acc = None
            if "BASE" in model_results:
                base_ece = model_results["BASE"]["metrics"].get("ece")
                base_acc = model_results["BASE"]["metrics"].get("accuracy")

            for eval_type in EVAL_TYPE_ORDER:
                if eval_type not in model_results:
                    continue

                m = model_results[eval_type]["metrics"]
                n = m.get("num_questions", "?")
                acc = m.get("accuracy", 0)
                ece = m.get("ece", 0)
                mce = m.get("mce", 0)
                overconf = m.get("overconfidence", 0)
                conf = m.get("mean_confidence", 0)
                unknown_rate = m.get("unknown_rate", 0) * 100
                random_rate = m.get("random_assignment_rate", 0) * 100

                line = (
                    f"{dataset:<10} {model:<18} │ {eval_type:<5} │ "
                    f"{n:>5} {acc:>7.4f} {ece:>7.4f} {mce:>7.4f} {overconf:>8.4f} {conf:>7.4f} │ "
                    f"{unknown_rate:>7.1f}% {random_rate:>7.1f}%"
                )
                lines.append(line)

            for compare_type in ["SFT", "GRPO"]:
                if compare_type in model_results and base_ece is not None:
                    cmp_m = model_results[compare_type]["metrics"]
                    delta_ece = cmp_m.get("ece", 0) - base_ece
                    delta_acc = cmp_m.get("accuracy", 0) - base_acc
                    label = f"Δ{compare_type[0]}"
                    lines.append(
                        f"{'':10} {'':18} │ {label:<5} │ "
                        f"{'':>5} {delta_acc:>+7.4f} {delta_ece:>+7.4f}"
                    )

            lines.append("-" * 130)

    lines.append("=" * 130)
    lines.append("")
    lines.append("Legend:")
    lines.append("  N:        Number of questions evaluated")
    lines.append("  Acc:      Accuracy")
    lines.append("  ECE:      Expected Calibration Error (lower is better)")
    lines.append("  MCE:      Maximum Calibration Error (lower is better)")
    lines.append("  OverConf: Overconfidence metric (lower is better)")
    lines.append("  Conf:     Mean confidence")
    lines.append("  Unknown%: Percentage of unparseable responses")
    lines.append("  Random%:  Percentage of questions with random assignment")
    lines.append("  ΔS:       Change from BASE to SFT   (negative ECE = improvement)")
    lines.append("  ΔG:       Change from BASE to GRPO  (negative ECE = improvement)")

    return "\n".join(lines)


def format_table_markdown(results: Dict[str, dict]) -> str:
    """Format results as markdown table."""
    organized = defaultdict(lambda: defaultdict(dict))
    for name, metrics in results.items():
        eval_type, model, dataset = parse_result_name(name)
        organized[dataset][model][eval_type] = {"name": name, "metrics": metrics}

    lines = []
    lines.append("# Calibration Results Summary")
    lines.append("")
    lines.append("| Dataset | Model | Type | N | Acc | ECE | MCE | OverConf | Conf | Unknown% | Random% |")
    lines.append("|---------|-------|------|---|-----|-----|-----|----------|------|----------|---------|")

    for dataset in sorted(organized.keys()):
        for model in sorted(organized[dataset].keys()):
            model_results = organized[dataset][model]
            for eval_type in EVAL_TYPE_ORDER:
                if eval_type not in model_results:
                    continue
                m = model_results[eval_type]["metrics"]
                n = m.get("num_questions", "?")
                acc = m.get("accuracy", 0)
                ece = m.get("ece", 0)
                mce = m.get("mce", 0)
                overconf = m.get("overconfidence", 0)
                conf = m.get("mean_confidence", 0)
                unknown_rate = m.get("unknown_rate", 0) * 100
                random_rate = m.get("random_assignment_rate", 0) * 100

                line = (
                    f"| {dataset} | {model} | {eval_type} | {n} | "
                    f"{acc:.4f} | {ece:.4f} | {mce:.4f} | {overconf:.4f} | "
                    f"{conf:.4f} | {unknown_rate:.1f}% | {random_rate:.1f}% |"
                )
                lines.append(line)

    return "\n".join(lines)


def format_table_csv(results: Dict[str, dict]) -> str:
    """Format results as CSV."""
    organized = defaultdict(lambda: defaultdict(dict))
    for name, metrics in results.items():
        eval_type, model, dataset = parse_result_name(name)
        organized[dataset][model][eval_type] = {"name": name, "metrics": metrics}

    lines = []
    lines.append("dataset,model,type,n,accuracy,ece,mce,overconfidence,mean_confidence,unknown_rate,random_rate")

    for dataset in sorted(organized.keys()):
        for model in sorted(organized[dataset].keys()):
            model_results = organized[dataset][model]
            for eval_type in EVAL_TYPE_ORDER:
                if eval_type not in model_results:
                    continue
                m = model_results[eval_type]["metrics"]
                n = m.get("num_questions", "")
                acc = m.get("accuracy", "")
                ece = m.get("ece", "")
                mce = m.get("mce", "")
                overconf = m.get("overconfidence", "")
                conf = m.get("mean_confidence", "")
                unknown_rate = m.get("unknown_rate", "")
                random_rate = m.get("random_assignment_rate", "")
                lines.append(f"{dataset},{model},{eval_type},{n},{acc},{ece},{mce},{overconf},{conf},{unknown_rate},{random_rate}")

    return "\n".join(lines)


def print_detailed_comparison(results: Dict[str, dict]):
    """Print detailed BASE vs SFT vs GRPO comparison."""
    organized = defaultdict(lambda: defaultdict(dict))
    for name, metrics in results.items():
        eval_type, model, dataset = parse_result_name(name)
        organized[dataset][model][eval_type] = metrics

    print("\n" + "=" * 90)
    print("DETAILED BASE vs SFT vs GRPO COMPARISON")
    print("=" * 90)

    for dataset in sorted(organized.keys()):
        print(f"\n{dataset}")
        print("-" * 70)

        for model in sorted(organized[dataset].keys()):
            model_results = organized[dataset][model]

            if "BASE" not in model_results:
                continue

            print(f"\n  {model}:")
            print(f"    {'Metric':<20} {'BASE':>10} {'SFT':>10} {'GRPO':>10} {'ΔSFT':>10} {'ΔGRPO':>10}")
            print(f"    {'-' * 70}")

            base = model_results["BASE"]
            sft = model_results.get("SFT")
            grpo = model_results.get("GRPO")

            for metric in ["accuracy", "ece", "mce", "overconfidence", "mean_confidence", "unknown_rate"]:
                b_val = base.get(metric, 0)
                s_val = sft.get(metric, 0) if sft else None
                g_val = grpo.get(metric, 0) if grpo else None

                if metric == "unknown_rate":
                    b_str = f"{b_val * 100:>9.1f}%"
                    s_str = f"{s_val * 100:>9.1f}%" if s_val is not None else f"{'N/A':>10}"
                    g_str = f"{g_val * 100:>9.1f}%" if g_val is not None else f"{'N/A':>10}"
                    ds_str = f"{(s_val - b_val) * 100:>+9.1f}%" if s_val is not None else f"{'':>10}"
                    dg_str = f"{(g_val - b_val) * 100:>+9.1f}%" if g_val is not None else f"{'':>10}"
                else:
                    b_str = f"{b_val:>10.4f}"
                    s_str = f"{s_val:>10.4f}" if s_val is not None else f"{'N/A':>10}"
                    g_str = f"{g_val:>10.4f}" if g_val is not None else f"{'N/A':>10}"
                    ds_str = f"{s_val - b_val:>+10.4f}" if s_val is not None else f"{'':>10}"
                    dg_str = f"{g_val - b_val:>+10.4f}" if g_val is not None else f"{'':>10}"

                print(f"    {metric:<20} {b_str} {s_str} {g_str} {ds_str} {dg_str}")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize calibration results")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--format", type=str, default="text", choices=["text", "markdown", "csv"])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--detailed", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    results = load_metrics(args.results_dir)

    if not results:
        print(f"No results found in {args.results_dir}")
        return

    print(f"Found {len(results)} result directories\n")

    if args.format == "text":
        output = format_table_text(results)
    elif args.format == "markdown":
        output = format_table_markdown(results)
    elif args.format == "csv":
        output = format_table_csv(results)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Saved to: {args.output}")
    else:
        print(output)

    if args.detailed:
        print_detailed_comparison(results)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Debug N Differences Across Models

Investigates why different models have different N (sample counts)
in calibration results, even when they should evaluate the same test set.
"""

import argparse
import os
import json
from pathlib import Path
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="./results/calibration")
    return parser.parse_args()


def load_detailed_results(results_dir: str):
    """Load detailed results from all evaluation folders."""
    results = {}
    
    for item in Path(results_dir).iterdir():
        if not item.is_dir():
            continue
        
        detailed_file = item / "detailed_results.json"
        if not detailed_file.exists():
            continue
        
        try:
            with open(detailed_file) as f:
                data = json.load(f)
            
            results[item.name] = {
                "n": len(data),
                "questions": set(d["question"] for d in data),
                "data": data,
            }
        except Exception as e:
            print(f"Error loading {item.name}: {e}")
    
    return results


def analyze_differences(results: dict):
    """Analyze why N differs across models."""
    
    print("=" * 80)
    print("N (SAMPLE COUNT) ANALYSIS")
    print("=" * 80)
    
    # Group by dataset
    by_dataset = defaultdict(list)
    for name, data in results.items():
        if "rad_vqa" in name.lower():
            by_dataset["rad_vqa"].append((name, data))
        elif "slake" in name.lower():
            by_dataset["slake"].append((name, data))
    
    for dataset, entries in by_dataset.items():
        print(f"\n{'─' * 80}")
        print(f"DATASET: {dataset.upper()}")
        print(f"{'─' * 80}")
        
        # Sort by N
        entries = sorted(entries, key=lambda x: x[1]["n"], reverse=True)
        
        print(f"\n{'Model':<60} {'N':>8}")
        print("-" * 70)
        for name, data in entries:
            print(f"{name:<60} {data['n']:>8}")
        
        # Find question differences
        if len(entries) >= 2:
            print(f"\n\nQUESTION SET COMPARISON:")
            print("-" * 70)
            
            # Get all unique questions
            all_questions = set()
            for _, data in entries:
                all_questions.update(data["questions"])
            
            print(f"Total unique questions across all models: {len(all_questions)}")
            
            # Check which questions each model has
            for name, data in entries:
                missing = all_questions - data["questions"]
                if missing:
                    print(f"\n{name}:")
                    print(f"  Has {len(data['questions'])} questions")
                    print(f"  Missing {len(missing)} questions:")
                    for q in list(missing)[:5]:
                        print(f"    - {q[:60]}...")
                    if len(missing) > 5:
                        print(f"    ... and {len(missing) - 5} more")
            
            # Find common questions
            common = entries[0][1]["questions"]
            for _, data in entries[1:]:
                common = common.intersection(data["questions"])
            
            print(f"\nQuestions common to ALL models: {len(common)}")
            
            # Check for questions only in some models
            print(f"\n\nPER-MODEL EXCLUSIVE QUESTIONS:")
            for name, data in entries:
                others = set()
                for other_name, other_data in entries:
                    if other_name != name:
                        others.update(other_data["questions"])
                
                exclusive = data["questions"] - others
                if exclusive:
                    print(f"\n{name} has {len(exclusive)} exclusive questions:")
                    for q in list(exclusive)[:3]:
                        print(f"    - {q[:60]}...")


def check_unknown_counts(results: dict):
    """Check unknown_count in detailed results."""
    
    print("\n\n" + "=" * 80)
    print("UNKNOWN COUNT ANALYSIS (from detailed_results)")
    print("=" * 80)
    
    for name, data in sorted(results.items()):
        if not data["data"]:
            continue
        
        # Check if detailed results have unknown_count
        sample = data["data"][0]
        if "unknown_count" not in sample:
            print(f"\n{name}: No unknown_count field")
            continue
        
        total_yes = sum(d.get("yes_count", 0) for d in data["data"])
        total_no = sum(d.get("no_count", 0) for d in data["data"])
        total_unknown = sum(d.get("unknown_count", 0) for d in data["data"])
        total = total_yes + total_no + total_unknown
        
        print(f"\n{name}:")
        print(f"  Questions evaluated: {len(data['data'])}")
        print(f"  Total samples: {total}")
        print(f"  Yes: {total_yes} ({100*total_yes/total:.1f}%)")
        print(f"  No: {total_no} ({100*total_no/total:.1f}%)")
        print(f"  Unknown: {total_unknown} ({100*total_unknown/total:.1f}%)")
        
        # Check for questions with high unknown rate
        high_unknown = [d for d in data["data"] if d.get("unknown_count", 0) > 50]
        if high_unknown:
            print(f"  Questions with >50% unknown responses: {len(high_unknown)}")


def main():
    args = parse_args()
    
    results = load_detailed_results(args.results_dir)
    
    if not results:
        print(f"No results found in {args.results_dir}")
        return
    
    analyze_differences(results)
    check_unknown_counts(results)
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

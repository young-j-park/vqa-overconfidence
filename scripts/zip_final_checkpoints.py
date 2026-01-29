#!/usr/bin/env python3
"""
Zip Final Checkpoints Only

Analyzes checkpoint directories and zips only the final epoch,
skipping intermediate checkpoints to save space.

Usage:
    # Show what's in each checkpoint (dry run)
    python scripts/zip_final_checkpoints.py --dry-run

    # Zip final checkpoints only
    python scripts/zip_final_checkpoints.py --output checkpoints_final.zip

    # Zip specific checkpoints
    python scripts/zip_final_checkpoints.py --checkpoints rad_vqa_qwen3vl_8b,slake_qwen3vl_8b
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path
from typing import List, Dict, Tuple
import re


def parse_args():
    parser = argparse.ArgumentParser(description="Zip final checkpoints only")
    parser.add_argument("--checkpoint_base", type=str, default="./checkpoints",
                       help="Base directory containing checkpoints")
    parser.add_argument("--checkpoints", type=str, default=None,
                       help="Comma-separated checkpoint names (default: all)")
    parser.add_argument("--output", type=str, default="checkpoints_final.zip",
                       help="Output zip filename")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be zipped without zipping")
    parser.add_argument("--include-optimizer", action="store_true",
                       help="Include optimizer states (larger files)")
    return parser.parse_args()


def get_dir_size(path: Path) -> int:
    """Get total size of directory in bytes."""
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def format_size(size_bytes: int) -> str:
    """Format size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def find_checkpoint_subdirs(ckpt_dir: Path) -> List[Path]:
    """Find checkpoint-N subdirectories sorted by number."""
    subdirs = []
    for item in ckpt_dir.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            # Extract number for proper sorting
            match = re.search(r'checkpoint-(\d+)', item.name)
            if match:
                subdirs.append((int(match.group(1)), item))
    
    # Sort by checkpoint number and return paths
    subdirs.sort(key=lambda x: x[0])
    return [p for _, p in subdirs]


def get_essential_files(directory: Path, include_optimizer: bool = False) -> List[Path]:
    """Get list of essential files to include from a directory."""
    essential_patterns = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "config.json",
        "experiment_metadata.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "preprocessor_config.json",
        "chat_template.json",
        "generation_config.json",
    ]
    
    if include_optimizer:
        essential_patterns.extend([
            "optimizer.pt",
            "scheduler.pt",
            "trainer_state.json",
            "training_args.bin",
        ])
    
    files = []
    for pattern in essential_patterns:
        file_path = directory / pattern
        if file_path.exists():
            files.append(file_path)
    
    return files


def analyze_checkpoint(ckpt_path: Path, include_optimizer: bool = False) -> Dict:
    """Analyze a checkpoint directory."""
    result = {
        "name": ckpt_path.name,
        "path": ckpt_path,
        "total_size": get_dir_size(ckpt_path),
        "checkpoint_subdirs": [],
        "final_checkpoint": None,
        "files_to_zip": [],
        "zip_size": 0,
    }
    
    # Find checkpoint subdirectories
    subdirs = find_checkpoint_subdirs(ckpt_path)
    result["checkpoint_subdirs"] = [s.name for s in subdirs]
    
    if subdirs:
        # Use the last checkpoint
        result["final_checkpoint"] = subdirs[-1]
        
        # Get files from final checkpoint
        final_files = get_essential_files(subdirs[-1], include_optimizer)
        result["files_to_zip"].extend(final_files)
    
    # Also get files from root directory
    root_files = get_essential_files(ckpt_path, include_optimizer)
    result["files_to_zip"].extend(root_files)
    
    # Calculate zip size
    result["zip_size"] = sum(f.stat().st_size for f in result["files_to_zip"])
    
    return result


def print_analysis(analyses: List[Dict]):
    """Print analysis summary."""
    print("\n" + "=" * 80)
    print("CHECKPOINT ANALYSIS")
    print("=" * 80)
    
    total_original = 0
    total_zip = 0
    
    for a in analyses:
        total_original += a["total_size"]
        total_zip += a["zip_size"]
        
        print(f"\n{a['name']}")
        print(f"  Total size:        {format_size(a['total_size'])}")
        print(f"  Checkpoint epochs: {len(a['checkpoint_subdirs'])}")
        if a["checkpoint_subdirs"]:
            print(f"    Available: {', '.join(a['checkpoint_subdirs'])}")
            if a["final_checkpoint"]:
                print(f"    Using:     {a['final_checkpoint'].name}")
        print(f"  Files to zip:      {len(a['files_to_zip'])}")
        print(f"  Zip size:          {format_size(a['zip_size'])}")
    
    print("\n" + "=" * 80)
    print(f"TOTAL ORIGINAL SIZE: {format_size(total_original)}")
    print(f"TOTAL ZIP SIZE:      {format_size(total_zip)}")
    print(f"SPACE SAVED:         {format_size(total_original - total_zip)} ({100*(1-total_zip/total_original):.1f}%)")
    print("=" * 80)


def create_zip(analyses: List[Dict], output_path: str, checkpoint_base: Path):
    """Create zip file with final checkpoints."""
    print(f"\nCreating zip: {output_path}")
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for a in analyses:
            ckpt_name = a["name"]
            
            for file_path in a["files_to_zip"]:
                # Calculate relative path within the zip
                rel_to_ckpt = file_path.relative_to(checkpoint_base)
                arcname = str(rel_to_ckpt)
                
                print(f"  Adding: {arcname}")
                zf.write(file_path, arcname)
    
    zip_size = os.path.getsize(output_path)
    print(f"\nCreated: {output_path} ({format_size(zip_size)})")


def main():
    args = parse_args()
    
    checkpoint_base = Path(args.checkpoint_base)
    
    if not checkpoint_base.exists():
        print(f"Checkpoint base not found: {checkpoint_base}")
        return
    
    # Get checkpoints to process
    if args.checkpoints:
        ckpt_names = [c.strip() for c in args.checkpoints.split(",")]
        ckpt_paths = [checkpoint_base / name for name in ckpt_names]
    else:
        ckpt_paths = [p for p in checkpoint_base.iterdir() if p.is_dir()]
    
    # Analyze each checkpoint
    analyses = []
    for ckpt_path in sorted(ckpt_paths):
        if not ckpt_path.exists():
            print(f"Warning: {ckpt_path} not found, skipping")
            continue
        analysis = analyze_checkpoint(ckpt_path, args.include_optimizer)
        analyses.append(analysis)
    
    # Print analysis
    print_analysis(analyses)
    
    # Create zip if not dry run
    if not args.dry_run:
        create_zip(analyses, args.output, checkpoint_base)
    else:
        print("\nDry run - no zip created. Remove --dry-run to create zip.")


if __name__ == "__main__":
    main()

"""
Utility functions for the Medical VQA framework.
"""

import os
import random
import json
from typing import Optional, Dict, Any, List, Union
from datetime import datetime

import numpy as np


def set_gpu(gpu_id: Union[int, List[int], str]) -> None:
    """Set which GPU(s) to use before importing torch.
    
    IMPORTANT: Call this BEFORE importing torch or any torch-dependent modules.
    
    Args:
        gpu_id: GPU index (int), list of indices, or comma-separated string
        
    Examples:
        set_gpu(0)           # Use GPU 0
        set_gpu(5)           # Use GPU 5
        set_gpu([0, 1])      # Use GPUs 0 and 1
        set_gpu("0,5,6")     # Use GPUs 0, 5, and 6
    """
    if isinstance(gpu_id, int):
        gpu_str = str(gpu_id)
    elif isinstance(gpu_id, list):
        gpu_str = ",".join(str(g) for g in gpu_id)
    else:
        gpu_str = str(gpu_id)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    print(f"[GPU] Set CUDA_VISIBLE_DEVICES={gpu_str}")


def get_available_gpus() -> List[int]:
    """Get list of available GPU indices.
    
    Returns:
        List of GPU indices
    """
    import torch
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


# Import torch after potential GPU setting
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory usage information.
    
    Returns:
        Dictionary with memory info in GB
    """
    if not torch.cuda.is_available():
        return {"available": False}
    
    return {
        "available": True,
        "num_gpus": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


def print_gpu_memory(prefix: str = "") -> None:
    """Print current GPU memory usage.
    
    Args:
        prefix: Optional prefix for the output
    """
    info = get_gpu_memory_info()
    
    if not info["available"]:
        print(f"{prefix}No GPU available")
        return
    
    print(f"{prefix}GPU Memory: {info['allocated_gb']:.2f} GB allocated, "
          f"{info['reserved_gb']:.2f} GB reserved")


def create_experiment_dir(
    base_dir: str,
    experiment_name: str,
    timestamp: bool = True,
) -> str:
    """Create an experiment output directory.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Name of the experiment
        timestamp: Whether to add timestamp to directory name
        
    Returns:
        Path to created directory
    """
    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{experiment_name}_{ts}"
    else:
        dir_name = experiment_name
    
    path = os.path.join(base_dir, dir_name)
    os.makedirs(path, exist_ok=True)
    
    return path


def save_json(data: Any, path: str, indent: int = 2) -> None:
    """Save data to JSON file.
    
    Args:
        data: Data to save
        path: Output path
        indent: JSON indentation
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(path: str) -> Any:
    """Load data from JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Loaded data
    """
    with open(path, "r") as f:
        return json.load(f)


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


class ExperimentLogger:
    """Simple experiment logger."""
    
    def __init__(self, output_dir: str, name: str = "experiment"):
        """Initialize logger.
        
        Args:
            output_dir: Directory for log files
            name: Experiment name
        """
        self.output_dir = output_dir
        self.name = name
        self.log_file = os.path.join(output_dir, f"{name}.log")
        self.metrics: Dict[str, Any] = {}
        
        os.makedirs(output_dir, exist_ok=True)
    
    def log(self, message: str, also_print: bool = True) -> None:
        """Log a message.
        
        Args:
            message: Message to log
            also_print: Whether to also print to console
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        
        with open(self.log_file, "a") as f:
            f.write(formatted + "\n")
        
        if also_print:
            print(formatted)
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics.
        
        Args:
            metrics: Dictionary of metric values
            step: Optional step number
        """
        if step is not None:
            self.metrics[step] = metrics
        else:
            self.metrics.update(metrics)
        
        # Format and log
        parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                 for k, v in metrics.items()]
        message = ", ".join(parts)
        
        if step is not None:
            message = f"Step {step}: {message}"
        
        self.log(message)
    
    def save_metrics(self) -> None:
        """Save all logged metrics to JSON."""
        path = os.path.join(self.output_dir, f"{self.name}_metrics.json")
        save_json(self.metrics, path)

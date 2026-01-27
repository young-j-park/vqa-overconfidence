"""
Medical VQA Training Framework

A modular framework for training and evaluating Vision-Language Models
on Medical Visual Question Answering tasks.

Supports:
- Multiple models: QwenVL3, InternVL3, LLaVA
- Multiple datasets: RAD-VQA, SLAKE (extensible)
- Training methods: SFT (supervised fine-tuning), GRPO (planned)
- Question type filtering: closed-only, open-only, mixed
- Dataset subsampling with seed control
"""

__version__ = "0.1.0"

"""TORCH-RaR training infrastructure for GRPO fine-tuning.

Uses TRL's GRPOTrainer with custom RaR reward functions to fine-tune
language models for Romanian toxicity classification.
"""

from torch_rar.training.config import GRPOTrainingConfig
from torch_rar.training.data_loader import load_training_dataset
from torch_rar.training.reward_function import RaRRewardFunction
from torch_rar.training.utils import extract_classification, format_prompt

__all__ = [
    "GRPOTrainingConfig",
    "load_training_dataset",
    "RaRRewardFunction",
    "extract_classification",
    "format_prompt",
]

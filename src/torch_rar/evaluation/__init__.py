"""TORCH-RaR evaluation infrastructure.

Evaluates trained models on Romanian toxicity classification
and computes metrics against baseline.
"""

from torch_rar.evaluation.evaluator import Evaluator
from torch_rar.evaluation.metrics import compute_metrics, print_comparison

__all__ = [
    "Evaluator",
    "compute_metrics",
    "print_comparison",
]

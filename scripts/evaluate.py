#!/usr/bin/env python3
"""Evaluation script for TORCH-RaR trained models.

Usage:
    # Evaluate trained model
    uv run python scripts/evaluate.py --model ./checkpoints/final

    # Evaluate baseline (untuned model)
    uv run python scripts/evaluate.py --model Qwen/Qwen3-8B --baseline

    # Compare baseline vs trained
    uv run python scripts/evaluate.py \
        --model ./checkpoints/final \
        --compare-baseline Qwen/Qwen3-8B

    # Quick test with fewer samples
    uv run python scripts/evaluate.py --model ./checkpoints/final --max-samples 20
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TORCH-RaR models on Romanian toxicity"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to model checkpoint or HuggingFace model name",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Treat --model as a standalone base model (no LoRA)",
    )
    parser.add_argument(
        "--compare-baseline",
        type=str,
        default=None,
        help="Base model name to compare against",
    )
    parser.add_argument(
        "--dataset",
        default="olimpia20/toxicity-dataset-ro-master",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of evaluation samples",
    )
    parser.add_argument(
        "--output",
        default="./output/eval_results.json",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3-8B",
        help="Base model name (for loading LoRA adapter)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    from torch_rar.training.config import GRPOTrainingConfig
    from torch_rar.evaluation.evaluator import Evaluator
    from torch_rar.evaluation.metrics import print_comparison

    config = GRPOTrainingConfig(base_model=args.base_model)

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device_name}")

    # Run baseline comparison if requested
    baseline_metrics = None
    if args.compare_baseline:
        print(f"\nEvaluating baseline: {args.compare_baseline}")
        print("-" * 40)
        baseline_eval = Evaluator(
            args.compare_baseline, config, is_baseline=True
        )
        baseline_metrics, baseline_results = baseline_eval.evaluate(
            dataset_name=args.dataset,
            split=args.split,
            max_samples=args.max_samples,
        )
        baseline_eval.save_results(
            baseline_metrics,
            baseline_results,
            args.output.replace(".json", "_baseline.json"),
        )
        print(f"Baseline accuracy: {baseline_metrics.accuracy:.1%}")
        print(f"Baseline F1:       {baseline_metrics.f1:.1%}")

        # Free GPU memory
        del baseline_eval
        torch.cuda.empty_cache()

    # Evaluate target model
    print(f"\nEvaluating model: {args.model}")
    print("-" * 40)
    evaluator = Evaluator(
        args.model, config, is_baseline=args.baseline
    )
    metrics, results = evaluator.evaluate(
        dataset_name=args.dataset,
        split=args.split,
        max_samples=args.max_samples,
    )
    evaluator.save_results(metrics, results, args.output)

    print(f"\nAccuracy:            {metrics.accuracy:.1%}")
    print(f"F1 Score:            {metrics.f1:.1%}")
    print(f"Precision:           {metrics.precision:.1%}")
    print(f"Recall:              {metrics.recall:.1%}")
    print(f"False Positive Rate: {metrics.false_positive_rate:.1%}")
    print(f"False Negative Rate: {metrics.false_negative_rate:.1%}")
    print(f"No Classification:   {metrics.no_classification}/{metrics.total}")

    # Print comparison if baseline was evaluated
    if baseline_metrics:
        print_comparison(baseline_metrics, metrics)


if __name__ == "__main__":
    main()

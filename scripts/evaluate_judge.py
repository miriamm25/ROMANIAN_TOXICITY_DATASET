#!/usr/bin/env python3
"""Evaluate the judge model (Qwen3-32B via Ollama) as a direct classifier.

This measures the "upper bound" — how well the judge itself classifies
Romanian toxicity. Useful for papers to show the gap between
student (8B) and teacher/judge (32B).

Usage (on server where Ollama is running):
    uv run python scripts/evaluate_judge.py
    uv run python scripts/evaluate_judge.py --max-samples 20  # quick test
"""

import argparse
import json
import re
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from datasets import load_dataset

from torch_rar.training.utils import extract_classification, label_to_int
from torch_rar.evaluation.metrics import ClassificationMetrics, compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate judge model as direct classifier"
    )
    parser.add_argument(
        "--judge-url",
        default="http://localhost:11434/v1",
        help="Ollama API base URL",
    )
    parser.add_argument(
        "--judge-model",
        default="qwen3:32b",
        help="Judge model name in Ollama",
    )
    parser.add_argument(
        "--dataset",
        default="olimpia20/toxicity-dataset-ro-master",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--split",
        default="test",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--output",
        default="./output/eval_results_judge.json",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout per request in seconds",
    )
    return parser.parse_args()


def classify_with_judge(
    text: str,
    session: requests.Session,
    judge_url: str,
    judge_model: str,
    timeout: int,
) -> str:
    """Send a Romanian text to the judge for direct classification.

    Returns the full completion text.
    """
    prompt = (
        "Analizează următorul text românesc și stabilește dacă este "
        "toxic sau non-toxic.\n"
        "Explică raționamentul tău pas cu pas, apoi oferă "
        "clasificarea finală.\n\n"
        f"Text: {text}\n\n"
        "Răspunde cu raționamentul tău urmat de clasificarea finală: "
        "TOXIC sau NON-TOXIC."
    )

    payload = {
        "model": judge_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6,
        "max_tokens": 2048,
    }

    response = session.post(
        f"{judge_url}/chat/completions",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()

    result = response.json()
    content = result["choices"][0]["message"]["content"]
    return content


def main():
    args = parse_args()
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})

    # Verify Ollama is reachable
    try:
        r = session.get(f"{args.judge_url}/models", timeout=5)
        print(f"Ollama OK: {args.judge_url}")
    except Exception as e:
        print(f"Cannot reach Ollama at {args.judge_url}: {e}")
        sys.exit(1)

    # Load dataset
    print(f"Loading dataset: {args.dataset} [{args.split}]")
    dataset = load_dataset(args.dataset, split=args.split)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    # Find columns
    text_col = None
    label_col = None
    for col in dataset.column_names:
        if col.lower() in ("text", "content", "comment"):
            text_col = col
        if col.lower() in ("label", "labels", "toxic", "toxicity"):
            label_col = col
    if not text_col or not label_col:
        print(f"Cannot find text/label columns in {dataset.column_names}")
        sys.exit(1)

    print(f"Evaluating {len(dataset)} samples with {args.judge_model}...")
    print("-" * 50)

    predictions = []
    labels = []
    detailed_results = []

    for i, sample in enumerate(dataset):
        text = str(sample[text_col]).strip()
        label = label_to_int(sample[label_col])

        if not text or label is None:
            continue

        try:
            completion = classify_with_judge(
                text, session, args.judge_url, args.judge_model, args.timeout
            )
        except Exception as e:
            print(f"  [{i+1}] ERROR: {e}")
            completion = ""

        # Strip <think> blocks before extracting classification
        clean_completion = re.sub(
            r"<think>.*?</think>", "", completion, flags=re.DOTALL
        ).strip()

        predicted_class = extract_classification(clean_completion or completion)
        predicted_label = (
            1 if predicted_class == "TOXIC"
            else 0 if predicted_class == "NON-TOXIC"
            else None
        )

        predictions.append(predicted_label)
        labels.append(label)

        detailed_results.append({
            "index": i,
            "text": text[:200],
            "label": label,
            "predicted": predicted_label,
            "predicted_class": predicted_class,
            "completion": completion[:2000],
            "correct": predicted_label == label,
        })

        if (i + 1) % 10 == 0:
            correct = sum(1 for r in detailed_results if r["correct"])
            print(
                f"  [{i+1}/{len(dataset)}] "
                f"Running accuracy: {correct / len(detailed_results):.1%}"
            )

    # Compute metrics
    metrics = compute_metrics(predictions, labels)

    # Save results
    output = {
        "metrics": {
            "accuracy": metrics.accuracy,
            "f1": metrics.f1,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "false_positive_rate": metrics.false_positive_rate,
            "false_negative_rate": metrics.false_negative_rate,
            "total": metrics.total,
            "correct": metrics.correct,
            "no_classification": metrics.no_classification,
        },
        "model": args.judge_model,
        "eval_type": "judge_as_classifier",
        "results": detailed_results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    total_evaluated = metrics.total + metrics.no_classification
    print(f"\nResults saved to {args.output}")
    print(f"\n{'='*50}")
    print(f"Judge Model ({args.judge_model}) as Classifier")
    print(f"{'='*50}")
    print(f"Accuracy:            {metrics.accuracy:.1%}")
    print(f"F1 Score:            {metrics.f1:.1%}")
    print(f"Precision:           {metrics.precision:.1%}")
    print(f"Recall:              {metrics.recall:.1%}")
    print(f"False Positive Rate: {metrics.false_positive_rate:.1%}")
    print(f"False Negative Rate: {metrics.false_negative_rate:.1%}")
    print(f"Classified:          {metrics.total}/{total_evaluated}")
    print(f"No Classification:   {metrics.no_classification}/{total_evaluated}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

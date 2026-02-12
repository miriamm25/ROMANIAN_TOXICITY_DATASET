"""Evaluation metrics for Romanian toxicity classification."""

from dataclasses import dataclass


@dataclass
class ClassificationMetrics:
    """Complete classification metrics.

    Attributes:
        accuracy: Overall correct / total.
        f1: Harmonic mean of precision and recall.
        precision: True positives / (true positives + false positives).
        recall: True positives / (true positives + false negatives).
        false_positive_rate: FP / (FP + TN) — legitimate criticism marked toxic.
        false_negative_rate: FN / (FN + TP) — implicit toxicity missed.
        total: Total number of samples evaluated.
        correct: Number of correct predictions.
        no_classification: Samples where model didn't produce TOXIC/NON-TOXIC.
    """

    accuracy: float
    f1: float
    precision: float
    recall: float
    false_positive_rate: float
    false_negative_rate: float
    total: int
    correct: int
    no_classification: int


def compute_metrics(
    predictions: list[int | None],
    labels: list[int],
) -> ClassificationMetrics:
    """Compute classification metrics from predictions and ground truth.

    Args:
        predictions: Model predictions (1=toxic, 0=non-toxic, None=no output).
        labels: Ground truth labels (1=toxic, 0=non-toxic).

    Returns:
        ClassificationMetrics with all computed values.
    """
    tp = fp = tn = fn = 0
    no_class = 0

    for pred, label in zip(predictions, labels):
        if pred is None:
            no_class += 1
            # Excluded from confusion matrix — reported separately
            continue

        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        elif pred == 0 and label == 1:
            fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return ClassificationMetrics(
        accuracy=accuracy,
        f1=f1,
        precision=precision,
        recall=recall,
        false_positive_rate=fpr,
        false_negative_rate=fnr,
        total=total,
        correct=tp + tn,
        no_classification=no_class,
    )


def print_comparison(
    baseline: ClassificationMetrics,
    trained: ClassificationMetrics,
) -> None:
    """Print side-by-side comparison of baseline vs trained model.

    Args:
        baseline: Metrics from the untuned base model.
        trained: Metrics from the GRPO-trained model.
    """
    def delta(new: float, old: float) -> str:
        diff = new - old
        sign = "+" if diff >= 0 else ""
        return f"{sign}{diff:.1%}"

    print("\n" + "=" * 60)
    print("TORCH-RaR GRPO Training Results")
    print("=" * 60)
    print(f"{'Metric':<25} {'Baseline':>10} {'Trained':>10} {'Delta':>10}")
    print("-" * 60)
    print(f"{'Accuracy':<25} {baseline.accuracy:>10.1%} {trained.accuracy:>10.1%} {delta(trained.accuracy, baseline.accuracy):>10}")
    print(f"{'F1 Score':<25} {baseline.f1:>10.1%} {trained.f1:>10.1%} {delta(trained.f1, baseline.f1):>10}")
    print(f"{'Precision':<25} {baseline.precision:>10.1%} {trained.precision:>10.1%} {delta(trained.precision, baseline.precision):>10}")
    print(f"{'Recall':<25} {baseline.recall:>10.1%} {trained.recall:>10.1%} {delta(trained.recall, baseline.recall):>10}")
    print(f"{'False Positive Rate':<25} {baseline.false_positive_rate:>10.1%} {trained.false_positive_rate:>10.1%} {delta(trained.false_positive_rate, baseline.false_positive_rate):>10}")
    print(f"{'False Negative Rate':<25} {baseline.false_negative_rate:>10.1%} {trained.false_negative_rate:>10.1%} {delta(trained.false_negative_rate, baseline.false_negative_rate):>10}")
    print(f"{'No Classification':<25} {baseline.no_classification:>10} {trained.no_classification:>10}")
    print(f"{'Total Samples':<25} {baseline.total:>10} {trained.total:>10}")
    print("=" * 60)

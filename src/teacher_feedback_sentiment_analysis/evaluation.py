"""Evaluation utilities for classification outputs."""

from __future__ import annotations

from collections.abc import Sequence


def evaluate_predictions(
    true_labels: Sequence[str], predicted_labels: Sequence[str]
) -> dict[str, float | int]:
    """Compute lightweight evaluation metrics."""
    if len(true_labels) != len(predicted_labels):
        raise ValueError(
            "true_labels and predicted_labels must have the same length."
        )
    if not true_labels:
        raise ValueError("Cannot evaluate empty label sequences.")

    labels = sorted(set(true_labels) | set(predicted_labels))
    accuracy = _accuracy(true_labels, predicted_labels)
    macro_precision, macro_recall, macro_f1 = _macro_scores(
        true_labels, predicted_labels, labels
    )

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "support": len(true_labels),
    }


def _accuracy(true_labels: Sequence[str], predicted_labels: Sequence[str]) -> float:
    matches = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
    return matches / len(true_labels)


def _macro_scores(
    true_labels: Sequence[str], predicted_labels: Sequence[str], labels: Sequence[str]
) -> tuple[float, float, float]:
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []

    for label in labels:
        tp = sum(
            1
            for true, pred in zip(true_labels, predicted_labels)
            if true == label and pred == label
        )
        fp = sum(
            1
            for true, pred in zip(true_labels, predicted_labels)
            if true != label and pred == label
        )
        fn = sum(
            1
            for true, pred in zip(true_labels, predicted_labels)
            if true == label and pred != label
        )

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    count = len(labels)
    return (
        sum(precisions) / count,
        sum(recalls) / count,
        sum(f1_scores) / count,
    )

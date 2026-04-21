"""Baseline models for sentiment classification."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from .types import FeedbackRecord


class MajorityClassBaseline:
    """Predict the most frequent class observed during training."""

    def __init__(self) -> None:
        self.majority_label: str | None = None

    def fit(self, records: Sequence[FeedbackRecord]) -> "MajorityClassBaseline":
        """Train baseline model by selecting the most common label."""
        return self.fit_labels([record.label for record in records])

    def fit_labels(self, labels: Sequence[str]) -> "MajorityClassBaseline":
        """Train baseline model from a label sequence."""
        if not labels:
            raise ValueError("Cannot fit MajorityClassBaseline with empty labels.")

        label_counts = Counter(labels)
        # Deterministic tie-breaker by label name after descending count.
        self.majority_label = sorted(
            label_counts.items(), key=lambda item: (-item[1], item[0])
        )[0][0]
        return self

    def predict_texts(self, texts: Sequence[str]) -> list[str]:
        """Predict labels for input texts."""
        if self.majority_label is None:
            raise RuntimeError("Model must be fitted before prediction.")
        return [self.majority_label for _ in texts]

    def predict_labels(self, sample_count: int) -> list[str]:
        """Predict labels for a fixed number of samples."""
        if sample_count < 0:
            raise ValueError("sample_count must be non-negative.")
        if self.majority_label is None:
            raise RuntimeError("Model must be fitted before prediction.")
        return [self.majority_label for _ in range(sample_count)]

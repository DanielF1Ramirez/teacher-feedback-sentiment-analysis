"""Shared data types for the sentiment analysis package."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeedbackRecord:
    """A single labeled teacher feedback comment."""

    text: str
    label: str


@dataclass(frozen=True)
class DatasetSplit:
    """Train/test split for text classification."""

    train_texts: list[str]
    test_texts: list[str]
    train_labels: list[str]
    test_labels: list[str]


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for reproducible model selection experiments."""

    text_column: str = "text"
    label_column: str = "label"
    test_size: float = 0.4
    random_state: int = 42
    selection_metric: str = "macro_f1"

"""Core package for teacher feedback sentiment analysis."""

from .baseline import MajorityClassBaseline
from .data_loading import load_feedback_csv, records_to_dataset, summarize_label_distribution
from .evaluation import evaluate_predictions
from .pipeline import (
    prepare_dataset_from_csv,
    run_majority_baseline_from_csv,
    run_model_selection_experiment_from_csv,
    split_dataset,
)
from .preprocessing import normalize_text, preprocess_records
from .types import DatasetSplit, ExperimentConfig, FeedbackRecord

__all__ = [
    "FeedbackRecord",
    "DatasetSplit",
    "ExperimentConfig",
    "MajorityClassBaseline",
    "load_feedback_csv",
    "records_to_dataset",
    "summarize_label_distribution",
    "normalize_text",
    "preprocess_records",
    "evaluate_predictions",
    "prepare_dataset_from_csv",
    "split_dataset",
    "run_majority_baseline_from_csv",
    "run_model_selection_experiment_from_csv",
]

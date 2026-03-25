"""Core package for teacher feedback sentiment analysis."""

from .baseline import MajorityClassBaseline
from .data_loading import load_feedback_csv
from .evaluation import evaluate_predictions
from .pipeline import run_majority_baseline_from_csv
from .preprocessing import normalize_text, preprocess_records
from .types import FeedbackRecord

__all__ = [
    "FeedbackRecord",
    "MajorityClassBaseline",
    "load_feedback_csv",
    "normalize_text",
    "preprocess_records",
    "evaluate_predictions",
    "run_majority_baseline_from_csv",
]

"""Reusable baseline pipeline utilities."""

from __future__ import annotations

from pathlib import Path

from .baseline import MajorityClassBaseline
from .data_loading import load_feedback_csv
from .evaluation import evaluate_predictions
from .preprocessing import preprocess_records


def run_majority_baseline_from_csv(
    csv_path: str | Path,
    text_column: str = "text",
    label_column: str = "label",
) -> dict[str, float | int | str]:
    """Execute load -> preprocess -> train baseline -> evaluate on one dataset."""
    records = load_feedback_csv(
        path=csv_path,
        text_column=text_column,
        label_column=label_column,
    )
    prepared_records = preprocess_records(records)

    model = MajorityClassBaseline().fit(prepared_records)
    true_labels = [record.label for record in prepared_records]
    predicted_labels = model.predict_texts([record.text for record in prepared_records])

    metrics = evaluate_predictions(true_labels, predicted_labels)
    metrics["majority_label"] = model.majority_label or ""
    return metrics

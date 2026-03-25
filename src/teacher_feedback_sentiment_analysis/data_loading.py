"""Data loading utilities."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from .types import FeedbackRecord


def load_feedback_csv(
    path: str | Path,
    text_column: str = "text",
    label_column: str = "label",
    encoding: str = "utf-8",
) -> list[FeedbackRecord]:
    """Load labeled feedback rows from a CSV file."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle)
        _validate_columns(reader.fieldnames, text_column, label_column)

        records: list[FeedbackRecord] = []
        for row_index, row in enumerate(reader, start=2):
            text_value = (row.get(text_column) or "").strip()
            label_value = (row.get(label_column) or "").strip()
            if not text_value or not label_value:
                raise ValueError(
                    f"Missing required values at row {row_index}: "
                    f"{text_column}='{text_value}', {label_column}='{label_value}'."
                )
            records.append(FeedbackRecord(text=text_value, label=label_value))

    if not records:
        raise ValueError(f"No valid rows found in CSV file: {csv_path}")

    return records


def _validate_columns(
    fieldnames: Iterable[str] | None, text_column: str, label_column: str
) -> None:
    if fieldnames is None:
        raise ValueError("CSV file does not contain a header row.")

    columns = {column.strip() for column in fieldnames if column}
    missing = [column for column in (text_column, label_column) if column not in columns]
    if missing:
        raise ValueError(f"CSV file is missing required columns: {missing}")

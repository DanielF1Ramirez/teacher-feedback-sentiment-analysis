"""Train and evaluate classical NLP models on a CSV dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = _project_root()
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from teacher_feedback_sentiment_analysis.pipeline import (  # noqa: E402
    run_model_selection_experiment_from_csv,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse script arguments."""
    parser = argparse.ArgumentParser(
        description="Train, compare, and tune lightweight classical NLP models."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "sample" / "teacher_feedback_sample.csv",
        help="Input CSV path with text and label columns.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "reports" / "metrics" / "model_selection_metrics.json",
        help="Output JSON path for experiment metrics.",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Name of text column in input CSV.",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Name of label column in input CSV.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.4,
        help="Fraction of data assigned to the test split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used by the split and tunable models.",
    )
    parser.add_argument(
        "--selection-metric",
        default="macro_f1",
        help="Metric used to rank the best model.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Execute the model selection workflow and persist metrics."""
    args = parse_args(argv)
    metrics = run_model_selection_experiment_from_csv(
        csv_path=args.input_csv,
        text_column=args.text_column,
        label_column=args.label_column,
        test_size=args.test_size,
        random_state=args.random_state,
        selection_metric=args.selection_metric,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Saved experiment metrics to: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

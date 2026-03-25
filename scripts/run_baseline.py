"""Run the baseline sentiment pipeline on a CSV dataset."""

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
    run_majority_baseline_from_csv,
)


def parse_args() -> argparse.Namespace:
    """Parse script arguments."""
    parser = argparse.ArgumentParser(
        description="Run the majority-class baseline pipeline on a CSV file."
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
        default=PROJECT_ROOT / "reports" / "metrics" / "baseline_metrics.json",
        help="Output JSON path for evaluation metrics.",
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
    return parser.parse_args()


def main() -> int:
    """Execute baseline pipeline and persist metrics."""
    args = parse_args()
    metrics = run_majority_baseline_from_csv(
        csv_path=args.input_csv,
        text_column=args.text_column,
        label_column=args.label_column,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Saved metrics to: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

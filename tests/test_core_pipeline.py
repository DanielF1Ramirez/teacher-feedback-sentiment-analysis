"""Lightweight tests for preprocessing, evaluation, and pipeline smoke flow."""

from __future__ import annotations

from pathlib import Path
import unittest

from teacher_feedback_sentiment_analysis.evaluation import evaluate_predictions
from teacher_feedback_sentiment_analysis.pipeline import run_majority_baseline_from_csv
from teacher_feedback_sentiment_analysis.preprocessing import normalize_text


class TestPreprocessing(unittest.TestCase):
    def test_normalize_text_applies_expected_cleanup(self) -> None:
        value = "  Great, Teacher!!!   "
        self.assertEqual(normalize_text(value), "great teacher")


class TestEvaluation(unittest.TestCase):
    def test_evaluate_predictions_returns_expected_macro_scores(self) -> None:
        true_labels = ["positive", "positive", "negative", "negative"]
        pred_labels = ["positive", "negative", "negative", "positive"]

        metrics = evaluate_predictions(true_labels, pred_labels)

        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["macro_precision"], 0.5)
        self.assertAlmostEqual(metrics["macro_recall"], 0.5)
        self.assertAlmostEqual(metrics["macro_f1"], 0.5)
        self.assertEqual(metrics["support"], 4)


class TestPipelineSmoke(unittest.TestCase):
    def test_run_majority_baseline_from_sample_csv(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        sample_csv = project_root / "data" / "sample" / "teacher_feedback_sample.csv"

        metrics = run_majority_baseline_from_csv(sample_csv)

        self.assertEqual(metrics["support"], 7)
        self.assertEqual(metrics["majority_label"], "positive")
        self.assertIn("accuracy", metrics)
        self.assertIn("macro_f1", metrics)


if __name__ == "__main__":
    unittest.main()

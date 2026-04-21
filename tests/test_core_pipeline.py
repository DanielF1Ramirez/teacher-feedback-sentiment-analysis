"""Tests for data preparation, training, evaluation, and CLI entrypoints."""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from teacher_feedback_sentiment_analysis.baseline import MajorityClassBaseline
from teacher_feedback_sentiment_analysis.data_loading import (
    load_feedback_csv,
    records_to_dataset,
)
from teacher_feedback_sentiment_analysis.evaluation import evaluate_predictions
from teacher_feedback_sentiment_analysis.pipeline import (
    prepare_dataset_from_csv,
    run_majority_baseline_from_csv,
    run_model_selection_experiment_from_csv,
    split_dataset,
)
from teacher_feedback_sentiment_analysis.preprocessing import normalize_text, preprocess_records


def _load_script_module(module_name: str, relative_path: str):
    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module from: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class TestDataLoading(unittest.TestCase):
    def test_load_feedback_csv_reads_expected_rows(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        sample_csv = project_root / "data" / "sample" / "teacher_feedback_sample.csv"

        records = load_feedback_csv(sample_csv)

        self.assertEqual(len(records), 7)
        self.assertEqual(records[0].label, "positive")

    def test_load_feedback_csv_raises_for_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_feedback_csv("missing.csv")

    def test_load_feedback_csv_raises_for_missing_columns(self) -> None:
        csv_path = Path("bad.csv")
        with patch.object(Path, "exists", return_value=True), patch.object(
            Path,
            "open",
            return_value=io.StringIO("text,target\nhello,positive\n"),
        ):
            with self.assertRaises(ValueError):
                load_feedback_csv(csv_path)

    def test_load_feedback_csv_raises_for_missing_row_values(self) -> None:
        csv_path = Path("bad.csv")
        with patch.object(Path, "exists", return_value=True), patch.object(
            Path,
            "open",
            return_value=io.StringIO("text,label\nhello,\n"),
        ):
            with self.assertRaises(ValueError):
                load_feedback_csv(csv_path)


class TestPreprocessing(unittest.TestCase):
    def test_normalize_text_applies_expected_cleanup(self) -> None:
        value = "  Great, Teacher!!!   "
        self.assertEqual(normalize_text(value), "great teacher")

    def test_prepare_dataset_from_csv_returns_normalized_features(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        sample_csv = project_root / "data" / "sample" / "teacher_feedback_sample.csv"

        texts, labels = prepare_dataset_from_csv(sample_csv)

        self.assertEqual(texts[0], "the teacher explains concepts very clearly")
        self.assertEqual(labels[0], "positive")

    def test_records_to_dataset_rejects_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            records_to_dataset([])

    def test_preprocess_records_lowercases_labels(self) -> None:
        records = load_feedback_csv(
            Path(__file__).resolve().parents[1]
            / "data"
            / "sample"
            / "teacher_feedback_sample.csv"
        )

        processed = preprocess_records(records)

        self.assertTrue(all(record.label == record.label.lower() for record in processed))


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
        self.assertEqual(metrics["labels"], ["negative", "positive"])
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 1]])


class TestBaseline(unittest.TestCase):
    def test_majority_baseline_requires_fit_before_predict(self) -> None:
        model = MajorityClassBaseline()
        with self.assertRaises(RuntimeError):
            model.predict_texts(["hello"])

    def test_majority_baseline_fit_labels_sets_majority_label(self) -> None:
        model = MajorityClassBaseline().fit_labels(["positive", "negative", "positive"])
        self.assertEqual(model.majority_label, "positive")


class TestTrainingPipeline(unittest.TestCase):
    def test_split_dataset_is_reproducible(self) -> None:
        texts = [f"text {index}" for index in range(8)]
        labels = [
            "positive",
            "positive",
            "positive",
            "positive",
            "negative",
            "negative",
            "negative",
            "negative",
        ]

        first_split = split_dataset(texts, labels, test_size=0.25, random_state=7)
        second_split = split_dataset(texts, labels, test_size=0.25, random_state=7)

        self.assertEqual(first_split, second_split)

    def test_run_majority_baseline_from_sample_csv(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        sample_csv = project_root / "data" / "sample" / "teacher_feedback_sample.csv"

        metrics = run_majority_baseline_from_csv(sample_csv)

        self.assertEqual(metrics["support"], 7)
        self.assertEqual(metrics["majority_label"], "positive")
        self.assertIn("accuracy", metrics)
        self.assertIn("macro_f1", metrics)

    def test_run_model_selection_experiment_returns_ranked_models(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        sample_csv = project_root / "data" / "sample" / "teacher_feedback_sample.csv"

        experiment = run_model_selection_experiment_from_csv(sample_csv)

        self.assertEqual(experiment["configuration"]["selection_metric"], "macro_f1")
        self.assertGreaterEqual(len(experiment["models"]), 4)
        self.assertEqual(experiment["selected_model"], experiment["models"][0])
        model_names = [model["model_name"] for model in experiment["models"]]
        self.assertIn("majority_class_baseline", model_names)
        self.assertIn("logistic_regression", model_names)
        self.assertIn("linear_svc", model_names)
        self.assertIn("multinomial_nb", model_names)
        self.assertIn("metrics", experiment["selected_model"])


class TestScripts(unittest.TestCase):
    def test_run_experiment_script_writes_json_report(self) -> None:
        module = _load_script_module("run_experiment", "scripts/run_experiment.py")
        output_path = Path("metrics.json")
        with patch.object(Path, "mkdir", return_value=None), patch.object(
            Path,
            "write_text",
        ) as mocked_write:
            exit_code = module.main(["--output-json", str(output_path)])

        self.assertEqual(exit_code, 0)
        payload = json.loads(mocked_write.call_args.args[0])
        self.assertIn("selected_model", payload)
        self.assertIn("models", payload)

    def test_run_baseline_script_writes_json_report(self) -> None:
        module = _load_script_module("run_baseline", "scripts/run_baseline.py")
        output_path = Path("baseline.json")
        with patch.object(Path, "mkdir", return_value=None), patch.object(
            Path,
            "write_text",
        ) as mocked_write:
            exit_code = module.main(["--output-json", str(output_path)])

        self.assertEqual(exit_code, 0)
        payload = json.loads(mocked_write.call_args.args[0])
        self.assertEqual(payload["majority_label"], "positive")


if __name__ == "__main__":
    unittest.main()

"""Reusable training and model selection pipeline utilities."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from .baseline import MajorityClassBaseline
from .data_loading import (
    load_feedback_csv,
    records_to_dataset,
    summarize_label_distribution,
)
from .evaluation import evaluate_predictions
from .modeling import build_candidate_pipelines, build_tuning_grids
from .preprocessing import preprocess_records
from .types import DatasetSplit, ExperimentConfig


def prepare_dataset_from_csv(
    csv_path: str | Path,
    text_column: str = "text",
    label_column: str = "label",
) -> tuple[list[str], list[str]]:
    """Load, preprocess, and extract text/label sequences from CSV data."""
    records = load_feedback_csv(
        path=csv_path,
        text_column=text_column,
        label_column=label_column,
    )
    prepared_records = preprocess_records(records)
    return records_to_dataset(prepared_records)


def split_dataset(
    texts: list[str],
    labels: list[str],
    test_size: float = 0.4,
    random_state: int = 42,
) -> DatasetSplit:
    """Create a reproducible stratified split for text classification."""
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length.")
    if not texts:
        raise ValueError("Cannot split an empty dataset.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    label_counts = Counter(labels)
    if len(label_counts) < 2:
        raise ValueError("At least two classes are required for model selection.")
    if min(label_counts.values()) < 2:
        raise ValueError(
            "Each class must have at least two samples for a stratified split."
        )

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    return DatasetSplit(
        train_texts=list(train_texts),
        test_texts=list(test_texts),
        train_labels=list(train_labels),
        test_labels=list(test_labels),
    )


def run_majority_baseline_from_csv(
    csv_path: str | Path,
    text_column: str = "text",
    label_column: str = "label",
) -> dict[str, Any]:
    """Execute load -> preprocess -> train baseline -> evaluate on one dataset."""
    texts, labels = prepare_dataset_from_csv(
        csv_path=csv_path,
        text_column=text_column,
        label_column=label_column,
    )
    model = MajorityClassBaseline().fit_labels(labels)
    predicted_labels = model.predict_texts(texts)

    metrics = evaluate_predictions(labels, predicted_labels)
    metrics["majority_label"] = model.majority_label or ""
    return metrics


def run_model_selection_experiment_from_csv(
    csv_path: str | Path,
    text_column: str = "text",
    label_column: str = "label",
    test_size: float = 0.4,
    random_state: int = 42,
    selection_metric: str = "macro_f1",
) -> dict[str, Any]:
    """Train, compare, and tune lightweight text classifiers from a CSV file."""
    config = ExperimentConfig(
        text_column=text_column,
        label_column=label_column,
        test_size=test_size,
        random_state=random_state,
        selection_metric=selection_metric,
    )
    texts, labels = prepare_dataset_from_csv(
        csv_path=csv_path,
        text_column=text_column,
        label_column=label_column,
    )
    split = split_dataset(
        texts=texts,
        labels=labels,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    model_results = [_evaluate_majority_baseline(split)]
    model_results.extend(
        _evaluate_candidate_models(split=split, random_state=config.random_state)
    )

    tuned_results, tuning_notes = _tune_top_models(
        split=split,
        model_results=model_results,
        selection_metric=config.selection_metric,
        random_state=config.random_state,
    )
    model_results.extend(tuned_results)
    ranked_results = _rank_results(model_results, config.selection_metric)
    best_result = ranked_results[0]

    notes = [
        (
            "This repository uses a very small sample dataset. Reported metrics are "
            "useful for reproducibility and comparison, not as a claim of broad "
            "generalization."
        )
    ]
    notes.extend(tuning_notes)

    return {
        "dataset": {
            "source_csv": str(Path(csv_path)),
            "total_samples": len(texts),
            "label_distribution": summarize_label_distribution(labels),
        },
        "configuration": {
            "text_column": config.text_column,
            "label_column": config.label_column,
            "test_size": config.test_size,
            "random_state": config.random_state,
            "selection_metric": config.selection_metric,
        },
        "split_summary": {
            "train_size": len(split.train_texts),
            "test_size": len(split.test_texts),
            "train_distribution": summarize_label_distribution(split.train_labels),
            "test_distribution": summarize_label_distribution(split.test_labels),
        },
        "models": ranked_results,
        "selected_model": best_result,
        "notes": notes,
    }


def _evaluate_majority_baseline(split: DatasetSplit) -> dict[str, Any]:
    model = MajorityClassBaseline().fit_labels(split.train_labels)
    predicted_labels = model.predict_texts(split.test_texts)
    metrics = evaluate_predictions(split.test_labels, predicted_labels)
    return {
        "model_name": "majority_class_baseline",
        "model_family": "baseline",
        "tuned": False,
        "best_params": {},
        "metrics": metrics,
        "metadata": {"majority_label": model.majority_label or ""},
    }


def _evaluate_candidate_models(
    split: DatasetSplit, random_state: int
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for model_name, pipeline in build_candidate_pipelines(random_state).items():
        pipeline.fit(split.train_texts, split.train_labels)
        predicted_labels = pipeline.predict(split.test_texts)
        results.append(
            {
                "model_name": model_name,
                "model_family": "classical_ml",
                "tuned": False,
                "best_params": {},
                "metrics": evaluate_predictions(split.test_labels, predicted_labels),
                "metadata": {},
            }
        )
    return results


def _tune_top_models(
    split: DatasetSplit,
    model_results: list[dict[str, Any]],
    selection_metric: str,
    random_state: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    tuning_grids = build_tuning_grids()
    folds = _cross_validation_folds(split.train_labels)
    if folds < 2:
        return [], [
            "Grid search was skipped because the training split is too small for "
            "stratified cross-validation."
        ]

    ranked_classical = [
        result
        for result in _rank_results(model_results, selection_metric)
        if result["model_family"] == "classical_ml"
    ]
    candidate_names = [result["model_name"] for result in ranked_classical[:2]]

    tuned_results: list[dict[str, Any]] = []
    for model_name in candidate_names:
        search = GridSearchCV(
            estimator=build_candidate_pipelines(random_state)[model_name],
            param_grid=tuning_grids[model_name],
            scoring="f1_macro",
            cv=StratifiedKFold(
                n_splits=folds,
                shuffle=True,
                random_state=random_state,
            ),
            n_jobs=None,
        )
        search.fit(split.train_texts, split.train_labels)
        predicted_labels = search.best_estimator_.predict(split.test_texts)
        tuned_results.append(
            {
                "model_name": f"{model_name}_tuned",
                "model_family": "classical_ml",
                "tuned": True,
                "best_params": dict(search.best_params_),
                "metrics": evaluate_predictions(split.test_labels, predicted_labels),
                "metadata": {
                    "base_model_name": model_name,
                    "cv_folds": folds,
                },
            }
        )

    return tuned_results, [
        (
            "Hyperparameter search is intentionally small because the tracked sample "
            "dataset is tiny."
        )
    ]


def _cross_validation_folds(labels: list[str]) -> int:
    label_counts = Counter(labels)
    return min(3, min(label_counts.values()))


def _rank_results(
    model_results: list[dict[str, Any]], selection_metric: str
) -> list[dict[str, Any]]:
    return sorted(
        model_results,
        key=lambda result: (
            -float(result["metrics"][selection_metric]),
            -float(result["metrics"]["accuracy"]),
            result["model_name"],
        ),
    )

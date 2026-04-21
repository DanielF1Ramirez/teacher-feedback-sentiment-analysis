"""Classical NLP model builders used by the training pipeline."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_candidate_pipelines(random_state: int) -> dict[str, Pipeline]:
    """Return candidate text classification pipelines."""
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=random_state,
                        solver="liblinear",
                    ),
                ),
            ]
        ),
        "linear_svc": Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer()),
                ("classifier", LinearSVC(random_state=random_state)),
            ]
        ),
        "multinomial_nb": Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer()),
                ("classifier", MultinomialNB()),
            ]
        ),
    }


def build_tuning_grids() -> dict[str, dict[str, list[object]]]:
    """Return deliberately small tuning grids for the strongest candidate models."""
    return {
        "logistic_regression": {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__sublinear_tf": [False, True],
            "classifier__C": [0.5, 1.0, 2.0],
        },
        "linear_svc": {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__sublinear_tf": [False, True],
            "classifier__C": [0.5, 1.0, 2.0],
        },
        "multinomial_nb": {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__sublinear_tf": [False, True],
            "classifier__alpha": [0.5, 1.0, 1.5],
        },
    }

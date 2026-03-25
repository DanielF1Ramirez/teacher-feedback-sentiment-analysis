# Current Status

## Date
2026-03-24

## Completed Work
- Baseline project configuration established:
  - `pyproject.toml` with package metadata and setuptools config (`src/` layout).
  - `requirements.txt` aligned to editable install (`-e .`) as dependency entrypoint.
  - `.gitignore` populated with Python/build/cache/environment ignores.
- Minimal modular NLP baseline implemented under `src/teacher_feedback_sentiment_analysis/`:
  - `types.py`: `FeedbackRecord` dataclass.
  - `data_loading.py`: CSV load + required-column/value validation.
  - `preprocessing.py`: deterministic text normalization and record preprocessing.
  - `baseline.py`: majority-class baseline classifier.
  - `evaluation.py`: accuracy + macro precision/recall/F1.
  - `pipeline.py`: reusable end-to-end baseline run from CSV.
  - `__init__.py`: public API exports.

## Stable Files and Modules
- Packaging/config: `pyproject.toml`, `requirements.txt`, `.gitignore`.
- Core baseline modules:
  - `src/teacher_feedback_sentiment_analysis/data_loading.py`
  - `src/teacher_feedback_sentiment_analysis/preprocessing.py`
  - `src/teacher_feedback_sentiment_analysis/baseline.py`
  - `src/teacher_feedback_sentiment_analysis/evaluation.py`
  - `src/teacher_feedback_sentiment_analysis/pipeline.py`
  - `src/teacher_feedback_sentiment_analysis/types.py`
- Context docs: `AGENTS.md`, `docs/project_context.md`.

## Remaining Gaps
- No CLI/entrypoint script yet in `scripts/` to run pipeline on repo data paths.
- No lightweight test suite in `tests/`.
- No CI workflow file in `.github/workflows/`.
- Repository documentation still incomplete (`README.md`, `CONTRIBUTING.md`, folder READMEs mostly empty).
- No inference-focused module/API contract beyond baseline pipeline helper.

## Deferred Decisions
- Final dataset contract for scripts/tests (exact sample CSV path and column names if different from defaults).
- Baseline extension strategy (keep stdlib-only vs. introduce `scikit-learn` for vectorized classical models).
- Output contract for metrics artifacts (console only vs. `reports/metrics/*.json` standard format).
- Tooling scope for CI (lint/test matrix and Python versions).

## Exact Next Recommended Step
- Create one script `scripts/run_baseline.py` that calls `run_majority_baseline_from_csv` on a small CSV in `data/sample/`, prints metrics, and saves them to `reports/metrics/baseline_metrics.json`.

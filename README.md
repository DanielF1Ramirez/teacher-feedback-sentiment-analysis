# Teacher Feedback Sentiment Analysis

Professional NLP repository for sentiment analysis over teacher feedback comments. The current delivery is a lightweight, reproducible classical ML workflow that compares a majority-class baseline against TF-IDF-based classifiers and selects the best model with `macro_f1`.

## Project Goal

This repository turns a small academic sentiment-analysis exercise into a portfolio-ready software project with:
- validated CSV ingestion,
- deterministic text normalization,
- reproducible train/test splitting,
- comparison of multiple classical models,
- constrained hyperparameter tuning,
- JSON experiment reports,
- automated tests and CI.

The tracked sample dataset is intentionally small. Reported metrics are useful for reproducibility and model comparison, not as a claim of production-grade generalization.

## Pipeline Overview

1. Load and validate a CSV dataset with `text` and `label` columns.
2. Normalize text and labels.
3. Create a stratified train/test split with a fixed `random_state`.
4. Evaluate the majority-class baseline.
5. Train and compare:
   - Logistic Regression
   - Linear SVC
   - Multinomial Naive Bayes
6. Tune the strongest candidates with a small grid search.
7. Select the best model by `macro_f1`.
8. Persist a JSON report under `reports/metrics/`.

## Repository Structure

```text
.
|-- data/
|   |-- sample/teacher_feedback_sample.csv
|-- docs/
|   |-- project_context.md
|   |-- current_status.md
|-- reports/
|   |-- metrics/
|   |-- README.md
|-- scripts/
|   |-- run_baseline.py
|   |-- run_experiment.py
|-- src/teacher_feedback_sentiment_analysis/
|   |-- baseline.py
|   |-- data_loading.py
|   |-- evaluation.py
|   |-- modeling.py
|   |-- pipeline.py
|   |-- preprocessing.py
|   |-- types.py
|-- tests/
|   |-- test_core_pipeline.py
```

## Installation

### Option A: `venv` + `pip`

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Option B: Conda

```bash
conda env create -f environment.yml
conda activate teacher-feedback-sentiment-analysis
```

## Run the Full Experiment

```bash
python scripts/run_experiment.py
```

Default input/output:
- input: `data/sample/teacher_feedback_sample.csv`
- output: `reports/metrics/model_selection_metrics.json`

Optional arguments:

```bash
python scripts/run_experiment.py \
  --input-csv data/sample/teacher_feedback_sample.csv \
  --output-json reports/metrics/model_selection_metrics.json \
  --text-column text \
  --label-column label \
  --test-size 0.4 \
  --random-state 42 \
  --selection-metric macro_f1
```

## Run the Baseline Only

```bash
python scripts/run_baseline.py
```

Default output:
- `reports/metrics/baseline_metrics.json`

## Expected JSON Outputs

`run_baseline.py` writes:
- `accuracy`
- `macro_precision`
- `macro_recall`
- `macro_f1`
- `support`
- `labels`
- `confusion_matrix`
- `majority_label`

`run_experiment.py` writes:
- dataset summary
- configuration
- split summary
- metrics for each evaluated model
- selected best model
- tuning notes and caveats

## Current Sample Results

From `reports/metrics/model_selection_metrics.json`:
- selected model: `linear_svc_tuned`
- selection metric: `macro_f1`
- test-split macro_f1: `1.0000`
- test-split accuracy: `1.0000`
- baseline macro_f1 on the same split: `0.2500`

From `reports/metrics/baseline_metrics.json` on the full sample:
- accuracy: `0.5714`
- macro_f1: `0.3636`
- macro_precision: `0.2857`
- macro_recall: `0.5000`
- majority_label: `positive`

## Data Contract

The training/evaluation scripts expect a CSV file with:
- a header row,
- one text column,
- one label column,
- no empty values in required fields.

By default the expected schema is:
- `text`
- `label`

The current version does not require environment variables.

## Testing

```bash
PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py" -v
```

The test suite covers:
- CSV validation and failure modes,
- preprocessing,
- evaluation metrics,
- reproducible splitting,
- baseline behavior,
- model-selection output shape,
- script-driven JSON report generation.

## Reproducibility Notes

- Main selection metric: `macro_f1`
- Default `random_state`: `42`
- Default `test_size`: `0.4`
- Hyperparameter search is intentionally small because the sample dataset is tiny.

## Limitations

- The tracked dataset is very small, so metric variance is high.
- This repository focuses on classical ML, not deep learning.
- The repo is designed for clarity, reproducibility, and professional presentation rather than benchmark maximization on a large corpus.
- The best score in the sample report comes from a holdout split with only three test examples, so it must be interpreted cautiously.

## CI

GitHub Actions installs dependencies and runs the unit test suite on push and pull request.

## License

This project is released under the MIT License. See `LICENSE`.

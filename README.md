# Teacher Feedback Sentiment Analysis

Professional NLP baseline repository for sentiment analysis on teacher feedback comments.

## Project Goal

This repository consolidates reusable code for a lightweight sentiment pipeline:
- load labeled feedback from CSV,
- preprocess text consistently,
- train a deterministic majority-class baseline,
- evaluate with fast classification metrics.

The goal is a clear, reproducible foundation that can be extended with richer classical ML models.

## Repository Structure

```text
.
|-- data/
|   |-- sample/teacher_feedback_sample.csv
|-- docs/
|   |-- project_context.md
|   |-- current_status.md
|-- reports/
|   |-- metrics/baseline_metrics.json
|-- scripts/
|   |-- run_baseline.py
|-- src/teacher_feedback_sentiment_analysis/
|   |-- data_loading.py
|   |-- preprocessing.py
|   |-- baseline.py
|   |-- evaluation.py
|   |-- pipeline.py
|-- tests/
|   |-- test_core_pipeline.py
```

## Quickstart

### Option A: `venv`

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

## Run Baseline Pipeline

```bash
python scripts/run_baseline.py
```

Default input/output:
- input: `data/sample/teacher_feedback_sample.csv`
- output: `reports/metrics/baseline_metrics.json`

## Run Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py" -v
```

## Current Baseline Metrics (Sample Data)

From `reports/metrics/baseline_metrics.json`:
- accuracy: `0.5714`
- macro_f1: `0.3636`
- macro_precision: `0.2857`
- macro_recall: `0.5000`
- majority_label: `positive`
- support: `7`

## CI

GitHub Actions workflow: `.github/workflows/ci.yml`  
Runs unit tests on push and pull request.

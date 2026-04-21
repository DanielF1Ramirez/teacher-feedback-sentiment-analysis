# Contributing

## Scope

This repository prioritizes small, reviewable updates to the teacher feedback sentiment analysis workflow.

## Development Workflow

1. Create focused changes.
2. Run fast local validations before committing:
   - `PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py" -v`
   - `python scripts/run_baseline.py` (when baseline behavior changes)
   - `python scripts/run_experiment.py` (when training, model selection, or reporting changes)
3. Keep commit messages in English and descriptive.

## Code Standards

- Keep reusable logic under `src/teacher_feedback_sentiment_analysis/`.
- Keep entrypoints in `scripts/`.
- Keep comments/docstrings in English.
- Avoid unnecessary dependencies and keep tuning lightweight.
- Do not introduce heavy training jobs as part of routine changes.

## Pull Request Checklist

- [ ] Changes are small and focused.
- [ ] Tests pass locally.
- [ ] Experiment script runs when model or reporting behavior changes.
- [ ] Documentation updated when behavior changes.
- [ ] No notebooks modified unless explicitly required.

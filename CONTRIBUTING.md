# Contributing

## Scope

This repository prioritizes small, reviewable updates to the teacher feedback sentiment baseline.

## Development Workflow

1. Create focused changes.
2. Run fast local validations before committing:
   - `PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py" -v`
   - `python scripts/run_baseline.py` (when script/data behavior changes)
3. Keep commit messages in English and descriptive.

## Code Standards

- Keep reusable logic under `src/teacher_feedback_sentiment_analysis/`.
- Keep entrypoints in `scripts/`.
- Keep comments/docstrings in English.
- Avoid unnecessary dependencies.
- Do not run heavy training jobs as part of routine changes.

## Pull Request Checklist

- [ ] Changes are small and focused.
- [ ] Tests pass locally.
- [ ] Documentation updated when behavior changes.
- [ ] No notebooks modified unless explicitly required.

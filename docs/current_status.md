# Current Status

## Date
2026-03-24

## Completed in this step
- Defined minimal packaging/build metadata in `pyproject.toml`.
- Established a single-source dependency approach via `requirements.txt` -> `-e .`.
- Added practical Python `.gitignore` entries to keep the repository clean.
- Implemented a minimal modular baseline flow in `src/`:
  - CSV loading and schema validation
  - text normalization preprocessing
  - majority-class baseline model
  - lightweight evaluation metrics
  - reusable end-to-end pipeline function

## Not done yet
- No entrypoint scripts under `scripts/`.
- No tests in `tests/`.
- No CI workflow files in `.github/workflows/`.
- Main documentation (`README.md`) remains to be authored.
- No dedicated inference module yet.

## Next suggested step
- Add one small runnable script in `scripts/` to execute the baseline pipeline against a sample CSV and save metrics.

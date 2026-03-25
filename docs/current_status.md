# Current Status

## Date
2026-03-24

## Completed in this step
- Defined minimal packaging/build metadata in `pyproject.toml`.
- Established a single-source dependency approach via `requirements.txt` -> `-e .`.
- Added practical Python `.gitignore` entries to keep the repository clean.

## Not done yet
- No preprocessing, training, evaluation, or inference modules implemented.
- No entrypoint scripts under `scripts/`.
- No tests in `tests/`.
- No CI workflow files in `.github/workflows/`.
- Main documentation (`README.md`) remains to be authored.

## Next suggested step
- Implement a minimal vertical slice in `src/` (load -> preprocess -> baseline model -> evaluate) and add one runnable script in `scripts/`.

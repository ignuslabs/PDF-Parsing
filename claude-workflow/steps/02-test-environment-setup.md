# 02 — Test Environment & Tooling

## Python & Packages
- Python >= 3.9
- Install with `pip install -r requirements.txt`
- Dev tooling: `pytest`, `pytest-cov`, `black`, `flake8`, `mypy`

## Commands
- `pytest -v` — all tests (verbose)
- `pytest tests/` — run tests folder
- Coverage: `pytest --cov=src --cov-report=term-missing`

## Local Settings
- Use virtualenv or Poetry.
- Provide `.env` for OCR and cache paths:
  ```env
  DOCLING_CACHE_DIR=./models_cache
  DOCLING_MAX_FILE_SIZE_MB=100
  DOCLING_MAX_PAGES=200
  OCR_LANGUAGE=eng
  ```

## Skips/Marks
- Use `@pytest.mark.slow` for OCR/table-heavy tests.
- Gate GPU/OS-specific tests with markers.

# 08 — CI Setup (pytest, coverage)

## GitHub Actions Example
```yaml
name: tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt
      - run: pip install -r requirements-dev.txt || true
      - run: pytest -v --maxfail=1 --disable-warnings --cov=src --cov-report=xml
```
- Mark heavy OCR/table tests as `@pytest.mark.slow` and gate with env var.

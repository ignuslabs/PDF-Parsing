Smart PDF Parser
================

Extract, search, verify, and export structured content from PDFs using Docling — with OCR, table structure, reading‑order awareness, and a streamlined Streamlit UI for quick workflows.

[![Docs (local)](https://img.shields.io/badge/docs-local-blue)](docs/index.md)
[![Documentation Status](https://pdf-parsing.readthedocs.io/en/latest/index.html)](https://pdf-parser.readthedocs.io/en/latest/?badge=latest)

Highlights
----------

- Intelligent parsing via Docling (OCR, tables, images, formulas)
- Precise bounding boxes and page numbers for every element
- Fast search (exact, fuzzy, semantic) with in-place verification overlay and rich export
- Test-driven design and fixtures for reliable iteration

Quick Start
-----------

- Install dependencies: `pip install -r requirements.txt`
- Run the app: `python run_app.py`
- Try a sample: open `tests/fixtures/text_simple.pdf` in the UI
- Use the **Search & Verify** workspace to find fields (IDs, amounts, dates) and mark them reviewed
◊
More details in the docs: `docs/index.md`.

Documentation
-------------

- Main docs: docs/index.md
- Design plan (KV extraction & header classification): docs/design/kv_extraction_implementation_plan.md

If you host on Read the Docs, update the badge/link above to your project slug, for example:

- Badge: `https://readthedocs.org/projects/<your-slug>/badge/?version=latest`
- Docs: `https://<your-slug>.readthedocs.io/en/latest/`

Features
--------

- Docling-powered parsing with optional OCR
- Table structure recognition (configurable accuracy vs. speed)
- Normalized bounding boxes for robust overlays and exports
- Streamlit UI: Parse, Search & Verify, Export
- JSON/CSV/Markdown/HTML exports

Development
-----------

- Run tests (unit/integration, excluding OCR/perf):
  - `pytest -v -m "not slow and not ocr and not performance"`
- Generate fixtures (PDF samples):
  - `python generate_test_fixtures.py`
- Optional OCR/performance suites:
  - `pytest -v -m ocr`
  - `pytest -v -m performance`

Key files:

- Parser: `src/core/parser.py`
- Data models: `src/core/models.py`
- Verification renderer: `src/verification/renderer.py`
- UI pages: `src/ui/pages/`

Tech Stack
----------

- Docling (document conversion, OCR/table/reading order)
- Streamlit (interactive UI)
- Pillow/OpenCV (images), pandas/numpy (data), pytest (tests)

Notes
-----

- Read the Docs: ensure `docs/` builds with your theme; see `.readthedocs.yaml` and `docs/conf.py`.
- If your Read the Docs project slug differs from `pdf-parser`, update the badge and link above.

# 03 — Test Fixtures (Sample PDFs)

Prepare **small** PDFs that target specific behaviors:

1. **text_simple.pdf** — single column text, headings, paragraphs.
2. **tables_basic.pdf** — 2 small tables (headers, merged cells).
3. **scanned_ocr_en.pdf** — scanned English page to validate OCR.
4. **multicolumn_rotated.pdf** — two-column layout, one rotated page.
5. **images_captions.pdf** — images with captions to test picture items.
6. **formulas_snippets.pdf** — includes math formulas.
7. **large_pages_light.pdf** — 20+ pages light content for perf sanity.

## Golden Outputs
For each fixture, store expected outputs under `tests/golden/`:
- `*.json` — `document.export_to_dict()`
- `*.md` — `document.export_to_markdown()`
- optional `*.html` — `export_to_html()`

Keep golden files **small** and **curated** to avoid churn.

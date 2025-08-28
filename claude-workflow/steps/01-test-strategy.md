# 01 — Test Strategy & Coverage Map

Goal: Define **what** to test before writing any code.

## Components Under Test
- **PDF Parser (Docling wrapper)**: layout analysis, OCR on/off, table extraction, page image generation.
- **Smart Search Engine**: exact/fuzzy/semantic matching, type filters, page filters, regional/coordinate search.
- **Interactive Verification System**: highlight overlays, coordinate transforms, verification state, export summaries.

## Coverage Map
- **Happy paths**: straight text PDFs, simple tables, single-language OCR.
- **Edge cases**: scanned PDFs (no embedded text), rotated pages, mixed languages, large tables, multi-column layouts.
- **Performance**: parsing time limits, memory caps, batch conversion sanity checks.
- **Determinism**: stable element ordering, stable bounding boxes for repeated runs.
- **I/O**: JSON/Markdown exports, report generation, download payloads.
- **Config toggles**: OCR on/off; table mode accurate/fast; images scale; page limits.

## Test Style
- **Unit tests**: pure functions/helpers and class methods.
- **Integration tests**: end-to-end parse+search flows.
- **Property tests**: idempotence of exports; bounding boxes remain within page bounds.
- **Golden files**: expected JSON/MD outputs for sample PDFs.

# 04 — Unit Tests: Core PDF Parser

## Targets
- Constructor wiring of Docling `DocumentConverter` with `PdfPipelineOptions`.
- `parse_document()` returns elements with page numbers and bounding boxes.
- OCR toggle impacts text yield for scanned PDFs.
- Table extraction produces row/col counts and cell mapping metadata.
- Page images available when `generate_page_images=True`.

## Example Test Skeletons
```python
# tests/test_parser_core.py
import pytest
from src.core.parser import DoclingParser

def test_parse_text_simple(fx_text_simple_pdf):
    p = DoclingParser(enable_ocr=False, enable_tables=True)
    elements = p.parse_document(fx_text_simple_pdf)
    assert len(elements) > 0
    assert all(e.page_number >= 1 for e in elements)

@pytest.mark.slow
def test_parse_scanned_with_ocr(fx_scanned_ocr_en_pdf):
    p = DoclingParser(enable_ocr=True)
    elements = p.parse_document(fx_scanned_ocr_en_pdf)
    assert any("sample" in e.text.lower() for e in elements)

def test_table_metadata(fx_tables_basic_pdf):
    p = DoclingParser(enable_tables=True)
    p.parse_document(fx_tables_basic_pdf)
    # inspect p.document.tables metadata or converted elements
```

## Invariants
- Bounding boxes within page bounds.
- Non-decreasing page numbers across elements when sorted by location.

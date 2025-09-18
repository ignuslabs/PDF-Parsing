import os
import pytest
import importlib.util
from pathlib import Path

from src.core.models import DocumentElement
from src.core.classifiers.header_classifier import is_heading
from src.core.kv_extraction import KeyValueExtractor, KVConfig


def _elem(text, x0=50, y0=700, x1=300, y1=720, page=1, etype="text", conf=0.95):
    return DocumentElement(
        text=text,
        element_type=etype,
        page_number=page,
        bbox={"x0": x0, "y0": y0, "x1": x1, "y1": y1},
        confidence=conf,
        metadata={"source": "unit_test"},
    )


def test_header_classifier_real_heading_vs_code_like():
    page_ctx = {"height": 792.0, "top_15_percent": 0.85 * 792.0}
    # True heading: top of page, structural words
    heading = _elem("Introduction", y0=720, y1=740)
    assert is_heading(heading, page_ctx) is True

    # Code-like string near top should not be heading
    code = _elem("INV-2025-00123", y0=720, y1=740)
    assert is_heading(code, page_ctx) is False

    # Another code-like with prefix
    code2 = _elem("PO# 887766", y0=720, y1=740)
    assert is_heading(code2, page_ctx) is False


def test_kv_extractor_finds_code_values_same_line():
    cfg = KVConfig(max_same_line_dx=200.0)
    kv = KeyValueExtractor(cfg)

    label = _elem("Invoice Number:", x0=50, y0=700, x1=180, y1=715)
    value = _elem("INV-2025-00123", x0=190, y0=700, x1=310, y1=715)

    # Include some noise
    noise = _elem("Lorem ipsum dolor sit amet", x0=50, y0=660, x1=400, y1=676)

    pairs = kv.extract([label, value, noise])

    assert pairs, "Expected at least one KV pair"
    found = any(p.label_text.lower().startswith("invoice") and "INV-2025" in p.value_text for p in pairs)
    assert found, "Expected to find invoice number pair"


HAS_DOCLING = importlib.util.find_spec("docling") is not None

@pytest.mark.skipif(
    not Path("/Users/joecorella/Desktop/PDF-Parsing/Test_PDFs").exists()
    and not Path("/Users/joecorella/Desktop/PDF Parsing/Test_PDFs").exists(),
    reason="Local PDF test directory not available",
)
@pytest.mark.skipif(
    not HAS_DOCLING,
    reason="Docling not available",
)
def test_parse_real_pdf_and_extract_codes():
    # Locate a PDF in the specified folder
    base1 = Path("/Users/joecorella/Desktop/PDF-Parsing/Test_PDFs")
    base2 = Path("/Users/joecorella/Desktop/PDF Parsing/Test_PDFs")
    base = base1 if base1.exists() else base2
    pdfs = list(base.glob("*.pdf"))
    assert pdfs, f"No PDFs found in {base}"

    # Lazy import to avoid hard dependency during collection
    from src.core.parser import DoclingParser

    parser = DoclingParser(enable_ocr=False, enable_tables=True, header_classifier_enabled=True)
    elements = parser.parse_document(pdfs[0])
    assert elements and len(elements) > 0

    # Run KV extraction on parsed elements
    kv = KeyValueExtractor()
    pairs = kv.extract(elements)

    # Heuristic check: at least one value looks like a code/id
    import re
    code_like = re.compile(r"\b(?:INV|PO|POL|ORD|MRN|ID|REF)[-# ]?[A-Z0-9]{4,}\b", re.I)
    assert any(code_like.search(p.value_text or "") for p in pairs), "Expected at least one code-like value"

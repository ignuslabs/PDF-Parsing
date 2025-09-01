"""
Unit tests for core PDF parser functionality.
Tests the DoclingParser class and its integration with Docling.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import os

# Import the classes we'll be testing (they don't exist yet - TDD approach)
from src.core.parser import DoclingParser
from src.core.models import DocumentElement


class TestDoclingParser:
    """Test suite for DoclingParser class."""

    @pytest.fixture
    def sample_pdf_path(self):
        """Path to a sample PDF fixture."""
        return Path("tests/fixtures/text_simple.pdf")

    @pytest.fixture
    def tables_pdf_path(self):
        """Path to tables PDF fixture."""
        return Path("tests/fixtures/tables_basic.pdf")

    @pytest.fixture
    def scanned_pdf_path(self):
        """Path to scanned OCR PDF fixture."""
        return Path("tests/fixtures/scanned_ocr_en.pdf")

    def test_parser_initialization_default_options(self):
        """Test parser initializes with default options."""
        parser = DoclingParser()

        assert parser is not None
        assert hasattr(parser, "enable_ocr")
        assert hasattr(parser, "enable_tables")
        assert hasattr(parser, "generate_page_images")

        # Default values
        assert parser.enable_ocr is False
        assert parser.enable_tables is True
        assert parser.generate_page_images is False

    def test_parser_initialization_custom_options(self):
        """Test parser initializes with custom options."""
        parser = DoclingParser(enable_ocr=True, enable_tables=False, generate_page_images=True)

        assert parser.enable_ocr is True
        assert parser.enable_tables is False
        assert parser.generate_page_images is True

    @pytest.mark.integration
    def test_parse_document_returns_elements(self, sample_pdf_path):
        """Test that parse_document returns a list of DocumentElements."""
        parser = DoclingParser(enable_ocr=False, enable_tables=True)

        # This test will fail initially since we haven't implemented the parser
        elements = parser.parse_document(sample_pdf_path)

        assert isinstance(elements, list)
        assert len(elements) > 0
        assert all(isinstance(elem, DocumentElement) for elem in elements)

    def test_parse_document_file_not_found(self):
        """Test parser handles non-existent files gracefully."""
        parser = DoclingParser()
        non_existent_path = Path("does_not_exist.pdf")

        with pytest.raises(FileNotFoundError):
            parser.parse_document(non_existent_path)

    @pytest.mark.integration
    def test_elements_have_page_numbers(self, sample_pdf_path):
        """Test that all parsed elements have valid page numbers."""
        parser = DoclingParser()
        elements = parser.parse_document(sample_pdf_path)

        for element in elements:
            assert hasattr(element, "page_number")
            assert element.page_number >= 1
            assert isinstance(element.page_number, int)

    @pytest.mark.integration
    def test_elements_have_bounding_boxes(self, sample_pdf_path):
        """Test that all elements have bounding box information."""
        parser = DoclingParser()
        elements = parser.parse_document(sample_pdf_path)

        for element in elements:
            assert hasattr(element, "bbox")
            assert element.bbox is not None

            # Bounding box should have required coordinates
            assert "x0" in element.bbox
            assert "y0" in element.bbox
            assert "x1" in element.bbox
            assert "y1" in element.bbox

            # Coordinates should be numeric
            assert isinstance(element.bbox["x0"], (int, float))
            assert isinstance(element.bbox["y0"], (int, float))
            assert isinstance(element.bbox["x1"], (int, float))
            assert isinstance(element.bbox["y1"], (int, float))

    @pytest.mark.integration
    def test_bounding_boxes_within_page_bounds(self, sample_pdf_path):
        """Test that bounding boxes are within reasonable page bounds."""
        parser = DoclingParser()
        elements = parser.parse_document(sample_pdf_path)

        # Typical page dimensions (letter size: 612x792 points)
        max_width = 1000  # Allow some margin for different page sizes
        max_height = 1000

        for element in elements:
            bbox = element.bbox

            # Coordinates should be non-negative and within page bounds
            assert bbox["x0"] >= 0
            assert bbox["y0"] >= 0
            assert bbox["x1"] <= max_width
            assert bbox["y1"] <= max_height

            # x1 should be >= x0, y1 should be >= y0
            assert bbox["x1"] >= bbox["x0"]
            assert bbox["y1"] >= bbox["y0"]

    @pytest.mark.integration
    def test_elements_sorted_by_page_and_position(self, sample_pdf_path):
        """Test that elements are sorted by page number and reading order."""
        parser = DoclingParser()
        elements = parser.parse_document(sample_pdf_path)

        # Check that page numbers are non-decreasing
        page_numbers = [elem.page_number for elem in elements]
        assert page_numbers == sorted(page_numbers)

    @pytest.mark.integration
    def test_table_extraction_enabled(self, tables_pdf_path):
        """Test that table extraction works when enabled."""
        parser = DoclingParser(enable_tables=True)
        elements = parser.parse_document(tables_pdf_path)

        # Should find table elements
        table_elements = [e for e in elements if e.element_type == "table"]
        assert len(table_elements) > 0

        # Table elements should have additional metadata
        for table_elem in table_elements:
            assert hasattr(table_elem, "metadata")
            assert table_elem.metadata is not None

    @pytest.mark.integration
    def test_table_extraction_disabled(self, tables_pdf_path):
        """Test that table extraction can be disabled."""
        parser = DoclingParser(enable_tables=False)
        elements = parser.parse_document(tables_pdf_path)

        # Should not find table-specific elements when disabled
        table_elements = [e for e in elements if e.element_type == "table"]
        # May still have text elements from tables, but no structured table elements
        # This behavior depends on implementation

    @pytest.mark.slow
    @pytest.mark.ocr
    def test_ocr_processing_enabled(self, scanned_pdf_path):
        """Test OCR processing on scanned PDFs."""
        parser = DoclingParser(enable_ocr=True)
        elements = parser.parse_document(scanned_pdf_path)

        # Should extract text from scanned PDF
        text_elements = [e for e in elements if e.element_type == "text" and e.text.strip()]
        assert len(text_elements) > 0

        # Should find specific text that we know is in the scanned PDF
        all_text = " ".join(e.text for e in text_elements)
        assert "scanned" in all_text.lower() or "document" in all_text.lower()

    @pytest.mark.integration
    def test_ocr_processing_disabled(self, scanned_pdf_path):
        """Test that OCR can be disabled (may extract less text)."""
        parser = DoclingParser(enable_ocr=False)
        elements = parser.parse_document(scanned_pdf_path)

        # Should still return elements (may be fewer without OCR)
        assert isinstance(elements, list)

    def test_page_image_generation_enabled(self, sample_pdf_path):
        """Test that page images can be generated when enabled."""
        parser = DoclingParser(generate_page_images=True)
        elements = parser.parse_document(sample_pdf_path)

        # Should have access to page images
        assert hasattr(parser, "page_images")
        # Implementation detail - may be stored differently

    @pytest.mark.integration
    def test_element_types_are_valid(self, sample_pdf_path):
        """Test that all elements have valid element types."""
        parser = DoclingParser()
        elements = parser.parse_document(sample_pdf_path)

        valid_types = {"text", "heading", "table", "image", "list", "formula"}

        for element in elements:
            assert hasattr(element, "element_type")
            assert element.element_type in valid_types

    @pytest.mark.integration
    def test_text_elements_have_content(self, sample_pdf_path):
        """Test that text elements contain actual text content."""
        parser = DoclingParser()
        elements = parser.parse_document(sample_pdf_path)

        text_elements = [e for e in elements if e.element_type in ("text", "heading")]
        assert len(text_elements) > 0

        for element in text_elements:
            assert hasattr(element, "text")
            assert isinstance(element.text, str)
            assert len(element.text.strip()) > 0

    @pytest.mark.integration
    def test_confidence_scores(self, sample_pdf_path):
        """Test that elements have confidence scores."""
        parser = DoclingParser()
        elements = parser.parse_document(sample_pdf_path)

        for element in elements:
            assert hasattr(element, "confidence")
            assert isinstance(element.confidence, (int, float))
            assert 0.0 <= element.confidence <= 1.0

    def test_parser_handles_empty_pdf(self):
        """Test parser behavior with empty or minimal PDF."""
        parser = DoclingParser()

        # Create a minimal test - this would need a truly empty PDF fixture
        # For now, we'll mock this scenario
        with patch.object(parser, "_convert_document") as mock_convert, \
             patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.stat") as mock_stat:
            
            # Mock file stats to simulate a valid file
            mock_stat.return_value.st_size = 1000  # Non-zero file size
            
            # Mock document conversion to return empty document
            mock_convert.return_value.document.texts = []
            mock_convert.return_value.document.tables = []
            mock_convert.return_value.document.pictures = []

            elements = parser.parse_document(Path("empty.pdf"))
            assert isinstance(elements, list)
            assert len(elements) == 0

    @pytest.mark.performance
    def test_parsing_performance(self, sample_pdf_path):
        """Test that parsing completes within reasonable time."""
        import time

        parser = DoclingParser()
        start_time = time.time()

        elements = parser.parse_document(sample_pdf_path)

        end_time = time.time()
        parsing_time = end_time - start_time

        # Should complete within 10 seconds for simple documents
        assert parsing_time < 10.0
        assert len(elements) > 0

    def test_multiple_documents_parsing(self, sample_pdf_path, tables_pdf_path):
        """Test parsing multiple documents with same parser instance."""
        parser = DoclingParser()

        elements1 = parser.parse_document(sample_pdf_path)
        elements2 = parser.parse_document(tables_pdf_path)

        assert len(elements1) > 0
        assert len(elements2) > 0

        # Results should be independent
        assert elements1 != elements2


# Fixtures for pytest
@pytest.fixture(scope="session", autouse=True)
def setup_test_fixtures():
    """Ensure test fixtures exist before running tests."""
    fixtures_dir = Path("tests/fixtures")
    required_fixtures = ["text_simple.pdf", "tables_basic.pdf", "scanned_ocr_en.pdf"]

    for fixture in required_fixtures:
        fixture_path = fixtures_dir / fixture
        if not fixture_path.exists():
            pytest.skip(f"Test fixture {fixture} not found. Run generate_test_fixtures.py first.")


@pytest.fixture
def mock_docling_document():
    """Mock Docling document for unit testing without actual PDF processing."""
    mock_doc = Mock()
    mock_doc.texts = [
        Mock(
            text="Sample text content",
            get_location=Mock(return_value=Mock(bbox=Mock(x0=100, y0=200, x1=300, y1=220))),
        )
    ]
    mock_doc.tables = []
    mock_doc.pictures = []
    return mock_doc

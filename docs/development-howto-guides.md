# Development How-To Guides

This collection of development-focused guides provides step-by-step instructions for testing, extending, and contributing to Smart PDF Parser. Each guide is designed to help developers accomplish specific development tasks efficiently.

## Testing How-To Guides

### How to Run the Test Suite

#### Setting Up the Test Environment

**Prerequisites**:
- Development dependencies installed (`pip install -r requirements-dev.txt`)
- Test fixtures generated
- Virtual environment activated

**Test Environment Setup**:

1. **Install Development Dependencies**:
   ```bash
   # Ensure you're in the project root and virtual environment is active
   pip install -r requirements-dev.txt
   
   # Verify test framework installation
   pytest --version
   python -c "import hypothesis; print('Hypothesis version:', hypothesis.__version__)"
   ```

2. **Generate Test Fixtures**:
   ```bash
   # Generate PDF test fixtures
   python generate_test_fixtures.py
   
   # Verify fixtures are created
   ls -la tests/fixtures/
   # Should show various PDF files for testing
   ```

3. **Verify Test Environment**:
   ```bash
   # Test basic imports
   python -c "from src.core.parser import DoclingParser; print('Parser: OK')"
   python -c "from src.core.search import SmartSearchEngine; print('Search: OK')"
   python -c "from src.verification.interface import VerificationInterface; print('Verification: OK')"
   ```

#### Running Different Test Categories

**Fast Tests (Default)**:
```bash
# Run fast tests only (excludes slow OCR and performance tests)
pytest tests/ -v

# More specific - exclude slow, OCR, and performance tests
pytest tests/ -v -m "not slow and not ocr and not performance"

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=html
```

**Integration Tests**:
```bash
# Run tests that use real PDF processing with Docling
pytest tests/ -v -m "integration"

# Run specific integration test files
pytest tests/test_parser_core.py -v -m "integration"
```

**OCR-specific Tests**:
```bash
# Run OCR-related tests (requires Tesseract)
pytest tests/ -v -m "ocr"

# Skip if Tesseract not available
pytest tests/ -v -m "ocr" --skip-tesseract-missing
```

**Performance Tests**:
```bash
# Run performance and memory tests
pytest tests/ -v -m "performance"

# Run with memory profiling
pytest tests/ -v -m "performance" --profile-memory

# Generate performance report
pytest tests/test_property_performance.py -v --benchmark-json=performance_report.json
```

**Property-based Tests**:
```bash
# Run Hypothesis property tests
pytest tests/ -v -m "property"

# Run with extended examples
pytest tests/ -v -m "property" --hypothesis-seed=12345 --hypothesis-max-examples=1000
```

#### Running Specific Tests

**Single Test Method**:
```bash
# Run specific test method
pytest tests/test_parser_core.py::TestDoclingParser::test_parse_document_returns_elements -v

# Run with debugging output
pytest tests/test_parser_core.py::TestDoclingParser::test_parse_document_returns_elements -v -s
```

**Test Class**:
```bash
# Run all tests in a class
pytest tests/test_parser_core.py::TestDoclingParser -v

# Run with fixture debugging
pytest tests/test_parser_core.py::TestDoclingParser -v --setup-show
```

**Test File**:
```bash
# Run entire test file
pytest tests/test_parser_core.py -v

# Run with failed test details
pytest tests/test_parser_core.py -v --tb=long
```

#### Debugging Failed Tests

**Verbose Output**:
```bash
# Show detailed output including print statements
pytest tests/test_failing.py -v -s

# Show local variables in tracebacks
pytest tests/test_failing.py -v --tb=long --showlocals
```

**Debug Mode**:
```bash
# Drop into debugger on failure
pytest tests/test_failing.py -v --pdb

# Drop into debugger on first failure
pytest tests/test_failing.py -v --pdb -x
```

**Logging Configuration**:
```bash
# Show log output during tests
pytest tests/ -v --log-cli-level=DEBUG

# Capture logs to file
pytest tests/ -v --log-file=test_debug.log --log-file-level=DEBUG
```

---

### How to Write New Tests

#### Test Structure and Organization

**Test File Organization**:
```
tests/
├── __init__.py
├── fixtures/              # Test PDF files
│   ├── text_simple.pdf
│   ├── tables_basic.pdf
│   └── scanned_ocr_en.pdf
├── golden/                # Expected output files
├── conftest.py            # Shared fixtures
├── test_parser_core.py    # Parser functionality tests
├── test_search_engine.py  # Search functionality tests
├── test_verification.py   # Verification system tests
└── test_property_performance.py  # Property and performance tests
```

**Test Naming Conventions**:
```python
# Test file names: test_[module_name].py
# Test class names: Test[ClassName]
# Test method names: test_[functionality]_[expected_behavior]

class TestDoclingParser:
    def test_parse_document_returns_elements(self):
        """Test that parse_document returns list of DocumentElement objects."""
        pass
    
    def test_parse_document_handles_invalid_pdf(self):
        """Test that parser gracefully handles corrupted PDF files."""
        pass
    
    def test_parse_document_preserves_text_formatting(self):
        """Test that text formatting is preserved during parsing."""
        pass
```

#### Writing Unit Tests

**Basic Unit Test Structure**:
```python
import pytest
from pathlib import Path
from src.core.parser import DoclingParser
from src.core.models import DocumentElement, ParsedDocument

class TestDoclingParser:
    """Test cases for the DoclingParser class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.parser = DoclingParser()
        self.test_pdf = Path("tests/fixtures/text_simple.pdf")
    
    def test_parser_initialization(self):
        """Test that parser initializes with default settings."""
        parser = DoclingParser()
        assert parser is not None
        assert hasattr(parser, 'parse_document')
        assert hasattr(parser, 'parse_document_full')
    
    def test_parse_simple_document(self):
        """Test parsing a simple text-based PDF."""
        # Arrange
        assert self.test_pdf.exists(), f"Test fixture {self.test_pdf} not found"
        
        # Act
        elements = self.parser.parse_document(str(self.test_pdf))
        
        # Assert
        assert isinstance(elements, list)
        assert len(elements) > 0
        assert all(isinstance(elem, DocumentElement) for elem in elements)
        
        # Check element properties
        first_element = elements[0]
        assert hasattr(first_element, 'text')
        assert hasattr(first_element, 'element_type')
        assert hasattr(first_element, 'confidence')
        assert 0.0 <= first_element.confidence <= 1.0
    
    def test_parse_document_with_invalid_path(self):
        """Test parser behavior with non-existent file."""
        with pytest.raises((FileNotFoundError, ValueError)):
            self.parser.parse_document("/nonexistent/file.pdf")
    
    def test_parse_document_full_returns_complete_object(self):
        """Test that parse_document_full returns ParsedDocument with metadata."""
        # Act
        parsed_doc = self.parser.parse_document_full(str(self.test_pdf))
        
        # Assert
        assert isinstance(parsed_doc, ParsedDocument)
        assert isinstance(parsed_doc.elements, list)
        assert isinstance(parsed_doc.metadata, dict)
        
        # Check required metadata fields
        assert 'source_path' in parsed_doc.metadata
        assert 'parsed_at' in parsed_doc.metadata
        assert 'total_elements' in parsed_doc.metadata
```

**Using Fixtures**:
```python
# conftest.py - shared fixtures
import pytest
from pathlib import Path
from src.core.parser import DoclingParser

@pytest.fixture
def sample_parser():
    """Provide a DoclingParser instance for tests."""
    return DoclingParser()

@pytest.fixture
def simple_text_pdf():
    """Provide path to simple text PDF fixture."""
    pdf_path = Path("tests/fixtures/text_simple.pdf")
    if not pdf_path.exists():
        pytest.skip(f"Test fixture {pdf_path} not available")
    return pdf_path

@pytest.fixture
def parsed_simple_document(sample_parser, simple_text_pdf):
    """Provide pre-parsed simple document for tests."""
    return sample_parser.parse_document_full(str(simple_text_pdf))

# Using fixtures in tests
def test_document_element_count(parsed_simple_document):
    """Test that simple document has expected number of elements."""
    assert len(parsed_simple_document.elements) >= 5
    assert len(parsed_simple_document.elements) <= 20  # Reasonable range
```

#### Writing Integration Tests

**Integration Test Structure**:
```python
import pytest
from src.core.parser import DoclingParser
from src.core.search import SmartSearchEngine
from src.verification.interface import VerificationInterface

@pytest.mark.integration
class TestDocumentProcessingWorkflow:
    """Integration tests for complete document processing workflow."""
    
    def setup_method(self):
        """Set up integrated components."""
        self.parser = DoclingParser()
        self.search_engine = SmartSearchEngine()
        self.verification = VerificationInterface()
    
    @pytest.mark.integration
    def test_complete_workflow(self, simple_text_pdf):
        """Test complete parse -> search -> verify workflow."""
        # 1. Parse document
        parsed_doc = self.parser.parse_document_full(str(simple_text_pdf))
        assert len(parsed_doc.elements) > 0
        
        # 2. Initialize search with parsed elements
        self.search_engine.load_document(parsed_doc.elements)
        
        # 3. Perform search
        search_results = self.search_engine.search("introduction", search_type="fuzzy")
        assert len(search_results) >= 0  # May or may not find matches
        
        # 4. Set up verification
        if search_results:
            verification_state = self.verification.create_session(parsed_doc.elements)
            assert verification_state is not None
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_large_document_processing(self):
        """Test processing of larger documents (integration + performance)."""
        large_pdf = Path("tests/fixtures/large_pages_light.pdf")
        if not large_pdf.exists():
            pytest.skip("Large document fixture not available")
        
        # Parse large document
        start_time = time.time()
        parsed_doc = self.parser.parse_document_full(str(large_pdf))
        parsing_time = time.time() - start_time
        
        # Performance assertions
        assert parsing_time < 120  # Should complete within 2 minutes
        assert len(parsed_doc.elements) > 50  # Should extract substantial content
        assert parsed_doc.get_average_confidence() > 0.7  # Reasonable confidence
```

#### Property-Based Testing with Hypothesis

**Property Test Examples**:
```python
from hypothesis import given, strategies as st, assume
import pytest
from src.core.models import DocumentElement

class TestDocumentElementProperties:
    """Property-based tests for DocumentElement."""
    
    @given(
        text=st.text(min_size=1, max_size=1000),
        element_type=st.sampled_from(['text', 'heading', 'table', 'image', 'formula']),
        page_number=st.integers(min_value=1, max_value=1000),
        confidence=st.floats(min_value=0.0, max_value=1.0),
        x0=st.floats(min_value=0, max_value=1000),
        y0=st.floats(min_value=0, max_value=1000),
        x1=st.floats(min_value=0, max_value=1000),
        y1=st.floats(min_value=0, max_value=1000)
    )
    @pytest.mark.property
    def test_document_element_creation_with_valid_data(
        self, text, element_type, page_number, confidence, x0, y0, x1, y1
    ):
        """Property: DocumentElement can be created with any valid input."""
        # Ensure bounding box is valid
        assume(x0 <= x1 and y0 <= y1)
        
        bbox = {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1}
        metadata = {}
        
        # Should not raise exception with valid data
        element = DocumentElement(
            text=text,
            element_type=element_type,
            page_number=page_number,
            bbox=bbox,
            confidence=confidence,
            metadata=metadata
        )
        
        # Properties should be preserved
        assert element.text == text
        assert element.element_type == element_type
        assert element.page_number == page_number
        assert element.confidence == confidence
        assert element.bbox == bbox
    
    @given(confidence=st.floats())
    @pytest.mark.property
    def test_invalid_confidence_raises_error(self, confidence):
        """Property: Invalid confidence values should raise ValueError."""
        assume(not (0.0 <= confidence <= 1.0))
        
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            DocumentElement(
                text="test",
                element_type="text", 
                page_number=1,
                bbox={'x0': 0, 'y0': 0, 'x1': 100, 'y1': 100},
                confidence=confidence,
                metadata={}
            )
```

---

### How to Add New Parsers

#### Parser Architecture Overview

**Parser Interface**:
```python
# src/core/parser.py - Base parser structure
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from src.core.models import DocumentElement, ParsedDocument

class BaseParser(ABC):
    """Abstract base class for document parsers."""
    
    @abstractmethod
    def parse_document(self, file_path: str, **kwargs) -> List[DocumentElement]:
        """Parse document and return list of elements."""
        pass
    
    @abstractmethod 
    def parse_document_full(self, file_path: str, **kwargs) -> ParsedDocument:
        """Parse document and return full parsed document with metadata."""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Return list of supported file formats."""
        pass
```

#### Creating a New Parser

**Example: Adding a Word Document Parser**

1. **Create Parser Class**:
```python
# src/core/parsers/docx_parser.py
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import docx
from docx.document import Document as DocxDocument

from ..models import DocumentElement, ParsedDocument
from .base_parser import BaseParser

class DocxParser(BaseParser):
    """Parser for Microsoft Word documents (.docx format)."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize DOCX parser with configuration."""
        self.config = config or {}
        self.default_confidence = self.config.get('default_confidence', 0.95)
    
    def get_supported_formats(self) -> List[str]:
        """Return supported file formats."""
        return ['.docx', '.doc']
    
    def parse_document(self, file_path: str, **kwargs) -> List[DocumentElement]:
        """Parse DOCX document and return elements."""
        elements = []
        doc = docx.Document(file_path)
        
        element_id = 0
        for para_idx, paragraph in enumerate(doc.paragraphs):
            if not paragraph.text.strip():
                continue
                
            # Determine element type based on style
            element_type = self._classify_paragraph(paragraph)
            
            # Create document element
            element = DocumentElement(
                text=paragraph.text.strip(),
                element_type=element_type,
                page_number=1,  # DOCX doesn't have explicit page numbers
                bbox=self._estimate_bbox(para_idx),
                confidence=self.default_confidence,
                metadata=self._extract_paragraph_metadata(paragraph)
            )
            elements.append(element)
            element_id += 1
        
        # Process tables
        for table_idx, table in enumerate(doc.tables):
            table_text = self._extract_table_text(table)
            if table_text:
                element = DocumentElement(
                    text=table_text,
                    element_type='table',
                    page_number=1,
                    bbox=self._estimate_table_bbox(table_idx),
                    confidence=self.default_confidence,
                    metadata={'table_index': table_idx, 'rows': len(table.rows)}
                )
                elements.append(element)
        
        return elements
    
    def parse_document_full(self, file_path: str, **kwargs) -> ParsedDocument:
        """Parse DOCX document and return full parsed document."""
        elements = self.parse_document(file_path, **kwargs)
        
        # Extract document metadata
        doc = docx.Document(file_path)
        core_props = doc.core_properties
        
        metadata = {
            'source_path': file_path,
            'filename': Path(file_path).name,
            'parsed_at': datetime.now().isoformat(),
            'parser_type': 'DocxParser',
            'total_elements': len(elements),
            'page_count': 1,  # DOCX is continuous
            'document_properties': {
                'title': core_props.title or '',
                'author': core_props.author or '',
                'subject': core_props.subject or '',
                'created': core_props.created.isoformat() if core_props.created else '',
                'modified': core_props.modified.isoformat() if core_props.modified else ''
            }
        }
        
        return ParsedDocument(elements=elements, metadata=metadata)
    
    def _classify_paragraph(self, paragraph) -> str:
        """Classify paragraph type based on style."""
        style_name = paragraph.style.name.lower()
        
        if 'heading' in style_name or 'title' in style_name:
            return 'heading'
        elif 'list' in style_name or paragraph.text.strip().startswith(('•', '-', '*')):
            return 'list'
        else:
            return 'text'
    
    def _estimate_bbox(self, para_idx: int) -> Dict[str, float]:
        """Estimate bounding box for paragraph (DOCX doesn't provide exact positions)."""
        # Rough estimation based on typical document layout
        y_position = 800 - (para_idx * 20)  # Approximate line spacing
        return {
            'x0': 72.0,    # Left margin
            'y0': y_position,
            'x1': 540.0,   # Right margin
            'y1': y_position + 15  # Line height
        }
    
    def _extract_paragraph_metadata(self, paragraph) -> Dict[str, Any]:
        """Extract metadata from paragraph formatting."""
        runs = paragraph.runs
        if not runs:
            return {}
        
        first_run = runs[0]
        return {
            'style_name': paragraph.style.name,
            'font_name': first_run.font.name,
            'font_size': first_run.font.size.pt if first_run.font.size else None,
            'bold': first_run.font.bold,
            'italic': first_run.font.italic,
            'underline': first_run.font.underline
        }
    
    def _extract_table_text(self, table) -> str:
        """Extract text content from table."""
        table_data = []
        for row in table.rows:
            row_data = [cell.text.strip() for cell in row.cells]
            table_data.append(' | '.join(row_data))
        return '\n'.join(table_data)
    
    def _estimate_table_bbox(self, table_idx: int) -> Dict[str, float]:
        """Estimate bounding box for table."""
        y_position = 600 - (table_idx * 100)
        return {
            'x0': 72.0,
            'y0': y_position,
            'x1': 540.0,
            'y1': y_position + 80
        }
```

2. **Add Parser Tests**:
```python
# tests/test_docx_parser.py
import pytest
from pathlib import Path
from src.core.parsers.docx_parser import DocxParser
from src.core.models import DocumentElement, ParsedDocument

class TestDocxParser:
    """Test cases for DOCX parser."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = DocxParser()
    
    def test_parser_initialization(self):
        """Test parser initializes correctly."""
        parser = DocxParser()
        assert parser is not None
        assert '.docx' in parser.get_supported_formats()
    
    @pytest.mark.integration
    def test_parse_simple_docx(self, simple_docx_fixture):
        """Test parsing a simple DOCX document."""
        elements = self.parser.parse_document(str(simple_docx_fixture))
        
        assert isinstance(elements, list)
        assert len(elements) > 0
        assert all(isinstance(elem, DocumentElement) for elem in elements)
    
    @pytest.mark.integration
    def test_parse_docx_full(self, simple_docx_fixture):
        """Test full parsing with metadata."""
        parsed_doc = self.parser.parse_document_full(str(simple_docx_fixture))
        
        assert isinstance(parsed_doc, ParsedDocument)
        assert 'parser_type' in parsed_doc.metadata
        assert parsed_doc.metadata['parser_type'] == 'DocxParser'
        assert 'document_properties' in parsed_doc.metadata
```

3. **Register Parser**:
```python
# src/core/parser_registry.py
from typing import Dict, Type
from .base_parser import BaseParser
from .docling_parser import DoclingParser
from .parsers.docx_parser import DocxParser

class ParserRegistry:
    """Registry for document parsers."""
    
    def __init__(self):
        self._parsers: Dict[str, Type[BaseParser]] = {}
        self._register_default_parsers()
    
    def _register_default_parsers(self):
        """Register built-in parsers."""
        self.register_parser('pdf', DoclingParser)
        self.register_parser('docx', DocxParser)
    
    def register_parser(self, format_key: str, parser_class: Type[BaseParser]):
        """Register a parser for a specific format."""
        self._parsers[format_key] = parser_class
    
    def get_parser(self, format_key: str) -> Type[BaseParser]:
        """Get parser class for format."""
        if format_key not in self._parsers:
            raise ValueError(f"No parser registered for format: {format_key}")
        return self._parsers[format_key]
    
    def get_available_formats(self) -> List[str]:
        """Get list of supported formats."""
        return list(self._parsers.keys())

# Global registry instance
parser_registry = ParserRegistry()
```

4. **Update UI Integration**:
```python
# src/ui/components/upload_handler.py
from src.core.parser_registry import parser_registry

def handle_file_upload(uploaded_file):
    """Handle file upload and determine appropriate parser."""
    file_extension = Path(uploaded_file.name).suffix.lower()
    
    # Map file extensions to parser format keys
    extension_map = {
        '.pdf': 'pdf',
        '.docx': 'docx',
        '.doc': 'docx'  # Treat .doc files same as .docx
    }
    
    if file_extension not in extension_map:
        st.error(f"Unsupported file format: {file_extension}")
        return None
    
    format_key = extension_map[file_extension]
    try:
        parser_class = parser_registry.get_parser(format_key)
        parser = parser_class()
        return parser
    except ValueError as e:
        st.error(str(e))
        return None
```

---

### How to Extend Search Functionality

#### Search Engine Architecture

**Current Search Implementation**:
```python
# src/core/search.py - Current structure
class SmartSearchEngine:
    """Multi-modal search engine with exact, fuzzy, and semantic matching."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.elements: List[DocumentElement] = []
        self.indices = {}  # Various search indices
    
    def search(self, query: str, search_type: str = "fuzzy", **filters) -> List[SearchResult]:
        """Main search method supporting different search types."""
        if search_type == "exact":
            return self._exact_search(query, **filters)
        elif search_type == "fuzzy":
            return self._fuzzy_search(query, **filters)  
        elif search_type == "semantic":
            return self._semantic_search(query, **filters)
        else:
            raise ValueError(f"Unknown search type: {search_type}")
```

#### Adding New Search Methods

**Example: Adding RegEx Search**

1. **Extend Search Engine**:
```python
# src/core/search_extensions/regex_search.py
import re
from typing import List, Dict, Any
from ..models import DocumentElement, SearchResult

class RegexSearchMixin:
    """Mixin to add regex search capabilities."""
    
    def _regex_search(self, pattern: str, **filters) -> List[SearchResult]:
        """Search using regular expressions."""
        try:
            # Compile regex pattern
            regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")
        
        results = []
        
        for element in self._filter_elements(**filters):
            matches = list(regex.finditer(element.text))
            
            for match in matches:
                # Calculate match confidence based on pattern complexity
                confidence = self._calculate_regex_confidence(match, pattern)
                
                # Extract context around match
                context = self._extract_match_context(element.text, match)
                
                result = SearchResult(
                    element=element,
                    score=confidence,
                    match_type='regex',
                    matched_text=match.group(0),
                    match_context=context
                )
                results.append(result)
        
        return self._rank_results(results)
    
    def _calculate_regex_confidence(self, match: re.Match, pattern: str) -> float:
        """Calculate confidence score for regex match."""
        # Base confidence for any match
        base_confidence = 0.8
        
        # Boost for exact word boundaries
        if r'\b' in pattern:
            base_confidence += 0.1
        
        # Boost for character classes and quantifiers (more specific patterns)
        if any(char in pattern for char in ['[', '{', '+', '*', '?']):
            base_confidence += 0.05
        
        # Reduce for very generic patterns
        if pattern in ['.', '.*', '.+']:
            base_confidence -= 0.3
        
        return min(1.0, base_confidence)
    
    def _extract_match_context(self, text: str, match: re.Match, context_chars: int = 50) -> str:
        """Extract context around regex match."""
        start = max(0, match.start() - context_chars)
        end = min(len(text), match.end() + context_chars)
        
        context = text[start:end]
        
        # Add ellipsis if we're not at the beginning/end
        if start > 0:
            context = '...' + context
        if end < len(text):
            context = context + '...'
        
        return context
```

2. **Update Main Search Engine**:
```python
# src/core/search.py - Updated with regex support
from .search_extensions.regex_search import RegexSearchMixin

class SmartSearchEngine(RegexSearchMixin):
    """Enhanced search engine with multiple search methods."""
    
    def search(self, query: str, search_type: str = "fuzzy", **filters) -> List[SearchResult]:
        """Main search method with regex support."""
        if search_type == "exact":
            return self._exact_search(query, **filters)
        elif search_type == "fuzzy":
            return self._fuzzy_search(query, **filters)
        elif search_type == "semantic":
            return self._semantic_search(query, **filters)
        elif search_type == "regex":
            return self._regex_search(query, **filters)
        else:
            raise ValueError(f"Unknown search type: {search_type}")
    
    def get_available_search_types(self) -> List[str]:
        """Get list of available search types."""
        return ["exact", "fuzzy", "semantic", "regex"]
```

3. **Add UI Support**:
```python
# src/ui/pages/2_🔍_Search.py - Updated search interface
import streamlit as st
import re

def render_search_interface():
    """Render search interface with regex support."""
    
    # Search type selection
    search_type = st.selectbox(
        "Search Type",
        options=["fuzzy", "exact", "semantic", "regex"],
        index=0,
        help="Choose search method: fuzzy (typo-tolerant), exact (precise match), semantic (meaning-based), regex (pattern matching)"
    )
    
    # Query input with help for regex
    if search_type == "regex":
        st.info("🔤 **Regex Pattern Examples:**")
        st.code("""
# Find email addresses
[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}

# Find phone numbers  
\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})

# Find dates (MM/DD/YYYY)
\d{1,2}/\d{1,2}/\d{4}

# Find currency amounts
\$\d+(?:,\d{3})*(?:\.\d{2})?

# Find section numbers
\d+\.\d+(?:\.\d+)*
        """)
        
        query = st.text_input(
            "Regex Pattern",
            placeholder="Enter regex pattern (e.g., \\d{4} for 4-digit numbers)",
            help="Enter a regular expression pattern. Use \\ to escape special characters."
        )
        
        # Validate regex pattern
        if query:
            try:
                re.compile(query)
                st.success("✓ Valid regex pattern")
            except re.error as e:
                st.error(f"❌ Invalid regex pattern: {e}")
    else:
        query = st.text_input("Search Query", placeholder="Enter search terms...")
```

4. **Add Tests for New Functionality**:
```python
# tests/test_regex_search.py
import pytest
from src.core.search import SmartSearchEngine
from src.core.models import DocumentElement

class TestRegexSearch:
    """Test cases for regex search functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.search_engine = SmartSearchEngine()
        
        # Create test elements with various patterns
        self.test_elements = [
            DocumentElement(
                text="Contact us at support@example.com or call (555) 123-4567",
                element_type="text", page_number=1, 
                bbox={'x0': 0, 'y0': 0, 'x1': 100, 'y1': 20},
                confidence=0.9, metadata={}
            ),
            DocumentElement(
                text="Meeting scheduled for 12/25/2024 at 2:30 PM",
                element_type="text", page_number=1,
                bbox={'x0': 0, 'y0': 20, 'x1': 100, 'y1': 40}, 
                confidence=0.9, metadata={}
            ),
            DocumentElement(
                text="Total amount: $1,234.56 (including tax)",
                element_type="text", page_number=1,
                bbox={'x0': 0, 'y0': 40, 'x1': 100, 'y1': 60},
                confidence=0.9, metadata={}
            )
        ]
        
        self.search_engine.load_document(self.test_elements)
    
    def test_regex_search_email_pattern(self):
        """Test regex search for email addresses."""
        pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        results = self.search_engine.search(pattern, search_type="regex")
        
        assert len(results) == 1
        assert "support@example.com" in results[0].matched_text
        assert results[0].match_type == "regex"
    
    def test_regex_search_phone_pattern(self):
        """Test regex search for phone numbers."""
        pattern = r'\(\d{3}\)\s\d{3}-\d{4}'
        results = self.search_engine.search(pattern, search_type="regex")
        
        assert len(results) == 1
        assert "(555) 123-4567" in results[0].matched_text
    
    def test_regex_search_currency_pattern(self):
        """Test regex search for currency amounts."""
        pattern = r'\$\d+(?:,\d{3})*(?:\.\d{2})?'
        results = self.search_engine.search(pattern, search_type="regex")
        
        assert len(results) == 1
        assert "$1,234.56" in results[0].matched_text
    
    def test_invalid_regex_pattern(self):
        """Test that invalid regex patterns raise appropriate errors."""
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            self.search_engine.search("[invalid", search_type="regex")
    
    def test_regex_search_with_filters(self):
        """Test regex search with element type filters."""
        pattern = r'\d+'
        results = self.search_engine.search(
            pattern, 
            search_type="regex",
            element_types=['text']
        )
        
        # Should find numbers in multiple elements
        assert len(results) > 0
        assert all(result.element.element_type == 'text' for result in results)
```

**Adding Advanced Search Features**:

**Contextual Search**:
```python
def _contextual_search(self, query: str, context_window: int = 2, **filters) -> List[SearchResult]:
    """Search considering surrounding context."""
    results = []
    
    for i, element in enumerate(self._filter_elements(**filters)):
        # Get context elements (previous and next)
        context_elements = []
        
        # Previous elements
        start_idx = max(0, i - context_window)
        context_elements.extend(self.elements[start_idx:i])
        
        # Current element
        context_elements.append(element)
        
        # Next elements  
        end_idx = min(len(self.elements), i + context_window + 1)
        context_elements.extend(self.elements[i+1:end_idx])
        
        # Combine text with context
        context_text = ' '.join(elem.text for elem in context_elements)
        
        # Search in combined context
        if self._matches_query(context_text, query):
            score = self._calculate_contextual_score(element, context_elements, query)
            result = SearchResult(
                element=element,
                score=score,
                match_type='contextual',
                matched_text=self._extract_match(element.text, query),
                match_context=context_text[:200] + "..." if len(context_text) > 200 else context_text
            )
            results.append(result)
    
    return results
```

This development guide provides comprehensive instructions for extending Smart PDF Parser's functionality through testing, new parsers, and enhanced search capabilities.
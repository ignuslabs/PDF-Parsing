# Testing Strategy

*How-to guide for testing approaches, test categories, and quality assurance*

## Overview

Smart PDF Parser uses a **comprehensive multi-layered testing strategy** built around Test-Driven Development (TDD) principles. Our testing approach ensures reliability, performance, and maintainability across the entire system.

## Test Architecture

### Test Pyramid Structure

```text
pyramid TB
    A[Unit Tests<br/>~70% of tests<br/>Fast, isolated, deterministic]
    B[Integration Tests<br/>~20% of tests<br/>Component interaction]  
    C[System Tests<br/>~10% of tests<br/>End-to-end workflows]
```

### Test Categories

Our tests are organized into **pytest markers** for selective execution:

```python
# Fast tests (default) - Run on every commit
pytest tests/ -v -m "not slow and not ocr and not performance"

# Slow tests - OCR and large document processing  
pytest tests/ -v -m "slow or ocr"

# Performance tests - Memory and timing benchmarks
pytest tests/ -v -m "performance"  

# Integration tests - End-to-end workflows
pytest tests/ -v -m "integration"
```

## Test Organization

### Directory Structure

```
tests/
├── fixtures/           # Test PDF samples and golden files
│   ├── text_simple.pdf
│   ├── tables_basic.pdf
│   ├── scanned_ocr_en.pdf
│   └── multicolumn_rotated.pdf
├── golden/            # Expected outputs for regression testing
│   ├── text_simple_expected.json
│   └── tables_basic_expected.json
├── test_parser_core.py      # Unit tests for PDF parser
├── test_search_engine.py    # Unit tests for search functionality  
├── test_verification.py     # Unit tests for verification system
└── test_property_performance.py  # Property-based and performance tests
```

### Test Fixtures

**Critical**: Always generate test fixtures before running tests:

```bash
python generate_test_fixtures.py
```

This creates standardized PDFs with known content for consistent testing:

- **text_simple.pdf**: Basic text document for parsing tests
- **tables_basic.pdf**: Document with simple table structures  
- **scanned_ocr_en.pdf**: Scanned document requiring OCR
- **multicolumn_rotated.pdf**: Complex layout with rotation
- **images_captions.pdf**: Document with images and captions
- **formulas_snippets.pdf**: Mathematical formulas and symbols
- **large_pages_light.pdf**: Multi-page document for performance testing

## Unit Testing

### Parser Tests (`test_parser_core.py`)

Tests the core `DoclingParser` class and its integration with Docling:

```python
class TestDoclingParser:
    
    @pytest.fixture
    def sample_pdf_path(self):
        return Path("tests/fixtures/text_simple.pdf")
    
    def test_parser_initialization_default_options(self):
        """Test parser initializes with default options."""
        parser = DoclingParser()
        assert parser.enable_ocr is False
        assert parser.enable_tables is True
        assert parser.generate_page_images is False
    
    @pytest.mark.slow 
    def test_parse_document_returns_elements(self, sample_pdf_path):
        """Test document parsing returns DocumentElement objects."""
        parser = DoclingParser()
        elements = parser.parse_document(sample_pdf_path)
        
        assert isinstance(elements, list)
        assert all(isinstance(elem, DocumentElement) for elem in elements)
        assert len(elements) > 0
```

#### Key Test Patterns

**Configuration Testing**:
```python
def test_parser_configuration_validation(self):
    """Test invalid configurations raise appropriate errors."""
    with pytest.raises(ValueError, match="Unsupported OCR engine"):
        DoclingParser(ocr_engine="invalid")
        
    with pytest.raises(ValueError, match="Image scale must be between"):
        DoclingParser(image_scale=10.0)
```

**Error Handling**:
```python
@patch('src.core.parser.DocumentConverter')
def test_parse_document_handles_docling_errors(self, mock_converter):
    """Test parser handles Docling conversion errors gracefully."""
    mock_converter.return_value.convert.side_effect = Exception("Docling error")
    
    parser = DoclingParser()
    with pytest.raises(DocumentParsingError):
        parser.parse_document("invalid.pdf")
```

**OCR Fallback Testing**:
```python  
@pytest.mark.ocr
def test_ocr_fallback_on_failure(self, scanned_pdf_path):
    """Test parser falls back gracefully when OCR fails."""
    parser = DoclingParser(enable_ocr=True)
    
    # Should not raise exception, may return fewer elements
    elements = parser.parse_document(scanned_pdf_path)
    assert isinstance(elements, list)
```

### Search Engine Tests (`test_search_engine.py`)

Tests the `SmartSearchEngine` multi-modal search capabilities:

```python
class TestSmartSearchEngine:
    
    @pytest.fixture
    def search_engine(self, sample_elements):
        return SmartSearchEngine(sample_elements)
    
    def test_exact_search_basic(self, search_engine):
        """Test exact string matching."""
        results = search_engine.search("specific text")
        assert len(results) > 0
        assert all("specific text" in result.element.text.lower() 
                  for result in results)
    
    def test_fuzzy_search_typos(self, search_engine):
        """Test fuzzy search handles typos."""
        results = search_engine.search("specfic txt", mode="fuzzy")
        assert len(results) > 0
        assert results[0].confidence > 0.7
```

#### Search Test Categories

**Exact Matching**:
```python
def test_exact_search_case_insensitive(self):
    """Test exact search ignores case."""
    results = self.engine.search("TABLE", mode="exact")
    assert any("table" in r.element.text.lower() for r in results)
```

**Fuzzy Matching**:  
```python
def test_fuzzy_search_similarity_threshold(self):
    """Test fuzzy search respects similarity thresholds."""
    results = self.engine.search("tabel", mode="fuzzy", min_similarity=0.8)
    assert all(r.confidence >= 0.8 for r in results)
```

**Filtering**:
```python
def test_element_type_filtering(self):
    """Test filtering by element type."""
    results = self.engine.search("", element_types=["heading"])
    assert all(r.element.element_type == "heading" for r in results)
    
def test_page_number_filtering(self):
    """Test filtering by page numbers."""  
    results = self.engine.search("", pages=[1, 2])
    assert all(r.element.page_num in [1, 2] for r in results)
```

### Verification Tests (`test_verification.py`)

Tests the interactive verification system:

```python
class TestVerificationInterface:
    
    def test_coordinate_transformation(self):
        """Test PDF to pixel coordinate transformation."""
        interface = VerificationInterface()
        
        pdf_coords = (100, 200, 300, 400)  # x1, y1, x2, y2
        pixel_coords = interface.pdf_to_pixel_coords(pdf_coords, page_height=800)
        
        # Should flip Y coordinates (PDF origin bottom-left, pixel top-left)
        assert pixel_coords[1] == 800 - 400  # y1 flipped
        assert pixel_coords[3] == 800 - 200  # y2 flipped
    
    def test_verification_state_management(self):
        """Test verification state tracking."""
        interface = VerificationInterface()
        
        # Mark element as verified
        interface.mark_verified("element_1", True, "Correct extraction")
        state = interface.get_verification_state("element_1")
        
        assert state.verified is True
        assert state.comment == "Correct extraction"
        assert state.timestamp is not None
```

## Integration Testing

### End-to-End Workflows

Integration tests verify complete user workflows:

```python
@pytest.mark.integration
class TestParseSearchVerifyWorkflow:
    
    def test_complete_document_processing(self, sample_pdf_path):
        """Test complete workflow: parse -> search -> verify -> export."""
        
        # 1. Parse document
        parser = DoclingParser(enable_tables=True)
        elements = parser.parse_document(sample_pdf_path)
        assert len(elements) > 0
        
        # 2. Search elements  
        engine = SmartSearchEngine(elements)
        results = engine.search("table")
        assert len(results) > 0
        
        # 3. Verify results
        interface = VerificationInterface()
        for result in results:
            interface.mark_verified(result.element.id, True)
            
        # 4. Export verified data
        export_data = interface.export_verified_elements()
        assert len(export_data) > 0
        assert all(elem['verified'] for elem in export_data)
```

### Configuration Integration

```python  
@pytest.mark.integration
def test_parser_configuration_integration(self):
    """Test parser works with all configuration combinations."""
    
    configs = [
        {"enable_ocr": True, "enable_tables": True},
        {"enable_ocr": False, "enable_tables": True},  
        {"enable_ocr": True, "enable_tables": False},
        {"generate_page_images": True},
    ]
    
    for config in configs:
        parser = DoclingParser(**config)
        elements = parser.parse_document(self.sample_pdf)
        assert isinstance(elements, list)
```

## Performance Testing

### Property-Based Testing

Using Hypothesis for property-based tests:

```python
from hypothesis import given, strategies as st

@pytest.mark.performance  
class TestPropertyPerformance:
    
    @given(st.text(min_size=1, max_size=1000))
    def test_search_query_performance(self, query_text):
        """Test search performance scales with query length."""
        engine = SmartSearchEngine(self.large_element_set)
        
        start_time = time.time()
        results = engine.search(query_text)
        duration = time.time() - start_time
        
        # Should complete within reasonable time regardless of query
        assert duration < 2.0
        assert isinstance(results, list)
    
    @given(st.integers(min_value=1, max_value=1000))
    def test_parser_memory_scaling(self, num_pages):
        """Test memory usage scales linearly with document size."""  
        assume(num_pages <= 100)  # Reasonable limit for CI
        
        parser = DoclingParser(max_pages=num_pages)
        initial_memory = psutil.Process().memory_info().rss
        
        elements = parser.parse_document(self.large_pdf)
        peak_memory = psutil.Process().memory_info().rss
        
        memory_increase = peak_memory - initial_memory
        # Memory should not exceed 100MB per page
        assert memory_increase < num_pages * 100 * 1024 * 1024
```

### Memory and Performance Benchmarks

```python
@pytest.mark.performance
class TestPerformanceBenchmarks:
    
    def test_parser_memory_usage(self):
        """Test parser memory usage stays within bounds."""
        parser = DoclingParser()
        
        # Measure memory before parsing
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Parse large document
        elements = parser.parse_document(self.large_pdf_path)
        peak_memory = process.memory_info().rss
        
        # Memory increase should be reasonable
        memory_increase_mb = (peak_memory - initial_memory) / 1024 / 1024
        assert memory_increase_mb < 500  # Less than 500MB increase
    
    def test_search_response_time(self):
        """Test search engine response times."""
        engine = SmartSearchEngine(self.large_element_set)
        
        queries = ["table", "figure", "conclusion", "methodology"]
        
        for query in queries:
            start_time = time.time()
            results = engine.search(query)
            duration = time.time() - start_time
            
            # Should respond within 100ms for typical queries
            assert duration < 0.1
            assert len(results) >= 0
```

## Test Execution Strategies

### Local Development Testing

**Fast feedback loop** (runs in ~30 seconds):
```bash
# Run only fast tests during development
pytest tests/ -v -m "not slow and not ocr and not performance" --maxfail=3

# Run specific test file
pytest tests/test_parser_core.py -v

# Run specific test method
pytest tests/test_parser_core.py::TestDoclingParser::test_parse_document_returns_elements -v

# Run with coverage
pytest tests/ -v -m "not slow" --cov=src --cov-report=term-missing
```

**Full test suite** (runs in ~5-10 minutes):
```bash
# Run all tests including slow ones
pytest tests/ -v

# Run only slow/OCR tests
pytest tests/ -v -m "slow or ocr"

# Run performance tests
pytest tests/ -v -m "performance"
```

### CI/CD Testing Strategy

Our GitHub Actions pipeline runs tests in **three stages**:

#### Stage 1: Fast Tests (All Python Versions)
- Unit tests without OCR/performance  
- Runs in parallel across Python 3.9, 3.10, 3.11
- Must complete in under 5 minutes
- Required for all PRs

#### Stage 2: Slow Tests (Python 3.11 only)  
- OCR processing tests
- Large document tests
- Integration tests
- Runs only on Python 3.11 to save CI time

#### Stage 3: Performance Tests (Main branch only)
- Memory usage benchmarks
- Performance regression detection  
- Property-based testing
- Runs only on pushes to main branch

### Test Data Management

#### Golden File Testing

**Creating golden files**:
```python  
def test_create_golden_output(self, sample_pdf_path):
    """Helper to create expected output files."""
    parser = DoclingParser()
    elements = parser.parse_document(sample_pdf_path)
    
    # Serialize to JSON for comparison
    output = [elem.to_dict() for elem in elements]
    
    golden_path = Path(f"tests/golden/{sample_pdf_path.stem}_expected.json")
    with open(golden_path, 'w') as f:
        json.dump(output, f, indent=2, sort_keys=True)
```

**Using golden files**:
```python
def test_parser_output_regression(self, sample_pdf_path):
    """Test parser output hasn't regressed."""
    parser = DoclingParser()
    elements = parser.parse_document(sample_pdf_path)
    
    actual = [elem.to_dict() for elem in elements]
    
    golden_path = Path(f"tests/golden/{sample_pdf_path.stem}_expected.json")  
    with open(golden_path) as f:
        expected = json.load(f)
        
    # Compare with tolerance for floating-point coordinates
    assert_elements_equal(actual, expected, tolerance=0.01)
```

## Test Quality Standards

### Coverage Requirements

- **Minimum overall coverage**: 85%
- **Critical modules coverage**: 95%
  - `src/core/parser.py`
  - `src/core/search.py` 
  - `src/core/models.py`
- **UI modules coverage**: 70% (UI testing is complex)

### Test Quality Metrics

**Test reliability**:
- Tests must be deterministic (no flaky tests)
- All external dependencies mocked in unit tests
- Proper cleanup in fixtures

**Test maintainability**:
- Clear test names describing the scenario
- Arrange-Act-Assert pattern
- Proper use of fixtures and parameterization

**Test performance**:
- Unit tests: < 100ms per test
- Integration tests: < 5s per test  
- Performance tests: < 30s per test

### Mocking Strategies

**External dependencies**:
```python
@patch('src.core.parser.DocumentConverter')
def test_parser_with_mocked_docling(self, mock_converter):
    """Test parser logic without actual Docling calls."""
    mock_result = Mock()
    mock_result.document.texts = [Mock(text="Sample text")]
    mock_converter.return_value.convert.return_value = mock_result
    
    parser = DoclingParser()
    elements = parser.parse_document("mock.pdf")
    
    assert len(elements) == 1
    assert elements[0].text == "Sample text"
```

**File system operations**:
```python
@patch('pathlib.Path.exists')  
@patch('pathlib.Path.open')
def test_export_without_filesystem(self, mock_open, mock_exists):
    """Test export functionality without touching filesystem."""
    mock_exists.return_value = True
    mock_file = Mock()
    mock_open.return_value.__enter__.return_value = mock_file
    
    interface = VerificationInterface()
    interface.export_to_json("mock_path.json")
    
    mock_file.write.assert_called()
```

## Debugging Test Failures

### Common Test Failure Patterns

**Docling version compatibility**:
```python
# Check Docling version in failing tests
import docling
print(f"Docling version: {docling.__version__}")

# Version-specific test skipping
pytest.mark.skipif(
    docling.__version__ < "1.0.0", 
    reason="Requires Docling >= 1.0.0"
)
```

**OCR environment issues**:
```python
@pytest.mark.ocr
def test_ocr_functionality(self):
    """Test OCR with proper environment checking."""
    try:
        import tesserocr
    except ImportError:
        pytest.skip("Tesseract not available")
        
    # Test OCR functionality
    parser = DoclingParser(enable_ocr=True)
    elements = parser.parse_document(self.scanned_pdf)
    assert len(elements) > 0
```

**Memory-related failures**:
```bash  
# Run single test with memory profiling
pytest tests/test_parser_core.py::test_large_document -v -s \
    --capture=no --tb=short
    
# Monitor memory during test execution
python -m memory_profiler tests/test_parser_core.py
```

### Test Debugging Tools

**Verbose output**:
```bash
# Show detailed test output
pytest tests/ -v -s --tb=long

# Show local variables in tracebacks
pytest tests/ -v --tb=long --showlocals

# Drop into debugger on failure
pytest tests/ -v --pdb
```

**Test selection**:
```bash
# Run failed tests from last run
pytest --lf

# Run tests matching pattern
pytest -k "test_parser" -v

# Run tests by marker
pytest -m "not slow" -v
```

---

*This testing strategy ensures robust, maintainable, and performant code through comprehensive test coverage and systematic quality assurance.*
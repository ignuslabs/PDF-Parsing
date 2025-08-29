# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/ -v -m "not slow and not ocr and not performance"    # Fast tests only
pytest tests/ -v -m "slow or ocr"                                 # Slow/OCR tests
pytest tests/ -v -m "performance"                                 # Performance tests

# Run single test
pytest tests/test_parser_core.py::TestDoclingParser::test_parse_document_returns_elements -v

# Generate test fixtures before running tests
python generate_test_fixtures.py
```

### Code Quality
```bash
# Format code
black src/ tests/

# Check formatting (no changes)
black --check --diff src/ tests/

# Lint code
flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503

# Type checking
mypy src/ --ignore-missing-imports --strict-optional

# Run all quality checks
black --check --diff src/ tests/ && flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503 && mypy src/ --ignore-missing-imports --strict-optional
```

### Installation
```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Install in development mode
pip install -e .
```

## Architecture Overview

### Core Components

**Smart PDF Parser** is built around three main subsystems that work with Docling for PDF processing:

1. **Parser Layer (`src/core/parser.py`)**
   - **DoclingParser**: Main parsing engine using IBM's Docling library
   - **Two interfaces**: `parse_document()` returns `List[DocumentElement]`, `parse_document_full()` returns `ParsedDocument` with metadata
   - **Docling integration**: Converts `result.document.texts`, `result.document.tables`, `result.document.pictures` to internal `DocumentElement` objects
   - **Fallback handling**: OCR failure → retry without OCR, Memory issues → reduced settings

2. **Search Engine (`src/core/search.py`)**
   - **SmartSearchEngine**: Multi-modal search with exact, fuzzy, and semantic matching
   - **Intelligent ranking**: Element type boosting (headings > text), confidence scoring, position factors
   - **Advanced filtering**: Element type, page numbers, spatial regions, metadata
   - **Performance optimization**: LRU caching, pre-built indices

3. **Verification System (`src/verification/`)**
   - **PDFRenderer**: Visual overlay system with coordinate transformation (PDF ↔ pixel coordinates)
   - **VerificationInterface**: Interactive verification workflow with state management
   - **Export capabilities**: JSON/CSV export with correction tracking

### Data Models (`src/core/models.py`)

**Key data structures with validation**:
- `DocumentElement`: Core parsed element with text, type, bbox, confidence
- `ParsedDocument`: Complete document with elements + metadata + page images
- `SearchResult`: Search hit with relevance scoring and match context

### Docling Integration Details

**Critical understanding**: Docling returns `result.document` with:
- `document.texts`: List of text elements (paragraphs, headings)
- `document.tables`: List of table structures  
- `document.pictures`: List of images/figures

**Parser responsibility**: Transform these into standardized `DocumentElement` objects with normalized bounding boxes and element types.

### Test Strategy

**Test structure follows TDD approach** from `claude-workflow/`:
- **Fast tests** (default): Core functionality without OCR/large files
- **Integration tests** (`-m integration`): Real PDF processing with Docling
- **Performance tests** (`-m performance`): Memory usage and timing benchmarks
- **Test fixtures**: Generated PDFs in `tests/fixtures/` for consistent testing

### Configuration

**Code standards enforced**:
- Line length: 100 characters (Black, Flake8)
- Type checking: Strict with mypy (except external libraries)
- Import handling: Docling, PIL, fuzzywuzzy ignored for missing imports

### Important Notes

**Docling coordinate system**: PDF coordinates (bottom-left origin) need transformation to pixel coordinates (top-left origin) for verification rendering.

**Parser interfaces**: Always use `parse_document()` for simple element lists, `parse_document_full()` when you need metadata/configuration info.

**Test fixture dependency**: Run `python generate_test_fixtures.py` before tests to ensure PDF samples exist.
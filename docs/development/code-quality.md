# Code Quality Standards

*How-to guide for code formatting, linting, type checking, and documentation standards*

## Overview

Smart PDF Parser maintains **strict code quality standards** through automated tooling, comprehensive type checking, and consistent documentation. These standards ensure maintainable, readable, and reliable code across the entire project.

## Code Formatting

### Black Formatting

We use **Black** as our opinionated code formatter with specific configuration:

#### Configuration (`pyproject.toml`)

```toml
[tool.black]
line-length = 100
target-version = ['py39']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''
```

#### Usage

```bash
# Format all code (modifies files)
black src/ tests/

# Check formatting without changes
black --check --diff src/ tests/

# Format specific file
black src/core/parser.py

# Show what would change
black --diff src/core/parser.py
```

#### Black Standards

**Line Length**: Maximum 100 characters
- Balances readability with modern screen sizes
- Allows side-by-side diffs in most environments

**String Quotes**: Double quotes preferred
```python
# Correct
message = "This is a message"
docstring = """This is a docstring"""

# Black will auto-convert single quotes
message = 'This gets converted to double quotes'
```

**Import Formatting**:
```python
# Black organizes imports consistently
from typing import List, Optional, Dict, Any
from pathlib import Path

import pandas as pd
import numpy as np

from src.core.models import DocumentElement
from src.utils.exceptions import DocumentParsingError
```

### IDE Integration

#### VS Code Settings

```json
{
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=100"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

#### PyCharm Settings

```
File → Settings → Tools → External Tools
Name: Black
Program: black  
Arguments: --line-length=100 $FilePath$
Working Directory: $ProjectFileDir$
```

## Linting with Flake8

### Configuration

We use **Flake8** for style guide enforcement with custom configuration:

```bash
# Command line usage
flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503
```

#### Key Rules

**Line Length**: 100 characters (matching Black)
**Ignored Rules**:
- `E203`: Whitespace before ':' (conflicts with Black)
- `W503`: Line break before binary operator (conflicts with Black)

### Common Flake8 Violations

#### Import Organization (E401, F401)

```python
# Wrong - multiple imports on one line
import os, sys, json

# Correct - separate lines
import os
import sys
import json

# Wrong - unused import
import pandas as pd  # F401 if unused

# Correct - remove unused imports
# import pandas as pd
```

#### Whitespace Issues (E302, E303)

```python  
# Wrong - missing blank lines
class DoclingParser:
    def __init__(self):
        pass
    def parse_document(self):  # E302: Expected 2 blank lines
        pass

# Correct - proper spacing  
class DoclingParser:
    def __init__(self):
        pass

    def parse_document(self):  # Two blank lines before method
        pass
```

#### Line Length (E501)

```python
# Wrong - line too long
def parse_document(self, file_path: str, enable_ocr: bool = False, enable_tables: bool = True, max_pages: Optional[int] = None) -> List[DocumentElement]:

# Correct - break long lines
def parse_document(
    self,
    file_path: str, 
    enable_ocr: bool = False,
    enable_tables: bool = True,
    max_pages: Optional[int] = None
) -> List[DocumentElement]:
```

#### Naming Conventions (N801, N802, N806)

```python
# Wrong - invalid naming
class doclingParser:  # N801: Class names should be CamelCase
    def ParseDocument(self):  # N802: Function names should be lowercase
        MY_VAR = "value"  # N806: Variable should be lowercase
        
# Correct - proper naming
class DoclingParser:
    def parse_document(self):
        my_var = "value"
```

## Type Checking with MyPy

### Configuration (`pyproject.toml`)

```toml
[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true

[[tool.mypy.overrides]]
module = [
    "docling.*",
    "reportlab.*", 
    "PIL.*",
    "fuzzywuzzy.*",
    "Levenshtein.*"
]
ignore_missing_imports = true
```

### Type Annotation Standards

#### Function Signatures

```python
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

# Complete type annotations required
def parse_document(
    self,
    file_path: Union[str, Path],
    enable_ocr: bool = False,
    max_pages: Optional[int] = None
) -> List[DocumentElement]:
    """Parse PDF document into structured elements."""
    pass

# Return type always specified
def get_element_count(self) -> int:
    return len(self.elements)

# Complex return types
def search_elements(self, query: str) -> Dict[str, Any]:
    return {
        "results": [],
        "count": 0,
        "query_time": 0.0
    }
```

#### Class Attribute Annotations

```python  
from typing import ClassVar, Optional
from dataclasses import dataclass

class DoclingParser:
    """PDF parser using Docling."""
    
    # Instance attributes
    enable_ocr: bool
    enable_tables: bool  
    max_pages: Optional[int]
    
    # Class variables  
    DEFAULT_MAX_PAGES: ClassVar[int] = 1000
    SUPPORTED_FORMATS: ClassVar[List[str]] = [".pdf"]
    
    def __init__(self, enable_ocr: bool = False) -> None:
        self.enable_ocr = enable_ocr
        self.enable_tables = True
        self.max_pages = None
```

#### Dataclass Type Annotations

```python
from dataclasses import dataclass
from typing import Optional, List, Tuple

@dataclass
class DocumentElement:
    """Represents a parsed document element."""
    
    text: str
    element_type: str
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    page_num: int
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        """Validate element after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")
```

#### Generic Types and Protocols

```python
from typing import TypeVar, Generic, Protocol, runtime_checkable

T = TypeVar('T')

class SearchEngine(Generic[T]):
    """Generic search engine for any element type."""
    
    def __init__(self, elements: List[T]) -> None:
        self.elements = elements
    
    def search(self, query: str) -> List[T]:
        pass

@runtime_checkable  
class Searchable(Protocol):
    """Protocol for searchable objects."""
    
    def get_text(self) -> str: ...
    def get_element_type(self) -> str: ...
```

### Common MyPy Issues

#### Missing Return Type

```python
# Error: Function is missing a return type annotation
def calculate_similarity(text1, text2):  # Missing -> float
    return 0.85

# Fixed
def calculate_similarity(text1: str, text2: str) -> float:
    return 0.85
```

#### Incompatible Types

```python  
# Error: Incompatible return value type
def get_page_count(self) -> int:
    return "5"  # str is not compatible with int

# Fixed
def get_page_count(self) -> int:
    return 5
```

#### Optional Handling

```python
# Error: Item "None" has no attribute "upper"
def process_text(self, text: Optional[str]) -> str:
    return text.upper()  # text might be None

# Fixed - explicit None check
def process_text(self, text: Optional[str]) -> str:
    if text is None:
        return ""
    return text.upper()

# Alternative - using Union
def process_text(self, text: Optional[str]) -> Optional[str]:
    return text.upper() if text else None
```

### MyPy Usage

```bash
# Check entire project
mypy src/ --ignore-missing-imports --strict-optional

# Check specific module
mypy src/core/parser.py --ignore-missing-imports

# Show error context
mypy src/ --ignore-missing-imports --show-error-context

# Generate HTML report
mypy src/ --ignore-missing-imports --html-report mypy_report/
```

## Documentation Standards

### Docstring Format

We use **Google-style docstrings** with complete type information:

```python
def parse_document(
    self,
    file_path: Union[str, Path],
    enable_ocr: bool = False,
    max_pages: Optional[int] = None
) -> List[DocumentElement]:
    """Parse PDF document into structured elements using Docling.
    
    This method processes a PDF file and extracts text, tables, and images
    as structured DocumentElement objects. OCR can be enabled for scanned
    documents.
    
    Args:
        file_path: Path to the PDF file to parse. Can be string or Path object.
        enable_ocr: Whether to enable OCR for scanned documents. Defaults to False.
        max_pages: Maximum number of pages to process. If None, processes all pages.
        
    Returns:
        List of DocumentElement objects containing parsed content. Each element
        includes text, bounding box coordinates, element type, and metadata.
        
    Raises:
        DocumentParsingError: If the PDF file cannot be processed.
        OCRError: If OCR is enabled but fails to process scanned content.
        ValidationError: If the parsed elements fail validation.
        
    Example:
        >>> parser = DoclingParser(enable_ocr=True)
        >>> elements = parser.parse_document("document.pdf", max_pages=10)
        >>> print(f"Parsed {len(elements)} elements")
        Parsed 45 elements
        
    Note:
        OCR processing can be memory-intensive for large documents. Consider
        setting max_pages for very large files.
    """
    pass
```

### Class Documentation

```python
class SmartSearchEngine:
    """Multi-modal search engine for document elements.
    
    Provides exact, fuzzy, and semantic search capabilities across parsed
    document elements. Supports filtering by element type, page numbers,
    and spatial regions.
    
    The search engine builds internal indices for performance and supports
    advanced ranking based on element type relevance and position within
    the document.
    
    Attributes:
        elements: List of DocumentElement objects to search.
        index_built: Whether search indices have been constructed.
        cache_enabled: Whether to cache search results for repeated queries.
        
    Example:
        >>> elements = parser.parse_document("doc.pdf")
        >>> engine = SmartSearchEngine(elements)
        >>> results = engine.search("quarterly revenue", mode="fuzzy")
        >>> print(f"Found {len(results)} matches")
    """
    
    def __init__(self, elements: List[DocumentElement]) -> None:
        """Initialize search engine with document elements.
        
        Args:
            elements: List of parsed document elements to index and search.
            
        Raises:
            ValidationError: If elements list is empty or contains invalid elements.
        """
        pass
```

### Module Documentation

```python
"""
Core PDF parser implementation using Docling.

This module provides the DoclingParser class which serves as the main interface
for parsing PDF documents into structured elements. It integrates with IBM's
Docling library to handle complex document layouts, tables, and OCR processing.

The parser supports various configuration options including:
- OCR enablement for scanned documents  
- Table structure recognition
- Page image generation for verification
- Memory optimization settings

Typical usage:
    >>> from src.core.parser import DoclingParser
    >>> parser = DoclingParser(enable_ocr=True, enable_tables=True)
    >>> elements = parser.parse_document("document.pdf")
    
Classes:
    DoclingParser: Main PDF parsing interface using Docling
    
Functions:
    validate_pdf_path: Utility function to validate PDF file paths
    
Exceptions:
    DocumentParsingError: Raised when document parsing fails
    OCRError: Raised when OCR processing encounters errors
"""

import os
import logging
# ... rest of module
```

## Code Organization Standards

### Import Organization

Following **PEP 8** import order with **isort** compatibility:

```python
"""Module docstring."""

# 1. Standard library imports
import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from datetime import datetime

# 2. Third-party imports  
import numpy as np
import pandas as pd
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions

# 3. Local application imports
from src.core.models import DocumentElement, ParsedDocument
from src.utils.exceptions import DocumentParsingError, OCRError
from src.verification.interface import VerificationInterface
```

### File Organization

```python
"""Module docstring at top."""

# Constants at module level
DEFAULT_OCR_LANGUAGE = "eng"
MAX_MEMORY_MB = 1000  
SUPPORTED_EXTENSIONS = [".pdf"]

# Type aliases for clarity
BoundingBox = Tuple[float, float, float, float]
ElementID = str

# Exception classes (if module-specific)
class ModuleSpecificError(Exception):
    """Exception specific to this module."""
    pass

# Helper functions before classes
def _validate_path(path: Union[str, Path]) -> Path:
    """Private helper function."""
    pass

def validate_pdf_path(path: Union[str, Path]) -> Path:
    """Public utility function."""
    pass

# Main classes
class DoclingParser:
    """Main class implementation."""
    pass
```

### Method Organization Within Classes

```python
class DoclingParser:
    """PDF parser using Docling."""
    
    # 1. Class variables
    SUPPORTED_FORMATS: ClassVar[List[str]] = [".pdf"]
    
    # 2. Initialization
    def __init__(self, enable_ocr: bool = False) -> None:
        pass
    
    # 3. Properties  
    @property
    def converter(self) -> DocumentConverter:
        """Lazy-initialized document converter."""
        pass
        
    # 4. Private methods (implementation details)
    def _validate_config(self) -> None:
        """Validate parser configuration."""
        pass
        
    def _create_converter(self) -> DocumentConverter:
        """Create configured Docling converter."""
        pass
    
    # 5. Public interface methods
    def parse_document(self, file_path: Union[str, Path]) -> List[DocumentElement]:
        """Main public parsing method."""
        pass
        
    def parse_document_full(self, file_path: Union[str, Path]) -> ParsedDocument:
        """Extended parsing with metadata."""
        pass
    
    # 6. Special methods last
    def __repr__(self) -> str:
        return f"DoclingParser(ocr={self.enable_ocr}, tables={self.enable_tables})"
```

## Error Handling Standards

### Exception Hierarchy

```python
# Base exception for all parser errors
class PDFParserError(Exception):
    """Base exception for PDF parser errors."""
    pass

# Specific exceptions with context
class DocumentParsingError(PDFParserError):
    """Raised when document parsing fails."""
    
    def __init__(self, message: str, document_path: str = None, cause: Exception = None):
        self.document_path = document_path
        self.cause = cause
        super().__init__(message)

# Usage in code
def parse_document(self, file_path: Union[str, Path]) -> List[DocumentElement]:
    """Parse document with proper error handling."""
    try:
        # Docling processing
        result = self.converter.convert(file_path)
        return self._convert_to_elements(result)
    except Exception as e:
        logger.error(f"Failed to parse {file_path}: {e}")
        raise DocumentParsingError(
            f"Document parsing failed: {e}",
            document_path=str(file_path),
            cause=e
        )
```

### Logging Standards

```python
import logging

# Module-level logger
logger = logging.getLogger(__name__)

class DoclingParser:
    """Parser with comprehensive logging."""
    
    def parse_document(self, file_path: Union[str, Path]) -> List[DocumentElement]:
        """Parse document with detailed logging."""
        logger.info(f"Starting document parsing: {file_path}")
        
        try:
            # Validate input
            validated_path = self._validate_path(file_path)
            logger.debug(f"Validated path: {validated_path}")
            
            # Process document  
            result = self.converter.convert(validated_path)
            logger.info(f"Docling conversion completed: {len(result.document.texts)} text elements")
            
            # Convert to internal format
            elements = self._convert_to_elements(result)
            logger.info(f"Parsed {len(elements)} total elements")
            
            return elements
            
        except Exception as e:
            logger.error(f"Parsing failed for {file_path}: {e}", exc_info=True)
            raise DocumentParsingError(f"Document parsing failed: {e}")
```

## Quality Automation

### Pre-commit Configuration

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        args: [--line-length=100]
        
  - repo: https://github.com/pycqa/flake8  
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=100, --extend-ignore=E203,W503]
        
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.0  
    hooks:
      - id: mypy
        args: [--ignore-missing-imports, --strict-optional]
```

### Makefile for Quality Checks

Create `Makefile`:

```makefile
.PHONY: format lint type-check quality test

# Format code with Black
format:
	black src/ tests/

# Check formatting  
format-check:
	black --check --diff src/ tests/

# Lint with Flake8
lint:
	flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503

# Type check with MyPy  
type-check:
	mypy src/ --ignore-missing-imports --strict-optional

# Run all quality checks
quality: format-check lint type-check
	@echo "All quality checks passed!"

# Run tests
test:
	pytest tests/ -v -m "not slow and not ocr and not performance"

# Complete check before commit
check: quality test
	@echo "Ready for commit!"
```

### VS Code Tasks Configuration

Create `.vscode/tasks.json`:

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Format with Black",
            "type": "shell", 
            "command": "black",
            "args": ["src/", "tests/"],
            "group": "build"
        },
        {
            "label": "Lint with Flake8",
            "type": "shell",
            "command": "flake8", 
            "args": ["src/", "tests/", "--max-line-length=100", "--extend-ignore=E203,W503"],
            "group": "build"
        },
        {
            "label": "Type Check",
            "type": "shell",
            "command": "mypy",
            "args": ["src/", "--ignore-missing-imports", "--strict-optional"], 
            "group": "build"
        },
        {
            "label": "Quality Check All",
            "dependsOrder": "sequence",
            "dependsOn": ["Format with Black", "Lint with Flake8", "Type Check"],
            "group": {
                "kind": "build",
                "isDefault": true
            }
        }
    ]
}
```

## Quality Metrics and Monitoring  

### Code Complexity

Monitor code complexity with **radon**:

```bash  
# Install radon
pip install radon

# Check cyclomatic complexity
radon cc src/ -a -nb

# Check maintainability index  
radon mi src/ -nb

# Check Halstead metrics
radon hal src/
```

### Documentation Coverage  

Monitor docstring coverage:

```bash
# Install pydocstyle
pip install pydocstyle

# Check docstring compliance
pydocstyle src/ --convention=google

# Generate documentation coverage report
pip install interrogate
interrogate -v src/
```

---

*These code quality standards ensure consistent, maintainable, and professional code across the Smart PDF Parser project.*
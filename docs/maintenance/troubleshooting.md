# Troubleshooting Procedures

*How-to guide for diagnosing and resolving common issues in Smart PDF Parser*

## Overview

This guide provides systematic procedures for diagnosing and resolving common issues in Smart PDF Parser. Each section includes symptoms, diagnosis steps, and resolution procedures with specific commands and code examples.

## Quick Diagnosis Checklist

### Initial System Check

```bash
#!/bin/bash
# Quick system health check script

echo "=== Smart PDF Parser Health Check ==="
echo "Date: $(date)"
echo

# 1. Process status
echo "1. Process Status:"
if pgrep -f "smart-pdf-parser" > /dev/null; then
    echo "✓ Application is running"
    ps -p $(pgrep -f "smart-pdf-parser") -o pid,rss,vsz,pmem --no-headers
else
    echo "✗ Application is not running"
fi
echo

# 2. Memory usage
echo "2. System Memory:"
free -h | grep -E "Mem:|Swap:"
echo

# 3. Disk space
echo "3. Disk Space:"
df -h | head -1
df -h | grep -E "/$|/tmp|/var"
echo

# 4. Recent errors  
echo "4. Recent Errors (last 10 lines):"
if [ -f "logs/smart-pdf-parser.log" ]; then
    tail -n 100 logs/smart-pdf-parser.log | grep -i error | tail -10
else
    echo "No log file found"
fi
echo

# 5. Dependencies
echo "5. Key Dependencies:"
python -c "
try:
    import docling
    print(f'✓ Docling: {docling.__version__}')
except ImportError:
    print('✗ Docling: Not installed')

try:
    import tesserocr
    print('✓ Tesseract OCR: Available')
except ImportError:
    print('✗ Tesseract OCR: Not available')
    
try:
    import streamlit
    print(f'✓ Streamlit: {streamlit.__version__}')
except ImportError:
    print('✗ Streamlit: Not installed')
"
```

### Common Issue Categories

**Performance Issues**:
- Slow document processing
- High memory usage
- Search timeouts
- UI responsiveness problems

**Processing Failures**:
- Document parsing errors
- OCR failures
- Memory exhaustion
- File format issues

**System Issues**:
- Session state corruption
- Dependency conflicts
- Configuration errors
- Network connectivity

## Document Processing Issues

### Slow Document Processing

#### Symptoms
```
- Processing times > 60 seconds for typical documents
- CPU usage consistently high (>80%)
- UI becomes unresponsive during processing
- Timeout errors in logs
```

#### Diagnosis

**Step 1: Check Document Characteristics**
```python
# Analyze document properties
import os
from pathlib import Path

def diagnose_document(file_path: str):
    """Diagnose document characteristics affecting performance."""
    
    path = Path(file_path)
    if not path.exists():
        print(f"File not found: {file_path}")
        return
    
    file_size_mb = os.path.getsize(path) / 1024 / 1024
    
    print(f"Document Analysis: {path.name}")
    print(f"File size: {file_size_mb:.1f} MB")
    
    # Try to get page count without full processing
    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(path)
        page_count = len(result.document.pages) if hasattr(result.document, 'pages') else 'Unknown'
        print(f"Pages: {page_count}")
        
        # Check for images/complex content
        if hasattr(result.document, 'pictures'):
            print(f"Images: {len(result.document.pictures)}")
        if hasattr(result.document, 'tables'):
            print(f"Tables: {len(result.document.tables)}")
            
    except Exception as e:
        print(f"Error analyzing document: {e}")
    
    # Performance recommendations
    if file_size_mb > 50:
        print("⚠️  Large file detected - consider processing in chunks")
    if file_size_mb > 100:
        print("⚠️  Very large file - may require memory optimization")

# Usage
diagnose_document("path/to/slow_document.pdf")
```

**Step 2: Monitor Resource Usage**
```bash
# Monitor processing in real-time
#!/bin/bash

PID=$(pgrep -f "smart-pdf-parser")
if [ -z "$PID" ]; then
    echo "Application not running"
    exit 1
fi

echo "Monitoring process $PID"
echo "Time,CPU%,Memory(MB),Threads"

for i in {1..60}; do
    STATS=$(ps -p $PID -o pcpu,rss,nlwp --no-headers)
    if [ $? -eq 0 ]; then
        CPU=$(echo $STATS | awk '{print $1}')
        MEM_MB=$(echo $STATS | awk '{print int($2/1024)}')
        THREADS=$(echo $STATS | awk '{print $3}')
        echo "$(date +%H:%M:%S),$CPU,$MEM_MB,$THREADS"
    else
        echo "Process ended"
        break
    fi
    sleep 1
done
```

#### Resolution

**Solution 1: Optimize Parser Configuration**
```python  
# Use optimized settings for large documents
from src.core.parser import DoclingParser

def create_optimized_parser(document_size_mb: float) -> DoclingParser:
    """Create parser with optimized settings based on document size."""
    
    if document_size_mb < 10:
        # Small documents - full processing
        return DoclingParser(
            enable_ocr=True,
            enable_tables=True,
            generate_page_images=True
        )
    elif document_size_mb < 50:
        # Medium documents - selective processing
        return DoclingParser(
            enable_ocr=False,  # Disable OCR for speed
            enable_tables=True,
            generate_page_images=False,  # Skip images
            table_mode="fast"  # Use fast table processing
        )
    else:
        # Large documents - minimal processing
        return DoclingParser(
            enable_ocr=False,
            enable_tables=False,  # Skip table processing
            generate_page_images=False,
            max_pages=100  # Limit pages processed
        )

# Usage
parser = create_optimized_parser(75.0)  # 75MB document
elements = parser.parse_document("large_document.pdf")
```

**Solution 2: Implement Chunked Processing**
```python
def process_large_document_chunked(file_path: str, chunk_size: int = 50) -> List[DocumentElement]:
    """Process large documents in chunks to manage memory."""
    
    all_elements = []
    parser = DoclingParser(enable_ocr=False, enable_tables=True)
    
    try:
        # First, determine total pages
        temp_parser = DoclingParser(max_pages=1)
        temp_result = temp_parser.parse_document_full(file_path)
        total_pages = temp_result.metadata.get("total_pages", 100)
        
        print(f"Processing {total_pages} pages in chunks of {chunk_size}")
        
        # Process in chunks
        for start_page in range(1, total_pages + 1, chunk_size):
            end_page = min(start_page + chunk_size - 1, total_pages)
            
            print(f"Processing pages {start_page}-{end_page}")
            
            # Configure parser for page range
            chunk_parser = DoclingParser(
                enable_ocr=False,
                enable_tables=True,
                page_range=(start_page, end_page)  # Hypothetical feature
            )
            
            chunk_elements = chunk_parser.parse_document(file_path)
            all_elements.extend(chunk_elements)
            
            # Force garbage collection between chunks
            import gc
            gc.collect()
            
    except Exception as e:
        logger.error(f"Chunked processing failed: {e}")
        raise
    
    return all_elements
```

### High Memory Usage

#### Symptoms
```
- Memory usage > 2GB for typical documents
- "Out of memory" errors
- System becomes unresponsive
- Process killed by OS
```

#### Diagnosis

**Memory Profiling**
```python
import psutil
import tracemalloc
from memory_profiler import profile

@profile
def debug_memory_usage():
    """Profile memory usage during parsing."""
    
    # Start tracing
    tracemalloc.start()
    
    # Monitor process memory
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    print(f"Initial memory: {initial_memory / 1024 / 1024:.1f} MB")
    
    # Parse document
    parser = DoclingParser(enable_ocr=True)
    elements = parser.parse_document("test_document.pdf")
    
    # Check memory after parsing
    final_memory = process.memory_info().rss
    print(f"Final memory: {final_memory / 1024 / 1024:.1f} MB")
    print(f"Memory increase: {(final_memory - initial_memory) / 1024 / 1024:.1f} MB")
    
    # Get top memory allocations
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
    
    # Top memory consumers
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    
    print("\nTop 10 memory allocations:")
    for stat in top_stats[:10]:
        print(stat)
    
    tracemalloc.stop()

# Run memory analysis
debug_memory_usage()
```

#### Resolution

**Solution 1: Memory-Optimized Processing**
```python
class MemoryOptimizedParser:
    """Parser with memory optimization techniques."""
    
    def __init__(self, memory_limit_mb: int = 1000):
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self.process = psutil.Process()
    
    def check_memory_usage(self) -> bool:
        """Check if memory usage is within limits."""
        current_memory = self.process.memory_info().rss
        return current_memory < self.memory_limit_bytes
    
    def parse_document_safe(self, file_path: str) -> List[DocumentElement]:
        """Parse document with memory monitoring and cleanup."""
        
        if not self.check_memory_usage():
            # Force garbage collection
            import gc
            gc.collect()
            
            if not self.check_memory_usage():
                raise MemoryError("Insufficient memory before parsing")
        
        # Use minimal settings for memory efficiency
        parser = DoclingParser(
            enable_ocr=False,  # OCR is memory-intensive
            enable_tables=True,
            generate_page_images=False  # Skip image generation
        )
        
        try:
            elements = parser.parse_document(file_path)
            
            # Monitor memory during processing
            if not self.check_memory_usage():
                logger.warning("Memory usage exceeded limit during parsing")
                
            return elements
            
        finally:
            # Cleanup
            del parser
            import gc
            gc.collect()

# Usage
memory_parser = MemoryOptimizedParser(memory_limit_mb=800)
elements = memory_parser.parse_document_safe("document.pdf")
```

**Solution 2: Streaming Processing**
```python
def process_document_streaming(file_path: str) -> Iterator[DocumentElement]:
    """Process document as a stream to minimize memory usage."""
    
    parser = DoclingParser(enable_ocr=False)
    
    # Process document page by page
    for page_num in range(1, get_page_count(file_path) + 1):
        # Create single-page parser
        page_parser = DoclingParser(
            max_pages=1,
            page_offset=page_num - 1  # Hypothetical feature
        )
        
        try:
            page_elements = page_parser.parse_document(file_path)
            
            # Yield elements one by one
            for element in page_elements:
                yield element
                
        finally:
            # Cleanup after each page
            del page_parser
            import gc
            gc.collect()

# Usage - process as stream
for element in process_document_streaming("large_document.pdf"):
    # Process individual elements without loading all into memory
    process_element(element)
```

## OCR Processing Issues

### OCR Engine Failures

#### Symptoms
```
- "OCR processing failed" errors
- Empty text extraction from scanned documents
- Tesseract/EasyOCR import errors
- Language detection failures
```

#### Diagnosis

**OCR Environment Check**
```bash
#!/bin/bash
# OCR diagnostics script

echo "=== OCR Environment Diagnostics ==="

# Check Tesseract installation
if command -v tesseract &> /dev/null; then
    echo "✓ Tesseract binary found: $(which tesseract)"
    echo "  Version: $(tesseract --version | head -1)"
    echo "  Languages: $(tesseract --list-langs | tail -n +2 | tr '\n' ' ')"
else
    echo "✗ Tesseract binary not found"
fi

# Check Python OCR bindings
python3 -c "
try:
    import tesserocr
    print('✓ tesserocr Python binding available')
    print(f'  Version: {tesserocr.tesseract_version()}')
except ImportError as e:
    print(f'✗ tesserocr not available: {e}')

try:
    import easyocr
    print('✓ EasyOCR available')
except ImportError as e:
    print(f'✗ EasyOCR not available: {e}')
"

# Check for common OCR issues
echo
echo "System Libraries:"
ldconfig -p | grep -E "(lept|tesseract)" | head -5
```

**OCR Processing Test**
```python
def test_ocr_functionality():
    """Test OCR engines with sample document."""
    
    # Test Tesseract
    try:
        import tesserocr
        from PIL import Image
        
        # Create test image with text
        test_image = Image.new('RGB', (200, 50), color='white')
        # Add text to image (implementation depends on PIL/drawing library)
        
        # Test OCR
        text = tesserocr.image_to_text(test_image)
        print(f"Tesseract test result: '{text.strip()}'")
        
        if not text.strip():
            print("⚠️  Tesseract produced no output")
        else:
            print("✓ Tesseract working correctly")
            
    except Exception as e:
        print(f"✗ Tesseract test failed: {e}")
    
    # Test EasyOCR
    try:
        import easyocr
        reader = easyocr.Reader(['en'])
        
        # Test with sample image
        results = reader.readtext(test_image, detail=0)
        print(f"EasyOCR test result: {results}")
        
        if results:
            print("✓ EasyOCR working correctly")
        else:
            print("⚠️  EasyOCR produced no results")
            
    except Exception as e:
        print(f"✗ EasyOCR test failed: {e}")

test_ocr_functionality()
```

#### Resolution

**Solution 1: OCR Engine Installation/Repair**
```bash
# Ubuntu/Debian OCR setup
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-fra tesseract-ocr-deu
sudo apt-get install -y libtesseract-dev libleptonica-dev

# Install Python bindings
pip install tesserocr easyocr

# macOS OCR setup
brew install tesseract
pip install tesserocr easyocr

# Verify installation
tesseract --version
python -c "import tesserocr; print('tesserocr working')"
python -c "import easyocr; print('easyocr working')"
```

**Solution 2: OCR Fallback Strategy**
```python
class RobustOCRProcessor:
    """OCR processor with multiple engines and fallback strategies."""
    
    def __init__(self):
        self.engines = []
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize available OCR engines."""
        
        # Try Tesseract
        try:
            import tesserocr
            self.engines.append(('tesseract', tesserocr))
            print("✓ Tesseract OCR available")
        except ImportError:
            print("⚠️  Tesseract OCR not available")
        
        # Try EasyOCR
        try:
            import easyocr
            reader = easyocr.Reader(['en'], gpu=False)  # CPU only
            self.engines.append(('easyocr', reader))
            print("✓ EasyOCR available")
        except ImportError:
            print("⚠️  EasyOCR not available")
    
    def extract_text_with_fallback(self, image_path: str, languages: List[str] = None) -> str:
        """Extract text using available OCR engines with fallback."""
        
        languages = languages or ['eng']
        
        for engine_name, engine in self.engines:
            try:
                print(f"Trying OCR with {engine_name}")
                
                if engine_name == 'tesseract':
                    text = self._extract_with_tesseract(image_path, languages, engine)
                elif engine_name == 'easyocr':
                    text = self._extract_with_easyocr(image_path, languages, engine)
                else:
                    continue
                
                if text and text.strip():
                    print(f"✓ Success with {engine_name}")
                    return text.strip()
                else:
                    print(f"⚠️  {engine_name} returned empty result")
                    
            except Exception as e:
                print(f"✗ {engine_name} failed: {e}")
                continue
        
        # No engine worked
        print("✗ All OCR engines failed")
        return ""
    
    def _extract_with_tesseract(self, image_path: str, languages: List[str], tesserocr) -> str:
        """Extract text using Tesseract."""
        from PIL import Image
        
        image = Image.open(image_path)
        lang_string = '+'.join(languages)
        
        return tesserocr.image_to_text(image, lang=lang_string)
    
    def _extract_with_easyocr(self, image_path: str, languages: List[str], reader) -> str:
        """Extract text using EasyOCR."""
        
        # Convert language codes if needed
        lang_map = {'eng': 'en', 'fra': 'fr', 'deu': 'de'}
        easyocr_langs = [lang_map.get(lang, lang) for lang in languages]
        
        results = reader.readtext(image_path, detail=0)
        return ' '.join(results)

# Usage
ocr_processor = RobustOCRProcessor()
text = ocr_processor.extract_text_with_fallback("scanned_page.png", ['eng'])
```

## Session State Issues

### Streamlit Session State Corruption

#### Symptoms
```
- "KeyError" in session state access
- UI components not responding
- Cached data becomes stale
- Page refresh required frequently
```

#### Diagnosis

**Session State Analysis**
```python
import streamlit as st

def diagnose_session_state():
    """Diagnose session state issues."""
    
    st.write("## Session State Diagnostics")
    
    # Show all session state keys
    st.write("### Current Session State Keys:")
    for key in st.session_state.keys():
        try:
            value = st.session_state[key]
            value_type = type(value).__name__
            value_size = len(str(value)) if hasattr(value, '__len__') else 0
            st.write(f"- `{key}`: {value_type} (size: {value_size})")
        except Exception as e:
            st.write(f"- `{key}`: Error accessing - {e}")
    
    # Show memory usage
    import sys
    session_size = sum(sys.getsizeof(v) for v in st.session_state.values())
    st.write(f"### Total Session State Size: {session_size / 1024:.1f} KB")
    
    # Check for problematic patterns
    large_items = []
    for key, value in st.session_state.items():
        size = sys.getsizeof(value)
        if size > 1024 * 100:  # > 100KB
            large_items.append((key, size))
    
    if large_items:
        st.warning("Large items in session state:")
        for key, size in large_items:
            st.write(f"- {key}: {size / 1024:.1f} KB")

# Add to your Streamlit app for debugging
if st.sidebar.button("Diagnose Session State"):
    diagnose_session_state()
```

#### Resolution

**Solution 1: Robust Session State Management**
```python
class SessionStateManager:
    """Robust session state management with error handling."""
    
    @staticmethod
    def safe_get(key: str, default=None, required_type=None):
        """Safely get value from session state with type checking."""
        
        try:
            if key in st.session_state:
                value = st.session_state[key]
                
                # Type checking if specified
                if required_type and not isinstance(value, required_type):
                    st.warning(f"Session state key '{key}' has unexpected type: {type(value)}")
                    st.session_state[key] = default
                    return default
                
                return value
            else:
                # Key doesn't exist, set default
                st.session_state[key] = default
                return default
                
        except Exception as e:
            st.error(f"Error accessing session state key '{key}': {e}")
            st.session_state[key] = default
            return default
    
    @staticmethod
    def safe_set(key: str, value, max_size_kb: int = 1024):
        """Safely set session state value with size limits."""
        
        try:
            # Check size limit
            import sys
            value_size = sys.getsizeof(value)
            
            if value_size > max_size_kb * 1024:
                st.warning(f"Value for '{key}' is too large ({value_size / 1024:.1f} KB)")
                return False
            
            st.session_state[key] = value
            return True
            
        except Exception as e:
            st.error(f"Error setting session state key '{key}': {e}")
            return False
    
    @staticmethod
    def cleanup_large_items(max_total_size_mb: int = 10):
        """Clean up large session state items."""
        
        import sys
        
        # Calculate current total size
        total_size = sum(sys.getsizeof(v) for v in st.session_state.values())
        
        if total_size > max_total_size_mb * 1024 * 1024:
            # Find and remove largest items
            item_sizes = [(k, sys.getsizeof(v)) for k, v in st.session_state.items()]
            item_sizes.sort(key=lambda x: x[1], reverse=True)
            
            removed_items = []
            for key, size in item_sizes:
                if total_size <= max_total_size_mb * 1024 * 1024:
                    break
                
                # Don't remove essential UI state
                if key in ['uploaded_file', 'parsed_elements', 'search_results']:
                    continue
                
                del st.session_state[key]
                total_size -= size
                removed_items.append(key)
            
            if removed_items:
                st.info(f"Cleaned up session state items: {', '.join(removed_items)}")

# Usage in Streamlit app
def initialize_app_state():
    """Initialize application state safely."""
    
    # Initialize core state variables
    SessionStateManager.safe_set('parsed_elements', [], max_size_kb=5120)  # 5MB limit
    SessionStateManager.safe_set('search_results', [])
    SessionStateManager.safe_set('verification_state', {})
    SessionStateManager.safe_set('processing_status', 'idle')
    
    # Cleanup if needed
    SessionStateManager.cleanup_large_items(max_total_size_mb=20)

# Call at app startup
initialize_app_state()
```

**Solution 2: Alternative State Storage**
```python
import sqlite3
import pickle
import hashlib
from pathlib import Path

class PersistentStateManager:
    """Alternative state management using SQLite for large data."""
    
    def __init__(self, db_path: str = "app_state.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for state storage."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS app_state (
                session_id TEXT,
                key TEXT,
                value BLOB,
                timestamp REAL,
                PRIMARY KEY (session_id, key)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def get_session_id(self) -> str:
        """Get or create session ID."""
        
        if 'session_id' not in st.session_state:
            # Create session ID from user session
            session_data = str(st.session_state.get('session_id', '')) + str(time.time())
            st.session_state['session_id'] = hashlib.md5(session_data.encode()).hexdigest()
        
        return st.session_state['session_id']
    
    def store_large_data(self, key: str, data):
        """Store large data in persistent storage."""
        
        session_id = self.get_session_id()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Serialize data
        serialized_data = pickle.dumps(data)
        
        cursor.execute("""
            INSERT OR REPLACE INTO app_state (session_id, key, value, timestamp)
            VALUES (?, ?, ?, ?)
        """, (session_id, key, serialized_data, time.time()))
        
        conn.commit()
        conn.close()
        
        # Store reference in session state
        st.session_state[f"{key}_ref"] = f"db:{session_id}:{key}"
    
    def load_large_data(self, key: str, default=None):
        """Load large data from persistent storage."""
        
        session_id = self.get_session_id()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT value FROM app_state 
            WHERE session_id = ? AND key = ?
        """, (session_id, key))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return pickle.loads(result[0])
        else:
            return default

# Usage
persistent_state = PersistentStateManager()

# Store large parsed document data
if elements:
    persistent_state.store_large_data('parsed_elements', elements)
    st.success(f"Stored {len(elements)} elements in persistent storage")

# Load data when needed
elements = persistent_state.load_large_data('parsed_elements', default=[])
```

## Configuration and Environment Issues

### Dependency Conflicts

#### Symptoms
```
- Import errors for core dependencies
- Version compatibility warnings
- Unexpected behavior after updates
- "Module not found" errors
```

#### Diagnosis

**Dependency Analysis**
```bash
#!/bin/bash
# Dependency diagnostics

echo "=== Dependency Analysis ==="

# Check Python version
echo "Python version: $(python --version)"

# Check key package versions
echo -e "\nKey Dependencies:"
python -c "
packages = ['docling', 'streamlit', 'pandas', 'numpy', 'pillow', 'opencv-python']

for pkg in packages:
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'Unknown')
        print(f'✓ {pkg}: {version}')
    except ImportError:
        print(f'✗ {pkg}: Not installed')
    except Exception as e:
        print(f'? {pkg}: Error - {e}')
"

# Check for conflicts
echo -e "\nChecking for conflicts:"
pip check

# Show dependency tree
echo -e "\nDependency tree for key packages:"
pip show docling streamlit pandas
```

#### Resolution

**Solution 1: Clean Environment Setup**
```bash
#!/bin/bash
# Clean dependency installation

# Create fresh virtual environment
python -m venv clean_env
source clean_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies in correct order
pip install --upgrade setuptools wheel

# Install core dependencies first
pip install numpy pandas pillow

# Install application dependencies
pip install -r requirements.txt

# Verify installation
python -c "
import docling
import streamlit
import pandas
import numpy
print('All dependencies installed successfully')
"
```

**Solution 2: Docker Environment**
```dockerfile
# Dockerfile for consistent environment
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY tests/ ./tests/

# Set environment variables
ENV PYTHONPATH=/app/src

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import src.core.parser; print('OK')" || exit 1

# Run application
CMD ["streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Emergency Recovery Procedures

### Complete System Reset

```bash
#!/bin/bash
# Emergency reset script

echo "=== Smart PDF Parser Emergency Reset ==="
echo "This will reset all application state and cached data"
read -p "Continue? (y/N): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Stop application
    pkill -f "smart-pdf-parser" 2>/dev/null || true
    pkill -f "streamlit" 2>/dev/null || true
    
    # Clear cache directories
    rm -rf ~/.streamlit/cache/ 2>/dev/null || true
    rm -rf .streamlit/ 2>/dev/null || true
    rm -rf __pycache__/ 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    
    # Clear logs
    rm -rf logs/*.log 2>/dev/null || true
    
    # Clear temporary files
    rm -rf /tmp/smart_pdf_parser_* 2>/dev/null || true
    
    # Reset database
    rm -f app_state.db 2>/dev/null || true
    
    # Recreate necessary directories
    mkdir -p logs
    
    echo "✓ System reset complete"
    echo "✓ Restart the application to continue"
else
    echo "Reset cancelled"
fi
```

### Data Recovery

```python
def recover_lost_session_data():
    """Attempt to recover lost session data."""
    
    recovery_sources = [
        "app_state.db",
        "logs/smart-pdf-parser.log",
        "/tmp/smart_pdf_parser_backup.json"
    ]
    
    recovered_data = {}
    
    for source in recovery_sources:
        if Path(source).exists():
            try:
                if source.endswith('.db'):
                    # Recover from SQLite
                    conn = sqlite3.connect(source)
                    cursor = conn.cursor()
                    cursor.execute("SELECT key, value FROM app_state")
                    for key, value in cursor.fetchall():
                        try:
                            recovered_data[key] = pickle.loads(value)
                        except:
                            pass
                    conn.close()
                    
                elif source.endswith('.log'):
                    # Recover from logs
                    with open(source, 'r') as f:
                        for line in f:
                            if 'parsed_elements' in line:
                                # Extract structured data from logs
                                pass
                                
                elif source.endswith('.json'):
                    # Recover from JSON backup
                    with open(source, 'r') as f:
                        data = json.load(f)
                        recovered_data.update(data)
                        
            except Exception as e:
                print(f"Failed to recover from {source}: {e}")
    
    return recovered_data
```

---

*This troubleshooting guide provides systematic approaches to diagnose and resolve the most common issues in Smart PDF Parser deployments.*
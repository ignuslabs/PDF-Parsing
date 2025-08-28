# Utilities Implementation Guide

## Overview

The utilities module provides essential helper functions, data processing tools, validation utilities, and common functionality used across the Smart PDF Parser application.

## Architecture

### Utility Module Structure

```
src/utils/
├── __init__.py            # Package initialization
├── validation.py          # Data validation utilities
├── text_processing.py     # Text processing and cleaning
├── coordinate_utils.py    # Coordinate system utilities
├── file_operations.py     # File handling utilities
├── logging_config.py      # Logging configuration
├── performance.py         # Performance monitoring
├── export_helpers.py      # Export format helpers
├── config_loader.py       # Configuration management
└── exceptions.py          # Custom exceptions
```

## Core Utility Modules

### Validation Utilities

```python
# src/utils/validation.py
from typing import Dict, List, Any, Optional, Union, Tuple
import re
from pathlib import Path
from dataclasses import dataclass
from src.core.models import DocumentElement

@dataclass
class ValidationResult:
    """Result of validation operation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
    def add_error(self, message: str):
        """Add validation error"""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str):
        """Add validation warning"""
        self.warnings.append(message)

class DocumentValidator:
    """Validates document structure and content"""
    
    @staticmethod
    def validate_element(element: DocumentElement) -> ValidationResult:
        """Validate a single document element"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Text validation
        if not element.text or not element.text.strip():
            result.add_warning("Element has empty or whitespace-only text")
        
        if len(element.text) > 10000:  # Arbitrary large text threshold
            result.add_warning("Element text is unusually long")
        
        # Element type validation
        valid_types = {'text', 'title', 'header', 'footer', 'table', 'image', 'caption'}
        if element.element_type not in valid_types:
            result.add_error(f"Invalid element type: {element.element_type}")
        
        # Page number validation
        if element.page_number < 1:
            result.add_error(f"Invalid page number: {element.page_number}")
        
        # Confidence validation
        if not 0.0 <= element.confidence <= 1.0:
            result.add_error(f"Invalid confidence score: {element.confidence}")
        
        # Bounding box validation
        bbox_result = BoundingBoxValidator.validate_bbox(element.bbox)
        result.errors.extend(bbox_result.errors)
        result.warnings.extend(bbox_result.warnings)
        if not bbox_result.is_valid:
            result.is_valid = False
        
        return result
    
    @staticmethod
    def validate_elements_consistency(elements: List[DocumentElement]) -> ValidationResult:
        """Validate consistency across multiple elements"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        if not elements:
            result.add_error("No elements provided for validation")
            return result
        
        # Check page number consistency
        page_numbers = set(e.page_number for e in elements)
        max_page = max(page_numbers)
        min_page = min(page_numbers)
        
        # Look for page gaps
        expected_pages = set(range(min_page, max_page + 1))
        missing_pages = expected_pages - page_numbers
        if missing_pages:
            result.add_warning(f"Missing pages: {sorted(missing_pages)}")
        
        # Check for duplicate content
        text_hashes = {}
        for element in elements:
            text_hash = hash(element.text.strip().lower())
            if text_hash in text_hashes:
                result.add_warning(f"Potential duplicate content found: {element.text[:50]}...")
            text_hashes[text_hash] = element
        
        # Validate element distribution per page
        page_element_counts = {}
        for element in elements:
            page_element_counts[element.page_number] = page_element_counts.get(element.page_number, 0) + 1
        
        # Check for pages with unusually few elements
        avg_elements_per_page = len(elements) / len(page_numbers)
        for page, count in page_element_counts.items():
            if count < avg_elements_per_page * 0.3:  # Less than 30% of average
                result.add_warning(f"Page {page} has unusually few elements ({count})")
        
        return result

class BoundingBoxValidator:
    """Validates bounding box coordinates"""
    
    @staticmethod
    def validate_bbox(bbox: Dict[str, float]) -> ValidationResult:
        """Validate bounding box structure and values"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Check required keys
        required_keys = {'x0', 'y0', 'x1', 'y1'}
        missing_keys = required_keys - set(bbox.keys())
        if missing_keys:
            result.add_error(f"Missing bounding box keys: {missing_keys}")
            return result
        
        # Check value types
        for key, value in bbox.items():
            if not isinstance(value, (int, float)):
                result.add_error(f"Bounding box value {key} must be numeric, got {type(value)}")
        
        # Check coordinate ordering
        if bbox['x0'] > bbox['x1']:
            result.add_error(f"Invalid x coordinates: x0 ({bbox['x0']}) > x1 ({bbox['x1']})")
        
        if bbox['y0'] > bbox['y1']:
            result.add_error(f"Invalid y coordinates: y0 ({bbox['y0']}) > y1 ({bbox['y1']})")
        
        # Check for zero-area boxes
        width = bbox['x1'] - bbox['x0']
        height = bbox['y1'] - bbox['y0']
        if width <= 0 or height <= 0:
            result.add_warning(f"Zero or negative area bounding box: {width}x{height}")
        
        # Check for reasonable coordinate ranges (assuming normalized coordinates)
        for key, value in bbox.items():
            if value < 0:
                result.add_warning(f"Negative coordinate {key}: {value}")
            if value > 1000:  # Arbitrary large coordinate threshold
                result.add_warning(f"Very large coordinate {key}: {value}")
        
        return result
    
    @staticmethod
    def validate_bbox_within_page(
        bbox: Dict[str, float], 
        page_width: float, 
        page_height: float
    ) -> ValidationResult:
        """Validate bounding box is within page bounds"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        if bbox['x0'] < 0 or bbox['x1'] > page_width:
            result.add_error(f"Bounding box extends beyond page width: {bbox}")
        
        if bbox['y0'] < 0 or bbox['y1'] > page_height:
            result.add_error(f"Bounding box extends beyond page height: {bbox}")
        
        return result

class FileValidator:
    """Validates file operations and paths"""
    
    @staticmethod
    def validate_pdf_file(file_path: Union[str, Path]) -> ValidationResult:
        """Validate PDF file"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        file_path = Path(file_path)
        
        # Check file exists
        if not file_path.exists():
            result.add_error(f"File does not exist: {file_path}")
            return result
        
        # Check file extension
        if file_path.suffix.lower() != '.pdf':
            result.add_warning(f"File does not have .pdf extension: {file_path}")
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size == 0:
            result.add_error("File is empty")
        elif file_size > 100 * 1024 * 1024:  # 100MB threshold
            result.add_warning(f"Large file size: {file_size / (1024*1024):.1f}MB")
        
        # Check PDF header
        try:
            with open(file_path, 'rb') as f:
                header = f.read(8)
                if not header.startswith(b'%PDF-'):
                    result.add_error("File does not have valid PDF header")
        except Exception as e:
            result.add_error(f"Error reading file: {str(e)}")
        
        return result
    
    @staticmethod
    def validate_output_path(output_path: Union[str, Path]) -> ValidationResult:
        """Validate output file path"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        output_path = Path(output_path)
        
        # Check parent directory exists or can be created
        parent_dir = output_path.parent
        if not parent_dir.exists():
            try:
                parent_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                result.add_error(f"Cannot create output directory: {str(e)}")
        
        # Check write permissions
        if parent_dir.exists() and not os.access(parent_dir, os.W_OK):
            result.add_error(f"No write permission for directory: {parent_dir}")
        
        # Check for file overwrite
        if output_path.exists():
            result.add_warning(f"Output file already exists and will be overwritten: {output_path}")
        
        return result
```

### Text Processing Utilities

```python
# src/utils/text_processing.py
import re
import unicodedata
from typing import List, Dict, Optional, Tuple, Set
import string

class TextProcessor:
    """Text processing and cleaning utilities"""
    
    def __init__(self):
        self.stopwords = self._load_stopwords()
    
    def _load_stopwords(self) -> Set[str]:
        """Load common English stopwords"""
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
            'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Normalize Unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C')
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text"""
        # Simple sentence boundary detection
        sentence_endings = r'[.!?]+\s+'
        sentences = re.split(sentence_endings, text)
        
        # Clean up sentences
        cleaned_sentences = []
        for sentence in sentences:
            cleaned = self.clean_text(sentence)
            if cleaned and len(cleaned) > 10:  # Filter very short sentences
                cleaned_sentences.append(cleaned)
        
        return cleaned_sentences
    
    def tokenize(self, text: str, remove_stopwords: bool = False) -> List[str]:
        """Tokenize text into words"""
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Split into tokens
        tokens = text.split()
        
        # Remove stopwords if requested
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        
        return tokens
    
    def extract_keywords(self, text: str, max_keywords: int = 20) -> List[Tuple[str, int]]:
        """Extract keywords with frequency counts"""
        tokens = self.tokenize(text, remove_stopwords=True)
        
        # Count word frequency
        word_freq = {}
        for token in tokens:
            if len(token) > 2:  # Filter very short words
                word_freq[token] = word_freq.get(token, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords[:max_keywords]
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using Jaccard similarity"""
        if not text1 or not text2:
            return 0.0
        
        tokens1 = set(self.tokenize(text1, remove_stopwords=True))
        tokens2 = set(self.tokenize(text2, remove_stopwords=True))
        
        if not tokens1 and not tokens2:
            return 1.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def detect_language(self, text: str) -> Optional[str]:
        """Simple language detection (placeholder implementation)"""
        # This is a simplified implementation
        # In production, use libraries like langdetect or polyglot
        
        text_lower = text.lower()
        
        # Check for common English words
        english_indicators = ['the', 'and', 'that', 'have', 'for', 'not', 'with', 'you']
        english_count = sum(1 for word in english_indicators if word in text_lower)
        
        # Check for common French words
        french_indicators = ['les', 'des', 'une', 'dans', 'pour', 'que', 'avec', 'sur']
        french_count = sum(1 for word in french_indicators if word in text_lower)
        
        # Check for common German words
        german_indicators = ['der', 'die', 'das', 'und', 'ist', 'mit', 'für', 'auf']
        german_count = sum(1 for word in german_indicators if word in text_lower)
        
        # Simple majority vote
        counts = {'en': english_count, 'fr': french_count, 'de': german_count}
        detected = max(counts, key=counts.get)
        
        return detected if counts[detected] > 0 else None

class TableTextProcessor:
    """Specialized text processing for table content"""
    
    @staticmethod
    def clean_table_cell(cell_text: str) -> str:
        """Clean text from table cell"""
        if not cell_text:
            return ""
        
        # Remove line breaks within cells
        cleaned = re.sub(r'\n+', ' ', cell_text)
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Strip whitespace
        cleaned = cleaned.strip()
        
        return cleaned
    
    @staticmethod
    def extract_numeric_values(text: str) -> List[float]:
        """Extract numeric values from text"""
        # Pattern for numbers (including decimals, percentages, currency)
        number_pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
        matches = re.findall(number_pattern, text)
        
        values = []
        for match in matches:
            try:
                values.append(float(match))
            except ValueError:
                continue
        
        return values
    
    @staticmethod
    def detect_table_headers(rows: List[List[str]]) -> List[int]:
        """Detect which rows are likely table headers"""
        if not rows:
            return []
        
        header_indices = []
        
        for i, row in enumerate(rows):
            if not row:
                continue
            
            # Check for header indicators
            header_score = 0
            
            # Check for all caps
            caps_count = sum(1 for cell in row if cell.isupper())
            header_score += caps_count * 2
            
            # Check for bold indicators (simplified)
            bold_indicators = ['**', '__', '<b>', '</b>']
            bold_count = sum(1 for cell in row 
                           for indicator in bold_indicators 
                           if indicator in cell)
            header_score += bold_count
            
            # Check position (first few rows more likely to be headers)
            if i < 3:
                header_score += 3 - i
            
            # Threshold for header detection
            if header_score >= 3:
                header_indices.append(i)
        
        return header_indices
```

### Coordinate Utilities

```python
# src/utils/coordinate_utils.py
from typing import Dict, Tuple, List, Optional
import math

class CoordinateUtils:
    """Coordinate system utilities and transformations"""
    
    @staticmethod
    def normalize_coordinates(
        bbox: Dict[str, float], 
        page_width: float, 
        page_height: float
    ) -> Dict[str, float]:
        """Normalize coordinates to [0,1] range"""
        return {
            'x0': bbox['x0'] / page_width,
            'y0': bbox['y0'] / page_height,
            'x1': bbox['x1'] / page_width,
            'y1': bbox['y1'] / page_height
        }
    
    @staticmethod
    def denormalize_coordinates(
        bbox: Dict[str, float],
        page_width: float,
        page_height: float
    ) -> Dict[str, float]:
        """Convert normalized coordinates back to absolute coordinates"""
        return {
            'x0': bbox['x0'] * page_width,
            'y0': bbox['y0'] * page_height,
            'x1': bbox['x1'] * page_width,
            'y1': bbox['y1'] * page_height
        }
    
    @staticmethod
    def calculate_bbox_area(bbox: Dict[str, float]) -> float:
        """Calculate bounding box area"""
        width = bbox['x1'] - bbox['x0']
        height = bbox['y1'] - bbox['y0']
        return width * height
    
    @staticmethod
    def calculate_bbox_center(bbox: Dict[str, float]) -> Tuple[float, float]:
        """Calculate bounding box center point"""
        center_x = (bbox['x0'] + bbox['x1']) / 2
        center_y = (bbox['y0'] + bbox['y1']) / 2
        return center_x, center_y
    
    @staticmethod
    def calculate_intersection(
        bbox1: Dict[str, float], 
        bbox2: Dict[str, float]
    ) -> Optional[Dict[str, float]]:
        """Calculate intersection of two bounding boxes"""
        # Calculate intersection coordinates
        x0 = max(bbox1['x0'], bbox2['x0'])
        y0 = max(bbox1['y0'], bbox2['y0'])
        x1 = min(bbox1['x1'], bbox2['x1'])
        y1 = min(bbox1['y1'], bbox2['y1'])
        
        # Check if intersection exists
        if x0 >= x1 or y0 >= y1:
            return None
        
        return {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1}
    
    @staticmethod
    def calculate_iou(
        bbox1: Dict[str, float], 
        bbox2: Dict[str, float]
    ) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        intersection = CoordinateUtils.calculate_intersection(bbox1, bbox2)
        
        if intersection is None:
            return 0.0
        
        intersection_area = CoordinateUtils.calculate_bbox_area(intersection)
        area1 = CoordinateUtils.calculate_bbox_area(bbox1)
        area2 = CoordinateUtils.calculate_bbox_area(bbox2)
        
        union_area = area1 + area2 - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    @staticmethod
    def expand_bbox(
        bbox: Dict[str, float], 
        padding: float
    ) -> Dict[str, float]:
        """Expand bounding box by padding amount"""
        return {
            'x0': bbox['x0'] - padding,
            'y0': bbox['y0'] - padding,
            'x1': bbox['x1'] + padding,
            'y1': bbox['y1'] + padding
        }
    
    @staticmethod
    def merge_bboxes(bboxes: List[Dict[str, float]]) -> Dict[str, float]:
        """Merge multiple bounding boxes into one enclosing box"""
        if not bboxes:
            return {'x0': 0, 'y0': 0, 'x1': 0, 'y1': 0}
        
        min_x0 = min(bbox['x0'] for bbox in bboxes)
        min_y0 = min(bbox['y0'] for bbox in bboxes)
        max_x1 = max(bbox['x1'] for bbox in bboxes)
        max_y1 = max(bbox['y1'] for bbox in bboxes)
        
        return {
            'x0': min_x0,
            'y0': min_y0,
            'x1': max_x1,
            'y1': max_y1
        }
    
    @staticmethod
    def point_in_bbox(
        point: Tuple[float, float], 
        bbox: Dict[str, float]
    ) -> bool:
        """Check if point is inside bounding box"""
        x, y = point
        return (bbox['x0'] <= x <= bbox['x1'] and 
                bbox['y0'] <= y <= bbox['y1'])
    
    @staticmethod
    def calculate_distance(
        point1: Tuple[float, float], 
        point2: Tuple[float, float]
    ) -> float:
        """Calculate Euclidean distance between two points"""
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    @staticmethod
    def bbox_to_corners(bbox: Dict[str, float]) -> List[Tuple[float, float]]:
        """Convert bounding box to list of corner points"""
        return [
            (bbox['x0'], bbox['y0']),  # Bottom-left
            (bbox['x1'], bbox['y0']),  # Bottom-right
            (bbox['x1'], bbox['y1']),  # Top-right
            (bbox['x0'], bbox['y1'])   # Top-left
        ]

class SpatialAnalyzer:
    """Advanced spatial analysis utilities"""
    
    @staticmethod
    def group_by_proximity(
        elements: List[Dict], 
        distance_threshold: float,
        get_center_func: callable
    ) -> List[List[Dict]]:
        """Group elements by spatial proximity"""
        if not elements:
            return []
        
        groups = []
        used = set()
        
        for i, element in enumerate(elements):
            if i in used:
                continue
            
            # Start new group
            group = [element]
            used.add(i)
            center1 = get_center_func(element)
            
            # Find nearby elements
            for j, other_element in enumerate(elements):
                if j in used:
                    continue
                
                center2 = get_center_func(other_element)
                distance = CoordinateUtils.calculate_distance(center1, center2)
                
                if distance <= distance_threshold:
                    group.append(other_element)
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    @staticmethod
    def detect_reading_order(
        elements: List[Dict],
        get_bbox_func: callable,
        column_threshold: float = 0.1
    ) -> List[Dict]:
        """Detect and sort elements in reading order"""
        if not elements:
            return []
        
        # Group elements by vertical position (rows)
        elements_with_y = [(element, get_bbox_func(element)['y0']) for element in elements]
        elements_with_y.sort(key=lambda x: x[1], reverse=True)  # Sort by y coordinate (top to bottom)
        
        # Group into rows based on y-coordinate similarity
        rows = []
        current_row = []
        current_y = None
        
        for element, y in elements_with_y:
            if current_y is None or abs(y - current_y) <= column_threshold:
                current_row.append(element)
                current_y = y
            else:
                if current_row:
                    rows.append(current_row)
                current_row = [element]
                current_y = y
        
        if current_row:
            rows.append(current_row)
        
        # Sort elements within each row by x-coordinate (left to right)
        reading_order = []
        for row in rows:
            row.sort(key=lambda el: get_bbox_func(el)['x0'])
            reading_order.extend(row)
        
        return reading_order
```

### Performance Monitoring

```python
# src/utils/performance.py
import time
import functools
import psutil
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    execution_time: float = 0.0
    memory_usage: float = 0.0  # MB
    cpu_usage: float = 0.0     # Percentage
    timestamp: datetime = field(default_factory=datetime.now)
    function_name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

class PerformanceMonitor:
    """Performance monitoring utilities"""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self._monitoring = False
        self._monitor_thread = None
    
    def timing_decorator(self, func):
        """Decorator to measure function execution time"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                end_memory = self._get_memory_usage()
                
                metrics = PerformanceMetrics(
                    execution_time=end_time - start_time,
                    memory_usage=end_memory - start_memory,
                    function_name=func.__name__,
                    parameters={
                        'args_count': len(args),
                        'kwargs_keys': list(kwargs.keys())
                    }
                )
                
                self.metrics_history.append(metrics)
                
                # Log performance if execution time is significant
                if metrics.execution_time > 1.0:  # More than 1 second
                    print(f"Performance: {func.__name__} took {metrics.execution_time:.3f}s")
        
        return wrapper
    
    def start_monitoring(self, interval: float = 1.0):
        """Start continuous system monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_system,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
    
    def _monitor_system(self, interval: float):
        """Monitor system resources"""
        while self._monitoring:
            metrics = PerformanceMetrics(
                memory_usage=self._get_memory_usage(),
                cpu_usage=psutil.cpu_percent(),
                function_name="system_monitor"
            )
            self.metrics_history.append(metrics)
            time.sleep(interval)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # Convert to MB
        except:
            return 0.0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        if not self.metrics_history:
            return {}
        
        execution_times = [m.execution_time for m in self.metrics_history if m.execution_time > 0]
        memory_usage = [m.memory_usage for m in self.metrics_history]
        
        return {
            'total_operations': len(self.metrics_history),
            'avg_execution_time': sum(execution_times) / len(execution_times) if execution_times else 0,
            'max_execution_time': max(execution_times) if execution_times else 0,
            'avg_memory_usage': sum(memory_usage) / len(memory_usage) if memory_usage else 0,
            'max_memory_usage': max(memory_usage) if memory_usage else 0,
            'function_counts': self._count_function_calls()
        }
    
    def _count_function_calls(self) -> Dict[str, int]:
        """Count function call frequencies"""
        counts = {}
        for metric in self.metrics_history:
            if metric.function_name:
                counts[metric.function_name] = counts.get(metric.function_name, 0) + 1
        return counts
    
    def clear_metrics(self):
        """Clear metrics history"""
        self.metrics_history.clear()

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

# Convenience decorators
def timed(func):
    """Convenience timing decorator"""
    return performance_monitor.timing_decorator(func)

def monitor_memory(threshold_mb: float = 100.0):
    """Decorator to monitor memory usage"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_memory = performance_monitor._get_memory_usage()
            
            result = func(*args, **kwargs)
            
            end_memory = performance_monitor._get_memory_usage()
            memory_delta = end_memory - start_memory
            
            if memory_delta > threshold_mb:
                print(f"Memory warning: {func.__name__} used {memory_delta:.1f}MB")
            
            return result
        return wrapper
    return decorator
```

### Configuration Management

```python
# src/utils/config_loader.py
import os
import json
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ParserConfig:
    """Parser configuration"""
    enable_ocr: bool = True
    enable_tables: bool = True
    enable_images: bool = False
    page_images: bool = True
    table_mode: str = "accurate"
    ocr_lang: str = "auto"
    max_pages: int = 100
    confidence_threshold: float = 0.5
    image_scale: float = 1.0

@dataclass
class SearchConfig:
    """Search configuration"""
    fuzzy_threshold: float = 0.8
    boost_headers: bool = True
    max_results: int = 50
    enable_semantic_search: bool = False
    semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2"

@dataclass
class UIConfig:
    """UI configuration"""
    page_title: str = "Smart PDF Parser"
    max_file_size_mb: int = 50
    default_theme: str = "light"
    enable_analytics: bool = True
    cache_size: int = 100

class ConfigManager:
    """Configuration management system"""
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        self.config_dir = Path(config_dir) if config_dir else Path("config")
        self.config_dir.mkdir(exist_ok=True)
        
        self._configs = {}
        self._load_all_configs()
    
    def _load_all_configs(self):
        """Load all configuration files"""
        config_files = {
            'parser': 'parser_config.yaml',
            'search': 'search_config.yaml',
            'ui': 'ui_config.yaml',
            'app': 'app_config.yaml'
        }
        
        for config_name, filename in config_files.items():
            config_path = self.config_dir / filename
            if config_path.exists():
                self._configs[config_name] = self._load_config_file(config_path)
            else:
                # Create default config file
                self._create_default_config(config_name, config_path)
    
    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                    return yaml.safe_load(f) or {}
                elif config_path.suffix.lower() == '.json':
                    return json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path}")
        except Exception as e:
            print(f"Error loading config file {config_path}: {e}")
            return {}
    
    def _create_default_config(self, config_name: str, config_path: Path):
        """Create default configuration file"""
        default_configs = {
            'parser': {
                'enable_ocr': True,
                'enable_tables': True,
                'enable_images': False,
                'page_images': True,
                'table_mode': 'accurate',
                'ocr_lang': 'auto',
                'max_pages': 100,
                'confidence_threshold': 0.5,
                'image_scale': 1.0
            },
            'search': {
                'fuzzy_threshold': 0.8,
                'boost_headers': True,
                'max_results': 50,
                'enable_semantic_search': False,
                'semantic_model': 'sentence-transformers/all-MiniLM-L6-v2'
            },
            'ui': {
                'page_title': 'Smart PDF Parser',
                'max_file_size_mb': 50,
                'default_theme': 'light',
                'enable_analytics': True,
                'cache_size': 100
            },
            'app': {
                'debug': False,
                'log_level': 'INFO',
                'temp_dir': 'temp',
                'output_dir': 'output'
            }
        }
        
        config_data = default_configs.get(config_name, {})
        
        try:
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
            self._configs[config_name] = config_data
        except Exception as e:
            print(f"Error creating default config file {config_path}: {e}")
            self._configs[config_name] = {}
    
    def get_parser_config(self) -> ParserConfig:
        """Get parser configuration"""
        config_data = self._configs.get('parser', {})
        return ParserConfig(**config_data)
    
    def get_search_config(self) -> SearchConfig:
        """Get search configuration"""
        config_data = self._configs.get('search', {})
        return SearchConfig(**config_data)
    
    def get_ui_config(self) -> UIConfig:
        """Get UI configuration"""
        config_data = self._configs.get('ui', {})
        return UIConfig(**config_data)
    
    def get_config_value(self, config_name: str, key: str, default: Any = None) -> Any:
        """Get specific configuration value"""
        return self._configs.get(config_name, {}).get(key, default)
    
    def update_config_value(self, config_name: str, key: str, value: Any):
        """Update specific configuration value"""
        if config_name not in self._configs:
            self._configs[config_name] = {}
        
        self._configs[config_name][key] = value
        self._save_config(config_name)
    
    def _save_config(self, config_name: str):
        """Save configuration to file"""
        config_filename = f"{config_name}_config.yaml"
        config_path = self.config_dir / config_filename
        
        try:
            with open(config_path, 'w') as f:
                yaml.dump(self._configs[config_name], f, default_flow_style=False)
        except Exception as e:
            print(f"Error saving config file {config_path}: {e}")

# Global config manager instance
config_manager = ConfigManager()
```

### Custom Exceptions

```python
# src/utils/exceptions.py
"""Custom exceptions for the Smart PDF Parser"""

class PDFParserError(Exception):
    """Base exception for PDF parser errors"""
    pass

class DocumentParsingError(PDFParserError):
    """Raised when document parsing fails"""
    def __init__(self, message: str, document_path: str = None):
        self.document_path = document_path
        super().__init__(message)

class SearchEngineError(PDFParserError):
    """Raised when search engine operations fail"""
    pass

class ValidationError(PDFParserError):
    """Raised when data validation fails"""
    def __init__(self, message: str, validation_errors: list = None):
        self.validation_errors = validation_errors or []
        super().__init__(message)

class ConfigurationError(PDFParserError):
    """Raised when configuration is invalid"""
    pass

class CoordinateError(PDFParserError):
    """Raised when coordinate operations fail"""
    pass

class ExportError(PDFParserError):
    """Raised when export operations fail"""
    pass

class VerificationError(PDFParserError):
    """Raised when verification operations fail"""
    pass
```

## Integration Examples

### Complete Utility Usage

```python
# Example: Complete document processing with utilities
from src.utils.validation import DocumentValidator, FileValidator
from src.utils.text_processing import TextProcessor
from src.utils.performance import performance_monitor, timed
from src.utils.config_loader import config_manager
from src.core.parser import DoclingParser

@timed
def process_document_with_validation(pdf_path: str):
    """Process document with full validation and monitoring"""
    
    # File validation
    file_result = FileValidator.validate_pdf_file(pdf_path)
    if not file_result.is_valid:
        raise ValidationError("Invalid PDF file", file_result.errors)
    
    # Get configuration
    parser_config = config_manager.get_parser_config()
    
    # Parse document
    parser = DoclingParser(
        enable_ocr=parser_config.enable_ocr,
        enable_tables=parser_config.enable_tables
    )
    
    document = parser.parse_document(pdf_path)
    
    # Validate elements
    text_processor = TextProcessor()
    for element in document.elements:
        # Validate element
        validation_result = DocumentValidator.validate_element(element)
        if not validation_result.is_valid:
            print(f"Element validation failed: {validation_result.errors}")
        
        # Process text
        element.text = text_processor.clean_text(element.text)
    
    # Validate overall consistency
    consistency_result = DocumentValidator.validate_elements_consistency(document.elements)
    if consistency_result.warnings:
        print(f"Consistency warnings: {consistency_result.warnings}")
    
    return document

# Usage
document = process_document_with_validation("sample.pdf")
print(performance_monitor.get_performance_summary())
```

This comprehensive utilities implementation provides robust support for validation, text processing, coordinate handling, performance monitoring, and configuration management across the Smart PDF Parser application.
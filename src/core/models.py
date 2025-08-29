"""
Data models for the Smart PDF Parser.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
import json


@dataclass
class DocumentElement:
    """Represents a single element extracted from a PDF document."""
    
    text: str
    element_type: str  # 'text', 'heading', 'table', 'image', 'list', 'formula'
    page_number: int
    bbox: Dict[str, float]  # {'x0', 'y0', 'x1', 'y1'}
    confidence: float
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validate the element after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        
        if self.page_number < 1:
            raise ValueError(f"Page number must be >= 1, got {self.page_number}")
        
        required_bbox_keys = {'x0', 'y0', 'x1', 'y1'}
        if not required_bbox_keys.issubset(self.bbox.keys()):
            missing = required_bbox_keys - set(self.bbox.keys())
            raise ValueError(f"Missing required bbox keys: {missing}")


@dataclass
class KeyValuePair:
    """Represents a key-value pair extracted from a PDF document.
    
    This class represents structured data extraction results where a label
    (key) has been associated with a corresponding value, typically found
    in forms, invoices, contracts, and other structured documents.
    
    Attributes:
        label_text: The label/key text (e.g., "Name:", "Invoice Number", "Date")
        value_text: The associated value text (e.g., "John A. Smith", "INV-2024-001")
        page_number: Page number where the pair was found (1-based)
        label_bbox: Label bounding box with keys 'x0', 'y0', 'x1', 'y1'
        value_bbox: Value bounding box with keys 'x0', 'y0', 'x1', 'y1'
        confidence: Overall confidence score for the key-value association [0.0, 1.0]
        metadata: Additional information about the extraction process
        
    Example metadata structure:
        {
            'label_score': 0.85,      # Confidence that the label was correctly identified
            'geom_score': 0.73,       # Spatial/geometric association confidence  
            'content_score': 0.92,    # Content/semantic association confidence
            'strategy': 'same_line',  # Extraction strategy used ('same_line', 'below', 'right', etc.)
            'distance': 15.2,         # Physical distance between label and value
            'alignment': 'horizontal' # Spatial alignment ('horizontal', 'vertical', 'diagonal')
        }
        
    Example usage:
        kv_pair = KeyValuePair(
            label_text="Customer Name:",
            value_text="John A. Smith",
            page_number=1,
            label_bbox={'x0': 50.0, 'y0': 100.0, 'x1': 150.0, 'y1': 120.0},
            value_bbox={'x0': 160.0, 'y0': 100.0, 'x1': 260.0, 'y1': 120.0},
            confidence=0.89,
            metadata={'strategy': 'same_line', 'label_score': 0.92, 'geom_score': 0.85}
        )
    """
    
    label_text: str
    value_text: str 
    page_number: int
    label_bbox: Dict[str, float]
    value_bbox: Dict[str, float]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate the key-value pair after initialization."""
        # Validate confidence score
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        
        # Validate page number
        if self.page_number < 1:
            raise ValueError(f"Page number must be >= 1, got {self.page_number}")
        
        # Validate text fields are non-empty strings
        if not isinstance(self.label_text, str) or not self.label_text.strip():
            raise ValueError("label_text must be a non-empty string")
        
        if not isinstance(self.value_text, str) or not self.value_text.strip():
            raise ValueError("value_text must be a non-empty string")
        
        # Validate bounding boxes
        self._validate_bbox(self.label_bbox, "label_bbox")
        self._validate_bbox(self.value_bbox, "value_bbox")
        
        # Validate metadata is a dictionary
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")
    
    def _validate_bbox(self, bbox: Dict[str, float], bbox_name: str):
        """Validate a bounding box dictionary.
        
        Args:
            bbox: The bounding box to validate
            bbox_name: Name of the bbox for error messages
        """
        if not isinstance(bbox, dict):
            raise ValueError(f"{bbox_name} must be a dictionary, got {type(bbox)}")
        
        required_keys = {'x0', 'y0', 'x1', 'y1'}
        if not required_keys.issubset(bbox.keys()):
            missing = required_keys - set(bbox.keys())
            raise ValueError(f"Missing required {bbox_name} keys: {missing}")
        
        # Validate coordinate types and values
        for key in required_keys:
            value = bbox[key]
            if not isinstance(value, (int, float)):
                raise ValueError(f"{bbox_name}['{key}'] must be a number, got {type(value)}")
            if value < 0:
                raise ValueError(f"{bbox_name}['{key}'] must be >= 0, got {value}")
        
        # Validate coordinate relationships
        if bbox['x0'] >= bbox['x1']:
            raise ValueError(f"{bbox_name}: x0 ({bbox['x0']}) must be less than x1 ({bbox['x1']})")
        
        if bbox['y0'] >= bbox['y1']:
            raise ValueError(f"{bbox_name}: y0 ({bbox['y0']}) must be less than y1 ({bbox['y1']})")


@dataclass
class SearchResult:
    """Represents a search result with relevance scoring."""
    
    element: DocumentElement
    score: float
    match_type: str  # 'exact', 'fuzzy', 'semantic'
    matched_text: str
    match_context: Optional[str] = None  # Context around the match
    
    def __post_init__(self):
        """Validate the search result after initialization."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be between 0 and 1, got {self.score}")
        
        if self.match_type not in ['exact', 'fuzzy', 'semantic']:
            raise ValueError(f"Invalid match type: {self.match_type}")


@dataclass
class VerificationState:
    """Represents the verification state of a document element."""
    
    element_id: int
    status: str  # 'pending', 'correct', 'incorrect', 'partial'
    timestamp: datetime
    verified_by: Optional[str] = None
    correction: Optional[str] = None
    corrected_text: Optional[str] = None  # The corrected text version
    verified_at: Optional[datetime] = None  # When verification was completed
    notes: Optional[str] = None
    
    def __post_init__(self):
        """Validate the verification state after initialization."""
        valid_statuses = {'pending', 'correct', 'incorrect', 'partial'}
        if self.status not in valid_statuses:
            raise ValueError(f"Invalid status: {self.status}. Must be one of {valid_statuses}")


@dataclass
class ParsedDocument:
    """Represents a parsed PDF document with all extracted elements."""
    
    elements: List[DocumentElement]
    metadata: Dict[str, Any]
    pages: Optional[Dict[int, Any]] = None  # Page-specific data
    
    def __post_init__(self):
        """Validate the parsed document after initialization."""
        if not isinstance(self.elements, list):
            raise ValueError("Elements must be a list")
        
        if not isinstance(self.metadata, dict):
            raise ValueError("Metadata must be a dictionary")
        
        # Ensure metadata has required fields
        if 'source_path' not in self.metadata:
            self.metadata['source_path'] = None
        if 'parsed_at' not in self.metadata:
            self.metadata['parsed_at'] = datetime.now().isoformat()
        if 'total_elements' not in self.metadata:
            self.metadata['total_elements'] = len(self.elements)
        if 'page_count' not in self.metadata:
            page_numbers = set(e.page_number for e in self.elements)
            self.metadata['page_count'] = max(page_numbers) if page_numbers else 0
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export the document to a dictionary for JSON serialization.
        
        Returns:
            Dictionary representation of the document
        """
        return {
            'metadata': self.metadata,
            'elements': [
                {
                    'text': element.text,
                    'element_type': element.element_type,
                    'page_number': element.page_number,
                    'bbox': element.bbox,
                    'confidence': element.confidence,
                    'metadata': element.metadata
                }
                for element in self.elements
            ],
            'pages': self.pages,
            'export_info': {
                'format': 'dict',
                'exported_at': datetime.now().isoformat(),
                'total_elements': len(self.elements)
            }
        }
    
    def export_to_json(self, indent: int = 2) -> str:
        """Export the document to JSON format.
        
        Args:
            indent: JSON indentation level
            
        Returns:
            JSON string representation
        """
        try:
            return json.dumps(self.export_to_dict(), indent=indent, ensure_ascii=False)
        except Exception as e:
            raise ValueError(f"Failed to export to JSON: {e}")
    
    def export_to_markdown(self) -> str:
        """Export the document to Markdown format.
        
        Returns:
            Markdown string representation
        """
        lines = []
        
        # Document header
        filename = self.metadata.get('filename', 'Document')
        lines.append(f"# {filename}")
        lines.append("")
        
        # Metadata section
        lines.append("## Document Information")
        lines.append("")
        lines.append(f"- **Source**: {self.metadata.get('source_path', 'Unknown')}")
        lines.append(f"- **Pages**: {self.metadata.get('page_count', 0)}")
        lines.append(f"- **Elements**: {len(self.elements)}")
        lines.append(f"- **Parsed**: {self.metadata.get('parsed_at', 'Unknown')}")
        lines.append("")
        
        # Group elements by page
        elements_by_page = {}
        for element in self.elements:
            page = element.page_number
            if page not in elements_by_page:
                elements_by_page[page] = []
            elements_by_page[page].append(element)
        
        # Export each page
        for page_num in sorted(elements_by_page.keys()):
            lines.append(f"## Page {page_num}")
            lines.append("")
            
            page_elements = elements_by_page[page_num]
            
            for element in page_elements:
                # Add element type as heading level based on type
                if element.element_type == 'heading':
                    lines.append(f"### {element.text}")
                elif element.element_type == 'table':
                    lines.append("### Table")
                    lines.append("")
                    lines.append("```")
                    lines.append(element.text)
                    lines.append("```")
                elif element.element_type == 'formula':
                    lines.append(f"**Formula**: {element.text}")
                elif element.element_type == 'list':
                    lines.append(element.text)
                elif element.element_type == 'code':
                    lines.append("```")
                    lines.append(element.text)
                    lines.append("```")
                else:
                    lines.append(element.text)
                
                lines.append("")
        
        return "\\n".join(lines)
    
    def export_to_html(self) -> str:
        """Export the document to HTML format.
        
        Returns:
            HTML string representation
        """
        html_lines = []
        
        # HTML header
        filename = self.metadata.get('filename', 'Document')
        html_lines.extend([
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='UTF-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            f"<title>{filename}</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }",
            ".header { background-color: #f4f4f4; padding: 20px; border-radius: 5px; margin-bottom: 20px; }",
            ".page { border-left: 3px solid #007cba; padding-left: 20px; margin: 20px 0; }",
            ".element { margin: 10px 0; }",
            ".heading { font-size: 1.2em; font-weight: bold; color: #333; }",
            ".table { background-color: #f9f9f9; padding: 10px; border-radius: 3px; }",
            ".formula { font-family: 'Courier New', monospace; background-color: #ffffcc; padding: 5px; }",
            ".code { background-color: #f4f4f4; padding: 10px; border-radius: 3px; font-family: 'Courier New', monospace; }",
            ".metadata { font-size: 0.9em; color: #666; margin-top: 5px; }",
            "</style>",
            "</head>",
            "<body>"
        ])
        
        # Document header
        html_lines.extend([
            "<div class='header'>",
            f"<h1>{filename}</h1>",
            "<div class='metadata'>",
            f"<p><strong>Source:</strong> {self.metadata.get('source_path', 'Unknown')}</p>",
            f"<p><strong>Pages:</strong> {self.metadata.get('page_count', 0)}</p>",
            f"<p><strong>Elements:</strong> {len(self.elements)}</p>",
            f"<p><strong>Parsed:</strong> {self.metadata.get('parsed_at', 'Unknown')}</p>",
            "</div>",
            "</div>"
        ])
        
        # Group elements by page
        elements_by_page = {}
        for element in self.elements:
            page = element.page_number
            if page not in elements_by_page:
                elements_by_page[page] = []
            elements_by_page[page].append(element)
        
        # Export each page
        for page_num in sorted(elements_by_page.keys()):
            html_lines.extend([
                f"<div class='page'>",
                f"<h2>Page {page_num}</h2>"
            ])
            
            page_elements = elements_by_page[page_num]
            
            for element in page_elements:
                # HTML escape the text
                escaped_text = (element.text
                               .replace("&", "&amp;")
                               .replace("<", "&lt;")
                               .replace(">", "&gt;")
                               .replace('"', "&quot;")
                               .replace("'", "&#39;"))
                
                element_class = element.element_type
                
                if element.element_type == 'heading':
                    html_lines.append(f"<div class='element {element_class}'><h3>{escaped_text}</h3></div>")
                elif element.element_type == 'table':
                    html_lines.extend([
                        f"<div class='element {element_class}'>",
                        "<h4>Table</h4>",
                        f"<pre>{escaped_text}</pre>",
                        "</div>"
                    ])
                elif element.element_type in ['formula', 'code']:
                    html_lines.append(f"<div class='element {element_class}'><pre>{escaped_text}</pre></div>")
                else:
                    html_lines.append(f"<div class='element {element_class}'><p>{escaped_text}</p></div>")
                
                # Add metadata if available
                if element.metadata:
                    confidence = element.confidence
                    html_lines.append(
                        f"<div class='metadata'>Confidence: {confidence:.3f} | "
                        f"Type: {element.element_type} | "
                        f"Position: ({element.bbox['x0']:.1f}, {element.bbox['y0']:.1f})</div>"
                    )
            
            html_lines.append("</div>")
        
        # HTML footer
        html_lines.extend([
            "</body>",
            "</html>"
        ])
        
        return "\\n".join(html_lines)
    
    def get_elements_by_type(self, element_type: str) -> List[DocumentElement]:
        """Get all elements of a specific type.
        
        Args:
            element_type: Type of elements to retrieve
            
        Returns:
            List of elements matching the type
        """
        return [e for e in self.elements if e.element_type == element_type]
    
    def get_elements_by_page(self, page_number: int) -> List[DocumentElement]:
        """Get all elements from a specific page.
        
        Args:
            page_number: Page number to retrieve elements from
            
        Returns:
            List of elements from the specified page
        """
        return [e for e in self.elements if e.page_number == page_number]
    
    def get_page_count(self) -> int:
        """Get the total number of pages in the document.
        
        Returns:
            Number of pages
        """
        if not self.elements:
            return 0
        return max(e.page_number for e in self.elements)
    
    def get_element_type_counts(self) -> Dict[str, int]:
        """Get count of elements by type.
        
        Returns:
            Dictionary mapping element types to counts
        """
        counts = {}
        for element in self.elements:
            counts[element.element_type] = counts.get(element.element_type, 0) + 1
        return counts
    
    def get_average_confidence(self) -> float:
        """Get average confidence score across all elements.
        
        Returns:
            Average confidence score (0.0 to 1.0)
        """
        if not self.elements:
            return 0.0
        return sum(e.confidence for e in self.elements) / len(self.elements)
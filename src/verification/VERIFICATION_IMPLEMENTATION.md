# Verification System Implementation Guide

## Overview

The Verification System provides interactive visual verification of PDF parsing results through coordinate-aware rendering and user interface components. It enables users to visually inspect, validate, and correct parsing results with pixel-perfect accuracy.

## Architecture

### Core Components

```
src/verification/
├── renderer.py          # Visual rendering engine
├── interface.py         # Streamlit verification interface
├── coordinate_transform.py  # Coordinate system transformations
├── overlay_manager.py   # Visual overlay management
└── export_handler.py    # Verification state export
```

### Data Flow

```
PDF Document → Parsed Elements → Coordinate Mapping → Visual Rendering → User Verification → Export
```

## Core Classes

### VerificationRenderer

The main rendering engine that creates visual overlays for parsed elements.

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.core.models import DocumentElement

@dataclass
class RenderConfig:
    """Configuration for verification rendering"""
    highlight_color: Tuple[int, int, int, int] = (255, 255, 0, 128)  # Yellow with alpha
    border_color: Tuple[int, int, int] = (255, 0, 0)  # Red border
    border_width: int = 2
    font_size: int = 12
    show_confidence: bool = True
    show_element_type: bool = True
    page_scale: float = 1.0

class VerificationRenderer:
    """Renders PDF elements with visual overlays for verification"""
    
    def __init__(self, config: RenderConfig = None):
        self.config = config or RenderConfig()
        self._font_cache = {}
    
    def render_page_with_overlays(
        self, 
        page_image: Image.Image,
        elements: List[DocumentElement],
        page_number: int
    ) -> Image.Image:
        """
        Render a page with element overlays for verification
        
        Args:
            page_image: PIL Image of the PDF page
            elements: List of parsed elements for this page
            page_number: Page number (1-indexed)
            
        Returns:
            PIL Image with overlays rendered
        """
        # Filter elements for this page
        page_elements = [e for e in elements if e.page_number == page_number]
        
        # Create overlay canvas
        overlay = Image.new('RGBA', page_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Render each element
        for element in page_elements:
            self._render_element_overlay(draw, element, page_image.size)
        
        # Composite overlay onto page image
        return Image.alpha_composite(
            page_image.convert('RGBA'), 
            overlay
        ).convert('RGB')
    
    def _render_element_overlay(
        self, 
        draw: ImageDraw.Draw, 
        element: DocumentElement,
        page_size: Tuple[int, int]
    ):
        """Render overlay for a single element"""
        # Convert normalized coordinates to pixel coordinates
        bbox = self._normalize_to_pixel_coords(element.bbox, page_size)
        
        # Draw bounding box
        draw.rectangle(
            bbox,
            fill=self.config.highlight_color,
            outline=self.config.border_color,
            width=self.config.border_width
        )
        
        # Add element type and confidence labels
        if self.config.show_element_type or self.config.show_confidence:
            self._draw_element_label(draw, element, bbox)
    
    def _normalize_to_pixel_coords(
        self, 
        bbox: Dict[str, float], 
        page_size: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """Convert normalized bbox to pixel coordinates"""
        width, height = page_size
        return (
            int(bbox['x0'] * width),
            int(bbox['y0'] * height),
            int(bbox['x1'] * width),
            int(bbox['y1'] * height)
        )
    
    def _draw_element_label(
        self, 
        draw: ImageDraw.Draw, 
        element: DocumentElement,
        bbox: Tuple[int, int, int, int]
    ):
        """Draw element type and confidence label"""
        labels = []
        if self.config.show_element_type:
            labels.append(element.element_type.upper())
        if self.config.show_confidence:
            labels.append(f"{element.confidence:.2f}")
        
        if labels:
            label_text = " | ".join(labels)
            font = self._get_font(self.config.font_size)
            
            # Position label at top-left of bounding box
            label_pos = (bbox[0] + 2, bbox[1] - 20)
            draw.text(label_pos, label_text, fill=(0, 0, 0), font=font)
    
    def _get_font(self, size: int) -> ImageFont.ImageFont:
        """Get cached font instance"""
        if size not in self._font_cache:
            try:
                self._font_cache[size] = ImageFont.truetype("arial.ttf", size)
            except OSError:
                self._font_cache[size] = ImageFont.load_default()
        return self._font_cache[size]
```

### CoordinateTransformer

Handles coordinate system transformations between different representations.

```python
class CoordinateTransformer:
    """Transforms coordinates between different coordinate systems"""
    
    @staticmethod
    def docling_to_pixel(
        bbox: Dict[str, float],
        page_width: float,
        page_height: float,
        image_width: int,
        image_height: int
    ) -> Tuple[int, int, int, int]:
        """
        Transform Docling coordinates to pixel coordinates
        
        Docling uses: (0,0) at bottom-left, normalized to page size
        Pixel uses: (0,0) at top-left, absolute pixel values
        """
        # Scale to image dimensions
        x0 = int((bbox['x0'] / page_width) * image_width)
        x1 = int((bbox['x1'] / page_width) * image_width)
        
        # Flip Y-axis and scale
        y0 = int(((page_height - bbox['y1']) / page_height) * image_height)
        y1 = int(((page_height - bbox['y0']) / page_height) * image_height)
        
        return (x0, y0, x1, y1)
    
    @staticmethod
    def pixel_to_docling(
        pixel_bbox: Tuple[int, int, int, int],
        page_width: float,
        page_height: float,
        image_width: int,
        image_height: int
    ) -> Dict[str, float]:
        """Transform pixel coordinates back to Docling format"""
        x0, y0, x1, y1 = pixel_bbox
        
        # Scale to page dimensions
        doc_x0 = (x0 / image_width) * page_width
        doc_x1 = (x1 / image_width) * page_width
        
        # Flip Y-axis and scale
        doc_y0 = page_height - ((y1 / image_height) * page_height)
        doc_y1 = page_height - ((y0 / image_height) * page_height)
        
        return {
            'x0': doc_x0, 'y0': doc_y0,
            'x1': doc_x1, 'y1': doc_y1
        }
    
    @staticmethod
    def validate_bbox(bbox: Dict[str, float]) -> bool:
        """Validate bounding box coordinates"""
        required_keys = {'x0', 'y0', 'x1', 'y1'}
        if not all(key in bbox for key in required_keys):
            return False
        
        return (
            bbox['x0'] <= bbox['x1'] and
            bbox['y0'] <= bbox['y1'] and
            all(isinstance(v, (int, float)) for v in bbox.values())
        )
```

### VerificationInterface

Streamlit-based interface for interactive verification.

```python
import streamlit as st
from typing import List, Dict, Any, Optional
from src.core.models import DocumentElement, ParsedDocument
from src.verification.renderer import VerificationRenderer, RenderConfig

class VerificationInterface:
    """Streamlit interface for PDF parsing verification"""
    
    def __init__(self, renderer: VerificationRenderer = None):
        self.renderer = renderer or VerificationRenderer()
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize Streamlit session state"""
        if 'verification_state' not in st.session_state:
            st.session_state.verification_state = {
                'current_page': 1,
                'verified_elements': set(),
                'flagged_elements': set(),
                'corrections': {},
                'filter_type': 'all'
            }
    
    def render_verification_ui(self, document: ParsedDocument):
        """Render the main verification interface"""
        st.title("📄 PDF Parsing Verification")
        
        # Sidebar controls
        self._render_sidebar_controls(document)
        
        # Main verification area
        self._render_main_verification_area(document)
        
        # Element inspector
        self._render_element_inspector(document)
        
        # Export controls
        self._render_export_controls(document)
    
    def _render_sidebar_controls(self, document: ParsedDocument):
        """Render sidebar with page navigation and filters"""
        with st.sidebar:
            st.header("Navigation")
            
            # Page selector
            total_pages = len(document.pages)
            current_page = st.selectbox(
                "Page",
                range(1, total_pages + 1),
                index=st.session_state.verification_state['current_page'] - 1,
                key="page_selector"
            )
            st.session_state.verification_state['current_page'] = current_page
            
            # Element type filter
            st.header("Filters")
            element_types = ['all'] + list(set(
                e.element_type for e in document.elements
            ))
            filter_type = st.selectbox(
                "Element Type",
                element_types,
                key="element_filter"
            )
            st.session_state.verification_state['filter_type'] = filter_type
            
            # Rendering options
            st.header("Display Options")
            show_confidence = st.checkbox("Show Confidence", value=True)
            show_element_type = st.checkbox("Show Element Type", value=True)
            highlight_opacity = st.slider("Highlight Opacity", 0.1, 1.0, 0.5)
            
            # Update renderer config
            self.renderer.config.show_confidence = show_confidence
            self.renderer.config.show_element_type = show_element_type
            self.renderer.config.highlight_color = (*self.renderer.config.highlight_color[:3], int(highlight_opacity * 255))
    
    def _render_main_verification_area(self, document: ParsedDocument):
        """Render main verification display area"""
        current_page = st.session_state.verification_state['current_page']
        filter_type = st.session_state.verification_state['filter_type']
        
        # Get page elements
        page_elements = [
            e for e in document.elements 
            if e.page_number == current_page
        ]
        
        # Apply filters
        if filter_type != 'all':
            page_elements = [
                e for e in page_elements 
                if e.element_type == filter_type
            ]
        
        # Get page image (would come from document.page_images in real implementation)
        page_image = self._get_page_image(document, current_page)
        
        if page_image:
            # Render with overlays
            rendered_image = self.renderer.render_page_with_overlays(
                page_image, page_elements, current_page
            )
            
            # Display image with click handling
            st.image(rendered_image, use_column_width=True)
            
            # Element selection interface
            self._render_element_selection(page_elements)
    
    def _render_element_selection(self, elements: List[DocumentElement]):
        """Render element selection and verification controls"""
        if not elements:
            st.info("No elements found on this page with current filters.")
            return
        
        st.subheader(f"Elements on Page ({len(elements)} found)")
        
        # Create element selection table
        for i, element in enumerate(elements):
            with st.expander(f"{element.element_type.title()} - {element.text[:50]}..."):
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                
                with col1:
                    st.text_area(
                        "Text Content",
                        value=element.text,
                        height=100,
                        key=f"text_{element.page_number}_{i}"
                    )
                
                with col2:
                    st.metric("Confidence", f"{element.confidence:.3f}")
                    st.text(f"Type: {element.element_type}")
                
                with col3:
                    if st.button("✅ Verify", key=f"verify_{element.page_number}_{i}"):
                        self._verify_element(element)
                    if st.button("🚩 Flag", key=f"flag_{element.page_number}_{i}"):
                        self._flag_element(element)
                
                with col4:
                    element_id = f"{element.page_number}_{i}"
                    if element_id in st.session_state.verification_state['verified_elements']:
                        st.success("Verified")
                    elif element_id in st.session_state.verification_state['flagged_elements']:
                        st.error("Flagged")
                    else:
                        st.info("Pending")
    
    def _render_element_inspector(self, document: ParsedDocument):
        """Render detailed element inspector"""
        st.subheader("Element Inspector")
        
        # Element selection for detailed view
        current_page = st.session_state.verification_state['current_page']
        page_elements = [
            e for e in document.elements 
            if e.page_number == current_page
        ]
        
        if page_elements:
            selected_idx = st.selectbox(
                "Select element for detailed inspection",
                range(len(page_elements)),
                format_func=lambda x: f"{page_elements[x].element_type}: {page_elements[x].text[:30]}..."
            )
            
            element = page_elements[selected_idx]
            
            # Display element details
            col1, col2 = st.columns(2)
            
            with col1:
                st.json({
                    "text": element.text,
                    "element_type": element.element_type,
                    "page_number": element.page_number,
                    "confidence": element.confidence
                })
            
            with col2:
                st.json({
                    "bounding_box": element.bbox,
                    "metadata": element.metadata
                })
    
    def _render_export_controls(self, document: ParsedDocument):
        """Render export and summary controls"""
        st.subheader("Export & Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Generate Report"):
                report = self._generate_verification_report(document)
                st.download_button(
                    "Download Verification Report",
                    data=report,
                    file_name="verification_report.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📝 Export Corrections"):
                corrections = self._export_corrections()
                st.download_button(
                    "Download Corrections",
                    data=corrections,
                    file_name="parsing_corrections.json",
                    mime="application/json"
                )
        
        with col3:
            if st.button("✅ Mark Complete"):
                self._mark_verification_complete(document)
    
    def _verify_element(self, element: DocumentElement):
        """Mark element as verified"""
        element_id = f"{element.page_number}_{hash(element.text)}"
        st.session_state.verification_state['verified_elements'].add(element_id)
        st.session_state.verification_state['flagged_elements'].discard(element_id)
        st.success(f"Element verified: {element.element_type}")
    
    def _flag_element(self, element: DocumentElement):
        """Flag element for review"""
        element_id = f"{element.page_number}_{hash(element.text)}"
        st.session_state.verification_state['flagged_elements'].add(element_id)
        st.session_state.verification_state['verified_elements'].discard(element_id)
        st.warning(f"Element flagged: {element.element_type}")
    
    def _generate_verification_report(self, document: ParsedDocument) -> str:
        """Generate verification summary report"""
        state = st.session_state.verification_state
        
        total_elements = len(document.elements)
        verified_count = len(state['verified_elements'])
        flagged_count = len(state['flagged_elements'])
        pending_count = total_elements - verified_count - flagged_count
        
        report = {
            "verification_summary": {
                "total_elements": total_elements,
                "verified": verified_count,
                "flagged": flagged_count,
                "pending": pending_count,
                "completion_rate": verified_count / total_elements if total_elements > 0 else 0
            },
            "flagged_elements": list(state['flagged_elements']),
            "corrections": state['corrections'],
            "timestamp": st.session_state.get('verification_timestamp')
        }
        
        return json.dumps(report, indent=2)
    
    def _get_page_image(self, document: ParsedDocument, page_number: int):
        """Get page image for rendering (placeholder implementation)"""
        # In real implementation, this would extract page image from document
        # For now, return None to indicate image loading is needed
        return None
```

## Advanced Features

### Interactive Bounding Box Adjustment

```python
class InteractiveBBoxEditor:
    """Interactive bounding box editor for verification interface"""
    
    def __init__(self, image_size: Tuple[int, int]):
        self.image_size = image_size
        self._selection_state = None
    
    def render_bbox_editor(self, element: DocumentElement) -> Dict[str, float]:
        """
        Render interactive bounding box editor
        Returns updated bbox coordinates
        """
        st.subheader("Bounding Box Editor")
        
        # Coordinate input fields
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            x0 = st.number_input("X0", value=element.bbox['x0'], step=0.01)
        with col2:
            y0 = st.number_input("Y0", value=element.bbox['y0'], step=0.01)
        with col3:
            x1 = st.number_input("X1", value=element.bbox['x1'], step=0.01)
        with col4:
            y1 = st.number_input("Y1", value=element.bbox['y1'], step=0.01)
        
        updated_bbox = {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1}
        
        # Validation
        if CoordinateTransformer.validate_bbox(updated_bbox):
            st.success("Valid bounding box")
        else:
            st.error("Invalid bounding box coordinates")
        
        return updated_bbox
```

### Verification State Persistence

```python
class VerificationStateManager:
    """Manages verification state persistence"""
    
    @staticmethod
    def save_verification_state(
        document_id: str, 
        state: Dict[str, Any],
        filepath: Optional[str] = None
    ):
        """Save verification state to disk"""
        if not filepath:
            filepath = f"verification_state_{document_id}.json"
        
        state_data = {
            "document_id": document_id,
            "timestamp": datetime.now().isoformat(),
            "verification_state": state
        }
        
        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2)
    
    @staticmethod
    def load_verification_state(filepath: str) -> Optional[Dict[str, Any]]:
        """Load verification state from disk"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return None
```

## Integration Points

### With Core Parser

```python
# Integration with DoclingParser
from src.core.parser import DoclingParser
from src.verification.interface import VerificationInterface

def run_verification_workflow(pdf_path: str):
    """Complete parsing and verification workflow"""
    # Parse document
    parser = DoclingParser(enable_ocr=True, enable_tables=True)
    document = parser.parse_document(pdf_path)
    
    # Launch verification interface
    interface = VerificationInterface()
    interface.render_verification_ui(document)
```

### With Search Engine

```python
# Verification of search results
def verify_search_results(
    search_results: List[Dict[str, Any]],
    query: str
) -> List[Dict[str, Any]]:
    """Verify search results through visual interface"""
    interface = VerificationInterface()
    
    # Convert search results to elements for verification
    elements = [result['element'] for result in search_results]
    
    # Create verification document
    verification_doc = ParsedDocument(
        elements=elements,
        pages=get_pages_from_elements(elements)
    )
    
    # Run verification
    interface.render_verification_ui(verification_doc)
    
    return get_verified_results()
```

## Configuration and Customization

### Render Configuration

```python
# Custom render configuration
custom_config = RenderConfig(
    highlight_color=(0, 255, 0, 100),  # Green highlights
    border_color=(0, 0, 255),          # Blue borders
    border_width=3,
    font_size=14,
    show_confidence=True,
    show_element_type=True,
    page_scale=1.5
)

renderer = VerificationRenderer(config=custom_config)
```

### Interface Customization

```python
# Custom verification interface
class CustomVerificationInterface(VerificationInterface):
    """Extended verification interface with custom features"""
    
    def _render_custom_controls(self, document: ParsedDocument):
        """Add custom verification controls"""
        st.subheader("Advanced Verification Tools")
        
        # Batch verification controls
        if st.button("Verify All High Confidence"):
            self._batch_verify_high_confidence(document)
        
        # Quality metrics
        self._display_quality_metrics(document)
    
    def _batch_verify_high_confidence(self, document: ParsedDocument):
        """Automatically verify elements above confidence threshold"""
        threshold = st.slider("Confidence Threshold", 0.5, 1.0, 0.9)
        
        high_confidence_elements = [
            e for e in document.elements 
            if e.confidence >= threshold
        ]
        
        for element in high_confidence_elements:
            self._verify_element(element)
        
        st.success(f"Verified {len(high_confidence_elements)} high-confidence elements")
```

## Performance Optimization

### Lazy Loading

```python
class OptimizedVerificationRenderer(VerificationRenderer):
    """Performance-optimized verification renderer"""
    
    def __init__(self, config: RenderConfig = None):
        super().__init__(config)
        self._rendered_cache = {}
        self._max_cache_size = 10
    
    def render_page_with_overlays(
        self, 
        page_image: Image.Image,
        elements: List[DocumentElement],
        page_number: int
    ) -> Image.Image:
        """Render with caching for performance"""
        cache_key = self._generate_cache_key(elements, page_number)
        
        if cache_key in self._rendered_cache:
            return self._rendered_cache[cache_key]
        
        # Render and cache
        rendered = super().render_page_with_overlays(
            page_image, elements, page_number
        )
        
        # Manage cache size
        if len(self._rendered_cache) >= self._max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._rendered_cache))
            del self._rendered_cache[oldest_key]
        
        self._rendered_cache[cache_key] = rendered
        return rendered
    
    def _generate_cache_key(
        self, 
        elements: List[DocumentElement], 
        page_number: int
    ) -> str:
        """Generate cache key for rendered page"""
        element_hash = hash(tuple(
            (e.text, e.element_type, str(e.bbox))
            for e in elements
        ))
        return f"{page_number}_{element_hash}_{hash(str(self.config))}"
```

## Testing Integration

### Verification Test Utilities

```python
def create_test_verification_state():
    """Create test verification state for testing"""
    return {
        'current_page': 1,
        'verified_elements': {'page_1_element_1', 'page_1_element_2'},
        'flagged_elements': {'page_1_element_3'},
        'corrections': {
            'page_1_element_1': {'text': 'corrected text'}
        },
        'filter_type': 'text'
    }

def assert_verification_quality(document: ParsedDocument, min_accuracy: float = 0.95):
    """Assert verification meets quality standards"""
    total_elements = len(document.elements)
    high_confidence = sum(1 for e in document.elements if e.confidence > 0.9)
    
    accuracy = high_confidence / total_elements if total_elements > 0 else 0
    assert accuracy >= min_accuracy, f"Verification accuracy {accuracy:.3f} below threshold {min_accuracy}"
```

This comprehensive implementation guide provides the foundation for building a robust PDF parsing verification system with interactive visual feedback, coordinate transformation, and export capabilities.
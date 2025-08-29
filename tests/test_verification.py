"""
Unit tests for interactive verification system.
Tests coordinate transformation, highlight rendering, and verification data export.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
from datetime import datetime
from PIL import Image
import numpy as np

# Import the classes we'll be testing (they don't exist yet - TDD approach)
from src.verification.renderer import PDFRenderer, CoordinateTransformer
from src.verification.interface import VerificationInterface, VerificationState
from src.core.models import DocumentElement


class TestPDFRenderer:
    """Test suite for PDFRenderer class."""
    
    @pytest.fixture
    def sample_bbox(self):
        """Sample bounding box in PDF coordinates."""
        return {'x0': 100, 'y0': 200, 'x1': 300, 'y1': 250}
    
    @pytest.fixture
    def mock_page_image(self):
        """Mock page image for testing."""
        # Create a small test image
        return Image.new('RGB', (600, 800), color='white')
    
    @pytest.fixture
    def pdf_renderer(self):
        """Create PDFRenderer instance for testing."""
        mock_parser = Mock()
        return PDFRenderer(pdf_parser=mock_parser)
    
    def test_renderer_initialization(self, pdf_renderer):
        """Test PDFRenderer initializes correctly."""
        assert pdf_renderer is not None
        assert hasattr(pdf_renderer, 'pdf_parser')
        assert hasattr(pdf_renderer, 'coordinate_transformer')
    
    def test_coordinate_transformation(self, pdf_renderer, sample_bbox, mock_page_image):
        """Test coordinate transformation from PDF to image coordinates."""
        # Transform PDF coordinates to image coordinates
        image_coords = pdf_renderer.transform_coordinates(sample_bbox, mock_page_image)
        
        assert isinstance(image_coords, dict)
        assert 'x0' in image_coords
        assert 'y0' in image_coords
        assert 'x1' in image_coords
        assert 'y1' in image_coords
        
        # Coordinates should be numeric
        assert isinstance(image_coords['x0'], (int, float))
        assert isinstance(image_coords['y0'], (int, float))
        assert isinstance(image_coords['x1'], (int, float))
        assert isinstance(image_coords['y1'], (int, float))
    
    def test_coordinates_within_image_bounds(self, pdf_renderer, sample_bbox, mock_page_image):
        """Test that transformed coordinates are within image bounds."""
        image_coords = pdf_renderer.transform_coordinates(sample_bbox, mock_page_image)
        image_width, image_height = mock_page_image.size
        
        # All coordinates should be within image bounds
        assert 0 <= image_coords['x0'] <= image_width
        assert 0 <= image_coords['y0'] <= image_height
        assert 0 <= image_coords['x1'] <= image_width
        assert 0 <= image_coords['y1'] <= image_height
        
        # x1 should be >= x0, y1 should be >= y0
        assert image_coords['x1'] >= image_coords['x0']
        assert image_coords['y1'] >= image_coords['y0']
    
    def test_coordinate_transformation_edge_cases(self, pdf_renderer, mock_page_image):
        """Test coordinate transformation with edge cases."""
        # Zero-width/height box
        zero_bbox = {'x0': 100, 'y0': 100, 'x1': 100, 'y1': 100}
        coords = pdf_renderer.transform_coordinates(zero_bbox, mock_page_image)
        assert coords['x0'] == coords['x1']
        assert coords['y0'] == coords['y1']
        
        # Very large coordinates
        large_bbox = {'x0': 0, 'y0': 0, 'x1': 10000, 'y1': 10000}
        coords = pdf_renderer.transform_coordinates(large_bbox, mock_page_image)
        # Should be clamped to image bounds
        assert coords['x1'] <= mock_page_image.size[0]
        assert coords['y1'] <= mock_page_image.size[1]
    
    def test_render_highlight_rectangle(self, pdf_renderer, sample_bbox, mock_page_image):
        """Test rendering highlight rectangles on images."""
        highlighted_image = pdf_renderer.render_highlight(
            mock_page_image, 
            sample_bbox,
            color=(255, 0, 0, 128),  # Semi-transparent red
            thickness=2
        )
        
        assert isinstance(highlighted_image, Image.Image)
        assert highlighted_image.size == mock_page_image.size
        # Image should be modified (different from original)
        assert highlighted_image != mock_page_image
    
    def test_render_multiple_highlights(self, pdf_renderer, mock_page_image):
        """Test rendering multiple highlight rectangles."""
        bboxes = [
            {'x0': 50, 'y0': 50, 'x1': 150, 'y1': 100},
            {'x0': 200, 'y0': 150, 'x1': 300, 'y1': 200},
            {'x0': 100, 'y0': 300, 'x1': 200, 'y1': 350}
        ]
        
        highlighted_image = pdf_renderer.render_multiple_highlights(
            mock_page_image, 
            bboxes,
            colors=[(255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128)]
        )
        
        assert isinstance(highlighted_image, Image.Image)
        assert highlighted_image.size == mock_page_image.size
    
    def test_highlight_colors_and_transparency(self, pdf_renderer, sample_bbox, mock_page_image):
        """Test different highlight colors and transparency levels."""
        # Test different colors
        red_image = pdf_renderer.render_highlight(mock_page_image, sample_bbox, color=(255, 0, 0, 100))
        blue_image = pdf_renderer.render_highlight(mock_page_image, sample_bbox, color=(0, 0, 255, 100))
        
        assert red_image != blue_image
        
        # Test different transparency levels
        opaque_image = pdf_renderer.render_highlight(mock_page_image, sample_bbox, color=(255, 0, 0, 255))
        transparent_image = pdf_renderer.render_highlight(mock_page_image, sample_bbox, color=(255, 0, 0, 50))
        
        assert opaque_image != transparent_image
    
    def test_get_page_image(self, pdf_renderer):
        """Test retrieving page images from PDF."""
        mock_page_num = 1
        
        with patch.object(pdf_renderer.pdf_parser, 'get_page_image') as mock_get:
            mock_get.return_value = Image.new('RGB', (600, 800), color='white')
            
            page_image = pdf_renderer.get_page_image(mock_page_num)
            
            assert isinstance(page_image, Image.Image)
            mock_get.assert_called_once_with(mock_page_num)
    
    def test_invalid_page_number(self, pdf_renderer):
        """Test handling of invalid page numbers."""
        with patch.object(pdf_renderer.pdf_parser, 'get_page_image') as mock_get:
            mock_get.side_effect = ValueError("Invalid page number")
            
            with pytest.raises(ValueError):
                pdf_renderer.get_page_image(-1)
    
    def test_coordinate_transformer_scaling(self):
        """Test coordinate transformation scaling calculations."""
        transformer = CoordinateTransformer()
        
        pdf_size = (612, 792)  # Standard letter size in points
        image_size = (600, 800)  # Image size in pixels
        
        scale_x, scale_y = transformer.calculate_scaling(pdf_size, image_size)
        
        assert isinstance(scale_x, float)
        assert isinstance(scale_y, float)
        assert scale_x > 0
        assert scale_y > 0
    
    def test_coordinate_transformer_aspect_ratio(self):
        """Test that coordinate transformer handles aspect ratio differences."""
        transformer = CoordinateTransformer()
        
        # Different aspect ratios
        pdf_size = (612, 792)  # 4:3 aspect ratio
        image_size = (800, 600)  # 16:9 aspect ratio
        
        bbox = {'x0': 100, 'y0': 100, 'x1': 200, 'y1': 200}
        
        transformed = transformer.transform_bbox(bbox, pdf_size, image_size)
        
        # Should maintain relative positioning
        assert transformed['x0'] < transformed['x1']
        assert transformed['y0'] < transformed['y1']


class TestVerificationInterface:
    """Test suite for VerificationInterface class."""
    
    @pytest.fixture
    def sample_elements(self):
        """Create sample elements for verification testing."""
        return [
            DocumentElement(
                text="Sample heading",
                element_type="heading",
                page_number=1,
                bbox={'x0': 100, 'y0': 50, 'x1': 300, 'y1': 80},
                confidence=0.9,
                metadata={}
            ),
            DocumentElement(
                text="Sample paragraph text",
                element_type="text",
                page_number=1,
                bbox={'x0': 100, 'y0': 100, 'x1': 400, 'y1': 140},
                confidence=0.85,
                metadata={}
            ),
            DocumentElement(
                text="Table data",
                element_type="table",
                page_number=1,
                bbox={'x0': 100, 'y0': 200, 'x1': 500, 'y1': 300},
                confidence=0.8,
                metadata={'rows': 3, 'columns': 2}
            )
        ]
    
    @pytest.fixture
    def verification_interface(self, sample_elements):
        """Create VerificationInterface instance for testing."""
        mock_renderer = Mock(spec=PDFRenderer)
        return VerificationInterface(elements=sample_elements, renderer=mock_renderer)
    
    def test_interface_initialization(self, verification_interface, sample_elements):
        """Test VerificationInterface initializes correctly."""
        assert verification_interface is not None
        assert len(verification_interface.elements) == len(sample_elements)
        assert hasattr(verification_interface, 'verification_states')
        assert hasattr(verification_interface, 'renderer')
    
    def test_initial_verification_states(self, verification_interface, sample_elements):
        """Test that initial verification states are set correctly."""
        states = verification_interface.verification_states
        
        assert len(states) == len(sample_elements)
        
        for i, element in enumerate(sample_elements):
            assert i in states
            assert states[i].status == "pending"
            assert states[i].timestamp is not None
            assert states[i].element_id == i
    
    def test_mark_element_correct(self, verification_interface):
        """Test marking an element as correct."""
        element_id = 0
        
        verification_interface.mark_element_correct(element_id)
        
        state = verification_interface.verification_states[element_id]
        assert state.status == "correct"
        assert state.verified_by is not None
        assert isinstance(state.timestamp, datetime)
    
    def test_mark_element_incorrect(self, verification_interface):
        """Test marking an element as incorrect with correction."""
        element_id = 1
        correction = "Corrected text content"
        
        verification_interface.mark_element_incorrect(element_id, correction=correction)
        
        state = verification_interface.verification_states[element_id]
        assert state.status == "incorrect"
        assert state.correction == correction
        assert state.verified_by is not None
        assert isinstance(state.timestamp, datetime)
    
    def test_mark_element_partial(self, verification_interface):
        """Test marking an element as partially correct."""
        element_id = 2
        notes = "Partially correct - missing some details"
        
        verification_interface.mark_element_partial(element_id, notes=notes)
        
        state = verification_interface.verification_states[element_id]
        assert state.status == "partial"
        assert state.notes == notes
        assert isinstance(state.timestamp, datetime)
    
    def test_invalid_element_id(self, verification_interface):
        """Test handling of invalid element IDs."""
        with pytest.raises((IndexError, KeyError)):
            verification_interface.mark_element_correct(999)
        
        with pytest.raises((IndexError, KeyError)):
            verification_interface.mark_element_incorrect(-1)
    
    def test_get_element_state(self, verification_interface):
        """Test retrieving element verification state."""
        element_id = 0
        
        # Initially pending
        state = verification_interface.get_element_state(element_id)
        assert state.status == "pending"
        
        # After marking correct
        verification_interface.mark_element_correct(element_id)
        state = verification_interface.get_element_state(element_id)
        assert state.status == "correct"
    
    def test_get_verification_summary(self, verification_interface):
        """Test getting verification summary statistics."""
        # Mark some elements
        verification_interface.mark_element_correct(0)
        verification_interface.mark_element_incorrect(1, correction="Fixed text")
        # Leave element 2 as pending
        
        summary = verification_interface.get_verification_summary()
        
        assert isinstance(summary, dict)
        assert 'total' in summary
        assert 'correct' in summary
        assert 'incorrect' in summary
        assert 'partial' in summary
        assert 'pending' in summary
        assert 'accuracy' in summary
        
        assert summary['total'] == 3
        assert summary['correct'] == 1
        assert summary['incorrect'] == 1
        assert summary['pending'] == 1
        assert 0 <= summary['accuracy'] <= 1
    
    def test_verification_summary_by_page(self, verification_interface):
        """Test getting verification summary by page."""
        verification_interface.mark_element_correct(0)
        verification_interface.mark_element_incorrect(1, correction="Fixed")
        
        by_page_summary = verification_interface.get_verification_summary_by_page()
        
        assert isinstance(by_page_summary, dict)
        assert 1 in by_page_summary  # Page 1 should exist
        
        page1_summary = by_page_summary[1]
        assert 'total' in page1_summary
        assert 'correct' in page1_summary
        assert 'incorrect' in page1_summary
        assert page1_summary['total'] == 3  # All elements on page 1
    
    def test_verification_summary_by_element_type(self, verification_interface):
        """Test getting verification summary by element type."""
        verification_interface.mark_element_correct(0)  # heading
        verification_interface.mark_element_incorrect(1)  # text
        
        by_type_summary = verification_interface.get_verification_summary_by_type()
        
        assert isinstance(by_type_summary, dict)
        assert 'heading' in by_type_summary
        assert 'text' in by_type_summary
        assert 'table' in by_type_summary
        
        heading_summary = by_type_summary['heading']
        assert heading_summary['total'] == 1
        assert heading_summary['correct'] == 1
    
    def test_export_verification_data_json(self, verification_interface):
        """Test exporting verification data as JSON."""
        # Mark some elements
        verification_interface.mark_element_correct(0)
        verification_interface.mark_element_incorrect(1, correction="Corrected")
        verification_interface.mark_element_partial(2, notes="Needs review")
        
        json_data = verification_interface.export_verification_data(format='json')
        
        assert isinstance(json_data, str)
        
        # Should be valid JSON
        parsed_data = json.loads(json_data)
        assert isinstance(parsed_data, dict)
        
        # Should contain expected sections
        assert 'summary' in parsed_data
        assert 'elements' in parsed_data
        assert 'by_page' in parsed_data
        assert 'by_type' in parsed_data
        assert 'export_timestamp' in parsed_data
        
        # Summary should have totals
        summary = parsed_data['summary']
        assert summary['total'] == 3
        assert summary['correct'] == 1
        assert summary['incorrect'] == 1
        assert summary['partial'] == 1
    
    def test_export_verification_data_csv(self, verification_interface):
        """Test exporting verification data as CSV."""
        verification_interface.mark_element_correct(0)
        verification_interface.mark_element_incorrect(1, correction="Fixed")
        
        csv_data = verification_interface.export_verification_data(format='csv')
        
        assert isinstance(csv_data, str)
        assert 'element_id' in csv_data
        assert 'status' in csv_data
        assert 'element_type' in csv_data
        assert 'page_number' in csv_data
        
        # Should have data rows (plus header)
        lines = csv_data.strip().split('\n')
        assert len(lines) >= 4  # Header + 3 data rows
    
    def test_export_with_corrections_only(self, verification_interface):
        """Test exporting only elements that need corrections."""
        verification_interface.mark_element_correct(0)
        verification_interface.mark_element_incorrect(1, correction="Fixed text")
        verification_interface.mark_element_partial(2, notes="Needs work")
        
        corrections_data = verification_interface.export_verification_data(
            format='json', 
            corrections_only=True
        )
        
        parsed_data = json.loads(corrections_data)
        elements = parsed_data['elements']
        
        # Should only include incorrect and partial elements
        assert len(elements) == 2
        assert all(elem['status'] in ['incorrect', 'partial'] for elem in elements)
    
    def test_load_verification_state(self, verification_interface):
        """Test loading verification state from saved data."""
        # Mark elements and export
        verification_interface.mark_element_correct(0)
        verification_interface.mark_element_incorrect(1, correction="Fixed")
        
        exported_data = verification_interface.export_verification_data(format='json')
        
        # Create new interface and load state
        new_interface = VerificationInterface(
            elements=verification_interface.elements,
            renderer=verification_interface.renderer
        )
        
        new_interface.load_verification_state(exported_data)
        
        # States should match
        assert new_interface.get_element_state(0).status == "correct"
        assert new_interface.get_element_state(1).status == "incorrect"
        assert new_interface.get_element_state(1).correction == "Fixed"
    
    def test_verification_progress_tracking(self, verification_interface):
        """Test tracking verification progress."""
        # Initially 0% complete
        progress = verification_interface.get_verification_progress()
        assert progress['percent_complete'] == 0.0
        assert progress['elements_verified'] == 0
        assert progress['elements_remaining'] == 3
        
        # Mark one element
        verification_interface.mark_element_correct(0)
        progress = verification_interface.get_verification_progress()
        assert progress['percent_complete'] == pytest.approx(33.33, rel=1e-2)
        assert progress['elements_verified'] == 1
        assert progress['elements_remaining'] == 2
        
        # Mark all elements
        verification_interface.mark_element_incorrect(1)
        verification_interface.mark_element_partial(2)
        progress = verification_interface.get_verification_progress()
        assert progress['percent_complete'] == 100.0
        assert progress['elements_verified'] == 3
        assert progress['elements_remaining'] == 0
    
    def test_bulk_verification_operations(self, verification_interface):
        """Test bulk verification operations."""
        element_ids = [0, 1]
        
        verification_interface.mark_elements_correct(element_ids)
        
        for element_id in element_ids:
            state = verification_interface.get_element_state(element_id)
            assert state.status == "correct"
    
    def test_verification_with_user_info(self, verification_interface):
        """Test verification with user information tracking."""
        user_id = "test_user"
        
        verification_interface.mark_element_correct(0, verified_by=user_id)
        
        state = verification_interface.get_element_state(0)
        assert state.verified_by == user_id
    
    def test_verification_undo_functionality(self, verification_interface):
        """Test undoing verification decisions."""
        element_id = 0
        
        # Mark as correct
        verification_interface.mark_element_correct(element_id)
        assert verification_interface.get_element_state(element_id).status == "correct"
        
        # Undo - should return to pending
        verification_interface.undo_verification(element_id)
        assert verification_interface.get_element_state(element_id).status == "pending"
    
    def test_verification_history(self, verification_interface):
        """Test verification history tracking."""
        element_id = 0
        
        # Multiple status changes
        verification_interface.mark_element_correct(element_id)
        verification_interface.mark_element_incorrect(element_id, correction="Fixed")
        verification_interface.mark_element_correct(element_id)
        
        history = verification_interface.get_verification_history(element_id)
        
        assert len(history) >= 3
        assert history[-1]['status'] == 'correct'  # Most recent
        
    def test_confidence_threshold_filtering(self, verification_interface):
        """Test filtering elements by confidence threshold for verification."""
        # Get elements below confidence threshold
        low_confidence_elements = verification_interface.get_elements_needing_verification(
            confidence_threshold=0.9
        )
        
        # Should include elements with confidence < 0.9
        assert len(low_confidence_elements) >= 2  # elements with 0.85 and 0.8 confidence


@pytest.fixture
def mock_verification_state():
    """Create a mock VerificationState for testing."""
    return VerificationState(
        element_id=0,
        status="pending",
        timestamp=datetime.now(),
        verified_by="test_user"
    )
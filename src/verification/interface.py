"""
Verification interface for interactive PDF parsing validation.
Provides state management and verification workflow for parsed elements.
"""

import json
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from io import StringIO

from src.core.models import DocumentElement
from src.verification.renderer import PDFRenderer


@dataclass
class VerificationState:
    """Represents the verification state of a document element."""
    
    element_id: int
    status: str  # 'pending', 'correct', 'incorrect', 'partial'
    timestamp: datetime
    verified_by: Optional[str] = None
    correction: Optional[str] = None
    notes: Optional[str] = None
    
    def __post_init__(self):
        """Validate the verification state after initialization."""
        valid_statuses = {'pending', 'correct', 'incorrect', 'partial'}
        if self.status not in valid_statuses:
            raise ValueError(f"Invalid status: {self.status}. Must be one of {valid_statuses}")


class VerificationInterface:
    """Interface for PDF parsing verification workflow."""
    
    def __init__(self, elements: List[DocumentElement], renderer: Optional[PDFRenderer] = None):
        """Initialize verification interface.
        
        Args:
            elements: List of document elements to verify
            renderer: PDFRenderer instance for visual rendering
        """
        self.elements = elements
        self.renderer = renderer or PDFRenderer()
        
        # Initialize verification states
        self.verification_states: Dict[int, VerificationState] = {}
        self._verification_history: Dict[int, List[Dict[str, Any]]] = {}
        
        self._init_verification_states()
    
    def _init_verification_states(self):
        """Initialize verification states for all elements."""
        for i, element in enumerate(self.elements):
            self.verification_states[i] = VerificationState(
                element_id=i,
                status='pending',
                timestamp=datetime.now()
            )
            self._verification_history[i] = [{
                'status': 'pending',
                'timestamp': datetime.now().isoformat(),
                'action': 'initialized'
            }]
    
    def mark_element_correct(self, element_id: int, verified_by: Optional[str] = None):
        """Mark an element as verified correct.
        
        Args:
            element_id: ID of element to mark as correct
            verified_by: User ID who verified the element
            
        Raises:
            IndexError: If element_id is invalid
        """
        if element_id not in self.verification_states:
            raise IndexError(f"Invalid element_id: {element_id}")
        
        state = self.verification_states[element_id]
        state.status = 'correct'
        state.timestamp = datetime.now()
        state.verified_by = verified_by or "system"  # Default to "system" if not provided
        state.correction = None  # Clear any previous correction
        
        self._add_history_entry(element_id, 'marked_correct', state.verified_by)
    
    def mark_element_incorrect(
        self, 
        element_id: int, 
        correction: Optional[str] = None,
        verified_by: Optional[str] = None
    ):
        """Mark an element as incorrect with optional correction.
        
        Args:
            element_id: ID of element to mark as incorrect
            correction: Corrected text content
            verified_by: User ID who verified the element
            
        Raises:
            IndexError: If element_id is invalid
        """
        if element_id not in self.verification_states:
            raise IndexError(f"Invalid element_id: {element_id}")
        
        state = self.verification_states[element_id]
        state.status = 'incorrect'
        state.timestamp = datetime.now()
        state.verified_by = verified_by or "system"  # Default to "system" if not provided
        state.correction = correction
        
        self._add_history_entry(element_id, 'marked_incorrect', state.verified_by, {'correction': correction})
    
    def mark_element_partial(
        self, 
        element_id: int, 
        notes: Optional[str] = None,
        verified_by: Optional[str] = None
    ):
        """Mark an element as partially correct.
        
        Args:
            element_id: ID of element to mark as partial
            notes: Notes about what needs improvement
            verified_by: User ID who verified the element
            
        Raises:
            IndexError: If element_id is invalid
        """
        if element_id not in self.verification_states:
            raise IndexError(f"Invalid element_id: {element_id}")
        
        state = self.verification_states[element_id]
        state.status = 'partial'
        state.timestamp = datetime.now()
        state.verified_by = verified_by
        state.notes = notes
        
        self._add_history_entry(element_id, 'marked_partial', verified_by, {'notes': notes})
    
    def get_element_state(self, element_id: int) -> VerificationState:
        """Get verification state for an element.
        
        Args:
            element_id: ID of element to get state for
            
        Returns:
            VerificationState for the element
            
        Raises:
            IndexError: If element_id is invalid
        """
        if element_id not in self.verification_states:
            raise IndexError(f"Invalid element_id: {element_id}")
        
        return self.verification_states[element_id]
    
    def get_verification_summary(self) -> Dict[str, Union[int, float]]:
        """Get overall verification summary statistics.
        
        Returns:
            Dictionary with verification statistics
        """
        total = len(self.verification_states)
        if total == 0:
            return {
                'total': 0, 'correct': 0, 'incorrect': 0, 
                'partial': 0, 'pending': 0, 'accuracy': 0.0
            }
        
        status_counts = {'correct': 0, 'incorrect': 0, 'partial': 0, 'pending': 0}
        
        for state in self.verification_states.values():
            status_counts[state.status] = status_counts.get(state.status, 0) + 1
        
        # Calculate accuracy (correct / (total - pending))
        verified_count = total - status_counts['pending']
        accuracy = status_counts['correct'] / verified_count if verified_count > 0 else 0.0
        
        return {
            'total': total,
            'correct': status_counts['correct'],
            'incorrect': status_counts['incorrect'],
            'partial': status_counts['partial'],
            'pending': status_counts['pending'],
            'accuracy': accuracy
        }
    
    def get_verification_summary_by_page(self) -> Dict[int, Dict[str, Union[int, float]]]:
        """Get verification summary by page.
        
        Returns:
            Dictionary mapping page numbers to verification summaries
        """
        page_summaries = {}
        
        for element_id, element in enumerate(self.elements):
            page_num = element.page_number
            
            if page_num not in page_summaries:
                page_summaries[page_num] = {
                    'total': 0, 'correct': 0, 'incorrect': 0, 
                    'partial': 0, 'pending': 0
                }
            
            page_summaries[page_num]['total'] += 1
            
            state = self.verification_states[element_id]
            page_summaries[page_num][state.status] += 1
        
        # Calculate accuracy for each page
        for page_summary in page_summaries.values():
            verified = page_summary['total'] - page_summary['pending']
            page_summary['accuracy'] = (
                page_summary['correct'] / verified if verified > 0 else 0.0
            )
        
        return page_summaries
    
    def get_verification_summary_by_type(self) -> Dict[str, Dict[str, Union[int, float]]]:
        """Get verification summary by element type.
        
        Returns:
            Dictionary mapping element types to verification summaries
        """
        type_summaries = {}
        
        for element_id, element in enumerate(self.elements):
            element_type = element.element_type
            
            if element_type not in type_summaries:
                type_summaries[element_type] = {
                    'total': 0, 'correct': 0, 'incorrect': 0, 
                    'partial': 0, 'pending': 0
                }
            
            type_summaries[element_type]['total'] += 1
            
            state = self.verification_states[element_id]
            type_summaries[element_type][state.status] += 1
        
        # Calculate accuracy for each type
        for type_summary in type_summaries.values():
            verified = type_summary['total'] - type_summary['pending']
            type_summary['accuracy'] = (
                type_summary['correct'] / verified if verified > 0 else 0.0
            )
        
        return type_summaries
    
    def get_verification_progress(self) -> Dict[str, Union[int, float]]:
        """Get verification progress statistics.
        
        Returns:
            Dictionary with progress information
        """
        total = len(self.verification_states)
        pending_count = sum(1 for state in self.verification_states.values() 
                          if state.status == 'pending')
        verified_count = total - pending_count
        
        return {
            'total_elements': total,
            'elements_verified': verified_count,
            'elements_remaining': pending_count,
            'percent_complete': (verified_count / total * 100.0) if total > 0 else 0.0
        }
    
    def mark_elements_correct(self, element_ids: List[int], verified_by: Optional[str] = None):
        """Mark multiple elements as correct in bulk.
        
        Args:
            element_ids: List of element IDs to mark as correct
            verified_by: User ID who verified the elements
        """
        for element_id in element_ids:
            self.mark_element_correct(element_id, verified_by)
    
    def undo_verification(self, element_id: int):
        """Undo verification decision, returning element to pending state.
        
        Args:
            element_id: ID of element to undo verification for
            
        Raises:
            IndexError: If element_id is invalid
        """
        if element_id not in self.verification_states:
            raise IndexError(f"Invalid element_id: {element_id}")
        
        state = self.verification_states[element_id]
        previous_status = state.status
        
        state.status = 'pending'
        state.timestamp = datetime.now()
        state.verified_by = None
        state.correction = None
        state.notes = None
        
        self._add_history_entry(element_id, 'undone', None, {'previous_status': previous_status})
    
    def get_verification_history(self, element_id: int) -> List[Dict[str, Any]]:
        """Get verification history for an element.
        
        Args:
            element_id: ID of element to get history for
            
        Returns:
            List of history entries
            
        Raises:
            IndexError: If element_id is invalid
        """
        if element_id not in self._verification_history:
            raise IndexError(f"Invalid element_id: {element_id}")
        
        return self._verification_history[element_id].copy()
    
    def get_elements_needing_verification(
        self, 
        confidence_threshold: float = 0.9
    ) -> List[Tuple[int, DocumentElement]]:
        """Get elements that need verification based on confidence threshold.
        
        Args:
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            List of tuples (element_id, element) that need verification
        """
        needing_verification = []
        
        for element_id, element in enumerate(self.elements):
            state = self.verification_states[element_id]
            
            # Include if pending or below confidence threshold
            if (state.status == 'pending' or 
                element.confidence < confidence_threshold):
                needing_verification.append((element_id, element))
        
        return needing_verification
    
    def export_verification_data(
        self, 
        format: str = 'json',
        corrections_only: bool = False
    ) -> str:
        """Export verification data in specified format.
        
        Args:
            format: Export format ('json' or 'csv')
            corrections_only: Only include elements needing corrections
            
        Returns:
            Exported data as string
            
        Raises:
            ValueError: If format is not supported
        """
        if format not in ['json', 'csv']:
            raise ValueError(f"Unsupported format: {format}")
        
        if format == 'json':
            return self._export_json(corrections_only)
        else:
            return self._export_csv(corrections_only)
    
    def _export_json(self, corrections_only: bool) -> str:
        """Export verification data as JSON.
        
        Args:
            corrections_only: Only include elements needing corrections
            
        Returns:
            JSON string
        """
        export_data = {
            'summary': self.get_verification_summary(),
            'by_page': self.get_verification_summary_by_page(),
            'by_type': self.get_verification_summary_by_type(),
            'export_timestamp': datetime.now().isoformat(),
            'corrections_only': corrections_only,
            'elements': []
        }
        
        for element_id, element in enumerate(self.elements):
            state = self.verification_states[element_id]
            
            # Filter if corrections_only is True
            if corrections_only and state.status not in ['incorrect', 'partial']:
                continue
            
            element_data = {
                'element_id': element_id,
                'text': element.text,
                'element_type': element.element_type,
                'page_number': element.page_number,
                'confidence': element.confidence,
                'bbox': element.bbox,
                'metadata': element.metadata,
                'status': state.status,  # Add status at top level for test compatibility
                'timestamp': state.timestamp.isoformat(),
                'verified_by': state.verified_by,
                'correction': state.correction,
                'notes': state.notes,
                'verification_state': {
                    'status': state.status,
                    'timestamp': state.timestamp.isoformat(),
                    'verified_by': state.verified_by,
                    'correction': state.correction,
                    'notes': state.notes
                }
            }
            
            export_data['elements'].append(element_data)
        
        return json.dumps(export_data, indent=2, ensure_ascii=False)
    
    def _export_csv(self, corrections_only: bool) -> str:
        """Export verification data as CSV.
        
        Args:
            corrections_only: Only include elements needing corrections
            
        Returns:
            CSV string
        """
        output = StringIO()
        
        fieldnames = [
            'element_id', 'text', 'element_type', 'page_number', 'confidence',
            'bbox_x0', 'bbox_y0', 'bbox_x1', 'bbox_y1',
            'status', 'timestamp', 'verified_by', 'correction', 'notes'
        ]
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for element_id, element in enumerate(self.elements):
            state = self.verification_states[element_id]
            
            # Filter if corrections_only is True
            if corrections_only and state.status not in ['incorrect', 'partial']:
                continue
            
            row = {
                'element_id': element_id,
                'text': element.text,
                'element_type': element.element_type,
                'page_number': element.page_number,
                'confidence': element.confidence,
                'bbox_x0': element.bbox['x0'],
                'bbox_y0': element.bbox['y0'],
                'bbox_x1': element.bbox['x1'],
                'bbox_y1': element.bbox['y1'],
                'status': state.status,
                'timestamp': state.timestamp.isoformat(),
                'verified_by': state.verified_by or '',
                'correction': state.correction or '',
                'notes': state.notes or ''
            }
            
            writer.writerow(row)
        
        return output.getvalue()
    
    def load_verification_state(self, exported_data: str):
        """Load verification state from exported JSON data.
        
        Args:
            exported_data: JSON string from previous export
            
        Raises:
            ValueError: If data format is invalid
            json.JSONDecodeError: If JSON is malformed
        """
        try:
            data = json.loads(exported_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}")
        
        if 'elements' not in data:
            raise ValueError("Invalid data format: missing 'elements' key")
        
        # Load verification states
        for element_data in data['elements']:
            element_id = element_data['element_id']
            
            if element_id >= len(self.elements):
                continue  # Skip if element doesn't exist
            
            verification_state = element_data['verification_state']
            
            state = self.verification_states[element_id]
            state.status = verification_state['status']
            state.timestamp = datetime.fromisoformat(verification_state['timestamp'])
            state.verified_by = verification_state.get('verified_by')
            state.correction = verification_state.get('correction')
            state.notes = verification_state.get('notes')
            
            # Add history entry for loaded state
            self._add_history_entry(element_id, 'loaded_from_export', None)
    
    def _add_history_entry(
        self, 
        element_id: int, 
        action: str, 
        user: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add entry to verification history.
        
        Args:
            element_id: Element ID
            action: Action performed
            user: User who performed action
            metadata: Additional metadata
        """
        if element_id not in self._verification_history:
            self._verification_history[element_id] = []
        
        state = self.verification_states[element_id]
        
        entry = {
            'action': action,
            'status': state.status,
            'timestamp': datetime.now().isoformat(),
            'user': user,
            'metadata': metadata or {}
        }
        
        self._verification_history[element_id].append(entry)
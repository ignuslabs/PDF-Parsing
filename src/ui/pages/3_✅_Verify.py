"""
Verify Page - Visual Document Verification

Provides visual verification of parsed document elements with overlays, 
element selection, and correction capabilities.
"""

# CRITICAL: Setup Python path FIRST, before any project imports
import sys
from pathlib import Path

# Add the src directory to Python path BEFORE importing project modules
src_path = Path(__file__).resolve().parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Standard library imports
import streamlit as st
from typing import List, Optional, Dict, Any, Tuple
import io
import base64
from PIL import Image
import numpy as np

try:
    from src.core.models import DocumentElement, ParsedDocument
    from src.verification.interface import VerificationInterface, VerificationState
    from src.verification.renderer import PDFRenderer, RenderConfig
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.error("Please ensure the application is running from the correct directory")
    st.stop()

st.set_page_config(
    page_title="Verify Documents",
    page_icon="✅",
    layout="wide"
)

def initialize_session_state():
    """Initialize session state for verification page."""
    if 'parsed_documents' not in st.session_state:
        st.session_state.parsed_documents = []
    
    if 'current_doc_index' not in st.session_state:
        st.session_state.current_doc_index = 0
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    
    if 'selected_element_id' not in st.session_state:
        st.session_state.selected_element_id = None
    
    if 'verification_interfaces' not in st.session_state:
        st.session_state.verification_interfaces = {}
    
    if 'pdf_renderers' not in st.session_state:
        st.session_state.pdf_renderers = {}

def get_current_document() -> Optional[ParsedDocument]:
    """Get the currently selected document."""
    if not st.session_state.parsed_documents:
        return None
    
    if st.session_state.current_doc_index >= len(st.session_state.parsed_documents):
        st.session_state.current_doc_index = 0
    
    return st.session_state.parsed_documents[st.session_state.current_doc_index]

def get_verification_interface(doc_index: int) -> Optional[VerificationInterface]:
    """Get or create verification interface for a document."""
    if doc_index not in st.session_state.verification_interfaces:
        doc = st.session_state.parsed_documents[doc_index]
        st.session_state.verification_interfaces[doc_index] = VerificationInterface(doc.elements)
    
    return st.session_state.verification_interfaces[doc_index]

def get_pdf_renderer(doc_index: int) -> Optional[PDFRenderer]:
    """Get or create PDF renderer for a document."""
    if doc_index not in st.session_state.pdf_renderers:
        config = RenderConfig(
            highlight_color=(255, 255, 0, 80),  # Semi-transparent yellow
            border_color=(255, 0, 0, 180),      # Red border
            border_width=2,
            show_element_type=True,
            show_confidence=True
        )
        st.session_state.pdf_renderers[doc_index] = PDFRenderer(config=config)
    
    return st.session_state.pdf_renderers[doc_index]

def get_page_elements(doc: ParsedDocument, page_number: int) -> List[DocumentElement]:
    """Get all elements for a specific page."""
    return [e for e in doc.elements if e.page_number == page_number]

def display_document_selector():
    """Display document selection interface."""
    if not st.session_state.parsed_documents:
        st.warning("No documents loaded. Please go to the Parse page to upload PDFs.")
        return False
    
    # Document selection
    doc_names = []
    for i, doc in enumerate(st.session_state.parsed_documents):
        name = doc.metadata.get('filename', f'Document {i+1}')
        element_count = len(doc.elements)
        page_count = doc.metadata.get('page_count', 'Unknown')
        doc_names.append(f"{name} ({element_count} elements, {page_count} pages)")
    
    selected_idx = st.selectbox(
        "Select Document",
        options=range(len(doc_names)),
        format_func=lambda i: doc_names[i],
        index=st.session_state.current_doc_index
    )
    
    if selected_idx != st.session_state.current_doc_index:
        st.session_state.current_doc_index = selected_idx
        st.session_state.current_page = 1
        st.session_state.selected_element_id = None
        st.rerun()
    
    return True

def display_page_navigation(doc: ParsedDocument):
    """Display page navigation controls."""
    max_pages = doc.metadata.get('page_count', len(doc.pages) if doc.pages else 1)
    
    col1, col2, col3, col4, col5 = st.columns([1, 1, 3, 1, 1])
    
    with col1:
        if st.button("⬅️ Prev", disabled=st.session_state.current_page <= 1):
            st.session_state.current_page = max(1, st.session_state.current_page - 1)
            st.session_state.selected_element_id = None
            st.rerun()
    
    with col2:
        if st.button("➡️ Next", disabled=st.session_state.current_page >= max_pages):
            st.session_state.current_page = min(max_pages, st.session_state.current_page + 1)
            st.session_state.selected_element_id = None
            st.rerun()
    
    with col3:
        page_number = st.slider(
            "Page",
            min_value=1,
            max_value=max_pages,
            value=st.session_state.current_page,
            format="Page %d of %d" % (st.session_state.current_page, max_pages)
        )
        
        if page_number != st.session_state.current_page:
            st.session_state.current_page = page_number
            st.session_state.selected_element_id = None
            st.rerun()
    
    with col4:
        if st.button("🏠 First"):
            st.session_state.current_page = 1
            st.session_state.selected_element_id = None
            st.rerun()
    
    with col5:
        if st.button("🔚 Last"):
            st.session_state.current_page = max_pages
            st.session_state.selected_element_id = None
            st.rerun()

def display_verification_stats(verification_interface: VerificationInterface):
    """Display verification progress statistics."""
    summary = verification_interface.get_verification_summary()
    progress = verification_interface.get_verification_progress()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Elements", 
            summary['total_elements'],
            help="Total number of elements in the document"
        )
    
    with col2:
        st.metric(
            "Verified", 
            summary['verified_elements'],
            delta=f"{progress['completion_percentage']:.1f}%",
            help="Number of elements that have been verified"
        )
    
    with col3:
        st.metric(
            "Corrections", 
            summary['total_corrections'],
            help="Number of elements that needed corrections"
        )
    
    with col4:
        st.metric(
            "Accuracy", 
            f"{summary['accuracy_percentage']:.1f}%",
            help="Percentage of elements that were correct"
        )

def render_page_with_overlays(
    doc: ParsedDocument, 
    page_number: int, 
    renderer: PDFRenderer
) -> Optional[Image.Image]:
    """Render a page with element overlays."""
    if not doc.pages or page_number not in doc.pages:
        st.error(f"Page {page_number} image not available")
        return None
    
    page_image = doc.pages[page_number]
    page_elements = get_page_elements(doc, page_number)
    
    if not page_elements:
        return page_image
    
    # Render overlays
    try:
        overlay_image = renderer.render_page_with_overlays(
            page_image,
            page_elements,
            page_number
        )
        return overlay_image
    except Exception as e:
        st.error(f"Failed to render overlays: {e}")
        return page_image

def display_element_details(element: DocumentElement, verification_interface: VerificationInterface):
    """Display detailed information about a selected element."""
    st.subheader(f"📋 Element Details")
    
    # Basic information
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Type:** {element.element_type.title()}")
        st.write(f"**Page:** {element.page_number}")
        st.write(f"**Confidence:** {element.confidence:.3f}")
    
    with col2:
        if element.bbox:
            st.write(f"**Position:** ({element.bbox['x0']:.1f}, {element.bbox['y0']:.1f}) to ({element.bbox['x1']:.1f}, {element.bbox['y1']:.1f})")
        st.write(f"**Element ID:** {element.metadata.get('element_id', 'Unknown')}")
    
    # Text content
    st.write("**Text Content:**")
    text_content = st.text_area(
        "Element text",
        value=element.text,
        height=100,
        key=f"element_text_{element.metadata.get('element_id', 'unknown')}",
        help="You can edit the text here to make corrections"
    )
    
    # Verification status
    element_id = element.metadata.get('element_id', 0)
    current_state = verification_interface.get_element_state(element_id)
    
    st.write(f"**Verification Status:** {current_state.status.value}")
    
    if current_state.notes:
        st.write(f"**Notes:** {current_state.notes}")
    
    if current_state.corrected_text:
        st.write(f"**Corrected Text:** {current_state.corrected_text}")
    
    # Verification actions
    st.subheader("✅ Verification Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Mark Correct", type="primary", use_container_width=True):
            verification_interface.mark_element_correct(element_id, verified_by="user")
            st.success("Marked as correct!")
            st.rerun()
    
    with col2:
        if st.button("❌ Mark Incorrect", type="secondary", use_container_width=True):
            corrected_text = text_content if text_content != element.text else None
            notes = "Text corrected by user" if corrected_text else "Marked as incorrect"
            
            verification_interface.mark_element_incorrect(
                element_id,
                corrected_text=corrected_text,
                notes=notes,
                verified_by="user"
            )
            st.success("Marked as incorrect!")
            st.rerun()
    
    with col3:
        if st.button("⚠️ Partially Correct", use_container_width=True):
            corrected_text = text_content if text_content != element.text else None
            notes = "Partially correct, minor adjustments made"
            
            verification_interface.mark_element_partial(
                element_id,
                corrected_text=corrected_text,
                notes=notes,
                verified_by="user"
            )
            st.success("Marked as partially correct!")
            st.rerun()
    
    # Additional notes
    with st.expander("📝 Add Notes"):
        notes = st.text_area(
            "Verification notes",
            placeholder="Add any additional notes about this element...",
            key=f"notes_{element_id}"
        )
        
        if st.button("Save Notes"):
            # Update the element's verification state with notes
            current_state.notes = notes
            st.success("Notes saved!")

def display_page_elements_list(page_elements: List[DocumentElement], verification_interface: VerificationInterface):
    """Display a list of elements on the current page."""
    st.subheader(f"📋 Page Elements ({len(page_elements)})")
    
    for i, element in enumerate(page_elements):
        element_id = element.metadata.get('element_id', i)
        state = verification_interface.get_element_state(element_id)
        
        # Status emoji
        status_emoji = {
            'pending': '⏳',
            'correct': '✅',
            'incorrect': '❌',
            'partial': '⚠️'
        }
        
        emoji = status_emoji.get(state.status.value, '⏳')
        
        # Create expandable element
        text_preview = element.text[:100] + "..." if len(element.text) > 100 else element.text
        
        with st.expander(f"{emoji} {element.element_type.title()} - {text_preview}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Full Text:** {element.text}")
                if element.bbox:
                    st.write(f"**Position:** ({element.bbox['x0']:.1f}, {element.bbox['y0']:.1f}) to ({element.bbox['x1']:.1f}, {element.bbox['y1']:.1f})")
            
            with col2:
                if st.button(f"Select", key=f"select_{element_id}"):
                    st.session_state.selected_element_id = element_id
                    st.rerun()
                
                st.write(f"**Confidence:** {element.confidence:.2f}")
                st.write(f"**Status:** {state.status.value}")

def main():
    """Main verify page function."""
    initialize_session_state()
    
    st.title("✅ Verify Documents")
    st.markdown("Review and verify parsed document elements with visual overlays.")
    
    # Document selection
    if not display_document_selector():
        return
    
    doc = get_current_document()
    if not doc:
        return
    
    verification_interface = get_verification_interface(st.session_state.current_doc_index)
    renderer = get_pdf_renderer(st.session_state.current_doc_index)
    
    # Verification statistics
    display_verification_stats(verification_interface)
    
    st.divider()
    
    # Page navigation
    display_page_navigation(doc)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📄 Page {st.session_state.current_page}")
        
        # Render page with overlays
        overlay_image = render_page_with_overlays(
            doc, 
            st.session_state.current_page, 
            renderer
        )
        
        if overlay_image:
            # Display the image
            st.image(
                overlay_image,
                caption=f"Page {st.session_state.current_page} with element overlays",
                use_container_width=True
            )
            
            # Click instruction
            st.info("💡 Click on highlighted elements in the image above to select them for verification.")
        else:
            st.error("Could not render page with overlays")
    
    with col2:
        # Element details or selection
        page_elements = get_page_elements(doc, st.session_state.current_page)
        
        if st.session_state.selected_element_id is not None:
            # Find the selected element
            selected_element = None
            for element in page_elements:
                if element.metadata.get('element_id') == st.session_state.selected_element_id:
                    selected_element = element
                    break
            
            if selected_element:
                display_element_details(selected_element, verification_interface)
            else:
                st.warning("Selected element not found on current page")
                st.session_state.selected_element_id = None
        else:
            # Show page elements list
            display_page_elements_list(page_elements, verification_interface)
    
    # Page summary
    st.divider()
    
    # Bulk actions
    st.subheader("🔄 Bulk Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("✅ Mark All Correct", use_container_width=True):
            element_ids = [e.metadata.get('element_id', i) for i, e in enumerate(page_elements)]
            verification_interface.mark_elements_correct(element_ids, verified_by="user")
            st.success(f"Marked {len(element_ids)} elements as correct!")
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset Page", use_container_width=True):
            for element in page_elements:
                element_id = element.metadata.get('element_id', 0)
                verification_interface.undo_verification(element_id)
            st.success("Reset all verifications on this page!")
            st.rerun()
    
    with col3:
        if st.button("📊 Export Progress", use_container_width=True):
            st.switch_page("pages/4_📊_Export.py")
    
    with col4:
        needing_verification = verification_interface.get_elements_needing_verification()
        if needing_verification and st.button("⏭️ Next Unverified", use_container_width=True):
            # Find next element needing verification
            next_element = needing_verification[0]
            next_page = next_element.page_number
            
            if next_page != st.session_state.current_page:
                st.session_state.current_page = next_page
            
            st.session_state.selected_element_id = next_element.metadata.get('element_id')
            st.rerun()

if __name__ == "__main__":
    main()
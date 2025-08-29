"""
Smart PDF Parser - Streamlit Application

Main entry point for the Smart PDF Parser UI application with document parsing,
search, verification, and export capabilities.
"""

import streamlit as st
from pathlib import Path
from typing import List, Dict, Any, Optional

# Configure Streamlit page
st.set_page_config(
    page_title="Smart PDF Parser",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize Streamlit session state with default values."""
    
    # Document storage
    if 'parsed_documents' not in st.session_state:
        st.session_state.parsed_documents = []
    
    if 'current_doc_index' not in st.session_state:
        st.session_state.current_doc_index = 0
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    
    # Element selection and verification
    if 'selected_element_id' not in st.session_state:
        st.session_state.selected_element_id = None
    
    if 'verification_states' not in st.session_state:
        st.session_state.verification_states = {}
    
    # Search functionality
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""
    
    if 'search_filters' not in st.session_state:
        st.session_state.search_filters = {
            'element_types': [],
            'page_range': None,
            'confidence_threshold': 0.0
        }
    
    # Parser configuration
    if 'parser_config' not in st.session_state:
        st.session_state.parser_config = {
            'enable_ocr': False,
            'enable_tables': True,
            'generate_page_images': True,
            'ocr_engine': 'tesseract',
            'ocr_language': 'eng',
            'table_mode': 'accurate',
            'image_scale': 1.0
        }
    
    # UI state
    if 'show_debug_info' not in st.session_state:
        st.session_state.show_debug_info = False
    
    if 'sidebar_expanded' not in st.session_state:
        st.session_state.sidebar_expanded = True

def get_current_document():
    """Get the currently selected document."""
    if not st.session_state.parsed_documents:
        return None
    
    if st.session_state.current_doc_index >= len(st.session_state.parsed_documents):
        st.session_state.current_doc_index = 0
    
    return st.session_state.parsed_documents[st.session_state.current_doc_index]

def get_current_elements():
    """Get elements from the current document."""
    doc = get_current_document()
    return doc.elements if doc else []

def display_sidebar():
    """Display sidebar navigation and controls."""
    with st.sidebar:
        st.title("📄 Smart PDF Parser")
        
        # Document selection
        if st.session_state.parsed_documents:
            st.subheader("Documents")
            
            doc_names = []
            for i, doc in enumerate(st.session_state.parsed_documents):
                name = doc.metadata.get('filename', f'Document {i+1}')
                doc_names.append(name)
            
            selected_doc = st.selectbox(
                "Select Document",
                options=range(len(doc_names)),
                format_func=lambda i: doc_names[i],
                index=st.session_state.current_doc_index,
                key="_doc_selector"
            )
            
            if selected_doc != st.session_state.current_doc_index:
                st.session_state.current_doc_index = selected_doc
                st.session_state.current_page = 1  # Reset to first page
                st.session_state.selected_element_id = None  # Clear selection
                st.rerun()
            
            # Current document info
            current_doc = get_current_document()
            if current_doc:
                st.info(f"""
                **Elements:** {len(current_doc.elements)}
                **Pages:** {current_doc.metadata.get('page_count', 'Unknown')}
                **Size:** {current_doc.metadata.get('file_size', 0) / 1024:.1f} KB
                """)
        else:
            st.info("No documents loaded. Go to the Parse page to upload PDFs.")
        
        st.divider()
        
        # Navigation shortcuts
        st.subheader("Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📄 Parse", use_container_width=True):
                st.switch_page("pages/1_📄_Parse.py")
        
        with col2:
            if st.button("🔍 Search", use_container_width=True):
                st.switch_page("pages/2_🔍_Search.py")
        
        col3, col4 = st.columns(2)
        with col3:
            if st.button("✅ Verify", use_container_width=True):
                st.switch_page("pages/3_✅_Verify.py")
        
        with col4:
            if st.button("📊 Export", use_container_width=True):
                st.switch_page("pages/4_📊_Export.py")
        
        st.divider()
        
        # Settings
        with st.expander("⚙️ Settings"):
            st.session_state.show_debug_info = st.checkbox(
                "Show debug information",
                value=st.session_state.show_debug_info
            )
            
            if st.button("Clear all data", type="secondary"):
                # Clear all session state
                for key in ['parsed_documents', 'search_results', 'verification_states']:
                    if key in st.session_state:
                        del st.session_state[key]
                initialize_session_state()
                st.success("All data cleared!")
                st.rerun()

def display_debug_info():
    """Display debug information if enabled."""
    if st.session_state.show_debug_info:
        with st.expander("🐛 Debug Information"):
            st.write("**Session State:**")
            
            debug_state = {
                'Documents': len(st.session_state.parsed_documents),
                'Current doc': st.session_state.current_doc_index,
                'Current page': st.session_state.current_page,
                'Selected element': st.session_state.selected_element_id,
                'Search results': len(st.session_state.search_results),
                'Verification states': len(st.session_state.verification_states)
            }
            
            st.json(debug_state)

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Display sidebar
    display_sidebar()
    
    # Main content area
    st.title("📄 Smart PDF Parser")
    st.markdown("""
    Welcome to Smart PDF Parser! This application provides comprehensive PDF parsing, 
    search, and verification capabilities using advanced document processing techniques.
    
    ## Getting Started
    
    1. **📄 Parse** - Upload and parse PDF documents
    2. **🔍 Search** - Find specific content in your documents
    3. **✅ Verify** - Review and correct parsed elements visually
    4. **📊 Export** - Download results in various formats
    
    Select an option from the sidebar or use the navigation pages at the top.
    """)
    
    # Show document overview if documents are loaded
    if st.session_state.parsed_documents:
        st.subheader("📋 Document Overview")
        
        # Create overview table
        overview_data = []
        for i, doc in enumerate(st.session_state.parsed_documents):
            metadata = doc.metadata
            overview_data.append({
                'Document': metadata.get('filename', f'Document {i+1}'),
                'Elements': len(doc.elements),
                'Pages': metadata.get('page_count', 'Unknown'),
                'Size (KB)': f"{metadata.get('file_size', 0) / 1024:.1f}",
                'Parsed': metadata.get('parsed_at', 'Unknown')[:16] if metadata.get('parsed_at') else 'Unknown'
            })
        
        st.dataframe(overview_data, use_container_width=True)
        
        # Element type distribution
        if get_current_elements():
            st.subheader("📊 Element Types")
            
            elements = get_current_elements()
            element_types = {}
            for element in elements:
                elem_type = element.element_type
                element_types[elem_type] = element_types.get(elem_type, 0) + 1
            
            # Display as columns
            type_cols = st.columns(min(len(element_types), 4))
            for i, (elem_type, count) in enumerate(element_types.items()):
                with type_cols[i % len(type_cols)]:
                    st.metric(elem_type.title(), count)
    
    # Display debug info if enabled
    display_debug_info()

if __name__ == "__main__":
    main()
# Streamlit UI Implementation Guide

## Overview

The Streamlit UI provides a comprehensive web-based interface for the Smart PDF Parser, integrating document upload, parsing configuration, search capabilities, and verification workflows into a cohesive user experience.

## Architecture

### Application Structure

```
src/ui/
├── app.py                 # Main Streamlit application
├── components/            # Reusable UI components
│   ├── upload_handler.py  # File upload management
│   ├── config_panel.py    # Configuration controls
│   ├── search_panel.py    # Search interface
│   └── results_display.py # Results visualization
├── pages/                 # Multi-page application
│   ├── parser_page.py     # PDF parsing interface
│   ├── search_page.py     # Search interface
│   └── verify_page.py     # Verification interface
├── utils/                 # UI utilities
│   ├── state_manager.py   # Session state management
│   ├── cache_manager.py   # Caching utilities
│   └── export_handler.py  # Export functionality
└── assets/                # Static assets
    ├── styles.css         # Custom CSS
    └── icons/            # UI icons
```

## Main Application

### Core App Structure

```python
import streamlit as st
import os
from typing import Optional, List, Dict, Any
from pathlib import Path

# Import core functionality
from src.core.parser import DoclingParser
from src.core.search import SmartSearchEngine
from src.core.models import ParsedDocument, DocumentElement
from src.verification.interface import VerificationInterface
from src.ui.components.upload_handler import FileUploadHandler
from src.ui.components.config_panel import ConfigurationPanel
from src.ui.components.search_panel import SearchPanel
from src.ui.components.results_display import ResultsDisplay
from src.ui.utils.state_manager import SessionStateManager

class SmartPDFParserApp:
    """Main Streamlit application for Smart PDF Parser"""
    
    def __init__(self):
        self.state_manager = SessionStateManager()
        self.upload_handler = FileUploadHandler()
        self.config_panel = ConfigurationPanel()
        self.search_panel = SearchPanel()
        self.results_display = ResultsDisplay()
        self._init_app_config()
    
    def _init_app_config(self):
        """Initialize Streamlit app configuration"""
        st.set_page_config(
            page_title="Smart PDF Parser",
            page_icon="📄",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/your-repo/smart-pdf-parser',
                'Report a bug': 'https://github.com/your-repo/smart-pdf-parser/issues',
                'About': "# Smart PDF Parser\nBuilt with Docling and Streamlit"
            }
        )
        
        # Load custom CSS
        self._load_custom_styles()
    
    def _load_custom_styles(self):
        """Load custom CSS styles"""
        css_file = Path(__file__).parent / "assets" / "styles.css"
        if css_file.exists():
            with open(css_file) as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    def run(self):
        """Run the main application"""
        # Initialize session state
        self.state_manager.initialize()
        
        # Main header
        self._render_header()
        
        # Sidebar navigation
        page = self._render_navigation()
        
        # Route to appropriate page
        if page == "Parse Documents":
            self._render_parser_page()
        elif page == "Search & Explore":
            self._render_search_page()
        elif page == "Verify Results":
            self._render_verification_page()
        elif page == "Export & Reports":
            self._render_export_page()
    
    def _render_header(self):
        """Render application header"""
        st.title("🔍 Smart PDF Parser")
        st.markdown(
            """
            **Intelligent PDF document processing with advanced search and verification**
            
            Upload PDFs, extract structured content, search with precision, and verify results
            through an interactive interface.
            """
        )
        
        # Status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            docs_count = len(st.session_state.get('parsed_documents', []))
            st.metric("Documents Processed", docs_count)
        
        with col2:
            elements_count = sum(
                len(doc.elements) for doc in st.session_state.get('parsed_documents', [])
            )
            st.metric("Elements Extracted", elements_count)
        
        with col3:
            search_count = st.session_state.get('search_count', 0)
            st.metric("Searches Performed", search_count)
        
        with col4:
            verified_count = len(st.session_state.get('verified_elements', set()))
            st.metric("Elements Verified", verified_count)
    
    def _render_navigation(self) -> str:
        """Render sidebar navigation"""
        with st.sidebar:
            st.header("Navigation")
            
            page = st.radio(
                "Select Page",
                [
                    "Parse Documents",
                    "Search & Explore", 
                    "Verify Results",
                    "Export & Reports"
                ],
                key="main_navigation"
            )
            
            # Quick actions
            st.header("Quick Actions")
            
            if st.button("🔄 Reset All Data"):
                if st.button("Confirm Reset", key="confirm_reset"):
                    self.state_manager.reset_session()
                    st.experimental_rerun()
            
            # System info
            st.header("System Info")
            st.info(f"Docling Version: {self._get_docling_version()}")
            st.info(f"Python Version: {self._get_python_version()}")
            
            return page
    
    def _render_parser_page(self):
        """Render PDF parsing interface"""
        st.header("📄 Parse PDF Documents")
        
        # File upload section
        uploaded_files = self.upload_handler.render_upload_interface()
        
        if uploaded_files:
            # Configuration panel
            config = self.config_panel.render_parser_config()
            
            # Parse button
            if st.button("🚀 Parse Documents", type="primary"):
                self._process_uploaded_files(uploaded_files, config)
            
        # Display parsed documents
        if st.session_state.get('parsed_documents'):
            self._display_parsed_documents()
    
    def _render_search_page(self):
        """Render search interface"""
        st.header("🔍 Search & Explore")
        
        if not st.session_state.get('parsed_documents'):
            st.warning("No documents parsed yet. Please parse documents first.")
            return
        
        # Search interface
        search_results = self.search_panel.render_search_interface(
            st.session_state.parsed_documents
        )
        
        # Display results
        if search_results:
            self.results_display.render_search_results(search_results)
    
    def _render_verification_page(self):
        """Render verification interface"""
        st.header("✅ Verify Results")
        
        if not st.session_state.get('parsed_documents'):
            st.warning("No documents to verify. Please parse documents first.")
            return
        
        # Document selector for verification
        doc_names = [doc.metadata.get('filename', f'Document {i+1}') 
                    for i, doc in enumerate(st.session_state.parsed_documents)]
        
        selected_doc_idx = st.selectbox(
            "Select document to verify",
            range(len(doc_names)),
            format_func=lambda x: doc_names[x]
        )
        
        selected_doc = st.session_state.parsed_documents[selected_doc_idx]
        
        # Launch verification interface
        verification_interface = VerificationInterface()
        verification_interface.render_verification_ui(selected_doc)
    
    def _render_export_page(self):
        """Render export and reporting interface"""
        st.header("📊 Export & Reports")
        
        if not st.session_state.get('parsed_documents'):
            st.warning("No documents to export. Please parse documents first.")
            return
        
        # Export options
        export_format = st.selectbox(
            "Export Format",
            ["JSON", "Markdown", "HTML", "CSV"]
        )
        
        # Document selection
        doc_options = ["All Documents"] + [
            doc.metadata.get('filename', f'Document {i+1}')
            for i, doc in enumerate(st.session_state.parsed_documents)
        ]
        
        selected_export = st.multiselect(
            "Select documents to export",
            doc_options,
            default=["All Documents"]
        )
        
        # Export button
        if st.button("📥 Generate Export"):
            export_data = self._generate_export(selected_export, export_format)
            
            # Download button
            st.download_button(
                label=f"Download {export_format} Export",
                data=export_data,
                file_name=f"pdf_parser_export.{export_format.lower()}",
                mime=self._get_mime_type(export_format)
            )
        
        # Analytics dashboard
        self._render_analytics_dashboard()
    
    def _process_uploaded_files(self, files: List[Any], config: Dict[str, Any]):
        """Process uploaded PDF files"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        parser = DoclingParser(
            enable_ocr=config['enable_ocr'],
            enable_tables=config['enable_tables'],
            enable_images=config['enable_images'],
            page_images=config['page_images']
        )
        
        parsed_documents = []
        
        for i, file in enumerate(files):
            status_text.text(f"Processing {file.name}...")
            
            # Save uploaded file temporarily
            temp_path = self._save_temp_file(file)
            
            try:
                # Parse document
                document = parser.parse_document(temp_path)
                document.metadata['filename'] = file.name
                document.metadata['upload_time'] = st.session_state['upload_time']
                
                parsed_documents.append(document)
                
                # Update progress
                progress_bar.progress((i + 1) / len(files))
                
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
            finally:
                # Clean up temp file
                os.unlink(temp_path)
        
        # Store in session state
        if 'parsed_documents' not in st.session_state:
            st.session_state.parsed_documents = []
        st.session_state.parsed_documents.extend(parsed_documents)
        
        status_text.text("Processing complete!")
        st.success(f"Successfully processed {len(parsed_documents)} documents")
    
    def _display_parsed_documents(self):
        """Display parsed documents summary"""
        st.subheader("Parsed Documents")
        
        for i, doc in enumerate(st.session_state.parsed_documents):
            with st.expander(f"📄 {doc.metadata.get('filename', f'Document {i+1}')}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Elements", len(doc.elements))
                    st.metric("Pages", len(set(e.page_number for e in doc.elements)))
                
                with col2:
                    element_types = {}
                    for element in doc.elements:
                        element_types[element.element_type] = element_types.get(element.element_type, 0) + 1
                    
                    for elem_type, count in element_types.items():
                        st.metric(f"{elem_type.title()}", count)
                
                with col3:
                    avg_confidence = sum(e.confidence for e in doc.elements) / len(doc.elements)
                    st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                    
                    # Quality indicators
                    high_confidence = sum(1 for e in doc.elements if e.confidence > 0.9)
                    quality_score = high_confidence / len(doc.elements)
                    st.metric("Quality Score", f"{quality_score:.3f}")
                
                # Element preview
                if st.checkbox(f"Show Elements Preview", key=f"preview_{i}"):
                    self._render_elements_preview(doc.elements[:10])  # Show first 10
    
    def _render_elements_preview(self, elements: List[DocumentElement]):
        """Render preview of document elements"""
        for element in elements:
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.text_area(
                        f"{element.element_type}",
                        value=element.text[:200] + ("..." if len(element.text) > 200 else ""),
                        height=60,
                        disabled=True,
                        key=f"preview_{hash(element.text)}"
                    )
                
                with col2:
                    st.write(f"**Page:** {element.page_number}")
                    st.write(f"**Conf:** {element.confidence:.3f}")
                
                with col3:
                    if st.button("🔍 Details", key=f"details_{hash(element.text)}"):
                        st.json({
                            "text": element.text,
                            "type": element.element_type,
                            "bbox": element.bbox,
                            "metadata": element.metadata
                        })
    
    def _render_analytics_dashboard(self):
        """Render analytics dashboard"""
        st.subheader("📈 Analytics Dashboard")
        
        if not st.session_state.get('parsed_documents'):
            return
        
        # Aggregate statistics
        all_elements = []
        for doc in st.session_state.parsed_documents:
            all_elements.extend(doc.elements)
        
        # Element type distribution
        st.subheader("Element Type Distribution")
        element_type_counts = {}
        for element in all_elements:
            element_type_counts[element.element_type] = element_type_counts.get(element.element_type, 0) + 1
        
        # Create bar chart
        import plotly.express as px
        import pandas as pd
        
        df_types = pd.DataFrame([
            {"Type": k, "Count": v} for k, v in element_type_counts.items()
        ])
        
        fig_types = px.bar(df_types, x="Type", y="Count", title="Elements by Type")
        st.plotly_chart(fig_types, use_container_width=True)
        
        # Confidence distribution
        st.subheader("Confidence Score Distribution")
        confidences = [e.confidence for e in all_elements]
        
        df_conf = pd.DataFrame({"Confidence": confidences})
        fig_conf = px.histogram(df_conf, x="Confidence", title="Confidence Score Distribution", nbins=20)
        st.plotly_chart(fig_conf, use_container_width=True)
        
        # Page distribution
        st.subheader("Elements per Page")
        page_counts = {}
        for element in all_elements:
            page_counts[element.page_number] = page_counts.get(element.page_number, 0) + 1
        
        df_pages = pd.DataFrame([
            {"Page": k, "Elements": v} for k, v in sorted(page_counts.items())
        ])
        
        fig_pages = px.line(df_pages, x="Page", y="Elements", title="Elements per Page")
        st.plotly_chart(fig_pages, use_container_width=True)
    
    def _save_temp_file(self, uploaded_file) -> str:
        """Save uploaded file to temporary location"""
        import tempfile
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return temp_path
    
    def _generate_export(self, selected_docs: List[str], format_type: str) -> str:
        """Generate export data in specified format"""
        # Implementation would handle different export formats
        documents_to_export = []
        
        if "All Documents" in selected_docs:
            documents_to_export = st.session_state.parsed_documents
        else:
            # Filter selected documents
            for doc in st.session_state.parsed_documents:
                if doc.metadata.get('filename') in selected_docs:
                    documents_to_export.append(doc)
        
        if format_type == "JSON":
            return self._export_as_json(documents_to_export)
        elif format_type == "Markdown":
            return self._export_as_markdown(documents_to_export)
        elif format_type == "HTML":
            return self._export_as_html(documents_to_export)
        elif format_type == "CSV":
            return self._export_as_csv(documents_to_export)
    
    def _export_as_json(self, documents: List[ParsedDocument]) -> str:
        """Export documents as JSON"""
        import json
        export_data = {
            "export_metadata": {
                "timestamp": st.session_state.get('export_time', ''),
                "document_count": len(documents),
                "total_elements": sum(len(doc.elements) for doc in documents)
            },
            "documents": [doc.export_to_dict() for doc in documents]
        }
        return json.dumps(export_data, indent=2)
    
    def _export_as_markdown(self, documents: List[ParsedDocument]) -> str:
        """Export documents as Markdown"""
        markdown_content = "# PDF Parser Export\n\n"
        
        for i, doc in enumerate(documents):
            markdown_content += f"## Document {i+1}: {doc.metadata.get('filename', 'Untitled')}\n\n"
            markdown_content += doc.export_to_markdown()
            markdown_content += "\n\n---\n\n"
        
        return markdown_content
    
    def _get_mime_type(self, format_type: str) -> str:
        """Get MIME type for export format"""
        mime_types = {
            "JSON": "application/json",
            "Markdown": "text/markdown",
            "HTML": "text/html",
            "CSV": "text/csv"
        }
        return mime_types.get(format_type, "text/plain")
    
    def _get_docling_version(self) -> str:
        """Get Docling version"""
        try:
            import docling
            return docling.__version__
        except:
            return "Unknown"
    
    def _get_python_version(self) -> str:
        """Get Python version"""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

# Main entry point
def main():
    """Main entry point for Streamlit app"""
    app = SmartPDFParserApp()
    app.run()

if __name__ == "__main__":
    main()
```

## UI Components

### File Upload Handler

```python
# src/ui/components/upload_handler.py
import streamlit as st
from typing import List, Any, Optional
import tempfile
import os
from datetime import datetime

class FileUploadHandler:
    """Handles file uploads and validation"""
    
    def __init__(self, max_file_size: int = 50):  # MB
        self.max_file_size = max_file_size * 1024 * 1024  # Convert to bytes
        
    def render_upload_interface(self) -> Optional[List[Any]]:
        """Render file upload interface"""
        st.subheader("📁 Upload PDF Documents")
        
        # Upload widget
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help=f"Maximum file size: {self.max_file_size // (1024*1024)}MB per file"
        )
        
        if uploaded_files:
            # Validate files
            valid_files = self._validate_uploads(uploaded_files)
            
            if valid_files:
                # Display upload summary
                self._display_upload_summary(valid_files)
                
                # Store upload time in session state
                st.session_state['upload_time'] = datetime.now().isoformat()
                
                return valid_files
        
        return None
    
    def _validate_uploads(self, files: List[Any]) -> List[Any]:
        """Validate uploaded files"""
        valid_files = []
        
        for file in files:
            # Size validation
            if file.size > self.max_file_size:
                st.error(f"File {file.name} is too large. Maximum size: {self.max_file_size // (1024*1024)}MB")
                continue
            
            # Format validation
            if not file.name.lower().endswith('.pdf'):
                st.error(f"File {file.name} is not a PDF file")
                continue
            
            # PDF header validation
            if not self._validate_pdf_header(file):
                st.error(f"File {file.name} does not appear to be a valid PDF")
                continue
            
            valid_files.append(file)
        
        return valid_files
    
    def _validate_pdf_header(self, file) -> bool:
        """Validate PDF file header"""
        try:
            # Read first few bytes to check PDF signature
            file.seek(0)
            header = file.read(8)
            file.seek(0)  # Reset for later use
            
            return header.startswith(b'%PDF-')
        except:
            return False
    
    def _display_upload_summary(self, files: List[Any]):
        """Display summary of uploaded files"""
        total_size = sum(file.size for file in files)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Files Selected", len(files))
        
        with col2:
            st.metric("Total Size", f"{total_size / (1024*1024):.1f} MB")
        
        with col3:
            avg_size = total_size / len(files) if files else 0
            st.metric("Average Size", f"{avg_size / (1024*1024):.1f} MB")
        
        # File details
        if st.checkbox("Show file details"):
            for file in files:
                st.write(f"📄 **{file.name}** - {file.size / (1024*1024):.1f} MB")
```

### Configuration Panel

```python
# src/ui/components/config_panel.py
import streamlit as st
from typing import Dict, Any

class ConfigurationPanel:
    """Parser configuration interface"""
    
    def render_parser_config(self) -> Dict[str, Any]:
        """Render parser configuration options"""
        st.subheader("⚙️ Parser Configuration")
        
        with st.expander("Parsing Options", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                enable_ocr = st.checkbox(
                    "Enable OCR",
                    value=True,
                    help="Extract text from scanned documents and images"
                )
                
                enable_tables = st.checkbox(
                    "Extract Tables",
                    value=True,
                    help="Detect and extract table structures"
                )
                
                enable_images = st.checkbox(
                    "Process Images",
                    value=False,
                    help="Extract and process embedded images"
                )
            
            with col2:
                page_images = st.checkbox(
                    "Generate Page Images",
                    value=True,
                    help="Create page images for verification"
                )
                
                table_mode = st.selectbox(
                    "Table Extraction Mode",
                    ["accurate", "fast"],
                    index=0,
                    help="Choose between accuracy and speed for table extraction"
                )
                
                ocr_lang = st.selectbox(
                    "OCR Language",
                    ["auto", "eng", "fra", "deu", "spa"],
                    index=0,
                    help="OCR language detection"
                )
        
        # Advanced options
        with st.expander("Advanced Options"):
            max_pages = st.number_input(
                "Maximum Pages to Process",
                min_value=1,
                max_value=1000,
                value=100,
                help="Limit processing to first N pages"
            )
            
            confidence_threshold = st.slider(
                "Minimum Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Filter out low-confidence extractions"
            )
            
            image_scale = st.slider(
                "Image Scale Factor",
                min_value=0.5,
                max_value=3.0,
                value=1.0,
                step=0.1,
                help="Scale factor for page images"
            )
        
        return {
            'enable_ocr': enable_ocr,
            'enable_tables': enable_tables,
            'enable_images': enable_images,
            'page_images': page_images,
            'table_mode': table_mode,
            'ocr_lang': ocr_lang,
            'max_pages': max_pages,
            'confidence_threshold': confidence_threshold,
            'image_scale': image_scale
        }
```

### Search Panel

```python
# src/ui/components/search_panel.py
import streamlit as st
from typing import List, Dict, Any, Optional
from src.core.search import SmartSearchEngine
from src.core.models import ParsedDocument

class SearchPanel:
    """Search interface component"""
    
    def render_search_interface(self, documents: List[ParsedDocument]) -> Optional[List[Dict[str, Any]]]:
        """Render search interface"""
        st.subheader("🔍 Smart Search")
        
        # Search query input
        query = st.text_input(
            "Search Query",
            placeholder="Enter search terms...",
            help="Search across all parsed documents"
        )
        
        if not query:
            return None
        
        # Search options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_mode = st.selectbox(
                "Search Mode",
                ["smart", "exact", "fuzzy", "semantic"],
                index=0,
                help="Choose search strategy"
            )
        
        with col2:
            element_types = st.multiselect(
                "Element Types",
                ["text", "title", "table", "image", "header", "footer"],
                default=[],
                help="Filter by element type"
            )
        
        with col3:
            max_results = st.number_input(
                "Max Results",
                min_value=1,
                max_value=100,
                value=20,
                help="Maximum number of results to return"
            )
        
        # Advanced search options
        with st.expander("Advanced Search Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                page_filter = st.text_input(
                    "Page Range",
                    placeholder="e.g., 1-5, 10",
                    help="Filter by page numbers"
                )
                
                confidence_min = st.slider(
                    "Minimum Confidence",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05
                )
            
            with col2:
                fuzzy_threshold = st.slider(
                    "Fuzzy Match Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.8,
                    step=0.05,
                    help="Similarity threshold for fuzzy matching"
                )
                
                boost_headers = st.checkbox(
                    "Boost Headers",
                    value=True,
                    help="Give higher relevance to headers and titles"
                )
        
        # Search button
        if st.button("🔍 Search", type="primary"):
            return self._perform_search(
                documents, query, search_mode, element_types,
                max_results, page_filter, confidence_min,
                fuzzy_threshold, boost_headers
            )
        
        return None
    
    def _perform_search(
        self,
        documents: List[ParsedDocument],
        query: str,
        search_mode: str,
        element_types: List[str],
        max_results: int,
        page_filter: str,
        confidence_min: float,
        fuzzy_threshold: float,
        boost_headers: bool
    ) -> List[Dict[str, Any]]:
        """Perform search across documents"""
        
        # Combine all elements from all documents
        all_elements = []
        for doc_idx, doc in enumerate(documents):
            for element in doc.elements:
                # Add document reference
                element.metadata['document_index'] = doc_idx
                element.metadata['document_filename'] = doc.metadata.get('filename', f'Document {doc_idx + 1}')
                all_elements.append(element)
        
        # Create search engine
        search_engine = SmartSearchEngine(
            all_elements,
            fuzzy_threshold=fuzzy_threshold,
            boost_headers=boost_headers
        )
        
        # Parse page filter
        page_numbers = self._parse_page_filter(page_filter) if page_filter else None
        
        # Perform search
        try:
            results = search_engine.search(
                query=query,
                mode=search_mode,
                element_types=element_types or None,
                page_numbers=page_numbers,
                min_confidence=confidence_min,
                max_results=max_results
            )
            
            # Update search count
            if 'search_count' not in st.session_state:
                st.session_state.search_count = 0
            st.session_state.search_count += 1
            
            return results
            
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            return []
    
    def _parse_page_filter(self, page_filter: str) -> List[int]:
        """Parse page filter string into list of page numbers"""
        try:
            pages = []
            parts = page_filter.split(',')
            
            for part in parts:
                part = part.strip()
                if '-' in part:
                    # Range (e.g., "1-5")
                    start, end = map(int, part.split('-'))
                    pages.extend(range(start, end + 1))
                else:
                    # Single page
                    pages.append(int(part))
            
            return sorted(list(set(pages)))  # Remove duplicates and sort
        except:
            st.warning(f"Invalid page filter format: {page_filter}")
            return []
```

## Session State Management

```python
# src/ui/utils/state_manager.py
import streamlit as st
from typing import Dict, Any, List
from datetime import datetime

class SessionStateManager:
    """Manages Streamlit session state"""
    
    def initialize(self):
        """Initialize session state with default values"""
        default_state = {
            'parsed_documents': [],
            'search_results': [],
            'verified_elements': set(),
            'flagged_elements': set(),
            'search_count': 0,
            'current_page': 1,
            'upload_time': None,
            'last_search_query': '',
            'verification_state': {},
            'export_history': []
        }
        
        for key, value in default_state.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def reset_session(self):
        """Reset session state"""
        keys_to_reset = [
            'parsed_documents', 'search_results', 'verified_elements',
            'flagged_elements', 'search_count', 'verification_state'
        ]
        
        for key in keys_to_reset:
            if key in st.session_state:
                if isinstance(st.session_state[key], set):
                    st.session_state[key] = set()
                elif isinstance(st.session_state[key], list):
                    st.session_state[key] = []
                elif isinstance(st.session_state[key], dict):
                    st.session_state[key] = {}
                else:
                    st.session_state[key] = 0
    
    def save_state_to_file(self, filepath: str):
        """Save session state to file"""
        import json
        
        # Convert sets to lists for JSON serialization
        serializable_state = {}
        for key, value in st.session_state.items():
            if isinstance(value, set):
                serializable_state[key] = list(value)
            elif hasattr(value, 'to_dict'):  # Custom objects with serialization
                serializable_state[key] = value.to_dict()
            elif isinstance(value, (str, int, float, bool, list, dict)):
                serializable_state[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(serializable_state, f, indent=2, default=str)
    
    def load_state_from_file(self, filepath: str):
        """Load session state from file"""
        import json
        
        try:
            with open(filepath, 'r') as f:
                loaded_state = json.load(f)
            
            # Update session state
            for key, value in loaded_state.items():
                if key in ['verified_elements', 'flagged_elements']:
                    st.session_state[key] = set(value)
                else:
                    st.session_state[key] = value
                    
            return True
        except Exception as e:
            st.error(f"Failed to load state: {str(e)}")
            return False
```

## Custom CSS Styling

```css
/* src/ui/assets/styles.css */

/* Main app styling */
.main-header {
    background: linear-gradient(90deg, #1f4e79 0%, #2d6aa0 100%);
    color: white;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}

/* Sidebar styling */
.css-1d391kg {
    background-color: #f8f9fa;
}

/* Success/error message styling */
.element-success {
    border-left: 4px solid #28a745;
    padding: 0.5rem;
    background-color: #d4edda;
    border-radius: 0.25rem;
}

.element-warning {
    border-left: 4px solid #ffc107;
    padding: 0.5rem;
    background-color: #fff3cd;
    border-radius: 0.25rem;
}

.element-error {
    border-left: 4px solid #dc3545;
    padding: 0.5rem;
    background-color: #f8d7da;
    border-radius: 0.25rem;
}

/* Search results styling */
.search-result {
    border: 1px solid #dee2e6;
    border-radius: 0.375rem;
    padding: 1rem;
    margin-bottom: 0.5rem;
    background-color: white;
}

.search-result:hover {
    box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
}

/* Confidence score styling */
.confidence-high {
    color: #28a745;
    font-weight: bold;
}

.confidence-medium {
    color: #ffc107;
    font-weight: bold;
}

.confidence-low {
    color: #dc3545;
    font-weight: bold;
}

/* Element type badges */
.element-badge {
    display: inline-block;
    padding: 0.25rem 0.5rem;
    margin: 0.125rem;
    font-size: 0.75rem;
    font-weight: 700;
    line-height: 1;
    text-align: center;
    white-space: nowrap;
    vertical-align: baseline;
    border-radius: 0.375rem;
}

.element-badge.text {
    background-color: #e3f2fd;
    color: #1565c0;
}

.element-badge.title {
    background-color: #f3e5f5;
    color: #7b1fa2;
}

.element-badge.table {
    background-color: #e8f5e8;
    color: #2e7d32;
}

.element-badge.image {
    background-color: #fff3e0;
    color: #ef6c00;
}

/* Responsive design */
@media (max-width: 768px) {
    .main-header {
        text-align: center;
    }
    
    .search-result {
        padding: 0.5rem;
    }
}
```

## Integration Examples

### Complete Usage Example

```python
# Run the Streamlit app
# streamlit run src/ui/app.py

# The app will provide:
# 1. File upload interface
# 2. Parser configuration
# 3. Document processing with progress tracking
# 4. Search interface with multiple modes
# 5. Verification interface with visual overlays
# 6. Export capabilities in multiple formats
# 7. Analytics dashboard with visualizations
```

This comprehensive UI implementation provides a complete web-based interface for the Smart PDF Parser, integrating all core functionality with an intuitive user experience.
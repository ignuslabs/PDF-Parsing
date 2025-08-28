"""
Export Page - Data Export and Analytics

Provides comprehensive export functionality for parsed documents,
search results, verification data, and analytics reports.
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
import json
import csv
import io
from typing import List, Optional, Dict, Any
from datetime import datetime
import zipfile
import pandas as pd

try:
    from core.models import DocumentElement, ParsedDocument
    from verification.interface import VerificationInterface
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.error("Please ensure the application is running from the correct directory")
    st.stop()

st.set_page_config(
    page_title="Export Data",
    page_icon="📊",
    layout="wide"
)

def initialize_session_state():
    """Initialize session state for export page."""
    if 'parsed_documents' not in st.session_state:
        st.session_state.parsed_documents = []
    
    if 'verification_interfaces' not in st.session_state:
        st.session_state.verification_interfaces = {}

def get_verification_interface(doc_index: int) -> Optional[VerificationInterface]:
    """Get verification interface for a document."""
    if doc_index in st.session_state.verification_interfaces:
        return st.session_state.verification_interfaces[doc_index]
    return None

def display_export_options():
    """Display export format and content options."""
    st.subheader("🎛️ Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Export Formats:**")
        export_json = st.checkbox("JSON (structured data)", value=True)
        export_csv = st.checkbox("CSV (tabular data)", value=True)
        export_markdown = st.checkbox("Markdown (readable text)", value=False)
        export_html = st.checkbox("HTML (web format)", value=False)
        export_text = st.checkbox("Plain Text", value=False)
    
    with col2:
        st.write("**Content Options:**")
        include_metadata = st.checkbox("Include metadata", value=True)
        include_bbox = st.checkbox("Include bounding boxes", value=True)
        include_verification = st.checkbox("Include verification data", value=True)
        corrections_only = st.checkbox("Corrections only", value=False)
    
    # Document selection
    if len(st.session_state.parsed_documents) > 1:
        st.write("**Document Selection:**")
        doc_names = [doc.metadata.get('filename', f'Document {i+1}') 
                    for i, doc in enumerate(st.session_state.parsed_documents)]
        
        selected_docs = st.multiselect(
            "Select documents to export",
            options=range(len(doc_names)),
            format_func=lambda i: doc_names[i],
            default=list(range(len(doc_names)))
        )
    else:
        selected_docs = list(range(len(st.session_state.parsed_documents)))
    
    return {
        'formats': {
            'json': export_json,
            'csv': export_csv,
            'markdown': export_markdown,
            'html': export_html,
            'text': export_text
        },
        'content': {
            'metadata': include_metadata,
            'bbox': include_bbox,
            'verification': include_verification,
            'corrections_only': corrections_only
        },
        'documents': selected_docs
    }

def export_to_json(documents: List[ParsedDocument], options: Dict[str, Any]) -> str:
    """Export documents to JSON format."""
    export_data = {
        'export_info': {
            'timestamp': datetime.now().isoformat(),
            'format': 'json',
            'document_count': len(documents),
            'options': options['content']
        },
        'documents': []
    }
    
    for i, doc in enumerate(documents):
        doc_data = {
            'document_id': i,
            'elements': []
        }
        
        # Add metadata if requested
        if options['content']['metadata']:
            doc_data['metadata'] = doc.metadata
        
        # Add verification summary if available
        verification_interface = get_verification_interface(i)
        if verification_interface and options['content']['verification']:
            doc_data['verification_summary'] = verification_interface.get_verification_summary()
        
        # Add elements
        for element in doc.elements:
            element_data = {
                'text': element.text,
                'element_type': element.element_type,
                'page_number': element.page_number,
                'confidence': element.confidence
            }
            
            if options['content']['bbox'] and element.bbox:
                element_data['bbox'] = element.bbox
            
            if options['content']['metadata']:
                element_data['metadata'] = element.metadata
            
            # Add verification data if available
            if verification_interface and options['content']['verification']:
                element_id = element.metadata.get('element_id', 0)
                verification_state = verification_interface.get_element_state(element_id)
                
                element_data['verification'] = {
                    'status': verification_state.status.value,
                    'verified_by': verification_state.verified_by,
                    'verified_at': verification_state.verified_at.isoformat() if verification_state.verified_at else None,
                    'corrected_text': verification_state.corrected_text,
                    'notes': verification_state.notes
                }
                
                # Skip elements if corrections_only is True and no corrections
                if options['content']['corrections_only'] and not verification_state.corrected_text:
                    continue
            
            doc_data['elements'].append(element_data)
        
        export_data['documents'].append(doc_data)
    
    return json.dumps(export_data, indent=2, ensure_ascii=False)

def export_to_csv(documents: List[ParsedDocument], options: Dict[str, Any]) -> str:
    """Export documents to CSV format."""
    rows = []
    
    for doc_idx, doc in enumerate(documents):
        verification_interface = get_verification_interface(doc_idx)
        
        for element in doc.elements:
            row = {
                'document_id': doc_idx,
                'document_name': doc.metadata.get('filename', f'Document {doc_idx + 1}'),
                'element_id': element.metadata.get('element_id', ''),
                'text': element.text,
                'element_type': element.element_type,
                'page_number': element.page_number,
                'confidence': element.confidence
            }
            
            # Add bounding box data
            if options['content']['bbox'] and element.bbox:
                row.update({
                    'bbox_x0': element.bbox['x0'],
                    'bbox_y0': element.bbox['y0'],
                    'bbox_x1': element.bbox['x1'],
                    'bbox_y1': element.bbox['y1']
                })
            
            # Add verification data
            if verification_interface and options['content']['verification']:
                element_id = element.metadata.get('element_id', 0)
                verification_state = verification_interface.get_element_state(element_id)
                
                row.update({
                    'verification_status': verification_state.status.value,
                    'verified_by': verification_state.verified_by or '',
                    'verified_at': verification_state.verified_at.isoformat() if verification_state.verified_at else '',
                    'corrected_text': verification_state.corrected_text or '',
                    'verification_notes': verification_state.notes or ''
                })
                
                # Skip if corrections_only and no corrections
                if options['content']['corrections_only'] and not verification_state.corrected_text:
                    continue
            
            rows.append(row)
    
    if not rows:
        return ""
    
    # Convert to CSV
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    
    return output.getvalue()

def export_to_markdown(documents: List[ParsedDocument], options: Dict[str, Any]) -> str:
    """Export documents to Markdown format."""
    md_content = []
    
    # Header
    md_content.append("# Smart PDF Parser Export Report")
    md_content.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    md_content.append("")
    
    for doc_idx, doc in enumerate(documents):
        doc_name = doc.metadata.get('filename', f'Document {doc_idx + 1}')
        md_content.append(f"## {doc_name}")
        
        # Document metadata
        if options['content']['metadata']:
            md_content.append("### Document Information")
            md_content.append(f"- **Elements:** {len(doc.elements)}")
            md_content.append(f"- **Pages:** {doc.metadata.get('page_count', 'Unknown')}")
            md_content.append(f"- **Size:** {doc.metadata.get('file_size', 0) / 1024:.1f} KB")
            md_content.append(f"- **Parsed:** {doc.metadata.get('parsed_at', 'Unknown')}")
            md_content.append("")
        
        # Verification summary
        verification_interface = get_verification_interface(doc_idx)
        if verification_interface and options['content']['verification']:
            summary = verification_interface.get_verification_summary()
            md_content.append("### Verification Summary")
            md_content.append(f"- **Total Elements:** {summary['total_elements']}")
            md_content.append(f"- **Verified:** {summary['verified_elements']}")
            md_content.append(f"- **Corrections:** {summary['total_corrections']}")
            md_content.append(f"- **Accuracy:** {summary['accuracy_percentage']:.1f}%")
            md_content.append("")
        
        # Elements by page
        current_page = None
        for element in sorted(doc.elements, key=lambda x: (x.page_number, x.bbox['y0'] if x.bbox else 0)):
            # Skip if corrections_only and no corrections
            if options['content']['corrections_only']:
                if verification_interface:
                    element_id = element.metadata.get('element_id', 0)
                    verification_state = verification_interface.get_element_state(element_id)
                    if not verification_state.corrected_text:
                        continue
            
            if element.page_number != current_page:
                current_page = element.page_number
                md_content.append(f"### Page {current_page}")
                md_content.append("")
            
            # Element header
            md_content.append(f"#### {element.element_type.title()}")
            
            # Element content
            if verification_interface and options['content']['verification']:
                element_id = element.metadata.get('element_id', 0)
                verification_state = verification_interface.get_element_state(element_id)
                
                if verification_state.corrected_text:
                    md_content.append(f"**Original:** {element.text}")
                    md_content.append(f"**Corrected:** {verification_state.corrected_text}")
                else:
                    md_content.append(element.text)
                
                if verification_state.notes:
                    md_content.append(f"*Notes: {verification_state.notes}*")
            else:
                md_content.append(element.text)
            
            md_content.append("")
    
    return "\n".join(md_content)

def export_to_html(documents: List[ParsedDocument], options: Dict[str, Any]) -> str:
    """Export documents to HTML format."""
    html_content = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "    <title>Smart PDF Parser Export Report</title>",
        "    <style>",
        "        body { font-family: Arial, sans-serif; margin: 40px; }",
        "        .document { margin-bottom: 40px; border-bottom: 2px solid #ccc; }",
        "        .element { margin: 20px 0; padding: 15px; border-left: 3px solid #007acc; }",
        "        .element-type { font-weight: bold; color: #007acc; }",
        "        .verification { background-color: #f9f9f9; padding: 10px; margin: 10px 0; }",
        "        .corrected { background-color: #fff3cd; }",
        "        .metadata { font-size: 0.9em; color: #666; }",
        "    </style>",
        "</head>",
        "<body>",
        f"    <h1>Smart PDF Parser Export Report</h1>",
        f"    <p><em>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>"
    ]
    
    for doc_idx, doc in enumerate(documents):
        doc_name = doc.metadata.get('filename', f'Document {doc_idx + 1}')
        html_content.append(f"    <div class='document'>")
        html_content.append(f"        <h2>{doc_name}</h2>")
        
        # Document metadata
        if options['content']['metadata']:
            html_content.append("        <div class='metadata'>")
            html_content.append(f"            <p><strong>Elements:</strong> {len(doc.elements)}</p>")
            html_content.append(f"            <p><strong>Pages:</strong> {doc.metadata.get('page_count', 'Unknown')}</p>")
            html_content.append(f"            <p><strong>Size:</strong> {doc.metadata.get('file_size', 0) / 1024:.1f} KB</p>")
            html_content.append("        </div>")
        
        # Elements
        verification_interface = get_verification_interface(doc_idx)
        current_page = None
        
        for element in sorted(doc.elements, key=lambda x: (x.page_number, x.bbox['y0'] if x.bbox else 0)):
            # Skip if corrections_only and no corrections
            if options['content']['corrections_only']:
                if verification_interface:
                    element_id = element.metadata.get('element_id', 0)
                    verification_state = verification_interface.get_element_state(element_id)
                    if not verification_state.corrected_text:
                        continue
            
            if element.page_number != current_page:
                current_page = element.page_number
                html_content.append(f"        <h3>Page {current_page}</h3>")
            
            css_class = "element"
            if verification_interface and options['content']['verification']:
                element_id = element.metadata.get('element_id', 0)
                verification_state = verification_interface.get_element_state(element_id)
                if verification_state.corrected_text:
                    css_class += " corrected"
            
            html_content.append(f"        <div class='{css_class}'>")
            html_content.append(f"            <div class='element-type'>{element.element_type.title()}</div>")
            
            # Element content
            if verification_interface and options['content']['verification']:
                element_id = element.metadata.get('element_id', 0)
                verification_state = verification_interface.get_element_state(element_id)
                
                if verification_state.corrected_text:
                    html_content.append(f"            <p><strong>Original:</strong> {element.text}</p>")
                    html_content.append(f"            <p><strong>Corrected:</strong> {verification_state.corrected_text}</p>")
                else:
                    html_content.append(f"            <p>{element.text}</p>")
                
                if verification_state.notes:
                    html_content.append(f"            <div class='verification'>Notes: {verification_state.notes}</div>")
            else:
                html_content.append(f"            <p>{element.text}</p>")
            
            html_content.append("        </div>")
        
        html_content.append("    </div>")
    
    html_content.extend([
        "</body>",
        "</html>"
    ])
    
    return "\n".join(html_content)

def export_to_text(documents: List[ParsedDocument], options: Dict[str, Any]) -> str:
    """Export documents to plain text format."""
    text_content = []
    
    text_content.append("SMART PDF PARSER EXPORT REPORT")
    text_content.append("=" * 50)
    text_content.append(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    text_content.append("")
    
    for doc_idx, doc in enumerate(documents):
        doc_name = doc.metadata.get('filename', f'Document {doc_idx + 1}')
        text_content.append(f"DOCUMENT: {doc_name}")
        text_content.append("-" * (len(doc_name) + 10))
        
        # Document metadata
        if options['content']['metadata']:
            text_content.append(f"Elements: {len(doc.elements)}")
            text_content.append(f"Pages: {doc.metadata.get('page_count', 'Unknown')}")
            text_content.append(f"Size: {doc.metadata.get('file_size', 0) / 1024:.1f} KB")
            text_content.append("")
        
        # Elements
        verification_interface = get_verification_interface(doc_idx)
        current_page = None
        
        for element in sorted(doc.elements, key=lambda x: (x.page_number, x.bbox['y0'] if x.bbox else 0)):
            # Skip if corrections_only and no corrections
            if options['content']['corrections_only']:
                if verification_interface:
                    element_id = element.metadata.get('element_id', 0)
                    verification_state = verification_interface.get_element_state(element_id)
                    if not verification_state.corrected_text:
                        continue
            
            if element.page_number != current_page:
                current_page = element.page_number
                text_content.append(f"\nPAGE {current_page}")
                text_content.append("-" * 10)
            
            text_content.append(f"\n[{element.element_type.upper()}]")
            
            # Element content
            if verification_interface and options['content']['verification']:
                element_id = element.metadata.get('element_id', 0)
                verification_state = verification_interface.get_element_state(element_id)
                
                if verification_state.corrected_text:
                    text_content.append(f"ORIGINAL: {element.text}")
                    text_content.append(f"CORRECTED: {verification_state.corrected_text}")
                else:
                    text_content.append(element.text)
                
                if verification_state.notes:
                    text_content.append(f"NOTES: {verification_state.notes}")
            else:
                text_content.append(element.text)
        
        text_content.append("\n" + "=" * 50 + "\n")
    
    return "\n".join(text_content)

def create_export_files(documents: List[ParsedDocument], options: Dict[str, Any]) -> Dict[str, str]:
    """Create export files based on selected options."""
    files = {}
    
    if options['formats']['json']:
        files['export.json'] = export_to_json(documents, options)
    
    if options['formats']['csv']:
        files['export.csv'] = export_to_csv(documents, options)
    
    if options['formats']['markdown']:
        files['export.md'] = export_to_markdown(documents, options)
    
    if options['formats']['html']:
        files['export.html'] = export_to_html(documents, options)
    
    if options['formats']['text']:
        files['export.txt'] = export_to_text(documents, options)
    
    return files

def display_export_preview(files: Dict[str, str]):
    """Display preview of export files."""
    if not files:
        return
    
    st.subheader("📋 Export Preview")
    
    # Create tabs for different formats
    tabs = st.tabs(list(files.keys()))
    
    for i, (filename, content) in enumerate(files.items()):
        with tabs[i]:
            if filename.endswith('.json'):
                st.json(json.loads(content))
            elif filename.endswith('.csv'):
                # Display as dataframe
                try:
                    df = pd.read_csv(io.StringIO(content))
                    st.dataframe(df, use_container_width=True)
                except:
                    st.text(content[:1000] + "..." if len(content) > 1000 else content)
            else:
                st.text(content[:1000] + "..." if len(content) > 1000 else content)
    
    # File sizes
    st.write("**File Sizes:**")
    for filename, content in files.items():
        size_kb = len(content.encode('utf-8')) / 1024
        st.write(f"- {filename}: {size_kb:.1f} KB")

def create_download_buttons(files: Dict[str, str]):
    """Create download buttons for export files."""
    if not files:
        return
    
    st.subheader("⬇️ Download Files")
    
    if len(files) == 1:
        # Single file download
        filename, content = list(files.items())[0]
        st.download_button(
            label=f"📁 Download {filename}",
            data=content.encode('utf-8'),
            file_name=filename,
            mime='text/plain',
            use_container_width=True
        )
    else:
        # Multiple files - create ZIP
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for filename, content in files.items():
                zip_file.writestr(filename, content.encode('utf-8'))
        
        st.download_button(
            label=f"📦 Download All Files (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"pdf_parser_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime='application/zip',
            use_container_width=True
        )
        
        st.write("**Individual Downloads:**")
        cols = st.columns(min(len(files), 3))
        for i, (filename, content) in enumerate(files.items()):
            with cols[i % len(cols)]:
                st.download_button(
                    label=f"📄 {filename}",
                    data=content.encode('utf-8'),
                    file_name=filename,
                    mime='text/plain',
                    use_container_width=True,
                    key=f"download_{filename}"
                )

def display_analytics():
    """Display analytics about the parsed documents."""
    if not st.session_state.parsed_documents:
        return
    
    st.subheader("📈 Document Analytics")
    
    # Overall statistics
    total_elements = sum(len(doc.elements) for doc in st.session_state.parsed_documents)
    total_pages = sum(doc.metadata.get('page_count', 0) for doc in st.session_state.parsed_documents)
    total_size = sum(doc.metadata.get('file_size', 0) for doc in st.session_state.parsed_documents)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Documents", len(st.session_state.parsed_documents))
    
    with col2:
        st.metric("Total Elements", total_elements)
    
    with col3:
        st.metric("Total Pages", total_pages)
    
    with col4:
        st.metric("Total Size", f"{total_size / 1024:.1f} KB")
    
    # Element type distribution across all documents
    all_types = {}
    for doc in st.session_state.parsed_documents:
        for element in doc.elements:
            elem_type = element.element_type
            all_types[elem_type] = all_types.get(elem_type, 0) + 1
    
    if all_types:
        st.write("**Element Type Distribution:**")
        df = pd.DataFrame(list(all_types.items()), columns=['Element Type', 'Count'])
        st.bar_chart(df.set_index('Element Type'))
    
    # Verification progress if available
    verification_stats = []
    for i, doc in enumerate(st.session_state.parsed_documents):
        verification_interface = get_verification_interface(i)
        if verification_interface:
            summary = verification_interface.get_verification_summary()
            verification_stats.append({
                'Document': doc.metadata.get('filename', f'Document {i+1}'),
                'Total': summary['total_elements'],
                'Verified': summary['verified_elements'],
                'Corrections': summary['total_corrections'],
                'Accuracy (%)': summary['accuracy_percentage']
            })
    
    if verification_stats:
        st.write("**Verification Progress:**")
        st.dataframe(verification_stats, use_container_width=True)

def main():
    """Main export page function."""
    initialize_session_state()
    
    st.title("📊 Export Data")
    st.markdown("Export your parsed documents and verification data in various formats.")
    
    # Check if documents are loaded
    if not st.session_state.parsed_documents:
        st.warning("No documents loaded. Please go to the Parse page to upload PDFs.")
        
        if st.button("📄 Go to Parse Page"):
            st.switch_page("pages/1_📄_Parse.py")
        return
    
    # Display analytics
    display_analytics()
    
    st.divider()
    
    # Export options
    options = display_export_options()
    
    # Generate export files
    if any(options['formats'].values()) and options['documents']:
        selected_documents = [st.session_state.parsed_documents[i] for i in options['documents']]
        
        if st.button("🚀 Generate Export Files", type="primary", use_container_width=True):
            with st.spinner("Generating export files..."):
                files = create_export_files(selected_documents, options)
                
                if files:
                    st.success(f"✅ Generated {len(files)} export file(s)!")
                    
                    # Display preview
                    display_export_preview(files)
                    
                    st.divider()
                    
                    # Download buttons
                    create_download_buttons(files)
                else:
                    st.error("No files were generated. Please check your export options.")

if __name__ == "__main__":
    main()
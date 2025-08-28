"""
Export Handler Utilities

Provides utilities for handling various export formats and operations.
"""

import streamlit as st
import io
import json
import csv
from typing import Dict, Any, List, Optional
from datetime import datetime
import zipfile
import base64


class ExportHandler:
    """Handles various export operations for the application."""
    
    def __init__(self):
        """Initialize export handler."""
        self.supported_formats = ['json', 'csv', 'markdown', 'html', 'text']
    
    def create_download_link(
        self, 
        data: str, 
        filename: str, 
        link_text: str = "Download",
        mime_type: str = "text/plain"
    ) -> str:
        """Create a download link for data.
        
        Args:
            data: Data to download
            filename: Name of the file
            link_text: Text for the download link
            mime_type: MIME type of the file
            
        Returns:
            HTML download link
        """
        b64_data = base64.b64encode(data.encode()).decode()
        return f'''
        <a href="data:{mime_type};base64,{b64_data}" 
           download="{filename}"
           style="text-decoration: none; 
                  background-color: #4CAF50; 
                  color: white; 
                  padding: 8px 16px; 
                  border-radius: 4px;
                  display: inline-block;
                  margin: 4px;">
           📁 {link_text}
        </a>
        '''
    
    def create_zip_download(
        self, 
        files: Dict[str, str], 
        zip_filename: str = None
    ) -> bytes:
        """Create a ZIP file from multiple files.
        
        Args:
            files: Dictionary of filename -> content
            zip_filename: Name of the ZIP file
            
        Returns:
            ZIP file bytes
        """
        if zip_filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            zip_filename = f"export_{timestamp}.zip"
        
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for filename, content in files.items():
                zip_file.writestr(filename, content.encode('utf-8'))
        
        return zip_buffer.getvalue()
    
    def format_json(self, data: Any, indent: int = 2) -> str:
        """Format data as JSON.
        
        Args:
            data: Data to format
            indent: JSON indentation
            
        Returns:
            JSON formatted string
        """
        return json.dumps(data, indent=indent, ensure_ascii=False, default=str)
    
    def format_csv(self, data: List[Dict[str, Any]]) -> str:
        """Format data as CSV.
        
        Args:
            data: List of dictionaries to format
            
        Returns:
            CSV formatted string
        """
        if not data:
            return ""
        
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        
        return output.getvalue()
    
    def format_markdown(self, data: Dict[str, Any]) -> str:
        """Format data as Markdown.
        
        Args:
            data: Data to format
            
        Returns:
            Markdown formatted string
        """
        md_lines = []
        
        # Title
        title = data.get('title', 'Export Report')
        md_lines.append(f"# {title}")
        md_lines.append("")
        
        # Metadata
        if 'metadata' in data:
            md_lines.append("## Metadata")
            for key, value in data['metadata'].items():
                md_lines.append(f"- **{key.title()}:** {value}")
            md_lines.append("")
        
        # Content sections
        if 'sections' in data:
            for section in data['sections']:
                md_lines.append(f"## {section['title']}")
                if 'description' in section:
                    md_lines.append(section['description'])
                md_lines.append("")
                
                if 'items' in section:
                    for item in section['items']:
                        if isinstance(item, dict):
                            md_lines.append(f"### {item.get('title', 'Item')}")
                            md_lines.append(item.get('content', ''))
                        else:
                            md_lines.append(f"- {item}")
                    md_lines.append("")
        
        return "\n".join(md_lines)
    
    def format_html(self, data: Dict[str, Any]) -> str:
        """Format data as HTML.
        
        Args:
            data: Data to format
            
        Returns:
            HTML formatted string
        """
        title = data.get('title', 'Export Report')
        
        html_lines = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "    <meta charset='UTF-8'>",
            "    <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            f"    <title>{title}</title>",
            "    <style>",
            "        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }",
            "        h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }",
            "        h2 { color: #666; border-left: 4px solid #4CAF50; padding-left: 10px; }",
            "        .metadata { background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0; }",
            "        .section { margin: 30px 0; }",
            "        .item { margin: 15px 0; padding: 10px; border: 1px solid #ddd; border-radius: 3px; }",
            "        table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
            "        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "        th { background-color: #f2f2f2; }",
            "    </style>",
            "</head>",
            "<body>",
            f"    <h1>{title}</h1>"
        ]
        
        # Metadata
        if 'metadata' in data:
            html_lines.append("    <div class='metadata'>")
            html_lines.append("        <h2>Metadata</h2>")
            for key, value in data['metadata'].items():
                html_lines.append(f"        <p><strong>{key.title()}:</strong> {value}</p>")
            html_lines.append("    </div>")
        
        # Content sections
        if 'sections' in data:
            for section in data['sections']:
                html_lines.append("    <div class='section'>")
                html_lines.append(f"        <h2>{section['title']}</h2>")
                
                if 'description' in section:
                    html_lines.append(f"        <p>{section['description']}</p>")
                
                if 'items' in section:
                    for item in section['items']:
                        html_lines.append("        <div class='item'>")
                        if isinstance(item, dict):
                            html_lines.append(f"            <h3>{item.get('title', 'Item')}</h3>")
                            html_lines.append(f"            <p>{item.get('content', '')}</p>")
                        else:
                            html_lines.append(f"            <p>{item}</p>")
                        html_lines.append("        </div>")
                
                html_lines.append("    </div>")
        
        html_lines.extend([
            "</body>",
            "</html>"
        ])
        
        return "\n".join(html_lines)
    
    def display_download_section(
        self, 
        files: Dict[str, str], 
        title: str = "Download Files"
    ):
        """Display download section with buttons for all files.
        
        Args:
            files: Dictionary of filename -> content
            title: Section title
        """
        if not files:
            return
        
        st.subheader(title)
        
        # Single file download
        if len(files) == 1:
            filename, content = list(files.items())[0]
            st.download_button(
                label=f"📁 Download {filename}",
                data=content.encode('utf-8'),
                file_name=filename,
                mime='text/plain',
                use_container_width=True
            )
        else:
            # Multiple files - create ZIP option
            zip_data = self.create_zip_download(files)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            st.download_button(
                label=f"📦 Download All Files (ZIP)",
                data=zip_data,
                file_name=f"export_{timestamp}.zip",
                mime='application/zip',
                use_container_width=True
            )
            
            # Individual downloads
            st.write("**Individual Downloads:**")
            cols = st.columns(min(len(files), 3))
            
            for i, (filename, content) in enumerate(files.items()):
                with cols[i % len(cols)]:
                    # Determine MIME type
                    if filename.endswith('.json'):
                        mime_type = 'application/json'
                    elif filename.endswith('.csv'):
                        mime_type = 'text/csv'
                    elif filename.endswith('.html'):
                        mime_type = 'text/html'
                    elif filename.endswith('.md'):
                        mime_type = 'text/markdown'
                    else:
                        mime_type = 'text/plain'
                    
                    st.download_button(
                        label=f"📄 {filename}",
                        data=content.encode('utf-8'),
                        file_name=filename,
                        mime=mime_type,
                        use_container_width=True,
                        key=f"download_{filename}_{i}"
                    )
    
    def display_file_previews(
        self, 
        files: Dict[str, str], 
        title: str = "File Previews",
        max_preview_length: int = 1000
    ):
        """Display previews of export files.
        
        Args:
            files: Dictionary of filename -> content
            title: Section title
            max_preview_length: Maximum length for preview
        """
        if not files:
            return
        
        st.subheader(title)
        
        # Create tabs for each file
        if len(files) > 1:
            tabs = st.tabs(list(files.keys()))
            
            for i, (filename, content) in enumerate(files.items()):
                with tabs[i]:
                    self._display_single_preview(filename, content, max_preview_length)
        else:
            filename, content = list(files.items())[0]
            self._display_single_preview(filename, content, max_preview_length)
    
    def _display_single_preview(
        self, 
        filename: str, 
        content: str, 
        max_length: int = 1000
    ):
        """Display preview of a single file.
        
        Args:
            filename: Name of the file
            content: File content
            max_length: Maximum length to display
        """
        # File info
        size_kb = len(content.encode('utf-8')) / 1024
        st.write(f"**File:** {filename} ({size_kb:.1f} KB)")
        
        # Preview based on file type
        if filename.endswith('.json'):
            try:
                json_data = json.loads(content)
                st.json(json_data)
            except json.JSONDecodeError:
                st.text(content[:max_length] + "..." if len(content) > max_length else content)
        
        elif filename.endswith('.csv'):
            try:
                # Try to display as dataframe
                import pandas as pd
                df = pd.read_csv(io.StringIO(content))
                st.dataframe(df, use_container_width=True)
            except Exception:
                st.text(content[:max_length] + "..." if len(content) > max_length else content)
        
        elif filename.endswith('.html'):
            # Show HTML rendered and as code
            with st.expander("Rendered HTML", expanded=False):
                st.components.v1.html(content, height=400, scrolling=True)
            
            with st.expander("HTML Source", expanded=True):
                st.code(content[:max_length] + "..." if len(content) > max_length else content, language='html')
        
        elif filename.endswith('.md'):
            # Show markdown rendered and as source
            with st.expander("Rendered Markdown", expanded=True):
                st.markdown(content[:max_length] + "..." if len(content) > max_length else content)
            
            with st.expander("Markdown Source", expanded=False):
                st.code(content[:max_length] + "..." if len(content) > max_length else content, language='markdown')
        
        else:
            # Plain text
            st.text(content[:max_length] + "..." if len(content) > max_length else content)
    
    def get_export_stats(self, files: Dict[str, str]) -> Dict[str, Any]:
        """Get statistics about export files.
        
        Args:
            files: Dictionary of filename -> content
            
        Returns:
            Statistics dictionary
        """
        if not files:
            return {}
        
        total_size = sum(len(content.encode('utf-8')) for content in files.values())
        
        stats = {
            'file_count': len(files),
            'total_size_bytes': total_size,
            'total_size_kb': total_size / 1024,
            'formats': list(set(f.split('.')[-1] for f in files.keys())),
            'largest_file': max(files.items(), key=lambda x: len(x[1]))[0],
            'smallest_file': min(files.items(), key=lambda x: len(x[1]))[0],
            'average_size_kb': (total_size / len(files)) / 1024
        }
        
        return stats


# Global export handler instance
export_handler = ExportHandler()
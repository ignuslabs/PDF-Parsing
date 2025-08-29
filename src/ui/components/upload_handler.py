"""
PDF Upload Handler Component

Provides a reusable component for handling PDF file uploads with validation
and progress tracking.
"""

import streamlit as st
from typing import List, Tuple, Optional
from pathlib import Path
import tempfile
import os


class PDFUploadHandler:
    """Handles PDF file uploads with validation and temporary file management."""
    
    def __init__(self, max_size_mb: int = 100, max_files: int = 10):
        """Initialize upload handler.
        
        Args:
            max_size_mb: Maximum file size in megabytes
            max_files: Maximum number of files allowed
        """
        self.max_size_mb = max_size_mb
        self.max_files = max_files
        self.max_size_bytes = max_size_mb * 1024 * 1024
    
    def validate_file(self, uploaded_file) -> Tuple[bool, str]:
        """Validate a single uploaded PDF file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Tuple of (is_valid, message)
        """
        # Check file extension
        if not uploaded_file.name.lower().endswith('.pdf'):
            return False, "File must be a PDF (.pdf extension)"
        
        # Check file size
        if uploaded_file.size == 0:
            return False, "File is empty"
        
        if uploaded_file.size > self.max_size_bytes:
            size_mb = uploaded_file.size / (1024 * 1024)
            return False, f"File too large ({size_mb:.1f}MB). Maximum size: {self.max_size_mb}MB"
        
        # Check PDF header
        try:
            uploaded_file.seek(0)
            header = uploaded_file.read(4)
            uploaded_file.seek(0)
            
            if header != b'%PDF':
                return False, "Invalid PDF file format"
        except Exception as e:
            return False, f"Error reading file: {str(e)}"
        
        return True, "Valid PDF file"
    
    def display_upload_widget(
        self, 
        label: str = "Upload PDF files",
        help_text: str = None,
        accept_multiple: bool = True,
        key: str = None
    ) -> List:
        """Display file upload widget with validation.
        
        Args:
            label: Label for the upload widget
            help_text: Help text to display
            accept_multiple: Whether to accept multiple files
            key: Unique key for the widget
            
        Returns:
            List of valid uploaded files
        """
        if help_text is None:
            help_text = f"Upload PDF files (max {self.max_size_mb}MB each, up to {self.max_files} files)"
        
        # File uploader
        uploaded_files = st.file_uploader(
            label,
            type=['pdf'],
            accept_multiple_files=accept_multiple,
            help=help_text,
            key=key
        )
        
        if not uploaded_files:
            return []
        
        # Ensure it's a list
        if not isinstance(uploaded_files, list):
            uploaded_files = [uploaded_files]
        
        # Check file count limit
        if len(uploaded_files) > self.max_files:
            st.error(f"Too many files ({len(uploaded_files)}). Maximum allowed: {self.max_files}")
            return []
        
        # Validate each file
        valid_files = []
        validation_results = []
        
        for uploaded_file in uploaded_files:
            is_valid, message = self.validate_file(uploaded_file)
            validation_results.append((uploaded_file, is_valid, message))
            
            if is_valid:
                valid_files.append(uploaded_file)
        
        # Display validation results
        self.display_validation_results(validation_results)
        
        return valid_files
    
    def display_validation_results(self, validation_results: List[Tuple]):
        """Display validation results for uploaded files.
        
        Args:
            validation_results: List of (file, is_valid, message) tuples
        """
        if not validation_results:
            return
        
        st.subheader("📋 File Validation")
        
        for uploaded_file, is_valid, message in validation_results:
            if is_valid:
                st.success(f"✅ {uploaded_file.name}: {message}")
            else:
                st.error(f"❌ {uploaded_file.name}: {message}")
    
    def create_temp_file(self, uploaded_file) -> Path:
        """Create a temporary file from uploaded file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Path to temporary file
        """
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_file.write(uploaded_file.read())
        temp_file.close()
        
        return Path(temp_file.name)
    
    def cleanup_temp_file(self, temp_path: Path):
        """Clean up temporary file.
        
        Args:
            temp_path: Path to temporary file
        """
        try:
            if temp_path.exists():
                os.unlink(temp_path)
        except Exception as e:
            st.warning(f"Failed to clean up temporary file {temp_path}: {e}")
    
    def display_file_info(self, uploaded_files: List) -> None:
        """Display information about uploaded files.
        
        Args:
            uploaded_files: List of uploaded file objects
        """
        if not uploaded_files:
            return
        
        st.subheader("📁 File Information")
        
        total_size = sum(f.size for f in uploaded_files)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Files", len(uploaded_files))
        
        with col2:
            st.metric("Total Size", f"{total_size / (1024*1024):.1f} MB")
        
        with col3:
            avg_size = total_size / len(uploaded_files) if uploaded_files else 0
            st.metric("Avg Size", f"{avg_size / (1024*1024):.1f} MB")
        
        # File details
        for i, uploaded_file in enumerate(uploaded_files):
            with st.expander(f"📄 {uploaded_file.name}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Name:** {uploaded_file.name}")
                    st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
                
                with col2:
                    st.write(f"**Type:** {uploaded_file.type}")
                    # You could add more file metadata here
    
    def display_upload_summary(self, valid_files: List, total_files: int) -> bool:
        """Display upload summary and return whether to proceed.
        
        Args:
            valid_files: List of valid files
            total_files: Total number of files uploaded
            
        Returns:
            Whether all files are valid and ready to process
        """
        if not total_files:
            return False
        
        valid_count = len(valid_files)
        invalid_count = total_files - valid_count
        
        if invalid_count == 0:
            st.success(f"🎉 All {valid_count} files are valid and ready to process!")
            return True
        elif valid_count == 0:
            st.error("❌ No valid files found. Please check your files and try again.")
            return False
        else:
            st.warning(f"⚠️ {valid_count} files are valid, {invalid_count} files have issues.")
            
            proceed = st.checkbox(
                f"Proceed with {valid_count} valid files?",
                help="Process only the valid files, ignoring invalid ones"
            )
            
            return proceed
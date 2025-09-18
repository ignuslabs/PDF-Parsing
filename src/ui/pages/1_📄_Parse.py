"""
Parse Page - PDF Upload and Document Parsing

Handles PDF file upload, parser configuration, and document parsing with progress tracking.
"""

# CRITICAL: Setup Python path FIRST, before any project imports
import sys
from pathlib import Path

# Add the src directory to Python path BEFORE importing project modules
src_path = Path(__file__).resolve().parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Standard library imports
import json
import re
from datetime import datetime
import streamlit as st
import tempfile
import os
import time
import logging
from typing import Optional

# Project imports (now that path is set)
try:
    from src.core.parser import DoclingParser
    from src.core.models import ParsedDocument
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.error("Please ensure the application is running from the correct directory")
    st.stop()

st.set_page_config(page_title="Parse Documents", page_icon="📄", layout="wide")

logger = logging.getLogger(__name__)


def initialize_session_state():
    """Initialize session state for the parse page."""
    if "parsed_documents" not in st.session_state:
        st.session_state.parsed_documents = []

    if "parser_config" not in st.session_state:
        st.session_state.parser_config = {
            "enable_ocr": False,
            "enable_tables": True,
            "generate_page_images": True,
            "ocr_engine": "tesseract",
            "ocr_language": "eng",
            "table_mode": "accurate",
            "image_scale": 1.0,
            "enable_kv_extraction": True,
            "header_classifier_enabled": True,
            "enable_form_fields": True,
            "export_raw_snapshot": False,
        }


def display_parser_configuration():
    """Display parser configuration options."""
    st.subheader("⚙️ Parser Configuration")

    with st.expander("Basic Settings", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.session_state.parser_config["enable_tables"] = st.checkbox(
                "Enable table extraction",
                value=st.session_state.parser_config["enable_tables"],
                help="Extract tables from documents",
            )

            st.session_state.parser_config["generate_page_images"] = st.checkbox(
                "Generate page images",
                value=st.session_state.parser_config["generate_page_images"],
                help="Required for verification functionality",
            )

        with col2:
            st.session_state.parser_config["enable_ocr"] = st.checkbox(
                "Enable OCR",
                value=st.session_state.parser_config["enable_ocr"],
                help="Extract text from scanned documents (slower)",
            )

            if st.session_state.parser_config["enable_ocr"]:
                st.session_state.parser_config["ocr_engine"] = st.selectbox(
                    "OCR Engine",
                    options=["tesseract", "easyocr"],
                    index=0 if st.session_state.parser_config["ocr_engine"] == "tesseract" else 1,
                    help="Choose OCR engine",
                )

    with st.expander("Advanced Settings"):
        col1, col2 = st.columns(2)

        with col1:
            st.session_state.parser_config["table_mode"] = st.selectbox(
                "Table extraction mode",
                options=["accurate", "fast"],
                index=0 if st.session_state.parser_config["table_mode"] == "accurate" else 1,
                help="Accurate mode is slower but more precise",
            )

        with col2:
            st.session_state.parser_config["image_scale"] = st.slider(
                "Image scale factor",
                min_value=0.5,
                max_value=2.0,
                value=st.session_state.parser_config["image_scale"],
                step=0.1,
                help="Higher values = better quality, larger file sizes",
            )

        if st.session_state.parser_config["enable_ocr"]:
            st.session_state.parser_config["ocr_language"] = st.text_input(
                "OCR Language(s)",
                value=st.session_state.parser_config["ocr_language"],
                help="Language codes (e.g., 'eng', 'fra', 'deu' for Tesseract or 'en', 'fr', 'de' for EasyOCR). Multiple languages: 'eng+fra'. Auto-converted between engines.",
            )

    with st.expander("🧠 Smart Extraction", expanded=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.session_state.parser_config["enable_kv_extraction"] = st.checkbox(
                "Enable Key-Value Extraction",
                value=st.session_state.parser_config.get("enable_kv_extraction", True),
                help="Extract field/value pairs (invoice numbers, claim IDs, totals) using the tuned detector",
            )

        with col2:
            st.session_state.parser_config["header_classifier_enabled"] = st.checkbox(
                "Use Safer Header Classifier",
                value=st.session_state.parser_config.get("header_classifier_enabled", True),
                help="Prevent codes and field values from being misclassified as headings",
            )

        with col3:
            st.session_state.parser_config["enable_form_fields"] = st.checkbox(
                "Extract PDF Form Fields",
                value=st.session_state.parser_config.get("enable_form_fields", True),
                help="Extract values from PDF form fields (AcroForm) that may not be captured by regular text extraction",
            )

        with col4:
            st.session_state.parser_config["export_raw_snapshot"] = st.checkbox(
                "Export Raw Snapshot",
                value=st.session_state.parser_config.get("export_raw_snapshot", False),
                help="Save raw extraction JSON to pdf_testing for inspection",
            )

        if st.session_state.parser_config.get("enable_kv_extraction", False):
            st.info(
                "💡 Key-Value extraction now spotlights invoice numbers, claim IDs, order codes, and other structured identifiers."
            )

        if st.session_state.parser_config.get("header_classifier_enabled", False):
            st.info(
                "💡 The safer header classifier keeps verification focused on real headings versus field values."
            )

        if st.session_state.parser_config.get("enable_form_fields", False):
            st.info(
                "💡 Form field extraction captures values stored in PDF form fields (AcroForm) that might not be visible through regular text extraction, such as filled-in values in government forms."
            )

        if st.session_state.parser_config.get("export_raw_snapshot", False):
            st.info(
                "💡 Raw snapshots are saved in the `pdf_testing` folder with timestamped filenames for auditing."
            )


def validate_pdf_file(uploaded_file) -> tuple[bool, str]:
    """Validate uploaded PDF file."""
    if not uploaded_file.name.lower().endswith(".pdf"):
        return False, "File must be a PDF"

    if uploaded_file.size == 0:
        return False, "File is empty"

    if uploaded_file.size > 100 * 1024 * 1024:  # 100MB limit
        return False, f"File too large ({uploaded_file.size / (1024*1024):.1f}MB). Maximum 100MB."

    # Check PDF header
    uploaded_file.seek(0)
    header = uploaded_file.read(4)
    uploaded_file.seek(0)

    if header != b"%PDF":
        return False, "Invalid PDF file format"

    return True, "Valid PDF file"


def create_parser_from_config():
    """Create DoclingParser instance from current configuration."""
    config = st.session_state.parser_config

    # Convert table_mode string to TableFormerMode enum
    from docling.datamodel.pipeline_options import TableFormerMode

    table_mode = (
        TableFormerMode.ACCURATE if config["table_mode"] == "accurate" else TableFormerMode.FAST
    )

    # Handle OCR language - convert to list if multiple languages
    ocr_lang = config["ocr_language"]
    if "+" in ocr_lang:
        ocr_lang = ocr_lang.split("+")

    return DoclingParser(
        enable_ocr=config["enable_ocr"],
        enable_tables=config["enable_tables"],
        generate_page_images=config["generate_page_images"],
        ocr_engine=config["ocr_engine"],
        ocr_lang=ocr_lang,
        table_mode=table_mode,
        image_scale=config["image_scale"],
        enable_kv_extraction=config.get("enable_kv_extraction", False),
        header_classifier_enabled=config.get("header_classifier_enabled", False),
        enable_form_fields=config.get("enable_form_fields", True),
    )


def _ordinal_suffix(day: int) -> str:
    if 10 <= day % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


def export_raw_snapshot(parsed_doc: ParsedDocument, original_filename: str) -> Optional[Path]:
    """Export parsed document data to pdf_testing as JSON."""
    try:
        base_dir = Path("pdf_testing")
        base_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now()
        month_part = now.strftime("%b")
        day = now.day
        date_part = f"{month_part}{day}{_ordinal_suffix(day)}"
        time_part = now.strftime("%H_%M")

        stem = Path(original_filename).stem or "document"
        safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "document"

        filename = f"{safe_stem}_{date_part}_{time_part}_extract.json"
        output_path = base_dir / filename

        data = parsed_doc.export_to_dict()

        pages = data.get("pages")
        if isinstance(pages, dict):
            sanitized_pages = {}
            for key, value in pages.items():
                page_info: Dict[str, Any] = {
                    "repr": str(type(value)),
                }
                if hasattr(value, "size"):
                    page_info["size"] = getattr(value, "size")
                if hasattr(value, "mode"):
                    page_info["mode"] = getattr(value, "mode")
                sanitized_pages[key] = page_info
            data["pages"] = sanitized_pages

        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

        logger.info("Saved raw snapshot to %s", output_path)
        return output_path
    except Exception as exc:
        logger.warning("Failed to save raw snapshot for %s: %s", original_filename, exc)
        return None


class StreamlitLogHandler(logging.Handler):
    """Custom log handler that captures log messages for later display."""
    
    def __init__(self):
        super().__init__()
        self.messages = []
        
    def emit(self, record):
        try:
            msg = self.format(record)
            self.messages.append((record.levelname, msg))
        except Exception:
            pass  # Silently ignore errors in log handler
    
    def display_messages(self, container):
        """Display all captured messages in the given container."""
        with container:
            for level, message in self.messages:
                if "🚀 Starting" in message or "🎉 Completed" in message:
                    st.success(message)
                elif "⏳ Step" in message:
                    st.info(message)
                elif "✅ Completed:" in message:
                    st.success(message)
                elif "📊 ETA" in message or "💾 Memory" in message:
                    st.text(message)
                elif level == "ERROR" or "❌" in message:
                    st.error(message)
                elif level == "WARNING":
                    st.warning(message)
                else:
                    st.text(message)


def parse_document_with_progress(pdf_path: Path, parser: DoclingParser, log_container) -> Optional[ParsedDocument]:
    """Parse a PDF document with real-time progress updates."""
    try:
        # Create and add custom log handler
        streamlit_handler = StreamlitLogHandler()
        streamlit_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        streamlit_handler.setFormatter(formatter)
        
        # Get the parser's logger and add our handler
        parser_logger = logging.getLogger('src.core.parser')
        parser_logger.addHandler(streamlit_handler)
        
        try:
            # Use the full parsing method to get metadata
            parsed_doc = parser.parse_document_full(pdf_path)
            
            # Display all captured log messages
            streamlit_handler.display_messages(log_container)
            
            return parsed_doc
        finally:
            # Remove the handler to avoid accumulating handlers
            parser_logger.removeHandler(streamlit_handler)
            
    except Exception as e:
        st.error(f"Failed to parse document: {str(e)}")
        return None


def parse_document(pdf_path: Path, parser: DoclingParser) -> Optional[ParsedDocument]:
    """Parse a PDF document and return the result."""
    try:
        # Use the full parsing method to get metadata
        parsed_doc = parser.parse_document_full(pdf_path)
        return parsed_doc
    except Exception as e:
        st.error(f"Failed to parse document: {str(e)}")
        return None


def display_parsing_results(parsed_doc: ParsedDocument):
    """Display parsing results summary."""
    st.subheader("✅ Parsing Results")

    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Elements", len(parsed_doc.elements))

    with col2:
        st.metric("Pages", parsed_doc.metadata.get("page_count", "Unknown"))

    with col3:
        page_images = len(parsed_doc.pages) if parsed_doc.pages else 0
        st.metric("Page Images", page_images)

    with col4:
        avg_confidence = (
            sum(e.confidence for e in parsed_doc.elements) / len(parsed_doc.elements)
            if parsed_doc.elements
            else 0
        )
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")

    # Element type breakdown
    st.subheader("📊 Element Types")
    element_types = {}
    for element in parsed_doc.elements:
        elem_type = element.element_type
        element_types[elem_type] = element_types.get(elem_type, 0) + 1

    if element_types:
        # Display as bar chart
        import pandas as pd

        df = pd.DataFrame(list(element_types.items()), columns=["Type", "Count"])
        st.bar_chart(df.set_index("Type"))

        # Display as columns
        type_cols = st.columns(min(len(element_types), 6))
        for i, (elem_type, count) in enumerate(element_types.items()):
            with type_cols[i % len(type_cols)]:
                st.metric(elem_type.title(), count)

    # Sample elements
    st.subheader("📋 Sample Elements")

    sample_size = min(5, len(parsed_doc.elements))
    for i, element in enumerate(parsed_doc.elements[:sample_size]):
        with st.expander(f"{element.element_type.title()} - Page {element.page_number}"):
            col1, col2 = st.columns([3, 1])

            with col1:
                text_preview = (
                    element.text[:200] + "..." if len(element.text) > 200 else element.text
                )
                st.write(f"**Text:** {text_preview}")

                if element.bbox:
                    st.write(
                        f"**Position:** ({element.bbox['x0']:.1f}, {element.bbox['y0']:.1f}) to ({element.bbox['x1']:.1f}, {element.bbox['y1']:.1f})"
                    )

            with col2:
                st.write(f"**Type:** {element.element_type}")
                st.write(f"**Confidence:** {element.confidence:.2f}")
                st.write(f"**Page:** {element.page_number}")


def main():
    """Main parse page function."""
    initialize_session_state()

    st.title("📄 Parse Documents")
    st.markdown("Upload and parse PDF documents with advanced extraction capabilities.")

    # Configuration section
    display_parser_configuration()

    st.divider()

    # File upload section
    st.subheader("📁 Upload PDF Files")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to parse",
    )

    if uploaded_files:
        # Validate all files first
        valid_files = []
        for uploaded_file in uploaded_files:
            is_valid, message = validate_pdf_file(uploaded_file)
            if is_valid:
                valid_files.append(uploaded_file)
                st.success(f"✅ {uploaded_file.name}: {message}")
            else:
                st.error(f"❌ {uploaded_file.name}: {message}")

        if valid_files:
            # Parse button
            if st.button("🚀 Parse Documents", type="primary", use_container_width=True):
                # Create parser
                try:
                    parser = create_parser_from_config()
                except Exception as e:
                    st.error(f"Failed to create parser: {str(e)}")
                    return

                logger.info("Using parser configuration from UI:\n%s", parser.describe_configuration())

                # Progress tracking containers
                progress_bar = st.progress(0, text="Initializing...")
                status_text = st.empty()
                
                # Create detailed progress log container
                st.subheader("🔍 Detailed Progress")
                log_container = st.container()
                
                results_container = st.container()

                parsed_docs = []

                for i, uploaded_file in enumerate(valid_files):
                    file_size_bytes = getattr(uploaded_file, "size", None)
                    file_size_kb = (file_size_bytes / 1024) if file_size_bytes else 0.0
                    logger.info(
                        "Starting parse for '%s' (%0.1f KB) [%d/%d]",
                        uploaded_file.name,
                        file_size_kb,
                        i + 1,
                        len(valid_files),
                    )

                    # Update main progress bar
                    file_progress = i / len(valid_files)
                    status_text.text(f"Parsing {uploaded_file.name} ({i+1}/{len(valid_files)})")
                    progress_bar.progress(file_progress, text=f"Processing {uploaded_file.name}...")

                    # Clear log container for this file
                    with log_container:
                        st.markdown(f"**Processing:** {uploaded_file.name}")
                        
                    # Create a container for this file's progress logs
                    file_log_container = log_container.container()

                    # Save uploaded file temporarily
                    try:
                        uploaded_file.seek(0)
                    except Exception:
                        pass
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = Path(tmp_file.name)

                    try:
                        # Parse document with detailed progress
                        start_time = time.time()
                        parsed_doc = parse_document_with_progress(tmp_path, parser, file_log_container)
                        parsing_time = time.time() - start_time

                        if parsed_doc:
                            parsed_doc.metadata["original_source_path"] = parsed_doc.metadata.get("source_path")
                            parsed_doc.metadata["source_path"] = uploaded_file.name
                            parsed_doc.metadata["filename"] = uploaded_file.name
                            # Add timing info to metadata
                            parsed_doc.metadata["parsing_time_seconds"] = parsing_time
                            parsed_docs.append(parsed_doc)

                            export_path = None
                            if st.session_state.parser_config.get("export_raw_snapshot", False):
                                export_path = export_raw_snapshot(parsed_doc, uploaded_file.name)

                            with file_log_container:
                                st.success(f"🎉 Successfully parsed {uploaded_file.name} in {parsing_time:.1f}s")
                                if export_path:
                                    st.info(f"Raw snapshot saved to `{export_path}`")
                                elif st.session_state.parser_config.get("export_raw_snapshot", False):
                                    st.warning("Raw snapshot export failed; check logs for details")
                                
                            status_text.success(
                                f"✅ Parsed {uploaded_file.name} in {parsing_time:.1f}s"
                            )
                        else:
                            with file_log_container:
                                st.error(f"❌ Failed to parse {uploaded_file.name}")
                            status_text.error(f"❌ Failed to parse {uploaded_file.name}")

                    finally:
                        # Clean up temporary file
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass

                # Update progress bar
                progress_bar.progress(1.0, text="Parsing complete!")

                if parsed_docs:
                    # Add to session state
                    st.session_state.parsed_documents.extend(parsed_docs)

                    # Display results
                    with results_container:
                        st.success(f"🎉 Successfully parsed {len(parsed_docs)} document(s)!")

                        # Show results for last parsed document
                        if parsed_docs:
                            display_parsing_results(parsed_docs[-1])

                        # Navigation buttons
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            if st.button("🔍 Search Documents", use_container_width=True):
                                st.switch_page("pages/2_🔍_Search.py")

                        with col2:
                            if st.button("✅ Verify Results", use_container_width=True):
                                st.switch_page("pages/3_✅_Verify.py")

                        with col3:
                            if st.button("📊 Export Data", use_container_width=True):
                                st.switch_page("pages/4_📊_Export.py")
                else:
                    st.error("No documents were successfully parsed.")

    # Show existing documents if any
    if st.session_state.parsed_documents:
        st.divider()
        st.subheader("📚 Loaded Documents")

        for i, doc in enumerate(st.session_state.parsed_documents):
            with st.expander(f"Document {i+1}: {doc.metadata.get('filename', 'Unknown')}"):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.write(f"**Elements:** {len(doc.elements)}")
                    st.write(f"**Pages:** {doc.metadata.get('page_count', 'Unknown')}")
                    st.write(f"**Size:** {doc.metadata.get('file_size', 0) / 1024:.1f} KB")

                with col2:
                    if st.button(f"Remove", key=f"remove_{i}", type="secondary"):
                        st.session_state.parsed_documents.pop(i)
                        st.rerun()


if __name__ == "__main__":
    main()

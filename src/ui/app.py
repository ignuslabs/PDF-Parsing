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
    page_title="Dashboard", page_icon="📊", layout="wide", initial_sidebar_state="expanded"
)


def initialize_session_state():
    """Initialize Streamlit session state with default values."""

    # Document storage
    if "parsed_documents" not in st.session_state:
        st.session_state.parsed_documents = []

    if "current_doc_index" not in st.session_state:
        st.session_state.current_doc_index = 0

    if "current_page" not in st.session_state:
        st.session_state.current_page = 1

    # Element selection and verification
    if "selected_element_id" not in st.session_state:
        st.session_state.selected_element_id = None

    if "verification_states" not in st.session_state:
        st.session_state.verification_states = {}

    # Search functionality
    if "search_results" not in st.session_state:
        st.session_state.search_results = []

    if "search_query" not in st.session_state:
        st.session_state.search_query = ""

    if "search_filters" not in st.session_state:
        st.session_state.search_filters = {
            "element_types": [],
            "page_range": None,
            "confidence_threshold": 0.0,
        }

    # Parser configuration
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

    # UI state
    if "show_debug_info" not in st.session_state:
        st.session_state.show_debug_info = False

    if "sidebar_expanded" not in st.session_state:
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


def display_topbar():
    """Display top navigation and document controls."""
    docs = st.session_state.parsed_documents

    top = st.container()
    with top:
        header_cols = st.columns([3, 1, 1, 1])

        with header_cols[0]:
            st.subheader("Documents")
            if docs:
                doc_names = [doc.metadata.get("filename", f"Document {i+1}") for i, doc in enumerate(docs)]
                selected_doc = st.selectbox(
                    "Select Document",
                    options=range(len(doc_names)),
                    format_func=lambda i: doc_names[i],
                    index=st.session_state.current_doc_index,
                    key="_doc_selector",
                )
                if selected_doc != st.session_state.current_doc_index:
                    st.session_state.current_doc_index = selected_doc
                    st.session_state.current_page = 1
                    st.session_state.selected_element_id = None
                    st.rerun()

                current_doc = get_current_document()
                if current_doc:
                    size_kb = current_doc.metadata.get("file_size")
                    info_lines = [
                        f"**Elements:** {len(current_doc.elements)}",
                        f"**Pages:** {current_doc.metadata.get('page_count', 'Unknown')}"
                    ]
                    if size_kb is not None:
                        info_lines.append(f"**Size:** {size_kb / 1024:.1f} KB")
                    kv_count = len(getattr(current_doc, "key_values", []) or [])
                    if kv_count:
                        info_lines.append(f"**Key-Value pairs:** {kv_count}")
                    st.caption("\n".join(info_lines))
            else:
                st.info("No documents loaded. Go to the Parse page to upload PDFs.")

        with header_cols[1]:
            st.markdown("##### Quick Actions")
            if st.button("📄 Parse", use_container_width=True):
                st.switch_page("pages/1_📄_Parse.py")

        with header_cols[2]:
            st.markdown("##### ")  # spacing
            if st.button("🔍 Search & Verify", use_container_width=True):
                st.switch_page("pages/3_✅_Verify.py")

        with header_cols[3]:
            st.markdown("##### ")
            if st.button("📊 Export", use_container_width=True):
                st.switch_page("pages/4_📊_Export.py")

        st.divider()

        with st.expander("⚙️ Settings"):
            st.session_state.show_debug_info = st.checkbox(
                "Show debug information", value=st.session_state.show_debug_info
            )

            if st.button("Clear all data", type="secondary"):
                for key in ["parsed_documents", "search_results", "verification_states"]:
                    if key in st.session_state:
                        del st.session_state[key]
                initialize_session_state()
                st.success("All data cleared!")
                st.rerun()


def show_dashboard():
    """Render the main dashboard overview."""
    st.title("Welcome to Smart PDF Parser")
    docs = st.session_state.parsed_documents
    doc_count = len(docs)
    element_total = sum(len(doc.elements) for doc in docs)
    kv_total = sum(len(getattr(doc, "key_values", []) or []) for doc in docs)
    docling_kv_total = sum(
        len([kv for kv in getattr(doc, "key_values", []) or [] if kv.metadata.get("source") == "docling_core"])
        for doc in docs
    )
    heuristic_kv_total = sum(
        len([kv for kv in getattr(doc, "key_values", []) or [] if kv.metadata.get("source") == "heuristic"])
        for doc in docs
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Documents", doc_count)
    with col2:
        st.metric("Total Elements", element_total)
    with col3:
        st.metric("Key-Value Pairs", kv_total)

    if kv_total:
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            st.caption(f"Docling derived: {docling_kv_total}")
        with sub_col2:
            st.caption(f"Inline/heuristic: {heuristic_kv_total}")

    st.markdown(
        """
**Next steps**
- Parse new documents on the Parse page
- Use Search & Verify to locate headers and matching values
- Export validated data once review is complete
"""
    )

    if docs:
        st.subheader("Recent documents")
        table_data = []
        for doc in docs[-10:][::-1]:
            meta = doc.metadata
            kv_values = getattr(doc, "key_values", []) or []
            docling_count = len([kv for kv in kv_values if kv.metadata.get("source") == "docling_core"])
            heuristic_count = len([kv for kv in kv_values if kv.metadata.get("source") == "heuristic"])
            table_data.append(
                {
                    "Document": meta.get("filename", "Unknown"),
                    "Pages": meta.get("page_count", "Unknown"),
                    "Elements": len(doc.elements),
                    "Docling KV": docling_count,
                    "Inline KV": heuristic_count,
                    "Parsed": (meta.get("parsed_at", "Unknown") or "Unknown")[:16],
                }
            )

        try:
            import pandas as pd

            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        except ImportError:
            st.table(table_data)

        # Optional visualization of key-value coverage per document
        if table_data and 'Docling KV' in table_data[0]:
            chart_data = {
                "Document": [row["Document"] for row in table_data],
                "Docling KV": [row["Docling KV"] for row in table_data],
                "Inline KV": [row["Inline KV"] for row in table_data],
            }
            st.bar_chart(chart_data, y=["Docling KV", "Inline KV"], x="Document")
    else:
        st.info("Upload documents from the Parse page to begin.")

def display_debug_info():
    """Display debug information if enabled."""
    if st.session_state.show_debug_info:
        with st.expander("🐛 Debug Information"):
            st.write("**Session State:**")

            debug_state = {
                "Documents": len(st.session_state.parsed_documents),
                "Current doc": st.session_state.current_doc_index,
                "Current page": st.session_state.current_page,
                "Selected element": st.session_state.selected_element_id,
                "Search results": len(st.session_state.search_results),
                "Verification states": len(st.session_state.verification_states),
            }

            st.json(debug_state)


def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()

    # Display top navigation
    display_topbar()

    show_dashboard()

    # Display debug info if enabled
    display_debug_info()


if __name__ == "__main__":
    main()

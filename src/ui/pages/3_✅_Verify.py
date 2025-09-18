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
import logging
import streamlit as st
from typing import List, Optional, Dict, Any
from PIL import Image

try:
    from src.core.models import DocumentElement, ParsedDocument, KeyValuePair
    from src.verification.interface import VerificationInterface
    from src.verification.renderer import PDFRenderer, RenderConfig
    from src.core.parser import DoclingParser
    from src.core.classifiers.header_classifier import is_code_like
    from src.core.search import SmartSearchEngine, SearchResult
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.error("Please ensure the application is running from the correct directory")
    st.stop()

st.set_page_config(page_title="Verify Documents", page_icon="✅", layout="wide")

logger = logging.getLogger(__name__)


def initialize_session_state():
    """Initialize session state for verification page."""
    if "parsed_documents" not in st.session_state:
        st.session_state.parsed_documents = []

    if "current_doc_index" not in st.session_state:
        st.session_state.current_doc_index = 0

    if "current_page" not in st.session_state:
        st.session_state.current_page = 1

    if "selected_element_id" not in st.session_state:
        st.session_state.selected_element_id = None

    if "verification_interfaces" not in st.session_state:
        st.session_state.verification_interfaces = {}

    if "pdf_renderers" not in st.session_state:
        st.session_state.pdf_renderers = {}

    if "show_kv_pairs" not in st.session_state:
        st.session_state.show_kv_pairs = False

    if "extracted_kv_pairs" not in st.session_state:
        st.session_state.extracted_kv_pairs = {}

    if "show_code_values_only" not in st.session_state:
        st.session_state.show_code_values_only = False

    if "highlight_code_values" not in st.session_state:
        st.session_state.highlight_code_values = True

    if "verify_search_query" not in st.session_state:
        st.session_state.verify_search_query = ""

    if "verify_search_mode" not in st.session_state:
        st.session_state.verify_search_mode = "Exact"

    if "verify_search_include_metadata" not in st.session_state:
        st.session_state.verify_search_include_metadata = False

    if "verify_search_min_confidence" not in st.session_state:
        st.session_state.verify_search_min_confidence = 0.0

    if "verify_search_results" not in st.session_state:
        st.session_state.verify_search_results = []

    if "verify_tag_filters" not in st.session_state:
        st.session_state.verify_tag_filters = []

    if "verify_search_engines" not in st.session_state:
        st.session_state.verify_search_engines = {}


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

    # Ensure backward compatibility by migrating existing verification states
    interface = st.session_state.verification_interfaces[doc_index]
    interface.migrate_verification_states()

    return interface


def get_pdf_renderer(doc_index: int) -> Optional[PDFRenderer]:
    """Get or create PDF renderer for a document."""
    if doc_index not in st.session_state.pdf_renderers:
        config = RenderConfig(
            highlight_color=(255, 255, 0, 80),  # Semi-transparent yellow
            border_color=(255, 0, 0),  # Red border
            border_width=2,
            show_element_type=True,
            show_confidence=True,
        )
        st.session_state.pdf_renderers[doc_index] = PDFRenderer(config=config)

    return st.session_state.pdf_renderers[doc_index]


def get_page_elements(doc: ParsedDocument, page_number: int) -> List[DocumentElement]:
    """Get all elements for a specific page."""
    return [e for e in doc.elements if e.page_number == page_number]


def extract_kv_pairs_for_document(doc: ParsedDocument, doc_index: int) -> List[KeyValuePair]:
    """Extract KV pairs for a document using the parser."""
    if doc_index in st.session_state.extracted_kv_pairs:
        return st.session_state.extracted_kv_pairs[doc_index]

    try:
        parser_config = getattr(st.session_state, "parser_config", {})

        base_pairs: List[KeyValuePair] = []
        docling_pairs = getattr(doc, "key_values", []) or []
        for pair in docling_pairs:
            if isinstance(pair, KeyValuePair):
                pair.metadata.setdefault("source", "docling_core")
                base_pairs.append(pair)

        combined_pairs = list(base_pairs)

        if parser_config.get("enable_kv_extraction", False):
            from docling.datamodel.pipeline_options import TableFormerMode

            table_mode = (
                TableFormerMode.ACCURATE
                if parser_config.get("table_mode") == "accurate"
                else TableFormerMode.FAST
            )

            ocr_lang = parser_config.get("ocr_language", "eng")
            if "+" in str(ocr_lang):
                ocr_lang = ocr_lang.split("+")

            parser = DoclingParser(
                enable_ocr=parser_config.get("enable_ocr", False),
                enable_tables=parser_config.get("enable_tables", True),
                generate_page_images=False,
                ocr_engine=parser_config.get("ocr_engine", "tesseract"),
                ocr_lang=ocr_lang,
                table_mode=table_mode,
                image_scale=parser_config.get("image_scale", 1.0),
                enable_kv_extraction=True,
                header_classifier_enabled=parser_config.get("header_classifier_enabled", False),
            )

            heuristic_pairs = parser.kv_extractor.extract(doc.elements) if parser.kv_extractor else []
            for hp in heuristic_pairs:
                hp.metadata.setdefault("source", "heuristic")

            # Merge, avoiding duplicates by label/value/page
            seen = {(p.label_text, p.value_text, p.page_number) for p in combined_pairs}
            for hp in heuristic_pairs:
                key = (hp.label_text, hp.value_text, hp.page_number)
                if key not in seen:
                    combined_pairs.append(hp)
                    seen.add(key)

        st.session_state.extracted_kv_pairs[doc_index] = combined_pairs
        return combined_pairs

    except Exception as e:
        st.error(f"Failed to extract KV pairs: {e}")
        logger.exception("KV extraction failed", exc_info=e)
        return []


def get_kv_pairs_for_page(kv_pairs: List[KeyValuePair], page_number: int) -> List[KeyValuePair]:
    """Get KV pairs for a specific page."""
    return [kv for kv in kv_pairs if kv.page_number == page_number]


def kv_has_code(kv: KeyValuePair) -> bool:
    """Return True if a KV pair's value looks like a code or identifier."""
    value_text = getattr(kv, "value_text", "") or ""
    return bool(value_text and is_code_like(value_text))


def get_verify_search_engine(doc_index: int) -> SmartSearchEngine:
    """Get or create a SmartSearchEngine for the specified document index."""
    engines = st.session_state.verify_search_engines
    if doc_index not in engines:
        if doc_index >= len(st.session_state.parsed_documents):
            raise IndexError(f"Document index {doc_index} out of range")
        doc = st.session_state.parsed_documents[doc_index]
        engines[doc_index] = SmartSearchEngine(doc.elements)
    return engines[doc_index]


def run_verify_search(
    doc_index: int,
    query: str,
    mode: str,
    include_metadata: bool,
    min_score: float,
) -> List[SearchResult]:
    """Execute a search against the current document."""
    engine = get_verify_search_engine(doc_index)
    logger.info(
        "Running %s search (metadata=%s, min_score=%s) on doc index %s", mode, include_metadata, min_score, doc_index
    )
    kwargs = {"include_metadata": include_metadata, "min_score": min_score}

    if mode.lower() == "exact":
        return engine.search_exact(query, **kwargs)
    if mode.lower() == "fuzzy":
        return engine.search_fuzzy(query, **kwargs)
    return engine.search_semantic(query, **kwargs)


def find_element_by_id(doc: ParsedDocument, element_id: Optional[int]) -> Optional[DocumentElement]:
    """Locate a document element by element_id."""
    if element_id is None:
        return None
    for element in doc.elements:
        if element.metadata.get("element_id") == element_id:
            return element
    return None


def go_to_element(element: DocumentElement) -> None:
    """Navigate to the page/element in the verification UI."""
    st.session_state.current_page = element.page_number
    element_id = element.metadata.get("element_id")
    st.session_state.selected_element_id = element_id
    logger.info(
        "Navigating to element_id=%s on page %s", element_id, element.page_number
    )
    st.rerun()


def go_to_kv_pair(kv: KeyValuePair) -> None:
    """Navigate to the KV pair's associated element (value preferred, fallback to label)."""
    target_ids = kv.metadata.get("value_element_ids", []) if kv.metadata else []
    target_id = next((eid for eid in target_ids if eid is not None), None)
    if target_id is None and kv.metadata:
        target_id = kv.metadata.get("label_element_id")

    st.session_state.current_page = kv.page_number
    st.session_state.selected_element_id = target_id
    logger.info(
        "Navigating to KV pair on page %s (target_element_id=%s, label='%s')",
        kv.page_number,
        target_id,
        kv.label_text,
    )
    st.rerun()


def display_search_panel(doc: ParsedDocument, doc_index: int, kv_pairs: List[KeyValuePair]):
    """Render the combined search and quick-filter controls."""
    st.subheader("🔍 Search & Filters")

    search_col1, search_col2 = st.columns([4, 1])
    query = search_col1.text_input(
        "Search this document",
        value=st.session_state.verify_search_query,
        placeholder="Enter keywords or phrases",
    )
    search_button = search_col2.button("Search", use_container_width=True)

    mode_options = ["Exact", "Fuzzy", "Semantic"]
    mode = st.selectbox(
        "Search mode",
        options=mode_options,
        index=mode_options.index(st.session_state.verify_search_mode)
        if st.session_state.verify_search_mode in mode_options
        else 0,
    )
    st.session_state.verify_search_mode = mode

    options_col1, options_col2 = st.columns(2)
    with options_col1:
        include_metadata = st.checkbox(
            "Search metadata",
            value=st.session_state.verify_search_include_metadata,
            help="Include element metadata fields in search",
        )
        st.session_state.verify_search_include_metadata = include_metadata

    with options_col2:
        min_confidence = st.slider(
            "Minimum confidence",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.verify_search_min_confidence,
            step=0.05,
        )
        st.session_state.verify_search_min_confidence = min_confidence

    executed_results: List[SearchResult] = []
    if search_button:
        if not query.strip():
            st.warning("Enter a search query to begin.")
            logger.info("Search button pressed without query")
        else:
            logger.info("Search initiated with query='%s'", query)
            executed_results = run_verify_search(
                doc_index,
                query,
                mode,
                include_metadata,
                min_confidence,
            )
            st.session_state.verify_search_query = query
            st.session_state.verify_search_results = [
                {
                    "element_id": res.element.metadata.get("element_id"),
                    "page": res.element.page_number,
                    "element_type": res.element.element_type,
                    "text": res.element.text,
                    "score": res.score,
                    "match_type": res.match_type,
                    "matched_text": res.matched_text,
                }
                for res in executed_results
            ]
            st.success(f"Found {len(executed_results)} matches")
            logger.info("Search completed with %d matches", len(executed_results))

    results_to_show = st.session_state.verify_search_results
    if results_to_show:
        st.markdown("### Search results")
        for idx, result in enumerate(results_to_show[:50]):
            with st.container():
                col_a, col_b = st.columns([4, 1])
                preview = result["text"][:150] + ("..." if len(result["text"]) > 150 else "")
                col_a.markdown(
                    f"**Page {result['page']}** · {result['element_type'].title()} · Score {result['score']:.2f}"
                )
                col_a.write(preview)
                if result.get("matched_text"):
                    col_a.caption(f"Match: {result['matched_text']}")

                if col_b.button("Go", key=f"verify_search_go_{idx}"):
                    element = find_element_by_id(doc, result.get("element_id"))
                    if element:
                        go_to_element(element)
                    else:
                        st.warning("Element no longer available in this document")
                        logger.warning("Search result element_id=%s not found", result.get("element_id"))

    available_tags = sorted({tag for kv in kv_pairs for tag in (kv.metadata or {}).get("tags", [])})
    if available_tags:
        st.markdown("### Quick field filters")
        selected_tags = st.multiselect(
            "Filter by field tags",
            options=available_tags,
            default=st.session_state.verify_tag_filters,
        )
        st.session_state.verify_tag_filters = selected_tags

        if selected_tags:
            logger.info("Tag filters applied: %s", selected_tags)
            filtered_kv = [
                kv for kv in kv_pairs if any(tag in selected_tags for tag in (kv.metadata or {}).get("tags", []))
            ]
            if filtered_kv:
                for idx, kv in enumerate(filtered_kv[:50]):
                    with st.container():
                        cols = st.columns([4, 1])
                        cols[0].markdown(
                            f"**Page {kv.page_number}** · {kv.label_text} = `{kv.value_text}`"
                        )
                        tags = kv.metadata.get("tags", []) if kv.metadata else []
                        if tags:
                            cols[0].caption(" ".join(f"`{tag}`" for tag in tags))
                        if cols[1].button("Go", key=f"quick_kv_go_{idx}"):
                            go_to_kv_pair(kv)
            else:
                st.info("No fields match the selected tags.")
                logger.info("No KV pairs matched tags %s", selected_tags)


def display_kv_pairs_toggle():
    """Display toggle for KV pairs visualization."""
    # Check if KV extraction was enabled during parsing
    parser_config = getattr(st.session_state, "parser_config", {})
    kv_enabled = parser_config.get("enable_kv_extraction", False)

    if not kv_enabled:
        with st.expander("🔬 Key-Value Pairs", expanded=False):
            st.info(
                "Key-Value extraction was not enabled during parsing. Please re-parse documents with 'Enable Key-Value Extraction' checked in the Parse page to see field pairs."
            )
        return False

    with st.expander("🔗 Key-Value Pairs", expanded=st.session_state.show_kv_pairs):
        st.session_state.show_kv_pairs = st.checkbox(
            "Show Field Pairs",
            value=st.session_state.show_kv_pairs,
            help="Display extracted key-value pairs with visual overlays on the page image",
        )

        if st.session_state.show_kv_pairs:
            toggles_col1, toggles_col2 = st.columns(2)
            with toggles_col1:
                st.session_state.highlight_code_values = st.checkbox(
                    "Highlight Code-Like Values",
                    value=st.session_state.get("highlight_code_values", True),
                    help="Use distinctive overlays for invoice numbers, IDs, and similar codes",
                )
            with toggles_col2:
                st.session_state.show_code_values_only = st.checkbox(
                    "Show Only Codes & IDs",
                    value=st.session_state.get("show_code_values_only", False),
                    help="Filter the list and overlays to just code-like values",
                )

            st.info(
                "💡 Key-Value pairs are shown with colored boxes: labels in one color, values in another, connected with lines."
            )

    return st.session_state.show_kv_pairs


def display_kv_pairs_list(kv_pairs: List[KeyValuePair], page_number: int):
    """Display list of KV pairs for a page."""
    page_kvs = get_kv_pairs_for_page(kv_pairs, page_number)

    if not page_kvs:
        st.info(f"No key-value pairs found on page {page_number}")
        return

    total_pairs = len(page_kvs)
    code_pairs = sum(1 for kv in page_kvs if kv_has_code(kv))
    show_codes_only = st.session_state.get("show_code_values_only", False)
    highlight_codes = st.session_state.get("highlight_code_values", True)

    st.subheader(f"📋 Field Pairs - Page {page_number}")
    metrics_col1, metrics_col2 = st.columns(2)
    metrics_col1.metric("Pairs on page", total_pairs)
    metrics_col2.metric("Code-like values", code_pairs)

    if show_codes_only:
        page_kvs = [kv for kv in page_kvs if kv_has_code(kv)]
        if not page_kvs:
            st.warning("No code-like values detected on this page.")
            return

    try:
        import pandas as pd
    except ImportError:
        pd = None

    rows: List[Dict[str, Any]] = []
    for kv in page_kvs:
        tags = (kv.metadata or {}).get("tags", [])
        rows.append(
            {
                "Label": kv.label_text,
                "Value": kv.value_text,
                "Tags": ", ".join(tags) if tags else "-",
                "Confidence": round(kv.confidence, 3),
                "Strategy": (kv.metadata or {}).get("strategy"),
                "Source": (kv.metadata or {}).get("source", "-"),
            }
        )

    if pd is not None:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.table(rows)

    options = {
        f"Page {kv.page_number}: {kv.label_text} → {kv.value_text[:40]}": idx
        for idx, kv in enumerate(page_kvs)
    }

    if options:
        selection = st.selectbox(
            "Jump to field",
            options=list(options.keys()),
            key=f"kv_select_page_{page_number}",
        )
        if st.button("Go to field", key=f"kv_pair_go_{page_number}"):
            selected_idx = options.get(selection)
            if selected_idx is not None:
                go_to_kv_pair(page_kvs[selected_idx])


def display_document_selector():
    """Display document selection interface."""
    if not st.session_state.parsed_documents:
        st.warning("No documents loaded. Please go to the Parse page to upload PDFs.")
        return False

    # Document selection
    doc_names = []
    for i, doc in enumerate(st.session_state.parsed_documents):
        name = doc.metadata.get("filename", f"Document {i+1}")
        element_count = len(doc.elements)
        page_count = doc.metadata.get("page_count", "Unknown")
        doc_names.append(f"{name} ({element_count} elements, {page_count} pages)")

    selected_idx = st.selectbox(
        "Select Document",
        options=range(len(doc_names)),
        format_func=lambda i: doc_names[i],
        index=st.session_state.current_doc_index,
    )

    if selected_idx != st.session_state.current_doc_index:
        st.session_state.current_doc_index = selected_idx
        st.session_state.current_page = 1
        st.session_state.selected_element_id = None
        st.session_state.verify_search_results = []
        st.session_state.verify_search_query = ""
        st.session_state.verify_tag_filters = []
        st.session_state.verify_search_engines = {}
        st.rerun()

    return True


def display_page_navigation(doc: ParsedDocument):
    """Display page navigation controls."""
    max_pages = doc.metadata.get("page_count", len(doc.pages) if doc.pages else 1)
    if not isinstance(max_pages, int) or max_pages < 1:
        max_pages = 1

    if max_pages == 1:
        st.session_state.current_page = 1
        col_prev, col_info, col_next = st.columns([1, 6, 1])
        with col_prev:
            st.button("⬅️ Prev", disabled=True)
        with col_info:
            st.markdown("**Single-page document**")
        with col_next:
            st.button("➡️ Next", disabled=True)
        return

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
            format="Page %d of %d" % (st.session_state.current_page, max_pages),
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
            summary["total_elements"],
            help="Total number of elements in the document",
        )

    with col2:
        st.metric(
            "Verified",
            summary["verified_elements"],
            delta=f"{progress['completion_percentage']:.1f}%",
            help="Number of elements that have been verified",
        )

    with col3:
        st.metric(
            "Corrections",
            summary["total_corrections"],
            help="Number of elements that needed corrections",
        )

    with col4:
        st.metric(
            "Accuracy",
            f"{summary['accuracy_percentage']:.1f}%",
            help="Percentage of elements that were correct",
        )


def render_page_with_overlays(
    doc: ParsedDocument,
    page_number: int,
    renderer: PDFRenderer,
    kv_pairs: Optional[List[KeyValuePair]] = None,
) -> Optional[Image.Image]:
    """Render a page with element overlays and optionally KV pairs."""
    if not doc.pages or page_number not in doc.pages:
        parser_cfg = doc.metadata.get("parser_config", {}) if isinstance(doc.metadata, dict) else {}
        logger.warning(
            "No page image for page %s (images_enabled=%s, pages_cached=%s)",
            page_number,
            parser_cfg.get("images_enabled"),
            list(doc.pages.keys()) if doc.pages else []
        )
        st.error(
            f"Page {page_number} image not available. Enable 'Generate Page Images' in parser settings and re-parse to view overlays."
        )
        return None

    page_image = doc.pages[page_number]
    page_elements = get_page_elements(doc, page_number)

    # Start with base image
    overlay_image = page_image

    # Render element overlays if there are elements
    if page_elements:
        try:
            overlay_image = renderer.render_page_with_overlays(
                overlay_image, page_elements, page_number
            )
        except Exception as e:
            logger.error("Failed to render element overlays for page %s: %s", page_number, e)
            st.error(f"Failed to render element overlays: {e}")

    # Render KV pairs overlays if requested and available
    if st.session_state.show_kv_pairs and kv_pairs:
        page_kvs = get_kv_pairs_for_page(kv_pairs, page_number)
        if st.session_state.get("show_code_values_only", False):
            page_kvs = [kv for kv in page_kvs if kv_has_code(kv)]

        if page_kvs:
            try:
                overlay_image = renderer.render_kv_pairs(
                    overlay_image,
                    page_kvs,
                    pdf_size=None,  # Let renderer auto-detect
                    highlight_codes=st.session_state.get("highlight_code_values", True),
                )
            except Exception as e:
                logger.error("Failed to render KV overlays for page %s: %s", page_number, e)
                st.error(f"Failed to render KV pair overlays: {e}")

    return overlay_image


def display_element_details(
    element: DocumentElement, verification_interface: VerificationInterface
):
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
            st.write(
                f"**Position:** ({element.bbox['x0']:.1f}, {element.bbox['y0']:.1f}) to ({element.bbox['x1']:.1f}, {element.bbox['y1']:.1f})"
            )
        st.write(f"**Element ID:** {element.metadata.get('element_id', 'Unknown')}")

    # Text content
    st.write("**Text Content:**")
    text_content = st.text_area(
        "Element text",
        value=element.text,
        height=100,
        key=f"element_text_{element.metadata.get('element_id', 'unknown')}",
        help="You can edit the text here to make corrections",
    )

    # Verification status
    element_id = element.metadata.get("element_id", 0)
    current_state = verification_interface.get_element_state(element_id)

    st.write(f"**Verification Status:** {current_state.status}")

    if current_state.notes:
        st.write(f"**Notes:** {current_state.notes}")

    if hasattr(current_state, "corrected_text") and getattr(current_state, "corrected_text", None):
        st.write(f"**Corrected Text:** {getattr(current_state, 'corrected_text', '')}")

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
                element_id, corrected_text=corrected_text, notes=notes, verified_by="user"
            )
            st.success("Marked as incorrect!")
            st.rerun()

    with col3:
        if st.button("⚠️ Partially Correct", use_container_width=True):
            corrected_text = text_content if text_content != element.text else None
            notes = "Partially correct, minor adjustments made"

            verification_interface.mark_element_partial(
                element_id, corrected_text=corrected_text, notes=notes, verified_by="user"
            )
            st.success("Marked as partially correct!")
            st.rerun()

    # Additional notes
    with st.expander("📝 Add Notes"):
        notes = st.text_area(
            "Verification notes",
            placeholder="Add any additional notes about this element...",
            key=f"notes_{element_id}",
        )

        if st.button("Save Notes"):
            # Update the element's verification state with notes
            current_state.notes = notes
            st.success("Notes saved!")


def display_page_elements_list(
    page_elements: List[DocumentElement], verification_interface: VerificationInterface
):
    """Display a list of elements on the current page."""
    st.subheader(f"📋 Page Elements ({len(page_elements)})")

    for i, element in enumerate(page_elements):
        element_id = element.metadata.get("element_id", i)
        state = verification_interface.get_element_state(element_id)

        # Status emoji
        status_emoji = {"pending": "⏳", "correct": "✅", "incorrect": "❌", "partial": "⚠️"}

        emoji = status_emoji.get(state.status, "⏳")

        # Create expandable element
        text_preview = element.text[:100] + "..." if len(element.text) > 100 else element.text

        with st.expander(f"{emoji} {element.element_type.title()} - {text_preview}"):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.write(f"**Full Text:** {element.text}")
                if element.bbox:
                    st.write(
                        f"**Position:** ({element.bbox['x0']:.1f}, {element.bbox['y0']:.1f}) to ({element.bbox['x1']:.1f}, {element.bbox['y1']:.1f})"
                    )

            with col2:
                if st.button(f"Select", key=f"select_{element_id}"):
                    st.session_state.selected_element_id = element_id
                    st.rerun()

                st.write(f"**Confidence:** {element.confidence:.2f}")
                st.write(f"**Status:** {state.status}")


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

    logger.info(
        "Verify page active for document '%s' (index=%s, pages=%s, elements=%s)",
        doc.metadata.get("filename"),
        st.session_state.current_doc_index,
        doc.metadata.get("page_count"),
        len(doc.elements),
    )

    verification_interface = get_verification_interface(st.session_state.current_doc_index)
    renderer = get_pdf_renderer(st.session_state.current_doc_index)

    # Verification statistics
    if verification_interface is not None:
        display_verification_stats(verification_interface)
    else:
        st.error("Verification interface not available for this document.")

    kv_pairs = extract_kv_pairs_for_document(doc, st.session_state.current_doc_index)
    show_kv = display_kv_pairs_toggle()
    if show_kv:
        if kv_pairs:
            st.success(f"✅ Extracted {len(kv_pairs)} key-value pairs from this document")
            logger.info("Displaying %d KV pairs for doc index %d", len(kv_pairs), st.session_state.current_doc_index)
        else:
            st.warning("No key-value pairs found in this document")
            logger.info("No KV pairs available for doc index %d", st.session_state.current_doc_index)

    display_search_panel(doc, st.session_state.current_doc_index, kv_pairs)

    st.divider()

    layout_left, layout_right = st.columns([3, 2])

    with layout_left:
        display_page_navigation(doc)

        st.subheader(f"📄 Page {st.session_state.current_page}")

        if renderer is not None:
            overlay_image = render_page_with_overlays(
                doc, st.session_state.current_page, renderer, kv_pairs=kv_pairs
            )
        else:
            overlay_image = None
            st.error("PDF renderer not available for this document.")

        if overlay_image:
            st.image(
                overlay_image,
                caption=f"Page {st.session_state.current_page} with element overlays",
                use_container_width=True,
            )
            st.caption(
                "Tip: Click highlighted regions above or use search results to jump directly to fields."
            )
        else:
            st.error("Could not render page with overlays")

    with layout_right:
        st.subheader("Inspector")
        page_elements = get_page_elements(doc, st.session_state.current_page)

        tab_selected, tab_fields, tab_elements = st.tabs([
            "Selected Element",
            "Field Directory",
            "Page Elements",
        ])

        with tab_selected:
            if st.session_state.selected_element_id is not None:
                selected_element = find_element_by_id(doc, st.session_state.selected_element_id)
                if selected_element and selected_element.page_number == st.session_state.current_page:
                    if verification_interface is not None:
                        display_element_details(selected_element, verification_interface)
                    else:
                        st.error("Verification interface not available.")
                else:
                    st.warning("Selected element not found on current page")
                    st.session_state.selected_element_id = None
            else:
                st.info("Select an element from the page, search results, or field directory to inspect it here.")

        with tab_fields:
            if show_kv and kv_pairs:
                display_kv_pairs_list(kv_pairs, st.session_state.current_page)
            elif show_kv:
                st.warning("No key-value pairs found for this page.")
            else:
                st.info("Enable ‘Show Field Pairs’ to view detected labels and values.")

        with tab_elements:
            if verification_interface is not None:
                display_page_elements_list(page_elements, verification_interface)
            else:
                st.error("Verification interface not available.")

    # Page summary
    st.divider()

    # Bulk actions
    st.subheader("🔄 Bulk Actions")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("✅ Mark All Correct", use_container_width=True):
            if verification_interface is not None:
                element_ids = [e.metadata.get("element_id", i) for i, e in enumerate(page_elements)]
                verification_interface.mark_elements_correct(element_ids, verified_by="user")
                st.success(f"Marked {len(element_ids)} elements as correct!")
                st.rerun()
            else:
                st.error("Verification interface not available.")

    with col2:
        if st.button("🔄 Reset Page", use_container_width=True):
            if verification_interface is not None:
                for element in page_elements:
                    element_id = element.metadata.get("element_id", 0)
                    verification_interface.undo_verification(element_id)
                st.success("Reset all verifications on this page!")
                st.rerun()
            else:
                st.error("Verification interface not available.")

    with col3:
        if st.button("📊 Export Progress", use_container_width=True):
            st.switch_page("pages/4_📊_Export.py")

    with col4:
        if verification_interface is not None:
            needing_verification = verification_interface.get_elements_needing_verification()
            if needing_verification and st.button("⏭️ Next Unverified", use_container_width=True):
                # Find next element needing verification
                element_id, next_element = needing_verification[0]
                next_page = next_element.page_number

                if next_page != st.session_state.current_page:
                    st.session_state.current_page = next_page

                st.session_state.selected_element_id = next_element.metadata.get("element_id")
                st.rerun()


if __name__ == "__main__":
    main()

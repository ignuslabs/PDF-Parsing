"""
Search Page - Advanced Document Search

Provides multi-mode search capabilities across parsed documents with
advanced filtering and result ranking.
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
from typing import List, Dict, Tuple, Any
import re

# Project imports (now that path is set)
try:
    from src.core.search import SmartSearchEngine, SearchResult
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.error("Please ensure the application is running from the correct directory")
    st.stop()

st.set_page_config(
    page_title="Search Documents",
    page_icon="🔍",
    layout="wide"
)

def initialize_session_state():
    """Initialize session state for search page."""
    if 'parsed_documents' not in st.session_state:
        st.session_state.parsed_documents = []
    
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""
    
    if 'search_filters' not in st.session_state:
        st.session_state.search_filters = {
            'element_types': [],
            'page_range': None,
            'confidence_threshold': 0.0,
            'document_indices': []
        }
    
    if 'search_engines' not in st.session_state:
        st.session_state.search_engines = {}

def get_search_engine(doc_index: int) -> SmartSearchEngine:
    """Get or create search engine for a document."""
    if doc_index not in st.session_state.search_engines:
        doc = st.session_state.parsed_documents[doc_index]
        st.session_state.search_engines[doc_index] = SmartSearchEngine(doc.elements)
    
    return st.session_state.search_engines[doc_index]

def display_search_interface():
    """Display the search interface with query input and filters."""
    st.subheader("🔍 Search Query")
    
    # Search input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "Enter search query",
            value=st.session_state.search_query,
            placeholder="Search for text, keywords, or phrases...",
            key="search_input"
        )
    
    with col2:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Advanced search options
    with st.expander("🎛️ Advanced Search Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Search mode
            search_mode = st.selectbox(
                "Search Mode",
                options=['exact', 'fuzzy', 'semantic'],
                index=0,
                help="Exact: Precise matches, Fuzzy: Similar matches, Semantic: Meaning-based matches"
            )
            
            # Element type filter
            available_types = set()
            for doc in st.session_state.parsed_documents:
                available_types.update(e.element_type for e in doc.elements)
            
            selected_types = st.multiselect(
                "Element Types",
                options=sorted(available_types),
                default=st.session_state.search_filters['element_types'],
                help="Filter by element types"
            )
        
        with col2:
            # Confidence threshold
            confidence_threshold = st.slider(
                "Minimum Confidence",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.search_filters['confidence_threshold'],
                step=0.05,
                help="Minimum confidence score for results"
            )
            
            # Document selection
            doc_names = [doc.metadata.get('filename', f'Document {i+1}') 
                        for i, doc in enumerate(st.session_state.parsed_documents)]
            
            selected_docs = st.multiselect(
                "Search in Documents",
                options=range(len(doc_names)),
                format_func=lambda i: doc_names[i],
                default=list(range(len(doc_names))),
                help="Select documents to search in"
            )
        
        # Page range filter
        page_range = st.text_input(
            "Page Range (optional)",
            placeholder="e.g., 1-5, 10, 15-20",
            help="Specify page numbers or ranges (e.g., '1-5, 10, 15-20')"
        )
    
    # Update session state
    if query != st.session_state.search_query:
        st.session_state.search_query = query
    
    st.session_state.search_filters.update({
        'element_types': selected_types,
        'confidence_threshold': confidence_threshold,
        'document_indices': selected_docs,
        'page_range': page_range if page_range.strip() else None
    })
    
    return query, search_mode, search_button

def parse_page_range(page_range_str: str) -> List[int]:
    """Parse page range string into list of page numbers."""
    if not page_range_str:
        return []
    
    pages = []
    try:
        parts = page_range_str.split(',')
        for part in parts:
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                pages.extend(range(start, end + 1))
            else:
                pages.append(int(part))
    except ValueError:
        st.error("Invalid page range format. Use format like '1-5, 10, 15-20'")
        return []
    
    return sorted(set(pages))

def perform_search(
    query: str, 
    search_mode: str, 
    filters: Dict[str, Any]
) -> List[Tuple[SearchResult, int]]:
    """Perform search across selected documents."""
    if not query.strip():
        return []
    
    all_results = []
    
    # Search in selected documents
    for doc_index in filters['document_indices']:
        search_engine = get_search_engine(doc_index)
        
        try:
            # Perform search based on mode
            if search_mode == 'exact':
                results = search_engine.search_exact(query)
            elif search_mode == 'fuzzy':
                results = search_engine.search_fuzzy(query)
            else:  # semantic
                results = search_engine.search_semantic(query)
            
            # Apply filters
            filtered_results = apply_filters(results, filters)
            
            # Add document index to results
            for result in filtered_results:
                all_results.append((result, doc_index))
                
        except Exception as e:
            st.error(f"Search failed for document {doc_index + 1}: {str(e)}")
    
    # Sort by relevance score
    all_results.sort(key=lambda x: x[0].score, reverse=True)
    
    return all_results

def apply_filters(results: List[SearchResult], filters: Dict[str, Any]) -> List[SearchResult]:
    """Apply filters to search results."""
    filtered = results
    
    # Element type filter
    if filters['element_types']:
        filtered = [r for r in filtered if r.element.element_type in filters['element_types']]
    
    # Confidence threshold filter
    if filters['confidence_threshold'] > 0:
        filtered = [r for r in filtered if r.element.confidence >= filters['confidence_threshold']]
    
    # Page range filter
    if filters['page_range']:
        allowed_pages = parse_page_range(filters['page_range'])
        if allowed_pages:
            filtered = [r for r in filtered if r.element.page_number in allowed_pages]
    
    return filtered

def display_search_results(results: List[Tuple[SearchResult, int]]):
    """Display search results with highlighting and navigation."""
    if not results:
        st.info("No results found. Try adjusting your search query or filters.")
        return
    
    st.subheader(f"📋 Search Results ({len(results)})")
    
    # Results summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Results", len(results))
    
    with col2:
        avg_score = sum(r[0].score for r in results) / len(results)
        st.metric("Avg Relevance", f"{avg_score:.3f}")
    
    with col3:
        pages = set(r[0].element.page_number for r in results)
        st.metric("Pages Found", len(pages))
    
    with col4:
        types = set(r[0].element.element_type for r in results)
        st.metric("Element Types", len(types))
    
    # Results display
    for i, (result, doc_index) in enumerate(results):
        element = result.element
        doc = st.session_state.parsed_documents[doc_index]
        doc_name = doc.metadata.get('filename', f'Document {doc_index + 1}')
        
        # Result card
        with st.expander(
            f"🎯 {element.element_type.title()} - Score: {result.score:.3f} - {doc_name}, Page {element.page_number}",
            expanded=i < 3  # Expand first 3 results
        ):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Highlight matching text
                highlighted_text = highlight_matches(element.text, st.session_state.search_query)
                st.markdown(f"**Text:** {highlighted_text}", unsafe_allow_html=True)
                
                if result.match_context:
                    st.write(f"**Match Context:** {result.match_context}")
                
                # Show position if available
                if element.bbox:
                    st.write(f"**Position:** ({element.bbox['x0']:.1f}, {element.bbox['y0']:.1f}) to ({element.bbox['x1']:.1f}, {element.bbox['y1']:.1f})")
            
            with col2:
                st.write(f"**Document:** {doc_name}")
                st.write(f"**Page:** {element.page_number}")
                st.write(f"**Type:** {element.element_type}")
                st.write(f"**Confidence:** {element.confidence:.3f}")
                st.write(f"**Relevance:** {result.score:.3f}")
                
                # Navigation buttons
                if st.button(f"Go to Verify", key=f"verify_{doc_index}_{element.metadata.get('element_id', i)}"):
                    # Set session state for verification page
                    st.session_state.current_doc_index = doc_index
                    st.session_state.current_page = element.page_number
                    st.session_state.selected_element_id = element.metadata.get('element_id')
                    st.switch_page("pages/3_✅_Verify.py")

def highlight_matches(text: str, query: str) -> str:
    """Highlight search matches in text."""
    if not query.strip():
        return text
    
    # Simple highlighting - could be improved with more sophisticated matching
    try:
        # Escape special regex characters in query
        escaped_query = re.escape(query.strip())
        
        # Case-insensitive highlighting
        pattern = re.compile(escaped_query, re.IGNORECASE)
        highlighted = pattern.sub(lambda m: f'<mark style="background-color: yellow;">{m.group()}</mark>', text)
        
        return highlighted
    except re.error:
        return text

def display_search_analytics(results: List[Tuple[SearchResult, int]]):
    """Display search analytics and insights."""
    if not results:
        return
    
    st.subheader("📊 Search Analytics")
    
    # Element type distribution
    type_counts = {}
    for result, doc_index in results:
        elem_type = result.element.element_type
        type_counts[elem_type] = type_counts.get(elem_type, 0) + 1
    
    if type_counts:
        st.write("**Results by Element Type:**")
        for elem_type, count in sorted(type_counts.items()):
            st.write(f"- {elem_type.title()}: {count}")
    
    # Page distribution
    page_counts = {}
    for result, doc_index in results:
        page_num = result.element.page_number
        page_counts[page_num] = page_counts.get(page_num, 0) + 1
    
    if page_counts:
        st.write("**Results by Page:**")
        # Show top 10 pages
        top_pages = sorted(page_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for page_num, count in top_pages:
            st.write(f"- Page {page_num}: {count} results")
    
    # Score distribution
    scores = [result.score for result, doc_index in results]
    if scores:
        st.write(f"**Score Range:** {min(scores):.3f} - {max(scores):.3f}")

def main():
    """Main search page function."""
    initialize_session_state()
    
    st.title("🔍 Search Documents")
    st.markdown("Search across your parsed documents with advanced filtering and ranking.")
    
    # Check if documents are loaded
    if not st.session_state.parsed_documents:
        st.warning("No documents loaded. Please go to the Parse page to upload PDFs.")
        
        if st.button("📄 Go to Parse Page"):
            st.switch_page("pages/1_📄_Parse.py")
        return
    
    # Search interface
    query, search_mode, search_button = display_search_interface()
    
    # Perform search
    if search_button and query and query.strip():
        with st.spinner("Searching..."):
            results = perform_search(query, search_mode, st.session_state.search_filters)
            st.session_state.search_results = results
    
    # Display results
    if st.session_state.search_results:
        display_search_results(st.session_state.search_results)
        
        # Analytics
        with st.expander("📈 Search Analytics"):
            display_search_analytics(st.session_state.search_results)
    
    # Search history/suggestions
    st.divider()
    
    with st.expander("💡 Search Tips"):
        st.markdown("""
        **Search Modes:**
        - **Exact**: Find exact phrase matches
        - **Fuzzy**: Find similar text with typos or variations
        - **Semantic**: Find text with similar meaning
        
        **Search Tips:**
        - Use quotes for exact phrases: "financial statement"
        - Use filters to narrow results by element type or page
        - Adjust confidence threshold to filter low-quality extractions
        - Use page ranges to search specific sections: "1-10, 20"
        
        **Examples:**
        - Find tables: Search for "table" with Element Type = table
        - Find headings: Use Element Type = heading filter
        - Find specific values: Use exact mode for numbers or codes
        """)

if __name__ == "__main__":
    main()
"""
Results Display Component

Provides reusable components for displaying parsing results, search results,
and verification data with consistent formatting.
"""

import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from datetime import datetime
import re

# Import types (will need to handle imports carefully in actual usage)
try:
    from ...core.models import DocumentElement, ParsedDocument
    from ...core.search import SearchResult
except ImportError:
    # Handle import for when running as part of the application
    pass


class ResultsDisplay:
    """Provides UI components for displaying various types of results."""

    def __init__(self):
        """Initialize results display."""
        self.element_type_colors = {
            "text": "#2E86AB",  # Blue
            "heading": "#A23B72",  # Purple
            "table": "#F18F01",  # Orange
            "list": "#C73E1D",  # Red
            "image": "#0F7B0F",  # Green
            "caption": "#7209B7",  # Dark Purple
            "formula": "#F72585",  # Pink
            "code": "#4361EE",  # Light Blue
            "footnote": "#B7094C",  # Dark Pink
        }

    def display_parsing_summary(self, parsed_doc, show_details: bool = True):
        """Display summary of parsing results.

        Args:
            parsed_doc: ParsedDocument object
            show_details: Whether to show detailed breakdown
        """
        st.subheader("✅ Parsing Summary")

        # Basic metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Elements", len(parsed_doc.elements))

        with col2:
            page_count = parsed_doc.metadata.get("page_count", "Unknown")
            st.metric("Pages", page_count)

        with col3:
            page_images = len(parsed_doc.pages) if parsed_doc.pages else 0
            st.metric("Page Images", page_images)

        with col4:
            if parsed_doc.elements:
                avg_confidence = sum(e.confidence for e in parsed_doc.elements) / len(
                    parsed_doc.elements
                )
                st.metric("Avg Confidence", f"{avg_confidence:.2f}")
            else:
                st.metric("Avg Confidence", "N/A")

        if not show_details:
            return

        # Element type breakdown
        self.display_element_type_breakdown(parsed_doc.elements)

        # Page distribution
        self.display_page_distribution(parsed_doc.elements)

        # Confidence distribution
        self.display_confidence_distribution(parsed_doc.elements)

    def display_element_type_breakdown(self, elements: List):
        """Display breakdown of elements by type.

        Args:
            elements: List of DocumentElement objects
        """
        if not elements:
            return

        element_types = {}
        for element in elements:
            elem_type = element.element_type
            element_types[elem_type] = element_types.get(elem_type, 0) + 1

        st.write("**Element Types:**")

        # Create DataFrame for visualization
        df = pd.DataFrame(list(element_types.items()), columns=["Type", "Count"])
        df["Percentage"] = (df["Count"] / df["Count"].sum() * 100).round(1)

        # Display as bar chart
        st.bar_chart(df.set_index("Type")["Count"])

        # Display as metrics
        type_cols = st.columns(min(len(element_types), 4))
        for i, (elem_type, count) in enumerate(sorted(element_types.items())):
            with type_cols[i % len(type_cols)]:
                percentage = count / len(elements) * 100
                st.metric(
                    elem_type.title(),
                    count,
                    delta=f"{percentage:.1f}%",
                    help=f"{count} {elem_type} elements ({percentage:.1f}% of total)",
                )

    def display_page_distribution(self, elements: List):
        """Display distribution of elements across pages.

        Args:
            elements: List of DocumentElement objects
        """
        if not elements:
            return

        page_counts = {}
        for element in elements:
            page_num = element.page_number
            page_counts[page_num] = page_counts.get(page_num, 0) + 1

        if len(page_counts) <= 1:
            return

        with st.expander("📄 Page Distribution"):
            # Create DataFrame
            df = pd.DataFrame(list(page_counts.items()), columns=["Page", "Elements"])
            df = df.sort_values("Page")

            # Display as line chart
            st.line_chart(df.set_index("Page"))

            # Summary stats
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Pages", len(page_counts))

            with col2:
                avg_elements = sum(page_counts.values()) / len(page_counts)
                st.metric("Avg Elements/Page", f"{avg_elements:.1f}")

            with col3:
                max_elements = max(page_counts.values())
                max_page = [k for k, v in page_counts.items() if v == max_elements][0]
                st.metric("Most Elements", f"{max_elements} (Page {max_page})")

    def display_confidence_distribution(self, elements: List):
        """Display confidence score distribution.

        Args:
            elements: List of DocumentElement objects
        """
        if not elements:
            return

        confidences = [e.confidence for e in elements]

        with st.expander("🎯 Confidence Distribution"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Min Confidence", f"{min(confidences):.3f}")

            with col2:
                st.metric("Avg Confidence", f"{sum(confidences)/len(confidences):.3f}")

            with col3:
                st.metric("Max Confidence", f"{max(confidences):.3f}")

            # Confidence histogram
            bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            confidence_bins = {}

            for conf in confidences:
                for i in range(len(bins) - 1):
                    if bins[i] <= conf <= bins[i + 1]:
                        bin_label = f"{bins[i]:.1f}-{bins[i+1]:.1f}"
                        confidence_bins[bin_label] = confidence_bins.get(bin_label, 0) + 1
                        break

            if confidence_bins:
                df = pd.DataFrame(list(confidence_bins.items()), columns=["Range", "Count"])
                st.bar_chart(df.set_index("Range"))

    def display_element_list(
        self,
        elements: List,
        title: str = "Elements",
        show_filters: bool = True,
        max_display: int = 10,
        allow_selection: bool = False,
    ) -> Optional[int]:
        """Display a list of elements with filtering options.

        Args:
            elements: List of DocumentElement objects
            title: Title for the list
            show_filters: Whether to show filter controls
            max_display: Maximum number of elements to display
            allow_selection: Whether to allow element selection

        Returns:
            Selected element ID if allow_selection is True
        """
        if not elements:
            st.info(f"No {title.lower()} found.")
            return None

        st.subheader(f"📋 {title} ({len(elements)})")

        # Filters
        filtered_elements = elements
        if show_filters:
            filtered_elements = self._apply_element_filters(elements)

        # Pagination
        if len(filtered_elements) > max_display:
            page_size = max_display
            total_pages = (len(filtered_elements) + page_size - 1) // page_size

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                page_num = st.slider(
                    "Page", min_value=1, max_value=total_pages, value=1, key=f"{title}_page_slider"
                )

            start_idx = (page_num - 1) * page_size
            end_idx = start_idx + page_size
            display_elements = filtered_elements[start_idx:end_idx]

            st.write(
                f"Showing {start_idx + 1}-{min(end_idx, len(filtered_elements))} of {len(filtered_elements)} elements"
            )
        else:
            display_elements = filtered_elements

        # Display elements
        selected_id = None
        for i, element in enumerate(display_elements):
            element_id = element.metadata.get("element_id", i)

            # Element container
            with st.expander(
                f"{element.element_type.title()} - Page {element.page_number} - {element.text[:50]}...",
                expanded=i < 3,
            ):
                col1, col2 = st.columns([3, 1])

                with col1:
                    self._display_element_details(element)

                with col2:
                    self._display_element_metadata(element)

                    if allow_selection:
                        if st.button(f"Select", key=f"select_{element_id}"):
                            selected_id = element_id

        return selected_id

    def display_search_results(
        self, results: List[Tuple], query: str = "", max_display: int = 20
    ) -> Optional[Tuple[int, int]]:  # Returns (doc_index, element_id)
        """Display search results with highlighting and navigation.

        Args:
            results: List of (SearchResult, doc_index) tuples
            query: Original search query for highlighting
            max_display: Maximum results to display

        Returns:
            Selected (doc_index, element_id) tuple if any
        """
        if not results:
            st.info("No search results found.")
            return None

        st.subheader(f"🔍 Search Results ({len(results)})")

        # Results summary
        self._display_search_summary(results)

        # Pagination for large result sets
        display_results = results[:max_display] if len(results) > max_display else results

        if len(results) > max_display:
            st.write(f"Showing top {max_display} results out of {len(results)}")

        # Display results
        selected = None
        for i, (search_result, doc_index) in enumerate(display_results):
            element = search_result.element
            element_id = element.metadata.get("element_id", i)

            # Result card
            relevance_color = self._get_relevance_color(search_result.score)

            with st.expander(
                f"🎯 Score: {search_result.score:.3f} | {element.element_type.title()} | Page {element.page_number}",
                expanded=i < 3,
            ):
                col1, col2 = st.columns([3, 1])

                with col1:
                    # Highlighted text
                    highlighted = self._highlight_text(element.text, query)
                    st.markdown(f"**Text:** {highlighted}", unsafe_allow_html=True)

                    if search_result.match_context:
                        st.write(f"**Context:** {search_result.match_context}")

                    if element.bbox:
                        st.write(
                            f"**Position:** ({element.bbox['x0']:.1f}, {element.bbox['y0']:.1f}) to ({element.bbox['x1']:.1f}, {element.bbox['y1']:.1f})"
                        )

                with col2:
                    st.write(f"**Document:** {doc_index + 1}")
                    st.write(f"**Page:** {element.page_number}")
                    st.write(f"**Type:** {element.element_type}")
                    st.write(f"**Confidence:** {element.confidence:.3f}")

                    # Relevance indicator
                    st.markdown(
                        f"<div style='background-color: {relevance_color}; padding: 5px; border-radius: 3px; text-align: center; color: white; font-weight: bold;'>"
                        f"Relevance: {search_result.score:.3f}"
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    if st.button("Go to Element", key=f"goto_{doc_index}_{element_id}"):
                        selected = (doc_index, element_id)

        return selected

    def _apply_element_filters(self, elements: List) -> List:
        """Apply filters to element list.

        Args:
            elements: List of elements to filter

        Returns:
            Filtered list of elements
        """
        # Get available filter values
        available_types = sorted(set(e.element_type for e in elements))
        available_pages = sorted(set(e.page_number for e in elements))

        # Filter controls
        col1, col2, col3 = st.columns(3)

        with col1:
            selected_types = st.multiselect(
                "Filter by Type",
                options=available_types,
                default=available_types,
                key="element_type_filter",
            )

        with col2:
            if len(available_pages) > 1:
                page_range = st.select_slider(
                    "Page Range",
                    options=available_pages,
                    value=(min(available_pages), max(available_pages)),
                    key="page_range_filter",
                )
            else:
                page_range = (available_pages[0], available_pages[0]) if available_pages else (1, 1)

        with col3:
            min_confidence = st.slider(
                "Min Confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
                key="confidence_filter",
            )

        # Apply filters
        filtered = elements

        if selected_types:
            filtered = [e for e in filtered if e.element_type in selected_types]

        if len(available_pages) > 1:
            filtered = [e for e in filtered if page_range[0] <= e.page_number <= page_range[1]]

        filtered = [e for e in filtered if e.confidence >= min_confidence]

        return filtered

    def _display_element_details(self, element):
        """Display detailed information about an element.

        Args:
            element: DocumentElement object
        """
        st.write(f"**Full Text:** {element.text}")

        if element.bbox:
            st.write(
                f"**Position:** ({element.bbox['x0']:.1f}, {element.bbox['y0']:.1f}) to ({element.bbox['x1']:.1f}, {element.bbox['y1']:.1f})"
            )

        if element.metadata and len(element.metadata) > 1:
            with st.expander("Additional Metadata"):
                for key, value in element.metadata.items():
                    if key != "element_id":
                        st.write(f"**{key.title()}:** {value}")

    def _display_element_metadata(self, element):
        """Display element metadata.

        Args:
            element: DocumentElement object
        """
        st.write(f"**Type:** {element.element_type}")
        st.write(f"**Page:** {element.page_number}")
        st.write(f"**Confidence:** {element.confidence:.3f}")

        # Color-coded confidence
        conf_color = self._get_confidence_color(element.confidence)
        st.markdown(
            f"<div style='background-color: {conf_color}; padding: 3px; border-radius: 3px; text-align: center; color: white; font-size: 12px;'>"
            f"{element.confidence:.3f}"
            "</div>",
            unsafe_allow_html=True,
        )

    def _display_search_summary(self, results: List[Tuple]):
        """Display summary statistics for search results.

        Args:
            results: List of search result tuples
        """
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Results", len(results))

        with col2:
            scores = [r[0].score for r in results]
            avg_score = sum(scores) / len(scores) if scores else 0
            st.metric("Avg Relevance", f"{avg_score:.3f}")

        with col3:
            pages = set(r[0].element.page_number for r in results)
            st.metric("Pages Found", len(pages))

        with col4:
            types = set(r[0].element.element_type for r in results)
            st.metric("Element Types", len(types))

    def _highlight_text(self, text: str, query: str) -> str:
        """Highlight search matches in text.

        Args:
            text: Text to highlight
            query: Search query

        Returns:
            Text with HTML highlighting
        """
        if not query.strip():
            return text

        try:
            escaped_query = re.escape(query.strip())
            pattern = re.compile(escaped_query, re.IGNORECASE)
            highlighted = pattern.sub(
                lambda m: f'<mark style="background-color: yellow; padding: 2px;">{m.group()}</mark>',
                text,
            )
            return highlighted
        except re.error:
            return text

    def _get_confidence_color(self, confidence: float) -> str:
        """Get color based on confidence score.

        Args:
            confidence: Confidence score (0-1)

        Returns:
            Color hex code
        """
        if confidence >= 0.8:
            return "#28a745"  # Green
        elif confidence >= 0.6:
            return "#ffc107"  # Yellow
        elif confidence >= 0.4:
            return "#fd7e14"  # Orange
        else:
            return "#dc3545"  # Red

    def _get_relevance_color(self, score: float) -> str:
        """Get color based on relevance score.

        Args:
            score: Relevance score

        Returns:
            Color hex code
        """
        if score >= 0.8:
            return "#1f77b4"  # Dark Blue
        elif score >= 0.6:
            return "#ff7f0e"  # Orange
        elif score >= 0.4:
            return "#2ca02c"  # Green
        else:
            return "#d62728"  # Red

"""
Unit tests for the smart search engine functionality.
Tests exact match, fuzzy search, semantic search, and filtering capabilities.
"""

import pytest
from unittest.mock import Mock, patch
from typing import List, Dict, Any

# Import the classes we'll be testing (they don't exist yet - TDD approach)
from src.core.search import SmartSearchEngine, SearchResult
from src.core.models import DocumentElement


class TestSmartSearchEngine:
    """Test suite for SmartSearchEngine class."""

    @pytest.fixture
    def sample_elements(self):
        """Create sample DocumentElements for testing."""
        elements = [
            DocumentElement(
                text="alpha beta gamma",
                element_type="text",
                page_number=1,
                bbox={"x0": 0, "y0": 0, "x1": 10, "y1": 10},
                confidence=0.9,
                metadata={},
            ),
            DocumentElement(
                text="alpha",
                element_type="heading",
                page_number=1,
                bbox={"x0": 0, "y0": 20, "x1": 10, "y1": 30},
                confidence=0.95,
                metadata={},
            ),
            DocumentElement(
                text="alpah beta",  # Typo for fuzzy matching test
                element_type="text",
                page_number=1,
                bbox={"x0": 0, "y0": 40, "x1": 10, "y1": 50},
                confidence=0.8,
                metadata={},
            ),
            DocumentElement(
                text="revenue table data",
                element_type="table",
                page_number=1,
                bbox={"x0": 0, "y0": 60, "x1": 100, "y1": 80},
                confidence=0.9,
                metadata={"rows": 3, "columns": 2},
            ),
            DocumentElement(
                text="paragraph content",
                element_type="text",
                page_number=2,
                bbox={"x0": 0, "y0": 0, "x1": 10, "y1": 10},
                confidence=0.85,
                metadata={},
            ),
            DocumentElement(
                text="chart visualization",
                element_type="image",
                page_number=2,
                bbox={"x0": 0, "y0": 20, "x1": 50, "y1": 70},
                confidence=0.9,
                metadata={"caption": "Data visualization chart"},
            ),
        ]
        return elements

    def test_search_engine_initialization(self, sample_elements):
        """Test search engine initializes with elements."""
        engine = SmartSearchEngine(sample_elements)

        assert engine is not None
        assert hasattr(engine, "elements")
        assert len(engine.elements) == len(sample_elements)

    def test_search_engine_empty_initialization(self):
        """Test search engine handles empty element list."""
        engine = SmartSearchEngine([])

        assert engine is not None
        assert len(engine.elements) == 0

        results = engine.search("any query")
        assert isinstance(results, list)
        assert len(results) == 0

    def test_exact_match_search(self, sample_elements):
        """Test exact text matching returns highest ranked results."""
        engine = SmartSearchEngine(sample_elements)

        results = engine.search("alpha beta")

        assert len(results) > 0
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)

        # First result should be exact match with highest score
        assert results[0].element.text == "alpha beta gamma"
        assert results[0].score > 0.8  # High confidence for exact match
        assert results[0].match_type == "exact"

    def test_single_word_exact_match(self, sample_elements):
        """Test single word exact matching."""
        engine = SmartSearchEngine(sample_elements)

        results = engine.search("alpha")

        assert len(results) >= 2  # Should match "alpha" and "alpha beta gamma"

        # Results should be sorted by relevance (score descending)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

        # Single word "alpha" should rank higher than partial match
        alpha_only = next((r for r in results if r.element.text == "alpha"), None)
        assert alpha_only is not None
        assert alpha_only.score >= 0.9  # High score for exact match

    def test_fuzzy_matching(self, sample_elements):
        """Test fuzzy matching finds similar text with typos."""
        engine = SmartSearchEngine(sample_elements)

        # Search for "alpha" should also match "alpah" (typo)
        results = engine.search("alpha", enable_fuzzy=True, fuzzy_threshold=0.7)

        # Should find both exact and fuzzy matches
        texts = [r.element.text for r in results]
        assert any("alpah" in text for text in texts)

        # Fuzzy matches should have lower scores than exact matches
        fuzzy_result = next((r for r in results if "alpah" in r.element.text), None)
        exact_result = next((r for r in results if r.element.text == "alpha"), None)

        assert fuzzy_result is not None
        assert exact_result is not None
        assert exact_result.score > fuzzy_result.score
        assert fuzzy_result.match_type == "fuzzy"

    def test_fuzzy_threshold_filtering(self, sample_elements):
        """Test that fuzzy threshold properly filters results."""
        engine = SmartSearchEngine(sample_elements)

        # High threshold should exclude poor matches
        results_strict = engine.search("alpha", enable_fuzzy=True, fuzzy_threshold=0.9)
        results_loose = engine.search("alpha", enable_fuzzy=True, fuzzy_threshold=0.5)

        # Loose threshold should return more results
        assert len(results_loose) >= len(results_strict)

        # All results should meet minimum threshold
        for result in results_strict:
            assert result.score >= 0.9

    def test_element_type_filtering(self, sample_elements):
        """Test filtering by element type."""
        engine = SmartSearchEngine(sample_elements)

        # Search only in tables
        table_results = engine.search("revenue", element_types=["table"])

        assert len(table_results) > 0
        assert all(r.element.element_type == "table" for r in table_results)

        # Should find the revenue table
        assert any("revenue" in r.element.text for r in table_results)

    def test_multiple_element_type_filtering(self, sample_elements):
        """Test filtering with multiple element types."""
        engine = SmartSearchEngine(sample_elements)

        # Search in both text and headings
        results = engine.search("alpha", element_types=["text", "heading"])

        assert len(results) > 0
        valid_types = {"text", "heading"}
        assert all(r.element.element_type in valid_types for r in results)

        # Should not include table or image elements
        assert not any(r.element.element_type in ["table", "image"] for r in results)

    def test_page_number_filtering(self, sample_elements):
        """Test filtering by page number."""
        engine = SmartSearchEngine(sample_elements)

        # Search only on page 1
        page1_results = engine.search("alpha", page_numbers=[1])

        assert len(page1_results) > 0
        assert all(r.element.page_number == 1 for r in page1_results)

        # Search on page 2 should return different results
        page2_results = engine.search("content", page_numbers=[2])

        assert len(page2_results) > 0
        assert all(r.element.page_number == 2 for r in page2_results)

    def test_multiple_page_filtering(self, sample_elements):
        """Test filtering with multiple pages."""
        engine = SmartSearchEngine(sample_elements)

        # Search on both pages
        multi_page_results = engine.search("text", page_numbers=[1, 2])

        valid_pages = {1, 2}
        assert all(r.element.page_number in valid_pages for r in multi_page_results)

    def test_region_based_search(self, sample_elements):
        """Test searching within specific coordinate regions."""
        engine = SmartSearchEngine(sample_elements)

        # Search in upper region (y < 50)
        region = {"x0": 0, "y0": 0, "x1": 100, "y1": 50}
        region_results = engine.search("alpha", region=region)

        assert len(region_results) > 0

        # All results should overlap with the region
        for result in region_results:
            bbox = result.element.bbox
            assert bbox["y0"] < 50 or bbox["y1"] < 50  # Some overlap with region

    def test_region_overlap_vs_containment(self, sample_elements):
        """Test that region search uses overlap by default, not strict containment."""
        engine = SmartSearchEngine(sample_elements)

        # Small region that partially overlaps with elements
        small_region = {"x0": 0, "y0": 0, "x1": 5, "y1": 5}
        overlap_results = engine.search("alpha", region=small_region, strict_containment=False)
        contained_results = engine.search("alpha", region=small_region, strict_containment=True)

        # Overlap should return more results than strict containment
        assert len(overlap_results) >= len(contained_results)

    def test_ranking_exact_over_fuzzy(self, sample_elements):
        """Test that exact matches rank higher than fuzzy matches."""
        engine = SmartSearchEngine(sample_elements)

        results = engine.search("alpha", enable_fuzzy=True)

        # Find exact and fuzzy matches
        exact_matches = [r for r in results if r.match_type == "exact"]
        fuzzy_matches = [r for r in results if r.match_type == "fuzzy"]

        if exact_matches and fuzzy_matches:
            # Best exact match should outrank best fuzzy match
            best_exact = max(exact_matches, key=lambda x: x.score)
            best_fuzzy = max(fuzzy_matches, key=lambda x: x.score)
            assert best_exact.score > best_fuzzy.score

    def test_heading_boost(self, sample_elements):
        """Test that headings receive ranking boost."""
        engine = SmartSearchEngine(sample_elements)

        results = engine.search("alpha")

        # Find heading and text results
        heading_results = [r for r in results if r.element.element_type == "heading"]
        text_results = [r for r in results if r.element.element_type == "text"]

        if heading_results and text_results:
            # Heading with same content should rank higher
            alpha_heading = next((r for r in heading_results if r.element.text == "alpha"), None)
            if alpha_heading:
                # Should have boost applied
                assert alpha_heading.score > 0.9

    def test_confidence_score_integration(self, sample_elements):
        """Test that element confidence affects search ranking."""
        engine = SmartSearchEngine(sample_elements)

        results = engine.search("alpha")

        # Results should consider element confidence in scoring
        for result in results:
            # Search score should be influenced by element confidence
            assert result.score <= 1.0
            assert result.score > 0.0

    def test_search_result_structure(self, sample_elements):
        """Test that search results have expected structure."""
        engine = SmartSearchEngine(sample_elements)

        results = engine.search("alpha")

        for result in results:
            assert hasattr(result, "element")
            assert hasattr(result, "score")
            assert hasattr(result, "match_type")
            assert hasattr(result, "matched_text")

            assert isinstance(result.element, DocumentElement)
            assert isinstance(result.score, (int, float))
            assert isinstance(result.match_type, str)
            assert isinstance(result.matched_text, str)

            assert 0.0 <= result.score <= 1.0
            assert result.match_type in ["exact", "fuzzy", "semantic"]

    def test_case_insensitive_search(self, sample_elements):
        """Test that search is case insensitive."""
        engine = SmartSearchEngine(sample_elements)

        lower_results = engine.search("alpha")
        upper_results = engine.search("ALPHA")
        mixed_results = engine.search("Alpha")

        # Should return same results regardless of case
        assert len(lower_results) == len(upper_results) == len(mixed_results)

        # Scores should be identical
        lower_scores = [r.score for r in lower_results]
        upper_scores = [r.score for r in upper_results]
        assert lower_scores == upper_scores

    def test_whitespace_normalization(self, sample_elements):
        """Test that extra whitespace doesn't affect search."""
        engine = SmartSearchEngine(sample_elements)

        normal_results = engine.search("alpha beta")
        spaced_results = engine.search("  alpha   beta  ")

        # Should return same results
        assert len(normal_results) == len(spaced_results)
        assert [r.score for r in normal_results] == [r.score for r in spaced_results]

    def test_empty_search_query(self, sample_elements):
        """Test handling of empty search queries."""
        engine = SmartSearchEngine(sample_elements)

        empty_results = engine.search("")
        whitespace_results = engine.search("   ")

        # Should return empty results for empty queries
        assert len(empty_results) == 0
        assert len(whitespace_results) == 0

    def test_no_matches_found(self, sample_elements):
        """Test behavior when no matches are found."""
        engine = SmartSearchEngine(sample_elements)

        results = engine.search("nonexistent_term_xyz123")

        assert isinstance(results, list)
        assert len(results) == 0

    def test_limit_results(self, sample_elements):
        """Test limiting number of search results."""
        engine = SmartSearchEngine(sample_elements)

        unlimited_results = engine.search("alpha")
        limited_results = engine.search("alpha", limit=2)

        assert len(limited_results) <= 2
        assert len(limited_results) <= len(unlimited_results)

        # Limited results should be the top-ranked ones
        if len(unlimited_results) >= 2:
            for i in range(len(limited_results)):
                assert limited_results[i].score == unlimited_results[i].score

    def test_search_in_metadata(self, sample_elements):
        """Test searching in element metadata."""
        engine = SmartSearchEngine(sample_elements)

        # Search for metadata content
        results = engine.search("caption", include_metadata=True)

        # Should find elements with matching metadata
        caption_results = [r for r in results if "caption" in r.matched_text.lower()]
        assert len(caption_results) > 0

    def test_semantic_search_placeholder(self, sample_elements):
        """Test semantic search capability (placeholder for future implementation)."""
        engine = SmartSearchEngine(sample_elements)

        # This test will initially fail until semantic search is implemented
        try:
            results = engine.semantic_search("financial data", top_k=3)
            assert isinstance(results, list)
            assert len(results) <= 3
        except NotImplementedError:
            # Acceptable for initial implementation
            pytest.skip("Semantic search not yet implemented")

    def test_search_performance(self, sample_elements):
        """Test that search completes within reasonable time."""
        import time

        # Create larger dataset
        large_dataset = sample_elements * 100  # 600 elements
        engine = SmartSearchEngine(large_dataset)

        start_time = time.time()
        results = engine.search("alpha")
        end_time = time.time()

        search_time = end_time - start_time

        # Should complete within 1 second for 600 elements
        assert search_time < 1.0
        assert len(results) > 0

    def test_concurrent_searches(self, sample_elements):
        """Test that engine handles concurrent searches correctly."""
        import threading
        import time

        engine = SmartSearchEngine(sample_elements)
        results_list = []

        def search_worker(query, results_container):
            results = engine.search(query)
            results_container.append(results)

        threads = []
        for query in ["alpha", "beta", "gamma", "revenue"]:
            thread = threading.Thread(target=search_worker, args=(query, results_list))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All searches should complete successfully
        assert len(results_list) == 4
        assert all(isinstance(results, list) for results in results_list)


@pytest.fixture
def mock_search_result():
    """Create a mock SearchResult for testing."""
    element = Mock(spec=DocumentElement)
    element.text = "sample text"
    element.element_type = "text"
    element.page_number = 1
    element.bbox = {"x0": 0, "y0": 0, "x1": 10, "y1": 10}
    element.confidence = 0.9

    return SearchResult(element=element, score=0.85, match_type="exact", matched_text="sample text")


def make_element(
    text: str, page: int = 1, element_type: str = "text", confidence: float = 0.9
) -> DocumentElement:
    """Helper function to create DocumentElements for testing."""
    return DocumentElement(
        text=text,
        element_type=element_type,
        page_number=page,
        bbox={"x0": 0, "y0": 0, "x1": 10, "y1": 10},
        confidence=confidence,
        metadata={},
    )

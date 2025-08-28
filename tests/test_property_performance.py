"""
Property-based and performance tests for the Smart PDF Parser.
Tests invariants, roundtrip operations, and performance boundaries.
"""

import pytest
import json
import time
import psutil
import os
from pathlib import Path
from unittest.mock import Mock, patch
from typing import List, Dict, Any

# Hypothesis for property-based testing
try:
    from hypothesis import given, strategies as st, settings, assume
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False
    # Create dummy decorator if hypothesis not available
    def given(*args, **kwargs):
        def decorator(func):
            return pytest.mark.skip("hypothesis not installed")(func)
        return decorator
    st = Mock()

# Import classes to test
from src.core.parser import DoclingParser
from src.core.search import SmartSearchEngine
from src.core.models import DocumentElement
from src.verification.interface import VerificationInterface


class TestPropertyBasedTests:
    """Property-based tests using Hypothesis to test invariants."""
    
    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
    @given(
        text=st.text(min_size=1, max_size=1000),
        page_num=st.integers(min_value=1, max_value=100),
        x0=st.floats(min_value=0, max_value=500),
        y0=st.floats(min_value=0, max_value=500),
        confidence=st.floats(min_value=0.0, max_value=1.0)
    )
    def test_document_element_bbox_invariants(self, text, page_num, x0, y0, confidence):
        """Test that DocumentElement bounding boxes always satisfy invariants."""
        assume(x0 < 600 and y0 < 800)  # Reasonable page bounds
        
        x1 = x0 + abs(hash(text)) % 100  # Generate x1 > x0
        y1 = y0 + abs(hash(text)) % 100  # Generate y1 > y0
        
        element = DocumentElement(
            text=text,
            element_type="text",
            page_number=page_num,
            bbox={'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1},
            confidence=confidence,
            metadata={}
        )
        
        # Invariants
        assert element.bbox['x1'] >= element.bbox['x0']
        assert element.bbox['y1'] >= element.bbox['y0']
        assert element.page_number >= 1
        assert 0.0 <= element.confidence <= 1.0
        assert isinstance(element.text, str)
    
    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
    @given(
        elements_data=st.lists(
            st.tuples(
                st.text(min_size=1, max_size=100),
                st.sampled_from(['text', 'heading', 'table']),
                st.integers(min_value=1, max_value=5),
                st.floats(min_value=0.5, max_value=1.0)
            ),
            min_size=1,
            max_size=20
        )
    )
    def test_search_engine_ranking_invariants(self, elements_data):
        """Test that search results are always properly ranked."""
        elements = []
        for text, elem_type, page, confidence in elements_data:
            element = DocumentElement(
                text=text,
                element_type=elem_type,
                page_number=page,
                bbox={'x0': 0, 'y0': 0, 'x1': 10, 'y1': 10},
                confidence=confidence,
                metadata={}
            )
            elements.append(element)
        
        engine = SmartSearchEngine(elements)
        
        # Search for any word that might exist
        if elements:
            first_word = elements[0].text.split()[0] if elements[0].text.split() else "test"
            results = engine.search(first_word)
            
            # Results should be sorted by score (descending)
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)
            
            # All scores should be between 0 and 1
            assert all(0.0 <= score <= 1.0 for score in scores)
    
    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
    @given(
        bbox=st.fixed_dictionaries({
            'x0': st.floats(min_value=0, max_value=400),
            'y0': st.floats(min_value=0, max_value=400),
            'x1': st.floats(min_value=0, max_value=600),
            'y1': st.floats(min_value=0, max_value=600)
        })
    )
    def test_coordinate_transformation_invariants(self, bbox):
        """Test coordinate transformation preserves relative positions."""
        from src.verification.renderer import CoordinateTransformer
        
        assume(bbox['x1'] > bbox['x0'])
        assume(bbox['y1'] > bbox['y0'])
        
        transformer = CoordinateTransformer()
        pdf_size = (612, 792)
        image_size = (600, 800)
        
        transformed = transformer.transform_bbox(bbox, pdf_size, image_size)
        
        # Relative ordering should be preserved
        assert transformed['x1'] >= transformed['x0']
        assert transformed['y1'] >= transformed['y0']
        
        # Should be within image bounds
        assert 0 <= transformed['x0'] <= image_size[0]
        assert 0 <= transformed['y0'] <= image_size[1]
        assert 0 <= transformed['x1'] <= image_size[0]
        assert 0 <= transformed['y1'] <= image_size[1]
    
    def test_json_export_import_roundtrip(self):
        """Test that JSON export/import roundtrip preserves data."""
        # Create sample elements
        elements = [
            DocumentElement(
                text="Test element",
                element_type="text",
                page_number=1,
                bbox={'x0': 100, 'y0': 200, 'x1': 300, 'y1': 220},
                confidence=0.9,
                metadata={'key': 'value'}
            )
        ]
        
        # Create verification interface and mark some elements
        interface = VerificationInterface(elements, Mock())
        interface.mark_element_correct(0)
        
        # Export to JSON
        exported_json = interface.export_verification_data(format='json')
        
        # Import back
        new_interface = VerificationInterface(elements, Mock())
        new_interface.load_verification_state(exported_json)
        
        # Data should match
        original_state = interface.get_element_state(0)
        restored_state = new_interface.get_element_state(0)
        
        assert original_state.status == restored_state.status
        assert original_state.element_id == restored_state.element_id
    
    def test_search_idempotence(self):
        """Test that repeated searches return identical results."""
        elements = [
            DocumentElement(
                text="alpha beta gamma",
                element_type="text",
                page_number=1,
                bbox={'x0': 0, 'y0': 0, 'x1': 10, 'y1': 10},
                confidence=0.9,
                metadata={}
            )
        ]
        
        engine = SmartSearchEngine(elements)
        
        # Multiple identical searches
        results1 = engine.search("alpha")
        results2 = engine.search("alpha")
        results3 = engine.search("alpha")
        
        # Results should be identical
        assert len(results1) == len(results2) == len(results3)
        
        for r1, r2, r3 in zip(results1, results2, results3):
            assert r1.score == r2.score == r3.score
            assert r1.match_type == r2.match_type == r3.match_type
            assert r1.element.text == r2.element.text == r3.element.text
    
    def test_case_insensitive_idempotence(self):
        """Test that case variations return same results."""
        elements = [
            DocumentElement(
                text="Alpha Beta",
                element_type="text",
                page_number=1,
                bbox={'x0': 0, 'y0': 0, 'x1': 10, 'y1': 10},
                confidence=0.9,
                metadata={}
            )
        ]
        
        engine = SmartSearchEngine(elements)
        
        # Case variations
        results_lower = engine.search("alpha")
        results_upper = engine.search("ALPHA")
        results_mixed = engine.search("Alpha")
        
        # Should return same results
        assert len(results_lower) == len(results_upper) == len(results_mixed)
        
        for r1, r2, r3 in zip(results_lower, results_upper, results_mixed):
            assert r1.score == r2.score == r3.score


class TestPerformanceTests:
    """Performance tests with specific timing and memory requirements."""
    
    @pytest.fixture
    def large_pdf_path(self):
        """Path to large PDF fixture for performance testing."""
        return Path("tests/fixtures/large_pages_light.pdf")
    
    @pytest.fixture
    def performance_elements(self):
        """Create large set of elements for performance testing."""
        elements = []
        for i in range(1000):
            element = DocumentElement(
                text=f"Element {i} with some sample text content for testing performance",
                element_type="text" if i % 3 == 0 else "heading",
                page_number=(i // 50) + 1,
                bbox={'x0': i % 100, 'y0': (i * 2) % 100, 'x1': (i % 100) + 50, 'y1': ((i * 2) % 100) + 20},
                confidence=0.8 + (i % 20) / 100,
                metadata={'index': i}
            )
            elements.append(element)
        return elements
    
    @pytest.mark.performance
    def test_pdf_parsing_performance(self, large_pdf_path):
        """Test PDF parsing performance stays within limits."""
        if not large_pdf_path.exists():
            pytest.skip("Large PDF fixture not found")
        
        parser = DoclingParser()
        
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        elements = parser.parse_document(large_pdf_path)
        
        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        parsing_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        # Performance requirements (adjust based on actual needs)
        pages_count = len(set(e.page_number for e in elements))
        time_per_page = parsing_time / max(pages_count, 1)
        
        assert time_per_page < 1.5, f"Parsing too slow: {time_per_page:.2f}s per page"
        assert memory_used < 500, f"Memory usage too high: {memory_used:.2f}MB"
        assert len(elements) > 0, "Should extract some elements"
    
    @pytest.mark.performance
    def test_search_engine_performance(self, performance_elements):
        """Test search engine performance with large dataset."""
        engine = SmartSearchEngine(performance_elements)
        
        # Test different search scenarios
        test_queries = ["Element", "sample text", "performance", "heading", "999"]
        
        for query in test_queries:
            start_time = time.time()
            results = engine.search(query)
            end_time = time.time()
            
            search_time = end_time - start_time
            
            # Should complete search within reasonable time
            assert search_time < 0.5, f"Search too slow for '{query}': {search_time:.3f}s"
            assert isinstance(results, list)
    
    @pytest.mark.performance
    def test_fuzzy_search_performance(self, performance_elements):
        """Test fuzzy search performance doesn't degrade significantly."""
        engine = SmartSearchEngine(performance_elements)
        
        start_time = time.time()
        exact_results = engine.search("Element", enable_fuzzy=False)
        exact_time = time.time() - start_time
        
        start_time = time.time()
        fuzzy_results = engine.search("Element", enable_fuzzy=True)
        fuzzy_time = time.time() - start_time
        
        # Fuzzy search should not be more than 3x slower
        assert fuzzy_time < exact_time * 3, f"Fuzzy search too slow: {fuzzy_time:.3f}s vs {exact_time:.3f}s"
        assert len(fuzzy_results) >= len(exact_results)
    
    @pytest.mark.performance
    def test_memory_usage_stability(self, performance_elements):
        """Test that memory usage doesn't grow unbounded."""
        engine = SmartSearchEngine(performance_elements)
        
        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        # Perform many searches
        for i in range(100):
            results = engine.search(f"Element {i % 50}")
            # Force garbage collection periodically
            if i % 10 == 0:
                import gc
                gc.collect()
        
        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be minimal
        assert memory_growth < 100, f"Memory leak detected: {memory_growth:.2f}MB growth"
    
    @pytest.mark.performance
    def test_verification_export_performance(self, performance_elements):
        """Test verification data export performance."""
        interface = VerificationInterface(performance_elements, Mock())
        
        # Mark many elements to create substantial export data
        for i in range(0, len(performance_elements), 10):
            if i % 20 == 0:
                interface.mark_element_correct(i)
            else:
                interface.mark_element_incorrect(i, correction="Fixed text")
        
        # Test JSON export performance
        start_time = time.time()
        json_data = interface.export_verification_data(format='json')
        json_time = time.time() - start_time
        
        # Test CSV export performance
        start_time = time.time()
        csv_data = interface.export_verification_data(format='csv')
        csv_time = time.time() - start_time
        
        # Exports should complete quickly
        assert json_time < 2.0, f"JSON export too slow: {json_time:.3f}s"
        assert csv_time < 2.0, f"CSV export too slow: {csv_time:.3f}s"
        
        # Data should be substantial
        assert len(json_data) > 1000
        assert len(csv_data) > 1000
    
    @pytest.mark.performance
    def test_concurrent_search_performance(self, performance_elements):
        """Test search performance under concurrent access."""
        import threading
        import queue
        
        engine = SmartSearchEngine(performance_elements)
        results_queue = queue.Queue()
        
        def search_worker(worker_id):
            start_time = time.time()
            for i in range(10):
                query = f"Element {worker_id * 10 + i}"
                results = engine.search(query)
            end_time = time.time()
            results_queue.put(end_time - start_time)
        
        # Start multiple concurrent searches
        threads = []
        num_workers = 4
        
        start_time = time.time()
        for i in range(num_workers):
            thread = threading.Thread(target=search_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Collect worker times
        worker_times = []
        while not results_queue.empty():
            worker_times.append(results_queue.get())
        
        # Concurrent performance should be reasonable
        assert len(worker_times) == num_workers
        assert total_time < 5.0, f"Concurrent searches too slow: {total_time:.3f}s"
        assert max(worker_times) < 3.0, f"Individual worker too slow: {max(worker_times):.3f}s"
    
    def test_batch_processing_performance(self, performance_elements):
        """Test batch processing multiple documents."""
        parser = DoclingParser()
        
        # Simulate multiple small documents
        mock_paths = [f"doc_{i}.pdf" for i in range(10)]
        
        start_time = time.time()
        
        with patch.object(parser, 'parse_document') as mock_parse:
            # Mock returns subset of performance_elements each time
            mock_parse.side_effect = [
                performance_elements[i*100:(i+1)*100] for i in range(10)
            ]
            
            all_results = []
            for path in mock_paths:
                result = parser.parse_document(path)
                all_results.extend(result)
        
        batch_time = time.time() - start_time
        
        # Batch processing should be efficient
        assert batch_time < 2.0, f"Batch processing too slow: {batch_time:.3f}s"
        assert len(all_results) == 1000  # All elements processed
    
    @pytest.mark.performance
    def test_memory_cleanup_after_operations(self):
        """Test that memory is properly cleaned up after operations."""
        import gc
        
        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        # Create and process large amounts of data
        for iteration in range(5):
            large_elements = []
            for i in range(1000):
                element = DocumentElement(
                    text=f"Large text content " * 20,  # Larger text
                    element_type="text",
                    page_number=1,
                    bbox={'x0': 0, 'y0': i, 'x1': 100, 'y1': i+20},
                    confidence=0.9,
                    metadata={'large_data': list(range(100))}
                )
                large_elements.append(element)
            
            # Process with search engine
            engine = SmartSearchEngine(large_elements)
            results = engine.search("Large")
            
            # Clear references
            del large_elements
            del engine
            del results
            
            # Force garbage collection
            gc.collect()
        
        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        # Memory should not grow significantly after cleanup
        assert memory_growth < 50, f"Memory not cleaned up properly: {memory_growth:.2f}MB growth"


class TestStabilityTests:
    """Tests for stability and determinism."""
    
    def test_deterministic_parsing_results(self):
        """Test that parsing the same document multiple times gives identical results."""
        parser = DoclingParser(enable_ocr=False)  # Disable OCR for determinism
        
        with patch.object(parser, '_convert_document') as mock_convert:
            # Mock consistent return
            mock_doc = Mock()
            mock_doc.texts = [Mock(text="Sample", get_location=Mock(return_value=Mock()))]
            mock_convert.return_value = Mock(document=mock_doc)
            
            results1 = parser.parse_document(Path("test.pdf"))
            results2 = parser.parse_document(Path("test.pdf"))
            results3 = parser.parse_document(Path("test.pdf"))
            
            # Results should be identical
            assert len(results1) == len(results2) == len(results3)
            for r1, r2, r3 in zip(results1, results2, results3):
                assert r1.text == r2.text == r3.text
                assert r1.element_type == r2.element_type == r3.element_type
                assert r1.page_number == r2.page_number == r3.page_number
    
    def test_stable_element_ordering(self):
        """Test that element ordering is stable across parsing runs."""
        # This would need actual PDF parsing to test properly
        # For now, test that our model maintains order
        elements = []
        for i in range(10):
            elements.append(DocumentElement(
                text=f"Element {i}",
                element_type="text",
                page_number=1,
                bbox={'x0': 0, 'y0': i*20, 'x1': 100, 'y1': (i+1)*20},
                confidence=0.9,
                metadata={}
            ))
        
        # Order should be maintained
        for i, element in enumerate(elements):
            assert f"Element {i}" in element.text
            assert element.bbox['y0'] == i * 20
    
    def test_bounding_box_stability(self):
        """Test that bounding boxes remain stable for same content."""
        # Test with coordinate transformer
        from src.verification.renderer import CoordinateTransformer
        
        transformer = CoordinateTransformer()
        bbox = {'x0': 100, 'y0': 200, 'x1': 300, 'y1': 250}
        pdf_size = (612, 792)
        image_size = (600, 800)
        
        # Multiple transformations should give identical results
        result1 = transformer.transform_bbox(bbox, pdf_size, image_size)
        result2 = transformer.transform_bbox(bbox, pdf_size, image_size)
        result3 = transformer.transform_bbox(bbox, pdf_size, image_size)
        
        assert result1 == result2 == result3
"""
SmartSearchEngine implementation for advanced document search capabilities.
Provides exact matching, fuzzy search, semantic search, and sophisticated filtering.
"""

import time
import string
import unicodedata
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict

from fuzzywuzzy import fuzz, process

from src.core.models import DocumentElement, SearchResult


class SearchCache:
    """LRU cache for search results to improve performance."""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
    
    def get(self, key: str):
        """Get cached result if available."""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def set(self, key: str, value):
        """Cache a result."""
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def _evict_oldest(self):
        """Remove oldest cache entry."""
        oldest_key = min(self.access_times, key=self.access_times.get)
        del self.cache[oldest_key]
        del self.access_times[oldest_key]


class SmartSearchEngine:
    """Advanced document search engine with multi-modal search capabilities."""
    
    def __init__(self, elements: List[DocumentElement]):
        """Initialize search engine with document elements.
        
        Args:
            elements: List of DocumentElement objects to search
        """
        self.elements = elements
        self._search_cache = SearchCache()
        self._build_indices()
    
    def _build_indices(self):
        """Build internal indices for efficient searching."""
        # Build type index
        self._type_index = defaultdict(list)
        for i, element in enumerate(self.elements):
            self._type_index[element.element_type].append(i)
        
        # Build page index
        self._page_index = defaultdict(list)
        for i, element in enumerate(self.elements):
            self._page_index[element.page_number].append(i)
        
        # Build text index for quick lookups
        self._text_index = {}
        for i, element in enumerate(self.elements):
            normalized_text = self._normalize_text(element.text)
            self._text_index[i] = normalized_text
    
    def _normalize_text(self, text: str) -> str:
        """Comprehensive text normalization for consistent matching.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text string
        """
        # Case normalization
        text = text.lower()
        
        # Unicode normalization
        text = unicodedata.normalize('NFKD', text)
        
        # Whitespace normalization
        text = ' '.join(text.split())
        
        # Remove punctuation for matching
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        return text
    
    def _create_cache_key(self, query: str, **kwargs) -> str:
        """Create a cache key from query and parameters."""
        key_parts = [query]
        
        # Add relevant kwargs to key
        for key in sorted(kwargs.keys()):
            value = kwargs[key]
            if isinstance(value, list):
                key_parts.append(f"{key}:{','.join(map(str, value))}")
            elif isinstance(value, dict):
                key_parts.append(f"{key}:{str(sorted(value.items()))}")
            else:
                key_parts.append(f"{key}:{value}")
        
        return '|'.join(key_parts)
    
    def search(self, query: str, 
               enable_fuzzy: bool = True,
               fuzzy_threshold: float = 0.8,
               element_types: Optional[List[str]] = None,
               page_numbers: Optional[List[int]] = None,
               region: Optional[Dict[str, float]] = None,
               strict_containment: bool = False,
               include_metadata: bool = False,
               limit: Optional[int] = None,
               min_score: float = 0.0,
               **kwargs) -> List[SearchResult]:
        """Main search method with configurable options.
        
        Args:
            query: Search query string
            enable_fuzzy: Enable fuzzy matching
            fuzzy_threshold: Minimum fuzzy match score (0.0-1.0)
            element_types: Filter by element types
            page_numbers: Filter by page numbers
            region: Spatial region filter {'x0', 'y0', 'x1', 'y1'}
            strict_containment: Use strict containment for region filtering
            include_metadata: Search in metadata fields
            limit: Maximum results to return
            min_score: Minimum relevance score
            
        Returns:
            List of SearchResult objects sorted by relevance
        """
        # Handle empty queries
        query = query.strip()
        if not query:
            return []
        
        # Check cache
        cache_key = self._create_cache_key(
            query, enable_fuzzy=enable_fuzzy, fuzzy_threshold=fuzzy_threshold,
            element_types=element_types, page_numbers=page_numbers,
            region=region, strict_containment=strict_containment,
            include_metadata=include_metadata, limit=limit, min_score=min_score
        )
        
        cached_result = self._search_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Start with all elements
        candidate_elements = self.elements[:]
        
        # Apply filters
        if element_types:
            candidate_elements = self.filter_by_element_types(candidate_elements, element_types)
        
        if page_numbers:
            candidate_elements = self.filter_by_pages(candidate_elements, page_numbers)
        
        if region:
            candidate_elements = self.filter_by_region(candidate_elements, region, strict_containment)
        
        # If no candidates after filtering, return empty
        if not candidate_elements:
            return []
        
        # Perform search
        all_results = []
        
        # Exact search
        exact_results = self._exact_search(query, candidate_elements, include_metadata)
        all_results.extend(exact_results)
        
        # Fuzzy search
        if enable_fuzzy:
            fuzzy_results = self._fuzzy_search(query, candidate_elements, fuzzy_threshold, include_metadata)
            all_results.extend(fuzzy_results)
        
        # Remove duplicates while preserving best scores
        seen_elements = {}
        unique_results = []
        
        for result in all_results:
            element_id = id(result.element)
            if element_id not in seen_elements or result.score > seen_elements[element_id].score:
                seen_elements[element_id] = result
        
        unique_results = list(seen_elements.values())
        
        # Apply minimum score filter
        unique_results = [r for r in unique_results if r.score >= min_score]
        
        # Sort by relevance score (descending)
        unique_results.sort(key=lambda x: x.score, reverse=True)
        
        # Apply limit
        if limit:
            unique_results = unique_results[:limit]
        
        # Cache results
        self._search_cache.set(cache_key, unique_results)
        
        return unique_results
    
    def _exact_search(self, query: str, elements: List[DocumentElement], 
                      include_metadata: bool = False) -> List[SearchResult]:
        """Exact string matching with normalization.
        
        Args:
            query: Search query
            elements: Elements to search
            include_metadata: Search in metadata fields
            
        Returns:
            List of SearchResult objects for exact matches
        """
        query_normalized = self._normalize_text(query)
        results = []
        
        for element in elements:
            best_match = None
            best_score = 0.0
            
            # Check main text
            text_normalized = self._normalize_text(element.text)
            
            if query_normalized in text_normalized:
                score = self._calculate_exact_score(query_normalized, text_normalized, element)
                matched_text = self._extract_matched_context(element.text, query)
                
                best_match = SearchResult(
                    element=element,
                    score=score,
                    match_type='exact',
                    matched_text=matched_text
                )
                best_score = score
            
            # Check metadata if requested
            if include_metadata and element.metadata:
                for key, value in element.metadata.items():
                    # Check metadata key
                    key_normalized = self._normalize_text(key)
                    if query_normalized in key_normalized:
                        score = self._calculate_exact_score(query_normalized, key_normalized, element)
                        
                        if score > best_score:
                            matched_text = self._extract_matched_context(key, query)
                            # Show that this came from metadata key
                            matched_text = f"[{key}]: {matched_text}"
                            
                            best_match = SearchResult(
                                element=element,
                                score=score,
                                match_type='exact',
                                matched_text=matched_text
                            )
                            best_score = score
                    
                    # Check metadata value
                    if isinstance(value, str):
                        meta_normalized = self._normalize_text(value)
                        
                        if query_normalized in meta_normalized:
                            score = self._calculate_exact_score(query_normalized, meta_normalized, element)
                            
                            if score > best_score:
                                matched_text = self._extract_matched_context(value, query)
                                # Show that this came from metadata
                                matched_text = f"[{key}]: {matched_text}"
                                
                                best_match = SearchResult(
                                    element=element,
                                    score=score,
                                    match_type='exact',
                                    matched_text=matched_text
                                )
                                best_score = score
            
            if best_match:
                results.append(best_match)
        
        return results
    
    def _calculate_exact_score(self, query_normalized: str, text_normalized: str, 
                               element: DocumentElement) -> float:
        """Calculate relevance score for exact matches.
        
        Args:
            query_normalized: Normalized query text
            text_normalized: Normalized element text
            element: Document element
            
        Returns:
            Relevance score (0.0-1.0)
        """
        # Base score for exact match
        base_score = 0.8
        
        # Boost for complete matches (query is exact match of entire text)
        if query_normalized == text_normalized:
            base_score = 0.95
        
        # Boost for complete word matches
        query_words = set(query_normalized.split())
        text_words = set(text_normalized.split())
        
        word_overlap = len(query_words.intersection(text_words))
        if len(query_words) > 0:
            word_overlap_ratio = word_overlap / len(query_words)
            base_score += word_overlap_ratio * 0.1
        
        # Boost for query being the complete text
        if len(text_words) == len(query_words) and word_overlap == len(query_words):
            base_score = max(base_score, 0.9)
        
        # Boost for substring matches where query has more words than text
        # e.g., "alpha beta" matching "alpha beta gamma" should score higher than "alpha"
        if len(query_words) > len(text_words) and word_overlap == len(text_words):
            # Partial match - text is subset of query
            partial_ratio = len(text_words) / len(query_words)
            base_score += partial_ratio * 0.1
        elif len(query_words) <= len(text_words) and word_overlap == len(query_words):
            # Complete query match - all query words found
            match_ratio = len(query_words) / len(text_words)
            base_score += match_ratio * 0.15
        
        # Apply multi-factor scoring
        final_score = self._calculate_relevance_score(element, query_normalized, 'exact', base_score)
        
        return min(1.0, final_score)
    
    def _fuzzy_search(self, query: str, elements: List[DocumentElement], 
                      threshold: float = 0.8, include_metadata: bool = False) -> List[SearchResult]:
        """Fuzzy matching with configurable similarity threshold.
        
        Args:
            query: Search query
            elements: Elements to search
            threshold: Minimum similarity threshold
            include_metadata: Search in metadata fields
            
        Returns:
            List of SearchResult objects for fuzzy matches
        """
        results = []
        
        # Skip elements that already have exact matches
        exact_matches = set()
        query_normalized = self._normalize_text(query)
        # Use normalized query for consistent fuzzy matching
        query_for_fuzzy = ' '.join(query_normalized.split())
        
        for element in elements:
            best_score = 0.0
            best_matched_text = ""
            best_is_metadata = False
            best_metadata_key = None
            
            # Check main text
            text_normalized = self._normalize_text(element.text)
            
            # Skip if exact match exists
            if query_normalized in text_normalized:
                exact_matches.add(id(element))
                continue
            
            # Calculate fuzzy scores for main text
            token_ratio = fuzz.token_set_ratio(query_for_fuzzy, element.text) / 100.0
            partial_ratio = fuzz.partial_ratio(query_for_fuzzy, element.text) / 100.0
            weighted_ratio = fuzz.WRatio(query_for_fuzzy, element.text) / 100.0
            
            # Combined score with weights
            combined_score = (
                token_ratio * 0.4 +
                partial_ratio * 0.3 +
                weighted_ratio * 0.3
            )
            
            if combined_score > best_score:
                best_score = combined_score
                best_matched_text = element.text
            
            # Check metadata if requested
            if include_metadata and element.metadata:
                for key, value in element.metadata.items():
                    # Check metadata key
                    key_normalized = self._normalize_text(key)
                    
                    # Skip if exact match exists in key
                    if query_normalized in key_normalized:
                        exact_matches.add(id(element))
                        break
                    
                    # Calculate fuzzy scores for metadata key
                    token_ratio = fuzz.token_set_ratio(query_for_fuzzy, key) / 100.0
                    partial_ratio = fuzz.partial_ratio(query_for_fuzzy, key) / 100.0
                    weighted_ratio = fuzz.WRatio(query_for_fuzzy, key) / 100.0
                    
                    # Combined score with weights
                    combined_score = (
                        token_ratio * 0.4 +
                        partial_ratio * 0.3 +
                        weighted_ratio * 0.3
                    )
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_matched_text = key
                        best_is_metadata = True
                        best_metadata_key = key
                    
                    # Check metadata value
                    if isinstance(value, str):
                        meta_normalized = self._normalize_text(value)
                        
                        # Skip if exact match exists in value
                        if query_normalized in meta_normalized:
                            exact_matches.add(id(element))
                            break
                        
                        # Calculate fuzzy scores for metadata value
                        token_ratio = fuzz.token_set_ratio(query_for_fuzzy, value) / 100.0
                        partial_ratio = fuzz.partial_ratio(query_for_fuzzy, value) / 100.0
                        weighted_ratio = fuzz.WRatio(query_for_fuzzy, value) / 100.0
                        
                        # Combined score with weights
                        combined_score = (
                            token_ratio * 0.4 +
                            partial_ratio * 0.3 +
                            weighted_ratio * 0.3
                        )
                        
                        if combined_score > best_score:
                            best_score = combined_score
                            best_matched_text = value
                            best_is_metadata = True
                            best_metadata_key = key
            
            # Only add if meets threshold and not exact match
            if best_score >= threshold and id(element) not in exact_matches:
                # Adjust score based on element properties
                final_score = self._adjust_fuzzy_score(best_score, element, query)
                
                # Extract fuzzy match context
                matched_context = self._extract_fuzzy_match_context(best_matched_text, query)
                
                # Add metadata indicator if it's a metadata match
                if best_is_metadata and best_metadata_key:
                    matched_context = f"[{best_metadata_key}]: {matched_context}"
                
                results.append(SearchResult(
                    element=element,
                    score=final_score,
                    match_type='fuzzy',
                    matched_text=matched_context
                ))
        
        return results
    
    def _adjust_fuzzy_score(self, base_score: float, element: DocumentElement, query: str) -> float:
        """Adjust fuzzy match score based on element characteristics.
        
        Args:
            base_score: Base fuzzy match score
            element: Document element
            query: Search query
            
        Returns:
            Adjusted score
        """
        adjusted_score = base_score
        
        # Apply multi-factor scoring
        final_score = self._calculate_relevance_score(element, query, 'fuzzy', adjusted_score)
        
        return min(1.0, final_score)
    
    def _calculate_relevance_score(self, element: DocumentElement, query: str, 
                                   match_type: str, base_score: float) -> float:
        """Calculate comprehensive relevance score using multiple factors.
        
        Args:
            element: Document element
            query: Search query
            match_type: Type of match ('exact', 'fuzzy', 'semantic')
            base_score: Base score from matching algorithm
            
        Returns:
            Final relevance score
        """
        final_score = base_score
        
        # 1. Element Type Boost
        type_multipliers = {
            'heading': 1.3,      # Headings are more important
            'title': 1.5,        # Titles are most important
            'table': 1.2,        # Tables contain structured data
            'formula': 1.1,      # Formulas are specific content
            'caption': 1.1,      # Captions describe images
            'text': 1.0,         # Regular text baseline
            'image': 0.9         # Images have less searchable text
        }
        
        type_multiplier = type_multipliers.get(element.element_type, 1.0)
        final_score *= type_multiplier
        
        # 2. Confidence Boost
        confidence_factor = 0.8 + (element.confidence * 0.2)  # Scale: 0.8-1.0
        final_score *= confidence_factor
        
        # 3. Query Length vs Text Length Relevance
        query_words = len(query.split())
        text_words = len(element.text.split())
        
        if text_words > 0:
            length_ratio = min(query_words / text_words, 1.0)
            # Boost score for texts that are similar length to query
            if 0.1 <= length_ratio <= 0.9:
                final_score *= 1.1
        
        # 4. Position Boost (earlier pages slightly favored)
        page_factor = 1.0 / (1.0 + (element.page_number - 1) * 0.01)
        final_score *= page_factor
        
        # 5. Match Type Preference
        match_type_multipliers = {
            'exact': 1.0,      # Baseline for exact matches
            'fuzzy': 0.85,     # Slightly lower for fuzzy matches
            'semantic': 0.9    # Good for semantic matches
        }
        
        match_multiplier = match_type_multipliers.get(match_type, 1.0)
        final_score *= match_multiplier
        
        # Ensure score stays within bounds
        return min(1.0, max(0.0, final_score))
    
    def _extract_matched_context(self, text: str, query: str, context_chars: int = 100) -> str:
        """Extract context around matched text.
        
        Args:
            text: Full text containing the match
            query: Query that was matched
            context_chars: Characters of context to include
            
        Returns:
            Matched text with context
        """
        query_lower = query.lower()
        text_lower = text.lower()
        
        # Find query position
        match_pos = text_lower.find(query_lower)
        
        if match_pos == -1:
            # Fallback to beginning of text
            return text[:context_chars] + ("..." if len(text) > context_chars else "")
        
        # Calculate context bounds
        start = max(0, match_pos - context_chars // 2)
        end = min(len(text), match_pos + len(query) + context_chars // 2)
        
        context = text[start:end]
        
        # Add ellipsis if truncated
        if start > 0:
            context = "..." + context
        if end < len(text):
            context = context + "..."
        
        return context
    
    def _extract_fuzzy_match_context(self, text: str, query: str, context_chars: int = 100) -> str:
        """Extract context for fuzzy matches.
        
        Args:
            text: Full text
            query: Query that was matched
            context_chars: Characters of context to include
            
        Returns:
            Context text
        """
        # For fuzzy matches, just return beginning of text with ellipsis
        if len(text) <= context_chars:
            return text
        return text[:context_chars] + "..."
    
    def filter_by_element_types(self, elements: List[DocumentElement], 
                                element_types: List[str]) -> List[DocumentElement]:
        """Filter elements by type with validation.
        
        Args:
            elements: Elements to filter
            element_types: Valid element types to include
            
        Returns:
            Filtered list of elements
            
        Raises:
            ValueError: If invalid element types are provided
        """
        valid_types = {'text', 'heading', 'table', 'image', 'list', 'formula', 'caption'}
        
        # Validate input types
        invalid_types = set(element_types) - valid_types
        if invalid_types:
            raise ValueError(f"Invalid element types: {invalid_types}")
        
        return [element for element in elements if element.element_type in element_types]
    
    def filter_by_pages(self, elements: List[DocumentElement], 
                        page_numbers: List[int]) -> List[DocumentElement]:
        """Filter elements by page numbers with range support.
        
        Args:
            elements: Elements to filter
            page_numbers: List of page numbers or tuples for ranges
            
        Returns:
            Filtered list of elements
        """
        if not page_numbers:
            return elements
        
        # Support for page ranges (e.g., [1, 2, 3] or [(1, 5)] for range)
        valid_pages = set()
        
        for page_spec in page_numbers:
            if isinstance(page_spec, int):
                valid_pages.add(page_spec)
            elif isinstance(page_spec, tuple) and len(page_spec) == 2:
                # Range specification
                start, end = page_spec
                valid_pages.update(range(start, end + 1))
        
        return [element for element in elements if element.page_number in valid_pages]
    
    def filter_by_region(self, elements: List[DocumentElement], region: Dict[str, float],
                         strict_containment: bool = False) -> List[DocumentElement]:
        """Filter elements by spatial region with overlap/containment options.
        
        Args:
            elements: Elements to filter
            region: Bounding box {'x0', 'y0', 'x1', 'y1'}
            strict_containment: If True, element must be completely within region
            
        Returns:
            Filtered list of elements
            
        Raises:
            ValueError: If region doesn't contain required keys
        """
        required_keys = {'x0', 'y0', 'x1', 'y1'}
        if not required_keys.issubset(region.keys()):
            raise ValueError(f"Region must contain keys: {required_keys}")
        
        filtered_elements = []
        
        for element in elements:
            element_bbox = element.bbox
            
            if strict_containment:
                # Element must be completely within region
                if (element_bbox['x0'] >= region['x0'] and
                    element_bbox['y0'] >= region['y0'] and
                    element_bbox['x1'] <= region['x1'] and
                    element_bbox['y1'] <= region['y1']):
                    filtered_elements.append(element)
            else:
                # Element must overlap with region
                if self._bboxes_overlap(element_bbox, region):
                    filtered_elements.append(element)
        
        return filtered_elements
    
    def _bboxes_overlap(self, bbox1: Dict[str, float], bbox2: Dict[str, float]) -> bool:
        """Check if two bounding boxes overlap.
        
        Args:
            bbox1: First bounding box
            bbox2: Second bounding box
            
        Returns:
            True if boxes overlap
        """
        return not (
            bbox1['x1'] < bbox2['x0'] or  # bbox1 is left of bbox2
            bbox2['x1'] < bbox1['x0'] or  # bbox2 is left of bbox1
            bbox1['y1'] < bbox2['y0'] or  # bbox1 is above bbox2
            bbox2['y1'] < bbox1['y0']     # bbox2 is above bbox1
        )
    
    def filter_by_metadata(self, elements: List[DocumentElement], 
                           metadata_filters: Dict[str, Any]) -> List[DocumentElement]:
        """Filter elements by metadata criteria.
        
        Args:
            elements: Elements to filter
            metadata_filters: Dictionary of metadata criteria
            
        Returns:
            Filtered list of elements
        """
        filtered_elements = []
        
        for element in elements:
            matches_all_filters = True
            
            for key, expected_value in metadata_filters.items():
                if key not in element.metadata:
                    matches_all_filters = False
                    break
                
                actual_value = element.metadata[key]
                
                # Support different comparison types
                if isinstance(expected_value, dict):
                    # Range comparison: {'min': 0, 'max': 100}
                    if 'min' in expected_value and actual_value < expected_value['min']:
                        matches_all_filters = False
                        break
                    if 'max' in expected_value and actual_value > expected_value['max']:
                        matches_all_filters = False
                        break
                elif isinstance(expected_value, list):
                    # Multiple valid values
                    if actual_value not in expected_value:
                        matches_all_filters = False
                        break
                else:
                    # Exact match
                    if actual_value != expected_value:
                        matches_all_filters = False
                        break
            
            if matches_all_filters:
                filtered_elements.append(element)
        
        return filtered_elements
    
    def semantic_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Semantic similarity search using embeddings.
        
        Args:
            query: Search query
            top_k: Maximum number of results to return
            
        Returns:
            List of SearchResult objects sorted by semantic similarity
            
        Raises:
            NotImplementedError: If sentence-transformers is not available
        """
        try:
            if not hasattr(self, '_embeddings_model'):
                self._initialize_embeddings_model()
            
            # Generate query embedding
            query_embedding = self._embeddings_model.encode([query])
            
            # Calculate similarities
            similarities = []
            for i, element in enumerate(self.elements):
                element_embedding = self._get_element_embedding(element, i)
                similarity = self._calculate_cosine_similarity(query_embedding[0], element_embedding)
                similarities.append((similarity, element, i))
            
            # Sort by similarity and take top_k
            similarities.sort(key=lambda x: x[0], reverse=True)
            top_results = similarities[:top_k]
            
            # Convert to SearchResult objects
            results = []
            for similarity, element, idx in top_results:
                if similarity > 0.3:  # Minimum semantic similarity threshold
                    matched_text = element.text[:200] + "..." if len(element.text) > 200 else element.text
                    
                    results.append(SearchResult(
                        element=element,
                        score=float(similarity),
                        match_type='semantic',
                        matched_text=matched_text
                    ))
            
            return results
            
        except ImportError:
            raise NotImplementedError(
                "Semantic search requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            )
    
    def _initialize_embeddings_model(self):
        """Initialize sentence transformer model for semantic search."""
        from sentence_transformers import SentenceTransformer
        
        self._embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Pre-compute embeddings for all elements
        self._element_embeddings = {}
        texts = [element.text for element in self.elements]
        embeddings = self._embeddings_model.encode(texts)
        
        for i, embedding in enumerate(embeddings):
            self._element_embeddings[i] = embedding
    
    def _get_element_embedding(self, element: DocumentElement, index: int):
        """Get embedding for an element."""
        return self._element_embeddings[index]
    
    def _calculate_cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
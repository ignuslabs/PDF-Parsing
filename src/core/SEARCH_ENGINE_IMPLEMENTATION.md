# SmartSearchEngine Implementation Guide

## 🎯 Overview

The `SmartSearchEngine` is an advanced document search system that provides multi-modal search capabilities including exact matching, fuzzy search, semantic search, and sophisticated filtering. It's designed to work with `DocumentElement` objects extracted by the `DoclingParser` and provides intelligent ranking based on relevance, element type, and content quality.

## 🏗️ Architecture

### Core Components

1. **Search Algorithm Engine**: Multi-strategy search implementation
2. **Ranking System**: Intelligent result scoring and ordering
3. **Filter Pipeline**: Advanced filtering by type, page, region, and metadata
4. **Index Management**: Efficient text indexing and retrieval structures

### Search Pipeline

```
Query Input → Preprocessing → Multi-Strategy Search → Filtering → Ranking → Results
     ↓             ↓              ↓                ↓          ↓         ↓
 Validation → Normalization → Exact/Fuzzy/Semantic → Type/Page → Score → SearchResult[]
```

## 🔧 Implementation Architecture

### Class Structure

```python
class SmartSearchEngine:
    def __init__(self, elements: List[DocumentElement]):
        """Initialize search engine with document elements"""
        
    def search(self, query: str, **kwargs) -> List[SearchResult]:
        """Main search method with configurable options"""
        
    def semantic_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Semantic similarity search using embeddings"""
        
    def filter_by_region(self, elements: List[DocumentElement], region: Dict) -> List[DocumentElement]:
        """Spatial filtering based on coordinate regions"""
```

## 🔍 Search Strategies

### 1. **Exact Matching**

High-precision text matching with case-insensitive search:

```python
def _exact_search(self, query: str, elements: List[DocumentElement]) -> List[SearchResult]:
    """Exact string matching with normalization"""
    query_normalized = self._normalize_text(query)
    results = []
    
    for element in elements:
        text_normalized = self._normalize_text(element.text)
        
        if query_normalized in text_normalized:
            # Calculate exact match score
            score = self._calculate_exact_score(query_normalized, text_normalized, element)
            
            results.append(SearchResult(
                element=element,
                score=score,
                match_type='exact',
                matched_text=self._extract_matched_context(element.text, query)
            ))
    
    return results

def _normalize_text(self, text: str) -> str:
    """Comprehensive text normalization"""
    # Case normalization
    text = text.lower()
    
    # Unicode normalization
    import unicodedata
    text = unicodedata.normalize('NFKD', text)
    
    # Whitespace normalization
    text = ' '.join(text.split())
    
    # Remove punctuation for matching (configurable)
    import string
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    return text
```

### 2. **Fuzzy Matching**

Advanced fuzzy search using Levenshtein distance and phonetic matching:

```python
def _fuzzy_search(self, query: str, elements: List[DocumentElement], 
                  threshold: float = 0.8) -> List[SearchResult]:
    """Fuzzy matching with configurable similarity threshold"""
    from fuzzywuzzy import fuzz, process
    results = []
    
    # Prepare search corpus
    corpus = [(element.text, element) for element in elements]
    
    # Use different fuzzy matching algorithms
    for text, element in corpus:
        # Token-based matching (best for partial matches)
        token_ratio = fuzz.token_set_ratio(query, text) / 100.0
        
        # Partial ratio (best for substring matches)
        partial_ratio = fuzz.partial_ratio(query, text) / 100.0
        
        # Weighted ratio (balanced approach)
        weighted_ratio = fuzz.WRatio(query, text) / 100.0
        
        # Combined score with weights
        combined_score = (
            token_ratio * 0.4 +
            partial_ratio * 0.3 +
            weighted_ratio * 0.3
        )
        
        if combined_score >= threshold:
            # Adjust score based on element properties
            final_score = self._adjust_fuzzy_score(combined_score, element, query)
            
            results.append(SearchResult(
                element=element,
                score=final_score,
                match_type='fuzzy',
                matched_text=self._extract_fuzzy_match_context(text, query)
            ))
    
    return results

def _adjust_fuzzy_score(self, base_score: float, element: DocumentElement, query: str) -> float:
    """Adjust fuzzy match score based on element characteristics"""
    adjusted_score = base_score
    
    # Boost score for higher confidence elements
    confidence_boost = element.confidence * 0.1
    adjusted_score = min(1.0, adjusted_score + confidence_boost)
    
    # Boost score for shorter text (more precise matches)
    length_factor = min(1.0, len(query) / len(element.text))
    length_boost = length_factor * 0.05
    adjusted_score = min(1.0, adjusted_score + length_boost)
    
    # Boost score for certain element types
    type_boosts = {
        'heading': 0.15,
        'title': 0.20,
        'table': 0.10,
        'formula': 0.05
    }
    
    if element.element_type in type_boosts:
        adjusted_score = min(1.0, adjusted_score + type_boosts[element.element_type])
    
    return adjusted_score
```

### 3. **Semantic Search**

Vector-based semantic similarity search using embeddings:

```python
def semantic_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
    """Semantic search using sentence embeddings"""
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
            results.append(SearchResult(
                element=element,
                score=float(similarity),
                match_type='semantic',
                matched_text=element.text[:200] + "..." if len(element.text) > 200 else element.text
            ))
    
    return results

def _initialize_embeddings_model(self):
    """Initialize sentence transformer model for semantic search"""
    try:
        from sentence_transformers import SentenceTransformer
        self._embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Pre-compute embeddings for all elements
        self._element_embeddings = {}
        texts = [element.text for element in self.elements]
        embeddings = self._embeddings_model.encode(texts)
        
        for i, embedding in enumerate(embeddings):
            self._element_embeddings[i] = embedding
            
    except ImportError:
        raise ImportError(
            "Semantic search requires sentence-transformers. "
            "Install with: pip install sentence-transformers"
        )

def _calculate_cosine_similarity(self, vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    import numpy as np
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)
```

## 🎯 Advanced Filtering System

### 1. **Element Type Filtering**

```python
def filter_by_element_types(self, elements: List[DocumentElement], 
                           element_types: List[str]) -> List[DocumentElement]:
    """Filter elements by type with validation"""
    valid_types = {'text', 'heading', 'table', 'image', 'list', 'formula', 'caption'}
    
    # Validate input types
    invalid_types = set(element_types) - valid_types
    if invalid_types:
        raise ValueError(f"Invalid element types: {invalid_types}")
    
    return [element for element in elements if element.element_type in element_types]
```

### 2. **Page-Based Filtering**

```python
def filter_by_pages(self, elements: List[DocumentElement], 
                   page_numbers: List[int]) -> List[DocumentElement]:
    """Filter elements by page numbers with range support"""
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
```

### 3. **Spatial/Region Filtering**

```python
def filter_by_region(self, elements: List[DocumentElement], region: Dict[str, float],
                    strict_containment: bool = False) -> List[DocumentElement]:
    """Filter elements by spatial region with overlap/containment options"""
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
    """Check if two bounding boxes overlap"""
    return not (
        bbox1['x1'] < bbox2['x0'] or  # bbox1 is left of bbox2
        bbox2['x1'] < bbox1['x0'] or  # bbox2 is left of bbox1
        bbox1['y1'] < bbox2['y0'] or  # bbox1 is above bbox2
        bbox2['y1'] < bbox1['y0']     # bbox2 is above bbox1
    )
```

### 4. **Metadata-Based Filtering**

```python
def filter_by_metadata(self, elements: List[DocumentElement], 
                      metadata_filters: Dict[str, Any]) -> List[DocumentElement]:
    """Filter elements by metadata criteria"""
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
```

## 📊 Intelligent Ranking System

### 1. **Multi-Factor Scoring**

```python
def _calculate_relevance_score(self, element: DocumentElement, query: str, 
                              match_type: str, base_score: float) -> float:
    """Calculate comprehensive relevance score"""
    
    # Start with base match score
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
    
    # 4. Position Boost (earlier elements slightly favored)
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
```

### 2. **Context-Aware Scoring**

```python
def _calculate_context_score(self, element: DocumentElement, query: str) -> float:
    """Calculate score based on contextual relevance"""
    context_score = 0.0
    
    # Check for query words in surrounding context
    element_words = set(element.text.lower().split())
    query_words = set(query.lower().split())
    
    # Word overlap ratio
    overlap = len(element_words.intersection(query_words))
    if len(query_words) > 0:
        word_overlap_ratio = overlap / len(query_words)
        context_score += word_overlap_ratio * 0.3
    
    # Check for query as complete phrase
    if query.lower() in element.text.lower():
        context_score += 0.4
    
    # Check for important keywords in vicinity
    important_keywords = {'important', 'key', 'main', 'primary', 'critical', 'essential'}
    text_words_lower = element.text.lower().split()
    
    keyword_boost = sum(1 for word in text_words_lower if word in important_keywords)
    context_score += min(0.1, keyword_boost * 0.02)
    
    # Proximity scoring for multi-word queries
    if len(query_words) > 1:
        proximity_score = self._calculate_word_proximity(element.text, query_words)
        context_score += proximity_score * 0.2
    
    return min(1.0, context_score)

def _calculate_word_proximity(self, text: str, query_words: set) -> float:
    """Calculate how close query words appear to each other in text"""
    text_words = text.lower().split()
    word_positions = {}
    
    # Find positions of query words in text
    for i, word in enumerate(text_words):
        if word in query_words:
            if word not in word_positions:
                word_positions[word] = []
            word_positions[word].append(i)
    
    if len(word_positions) < 2:
        return 0.0
    
    # Calculate minimum distance between different query words
    min_distance = float('inf')
    words_found = list(word_positions.keys())
    
    for i in range(len(words_found)):
        for j in range(i + 1, len(words_found)):
            word1_positions = word_positions[words_found[i]]
            word2_positions = word_positions[words_found[j]]
            
            for pos1 in word1_positions:
                for pos2 in word2_positions:
                    distance = abs(pos1 - pos2)
                    min_distance = min(min_distance, distance)
    
    if min_distance == float('inf'):
        return 0.0
    
    # Convert distance to score (closer = higher score)
    proximity_score = 1.0 / (1.0 + min_distance)
    return proximity_score
```

## 🚀 Advanced Features

### 1. **Query Expansion**

```python
def _expand_query(self, query: str) -> List[str]:
    """Expand query with synonyms and related terms"""
    expanded_queries = [query]
    
    # Add stemmed versions
    try:
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        
        words = query.split()
        stemmed_words = [stemmer.stem(word) for word in words]
        stemmed_query = ' '.join(stemmed_words)
        
        if stemmed_query != query:
            expanded_queries.append(stemmed_query)
            
    except ImportError:
        pass  # NLTK not available
    
    # Add synonym expansions (if wordnet available)
    try:
        from nltk.corpus import wordnet
        
        synonyms = set()
        for word in query.split():
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym.lower() != word.lower():
                        synonyms.add(synonym)
        
        # Add top synonyms as alternative queries
        for synonym in list(synonyms)[:3]:  # Limit to top 3
            synonym_query = query.replace(word, synonym, 1)
            expanded_queries.append(synonym_query)
            
    except ImportError:
        pass  # WordNet not available
    
    return expanded_queries
```

### 2. **Multi-Language Support**

```python
def _detect_language(self, text: str) -> str:
    """Detect text language for appropriate processing"""
    try:
        from langdetect import detect
        return detect(text)
    except:
        return 'en'  # Default to English

def _process_multilingual_query(self, query: str) -> str:
    """Process query based on detected language"""
    language = self._detect_language(query)
    
    # Language-specific normalization
    if language == 'de':  # German
        query = self._normalize_german_text(query)
    elif language == 'fr':  # French
        query = self._normalize_french_text(query)
    elif language == 'es':  # Spanish
        query = self._normalize_spanish_text(query)
    
    return query
```

### 3. **Caching and Performance Optimization**

```python
class SearchCache:
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
    
    def get(self, key: str):
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def set(self, key: str, value):
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def _evict_oldest(self):
        oldest_key = min(self.access_times, key=self.access_times.get)
        del self.cache[oldest_key]
        del self.access_times[oldest_key]

# Usage in SmartSearchEngine
def search(self, query: str, **kwargs) -> List[SearchResult]:
    """Main search with caching"""
    
    # Create cache key
    cache_key = self._create_cache_key(query, kwargs)
    
    # Check cache first
    cached_result = self._search_cache.get(cache_key)
    if cached_result is not None:
        return cached_result
    
    # Perform search
    results = self._perform_search(query, **kwargs)
    
    # Cache results
    self._search_cache.set(cache_key, results)
    
    return results
```

## 🧪 Usage Examples

### Basic Search

```python
from src.core.search import SmartSearchEngine
from src.core.models import DocumentElement

# Initialize with parsed elements
elements = [...]  # From DoclingParser
search_engine = SmartSearchEngine(elements)

# Simple text search
results = search_engine.search("revenue analysis")

# Display results
for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Type: {result.element.element_type}")
    print(f"Text: {result.matched_text}")
    print("---")
```

### Advanced Search with Filters

```python
# Search with multiple filters
results = search_engine.search(
    query="financial data",
    element_types=['table', 'text'],
    page_numbers=[1, 2, 3],
    enable_fuzzy=True,
    fuzzy_threshold=0.7,
    limit=10
)

# Regional search
region = {'x0': 100, 'y0': 100, 'x1': 500, 'y1': 400}
regional_results = search_engine.search(
    query="key findings",
    region=region,
    strict_containment=False
)
```

### Semantic Search

```python
# Initialize with semantic capabilities
search_engine = SmartSearchEngine(elements)

# Perform semantic search
semantic_results = search_engine.semantic_search(
    query="machine learning algorithms performance",
    top_k=5
)

# Results will include semantically similar content
# even if exact words don't match
```

### Batch Search Operations

```python
# Multiple query search
queries = ["revenue", "profit", "loss", "growth"]
batch_results = {}

for query in queries:
    batch_results[query] = search_engine.search(query, limit=5)

# Analyze search performance
for query, results in batch_results.items():
    avg_score = sum(r.score for r in results) / len(results) if results else 0
    print(f"Query '{query}': {len(results)} results, avg score: {avg_score:.3f}")
```

## 📚 Integration Points

### With DoclingParser
```python
# Parse document and search
parser = DoclingParser()
elements = parser.parse_document("document.pdf")

search_engine = SmartSearchEngine(elements)
results = search_engine.search("important findings")
```

### With Verification System
```python
# Search and verify results
results = search_engine.search("data table")

from src.verification.interface import VerificationInterface
verification = VerificationInterface(
    elements=[r.element for r in results],
    renderer=renderer
)
```

## 🔧 Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_fuzzy` | bool | True | Enable fuzzy matching |
| `fuzzy_threshold` | float | 0.8 | Minimum fuzzy match score |
| `enable_semantic` | bool | False | Enable semantic search |
| `element_types` | List[str] | None | Filter by element types |
| `page_numbers` | List[int] | None | Filter by page numbers |
| `region` | Dict | None | Spatial region filter |
| `include_metadata` | bool | False | Search in metadata fields |
| `limit` | int | None | Maximum results to return |
| `min_score` | float | 0.0 | Minimum relevance score |

This comprehensive implementation provides a robust, scalable search engine that can handle diverse document types and search requirements while maintaining high performance and accuracy.
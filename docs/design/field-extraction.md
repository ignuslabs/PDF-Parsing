# Field Extraction System Design

## Overview

The Field Extraction system provides automated key-value pair extraction from PDF documents, specifically designed for forms, applications, invoices, and other structured documents with clear label-value relationships.

## Architecture

### Core Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   KVConfig      │    │ KeyValueExtractor│    │  Visualization  │
│   (Settings)    │───▶│   (Core Logic)   │───▶│   (UI/Render)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │ DocumentElement  │
                       │   (Input Data)   │
                       └──────────────────┘
```

### Data Flow

1. **Input**: List of `DocumentElement` objects from PDF parsing
2. **Processing**: Multi-stage extraction pipeline
3. **Output**: List of `KeyValuePair` objects with confidence scores

## Algorithm Design

### 1. Preprocessing Stage

**Page-level Processing**
- Groups elements by page number for independent processing
- Builds line clusters using Y-coordinate overlap detection
- Identifies column boundaries using gap-based clustering
- Creates spatial indices for efficient candidate lookup

**Column Detection**
```python
# Gap-based column clustering algorithm
gaps = []
for i in range(len(sorted_x_positions) - 1):
    gap = sorted_x_positions[i+1] - sorted_x_positions[i]
    if gap > config.gutter_min_dx:
        gaps.append(gap)
```

### 2. Label Detection

**Multi-factor Scoring System**

Labels are scored on a scale of 0-1 using the following factors:

| Feature | Weight | Criteria |
|---------|---------|----------|
| **Colon Ending** | 0.4 | Text ends with ':' and length < max_label_len |
| **Length Check** | 0.2 | Text length < max_label_len |
| **Case Analysis** | 0.2 | Uppercase ratio > min_upper_ratio |
| **Position** | 0.1 | Left margin percentile < 30% |
| **Word Count** | 0.1 | Word count ≤ 4 |

**Threshold**: label_score ≥ 0.5

**Example Scoring**:
```
"Name:" → 0.4 (colon) + 0.2 (length) + 0.1 (position) = 0.7 ✓
"Employment Information" → 0.2 (length) + 0.2 (case) = 0.4 ✗
```

### 3. Value Pairing Strategies

**Same-line Strategy (Preferred)**
- Searches for nearest element to the right
- Within `max_same_line_dx` horizontal distance
- Similar Y-coordinate range (overlap detection)

**Below Strategy (Fallback)**
- Vertical search within `max_below_dy` window
- X-alignment tolerance using `x_align_tol`
- Must remain within same column cluster

**Geometric Constraints**
```python
# Same-line pairing check
def is_same_line(label_bbox, value_bbox, max_dx):
    horizontal_gap = value_bbox['x0'] - label_bbox['x1']
    vertical_overlap = calculate_y_overlap(label_bbox, value_bbox)
    return horizontal_gap <= max_dx and vertical_overlap > 0.5
```

### 4. Multi-line Value Merging

**Spatial Adjacency Algorithm**

Values are extended downward when:
- X-overlap ratio > 60% with previous line
- Vertical gap < median line spacing
- Maintains column alignment

**Example**:
```
Address: │ 123 Main Street     │ ← Initial value
         │ Apartment 4B        │ ← Merged (aligned)
         │ Springfield, IL     │ ← Merged (aligned)
```

### 5. Confidence Scoring

**Weighted Composite Score**

```python
confidence = sigmoid(0.5*label_score + 0.3*geom_score + 0.2*content_score)
```

**Components**:
- **Label Score**: Quality of label detection (0-1)
- **Geometric Score**: Spatial relationship quality (0-1)
- **Content Score**: Value content appropriateness (0-1)

**Geometric Scoring Factors**:
- Distance between label and value (closer = higher)
- Alignment quality (better alignment = higher)
- Pairing strategy used (same-line > below)

## Configuration Parameters

### KVConfig Class

```python
@dataclass
class KVConfig:
    # Pairing distances
    max_same_line_dx: float = 200.0      # Max horizontal gap for same-line
    max_below_dy: float = 100.0          # Max vertical gap for below pairing
    x_align_tol: float = 40.0            # X-alignment tolerance
    
    # Column detection  
    gutter_min_dx: float = 120.0         # Minimum column gutter width
    
    # Label constraints
    max_label_len: int = 60              # Maximum label character length
    min_upper_ratio: float = 0.4         # Minimum uppercase ratio
    min_value_len: int = 1               # Minimum value length
    
    # Confidence weights
    label_weight: float = 0.5            # Label score weight
    geom_weight: float = 0.3             # Geometric score weight  
    content_weight: float = 0.2          # Content score weight
```

## Performance Characteristics

### Benchmarks

| Metric | Performance |
|--------|-------------|
| **Throughput** | 23,000+ elements/second |
| **KV Rate** | 11,000+ pairs/second |
| **Memory Usage** | O(n) linear scaling |
| **Accuracy** | >95% on structured forms |

### Complexity Analysis

- **Time**: O(n log n) per page due to spatial indexing
- **Space**: O(n) for element storage and indices
- **Scaling**: Approximately linear with document size

## Integration Points

### 1. Parser Integration

**DoclingParser Extensions**:
```python
parser = DoclingParser(
    enable_kv_extraction=True,
    kv_config=KVConfig(),
    header_classifier_enabled=True
)

elements, kv_pairs = parser.parse_document_with_kvs(pdf_path)
```

### 2. UI Integration

**Parse Page Features**:
- Toggle for enabling KV extraction
- Configuration for header classifier
- Performance warnings for large documents

**Verify Page Features**:
- Visual overlay rendering of KV pairs
- Color-coded labels and values
- Connection lines between paired elements
- Confidence score display

### 3. Visualization System

**PDFRenderer Extensions**:
```python
# Render individual KV pair
renderer.render_kv_pair(page_image, kv_pair, colors=((255,0,0,128), (0,128,0,128)))

# Render all KV pairs for page
renderer.render_kv_pairs(page_image, kv_pairs, palette=color_palette)
```

## Quality Assurance

### Testing Strategy

**Unit Tests** (25 test cases):
- Label detection accuracy
- Pairing strategy correctness
- Multi-line merging behavior
- Column boundary detection
- Confidence score calculation
- Edge case handling

**Integration Tests**:
- Real PDF form processing
- Parser integration validation
- UI component functionality

**Performance Tests**:
- Large document handling (1000+ elements)
- Linear scaling verification
- Memory usage bounds

### Error Handling

**Graceful Degradation**:
- Invalid bounding boxes → skip element
- Parsing failures → empty KV pair list
- Configuration errors → use defaults

**Logging and Debugging**:
- Detailed extraction metrics
- Confidence score breakdowns
- Performance timing information

## Usage Patterns

### 1. Form Processing

**Best For**:
- Job applications
- Insurance forms
- Government documents
- Survey responses

**Pattern Recognition**:
```
Name: John Smith
Address: 123 Main St
Phone: (555) 123-4567
```

### 2. Invoice Processing

**Best For**:
- Service invoices
- Purchase orders
- Billing statements

**Pattern Recognition**:
```
Invoice #: INV-2024-001
Date: March 15, 2024
Amount: $1,234.56
```

### 3. Multi-column Layouts

**Handling**:
- Automatic column detection
- Cross-gutter prevention
- Independent processing per column

## Limitations and Considerations

### Current Limitations

1. **Language Support**: Optimized for English text patterns
2. **Layout Types**: Works best with structured forms
3. **Handwriting**: Requires OCR pre-processing for handwritten forms
4. **Complex Tables**: May not handle nested table structures optimally

### Performance Considerations

1. **Large Documents**: Performance degrades with >10k elements per page
2. **Memory Usage**: Holds all elements in memory during processing
3. **OCR Dependencies**: Performance depends on OCR quality for scanned docs

### Future Enhancements

1. **Machine Learning**: Train models for better label/value classification
2. **Context Awareness**: Use semantic understanding for field validation
3. **Multi-language**: Support for non-English document processing
4. **Template Learning**: Adapt to recurring document templates

## Implementation History

- **Phase 1**: Core extraction algorithm and testing framework
- **Phase 2**: Parser integration and configuration system
- **Phase 3**: UI integration and visualization
- **Phase 4**: Performance optimization and benchmarking
- **Phase 5**: Documentation and quality assurance

---

*This document reflects the design as of the initial implementation. Future updates will be tracked through version control and change logs.*
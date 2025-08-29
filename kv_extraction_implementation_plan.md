# Field (Label–Value) Extraction and Header-Classification Plan

This document provides a precise, step-by-step implementation plan to add robust form-like field (label–value) extraction and improve header vs. body classification without breaking existing functionality. Follow the steps in order. Each step includes intent, files to add/modify, function signatures, acceptance criteria, and commands to run.

## Scope & Principles

- Maintain backward compatibility by keeping all new behavior behind feature flags and optional entry points.
- Start with tests (TDD), add fixtures, then implement minimal code to satisfy tests.
- Prefer geometry-first heuristics with tunable thresholds; optional ML can be added later.
- Preserve current parser behavior and interfaces used by UI and verification.

## Prerequisites

- Python environment activated with project requirements.
- Test fixtures present or generated via `python generate_test_fixtures.py`.
- Ensure `docling` is installed per `requirements.txt`.

## High-Level Deliverables

- New data model for KV pairs.
- Header classifier module with improved rules and context features.
- KV extractor module that pairs labels and values using geometry and content heuristics.
- Parser integration behind a feature flag; optional method to return KV pairs.
- Tests: unit, integration, and basic property tests for KV/headers.
- Optional verification overlay support and UI toggles (non-breaking, additive).

---

## Step 0 — Baseline Validation (No Code Changes)

- Verify current tests run (or are skipped by markers) to establish a baseline.
- Command:
  - `pytest -q -m "not slow and not ocr and not performance"`
  - Optional OCR tests: `pytest -q -m ocr`

Acceptance:
- Existing tests pass (or known skips remain). No new failures introduced.

---

## Step 1 — Add Form-like PDF Fixtures

Intent: Create a realistic government-form style PDF to drive label–value extraction tests.

Files to modify:
- `generate_test_fixtures.py`: add `create_forms_basic_pdf(filepath)` that generates fields like:
  - Name: John A. Smith
  - Date of Birth: 01/23/1980
  - Address: 123 Main St. (multi-line value)
  - SSN: 123-45-6789 (two-column variant on same page)

Task specifics:
- Ensure both right-of-label and below-label value placements exist.
- Include multi-line values and two-column layout with a clear gutter.
- Insert a heading (e.g., “APPLICATION FORM”) to test heading rules.

Command:
- `python generate_test_fixtures.py`

Acceptance:
- `tests/fixtures/forms_basic.pdf` exists and opens.

---

## Step 2 — Add Data Model for KV Pairs

Intent: Define a serializable structure for label–value pairs with geometry and confidence.

Files to modify:
- `src/core/models.py`

Add:
- `@dataclass KeyValuePair` with fields:
  - `label_text: str`
  - `value_text: str`
  - `page_number: int`
  - `label_bbox: Dict[str, float]`
  - `value_bbox: Dict[str, float]`
  - `confidence: float`
  - `metadata: Dict[str, Any]` (e.g., `{'label_score': float, 'geom_score': float, 'content_score': float, 'strategy': 'same_line|below'}`)

Acceptance:
- Importable dataclass with validation similar to `DocumentElement` (confidence in [0,1], bbox keys present, page >= 1).

---

## Step 3 — Header Classifier Tests (TDD)

Intent: Lock in improved rules to avoid misclassifying values (e.g., person names) as headings.

Files to add:
- `tests/test_header_classifier.py`

Test cases:
- Classifies short, all-caps, colon-ended text near page top as heading.
- Does not classify values like `"John A. Smith"` or `"123-45-6789"` as headings.
- Treats tokens like `"Section 1:"` as heading.
- Leverages page-relative position percentiles; being within top 15% boosts heading.

Design:
- Use fabricated `DocumentElement` instances with `bbox` and `page_number`.
- Provide a minimal `PageContext` helper (line height stats, page width/height) mocked in tests.

Acceptance:
- Failing tests exist verifying the stricter/safer heading detection behavior.

---

## Step 4 — Implement Header Classifier (Additive)

Files to add:
- `src/core/classifiers/header_classifier.py`

Public API:
- `def is_heading(element: DocumentElement, page_context: dict) -> bool`

Rules (starting weights):
- Uppercase ratio > 0.6 and length < 80 → heading boost.
- Ends with `:` and length < 60 → heading boost.
- Page top percentile (e.g., `y0` within top 15% of page) → heading boost.
- Negative evidence: looks like a person name (capitalization pattern), a date, or alphanumeric-looking/numeric-only; near a detected label → demote.

Integration (non-breaking):
- In `src/core/parser.py`, gate a call in `_determine_text_element_type` via a flag (see Step 7). Default remains current heuristics.

Acceptance:
- `tests/test_header_classifier.py` passes.

---

## Step 5 — KV Extraction Tests (TDD)

Intent: Formalize label detection and value pairing rules.

Files to add:
- `tests/test_kv_extraction.py`

Test scenarios:
- Label detection: short text, endswith colon, left-aligned; reject long paragraphs.
- Same-line right pairing preferred; fallback to below within vertical window.
- Multi-line values merged by vertical adjacency and x-alignment.
- Two-column layout: prevent cross-gutter pairing; remain in the same column cluster.
- Confidence score between 0–1 with expected composition.
- No duplicate assignment of a value to multiple labels on the same page.

Test utilities:
- Helper to fabricate `DocumentElement` lists for given page sizes.
- Use `forms_basic.pdf` via `DoclingParser.parse_document(...)` in an integration test to assert expected pairs (e.g., Name → John A. Smith).

Acceptance:
- Tests are present and initially fail.

---

## Step 6 — Implement KV Extraction Module

Files to add:
- `src/core/kv_extraction.py`

Public API:
- `@dataclass
  class KVConfig:` with tunables: `max_same_line_dx`, `max_below_dy`, `x_align_tol`, `gutter_min_dx`, `min_value_len`, `max_label_len`, `min_upper_ratio`, etc.
- `class KeyValueExtractor:`
  - `def __init__(self, config: Optional[KVConfig] = None): ...`
  - `def extract(self, elements: List[DocumentElement]) -> List[KeyValuePair]`

Algorithm outline:
- Preprocess per page: group elements into lines via y-overlap; compute page stats; cluster columns by x using 1D k-means or gap detection; build a KD-tree over candidate value positions.
- Label detection score:
  - Features: endswith colon (+0.4), len < `max_label_len` (+0.2), uppercase ratio > `min_upper_ratio` (+0.2), left margin percentile < 0.3 (+0.1), word count <= 4 (+0.1).
  - Threshold: label_score >= 0.5.
- Candidate search:
  - Same-line: nearest element to the right within `max_same_line_dx` and similar y-range.
  - Fallback below: vertical window up to `max_below_dy`, similar x-range (`x_align_tol`), within same column cluster.
  - Exclude element types `table`, `image`, `heading`, `code`, `formula` from values.
- Multi-line merge: keep extending down while x-overlap ratio > 0.6 and vertical gap below median line gap.
- Confidence: `sigmoid(w1*label_score + w2*geom_score + w3*content_score)` with default weights `(0.5,0.3,0.2)`.
- Return `KeyValuePair` instances with `metadata` holding scores, strategy (`same_line|below`), and distances.

Acceptance:
- `tests/test_kv_extraction.py` unit tests pass for synthetic scenarios.

---

## Step 7 — Parser Integration Behind Feature Flags

Intent: Expose KV extraction without breaking current APIs.

Files to modify:
- `src/core/parser.py`

Add config:
- New optional flags in `DoclingParser.__init__` (default off):
  - `enable_kv_extraction: bool = False`
  - `kv_config: Optional[KVConfig] = None`
  - `header_classifier_enabled: bool = False`

Integration points:
- In `_determine_text_element_type`, if `header_classifier_enabled` use `is_heading(...)` to override/confirm ‘heading’; otherwise keep current logic.
- Add new method:
  - `def parse_document_with_kvs(self, pdf_path: Path) -> Tuple[List[DocumentElement], List[KeyValuePair]]`
  - Internally: call `parse_document(...)` to get elements; if `enable_kv_extraction`, run `KeyValueExtractor.extract(elements)`; else return empty list for pairs.

Acceptance:
- Existing tests still pass.
- New integration tests using `forms_basic.pdf` can call `parse_document_with_kvs(...)` and assert expected pairs.

---

## Step 8 — Verification Renderer (Optional, Additive)

Intent: Visualize KV pairs on page images to aid manual verification.

Files to modify:
- `src/verification/renderer.py`

Add methods:
- `def render_kv_pair(self, page_image, kv: KeyValuePair, colors=((255,0,0,128),(0,128,0,128))):` draw label and value boxes in different colors and a leader line connecting them.
- `def render_kv_pairs(self, page_image, kvs: List[KeyValuePair], palette=...)`

Acceptance:
- New methods are unit-tested minimally (mock image, bbox transforms) or exercised by an integration smoke test.

---

## Step 9 — UI Toggles (Optional, Additive)

Intent: Allow users to enable KV extraction and view results in the Verify page.

Files to modify:
- `src/ui/pages/1_📄_Parse.py`: add parser options toggles `Enable KV Extraction`, `Use Safer Header Classifier`.
- `src/ui/pages/3_✅_Verify.py`: add “Show Field Pairs” toggle; list KV pairs with confidence; clicking highlights both boxes in the page image.

Acceptance:
- When toggles are off, current UI behavior remains unchanged.
- When toggles are on (and images enabled), KV overlays appear without exceptions.

---

## Step 10 — Performance & Threshold Calibration

Intent: Ensure minimal overhead and tune defaults.

Tasks:
- Add benchmarks in `tests/test_property_performance.py` for KV extraction on `large_pages_light.pdf`; assert added time < 15% per page.
- Expose `KVConfig` via UI advanced settings (optional).
- Tune defaults for `max_same_line_dx`, `max_below_dy`, `x_align_tol`, and label threshold based on fixtures.

Acceptance:
- Performance tests stay within set limits; parser memory footprint acceptable.

---

## Step 11 — Documentation

Files to add/modify:
- `docs/design/field-extraction.md`: deeper design, rules, diagrams (optional if this plan suffices).
- `docs/index.md`: add a section linking to KV extraction feature and how to enable flags.

Content:
- Explain heuristics, feature flags, limitations, and verification workflow.

Acceptance:
- Docs build without warnings; links valid.

---

## Step 12 — Quality Gates & Rollout

- CI gates (if applicable):
  - All unit/integration tests pass.
  - Performance checks pass.
- Rollout strategy:
  - Keep `enable_kv_extraction=False` and `header_classifier_enabled=False` by default for initial release.
  - After validation, consider flipping defaults in a minor version.
- Rollback:
  - Since features are behind flags and additive, rollback is achieved by disabling flags; code paths remain dormant.

---

## Detailed Interfaces and Pseudocode

### `src/core/models.py`

```python
@dataclass
class KeyValuePair:
    label_text: str
    value_text: str
    page_number: int
    label_bbox: Dict[str, float]
    value_bbox: Dict[str, float]
    confidence: float
    metadata: Dict[str, Any]
```

### `src/core/classifiers/header_classifier.py`

```python
def is_heading(element: DocumentElement, page_context: dict) -> bool:
    text = element.text.strip()
    feats = {
        'len': len(text),
        'upper_ratio': sum(c.isupper() for c in text if c.isalpha()) / max(1, sum(c.isalpha() for c in text)),
        'ends_colon': text.endswith(':'),
        'is_numeric_like': looks_numeric_or_date(text),
        'y_percentile': page_context.get('y_percentile', 1.0),
    }
    score = 0.0
    if feats['upper_ratio'] > 0.6 and feats['len'] < 80: score += 0.4
    if feats['ends_colon'] and feats['len'] < 60: score += 0.3
    if feats['y_percentile'] < 0.15: score += 0.2
    if feats['is_numeric_like']: score -= 0.5
    if looks_like_person_name(text): score -= 0.5
    return score >= 0.5
```

### `src/core/kv_extraction.py`

```python
@dataclass
class KVConfig:
    max_same_line_dx: float = 200.0
    max_below_dy: float = 100.0
    x_align_tol: float = 40.0
    gutter_min_dx: float = 120.0
    max_label_len: int = 60
    min_upper_ratio: float = 0.4
    min_value_len: int = 1

class KeyValueExtractor:
    def __init__(self, config: Optional[KVConfig] = None):
        self.cfg = config or KVConfig()

    def extract(self, elements: List[DocumentElement]) -> List[KeyValuePair]:
        pairs: List[KeyValuePair] = []
        for page, elems in group_by_page(elements).items():
            lines = cluster_lines(elems)
            columns = cluster_columns(elems)
            candidates = build_value_index(elems)
            for el in elems:
                label_score = score_label(el, self.cfg)
                if label_score < 0.5: continue
                value, geom_score, strategy = find_value(el, candidates, lines, columns, self.cfg)
                if not value: continue
                merged_value = merge_multiline(value, lines, self.cfg)
                content_score = score_value_content(merged_value)
                conf = sigmoid(0.5*label_score + 0.3*geom_score + 0.2*content_score)
                pairs.append(make_kv(el, merged_value, page, conf, label_score, geom_score, content_score, strategy))
        return dedupe_pairs(pairs)
```

---

## Testing Commands

- Unit and integration (no OCR/perf):
  - `pytest -q -m "not slow and not ocr and not performance"`
- Forms/KV specific (mark with `@pytest.mark.forms` in new tests):
  - `pytest -q -m forms`
- OCR and slow suites (optional):
  - `pytest -q -m ocr`
  - `pytest -q -m performance`

---

## Acceptance Checklist

- KV and header tests added and passing.
- Existing parser tests untouched and passing.
- Feature flags default off; enabling them surfaces KV pairs and improved headings.
- Verification/UI additions are optional and do not alter defaults.
- Performance within thresholds on large fixtures.
- Documentation updated and linked from `docs/index.md`.

---

## Risks & Mitigations

- OCR noise: merge multi-line values; prefer geometry over text-only signals.
- Multi-column confusion: enforce same-column constraint and gutter threshold.
- International forms: support labels without colons via short-length + left alignment heuristics.
- Ambiguity: avoid duplicate value assignment; tie-break by distance then reading order.

---

## Rollout Strategy

- Ship disabled-by-default flags.
- Encourage validation on internal form sets; capture feedback.
- Consider enabling safer header classifier by default in a minor release after validation.


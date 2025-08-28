# 05 — Unit Tests: Smart Search Engine

## Targets
- Exact match, word-level match, fuzzy threshold behavior.
- Type filters (text/table/image/header).
- Page filters and region search.
- Ranking: exact > word > fuzzy; headers boosted.

## Example Test Skeletons
```python
# tests/test_search_engine.py
from src.core.models import DocumentElement
from src.core.search import SmartSearchEngine

def make_el(text, page=1, et="text"):
    return DocumentElement(text=text, element_type=et, page_number=page,
                           bbox={'x0':0,'y0':0,'x1':10,'y1':10},
                           confidence=0.9, metadata={})

def test_exact_match_ranks_higher():
    elems = [make_el("alpha beta"), make_el("alpha"), make_el("alpah")]  # typo
    eng = SmartSearchEngine(elems)
    res = eng.search("alpha beta")
    assert res[0]['element'].text == "alpha beta"

def test_type_filter_tables_only():
    elems = [make_el("revenue table", et="table"), make_el("paragraph", et="text")]
    eng = SmartSearchEngine(elems)
    res = eng.search("revenue", element_types=["table"])
    assert all(r['element'].element_type == "table" for r in res)
```

## Invariants
- Results sorted by relevance descending.
- Region search returns overlaps (not strict containment) by default.

# 06 — Unit Tests: Interactive Verification

## Targets
- Coordinate transform (PDF coords -> image coords).
- Highlight rectangles render within image bounds.
- Verification state recorded per element (correct/incorrect/partial) with timestamps.
- Export JSON summary structure (totals, accuracy, by_page).

## Example Test Skeletons
```python
# tests/test_verification.py
from src.verification.renderer import PDFRenderer
from src.verification.interface import VerificationInterface

def test_transform_coordinates_in_bounds(fake_page_image, sample_bbox):
    r = PDFRenderer(pdf_parser=None)  # patch page size assumptions if needed
    coords = r.transform_coordinates(sample_bbox, fake_page_image)
    assert coords['x0'] >= 0 and coords['y0'] >= 0

def test_export_verification_json(sample_results):
    v = VerificationInterface(...)
    # simulate clicks/marks
    payload = v.export_verification_data()
    assert '"total":' in payload and '"by_page":' in payload
```

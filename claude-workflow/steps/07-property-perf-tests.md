# 07 — Property & Performance Tests

## Property Tests
- Export-Import roundtrip: JSON -> model -> JSON is stable for key fields.
- Bounding boxes always within page extents.
- Search idempotence under case changes and minor whitespace tweaks.

## Performance Guards
- Parsing time per page stays under budget (e.g., < 1.5s/page on baseline).
- Memory stays under cap (no unbounded growth) in batch parse.

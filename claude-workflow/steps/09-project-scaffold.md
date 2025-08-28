# 09 — Project Scaffold

## Directories
```
smart-pdf-parser/
├── src/
│   ├── core/              # Docling wrapper & search
│   ├── verification/      # Renderer + Interface
│   ├── utils/
│   └── ui/                # Streamlit app
├── tests/
│   ├── golden/
│   └── fixtures/
├── data/
│   └── samples/
├── config/
├── requirements.txt
├── requirements-dev.txt
├── .env.example
└── README.md
```

## Minimal Files to Generate
- `src/core/parser.py`, `src/core/search.py`, `src/core/models.py`
- `src/verification/renderer.py`, `src/verification/interface.py`
- `src/ui/app.py`
- `tests/test_installation.py`, `tests/test_parser_core.py`, `tests/test_search_engine.py`

## Commands
- Run UI: `python -m streamlit run src/ui/app.py`
- Tests: `pytest -v`

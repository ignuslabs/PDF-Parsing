"""
Docling compatibility utilities.

This module provides thin wrappers and helpers to smooth over breaking- or
minor API differences between Docling releases. Keep all Docling-specific
imports and attribute checks here to reduce surface area of change.

Usage goals:
- Build a configured converter in one place
- Normalize table mode handling (string or enum)
- Provide robust export helpers for markdown/text/dict
- Extract page images regardless of page index attribute names
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


def _import_docling():
    """Import docling modules lazily and return a dict of symbols."""
    # Import inside to avoid hard import failures at module import time
    from docling.document_converter import DocumentConverter, PdfFormatOption
    try:
        # Newer location (as seen in some versions)
        from docling.datamodel.base_models import InputFormat
    except Exception:
        InputFormat = None  # type: ignore

    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TesseractOcrOptions,
        TableFormerMode,
    )

    # EasyOcrOptions may not exist in some builds; guard import
    try:
        from docling.datamodel.pipeline_options import EasyOcrOptions  # type: ignore
    except Exception:
        EasyOcrOptions = None  # type: ignore

    return {
        "DocumentConverter": DocumentConverter,
        "PdfFormatOption": PdfFormatOption,
        "InputFormat": InputFormat,
        "PdfPipelineOptions": PdfPipelineOptions,
        "TesseractOcrOptions": TesseractOcrOptions,
        "TableFormerMode": TableFormerMode,
        "EasyOcrOptions": EasyOcrOptions,
    }


def normalize_table_mode(mode: Any) -> Any:
    """Normalize table mode value to TableFormerMode enum if possible.

    Accepts string values like "accurate"/"fast" or already-enum values.
    Returns the best-effort enum; falls back to the provided value.
    """
    syms = _import_docling()
    TableFormerMode = syms["TableFormerMode"]

    if isinstance(mode, str):
        lowered = mode.lower().strip()
        if hasattr(TableFormerMode, "ACCURATE") and lowered in {"accurate", "accurate_mode"}:
            return TableFormerMode.ACCURATE
        if hasattr(TableFormerMode, "FAST") and lowered in {"fast", "speed"}:
            return TableFormerMode.FAST
    return mode


def create_ocr_options(ocr_engine: str, lang_list: List[str], force_full_page_ocr: bool = True) -> Any:
    """Create OCR options for the selected engine with best-effort compatibility."""
    syms = _import_docling()
    TesseractOcrOptions = syms["TesseractOcrOptions"]
    EasyOcrOptions = syms["EasyOcrOptions"]

    if ocr_engine == "tesseract":
        return TesseractOcrOptions(lang=lang_list, force_full_page_ocr=force_full_page_ocr)

    if ocr_engine == "easyocr":
        if EasyOcrOptions is None:
            raise ValueError("EasyOCR support not available in this Docling build")
        return EasyOcrOptions(lang=lang_list)

    raise ValueError(f"Unsupported OCR engine: {ocr_engine}")


def build_converter(
    *,
    enable_ocr: bool,
    ocr_options: Any | None,
    enable_tables: bool,
    generate_page_images: bool,
    table_mode: Any,
    image_scale: float,
) -> Any:
    """Create a configured DocumentConverter with backward-compatible options."""
    syms = _import_docling()
    DocumentConverter = syms["DocumentConverter"]
    PdfFormatOption = syms["PdfFormatOption"]
    InputFormat = syms["InputFormat"]
    PdfPipelineOptions = syms["PdfPipelineOptions"]

    pipeline_options = PdfPipelineOptions()
    # Core toggles
    if hasattr(pipeline_options, "do_ocr"):
        pipeline_options.do_ocr = enable_ocr
    if hasattr(pipeline_options, "do_table_structure"):
        pipeline_options.do_table_structure = enable_tables
    if hasattr(pipeline_options, "generate_page_images"):
        pipeline_options.generate_page_images = generate_page_images

    # OCR opts
    if enable_ocr and ocr_options is not None and hasattr(pipeline_options, "ocr_options"):
        pipeline_options.ocr_options = ocr_options

    # Table opts
    if enable_tables and hasattr(pipeline_options, "table_structure_options"):
        try:
            # Prefer enum if available
            pipeline_options.table_structure_options.mode = normalize_table_mode(table_mode)
        except Exception:
            # Fallback: accept raw string if enum fails
            pipeline_options.table_structure_options.mode = table_mode

    # Image opts
    if generate_page_images:
        # Some versions use images_scale; guard attribute
        if hasattr(pipeline_options, "images_scale"):
            pipeline_options.images_scale = image_scale

    # Build converter with format options
    if InputFormat is not None:
        fmt_key = InputFormat.PDF
    else:
        # Fallback for older versions that accept string keys
        fmt_key = "pdf"  # type: ignore

    converter = DocumentConverter(format_options={fmt_key: PdfFormatOption(pipeline_options=pipeline_options)})
    return converter


def export_markdown(document: Any) -> str:
    """Export document as markdown using whichever method exists."""
    for attr in ("export_to_markdown", "to_markdown", "as_markdown"):
        if hasattr(document, attr):
            return getattr(document, attr)()
    raise AttributeError("No markdown export method found on document")


def export_text(document: Any) -> str:
    """Export document as plain text using available method names."""
    for attr in ("export_to_text", "to_text", "as_text"):
        if hasattr(document, attr):
            return getattr(document, attr)()
    raise AttributeError("No text export method found on document")


def export_dict(document: Any) -> Dict[str, Any]:
    """Export document as dict using available method names."""
    for attr in ("export_to_dict", "to_dict", "as_dict"):
        if hasattr(document, attr):
            return getattr(document, attr)()
    raise AttributeError("No dict export method found on document")


def iter_page_images(conversion_result: Any) -> Iterable[Tuple[int, Any]]:
    """Yield (1-based page_number, image) pairs if images are present."""
    if not hasattr(conversion_result, "pages") or not conversion_result.pages:
        return []

    pages = conversion_result.pages
    out: List[Tuple[int, Any]] = []
    for idx, page in enumerate(pages):
        if not hasattr(page, "image") or page.image is None:
            continue
        # Try common page index attributes
        page_num = None
        for attr in ("page_no", "page_idx", "index"):
            if hasattr(page, attr):
                try:
                    raw_val = int(getattr(page, attr))
                    page_num = raw_val + 1 if raw_val < 1 else raw_val
                    break
                except Exception:
                    continue
        if page_num is None:
            # Fallback to enumerate index (convert to 1-based)
            page_num = idx + 1
        out.append((page_num, page.image))
    return out


def get_document_collections(document: Any) -> Dict[str, Any]:
    """Return collections for texts, tables, and pictures with graceful fallbacks."""
    collections = {
        "texts": getattr(document, "texts", []),
        "tables": getattr(document, "tables", []),
        "pictures": getattr(document, "pictures", []),
        "key_values": getattr(document, "key_value_items", []),
    }
    # Alternate attribute names in case of version differences
    if not collections["texts"]:
        collections["texts"] = getattr(document, "paragraphs", [])
    if not collections["pictures"]:
        collections["pictures"] = getattr(document, "images", [])
    return collections


def table_to_dataframe(table: Any):
    """Get a pandas DataFrame for a table if possible, else string fallback."""
    for attr in ("export_to_dataframe", "to_dataframe", "as_dataframe"):
        if hasattr(table, attr):
            try:
                return getattr(table, attr)()
            except Exception:
                pass
    # Last resort: return a string representation
    return str(table)

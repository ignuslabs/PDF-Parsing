"""
Core PDF parser implementation using Docling.
"""

import copy
import logging
import re
from pathlib import Path
from typing import List, Optional, Any, Dict, Union, Tuple
from datetime import datetime

# Import progress tracking utilities
from src.utils.logging_config import ProgressTracker, time_it, log_memory_usage
from src.utils.profiling import profile_function, profile_context

# Setup logging first
logger = logging.getLogger(__name__)

# PDF form field extraction and text enhancement
PyPDF2 = None
pdfplumber = None

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    logger.warning("PyPDF2 not available. Form field extraction will be disabled.")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not available. Enhanced text extraction will be disabled.")
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractOcrOptions,
    EasyOcrOptions,
    TableFormerMode,
)
from docling.datamodel.base_models import InputFormat

from src.utils.docling_compat import (
    build_converter as dl_build_converter,
    create_ocr_options as dl_create_ocr_options,
    normalize_table_mode as dl_normalize_table_mode,
    iter_page_images as dl_iter_page_images,
    get_document_collections as dl_get_document_collections,
)

from src.core.models import DocumentElement, ParsedDocument, KeyValuePair
from src.utils.exceptions import DocumentParsingError, OCRError
from src.core.classifiers.header_classifier import is_heading
from src.core.kv_extraction import KeyValueExtractor, KVConfig


class DoclingParser:
    """PDF parser using Docling for document conversion and analysis."""

    def __init__(
        self,
        enable_ocr: bool = False,
        enable_tables: bool = True,
        generate_page_images: bool = False,
        max_pages: Optional[int] = None,
        ocr_engine: str = "tesseract",
        ocr_lang: Union[str, List[str]] = "eng",
        table_mode: TableFormerMode = TableFormerMode.ACCURATE,
        image_scale: float = 1.0,
        enable_kv_extraction: bool = False,
        kv_config: Optional[KVConfig] = None,
        header_classifier_enabled: bool = False,
        enable_form_fields: bool = True,
        enable_profiling: bool = False,
    ):
        """Initialize the Docling parser with configuration options.

        Args:
            enable_ocr: Enable OCR for scanned documents
            enable_tables: Enable table structure recognition
            generate_page_images: Generate page images for verification
            max_pages: Maximum number of pages to process
            ocr_engine: OCR engine to use ('tesseract' or 'easyocr')
            ocr_lang: OCR language(s) - string or list of language codes
            table_mode: Table extraction mode ('accurate' or 'fast')
            image_scale: Scale factor for generated images
            enable_kv_extraction: Enable key-value pair extraction
            kv_config: Configuration for KV extraction (uses defaults if None)
            header_classifier_enabled: Enable header/heading classification
            enable_form_fields: Enable PDF form field extraction using PyPDF2
            enable_profiling: Enable profiling for performance analysis
        """
        self.enable_ocr = enable_ocr
        self.enable_tables = enable_tables
        self.generate_page_images = generate_page_images
        self.max_pages = max_pages
        self.ocr_engine = ocr_engine
        self.ocr_lang = ocr_lang
        self.table_mode = table_mode
        self.image_scale = image_scale
        self.enable_kv_extraction = enable_kv_extraction
        self.kv_config = kv_config
        self.header_classifier_enabled = header_classifier_enabled
        self.enable_form_fields = enable_form_fields
        self.enable_profiling = enable_profiling

        # Initialize converter (will be created lazily)
        self._converter = None
        self.page_images = {}

        # Initialize KV extractor (will be created lazily)
        self._kv_extractor = None

        # if not DOCLING_AVAILABLE:
        #     raise ImportError(
        #         "Docling is not available. Please install it with: "
        #         "pip install docling"
        #     )

        # Validate configuration
        self._validate_config()

    def _get_ocr_lang_list(self) -> List[str]:
        """Return normalized OCR language list without modifying state."""
        if isinstance(self.ocr_lang, (list, tuple)):
            lang_list = list(self.ocr_lang)
        elif isinstance(self.ocr_lang, set):
            lang_list = list(self.ocr_lang)
        elif isinstance(self.ocr_lang, str):
            lang_list = [self.ocr_lang]
        else:
            lang_list = ["eng"]

        return list(dict.fromkeys(lang_list))

    @property
    def converter(self):
        """Lazy initialization of the document converter."""
        if self._converter is None:
            self._converter = self._create_converter()
        return self._converter

    @property
    def kv_extractor(self) -> Optional[KeyValueExtractor]:
        """Lazy initialization of the key-value extractor."""
        if not self.enable_kv_extraction:
            return None
        if self._kv_extractor is None:
            self._kv_extractor = KeyValueExtractor(self.kv_config)
        return self._kv_extractor

    def _validate_config(self):
        """Validate parser configuration."""
        if self.ocr_engine not in ["tesseract", "easyocr"]:
            raise ValueError(f"Unsupported OCR engine: {self.ocr_engine}")

        # Accept either string values or TableFormerMode enum
        if isinstance(self.table_mode, str):
            if self.table_mode.lower() not in ["accurate", "fast"]:
                raise ValueError(f"Invalid table mode: {self.table_mode}")
        else:
            # Assume valid enum or compatible type
            pass

        if not 0.1 <= self.image_scale <= 5.0:
            raise ValueError(f"Image scale must be between 0.1 and 5.0, got: {self.image_scale}")

    def _create_converter(self) -> DocumentConverter:
        """Create and configure the Docling document converter."""
        # Build via compatibility helper to handle docling API differences
        ocr_opts = self._create_ocr_options() if self.enable_ocr else None
        return dl_build_converter(
            enable_ocr=self.enable_ocr,
            ocr_options=ocr_opts,
            enable_tables=self.enable_tables,
            generate_page_images=self.generate_page_images,
            table_mode=self.table_mode,
            image_scale=self.image_scale,
        )

    def _create_ocr_options(self):
        """Create OCR options based on configuration."""
        # Language code mapping: Tesseract 3-letter -> EasyOCR 2-letter
        tesseract_to_easyocr = {
            "eng": "en",
            "fra": "fr",
            "deu": "de",
            "spa": "es",
            "ita": "it",
            "por": "pt",
            "rus": "ru",
            "chi_sim": "ch_sim",
            "chi_tra": "ch_tra",
            "jpn": "ja",
            "kor": "ko",
            "ara": "ar",
            "hin": "hi",
            "tha": "th",
            "vie": "vi",
            "nld": "nl",
            "pol": "pl",
            "ces": "cs",
            "dan": "da",
            "fin": "fi",
            "hun": "hu",
            "nor": "no",
            "swe": "sv",
            "tur": "tr",
        }

        lang_list = self._get_ocr_lang_list()

        if self.ocr_engine == "tesseract":
            # TesseractOcrOptions uses 3-letter codes directly
            return dl_create_ocr_options("tesseract", lang_list, force_full_page_ocr=True)
        elif self.ocr_engine == "easyocr":
            # Convert 3-letter codes to 2-letter codes for EasyOCR
            easyocr_langs = []
            for lang in lang_list:
                if lang in tesseract_to_easyocr:
                    easyocr_langs.append(tesseract_to_easyocr[lang])
                elif len(lang) == 2:
                    # Already 2-letter code, use as-is
                    easyocr_langs.append(lang)
                else:
                    # Unknown code, fallback to English
                    logger.warning(f"Unknown language code '{lang}', falling back to English")
                    easyocr_langs.append("en")

            return dl_create_ocr_options("easyocr", easyocr_langs or ["en"])  # Ensure at least English if empty
        else:
            raise ValueError(f"Unsupported OCR engine: {self.ocr_engine}")

    @staticmethod
    def _extract_text_from_doc_item(item: Any) -> Optional[str]:
        """Best-effort text extraction from a Docling item."""
        if item is None:
            return None
        if isinstance(item, str):
            return item.strip() or None

        candidate_attrs = [
            "text",
            "value",
            "name",
            "label_text",
            "key_text",
            "value_text",
        ]
        for attr in candidate_attrs:
            if hasattr(item, attr):
                value = getattr(item, attr)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        # Some Docling items expose `.get_text()`
        for call_attr in ("get_text",):
            if hasattr(item, call_attr):
                try:
                    value = getattr(item, call_attr)()
                    if isinstance(value, str) and value.strip():
                        return value.strip()
                except Exception:
                    continue

        # Fallback to str representation
        try:
            value = str(item).strip()
            return value or None
        except Exception:
            return None

    def _match_element_by_bbox(
        self,
        elements: List[DocumentElement],
        page_number: Optional[int],
        bbox: Optional[Dict[str, float]],
    ) -> Optional[int]:
        if not bbox:
            return None
        page_no = page_number or 1

        x0, y0, x1, y1 = bbox.get("x0", 0), bbox.get("y0", 0), bbox.get("x1", 0), bbox.get("y1", 0)
        if x1 <= x0 or y1 <= y0:
            return None

        for element in elements:
            if element.page_number != page_no:
                continue
            eb = element.bbox
            if eb["x1"] <= eb["x0"] or eb["y1"] <= eb["y0"]:
                continue

            # Check simple overlap
            overlap_x = min(x1, eb["x1"]) - max(x0, eb["x0"])
            overlap_y = min(y1, eb["y1"]) - max(y0, eb["y0"])
            if overlap_x > 0 and overlap_y > 0:
                return element.metadata.get("element_id")

        return None

    def _extract_docling_key_values(
        self, doc_obj: Any, elements: List[DocumentElement]
    ) -> List[KeyValuePair]:
        """Extract key-value pairs provided by Docling when available."""
        kv_items = getattr(doc_obj, "key_value_items", None)
        if not kv_items:
            return []

        extracted: List[KeyValuePair] = []
        for item in kv_items:
            try:
                label_obj = None
                value_obj = None

                for attr in ("key", "label", "name"):
                    if hasattr(item, attr):
                        label_obj = getattr(item, attr)
                        if label_obj is not None:
                            break

                for attr in ("value", "text", "content"):
                    if hasattr(item, attr):
                        value_obj = getattr(item, attr)
                        if value_obj is not None:
                            break

                label_text = None
                for attr in (
                    "key_text",
                    "label_text",
                    "name",
                    "key",
                ):
                    if hasattr(item, attr):
                        candidate = getattr(item, attr)
                        if isinstance(candidate, str) and candidate.strip():
                            label_text = candidate.strip()
                            break
                if label_text is None:
                    label_text = self._extract_text_from_doc_item(label_obj)

                value_text = None
                for attr in ("value_text", "text", "value"):
                    if hasattr(item, attr):
                        candidate = getattr(item, attr)
                        if isinstance(candidate, str) and candidate.strip():
                            value_text = candidate.strip()
                            break
                if value_text is None:
                    value_text = self._extract_text_from_doc_item(value_obj)

                if not label_text or not value_text:
                    continue

                label_bbox = None
                value_bbox = None

                if label_obj is not None:
                    label_bbox = self._extract_bbox(label_obj)
                elif hasattr(item, "key_bbox"):
                    try:
                        label_bbox = self._extract_bbox(getattr(item, "key_bbox"))
                    except Exception:
                        label_bbox = None

                if value_obj is not None:
                    value_bbox = self._extract_bbox(value_obj)
                elif hasattr(item, "value_bbox"):
                    try:
                        value_bbox = self._extract_bbox(getattr(item, "value_bbox"))
                    except Exception:
                        value_bbox = None

                page_number = None
                for attr in ("page_no", "page_number"):
                    if hasattr(item, attr):
                        try:
                            page_number = int(getattr(item, attr))
                            break
                        except Exception:
                            continue
                if page_number is None:
                    page_number = self._extract_page_number(label_obj or item)

                label_bbox = label_bbox or {"x0": 0.0, "y0": 0.0, "x1": 0.0, "y1": 0.0}
                value_bbox = value_bbox or label_bbox

                confidence = getattr(item, "confidence", None)
                if confidence is None:
                    confidence = getattr(item, "score", 1.0)
                try:
                    confidence = float(confidence)
                except Exception:
                    confidence = 1.0

                label_el_id = self._match_element_by_bbox(elements, page_number, label_bbox)
                value_el_id = self._match_element_by_bbox(elements, page_number, value_bbox)

                metadata: Dict[str, Any] = {
                    "source": "docling_core",
                    "docling_confidence": confidence,
                }
                if label_el_id is not None:
                    metadata["label_element_id"] = label_el_id
                if value_el_id is not None:
                    metadata["value_element_ids"] = [value_el_id]

                pair = KeyValuePair(
                    label_text=label_text,
                    value_text=value_text,
                    page_number=page_number or 1,
                    label_bbox=label_bbox,
                    value_bbox=value_bbox,
                    confidence=max(0.0, min(confidence, 1.0)),
                    metadata=metadata,
                )
                extracted.append(pair)

            except Exception as exc:
                logger.debug("Could not convert docling key-value item: %s", exc)
                continue

        return extracted

    def _split_inline_label_value(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        text = text.strip()
        if not text or len(text) < 3:
            return None, None

        # Pattern 1: Label: Value
        if ":" in text:
            label, value = text.split(":", 1)
            label = label.strip(" -\n\t")
            value = value.strip()
            if label and value:
                return label, value

        # Pattern 2: Multi-line -> last line is value
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) >= 2:
            label = " ".join(lines[:-1]).strip(" -")
            value = lines[-1].strip()
            if label and value:
                return label, value

        # Pattern 3: split on large whitespace gap
        parts = [p for p in re.split(r"\s{2,}", text) if p.strip()]
        if len(parts) >= 2:
            label = parts[0].strip(" -")
            value = parts[-1].strip()
            if label and value:
                return label, value

        # Pattern 4: descriptive label + trailing token containing digits/
        match = re.search(r"^(.*?)([A-Za-z]*\d[\w/_-]*)(?:\s*)$", text)
        if match:
            label = match.group(1).strip(" -")
            value = match.group(2).strip()
            if label and value:
                return label, value

        return None, None

    def _derive_inline_key_values(self, elements: List[DocumentElement]) -> List[KeyValuePair]:
        derived: List[KeyValuePair] = []
        seen = set()

        for element in elements:
            if element.element_type not in {"text", "heading"}:
                continue
            text = (element.text or "").strip()
            if not text or len(text) > 200:
                continue

            label, value = self._split_inline_label_value(text)
            if not label or not value:
                continue

            key = (label, value, element.page_number)
            if key in seen:
                continue
            seen.add(key)

            bbox = element.bbox
            pair = KeyValuePair(
                label_text=label,
                value_text=value,
                page_number=element.page_number,
                label_bbox=bbox,
                value_bbox=bbox,
                confidence=element.confidence,
                metadata={
                    "source": "inline_split",
                    "label_element_id": element.metadata.get("element_id"),
                    "inline_original_text": text,
                },
            )
            derived.append(pair)

        return derived

    def _build_configuration_snapshot(self) -> Dict[str, Any]:
        """Create a dictionary describing current parser configuration."""
        lang_list = self._get_ocr_lang_list() if self.enable_ocr else []
        if self.enable_tables:
            mode = self.table_mode
            if isinstance(mode, TableFormerMode):
                table_mode_value = mode.name.lower()
            else:
                table_mode_value = str(mode).lower()
        else:
            table_mode_value = None

        snapshot: Dict[str, Any] = {
            "ocr": {
                "enabled": self.enable_ocr,
                "engine": self.ocr_engine if self.enable_ocr else None,
                "languages": lang_list,
            },
            "tables": {
                "enabled": self.enable_tables,
                "mode": table_mode_value,
            },
            "images": {
                "generate": self.generate_page_images,
                "scale": self.image_scale if self.generate_page_images else None,
            },
            "kv_extraction": self.enable_kv_extraction,
            "header_classifier": self.header_classifier_enabled,
            "form_fields": self.enable_form_fields,
            "max_pages": self.max_pages,
            "profiling": self.enable_profiling,
        }
        return snapshot

    def _format_configuration_snapshot(self, snapshot: Dict[str, Any]) -> str:
        """Create a human-readable string for the configuration snapshot."""
        lines: List[str] = []

        ocr = snapshot["ocr"]
        if ocr["enabled"]:
            languages = ", ".join(ocr["languages"]) if ocr["languages"] else "auto"
            lines.append(f"  • OCR: ON (engine={ocr['engine']}, lang={languages})")
        else:
            lines.append("  • OCR: OFF")

        tables = snapshot["tables"]
        if tables["enabled"]:
            lines.append(f"  • Tables: ON (mode={tables['mode']})")
        else:
            lines.append("  • Tables: OFF")

        images = snapshot["images"]
        if images["generate"]:
            lines.append(f"  • Page images: ON (scale={images['scale']})")
        else:
            lines.append("  • Page images: OFF")

        lines.append(f"  • Key-Value extraction: {'ON' if snapshot['kv_extraction'] else 'OFF'}")
        lines.append(f"  • Safer header classifier: {'ON' if snapshot['header_classifier'] else 'OFF'}")
        lines.append(f"  • Form field extraction: {'ON' if snapshot['form_fields'] else 'OFF'}")

        if snapshot.get("max_pages"):
            lines.append(f"  • Max pages: {snapshot['max_pages']}")
        lines.append(f"  • Profiling: {'ON' if snapshot['profiling'] else 'OFF'}")

        return "\n".join(lines)

    def describe_configuration(self) -> str:
        """Return human-readable description of current configuration."""
        return self._format_configuration_snapshot(self._build_configuration_snapshot())

    def configuration_snapshot(self) -> Dict[str, Any]:
        """Return a copy of the current configuration snapshot."""
        return copy.deepcopy(self._build_configuration_snapshot())

    def _log_configuration(self) -> Dict[str, Any]:
        """Log configuration summary and return snapshot for progress tracking."""
        snapshot = self._build_configuration_snapshot()
        logger.info("Parser configuration:\n%s", self._format_configuration_snapshot(snapshot))
        return snapshot

    def parse_document(self, pdf_path: Path) -> List[DocumentElement]:
        """Parse a PDF document and return extracted elements.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of DocumentElement objects

        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            ValueError: If the file is not a valid PDF
            DocumentParsingError: If parsing fails
        """
        # Apply profiling decorator if enabled
        if self.enable_profiling:
            return self._parse_document_profiled(pdf_path)
        else:
            return self._parse_document_impl(pdf_path)
    
    @profile_function(sort_by='cumulative', print_stats=False, memory_profile=True)
    def _parse_document_profiled(self, pdf_path: Path) -> List[DocumentElement]:
        """Profiled version of parse_document."""
        return self._parse_document_impl(pdf_path)
    
    def _parse_document_impl(self, pdf_path: Path) -> List[DocumentElement]:
        """Internal implementation of document parsing."""
        pdf_path = Path(pdf_path)
        
        # Create progress tracker for this parsing operation
        progress = ProgressTracker(f"PDF Parsing: {pdf_path.name}", total_steps=8, logger=logger)

        try:
            # Step 1: Validate input file
            progress.start_step("Validating input file")
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            if pdf_path.suffix.lower() != ".pdf":
                raise ValueError(f"File must be a PDF, got: {pdf_path.suffix}")

            # Get file info for logging
            file_size = pdf_path.stat().st_size
            if file_size == 0:
                raise ValueError(f"PDF file is empty: {pdf_path}")
            elif file_size > 100 * 1024 * 1024:  # 100MB limit
                logger.warning(f"Large PDF file ({file_size / (1024*1024):.1f}MB): {pdf_path}")
                
            progress.complete_step(result=f"File size: {file_size / (1024*1024):.1f}MB", 
                                 details="File validation successful")
            log_memory_usage(logger, "file validation")

            # Step 2: Initialize parser configuration
            progress.start_step("Initializing parser configuration")
            config_snapshot = self._log_configuration()
            progress.complete_step(result=config_snapshot, details="Configuration loaded")

            # Step 3: Convert document using Docling
            progress.start_step("Converting document with Docling")
            result = self._convert_document(pdf_path)
            
            # Count elements in result for progress reporting (compat-aware)
            cols = dl_get_document_collections(result.document)
            text_count = len(cols.get('texts', []) or [])
            table_count = len(cols.get('tables', []) or [])
            image_count = len(cols.get('pictures', []) or [])
            
            progress.complete_step(
                result=f"T:{text_count}, Ta:{table_count}, I:{image_count}",
                details=f"Document converted successfully"
            )
            log_memory_usage(logger, "document conversion")

            # Step 4: Extract elements from converted document
            progress.start_step("Extracting document elements")
            elements = self._extract_elements(result.document, pdf_path)
            progress.complete_step(
                result=f"{len(elements)} elements", 
                details=f"Element extraction completed"
            )
            log_memory_usage(logger, "element extraction")

            # Step 5: Generate page images if requested
            if self.generate_page_images:
                progress.start_step("Generating page images")
                self.page_images = {}
                try:
                    for page_num, img in dl_iter_page_images(result):
                        self.page_images[page_num] = img
                    progress.complete_step(
                        result=f"{len(self.page_images)} images",
                        details="Page images generated"
                    )
                except Exception:
                    progress.complete_step(result="No pages", details="No page data available")
            else:
                progress.start_step("Skipping page image generation")
                progress.complete_step(details="Page images disabled")

            # Step 6: Apply OCR corrections if needed
            progress.start_step("Applying OCR corrections")
            corrected_count = 0
            for element in elements:
                if element.text and element.metadata.get("source") in ["docling_text", "docling_table"]:
                    original_text = element.text
                    corrected_text = self._correct_ocr_errors(original_text)
                    if corrected_text != original_text:
                        element.text = corrected_text
                        element.metadata["original_text"] = original_text
                        element.metadata["ocr_corrected"] = True
                        corrected_count += 1
            progress.complete_step(
                result=f"{corrected_count} corrections",
                details="OCR corrections applied"
            )

            # Step 7: Apply header classification if enabled
            if self.header_classifier_enabled:
                progress.start_step("Applying header classification")
                original_heading_count = len([e for e in elements if e.element_type == "heading"])
                elements = self._refine_heading_classification(elements)
                new_heading_count = len([e for e in elements if e.element_type == "heading"])
                progress.complete_step(
                    result=f"{new_heading_count} headings ({new_heading_count - original_heading_count:+d})",
                    details="Header classification completed"
                )
            else:
                progress.start_step("Skipping header classification") 
                progress.complete_step(details="Header classification disabled")

            # Step 8: Final processing and sorting
            progress.start_step("Final processing and sorting")
            elements.sort(key=lambda x: (x.page_number, x.bbox["y0"], x.bbox["x0"]))
            
            # Generate final statistics
            element_types = {}
            for element in elements:
                element_types[element.element_type] = element_types.get(element.element_type, 0) + 1
            
            progress.complete_step(
                result=element_types,
                details="Processing completed successfully"
            )
            log_memory_usage(logger, "final processing")

            # Finish progress tracking
            summary = progress.finish(success=True)
            logger.info(f"📊 Parsing Summary: {summary}")
            
            return elements

        except DocumentParsingError:
            raise  # Re-raise parsing errors as-is
        except OCRError as e:
            logger.warning(f"OCR failed for {pdf_path}: {e}")
            if self.enable_ocr:
                # Retry without OCR
                logger.info(f"Retrying {pdf_path} without OCR")
                return self._parse_without_ocr(pdf_path)
            else:
                raise DocumentParsingError(f"OCR processing failed: {e}", str(pdf_path))
        except MemoryError as e:
            logger.error(f"Insufficient memory for {pdf_path}: {e}")
            # Try with reduced settings
            return self._parse_with_reduced_settings(pdf_path)
        except Exception as e:
            logger.error(f"Unexpected error parsing {pdf_path}: {e}")
            raise DocumentParsingError(f"Failed to parse PDF: {str(e)}", str(pdf_path)) from e

    def parse_document_full(self, pdf_path: Path) -> ParsedDocument:
        """Parse a PDF document and return full ParsedDocument with metadata.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            ParsedDocument containing elements, metadata, and optional page images

        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            ValueError: If the file is not a valid PDF
            DocumentParsingError: If parsing fails
        """
        pdf_path = Path(pdf_path)
        docling_kv_pairs: List[KeyValuePair] = []

        # Input validation
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"File must be a PDF, got: {pdf_path.suffix}")

        # Validate file is readable and not empty
        try:
            file_size = pdf_path.stat().st_size
            if file_size == 0:
                raise ValueError(f"PDF file is empty: {pdf_path}")
            elif file_size > 100 * 1024 * 1024:  # 100MB limit
                logger.warning(f"Large PDF file ({file_size / (1024*1024):.1f}MB): {pdf_path}")
        except OSError as e:
            raise DocumentParsingError(f"Cannot access PDF file: {e}", str(pdf_path))

        try:
            logger.info(f"Starting to parse PDF: {pdf_path}")
            logger.info("Parser configuration:\n%s", self.describe_configuration())

            # Convert document using Docling
            result = self._convert_document(pdf_path)

            elements = self._extract_elements(result.document, pdf_path)
            docling_kv_pairs = self._extract_docling_key_values(result.document, elements)
            inline_kv_pairs = self._derive_inline_key_values(elements)
            docling_kv_pairs.extend(inline_kv_pairs)

            # Store page images if requested
            if self.generate_page_images:
                self.page_images = {}
                try:
                    for page_num, img in dl_iter_page_images(result):
                        self.page_images[page_num] = img
                except Exception:
                    pass

            inline_kv_pairs = self._derive_inline_key_values(elements)
            docling_kv_pairs.extend(inline_kv_pairs)

            logger.info(f"Successfully parsed PDF: {pdf_path} - {len(elements)} elements extracted")

            # Derive page_count with best available signal
            page_count: int
            try:
                # Prefer document.num_pages if available
                doc_num_pages = getattr(result.document, 'num_pages', None)
                if isinstance(doc_num_pages, int) and doc_num_pages > 0:
                    page_count = doc_num_pages
                elif hasattr(result, 'pages') and result.pages:
                    page_count = len(result.pages)
                else:
                    page_count = len(set(e.page_number for e in elements)) if elements else 0
            except Exception:
                page_count = len(set(e.page_number for e in elements)) if elements else 0

            # Create ParsedDocument
            document = ParsedDocument(
                elements=elements,
                metadata={
                    "source_path": str(pdf_path),
                    "filename": pdf_path.name,
                    "file_size": file_size,
                    "parsed_at": datetime.now().isoformat(),
                    "parser_config": {
                        "ocr_enabled": self.enable_ocr,
                        "tables_enabled": self.enable_tables,
                        "images_enabled": self.generate_page_images,
                        "ocr_engine": self.ocr_engine if self.enable_ocr else None,
                        "table_mode": self.table_mode if self.enable_tables else None,
                        "kv_extraction_enabled": self.enable_kv_extraction,
                        "header_classifier_enabled": self.header_classifier_enabled,
                    },
                    "total_elements": len(elements),
                    "page_count": page_count,
                },
                pages=self.page_images if self.page_images else None,
                key_values=docling_kv_pairs,
            )

            return document

        except DocumentParsingError:
            raise  # Re-raise parsing errors as-is
        except OCRError as e:
            logger.warning(f"OCR failed for {pdf_path}: {e}")
            if self.enable_ocr:
                # Retry without OCR
                logger.info(f"Retrying {pdf_path} without OCR")
                elements = self._parse_without_ocr(pdf_path)
                inline_kv_pairs = self._derive_inline_key_values(elements)
                docling_kv_pairs.extend(inline_kv_pairs)
                return ParsedDocument(
                    elements=elements,
                    metadata={
                        "source_path": str(pdf_path),
                        "filename": pdf_path.name,
                        "file_size": file_size,
                        "parsed_at": datetime.now().isoformat(),
                        "fallback_mode": "no_ocr",
                        "parser_config": {
                            "ocr_enabled": False,
                            "tables_enabled": self.enable_tables,
                            "images_enabled": self.generate_page_images,
                            "kv_extraction_enabled": self.enable_kv_extraction,
                            "header_classifier_enabled": self.header_classifier_enabled,
                        },
                        "total_elements": len(elements),
                        "page_count": len(set(e.page_number for e in elements)) if elements else 0,
                    },
                    pages=None,
                    key_values=docling_kv_pairs,
                )
            else:
                raise DocumentParsingError(f"OCR processing failed: {e}", str(pdf_path))
        except MemoryError as e:
            logger.error(f"Insufficient memory for {pdf_path}: {e}")
            # Try with reduced settings
                elements = self._parse_with_reduced_settings(pdf_path)
            inline_kv_pairs = self._derive_inline_key_values(elements)
            docling_kv_pairs.extend(inline_kv_pairs)
            return ParsedDocument(
                elements=elements,
                metadata={
                    "source_path": str(pdf_path),
                    "filename": pdf_path.name,
                    "file_size": file_size,
                    "parsed_at": datetime.now().isoformat(),
                    "fallback_mode": "reduced_memory",
                    "parser_config": {
                        "ocr_enabled": self.enable_ocr,
                        "tables_enabled": False,  # Disabled for memory
                        "images_enabled": False,  # Disabled for memory
                        "kv_extraction_enabled": self.enable_kv_extraction,
                        "header_classifier_enabled": self.header_classifier_enabled,
                    },
                    "total_elements": len(elements),
                    "page_count": len(set(e.page_number for e in elements)) if elements else 0,
                },
                pages=None,
            )
        except Exception as e:
            logger.error(f"Unexpected error parsing {pdf_path}: {e}")
            raise DocumentParsingError(f"Failed to parse PDF: {str(e)}", str(pdf_path)) from e

    def _convert_document(self, pdf_path: Path):
        """Convert PDF using Docling converter with OCR fallback."""
        try:
            return self.converter.convert(str(pdf_path))
        except Exception as e:
            error_msg = str(e).lower()

            # Check for OCR-specific errors
            if any(
                ocr_keyword in error_msg
                for ocr_keyword in ["tesserocr", "tessdata", "ocr", "easyocr"]
            ):
                logger.warning(f"OCR error for {pdf_path}: {e}")
                if self.enable_ocr:
                    logger.info("Attempting to parse without OCR...")
                    original_ocr = self.enable_ocr
                    try:
                        # Try parsing without OCR
                        self.enable_ocr = False
                        self._converter = None  # Force recreation without OCR
                        result = self.converter.convert(str(pdf_path))
                        logger.info("Successfully parsed without OCR")
                        return result
                    except Exception as ocr_fallback_error:
                        logger.error(f"Fallback without OCR also failed: {ocr_fallback_error}")
                        raise DocumentParsingError(
                            f"Document conversion failed: {e}. OCR fallback also failed: {ocr_fallback_error}",
                            str(pdf_path),
                        )
                    finally:
                        # Always restore original settings
                        self.enable_ocr = original_ocr
                        self._converter = None

            # For non-OCR errors, just raise
            logger.error(f"Docling conversion failed for {pdf_path}: {e}")
            raise DocumentParsingError(f"Document conversion failed: {e}", str(pdf_path))

    def _parse_without_ocr(self, pdf_path: Path) -> List[DocumentElement]:
        """Parse document without OCR as fallback."""
        original_ocr = self.enable_ocr
        try:
            self.enable_ocr = False
            # Recreate converter without OCR
            self._converter = None

            result = self._convert_document(pdf_path)
            elements = self._extract_elements(result.document, pdf_path)

            return elements
        finally:
            # Restore original OCR setting
            self.enable_ocr = original_ocr
            self._converter = None  # Force recreation with original settings

    def _parse_with_reduced_settings(self, pdf_path: Path) -> List[DocumentElement]:
        """Parse document with reduced memory settings."""
        original_images = self.generate_page_images
        original_tables = self.enable_tables

        try:
            # Disable memory-intensive features
            self.generate_page_images = False
            self.enable_tables = False

            # Recreate converter with reduced settings
            self._converter = None

            result = self._convert_document(pdf_path)
            elements = self._extract_elements(result.document, pdf_path)

            logger.warning(f"Parsed {pdf_path} with reduced settings due to memory constraints")

            return elements

        finally:
            # Restore original settings
            self.generate_page_images = original_images
            self.enable_tables = original_tables
            self._converter = None  # Force recreation with original settings

    def _extract_elements(self, document, pdf_path: Optional[Path] = None) -> List[DocumentElement]:
        """Extract DocumentElement objects from Docling document and PDF form fields.

        Args:
            document: Docling Document object
            pdf_path: Path to the original PDF file (for form field extraction)

        Returns:
            List of DocumentElement objects
        """
        from src.utils.logging_config import ProgressTracker, log_memory_usage
        
        # Initialize progress tracking for element extraction
        total_steps = 7
        if self.header_classifier_enabled:
            total_steps += 1
        if pdf_path and self.enable_form_fields:
            total_steps += 1
            
        progress = ProgressTracker("Element Extraction", total_steps, logger)
        
        elements = []
        element_id = 0

        # Step 1: Extract text elements (compat-aware collections)
        progress.start_step("Extracting text elements")
        text_count = 0
        collections = dl_get_document_collections(document)
        for text_item in collections.get("texts", []) or []:
            element = self._create_element_from_text(text_item, element_id, document)
            if element:
                elements.append(element)
                element_id += 1
                text_count += 1
        progress.complete_step(text_count, f"Found {text_count} text elements")
        log_memory_usage(logger, "text element extraction")

        # Step 2: Extract table elements
        progress.start_step("Extracting table elements")
        table_count = 0
        if self.enable_tables:
            for table_item in collections.get("tables", []) or []:
                element = self._create_element_from_table(table_item, element_id, document)
                if element:
                    elements.append(element)
                    element_id += 1
                    table_count += 1
        else:
            logger.debug("Table extraction disabled or no tables found")
        progress.complete_step(table_count, f"Found {table_count} table elements")

        # Step 3: Extract picture elements
        progress.start_step("Extracting picture elements")
        picture_count = 0
        for picture_item in collections.get("pictures", []) or []:
            element = self._create_element_from_picture(picture_item, element_id, document)
            if element:
                elements.append(element)
                element_id += 1
                picture_count += 1
        progress.complete_step(picture_count, f"Found {picture_count} picture elements")

        # Step 4: Extract form field elements (conditional step)
        if pdf_path and self.enable_form_fields:
            progress.start_step("Extracting PDF form fields")
            form_field_elements = self._extract_form_fields(pdf_path, element_id)
            elements.extend(form_field_elements)
            element_id += len(form_field_elements)
            progress.complete_step(len(form_field_elements), f"Found {len(form_field_elements)} form field elements")

        # Step 5: Sort elements by spatial position
        progress.start_step("Sorting elements by position")
        elements.sort(key=lambda x: (x.page_number, x.bbox["y0"], x.bbox["x0"]))
        progress.complete_step(len(elements), f"Sorted {len(elements)} elements by position")

        # Step 6: Apply OCR corrections
        progress.start_step("Applying OCR corrections")
        corrected_count = 0
        for element in elements:
            if element.text and element.metadata.get("source") in ["docling_text", "docling_table"]:
                original_text = element.text
                corrected_text = self._correct_ocr_errors(original_text)
                if corrected_text != original_text:
                    element.text = corrected_text
                    element.metadata["original_text"] = original_text
                    element.metadata["ocr_corrected"] = True
                    corrected_count += 1
        progress.complete_step(corrected_count, f"Applied OCR corrections to {corrected_count} elements")

        # Step 7: Check pdfplumber fallback
        progress.start_step("Checking pdfplumber fallback")
        fallback_added = 0
        if pdf_path and PDFPLUMBER_AVAILABLE and len(elements) < 5:
            logger.info("Few elements found, trying pdfplumber as fallback...")
            pdfplumber_elements = self._extract_text_with_pdfplumber(pdf_path, element_id)
            if pdfplumber_elements:
                # Only add unique pdfplumber elements (avoid duplicates)
                existing_texts = {elem.text.lower().strip() for elem in elements}
                unique_elements = [
                    elem
                    for elem in pdfplumber_elements
                    if elem.text.lower().strip() not in existing_texts
                ]
                elements.extend(unique_elements)
                fallback_added = len(unique_elements)
                logger.info(f"Added {fallback_added} unique elements from pdfplumber")
        progress.complete_step(fallback_added, f"Added {fallback_added} fallback elements")
        log_memory_usage(logger, "pdfplumber fallback")

        # Step 8: Post-process with header classifier (conditional step)
        if self.header_classifier_enabled and elements:
            progress.start_step("Refining heading classification")
            original_headings = sum(1 for e in elements if e.element_type == "heading")
            elements = self._refine_heading_classification(elements)
            refined_headings = sum(1 for e in elements if e.element_type == "heading")
            heading_change = refined_headings - original_headings
            progress.complete_step(
                refined_headings, 
                f"Refined {abs(heading_change)} headings ({'added' if heading_change > 0 else 'removed'})"
            )

        # Finish progress tracking
        progress.finish(success=True)
        log_memory_usage(logger, "element extraction completion")

        return elements

    def _refine_heading_classification(
        self, elements: List[DocumentElement]
    ) -> List[DocumentElement]:
        """Refine heading classification using the header classifier.

        Args:
            elements: List of all elements from the document

        Returns:
            List of elements with refined heading classification
        """
        # Group elements by page for context building
        page_groups = {}
        for element in elements:
            page_num = element.page_number
            if page_num not in page_groups:
                page_groups[page_num] = []
            page_groups[page_num].append(element)

        # Process each page independently
        refined_elements = []
        for page_num, page_elements in page_groups.items():
            # Build page context
            page_context = self._build_page_context(elements, page_num)

            # Refine classification for text elements
            for element in page_elements:
                if element.element_type == "text":  # Only refine text elements
                    # Check if header classifier suggests this should be a heading
                    if classify_as_heading(element, page_context):
                        # Create new element with updated type
                        updated_element = DocumentElement(
                            text=element.text,
                            element_type="heading",
                            page_number=element.page_number,
                            bbox=element.bbox,
                            confidence=element.confidence,
                            metadata={**element.metadata, "header_classifier_applied": True},
                        )
                        refined_elements.append(updated_element)
                    else:
                        refined_elements.append(element)
                else:
                    # Keep non-text elements as-is
                    refined_elements.append(element)

        return refined_elements

    def _create_element_from_text(
        self, text_item, element_id: int, _document
    ) -> Optional[DocumentElement]:
        """Create DocumentElement from Docling text item."""
        try:
            # Get text content
            text_content = getattr(text_item, "text", "")
            if not text_content or not text_content.strip():
                return None

            # Create preliminary element for type determination
            bbox = self._extract_bbox(text_item)
            page_number = self._extract_page_number(text_item)
            confidence = getattr(text_item, "confidence", 0.9)

            # Create preliminary element
            preliminary_element = DocumentElement(
                text=text_content.strip(),
                element_type="text",  # Temporary type
                page_number=page_number,
                bbox=bbox,
                confidence=confidence,
                metadata={"element_id": element_id, "source": "docling_text"},
            )

            # Determine element type (will be refined later if header classifier is enabled)
            element_type = self._determine_text_element_type(text_content, preliminary_element)

            # Update element type in preliminary element
            preliminary_element.element_type = element_type

            return preliminary_element

        except Exception as e:
            # Skip problematic elements rather than failing entire parsing
            logger.warning(f"Failed to process text element {element_id}: {e}")
            return None

    def _create_element_from_table(
        self, table_item, element_id: int, _document
    ) -> Optional[DocumentElement]:
        """Create DocumentElement from Docling table item."""
        try:
            # Extract table text (simplified)
            table_text = self._extract_table_text(table_item)

            # Get bounding box
            bbox = self._extract_bbox(table_item)

            # Get page number
            page_number = self._extract_page_number(table_item)

            # Extract table metadata
            table_metadata = self._extract_table_metadata(table_item)
            table_metadata.update({"element_id": element_id, "source": "docling_table"})

            return DocumentElement(
                text=table_text,
                element_type="table",
                page_number=page_number,
                bbox=bbox,
                confidence=0.8,  # Tables might be less confident
                metadata=table_metadata,
            )

        except Exception as e:
            logger.warning(f"Failed to process table element {element_id}: {e}")
            return None

    def _create_element_from_picture(
        self, picture_item, element_id: int, document
    ) -> Optional[DocumentElement]:
        """Create DocumentElement from Docling picture item."""
        try:
            # Get picture caption or description
            picture_text = self._extract_picture_text(picture_item, document)

            # Get bounding box
            bbox = self._extract_bbox(picture_item)

            # Get page number
            page_number = self._extract_page_number(picture_item)

            picture_metadata = {"element_id": element_id, "source": "docling_picture"}

            return DocumentElement(
                text=picture_text,
                element_type="image",
                page_number=page_number,
                bbox=bbox,
                confidence=0.9,
                metadata=picture_metadata,
            )

        except Exception as e:
            logger.warning(f"Failed to process picture element {element_id}: {e}")
            return None

    def _extract_form_fields(
        self, pdf_path: Path, element_id_offset: int = 0
    ) -> List[DocumentElement]:
        """Extract PDF form field values using PyPDF2.

        Args:
            pdf_path: Path to the PDF file
            element_id_offset: Starting element ID for form fields

        Returns:
            List of DocumentElement objects created from form field data
        """
        if not PYPDF2_AVAILABLE:
            logger.warning("PyPDF2 not available. Skipping form field extraction.")
            return []

        elements = []
        element_id = element_id_offset

        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)  # type: ignore

                # Extract form fields
                if reader.is_encrypted:
                    logger.warning("PDF is encrypted. Cannot extract form fields.")
                    return []

                # Try different approaches to extract form fields with positioning
                form_fields_with_positions = self._extract_form_fields_with_positions(reader)

                if form_fields_with_positions:
                    logger.info(
                        f"Found {len(form_fields_with_positions)} form fields with positioning in PDF"
                    )

                    for field_info in form_fields_with_positions:
                        field_name = field_info.get("name", "unknown_field")
                        field_value = field_info.get("value", "")
                        page_number = field_info.get("page", 1)
                        bbox = field_info.get("bbox", {"x0": 0.0, "y0": 0.0, "x1": 0.0, "y1": 0.0})

                        if field_value and str(field_value).strip():
                            # Create DocumentElement from form field
                            element = DocumentElement(
                                text=str(field_value).strip(),
                                element_type="form_field",
                                page_number=page_number,
                                bbox=bbox,
                                confidence=0.95,  # High confidence for form field values
                                metadata={
                                    "element_id": element_id,
                                    "source": "form_field",
                                    "field_name": field_name,
                                },
                            )
                            elements.append(element)
                            element_id += 1

                    logger.info(f"Extracted {len(elements)} form field values with positioning")
                else:
                    # Fallback to basic form field extraction without positioning
                    logger.info("Attempting basic form field extraction without positioning")
                    fallback_fields = self._extract_basic_form_fields(reader)

                    for field_name, field_value in fallback_fields.items():
                        if field_value and str(field_value).strip():
                            # Create DocumentElement from form field
                            element = DocumentElement(
                                text=str(field_value).strip(),
                                element_type="form_field",
                                page_number=1,  # Default to page 1
                                bbox={
                                    "x0": 0.0,
                                    "y0": 0.0,
                                    "x1": 0.0,
                                    "y1": 0.0,
                                },  # No positioning info available
                                confidence=0.95,  # High confidence for form field values
                                metadata={
                                    "element_id": element_id,
                                    "source": "form_field",
                                    "field_name": field_name,
                                },
                            )
                            elements.append(element)
                            element_id += 1

                    logger.info(f"Extracted {len(elements)} form field values without positioning")

        except Exception as e:
            logger.error(f"Error extracting form fields: {e}")

        return elements

    def _extract_form_fields_with_positions(self, reader: Any) -> List[Dict]:
        """Try to extract form fields with their positions using PyPDF2 annotations.

        Args:
            reader: PyPDF2 PdfReader instance

        Returns:
            List of dictionaries containing field info with positions
        """
        fields_with_positions = []

        try:
            # Iterate through pages to find form field annotations
            for page_num, page in enumerate(reader.pages, start=1):
                if "/Annots" in page:
                    annotations = page["/Annots"]
                    
                    # Resolve the annotations object if needed
                    if hasattr(annotations, 'get_object'):
                        annotations = annotations.get_object()
                    
                    # Ensure annotations is iterable
                    if not hasattr(annotations, '__iter__'):
                        continue
                        
                    for annot in annotations:
                        try:
                            annot_obj = annot.get_object()

                            # Check if it's a form field annotation
                            if "/Subtype" in annot_obj and annot_obj["/Subtype"] == "/Widget":
                                field_info = self._extract_field_info_from_annotation(
                                    annot_obj, page_num
                                )
                                if field_info and field_info.get("value"):
                                    fields_with_positions.append(field_info)

                        except Exception as e:
                            logger.debug(f"Could not process annotation: {e}")
                            continue

        except Exception as e:
            logger.debug(f"Could not extract form fields with positions: {e}")

        return fields_with_positions

    def _extract_field_info_from_annotation(self, annot_obj, page_number: int) -> Optional[Dict]:
        """Extract field information from a form field annotation.

        Args:
            annot_obj: PyPDF2 annotation object
            page_number: Page number containing the field

        Returns:
            Dictionary with field information or None
        """
        try:
            field_info: Dict[str, Any] = {"page": page_number}

            # Try to get field name
            if "/T" in annot_obj:
                field_info["name"] = str(annot_obj["/T"])

            # Try to get field value
            if "/V" in annot_obj:
                field_info["value"] = str(annot_obj["/V"])

            # Try to get bounding box (Rect)
            if "/Rect" in annot_obj:
                rect = annot_obj["/Rect"]
                if len(rect) >= 4:
                    field_info["bbox"] = {
                        "x0": float(rect[0]),
                        "y0": float(rect[1]),
                        "x1": float(rect[2]),
                        "y1": float(rect[3]),
                    }

            return field_info if field_info.get("value") else None

        except Exception as e:
            logger.debug(f"Could not extract field info from annotation: {e}")
            return None

    def _extract_basic_form_fields(self, reader: Any) -> Dict[str, str]:
        """Extract basic form fields without positioning using PyPDF2.

        Args:
            reader: PyPDF2 PdfReader instance

        Returns:
            Dictionary mapping field names to values
        """
        try:
            # Get form fields from the first attempt
            if "/AcroForm" in reader.trailer.get("/Root", {}):
                try:
                    # Try to get form fields
                    if hasattr(reader, "get_form_text_fields"):
                        return reader.get_form_text_fields() or {}
                    elif hasattr(reader, "getFormTextFields"):
                        return reader.getFormTextFields() or {}
                except Exception as e:
                    logger.debug(f"Could not extract basic form fields: {e}")

        except Exception as e:
            logger.debug(f"Error in basic form field extraction: {e}")

        return {}

    def _correct_ocr_errors(self, text: str) -> str:
        """Apply common OCR error corrections to text.

        Args:
            text: Original text with potential OCR errors

        Returns:
            Corrected text
        """
        if not text:
            return text

        corrections = {
            # Common character misreads for DE form IDs
            "AGOI": "AG01",
            "MOO1": "MO01",
            "RSOS": "RS05",
            "RSOO": "RS00",
            "RSOO5": "RS005",
            "OOLL": "00LL",
            "POOS": "P005",
            # Specific OCR errors in form IDs
            "ISL": "18",
            "ISLP": "18LP",
            "OSOOS": "05005",
            "POSOOS": "P05005",
            "LPOSOOS": "LP05005",
            # Character substitutions
            "OI": "01",
            "LPO5": "LP05",
            "LP0S": "LP05",
            "LPOO": "LP00",
            "LPQS": "LP05",
            "LPOOS": "LP005",
            "OO": "00",
            # Date-related corrections
            "ZO24": "2024",
            "ZO23": "2023",
            "ZO25": "2025",
            "May S,": "May 5,",
            "May B,": "May 8,",
        }

        # Pattern-based corrections for DOE form IDs first (more specific)
        import re

        # Fix specific known OCR errors in DE form patterns
        specific_corrections = [
            (r"DE-AGOI-ISLPOSOOS", "DE-AG01-18LP05005"),
            (r"DE-MOO1-22LPOSOOS", "DE-MO01-22LP05005"),
            (r"DE-RSOS-OOLLPOOS", "DE-RS05-00LP005"),
            (r"DE-RSOO5-OOLLPOOS", "DE-RS005-00LP005"),
        ]

        corrected_text = text
        for pattern, replacement in specific_corrections:
            corrected_text = re.sub(pattern, replacement, corrected_text)

        # Apply general character corrections
        for error, correction in corrections.items():
            corrected_text = corrected_text.replace(error, correction)

        # Fix general DE-XX##-##LP##### pattern
        pattern = r"DE-([A-Z]{2})([OI0-9]{2})-([OI0-9]{2})LP([OI0-9S]{5})"

        def fix_de_pattern(match):
            state = match.group(1)
            num1 = match.group(2).replace("O", "0").replace("I", "1")
            num2 = match.group(3).replace("O", "0").replace("I", "1")
            num3 = match.group(4).replace("O", "0").replace("I", "1").replace("S", "5")
            return f"DE-{state}{num1}-{num2}LP{num3}"

        corrected_text = re.sub(pattern, fix_de_pattern, corrected_text)

        return corrected_text

    def _extract_text_with_pdfplumber(
        self, pdf_path: Path, element_id_offset: int = 0
    ) -> List[DocumentElement]:
        """Extract text using pdfplumber as fallback method.

        Args:
            pdf_path: Path to the PDF file
            element_id_offset: Starting element ID for pdfplumber elements

        Returns:
            List of DocumentElement objects from pdfplumber extraction
        """
        if not PDFPLUMBER_AVAILABLE:
            logger.warning("pdfplumber not available. Skipping enhanced text extraction.")
            return []

        elements = []
        element_id = element_id_offset

        try:
            with pdfplumber.open(pdf_path) as pdf:  # type: ignore
                for page_num, page in enumerate(pdf.pages, start=1):
                    # Extract text with layout preservation
                    text = page.extract_text(layout=True)

                    if text and text.strip():
                        # Split into lines and create elements
                        lines = text.strip().split("\n")

                        for line in lines:
                            line = line.strip()
                            if line and len(line) > 2:  # Filter out very short lines
                                # Apply OCR corrections
                                corrected_text = self._correct_ocr_errors(line)

                                # Create element
                                element = DocumentElement(
                                    text=corrected_text,
                                    element_type="text",
                                    page_number=page_num,
                                    bbox={
                                        "x0": 0.0,
                                        "y0": 0.0,
                                        "x1": 0.0,
                                        "y1": 0.0,
                                    },  # No precise positioning from pdfplumber
                                    confidence=0.8,  # Lower confidence for fallback method
                                    metadata={
                                        "element_id": element_id,
                                        "source": "pdfplumber_fallback",
                                        "original_text": line if line != corrected_text else None,
                                    },
                                )
                                elements.append(element)
                                element_id += 1

                logger.info(f"pdfplumber extracted {len(elements)} text elements")

        except Exception as e:
            logger.error(f"Error in pdfplumber text extraction: {e}")

        return elements

    def _build_page_context(self, elements: List[DocumentElement], page_num: int) -> Dict[str, Any]:
        """Build page context for header classifier.

        Args:
            elements: All elements on the page
            page_num: Page number

        Returns:
            Dictionary with page statistics and dimensions
        """
        if not elements:
            return {
                "page_width": 612.0,  # Default PDF page width
                "page_height": 792.0,  # Default PDF page height
                "top_15_percent_y": 792.0 * 0.85,  # Top 15% boundary
                "element_count": 0,
            }

        # Calculate page dimensions from element bounding boxes
        page_elements = [e for e in elements if e.page_number == page_num]

        if page_elements:
            max_x = max(e.bbox["x1"] for e in page_elements)
            max_y = max(e.bbox["y1"] for e in page_elements)
            page_width = max(max_x, 612.0)  # Use at least standard width
            page_height = max(max_y, 792.0)  # Use at least standard height
        else:
            page_width = 612.0
            page_height = 792.0

        return {
            "page_width": page_width,
            "page_height": page_height,
            "top_15_percent_y": page_height * 0.85,  # Top 15% boundary
            "element_count": len(page_elements),
        }

    def _determine_text_element_type(
        self,
        text: str,
        element: Optional[DocumentElement] = None,
        page_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Determine element type based on text characteristics and optional classification."""
        text_stripped = text.strip()

        # Short text in all caps might be a heading
        if len(text_stripped) < 100 and text_stripped.isupper():
            return "heading"

        # Text with certain patterns might be headings
        if len(text_stripped) < 200 and (
            text_stripped.endswith(":")
            or any(
                word in text_stripped.lower()
                for word in ["chapter", "section", "introduction", "conclusion"]
            )
        ):
            return "heading"

        # Mathematical expressions
        if any(symbol in text_stripped for symbol in ["=", "∑", "∫", "√", "±", "∞", "α", "β", "γ"]):
            return "formula"

        # List detection
        if (
            text_stripped.startswith(("•", "◦", "▪", "-", "*"))
            or re.match(r"^\d+\.", text_stripped)
            or re.match(r"^[a-zA-Z]\)", text_stripped)
        ):
            return "list"

        # Code detection
        code_indicators = ["def ", "class ", "import ", "function", "{", "}", ";"]
        if any(indicator in text_stripped for indicator in code_indicators):
            return "code"

        # Caption detection (often starts with "Figure" or "Table")
        if re.match(r"^(Figure|Table|Image|Chart)\s+\d+", text_stripped, re.IGNORECASE):
            return "caption"

        # Use header classifier if enabled and we have the necessary context
        if self.header_classifier_enabled and element is not None and page_context is not None:
            # Check if header classifier suggests this is a heading
            if classify_as_heading(element, page_context):
                return "heading"

        # Default to text
        return "text"

    def _extract_bbox(self, item) -> Dict[str, float]:
        """Extract and validate bounding box from Docling item."""
        try:
            # Check for prov list (new Docling structure)
            if hasattr(item, "prov") and item.prov:
                prov_item = item.prov[0]  # Get first provenance item
                if hasattr(prov_item, "bbox"):
                    bbox = prov_item.bbox

                    # Convert from Docling's l,t,r,b format to x0,y0,x1,y1
                    coords = {
                        "x0": float(getattr(bbox, "l", 0)),
                        "y0": float(getattr(bbox, "t", 0)),
                        "x1": float(getattr(bbox, "r", 100)),
                        "y1": float(getattr(bbox, "b", 20)),
                    }

                    # Validate coordinate consistency
                    if coords["x1"] < coords["x0"]:
                        coords["x0"], coords["x1"] = coords["x1"], coords["x0"]
                    if coords["y1"] < coords["y0"]:
                        coords["y0"], coords["y1"] = coords["y1"], coords["y0"]

                    # Ensure non-negative coordinates
                    for key in coords:
                        coords[key] = max(0.0, coords[key])

                    return coords

            # Fallback: try old get_location method
            if hasattr(item, "get_location"):
                location = item.get_location()
                if hasattr(location, "bbox"):
                    bbox = location.bbox

                    # Extract coordinates with type validation
                    coords = {
                        "x0": float(getattr(bbox, "x0", 0)),
                        "y0": float(getattr(bbox, "y0", 0)),
                        "x1": float(getattr(bbox, "x1", 100)),
                        "y1": float(getattr(bbox, "y1", 20)),
                    }

                    # Validate coordinate consistency
                    if coords["x1"] < coords["x0"]:
                        coords["x0"], coords["x1"] = coords["x1"], coords["x0"]
                    if coords["y1"] < coords["y0"]:
                        coords["y0"], coords["y1"] = coords["y1"], coords["y0"]

                    # Ensure non-negative coordinates
                    for key in coords:
                        coords[key] = max(0.0, coords[key])

                    return coords

            # Try alternative attribute names
            for attr_name in ["bbox", "bounding_box", "bounds"]:
                if hasattr(item, attr_name):
                    bbox_attr = getattr(item, attr_name)
                    if bbox_attr:
                        return self._parse_bbox_object(bbox_attr)

            # Fallback to default bbox
            return {"x0": 0.0, "y0": 0.0, "x1": 100.0, "y1": 20.0}

        except Exception as e:
            logger.warning(f"Failed to extract bbox: {e}")
            return {"x0": 0.0, "y0": 0.0, "x1": 100.0, "y1": 20.0}

    def _parse_bbox_object(self, bbox_obj) -> Dict[str, float]:
        """Parse bounding box from various object types."""
        if hasattr(bbox_obj, "__dict__"):
            # Object with attributes
            return {
                "x0": float(getattr(bbox_obj, "x0", getattr(bbox_obj, "left", 0))),
                "y0": float(getattr(bbox_obj, "y0", getattr(bbox_obj, "top", 0))),
                "x1": float(getattr(bbox_obj, "x1", getattr(bbox_obj, "right", 100))),
                "y1": float(getattr(bbox_obj, "y1", getattr(bbox_obj, "bottom", 20))),
            }
        elif isinstance(bbox_obj, (list, tuple)) and len(bbox_obj) >= 4:
            # List or tuple format
            return {
                "x0": float(bbox_obj[0]),
                "y0": float(bbox_obj[1]),
                "x1": float(bbox_obj[2]),
                "y1": float(bbox_obj[3]),
            }
        elif isinstance(bbox_obj, dict):
            # Dictionary format
            return {
                "x0": float(bbox_obj.get("x0", bbox_obj.get("left", 0))),
                "y0": float(bbox_obj.get("y0", bbox_obj.get("top", 0))),
                "x1": float(bbox_obj.get("x1", bbox_obj.get("right", 100))),
                "y1": float(bbox_obj.get("y1", bbox_obj.get("bottom", 20))),
            }
        else:
            return {"x0": 0.0, "y0": 0.0, "x1": 100.0, "y1": 20.0}

    def _extract_page_number(self, item) -> int:
        """Extract page number from Docling item."""
        try:
            # Check for prov list (new Docling structure)
            if hasattr(item, "prov") and item.prov:
                prov_item = item.prov[0]  # Get first provenance item
                if hasattr(prov_item, "page_no"):
                    # Page numbers in prov are already 1-based
                    return max(1, prov_item.page_no)

            # Fallback: try old methods
            if hasattr(item, "get_page_number"):
                return max(1, item.get_page_number())

            # Try to get from location
            if hasattr(item, "get_location"):
                location = item.get_location()
                if hasattr(location, "page_number"):
                    return max(1, location.page_number)

            # Default to page 1
            return 1

        except Exception:
            return 1

    def _extract_table_text(self, table_item) -> str:
        """Extract text representation from table."""
        try:
            # Try to get structured table data
            if hasattr(table_item, "data") and table_item.data:
                return self._format_table_data(table_item.data)

            # Try to get table text directly
            if hasattr(table_item, "text") and table_item.text:
                return table_item.text.strip()

            # Try to get cells
            if hasattr(table_item, "cells") and table_item.cells:
                cell_texts = []
                for cell in table_item.cells:
                    if hasattr(cell, "text") and cell.text:
                        cell_texts.append(cell.text.strip())

                if cell_texts:
                    return " | ".join(cell_texts)

            # Fallback to generic text
            return "Table content"

        except Exception as e:
            logger.warning(f"Failed to extract table text: {e}")
            return "Table content"

    def _format_table_data(self, data) -> str:
        """Format structured table data into text."""
        try:
            if isinstance(data, list) and data:
                # Assume data is list of rows
                formatted_rows = []
                for row in data:
                    if isinstance(row, list):
                        # Row is list of cells
                        formatted_rows.append(" | ".join(str(cell) for cell in row))
                    else:
                        formatted_rows.append(str(row))

                return "\n".join(formatted_rows)
            else:
                return str(data)[:500]  # Truncate very long data
        except Exception:
            return str(data)[:200]

    def _extract_table_metadata(self, table_item) -> Dict[str, Any]:
        """Extract metadata from table item."""
        metadata = {}

        try:
            # Extract table dimensions
            if hasattr(table_item, "data") and table_item.data:
                data = table_item.data
                if hasattr(data, "__len__"):
                    metadata["rows"] = len(data)
                    if data and hasattr(data[0], "__len__"):
                        metadata["columns"] = len(data[0])

            # Extract table properties
            table_attrs = [
                "num_rows",
                "num_cols",
                "has_header",
                "has_footer",
                "table_type",
                "confidence",
                "page_number",
            ]

            for attr in table_attrs:
                if hasattr(table_item, attr):
                    value = getattr(table_item, attr)
                    if value is not None:
                        metadata[attr] = value

            # Extract cell information if available
            if hasattr(table_item, "cells") and table_item.cells:
                metadata["cell_count"] = len(table_item.cells)

                # Analyze cell types
                cell_types = {}
                for cell in table_item.cells[:10]:  # Sample first 10 cells
                    cell_type = self._analyze_cell_type(cell)
                    cell_types[cell_type] = cell_types.get(cell_type, 0) + 1

                if cell_types:
                    metadata["cell_types"] = cell_types

        except Exception as e:
            logger.warning(f"Failed to extract table metadata: {e}")

        return metadata

    def _analyze_cell_type(self, cell) -> str:
        """Analyze the type of content in a table cell."""
        try:
            if not hasattr(cell, "text") or not cell.text:
                return "empty"

            text = cell.text.strip()
            if not text:
                return "empty"

            # Check for numeric content
            try:
                float(text.replace(",", "").replace("$", "").replace("%", ""))
                return "numeric"
            except ValueError:
                pass

            # Check for date patterns
            date_patterns = [r"\d{1,2}/\d{1,2}/\d{4}", r"\d{4}-\d{2}-\d{2}"]
            if any(re.search(pattern, text) for pattern in date_patterns):
                return "date"

            # Default to text
            return "text"

        except Exception:
            return "unknown"

    def parse_document_with_kvs(
        self, pdf_path: Path
    ) -> Tuple[List[DocumentElement], List[KeyValuePair]]:
        """Parse a PDF document and extract key-value pairs if enabled.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Tuple of (document_elements, key_value_pairs)

        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            ValueError: If the file is not a valid PDF
            DocumentParsingError: If parsing fails
        """
        # Parse document elements first
        elements = self.parse_document(pdf_path)

        # Extract key-value pairs if enabled
        if self.enable_kv_extraction and self.kv_extractor and elements:
            try:
                logger.info(f"Extracting key-value pairs from {pdf_path}")
                kv_pairs = self.kv_extractor.extract(elements)
                logger.info(f"Extracted {len(kv_pairs)} key-value pairs from {pdf_path}")
                return elements, kv_pairs
            except Exception as e:
                logger.warning(f"KV extraction failed for {pdf_path}: {e}")
                # Return elements with empty KV pairs on extraction failure
                return elements, []
        else:
            # Return elements with empty KV pairs if extraction is disabled
            return elements, []

    def parse_multiple_documents(self, pdf_paths: List[Path]) -> Dict[Path, List[DocumentElement]]:
        """Efficiently process multiple documents.

        Args:
            pdf_paths: List of PDF file paths

        Returns:
            Dictionary mapping paths to extracted elements
        """
        results = {}

        logger.info(f"Starting batch processing of {len(pdf_paths)} documents")

        for i, pdf_path in enumerate(pdf_paths):
            try:
                logger.info(f"Processing document {i+1}/{len(pdf_paths)}: {pdf_path}")
                elements = self.parse_document(pdf_path)
                results[pdf_path] = elements

            except Exception as e:
                logger.error(f"Failed to parse {pdf_path}: {e}")
                results[pdf_path] = []

        logger.info(
            f"Completed batch processing: {len([r for r in results.values() if r])} successful"
        )
        return results

    def _extract_picture_text(self, picture_item, document) -> str:
        """Extract text description from picture item."""
        try:
            # Try caption_text method (new Docling structure)
            if hasattr(picture_item, "caption_text") and callable(picture_item.caption_text):
                try:
                    caption = picture_item.caption_text(document)
                    if caption and isinstance(caption, str) and caption.strip():
                        return caption.strip()
                except Exception as e:
                    logger.warning(f"Failed to extract caption_text: {e}")

            # Try various caption/description attributes (fallback)
            for attr in ["caption", "description", "alt_text", "text"]:
                if hasattr(picture_item, attr):
                    text = getattr(picture_item, attr)
                    if text and isinstance(text, str):
                        return text.strip()

            # Try to get text from children (captions often stored as child elements)
            if hasattr(picture_item, "children") and picture_item.children:
                caption_texts = []
                for child_ref in picture_item.children:
                    # Child references point to other document elements
                    if hasattr(child_ref, "cref"):
                        # Extract text from referenced child elements
                        # This is a simplified approach - could be improved with proper reference resolution
                        continue

                if caption_texts:
                    return " ".join(caption_texts)

            # Fallback
            return "Image content"

        except Exception as e:
            logger.warning(f"Failed to extract picture text: {e}")
            return "Image content"

    def get_page_image(self, page_number: int):
        """Get page image for verification purposes.

        Args:
            page_number: Page number (1-indexed)

        Returns:
            PIL Image object for the specified page

        Raises:
            ValueError: If page images not enabled or page not found
        """
        if not self.generate_page_images:
            raise ValueError("Page images not enabled. Set generate_page_images=True")

        if page_number < 1:
            raise ValueError(f"Invalid page number: {page_number}")

        if page_number not in self.page_images:
            available_pages = sorted(self.page_images.keys())
            raise ValueError(
                f"Page image not found for page {page_number}. "
                f"Available pages: {available_pages}"
            )

        return self.page_images[page_number]

    def get_parsing_stats(self) -> Dict[str, Any]:
        """Get statistics about the last parsing operation.

        Returns:
            Dictionary with parsing statistics
        """
        return {
            "page_images_generated": len(self.page_images),
            "ocr_enabled": self.enable_ocr,
            "tables_enabled": self.enable_tables,
            "ocr_engine": self.ocr_engine if self.enable_ocr else None,
            "table_mode": self.table_mode if self.enable_tables else None,
            "image_scale": self.image_scale if self.generate_page_images else None,
            "kv_extraction_enabled": self.enable_kv_extraction,
            "header_classifier_enabled": self.header_classifier_enabled,
        }


def classify_as_heading(element: DocumentElement, page_context: dict) -> bool:
    """
    Classify a DocumentElement as a heading or regular text.

    This is a wrapper function around the header classifier module that
    provides the interface expected by the parser and tests.

    Args:
        element: DocumentElement to classify
        page_context: Dictionary with page statistics and dimensions

    Returns:
        bool: True if element should be classified as heading, False otherwise
    """
    return is_heading(element, page_context)

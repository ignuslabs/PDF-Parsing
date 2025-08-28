"""
Core PDF parser implementation using Docling.
"""

import os
import logging
import re
from pathlib import Path
from typing import List, Optional, Any, Dict, Union
from dataclasses import dataclass
from datetime import datetime
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
        PdfPipelineOptions, 
        TesseractOcrOptions, 
        EasyOcrOptions,
        TableStructureOptions,
        TableFormerMode
    )
from docling.datamodel.base_models import InputFormat

from src.core.models import DocumentElement, ParsedDocument
from src.utils.exceptions import DocumentParsingError, OCRError

# Setup logging
logger = logging.getLogger(__name__)


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
        image_scale: float = 1.0

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


            ADDD TableStructureOptions 
        """
        self.enable_ocr = enable_ocr
        self.enable_tables = enable_tables
        self.generate_page_images = generate_page_images
        self.max_pages = max_pages
        self.ocr_engine = ocr_engine
        self.ocr_lang = ocr_lang
        self.table_mode = table_mode
        self.image_scale = image_scale
        
        # Initialize converter (will be created lazily)
        self._converter = None
        self.page_images = {}
        
        # if not DOCLING_AVAILABLE:
        #     raise ImportError(
        #         "Docling is not available. Please install it with: "
        #         "pip install docling"
        #     )
        
        # Validate configuration
        self._validate_config()
    
    @property
    def converter(self):
        """Lazy initialization of the document converter."""
        if self._converter is None:
            self._converter = self._create_converter()
        return self._converter
    
    def _validate_config(self):
        """Validate parser configuration."""
        if self.ocr_engine not in ['tesseract', 'easyocr']:
            raise ValueError(f"Unsupported OCR engine: {self.ocr_engine}")
        
        if self.table_mode not in ['accurate', 'fast']:
            raise ValueError(f"Invalid table mode: {self.table_mode}")
        
        if not 0.1 <= self.image_scale <= 5.0:
            raise ValueError(f"Image scale must be between 0.1 and 5.0, got: {self.image_scale}")
    
    def _create_converter(self) -> DocumentConverter:
        """Create and configure the Docling document converter."""
        # Configure pipeline options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = self.enable_ocr
        pipeline_options.do_table_structure = self.enable_tables
        pipeline_options.generate_page_images = self.generate_page_images
        
        # Configure OCR options
        if self.enable_ocr:
            pipeline_options.ocr_options = self._create_ocr_options()
        
        # Configure table options
        if self.enable_tables and hasattr(pipeline_options, 'table_structure_options'):
            pipeline_options.table_structure_options.mode = self.table_mode
        
        # Configure image options
        if self.generate_page_images:
            pipeline_options.images_scale = self.image_scale
        
        # Create converter with PDF format options
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        return converter
    
    def _create_ocr_options(self):
        """Create OCR options based on configuration."""
        if self.ocr_engine == "tesseract":
            # TesseractOcrOptions expects a list for lang parameter
            if isinstance(self.ocr_lang, list):
                lang_list = self.ocr_lang
            else:
                lang_list = [self.ocr_lang]
            
            return TesseractOcrOptions(
                lang=lang_list,
                force_full_page_ocr=True
            )
        elif self.ocr_engine == "easyocr":
            lang_list = self.ocr_lang if isinstance(self.ocr_lang, list) else [self.ocr_lang]
            return EasyOcrOptions(
                lang=lang_list
            )
        else:
            raise ValueError(f"Unsupported OCR engine: {self.ocr_engine}")
    
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
        pdf_path = Path(pdf_path)
        
        # Input validation
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if pdf_path.suffix.lower() != '.pdf':
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
            
            # Convert document using Docling
            result = self._convert_document(pdf_path)
            
            # Extract elements from the converted document
            elements = self._extract_elements(result.document)
            
            # Store page images if requested
            if self.generate_page_images and hasattr(result, 'pages'):
                # Convert result.pages list to dict mapping page numbers to images
                self.page_images = {}
                for page in result.pages:
                    if hasattr(page, 'image') and page.image and hasattr(page, 'page_no'):
                        # Convert from 0-based to 1-based page numbering
                        page_num = page.page_no + 1
                        self.page_images[page_num] = page.image
            
            logger.info(f"Successfully parsed PDF: {pdf_path} - {len(elements)} elements extracted")
            
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
        
        # Input validation
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if pdf_path.suffix.lower() != '.pdf':
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
            
            # Convert document using Docling
            result = self._convert_document(pdf_path)
            
            # Extract elements from the converted document
            elements = self._extract_elements(result.document)
            
            # Store page images if requested
            if self.generate_page_images and hasattr(result, 'pages'):
                # Convert result.pages list to dict mapping page numbers to images
                self.page_images = {}
                for page in result.pages:
                    if hasattr(page, 'image') and page.image and hasattr(page, 'page_no'):
                        # Convert from 0-based to 1-based page numbering
                        page_num = page.page_no + 1
                        self.page_images[page_num] = page.image
            
            logger.info(f"Successfully parsed PDF: {pdf_path} - {len(elements)} elements extracted")
            
            # Create ParsedDocument
            document = ParsedDocument(
                elements=elements,
                metadata={
                    'source_path': str(pdf_path),
                    'filename': pdf_path.name,
                    'file_size': file_size,
                    'parsed_at': datetime.now().isoformat(),
                    'parser_config': {
                        'ocr_enabled': self.enable_ocr,
                        'tables_enabled': self.enable_tables,
                        'images_enabled': self.generate_page_images,
                        'ocr_engine': self.ocr_engine if self.enable_ocr else None,
                        'table_mode': self.table_mode if self.enable_tables else None
                    },
                    'total_elements': len(elements),
                    'page_count': len(set(e.page_number for e in elements)) if elements else 0
                },
                pages=self.page_images if self.page_images else None
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
                return ParsedDocument(
                    elements=elements,
                    metadata={
                        'source_path': str(pdf_path),
                        'filename': pdf_path.name,
                        'file_size': file_size,
                        'parsed_at': datetime.now().isoformat(),
                        'fallback_mode': 'no_ocr',
                        'parser_config': {
                            'ocr_enabled': False,
                            'tables_enabled': self.enable_tables,
                            'images_enabled': self.generate_page_images
                        },
                        'total_elements': len(elements),
                        'page_count': len(set(e.page_number for e in elements)) if elements else 0
                    },
                    pages=None
                )
            else:
                raise DocumentParsingError(f"OCR processing failed: {e}", str(pdf_path))
        except MemoryError as e:
            logger.error(f"Insufficient memory for {pdf_path}: {e}")
            # Try with reduced settings
            elements = self._parse_with_reduced_settings(pdf_path)
            return ParsedDocument(
                elements=elements,
                metadata={
                    'source_path': str(pdf_path),
                    'filename': pdf_path.name,
                    'file_size': file_size,
                    'parsed_at': datetime.now().isoformat(),
                    'fallback_mode': 'reduced_memory',
                    'parser_config': {
                        'ocr_enabled': self.enable_ocr,
                        'tables_enabled': False,  # Disabled for memory
                        'images_enabled': False   # Disabled for memory
                    },
                    'total_elements': len(elements),
                    'page_count': len(set(e.page_number for e in elements)) if elements else 0
                },
                pages=None
            )
        except Exception as e:
            logger.error(f"Unexpected error parsing {pdf_path}: {e}")
            raise DocumentParsingError(f"Failed to parse PDF: {str(e)}", str(pdf_path)) from e
    
    def _convert_document(self, pdf_path: Path):
        """Convert PDF using Docling converter."""
        try:
            return self.converter.convert(str(pdf_path))
        except Exception as e:
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
            elements = self._extract_elements(result.document)
            
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
            elements = self._extract_elements(result.document)
            
            logger.warning(f"Parsed {pdf_path} with reduced settings due to memory constraints")
            
            return elements
            
        finally:
            # Restore original settings
            self.generate_page_images = original_images
            self.enable_tables = original_tables
            self._converter = None  # Force recreation with original settings
    
    def _extract_elements(self, document) -> List[DocumentElement]:
        """Extract DocumentElement objects from Docling document.
        
        Args:
            document: Docling Document object
            
        Returns:
            List of DocumentElement objects
        """
        elements = []
        element_id = 0
        
        # Extract text elements
        if hasattr(document, 'texts'):
            for text_item in document.texts:
                element = self._create_element_from_text(text_item, element_id, document)
                if element:
                    elements.append(element)
                    element_id += 1
        
        # Extract table elements
        if hasattr(document, 'tables') and self.enable_tables:
            for table_item in document.tables:
                element = self._create_element_from_table(table_item, element_id, document)
                if element:
                    elements.append(element)
                    element_id += 1
        
        # Extract picture elements
        if hasattr(document, 'pictures'):
            for picture_item in document.pictures:
                element = self._create_element_from_picture(picture_item, element_id, document)
                if element:
                    elements.append(element)
                    element_id += 1
        
        # Sort elements by page number and position
        elements.sort(key=lambda x: (x.page_number, x.bbox['y0'], x.bbox['x0']))
        
        return elements
    
    def _create_element_from_text(self, text_item, element_id: int, document) -> Optional[DocumentElement]:
        """Create DocumentElement from Docling text item."""
        try:
            # Get text content
            text_content = getattr(text_item, 'text', '')
            if not text_content or not text_content.strip():
                return None
            
            # Determine element type (simplified heuristic)
            element_type = self._determine_text_element_type(text_content)
            
            # Get bounding box
            bbox = self._extract_bbox(text_item)
            
            # Get page number
            page_number = self._extract_page_number(text_item)
            
            # Get confidence (default to high for text)
            confidence = getattr(text_item, 'confidence', 0.9)
            
            return DocumentElement(
                text=text_content.strip(),
                element_type=element_type,
                page_number=page_number,
                bbox=bbox,
                confidence=confidence,
                metadata={'element_id': element_id, 'source': 'docling_text'}
            )
            
        except Exception as e:
            # Skip problematic elements rather than failing entire parsing
            print(f"Warning: Failed to process text element {element_id}: {e}")
            return None
    
    def _create_element_from_table(self, table_item, element_id: int, document) -> Optional[DocumentElement]:
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
            table_metadata.update({
                'element_id': element_id,
                'source': 'docling_table'
            })
            
            return DocumentElement(
                text=table_text,
                element_type='table',
                page_number=page_number,
                bbox=bbox,
                confidence=0.8,  # Tables might be less confident
                metadata=table_metadata
            )
            
        except Exception as e:
            print(f"Warning: Failed to process table element {element_id}: {e}")
            return None
    
    def _create_element_from_picture(self, picture_item, element_id: int, document) -> Optional[DocumentElement]:
        """Create DocumentElement from Docling picture item."""
        try:
            # Get picture caption or description
            picture_text = self._extract_picture_text(picture_item, document)
            
            # Get bounding box
            bbox = self._extract_bbox(picture_item)
            
            # Get page number
            page_number = self._extract_page_number(picture_item)
            
            picture_metadata = {
                'element_id': element_id,
                'source': 'docling_picture'
            }
            
            return DocumentElement(
                text=picture_text,
                element_type='image',
                page_number=page_number,
                bbox=bbox,
                confidence=0.9,
                metadata=picture_metadata
            )
            
        except Exception as e:
            print(f"Warning: Failed to process picture element {element_id}: {e}")
            return None
    
    def _determine_text_element_type(self, text: str) -> str:
        """Determine element type based on text characteristics."""
        text_stripped = text.strip()
        
        # Short text in all caps might be a heading
        if len(text_stripped) < 100 and text_stripped.isupper():
            return 'heading'
        
        # Text with certain patterns might be headings
        if (len(text_stripped) < 200 and 
            (text_stripped.endswith(':') or 
             any(word in text_stripped.lower() for word in ['chapter', 'section', 'introduction', 'conclusion']))):
            return 'heading'
        
        # Mathematical expressions
        if any(symbol in text_stripped for symbol in ['=', '∑', '∫', '√', '±', '∞', 'α', 'β', 'γ']):
            return 'formula'
        
        # List detection
        if (text_stripped.startswith(('•', '◦', '▪', '-', '*')) or 
            re.match(r'^\d+\.', text_stripped) or 
            re.match(r'^[a-zA-Z]\)', text_stripped)):
            return 'list'
        
        # Code detection
        code_indicators = ['def ', 'class ', 'import ', 'function', '{', '}', ';']
        if any(indicator in text_stripped for indicator in code_indicators):
            return 'code'
        
        # Caption detection (often starts with "Figure" or "Table")
        if re.match(r'^(Figure|Table|Image|Chart)\s+\d+', text_stripped, re.IGNORECASE):
            return 'caption'
        
        # Default to text
        return 'text'
    
    def _extract_bbox(self, item) -> Dict[str, float]:
        """Extract and validate bounding box from Docling item."""
        try:
            # Check for prov list (new Docling structure)
            if hasattr(item, 'prov') and item.prov:
                prov_item = item.prov[0]  # Get first provenance item
                if hasattr(prov_item, 'bbox'):
                    bbox = prov_item.bbox
                    
                    # Convert from Docling's l,t,r,b format to x0,y0,x1,y1
                    coords = {
                        'x0': float(getattr(bbox, 'l', 0)),
                        'y0': float(getattr(bbox, 't', 0)),
                        'x1': float(getattr(bbox, 'r', 100)),
                        'y1': float(getattr(bbox, 'b', 20))
                    }
                    
                    # Validate coordinate consistency
                    if coords['x1'] < coords['x0']:
                        coords['x0'], coords['x1'] = coords['x1'], coords['x0']
                    if coords['y1'] < coords['y0']:
                        coords['y0'], coords['y1'] = coords['y1'], coords['y0']
                    
                    # Ensure non-negative coordinates
                    for key in coords:
                        coords[key] = max(0.0, coords[key])
                    
                    return coords
            
            # Fallback: try old get_location method
            if hasattr(item, 'get_location'):
                location = item.get_location()
                if hasattr(location, 'bbox'):
                    bbox = location.bbox
                    
                    # Extract coordinates with type validation
                    coords = {
                        'x0': float(getattr(bbox, 'x0', 0)),
                        'y0': float(getattr(bbox, 'y0', 0)),
                        'x1': float(getattr(bbox, 'x1', 100)),
                        'y1': float(getattr(bbox, 'y1', 20))
                    }
                    
                    # Validate coordinate consistency
                    if coords['x1'] < coords['x0']:
                        coords['x0'], coords['x1'] = coords['x1'], coords['x0']
                    if coords['y1'] < coords['y0']:
                        coords['y0'], coords['y1'] = coords['y1'], coords['y0']
                    
                    # Ensure non-negative coordinates
                    for key in coords:
                        coords[key] = max(0.0, coords[key])
                    
                    return coords
            
            # Try alternative attribute names
            for attr_name in ['bbox', 'bounding_box', 'bounds']:
                if hasattr(item, attr_name):
                    bbox_attr = getattr(item, attr_name)
                    if bbox_attr:
                        return self._parse_bbox_object(bbox_attr)
            
            # Fallback to default bbox
            return {'x0': 0.0, 'y0': 0.0, 'x1': 100.0, 'y1': 20.0}
            
        except Exception as e:
            logger.warning(f"Failed to extract bbox: {e}")
            return {'x0': 0.0, 'y0': 0.0, 'x1': 100.0, 'y1': 20.0}
    
    def _parse_bbox_object(self, bbox_obj) -> Dict[str, float]:
        """Parse bounding box from various object types."""
        if hasattr(bbox_obj, '__dict__'):
            # Object with attributes
            return {
                'x0': float(getattr(bbox_obj, 'x0', getattr(bbox_obj, 'left', 0))),
                'y0': float(getattr(bbox_obj, 'y0', getattr(bbox_obj, 'top', 0))),
                'x1': float(getattr(bbox_obj, 'x1', getattr(bbox_obj, 'right', 100))),
                'y1': float(getattr(bbox_obj, 'y1', getattr(bbox_obj, 'bottom', 20)))
            }
        elif isinstance(bbox_obj, (list, tuple)) and len(bbox_obj) >= 4:
            # List or tuple format
            return {
                'x0': float(bbox_obj[0]),
                'y0': float(bbox_obj[1]),
                'x1': float(bbox_obj[2]),
                'y1': float(bbox_obj[3])
            }
        elif isinstance(bbox_obj, dict):
            # Dictionary format
            return {
                'x0': float(bbox_obj.get('x0', bbox_obj.get('left', 0))),
                'y0': float(bbox_obj.get('y0', bbox_obj.get('top', 0))),
                'x1': float(bbox_obj.get('x1', bbox_obj.get('right', 100))),
                'y1': float(bbox_obj.get('y1', bbox_obj.get('bottom', 20)))
            }
        else:
            return {'x0': 0.0, 'y0': 0.0, 'x1': 100.0, 'y1': 20.0}
    
    def _extract_page_number(self, item) -> int:
        """Extract page number from Docling item."""
        try:
            # Check for prov list (new Docling structure)
            if hasattr(item, 'prov') and item.prov:
                prov_item = item.prov[0]  # Get first provenance item
                if hasattr(prov_item, 'page_no'):
                    # Page numbers in prov are already 1-based
                    return max(1, prov_item.page_no)
            
            # Fallback: try old methods
            if hasattr(item, 'get_page_number'):
                return max(1, item.get_page_number())
            
            # Try to get from location
            if hasattr(item, 'get_location'):
                location = item.get_location()
                if hasattr(location, 'page_number'):
                    return max(1, location.page_number)
            
            # Default to page 1
            return 1
            
        except Exception:
            return 1
    
    def _extract_table_text(self, table_item) -> str:
        """Extract text representation from table."""
        try:
            # Try to get structured table data
            if hasattr(table_item, 'data') and table_item.data:
                return self._format_table_data(table_item.data)
            
            # Try to get table text directly
            if hasattr(table_item, 'text') and table_item.text:
                return table_item.text.strip()
            
            # Try to get cells
            if hasattr(table_item, 'cells') and table_item.cells:
                cell_texts = []
                for cell in table_item.cells:
                    if hasattr(cell, 'text') and cell.text:
                        cell_texts.append(cell.text.strip())
                
                if cell_texts:
                    return ' | '.join(cell_texts)
            
            # Fallback to generic text
            return 'Table content'
            
        except Exception as e:
            logger.warning(f"Failed to extract table text: {e}")
            return 'Table content'
    
    def _format_table_data(self, data) -> str:
        """Format structured table data into text."""
        try:
            if isinstance(data, list) and data:
                # Assume data is list of rows
                formatted_rows = []
                for row in data:
                    if isinstance(row, list):
                        # Row is list of cells
                        formatted_rows.append(' | '.join(str(cell) for cell in row))
                    else:
                        formatted_rows.append(str(row))
                
                return '\n'.join(formatted_rows)
            else:
                return str(data)[:500]  # Truncate very long data
        except Exception:
            return str(data)[:200]
    
    def _extract_table_metadata(self, table_item) -> Dict[str, Any]:
        """Extract metadata from table item."""
        metadata = {}
        
        try:
            # Extract table dimensions
            if hasattr(table_item, 'data') and table_item.data:
                data = table_item.data
                if hasattr(data, '__len__'):
                    metadata['rows'] = len(data)
                    if data and hasattr(data[0], '__len__'):
                        metadata['columns'] = len(data[0])
            
            # Extract table properties
            table_attrs = [
                'num_rows', 'num_cols', 'has_header', 'has_footer',
                'table_type', 'confidence', 'page_number'
            ]
            
            for attr in table_attrs:
                if hasattr(table_item, attr):
                    value = getattr(table_item, attr)
                    if value is not None:
                        metadata[attr] = value
            
            # Extract cell information if available
            if hasattr(table_item, 'cells') and table_item.cells:
                metadata['cell_count'] = len(table_item.cells)
                
                # Analyze cell types
                cell_types = {}
                for cell in table_item.cells[:10]:  # Sample first 10 cells
                    cell_type = self._analyze_cell_type(cell)
                    cell_types[cell_type] = cell_types.get(cell_type, 0) + 1
                
                if cell_types:
                    metadata['cell_types'] = cell_types
                    
        except Exception as e:
            logger.warning(f"Failed to extract table metadata: {e}")
        
        return metadata
    
    def _analyze_cell_type(self, cell) -> str:
        """Analyze the type of content in a table cell."""
        try:
            if not hasattr(cell, 'text') or not cell.text:
                return 'empty'
            
            text = cell.text.strip()
            if not text:
                return 'empty'
            
            # Check for numeric content
            try:
                float(text.replace(',', '').replace('$', '').replace('%', ''))
                return 'numeric'
            except ValueError:
                pass
            
            # Check for date patterns
            date_patterns = [r'\d{1,2}/\d{1,2}/\d{4}', r'\d{4}-\d{2}-\d{2}']
            if any(re.search(pattern, text) for pattern in date_patterns):
                return 'date'
            
            # Default to text
            return 'text'
            
        except Exception:
            return 'unknown'
    
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
        
        logger.info(f"Completed batch processing: {len([r for r in results.values() if r])} successful")
        return results
    
    def _extract_picture_text(self, picture_item, document) -> str:
        """Extract text description from picture item."""
        try:
            # Try caption_text method (new Docling structure)
            if hasattr(picture_item, 'caption_text') and callable(picture_item.caption_text):
                try:
                    caption = picture_item.caption_text(document)
                    if caption and caption.strip():
                        return caption.strip()
                except Exception as e:
                    logger.warning(f"Failed to extract caption_text: {e}")
            
            # Try various caption/description attributes (fallback)
            for attr in ['caption', 'description', 'alt_text', 'text']:
                if hasattr(picture_item, attr):
                    text = getattr(picture_item, attr)
                    if text and isinstance(text, str):
                        return text.strip()
            
            # Try to get text from children (captions often stored as child elements)
            if hasattr(picture_item, 'children') and picture_item.children:
                caption_texts = []
                for child_ref in picture_item.children:
                    # Child references point to other document elements
                    if hasattr(child_ref, 'cref'):
                        # Extract text from referenced child elements
                        # This is a simplified approach - could be improved with proper reference resolution
                        continue
                
                if caption_texts:
                    return ' '.join(caption_texts)
            
            # Fallback
            return 'Image content'
            
        except Exception as e:
            logger.warning(f"Failed to extract picture text: {e}")
            return 'Image content'
    
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
            'page_images_generated': len(self.page_images),
            'ocr_enabled': self.enable_ocr,
            'tables_enabled': self.enable_tables,
            'ocr_engine': self.ocr_engine if self.enable_ocr else None,
            'table_mode': self.table_mode if self.enable_tables else None,
            'image_scale': self.image_scale if self.generate_page_images else None
        }
#!/usr/bin/env python3
"""
Comprehensive PDF parsing demonstration script.
Tests tesseract, pdfplumber, PyPDF2, and docling with all export formats.

IMPORTANT NOTES:
- PyPDF2: Only works with text-based PDFs (not scanned/image PDFs)
- pdfplumber: Works best with machine-generated PDFs (limited OCR support)
- Tesseract: Requires pdf2image and proper OCR setup
- Docling: Has built-in OCR support when configured properly
"""

import os
import sys
import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import traceback
from datetime import datetime

# PDF Processing Libraries
try:
    import tesserocr
    from PIL import Image
    TESSEROCR_AVAILABLE = True
except ImportError:
    TESSEROCR_AVAILABLE = False
    print("Warning: tesserocr not available")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("Warning: pdfplumber not available")

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    print("Warning: PyPDF2 not available")

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TesseractOcrOptions,
        TableFormerMode,
    )
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    print("Warning: docling not available")

# Docling compatibility helpers
try:
    from src.utils.docling_compat import (
        build_converter as dl_build_converter,
        create_ocr_options as dl_create_ocr_options,
        normalize_table_mode as dl_normalize_table_mode,
        export_markdown as dl_export_markdown,
        export_text as dl_export_text,
        export_dict as dl_export_dict,
        get_document_collections as dl_get_document_collections,
        table_to_dataframe as dl_table_to_dataframe,
    )
except Exception:
    # Keep script working even if import path differs when run standalone
    dl_build_converter = dl_create_ocr_options = dl_normalize_table_mode = None
    dl_export_markdown = dl_export_text = dl_export_dict = None
    dl_get_document_collections = dl_table_to_dataframe = None

# For PDF to image conversion (needed for tesseract)
try:
    import pdf2image
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    print("Warning: pdf2image not available (needed for tesseract OCR)")


def ensure_directory(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def detect_pdf_type(pdf_path: Path) -> Tuple[str, float]:
    """
    Detect if a PDF is text-based or scanned/image-based.
    Returns: (type, confidence) where type is 'text', 'scanned', or 'mixed'
    """
    try:
        if PYPDF2_AVAILABLE:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                total_pages = len(reader.pages)
                pages_with_text = 0
                total_text_length = 0

                for page in reader.pages:
                    text = page.extract_text()
                    if not text:
                        continue
                    text = text.strip()
                    # Remove common formatting characters
                    clean_text = ''.join(c for c in text if c.isalnum() or c.isspace())
                    if len(clean_text) > 50:  # More than 50 alphanumeric chars
                        pages_with_text += 1
                        total_text_length += len(clean_text)

                text_ratio = pages_with_text / total_pages if total_pages > 0 else 0
                avg_text_per_page = total_text_length / total_pages if total_pages > 0 else 0

                if text_ratio > 0.8 and avg_text_per_page > 100:
                    return ("text", text_ratio)
                elif text_ratio < 0.2 or avg_text_per_page < 50:
                    return ("scanned", 1 - text_ratio)
                else:
                    return ("mixed", 0.5)
    except Exception as e:
        print(f"Could not detect PDF type: {e}")

    return ("unknown", 0.0)


def process_with_tesseract(pdf_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Process PDF with tesseract OCR."""
    results = {"success": False, "exports": [], "error": None}

    if not TESSEROCR_AVAILABLE:
        results["error"] = "tesserocr not installed"
        return results

    if not PDF2IMAGE_AVAILABLE:
        results["error"] = "pdf2image not installed (needed for OCR)"
        return results

    try:
        print("\n=== Processing with Tesseract ===")
        tesseract_dir = output_dir / "tesseract"
        ensure_directory(tesseract_dir)

        # Convert PDF to images
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path)

        all_text = []
        all_hocr = []
        all_tsv = []

        for i, img in enumerate(images):
            print(f"Processing page {i+1}/{len(images)}...")

            with tesserocr.PyTessBaseAPI() as api:
                api.SetImage(img)

                # Get plain text
                text = api.GetUTF8Text()
                all_text.append(f"--- Page {i+1} ---\n{text}")

                # Get hOCR (HTML with positional data)
                hocr = api.GetHOCRText(0)
                all_hocr.append(hocr)

                # Get TSV output with bounding boxes
                tsv = api.GetTSVText(0)
                all_tsv.append(tsv)

                # Get confidence
                confidence = api.MeanTextConf()
                print(f"  Page {i+1} confidence: {confidence:.2f}%")

        # Save text output
        text_path = tesseract_dir / "text.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(all_text))
        results["exports"].append(str(text_path))
        print(f"  ✓ Saved text to {text_path}")

        # Save hOCR output
        hocr_path = tesseract_dir / "hocr.html"
        with open(hocr_path, 'w', encoding='utf-8') as f:
            f.write("<html><body>")
            f.write("\n".join(all_hocr))
            f.write("</body></html>")
        results["exports"].append(str(hocr_path))
        print(f"  ✓ Saved hOCR to {hocr_path}")

        # Save TSV output
        tsv_path = tesseract_dir / "tsv_data.tsv"
        with open(tsv_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(all_tsv))
        results["exports"].append(str(tsv_path))
        print(f"  ✓ Saved TSV data to {tsv_path}")

        results["success"] = True

    except Exception as e:
        results["error"] = str(e)
        print(f"  ✗ Error: {e}")
        traceback.print_exc()

    return results


def process_with_pdfplumber(pdf_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Process PDF with pdfplumber (works best with machine-generated PDFs)."""
    results = {"success": False, "exports": [], "error": None, "warnings": []}

    if not PDFPLUMBER_AVAILABLE:
        results["error"] = "pdfplumber not installed"
        return results

    try:
        print("\n=== Processing with pdfplumber ===")

        # Detect PDF type
        pdf_type, confidence = detect_pdf_type(pdf_path)
        if pdf_type == "scanned":
            warning = "⚠️  WARNING: This appears to be a scanned PDF. pdfplumber works best with machine-generated PDFs."
            print(warning)
            print("   For scanned PDFs, consider using Docling with OCR or converting with OCRmyPDF first.")
            results["warnings"].append(warning)
        elif pdf_type == "mixed":
            print("   Note: This PDF contains mixed content (text and scanned images).")

        plumber_dir = output_dir / "pdfplumber"
        ensure_directory(plumber_dir)

        # Open with different settings for better extraction
        with pdfplumber.open(pdf_path, laparams={
            "line_overlap": 0.5,
            "char_margin": 2.0,
            "word_margin": 0.1,
            "boxes_flow": 0.5
        }) as pdf:
            # Extract metadata
            metadata = pdf.metadata
            metadata_path = plumber_dir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, default=str)
            results["exports"].append(str(metadata_path))
            print(f"  ✓ Saved metadata to {metadata_path}")

            # Extract text from all pages
            all_text = []
            all_tables = []
            page_info = []

            for i, page in enumerate(pdf.pages):
                print(f"Processing page {i+1}/{len(pdf.pages)}...")

                # Extract text
                text = page.extract_text()
                if text:
                    all_text.append(f"--- Page {i+1} ---\n{text}")

                # Extract tables
                tables = page.extract_tables()
                for j, table in enumerate(tables):
                    all_tables.append({
                        "page": i+1,
                        "table_index": j+1,
                        "data": table
                    })

                # Get page info
                page_info.append({
                    "page": i+1,
                    "width": page.width,
                    "height": page.height,
                    "rotation": (page.get('/Rotate', 0) if hasattr(page, 'get') else (getattr(page, 'rotation', None) or 0)),
                    "char_count": len(page.chars) if hasattr(page, 'chars') else 0
                })

            # Save text
            text_path = plumber_dir / "text.txt"
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write("\n\n".join(all_text))
            results["exports"].append(str(text_path))
            print(f"  ✓ Saved text to {text_path}")

            # Save tables as JSON
            if all_tables:
                tables_json_path = plumber_dir / "tables.json"
                with open(tables_json_path, 'w', encoding='utf-8') as f:
                    json.dump(all_tables, f, indent=2)
                results["exports"].append(str(tables_json_path))
                print(f"  ✓ Saved tables (JSON) to {tables_json_path}")

                # Save tables as CSV (first table only for demo)
                if all_tables[0]["data"]:
                    tables_csv_path = plumber_dir / "tables.csv"
                    with open(tables_csv_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        for row in all_tables[0]["data"]:
                            writer.writerow(row)
                    results["exports"].append(str(tables_csv_path))
                    print(f"  ✓ Saved first table (CSV) to {tables_csv_path}")

            # Save page info
            page_info_path = plumber_dir / "page_info.json"
            with open(page_info_path, 'w', encoding='utf-8') as f:
                json.dump(page_info, f, indent=2)
            results["exports"].append(str(page_info_path))
            print(f"  ✓ Saved page info to {page_info_path}")

        results["success"] = True

    except Exception as e:
        results["error"] = str(e)
        print(f"  ✗ Error: {e}")
        traceback.print_exc()

    return results


def process_with_pypdf2(pdf_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Process PDF with PyPDF2 (only works with text-based PDFs, not scanned)."""
    results = {"success": False, "exports": [], "error": None, "warnings": []}

    if not PYPDF2_AVAILABLE:
        results["error"] = "PyPDF2 not installed"
        return results

    try:
        print("\n=== Processing with PyPDF2 ===")

        # Detect PDF type
        pdf_type, confidence = detect_pdf_type(pdf_path)
        if pdf_type == "scanned":
            warning = "⚠️  WARNING: This appears to be a scanned/image PDF. PyPDF2 cannot extract text from images."
            print(warning)
            print("   PyPDF2 only works with text-based PDFs. Use Docling with OCR or Tesseract for scanned PDFs.")
            results["warnings"].append(warning)
        elif pdf_type == "mixed":
            print("   Note: This PDF contains mixed content. PyPDF2 will only extract embedded text, not text from images.")

        pypdf_dir = output_dir / "pypdf2"
        ensure_directory(pypdf_dir)

        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            # Extract metadata
            metadata = {}
            if pdf_reader.metadata:
                metadata = {
                    "title": pdf_reader.metadata.get('/Title', ''),
                    "author": pdf_reader.metadata.get('/Author', ''),
                    "subject": pdf_reader.metadata.get('/Subject', ''),
                    "creator": pdf_reader.metadata.get('/Creator', ''),
                    "producer": pdf_reader.metadata.get('/Producer', ''),
                    "creation_date": str(pdf_reader.metadata.get('/CreationDate', '')),
                    "modification_date": str(pdf_reader.metadata.get('/ModDate', '')),
                }

            metadata_path = pypdf_dir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            results["exports"].append(str(metadata_path))
            print(f"  ✓ Saved metadata to {metadata_path}")

            # Extract text and page info
            all_text = []
            page_info = []

            for i, page in enumerate(pdf_reader.pages):
                print(f"Processing page {i+1}/{len(pdf_reader.pages)}...")

                # Extract text
                text = page.extract_text()
                all_text.append(f"--- Page {i+1} ---\n{text}")

                # Get page info
                mediabox = page.mediabox
                page_info.append({
                    "page": i+1,
                    "width": float(mediabox.width),
                    "height": float(mediabox.height),
                    "rotation": (page.get('/Rotate', 0) if hasattr(page, 'get') else (getattr(page, 'rotation', None) or 0)),
                })

            # Save text
            text_path = pypdf_dir / "text.txt"
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write("\n\n".join(all_text))
            results["exports"].append(str(text_path))
            print(f"  ✓ Saved text to {text_path}")

            # Save page info
            page_info_path = pypdf_dir / "page_info.json"
            with open(page_info_path, 'w', encoding='utf-8') as f:
                json.dump(page_info, f, indent=2)
            results["exports"].append(str(page_info_path))
            print(f"  ✓ Saved page info to {page_info_path}")

            # Save document info
            doc_info = {
                "page_count": len(pdf_reader.pages),
                "is_encrypted": pdf_reader.is_encrypted,
                "metadata": metadata,
                "pdf_version": pdf_reader.pdf_header if hasattr(pdf_reader, 'pdf_header') else None,
            }
            doc_info_path = pypdf_dir / "document_info.json"
            with open(doc_info_path, 'w', encoding='utf-8') as f:
                json.dump(doc_info, f, indent=2)
            results["exports"].append(str(doc_info_path))
            print(f"  ✓ Saved document info to {doc_info_path}")

        results["success"] = True

    except Exception as e:
        results["error"] = str(e)
        print(f"  ✗ Error: {e}")
        traceback.print_exc()

    return results


def process_with_docling(pdf_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Process PDF with docling."""
    results = {"success": False, "exports": [], "error": None}

    if not DOCLING_AVAILABLE:
        results["error"] = "docling not installed"
        return results

    try:
        print("\n=== Processing with Docling ===")
        docling_dir = output_dir / "docling"
        ensure_directory(docling_dir)

        # Try with OCR first for better text extraction
        print("Attempting with OCR enabled...")
        try:
            # Create converter via compatibility helper if available
            if dl_build_converter is not None and dl_create_ocr_options is not None:
                ocr_options = dl_create_ocr_options("tesseract", ["eng"], force_full_page_ocr=True)
                converter = dl_build_converter(
                    enable_ocr=True,
                    ocr_options=ocr_options,
                    enable_tables=True,
                    generate_page_images=False,
                    table_mode="accurate",
                    image_scale=1.0,
                )
            else:
                # Fallback to direct construction
                ocr_options = TesseractOcrOptions(force_full_page_ocr=True)
                pipeline_options = PdfPipelineOptions(
                    do_ocr=True,
                    ocr_options=ocr_options,
                    do_table_structure=True,
                )
                pdf_format = PdfFormatOption(pipeline_options=pipeline_options)
                converter = DocumentConverter(format_options={"pdf": pdf_format})

            print("Converting document with OCR...")
            result = converter.convert(pdf_path)
            print("  ✓ OCR processing successful")

        except Exception as ocr_error:
            print(f"  OCR failed: {ocr_error}")
            print("  Falling back to non-OCR processing...")

            # Fallback to non-OCR processing
            if dl_build_converter is not None:
                converter = dl_build_converter(
                    enable_ocr=False,
                    ocr_options=None,
                    enable_tables=True,
                    generate_page_images=False,
                    table_mode="accurate",
                    image_scale=1.0,
                )
            else:
                pipeline_options = PdfPipelineOptions(
                    do_ocr=False,
                    do_table_structure=True,
                )
                pdf_format = PdfFormatOption(pipeline_options=pipeline_options)
                converter = DocumentConverter(format_options={"pdf": pdf_format})

            print("Converting document without OCR...")
            result = converter.convert(pdf_path)

        # Export as Markdown
        markdown_path = docling_dir / "markdown.md"
        try:
            markdown_content = (
                dl_export_markdown(result.document) if dl_export_markdown else result.document.export_to_markdown()
            )
        except Exception:
            markdown_content = result.document.export_to_markdown()
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        results["exports"].append(str(markdown_path))
        print(f"  ✓ Saved markdown to {markdown_path}")

        # Export as JSON (full document structure)
        json_path = docling_dir / "document.json"
        try:
            json_content = (
                dl_export_dict(result.document) if dl_export_dict else result.document.export_to_dict()
            )
        except Exception:
            json_content = result.document.export_to_dict()
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_content, f, indent=2, ensure_ascii=False)
        results["exports"].append(str(json_path))
        print(f"  ✓ Saved JSON document to {json_path}")

        # Export as plain text
        text_path = docling_dir / "text.txt"
        try:
            text_content = (
                dl_export_text(result.document) if dl_export_text else result.document.export_to_text()
            )
        except Exception:
            text_content = result.document.export_to_text()
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
        results["exports"].append(str(text_path))
        print(f"  ✓ Saved text to {text_path}")

        # Export document elements with types and confidence
        elements = []
        collections = (
            dl_get_document_collections(result.document)
            if dl_get_document_collections
            else {
                "texts": getattr(result.document, "texts", []),
                "tables": getattr(result.document, "tables", []),
                "pictures": getattr(result.document, "pictures", []),
            }
        )

        for item in collections.get("texts", []):
            elements.append({
                "type": "text",
                "content": getattr(item, 'text', None),
                "label": getattr(item, 'label', None),
            })

        for table in collections.get("tables", []):
            try:
                df = dl_table_to_dataframe(table) if dl_table_to_dataframe else table.export_to_dataframe()
                table_content = df.to_dict() if hasattr(df, 'to_dict') else str(df)
            except Exception:
                table_content = str(table)
            elements.append({
                "type": "table",
                "content": table_content,
            })

        for picture in collections.get("pictures", []):
            elements.append({
                "type": "picture",
                "caption": getattr(picture, 'caption', None),
            })

        elements_path = docling_dir / "elements.json"
        with open(elements_path, 'w', encoding='utf-8') as f:
            json.dump(elements, f, indent=2, ensure_ascii=False, default=str)
        results["exports"].append(str(elements_path))
        print(f"  ✓ Saved elements to {elements_path}")

        # Export metadata
        try:
            page_count = len(result.pages) if hasattr(result, 'pages') else None
        except Exception:
            page_count = None
        collections = (
            dl_get_document_collections(result.document)
            if dl_get_document_collections
            else {
                "texts": getattr(result.document, "texts", []),
                "tables": getattr(result.document, "tables", []),
                "pictures": getattr(result.document, "pictures", []),
            }
        )
        metadata = {
            "page_count": page_count,
            "text_count": len(collections.get("texts", [])),
            "table_count": len(collections.get("tables", [])),
            "picture_count": len(collections.get("pictures", [])),
        }
        metadata_path = docling_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        results["exports"].append(str(metadata_path))
        print(f"  ✓ Saved metadata to {metadata_path}")

        results["success"] = True

    except Exception as e:
        results["error"] = str(e)
        print(f"  ✗ Error: {e}")
        traceback.print_exc()

    return results


def print_summary(all_results: Dict[str, Dict[str, Any]]) -> None:
    """Print a summary of all processing results."""
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)

    for library, results in all_results.items():
        print(f"\n{library}:")
        if results["success"]:
            print(f"  ✓ Success - {len(results['exports'])} files exported")
            for export in results["exports"]:
                print(f"    - {export}")
        else:
            print(f"  ✗ Failed: {results['error']}")

    print("\n" + "="*60)
    print("All exports completed!")
    print("="*60)


def main():
    """Main execution function."""
    # Parse command line arguments
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
    else:
        # Use default test PDF
        pdf_path = Path("tests/fixtures/text_simple.pdf")
        if not pdf_path.exists():
            pdf_path = Path("tests/test.pdf")
        if not pdf_path.exists():
            print("Error: No PDF file specified and no default test PDF found.")
            print(f"Usage: {sys.argv[0]} <pdf_file>")
            sys.exit(1)

    if not pdf_path.exists():
        print(f"Error: PDF file '{pdf_path}' not found.")
        sys.exit(1)

    print(f"Processing PDF: {pdf_path}")
    print(f"File size: {pdf_path.stat().st_size / 1024:.2f} KB")

    # Detect PDF type
    pdf_type, confidence = detect_pdf_type(pdf_path)
    print(f"PDF Type: {pdf_type.capitalize()} (confidence: {confidence:.1%})")
    if pdf_type == "scanned":
        print("\n📌 IMPORTANT: This is a scanned/image-based PDF.")
        print("   - PyPDF2 and pdfplumber will NOT extract meaningful text")
        print("   - Use Docling with OCR enabled or Tesseract for best results")
        print("   - Consider pre-processing with OCRmyPDF for better compatibility")

    # Create output directory with timestamp inside pdf_testing folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path("pdf_testing")
    output_dir = base_dir / f"outputs_{timestamp}"
    ensure_directory(output_dir)
    print(f"Output directory: {output_dir}")

    # Process with each library
    all_results = {}

    # Tesseract OCR
    all_results["Tesseract"] = process_with_tesseract(pdf_path, output_dir)

    # pdfplumber
    all_results["pdfplumber"] = process_with_pdfplumber(pdf_path, output_dir)

    # PyPDF2
    all_results["PyPDF2"] = process_with_pypdf2(pdf_path, output_dir)

    # Docling
    all_results["Docling"] = process_with_docling(pdf_path, output_dir)

    # Print summary
    print_summary(all_results)

    # Save summary to file
    summary_path = output_dir / "processing_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()

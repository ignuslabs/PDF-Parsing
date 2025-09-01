"""Custom exceptions for the Smart PDF Parser"""

from typing import List, Any, Optional


class PDFParserError(Exception):
    """Base exception for PDF parser errors"""

    pass


class DocumentParsingError(PDFParserError):
    """Raised when document parsing fails"""

    def __init__(self, message: str, document_path: Optional[str] = None) -> None:
        self.document_path = document_path
        super().__init__(message)


class OCRError(PDFParserError):
    """Raised when OCR processing fails"""

    pass


class ValidationError(PDFParserError):
    """Raised when data validation fails"""

    def __init__(self, message: str, validation_errors: Optional[List[Any]] = None) -> None:
        self.validation_errors = validation_errors or []
        super().__init__(message)


class ConfigurationError(PDFParserError):
    """Raised when configuration is invalid"""

    pass


class CoordinateError(PDFParserError):
    """Raised when coordinate operations fail"""

    pass


class ExportError(PDFParserError):
    """Raised when export operations fail"""

    pass


class VerificationError(PDFParserError):
    """Raised when verification operations fail"""

    pass

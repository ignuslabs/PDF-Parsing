"""
Header classification module for PDF parsing.

This module provides functionality to classify document elements as headings
or regular text based on various heuristics and patterns.
"""

from .header_classifier import is_heading

__all__ = ["is_heading"]

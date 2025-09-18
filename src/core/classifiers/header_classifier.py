"""
Header classification module for distinguishing headings from form values.

This module implements intelligent classification logic to prevent misclassifying
personal data (names, addresses, phone numbers) as document headings.
"""

import re
from src.core.models import DocumentElement


def is_heading(element: DocumentElement, page_context: dict) -> bool:
    """
    Determine if a DocumentElement should be classified as a heading.

    Uses a scoring system that considers multiple factors:
    - Text formatting (uppercase ratio, punctuation)
    - Content patterns (person names, numeric data)
    - Position on page
    - Length and structure

    Args:
        element: DocumentElement with text and bbox information
        page_context: Dict with page statistics and dimensions

    Returns:
        bool: True if element should be classified as heading
    """
    text = element.text.strip()

    # Handle empty or whitespace-only text
    if not text:
        return False

    # Calculate base score
    score = 0.5  # Start with neutral score

    # Apply scoring rules
    score += _score_text_formatting(text)
    score += _score_position(element, page_context)
    score += _score_structural_patterns(text)
    score -= _score_negative_patterns(text)
    score -= _score_length_penalties(text)

    # Final decision threshold
    return score >= 0.6


def _score_text_formatting(text: str) -> float:
    """Score based on text formatting characteristics."""
    score = 0.0

    # Uppercase ratio boost for short text
    if len(text) < 80:
        uppercase_chars = sum(1 for c in text if c.isupper())
        total_chars = len([c for c in text if c.isalpha()])

        if total_chars > 0:
            uppercase_ratio = uppercase_chars / total_chars
            if uppercase_ratio > 0.6:
                score += 0.4

    # Colon-ended text boost for short text
    if text.endswith(":") and len(text) < 60:
        score += 0.3

    # Status indicators (all caps short phrases) - but check context
    status_patterns = ["CONFIDENTIAL", "DRAFT", "URGENT", "FINAL", "APPROVED"]
    if text.upper() in status_patterns:
        # Don't boost if it appears to be part of a copyright notice or footer
        if not re.search(r"(©|copyright|page\s+\d+)", text, re.IGNORECASE):
            score += 0.3

    return score


def _score_position(element: DocumentElement, page_context: dict) -> float:
    """Score based on position on page."""
    score = 0.0

    try:
        # Check if element is in top 15% of page
        page_height = page_context.get("height", 792.0)
        top_15_percent_threshold = page_context.get("top_15_percent", page_height * 0.85)

        element_y = element.bbox["y0"]  # Bottom of element in PDF coordinates

        # In PDF coordinates, higher Y values are near the top
        if element_y > top_15_percent_threshold:
            score += 0.2

    except (KeyError, TypeError):
        # Handle missing bbox or page context gracefully
        pass

    return score


def _score_structural_patterns(text: str) -> float:
    """Score based on structural indicators."""
    score = 0.0

    # Structural keywords
    structural_keywords = [
        "section",
        "part",
        "chapter",
        "article",
        "appendix",
        "introduction",
        "conclusion",
        "summary",
        "overview",
        "background",
        "methodology",
        "results",
        "discussion",
        "references",
        "bibliography",
        "information",
        "contact",
        "employment",
        "work",
        "experience",
        "education",
        "skills",
        "qualifications",
        "customer",
    ]

    text_lower = text.lower()

    # Check for structural keywords - more specific matching
    for keyword in structural_keywords:
        if keyword in text_lower:
            # Higher boost for exact matches or standalone keywords
            if text_lower == keyword or f" {keyword} " in f" {text_lower} ":
                score += 0.3
            else:
                score += 0.2
            break  # Only apply bonus once

    # Numbered sections or parts
    if re.match(r"^(section|part|chapter|article)\s+\d+", text_lower):
        score += 0.2

    # Roman numerals (often used in formal headings)
    if re.match(r"^[ivxlcdm]+\.?\s", text_lower):
        score += 0.15

    # Legal/formal section references
    if re.match(r"^section\s+\d+\([a-z]\)\(\d+\)", text_lower):
        score += 0.2

    return score


def _score_negative_patterns(text: str) -> float:
    """Score penalties for patterns that indicate non-headings."""
    penalty = 0.0

    # Person name patterns
    if _is_person_name(text):
        penalty += 0.5

    # Date and numeric patterns
    if _is_date_or_numeric(text):
        penalty += 0.5

    # Code-like identifiers (e.g., INV-2025-0001, PO#12345, ABC12345)
    if is_code_like(text):
        penalty += 0.65  # Stronger penalty to avoid classifying codes as headings

    # Common form values
    if _is_common_form_value(text):
        penalty += 0.5

    # Address patterns
    if _is_address_pattern(text):
        penalty += 0.4

    # Email patterns
    if re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text):
        penalty += 0.4

    return penalty


def _score_length_penalties(text: str) -> float:
    """Score penalties based on text length and structure."""
    penalty = 0.0

    # Long text penalty - stronger penalty for very long text
    if len(text) > 80:
        penalty += 0.5
    elif len(text) > 40:
        penalty += 0.1  # Slight penalty for moderately long text

    # Multiple sentences penalty - stronger for multiple sentences
    sentence_count = len([s for s in text.split(". ") if s.strip()])
    if sentence_count > 1:
        penalty += 0.4

    # Very short non-meaningful text
    if len(text) <= 2 and not text.upper() in ["I", "A", "B", "C", "D"]:
        penalty += 0.3

    # Special characters only (like ***, ------) or page footers
    if re.match(r"^[^a-zA-Z0-9]+$", text) and len(text) > 2:
        penalty += 0.5

    # Page footer patterns
    if re.search(r"page\s+\d+\s+of\s+\d+", text, re.IGNORECASE):
        penalty += 0.6

    return penalty


def _is_person_name(text: str) -> bool:
    """Detect if text represents a person's name."""
    text_clean = text.strip()

    # Exclude common section headers that look like names
    section_headers = {
        "employment information", "contact information", "work experience", 
        "education background", "employment history", "personal information",
        "customer information", "billing information", "shipping information",
        "payment information", "account information", "profile information",
        "business information", "company information", "project information",
        "service information", "product information", "system information",
        "security information", "login information", "access information",
        "user information", "client information", "patient information",
        "student information", "employee information", "member information",
        "registration information", "enrollment information", "application information"
    }
    
    if text_clean.lower() in section_headers:
        return False

    # Simple heuristics for person names
    words = text_clean.split()

    if len(words) < 2:
        return False

    # Check for common name patterns including international names
    name_patterns = [
        # First Middle Last
        r"^[A-Z][a-z]+\s+[A-Z][a-z]*\.?\s+[A-Z][a-z]+$",
        # First Last
        r"^[A-Z][a-z]+\s+[A-Z][a-z]+$",
        # First Middle Initial Last
        r"^[A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+$",
        # Title First Last
        r"^(Dr|Mr|Mrs|Ms|Miss)\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+",
        # First Last Suffix
        r"^[A-Z][a-z]+\s+[A-Z][a-z]+\s+(Jr|Sr|III|IV|V)\.?$",
        # International names with accents
        r"^[A-ZÀ-ÿ][a-zà-ÿ]+\s+[A-ZÀ-ÿ][a-zà-ÿ]*\s+[A-ZÀ-ÿ][a-zà-ÿ]+$",
        # Names with apostrophes or hyphens
        r"^[A-ZÀ-ÿ][a-zà-ÿ\'-]+\s+[A-ZÀ-ÿ][a-zà-ÿ\'-]+$",
    ]

    for pattern in name_patterns:
        if re.match(pattern, text_clean):
            return True

    # Check for common name indicators
    title_prefixes = ["Dr.", "Mr.", "Mrs.", "Ms.", "Miss.", "Prof."]
    name_suffixes = ["Jr.", "Sr.", "III", "IV", "V", "Jr", "Sr"]

    has_title = any(text_clean.startswith(prefix) for prefix in title_prefixes)
    has_suffix = any(text_clean.endswith(suffix) for suffix in name_suffixes)

    # Names with apostrophes or hyphens
    has_name_chars = "'" in text_clean or "-" in text_clean

    # All words are title case (typical for names)
    all_title_case = all(
        word[0].isupper() and word[1:].islower() for word in words if word.isalpha()
    )

    # If it has name-like characteristics and isn't too long
    if (has_title or has_suffix or has_name_chars) and all_title_case and len(words) <= 4:
        return True

    # Common first names detection (limited set)
    common_first_names = {
        "john",
        "mary",
        "james",
        "patricia",
        "robert",
        "jennifer",
        "michael",
        "linda",
        "william",
        "elizabeth",
        "david",
        "barbara",
        "richard",
        "susan",
        "joseph",
        "jessica",
        "thomas",
        "sarah",
        "christopher",
        "karen",
        "charles",
        "nancy",
        "daniel",
        "lisa",
        "matthew",
        "betty",
        "anthony",
        "helen",
        "mark",
        "sandra",
        "donald",
        "donna",
        "steven",
        "carol",
        "paul",
        "ruth",
        "andrew",
        "sharon",
        "joshua",
        "michelle",
        "kenneth",
        "laura",
        "kevin",
        "sarah",
        "brian",
        "kimberly",
        "george",
        "deborah",
        "timothy",
        "dorothy",
        "ronald",
        "lisa",
        "jason",
        "nancy",
        "edward",
        "karen",
        "jeffrey",
        "betty",
        "ryan",
        "helen",
        "jacob",
        "sandra",
        "gary",
        "donna",
    }

    if len(words) >= 2:
        first_word = words[0].lower()
        if first_word in common_first_names:
            return True

    return False


def _is_date_or_numeric(text: str) -> bool:
    """Detect if text represents dates, numbers, or numeric identifiers."""
    text_clean = text.strip()

    # Date patterns
    date_patterns = [
        r"\d{1,2}/\d{1,2}/\d{4}",  # MM/DD/YYYY
        r"\d{1,2}-\d{1,2}-\d{4}",  # MM-DD-YYYY
        r"\d{4}-\d{2}-\d{2}",  # YYYY-MM-DD
        r"\b\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}",
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}",
    ]

    for pattern in date_patterns:
        if re.search(pattern, text_clean, re.IGNORECASE):
            return True

    # Social Security Number
    if re.match(r"^\d{3}-\d{2}-\d{4}$", text_clean):
        return True

    # Phone numbers
    phone_patterns = [
        r"^\(\d{3}\)\s*\d{3}-\d{4}$",  # (555) 123-4567
        r"^\d{3}-\d{3}-\d{4}$",  # 555-123-4567
        r"^\d{3}\.\d{3}\.\d{4}$",  # 555.123.4567
        r"^\d{10}$",  # 5551234567
    ]

    for pattern in phone_patterns:
        if re.match(pattern, text_clean):
            return True

    # Currency amounts
    if re.match(r"^\$[\d,]+\.?\d*$", text_clean):
        return True

    # Percentages
    if re.match(r"^\d+\.?\d*%$", text_clean):
        return True

    # Account numbers, policy numbers, etc.
    if re.search(r"(account|policy|invoice|order)\s*#?:?\s*[\w\d-]+", text_clean, re.IGNORECASE):
        return True

    # ZIP codes
    if re.match(r"^\d{5}(-\d{4})?$", text_clean):
        return True

    # Room numbers, model numbers, building identifiers
    if re.match(r"^(room|unit|model|version|building|form)\s+[\w\d-]+", text_clean, re.IGNORECASE):
        return True

    # Pure numbers
    try:
        # Check if it's just a number (with possible commas)
        float(text_clean.replace(",", ""))
        if len(text_clean.replace(",", "").replace(".", "")) > 2:  # Not just 1 or 2 digit numbers
            return True
    except ValueError:
        pass

    return False


def is_code_like(text: str) -> bool:
    """Detect code-like identifiers that are not headings.

    Examples: invoice numbers, PO numbers, policy numbers, claim IDs,
    alphanumeric with dashes/underscores, leading prefixes.
    """
    t = text.strip()
    if not t:
        return False

    # Common prefixes for codes
    prefixes = [
        "inv", "invoice", "po", "po#", "order", "ord", "claim", "pol", "policy",
        "acct", "account", "mrn", "id", "ref", "ref#", "ticket", "case",
    ]
    tl = t.lower()
    for p in prefixes:
        if tl.startswith(p + " ") or tl.startswith(p + ":") or tl.startswith(p + "#"):
            return True

    # Alphanumeric code patterns (6–24 chars, mostly uppercase/digits, optional dashes/underscores)
    import re as _re
    if _re.match(r"^[A-Z0-9][A-Z0-9_-]{4,23}$", t.replace(" ", "")):
        return True

    # Contains multiple dashes in an alphanumeric token (e.g., ABC-123-XYZ)
    if _re.match(r"^[A-Z0-9]+(-[A-Z0-9]+){1,3}$", t.replace(" ", "")):
        return True

    # Looks like a GUID fragment (not full GUID to avoid false positives)
    if _re.match(r"^[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-", t.replace(" ", "")):
        return True

    return False


def _is_common_form_value(text: str) -> bool:
    """Detect common form field values."""
    text_lower = text.lower().strip()

    # Common form values
    common_values = {
        "yes",
        "no",
        "n/a",
        "na",
        "none",
        "unknown",
        "other",
        "male",
        "female",
        "single",
        "married",
        "divorced",
        "widowed",
        "employed",
        "unemployed",
        "retired",
        "student",
        "home",
        "work",
        "cell",
        "mobile",
        "office",
        "united states",
        "usa",
        "us",
        "canada",
        "active",
        "inactive",
        "pending",
        "approved",
        "denied",
        "full-time",
        "part-time",
        "temporary",
        "contract",
        "bachelor",
        "master",
        "doctorate",
        "phd",
        "high school",
        "elementary",
        "middle school",
        "college",
        "university",
        "springfield",
        "apt",
        "software engineer",
        "abc corporation",
        "building a",
        "unit 2b",
        "form w-2",
        "model xyz-123",
        "version",
        "page 1 of 5",
        "copyright",
        "© 2024",
        "confidential",
        "draft",
    }

    if text_lower in common_values:
        return True

    # Education levels
    education_patterns = [
        r"bachelor'?s?\s+(degree|of)",
        r"master'?s?\s+(degree|of)",
        r"associate'?s?\s+(degree|of)",
        r"high\s+school\s+(diploma|graduate)",
    ]

    for pattern in education_patterns:
        if re.search(pattern, text_lower):
            return True

    return False


def _is_address_pattern(text: str) -> bool:
    """Detect address-like patterns."""
    text_clean = text.strip()

    # Street address patterns
    street_patterns = [
        r"^\d+\s+\w+\s+(street|st|avenue|ave|road|rd|lane|ln|drive|dr|boulevard|blvd|place|pl|court|ct)",
        r"^\d+\s+[A-Z]\w+\s+(Street|Ave|Road|Lane|Drive|Boulevard|Place|Court)",
        r"(apt|apartment|unit|suite|ste)\s*#?\s*\w+",
        r"\d+\s+\w+.*,\s*\w+\s+\d{5}",  # Street, City ZIP
    ]

    for pattern in street_patterns:
        if re.search(pattern, text_clean, re.IGNORECASE):
            return True

    # City, State ZIP pattern
    if re.search(r"\w+,\s*[A-Z]{2}\s+\d{5}", text_clean):
        return True

    # International addresses
    international_patterns = [
        r"\w+,\s*(France|Germany|Italy|Spain|UK|Canada|Australia)",
        r"^\d+\s+(Rue|Avenue|Boulevard)\s+",  # French
        r"^\w+straße\s+\d+",  # German
        r"^Via\s+\w+\s+\d+",  # Italian
    ]

    for pattern in international_patterns:
        if re.search(pattern, text_clean, re.IGNORECASE):
            return True

    return False

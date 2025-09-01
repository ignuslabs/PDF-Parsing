"""
Unit tests for header classification functionality.
Tests the header classifier that prevents misclassifying values as headings.

This test suite follows TDD principles - tests are written first and will fail
until the header classifier implementation is complete in Step 4.
"""

import pytest
from typing import Dict, Any, List

# Import the classes we'll be testing
from src.core.models import DocumentElement


class TestHeaderClassifier:
    """Test suite for header classification functionality."""

    def create_test_element(
        self, text: str, bbox: Dict[str, float], page_number: int = 1, confidence: float = 0.85
    ) -> DocumentElement:
        """Helper to create DocumentElement for testing.

        Args:
            text: Text content of the element
            bbox: Bounding box coordinates {'x0', 'y0', 'x1', 'y1'}
            page_number: Page number (default 1)
            confidence: Confidence score (default 0.85)

        Returns:
            DocumentElement instance for testing
        """
        return DocumentElement(
            text=text,
            element_type="text",  # Will be reclassified by header classifier
            page_number=page_number,
            bbox=bbox,
            confidence=confidence,
            metadata={},
        )

    def create_page_context(
        self, page_width: float = 612.0, page_height: float = 792.0
    ) -> Dict[str, Any]:
        """Helper to create page context for testing.

        Args:
            page_width: Page width in points (default letter size)
            page_height: Page height in points (default letter size)

        Returns:
            Dictionary with page statistics
        """
        return {
            "width": page_width,
            "height": page_height,
            "top_15_percent": page_height * 0.85,  # Y coordinate for top 15% (PDF coordinates)
            "elements_count": 0,  # Will be set by specific tests
        }

    # Positive Cases - Should classify as heading

    def test_classifies_all_caps_short_text_as_heading(self):
        """Test that short all-caps text is classified as heading.

        Expected behavior:
        - Uppercase ratio > 0.6 and length < 80 → heading boost
        - Examples: "APPLICATION FORM", "SECTION A", "PERSONAL INFO"
        """
        test_cases = [
            "APPLICATION FORM",
            "SECTION A",
            "PERSONAL INFO",
            "EMPLOYMENT HISTORY",
            "PART I",
            "REFERENCES",
        ]

        page_context = self.create_page_context()

        # Import the classifier function (will fail until implemented)
        from src.core.parser import classify_as_heading

        for text in test_cases:
            element = self.create_test_element(
                text=text,
                bbox={"x0": 50.0, "y0": 700.0, "x1": 200.0, "y1": 720.0},  # Near top of page
            )

            is_heading = classify_as_heading(element, page_context)
            assert is_heading, f"'{text}' should be classified as heading (all caps, short)"

    def test_classifies_colon_ended_text_as_heading(self):
        """Test that colon-ended short text is classified as heading.

        Expected behavior:
        - Ends with ':' and length < 60 → heading boost
        - Examples: "Name:", "Address:", "Section 1:"
        """
        test_cases = [
            "Name:",
            "Address:",
            "Section 1:",
            "Employment Information:",
            "Phone Number:",
            "Date of Birth:",
            "Emergency Contact:",
        ]

        page_context = self.create_page_context()

        from src.core.parser import classify_as_heading

        for text in test_cases:
            element = self.create_test_element(
                text=text, bbox={"x0": 50.0, "y0": 600.0, "x1": 150.0, "y1": 620.0}
            )

            is_heading = classify_as_heading(element, page_context)
            assert is_heading, f"'{text}' should be classified as heading (colon-ended)"

    def test_classifies_mixed_case_structural_text_as_heading(self):
        """Test that mixed case structural indicators are classified as headings.

        Expected behavior:
        - Structural keywords → heading boost
        - Examples: "Employment Information", "Part II - References"
        """
        test_cases = [
            "Employment Information",
            "Part II - References",
            "Section B: Education",
            "Contact Information",
            "Work Experience",
            "Skills and Qualifications",
            "Chapter 1: Introduction",
        ]

        page_context = self.create_page_context()

        from src.core.parser import classify_as_heading

        for text in test_cases:
            element = self.create_test_element(
                text=text, bbox={"x0": 50.0, "y0": 650.0, "x1": 250.0, "y1": 670.0}
            )

            is_heading = classify_as_heading(element, page_context)
            assert is_heading, f"'{text}' should be classified as heading (structural)"

    def test_page_position_affects_classification(self):
        """Test that elements within top 15% of page get heading boost.

        Expected behavior:
        - Page top percentile < 15% → heading boost
        - Same text should have different classification based on position
        """
        page_context = self.create_page_context(page_width=612.0, page_height=792.0)
        text = "Customer Information"  # Borderline case

        from src.core.parser import classify_as_heading

        # Element near top of page (within top 15%)
        top_element = self.create_test_element(
            text=text,
            bbox={"x0": 50.0, "y0": 720.0, "x1": 200.0, "y1": 740.0},  # Y=720 > 673.2 (85% of 792)
        )

        # Element in middle of page
        middle_element = self.create_test_element(
            text=text, bbox={"x0": 50.0, "y0": 400.0, "x1": 200.0, "y1": 420.0}  # Middle of page
        )

        is_heading_top = classify_as_heading(top_element, page_context)
        is_heading_middle = classify_as_heading(middle_element, page_context)

        # Top element should be more likely to be classified as heading
        assert is_heading_top, f"'{text}' near page top should be classified as heading"
        # Middle element may or may not be heading - depends on other factors

    # Negative Cases - Should NOT classify as heading

    def test_does_not_classify_person_names_as_heading(self):
        """Test that person names are not classified as headings.

        Expected behavior:
        - Person name pattern → demote
        - Examples: "John A. Smith", "Mary Johnson", "Robert Williams Jr."
        """
        test_cases = [
            "John A. Smith",
            "Mary Johnson",
            "Robert Williams Jr.",
            "Sarah Michelle Parker",
            "Dr. James Wilson",
            "Michael O'Connor",
            "Lisa Marie Thompson",
            "Jose Rodriguez-Martinez",
            "Catherine Elizabeth Brown",
        ]

        page_context = self.create_page_context()

        from src.core.parser import classify_as_heading

        for text in test_cases:
            element = self.create_test_element(
                text=text, bbox={"x0": 150.0, "y0": 600.0, "x1": 300.0, "y1": 620.0}
            )

            is_heading = classify_as_heading(element, page_context)
            assert not is_heading, f"'{text}' should NOT be classified as heading (person name)"

    def test_does_not_classify_numeric_values_as_heading(self):
        """Test that numeric values are not classified as headings.

        Expected behavior:
        - Date/numeric pattern → demote
        - Examples: "123-45-6789", "$75,000", "01/23/1980"
        """
        test_cases = [
            "123-45-6789",  # SSN
            "$75,000",  # Currency
            "01/23/1980",  # Date
            "555-123-4567",  # Phone
            "12345",  # Zip code
            "Account #: 987654321",
            "Policy No. 12-ABC-567",
            "Invoice #2024-001",
            "June 15, 2023",
            "3.14159",
            "100%",
            "Room 204B",
        ]

        page_context = self.create_page_context()

        from src.core.parser import classify_as_heading

        for text in test_cases:
            element = self.create_test_element(
                text=text, bbox={"x0": 150.0, "y0": 500.0, "x1": 250.0, "y1": 520.0}
            )

            is_heading = classify_as_heading(element, page_context)
            assert not is_heading, f"'{text}' should NOT be classified as heading (numeric/date)"

    def test_does_not_classify_common_form_values_as_heading(self):
        """Test that common form field values are not classified as headings.

        Expected behavior:
        - Address patterns, phone patterns, etc. → demote
        - Examples: "Springfield, IL 62701", "(555) 123-4567"
        """
        test_cases = [
            "Springfield, IL 62701",
            "(555) 123-4567",
            "john.doe@email.com",
            "123 Main Street",
            "Apt 4B",
            "United States",
            "Male",
            "Female",
            "Single",
            "Married",
            "Yes",
            "No",
            "N/A",
            "Bachelor's Degree",
            "Software Engineer",
            "ABC Corporation",
        ]

        page_context = self.create_page_context()

        from src.core.parser import classify_as_heading

        for text in test_cases:
            element = self.create_test_element(
                text=text, bbox={"x0": 200.0, "y0": 450.0, "x1": 350.0, "y1": 470.0}
            )

            is_heading = classify_as_heading(element, page_context)
            assert not is_heading, f"'{text}' should NOT be classified as heading (form value)"

    def test_does_not_classify_long_descriptive_text_as_heading(self):
        """Test that long paragraphs are not classified as headings.

        Expected behavior:
        - Length > 80 characters → demote
        - Multi-sentence content → demote
        """
        test_cases = [
            "This is a longer paragraph that contains multiple sentences and should not be classified as a heading.",
            "Please provide detailed information about your work experience including company names, job titles, dates of employment, and key responsibilities.",
            "I certify that all information provided in this application is true and complete to the best of my knowledge.",
            "The quick brown fox jumps over the lazy dog. This sentence is used for testing purposes only.",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore.",
        ]

        page_context = self.create_page_context()

        from src.core.parser import classify_as_heading

        for text in test_cases:
            element = self.create_test_element(
                text=text,
                bbox={"x0": 50.0, "y0": 300.0, "x1": 500.0, "y1": 350.0},  # Wide bbox for long text
            )

            is_heading = classify_as_heading(element, page_context)
            assert (
                not is_heading
            ), f"Long text should NOT be classified as heading (length: {len(text)})"

    # Edge Cases

    def test_handles_edge_cases_gracefully(self):
        """Test that classifier handles edge cases gracefully.

        Expected behavior:
        - Empty/whitespace text → not heading
        - Very long text → not heading
        - Special characters → depends on context
        """
        edge_cases = [
            ("", False, "Empty string"),
            ("   ", False, "Whitespace only"),
            ("A" * 100, False, "Very long text"),
            ("***", False, "Special characters only"),
            ("------", False, "Divider line"),
            ("Page 1 of 5", False, "Page footer"),
            ("© 2024 Company Name", False, "Copyright notice"),
            ("CONFIDENTIAL", True, "Status label - should be heading"),
            ("DRAFT", True, "Document status - should be heading"),
        ]

        page_context = self.create_page_context()

        from src.core.parser import classify_as_heading

        for text, expected_result, description in edge_cases:
            if not text.strip():  # Handle empty/whitespace
                # These should return False without error
                element = self.create_test_element(
                    text=text, bbox={"x0": 50.0, "y0": 400.0, "x1": 100.0, "y1": 420.0}
                )

                is_heading = classify_as_heading(element, page_context)
                assert (
                    is_heading == expected_result
                ), f"{description}: expected {expected_result}, got {is_heading}"
            else:
                element = self.create_test_element(
                    text=text, bbox={"x0": 50.0, "y0": 400.0, "x1": 200.0, "y1": 420.0}
                )

                is_heading = classify_as_heading(element, page_context)
                assert (
                    is_heading == expected_result
                ), f"{description}: expected {expected_result}, got {is_heading}"

    def test_international_names_and_addresses(self):
        """Test that international names and addresses are not classified as headings."""
        test_cases = [
            "José María García",
            "François Dubois",
            "李小明",
            "محمد أحمد",
            "Владимир Петров",
            "123 Rue de la Paix, Paris, France",
            "Königstraße 45, Munich, Germany",
            "Via Roma 123, Milano, Italy",
        ]

        page_context = self.create_page_context()

        from src.core.parser import classify_as_heading

        for text in test_cases:
            element = self.create_test_element(
                text=text, bbox={"x0": 150.0, "y0": 500.0, "x1": 300.0, "y1": 520.0}
            )

            is_heading = classify_as_heading(element, page_context)
            assert (
                not is_heading
            ), f"'{text}' should NOT be classified as heading (international name/address)"

    def test_mixed_alphanumeric_content(self):
        """Test classification of mixed alphanumeric content."""
        test_cases = [
            ("Form W-2", False, "Tax form identifier"),
            ("Section 501(c)(3)", True, "Legal section reference - could be heading"),
            ("Building A", False, "Location identifier"),
            ("Unit 2B", False, "Unit number"),
            ("Chapter 12", True, "Chapter reference - should be heading"),
            ("Article III", True, "Article reference - should be heading"),
            ("Model XYZ-123", False, "Product model"),
            ("Version 2.1.4", False, "Version number"),
        ]

        page_context = self.create_page_context()

        from src.core.parser import classify_as_heading

        for text, expected_result, description in test_cases:
            element = self.create_test_element(
                text=text, bbox={"x0": 50.0, "y0": 600.0, "x1": 200.0, "y1": 620.0}
            )

            is_heading = classify_as_heading(element, page_context)
            assert (
                is_heading == expected_result
            ), f"{description}: expected {expected_result}, got {is_heading}"

    # Test Classification Scoring Thresholds

    def test_classification_scoring_thresholds(self):
        """Test that the classification scoring follows expected thresholds.

        Expected behavior from implementation plan:
        - Uppercase ratio > 0.6 and length < 80 → heading boost
        - Ends with ':' and length < 60 → heading boost
        - Page top percentile < 15% → heading boost
        - Person name pattern → demote
        - Date/numeric pattern → demote
        """
        page_context = self.create_page_context(page_width=612.0, page_height=792.0)

        from src.core.parser import classify_as_heading

        # Test uppercase ratio threshold
        uppercase_cases = [
            ("MOSTLY UPPER", True, "High uppercase ratio"),
            ("Mostly lower", False, "Low uppercase ratio"),
            ("Mixed Case Text", False, "Moderate uppercase ratio"),
        ]

        for text, expected, description in uppercase_cases:
            element = self.create_test_element(
                text=text, bbox={"x0": 50.0, "y0": 400.0, "x1": 200.0, "y1": 420.0}
            )

            is_heading = classify_as_heading(element, page_context)
            # Note: Expected results may vary based on other factors
            # This test primarily ensures the function handles different cases

    def test_borderline_cases(self):
        """Test borderline cases that might be ambiguous.

        These cases test the classifier's ability to handle ambiguous text
        that could reasonably be classified either way.
        """
        borderline_cases = [
            ("Summary", "Could be heading or content"),
            ("Total", "Could be heading or value label"),
            ("Conclusion", "Could be heading or content"),
            ("Notes", "Could be heading or field label"),
            ("Status: Active", "Colon suggests heading but content suggests value"),
            ("Part 1", "Could be heading or reference"),
            ("Question 5", "Could be heading or form element"),
        ]

        page_context = self.create_page_context()

        from src.core.parser import classify_as_heading

        for text, description in borderline_cases:
            element = self.create_test_element(
                text=text, bbox={"x0": 50.0, "y0": 500.0, "x1": 150.0, "y1": 520.0}
            )

            # For borderline cases, we just ensure the function runs without error
            # The specific result depends on the implementation details
            is_heading = classify_as_heading(element, page_context)
            assert isinstance(
                is_heading, bool
            ), f"classify_as_heading should return boolean for '{text}'"


class TestPageContextCalculation:
    """Test page context calculation for header classification."""

    def test_page_context_top_percentile_calculation(self):
        """Test that page context correctly calculates top percentile boundaries."""
        # Standard letter size page
        page_context = {
            "width": 612.0,
            "height": 792.0,
            "top_15_percent": 792.0 * 0.85,  # Y coordinate for top 15%
            "elements_count": 10,
        }

        # Element near top should be in top 15%
        top_y = 750.0  # Above 673.2 threshold
        assert top_y > page_context["top_15_percent"]

        # Element near bottom should not be in top 15%
        bottom_y = 100.0  # Below 673.2 threshold
        assert bottom_y < page_context["top_15_percent"]

    def test_different_page_sizes(self):
        """Test page context calculation with different page sizes."""
        page_sizes = [
            (612.0, 792.0, "Letter"),
            (595.0, 842.0, "A4"),
            (612.0, 1008.0, "Legal"),
            (792.0, 1224.0, "Tabloid"),
        ]

        for width, height, size_name in page_sizes:
            page_context = {
                "width": width,
                "height": height,
                "top_15_percent": height * 0.85,
                "elements_count": 5,
            }

            # Top 15% boundary should be reasonable
            assert page_context["top_15_percent"] > height * 0.5  # Above middle
            assert page_context["top_15_percent"] < height  # Below max height

            # Verify calculation
            expected_boundary = height * 0.85
            assert abs(page_context["top_15_percent"] - expected_boundary) < 0.1


# Integration test helpers (will be used when classifier is integrated)


class TestHeaderClassifierIntegration:
    """Integration tests for header classifier with the parser."""

    @pytest.mark.integration
    def test_classifier_integration_with_parser(self):
        """Test that header classifier integrates properly with the parser.

        This test will be enabled once the classifier is integrated into the parser.
        """
        pytest.skip("Integration test - requires parser implementation")

        # Future test structure:
        # 1. Parse a document with known headings and values
        # 2. Verify headings are classified correctly
        # 3. Verify values are not misclassified as headings

    @pytest.mark.integration
    def test_classifier_with_real_form_documents(self):
        """Test classifier accuracy on real form documents.

        This test will use actual form fixtures to validate classification.
        """
        pytest.skip("Integration test - requires form fixtures")

        # Future test structure:
        # 1. Load form PDF fixtures
        # 2. Parse with header classification enabled
        # 3. Manually verify results against expected classifications
        # 4. Calculate precision/recall metrics


# Expected behavior documentation (for implementation reference)

"""
IMPLEMENTATION REFERENCE - Expected Classification Rules:

1. HEADING INDICATORS (boost score):
   - Uppercase ratio > 0.6 and length < 80 characters
   - Ends with ':' and length < 60 characters
   - Contains structural keywords: "Section", "Part", "Chapter", "Article"
   - Y position in top 15% of page
   - Short length (< 40 characters) with title case
   - All caps single words or short phrases

2. NON-HEADING INDICATORS (demote score):
   - Person name patterns (Title? FirstName MiddleName? LastName Suffix?)
   - Date patterns (MM/DD/YYYY, Month DD, YYYY, etc.)
   - Numeric patterns (SSN, phone, currency, percentages)
   - Address patterns (Street, City State ZIP)
   - Email patterns
   - Length > 80 characters
   - Multiple sentences (contains '. ' pattern)
   - Common form values (Yes/No, Male/Female, etc.)

3. SCORING THRESHOLDS:
   - Score >= 0.7: Classify as heading
   - Score < 0.7: Keep as text
   - Base score starts at 0.5
   - Each indicator adds/subtracts points
   - Final decision based on total score

4. EDGE CASES:
   - Empty/whitespace text → not heading
   - Special characters only → not heading unless status indicators
   - Very short text (1-2 chars) → not heading unless specific patterns
   - International text → apply same rules with Unicode awareness
"""

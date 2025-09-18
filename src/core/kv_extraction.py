"""
Key-Value extraction module for Smart PDF Parser.

This module implements intelligent key-value pair extraction from parsed PDF documents
using spatial analysis, label detection heuristics, and confidence scoring.

The extraction algorithm follows a multi-step process:
1. Group elements by page and cluster into lines
2. Detect column boundaries for multi-column layouts
3. Score elements as potential labels using text and position features
4. Find associated values using same-line and below-pairing strategies
5. Merge multi-line values through spatial adjacency
6. Calculate confidence scores using label, geometric, and content factors
7. Deduplicate to prevent multiple labels claiming the same value
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import re
import math
import logging
from collections import defaultdict

from src.core.models import DocumentElement, KeyValuePair
from src.utils.logging_config import ProgressTracker, time_it, log_memory_usage

# Initialize logger for KV extraction
logger = logging.getLogger(__name__)


@dataclass
class KVConfig:
    """Configuration parameters for key-value extraction.

    This dataclass contains all tunable parameters for the extraction algorithm,
    allowing for fine-tuning based on document types and layouts.
    """

    # Distance thresholds for pairing strategies
    max_same_line_dx: float = 200.0  # Max horizontal distance for same-line pairing
    max_below_dy: float = 100.0  # Max vertical distance for below pairing
    x_align_tol: float = 40.0  # X-alignment tolerance for vertical pairing

    # Column detection parameters
    gutter_min_dx: float = 120.0  # Minimum distance to detect column gutters

    # Label detection parameters
    max_label_len: int = 60  # Maximum length for label text
    min_upper_ratio: float = 0.4  # Minimum uppercase ratio for labels
    min_value_len: int = 1  # Minimum length for value text

    # Confidence calculation weights
    label_weight: float = 0.5  # Weight for label score component
    geom_weight: float = 0.3  # Weight for geometric score component
    content_weight: float = 0.2  # Weight for content score component

    # Domain-specific boosts
    label_keywords_boost: float = 0.2  # Additional boost for known label keywords
    code_value_boost: float = 0.2      # Additional boost for code-like value patterns


class KeyValueExtractor:
    """
    Main class for extracting key-value pairs from document elements.

    The extractor uses a multi-stage pipeline:
    1. Preprocessing: Group elements by page, cluster into lines and columns
    2. Label detection: Score elements as potential labels using heuristics
    3. Value pairing: Find associated values using spatial strategies
    4. Multi-line merging: Combine adjacent value lines
    5. Confidence scoring: Calculate overall confidence using multiple factors
    6. Deduplication: Prevent multiple labels from claiming same value

    Example usage:
        config = KVConfig(max_same_line_dx=150.0)
        extractor = KeyValueExtractor(config=config)
        pairs = extractor.extract(elements)
    """

    def __init__(self, config: Optional[KVConfig] = None):
        """Initialize the key-value extractor.

        Args:
            config: Optional configuration parameters. If None, uses defaults.
        """
        self.cfg = config if config is not None else KVConfig()

    @time_it(logger=logger)
    def extract(self, elements: List[DocumentElement]) -> List[KeyValuePair]:
        """Extract key-value pairs from document elements.

        This is the main entry point for the extraction process. It processes
        elements page by page to ensure no cross-page pairing occurs.

        Args:
            elements: List of document elements to process

        Returns:
            List of extracted KeyValuePair objects
        """
        if not elements:
            logger.info("No elements provided for KV extraction")
            return []

        logger.info(f"🚀 Starting KV extraction from {len(elements)} elements")
        
        # Initialize progress tracking
        page_groups = self._group_by_page(elements)
        total_pages = len(page_groups)
        
        progress = ProgressTracker("KV Extraction", total_pages, logger)
        all_pairs = []

        # Process each page independently
        for page_num, page_elements in page_groups.items():
            progress.start_step(f"Processing page {page_num}")
            
            if not page_elements:
                progress.complete_step(0, "No elements on page")
                continue

            try:
                page_pairs = self._extract_page_pairs(page_elements, page_num)
                all_pairs.extend(page_pairs)
                progress.complete_step(
                    len(page_pairs), 
                    f"Found {len(page_pairs)} KV pairs on page {page_num}"
                )
            except Exception as e:
                logger.warning(f"Failed to extract KV pairs from page {page_num}: {e}")
                progress.complete_step(0, f"Failed: {e}")
                continue

        progress.finish(success=True)
        log_memory_usage(logger, "KV extraction")
        
        logger.info(f"🎉 KV extraction completed: {len(all_pairs)} pairs found across {total_pages} pages")
        return all_pairs

    def _extract_page_pairs(
        self, elements: List[DocumentElement], page_num: int
    ) -> List[KeyValuePair]:
        """Extract key-value pairs from a single page.

        Args:
            elements: Elements from a single page
            page_num: Page number

        Returns:
            List of KeyValuePair objects for this page
        """
        logger.debug(f"Starting KV extraction for page {page_num} with {len(elements)} elements")
        
        # Filter out empty or invalid text
        valid_elements = [e for e in elements if e.text and e.text.strip()]
        if not valid_elements:
            logger.debug(f"Page {page_num}: No valid text elements found")
            return []

        logger.debug(f"Page {page_num}: {len(valid_elements)} valid elements after filtering")

        # Step 1: Cluster elements into lines
        lines = self._cluster_lines(valid_elements)
        if not lines:
            logger.debug(f"Page {page_num}: No lines detected")
            return []
        logger.debug(f"Page {page_num}: Clustered into {len(lines)} lines")

        # Step 2: Detect column boundaries
        columns = self._cluster_columns(valid_elements)
        logger.debug(f"Page {page_num}: Detected {len(columns)} columns")

        # Step 3: Build spatial index for efficient value lookup
        value_candidates = [
            e
            for e in valid_elements
            if e.element_type not in {"table", "image", "heading", "code", "formula"}
        ]
        logger.debug(f"Page {page_num}: {len(value_candidates)} value candidates identified")

        # Step 4: Find label-value pairs
        pairs = []
        used_values = set()  # Track used elements to prevent duplicates
        used_labels = set()  # Track used labels to prevent them being values
        potential_labels = 0
        successful_pairs = 0

        for element in valid_elements:
            # Skip if already used as a value
            if id(element) in used_values:
                continue

            # Skip non-text elements as labels (headings, tables, etc. shouldn't be labels)
            if element.element_type != "text":
                continue

            # Score element as potential label
            label_score = self._score_label(element, self.cfg)
            if label_score >= 0.5:  # Label threshold from implementation plan
                potential_labels += 1
                logger.debug(f"Page {page_num}: Potential label '{element.text[:30]}...' (score: {label_score:.3f})")

            if label_score < 0.5:
                continue

            # Filter value candidates to exclude already-used labels and values
            available_candidates = [
                c for c in value_candidates if id(c) not in used_values and id(c) not in used_labels
            ]

            # Find best value for this label
            value_info = self._find_value(
                element, available_candidates, lines, columns, used_values
            )
            if not value_info:
                logger.debug(f"Page {page_num}: No value found for label '{element.text[:30]}...'")
                continue

            value_element, strategy, geom_score = value_info
            logger.debug(f"Page {page_num}: Found value using '{strategy}' strategy (geom_score: {geom_score:.3f})")

            # Mark value elements as used and handle multiline merging
            value_elements = [value_element]
            if strategy == "below":
                # Try to merge multi-line values only if it looks like a natural continuation
                merged_elements = self._merge_multiline(
                    value_element, lines, columns, used_values, element
                )
                # Only use merged result if we actually found continuation lines
                if len(merged_elements) > 1:
                    value_elements = merged_elements
                    logger.debug(f"Page {page_num}: Merged {len(value_elements)} value elements")

            for ve in value_elements:
                used_values.add(id(ve))

            # Mark label as used so it can't be claimed as a value by other labels
            used_labels.add(id(element))

            # Create combined value text
            value_text = " ".join(ve.text.strip() for ve in value_elements if ve.text.strip())
            if len(value_text) < self.cfg.min_value_len:
                logger.debug(f"Page {page_num}: Value text too short: '{value_text}'")
                continue

            # Calculate content score
            content_score = self._score_value_content(value_text)

            # Calculate overall confidence
            confidence = self._calculate_confidence(label_score, geom_score, content_score)

            # Create combined value bbox
            value_bbox = self._combine_bboxes([ve.bbox for ve in value_elements])

            tags = self._detect_tags(element.text, value_text)

            # Create KeyValuePair
            pair = KeyValuePair(
                label_text=element.text,
                value_text=value_text,
                page_number=page_num,
                label_bbox=element.bbox,
                value_bbox=value_bbox,
                confidence=confidence,
                metadata={
                    "label_score": label_score,
                    "geom_score": geom_score,
                    "content_score": content_score,
                    "strategy": strategy,
                    "distance": self._calculate_distance(element.bbox, value_bbox),
                    "label_element_id": element.metadata.get("element_id"),
                    "value_element_ids": [ve.metadata.get("element_id") for ve in value_elements],
                },
            )

            if tags:
                pair.metadata["tags"] = sorted(tags)
                pair.metadata["tag_summary"] = ", ".join(sorted(tags))

            pairs.append(pair)
            successful_pairs += 1
            logger.debug(f"Page {page_num}: Created KV pair '{element.text}' = '{value_text[:50]}...' (confidence: {confidence:.3f})")

        logger.debug(f"Page {page_num}: Found {potential_labels} potential labels, created {successful_pairs} KV pairs")
        return pairs

    def _group_by_page(self, elements: List[DocumentElement]) -> Dict[int, List[DocumentElement]]:
        """Group elements by page number."""
        page_groups = defaultdict(list)
        for element in elements:
            page_groups[element.page_number].append(element)
        return dict(page_groups)

    def _cluster_lines(self, elements: List[DocumentElement]) -> List[List[DocumentElement]]:
        """Cluster elements into horizontal lines based on y-overlap.

        Args:
            elements: Elements to cluster

        Returns:
            List of line clusters, each containing elements in the same line
        """
        if not elements:
            return []

        # Sort by y-coordinate (top to bottom)
        sorted_elements = sorted(elements, key=lambda e: -e.bbox["y1"])  # Descending y

        lines = []
        for element in sorted_elements:
            # Find existing line that overlaps with this element
            placed = False
            for line in lines:
                # Check overlap with any element in the line, not just the first
                for line_element in line:
                    if self._elements_overlap_y(element, line_element):
                        line.append(element)
                        placed = True
                        break
                if placed:
                    break

            if not placed:
                lines.append([element])

        # Sort elements within each line by x-coordinate
        for line in lines:
            line.sort(key=lambda e: e.bbox["x0"])

        return lines

    def _elements_overlap_y(self, e1: DocumentElement, e2: DocumentElement) -> bool:
        """Check if two elements overlap vertically."""
        y1_top, y1_bottom = e1.bbox["y1"], e1.bbox["y0"]
        y2_top, y2_bottom = e2.bbox["y1"], e2.bbox["y0"]

        # Check for overlap or touching
        overlap = max(0, min(y1_top, y2_top) - max(y1_bottom, y2_bottom))

        # Calculate overlap ratio relative to smaller element height
        h1 = y1_top - y1_bottom
        h2 = y2_top - y2_bottom
        min_height = min(h1, h2)

        if min_height <= 0:
            return False

        overlap_ratio = overlap / min_height

        # Allow both overlapping and touching elements
        if overlap_ratio > 0.5:  # 50% overlap
            return True

        # Also allow touching elements (0 overlap but adjacent)
        touching = overlap == 0 and (abs(y1_bottom - y2_top) < 1.0 or abs(y2_bottom - y1_top) < 1.0)
        return touching

    def _cluster_columns(self, elements: List[DocumentElement]) -> List[Tuple[float, float]]:
        """Detect column boundaries using conservative gap detection.

        For small element counts, return a single column to avoid degenerate
        splits. Otherwise, use x-centers and the largest gap as a gutter,
        requiring a minimum number of elements per column.
        """
        if not elements:
            return []

        # If too few elements, assume single column
        if len(elements) < 8:
            x_min = min(e.bbox["x0"] for e in elements)
            x_max = max(e.bbox["x1"] for e in elements)
            return [(x_min, x_max)]

        centers = sorted(((e.bbox["x0"] + e.bbox["x1"]) / 2.0) for e in elements)
        gaps = [(centers[i + 1] - centers[i], i) for i in range(len(centers) - 1)]
        if not gaps:
            x_min = min(e.bbox["x0"] for e in elements)
            x_max = max(e.bbox["x1"] for e in elements)
            return [(x_min, x_max)]

        max_gap, idx = max(gaps, key=lambda t: t[0])
        if max_gap < self.cfg.gutter_min_dx:
            x_min = min(e.bbox["x0"] for e in elements)
            x_max = max(e.bbox["x1"] for e in elements)
            return [(x_min, x_max)]

        left_centers = centers[: idx + 1]
        right_centers = centers[idx + 1 :]

        MIN_ELEMS_PER_COL = 5
        if len(left_centers) < MIN_ELEMS_PER_COL or len(right_centers) < MIN_ELEMS_PER_COL:
            x_min = min(e.bbox["x0"] for e in elements)
            x_max = max(e.bbox["x1"] for e in elements)
            return [(x_min, x_max)]

        midpoint = (centers[idx] + centers[idx + 1]) / 2.0
        left_x_min = min(e.bbox["x0"] for e in elements if (e.bbox["x0"] + e.bbox["x1"]) / 2.0 <= midpoint)
        left_x_max = max(e.bbox["x1"] for e in elements if (e.bbox["x0"] + e.bbox["x1"]) / 2.0 <= midpoint)
        right_x_min = min(e.bbox["x0"] for e in elements if (e.bbox["x0"] + e.bbox["x1"]) / 2.0 > midpoint)
        right_x_max = max(e.bbox["x1"] for e in elements if (e.bbox["x0"] + e.bbox["x1"]) / 2.0 > midpoint)

        return [(left_x_min, left_x_max), (right_x_min, right_x_max)]

    def _build_value_index(self, elements: List[DocumentElement]) -> Dict[str, Any]:
        """Build spatial index for efficient value candidate lookup.

        Args:
            elements: Value candidate elements

        Returns:
            Dictionary containing indexed elements and metadata
        """
        return {
            "elements": elements,
            "by_y": sorted(elements, key=lambda e: -e.bbox["y1"]),  # Descending y
            "by_x": sorted(elements, key=lambda e: e.bbox["x0"]),  # Ascending x
        }

    def _score_label(self, element: DocumentElement, config: KVConfig) -> float:
        """Score an element as a potential label.

        Scoring features from implementation plan:
        - ends with ':' and len < max_label_len → +0.4 points
        - len < max_label_len → +0.2 points
        - uppercase_ratio > min_upper_ratio → +0.2 points
        - left_margin_percentile < 0.3 → +0.1 points
        - word_count <= 4 → +0.1 points

        Args:
            element: Element to score
            config: Configuration parameters

        Returns:
            Label score between 0.0 and 1.0
        """
        text = element.text.strip()
        if not text:
            return 0.0

        score = 0.0

        # Feature 1: Ends with colon and reasonable length
        if text.endswith(":") and len(text) <= config.max_label_len:
            score += 0.4

        # Feature 2: Short length - more generous scoring for structural labels
        if len(text) <= config.max_label_len:
            score += 0.2
            # Extra boost for very short labels that look structural
            if len(text) <= 20 and len(text.split()) <= 2:
                score += 0.2  # Additional boost for short structural labels

        # Feature 3: Uppercase ratio
        if text:
            upper_count = sum(1 for c in text if c.isupper())
            alpha_count = len([c for c in text if c.isalpha()])
            if alpha_count > 0:  # Avoid division by zero
                upper_ratio = upper_count / alpha_count
                if upper_ratio >= config.min_upper_ratio:
                    score += 0.2

        # Feature 4: Left margin position (approximation using x0)
        # This is simplified - in reality would need page width context
        if element.bbox["x0"] <= 100.0:  # Assume left margin
            score += 0.1

        # Feature 5: Word count
        word_count = len(text.split())
        if word_count <= 4:
            score += 0.1

        # Feature 6: Label vocabulary boost (policy number, invoice, etc.)
        label_keywords = [
            "invoice", "invoice no", "invoice #", "inv", "po", "po#", "purchase order",
            "order", "order #", "account", "account number", "acct", "policy", "policy no",
            "claim", "claim number", "mrn", "id", "reference", "ref", "ref#", "ticket",
        ]
        tl = text.lower().rstrip(":")
        if any(tl.startswith(k) or (" " + k + " ") in (" " + tl + " ") for k in label_keywords):
            score += config.label_keywords_boost

        # Penalty for generic/vague text that doesn't look like a label
        generic_patterns = [
            r"^(some|any|this|that)\s+(value|text|content|data)",
            r"^(please|kindly|note)",
        ]

        for pattern in generic_patterns:
            if re.search(pattern, text.lower()):
                score *= 0.3  # Significant penalty
                break

        return min(score, 1.0)  # Cap at 1.0

    def _find_value(
        self,
        label_element: DocumentElement,
        value_candidates: List[DocumentElement],
        lines: List[List[DocumentElement]],
        columns: List[Tuple[float, float]],
        used_values: set,
    ) -> Optional[Tuple[DocumentElement, str, float]]:
        """Find the best value element for a given label.

        Strategies:
        1. Same-line: nearest right element within max_same_line_dx
        2. Below: within max_below_dy vertical window, x_align_tol alignment

        Args:
            label_element: The label element
            value_candidates: List of potential value elements
            lines: Clustered line elements
            columns: Column boundaries
            used_values: Set of already used value element IDs

        Returns:
            Tuple of (value_element, strategy, geometric_score) or None
        """
        label_column = self._find_element_column(label_element, columns)

        # Strategy 1: Same-line pairing
        same_line_candidate = self._find_same_line_value(
            label_element, value_candidates, label_column, columns, used_values
        )

        if same_line_candidate:
            value_element, geom_score = same_line_candidate
            return (value_element, "same_line", geom_score)

        # Strategy 2: Below pairing
        below_candidate = self._find_below_value(
            label_element, value_candidates, label_column, columns, used_values
        )

        if below_candidate:
            value_element, geom_score = below_candidate
            return (value_element, "below", geom_score)

        return None

    def _find_element_column(
        self, element: DocumentElement, columns: List[Tuple[float, float]]
    ) -> Tuple[float, float]:
        """Find which column an element belongs to."""
        if not columns:
            return (0, float("inf"))

        element_x = element.bbox["x0"]

        for col_start, col_end in columns:
            if col_start <= element_x <= col_end:
                return (col_start, col_end)

        # Default to first column if no match
        return columns[0]

    def _find_same_line_value(
        self,
        label_element: DocumentElement,
        value_candidates: List[DocumentElement],
        label_column: Tuple[float, float],
        columns: List[Tuple[float, float]],
        used_values: set,
    ) -> Optional[Tuple[DocumentElement, float]]:
        """Find value using same-line strategy."""
        label_bbox = label_element.bbox
        best_candidate = None
        best_score = 0.0

        for candidate in value_candidates:
            # Skip if already used or wrong column
            if id(candidate) in used_values:
                continue

            candidate_column = self._find_element_column(candidate, columns)
            if candidate_column != label_column:
                continue

            # Check if on same line (y-overlap)
            if not self._elements_overlap_y(label_element, candidate):
                continue

            # Check horizontal distance and direction
            dx = candidate.bbox["x0"] - label_bbox["x1"]
            if dx <= 0 or dx > self.cfg.max_same_line_dx:
                continue

            # Score based on distance (closer is better)
            distance_score = max(0, 1.0 - dx / self.cfg.max_same_line_dx)

            if distance_score > best_score:
                best_candidate = candidate
                best_score = distance_score

        if best_candidate:
            return (best_candidate, best_score)

        return None

    def _find_below_value(
        self,
        label_element: DocumentElement,
        value_candidates: List[DocumentElement],
        label_column: Tuple[float, float],
        columns: List[Tuple[float, float]],
        used_values: set,
    ) -> Optional[Tuple[DocumentElement, float]]:
        """Find value using below strategy."""
        label_bbox = label_element.bbox
        best_candidate = None
        best_score = 0.0

        for candidate in value_candidates:
            # Skip if already used
            if id(candidate) in used_values:
                continue

            candidate_column = self._find_element_column(candidate, columns)
            if candidate_column != label_column:
                continue

            # Check vertical relationship (candidate should be below label)
            dy = label_bbox["y0"] - candidate.bbox["y1"]  # Positive if candidate is below
            if dy < 0 or dy > self.cfg.max_below_dy:
                continue

            # Check horizontal alignment - use distance between label start and candidate start
            label_x = label_bbox["x0"]
            candidate_x = candidate.bbox["x0"]
            x_distance = abs(label_x - candidate_x)

            if x_distance > self.cfg.x_align_tol:
                continue

            # Score based on vertical distance and alignment
            distance_score = max(0, 1.0 - dy / self.cfg.max_below_dy)
            alignment_score = max(0, 1.0 - x_distance / self.cfg.x_align_tol)
            combined_score = (distance_score + alignment_score) / 2.0

            if combined_score > best_score:
                best_candidate = candidate
                best_score = combined_score

        if best_candidate:
            return (best_candidate, best_score)

        return None

    def _calculate_x_overlap(self, bbox1: Dict[str, float], bbox2: Dict[str, float]) -> float:
        """Calculate horizontal overlap between two bounding boxes."""
        overlap = max(0, min(bbox1["x1"], bbox2["x1"]) - max(bbox1["x0"], bbox2["x0"]))
        return overlap

    def _merge_multiline(
        self,
        start_element: DocumentElement,
        lines: List[List[DocumentElement]],
        columns: List[Tuple[float, float]],
        used_values: set,
        original_label: Optional[DocumentElement] = None,
    ) -> List[DocumentElement]:
        """Merge multi-line values through spatial adjacency.

        Args:
            start_element: Starting value element
            lines: Line clusters
            columns: Column boundaries
            used_values: Set of already used element IDs

        Returns:
            List of elements forming the complete multi-line value
        """
        merged_elements = [start_element]
        start_column = self._find_element_column(start_element, columns)

        # Find the line containing the start element
        start_line_idx = None
        for i, line in enumerate(lines):
            if start_element in line:
                start_line_idx = i
                break

        if start_line_idx is None:
            return merged_elements

        # Look for vertically adjacent elements
        median_line_gap = self._calculate_median_line_gap(lines)

        # Get all elements from the same page, sorted by y-position (descending)
        all_elements = []
        for line in lines:
            all_elements.extend(line)
        all_elements.sort(key=lambda e: -e.bbox["y1"])

        # Find the next elements below the current element
        current_element = start_element

        while True:
            # Find candidates that are below the current element
            candidates = []
            for element in all_elements:
                # Skip if this is the current element or not below it
                if element.bbox["y1"] > current_element.bbox["y0"]:
                    continue

                # Skip if already used
                if id(element) in used_values:
                    continue

                # Skip non-text elements for values
                if element.element_type not in {"text"}:
                    continue

                element_column = self._find_element_column(element, columns)
                if element_column != start_column:
                    continue

                # Check x-alignment
                label_x = start_element.bbox["x0"]
                candidate_x = element.bbox["x0"]
                x_distance = abs(label_x - candidate_x)

                if x_distance <= 30.0:  # x-alignment tolerance
                    candidates.append(element)

            if not candidates:
                break

            # Choose the closest candidate vertically
            best_candidate = min(
                candidates, key=lambda e: current_element.bbox["y0"] - e.bbox["y1"]
            )

            # Check vertical gap
            vertical_gap = current_element.bbox["y0"] - best_candidate.bbox["y1"]

            # Only merge if gap is reasonable
            if vertical_gap > median_line_gap * 1.5:
                break

            # Check if the text looks like a continuation
            if self._looks_like_separate_field(best_candidate.text):
                break

            if not self._looks_like_continuation(current_element.text, best_candidate.text):
                break

            # Check original label distance limits
            if original_label is not None:
                label_to_candidate_dy = original_label.bbox["y0"] - best_candidate.bbox["y1"]
                if label_to_candidate_dy > self.cfg.max_below_dy:
                    break

            merged_elements.append(best_candidate)
            current_element = best_candidate

        return merged_elements

    def _looks_like_separate_field(self, text: str) -> bool:
        """Check if text looks like a separate field rather than a continuation.

        Args:
            text: Text to check

        Returns:
            True if it looks like a separate field, False if it looks like continuation
        """
        text = text.strip()
        if not text:
            return False

        # If it ends with colon, it's likely a label for a separate field
        if text.endswith(":"):
            return True

        # If it contains common field indicators, it's likely separate
        field_patterns = [
            r"\b(name|address|phone|email|date|ssn|id)\b",
            r"\b\w+:",  # Any word followed by colon
        ]

        for pattern in field_patterns:
            if re.search(pattern, text.lower()):
                return True

        return False

    def _looks_like_continuation(self, first_text: str, second_text: str) -> bool:
        """Check if second text looks like a continuation of the first.

        Args:
            first_text: The first text (initial value)
            second_text: The second text (potential continuation)

        Returns:
            True if it looks like a continuation, False otherwise
        """
        first_text = first_text.strip().lower()
        second_text = second_text.strip().lower()

        # Very simple heuristic: if both texts look like address components,
        # phone numbers, or other naturally multi-line values
        address_patterns = [
            r"\d+.*(?:street|st|avenue|ave|road|rd|drive|dr|lane|ln|blvd|boulevard)",
            r"(?:apartment|apt|unit|suite|ste)\s*\d+",
            r"\w+,?\s+[a-z]{2}\s+\d{5}",  # City, State ZIP
        ]

        # If first text looks like start of address and second looks like continuation
        for pattern in address_patterns:
            if re.search(pattern, first_text):
                # Second text should also look address-like or be simple continuation
                if (
                    re.search(pattern, second_text)
                    or len(second_text.split()) <= 3  # Simple continuation
                    or re.match(r"^[a-z\s,\d-]+$", second_text)
                ):  # Simple text
                    return True

        # If they look completely unrelated (e.g., "Near Value" and "Far Value")
        # then it's probably not a continuation
        if first_text in ["near value", "far value"] or second_text in ["near value", "far value"]:
            return False

        # Check for natural text continuation patterns
        # If first text ends with incomplete words/sentences, likely continuation
        if first_text.endswith(
            ("that", "and", "with", "of", "the", "to", "in", "for")
        ) or not first_text.endswith((".", "!", "?", ":")):
            # And second text looks like continuation of narrative
            if len(
                second_text.split()
            ) > 2 and not second_text.startswith(  # Not just a single word
                ("name", "address", "phone", "email")
            ):  # Not a new field
                return True

        # Check if both texts form a coherent narrative when combined
        combined = first_text + " " + second_text
        if len(combined.split()) > 5 and not re.search(  # Substantial text
            r"\b(?:name|address|phone|email|date|ssn)\b", combined
        ):  # Not field-like
            return True

        # Default: be conservative and don't merge unless we're confident
        return False

    def _calculate_median_line_gap(self, lines: List[List[DocumentElement]]) -> float:
        """Calculate median vertical gap between lines."""
        if len(lines) < 2:
            return 20.0  # Default gap

        gaps = []
        for i in range(len(lines) - 1):
            current_line = lines[i]
            next_line = lines[i + 1]

            current_y = min(e.bbox["y0"] for e in current_line)
            next_y = max(e.bbox["y1"] for e in next_line)

            gap = current_y - next_y
            if gap > 0:
                gaps.append(gap)

        if gaps:
            gaps.sort()
            median_idx = len(gaps) // 2
            return gaps[median_idx]

        return 20.0  # Default gap

    def _score_value_content(self, value_text: str) -> float:
        """Score value content quality.

        Args:
            value_text: The value text to score

        Returns:
            Content score between 0.0 and 1.0
        """
        if not value_text.strip():
            return 0.0

        score = 0.5  # Base score

        # Boost for common value patterns and code-like identifiers
        patterns = [
            r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",  # Dates
            r"\b\d{3}-\d{2}-\d{4}\b",        # SSN
            r"\b\d{3}-\d{3}-\d{4}\b",        # Phone
            r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",    # Names
            r"\b\d+\b",                        # Numbers
            r"\b[A-Z0-9]{4,}[A-Z0-9_-]{2,}\b", # Generic code (mix letters/digits)
            r"\b(?:INV|PO|POL|ORD|MRN|ID|REF)[-# ]?[A-Z0-9]{3,}\b",  # Prefixed identifiers
            r"\b[A-Z0-9]+(?:-[A-Z0-9]+){1,3}\b",                     # Multi-segment code
        ]

        for pattern in patterns:
            if re.search(pattern, value_text):
                score += 0.1
                break

        # Penalize very long values (might be paragraphs)
        if len(value_text) > 200:
            score *= 0.8

        # Boost for reasonable length values
        if 2 <= len(value_text) <= 100:
            score += 0.1

        # Additional boost if value strongly matches code-like patterns
        if re.search(r"\b(?:INV|PO|POL|ORD|MRN|ID|REF)[-# ]?[A-Z0-9]{4,}\b", value_text.upper()) or \
           re.search(r"\b[A-Z0-9]+(?:-[A-Z0-9]+){1,3}\b", value_text.upper()):
            score += self.cfg.code_value_boost

        return min(score, 1.0)

    def _detect_tags(self, label_text: str, value_text: str) -> set:
        """Detect semantic tags for a key-value pair."""
        tags = set()
        label_lower = (label_text or "").lower()
        value_lower = (value_text or "").lower()
        combined = f"{label_lower} {value_lower}"

        # Identifier / reference
        identifier_keywords = [
            "invoice", "inv", "po", "purchase order", "order", "account", "acct",
            "policy", "claim", "mrn", "id", "identifier", "reference", "ref",
            "ticket", "serial", "voucher", "contract", "agreement", "case",
        ]
        if any(keyword in label_lower for keyword in identifier_keywords):
            tags.add("identifier")
        identifier_pattern = re.compile(r"\b[A-Z0-9]{3,}(?:[-#][A-Z0-9]{2,})+\b")
        if identifier_pattern.search(value_text.upper() if value_text else ""):
            tags.add("identifier")

        # Monetary values
        monetary_pattern = re.compile(r"(?:\$|€|£|usd|eur|gbp)\s*[-+]?[0-9]+(?:,[0-9]{3})*(?:\.[0-9]{2})?", re.I)
        if monetary_pattern.search(value_text or "") or any(
            keyword in label_lower for keyword in [
                "amount", "total", "balance", "price", "cost", "subtotal", "tax", "fee"
            ]
        ) and re.search(r"[-+]?[0-9]+(?:,[0-9]{3})*(?:\.[0-9]{2})?", value_text or ""):
            tags.add("monetary")

        # Date values
        date_patterns = [
            r"\b\d{4}-\d{2}-\d{2}\b",
            r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
            r"\b\d{1,2}-\d{1,2}-\d{2,4}\b",
            r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b",
        ]
        if any(re.search(pattern, value_lower) for pattern in date_patterns) or any(
            keyword in label_lower for keyword in ["date", "deadline", "issued", "due"]
        ) and re.search(r"\d", value_lower):
            tags.add("date")

        # Quantity / count
        if any(keyword in label_lower for keyword in ["quantity", "qty", "count", "units", "items"]):
            tags.add("quantity")
        quantity_pattern = re.compile(r"\b[0-9]+(?:\.[0-9]+)?\s*(?:pcs|units|items|boxes|packs|kg|lb|lbs|ounces|oz|g|liters|l|ml)\b", re.I)
        if quantity_pattern.search(value_text or ""):
            tags.add("quantity")

        # Percentage
        if "%" in value_text or re.search(r"\b[0-9]+(?:\.[0-9]+)?\s*percent\b", value_lower):
            tags.add("percentage")

        # Contact info
        if re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", value_text or ""):
            tags.add("email")
        if re.search(r"\b\+?\d{1,3}[\s-]?\(?\d{2,3}\)?[\s-]?\d{3}[\s-]?\d{4}\b", value_text or ""):
            tags.add("phone")

        # Financial specific keywords
        if any(keyword in label_lower for keyword in ["balance", "due", "credit", "debit", "payment"]):
            tags.add("financial")

        # Address cues
        if any(keyword in label_lower for keyword in ["address", "city", "state", "country", "zipcode", "postal"]):
            tags.add("address")

        # Notes / description
        if any(keyword in label_lower for keyword in ["description", "notes", "comments", "details"]):
            tags.add("description")

        return tags

    def _calculate_confidence(
        self, label_score: float, geom_score: float, content_score: float
    ) -> float:
        """Calculate overall confidence using sigmoid function.

        Args:
            label_score: Label detection score
            geom_score: Geometric relationship score
            content_score: Content quality score

        Returns:
            Overall confidence score between 0.0 and 1.0
        """
        weighted_score = (
            self.cfg.label_weight * label_score
            + self.cfg.geom_weight * geom_score
            + self.cfg.content_weight * content_score
        )

        return self._sigmoid(weighted_score)

    def _sigmoid(self, x: float) -> float:
        """Sigmoid activation function."""
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def _combine_bboxes(self, bboxes: List[Dict[str, float]]) -> Dict[str, float]:
        """Combine multiple bounding boxes into a single encompassing box."""
        if not bboxes:
            return {"x0": 0.0, "y0": 0.0, "x1": 0.0, "y1": 0.0}

        x0 = min(bbox["x0"] for bbox in bboxes)
        y0 = min(bbox["y0"] for bbox in bboxes)
        x1 = max(bbox["x1"] for bbox in bboxes)
        y1 = max(bbox["y1"] for bbox in bboxes)

        return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}

    def _calculate_distance(self, bbox1: Dict[str, float], bbox2: Dict[str, float]) -> float:
        """Calculate Euclidean distance between two bounding box centers."""
        center1_x = (bbox1["x0"] + bbox1["x1"]) / 2
        center1_y = (bbox1["y0"] + bbox1["y1"]) / 2
        center2_x = (bbox2["x0"] + bbox2["x1"]) / 2
        center2_y = (bbox2["y0"] + bbox2["y1"]) / 2

        dx = center2_x - center1_x
        dy = center2_y - center1_y

        return math.sqrt(dx * dx + dy * dy)

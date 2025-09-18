"""
PDF verification rendering engine with coordinate transformation capabilities.
Provides visual overlays for parsed elements to enable interactive verification.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import logging
from PIL import Image, ImageDraw, ImageFont

from src.core.models import DocumentElement, KeyValuePair
from src.core.classifiers.header_classifier import is_code_like
from src.utils.logging_config import time_it

# Initialize logger for renderer
logger = logging.getLogger(__name__)


@dataclass
class RenderConfig:
    """Configuration for verification rendering."""

    highlight_color: Tuple[int, int, int, int] = (255, 255, 0, 128)  # Yellow with alpha
    border_color: Tuple[int, int, int] = (255, 0, 0)  # Red border
    border_width: int = 2
    font_size: int = 12
    show_confidence: bool = True
    show_element_type: bool = True
    page_scale: float = 1.0


class CoordinateTransformer:
    """Transforms coordinates between different coordinate systems."""

    @staticmethod
    def calculate_scaling(
        pdf_size: Tuple[float, float], image_size: Tuple[int, int]
    ) -> Tuple[float, float]:
        """Calculate scaling factors from PDF to image coordinates.

        Args:
            pdf_size: PDF page dimensions (width, height) in points
            image_size: Image dimensions (width, height) in pixels

        Returns:
            Tuple of (scale_x, scale_y) factors
        """
        pdf_width, pdf_height = pdf_size
        image_width, image_height = image_size

        scale_x = image_width / pdf_width if pdf_width > 0 else 1.0
        scale_y = image_height / pdf_height if pdf_height > 0 else 1.0

        return scale_x, scale_y

    @staticmethod
    def transform_bbox(
        bbox: Dict[str, float], pdf_size: Tuple[float, float], image_size: Tuple[int, int]
    ) -> Dict[str, int]:
        """Transform bounding box from PDF to image coordinates.

        Args:
            bbox: PDF bounding box with keys 'x0', 'y0', 'x1', 'y1'
            pdf_size: PDF page dimensions (width, height)
            image_size: Image dimensions (width, height)

        Returns:
            Transformed bounding box with integer pixel coordinates
        """
        scale_x, scale_y = CoordinateTransformer.calculate_scaling(pdf_size, image_size)
        image_width, image_height = image_size

        # Transform and clamp to image bounds
        x0 = max(0, min(image_width, int(bbox["x0"] * scale_x)))
        y0 = max(0, min(image_height, int(bbox["y0"] * scale_y)))
        x1 = max(0, min(image_width, int(bbox["x1"] * scale_x)))
        y1 = max(0, min(image_height, int(bbox["y1"] * scale_y)))

        # Ensure x1 >= x0 and y1 >= y0
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0

        return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}

    @staticmethod
    def docling_to_pixel(
        bbox: Dict[str, float],
        page_width: float,
        page_height: float,
        image_width: int,
        image_height: int,
    ) -> Tuple[int, int, int, int]:
        """Transform Docling coordinates to pixel coordinates.

        Docling uses: (0,0) at bottom-left, normalized to page size
        Pixel uses: (0,0) at top-left, absolute pixel values

        Args:
            bbox: Docling bounding box
            page_width: PDF page width
            page_height: PDF page height
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            Tuple of (x0, y0, x1, y1) in pixel coordinates
        """
        # Scale to image dimensions
        x0 = int((bbox["x0"] / page_width) * image_width)
        x1 = int((bbox["x1"] / page_width) * image_width)

        # Flip Y-axis and scale
        y0 = int(((page_height - bbox["y1"]) / page_height) * image_height)
        y1 = int(((page_height - bbox["y0"]) / page_height) * image_height)

        # Clamp to bounds and ensure valid ordering
        x0 = max(0, min(image_width, x0))
        x1 = max(0, min(image_width, x1))
        y0 = max(0, min(image_height, y0))
        y1 = max(0, min(image_height, y1))

        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0

        return (x0, y0, x1, y1)

    @staticmethod
    def pixel_to_docling(
        pixel_bbox: Tuple[int, int, int, int],
        page_width: float,
        page_height: float,
        image_width: int,
        image_height: int,
    ) -> Dict[str, float]:
        """Transform pixel coordinates back to Docling format.

        Args:
            pixel_bbox: Pixel coordinates (x0, y0, x1, y1)
            page_width: PDF page width
            page_height: PDF page height
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            Docling format bounding box
        """
        x0, y0, x1, y1 = pixel_bbox

        # Scale to page dimensions
        doc_x0 = (x0 / image_width) * page_width if image_width > 0 else 0
        doc_x1 = (x1 / image_width) * page_width if image_width > 0 else 0

        # Flip Y-axis and scale
        doc_y0 = page_height - ((y1 / image_height) * page_height) if image_height > 0 else 0
        doc_y1 = page_height - ((y0 / image_height) * page_height) if image_height > 0 else 0

        return {"x0": doc_x0, "y0": doc_y0, "x1": doc_x1, "y1": doc_y1}

    @staticmethod
    def validate_bbox(bbox: Dict[str, float]) -> bool:
        """Validate bounding box coordinates.

        Args:
            bbox: Bounding box to validate

        Returns:
            True if valid, False otherwise
        """
        required_keys = {"x0", "y0", "x1", "y1"}
        if not all(key in bbox for key in required_keys):
            return False

        try:
            return (
                bbox["x0"] <= bbox["x1"]
                and bbox["y0"] <= bbox["y1"]
                and all(isinstance(v, (int, float)) for v in bbox.values())
            )
        except (TypeError, KeyError):
            return False


class PDFRenderer:
    """Renders PDF elements with visual overlays for verification."""

    def __init__(self, pdf_parser=None, config: Optional[RenderConfig] = None):
        """Initialize PDFRenderer.

        Args:
            pdf_parser: Parser instance for accessing page images
            config: Rendering configuration
        """
        self.pdf_parser = pdf_parser
        self.config = config or RenderConfig()
        self.coordinate_transformer = CoordinateTransformer()
        self._font_cache = {}

    def get_page_image(self, page_number: int) -> Image.Image:
        """Get page image from PDF parser.

        Args:
            page_number: Page number (1-indexed)

        Returns:
            PIL Image of the page

        Raises:
            ValueError: If invalid page number or parser not available
        """
        if not self.pdf_parser:
            raise ValueError("PDF parser not available")

        if page_number < 1:
            raise ValueError(f"Invalid page number: {page_number}")

        return self.pdf_parser.get_page_image(page_number)

    def transform_coordinates(
        self,
        bbox: Dict[str, float],
        page_image: Image.Image,
        pdf_size: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, int]:
        """Transform coordinates from PDF to image space.

        Args:
            bbox: PDF bounding box
            page_image: Page image for size reference
            pdf_size: Optional PDF page size, defaults to image size

        Returns:
            Transformed coordinates in image space
        """
        if pdf_size is None:
            # Assume PDF size matches image size if not provided
            pdf_size = page_image.size

        return self.coordinate_transformer.transform_bbox(bbox, pdf_size, page_image.size)

    @time_it(logger=logger)
    def render_highlight(
        self,
        page_image: Image.Image,
        bbox: Dict[str, float],
        color: Optional[Tuple[int, int, int, int]] = None,
        thickness: Optional[int] = None,
        pdf_size: Optional[Tuple[float, float]] = None,
    ) -> Image.Image:
        """Render a single highlight rectangle on image.

        Args:
            page_image: Base page image
            bbox: Bounding box to highlight
            color: RGBA color tuple, defaults to config color
            thickness: Border thickness, defaults to config thickness
            pdf_size: PDF page dimensions

        Returns:
            Image with highlight overlay
        """
        if color is None:
            color = self.config.highlight_color
        if thickness is None:
            thickness = self.config.border_width

        # Convert to RGBA for transparency support
        result_image = page_image.convert("RGBA")

        # Create overlay
        overlay = Image.new("RGBA", result_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Transform coordinates
        pixel_coords = self.transform_coordinates(bbox, page_image, pdf_size)
        rect_coords = (
            pixel_coords["x0"],
            pixel_coords["y0"],
            pixel_coords["x1"],
            pixel_coords["y1"],
        )

        # Draw filled rectangle with transparency
        draw.rectangle(rect_coords, fill=color, outline=color[:3], width=thickness)

        # Composite overlay onto base image
        result_image = Image.alpha_composite(result_image, overlay)

        return result_image.convert("RGB")

    @time_it(logger=logger)
    def render_multiple_highlights(
        self,
        page_image: Image.Image,
        bboxes: List[Dict[str, float]],
        colors: Optional[List[Tuple[int, int, int, int]]] = None,
        thickness: Optional[int] = None,
        pdf_size: Optional[Tuple[float, float]] = None,
    ) -> Image.Image:
        """Render multiple highlight rectangles on image.

        Args:
            page_image: Base page image
            bboxes: List of bounding boxes to highlight
            colors: List of colors for each bbox, defaults to config color
            thickness: Border thickness
            pdf_size: PDF page dimensions

        Returns:
            Image with all highlights overlaid
        """
        if not bboxes:
            return page_image

        if colors is None:
            colors = [self.config.highlight_color] * len(bboxes)
        elif len(colors) < len(bboxes):
            # Extend colors list if needed
            colors.extend([self.config.highlight_color] * (len(bboxes) - len(colors)))

        if thickness is None:
            thickness = self.config.border_width

        # Convert to RGBA
        result_image = page_image.convert("RGBA")

        # Create single overlay for all highlights
        overlay = Image.new("RGBA", result_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Draw all highlights on overlay
        for bbox, color in zip(bboxes, colors):
            pixel_coords = self.transform_coordinates(bbox, page_image, pdf_size)
            rect_coords = (
                pixel_coords["x0"],
                pixel_coords["y0"],
                pixel_coords["x1"],
                pixel_coords["y1"],
            )

            draw.rectangle(rect_coords, fill=color, outline=color[:3], width=thickness)

        # Composite overlay onto base image
        result_image = Image.alpha_composite(result_image, overlay)

        return result_image.convert("RGB")

    def render_page_with_overlays(
        self,
        page_image: Image.Image,
        elements: List[DocumentElement],
        page_number: int,
        pdf_size: Optional[Tuple[float, float]] = None,
    ) -> Image.Image:
        """Render a page with element overlays for verification.

        Args:
            page_image: PIL Image of the PDF page
            elements: List of parsed elements for this page
            page_number: Page number (1-indexed)
            pdf_size: PDF page dimensions

        Returns:
            PIL Image with overlays rendered
        """
        # Filter elements for this page
        page_elements = [e for e in elements if e.page_number == page_number]

        if not page_elements:
            return page_image

        # Create overlay canvas
        result_image = page_image.convert("RGBA")
        overlay = Image.new("RGBA", result_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Render each element
        for element in page_elements:
            self._render_element_overlay(draw, element, result_image.size, pdf_size)

        # Composite overlay onto page image
        result_image = Image.alpha_composite(result_image, overlay)

        return result_image.convert("RGB")

    def _render_element_overlay(
        self,
        draw: Any,
        element: DocumentElement,
        image_size: Tuple[int, int],
        pdf_size: Optional[Tuple[float, float]] = None,
    ):
        """Render overlay for a single element.

        Args:
            draw: ImageDraw instance
            element: Document element to render
            image_size: Image dimensions
            pdf_size: PDF page dimensions
        """
        # Transform coordinates
        if pdf_size:
            # Use PDF size for accurate transformation
            scale_x, scale_y = self.coordinate_transformer.calculate_scaling(pdf_size, image_size)
            pixel_bbox = {
                "x0": int(element.bbox["x0"] * scale_x),
                "y0": int(element.bbox["y0"] * scale_y),
                "x1": int(element.bbox["x1"] * scale_x),
                "y1": int(element.bbox["y1"] * scale_y),
            }
        else:
            # Assume normalized coordinates
            pixel_bbox = self._normalize_to_pixel_coords(element.bbox, image_size)

        rect_coords = (pixel_bbox["x0"], pixel_bbox["y0"], pixel_bbox["x1"], pixel_bbox["y1"])

        # Draw bounding box
        draw.rectangle(
            rect_coords,
            fill=self.config.highlight_color,
            outline=self.config.border_color,
            width=self.config.border_width,
        )

        # Add element type and confidence labels
        if self.config.show_element_type or self.config.show_confidence:
            self._draw_element_label(draw, element, rect_coords)

    def _normalize_to_pixel_coords(
        self, bbox: Dict[str, float], image_size: Tuple[int, int]
    ) -> Dict[str, int]:
        """Convert normalized bbox to pixel coordinates.

        Args:
            bbox: Normalized bounding box (0-1 range)
            image_size: Image dimensions

        Returns:
            Pixel coordinates
        """
        width, height = image_size
        return {
            "x0": max(0, min(width, int(bbox["x0"] * width))),
            "y0": max(0, min(height, int(bbox["y0"] * height))),
            "x1": max(0, min(width, int(bbox["x1"] * width))),
            "y1": max(0, min(height, int(bbox["y1"] * height))),
        }

    def _draw_element_label(
        self, draw: Any, element: DocumentElement, bbox: Tuple[int, int, int, int]
    ):
        """Draw element type and confidence label.

        Args:
            draw: ImageDraw instance
            element: Document element
            bbox: Pixel bounding box coordinates
        """
        labels = []
        if self.config.show_element_type:
            labels.append(element.element_type.upper())
        if self.config.show_confidence:
            labels.append(f"{element.confidence:.2f}")

        if labels:
            label_text = " | ".join(labels)
            font = self._get_font(self.config.font_size)

            # Position label at top-left of bounding box
            label_pos = (bbox[0] + 2, max(0, bbox[1] - 20))

            # Draw text with background for readability
            try:
                # Get text size for background
                if hasattr(draw, "textbbox"):
                    # PIL >= 8.0.0
                    text_bbox = draw.textbbox(label_pos, label_text, font=font)
                    background_coords = (
                        text_bbox[0] - 2,
                        text_bbox[1] - 1,
                        text_bbox[2] + 2,
                        text_bbox[3] + 1,
                    )
                else:
                    # Fallback for older PIL versions
                    text_size = draw.textsize(label_text, font=font)
                    background_coords = (
                        label_pos[0] - 2,
                        label_pos[1] - 1,
                        label_pos[0] + text_size[0] + 2,
                        label_pos[1] + text_size[1] + 1,
                    )

                # Draw background
                draw.rectangle(background_coords, fill=(255, 255, 255, 200))

                # Draw text
                draw.text(label_pos, label_text, fill=(0, 0, 0), font=font)

            except Exception:
                # Fallback: just draw text without background
                draw.text(label_pos, label_text, fill=(0, 0, 0), font=font)

    def _get_font(self, size: int) -> ImageFont.ImageFont:
        """Get cached font instance.

        Args:
            size: Font size

        Returns:
            ImageFont instance
        """
        if size not in self._font_cache:
            try:
                # Try to load a system font
                font_paths = [
                    "/System/Library/Fonts/Arial.ttf",  # macOS
                    "/Windows/Fonts/arial.ttf",  # Windows
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
                    "arial.ttf",  # Generic fallback
                ]

                font = None
                for font_path in font_paths:
                    try:
                        font = ImageFont.truetype(font_path, size)
                        break
                    except (OSError, IOError):
                        continue

                if font is None:
                    font = ImageFont.load_default()

                self._font_cache[size] = font

            except Exception:
                self._font_cache[size] = ImageFont.load_default()

        return self._font_cache[size]

    def render_kv_pair(
        self,
        page_image: Image.Image,
        kv: KeyValuePair,
        colors: Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]] = None,
        pdf_size: Optional[Tuple[float, float]] = None,
        highlight_codes: bool = True,
    ) -> Image.Image:
        """Render a single key-value pair with connection line.

        Args:
            page_image: Base page image
            kv: KeyValuePair to render
            colors: Tuple of (label_color, value_color) with RGBA values
            pdf_size: PDF page dimensions for coordinate transformation

        Returns:
            Image with KV pair rendered
        """
        result_image = page_image.convert("RGBA")
        overlay = Image.new("RGBA", result_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        default_colors = ((255, 0, 0, 128), (0, 128, 0, 128))
        label_color, value_color = colors if colors is not None else default_colors

        is_code = bool(highlight_codes and getattr(kv, "value_text", "") and is_code_like(kv.value_text))
        if is_code:
            label_color = (110, 38, 204, 190)
            value_color = (0, 186, 255, 200)

        # Transform coordinates to pixel space
        if pdf_size:
            scale_x, scale_y = self.coordinate_transformer.calculate_scaling(
                pdf_size, result_image.size
            )

            # Transform label bbox
            label_bbox = {
                "x0": int(kv.label_bbox["x0"] * scale_x),
                "y0": int(kv.label_bbox["y0"] * scale_y),
                "x1": int(kv.label_bbox["x1"] * scale_x),
                "y1": int(kv.label_bbox["y1"] * scale_y),
            }

            # Transform value bbox
            value_bbox = {
                "x0": int(kv.value_bbox["x0"] * scale_x),
                "y0": int(kv.value_bbox["y0"] * scale_y),
                "x1": int(kv.value_bbox["x1"] * scale_x),
                "y1": int(kv.value_bbox["y1"] * scale_y),
            }
        else:
            label_bbox = self._normalize_to_pixel_coords(kv.label_bbox, result_image.size)
            value_bbox = self._normalize_to_pixel_coords(kv.value_bbox, result_image.size)

        # Draw label rectangle
        label_coords = (label_bbox["x0"], label_bbox["y0"], label_bbox["x1"], label_bbox["y1"])
        draw.rectangle(label_coords, fill=label_color, outline=label_color[:3], width=2)

        # Draw value rectangle
        value_coords = (value_bbox["x0"], value_bbox["y0"], value_bbox["x1"], value_bbox["y1"])
        draw.rectangle(value_coords, fill=value_color, outline=value_color[:3], width=2)

        if is_code:
            font = self._get_font(self.config.font_size)
            badge_text = "🆔 Code"
            badge_pos = (value_bbox["x0"], max(value_bbox["y0"] - (self.config.font_size + 4), 0))
            self._draw_text_with_background(draw, badge_text, badge_pos, font, (0, 0, 0, 180))

        # Draw connection line between label and value
        self._draw_connection_line(
            draw, label_coords, value_coords, color=(100, 100, 100, 180), width=2
        )

        # Add confidence score label
        if self.config.show_confidence and kv.confidence:
            conf_text = f"Conf: {kv.confidence:.2f}"
            font = self._get_font(self.config.font_size - 2)

            # Position confidence label near the value
            conf_pos = (value_bbox["x0"], value_bbox["y1"] + 2)
            self._draw_text_with_background(draw, conf_text, conf_pos, font, (255, 255, 255, 200))

        # Composite overlay
        result_image = Image.alpha_composite(result_image, overlay)
        return result_image.convert("RGB")

    def render_kv_pairs(
        self,
        page_image: Image.Image,
        kvs: List[KeyValuePair],
        palette: Optional[List[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]]] = None,
        pdf_size: Optional[Tuple[float, float]] = None,
        highlight_codes: bool = True,
    ) -> Image.Image:
        """Render multiple key-value pairs on a page image.

        Args:
            page_image: Base page image
            kvs: List of KeyValuePair objects to render
            palette: List of color pairs for labels and values. Cycles through if fewer than pairs.
            pdf_size: PDF page dimensions for coordinate transformation

        Returns:
            Image with all KV pairs rendered
        """
        if not kvs:
            return page_image

        # Default color palette for multiple KV pairs
        if palette is None:
            palette = [
                ((255, 0, 0, 128), (0, 128, 0, 128)),  # Red label, Green value
                ((0, 0, 255, 128), (255, 165, 0, 128)),  # Blue label, Orange value
                ((128, 0, 128, 128), (255, 255, 0, 128)),  # Purple label, Yellow value
                ((0, 128, 128, 128), (255, 192, 203, 128)),  # Teal label, Pink value
                ((165, 42, 42, 128), (0, 255, 255, 128)),  # Brown label, Cyan value
            ]

        result_image = page_image

        # Render each KV pair with cycling colors
        for i, kv in enumerate(kvs):
            color_pair = palette[i % len(palette)]
            result_image = self.render_kv_pair(
                result_image,
                kv,
                colors=color_pair,
                pdf_size=pdf_size,
                highlight_codes=highlight_codes,
            )

        return result_image

    def _draw_connection_line(
        self,
        draw,
        label_coords: Tuple[int, int, int, int],
        value_coords: Tuple[int, int, int, int],
        color: Tuple[int, int, int, int] = (100, 100, 100, 180),
        width: int = 2,
    ):
        """Draw a connection line between label and value rectangles.

        Args:
            draw: ImageDraw instance
            label_coords: Label rectangle coordinates (x0, y0, x1, y1)
            value_coords: Value rectangle coordinates (x0, y0, x1, y1)
            color: Line color with alpha
            width: Line width
        """
        # Calculate connection points (center of closest edges)
        label_center_x = (label_coords[0] + label_coords[2]) // 2
        label_center_y = (label_coords[1] + label_coords[3]) // 2
        value_center_x = (value_coords[0] + value_coords[2]) // 2
        value_center_y = (value_coords[1] + value_coords[3]) // 2

        # Determine connection strategy based on relative positions
        label_right = label_coords[2]
        label_bottom = label_coords[3]
        value_left = value_coords[0]
        value_top = value_coords[1]

        # Same-line pairing (horizontal connection)
        if abs(label_center_y - value_center_y) < 20 and value_left > label_right:
            start_point = (label_right, label_center_y)
            end_point = (value_left, value_center_y)
        # Below pairing (vertical connection)
        elif value_center_y > label_bottom:
            start_point = (label_center_x, label_bottom)
            end_point = (value_center_x, value_top)
        # Default: connect centers with a line
        else:
            start_point = (label_center_x, label_center_y)
            end_point = (value_center_x, value_center_y)

        # Draw line
        draw.line([start_point, end_point], fill=color[:3], width=width)

        # Draw small circles at connection points for better visibility
        circle_radius = 3
        draw.ellipse(
            [
                start_point[0] - circle_radius,
                start_point[1] - circle_radius,
                start_point[0] + circle_radius,
                start_point[1] + circle_radius,
            ],
            fill=color[:3],
        )

        draw.ellipse(
            [
                end_point[0] - circle_radius,
                end_point[1] - circle_radius,
                end_point[0] + circle_radius,
                end_point[1] + circle_radius,
            ],
            fill=color[:3],
        )

    def _draw_text_with_background(
        self,
        draw,
        text: str,
        position: Tuple[int, int],
        font,
        bg_color: Tuple[int, int, int, int] = (255, 255, 255, 200),
        text_color: Tuple[int, int, int] = (0, 0, 0),
    ):
        """Draw text with a background rectangle.

        Args:
            draw: ImageDraw instance
            text: Text to draw
            position: Text position (x, y)
            font: Font to use
            bg_color: Background color with alpha
            text_color: Text color
        """
        try:
            # Get text dimensions
            bbox = draw.textbbox(position, text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Draw background rectangle with padding
            padding = 2
            bg_coords = (
                position[0] - padding,
                position[1] - padding,
                position[0] + text_width + padding,
                position[1] + text_height + padding,
            )
            draw.rectangle(bg_coords, fill=bg_color)

            # Draw text
            draw.text(position, text, fill=text_color, font=font)

        except AttributeError:
            # Fallback for older PIL versions
            try:
                text_size = draw.textsize(text, font=font)
                text_width, text_height = text_size

                padding = 2
                bg_coords = (
                    position[0] - padding,
                    position[1] - padding,
                    position[0] + text_width + padding,
                    position[1] + text_height + padding,
                )
                draw.rectangle(bg_coords, fill=bg_color)
                draw.text(position, text, fill=text_color, font=font)

            except Exception:
                # Ultimate fallback: just draw text
                draw.text(position, text, fill=text_color, font=font)

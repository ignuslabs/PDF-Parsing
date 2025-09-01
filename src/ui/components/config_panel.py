"""
Parser Configuration Panel Component

Provides a reusable component for configuring parser settings with
presets and validation.
"""

import streamlit as st
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class ConfigPreset:
    """Configuration preset for different use cases."""

    name: str
    description: str
    config: Dict[str, Any]


class ParserConfigPanel:
    """Provides UI for configuring parser settings."""

    def __init__(self):
        """Initialize configuration panel."""
        self.presets = self._create_presets()

    def _create_presets(self) -> List[ConfigPreset]:
        """Create configuration presets."""
        return [
            ConfigPreset(
                name="Fast Processing",
                description="Quick parsing with basic features",
                config={
                    "enable_ocr": False,
                    "enable_tables": True,
                    "generate_page_images": False,
                    "ocr_engine": "tesseract",
                    "ocr_language": "eng",
                    "table_mode": "fast",
                    "image_scale": 1.0,
                },
            ),
            ConfigPreset(
                name="Standard Processing",
                description="Balanced speed and accuracy",
                config={
                    "enable_ocr": False,
                    "enable_tables": True,
                    "generate_page_images": True,
                    "ocr_engine": "tesseract",
                    "ocr_language": "eng",
                    "table_mode": "accurate",
                    "image_scale": 1.0,
                },
            ),
            ConfigPreset(
                name="High Accuracy",
                description="Best quality extraction with OCR",
                config={
                    "enable_ocr": True,
                    "enable_tables": True,
                    "generate_page_images": True,
                    "ocr_engine": "tesseract",
                    "ocr_language": "eng",
                    "table_mode": "accurate",
                    "image_scale": 1.2,
                },
            ),
            ConfigPreset(
                name="Scanned Documents",
                description="Optimized for scanned PDFs with OCR",
                config={
                    "enable_ocr": True,
                    "enable_tables": True,
                    "generate_page_images": True,
                    "ocr_engine": "easyocr",
                    "ocr_language": "eng",
                    "table_mode": "accurate",
                    "image_scale": 1.5,
                },
            ),
            ConfigPreset(
                name="Verification Mode",
                description="Full features for verification workflow",
                config={
                    "enable_ocr": False,
                    "enable_tables": True,
                    "generate_page_images": True,
                    "ocr_engine": "tesseract",
                    "ocr_language": "eng",
                    "table_mode": "accurate",
                    "image_scale": 1.0,
                },
            ),
        ]

    def display_preset_selector(self, current_config: Dict[str, Any]) -> Dict[str, Any]:
        """Display preset selector and apply selected preset.

        Args:
            current_config: Current configuration dictionary

        Returns:
            Updated configuration dictionary
        """
        st.subheader("🎯 Configuration Presets")

        # Preset selection
        preset_names = ["Custom"] + [p.name for p in self.presets]

        # Find current preset if any
        current_preset_idx = 0  # Default to "Custom"
        for i, preset in enumerate(self.presets):
            if self._configs_match(current_config, preset.config):
                current_preset_idx = i + 1
                break

        selected_preset = st.selectbox(
            "Choose configuration preset",
            options=range(len(preset_names)),
            format_func=lambda i: preset_names[i],
            index=current_preset_idx,
            help="Select a preset or choose 'Custom' for manual configuration",
        )

        # Apply preset if selected
        if selected_preset > 0:  # Not "Custom"
            preset = self.presets[selected_preset - 1]
            st.info(f"**{preset.name}:** {preset.description}")

            if st.button("Apply Preset", key="apply_preset"):
                st.success(f"Applied {preset.name} configuration!")
                return preset.config.copy()

        return current_config

    def display_basic_settings(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Display basic parser settings.

        Args:
            config: Current configuration dictionary

        Returns:
            Updated configuration dictionary
        """
        st.subheader("⚙️ Basic Settings")

        with st.expander("Core Features", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                config["enable_tables"] = st.checkbox(
                    "Extract Tables",
                    value=config.get("enable_tables", True),
                    help="Extract table structures from documents",
                )

                config["generate_page_images"] = st.checkbox(
                    "Generate Page Images",
                    value=config.get("generate_page_images", True),
                    help="Required for verification functionality",
                )

            with col2:
                config["enable_ocr"] = st.checkbox(
                    "Enable OCR",
                    value=config.get("enable_ocr", False),
                    help="Extract text from scanned documents (slower)",
                )

                if config["enable_ocr"]:
                    config["ocr_engine"] = st.selectbox(
                        "OCR Engine",
                        options=["tesseract", "easyocr"],
                        index=0 if config.get("ocr_engine", "tesseract") == "tesseract" else 1,
                        help="Choose OCR engine",
                    )

        return config

    def display_advanced_settings(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Display advanced parser settings.

        Args:
            config: Current configuration dictionary

        Returns:
            Updated configuration dictionary
        """
        with st.expander("🔧 Advanced Settings"):
            col1, col2 = st.columns(2)

            with col1:
                # Table extraction settings
                if config.get("enable_tables", True):
                    config["table_mode"] = st.selectbox(
                        "Table Extraction Mode",
                        options=["accurate", "fast"],
                        index=0 if config.get("table_mode", "accurate") == "accurate" else 1,
                        help="Accurate mode is slower but more precise",
                    )

                # OCR language settings
                if config.get("enable_ocr", False):
                    config["ocr_language"] = st.text_input(
                        "OCR Languages",
                        value=config.get("ocr_language", "eng"),
                        help="Language codes (e.g., 'eng', 'fra', 'deu'). Multiple: 'eng+fra'",
                    )

            with col2:
                # Image generation settings
                if config.get("generate_page_images", True):
                    config["image_scale"] = st.slider(
                        "Image Scale Factor",
                        min_value=0.5,
                        max_value=2.0,
                        value=config.get("image_scale", 1.0),
                        step=0.1,
                        help="Higher values = better quality, larger file sizes",
                    )

                # Memory optimization hint
                if config.get("enable_ocr", False) and config.get("generate_page_images", True):
                    st.warning(
                        "⚠️ OCR + Page Images uses more memory. Consider reducing image scale for large documents."
                    )

        return config

    def display_performance_indicators(self, config: Dict[str, Any]):
        """Display performance indicators based on configuration.

        Args:
            config: Current configuration dictionary
        """
        st.subheader("📊 Performance Estimate")

        # Calculate performance scores (simplified)
        speed_score = 100
        quality_score = 50
        memory_usage = 20

        if config.get("enable_ocr", False):
            speed_score -= 40
            quality_score += 30
            memory_usage += 30

        if config.get("enable_tables", True):
            speed_score -= 10
            quality_score += 20
            memory_usage += 10

            if config.get("table_mode", "accurate") == "accurate":
                speed_score -= 10
                quality_score += 10

        if config.get("generate_page_images", True):
            speed_score -= 5
            memory_usage += 20

            scale = config.get("image_scale", 1.0)
            if scale > 1.0:
                memory_usage += int((scale - 1.0) * 20)

        # Normalize scores
        speed_score = max(10, min(100, speed_score))
        quality_score = max(10, min(100, quality_score))
        memory_usage = min(100, memory_usage)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Speed", f"{speed_score}/100", help="Estimated processing speed")

        with col2:
            st.metric("Quality", f"{quality_score}/100", help="Estimated extraction quality")

        with col3:
            st.metric("Memory Usage", f"{memory_usage}/100", help="Estimated memory consumption")

        # Recommendations
        if speed_score < 30:
            st.warning(
                "⚠️ **Slow configuration detected.** Consider disabling OCR or using fast table mode for better speed."
            )

        if memory_usage > 70:
            st.warning(
                "⚠️ **High memory usage.** Consider reducing image scale or disabling page image generation for large documents."
            )

    def display_full_panel(self, session_state_key: str = "parser_config") -> Dict[str, Any]:
        """Display complete configuration panel.

        Args:
            session_state_key: Key in session state to store configuration

        Returns:
            Current configuration dictionary
        """
        # Initialize config in session state if not exists
        if session_state_key not in st.session_state:
            st.session_state[session_state_key] = {
                "enable_ocr": False,
                "enable_tables": True,
                "generate_page_images": True,
                "ocr_engine": "tesseract",
                "ocr_language": "eng",
                "table_mode": "accurate",
                "image_scale": 1.0,
            }

        config = st.session_state[session_state_key]

        # Display preset selector
        config = self.display_preset_selector(config)

        # Display basic settings
        config = self.display_basic_settings(config)

        # Display advanced settings
        config = self.display_advanced_settings(config)

        # Performance indicators
        self.display_performance_indicators(config)

        # Update session state
        st.session_state[session_state_key] = config

        return config

    def _configs_match(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> bool:
        """Check if two configurations match.

        Args:
            config1: First configuration
            config2: Second configuration

        Returns:
            True if configurations match
        """
        for key in config2:
            if config1.get(key) != config2[key]:
                return False
        return True

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration and return any issues.

        Args:
            config: Configuration to validate

        Returns:
            List of validation error messages
        """
        issues = []

        # Check OCR language format
        if config.get("enable_ocr", False):
            ocr_lang = config.get("ocr_language", "eng")
            if not ocr_lang or not isinstance(ocr_lang, str):
                issues.append("OCR language must be specified when OCR is enabled")
            elif not all(len(lang) >= 2 for lang in ocr_lang.replace("+", " ").split()):
                issues.append("OCR language codes must be at least 2 characters")

        # Check image scale
        image_scale = config.get("image_scale", 1.0)
        if not isinstance(image_scale, (int, float)) or not 0.1 <= image_scale <= 3.0:
            issues.append("Image scale must be between 0.1 and 3.0")

        # Check logical dependencies
        if not config.get("generate_page_images", True):
            st.info(
                "💡 **Note:** Page images are disabled. Verification functionality will be limited."
            )

        return issues

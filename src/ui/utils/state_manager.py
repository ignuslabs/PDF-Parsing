"""
Session State Manager

Provides utilities for managing Streamlit session state with proper
initialization, validation, and persistence.
"""

import streamlit as st
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
import json


class StateManager:
    """Manages Streamlit session state with validation and persistence."""

    def __init__(self):
        """Initialize state manager."""
        self._validators = {}
        self._default_values = {}
        self._callbacks = {}

    def register_state(
        self,
        key: str,
        default_value: Any,
        validator: Optional[Callable] = None,
        callback: Optional[Callable] = None,
    ):
        """Register a state variable with default value and optional validator.

        Args:
            key: Session state key
            default_value: Default value if key doesn't exist
            validator: Optional validation function
            callback: Optional callback when value changes
        """
        self._default_values[key] = default_value
        if validator:
            self._validators[key] = validator
        if callback:
            self._callbacks[key] = callback

        # Initialize if not exists
        if key not in st.session_state:
            st.session_state[key] = default_value

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from session state with fallback.

        Args:
            key: Session state key
            default: Default value if key doesn't exist

        Returns:
            Value from session state or default
        """
        if key in self._default_values:
            return st.session_state.get(key, self._default_values[key])
        return st.session_state.get(key, default)

    def set(self, key: str, value: Any, validate: bool = True) -> bool:
        """Set value in session state with optional validation.

        Args:
            key: Session state key
            value: Value to set
            validate: Whether to run validation

        Returns:
            True if value was set successfully
        """
        # Validate if validator exists
        if validate and key in self._validators:
            validator = self._validators[key]
            try:
                if not validator(value):
                    st.error(f"Invalid value for {key}: {value}")
                    return False
            except Exception as e:
                st.error(f"Validation error for {key}: {e}")
                return False

        # Store previous value for callback
        old_value = st.session_state.get(key)

        # Set new value
        st.session_state[key] = value

        # Run callback if exists and value changed
        if key in self._callbacks and old_value != value:
            try:
                self._callbacks[key](old_value, value)
            except Exception as e:
                st.error(f"Callback error for {key}: {e}")

        return True

    def update(self, updates: Dict[str, Any], validate: bool = True) -> bool:
        """Update multiple values in session state.

        Args:
            updates: Dictionary of key-value pairs to update
            validate: Whether to run validation

        Returns:
            True if all updates were successful
        """
        success = True
        for key, value in updates.items():
            if not self.set(key, value, validate):
                success = False
        return success

    def reset(self, key: str = None):
        """Reset session state key(s) to default values.

        Args:
            key: Specific key to reset, or None to reset all registered keys
        """
        if key:
            if key in self._default_values:
                st.session_state[key] = self._default_values[key]
            elif key in st.session_state:
                del st.session_state[key]
        else:
            # Reset all registered keys
            for k, default in self._default_values.items():
                st.session_state[k] = default

    def clear_all(self):
        """Clear all session state (use with caution)."""
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        # Reinitialize registered defaults
        for key, default in self._default_values.items():
            st.session_state[key] = default

    def validate_all(self) -> List[str]:
        """Validate all registered state variables.

        Returns:
            List of validation error messages
        """
        errors = []

        for key, validator in self._validators.items():
            if key in st.session_state:
                try:
                    value = st.session_state[key]
                    if not validator(value):
                        errors.append(f"Invalid value for {key}: {value}")
                except Exception as e:
                    errors.append(f"Validation error for {key}: {e}")

        return errors

    def export_state(self, keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """Export session state to dictionary.

        Args:
            keys: Specific keys to export, or None for all

        Returns:
            Dictionary with exported state
        """
        if keys is None:
            keys = list(self._default_values.keys())

        exported = {}
        for key in keys:
            if key in st.session_state:
                value = st.session_state[key]
                # Only export JSON-serializable values
                try:
                    json.dumps(value)
                    exported[key] = value
                except (TypeError, ValueError):
                    # Skip non-serializable values
                    pass

        return exported

    def import_state(self, state_dict: Dict[str, Any], validate: bool = True) -> bool:
        """Import state from dictionary.

        Args:
            state_dict: Dictionary with state values
            validate: Whether to validate imported values

        Returns:
            True if import was successful
        """
        return self.update(state_dict, validate)

    def get_state_info(self) -> Dict[str, Any]:
        """Get information about current session state.

        Returns:
            Dictionary with state information
        """
        return {
            "registered_keys": list(self._default_values.keys()),
            "total_keys": len(st.session_state),
            "registered_count": len(self._default_values),
            "validators": list(self._validators.keys()),
            "callbacks": list(self._callbacks.keys()),
            "memory_usage_estimate": len(str(st.session_state)),
        }


class DocumentStateManager(StateManager):
    """Extended state manager for document-related state."""

    def __init__(self):
        """Initialize document state manager with common validators."""
        super().__init__()
        self._setup_document_state()

    def _setup_document_state(self):
        """Setup common document state variables."""
        # Document management
        self.register_state("parsed_documents", [], self._validate_document_list)
        self.register_state("current_doc_index", 0, self._validate_doc_index)
        self.register_state("current_page", 1, self._validate_page_number)

        # Element selection and verification
        self.register_state("selected_element_id", None)
        self.register_state("verification_interfaces", {})
        self.register_state("pdf_renderers", {})

        # Search functionality
        self.register_state("search_results", [])
        self.register_state("search_query", "", self._validate_search_query)
        self.register_state(
            "search_filters",
            {
                "element_types": [],
                "page_range": None,
                "confidence_threshold": 0.0,
                "document_indices": [],
            },
        )

        # Parser configuration
        self.register_state(
            "parser_config",
            {
                "enable_ocr": False,
                "enable_tables": True,
                "generate_page_images": True,
                "ocr_engine": "tesseract",
                "ocr_language": "eng",
                "table_mode": "accurate",
                "image_scale": 1.0,
            },
            self._validate_parser_config,
        )

        # UI state
        self.register_state("show_debug_info", False)
        self.register_state("sidebar_expanded", True)

    def _validate_document_list(self, value) -> bool:
        """Validate document list."""
        return isinstance(value, list)

    def _validate_doc_index(self, value) -> bool:
        """Validate document index."""
        if not isinstance(value, int) or value < 0:
            return False

        docs = self.get("parsed_documents", [])
        return value < len(docs) if docs else value == 0

    def _validate_page_number(self, value) -> bool:
        """Validate page number."""
        return isinstance(value, int) and value >= 1

    def _validate_search_query(self, value) -> bool:
        """Validate search query."""
        return isinstance(value, str)

    def _validate_parser_config(self, value) -> bool:
        """Validate parser configuration."""
        if not isinstance(value, dict):
            return False

        required_keys = [
            "enable_ocr",
            "enable_tables",
            "generate_page_images",
            "ocr_engine",
            "ocr_language",
            "table_mode",
            "image_scale",
        ]

        return all(key in value for key in required_keys)

    def get_current_document(self):
        """Get the currently selected document."""
        documents = self.get("parsed_documents", [])
        if not documents:
            return None

        doc_index = self.get("current_doc_index", 0)
        if doc_index >= len(documents):
            # Reset to first document if index is invalid
            self.set("current_doc_index", 0)
            doc_index = 0

        return documents[doc_index]

    def get_current_elements(self):
        """Get elements from the current document."""
        doc = self.get_current_document()
        return doc.elements if doc else []

    def add_document(self, document) -> int:
        """Add a new document to the state.

        Args:
            document: ParsedDocument to add

        Returns:
            Index of the added document
        """
        documents = self.get("parsed_documents", [])
        documents.append(document)
        self.set("parsed_documents", documents)
        return len(documents) - 1

    def remove_document(self, index: int) -> bool:
        """Remove a document from the state.

        Args:
            index: Index of document to remove

        Returns:
            True if document was removed
        """
        documents = self.get("parsed_documents", [])
        if 0 <= index < len(documents):
            documents.pop(index)
            self.set("parsed_documents", documents)

            # Adjust current document index if necessary
            current_index = self.get("current_doc_index", 0)
            if current_index >= len(documents):
                self.set("current_doc_index", max(0, len(documents) - 1))

            return True
        return False

    def clear_documents(self):
        """Clear all documents and related state."""
        self.set("parsed_documents", [])
        self.set("current_doc_index", 0)
        self.set("current_page", 1)
        self.set("selected_element_id", None)
        self.set("search_results", [])
        self.set("verification_interfaces", {})
        self.set("pdf_renderers", {})


# Global state manager instance
state_manager = DocumentStateManager()

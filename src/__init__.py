"""Smart PDF Parser package."""

try:
    # Prefer runtime package version if installed
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("smart-pdf-parser")
except Exception:
    # Fallback to declared project version
    __version__ = "0.1.0"

__all__ = ["__version__"]

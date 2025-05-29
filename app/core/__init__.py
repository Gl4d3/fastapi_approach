"""
Core application modules for document processing.
"""

from .config import get_settings
from .document_processor import DocumentProcessor

__all__ = ["get_settings", "DocumentProcessor"]
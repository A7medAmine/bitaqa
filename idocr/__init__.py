"""idocr - template-based ID card field extraction (OpenCV + OCR + Pydantic)."""

from .pipeline import extract_card
from .template import load_template

__all__ = ["extract_card", "load_template"]

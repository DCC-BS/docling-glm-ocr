"""Docling plugin entry point registering the GLM-OCR remote OCR engine."""

from docling_glm_ocr.model import GlmOcrRemoteModel


def ocr_engines() -> dict[str, list[type]]:
    """Return the OCR engine classes provided by this plugin."""
    return {"ocr_engines": [GlmOcrRemoteModel]}

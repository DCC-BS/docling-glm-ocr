"""Docling plugin entry point registering the GLM-OCR remote OCR engine."""

from docling_glm_ocr.model import GlmOcrRemoteModel


def ocr_engines():
    return {"ocr_engines": [GlmOcrRemoteModel]}

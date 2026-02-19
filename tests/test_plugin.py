"""Tests for the docling plugin entry point."""

from __future__ import annotations

from docling.models.base_ocr_model import BaseOcrModel

from docling_glm_ocr.plugin import ocr_engines


def test_ocr_engines_returns_dict():
    result = ocr_engines()
    assert isinstance(result, dict)
    assert "ocr_engines" in result


def test_ocr_engines_contains_model_class():
    result = ocr_engines()
    classes = result["ocr_engines"]
    assert len(classes) == 1
    assert issubclass(classes[0], BaseOcrModel)


def test_registered_model_name():
    cls = ocr_engines()["ocr_engines"][0]
    assert cls.__name__ == "GlmOcrRemoteModel"

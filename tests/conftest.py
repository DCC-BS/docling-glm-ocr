"""Pytest configuration for docling_glm_ocr tests."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from docling_glm_ocr.options import GlmOcrRemoteOptions


def pytest_collection_modifyitems(config, items):
    """Auto-skip e2e tests when no vLLM server URL is configured."""
    if os.environ.get("GLMOCR_REMOTE_OCR_API_URL"):
        return
    skip_e2e = pytest.mark.skip(reason="GLMOCR_REMOTE_OCR_API_URL not set")
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip_e2e)


@pytest.fixture
def default_options() -> GlmOcrRemoteOptions:
    return GlmOcrRemoteOptions()


@pytest.fixture
def mock_model():
    """Create a GlmOcrRemoteModel with a mocked HTTP client."""
    from docling.datamodel.accelerator_options import AcceleratorOptions

    from docling_glm_ocr.model import GlmOcrRemoteModel

    opts = GlmOcrRemoteOptions(
        api_url="http://test-server:8000/v1/chat/completions",
        model_name="test-model",
    )

    with patch("httpx.Client") as mock_client_cls:
        model = GlmOcrRemoteModel(
            enabled=True,
            artifacts_path=None,
            options=opts,
            accelerator_options=AcceleratorOptions(),
        )
        yield model, mock_client_cls.return_value

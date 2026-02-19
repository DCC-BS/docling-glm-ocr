"""End-to-end tests requiring a running vLLM server.

Skipped automatically unless ``GLMOCR_REMOTE_OCR_API_URL`` is set::

    GLMOCR_REMOTE_OCR_API_URL=http://host:port/v1/chat/completions pytest -m e2e
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image, ImageDraw

from docling_glm_ocr.model import GlmOcrRemoteModel
from docling_glm_ocr.options import GlmOcrRemoteOptions

pytestmark = pytest.mark.e2e

VLLM_URL = os.environ.get("GLMOCR_REMOTE_OCR_API_URL", "")


@pytest.fixture
def e2e_model():
    """Create a GlmOcrRemoteModel pointing at the real vLLM server."""
    from docling.datamodel.accelerator_options import AcceleratorOptions

    opts = GlmOcrRemoteOptions(api_url=VLLM_URL)
    return GlmOcrRemoteModel(
        enabled=True,
        artifacts_path=None,
        options=opts,
        accelerator_options=AcceleratorOptions(),
    )


def _make_text_image(text: str, size: tuple[int, int] = (400, 100)) -> Image.Image:
    """Render *text* onto a white PIL image for OCR recognition."""
    img = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 30), text, fill="black")
    return img


class TestRecogniseCropE2E:
    """Smoke-tests that hit the real vLLM endpoint via ``_recognise_crop``."""

    def test_recognises_printed_text(self, e2e_model):
        img = _make_text_image("Hello World")
        result = e2e_model._recognise_crop(img)
        assert isinstance(result, str)
        assert len(result.strip()) > 0
        assert "Hello World" in result

    def test_recognises_digits(self, e2e_model):
        img = _make_text_image("12345 67890")
        result = e2e_model._recognise_crop(img)
        assert any(ch.isdigit() for ch in result)
        assert "12345 67890" in result

    def test_blank_image_returns_string(self, e2e_model):
        img = Image.new("RGB", (200, 200), color="white")
        result = e2e_model._recognise_crop(img)
        assert isinstance(result, str)


class TestCallE2E:
    """Integration test of the full ``__call__`` path with a real server."""

    @staticmethod
    def _make_ocr_rect(*, left=0, top=0, right=50, bottom=50):
        rect = MagicMock()
        rect.area.return_value = (right - left) * (bottom - top)
        rect.l, rect.t, rect.r, rect.b = left, top, right, bottom
        return rect

    def test_full_call_produces_cells(self, e2e_model):
        page = MagicMock()
        page._backend.is_valid.return_value = True
        page._backend.get_page_image.return_value = _make_text_image("Integration test")

        rect = self._make_ocr_rect(left=0, top=0, right=400, bottom=100)

        with (
            patch.object(e2e_model, "get_ocr_rects", return_value=[rect]),
            patch.object(e2e_model, "post_process_cells") as mock_post,
            patch("docling_glm_ocr.model.TimeRecorder"),
        ):
            pages = list(e2e_model(MagicMock(), [page]))

        assert len(pages) == 1
        mock_post.assert_called_once()
        cells = mock_post.call_args[0][0]
        assert len(cells) >= 1
        assert len(cells[0].text.strip()) > 0

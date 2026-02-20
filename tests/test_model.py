"""Tests for GlmOcrRemoteModel."""

from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

import httpx
import pytest
from docling.datamodel.pipeline_options import OcrOptions
from PIL import Image

from docling_glm_ocr.model import GlmOcrRemoteModel, _pil_to_base64_uri
from docling_glm_ocr.options import GlmOcrRemoteOptions


class TestGetOptionsType:
    def test_returns_options_class(self):
        assert GlmOcrRemoteModel.get_options_type() is GlmOcrRemoteOptions

    def test_is_subclass_of_base(self):
        assert issubclass(GlmOcrRemoteModel.get_options_type(), OcrOptions)


class TestPilToBase64Uri:
    def test_produces_data_uri(self):
        img = Image.new("RGB", (10, 10), color="red")
        uri = _pil_to_base64_uri(img)
        assert uri.startswith("data:image/png;base64,")

    def test_base64_is_decodable(self):
        img = Image.new("RGB", (10, 10), color="blue")
        uri = _pil_to_base64_uri(img)
        b64_part = uri.split(",", 1)[1]
        decoded = base64.b64decode(b64_part)
        assert len(decoded) > 0

    def test_custom_format(self):
        img = Image.new("RGB", (10, 10))
        uri = _pil_to_base64_uri(img, fmt="JPEG")
        assert uri.startswith("data:image/jpeg;base64,")


class TestRecogniseCrop:
    def test_sends_correct_payload_structure(self, mock_model):
        model, mock_client = mock_model

        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Hello World"}}]}
        mock_client.post.return_value = mock_response

        img = Image.new("RGB", (50, 50))
        model._recognise_crop(img)

        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "http://test-server:8000/v1/chat/completions"

        payload = call_args[1]["json"]
        assert payload["model"] == "test-model"
        assert payload["max_tokens"] == 16384
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"

        content = payload["messages"][0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/")

    def test_returns_extracted_text(self, mock_model):
        model, mock_client = mock_model

        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Extracted text content"}}]}
        mock_client.post.return_value = mock_response

        img = Image.new("RGB", (50, 50))
        result = model._recognise_crop(img)

        assert result == "Extracted text content"

    def test_empty_choices_returns_empty_string(self, mock_model):
        model, mock_client = mock_model

        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": []}
        mock_client.post.return_value = mock_response

        img = Image.new("RGB", (50, 50))
        result = model._recognise_crop(img)

        assert result == ""

    def test_missing_content_returns_empty_string(self, mock_model):
        model, mock_client = mock_model

        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {}}]}
        mock_client.post.return_value = mock_response

        img = Image.new("RGB", (50, 50))
        result = model._recognise_crop(img)

        assert result == ""

    def test_raises_on_http_error(self, mock_model):
        model, mock_client = mock_model

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Internal Server Error",
            request=MagicMock(),
            response=MagicMock(status_code=500),
        )
        mock_client.post.return_value = mock_response

        img = Image.new("RGB", (50, 50))
        with pytest.raises(httpx.HTTPStatusError):
            model._recognise_crop(img)


class TestDisabledModel:
    def test_disabled_model_yields_pages_unchanged(self):
        from docling.datamodel.accelerator_options import AcceleratorOptions

        opts = GlmOcrRemoteOptions()
        model = GlmOcrRemoteModel(
            enabled=False,
            artifacts_path=None,
            options=opts,
            accelerator_options=AcceleratorOptions(),
        )

        mock_page = MagicMock()
        pages = list(model(MagicMock(), [mock_page]))
        assert pages == [mock_page]

    def test_disabled_model_has_no_client(self):
        from docling.datamodel.accelerator_options import AcceleratorOptions

        opts = GlmOcrRemoteOptions()
        model = GlmOcrRemoteModel(
            enabled=False,
            artifacts_path=None,
            options=opts,
            accelerator_options=AcceleratorOptions(),
        )

        assert not hasattr(model, "_client")


class TestModelInit:
    def test_enabled_model_creates_client(self, mock_model):
        model, _ = mock_model
        assert hasattr(model, "_client")

    def test_scale_is_set(self, mock_model):
        model, _ = mock_model
        assert model.scale == 3

    def test_options_stored(self, mock_model):
        model, _ = mock_model
        assert isinstance(model.options, GlmOcrRemoteOptions)
        assert model.options.model_name == "test-model"


class TestCall:
    """Tests for GlmOcrRemoteModel.__call__."""

    @staticmethod
    def _make_ocr_rect(*, left=0, top=0, right=50, bottom=50, area=2500):
        rect = MagicMock()
        rect.area.return_value = area
        rect.l, rect.t, rect.r, rect.b = left, top, right, bottom
        return rect

    def test_valid_page_produces_text_cells(self, mock_model):
        model, mock_client = mock_model
        page = MagicMock()
        page._backend.is_valid.return_value = True
        page._backend.get_page_image.return_value = Image.new("RGB", (100, 100))

        ocr_rect = self._make_ocr_rect()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "Hello World"}}]}
        mock_client.post.return_value = mock_resp

        with (
            patch.object(model, "get_ocr_rects", return_value=[ocr_rect]),
            patch.object(model, "post_process_cells") as mock_post,
            patch("docling_glm_ocr.model.TimeRecorder"),
        ):
            pages = list(model(MagicMock(), [page]))

        assert len(pages) == 1
        mock_post.assert_called_once()
        cells = mock_post.call_args[0][0]
        assert len(cells) == 1
        assert cells[0].text == "Hello World"
        assert cells[0].from_ocr is True
        assert cells[0].confidence == 1.0

    def test_cell_index_matches_enumerate_position(self, mock_model):
        model, mock_client = mock_model
        page = MagicMock()
        page._backend.is_valid.return_value = True
        page._backend.get_page_image.return_value = Image.new("RGB", (100, 100))

        skip_rect = self._make_ocr_rect(area=0)
        real_rect = self._make_ocr_rect(left=50, right=100)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "Text"}}]}
        mock_client.post.return_value = mock_resp

        with (
            patch.object(model, "get_ocr_rects", return_value=[skip_rect, real_rect]),
            patch.object(model, "post_process_cells") as mock_post,
            patch("docling_glm_ocr.model.TimeRecorder"),
        ):
            list(model(MagicMock(), [page]))

        cells = mock_post.call_args[0][0]
        assert len(cells) == 1
        assert cells[0].index == 1

    def test_invalid_backend_yields_page_unchanged(self, mock_model):
        model, mock_client = mock_model
        page = MagicMock()
        page._backend.is_valid.return_value = False

        pages = list(model(MagicMock(), [page]))

        assert pages == [page]
        mock_client.post.assert_not_called()

    def test_zero_area_rect_skipped(self, mock_model):
        model, mock_client = mock_model
        page = MagicMock()
        page._backend.is_valid.return_value = True

        ocr_rect = self._make_ocr_rect(area=0)

        with (
            patch.object(model, "get_ocr_rects", return_value=[ocr_rect]),
            patch.object(model, "post_process_cells") as mock_post,
            patch("docling_glm_ocr.model.TimeRecorder"),
        ):
            pages = list(model(MagicMock(), [page]))

        assert len(pages) == 1
        cells = mock_post.call_args[0][0]
        assert len(cells) == 0
        mock_client.post.assert_not_called()

    def test_blank_text_skipped(self, mock_model):
        model, mock_client = mock_model
        page = MagicMock()
        page._backend.is_valid.return_value = True
        page._backend.get_page_image.return_value = Image.new("RGB", (100, 100))

        ocr_rect = self._make_ocr_rect()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "   \n  "}}]}
        mock_client.post.return_value = mock_resp

        with (
            patch.object(model, "get_ocr_rects", return_value=[ocr_rect]),
            patch.object(model, "post_process_cells") as mock_post,
            patch("docling_glm_ocr.model.TimeRecorder"),
        ):
            list(model(MagicMock(), [page]))

        cells = mock_post.call_args[0][0]
        assert len(cells) == 0

    def test_http_error_logged_and_continued(self, mock_model):
        model, mock_client = mock_model
        model.options.max_retries = 1
        model.options.retry_backoff_factor = 0.0
        model.options.max_concurrent_requests = 1

        page = MagicMock()
        page._backend.is_valid.return_value = True
        page._backend.get_page_image.return_value = Image.new("RGB", (100, 100))

        bad_rect = self._make_ocr_rect(left=0, top=0, right=50, bottom=50)
        good_rect = self._make_ocr_rect(left=50, top=0, right=100, bottom=50)

        error_resp = MagicMock()
        error_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error",
            request=MagicMock(),
            response=MagicMock(status_code=500),
        )
        ok_resp = MagicMock()
        ok_resp.json.return_value = {"choices": [{"message": {"content": "Good"}}]}
        mock_client.post.side_effect = [error_resp, error_resp, ok_resp]

        with (
            patch.object(model, "get_ocr_rects", return_value=[bad_rect, good_rect]),
            patch.object(model, "post_process_cells") as mock_post,
            patch("docling_glm_ocr.model.TimeRecorder"),
        ):
            pages = list(model(MagicMock(), [page]))

        assert len(pages) == 1
        cells = mock_post.call_args[0][0]
        assert len(cells) == 1
        assert cells[0].text == "Good"

    def test_multiple_rects_produce_multiple_cells(self, mock_model):
        model, mock_client = mock_model
        model.options.max_concurrent_requests = 1

        page = MagicMock()
        page._backend.is_valid.return_value = True
        page._backend.get_page_image.return_value = Image.new("RGB", (100, 100))

        rect1 = self._make_ocr_rect(left=0, top=0, right=50, bottom=50)
        rect2 = self._make_ocr_rect(left=50, top=0, right=100, bottom=50)
        resp1 = MagicMock()
        resp1.json.return_value = {"choices": [{"message": {"content": "Cell one"}}]}
        resp2 = MagicMock()
        resp2.json.return_value = {"choices": [{"message": {"content": "Cell two"}}]}
        mock_client.post.side_effect = [resp1, resp2]

        with (
            patch.object(model, "get_ocr_rects", return_value=[rect1, rect2]),
            patch.object(model, "post_process_cells") as mock_post,
            patch("docling_glm_ocr.model.TimeRecorder"),
        ):
            list(model(MagicMock(), [page]))

        cells = mock_post.call_args[0][0]
        assert len(cells) == 2
        assert cells[0].text == "Cell one"
        assert cells[1].text == "Cell two"

    def test_no_ocr_rects_yields_page(self, mock_model):
        model, _ = mock_model
        page = MagicMock()
        page._backend.is_valid.return_value = True

        with (
            patch.object(model, "get_ocr_rects", return_value=[]),
            patch.object(model, "post_process_cells") as mock_post,
            patch("docling_glm_ocr.model.TimeRecorder"),
        ):
            pages = list(model(MagicMock(), [page]))

        assert len(pages) == 1
        cells = mock_post.call_args[0][0]
        assert len(cells) == 0

    def test_multiple_pages_all_yielded(self, mock_model):
        model, mock_client = mock_model
        pages_in = []
        for _ in range(3):
            p = MagicMock()
            p._backend.is_valid.return_value = True
            p._backend.get_page_image.return_value = Image.new("RGB", (100, 100))
            pages_in.append(p)

        rect = self._make_ocr_rect()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "T"}}]}
        mock_client.post.return_value = mock_resp

        with (
            patch.object(model, "get_ocr_rects", return_value=[rect]),
            patch.object(model, "post_process_cells"),
            patch("docling_glm_ocr.model.TimeRecorder"),
        ):
            pages_out = list(model(MagicMock(), pages_in))

        assert len(pages_out) == 3

    def test_get_page_image_called_with_correct_scale(self, mock_model):
        model, mock_client = mock_model
        page = MagicMock()
        page._backend.is_valid.return_value = True
        page._backend.get_page_image.return_value = Image.new("RGB", (100, 100))

        ocr_rect = self._make_ocr_rect()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "X"}}]}
        mock_client.post.return_value = mock_resp

        with (
            patch.object(model, "get_ocr_rects", return_value=[ocr_rect]),
            patch.object(model, "post_process_cells"),
            patch("docling_glm_ocr.model.TimeRecorder"),
        ):
            list(model(MagicMock(), [page]))

        page._backend.get_page_image.assert_called_once_with(scale=3, cropbox=ocr_rect)

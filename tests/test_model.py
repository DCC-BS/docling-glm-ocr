"""Tests for GlmOcrRemoteModel."""

from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

import httpx
import pytest
from docling.datamodel.base_models import DoclingComponentType
from docling.datamodel.pipeline_options import OcrOptions
from PIL import Image

from docling_glm_ocr.model import GlmOcrRemoteModel, _pil_to_base64_uri
from docling_glm_ocr.options import GlmOcrRemoteOptions


def _make_ok_response(text: str = "Hello") -> MagicMock:
    """Return a mock httpx response that looks like a successful vLLM reply."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"choices": [{"message": {"content": text}}]}
    return resp


def _make_http_error_response(status_code: int) -> MagicMock:
    """Return a mock httpx response that triggers raise_for_status."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = f"HTTP {status_code} error body"
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        f"HTTP {status_code}",
        request=MagicMock(),
        response=MagicMock(status_code=status_code, text=f"HTTP {status_code} error body"),
    )
    return resp


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
        mock_client.post.return_value = _make_ok_response("Hello World")

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
        mock_client.post.return_value = _make_ok_response("Extracted text content")

        img = Image.new("RGB", (50, 50))
        result = model._recognise_crop(img)

        assert result == "Extracted text content"

    def test_empty_choices_returns_empty_string(self, mock_model):
        model, mock_client = mock_model
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"choices": []}
        mock_client.post.return_value = resp

        img = Image.new("RGB", (50, 50))
        result = model._recognise_crop(img)

        assert result == ""

    def test_missing_content_returns_empty_string(self, mock_model):
        model, mock_client = mock_model
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"choices": [{"message": {}}]}
        mock_client.post.return_value = resp

        img = Image.new("RGB", (50, 50))
        result = model._recognise_crop(img)

        assert result == ""

    def test_raises_on_http_error(self, mock_model):
        model, mock_client = mock_model
        mock_client.post.return_value = _make_http_error_response(500)

        img = Image.new("RGB", (50, 50))
        with pytest.raises(httpx.HTTPStatusError):
            model._recognise_crop(img)

    def test_4xx_raises_http_status_error(self, mock_model):
        model, mock_client = mock_model
        mock_client.post.return_value = _make_http_error_response(400)

        img = Image.new("RGB", (50, 50))
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            model._recognise_crop(img)
        assert exc_info.value.response.status_code == 400


class TestRecogniseCropWithRetry:
    def test_success_on_first_attempt(self, mock_model):
        model, mock_client = mock_model
        mock_client.post.return_value = _make_ok_response("text")

        result = model._recognise_crop_with_retry(Image.new("RGB", (10, 10)))

        assert result == "text"
        assert mock_client.post.call_count == 1

    def test_4xx_not_retried(self, mock_model):
        """4xx is deterministic — must raise immediately without any retry."""
        model, mock_client = mock_model
        model.options.max_retries = 3
        mock_client.post.return_value = _make_http_error_response(400)

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            model._recognise_crop_with_retry(Image.new("RGB", (10, 10)))

        assert exc_info.value.response.status_code == 400
        assert mock_client.post.call_count == 1  # no retries

    def test_5xx_retried_up_to_max_retries(self, mock_model):
        """5xx triggers retries up to max_retries then re-raises."""
        model, mock_client = mock_model
        model.options.max_retries = 2
        model.options.retry_backoff_factor = 0.0
        mock_client.post.return_value = _make_http_error_response(503)

        with pytest.raises(httpx.HTTPStatusError):
            model._recognise_crop_with_retry(Image.new("RGB", (10, 10)))

        assert mock_client.post.call_count == 3  # 1 attempt + 2 retries

    def test_5xx_succeeds_after_retry(self, mock_model):
        """5xx on first attempt, success on second — returns text."""
        model, mock_client = mock_model
        model.options.max_retries = 2
        model.options.retry_backoff_factor = 0.0
        mock_client.post.side_effect = [
            _make_http_error_response(503),
            _make_ok_response("recovered"),
        ]

        result = model._recognise_crop_with_retry(Image.new("RGB", (10, 10)))

        assert result == "recovered"
        assert mock_client.post.call_count == 2

    def test_network_error_retried(self, mock_model):
        """Pure network errors (no response) are retried."""
        model, mock_client = mock_model
        model.options.max_retries = 1
        model.options.retry_backoff_factor = 0.0
        mock_client.post.side_effect = [
            httpx.ConnectError("connection refused"),
            _make_ok_response("after retry"),
        ]

        result = model._recognise_crop_with_retry(Image.new("RGB", (10, 10)))

        assert result == "after retry"


class TestProcessCrop:
    """Tests for GlmOcrRemoteModel._process_crop."""

    def test_none_image_returns_none_none(self, mock_model):
        model, _ = mock_model
        cell, error = model._process_crop(0, MagicMock(), None)
        assert cell is None
        assert error is None

    def test_success_returns_text_cell_and_no_error(self, mock_model):
        model, mock_client = mock_model
        mock_client.post.return_value = _make_ok_response("Some text")

        ocr_rect = MagicMock()
        ocr_rect.l, ocr_rect.t, ocr_rect.r, ocr_rect.b = 0, 0, 100, 50

        cell, error = model._process_crop(5, ocr_rect, Image.new("RGB", (100, 50)))

        assert cell is not None
        assert cell.text == "Some text"
        assert cell.index == 5
        assert cell.from_ocr is True
        assert cell.confidence == 1.0
        assert error is None

    def test_blank_text_returns_none_none(self, mock_model):
        model, mock_client = mock_model
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"choices": [{"message": {"content": "   \n  "}}]}
        mock_client.post.return_value = resp

        cell, error = model._process_crop(0, MagicMock(), Image.new("RGB", (10, 10)))

        assert cell is None
        assert error is None

    def test_4xx_returns_none_with_error_msg(self, mock_model):
        model, mock_client = mock_model
        mock_client.post.return_value = _make_http_error_response(400)

        cell, error = model._process_crop(3, MagicMock(), Image.new("RGB", (10, 10)))

        assert cell is None
        assert error is not None
        assert "400" in error
        assert "index=3" in error

    def test_5xx_after_retries_returns_none_with_error_msg(self, mock_model):
        model, mock_client = mock_model
        model.options.max_retries = 0
        mock_client.post.return_value = _make_http_error_response(500)

        cell, error = model._process_crop(7, MagicMock(), Image.new("RGB", (10, 10)))

        assert cell is None
        assert error is not None
        assert "index=7" in error

    def test_network_error_after_retries_returns_none_with_error_msg(self, mock_model):
        model, mock_client = mock_model
        model.options.max_retries = 0
        model.options.retry_backoff_factor = 0.0
        mock_client.post.side_effect = httpx.ConnectError("refused")

        cell, error = model._process_crop(2, MagicMock(), Image.new("RGB", (10, 10)))

        assert cell is None
        assert error is not None
        assert "index=2" in error


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

    def test_disabled_model_get_client_raises(self):
        from docling.datamodel.accelerator_options import AcceleratorOptions

        opts = GlmOcrRemoteOptions()
        model = GlmOcrRemoteModel(
            enabled=False,
            artifacts_path=None,
            options=opts,
            accelerator_options=AcceleratorOptions(),
        )

        with pytest.raises(RuntimeError, match="not enabled"):
            model._get_client()


class TestModelInit:
    def test_enabled_model_get_client_returns_client(self, mock_model):
        model, mock_client = mock_model
        assert model._get_client() is mock_client

    def test_scale_is_set(self, mock_model):
        model, _ = mock_model
        assert model.options.scale == pytest.approx(3.0)

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
        mock_client.post.return_value = _make_ok_response("Hello World")

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
        mock_client.post.return_value = _make_ok_response("Text")

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
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"choices": [{"message": {"content": "   \n  "}}]}
        mock_client.post.return_value = resp

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

        mock_client.post.side_effect = [
            _make_http_error_response(500),
            _make_http_error_response(500),
            _make_ok_response("Good"),
        ]

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

    def test_4xx_error_adds_error_item_to_conv_res(self, mock_model):
        """4xx crop failure must be recorded in conv_res.errors."""
        model, mock_client = mock_model
        model.options.max_concurrent_requests = 1

        page = MagicMock()
        page._backend.is_valid.return_value = True
        page._backend.get_page_image.return_value = Image.new("RGB", (10, 10))

        mock_client.post.return_value = _make_http_error_response(400)
        conv_res = MagicMock()
        conv_res.errors = []

        ocr_rect = self._make_ocr_rect()
        with (
            patch.object(model, "get_ocr_rects", return_value=[ocr_rect]),
            patch.object(model, "post_process_cells"),
            patch("docling_glm_ocr.model.TimeRecorder"),
        ):
            list(model(conv_res, [page]))

        assert len(conv_res.errors) == 1
        err = conv_res.errors[0]
        assert err.component_type == DoclingComponentType.MODEL
        assert "400" in err.error_message

    def test_successful_crop_does_not_add_errors(self, mock_model):
        model, mock_client = mock_model
        page = MagicMock()
        page._backend.is_valid.return_value = True
        page._backend.get_page_image.return_value = Image.new("RGB", (100, 100))

        mock_client.post.return_value = _make_ok_response("text")
        conv_res = MagicMock()
        conv_res.errors = []

        ocr_rect = self._make_ocr_rect()
        with (
            patch.object(model, "get_ocr_rects", return_value=[ocr_rect]),
            patch.object(model, "post_process_cells"),
            patch("docling_glm_ocr.model.TimeRecorder"),
        ):
            list(model(conv_res, [page]))

        assert conv_res.errors == []

    def test_multiple_rects_produce_multiple_cells(self, mock_model):
        model, mock_client = mock_model
        model.options.max_concurrent_requests = 1

        page = MagicMock()
        page._backend.is_valid.return_value = True
        page._backend.get_page_image.return_value = Image.new("RGB", (100, 100))

        rect1 = self._make_ocr_rect(left=0, top=0, right=50, bottom=50)
        rect2 = self._make_ocr_rect(left=50, top=0, right=100, bottom=50)
        mock_client.post.side_effect = [
            _make_ok_response("Cell one"),
            _make_ok_response("Cell two"),
        ]

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
        mock_client.post.return_value = _make_ok_response("T")

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
        mock_client.post.return_value = _make_ok_response("X")

        with (
            patch.object(model, "get_ocr_rects", return_value=[ocr_rect]),
            patch.object(model, "post_process_cells"),
            patch("docling_glm_ocr.model.TimeRecorder"),
        ):
            list(model(MagicMock(), [page]))

        page._backend.get_page_image.assert_called_once_with(scale=3, cropbox=ocr_rect)

    def test_scale_capped_for_large_crop(self, mock_model):
        """A crop whose native area would exceed max_image_pixels at configured scale
        must be fetched at a reduced scale."""
        model, mock_client = mock_model
        # 1000x1000 native crop @ scale=3 → 9_000_000 pixels > max_image_pixels=4_500_000
        # Expected capped scale: sqrt(4_500_000 / 1_000_000) ≈ 2.12
        model.options.max_image_pixels = 4_500_000
        page = MagicMock()
        page._backend.is_valid.return_value = True
        page._backend.get_page_image.return_value = Image.new("RGB", (100, 100))

        ocr_rect = self._make_ocr_rect(left=0, top=0, right=1000, bottom=1000, area=1_000_000)
        mock_client.post.return_value = _make_ok_response("text")

        with (
            patch.object(model, "get_ocr_rects", return_value=[ocr_rect]),
            patch.object(model, "post_process_cells"),
            patch("docling_glm_ocr.model.TimeRecorder"),
        ):
            list(model(MagicMock(), [page]))

        actual_scale = page._backend.get_page_image.call_args[1]["scale"]
        assert actual_scale < model.options.scale
        assert actual_scale == pytest.approx((4_500_000 / 1_000_000) ** 0.5, rel=1e-3)

    def test_scale_not_capped_for_small_crop(self, mock_model):
        """A crop within max_image_pixels at configured scale uses the full scale."""
        model, mock_client = mock_model
        # 100x100 native crop @ scale=3 → 90_000 pixels << max_image_pixels
        model.options.max_image_pixels = 4_500_000
        page = MagicMock()
        page._backend.is_valid.return_value = True
        page._backend.get_page_image.return_value = Image.new("RGB", (100, 100))

        ocr_rect = self._make_ocr_rect(left=0, top=0, right=100, bottom=100, area=10_000)
        mock_client.post.return_value = _make_ok_response("text")

        with (
            patch.object(model, "get_ocr_rects", return_value=[ocr_rect]),
            patch.object(model, "post_process_cells"),
            patch("docling_glm_ocr.model.TimeRecorder"),
        ):
            list(model(MagicMock(), [page]))

        actual_scale = page._backend.get_page_image.call_args[1]["scale"]
        assert actual_scale == model.options.scale

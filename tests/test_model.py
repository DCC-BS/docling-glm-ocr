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
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello World"}}]
        }
        mock_client.post.return_value = mock_response

        img = Image.new("RGB", (50, 50))
        result = model._recognise_crop(img)

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
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Extracted text content"}}]
        }
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

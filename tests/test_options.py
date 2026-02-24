"""Tests for GlmOcrRemoteOptions."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from docling_glm_ocr.options import GlmOcrRemoteOptions


class TestKind:
    def test_kind_value(self):
        assert GlmOcrRemoteOptions.kind == "glm-ocr-remote"

    def test_kind_is_class_var(self):
        opts = GlmOcrRemoteOptions()
        assert "kind" not in opts.model_fields


class TestDefaults:
    def test_default_api_url(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GLMOCR_REMOTE_OCR_API_URL", None)
            # Default is evaluated at class definition time, so we
            # just check the current default makes sense
            opts = GlmOcrRemoteOptions()
            assert "chat/completions" in opts.api_url

    def test_default_model_name(self):
        opts = GlmOcrRemoteOptions()
        assert opts.model_name == "zai-org/GLM-OCR"

    def test_default_timeout(self):
        opts = GlmOcrRemoteOptions()
        assert opts.timeout == 120

    def test_default_max_tokens(self):
        opts = GlmOcrRemoteOptions()
        assert opts.max_tokens == 16384

    def test_default_lang(self):
        opts = GlmOcrRemoteOptions()
        assert opts.lang == ["en"]

    def test_prompt_is_nonempty(self):
        opts = GlmOcrRemoteOptions()
        assert len(opts.prompt.strip()) > 0

    def test_default_scale(self):
        opts = GlmOcrRemoteOptions()
        assert opts.scale == 3.0

    def test_default_max_image_pixels(self):
        opts = GlmOcrRemoteOptions()
        assert opts.max_image_pixels == 4_500_000

    def test_default_max_concurrent_requests(self):
        opts = GlmOcrRemoteOptions()
        assert opts.max_concurrent_requests == 10

    def test_default_max_retries(self):
        opts = GlmOcrRemoteOptions()
        assert opts.max_retries == 3

    def test_default_retry_backoff_factor(self):
        opts = GlmOcrRemoteOptions()
        assert opts.retry_backoff_factor == 2.0


class TestCustomValues:
    def test_custom_api_url(self):
        opts = GlmOcrRemoteOptions(api_url="http://my-server:9000/v1/chat/completions")
        assert opts.api_url == "http://my-server:9000/v1/chat/completions"

    def test_custom_model_name(self):
        opts = GlmOcrRemoteOptions(model_name="my-org/my-model")
        assert opts.model_name == "my-org/my-model"

    def test_custom_timeout(self):
        opts = GlmOcrRemoteOptions(timeout=30)
        assert opts.timeout == 30

    def test_custom_max_tokens(self):
        opts = GlmOcrRemoteOptions(max_tokens=4096)
        assert opts.max_tokens == 4096

    def test_custom_prompt(self):
        opts = GlmOcrRemoteOptions(prompt="OCR this.")
        assert opts.prompt == "OCR this."

    def test_custom_lang(self):
        opts = GlmOcrRemoteOptions(lang=["en", "de"])
        assert opts.lang == ["en", "de"]


class TestValidation:
    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            GlmOcrRemoteOptions(unknown_field="value")


class TestInheritance:
    def test_is_ocr_options_subclass(self):
        from docling.datamodel.pipeline_options import OcrOptions

        assert issubclass(GlmOcrRemoteOptions, OcrOptions)

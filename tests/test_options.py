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
            opts = GlmOcrRemoteOptions()
            assert "chat/completions" in opts.api_url

    def test_default_model_name(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GLMOCR_REMOTE_OCR_MODEL_NAME", None)
            opts = GlmOcrRemoteOptions()
            assert opts.model_name == "zai-org/GLM-OCR"

    def test_default_timeout(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GLMOCR_REMOTE_OCR_TIMEOUT", None)
            opts = GlmOcrRemoteOptions()
            assert opts.timeout == 120

    def test_default_max_tokens(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GLMOCR_REMOTE_OCR_MAX_TOKENS", None)
            opts = GlmOcrRemoteOptions()
            assert opts.max_tokens == 16384

    def test_default_lang(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GLMOCR_REMOTE_OCR_LANG", None)
            opts = GlmOcrRemoteOptions()
            assert opts.lang == ["en"]

    def test_prompt_is_nonempty(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GLMOCR_REMOTE_OCR_PROMPT", None)
            opts = GlmOcrRemoteOptions()
            assert len(opts.prompt.strip()) > 0

    def test_default_scale(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GLMOCR_REMOTE_OCR_SCALE", None)
            opts = GlmOcrRemoteOptions()
            assert opts.scale == 3.0

    def test_default_max_image_pixels(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GLMOCR_REMOTE_OCR_MAX_IMAGE_PIXELS", None)
            opts = GlmOcrRemoteOptions()
            assert opts.max_image_pixels == 4_500_000

    def test_default_max_concurrent_requests(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GLMOCR_REMOTE_OCR_MAX_CONCURRENT_REQUESTS", None)
            opts = GlmOcrRemoteOptions()
            assert opts.max_concurrent_requests == 10

    def test_default_max_retries(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GLMOCR_REMOTE_OCR_MAX_RETRIES", None)
            opts = GlmOcrRemoteOptions()
            assert opts.max_retries == 3

    def test_default_retry_backoff_factor(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GLMOCR_REMOTE_OCR_RETRY_BACKOFF_FACTOR", None)
            opts = GlmOcrRemoteOptions()
            assert opts.retry_backoff_factor == 2.0


class TestEnvVars:
    def test_env_api_url(self):
        with patch.dict(os.environ, {"GLMOCR_REMOTE_OCR_API_URL": "http://remote:9000/v1/chat/completions"}):
            opts = GlmOcrRemoteOptions()
            assert opts.api_url == "http://remote:9000/v1/chat/completions"

    def test_env_model_name(self):
        with patch.dict(os.environ, {"GLMOCR_REMOTE_OCR_MODEL_NAME": "my-org/my-glm-model"}):
            opts = GlmOcrRemoteOptions()
            assert opts.model_name == "my-org/my-glm-model"

    def test_env_prompt(self):
        with patch.dict(os.environ, {"GLMOCR_REMOTE_OCR_PROMPT": "Just OCR it."}):
            opts = GlmOcrRemoteOptions()
            assert opts.prompt == "Just OCR it."

    def test_env_timeout(self):
        with patch.dict(os.environ, {"GLMOCR_REMOTE_OCR_TIMEOUT": "60"}):
            opts = GlmOcrRemoteOptions()
            assert opts.timeout == 60.0

    def test_env_max_tokens(self):
        with patch.dict(os.environ, {"GLMOCR_REMOTE_OCR_MAX_TOKENS": "8192"}):
            opts = GlmOcrRemoteOptions()
            assert opts.max_tokens == 8192

    def test_env_scale(self):
        with patch.dict(os.environ, {"GLMOCR_REMOTE_OCR_SCALE": "2.0"}):
            opts = GlmOcrRemoteOptions()
            assert opts.scale == 2.0

    def test_env_max_image_pixels(self):
        with patch.dict(os.environ, {"GLMOCR_REMOTE_OCR_MAX_IMAGE_PIXELS": "2000000"}):
            opts = GlmOcrRemoteOptions()
            assert opts.max_image_pixels == 2_000_000

    def test_env_max_concurrent_requests(self):
        with patch.dict(os.environ, {"GLMOCR_REMOTE_OCR_MAX_CONCURRENT_REQUESTS": "5"}):
            opts = GlmOcrRemoteOptions()
            assert opts.max_concurrent_requests == 5

    def test_env_max_retries(self):
        with patch.dict(os.environ, {"GLMOCR_REMOTE_OCR_MAX_RETRIES": "1"}):
            opts = GlmOcrRemoteOptions()
            assert opts.max_retries == 1

    def test_env_retry_backoff_factor(self):
        with patch.dict(os.environ, {"GLMOCR_REMOTE_OCR_RETRY_BACKOFF_FACTOR": "1.5"}):
            opts = GlmOcrRemoteOptions()
            assert opts.retry_backoff_factor == 1.5

    def test_env_lang_single(self):
        with patch.dict(os.environ, {"GLMOCR_REMOTE_OCR_LANG": "de"}):
            opts = GlmOcrRemoteOptions()
            assert opts.lang == ["de"]

    def test_env_lang_multiple(self):
        with patch.dict(os.environ, {"GLMOCR_REMOTE_OCR_LANG": "en,de,fr"}):
            opts = GlmOcrRemoteOptions()
            assert opts.lang == ["en", "de", "fr"]

    def test_env_var_overridden_by_explicit_arg_str(self):
        with patch.dict(os.environ, {"GLMOCR_REMOTE_OCR_MODEL_NAME": "env-org/env-model"}):
            opts = GlmOcrRemoteOptions(model_name="explicit-org/explicit-model")
            assert opts.model_name == "explicit-org/explicit-model"

    def test_env_var_overridden_by_explicit_arg_float(self):
        with patch.dict(os.environ, {"GLMOCR_REMOTE_OCR_TIMEOUT": "999"}):
            opts = GlmOcrRemoteOptions(timeout=30.0)
            assert opts.timeout == 30.0

    def test_env_var_overridden_by_explicit_arg_int(self):
        with patch.dict(os.environ, {"GLMOCR_REMOTE_OCR_MAX_RETRIES": "99"}):
            opts = GlmOcrRemoteOptions(max_retries=1)
            assert opts.max_retries == 1

    def test_env_var_overridden_by_explicit_arg_list(self):
        with patch.dict(os.environ, {"GLMOCR_REMOTE_OCR_LANG": "fr,es"}):
            opts = GlmOcrRemoteOptions(lang=["en"])
            assert opts.lang == ["en"]


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

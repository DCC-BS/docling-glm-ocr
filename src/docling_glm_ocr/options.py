"""Configuration model for the GLM-OCR remote OCR engine."""

from __future__ import annotations

import os
from typing import ClassVar, Literal

from docling.datamodel.pipeline_options import OcrOptions
from pydantic import ConfigDict, Field

_DEFAULT_PROMPT = (
    "Recognize the text in the image and output in Markdown format.\n"
    "Preserve the original layout (headings/paragraphs/tables/formulas).\n"
    "Do not fabricate content that does not exist in the image."
)

_DEFAULT_API_URL = "http://localhost:8001/v1/chat/completions"


class GlmOcrRemoteOptions(OcrOptions):
    """Options for the remote GLM-OCR OCR engine.

    The engine sends each page crop to a vLLM-hosted GLM-OCR model via its
    OpenAI-compatible chat completion endpoint and returns the recognised text.

    All options fall back to environment variables when not set explicitly,
    allowing configuration without code changes (e.g. in Docker / Compose
    deployments).

    Attributes:
        api_url: OpenAI-compatible chat completion URL of the vLLM server.
            Falls back to the ``GLMOCR_REMOTE_OCR_API_URL`` env var, then to
            a localhost default.
        model_name: ``model`` parameter sent in the chat completion request.
            Falls back to the ``GLMOCR_REMOTE_OCR_MODEL_NAME`` env var.
        prompt: Text prompt sent alongside each image crop.
            Falls back to the ``GLMOCR_REMOTE_OCR_PROMPT`` env var.
        timeout: HTTP request timeout in seconds per crop.
            Falls back to the ``GLMOCR_REMOTE_OCR_TIMEOUT`` env var.
        max_tokens: Maximum tokens for the chat completion response.
            Falls back to the ``GLMOCR_REMOTE_OCR_MAX_TOKENS`` env var.
        scale: Render scale applied to each crop before encoding.  Higher
            values improve recognition of small text at the cost of larger
            payloads.  Falls back to the ``GLMOCR_REMOTE_OCR_SCALE`` env var.
        max_image_pixels: Pixel budget per crop.  The scale is reduced
            automatically when a crop would exceed this limit, keeping the
            vLLM encoder token count within its pre-allocated cache.
            Falls back to the ``GLMOCR_REMOTE_OCR_MAX_IMAGE_PIXELS`` env var.
        max_concurrent_requests: Number of worker threads (and thus
            concurrent HTTP requests) used per page.
            Falls back to the ``GLMOCR_REMOTE_OCR_MAX_CONCURRENT_REQUESTS``
            env var.
        max_retries: How many times a 5xx or network error is retried before
            the crop is recorded as a conversion error.
            Falls back to the ``GLMOCR_REMOTE_OCR_MAX_RETRIES`` env var.
        retry_backoff_factor: Multiplier for the exponential back-off between
            retries.  Delay before retry *n* is ``retry_backoff_factor ** n``
            seconds.  Falls back to the
            ``GLMOCR_REMOTE_OCR_RETRY_BACKOFF_FACTOR`` env var.
        lang: Language hint list (passed to docling).  Falls back to the
            ``GLMOCR_REMOTE_OCR_LANG`` env var as a comma-separated string.
    """

    kind: ClassVar[Literal["glm-ocr-remote"]] = "glm-ocr-remote"

    lang: list[str] = Field(default_factory=lambda: os.environ.get("GLMOCR_REMOTE_OCR_LANG", "en").split(","))
    api_url: str = Field(default_factory=lambda: os.environ.get("GLMOCR_REMOTE_OCR_API_URL", _DEFAULT_API_URL))
    model_name: str = Field(default_factory=lambda: os.environ.get("GLMOCR_REMOTE_OCR_MODEL_NAME", "zai-org/GLM-OCR"))
    prompt: str = Field(default_factory=lambda: os.environ.get("GLMOCR_REMOTE_OCR_PROMPT", _DEFAULT_PROMPT))
    timeout: float = Field(default_factory=lambda: float(os.environ.get("GLMOCR_REMOTE_OCR_TIMEOUT", "120")))
    max_tokens: int = Field(default_factory=lambda: int(os.environ.get("GLMOCR_REMOTE_OCR_MAX_TOKENS", "16384")))
    scale: float = Field(default_factory=lambda: float(os.environ.get("GLMOCR_REMOTE_OCR_SCALE", "3.0")))
    max_image_pixels: int = Field(
        default_factory=lambda: int(os.environ.get("GLMOCR_REMOTE_OCR_MAX_IMAGE_PIXELS", "4500000"))
    )
    max_concurrent_requests: int = Field(
        default_factory=lambda: int(os.environ.get("GLMOCR_REMOTE_OCR_MAX_CONCURRENT_REQUESTS", "10"))
    )
    max_retries: int = Field(default_factory=lambda: int(os.environ.get("GLMOCR_REMOTE_OCR_MAX_RETRIES", "3")))
    retry_backoff_factor: float = Field(
        default_factory=lambda: float(os.environ.get("GLMOCR_REMOTE_OCR_RETRY_BACKOFF_FACTOR", "2.0"))
    )

    model_config = ConfigDict(extra="forbid")

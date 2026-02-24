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


class GlmOcrRemoteOptions(OcrOptions):
    """Options for the remote GLM-OCR OCR engine.

    The engine sends each page crop to a vLLM-hosted GLM-OCR model via its
    OpenAI-compatible chat completion endpoint and returns the recognised text.

    Attributes:
        api_url: OpenAI-compatible chat completion URL of the vLLM server.
            Falls back to the ``GLMOCR_REMOTE_OCR_API_URL`` env var, then to
            a localhost default.
        model_name: ``model`` parameter sent in the chat completion request.
        prompt: Text prompt sent alongside each image crop.
            Falls back to the ``GLMOCR_REMOTE_OCR_PROMPT`` env var.
        timeout: HTTP request timeout in seconds per crop.
        max_tokens: Maximum tokens for the chat completion response.
        scale: Render scale applied to each crop before encoding.  Higher
            values improve recognition of small text at the cost of larger
            payloads.
        max_image_pixels: Pixel budget per crop.  The scale is reduced
            automatically when a crop would exceed this limit, keeping the
            vLLM encoder token count within its pre-allocated cache.
        max_concurrent_requests: Number of worker threads (and thus
            concurrent HTTP requests) used per page.
        max_retries: How many times a 5xx or network error is retried before
            the crop is recorded as a conversion error.
        retry_backoff_factor: Multiplier for the exponential back-off between
            retries.  Delay before retry *n* is ``retry_backoff_factor ** n``
            seconds.
    """

    kind: ClassVar[Literal["glm-ocr-remote"]] = "glm-ocr-remote"

    lang: list[str] = Field(default_factory=lambda: ["en"])
    api_url: str = Field(
        default_factory=lambda: os.environ.get(
            "GLMOCR_REMOTE_OCR_API_URL",
            "http://localhost:8001/v1/chat/completions",
        )
    )
    model_name: str = "zai-org/GLM-OCR"
    prompt: str = Field(default_factory=lambda: os.environ.get("GLMOCR_REMOTE_OCR_PROMPT", _DEFAULT_PROMPT))
    timeout: float = 120
    max_tokens: int = 16384
    scale: float = 3.0
    max_image_pixels: int = 4_500_000
    max_concurrent_requests: int = 10
    max_retries: int = 3
    retry_backoff_factor: float = 2.0

    model_config = ConfigDict(extra="forbid")

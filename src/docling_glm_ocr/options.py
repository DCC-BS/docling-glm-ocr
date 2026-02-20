"""Configuration model for the GLM-OCR remote OCR engine."""

from __future__ import annotations

import os
from typing import ClassVar, Literal

from docling.datamodel.pipeline_options import OcrOptions
from pydantic import ConfigDict, Field


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
    """

    kind: ClassVar[Literal["glm-ocr-remote"]] = "glm-ocr-remote"

    lang: list[str] = Field(default_factory=lambda: ["en"])
    api_url: str = os.environ.get(
        "GLMOCR_REMOTE_OCR_API_URL",
        "http://localhost:8001/v1/chat/completions",
    )
    model_name: str = "zai-org/GLM-OCR"
    prompt: str = os.environ.get(
        "GLMOCR_REMOTE_OCR_PROMPT",
        """Recognize the text in the image and output in Markdown format.
      Preserve the original layout (headings/paragraphs/tables/formulas).
      Do not fabricate content that does not exist in the image.""",
    )
    timeout: float = 120
    max_tokens: int = 16384
    scale: float = 3.0
    max_concurrent_requests: int = 10
    max_retries: int = 3
    retry_backoff_factor: float = 2.0

    model_config = ConfigDict(extra="forbid")

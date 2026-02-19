"""GLM-OCR remote OCR engine for the docling standard pipeline.

Sends each page crop to a vLLM-hosted GLM-OCR model and returns the
recognised text as ``TextCell`` objects that docling merges with its
standard-pipeline output.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import TYPE_CHECKING

import httpx
from docling.models.base_ocr_model import BaseOcrModel
from docling.utils.profiling import TimeRecorder
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling_glm_ocr.options import GlmOcrRemoteOptions

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import Page
    from docling.datamodel.document import ConversionResult
    from docling.datamodel.pipeline_options import OcrOptions
    from PIL import Image

logger = logging.getLogger(__name__)


def _pil_to_base64_uri(image: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{b64}"


class GlmOcrRemoteModel(BaseOcrModel):
    """OCR engine that delegates recognition to a remote GLM-OCR vLLM server."""

    def __init__(
        self,
        *,
        enabled: bool,
        artifacts_path: Path | None,
        options: GlmOcrRemoteOptions,
        accelerator_options: AcceleratorOptions,
    ) -> None:
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: GlmOcrRemoteOptions
        self.scale = 3  # 72 dpi → 216 dpi

        if self.enabled:
            self._client = httpx.Client(timeout=self.options.timeout)
            logger.info(
                "GlmOcrRemoteModel initialised: api_url=%s  model=%s",
                self.options.api_url,
                self.options.model_name,
            )

    def _recognise_crop(self, image: Image.Image) -> str:
        """Send a single cropped image to the remote GLM-OCR endpoint."""
        data_uri = _pil_to_base64_uri(image)
        payload = {
            "model": self.options.model_name,
            "max_tokens": self.options.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.options.prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
        }
        resp = self._client.post(self.options.api_url, json=payload)
        resp.raise_for_status()
        choices = resp.json().get("choices", [])
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "")

    def __call__(
        self,
        conv_res: ConversionResult,
        page_batch: Iterable[Page],
    ) -> Iterable[Page]:
        """Run OCR on each page crop and yield pages with recognised text cells."""
        if not self.enabled:
            yield from page_batch
            return

        for page in page_batch:
            if page._backend is None or not page._backend.is_valid():  # noqa: SLF001
                yield page
                continue

            with TimeRecorder(conv_res, "ocr"):
                ocr_rects = self.get_ocr_rects(page)
                all_ocr_cells: list[TextCell] = []

                for cell_idx, ocr_rect in enumerate(ocr_rects):
                    if ocr_rect.area() == 0:
                        continue

                    high_res_image = page._backend.get_page_image(  # noqa: SLF001
                        scale=self.scale,
                        cropbox=ocr_rect,
                    )

                    try:
                        text = self._recognise_crop(high_res_image)
                    except httpx.HTTPError:
                        logger.exception("GLM-OCR remote call failed for crop %d", cell_idx)
                        continue

                    if not text.strip():
                        continue

                    all_ocr_cells.append(
                        TextCell(
                            index=cell_idx,
                            text=text,
                            orig=text,
                            from_ocr=True,
                            confidence=1.0,
                            rect=BoundingRectangle.from_bounding_box(
                                BoundingBox.from_tuple(
                                    coord=(
                                        ocr_rect.l,
                                        ocr_rect.t,
                                        ocr_rect.r,
                                        ocr_rect.b,
                                    ),
                                    origin=CoordOrigin.TOPLEFT,
                                )
                            ),
                        )
                    )

                self.post_process_cells(all_ocr_cells, page)

            yield page

    @classmethod
    def get_options_type(cls) -> type[OcrOptions]:
        """Return the options class for this OCR engine."""
        return GlmOcrRemoteOptions

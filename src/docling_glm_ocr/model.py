"""GLM-OCR remote OCR engine for the docling standard pipeline.

Sends each page crop to a vLLM-hosted GLM-OCR model and returns the
recognised text as ``TextCell`` objects that docling merges with its
standard-pipeline output.
"""

from __future__ import annotations

import base64
import concurrent.futures
import io
import logging
import time
from typing import TYPE_CHECKING

import httpx
from docling.datamodel.base_models import DoclingComponentType, ErrorItem
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

_HTTP_CLIENT_ERROR_MIN = 400
_HTTP_SERVER_ERROR_MIN = 500


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
        self.scale = self.options.scale

        if self.enabled:
            limits = httpx.Limits(
                max_connections=self.options.max_concurrent_requests,
                max_keepalive_connections=self.options.max_concurrent_requests,
            )
            self._client = httpx.Client(timeout=self.options.timeout, limits=limits)
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
        if resp.status_code >= _HTTP_CLIENT_ERROR_MIN:
            logger.error(
                "vLLM returned HTTP %d for %s: %s",
                resp.status_code,
                self.options.api_url,
                resp.text,
            )
        resp.raise_for_status()
        choices = resp.json().get("choices", [])
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "")

    def _recognise_crop_with_retry(self, image: Image.Image) -> str:
        """Send a single cropped image with retry logic.

        4xx responses are not retried — they represent a deterministic client error
        (e.g. image too large for the server's encoder cache) that will never resolve
        by repeating the same request. Only 5xx and network/timeout errors are retried.
        """
        max_retries = self.options.max_retries
        backoff = self.options.retry_backoff_factor

        for attempt in range(max_retries + 1):
            try:
                return self._recognise_crop(image)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < _HTTP_SERVER_ERROR_MIN:
                    # 4xx — deterministic client error, do not retry
                    raise
                # 5xx — server error, may be transient
                if attempt == max_retries:
                    logger.exception("GLM-OCR remote call failed after %d retries", max_retries)
                    raise
            except httpx.HTTPError:
                # Network / timeout errors — always retry
                if attempt == max_retries:
                    logger.exception("GLM-OCR remote call failed after %d retries", max_retries)
                    raise

            sleep_time = backoff**attempt
            logger.warning(
                "GLM-OCR remote call failed (attempt %d/%d). Retrying in %.1f seconds...",
                attempt + 1,
                max_retries,
                sleep_time,
            )
            time.sleep(sleep_time)
        return ""

    def _process_crop(
        self,
        cell_idx: int,
        ocr_rect: BoundingBox,
        image: Image.Image | None,
    ) -> tuple[TextCell | None, str | None]:
        if image is None:
            return None, None

        try:
            text = self._recognise_crop_with_retry(image)
        except httpx.HTTPStatusError as exc:
            # 4xx — deterministic rejection (e.g. image too large for encoder cache)
            msg = (
                f"OCR crop index={cell_idx} rejected by vLLM:"
                f" HTTP {exc.response.status_code} — {exc.response.text[:200]}"
            )
            return None, msg
        except httpx.HTTPError as exc:
            # 5xx / network — exhausted retries
            msg = f"OCR crop index={cell_idx} failed after {self.options.max_retries} retries: {exc}"
            return None, msg

        if not text.strip():
            return None, None

        return TextCell(
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
        ), None

    def _collect_crops(
        self,
        page: Page,
        ocr_rects: list[BoundingBox],
    ) -> list[tuple[int, BoundingBox, Image.Image | None]]:
        """Extract crop images for all OCR regions sequentially.

        Images are gathered in the calling thread to avoid thread-safety issues
        with the PDF backend. Scale is capped so the extracted image never
        exceeds ``max_image_pixels``, keeping the vLLM encoder token count
        within its pre-allocated cache.
        """
        crop_data: list[tuple[int, BoundingBox, Image.Image | None]] = []
        for cell_idx, ocr_rect in enumerate(ocr_rects):
            if ocr_rect.area() == 0:
                crop_data.append((cell_idx, ocr_rect, None))
                continue

            crop_w = ocr_rect.r - ocr_rect.l
            crop_h = ocr_rect.b - ocr_rect.t
            native_pixels = crop_w * crop_h
            if native_pixels > 0:
                max_safe_scale = (self.options.max_image_pixels / native_pixels) ** 0.5
                actual_scale = min(self.scale, max_safe_scale)
            else:
                actual_scale = self.scale
            if actual_scale < self.scale:
                logger.debug(
                    "Crop (%dx%d page-units) would exceed max_image_pixels=%d at scale=%.1f; reducing to scale=%.2f",
                    int(crop_w),
                    int(crop_h),
                    self.options.max_image_pixels,
                    self.scale,
                    actual_scale,
                )
            high_res_image = page._backend.get_page_image(  # noqa: SLF001
                scale=actual_scale,
                cropbox=ocr_rect,
            )
            crop_data.append((cell_idx, ocr_rect, high_res_image))
        return crop_data

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

                crop_data = self._collect_crops(page, ocr_rects)

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.options.max_concurrent_requests
                ) as executor:
                    futures = [
                        executor.submit(self._process_crop, cell_idx, ocr_rect, image)
                        for cell_idx, ocr_rect, image in crop_data
                    ]

                    for future in concurrent.futures.as_completed(futures):
                        cell, error_msg = future.result()
                        if cell is not None:
                            all_ocr_cells.append(cell)
                        if error_msg is not None:
                            conv_res.errors.append(
                                ErrorItem(
                                    component_type=DoclingComponentType.MODEL,
                                    module_name=__name__,
                                    error_message=error_msg,
                                )
                            )

                # Sort by index to maintain deterministic sequential order
                all_ocr_cells.sort(key=lambda c: c.index)

                self.post_process_cells(all_ocr_cells, page)

            yield page

    @classmethod
    def get_options_type(cls) -> type[OcrOptions]:
        """Return the options class for this OCR engine."""
        return GlmOcrRemoteOptions

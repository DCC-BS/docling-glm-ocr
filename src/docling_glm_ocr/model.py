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
import threading
import time
from typing import TYPE_CHECKING, Final

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

_HTTP_CLIENT_ERROR_MIN: Final = 400
_HTTP_SERVER_ERROR_MIN: Final = 500


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
        """Initialise the OCR engine.

        The httpx client is created lazily per thread on first use, so the
        model is safe to call from concurrent worker threads.  No client is
        created when ``enabled`` is ``False``.
        """
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: GlmOcrRemoteOptions
        self._local = threading.local()

        if self.enabled:
            logger.info(
                "GlmOcrRemoteModel initialised: api_url=%s  model=%s",
                self.options.api_url,
                self.options.model_name,
            )

    def _get_client(self) -> httpx.Client:
        """Return the thread-local httpx client, creating it on first use per thread.

        Each worker thread gets its own ``httpx.Client`` so concurrent requests
        from the ``ThreadPoolExecutor`` never share connection state.

        Raises:
            RuntimeError: If the model is not enabled.
        """
        if not self.enabled:
            msg = "GlmOcrRemoteModel is not enabled"
            raise RuntimeError(msg)
        if not hasattr(self._local, "client"):
            limits = httpx.Limits(
                max_connections=self.options.max_concurrent_requests,
                max_keepalive_connections=self.options.max_concurrent_requests,
            )
            self._local.client = httpx.Client(timeout=self.options.timeout, limits=limits)
        return self._local.client

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
        resp = self._get_client().post(self.options.api_url, json=payload)
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
                logger.debug("HTTP %d from vLLM on attempt %d", exc.response.status_code, attempt + 1)
            except httpx.HTTPError as exc:
                # Network / timeout errors — always retry
                if attempt == max_retries:
                    logger.exception("GLM-OCR remote call failed after %d retries", max_retries)
                    raise
                logger.debug("Network error on attempt %d: %s", attempt + 1, exc)

            sleep_time = backoff**attempt
            logger.warning(
                "GLM-OCR remote call failed (attempt %d/%d). Retrying in %.1f seconds...",
                attempt + 1,
                max_retries + 1,
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
        """OCR a single pre-extracted crop and convert it to a ``TextCell``.

        Designed to run inside a ``ThreadPoolExecutor`` worker.  Returns a
        ``(cell, None)`` pair on success, ``(None, error_message)`` when the
        remote call fails, or ``(None, None)`` when the image is empty or the
        model returns no text.

        The bounding box is converted from the page's native coordinate space
        to a ``BoundingRectangle`` with ``CoordOrigin.TOPLEFT`` as required by
        docling's text-cell contract.
        """
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
        backend = page._backend  # noqa: SLF001
        if backend is None:
            return []
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
                actual_scale = min(self.options.scale, max_safe_scale)
            else:
                actual_scale = self.options.scale
            if actual_scale < self.options.scale:
                logger.debug(
                    "Crop (%dx%d page-units) would exceed max_image_pixels=%d at scale=%.1f; reducing to scale=%.2f",
                    int(crop_w),
                    int(crop_h),
                    self.options.max_image_pixels,
                    self.options.scale,
                    actual_scale,
                )
            high_res_image = backend.get_page_image(
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

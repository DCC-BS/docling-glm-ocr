# docling-glm-ocr

A docling OCR plugin that delegates text recognition to a remote
[GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) model served by vLLM.

---

<p align="center">
  <a href="https://github.com/DCC-BS/docling-glm-ocr">GitHub</a>
  &nbsp;|&nbsp;
  <a href="https://pypi.org/project/docling-glm-ocr/">PyPI</a>
</p>

---

[![PyPI version](https://img.shields.io/pypi/v/docling-glm-ocr.svg)](https://pypi.org/project/docling-glm-ocr/)
[![Python versions](https://img.shields.io/pypi/pyversions/docling-glm-ocr.svg)](https://pypi.org/project/docling-glm-ocr/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/DCC-BS/docling-glm-ocr/blob/main/LICENSE)
[![CI](https://github.com/DCC-BS/docling-glm-ocr/actions/workflows/main.yml/badge.svg)](https://github.com/DCC-BS/docling-glm-ocr/actions/workflows/main.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![codecov](https://codecov.io/gh/DCC-BS/docling-glm-ocr/graph/badge.svg?token=LOPIP1UZOC)](https://codecov.io/gh/DCC-BS/docling-glm-ocr)

## Overview

`docling-glm-ocr` is a [docling](https://github.com/DS4SD/docling) plugin that
replaces the built-in OCR stage with a call to a remote
[GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) model hosted on a
[vLLM](https://github.com/vllm-project/vllm) server.

Each page crop is sent to the vLLM OpenAI-compatible chat completion endpoint
as a base64-encoded image. The model returns Markdown-formatted text which
docling merges back into the document structure.

The plugin registers itself under the `"glm-ocr-remote"` OCR engine key so it
can be selected per-request through docling or docling-serve without changing
application code.

## Requirements

- Python 3.13+
- A running vLLM server hosting `zai-org/GLM-OCR` (or any compatible model)

## Installation

```bash
# with uv (recommended)
uv add docling-glm-ocr

# with pip
pip install docling-glm-ocr
```

## Usage

### Python SDK

```python
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from docling_glm_ocr import GlmOcrRemoteOptions

pipeline_options = PdfPipelineOptions(
    allow_external_plugins=True,
    ocr_options=GlmOcrRemoteOptions(
        api_url="http://localhost:8001/v1/chat/completions",
        model_name="zai-org/GLM-OCR",
    ),
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
result = converter.convert("document.pdf")
print(result.document.export_to_markdown())
```

### docling-serve

Select the engine per-request via the standard API:

```bash
curl -X POST http://localhost:5001/v1/convert/source \
  -H 'Content-Type: application/json' \
  -d '{
    "options": {
      "ocr_engine": "glm-ocr-remote"
    },
    "sources": [{"kind": "http", "url": "https://arxiv.org/pdf/2501.17887"}]
  }'
```

The server must have `DOCLING_SERVE_ALLOW_EXTERNAL_PLUGINS=true` set so the
plugin is loaded automatically.

## Configuration

All options can be set via environment variables (useful for Docker / Compose
deployments) or programmatically via `GlmOcrRemoteOptions`.  Explicit
constructor arguments always take precedence over environment variables.

### Environment variables

| Variable | Description | Default |
|---|---|---|
| `GLMOCR_REMOTE_OCR_API_URL` | vLLM chat completion URL | `http://localhost:8001/v1/chat/completions` |
| `GLMOCR_REMOTE_OCR_MODEL_NAME` | Model name sent to vLLM | `zai-org/GLM-OCR` |
| `GLMOCR_REMOTE_OCR_PROMPT` | Text prompt sent with each image crop | see below |
| `GLMOCR_REMOTE_OCR_TIMEOUT` | HTTP timeout per crop (seconds) | `120` |
| `GLMOCR_REMOTE_OCR_MAX_TOKENS` | Max tokens per completion | `16384` |
| `GLMOCR_REMOTE_OCR_SCALE` | Image crop rendering scale | `3.0` |
| `GLMOCR_REMOTE_OCR_MAX_IMAGE_PIXELS` | Pixel budget per crop | `4500000` |
| `GLMOCR_REMOTE_OCR_MAX_CONCURRENT_REQUESTS` | Max concurrent API requests | `10` |
| `GLMOCR_REMOTE_OCR_MAX_RETRIES` | Max retry attempts for HTTP errors | `3` |
| `GLMOCR_REMOTE_OCR_RETRY_BACKOFF_FACTOR` | Exponential backoff factor for retries | `2.0` |
| `GLMOCR_REMOTE_OCR_LANG` | Comma-separated language hint(s) | `en` |

### `GlmOcrRemoteOptions`

All options can also be set programmatically via `GlmOcrRemoteOptions`:

| Option | Type | Description | Default |
|---|---|---|---|
| `api_url` | `str` | OpenAI-compatible chat completion URL | `GLMOCR_REMOTE_OCR_API_URL` env or `http://localhost:8001/v1/chat/completions` |
| `model_name` | `str` | Model name sent to vLLM | `GLMOCR_REMOTE_OCR_MODEL_NAME` env or `zai-org/GLM-OCR` |
| `prompt` | `str` | Text prompt for each image crop | `GLMOCR_REMOTE_OCR_PROMPT` env or default prompt |
| `timeout` | `float` | HTTP timeout per crop (seconds) | `GLMOCR_REMOTE_OCR_TIMEOUT` env or `120` |
| `max_tokens` | `int` | Max tokens per completion | `GLMOCR_REMOTE_OCR_MAX_TOKENS` env or `16384` |
| `scale` | `float` | Image crop rendering scale | `GLMOCR_REMOTE_OCR_SCALE` env or `3.0` |
| `max_image_pixels` | `int` | Pixel budget per crop | `GLMOCR_REMOTE_OCR_MAX_IMAGE_PIXELS` env or `4500000` |
| `max_concurrent_requests` | `int` | Max concurrent API requests | `GLMOCR_REMOTE_OCR_MAX_CONCURRENT_REQUESTS` env or `10` |
| `max_retries` | `int` | Max retry attempts for HTTP errors | `GLMOCR_REMOTE_OCR_MAX_RETRIES` env or `3` |
| `retry_backoff_factor` | `float` | Exponential backoff factor for retries | `GLMOCR_REMOTE_OCR_RETRY_BACKOFF_FACTOR` env or `2.0` |
| `lang` | `list[str]` | Language hint (passed to docling) | `GLMOCR_REMOTE_OCR_LANG` env (comma-separated) or `["en"]` |

Default prompt:

```
Recognize the text in the image and output in Markdown format.
Preserve the original layout (headings/paragraphs/tables/formulas).
Do not fabricate content that does not exist in the image.
```

## Architecture

```mermaid
flowchart LR
    subgraph docling
        Pipeline --> GlmOcrRemoteModel
    end

    subgraph vLLM
        GLMOCR["zai-org/GLM-OCR"]
    end

    GlmOcrRemoteModel -- "POST /v1/chat/completions\n(base64 image)" --> GLMOCR
    GLMOCR -- "Markdown text" --> GlmOcrRemoteModel
```

For each page the model:
1. Collects OCR regions from the docling layout analysis
2. Renders each region using the page backend (scale configurable, default 3×)
3. Encodes the crop as a base64 PNG data URI
4. POSTs concurrent chat completion requests to the vLLM endpoint (with retry logic)
5. Returns the recognised text as `TextCell` objects for docling to merge

## Starting a GLM-OCR vLLM server

```bash
docker run -d \
  --rm --name ocr-glm \
  --gpus device=0 \
  --ipc=host \
  -p 8001:8000 \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -e "HF_TOKEN=${HF_TOKEN}" \
  --entrypoint /bin/bash \
  vllm/vllm-openai:latest \
  -c "uv pip install --system --upgrade transformers && \
      exec vllm serve zai-org/GLM-OCR \
        --served-model-name zai-org/GLM-OCR \
        --port 8000 \
        --trust-remote-code"
```

The plugin will connect to `http://localhost:8001/v1/chat/completions` by default.

## Development

### Setup

```bash
git clone https://github.com/DCC-BS/docling-glm-ocr.git
cd docling-glm-ocr
make install
```

### Available commands

```
make install     Install dependencies and pre-commit hooks
make check       Run all quality checks (ruff lint, format, ty type check)
make test        Run tests with coverage report
make build       Build distribution packages
make publish     Publish to PyPI
```

### Running tests

```bash
make test
```

Tests are in `tests/` and use [pytest](https://pytest.org).
Coverage reports are generated at `coverage.xml` and printed to the terminal.

#### End-to-end tests

The e2e tests hit a real vLLM server and are **skipped by default**.
To run them, set the server URL and use the `e2e` marker:

```bash
GLMOCR_REMOTE_OCR_API_URL=http://localhost:8001/v1/chat/completions pytest -m e2e
```

### Code quality

This project uses:

- **[ruff](https://github.com/astral-sh/ruff)** – linting and formatting
- **[ty](https://github.com/astral-sh/ty)** – type checking
- **[pre-commit](https://pre-commit.com/)** – pre-commit hooks

Run all checks:

```bash
make check
```

### Releasing

Releases are published to PyPI automatically.
Update the version in `pyproject.toml`, then trigger the **Publish** workflow from GitHub Actions:

```
GitHub → Actions → Publish to PyPI → Run workflow
```

The workflow tags the commit, builds the package, and publishes to PyPI via trusted publishing.

## License

[MIT](LICENSE) © Data Competence Center Basel-Stadt

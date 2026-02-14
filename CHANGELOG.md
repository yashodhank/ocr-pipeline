# CHANGELOG


## v0.1.1 (2026-02-14)

### Bug Fixes

- **ci**: Skip PyPI publish when PYPI_API_TOKEN is not configured
  ([`e932377`](https://github.com/yashodhank/ocr-pipeline/commit/e932377d4b7b0e97dae9efb87f1b6ec593402650))

Root cause: pypa/gh-action-pypi-publish attempts OIDC trusted publishing first, which fails when no
  trusted publisher is configured on PyPI. The action doesn't fall back to token auth — it errors
  out entirely.

Fix: check for PYPI_API_TOKEN secret before attempting publish, emit a warning annotation instead of
  failing the workflow.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.1.0 (2026-02-14)

### Bug Fixes

- **ci**: Fix release workflow and improve CI pipeline
  ([`3b41ab5`](https://github.com/yashodhank/ocr-pipeline/commit/3b41ab5976437201af9ace6c47cbcfe5477c100b))

- Set build: false on semantic-release action to prevent build inside its Docker container where
  python -m build is unavailable - Remove build_command from pyproject.toml semantic_release config
  - Reuse CI workflow from release via workflow_call - Remove redundant test/system-dep steps from
  release job - Add pip caching to both workflows - Drop Windows from CI matrix (Linux + macOS only)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Features

- Add Docker and Docker Compose for local development
  ([`d22ac71`](https://github.com/yashodhank/ocr-pipeline/commit/d22ac713d4038ebe0cc116921931473c8dbdc35a))

Dockerfile with python:3.12-slim, tesseract, poppler, and language packs. Docker Compose with
  ocr-pipeline and ollama services for easy local testing.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- Multilingual OCR pipeline with automatic backend selection, parallel processing, and confidence
  scoring.
  ([`a5722ab`](https://github.com/yashodhank/ocr-pipeline/commit/a5722ab9de1e29735bb130709c7810ec0c61c4df))

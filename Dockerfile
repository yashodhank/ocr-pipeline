FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-hin \
    tesseract-ocr-mar \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY ocr_engine.py ocr_cli.py ocr_mcp_server.py ./

RUN pip install --no-cache-dir .

ENTRYPOINT ["ocr-pipeline"]

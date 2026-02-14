#!/usr/bin/env python3
"""
OCR MCP Server — Expose OCR pipeline as MCP tools.

Tools:
    ocr_single_file  — OCR one file (PDF/image)
    ocr_batch_files   — OCR multiple files
    list_ocr_models   — List available Ollama models
    get_ocr_status    — System status check

Usage:
    python ocr_mcp_server.py              # streamable-http (default)
    python ocr_mcp_server.py --stdio      # stdio transport (for Claude Desktop)
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP

from ocr_engine import (
    BatchResult,
    FileResult,
    OCRConfig,
    PageResult,
    collect_inputs,
    detect_best_model,
    list_available_models,
    ocr_batch,
    ocr_file,
    validate_languages,
    PYTESSERACT_AVAILABLE,
    PDF2IMAGE_AVAILABLE,
    HTTPX_AVAILABLE,
    DEFAULT_OLLAMA_BASE_URL,
)

mcp = FastMCP("ocr-pipeline")

# Track active jobs for status queries
_active_jobs: dict[str, dict] = {}


def _page_result_to_dict(p: PageResult) -> dict:
    return {
        "page": p.page,
        "status": p.status,
        "text": p.text,
        "backend": p.backend,
        "confidence": round(p.confidence, 1),
        "error": p.error,
        "processing_time": round(p.processing_time, 2),
    }


def _file_result_to_dict(r: FileResult) -> dict:
    return {
        "input_path": r.input_path,
        "status": r.status,
        "text": r.text,
        "backend": r.backend,
        "error": r.error,
        "processing_time": round(r.processing_time, 2),
        "pages": [_page_result_to_dict(p) for p in r.pages],
    }


@mcp.tool()
async def ocr_single_file(
    file_path: str,
    lang: str = "eng",
    backend: str = "auto",
    model: Optional[str] = None,
) -> dict:
    """OCR a single file (PDF or image). Returns extracted text with confidence scores.

    Args:
        file_path: Path to the PDF or image file.
        lang: Languages separated by +, e.g. "eng", "hin", "eng+hin+mar".
        backend: OCR backend — "auto", "ollama", or "pytesseract".
        model: Ollama model name (auto-detected if omitted).
    """
    languages = lang.split("+")
    config = OCRConfig(
        backend=backend,
        model=model,
        languages=languages,
    )

    job_id = f"single_{int(time.time())}_{Path(file_path).name}"
    _active_jobs[job_id] = {"status": "running", "file": file_path, "started": time.time()}

    try:
        result = await ocr_file(Path(file_path), config)
        _active_jobs[job_id]["status"] = "completed"
        return _file_result_to_dict(result)
    except Exception as e:
        _active_jobs[job_id]["status"] = "error"
        _active_jobs[job_id]["error"] = str(e)
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def ocr_batch_files(
    file_paths: list[str],
    lang: str = "eng",
    backend: str = "auto",
    model: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """OCR multiple files (PDFs and/or images) in batch.

    Args:
        file_paths: List of file paths or directories to process.
        lang: Languages separated by +, e.g. "eng+hin".
        backend: OCR backend — "auto", "ollama", or "pytesseract".
        model: Ollama model name (auto-detected if omitted).
        output_dir: Directory to write results (optional).
    """
    languages = lang.split("+")
    config = OCRConfig(
        backend=backend,
        model=model,
        languages=languages,
    )

    job_id = f"batch_{int(time.time())}_{len(file_paths)}files"
    _active_jobs[job_id] = {"status": "running", "files": len(file_paths), "started": time.time()}

    try:
        batch = await ocr_batch(file_paths, config)
        _active_jobs[job_id]["status"] = "completed"

        result = {
            "total_files": batch.total_files,
            "successful_files": batch.successful_files,
            "total_time": round(batch.total_time, 2),
            "results": [_file_result_to_dict(r) for r in batch.results],
        }

        if output_dir:
            from ocr_engine import write_output
            write_output(batch.results, Path(output_dir), fmt="json", indent=True)
            result["output_dir"] = output_dir

        return result
    except Exception as e:
        _active_jobs[job_id]["status"] = "error"
        _active_jobs[job_id]["error"] = str(e)
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def list_ocr_models() -> dict:
    """List available Ollama vision/OCR models.

    Returns a list of models with their names and sizes.
    """
    models = await list_available_models()
    return {
        "models": [
            {
                "name": m.get("model", m.get("name", "unknown")),
                "size": m.get("size", 0),
                "modified_at": m.get("modified_at", ""),
            }
            for m in models
        ],
        "count": len(models),
    }


@mcp.tool()
async def get_ocr_status(job_id: Optional[str] = None) -> dict:
    """Check OCR system status and optionally query a specific job.

    Args:
        job_id: Optional job ID to check status of a specific OCR job.
    """
    status = {
        "pytesseract_available": PYTESSERACT_AVAILABLE,
        "pdf2image_available": PDF2IMAGE_AVAILABLE,
        "httpx_available": HTTPX_AVAILABLE,
        "active_jobs": len([j for j in _active_jobs.values() if j.get("status") == "running"]),
        "total_jobs": len(_active_jobs),
    }

    # Check Ollama connectivity
    models = await list_available_models()
    status["ollama_available"] = len(models) > 0
    status["ollama_model_count"] = len(models)

    # Check tessdata languages
    lang_status = validate_languages(["eng", "hin", "mar"])
    status["tessdata_languages"] = lang_status

    if job_id and job_id in _active_jobs:
        status["job"] = _active_jobs[job_id]

    return status


def main():
    parser = argparse.ArgumentParser(description="OCR Pipeline MCP Server")
    parser.add_argument("--stdio", action="store_true", help="Use stdio transport (for Claude Desktop)")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port (default: 8000)")
    args = parser.parse_args()

    if args.stdio:
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()

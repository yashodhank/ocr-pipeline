#!/usr/bin/env python3
"""
OCR Engine — Core library for the OCR pipeline.

Provides async OCR processing with Ollama (vision LLMs) and PyTesseract backends,
language support for English, Hindi, and Marathi (Devanagari), parallel page
processing, confidence scoring, and automatic fallback.

Public API:
    ocr_file(file_path, config, progress_callback) -> FileResult
    ocr_batch(input_paths, config, recursive, progress_callback) -> BatchResult
    list_available_models(base_url) -> list[dict]
    detect_best_model(languages, base_url) -> str | None
    collect_inputs(inputs, recursive) -> list[Path]
    write_output(results, output_dir, fmt, indent)
"""

import asyncio
import base64
import io
import json
import logging
import os
import re
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# --- Imports with graceful fallbacks ---

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

try:
    import pdf2image
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# macOS-specific: use spawn for multiprocessing safety
import multiprocessing
if sys.platform == "darwin":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # already set

log = logging.getLogger("ocr-engine")

# --- Constants ---

SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}

TEMP_DIR = Path(tempfile.gettempdir()) / "ocr_pipeline"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_LANGUAGES = {"eng", "hin", "mar"}

DEVANAGARI_PREFERRED_MODELS = ["qwen2.5-vl", "qwen3-vl", "llava", "glm-ocr"]

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

# --- Language-aware OCR prompts ---

LANGUAGE_PROMPTS: Dict[str, str] = {
    "eng": (
        "Extract all visible text from this document image. "
        "Preserve structure: headings, paragraphs, lists, tables. "
        "Return plain text only — no explanations, no markdown formatting."
    ),
    "hin": (
        "इस दस्तावेज़ छवि से सभी दृश्य पाठ निकालें। "
        "Hindi text in Devanagari script (हिन्दी) — extract in original Devanagari. "
        "Do NOT transliterate to Latin script. Preserve all Devanagari characters exactly. "
        "Preserve structure: headings, paragraphs, lists. Return plain text only."
    ),
    "mar": (
        "या दस्तऐवज प्रतिमेतील सर्व दृश्य मजकूर काढा। "
        "Marathi text in Devanagari script (मराठी) — extract in original Devanagari. "
        "Do NOT transliterate to Latin script. Preserve all Devanagari characters exactly. "
        "Preserve structure: headings, paragraphs, lists. Return plain text only."
    ),
}


def _build_ocr_prompt(languages: List[str]) -> str:
    """Build a combined OCR prompt for the given language set."""
    if len(languages) == 1 and languages[0] in LANGUAGE_PROMPTS:
        return LANGUAGE_PROMPTS[languages[0]]

    has_devanagari = any(l in ("hin", "mar") for l in languages)
    has_english = "eng" in languages

    parts = [
        "Extract all visible text from this document image. "
        "Preserve structure: headings, paragraphs, lists, tables."
    ]

    if has_devanagari and has_english:
        lang_names = []
        if "eng" in languages:
            lang_names.append("English")
        if "hin" in languages:
            lang_names.append("Hindi (हिन्दी)")
        if "mar" in languages:
            lang_names.append("Marathi (मराठी)")
        parts.append(
            f"This document contains text in: {', '.join(lang_names)}. "
            "Extract ALL text in its original script. "
            "Keep Devanagari text in Devanagari — do NOT transliterate to Latin."
        )
    elif has_devanagari:
        parts.append(
            "Extract text in original Devanagari script. "
            "Do NOT transliterate to Latin script."
        )

    parts.append("Return plain text only — no explanations.")
    return " ".join(parts)


# --- Data Models ---

@dataclass
class OCRConfig:
    """Configuration for an OCR run."""
    backend: str = "auto"  # "auto", "ollama", "pytesseract"
    model: Optional[str] = None
    languages: List[str] = field(default_factory=lambda: ["eng"])
    temperature: float = 0.1
    max_retries: int = 3
    max_dim: int = 2048
    timeout: int = 120
    workers: int = 3
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    save_debug_img: bool = False


@dataclass
class PageResult:
    """Result from processing a single page."""
    page: int
    status: str  # "success", "error", "timeout"
    text: str = ""
    backend: str = "none"
    confidence: float = 0.0
    error: Optional[str] = None
    processing_time: float = 0.0


@dataclass
class FileResult:
    """Result from processing a single file."""
    input_path: str
    status: str  # "success", "error", "partial"
    text: str = ""
    backend: str = "none"
    pages: List[PageResult] = field(default_factory=list)
    error: Optional[str] = None
    processing_time: float = 0.0


@dataclass
class BatchResult:
    """Result from processing a batch of files."""
    results: List[FileResult] = field(default_factory=list)
    total_files: int = 0
    successful_files: int = 0
    total_time: float = 0.0


# --- Image Utilities ---

def _load_image_safe(img_path: Path) -> Optional["Image.Image"]:
    """Load image safely with PIL."""
    try:
        img = Image.open(img_path)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        return img
    except Exception as e:
        log.error("Failed to load image %s: %s", img_path, e)
        return None


def _preprocess_image(page: "Image.Image", max_dim: int = 2048) -> "Image.Image":
    """Preprocess image: resize, grayscale (PIL-only for macOS safety)."""
    w, h = page.size
    scale = min(1.0, max_dim / max(w, h))
    if scale < 1.0:
        page = page.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    if page.mode != "L":
        page = page.convert("L")
    return page


def _encode_image_to_base64(img: "Image.Image", fmt: str = "PNG") -> str:
    """Convert PIL Image to base64-encoded string."""
    buf = io.BytesIO()
    if img.mode == "L":
        img = img.convert("RGB")
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# --- Input Collection ---

def collect_inputs(inputs: List[str], recursive: bool = False) -> List[Path]:
    """Collect all valid image/PDF files from input list."""
    file_paths = []
    for inp in inputs:
        p = Path(inp)
        if p.is_file():
            if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                file_paths.append(p)
            else:
                log.warning("Unsupported file extension: %s", p.name)
        elif p.is_dir():
            pattern = "**/*" if recursive else "*"
            for f in p.glob(pattern):
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                    file_paths.append(f)
        else:
            log.warning("Skipping missing path: %s", inp)
    return sorted(set(file_paths))


# --- Language Validation ---

def validate_languages(languages: List[str]) -> Dict[str, bool]:
    """Check which tessdata language packs are available."""
    result = {}
    for lang in languages:
        if lang not in SUPPORTED_LANGUAGES:
            result[lang] = False
            continue
        if not PYTESSERACT_AVAILABLE:
            result[lang] = False
            continue
        try:
            available = pytesseract.get_languages()
            result[lang] = lang in available
        except Exception:
            result[lang] = False
    return result


# --- Model Detection ---

async def list_available_models(base_url: str = DEFAULT_OLLAMA_BASE_URL) -> List[dict]:
    """List available Ollama models."""
    if not HTTPX_AVAILABLE:
        return []
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{base_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            return data.get("models", [])
    except Exception as e:
        log.error("Failed to list Ollama models: %s", e)
        return []


def detect_best_model(
    languages: List[str],
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
) -> Optional[str]:
    """Synchronously detect the best OCR model from Ollama.

    Prefers Devanagari-capable models when Hindi/Marathi requested.
    """
    try:
        import ollama as ollama_lib
        models_resp = ollama_lib.list()
        models = models_resp.get("models", [])
        model_names = [m.get("model", m.get("name", "")) for m in models]
    except Exception as e:
        log.error("Failed to list Ollama models: %s", e)
        return None

    if not model_names:
        return None

    has_devanagari = any(l in ("hin", "mar") for l in languages)

    if has_devanagari:
        for preferred in DEVANAGARI_PREFERRED_MODELS:
            for name in model_names:
                if preferred in name.lower():
                    return name

    # General preference: vision/OCR models
    for name in model_names:
        lower = name.lower()
        if any(tag in lower for tag in ["ocr", "vision", "llava", "qwen", "glm"]):
            return name

    # Last resort: first model with multimodal hints
    for name in model_names:
        lower = name.lower()
        if any(tag in lower for tag in ["llava", "vl"]):
            return name

    return None


# --- Async Ollama Client ---

async def _ollama_ocr_async(
    image: "Image.Image",
    model: str,
    languages: List[str],
    temperature: float,
    timeout: int,
    max_dim: int,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
) -> Tuple[str, bool]:
    """Async Ollama OCR using httpx. Returns (text, success)."""
    if not HTTPX_AVAILABLE:
        return ("", False)

    proc_img = _preprocess_image(image, max_dim=max_dim)
    b64_img = _encode_image_to_base64(proc_img)
    prompt = _build_ocr_prompt(languages)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [b64_img],
            }
        ],
        "options": {"temperature": temperature},
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("message", {}).get("content", "").strip()
            return (text, bool(text))
    except httpx.TimeoutException:
        log.warning("Ollama OCR timed out after %ds", timeout)
        return ("", False)
    except Exception as e:
        log.warning("Ollama OCR error: %s", e)
        return ("", False)


# --- PyTesseract OCR ---

def _pytesseract_ocr(
    image: "Image.Image",
    languages: List[str],
    max_dim: int = 2048,
) -> Tuple[str, float]:
    """Run pytesseract OCR with language support. Returns (text, confidence)."""
    if not PYTESSERACT_AVAILABLE:
        return ("", 0.0)

    proc_img = _preprocess_image(image, max_dim=max_dim)
    lang_str = "+".join(languages)
    config = "--psm 6 --oem 3"

    try:
        text = pytesseract.image_to_string(proc_img, lang=lang_str, config=config).strip()
    except Exception as e:
        log.error("PyTesseract OCR failed: %s", e)
        return ("", 0.0)

    # Confidence scoring via image_to_data
    confidence = 0.0
    try:
        data = pytesseract.image_to_data(proc_img, lang=lang_str, config=config, output_type=pytesseract.Output.DICT)
        confs = [int(c) for c in data.get("conf", []) if str(c).lstrip("-").isdigit() and int(c) > 0]
        if confs:
            confidence = sum(confs) / len(confs)
    except Exception:
        pass

    return (text, confidence)


# Helper for ProcessPoolExecutor (must be picklable — accepts bytes, not PIL Image)
def _pytesseract_ocr_from_bytes(
    img_bytes: bytes,
    languages: List[str],
    max_dim: int,
) -> Tuple[str, float]:
    """Run pytesseract on image bytes (for use in ProcessPoolExecutor)."""
    from PIL import Image as _Image
    img = _Image.open(io.BytesIO(img_bytes))
    return _pytesseract_ocr(img, languages, max_dim)


def _image_to_bytes(img: "Image.Image") -> bytes:
    """Serialize PIL Image to PNG bytes for pickling."""
    buf = io.BytesIO()
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(buf, format="PNG")
    return buf.getvalue()


# --- Post-Processing ---

def _post_process_ocr(text: str, languages: List[str]) -> str:
    """Clean up OCR output."""
    if not text:
        return text

    # Normalize multiple blank lines
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    # Remove trailing whitespace on lines
    text = "\n".join(line.rstrip() for line in text.splitlines())

    # Fix hyphenated line breaks (English only)
    has_english = "eng" in languages
    if has_english:
        text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    return text.strip()


def _estimate_confidence(text: str, languages: List[str], backend: str) -> float:
    """Heuristic confidence scoring based on text characteristics."""
    if not text or not text.strip():
        return 0.0

    score = 50.0  # base

    # Length bonus
    length = len(text.strip())
    if length > 100:
        score += 15.0
    elif length > 20:
        score += 5.0

    # Character class consistency
    has_devanagari = any(l in ("hin", "mar") for l in languages)
    if has_devanagari:
        devanagari_chars = sum(1 for c in text if "\u0900" <= c <= "\u097F")
        total_alpha = sum(1 for c in text if c.isalpha())
        if total_alpha > 0:
            ratio = devanagari_chars / total_alpha
            if ratio > 0.5:
                score += 20.0  # good Devanagari content
            elif ratio > 0.1:
                score += 10.0
    else:
        ascii_alpha = sum(1 for c in text if c.isascii() and c.isalpha())
        total_chars = sum(1 for c in text if not c.isspace())
        if total_chars > 0:
            ratio = ascii_alpha / total_chars
            if ratio > 0.6:
                score += 20.0
            elif ratio > 0.3:
                score += 10.0

    # Backend bonus
    if "ollama" in backend:
        score += 10.0
    elif backend == "pytesseract":
        score += 5.0

    return min(score, 100.0)


# --- Page-Level Parallel Processing ---

async def _process_single_page_ollama(
    page_num: int,
    image: "Image.Image",
    config: OCRConfig,
    semaphore: asyncio.Semaphore,
) -> PageResult:
    """Process a single page with Ollama (async, with semaphore for concurrency control)."""
    start = time.monotonic()
    async with semaphore:
        retries = 0
        current_max_dim = config.max_dim
        while retries <= config.max_retries:
            text, success = await _ollama_ocr_async(
                image=image,
                model=config.model,
                languages=config.languages,
                temperature=config.temperature,
                timeout=config.timeout,
                max_dim=current_max_dim,
                base_url=config.ollama_base_url,
            )
            if success:
                text = _post_process_ocr(text, config.languages)
                backend = f"ollama/{config.model}"
                confidence = _estimate_confidence(text, config.languages, backend)
                return PageResult(
                    page=page_num,
                    status="success",
                    text=text,
                    backend=backend,
                    confidence=confidence,
                    processing_time=time.monotonic() - start,
                )
            retries += 1
            # OOM mitigation: halve max_dim on retry
            current_max_dim = max(512, current_max_dim // 2)
            log.warning("Page %d: Ollama retry %d/%d (max_dim=%d)", page_num, retries, config.max_retries, current_max_dim)

    # Ollama failed — try pytesseract fallback
    log.info("Page %d: Ollama failed, falling back to pytesseract", page_num)
    return _process_single_page_pytesseract(page_num, image, config, start_time=start)


def _process_single_page_pytesseract(
    page_num: int,
    image: "Image.Image",
    config: OCRConfig,
    start_time: Optional[float] = None,
) -> PageResult:
    """Process a single page with pytesseract (sync)."""
    start = start_time or time.monotonic()
    try:
        text, tess_confidence = _pytesseract_ocr(image, config.languages, config.max_dim)
        text = _post_process_ocr(text, config.languages)
        confidence = tess_confidence if tess_confidence > 0 else _estimate_confidence(text, config.languages, "pytesseract")
        return PageResult(
            page=page_num,
            status="success",
            text=text,
            backend="pytesseract",
            confidence=confidence,
            processing_time=time.monotonic() - start,
        )
    except Exception as e:
        return PageResult(
            page=page_num,
            status="error",
            error=str(e),
            processing_time=time.monotonic() - start,
        )


async def _process_pages_parallel(
    pages: List["Image.Image"],
    config: OCRConfig,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> List[PageResult]:
    """Process pages in parallel using the configured backend."""
    total = len(pages)
    results: List[PageResult] = []

    use_ollama = config.backend in ("auto", "ollama") and config.model is not None

    if use_ollama:
        # Async gather with semaphore for I/O-bound Ollama calls
        semaphore = asyncio.Semaphore(config.workers)
        tasks = []
        for idx, page in enumerate(pages):
            tasks.append(_process_single_page_ollama(idx + 1, page, config, semaphore))

        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            if progress_callback:
                progress_callback(result.page, total, result.status)
    else:
        # CPU-bound pytesseract: use ThreadPoolExecutor (tesseract shells out, releases GIL)
        loop = asyncio.get_event_loop()
        with ProcessPoolExecutor(max_workers=min(config.workers, os.cpu_count() or 1)) as pool:
            futures = []
            page_bytes_list = []
            for idx, page in enumerate(pages):
                img_bytes = _image_to_bytes(page)
                page_bytes_list.append(img_bytes)
                future = loop.run_in_executor(
                    pool,
                    _pytesseract_ocr_from_bytes,
                    img_bytes,
                    config.languages,
                    config.max_dim,
                )
                futures.append((idx + 1, future))

            for page_num, future in futures:
                start = time.monotonic()
                try:
                    text, confidence = await future
                    text = _post_process_ocr(text, config.languages)
                    if confidence <= 0:
                        confidence = _estimate_confidence(text, config.languages, "pytesseract")
                    result = PageResult(
                        page=page_num,
                        status="success",
                        text=text,
                        backend="pytesseract",
                        confidence=confidence,
                        processing_time=time.monotonic() - start,
                    )
                except Exception as e:
                    result = PageResult(
                        page=page_num,
                        status="error",
                        error=str(e),
                        processing_time=time.monotonic() - start,
                    )
                results.append(result)
                if progress_callback:
                    progress_callback(page_num, total, result.status)

    # Sort by page number
    results.sort(key=lambda r: r.page)
    return results


# --- Public API ---

async def ocr_file(
    file_path: Path,
    config: OCRConfig,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> FileResult:
    """OCR a single file (PDF or image). Returns FileResult."""
    start = time.monotonic()
    file_path = Path(file_path)
    result = FileResult(input_path=str(file_path), status="error")

    if not file_path.exists():
        result.error = f"File not found: {file_path}"
        return result

    try:
        # Load pages
        if file_path.suffix.lower() == ".pdf":
            if not PDF2IMAGE_AVAILABLE:
                result.error = "pdf2image not available — PDF support disabled"
                return result
            try:
                pages = pdf2image.convert_from_path(str(file_path), dpi=300, fmt="tiff")
            except Exception as e:
                result.error = f"PDF conversion failed: {e}"
                return result
        else:
            img = _load_image_safe(file_path)
            if img is None:
                result.error = "Failed to load image"
                return result
            pages = [img]

        # Auto-detect model if needed
        if config.backend in ("auto", "ollama") and config.model is None:
            config.model = detect_best_model(config.languages, config.ollama_base_url)
            if config.model is None and config.backend == "ollama":
                result.error = "Ollama backend selected but no model found"
                return result

        # Process pages
        page_results = await _process_pages_parallel(pages, config, progress_callback)
        result.pages = page_results

        # Combine text
        texts = [pr.text for pr in page_results if pr.text]
        result.text = "\n\n".join(texts)

        # Determine overall status
        success_count = sum(1 for pr in page_results if pr.status == "success")
        if success_count == len(page_results):
            result.status = "success"
        elif success_count > 0:
            result.status = "partial"
        else:
            result.status = "error"

        # Determine backend used
        backends = {pr.backend for pr in page_results if pr.backend != "none"}
        if len(backends) > 1:
            result.backend = "mixed"
        elif backends:
            result.backend = backends.pop()

    except Exception as e:
        result.error = str(e)
        log.error("Unexpected error processing %s: %s", file_path, e)

    result.processing_time = time.monotonic() - start
    return result


async def ocr_batch(
    input_paths: List[str],
    config: OCRConfig,
    recursive: bool = False,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> BatchResult:
    """OCR a batch of files/directories. Returns BatchResult."""
    start = time.monotonic()
    files = collect_inputs(input_paths, recursive=recursive)
    batch = BatchResult(total_files=len(files))

    for idx, file_path in enumerate(files):
        log.info("Processing file %d/%d: %s", idx + 1, len(files), file_path.name)
        if progress_callback:
            progress_callback(str(file_path), idx + 1, len(files))

        file_result = await ocr_file(file_path, config)
        batch.results.append(file_result)

        if file_result.status in ("success", "partial"):
            batch.successful_files += 1

    batch.total_time = time.monotonic() - start
    return batch


# --- Output ---

def write_output(
    results: List[FileResult],
    output_dir: Path,
    fmt: str = "json",
    indent: bool = False,
) -> None:
    """Write OCR results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build JSON data
    json_data = {
        "results": [
            {
                "input": r.input_path,
                "status": r.status,
                "text": r.text,
                "backend": r.backend,
                "processing_time": round(r.processing_time, 2),
                "error": r.error,
                "pages": [
                    {
                        "page": p.page,
                        "status": p.status,
                        "text": p.text,
                        "backend": p.backend,
                        "confidence": round(p.confidence, 1),
                        "error": p.error,
                        "processing_time": round(p.processing_time, 2),
                    }
                    for p in r.pages
                ],
            }
            for r in results
        ],
        "summary": {
            "total_files": len(results),
            "successful_files": sum(1 for r in results if r.status in ("success", "partial")),
            "total_pages": sum(len(r.pages) for r in results),
            "successful_pages": sum(
                sum(1 for p in r.pages if p.status == "success")
                for r in results
            ),
        },
    }

    # Always write JSON
    json_path = output_dir / "ocr_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2 if indent else None, ensure_ascii=False)
    log.info("Wrote JSON: %s", json_path)

    # Write plain text
    if fmt in ("txt", "md"):
        ext = fmt
        txt_path = output_dir / f"ocr_output.{ext}"
        with open(txt_path, "w", encoding="utf-8") as f:
            for r in results:
                if r.status in ("success", "partial"):
                    f.write(f"File: {r.input_path}\n")
                    f.write("---\n")
                    f.write(r.text)
                    f.write("\n\n")
        log.info("Wrote %s: %s", fmt.upper(), txt_path)

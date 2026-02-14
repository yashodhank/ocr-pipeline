#!/usr/bin/env python3
"""
Batch OCR pipeline with LLM fallback, robust error handling, and macOS-optimized for:
- Apple Silicon (M1/M2/M3) compatibility
- Safe multiprocessing (spawn start method)
- Fallback to PIL if cv2 not available
- Homebrew-installed poppler/tesseract
"""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# --- Imports with graceful fallbacks ---
try:
    import pdf2image
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    print("⚠️ pdf2image not installed — PDF support will be disabled.", file=sys.stderr)

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️ ollama not installed — Ollama fallback will be disabled.", file=sys.stderr)

try:
    import cv2  # OpenCV (optional)
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("ℹ️ cv2 not found — falling back to PIL for image preprocessing.", file=sys.stderr)

try:
    from PIL import Image
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError as e:
    PYTESSERACT_AVAILABLE = False
    print(f"❌ PyTesseract or Pillow not available: {e}", file=sys.stderr)
    sys.exit(1)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = lambda x, *a, **kw: x  # no-op fallback

# macOS-specific: use spawn for multiprocessing safety
import multiprocessing
if sys.platform == "darwin":
    multiprocessing.set_start_method("spawn", force=True)

# --- Constants ---
MAX_IMAGE_SIZE_OCR = 2048  # max dimension for OCR
TEMP_DIR = Path(tempfile.gettempdir()) / "batch_ocr"

# Ensure temp dir exists
TEMP_DIR.mkdir(parents=True, exist_ok=True)


# --- Logging (lightweight wrapper) ---
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# --- Helper Functions ---

def _collect_inputs(inputs: List[str], recursive: bool = False) -> List[Path]:
    """Collect all valid image/PDF files from input list."""
    file_paths = []
    for inp in inputs:
        p = Path(inp)
        if p.is_file():
            if p.suffix.lower() in {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"}:
                file_paths.append(p)
            else:
                log.warning(f"Unsupported file extension: {p.name}")
        elif p.is_dir():
            pattern = "**/*" if recursive else "*"
            for f in p.glob(pattern):
                if f.is_file() and f.suffix.lower() in {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"}:
                    file_paths.append(f)
        else:
            log.warning(f"Skipping missing path: {inp}")
    return sorted(file_paths)


def _load_image_safe(img_path: Path) -> Optional[Image.Image]:
    """Load image safely, with OpenCV fallback (optional) or pure PIL."""
    try:
        img = Image.open(img_path)
        # Convert to RGB if necessary (e.g., palette/P mode)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        return img
    except Exception as e:
        log.error(f"Failed to load image {img_path}: {e}")
        return None


def _detect_ocr_model() -> Optional[str]:
    """Try to auto-detect the best OCR model from `ollama list`."""
    if not OLLAMA_AVAILABLE:
        return None
    try:
        models = ollama.list()
        for model in models.get("models", []):
            name = model.get("model", model.get("name", ""))
            if any(tag in name.lower() for tag in ["ocr", "vision", "llava"]):
                return name
        # fallback: return first multimodal model
        for model in models.get("models", []):
            name = model.get("model", model.get("name", ""))
            if "llava" in name or "gpt-4o" in name:  # user may have added OpenAI-compatible proxy
                return name
        return None
    except Exception as e:
        log.error(f"Failed to list Ollama models: {e}")
        return None


def _preprocess_image(page: Image.Image, max_dim: int = MAX_IMAGE_SIZE_OCR) -> Image.Image:
    """Preprocess image: resize, grayscale, denoise (PIL-only for macOS safety)."""
    # Resize if too large
    w, h = page.size
    scale = min(1.0, max_dim / max(w, h))
    if scale < 1.0:
        page = page.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Grayscale & denoise (basic)
    if page.mode != "L":
        page = page.convert("L")
    # No heavy denoising (preserve text detail); simple threshold can help
    # (But avoid hard binarization — let OCR decide)
    return page


def _run_ocr_on_page(
    page_num: int,
    total_pages: int,
    page: Image.Image,
    model: Optional[str],
    temp: float,
    use_ollama: bool,
    max_retries: int,
    max_dim: int,
) -> Dict[str, Any]:
    """Run OCR on a single page (preprocessed), with retry + fallback."""
    res: Dict[str, Any] = {
        "page": page_num,
        "status": "error",
        "text": "",
        "backend": "none",
        "error": None,
    }

    try:
        # Preprocess
        proc_img = _preprocess_image(page, max_dim=max_dim)

        # Fallback if no Ollama or model unavailable
        if not use_ollama or not OLLAMA_AVAILABLE or not model:
            log.info(f"📄 Page {page_num}/{total_pages}: using pytesseract fallback")
            res["backend"] = "pytesseract"
            try:
                res["text"] = pytesseract.image_to_string(proc_img).strip()
                res["status"] = "success"
            except Exception as e:
                res["error"] = str(e)
                log.error(f"❌ PyTesseract failed on page {page_num}: {e}")
            return res

        # Try Ollama OCR (multimodal)
        retries = 0
        ocr_prompt = (
            "Describe this document page in detail. "
            "Extract all visible text (including headings, paragraphs, tables, footers), "
            "preserving structure (newlines, bullets, indentation). "
            "Return plain text only — no explanations."
        )
        while retries <= max_retries:
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=TEMP_DIR) as tmp:
                    proc_img.save(tmp.name, format="PNG")
                    img_path = tmp.name

                try:
                    response = ollama.chat(
                        model=model,
                        messages=[
                            {
                                "role": "user",
                                "content": ocr_prompt,
                                "images": [img_path],
                            }
                        ],
                        options={"temperature": temp},
                    )
                    ocr_text = response["message"]["content"].strip()
                    if ocr_text:
                        res["text"] = ocr_text
                        res["status"] = "success"
                        res["backend"] = f"ollama/{model}"
                        log.info(
                            f"✅ Page {page_num}/{total_pages}: Ollama OCR succeeded ({len(ocr_text)} chars)"
                        )
                        return res
                finally:
                    try:
                        os.unlink(img_path)
                    except Exception:
                        pass

            except Exception as e:
                retries += 1
                if retries > max_retries:
                    res["error"] = str(e)
                    log.error(
                        f"❌ Ollama OCR failed on page {page_num} after {max_retries} retries: {e}"
                    )
                    res["backend"] = "none"
                else:
                    log.warning(
                        f"⚠️ Page {page_num}: Ollama OCR failed, retry {retries}/{max_retries}... ({e})"
                    )

        return res

    except Exception as e:
        res["error"] = str(e)
        log.error(f"❌ Unexpected error on page {page_num}: {e}")
        return res


def _process_single_file(
    img_path: Path,
    model: Optional[str],
    lang: str,
    use_ollama: bool,
    temp: float,
    max_retries: int,
    output_dir: Path,
    save_debug_img: bool,
) -> Dict[str, Any]:
    """Process a single file (PDF or image) into OCR result."""
    result: Dict[str, Any] = {
        "status": "error",
        "text": "",
        "backend": "none",
        "pages": [],
        "input": str(img_path),
        "error": None,
    }

    try:
        # PDF handling
        if img_path.suffix.lower() == ".pdf":
            if not PDF2IMAGE_AVAILABLE:
                result["error"] = "pdf2image not available — PDF support disabled"
                log.error(f"❌ PDF processing skipped for {img_path.name}: pdf2image not installed")
                return result

            # Convert PDF to images safely
            try:
                # Use high DPI for better OCR
                pages = pdf2image.convert_from_path(
                    str(img_path),
                    dpi=300,  # good for OCR
                    fmt="tiff",  # lossless
                    # path_to_poppler=None,  # Let poppler autodetect or set explicitly
                )
            except Exception as e:
                result["error"] = f"PDF conversion failed: {e}"
                log.error(f"❌ PDF conversion failed for {img_path.name}: {e}")
                return result

        else:
            # Load single image
            img = _load_image_safe(img_path)
            if img is None:
                result["error"] = "Failed to load image"
                return result
            pages = [img]

        result["status"] = "success"

        # Process pages
        for idx, page in enumerate(pages):
            page_res = _run_ocr_on_page(
                page_num=idx + 1,
                total_pages=len(pages),
                page=page,
                model=model,
                temp=temp,
                use_ollama=use_ollama,
                max_retries=max_retries,
                max_dim=MAX_IMAGE_SIZE_OCR,
            )
            result["pages"].append(page_res)
            result["text"] += page_res["text"] + "\n\n"

            # Save debug image if requested (optional)
            if save_debug_img:
                debug_path = (
                    output_dir / f"{img_path.stem}_page_{idx + 1}.png"
                )
                _preprocess_image(page).save(debug_path, "PNG")

        result["text"] = result["text"].strip()
        result["backend"] = "mixed" if any(
            p["backend"] == "pytesseract" for p in result["pages"]
        ) else (
            result["pages"][0]["backend"] if result["pages"] else "none"
        )

    except Exception as e:
        result["error"] = str(e)
        log.error(f"❌ Unexpected error processing {img_path}: {e}")

    return result


def _write_output(
    results: List[Dict[str, Any]],
    output_dir: Path,
    format: str,
    indent: bool,
) -> None:
    """Write OCR results to files."""
    # Prepare JSON output
    json_data = {
        "results": results,
        "summary": {
            "total_files": len(results),
            "successful_files": sum(1 for r in results if r["status"] == "success"),
            "total_pages": sum(len(r["pages"]) for r in results),
            "total_pages_success": sum(
                sum(1 for p in r["pages"] if p["status"] == "success")
                for r in results
            ),
        },
    }

    # Write JSON (always)
    json_path = output_dir / "ocr_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=indent, ensure_ascii=False)
    log.info(f"✅ Wrote JSON summary to {json_path}")

    # Write plain text (concatenated)
    if format == "txt":
        txt_path = output_dir / "ocr_output.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for result in results:
                if result["status"] == "success":
                    f.write(f"File: {result['input']}\n")
                    f.write("---\n")
                    f.write(result["text"])
                    f.write("\n\n")
        log.info(f"✅ Wrote plain text to {txt_path}")


# --- Main Entry Point ---

def main():
    parser = argparse.ArgumentParser(
        description="Batch OCR using Ollama or fallback to pytesseract."
    )
    parser.add_argument("inputs", nargs="+", help="Input files or directories")
    parser.add_argument("--recursive", action="store_true", help="Recursively process directories")
    parser.add_argument(
        "--ocr-backend", 
        dest="ocr_backend", 
        choices=["auto", "ollama", "pytesseract"], 
        default="auto", 
        help="OCR backend"
    )
    #parser.spawn("ocr-backend", choices=["auto", "ollama", "pytesseract"], default="auto", help="OCR backend")
    parser.add_argument("--ocr-backend", dest="ocr_backend", choices=["auto", "ollama", "pytesseract"], default="auto", help="OCR backend")
    parser.add_argument("--model", type=str, default=None, help="Ollama model name (if auto, tries to detect)")
    parser.add_argument("--temp", type=float, default=0.1, help="Temperature for Ollama")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries for Ollama OCR")
    parser.add_argument("--output-dir", type=str, default="ocr_output", help="Output directory")
    parser.add_argument("--save-debug-img", action="store_true", help="Save preprocessed images")
    parser.add_argument("--format", choices=["json", "txt"], default="json", help="Output format")
    parser.add_argument("--indent", action="store_true", help="Pretty-print JSON output")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: auto)")
    args = parser.parse_args()

    # Validate inputs
    input_files = _collect_inputs(args.inputs, recursive=args.recursive)
    if not input_files:
        log.error("❌ No valid input files found.")
        return

    # Ensure output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect OCR backend
    ollama_backend = False
    model = None
    if args.ocr_backend == "auto":
        model = _detect_ocr_model()
        ollama_backend = model is not None
    elif args.ocr_backend == "ollama":
        model = args.model or _detect_ocr_model()
        if not model:
            log.error("❌ Ollama backend selected but no model found.")
            return
        ollama_backend = True
    elif args.ocr_backend == "pytesseract":
        ollama_backend = False

    log.info(f"✅ OCR backend: {'ollama' if ollama_backend else 'pytesseract'}")
    if model:
        log.info(f"🧠 Model: {model}")

    # Process files in parallel
    log.info(f"🔄 Processing {len(input_files)} files with {args.workers or multiprocessing.cpu_count()} workers...")
    results: List[Dict[str, Any]] = []

    # Use ProcessPoolExecutor for multiprocessing
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _process_single_file,
                f,
                model,
                "eng",  # lang placeholder (pytesseract not used in fallback)
                ollama_backend,
                args.temp,
                args.max_retries,
                output_dir,
                args.save_debug_img,
            ): f for f in input_files
        }

        if TQDM_AVAILABLE:
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    log.error(f"❌ Exception in worker: {e}")
        else:
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    log.error(f"❌ Exception in worker: {e}")

    # Write results
    _write_output(results, output_dir, format=args.format, indent=args.indent)


if __name__ == "__main__":
    main()

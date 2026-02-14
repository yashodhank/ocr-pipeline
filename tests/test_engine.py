"""Unit tests for ocr_engine.py."""

import io
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from ocr_engine import (
    OCRConfig,
    PageResult,
    FileResult,
    BatchResult,
    _build_ocr_prompt,
    _encode_image_to_base64,
    _estimate_confidence,
    _post_process_ocr,
    _preprocess_image,
    collect_inputs,
    validate_languages,
    SUPPORTED_EXTENSIONS,
)


# --- collect_inputs ---

class TestCollectInputs:
    def test_single_file(self, tmp_path):
        f = tmp_path / "doc.pdf"
        f.touch()
        result = collect_inputs([str(f)])
        assert len(result) == 1
        assert result[0] == f

    def test_single_image(self, tmp_path):
        f = tmp_path / "photo.jpg"
        f.touch()
        result = collect_inputs([str(f)])
        assert len(result) == 1

    def test_unsupported_extension(self, tmp_path):
        f = tmp_path / "readme.txt"
        f.touch()
        result = collect_inputs([str(f)])
        assert len(result) == 0

    def test_directory_non_recursive(self, tmp_path):
        (tmp_path / "a.pdf").touch()
        (tmp_path / "b.png").touch()
        (tmp_path / "c.txt").touch()
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "d.jpg").touch()
        result = collect_inputs([str(tmp_path)], recursive=False)
        assert len(result) == 2  # a.pdf, b.png (not sub/d.jpg, not c.txt)

    def test_directory_recursive(self, tmp_path):
        (tmp_path / "a.pdf").touch()
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.jpg").touch()
        result = collect_inputs([str(tmp_path)], recursive=True)
        assert len(result) == 2

    def test_tif_extension_supported(self, tmp_path):
        f = tmp_path / "scan.tif"
        f.touch()
        result = collect_inputs([str(f)])
        assert len(result) == 1

    def test_tiff_extension_supported(self, tmp_path):
        f = tmp_path / "scan.tiff"
        f.touch()
        result = collect_inputs([str(f)])
        assert len(result) == 1

    def test_missing_path(self):
        result = collect_inputs(["/nonexistent/path/file.pdf"])
        assert len(result) == 0

    def test_deduplication(self, tmp_path):
        f = tmp_path / "doc.pdf"
        f.touch()
        result = collect_inputs([str(f), str(f)])
        assert len(result) == 1

    def test_webp_supported(self, tmp_path):
        f = tmp_path / "image.webp"
        f.touch()
        result = collect_inputs([str(f)])
        assert len(result) == 1


# --- _build_ocr_prompt ---

class TestBuildOCRPrompt:
    def test_english_only(self):
        prompt = _build_ocr_prompt(["eng"])
        assert "Extract all visible text" in prompt
        assert "Devanagari" not in prompt

    def test_hindi_only(self):
        prompt = _build_ocr_prompt(["hin"])
        assert "हिन्दी" in prompt
        assert "transliterate" in prompt.lower()

    def test_marathi_only(self):
        prompt = _build_ocr_prompt(["mar"])
        assert "मराठी" in prompt
        assert "transliterate" in prompt.lower()

    def test_multilingual_eng_hin(self):
        prompt = _build_ocr_prompt(["eng", "hin"])
        assert "English" in prompt
        assert "Hindi" in prompt
        assert "Devanagari" in prompt

    def test_multilingual_eng_hin_mar(self):
        prompt = _build_ocr_prompt(["eng", "hin", "mar"])
        assert "English" in prompt
        assert "Hindi" in prompt
        assert "Marathi" in prompt

    def test_unknown_language_still_works(self):
        prompt = _build_ocr_prompt(["fra"])
        assert "Extract all visible text" in prompt


# --- _preprocess_image ---

class TestPreprocessImage:
    def test_resize_large_image(self):
        img = Image.new("RGB", (4000, 3000))
        result = _preprocess_image(img, max_dim=2048)
        assert max(result.size) <= 2048

    def test_no_resize_small_image(self):
        img = Image.new("RGB", (800, 600))
        result = _preprocess_image(img, max_dim=2048)
        assert result.size[0] == 800
        assert result.size[1] == 600

    def test_converts_to_grayscale(self):
        img = Image.new("RGB", (100, 100))
        result = _preprocess_image(img)
        assert result.mode == "L"

    def test_already_grayscale(self):
        img = Image.new("L", (100, 100))
        result = _preprocess_image(img)
        assert result.mode == "L"

    def test_aspect_ratio_preserved(self):
        img = Image.new("RGB", (4000, 2000))
        result = _preprocess_image(img, max_dim=2000)
        # 4000:2000 = 2:1 ratio should be preserved
        w, h = result.size
        assert abs(w / h - 2.0) < 0.01


# --- _post_process_ocr ---

class TestPostProcessOCR:
    def test_empty_text(self):
        assert _post_process_ocr("", ["eng"]) == ""

    def test_normalize_blank_lines(self):
        text = "Hello\n\n\n\n\nWorld"
        result = _post_process_ocr(text, ["eng"])
        assert result == "Hello\n\nWorld"

    def test_trailing_whitespace(self):
        text = "Hello   \nWorld   "
        result = _post_process_ocr(text, ["eng"])
        assert result == "Hello\nWorld"

    def test_hyphen_fix_english(self):
        text = "for-\nmat"
        result = _post_process_ocr(text, ["eng"])
        assert result == "format"

    def test_no_hyphen_fix_for_hindi(self):
        text = "for-\nmat"
        result = _post_process_ocr(text, ["hin"])
        # Hindi-only mode should NOT fix English hyphenation
        assert "for" in result  # hyphen not removed

    def test_devanagari_preserved(self):
        text = "यह एक परीक्षण है"
        result = _post_process_ocr(text, ["hin"])
        assert "परीक्षण" in result


# --- _estimate_confidence ---

class TestEstimateConfidence:
    def test_empty_text(self):
        assert _estimate_confidence("", ["eng"], "pytesseract") == 0.0

    def test_whitespace_only(self):
        assert _estimate_confidence("   \n  ", ["eng"], "pytesseract") == 0.0

    def test_long_english_text(self):
        text = "This is a long English text that should score well. " * 5
        score = _estimate_confidence(text, ["eng"], "pytesseract")
        assert score > 50.0

    def test_ollama_backend_bonus(self):
        text = "Some text"
        score_ollama = _estimate_confidence(text, ["eng"], "ollama/model")
        score_tess = _estimate_confidence(text, ["eng"], "pytesseract")
        assert score_ollama > score_tess

    def test_devanagari_text_hindi(self):
        text = "यह एक परीक्षण है जो हिंदी में लिखा गया है और काफी लम्बा है"
        score = _estimate_confidence(text, ["hin"], "pytesseract")
        assert score > 50.0

    def test_max_100(self):
        text = "A" * 1000
        score = _estimate_confidence(text, ["eng"], "ollama/model")
        assert score <= 100.0


# --- validate_languages ---

class TestValidateLanguages:
    def test_unsupported_language(self):
        result = validate_languages(["xyz"])
        assert result["xyz"] is False

    @patch("ocr_engine.PYTESSERACT_AVAILABLE", False)
    def test_no_pytesseract(self):
        result = validate_languages(["eng"])
        assert result["eng"] is False

    @patch("ocr_engine.PYTESSERACT_AVAILABLE", True)
    @patch("ocr_engine.pytesseract")
    def test_eng_available(self, mock_tess):
        mock_tess.get_languages.return_value = ["eng", "osd"]
        result = validate_languages(["eng"])
        assert result["eng"] is True

    @patch("ocr_engine.PYTESSERACT_AVAILABLE", True)
    @patch("ocr_engine.pytesseract")
    def test_hin_missing(self, mock_tess):
        mock_tess.get_languages.return_value = ["eng", "osd"]
        result = validate_languages(["hin"])
        assert result["hin"] is False


# --- _encode_image_to_base64 ---

class TestEncodeImage:
    def test_rgb_image(self):
        img = Image.new("RGB", (10, 10), color="red")
        b64 = _encode_image_to_base64(img)
        assert len(b64) > 0

    def test_grayscale_image(self):
        img = Image.new("L", (10, 10))
        b64 = _encode_image_to_base64(img)
        assert len(b64) > 0  # should convert L -> RGB before encoding


# --- Data models ---

class TestDataModels:
    def test_ocr_config_defaults(self):
        config = OCRConfig()
        assert config.backend == "auto"
        assert config.languages == ["eng"]
        assert config.temperature == 0.1
        assert config.max_retries == 3

    def test_page_result_defaults(self):
        pr = PageResult(page=1, status="success")
        assert pr.text == ""
        assert pr.confidence == 0.0

    def test_file_result_defaults(self):
        fr = FileResult(input_path="/test.pdf", status="success")
        assert fr.pages == []
        assert fr.error is None

    def test_batch_result_defaults(self):
        br = BatchResult()
        assert br.total_files == 0
        assert br.results == []


# --- Supported Extensions ---

class TestSupportedExtensions:
    def test_pdf_supported(self):
        assert ".pdf" in SUPPORTED_EXTENSIONS

    def test_tif_supported(self):
        assert ".tif" in SUPPORTED_EXTENSIONS

    def test_tiff_supported(self):
        assert ".tiff" in SUPPORTED_EXTENSIONS

    def test_webp_supported(self):
        assert ".webp" in SUPPORTED_EXTENSIONS

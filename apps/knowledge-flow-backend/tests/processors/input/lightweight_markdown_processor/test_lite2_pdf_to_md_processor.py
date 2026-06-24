# Copyright Thales 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from knowledge_flow_backend.core.processors.input.lightweight_markdown_processor.lite2_pdf_to_md_processor import (
    LiteMarkdownOptions,
    LitePdfMarkdownProcessor,
    LitePdfToMdExtractor,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

ASSETS = Path(__file__).parent.parent.parent.parent / "assets"
SAMPLE_PDF = ASSETS / "sample.pdf"


def _make_fitz_doc(page_count: int = 3, metadata: dict | None = None):
    """Build a minimal fitz.Document-shaped mock."""
    doc = MagicMock()
    doc.page_count = page_count
    doc.metadata = metadata or {"title": "Test Doc", "author": "Fred", "subject": "", "producer": "", "creator": ""}
    return doc


def _make_pymupdf4llm_pages(texts: list[str]) -> list[dict]:
    return [{"text": t, "md": t, "metadata": {"page_number": i + 1}} for i, t in enumerate(texts)]


# ── extract_file_metadata ─────────────────────────────────────────────────────

class TestExtractFileMetadata:
    """Regression tests for the use-after-close bug fixed in QUALITY-04 Phase 1."""

    def test_page_count_read_before_close(self, tmp_path):
        """page_count must be captured before doc.close() — regression for QUALITY-04."""
        processor = LitePdfMarkdownProcessor()
        fake_doc = _make_fitz_doc(page_count=7)

        with patch("knowledge_flow_backend.core.processors.input.lightweight_markdown_processor.lite2_pdf_to_md_processor.fitz.open", return_value=fake_doc):
            result = processor.extract_file_metadata(tmp_path / "test.pdf")

        assert result["page_count"] == 7
        fake_doc.close.assert_called_once()

    def test_metadata_fields_returned(self, tmp_path):
        """All expected metadata keys must be present in the returned dict."""
        processor = LitePdfMarkdownProcessor()
        fake_doc = _make_fitz_doc(metadata={"title": "MyDoc", "author": "Alice", "subject": "S", "producer": "P", "creator": "C"})

        with patch("knowledge_flow_backend.core.processors.input.lightweight_markdown_processor.lite2_pdf_to_md_processor.fitz.open", return_value=fake_doc):
            result = processor.extract_file_metadata(tmp_path / "test.pdf")

        assert result["title"] == "MyDoc"
        assert result["author"] == "Alice"
        assert result["extras"]["pdf.subject"] == "S"
        assert result["extras"]["pdf.producer"] == "P"
        assert result["extras"]["pdf.creator"] == "C"

    def test_returns_error_key_on_exception(self, tmp_path):
        """On fitz.open failure, returns dict with 'error' key — no exception raised."""
        processor = LitePdfMarkdownProcessor()
        with patch("knowledge_flow_backend.core.processors.input.lightweight_markdown_processor.lite2_pdf_to_md_processor.fitz.open", side_effect=Exception("corrupt")):
            result = processor.extract_file_metadata(tmp_path / "broken.pdf")

        assert "error" in result
        assert result["document_name"] == "broken.pdf"


# ── check_file_validity ───────────────────────────────────────────────────────

class TestCheckFileValidity:
    def test_valid_pdf_returns_true(self, tmp_path):
        processor = LitePdfMarkdownProcessor()
        fake_doc = _make_fitz_doc(page_count=2)
        with patch("knowledge_flow_backend.core.processors.input.lightweight_markdown_processor.lite2_pdf_to_md_processor.fitz.open", return_value=fake_doc):
            assert processor.check_file_validity(tmp_path / "ok.pdf") is True

    def test_zero_page_pdf_returns_false(self, tmp_path):
        processor = LitePdfMarkdownProcessor()
        fake_doc = _make_fitz_doc(page_count=0)
        with patch("knowledge_flow_backend.core.processors.input.lightweight_markdown_processor.lite2_pdf_to_md_processor.fitz.open", return_value=fake_doc):
            assert processor.check_file_validity(tmp_path / "empty.pdf") is False

    def test_corrupt_pdf_returns_false(self, tmp_path):
        processor = LitePdfMarkdownProcessor()
        with patch("knowledge_flow_backend.core.processors.input.lightweight_markdown_processor.lite2_pdf_to_md_processor.fitz.open", side_effect=Exception("not a PDF")):
            assert processor.check_file_validity(tmp_path / "bad.pdf") is False


# ── LitePdfToMdExtractor._extract_pymupdf4llm ────────────────────────────────

class TestExtractPymupdf4llm:
    def test_digital_pdf_returns_text(self, tmp_path):
        """Normal digital PDF: pages with text → non-empty LiteMarkdownResult."""
        extractor = LitePdfToMdExtractor()
        pages = _make_pymupdf4llm_pages(["Page one text.", "Page two text."])
        with patch("knowledge_flow_backend.core.processors.input.lightweight_markdown_processor.lite2_pdf_to_md_processor.pymupdf4llm.to_markdown", return_value=pages):
            result = extractor._extract_pymupdf4llm(tmp_path / "doc.pdf", LiteMarkdownOptions())

        assert result.total_chars > 0
        assert "Page one text." in result.markdown
        assert result.page_count == 2
        assert result.truncated is False
        assert result.extras.get("engine") == "pymupdf4llm"

    def test_empty_pages_yield_zero_chars(self, tmp_path):
        """Scanned PDF path: pymupdf4llm returns pages with no text."""
        extractor = LitePdfToMdExtractor()
        pages = _make_pymupdf4llm_pages(["", "", ""])
        with patch("knowledge_flow_backend.core.processors.input.lightweight_markdown_processor.lite2_pdf_to_md_processor.pymupdf4llm.to_markdown", return_value=pages):
            result = extractor._extract_pymupdf4llm(tmp_path / "scanned.pdf", LiteMarkdownOptions())

        assert result.total_chars <= 0
        assert result.truncated is False

    def test_truncation_at_max_chars(self, tmp_path):
        """Pages that would exceed max_chars are skipped; truncated=True."""
        extractor = LitePdfToMdExtractor()
        long_text = "A" * 200
        pages = _make_pymupdf4llm_pages([long_text, long_text, long_text])
        opts = LiteMarkdownOptions(max_chars=100)
        with patch("knowledge_flow_backend.core.processors.input.lightweight_markdown_processor.lite2_pdf_to_md_processor.pymupdf4llm.to_markdown", return_value=pages):
            result = extractor._extract_pymupdf4llm(tmp_path / "long.pdf", opts)

        assert result.truncated is True
        assert result.total_chars <= 100


# ── LitePdfToMdExtractor.extract (fallback path) ─────────────────────────────

class TestExtractFallback:
    def test_pymupdf4llm_failure_falls_back_to_markitdown(self, tmp_path):
        """When pymupdf4llm raises, markitdown is used and result is returned."""
        extractor = LitePdfToMdExtractor()
        markitdown_result = MagicMock()
        markitdown_result.markdown = "Fallback text from markitdown."

        with patch("knowledge_flow_backend.core.processors.input.lightweight_markdown_processor.lite2_pdf_to_md_processor.pymupdf4llm.to_markdown", side_effect=RuntimeError("pymupdf4llm crashed")):
            with patch.object(extractor._md, "convert", return_value=markitdown_result):
                result = extractor.extract(tmp_path / "doc.pdf")

        assert "Fallback text" in result.markdown
        assert result.extras.get("engine") == "markitdown"

    def test_both_engines_fail_raises(self, tmp_path):
        """When both pymupdf4llm and markitdown fail, the exception propagates."""
        extractor = LitePdfToMdExtractor()
        with patch("knowledge_flow_backend.core.processors.input.lightweight_markdown_processor.lite2_pdf_to_md_processor.pymupdf4llm.to_markdown", side_effect=RuntimeError("fail")):
            with patch.object(extractor._md, "convert", side_effect=RuntimeError("markitdown fail")):
                with pytest.raises(RuntimeError, match="markitdown fail"):
                    extractor.extract(tmp_path / "doc.pdf")

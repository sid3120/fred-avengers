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

from knowledge_flow_backend.core.processors.input.common.pdf_pre_flight_classifier import (
    PdfDocumentType,
    PdfPreFlightClassifier,
)

_MODULE = "knowledge_flow_backend.core.processors.input.common.pdf_pre_flight_classifier"


def _make_page(text: str, image_count: int = 0) -> MagicMock:
    page = MagicMock()
    page.get_text.return_value = text
    page.get_images.return_value = [MagicMock()] * image_count
    return page


def _make_doc(pages: list) -> MagicMock:
    doc = MagicMock()
    doc.page_count = len(pages)
    doc.__getitem__ = lambda self, i: pages[i]
    return doc


# ── digital-native PDF ────────────────────────────────────────────────────────

def test_digital_native_pdf_returns_native_text(tmp_path):
    """Pages with plenty of text → NATIVE_TEXT regardless of images."""
    pages = [
        _make_page("This page has lots of readable text content.", image_count=0),
        _make_page("Another page with substantial textual content here.", image_count=1),
    ]
    with patch(f"{_MODULE}.fitz.open", return_value=_make_doc(pages)):
        result = PdfPreFlightClassifier.classify(tmp_path / "doc.pdf")
    assert result == PdfDocumentType.NATIVE_TEXT


# ── scanned PDF ───────────────────────────────────────────────────────────────

def test_scanned_pdf_returns_scanned(tmp_path):
    """Pages with no text but with images → SCANNED (all 3 sampled pages qualify)."""
    pages = [
        _make_page("", image_count=1),
        _make_page("   ", image_count=1),
        _make_page("", image_count=2),
    ]
    with patch(f"{_MODULE}.fitz.open", return_value=_make_doc(pages)):
        result = PdfPreFlightClassifier.classify(tmp_path / "scanned.pdf")
    assert result == PdfDocumentType.SCANNED


# ── mixed PDF (below 50% scanned threshold) ───────────────────────────────────

def test_mixed_pdf_below_threshold_returns_native_text(tmp_path):
    """1 scanned page out of 3 sampled = 33% < 50% threshold → NATIVE_TEXT."""
    pages = [
        _make_page("", image_count=1),                                   # scanned
        _make_page("Substantial text content on this page.", image_count=0),  # native
        _make_page("More readable content present here.", image_count=0),     # native
    ]
    with patch(f"{_MODULE}.fitz.open", return_value=_make_doc(pages)):
        result = PdfPreFlightClassifier.classify(tmp_path / "mixed.pdf")
    assert result == PdfDocumentType.NATIVE_TEXT


# ── edge: empty PDF (0 pages) ─────────────────────────────────────────────────

def test_empty_pdf_returns_native_text(tmp_path):
    """PDF with zero pages → NATIVE_TEXT (fail-safe, nothing to classify)."""
    with patch(f"{_MODULE}.fitz.open", return_value=_make_doc([])):
        result = PdfPreFlightClassifier.classify(tmp_path / "empty.pdf")
    assert result == PdfDocumentType.NATIVE_TEXT


# ── edge: fitz.open raises ────────────────────────────────────────────────────

def test_corrupt_file_returns_native_text(tmp_path):
    """Any exception from fitz → NATIVE_TEXT (classifier never blocks ingestion)."""
    with patch(f"{_MODULE}.fitz.open", side_effect=Exception("not a PDF")):
        result = PdfPreFlightClassifier.classify(tmp_path / "bad.pdf")
    assert result == PdfDocumentType.NATIVE_TEXT


# ── edge: image-free scanned page (no images detected) ───────────────────────

def test_text_poor_but_no_images_returns_native_text(tmp_path):
    """Pages with near-zero text but NO images do not count as scanned."""
    pages = [
        _make_page("", image_count=0),   # empty text, no images → not scanned
        _make_page("", image_count=0),
        _make_page("", image_count=0),
    ]
    with patch(f"{_MODULE}.fitz.open", return_value=_make_doc(pages)):
        result = PdfPreFlightClassifier.classify(tmp_path / "blank.pdf")
    assert result == PdfDocumentType.NATIVE_TEXT

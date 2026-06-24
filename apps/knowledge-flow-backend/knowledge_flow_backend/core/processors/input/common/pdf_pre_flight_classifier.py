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

"""
Cheap pre-flight classification of PDF documents before extraction begins.

Uses PyMuPDF (fitz) — already a project dependency — with no network calls
and no heavy model loading. Fail-safe: any exception returns NATIVE_TEXT so
the normal extraction path is never blocked by classifier failure.
"""

import logging
from enum import Enum
from pathlib import Path

import fitz

logger = logging.getLogger(__name__)

_SAMPLE_PAGES = 3
_MIN_TEXT_CHARS = 50
_SCANNED_PAGE_RATIO = 0.5


class PdfDocumentType(str, Enum):
    NATIVE_TEXT = "native_text"
    SCANNED = "scanned"


class PdfPreFlightClassifier:
    """Stateless PDF classifier. All methods are static."""

    @staticmethod
    def classify(file_path: Path) -> PdfDocumentType:
        """
        Classify a PDF as NATIVE_TEXT or SCANNED.

        Samples the first _SAMPLE_PAGES pages. A page counts as "scanned" when
        it has fewer than _MIN_TEXT_CHARS of extractable text AND contains at
        least one embedded image. If the fraction of scanned pages meets or
        exceeds _SCANNED_PAGE_RATIO, the document is classified SCANNED.

        Any exception (corrupt file, unexpected fitz error) returns NATIVE_TEXT
        so the caller always falls through to the normal extraction path.
        """
        try:
            doc = fitz.open(str(file_path))
            page_count = doc.page_count
            sample_n = min(_SAMPLE_PAGES, page_count)
            if sample_n == 0:
                doc.close()
                return PdfDocumentType.NATIVE_TEXT

            scanned_pages = 0
            for i in range(sample_n):
                page = doc[i]
                text_len = len(page.get_text("text").strip())
                has_images = bool(page.get_images())
                if text_len < _MIN_TEXT_CHARS and has_images:
                    scanned_pages += 1
            doc.close()

            ratio = scanned_pages / sample_n
            result = PdfDocumentType.SCANNED if ratio >= _SCANNED_PAGE_RATIO else PdfDocumentType.NATIVE_TEXT
            logger.debug(
                "PdfPreFlightClassifier: %s → %s (scanned pages %d/%d)",
                file_path.name,
                result.value,
                scanned_pages,
                sample_n,
            )
            return result
        except Exception as exc:
            logger.warning(
                "PdfPreFlightClassifier: classification failed for %s — %s. Defaulting to NATIVE_TEXT.",
                file_path.name,
                exc,
            )
            return PdfDocumentType.NATIVE_TEXT

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

import asyncio

import pytest
from fred_core import KeycloakUser

from knowledge_flow_backend.common.document_structures import DocumentMetadata
from knowledge_flow_backend.features.ingestion.service import IngestionService


@pytest.fixture
def sample_docx(tmp_path):
    sample_path = tmp_path / "sample.docx"
    sample_path.write_bytes(b"Dummy DOCX content")
    return sample_path


@pytest.fixture
def output_dir(tmp_path):
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    return out_dir


@pytest.fixture
def test_user():
    return KeycloakUser(uid="test-user", username="testuser", email="testuser@localhost", roles=["admin"], groups=["admins"])


def test_extract_and_save_metadata(sample_docx, metadata_store, test_user):
    service = IngestionService()

    # 🔍 Extract metadata
    metadata = asyncio.run(
        service.extract_metadata(
            test_user,
            sample_docx,
            tags=["test"],
            source_tag="uploads",
        )
    )
    assert isinstance(metadata, DocumentMetadata)
    assert metadata.document_uid is not None
    assert "test" in set(metadata.tags.tag_ids + metadata.tags.tag_names)
    assert metadata.source_tag == "uploads"

    # 💾 Save metadata and reload it
    asyncio.run(service.save_metadata(test_user, metadata))
    restored = asyncio.run(service.get_metadata(test_user, metadata.document_uid))
    assert restored is not None
    assert restored.document_uid == metadata.document_uid
    assert "test" in set(restored.tags.tag_ids + restored.tags.tag_names)


def test_process_input(sample_docx, output_dir, test_user):
    service = IngestionService()
    metadata = asyncio.run(service.extract_metadata(test_user, sample_docx, tags=["test"], source_tag="fred"))

    # ⚙️ Process the file into output directory
    service.process_input(test_user, sample_docx, output_dir, metadata)

    # ✅ Check expected output
    output_file = output_dir / "output.md"
    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_process_input_then_output(sample_docx, output_dir, test_user):
    service = IngestionService()
    metadata = asyncio.run(service.extract_metadata(test_user, sample_docx, tags=["test"], source_tag="fred"))

    # First process input
    service.process_input(test_user, sample_docx, output_dir, metadata)

    # Then process output
    result = service.process_output(test_user, sample_docx.name, output_dir, metadata)
    assert result is not None


def test_get_preview_file_fallback(sample_docx, output_dir, test_user):
    service = IngestionService()
    metadata = asyncio.run(service.extract_metadata(test_user, sample_docx, tags=["test"], source_tag="fred"))

    # Write a dummy preview
    preview = output_dir / "table.csv"
    preview.write_text("a,b,c\n1,2,3", encoding="utf-8")

    found = service.get_preview_file(test_user, metadata, output_dir)
    assert found.name == "table.csv"


# ── adaptive PDF routing (QUALITY-04) ────────────────────────────────────────

import pathlib
from unittest.mock import MagicMock, patch

from knowledge_flow_backend.common.structures import IngestionProcessingProfile
from knowledge_flow_backend.core.processors.input.common.pdf_pre_flight_classifier import PdfDocumentType

_CLASSIFIER = "knowledge_flow_backend.features.ingestion.ingestion_service.PdfPreFlightClassifier.classify"
_PIPELINE_MGR = "knowledge_flow_backend.features.ingestion.ingestion_service.IngestionService.pipeline_manager"


@pytest.fixture
def sample_pdf(tmp_path):
    p = tmp_path / "sample.pdf"
    p.write_bytes(b"%PDF-1.4 dummy")
    return p


@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock()
    pipeline.process_input.return_value = None
    return pipeline


def test_scanned_pdf_with_fast_profile_upgrades_to_medium(sample_pdf, output_dir, test_user, mock_pipeline):
    """fast + scanned → effective profile upgraded to medium before pipeline dispatch."""
    service = IngestionService()
    metadata = MagicMock(spec=DocumentMetadata)

    captured_profiles = []

    def capture_pipeline(meta, profile):
        captured_profiles.append(profile)
        return mock_pipeline

    with patch(_CLASSIFIER, return_value=PdfDocumentType.SCANNED):
        with patch.object(service.pipeline_manager, "get_pipeline_for_metadata", side_effect=capture_pipeline):
            service.process_input(test_user, sample_pdf, output_dir, metadata, profile=IngestionProcessingProfile.fast)

    assert captured_profiles[0] == IngestionProcessingProfile.medium


def test_digital_pdf_with_fast_profile_stays_fast(sample_pdf, output_dir, test_user, mock_pipeline):
    """fast + digital-native → profile unchanged."""
    service = IngestionService()
    metadata = MagicMock(spec=DocumentMetadata)

    captured_profiles = []

    def capture_pipeline(meta, profile):
        captured_profiles.append(profile)
        return mock_pipeline

    with patch(_CLASSIFIER, return_value=PdfDocumentType.NATIVE_TEXT):
        with patch.object(service.pipeline_manager, "get_pipeline_for_metadata", side_effect=capture_pipeline):
            service.process_input(test_user, sample_pdf, output_dir, metadata, profile=IngestionProcessingProfile.fast)

    assert captured_profiles[0] == IngestionProcessingProfile.fast


def test_scanned_pdf_with_medium_profile_unchanged(sample_pdf, output_dir, test_user, mock_pipeline):
    """Classifier only fires for fast profile — medium stays medium even for scanned PDFs."""
    service = IngestionService()
    metadata = MagicMock(spec=DocumentMetadata)

    captured_profiles = []

    def capture_pipeline(meta, profile):
        captured_profiles.append(profile)
        return mock_pipeline

    with patch(_CLASSIFIER, return_value=PdfDocumentType.SCANNED) as mock_classify:
        with patch.object(service.pipeline_manager, "get_pipeline_for_metadata", side_effect=capture_pipeline):
            service.process_input(test_user, sample_pdf, output_dir, metadata, profile=IngestionProcessingProfile.medium)

    mock_classify.assert_not_called()
    assert captured_profiles[0] == IngestionProcessingProfile.medium


def test_non_pdf_file_bypasses_classifier(output_dir, test_user, mock_pipeline, tmp_path):
    """Classifier never called for non-PDF suffixes."""
    docx_path = tmp_path / "report.docx"
    docx_path.write_bytes(b"PK dummy docx")
    service = IngestionService()
    metadata = MagicMock(spec=DocumentMetadata)

    with patch(_CLASSIFIER) as mock_classify:
        with patch.object(service.pipeline_manager, "get_pipeline_for_metadata", return_value=mock_pipeline):
            service.process_input(test_user, docx_path, output_dir, metadata, profile=IngestionProcessingProfile.fast)

    mock_classify.assert_not_called()

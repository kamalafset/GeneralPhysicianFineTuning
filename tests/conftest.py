import pymupdf as fitz
import pytest

from src.document_service import DocumentService
from src.embeddings import EmbeddingPipeline
from src.retrieval_service import RetrievalService
from src.vector_store import ChromaVectorStore


@pytest.fixture(scope="session")
def shared_pipeline():
    return EmbeddingPipeline()  # load the model once for the whole test session


@pytest.fixture
def store(tmp_path, shared_pipeline):
    s = ChromaVectorStore(persist_dir=str(tmp_path / "chroma"), collection_name="lab_reports")
    s.embedding_pipeline = shared_pipeline
    return s


@pytest.fixture
def docs(store, tmp_path):
    return DocumentService(store, uploads_dir=tmp_path / "uploads")


@pytest.fixture
def search(store):
    return RetrievalService(store)


def make_pdf(patient_id="P0001", report_id="R0001", report_date="2026-03-13",
             hemoglobin="12.2", name="Test Person", header=True) -> bytes:
    """Synthetic lab report in the same layout as data/reports (one cell per line)."""
    if header:
        lines = [
            "SYNTHETIC DIAGNOSTICS LAB   (synthetic data - not a real patient)",
            f"Patient: {name}   |   Patient ID: {patient_id}   |   Age/Sex: 30/M",
            f"Report ID: {report_id}   |   Report Date: {report_date}",
            "Complete Blood Count (CBC)", "Test", "Result", "Unit", "Reference Range", "Flag",
            "Hemoglobin", hemoglobin, "g/dL", "13 - 17", "LOW",
            "Platelet Count", "196.1", "x10^3/uL", "150 - 410",
        ]
    else:
        lines = ["Clinical note: patient reports mild fatigue. Hemoglobin 12.2 g/dL noted."]
    doc = fitz.open()
    page = doc.new_page()
    y = 50
    for line in lines:
        page.insert_text((40, y), line, fontsize=10)
        y += 16
    return doc.tobytes()

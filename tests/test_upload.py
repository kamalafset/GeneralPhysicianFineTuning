from pathlib import Path

from tests.conftest import make_pdf


def test_successful_upload_stores_metadata_and_file(docs, store):
    res = docs.process_upload("p0001", "my report.pdf", make_pdf())
    assert res.status == "indexed" and res.ok
    assert res.report_id == "R0001" and res.report_date == "2026-03-13"
    assert res.version == 1 and res.chunks >= 1
    assert Path(res.stored_path).exists()                      # original persisted
    assert "P0001" in Path(res.stored_path).parts               # patient-associated dir
    assert Path(res.stored_path).name != "my report.pdf"        # internal unique name

    got = store.collection.get(where={"patient_id": "P0001"}, include=["metadatas"])
    assert len(got["ids"]) == res.chunks
    for m in got["metadatas"]:
        for key in ("patient_id", "report_id", "report_date", "source", "page", "uploaded_at", "checksum"):
            assert key in m
        assert m["original_filename"] == "my report.pdf" and m["is_active"] is True


def test_rejects_non_pdf_empty_corrupt_and_oversized(docs, store):
    assert docs.process_upload("P0001", "a.pdf", b"").status == "rejected"
    assert docs.process_upload("P0001", "a.pdf", b"hello world, not a pdf").status == "rejected"
    assert docs.process_upload("P0001", "a.pdf", b"%PDF-1.4\ngarbage").status == "rejected"
    docs.max_bytes = 100
    assert "too large" in docs.process_upload("P0001", "a.pdf", make_pdf()).message
    assert store.collection.count() == 0


def test_invalid_patient_id_rejected(docs):
    assert docs.process_upload("P1; drop", "a.pdf", make_pdf()).status == "rejected"


def test_patient_id_mismatch_rejected_and_nothing_stored(docs, store, tmp_path):
    res = docs.process_upload("P0002", "a.pdf", make_pdf(patient_id="P0001"))
    assert res.status == "rejected" and "mismatch" in res.message.lower()
    assert store.collection.count() == 0
    assert not list((tmp_path / "uploads").rglob("*.pdf"))      # staged file removed


def test_report_without_patient_id_accepted_with_warning(docs, store):
    res = docs.process_upload("P0001", "note.pdf", make_pdf(header=False))
    assert res.ok and res.warnings
    m = store.collection.get(include=["metadatas"])["metadatas"][0]
    assert m["patient_id_verified"] is False and m["patient_id"] == "P0001"

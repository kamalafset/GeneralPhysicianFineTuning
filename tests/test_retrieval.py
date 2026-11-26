import pytest

from src.retrieval_service import SearchInputError
from tests.conftest import make_pdf

Q = "What is the Hemoglobin count?"


def test_retrieves_relevant_chunk_with_metadata(docs, search):
    docs.process_upload("P0001", "a.pdf", make_pdf(hemoglobin="12.2"))
    resp = search.search("P0001", "What is the Hemoglobin count for Patient ID P0001?")
    assert resp.results and len(resp.results) <= 5
    top = resp.results[0]
    assert "Hemoglobin: 12.2" in top.text
    assert (top.report_id, top.report_date, top.page) == ("R0001", "2026-03-13", 1)
    assert 0 < top.score <= 1


def test_patient_isolation_never_falls_back(docs, search):
    docs.process_upload("P0001", "a.pdf", make_pdf("P0001", "R0001", hemoglobin="12.2"))
    docs.process_upload("P0002", "b.pdf", make_pdf("P0002", "R0002", hemoglobin="15.5"))

    r1 = search.search("P0001", Q)
    r2 = search.search("P0002", Q)
    assert r1.results and all(c.patient_id == "P0001" for c in r1.results)
    assert all("15.5" not in c.text for c in r1.results)
    assert r2.results and all("12.2" not in c.text for c in r2.results)

    unknown = search.search("P0099", Q)                 # no fallback to anyone else
    assert unknown.results == [] and "No reports" in unknown.message


def test_question_naming_another_patient_is_refused(docs, search):
    docs.process_upload("P0002", "b.pdf", make_pdf("P0002", "R0002"))
    with pytest.raises(SearchInputError):
        search.search("P0001", "What is the Hemoglobin for Patient ID P0002?")


def test_select_specific_report_and_default_latest_version(docs, search):
    docs.process_upload("P0001", "a.pdf", make_pdf(report_id="R0001", report_date="2026-01-01", hemoglobin="11.1"))
    docs.process_upload("P0001", "b.pdf", make_pdf(report_id="R0002", report_date="2026-06-01", hemoglobin="13.3"))
    only_r1 = search.search("P0001", Q, report_id="R0001")
    assert only_r1.results and {c.report_id for c in only_r1.results} == {"R0001"}
    all_reports = search.search("P0001", Q)
    assert {c.report_id for c in all_reports.results} == {"R0001", "R0002"}

    docs.process_upload("P0001", "b2.pdf", make_pdf(report_id="R0002", report_date="2026-06-01", hemoglobin="9.9"), allow_replace=True)
    latest = search.search("P0001", Q)
    texts = " ".join(c.text for c in latest.results)
    assert "9.9" in texts and "13.3" not in texts       # superseded hidden by default
    with_old = search.search("P0001", Q, include_superseded=True)
    assert "13.3" in " ".join(c.text for c in with_old.results)


def test_irrelevant_query_returns_clear_message(docs, search):
    docs.process_upload("P0001", "a.pdf", make_pdf())
    search.min_similarity = 0.99                        # nothing can clear this bar
    resp = search.search("P0001", Q)
    assert resp.results == [] and "No relevant information" in resp.message


def test_input_validation(search):
    with pytest.raises(SearchInputError):
        search.search("P0001", "   ")
    with pytest.raises(SearchInputError):
        search.search("bad id!", Q)


def test_legacy_chunks_without_version_fields_are_searchable(store, search):
    store.collection.add(
        ids=["legacy1"], documents=["Hemoglobin: 13.0 g/dL (ref 13 - 17)"],
        embeddings=[store.embedding_pipeline.model.encode("Hemoglobin: 13.0 g/dL").tolist()],
        metadatas=[{"patient_id": "P0003", "report_id": "R0003", "source": "data\\reports\\R0003_P0003.pdf", "page": 0}],
    )
    resp = search.search("P0003", Q)
    assert resp.results and resp.results[0].report_name == "R0003_P0003.pdf"

"""Grounded-answer tests. Synthetic patients only; the LLM is a scripted fake (no network)."""
import pytest

from src import config
from src.answer_service import AnswerService
from src.ui_text import format_source, safe_markdown
from tests.conftest import make_pdf


class FakeLLM:
    def __init__(self, reply="", fail=False):
        self.reply, self.fail, self.calls = reply, fail, []

    def generate(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        if self.fail:
            raise RuntimeError("boom")
        return self.reply


@pytest.fixture
def make_svc(store, tmp_path):
    def _make(llm=None):
        return AnswerService(store, llm=llm, llm_factory=lambda: None, uploads_dir=tmp_path / "uploads")
    return _make


@pytest.fixture
def svc(make_svc):
    return make_svc()


@pytest.fixture
def two_patients(docs):
    docs.process_upload("P0001", "R0001_P0001.pdf", make_pdf("P0001", "R0001", name="Arjun Reddy", hemoglobin="12.2"))
    docs.process_upload("P0002", "R0002_P0002.pdf", make_pdf("P0002", "R0002", name="Meera Nair", hemoglobin="15.5"))


# 1. Patient ID inside the question -------------------------------------------------
def test_patient_id_in_question(svc, two_patients):
    res = svc.answer_question("What is the hemoglobin count for patient P0001?")
    assert res["status"] == "ok" and res["patient_id"] == "P0001"
    assert "12.2 g/dL" in res["answer"] and "LOW" in res["answer"] and "13–17 g/dL" in res["answer"]


# 2. Valid patient name -------------------------------------------------------------
def test_patient_name_in_question(svc, two_patients):
    res = svc.answer_question("What is the hemoglobin count for Arjun Reddy?")
    assert res["status"] == "ok" and res["patient_id"] == "P0001" and "12.2 g/dL" in res["answer"]


# 3. Patient ID supplied separately -------------------------------------------------
def test_patient_id_field(svc, two_patients):
    res = svc.answer_question("What is my hemoglobin?", patient_id_field="p0002")
    assert res["patient_id"] == "P0002" and "15.5 g/dL" in res["answer"]


# 4. Missing / ambiguous identity ---------------------------------------------------
def test_missing_identity_asks_for_clarification_and_retrieves_nothing(svc, two_patients):
    res = svc.answer_question("What is my hemoglobin?")
    assert res["status"] == "identity_error" and res["sources"] == []
    assert "12.2" not in res["answer"] and "15.5" not in res["answer"]


def test_unknown_id_and_unknown_name(svc, two_patients):
    assert "No patient found" in svc.answer_question("hemoglobin for P0099")["answer"]
    assert svc.answer_question("hemoglobin for Nobody Known")["status"] == "identity_error"


def test_ambiguous_name_is_rejected(docs, svc):
    docs.process_upload("P0001", "a.pdf", make_pdf("P0001", "R0001", name="Sam Roy"))
    docs.process_upload("P0002", "b.pdf", make_pdf("P0002", "R0002", name="Sam Roy"))
    res = svc.answer_question("What is the hemoglobin for Sam Roy?")
    assert res["status"] == "identity_error" and "P0001" in res["answer"] and "P0002" in res["answer"]


def test_id_is_never_invented_from_a_name_in_other_case(svc, two_patients):
    assert svc.answer_question("hemoglobin of arjun reddy")["patient_id"] == "P0001"  # registry lookup, not a guess


# 5. Conflicting identifiers --------------------------------------------------------
def test_field_and_question_id_conflict(svc, two_patients):
    res = svc.answer_question("What is the hemoglobin for P0002?", patient_id_field="P0001")
    assert res["status"] == "identity_error" and "P0002" in res["answer"] and "P0001" in res["answer"]
    assert "15.5" not in res["answer"]


def test_field_vs_other_patients_name_and_id_vs_name_conflict(svc, two_patients):
    assert svc.answer_question("hemoglobin for Meera Nair", patient_id_field="P0001")["status"] == "identity_error"
    assert svc.answer_question("hemoglobin for P0001 Meera Nair")["status"] == "identity_error"


def test_authenticated_identity_cannot_be_overridden(svc, two_patients):
    res = svc.answer_question("hemoglobin for P0002", authenticated_patient_id="P0001")
    assert res["status"] == "identity_error" and "15.5" not in res["answer"]
    assert svc.answer_question("hemoglobin for P0001", patient_id_field="P0002", authenticated_patient_id="P0001")["status"] == "identity_error"
    ok = svc.answer_question("What is my hemoglobin?", authenticated_patient_id="P0001")
    assert ok["status"] == "ok" and "12.2 g/dL" in ok["answer"]


# 6. Numbers, units and hallucination guard ----------------------------------------
def test_values_units_and_unflagged_result(svc, docs):
    docs.process_upload("P0001", "a.pdf", make_pdf())
    res = svc.answer_question("What is the platelet count for P0001?")
    assert "196.1 x10^3/uL" in res["answer"] and "does not flag" in res["answer"] and "150–410" in res["answer"]


def test_llm_wording_accepted_when_numbers_check_out(make_svc, two_patients):
    llm = FakeLLM("Patient P0001's hemoglobin is 12.2 g/dL, marked LOW against the range of 13–17 g/dL.")
    res = make_svc(llm).answer_question("What is the hemoglobin count for patient P0001?")
    assert res["method"] == "llm" and res["answer"].startswith("Patient P0001's hemoglobin is 12.2")
    system, user = llm.calls[0]
    assert "untrusted" in system and "Never invent" in system and "Arjun" not in user


@pytest.mark.parametrize("bad", ["Hemoglobin is 99 g/dL.", "It is 12.2 g/dL, normal range 12-20 g/dL.", ""])
def test_hallucinated_or_empty_llm_answer_falls_back_to_report_values(make_svc, two_patients, bad):
    res = make_svc(FakeLLM(bad)).answer_question("What is the hemoglobin for P0001?")
    assert res["method"] == "deterministic" and "12.2 g/dL" in res["answer"] and "99" not in res["answer"]


def test_llm_failure_does_not_break_lab_questions(make_svc, two_patients):
    res = make_svc(FakeLLM(fail=True)).answer_question("What is the hemoglobin for P0001?")
    assert res["status"] == "ok" and "12.2 g/dL" in res["answer"]


# 7. Missing results ---------------------------------------------------------------
def test_missing_test_result(svc, two_patients):
    res = svc.answer_question("What is the HbA1c level for patient P0001?")
    assert res["status"] == "not_found" and "HbA1c" in res["answer"] and res["sources"] == []
    assert not any(ch.isdigit() for ch in res["answer"].replace("P0001", "").replace("HbA1c", ""))  # no invented value


def test_general_question_without_llm_does_not_guess(svc, two_patients):
    res = svc.answer_question("Does P0001 have any allergies?")
    assert res["status"] in {"unavailable", "not_found"} and "12.2" not in res["answer"]


def test_flagged_results_listing(svc, two_patients):
    res = svc.answer_question("Which blood test results are marked HIGH or LOW for P0001?")
    assert res["status"] == "ok" and "Hemoglobin: 12.2 g/dL - LOW" in res["answer"]
    assert "Platelet" not in res["answer"]  # unflagged


# 8-9. Multiple reports, latest by date ---------------------------------------------
@pytest.fixture
def history(docs):
    # uploaded out of date order on purpose: the newest REPORT is not the last upload
    docs.process_upload("P0001", "new.pdf", make_pdf(report_id="R0003", report_date="2026-09-01", hemoglobin="13.4"))
    docs.process_upload("P0001", "old.pdf", make_pdf(report_id="R0001", report_date="2026-01-01", hemoglobin="11.1"))
    docs.process_upload("P0001", "mid.pdf", make_pdf(report_id="R0002", report_date="2026-05-01", hemoglobin="12.0"))


def test_trend_uses_all_reports_in_date_order(svc, history):
    res = svc.answer_question("How has my hemoglobin changed across my previous reports?", patient_id_field="P0001")
    a = res["answer"]
    assert res["intent"] == "trend"
    assert a.index("11.1") < a.index("12.0") < a.index("13.4") and "increased" in a
    assert {s["report_id"] for s in res["sources"]} == {"R0001", "R0002", "R0003"}


def test_latest_value_is_chosen_by_report_date(svc, history):
    res = svc.answer_question("What is my hemoglobin?", patient_id_field="P0001")
    assert "13.4 g/dL" in res["answer"] and "11.1" not in res["answer"]
    assert [s["report_date"] for s in res["sources"]] == ["2026-09-01"]


def test_summarize_latest_report_sends_only_latest_report_to_llm(make_svc, history):
    llm = FakeLLM("Your latest report shows hemoglobin of 13.4 g/dL.")
    res = make_svc(llm).answer_question("Summarize my latest medical report", patient_id_field="P0001")
    assert res["intent"] == "summary" and res["method"] == "llm"
    prompt = llm.calls[0][1]
    assert "13.4" in prompt and "11.1" not in prompt and "12.0" not in prompt
    assert [s["report_date"] for s in res["sources"]] == ["2026-09-01"]


def test_selected_report_limits_scope(svc, history):
    res = svc.answer_question("What is my hemoglobin?", patient_id_field="P0001", report_id="R0001")
    assert "11.1 g/dL" in res["answer"] and "13.4" not in res["answer"]
    assert svc.answer_question("hemoglobin", patient_id_field="P0001", report_id="R9999")["status"] == "not_found"


# 10. Patient isolation ------------------------------------------------------------
def test_isolation_between_patients(make_svc, two_patients):
    llm = FakeLLM("ok")
    s = make_svc(llm)
    for pid, own, other in (("P0001", "12.2", "15.5"), ("P0002", "15.5", "12.2")):
        res = s.search_and_summarize("What is the hemoglobin?", pid)
        assert own in res["answer"] and other not in res["answer"]
        assert {x["report_id"] for x in res["sources"]} == {"R0001" if pid == "P0001" else "R0002"}
    s.answer_question("Summarize my latest report", patient_id_field="P0001")
    assert "15.5" not in llm.calls[-1][1] and "Meera" not in llm.calls[-1][1]
    assert s.search_and_summarize("hemoglobin", "P0099")["status"] == "not_found"


def test_original_pdf_only_for_own_reports(svc, two_patients):
    assert svc.read_report_pdf("P0001", "R0001").startswith(b"%PDF-")
    assert svc.read_report_pdf("P0001", "R0002") is None  # P0002's report
    assert svc.read_report_pdf("P0003", "R0001") is None


# 11-12. Source attribution and no raw output --------------------------------------
def test_source_attribution_and_no_internal_fields(svc, two_patients):
    res = svc.answer_question("What is the hemoglobin count for patient P0001?")
    assert res["sources"] == [{"report_id": "R0001", "report_name": "R0001_P0001.pdf",
                               "report_date": "2026-03-13", "page": 1}]
    assert set(res) == {"status", "answer", "patient_id", "sources", "intent", "method"}
    blob = repr(res)
    for forbidden in ("distance", "embedding", "upload_id", "checksum", "(ref ", "Hemoglobin: 12.2"):
        assert forbidden not in blob
    assert format_source(res["sources"][0]) == "Source: R0001\\_P0001.pdf  \nPage: 1  \nReport Date: 2026-03-13"


def test_untrusted_text_is_escaped():
    assert "<" not in safe_markdown("<img src=x> [a](http://e.com)").replace("\\<", "")
    assert "\\[" in safe_markdown("[a](http://e.com)")


def test_streamlit_search_tab_shows_only_answer_and_source(docs, tmp_path, monkeypatch, two_patients):
    from streamlit.testing.v1 import AppTest
    import streamlit as st

    monkeypatch.setattr(config, "CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(config, "LLM_PROVIDER", "none")
    st.cache_resource.clear()
    at = AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=180).run()
    assert not at.exception
    at.text_input(key="s_q").set_value("What is the hemoglobin count for patient P0001?")
    at.button(key="s_btn").click().run()
    assert not at.exception

    shown = " ".join(e.value for e in list(at.markdown) + list(at.text) + list(at.caption) + list(at.warning) + list(at.error))
    assert "12.2 g/dL" in shown and "R0001" in shown and "Page: 1" in shown and "2026-03-13" in shown
    for forbidden in ("distance", "score", "chunk", "(ref ", "Hemoglobin: 12.2"):
        assert forbidden not in shown
    assert len(at.expander) == 0 and len(at.json) == 0 and len(at.dataframe) == 0
    st.cache_resource.clear()

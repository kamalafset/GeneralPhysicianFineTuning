"""Streamlit UI: upload medical reports and ask questions that are answered from them.

Run:  uv run streamlit run app.py
"""
import logging
import re

import streamlit as st

from src import config
from src.answer_service import AnswerService
from src.document_service import DocumentService
from src.patient_scope import LocalDemoPatientResolver
from src.ui_text import format_source, safe_markdown
from src.vector_store import ChromaVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")

st.set_page_config(page_title="Medical Report RAG", page_icon="🩺", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model and vector store...")
def get_services():
    """ONE store (=> one embedding model + one collection) shared by upload and search."""
    store = ChromaVectorStore(
        persist_dir=config.CHROMA_DIR,
        collection_name=config.COLLECTION_NAME,
        embedding_model=config.EMBEDDING_MODEL,
    )
    return DocumentService(store), AnswerService(store)


# Prototype: trusts the typed ID. In production swap for a resolver that reads the
# authenticated session; nothing below needs to change.
resolver = LocalDemoPatientResolver()
documents, answers = get_services()


def report_rows(patient_id: str):
    return [
        {
            "Report ID": r.report_id,
            "Report date": r.report_date or "-",
            "File": r.name,
            "Uploaded at (UTC)": r.uploaded_at or "(seed data)",
            "Version": r.version,
            "Status": "active" if r.is_active else "superseded",
            "Pages": r.pages,
            "Chunks": r.chunks,
        }
        for r in documents.list_reports(patient_id)
    ]


st.title("🩺 General Physician Report Assistant")
st.caption("Prototype - synthetic data only. Do not upload real patient records.")
tab_upload, tab_search = st.tabs(["1. Upload Medical Report", "2. Search Medical Reports"])

# ----------------------------------------------------------------------------- upload
with tab_upload:
    up_pid_raw = st.text_input("Patient ID", placeholder="P0001", key="up_pid")
    file = st.file_uploader("Medical report (PDF)", type=["pdf"], key="up_file")
    replace = st.checkbox(
        "This is an updated version of an existing report (keep old version as history)",
        key="up_replace",
    )
    if st.button("Upload and Process", type="primary", key="up_btn"):
        if not file:
            st.error("Please choose a PDF file.")
        else:
            try:
                pid = resolver.resolve(up_pid_raw)
            except ValueError as exc:
                st.error(str(exc))
            else:
                with st.status("Processing report...", expanded=True) as status:
                    st.write("Validating, extracting, cleaning, embedding and indexing...")
                    res = documents.process_upload(pid, file.name, file.getvalue(), allow_replace=replace)
                    status.update(label=res.message, state="complete" if res.ok else "error")
                if res.ok:
                    st.success(res.message)
                    st.json(
                        {
                            "patient_id": res.patient_id,
                            "report_id": res.report_id,
                            "report_date": res.report_date or None,
                            "version": res.version,
                            "pages": res.pages,
                            "chunks_stored": res.chunks,
                            "checksum_sha256": res.checksum,
                            "stored_file": res.stored_path,
                            "collection": config.COLLECTION_NAME,
                            "superseded_uploads": res.superseded_upload_ids,
                        }
                    )
                elif res.status == "needs_confirmation":
                    st.warning(res.message + " Tick the checkbox above and press the button again.")
                elif res.status == "duplicate":
                    st.info(res.message)
                else:
                    st.error(res.message)
                for w in res.warnings:
                    st.warning(w)

    st.subheader("Previously uploaded reports")
    hist_pid_raw = up_pid_raw
    if hist_pid_raw:
        try:
            rows = report_rows(resolver.resolve(hist_pid_raw))
            if rows:
                st.dataframe(rows, width="stretch", hide_index=True)
            else:
                st.info("No reports stored for this patient yet.")
        except ValueError as exc:
            st.error(str(exc))
    else:
        st.caption("Enter a Patient ID to see their reports.")

# ----------------------------------------------------------------------------- search
def get_authenticated_patient_id():
    """Production hook: return the signed-in patient's ID from the auth session (e.g. OIDC subject mapping).
    While it returns None the prototype falls back to the Patient ID field / question."""
    return st.session_state.get("authenticated_patient_id")


with tab_search:
    s_pid_raw = st.text_input("Patient ID (optional)", placeholder="P0001", key="s_pid",
                              help="Optional. You can also include the Patient ID or the patient's full name in your question.")
    question = st.text_input("Question", placeholder="What is the hemoglobin count for patient P0001?", key="s_q")

    report_filter = None
    options = {"All Reports": None}
    if s_pid_raw.strip():
        try:
            for r in documents.list_reports(resolver.resolve(s_pid_raw), include_superseded=False):
                options[f"{r.report_date or 'no date'} - {r.name}"] = r.report_id
        except ValueError:
            pass  # the invalid ID is reported when the question is submitted
    choice = st.selectbox("Report", list(options), key="s_report")
    report_filter = options[choice]

    if st.button("Get Answer", type="primary", key="s_btn"):
        if not question.strip():
            st.error("Please enter a question.")
        else:
            try:
                with st.spinner("Searching the reports..."):
                    res = answers.answer_question(
                        question,
                        patient_id_field=s_pid_raw,
                        authenticated_patient_id=get_authenticated_patient_id(),
                        top_k=config.DEFAULT_TOP_K,
                        report_id=report_filter,
                    )
            except Exception:
                logger.exception("Answer generation failed")
                st.error("Something went wrong while answering. See the server log.")
            else:
                if res["status"] == "ok":
                    st.subheader("Answer")
                    st.markdown(safe_markdown(res["answer"]))
                    for src in res["sources"]:
                        st.markdown(format_source(src))
                    for rid in dict.fromkeys(s["report_id"] for s in res["sources"]):
                        pdf = answers.read_report_pdf(res["patient_id"], rid)
                        if pdf:
                            st.download_button("View Original Report", pdf, file_name=f"{rid}.pdf",
                                               mime="application/pdf", key=f"dl_{rid}")
                elif res["status"] in {"not_found", "needs_clarification"}:
                    st.warning(res["answer"])
                else:
                    st.error(res["answer"])

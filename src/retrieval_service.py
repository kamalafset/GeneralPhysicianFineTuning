"""Patient-scoped semantic search over ChromaDB. Retrieval only - no LLM."""
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from src import config
from src.document_service import display_name
from src.patient_scope import build_where, normalize_patient_id
from src.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)

_PATIENT_REF_RE = re.compile(r"\bP\d+\b", re.IGNORECASE)  # same convention as src/search.py
MAX_QUESTION_CHARS = 500


class SearchInputError(ValueError):
    """Bad input; the message is safe to show to the user."""


@dataclass
class RetrievedChunk:
    text: str
    score: float  # cosine similarity, 0..1 (higher = more similar)
    patient_id: str
    report_id: str
    report_name: str
    report_date: str
    page: int  # 1-based for display (stored value is 0-based)
    version: int
    is_active: bool
    uploaded_at: str
    chunk_id: str


@dataclass
class SearchResponse:
    patient_id: str
    question: str
    results: List[RetrievedChunk] = field(default_factory=list)
    message: str = ""


class RetrievalService:
    def __init__(
        self,
        store: ChromaVectorStore,
        default_top_k: int = config.DEFAULT_TOP_K,
        min_similarity: float = config.MIN_SIMILARITY,
    ):
        self.store = store
        self.default_top_k = default_top_k
        self.min_similarity = min_similarity

    def search(
        self,
        patient_id: str,
        question: str,
        report_id: Optional[str] = None,
        top_k: Optional[int] = None,
        include_superseded: bool = False,
    ) -> SearchResponse:
        try:
            patient_id = normalize_patient_id(patient_id)
        except ValueError as exc:
            raise SearchInputError(str(exc)) from exc

        question = (question or "").strip()
        if not question:
            raise SearchInputError("Please enter a question.")
        if len(question) > MAX_QUESTION_CHARS:
            raise SearchInputError(f"Question is too long (max {MAX_QUESTION_CHARS} characters).")

        # The Patient ID field is authoritative. A different ID in the question is
        # refused rather than silently switching patients.
        others = {m.upper() for m in _PATIENT_REF_RE.findall(question)} - {patient_id}
        if others:
            raise SearchInputError(
                f"Your question mentions {', '.join(sorted(others))}, but the selected patient is "
                f"{patient_id}. Searches are limited to one patient - correct the ID or the question."
            )

        k = max(1, min(int(top_k or self.default_top_k), 20))
        where = build_where(
            patient_id,
            report_id=report_id or None,
            active_only=(report_id is None and not include_superseded),
        )
        raw = self.store.query(query_text=question, top_k=k, where=where)

        results: List[RetrievedChunk] = []
        for r in raw:
            meta = r.get("metadata") or {}
            # Defense in depth: never trust the filter alone.
            if str(meta.get("patient_id")) != patient_id:
                logger.error("Filter leak: chunk %s has patient_id=%s (wanted %s)", r.get("id"), meta.get("patient_id"), patient_id)
                continue
            score = max(0.0, min(1.0, 1.0 - float(r["distance"])))
            if score < self.min_similarity:
                continue
            results.append(
                RetrievedChunk(
                    text=r["text"],
                    score=score,
                    patient_id=patient_id,
                    report_id=str(meta.get("report_id", "")),
                    report_name=display_name(meta),
                    report_date=str(meta.get("report_date", "")),
                    page=int(meta.get("page", 0)) + 1,
                    version=int(meta.get("version", 1)),
                    is_active=meta.get("is_active", True) is not False,
                    uploaded_at=str(meta.get("uploaded_at", "")),
                    chunk_id=str(r.get("id", "")),
                )
            )

        message = ""
        if not results:
            message = f"No relevant information was found in the reports for patient {patient_id}."
            if not raw and not self._has_any_report(patient_id):
                message = f"No reports have been uploaded for patient {patient_id}."
        logger.info("Search patient=%s report=%s hits=%d", patient_id, report_id, len(results))
        return SearchResponse(patient_id=patient_id, question=question, results=results, message=message)

    def _has_any_report(self, patient_id: str) -> bool:
        res = self.store.collection.get(where=build_where(patient_id), limit=1)
        return bool(res["ids"])

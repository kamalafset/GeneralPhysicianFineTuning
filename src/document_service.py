"""Upload pipeline: validate -> store -> load -> clean -> verify -> chunk -> embed -> index.

Reuses the existing modules (data_loader, data_cleaner, EmbeddingPipeline,
ChromaVectorStore). Nothing here trains or fine-tunes a model.

History policy: reports are never deleted. Uploading a changed version of an
existing report keeps the old chunks, flags them `is_active=False` and indexes
the new ones as `version+1`. Search ignores superseded versions by default.
"""
import hashlib
import logging
import re
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from io import BytesIO
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, List

from pypdf import PdfReader

from src import config
from src.data_cleaner import clean_documents
from src.data_loader import load_all_documents
from src.patient_scope import build_where, normalize_patient_id
from src.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

# Only these keys are copied from loaded/cleaned documents. PDFs are untrusted:
# PyPDFLoader copies arbitrary PDF info-dict entries into metadata, which could
# otherwise inject keys like patient_id / is_active.
_ALLOWED_DOC_META = {"page", "total_pages", "doc_type", "patient_name", "age", "sex"}


class ReportValidationError(ValueError):
    """The upload was rejected; the message is safe to show to the user."""


@dataclass
class UploadResult:
    status: str  # indexed | replaced | duplicate | needs_confirmation | rejected
    message: str
    patient_id: str = ""
    report_id: str = ""
    upload_id: str = ""
    version: int = 0
    chunks: int = 0
    pages: int = 0
    report_date: str = ""
    checksum: str = ""
    stored_path: str = ""
    warnings: List[str] = field(default_factory=list)
    superseded_upload_ids: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in {"indexed", "replaced"}


@dataclass
class ReportSummary:
    report_id: str
    report_date: str
    name: str
    uploaded_at: str
    version: int
    is_active: bool
    pages: int
    chunks: int
    source: str


def display_name(meta: Dict[str, Any]) -> str:
    """Human-friendly report name; works for legacy chunks (only `source` path)."""
    return meta.get("original_filename") or PureWindowsPath(str(meta.get("source", "unknown"))).name


def sanitize_filename(name: str) -> str:
    """Display-only: the stored file always gets an internal UUID name."""
    base = PureWindowsPath(name or "report.pdf").name
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip(" .") or "report.pdf"
    return base[:100]


class DocumentService:
    def __init__(
        self,
        store: ChromaVectorStore,
        uploads_dir: Path = config.UPLOADS_DIR,
        max_bytes: int = config.MAX_UPLOAD_MB * 1024 * 1024,
        max_pages: int = config.MAX_PDF_PAGES,
    ):
        self.store = store
        self.collection = store.collection
        self.uploads_dir = Path(uploads_dir)
        self.max_bytes = max_bytes
        self.max_pages = max_pages

    # ------------------------------------------------------------------ validation
    def validate_pdf(self, data: bytes) -> int:
        """Return page count or raise ReportValidationError. Does not extract text."""
        if not data:
            raise ReportValidationError("The uploaded file is empty.")
        if len(data) > self.max_bytes:
            raise ReportValidationError(
                f"File is too large ({len(data) / 1_048_576:.1f} MB). "
                f"Limit is {self.max_bytes // 1_048_576} MB."
            )
        if not data.startswith(b"%PDF-"):  # magic bytes, not the file extension
            raise ReportValidationError("The file is not a valid PDF (bad file signature).")
        try:
            reader = PdfReader(BytesIO(data))
            if reader.is_encrypted:
                raise ReportValidationError("Encrypted/password-protected PDFs are not supported.")
            pages = len(reader.pages)
        except ReportValidationError:
            raise
        except Exception as exc:
            logger.warning("PDF parse failed: %s", exc)
            raise ReportValidationError("The PDF is corrupted or cannot be read.") from exc
        if pages < 1:
            raise ReportValidationError("The PDF has no pages.")
        if pages > self.max_pages:
            raise ReportValidationError(f"The PDF has {pages} pages; the limit is {self.max_pages}.")
        return pages

    # ------------------------------------------------------------------ main flow
    def process_upload(
        self,
        patient_id: str,
        filename: str,
        data: bytes,
        allow_replace: bool = False,
    ) -> UploadResult:
        """Run the full pipeline. Never raises for expected problems; returns a status."""
        try:
            return self._process(patient_id, filename, data, allow_replace)
        except ReportValidationError as exc:
            logger.info("Upload rejected: %s", exc)
            return UploadResult(status="rejected", message=str(exc))
        except Exception:
            logger.exception("Unexpected error while processing upload")
            return UploadResult(
                status="rejected",
                message="Unexpected error while processing the report. Nothing was indexed.",
            )

    def _process(self, patient_id: str, filename: str, data: bytes, allow_replace: bool) -> UploadResult:
        try:
            patient_id = normalize_patient_id(patient_id)
        except ValueError as exc:
            raise ReportValidationError(str(exc)) from exc

        self.validate_pdf(data)
        checksum = hashlib.sha256(data).hexdigest()

        # Same bytes already indexed for this patient -> never index twice.
        dup = self.collection.get(
            where=build_where(patient_id, checksum=checksum), limit=1, include=["metadatas"]
        )
        if dup["ids"]:
            m = dup["metadatas"][0]
            return UploadResult(
                status="duplicate",
                patient_id=patient_id,
                report_id=str(m.get("report_id", "")),
                checksum=checksum,
                message=(
                    f"This exact file is already indexed for {patient_id} as report "
                    f"'{m.get('report_id', '?')}' ({display_name(m)}). Nothing was changed."
                ),
            )

        upload_id = uuid.uuid4().hex
        report_dir = self.uploads_dir / patient_id / upload_id
        stored = report_dir / f"{upload_id}.pdf"
        try:
            report_dir.mkdir(parents=True, exist_ok=True)
            stored.write_bytes(data)
            result = self._index_stored_file(
                patient_id, sanitize_filename(filename), checksum, upload_id, report_dir, stored, allow_replace
            )
        except BaseException:
            self._rollback(upload_id, report_dir)
            raise
        if not result.ok:  # e.g. needs_confirmation: leave no staged file behind
            self._rollback(upload_id, report_dir)
        return result

    def _index_stored_file(
        self,
        patient_id: str,
        original_name: str,
        checksum: str,
        upload_id: str,
        report_dir: Path,
        stored: Path,
        allow_replace: bool,
    ) -> UploadResult:
        warnings: List[str] = []

        # 3. extract (existing loader reads a directory; the dir holds only this file)
        raw = load_all_documents(str(report_dir))
        if not raw:
            raise ReportValidationError("No text could be read from the PDF (is it a scanned image?).")
        # 4. clean
        docs = clean_documents(raw)
        if not docs:
            raise ReportValidationError("The PDF contains no usable text after cleaning.")

        # 6. patient identity check (identifiers found inside the document)
        found_pids = {str(d.metadata["patient_id"]).strip().upper() for d in docs if d.metadata.get("patient_id")}
        if len(found_pids) > 1:
            raise ReportValidationError(
                f"The report mentions several patient IDs ({', '.join(sorted(found_pids))}). Rejected for review."
            )
        verified = bool(found_pids)
        if found_pids and next(iter(found_pids)) != patient_id:
            raise ReportValidationError(
                f"Patient ID mismatch: you selected {patient_id} but the report belongs to "
                f"{next(iter(found_pids))}. Rejected - nothing was stored."
            )
        if not verified:
            warnings.append("No patient ID was found inside the report, so it could not be cross-checked.")

        found_rids = {str(d.metadata["report_id"]).strip() for d in docs if d.metadata.get("report_id")}
        if len(found_rids) > 1:
            raise ReportValidationError("The PDF contains more than one report ID. Rejected for review.")
        report_id = next(iter(found_rids)) if found_rids else f"UPL-{upload_id[:8].upper()}"
        if not _SAFE_ID_RE.match(report_id):
            raise ReportValidationError("The report ID inside the PDF has an unsupported format.")
        if not found_rids:
            warnings.append(f"No report ID found in the PDF; assigned '{report_id}'.")

        report_date = ""
        for d in docs:
            raw_date = str(d.metadata.get("report_date", "")).strip()
            if raw_date:
                try:
                    report_date = date.fromisoformat(raw_date).isoformat()
                except ValueError:
                    warnings.append(f"Ignored unparseable report date '{raw_date[:30]}'.")
                break

        # 11. same report_id already active with different content -> versioning
        previous = self.collection.get(
            where=build_where(patient_id, report_id=report_id, active_only=True), include=["metadatas"]
        )
        version, superseded_ids = 1, []
        if previous["ids"]:
            if not allow_replace:
                return UploadResult(
                    status="needs_confirmation",
                    patient_id=patient_id,
                    report_id=report_id,
                    checksum=checksum,
                    message=(
                        f"Report '{report_id}' already exists for {patient_id} with different content. "
                        "Confirm to store this file as a new version (the old version is kept as history "
                        "and excluded from default search)."
                    ),
                )
            version = max(int(m.get("version", 1)) for m in previous["metadatas"]) + 1
            superseded_ids = sorted({str(m.get("upload_id", "")) for m in previous["metadatas"]})

        # 5. authoritative metadata (overrides anything the PDF claimed)
        uploaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for d in docs:
            keep = {k: v for k, v in d.metadata.items() if k in _ALLOWED_DOC_META}
            keep.update(
                patient_id=patient_id,
                report_id=report_id,
                source=str(stored),  # unique per upload -> unique chunk ids per version
                original_filename=original_name,
                page=int(d.metadata.get("page", 0)),
                uploaded_at=uploaded_at,
                upload_id=upload_id,
                checksum=checksum,
                version=version,
                is_active=True,
                patient_id_verified=verified,
            )
            if report_date:
                keep["report_date"] = report_date
            d.metadata = keep

        # 7-9. chunk, embed with the SAME model used for search, upsert into the one collection
        pipe = self.store.embedding_pipeline
        chunks = pipe.chunk_documents(docs)
        if not chunks:
            raise ReportValidationError("No chunks were produced from the report.")
        embeddings = pipe.embed_chunks(chunks)
        self.store.add_embeddings(embeddings, chunks)

        # Deactivate old versions only after the new one is safely stored.
        if previous["ids"]:
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self.collection.update(
                ids=previous["ids"],
                metadatas=[{"is_active": False, "superseded_by": upload_id, "superseded_at": now}] * len(previous["ids"]),
            )

        logger.info("Indexed %s/%s v%d: %d chunks", patient_id, report_id, version, len(chunks))
        return UploadResult(
            status="replaced" if superseded_ids else "indexed",
            message=f"Report '{report_id}' indexed" + (f" as version {version}." if superseded_ids else "."),
            patient_id=patient_id,
            report_id=report_id,
            upload_id=upload_id,
            version=version,
            chunks=len(chunks),
            pages=len({c.metadata.get("page") for c in chunks}),
            report_date=report_date,
            checksum=checksum,
            stored_path=str(stored),
            warnings=warnings,
            superseded_upload_ids=superseded_ids,
        )

    def _rollback(self, upload_id: str, report_dir: Path) -> None:
        """Undo a half-finished upload: remove its chunks and its stored file."""
        try:
            self.collection.delete(where={"upload_id": upload_id})
        except Exception:
            logger.exception("Rollback: failed to delete chunks for %s", upload_id)
        shutil.rmtree(report_dir, ignore_errors=True)

    # ------------------------------------------------------------------ history
    def list_reports(self, patient_id: str, include_superseded: bool = True) -> List[ReportSummary]:
        """One row per stored file for this patient (legacy + uploaded), newest first."""
        res = self.collection.get(
            where=build_where(patient_id, active_only=not include_superseded), include=["metadatas"]
        )
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for m in res["metadatas"] or []:
            groups.setdefault(str(m.get("source", "")), []).append(m)
        rows = []
        for source, metas in groups.items():
            m = metas[0]
            rows.append(
                ReportSummary(
                    report_id=str(m.get("report_id", "")),
                    report_date=str(m.get("report_date", "")),
                    name=display_name(m),
                    uploaded_at=str(m.get("uploaded_at", "")),
                    version=int(m.get("version", 1)),
                    is_active=m.get("is_active", True) is not False,
                    pages=len({x.get("page") for x in metas}),
                    chunks=len(metas),
                    source=source,
                )
            )
        rows.sort(key=lambda r: (r.report_date, r.uploaded_at, r.version), reverse=True)
        return rows

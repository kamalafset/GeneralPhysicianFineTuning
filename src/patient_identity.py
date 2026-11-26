"""Resolve WHICH patient a request is about, before any record is retrieved.

Priority (prototype): authenticated identity > Patient ID field > ID in the question >
full patient name found in a trusted registry > ask for clarification.
Names are looked up in a registry built from stored report metadata; an ID is never
derived from a name by guessing. In production the authenticated identity is the only
source of truth - anything typed or asked about can only be *checked against* it.
"""
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from src.patient_scope import normalize_patient_id

_ID_IN_TEXT_RE = re.compile(r"\bP\d+\b", re.IGNORECASE)  # same convention as src/search.py


@dataclass
class IdentityResult:
    patient_id: Optional[str] = None
    error: Optional[str] = None  # safe to show to the user
    source: str = ""  # authenticated | field | question_id | question_name

    @property
    def ok(self) -> bool:
        return self.patient_id is not None


class PatientRegistry:
    """patient_id <-> patient_name from ChromaDB metadata (refreshed whenever the collection size changes)."""

    def __init__(self, collection):
        self._collection = collection
        self._count = -1
        self.id_to_name: Dict[str, str] = {}
        self._name_to_ids: Dict[str, Set[str]] = {}

    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()

    def refresh(self) -> None:
        count = self._collection.count()
        if count == self._count:
            return
        id_to_name: Dict[str, str] = {}
        name_to_ids: Dict[str, Set[str]] = {}
        for meta in self._collection.get(include=["metadatas"]).get("metadatas") or []:
            pid, name = meta.get("patient_id"), meta.get("patient_name")
            if pid:
                id_to_name.setdefault(str(pid).upper(), str(name or ""))
            if pid and name:
                name_to_ids.setdefault(self._norm(str(name)), set()).add(str(pid).upper())
        self.id_to_name, self._name_to_ids, self._count = id_to_name, name_to_ids, count

    def knows(self, patient_id: str) -> bool:
        self.refresh()
        return patient_id in self.id_to_name

    def ids_named_in(self, text: str) -> Set[str]:
        self.refresh()
        q = self._norm(text)
        found: Set[str] = set()
        for name, ids in self._name_to_ids.items():
            if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", q):
                found |= ids
        return found

    def label(self, patient_id: str) -> str:
        name = self.id_to_name.get(patient_id, "")
        return f"{name} ({patient_id})" if name else patient_id


def resolve_patient_identity(
    question: str,
    registry: PatientRegistry,
    field_patient_id: Optional[str] = None,
    authenticated_patient_id: Optional[str] = None,
) -> IdentityResult:
    q_ids = {m.upper() for m in _ID_IN_TEXT_RE.findall(question or "")}
    q_name_ids = registry.ids_named_in(question or "")

    def bad(msg: str) -> IdentityResult:
        return IdentityResult(error=msg)

    field_id: Optional[str] = None
    if (field_patient_id or "").strip():
        try:
            field_id = normalize_patient_id(field_patient_id)
        except ValueError as exc:
            return bad(str(exc))

    # 1-2. authenticated identity, else the Patient ID field
    chosen: Optional[str] = None
    source = ""
    if authenticated_patient_id:
        chosen, source = normalize_patient_id(authenticated_patient_id), "authenticated"
        if field_id and field_id != chosen:
            return bad("You can only access your own reports. The Patient ID entered does not match your account.")
    elif field_id:
        chosen, source = field_id, "field"

    # Whatever the question mentions must agree with the chosen identity - it never overrides it.
    if chosen:
        other_ids = q_ids - {chosen}
        other_names = q_name_ids - {chosen}
        if other_ids or other_names:
            mentioned = ", ".join(sorted(other_ids | {registry.label(i) for i in other_names}))
            return bad(
                f"Your question mentions {mentioned}, but this search is for patient {chosen}. "
                "Searches are limited to one patient - correct the Patient ID or the question."
            )
        return IdentityResult(patient_id=chosen, source=source)

    # 3. Patient ID inside the question
    if len(q_ids) > 1:
        return bad(f"Your question mentions more than one patient ({', '.join(sorted(q_ids))}). Ask about one patient at a time.")
    if q_ids:
        pid = next(iter(q_ids))
        if q_name_ids and pid not in q_name_ids:
            return bad(
                f"Patient ID {pid} does not match the name given in your question "
                f"({', '.join(sorted(registry.label(i) for i in q_name_ids))}). Please check the name and ID."
            )
        if not registry.knows(pid):
            return bad(f"No patient found with ID {pid}.")
        return IdentityResult(patient_id=pid, source="question_id")

    # 4. explicit full name resolved against the registry
    if len(q_name_ids) == 1:
        return IdentityResult(patient_id=next(iter(q_name_ids)), source="question_name")
    if len(q_name_ids) > 1:
        options: List[str] = sorted(registry.label(i) for i in q_name_ids)
        return bad(f"More than one patient matches: {', '.join(options)}. Please enter the Patient ID.")

    # 5. nothing usable
    return bad("Please enter a Patient ID, or include the patient's ID (e.g. P0001) or full name in your question.")

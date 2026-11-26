"""Patient identity + access scoping.

Every read/write against ChromaDB goes through `build_where()`, so patient
isolation is enforced in one place rather than scattered across the code.

Production note: replace `LocalDemoPatientResolver` with a resolver that reads
the authenticated session (e.g. JWT/OIDC subject -> patient_id). Services only
ever receive an already-resolved patient_id, so nothing else has to change.
"""
import re
from typing import Any, Dict, Optional, Protocol

_PATIENT_ID_RE = re.compile(r"^[A-Z0-9][A-Z0-9_-]{1,31}$")


class InvalidPatientIdError(ValueError):
    pass


def normalize_patient_id(raw: Optional[str]) -> str:
    """Trim + upper-case, then validate. Also blocks anything that could be abused
    as a filter value (whitespace, operators, wildcards)."""
    pid = (raw or "").strip().upper()
    if not _PATIENT_ID_RE.match(pid):
        raise InvalidPatientIdError(
            "Invalid Patient ID. Use 2-32 letters/digits/-/_ (for example P0001)."
        )
    return pid


class PatientResolver(Protocol):
    def resolve(self, requested_patient_id: Optional[str] = None) -> str:
        """Return the patient_id the current caller is allowed to access."""


class LocalDemoPatientResolver:
    """Prototype only: trusts whatever ID the user typed (synthetic data)."""

    def resolve(self, requested_patient_id: Optional[str] = None) -> str:
        return normalize_patient_id(requested_patient_id)


def build_where(
    patient_id: str,
    *,
    report_id: Optional[str] = None,
    active_only: bool = False,
    **extra: Any,
) -> Dict[str, Any]:
    """Chroma metadata filter that ALWAYS contains an exact patient_id match.

    `is_active != False` (rather than `== True`) deliberately keeps legacy chunks
    indexed before versioning existed (they have no `is_active` field).
    """
    clauses = [{"patient_id": normalize_patient_id(patient_id)}]
    if report_id:
        clauses.append({"report_id": report_id})
    if active_only:
        clauses.append({"is_active": {"$ne": False}})
    clauses.extend({k: v} for k, v in extra.items())
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}

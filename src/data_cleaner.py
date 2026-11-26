import re
from typing import List, Optional

from langchain_core.documents import Document

BANNER_RE = re.compile(r"^(SYNTHETIC DIAGNOSTICS LAB|RAG TEST REPORT|SYNTHETIC TEST DATA|Page \d+$).*$", re.IGNORECASE)
PATIENT_RE = re.compile(
    r"Patient:\s*(?P<patient_name>.+?)\s*\|?\s*Patient ID:\s*(?P<patient_id>\S+)"
    r"\s*\|?\s*Age/Sex:\s*(?P<age>\d+)\s*/\s*(?P<sex>\w)"
)
REPORT_RE = re.compile(r"Report ID:\s*(?P<report_id>\S+)\s*\|?\s*Report Date:\s*(?P<report_date>\S+)")
TABLE_HEADER = ["Test", "Result", "Unit", "Reference Range", "Flag"]
FLAGS = {"LOW", "HIGH", "CRITICAL", "ABNORMAL", "NORMAL"}
NUMBER_RE = re.compile(r"^[<>]?\s*-?\d+(\.\d+)?$")


def normalize_text(text: str) -> str:
    """Generic cleanup: fix encoding artefacts, strip control chars, collapse whitespace."""
    text = text.replace("\x00", "").replace(" ", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _parse_table_rows(lines: List[str], i: int):
    """Parse one-cell-per-line table rows starting at index i. Returns (rows, next_index)."""
    rows = []
    while i < len(lines):
        name = lines[i]
        # stop at the next section title, remarks, or end of table
        if name.startswith("Pathologist remarks") or (i + 1 < len(lines) and lines[i + 1] == "Test"):
            break
        if i + 3 >= len(lines) or not NUMBER_RE.match(lines[i + 1]):
            break
        result, unit, ref = lines[i + 1], lines[i + 2], lines[i + 3]
        i += 4
        flag = ""
        if i < len(lines) and lines[i].upper() in FLAGS:
            flag = lines[i].upper()
            i += 1
        rows.append((name, result, unit, ref, flag))
    return rows, i


def clean_report_text(text: str) -> tuple[str, dict]:
    """Turn a raw lab-report page into readable 'Test: value unit (ref) [FLAG]' lines
    and extract patient/report details as metadata."""
    meta: dict = {}
    lines = [l for l in normalize_text(text).split("\n") if l]
    out: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if BANNER_RE.match(line):
            i += 1
            continue
        if (m := PATIENT_RE.search(line)):
            meta.update(m.groupdict())
            meta["age"] = int(meta["age"])
            i += 1
            continue
        if (m := REPORT_RE.search(line)):
            meta.update(m.groupdict())
            i += 1
            continue
        # section title followed by the table header cells
        if lines[i + 1 : i + 1 + len(TABLE_HEADER)] == TABLE_HEADER:
            out.append(f"\n{line}")
            rows, i = _parse_table_rows(lines, i + 1 + len(TABLE_HEADER))
            for name, result, unit, ref, flag in rows:
                out.append(f"{name}: {result} {unit} (ref {ref})" + (f" [{flag}]" if flag else ""))
            continue
        if line.startswith("Pathologist remarks"):
            # re-join hard-wrapped remark lines into a single paragraph
            out.append("\n" + " ".join(lines[i:]))
            break
        out.append(line)
        i += 1
    return "\n".join(out).strip(), meta


def clean_documents(docs: List[Document], min_chars: int = 20) -> List[Document]:
    """Clean loaded documents: normalise text, restructure lab reports, attach metadata,
    drop empty/near-empty pages and exact duplicates."""
    cleaned: List[Document] = []
    seen = set()
    for doc in docs:
        text = doc.page_content or ""
        meta = dict(doc.metadata)
        if "SYNTHETIC DIAGNOSTICS LAB" in text:
            text, extra = clean_report_text(text)
            meta.update(extra)
            meta["doc_type"] = "lab_report"
        else:
            text = normalize_text(text)
        if len(text) < min_chars:
            continue
        key = (meta.get("source"), meta.get("page"), text)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(Document(page_content=text, metadata=meta))
    return cleaned


if __name__ == "__main__":
    from data_loader import load_all_documents

    from pathlib import Path

    raw = load_all_documents(str(Path(__file__).resolve().parent.parent / "data" / "reports"))
    docs = clean_documents(raw)
    print(f"Raw: {len(raw)}  Cleaned: {len(docs)}")
    print(docs[0].metadata)
    print(docs[0].page_content)

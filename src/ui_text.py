"""Patient-facing text helpers (kept out of app.py so they are unit-testable without Streamlit)."""
import re
from typing import Dict


def safe_markdown(text: str) -> str:
    """Model output is untrusted: escape markdown/HTML so it cannot render links, images or tags."""
    escaped = re.sub(r"([\\`*_{}\[\]<>#|~$!])", r"\\\1", text)
    return escaped.replace("\n", "  \n")


def format_source(src: Dict) -> str:
    """The only source fields a patient sees: file name, page, report date."""
    return safe_markdown(
        f"Source: {src['report_name']}\n"
        f"Page: {src['page']}\n"
        f"Report Date: {src['report_date'] or 'not stated'}"
    )

"""Deterministic reading of lab results from the *cleaned* report text.

data_cleaner.py writes every result as one line:
    "Hemoglobin: 12.2 g/dL (ref 13 - 17) [LOW]"
Reading the value back with a regex (instead of trusting an LLM) gives exact numbers,
units and flags, which AnswerService then uses to answer - and to validate LLM output.
"""
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple

LAB_LINE_RE = re.compile(
    r"^(?P<test>[^:\n]+?): (?P<value>[<>]?\s*-?\d+(?:\.\d+)?) (?P<unit>.+?) "
    r"\(ref (?P<ref>.+?)\)(?: \[(?P<flag>[A-Z]+)\])?\s*$"
)
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


@dataclass(frozen=True)
class LabRow:
    test: str
    value: str
    unit: str
    ref: str
    flag: str  # "" when the report does not flag the result
    report_id: str
    page: int  # 1-based

    @property
    def key(self) -> str:
        return norm_name(self.test)


def norm_name(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def parse_rows(text: str, report_id: str, page: int) -> List[LabRow]:
    rows = []
    for line in text.splitlines():
        m = LAB_LINE_RE.match(line.strip())
        if m:
            rows.append(
                LabRow(
                    test=m["test"].strip(),
                    value=re.sub(r"\s+", "", m["value"]),
                    unit=m["unit"].strip(),
                    ref=m["ref"].strip(),
                    flag=m["flag"] or "",
                    report_id=report_id,
                    page=page,
                )
            )
    return rows


def format_ref(ref: str) -> str:
    """'13 - 17' -> '13–17' (display only; digits are untouched)."""
    return re.sub(r"(?<=\d)\s*-\s*(?=\d)", "–", ref)


def numbers_in(text: str) -> Set[str]:
    return set(NUMBER_RE.findall(text))


# Test names + the ways patients refer to them. Tests that appear only in uploaded reports are
# added dynamically from their own names (see find_tests).
KNOWN_TESTS: Dict[str, List[str]] = {
    "Hemoglobin": ["hemoglobin", "haemoglobin", "hb", "hgb"],
    "RBC Count": ["rbc count", "rbc", "red blood cell count", "red blood cells"],
    "Hematocrit (PCV)": ["hematocrit", "pcv", "packed cell volume"],
    "MCV": ["mcv"],
    "MCH": ["mch"],
    "MCHC": ["mchc"],
    "Total WBC Count": ["total wbc count", "wbc count", "wbc", "white blood cell count", "white blood cells"],
    "Neutrophils": ["neutrophils"],
    "Lymphocytes": ["lymphocytes"],
    "Platelet Count": ["platelet count", "platelets", "platelet"],
    "Total Cholesterol": ["total cholesterol", "cholesterol"],
    "LDL Cholesterol": ["ldl cholesterol", "ldl"],
    "HDL Cholesterol": ["hdl cholesterol", "hdl"],
    "Triglycerides": ["triglycerides"],
    "TSH": ["tsh", "thyroid stimulating hormone"],
    "Serum Creatinine": ["serum creatinine", "creatinine"],
    "Blood Urea Nitrogen (BUN)": ["blood urea nitrogen", "bun", "urea nitrogen"],
    "HbA1c (Glycated Hemoglobin)": ["hba1c", "a1c", "hb a1c", "glycated hemoglobin", "glycosylated hemoglobin"],
    "Estimated Average Glucose (eAG)": ["estimated average glucose", "eag"],
    "Fasting Plasma Glucose": ["fasting plasma glucose", "fasting glucose", "fasting blood sugar", "fbs"],
}


def find_tests(question: str, extra_test_names: Iterable[str] = ()) -> List[str]:
    """Canonical test names mentioned in the question. The longest mention wins, so
    'glycated hemoglobin' selects HbA1c, not Hemoglobin, and 'LDL cholesterol' not Total Cholesterol."""
    vocab: Dict[str, List[str]] = dict(KNOWN_TESTS)
    known_keys = {norm_name(k) for k in vocab}
    for name in extra_test_names:
        if norm_name(name) in known_keys:
            continue
        paren = re.search(r"\(([^)]+)\)", name)
        vocab[name] = [norm_name(name.split("(")[0]), norm_name(paren.group(1)) if paren else ""]
        known_keys.add(norm_name(name))

    q = f" {norm_name(question)} "
    hits: List[Tuple[int, int, str]] = []
    for test, phrases in vocab.items():
        for phrase in {norm_name(p) for p in phrases if p}:
            for m in re.finditer(rf"(?<=\s){re.escape(phrase)}(?=\s)", q):
                hits.append((m.start(), m.end(), test))
    kept: List[str] = []
    for s, e, t in hits:
        swallowed = any(t2 != t and s2 <= s and e <= e2 and (e2 - s2) > (e - s) for s2, e2, t2 in hits)
        if not swallowed and t not in kept:
            kept.append(t)
    return kept

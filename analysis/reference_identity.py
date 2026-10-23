"""Deterministic paper identities, shared by snapshot and external loaders.

Only syntactic aliases are merged. Measurements and raw references are retained.
DOI suffixes are case-insensitive; no title similarity or author heuristics apply.
"""
import re
import unicodedata
from urllib.parse import unquote

import pandas as pd


def canonical_reference(value):
    if pd.isna(value) or not str(value).strip():
        raise ValueError("Missing paper reference: cannot assign a valid study group")
    text = unicodedata.normalize("NFKC", unquote(str(value))).strip()
    text = re.sub(r"^(?:https?://)?(?:dx\.)?doi\.org/", "", text, flags=re.I)
    text = re.sub(r"^doi\s*:\s*", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).lower().rstrip(". ")


def normalize_references(df):
    out = df.copy()
    if "report.reference_raw" not in out:
        out["report.reference_raw"] = out["report.reference"]
    out["report.reference"] = out["report.reference_raw"].map(canonical_reference)
    return out

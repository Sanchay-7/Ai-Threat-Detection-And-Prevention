"""Lightweight XSS preprocessing stub.

Current logic is minimal: it copies/cleans text rows. Replace with real cleaning if you add a larger corpus.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd


def _clean(text: str) -> str:
    return "" if text is None else str(text).strip()


def preprocess(raw_path: str | Path, out_path: str | Path):
    raw_path = Path(raw_path)
    out_path = Path(out_path)
    df = pd.read_csv(raw_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Input CSV must contain 'text' and 'label' columns")
    df["text"] = df["text"].map(_clean)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path

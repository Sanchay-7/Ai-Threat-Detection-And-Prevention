"""SQL injection inference: regex first, ML if available."""
from __future__ import annotations
import re
import sys
from pathlib import Path
import joblib
import numpy as np

import config

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SIG_PATTERNS = [
    re.compile(r"'\s*or\s*1=1", re.IGNORECASE),
    re.compile(r"union\s+select", re.IGNORECASE),
    re.compile(r"sleep\s*\(", re.IGNORECASE),
    re.compile(r"benchmark\s*\(", re.IGNORECASE),
    re.compile(r"information_schema", re.IGNORECASE),
]

_ensemble = None
_vectorizer = None


def _load_artifacts():
    """Lazy-load vectorizer and ensemble, with backward compatibility."""
    global _ensemble, _vectorizer

    # Prefer new artifact files
    vec_path = Path("sql_injection/sql_vectorizer.pkl")
    ens_path = Path("sql_injection/sql_ensemble.pkl")

    if _ensemble is None or _vectorizer is None:
        if vec_path.exists() and ens_path.exists():
            _vectorizer = joblib.load(vec_path)
            _ensemble = joblib.load(ens_path)
        elif Path(config.SQL_MODEL_PATH).exists():
            # Backward compatible: single model file with its own vectorizer
            _ensemble = joblib.load(config.SQL_MODEL_PATH)
            _vectorizer = None  # Some older models embed vectorization internally
    return _vectorizer, _ensemble


def _regex_hit(text: str) -> bool:
    return any(p.search(text) for p in _SIG_PATTERNS)


def predict(text: str):
    text = text or ""
    if _regex_hit(text):
        return {"decision": True, "score": 1.0, "reason": "signature_sql"}

    vectorizer, model = _load_artifacts()
    if model is None:
        return {"decision": False, "score": 0.0, "reason": "model_missing"}

    try:
        if vectorizer is not None:
            X = vectorizer.transform([text])
            proba = float(model.predict_proba(X)[0][1])
        else:
            proba = float(model.predict_proba([text])[0][1])
    except Exception:
        proba = 0.0
    decision = proba >= config.SQL_THRESHOLD
    return {"decision": decision, "score": proba, "reason": "ml_sql" if decision else "normal"}


if __name__ == "__main__":
    print(predict("' OR 1=1 --"))

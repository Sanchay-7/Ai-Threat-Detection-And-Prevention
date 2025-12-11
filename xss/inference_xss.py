"""XSS inference: signature/regex first, then ML model if available."""
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
    re.compile(r"<script", re.IGNORECASE),
    re.compile(r"onerror\s*=", re.IGNORECASE),
    re.compile(r"javascript:\s*", re.IGNORECASE),
    re.compile(r"<svg[\s\S]*onload", re.IGNORECASE),
    re.compile(r"<iframe", re.IGNORECASE),
]

_model = None


def _load_model():
    global _model
    if _model is None and Path(config.XSS_MODEL_PATH).exists():
        _model = joblib.load(config.XSS_MODEL_PATH)
    return _model


def _regex_hit(text: str) -> bool:
    return any(p.search(text) for p in _SIG_PATTERNS)


def predict(text: str):
    text = text or ""
    if _regex_hit(text):
        return {"decision": True, "score": 1.0, "reason": "signature_xss"}

    model = _load_model()
    if model is None:
        return {"decision": False, "score": 0.0, "reason": "model_missing"}

    try:
        proba = float(model.predict_proba([text])[0][1])
    except Exception:
        proba = 0.0
    decision = proba >= config.XSS_THRESHOLD
    return {"decision": decision, "score": proba, "reason": "ml_xss" if decision else "normal"}


if __name__ == "__main__":
    print(predict("<script>alert(1)</script>"))

"""Unified detection entrypoint for DDoS, XSS, SQLi."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detector import HybridDetector  # noqa: E402
from xss.inference_xss import predict as predict_xss  # noqa: E402
from sql_injection.inference_sql import predict as predict_sql  # noqa: E402

_ddos = None

def detect_ddos(features, payload=""):
    global _ddos
    if _ddos is None:
        _ddos = HybridDetector()
        _ddos.load_models()
    return _ddos.hybrid_decision(features, payload)

def detect_xss(text: str):
    return predict_xss(text)

def detect_sql(text: str):
    return predict_sql(text)


def detect_all(features, payload=""):
    return {
        "ddos": detect_ddos(features, payload),
        "xss": detect_xss(payload if isinstance(payload, str) else ""),
        "sql": detect_sql(payload if isinstance(payload, str) else ""),
    }

if __name__ == "__main__":
    sample = [1.0, 0.1, 0.5, 200]
    print(detect_all(sample, payload=""))

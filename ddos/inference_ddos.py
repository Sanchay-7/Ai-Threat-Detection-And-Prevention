"""Basic DDoS inference entrypoint using HybridDetector."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detector import HybridDetector  # noqa: E402

_detector = None

def load_detector() -> HybridDetector:
    global _detector
    if _detector is None:
        _detector = HybridDetector()
        _detector.load_models()
    return _detector


def score(features, payload=""):
    hd = load_detector()
    return hd.hybrid_decision(features, payload)

if __name__ == "__main__":
    sample = [1.0, 0.1, 0.5, 200]
    print(score(sample, payload=""))

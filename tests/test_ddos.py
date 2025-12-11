import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unified_api.detect import detect_ddos


def test_ddos_detect_returns_dict():
    res = detect_ddos([0.1, 0.1, 0.1, 10], payload="")
    assert isinstance(res, dict)
    assert "decision" in res and "score" in res

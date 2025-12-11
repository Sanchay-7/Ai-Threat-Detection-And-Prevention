import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unified_api.detect import detect_xss


def test_xss_signature_hits():
    res = detect_xss("<script>alert(1)</script>")
    assert res["decision"] is True
    assert "xss" in res["reason"]

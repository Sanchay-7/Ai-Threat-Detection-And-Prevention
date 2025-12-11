import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unified_api.detect import detect_sql


def test_sql_signature_hits():
    res = detect_sql("' OR 1=1 --")
    assert res["decision"] is True
    assert "sql" in res["reason"]

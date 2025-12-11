"""Train DDoS models wrapper reusing root training pipeline."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train as root_train  # noqa: E402

if __name__ == "__main__":
    root_train.main()

# tests/adsorbent_ml/conftest.py
"""Put adsorbent-ml/data on sys.path so the fitting library is importable
as a plain module (it is a library + CLI, not an installed package)."""

import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "adsorbent-ml" / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

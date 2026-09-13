# tests/adsorbent_ml/conftest.py
"""Put adsorbent-ml subpackages on sys.path so the libraries are importable
as plain modules (library + thin CLI, not installed packages — same pattern
as fit_da)."""

import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[2] / "adsorbent-ml"
for _sub in ("data", "models", "training", "eval", "features"):
    _d = ML_ROOT / _sub
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

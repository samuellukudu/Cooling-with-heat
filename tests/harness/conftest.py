# tests/harness/conftest.py
"""Harness test configuration: put Materials/ on sys.path so the V1 parity
tests can import the canonical ``cooling_physics`` module directly."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MATERIALS_DIR = REPO_ROOT / "Materials"
if str(MATERIALS_DIR) not in sys.path:
    sys.path.insert(0, str(MATERIALS_DIR))

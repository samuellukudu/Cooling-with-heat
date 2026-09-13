# tests/harness/conftest.py
"""Harness test configuration: put the vendored V1 reference oracle on
sys.path so the parity tests can import the canonical ``cooling_physics``
module directly (vendored from the archived Materials/ effort)."""

import sys
from pathlib import Path

REFERENCE_DIR = Path(__file__).resolve().parent / "reference"
if str(REFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(REFERENCE_DIR))

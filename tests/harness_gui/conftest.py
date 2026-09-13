"""Offscreen tests for the simulation launcher (harness.gui).

Follows the parked-workbench test pattern: a module-scoped offscreen
QApplication, workers run synchronously, tiny budgets.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app

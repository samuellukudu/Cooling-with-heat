"""Offscreen MainWindow smoke test."""

import pytest
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    yield app
    # don't quit here — let pytest handle


def test_mainwindow_creates(qapp):
    from harness.gui.app import MainWindow

    w = MainWindow()
    assert w.graph is not None
    assert len(w.scene._nodes) > 0
    assert w.scopes is not None
    # cleanup
    w.close()
    w._thread.quit()
    w._thread.wait(2000)
    assert not w._thread.isRunning()


def test_mainwindow_palette_and_inspector(qapp):
    from harness.gui.app import MainWindow

    w = MainWindow()
    # palette should have blocks
    assert w.palette.list.count() > 0
    # inspector initially empty
    assert w.inspector._current_id is None
    # select first node
    nid = w.graph.nodes[0].id
    w.scene._nodes[nid].setSelected(True)
    # trigger selection
    w._on_selection_changed()
    assert w.inspector._current_id == nid
    w.close()
    w._thread.quit()
    w._thread.wait(2000)


def test_mainwindow_run_cycle0d_offscreen(qapp):
    """Run a tiny Cycle0D search offscreen via MainWindow's worker (direct call)."""
    from harness.gui.app import MainWindow
    from harness.gui.model import ExperimentGraph

    w = MainWindow()
    # use tiny budget
    for n in w.graph.nodes:
        if n.type == "Optimizer":
            n.data["budget"] = 8
    # connect to finished
    payloads = []
    errors = []
    w._worker.finished.connect(lambda p: payloads.append(p))
    w._worker.failed.connect(lambda s: errors.append(s))
    # run synchronously via worker (not via MainWindow.run_graph which uses QTimer)
    w._worker.run(w.graph)
    qapp.processEvents()
    assert not errors, f"failed: {errors}"
    assert payloads
    assert "COP" in payloads[0].metrics
    w.close()
    w._thread.quit()
    w._thread.wait(2000)

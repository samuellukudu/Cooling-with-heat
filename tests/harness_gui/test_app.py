"""Launcher MainWindow + kit widget tests (offscreen, synchronous)."""

import time

from harness.gui import runner
from harness.gui.app import ConfigPanel, MainWindow
from harness.gui.kit import CompareScope, DataFrameScope, JobRunner


def test_config_panel_forms_are_registry_driven(qapp):
    panel = ConfigPanel(lambda _s: None)
    panel.env_combo.setCurrentText("Bed1D-v0")
    # dynamic from the factory signature + design space probe
    assert "n_cells" in panel._env_widgets
    assert "q_sat_kg_kg" in panel._design_widgets
    assert "budget" in panel._backend_widgets
    spec = panel.current_spec()
    assert spec.env_name == "Bed1D-v0"
    assert spec.mode == "evaluate"


def test_main_window_construct_and_generate_python(qapp):
    w = MainWindow()
    w.panel.env_combo.setCurrentText("Cycle0D-v0")
    w._show_python()
    code = w.scopes.python.edit.toPlainText()
    compile(code, "script", "exec")
    assert "harness.make(" in code


def test_main_window_run_evaluate_synchronous(qapp):
    w = MainWindow()
    w.panel.env_combo.setCurrentText("Cycle0D-v0")
    w.panel.mode_combo.setCurrentText("evaluate")
    spec = w.panel.current_spec()
    payload = w._execute(spec, type("W", (), {
        "log": type("S", (), {"emit": staticmethod(lambda _s: None)})(),
        "cancel_requested": False,
    })())
    w._on_finished(payload)
    assert "COP" in w.scopes.metrics.table.item(0, 0).text() or \
        w.scopes.metrics.table.rowCount() >= 1
    # store/compare flow
    w._store("a")
    w._store("b")
    w._compare()
    assert w.scopes.compare.table.rowCount() >= 1


def test_job_runner_threaded_dispatch(qapp):
    """The queued started-signal dispatch really runs the job off the UI thread."""
    runner_obj = JobRunner()
    seen = {}

    import threading

    def job(worker):
        seen["thread"] = threading.current_thread().name
        seen["cancelled_flag"] = worker.cancel_requested
        return {"ok": True}

    runner_obj.finished.connect(lambda payload: seen.setdefault("payload", payload))
    runner_obj.start("job", job)
    deadline = time.time() + 10
    while "payload" not in seen and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    runner_obj.shutdown()
    assert seen["payload"] == {"ok": True}
    assert seen["thread"] != threading.current_thread().name


def test_dataframe_and_compare_scopes(qapp):
    import pandas as pd

    df = pd.DataFrame({"metric": ["COP", "SCP"], "value": [0.5, 120.0]})
    table = DataFrameScope()
    table.set_dataframe(df, title="t")
    assert table.table.rowCount() == 2
    assert table.label.text() == "t"
    cmp_scope = CompareScope()
    cmp_scope.set_compare({"COP": 0.5}, {"COP": 0.6}, a_label="A", b_label="B")
    assert cmp_scope.table.rowCount() == 1

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


def test_field_scope_plays_back_and_lands_tab(qapp):
    import numpy as np

    from harness.gui.scopes import FieldScope

    scope = FieldScope()
    n, nc = 24, 8
    fields = {
        "T_K": np.linspace(300, 350, n * nc).reshape(n, nc),
        "q_kg_kg": np.linspace(0.2, 0.05, n * nc).reshape(n, nc),
        "t_s": np.linspace(0, 120, n),
        "x_m": np.linspace(0.0001, 0.0019, nc),
    }
    scope.set_fields(fields, kind="bed1d")
    assert scope.slider.maximum() == n - 1
    scope.slider.setValue(5)
    assert "frame 6/24" in scope.time_label.text()
    scope._toggle()
    assert scope._timer.isActive()
    scope._toggle()
    assert not scope._timer.isActive()

    w = MainWindow()
    w.panel.env_combo.setCurrentText("Bed1D-v0")
    w.panel.mode_combo.setCurrentText("evaluate")
    spec = w.panel.current_spec()
    payload = w._execute(spec, type("W", (), {
        "log": type("S", (), {"emit": staticmethod(lambda _s: None)})(),
        "cancel_requested": False,
    })())
    assert payload.trace is not None and payload.trace.fields is not None
    w.scopes.setCurrentWidget(w.scopes.log)  # passive watcher on the Log tab
    w._on_finished(payload)
    assert w.scopes.currentWidget() is w.scopes.field

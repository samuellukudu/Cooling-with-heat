"""MainWindow — the RL & simulation launcher (PyQt6).

Pick an environment from the harness registry, configure material/profile/
kwargs in forms generated from the factory signatures and design space,
choose evaluate/optimize/sweep, run on a background thread, and watch the
scopes. The node-canvas composer remains parked in ``attic/``.
"""

from __future__ import annotations

import sys
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from . import runner
from .kit import DARK_QSS, JobRunner
from .scopes import ScopeTabs


class ConfigPanel(QWidget):
    """Left-hand form: env, materials, kwargs, design, backend, sweep."""

    def __init__(self, status) -> None:
        super().__init__()
        self.status = status
        self._env_widgets: dict[str, tuple[str, Any]] = {}
        self._design_widgets: dict[str, tuple[float, QDoubleSpinBox]] = {}
        self._backend_widgets: dict[str, tuple[str, Any]] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)

        form = QFormLayout()
        self.env_combo = QComboBox()
        self.env_combo.addItems(list(runner.env_names()))
        self.env_combo.currentTextChanged.connect(self._rebuild_env_form)
        form.addRow("Environment", self.env_combo)

        self.material_combo = QComboBox()
        self.material_combo.setEditable(True)
        self.material_combo.addItems(self._registry_names("materials"))
        form.addRow("Material", self.material_combo)

        self.material_b_edit = QLineEdit()
        self.material_b_edit.setPlaceholderText("(optional — composite pair)")
        form.addRow("Material B", self.material_b_edit)

        self.profile_combo = QComboBox()
        self.profile_combo.setEditable(True)
        self.profile_combo.addItems(self._registry_names("profiles"))
        form.addRow("Profile", self.profile_combo)
        outer.addLayout(form)

        self.env_group = QGroupBox("Environment parameters")
        self.env_form = QFormLayout(self.env_group)
        outer.addWidget(self.env_group)

        self.design_group = QGroupBox("Design (start point / overrides)")
        self.design_form = QFormLayout(self.design_group)
        outer.addWidget(self.design_group)

        mode_group = QGroupBox("Run")
        mode_form = QFormLayout(mode_group)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["evaluate", "optimize"])
        mode_form.addRow("Mode", self.mode_combo)
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["grad", "search", "rl"])
        self.backend_combo.currentTextChanged.connect(self._rebuild_backend_form)
        mode_form.addRow("Backend", self.backend_combo)
        weights = QHBoxLayout()
        self.cop_spin = QDoubleSpinBox()
        self.cop_spin.setRange(0.0, 10.0)
        self.cop_spin.setSingleStep(0.05)
        self.cop_spin.setValue(0.35)
        self.scp_spin = QDoubleSpinBox()
        self.scp_spin.setRange(0.0, 10.0)
        self.scp_spin.setSingleStep(0.05)
        self.scp_spin.setValue(0.30)
        for w, lab in ((self.cop_spin, "COP"), (self.scp_spin, "SCP")):
            weights.addWidget(QLabel(lab))
            weights.addWidget(w)
        mode_form.addRow("Objective weights", weights)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2**31 - 1)
        mode_form.addRow("Seed", self.seed_spin)
        self.trace_check = QCheckBox("Collect trace")
        self.trace_check.setChecked(True)
        mode_form.addRow("", self.trace_check)
        self.backend_form = QFormLayout()
        mode_form.addRow(self.backend_form)
        outer.addWidget(mode_group)

        sweep_group = QGroupBox("Sweep")
        sweep_form = QFormLayout(sweep_group)
        self.sweep_axis = QComboBox()
        self.sweep_axis.addItems(["none", "t_switch", "material"])
        sweep_form.addRow("Axis", self.sweep_axis)
        self.sweep_min = QDoubleSpinBox()
        self.sweep_min.setRange(-1e6, 1e6)
        self.sweep_min.setValue(60.0)
        self.sweep_max = QDoubleSpinBox()
        self.sweep_max.setRange(-1e6, 1e6)
        self.sweep_max.setValue(600.0)
        sweep_form.addRow("min / max", self._pair(self.sweep_min, self.sweep_max))
        self.sweep_steps = QSpinBox()
        self.sweep_steps.setRange(2, 64)
        self.sweep_steps.setValue(6)
        sweep_form.addRow("Steps", self.sweep_steps)
        outer.addWidget(sweep_group)
        outer.addStretch()

        # initial builds (combo currentTextChanged fired before slots existed)
        self._rebuild_backend_form()
        self._rebuild_env_form()

    @staticmethod
    def _pair(a: QWidget, b: QWidget) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(a)
        lay.addWidget(b)
        return w

    @staticmethod
    def _registry_names(kind: str) -> list[str]:
        try:
            import harness  # noqa: PLC0415

            return list(harness.REGISTRIES[kind].names())
        except Exception:
            return []

    def spec_kwargs(self) -> dict[str, Any]:
        return {k: v for k, v in self._collect(self._env_widgets).items()}

    def _collect(self, store: dict[str, tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, (kind, w) in store.items():
            if kind == "bool":
                out[name] = w.isChecked()
            elif kind == "int":
                out[name] = int(w.value())
            elif kind == "float":
                out[name] = float(w.value())
            else:
                txt = w.text().strip() if hasattr(w, "text") else ""
                if txt != "":
                    out[name] = txt
        return out

    def current_spec(self) -> runner.RunSpec:
        overrides = {}
        for key, (default, w) in self._design_widgets.items():
            if abs(float(w.value()) - default) > 1e-12:
                overrides[key] = float(w.value())
        backend_kwargs = self._collect(self._backend_widgets)
        return runner.RunSpec(
            env_name=self.env_combo.currentText(),
            material_ref=self.material_combo.currentText().strip(),
            profile_ref=self.profile_combo.currentText().strip(),
            material_b_ref=self.material_b_edit.text().strip(),
            env_kwargs=self.spec_kwargs(),
            design_overrides=overrides,
            mode=self.mode_combo.currentText(),
            backend=self.backend_combo.currentText(),
            cop_weight=float(self.cop_spin.value()),
            scp_weight=float(self.scp_spin.value()),
            seed=int(self.seed_spin.value()),
            collect_trace=self.trace_check.isChecked(),
            backend_kwargs=backend_kwargs,
            sweep_axis=self.sweep_axis.currentText(),
            sweep_min=float(self.sweep_min.value()),
            sweep_max=float(self.sweep_max.value()),
            sweep_steps=int(self.sweep_steps.value()),
        )

    def refresh_design(self) -> None:
        """Probe-build the configured problem and rebuild the design form."""
        while self.design_form.rowCount():
            self.design_form.removeRow(0)
        self._design_widgets.clear()
        try:
            problem = runner.build_problem(self.current_spec())
        except Exception as exc:  # noqa: BLE001
            self.status(f"design probe failed: {exc}")
            return
        ds = getattr(problem, "design_space", None)
        if ds is None:
            self.design_form.addRow(QLabel("(static problem — no design space)"))
            return
        for key in ds.keys:
            lo_list, hi_list = ds.bounds_for([key])
            default = float(ds.defaults.get(key, 0.0))
            spin = QDoubleSpinBox()
            spin.setDecimals(6)
            spin.setRange(float(lo_list[0]), float(hi_list[0]))
            spin.setValue(default)
            self.design_form.addRow(key, spin)
            self._design_widgets[key] = (default, spin)

    def _rebuild_env_form(self) -> None:
        while self.env_form.rowCount():
            self.env_form.removeRow(0)
        self._env_widgets.clear()
        try:
            params = runner.env_factory_params(self.env_combo.currentText())
        except Exception as exc:  # noqa: BLE001
            self.status(f"introspection failed: {exc}")
            return
        for name, info in params.items():
            if name in runner.SPECIAL_PARAMS:
                continue
            w = self._widget_for(info)
            self.env_form.addRow(name, w)
            self._env_widgets[name] = (info["kind"], w)
        self.refresh_design()

    def _rebuild_backend_form(self) -> None:
        while self.backend_form.rowCount():
            self.backend_form.removeRow(0)
        self._backend_widgets.clear()
        params = runner.backend_params().get(self.backend_combo.currentText(), {})
        for name, info in params.items():
            if name == "total_timesteps":
                continue
            w = self._widget_for(info, lo=1 if info["kind"] == "int" else 0)
            self.backend_form.addRow(name, w)
            self._backend_widgets[name] = (info["kind"], w)

    @staticmethod
    def _widget_for(info: dict[str, Any], lo: int = 0) -> Any:
        kind, default = info["kind"], info["default"]
        if kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(default))
            return w
        if kind == "int":
            w = QSpinBox()
            w.setRange(lo, 10**9)
            w.setValue(int(default) if default is not None else 0)
            return w
        if kind == "float":
            w = QDoubleSpinBox()
            w.setRange(-1e9, 1e9)
            w.setDecimals(6)
            w.setValue(float(default) if default is not None else 0.0)
            return w
        line = QLineEdit()
        if default is not None:
            line.setText(str(default))
        return line


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        # VRAM hygiene BEFORE anything imports harness/jax: no preallocation,
        # freed memory returned to the driver (4 GB shared-GPU reality).
        from .. import gpu  # noqa: PLC0415

        gpu.configure()
        self.setWindowTitle("Harness — RL & simulation lab")
        self.resize(1360, 860)
        self.setStyleSheet(DARK_QSS)

        self._result_a = self._result_b = None
        self._last_payload = None

        self.panel = ConfigPanel(self._status_msg)
        panel_scroll = QScrollArea()
        panel_scroll.setWidget(self.panel)
        panel_scroll.setWidgetResizable(True)
        panel_scroll.setMinimumWidth(360)

        self.scopes = ScopeTabs(self)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setHandleWidth(6)
        splitter.addWidget(panel_scroll)
        splitter.addWidget(self.scopes)
        splitter.setSizes([420, 940])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        self.runner_thread = JobRunner(self)
        self.runner_thread.started.connect(lambda label: (self._status_msg(label),
                                                          self.scopes.append_log(f"## {label}")))
        self.runner_thread.log.connect(self.scopes.append_log)
        self.runner_thread.finished.connect(self._on_finished)
        self.runner_thread.failed.connect(self._on_failed)

        self._build_toolbar()
        self._status = QStatusBar(self)
        self.setStatusBar(self._status)
        self._status_msg("ready — pick an environment and Run")

    # -- toolbar ---------------------------------------------------------------

    def _build_toolbar(self) -> None:
        tb = QToolBar("run")
        tb.setMovable(False)
        self.addToolBar(tb)
        self.run_btn = QAction("▶ Run", self)
        self.run_btn.setShortcut(QKeySequence("Ctrl+R"))
        self.run_btn.triggered.connect(self.run)
        tb.addAction(self.run_btn)
        self.stop_btn = QAction("■ Stop", self)
        self.stop_btn.triggered.connect(self._stop)
        tb.addAction(self.stop_btn)
        tb.addSeparator()
        store_a = QAction("Store A", self)
        store_a.triggered.connect(lambda: self._store("a"))
        tb.addAction(store_a)
        store_b = QAction("Store B", self)
        store_b.triggered.connect(lambda: self._store("b"))
        tb.addAction(store_b)
        compare_btn = QAction("Compare A/B", self)
        compare_btn.triggered.connect(self._compare)
        tb.addAction(compare_btn)
        tb.addSeparator()
        py_btn = QAction("Generate Python", self)
        py_btn.setShortcut(QKeySequence("Ctrl+P"))
        py_btn.triggered.connect(self._show_python)
        tb.addAction(py_btn)

    # -- actions ---------------------------------------------------------------

    def run(self) -> None:
        spec = self.panel.current_spec()
        self.run_btn.setEnabled(False)
        self.runner_thread.start(
            f"Running {spec.env_name} ({spec.sweep_axis or spec.mode})",
            lambda worker, spec=spec: self._execute(spec, worker),
        )

    @staticmethod
    def _execute(spec, worker):
        return runner.execute(
            spec,
            log=worker.log.emit,
            is_cancelled=lambda: worker.cancel_requested,
        )
    def _stop(self) -> None:
        self.runner_thread.request_cancel()
        self._status_msg("cancel requested (takes effect between sweep points)")

    def _store(self, slot: str) -> None:
        payload = getattr(self, "_last_payload", None)
        if payload is None:
            self._status_msg("nothing to store yet — Run first")
            return
        setattr(self, f"_result_{slot}", payload)
        self._status_msg(f"stored result in slot {slot.upper()}")

    def _compare(self) -> None:
        a, b = self._result_a, self._result_b
        if a is None and b is None:
            self._status_msg("store A and/or B first")
            return
        self.scopes.compare.set_compare(a, b, title="Compare — stored results")
        self.scopes.setCurrentWidget(self.scopes.compare)

    def _show_python(self) -> None:
        from . import runner as runner_mod  # noqa: PLC0415

        self.scopes.python.set_text(
            runner_mod.python_script(self.panel.current_spec()),
            title="Python — reproducible script")
        self.scopes.setCurrentWidget(self.scopes.python)

    # -- worker callbacks --------------------------------------------------------

    def _on_finished(self, payload) -> None:
        self.run_btn.setEnabled(True)
        self._last_payload = payload
        self.scopes.set_result(payload)
        self._status_msg(f"done in {payload.elapsed_s:.2f}s")

    def _on_failed(self, message: str) -> None:
        self.run_btn.setEnabled(True)
        self.scopes.append_log(f"FAILED: {message}")
        self._status_msg("run failed — see Log tab")

    def _status_msg(self, text: str) -> None:
        self._status.showMessage(text)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.runner_thread.shutdown()
        super().closeEvent(event)


def run() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    app.exec()


if __name__ == "__main__":  # pragma: no cover
    run()

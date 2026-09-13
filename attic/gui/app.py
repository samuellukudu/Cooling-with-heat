"""MainWindow — the PyQt6 desktop workbench."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, QThread, pyqtSlot
from PyQt6.QtGui import QAction, QColor, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .command_palette import CommandPalette, apply_patch, parse_command
from .executor import ExecutorResult, JobWorker
from .inspector import InspectorDock
from .model import ExperimentGraph
from .palette import PaletteDock
from .scene import CanvasScene, CanvasView
from .scopes import ScopeTabs


DARK_QSS = """
QMainWindow { background: #0f1115; }
QDockWidget { background: #0f1115; color: #e2e8f0; }
QDockWidget::title { background: #1b1e24; padding: 4px 8px; font-size: 8.5pt; color: #94a3b8; }
QToolBar { background: #0f1115; spacing: 6px; padding: 4px; border: none; }
QStatusBar { background: #0f1115; color: #64748b; font-size: 7.5pt; }
QTabWidget::pane { border: 1px solid #1e222a; background: #0f1115; }
QTabBar::tab { background: #1b1e24; color: #94a3b8; padding: 5px 10px; margin-right: 2px; border-top-left-radius: 6px; border-top-right-radius: 6px; }
QTabBar::tab:selected { background: #1e293b; color: #e2e8f0; }
QListWidget, QTableWidget, QTableView, QTextEdit, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: #111318; color: #e2e8f0; border: 1px solid #1e222a; border-radius: 6px; padding: 3px;
}
QListWidget::item:selected { background: #1e293b; color: #e2e8f0; }
QHeaderView::section { background: #1b1e24; color: #94a3b8; padding: 4px; border: none; font-size: 7.5pt; }
QPushButton { background: #1e293b; color: #e2e8f0; border: 1px solid #2a2f3a; border-radius: 6px; padding: 4px 8px; font-size: 8.5pt; }
QPushButton:hover { background: #334155; }
QPushButton:pressed { background: #0f1115; }
QPushButton:disabled { color: #475569; background: #111318; }
QCheckBox { color: #cbd5e1; spacing: 6px; }
QLabel { color: #cbd5e1; }
"""


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("harness — Cooling-with-Heat Workbench  [PyQt6 desktop]")
        self.resize(1380, 860)
        self.setStyleSheet(DARK_QSS)

        # model
        self.graph = ExperimentGraph.default_cycle0d()
        self._file_path: Path | None = None
        self._dirty = False

        # scene / view
        self.scene = CanvasScene(self)
        self.scene.set_graph(self.graph)
        self.scene.graph_changed.connect(self._on_graph_changed)
        self.scene.selection_changed.connect(self._on_selection_changed)
        self.scene.node_double_clicked.connect(self._on_node_double_clicked)
        self.view = CanvasView(self.scene, self)
        self.view.request_inspect.connect(self._on_request_inspect)

        # docks — all closable/movable/floatable (fixes "cannot minimize/close")
        self.palette = PaletteDock(self)
        self.palette.block_requested.connect(self._on_add_block)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.palette)

        self.inspector = InspectorDock(self)
        self.inspector.node_changed.connect(self._on_node_patch)
        self.inspector.delete_requested.connect(self._on_delete_node)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.inspector)

        # scopes as central-tab widget + also hideable via splitter collapse
        self.scopes = ScopeTabs(self)
        self.scopes.setMinimumWidth(220)

        # central splitter: view | scopes — right pane collapsible (double-click handle to hide)
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setHandleWidth(6)
        splitter.addWidget(self.view)
        splitter.addWidget(self.scopes)
        splitter.setSizes([880, 500])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setChildrenCollapsible(True)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, True)
        self._splitter = splitter  # keep ref for View toggles
        self.setCentralWidget(splitter)

        # menu bar
        self._build_menus()

        # toolbar
        self._build_toolbar()

        # status bar
        self._status = QStatusBar(self)
        self.setStatusBar(self._status)
        self._status_label = QLabel("Ready — build a graph: Material → Profile → Physics → Objective → Optimizer → Scope")
        self._status_label.setStyleSheet("color:#64748b; font-size:7.5pt;")
        self._status.addWidget(self._status_label, 1)
        self._validation_label = QLabel("")
        self._validation_label.setStyleSheet("color:#f59e0b; font-size:7.5pt;")
        self._status.addPermanentWidget(self._validation_label)

        # worker thread
        self._thread = QThread(self)
        self._worker = JobWorker()
        self._worker.moveToThread(self._thread)
        self._worker.started.connect(self._on_job_started)
        self._worker.progressed.connect(self._on_job_progress)
        self._worker.finished.connect(self._on_job_finished)
        self._worker.failed.connect(self._on_job_failed)
        self._worker.log.connect(self._on_job_log)
        self._thread.start()
        self._running = False
        # Compare history: keep last two successful results for diff (G3/G4)
        self._last_result: Any | None = None
        self._prev_result: Any | None = None

        # actions / shortcuts
        self._build_actions()

        self._refresh_validation()
        self._on_selection_changed()

    # -- toolbar ------------------------------------------------------------

    def _build_toolbar(self) -> None:
        bar = QToolBar("Main", self)
        bar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, bar)

        def act(text: str, tip: str, slot: Any) -> QAction:
            a = QAction(text, self)
            a.setToolTip(tip)
            a.triggered.connect(slot)
            bar.addAction(a)
            return a

        act("New", "New graph (Cycle0D demo)", self.new_graph)
        act("Open…", "Open *.harness.json", self.open_graph)
        self._save_act = act("Save", "Save graph", self.save_graph)
        act("Save As…", "Save graph as…", self.save_graph_as)
        bar.addSeparator()
        self._run_btn = QPushButton("▶  Run")
        self._run_btn.setToolTip("Execute the experiment graph (validates, then harness.optimize/rollout in worker thread)")
        self._run_btn.setStyleSheet("background:#16a34a; color:white; font-weight:600; padding:5px 14px; border-radius:6px;")
        self._run_btn.clicked.connect(self.run_graph)
        bar.addWidget(self._run_btn)
        self._stop_btn = QPushButton("■ Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        bar.addWidget(self._stop_btn)
        bar.addSeparator()
        pal_btn = QPushButton("⌘  Command…")
        pal_btn.setToolTip("Wolfram-style command palette (Ctrl+K)")
        pal_btn.clicked.connect(self.open_palette)
        bar.addWidget(pal_btn)
        py_btn = QPushButton("Generate Python")
        py_btn.setToolTip("Copy reproducible harness script to clipboard / save .py")
        py_btn.clicked.connect(self.generate_python)
        bar.addWidget(py_btn)
        bar.addSeparator()
        # run opts
        bar.addWidget(QLabel("seed"))
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 999999)
        self._seed_spin.setValue(int(self.graph.seed))
        self._seed_spin.valueChanged.connect(lambda v: self._set_graph_attr("seed", int(v)))
        bar.addWidget(self._seed_spin)
        self._trace_check = QCheckBox("Collect trace")
        self._trace_check.setChecked(bool(self.graph.collect_trace))
        self._trace_check.toggled.connect(lambda v: self._set_graph_attr("collect_trace", bool(v)))
        bar.addWidget(self._trace_check)
        # presets
        preset_btn = QPushButton("Presets ▾")
        preset_btn.setToolTip("Load a preset graph")
        # simple menu via click
        from PyQt6.QtWidgets import QMenu  # noqa: WPS433

        menu = QMenu(preset_btn)
        for name in ("Cycle0D demo", "Bed1D demo"):
            a = menu.addAction(name)
            a.triggered.connect(lambda _checked=False, n=name: self.load_preset(n))
        preset_btn.setMenu(menu)
        bar.addWidget(preset_btn)

    def _build_menus(self) -> None:
        mb = self.menuBar()

        def _act(menu: Any, text: str, slot: Any, shortcut: Any = None) -> QAction:
            a = QAction(text, self)
            if shortcut is not None:
                a.setShortcut(shortcut)
            a.triggered.connect(slot)
            menu.addAction(a)
            return a

        m_file = mb.addMenu("&File")
        _act(m_file, "New (Cycle0D)", self.new_graph, QKeySequence("Ctrl+N"))
        _act(m_file, "Open…", self.open_graph, QKeySequence.StandardKey.Open)
        _act(m_file, "Save", self.save_graph, QKeySequence.StandardKey.Save)
        _act(m_file, "Save As…", self.save_graph_as, QKeySequence("Ctrl+Shift+S"))
        m_file.addSeparator()
        _act(m_file, "Generate Python…", self.generate_python, QKeySequence("Ctrl+G"))
        m_file.addSeparator()
        _act(m_file, "Quit", self.close, QKeySequence.StandardKey.Quit)

        m_exp = mb.addMenu("&Experiment")
        _act(m_exp, "▶ Run", self.run_graph, QKeySequence("Ctrl+R"))
        _act(m_exp, "■ Stop", self._on_stop, QKeySequence("Ctrl+."))  # type: ignore[arg-type]
        m_exp.addSeparator()
        _act(m_exp, "Presets → Cycle0D demo", lambda: self.load_preset("Cycle0D demo"))
        _act(m_exp, "Presets → Bed1D demo", lambda: self.load_preset("Bed1D demo"))
        m_exp.addSeparator()
        _act(m_exp, "Ranking (datacenter, 6 anchors)…", lambda: self.run_ranking("datacenter"))
        _act(m_exp, "Ranking (all profiles, 6 anchors)…", lambda: self.run_ranking_all())
        _act(m_exp, "Calibration (Uyun/Sztekler subset)…", self.run_calibration)
        m_exp.addSeparator()
        _act(m_exp, "Sweep — Bed1D t_switch…", self.run_sweep)
        _act(m_exp, "Sweep — graph Sweep node (executor)…", self.run_sweep_via_graph)
        _act(m_exp, "Compare — last two results diff…", self.run_compare)
        _act(m_exp, "V3 oracle-limit check…", self.run_v3)

        m_view = mb.addMenu("&View")
        _act(m_view, "Command palette…", self.open_palette, QKeySequence("Ctrl+K"))
        _act(m_view, "Focus Canvas", lambda: self.view.setFocus(), QKeySequence("Ctrl+1"))
        _act(m_view, "Focus Inspector", lambda: self.inspector.setFocus(), QKeySequence("Ctrl+2"))
        m_view.addSeparator()
        # Dock/pane visibility — fixes "cannot minimize/close" (image): each pane now
        # toggleable via toggleViewAction and splitter collapse.
        m_view.addAction(self.palette.toggleViewAction())
        m_view.addAction(self.inspector.toggleViewAction())
        # Scopes is central splitter pane, not a dock — provide explicit show/hide
        self._scopes_action = QAction("Results (Scopes)", self)
        self._scopes_action.setCheckable(True)
        self._scopes_action.setChecked(True)
        self._scopes_action.setShortcut(QKeySequence("Ctrl+3"))
        self._scopes_action.setStatusTip("Show/hide the right-hand Results tabs (Metrics/History/Trace/Ranking...)")
        self._scopes_action.triggered.connect(self._toggle_scopes)
        m_view.addAction(self._scopes_action)
        _act(m_view, "Reset Layout", self._reset_layout)

        m_help = mb.addMenu("&Help")
        _act(m_help, "Shortcuts…", self._show_help)
        _act(m_help, "About harness GUI…", self._show_about)

    def _show_help(self) -> None:
        QMessageBox.information(self, "Shortcuts",
            "harness GUI — shortcuts\n\n"
            "Ctrl+K / Ctrl+P  Command palette (Wolfram)\n"
            "Ctrl+R  Run graph\n"
            "Ctrl+S  Save   •   Ctrl+O  Open\n"
            "Ctrl+G  Generate Python\n"
            "Delete  Remove selected block\n"
            "Space-drag / Middle-drag  Pan canvas\n"
            "Wheel  Zoom   •   Double-click node → Inspector\n"
            "Right-drag a port to wire  •  Budget ≤ 30 for Bed1D demo")

    def _show_about(self) -> None:
        try:
            import harness as _h  # noqa: WPS433
            hv = _h.__version__
        except Exception:
            hv = "?"
        QMessageBox.about(self, "About harness GUI",
            f"harness GUI  —  Cooling-with-Heat Workbench (PyQt6)\n\n"
            f"harness {hv}  •  ExperimentGraph JSON + JobWorker (QThread)\n"
            f"Headless harness stays pure (lint-imports: 3 kept)\n\n"
            f"Palette → Canvas → Inspector → Run → Scopes\n"
            f"See harness/gui/README.md and DESIGN.md §3–§6.")

    def _toggle_scopes(self, checked: bool) -> None:
        self.scopes.setVisible(checked)
        # when hidden, give canvas full width; when shown, restore split
        if checked and hasattr(self, "_splitter"):
            self._splitter.setSizes([720, 660])

    def _reset_layout(self) -> None:
        self.palette.setVisible(True)
        self.inspector.setVisible(True)
        self._scopes_action.setChecked(True)
        self._toggle_scopes(True)
        self.palette.toggleViewAction().setChecked(True)
        self.inspector.toggleViewAction().setChecked(True)
        if hasattr(self, "_splitter"):
            self._splitter.setSizes([880, 500])

    def _build_actions(self) -> None:
        # Ctrl+K palette
        a = QAction(self)
        a.setShortcut(QKeySequence("Ctrl+K"))
        a.triggered.connect(self.open_palette)
        self.addAction(a)
        # Ctrl+P also
        b = QAction(self)
        b.setShortcut(QKeySequence("Ctrl+P"))
        b.triggered.connect(self.open_palette)
        self.addAction(b)
        # Ctrl+S save
        c = QAction(self)
        c.setShortcut(QKeySequence.StandardKey.Save)
        c.triggered.connect(self.save_graph)
        self.addAction(c)
        # Ctrl+O open
        d = QAction(self)
        d.setShortcut(QKeySequence.StandardKey.Open)
        d.triggered.connect(self.open_graph)
        self.addAction(d)
        # Ctrl+R run
        e = QAction(self)
        e.setShortcut(QKeySequence("Ctrl+R"))
        e.triggered.connect(self.run_graph)
        self.addAction(e)
        # Ctrl+3 toggle Results pane
        f = QAction(self)
        f.setShortcut(QKeySequence("Ctrl+3"))
        f.triggered.connect(lambda: self._scopes_action.toggle())
        self.addAction(f)

    # -- graph mutators -----------------------------------------------------

    def _on_add_block(self, block: dict[str, Any]) -> None:
        # place near center of view
        center = self.view.mapToScene(self.view.viewport().rect().center())
        spec = self.graph.add_node(block["type"], x=float(center.x() - 80), y=float(center.y() - 35),
                                   data=dict(block.get("data", {})), label=str(block.get("label", "")))
        # reflect in scene directly (model already mutated — avoid double add)
        # scene.add_node would re-append to graph; instead create item manually
        from .scene import NodeItem  # noqa: WPS433

        item = NodeItem(spec)
        self.scene.addItem(item)
        self.scene._nodes[spec.id] = item  # type: ignore[attr-defined]
        self._dirty = True
        self._refresh_validation()

    def _on_node_patch(self, node_id: str, patch: dict[str, Any]) -> None:
        node = self.graph.node_by_id(node_id)
        if not node:
            return
        # Research-fit: None means delete key unless the field is nullable
        # (dt_phys_s / material_b legitimately use None for "auto/none").
        _nullable = {"dt_phys_s", "material_b"}
        for k, v in list(patch.items()):
            if v is None and k not in _nullable:
                # delete custom field
                if k in node.data:
                    del node.data[k]
                patch.pop(k, None)
            elif v is None and k in _nullable:
                node.data[k] = None
                patch.pop(k, None)
        if patch:
            node.data.update(patch)
        # when Physics kind changes, ensure defaults for the new kind are present
        if node.type == "Physics" and "kind" in patch:
            from .model import DEFAULT_NODE_DATA, PHYSICS_KWARGS  # noqa: WPS433
            kind = patch["kind"]
            allowed = PHYSICS_KWARGS.get(kind, set())
            defaults = DEFAULT_NODE_DATA.get("Physics", {})
            for k in allowed:
                if k not in node.data or node.data[k] is None:
                    if k in defaults:
                        node.data[k] = defaults[k]
        self.scene.refresh_node(node_id)
        # keep inspector in sync (e.g. after kind switch, show new fields immediately)
        # refresh inspector with updated data
        try:
            cur = self.scene.selected_node_ids()
            if node_id in cur:
                self.inspector.show_node(node.id, node.type, dict(node.data))
        except Exception:
            pass
        self._dirty = True
        self._refresh_validation()

    def _on_duplicate_node(self, node_id: str) -> None:
        item = self.scene.duplicate_node(node_id)  # type: ignore[attr-defined]
        if item:
            self._dirty = True
            self._refresh_validation()
            # select the duplicate
            try:
                self.scene.clearSelection()
                item.setSelected(True)
                self._on_selection_changed()
            except Exception:
                pass

    def _on_delete_node(self, node_id: str) -> None:
        self.scene.remove_node(node_id)
        self._dirty = True
        self._refresh_validation()
        self.inspector.clear_selection()

    def _set_graph_attr(self, key: str, value: Any) -> None:
        setattr(self.graph, key, value)
        self._dirty = True

    def _on_graph_changed(self) -> None:
        self._dirty = True
        self._refresh_validation()

    def _on_selection_changed(self) -> None:
        ids = self.scene.selected_node_ids()
        if not ids:
            self.inspector.clear_selection()
            return
        nid = ids[0]
        node = self.graph.node_by_id(nid)
        if node:
            self.inspector.show_node(node.id, node.type, dict(node.data))

    def _on_node_double_clicked(self, nid: str) -> None:
        node = self.graph.node_by_id(nid)
        if node:
            self.inspector.show_node(node.id, node.type, dict(node.data))
            self.inspector.raise_()

    def _on_request_inspect(self, nid: str) -> None:
        self._on_node_double_clicked(nid)

    def _refresh_validation(self) -> None:
        errs = self.graph.validate()
        hard = [e for e in errs if e["code"] not in ("schedule_on_static",)]
        if hard:
            self._validation_label.setText(f"⚠ {len(hard)} issue(s) — see Log")
            self._validation_label.setStyleSheet("color:#f59e0b; font-size:7.5pt;")
            # mark edges bad if port mismatch
            bad_edge_ids = {e["node"] for e in errs if e["code"] in ("bad_port", "port_mismatch", "bad_edge")}
            self.scene.mark_bad_edges(bad_edge_ids)
        else:
            # show soft warnings
            soft = [e for e in errs if e["code"] == "schedule_on_static"]
            if soft:
                self._validation_label.setText(f"ⓘ {soft[0]['message'][:60]}")
                self._validation_label.setStyleSheet("color:#38bdf8; font-size:7.5pt;")
            else:
                self._validation_label.setText("✓ valid")
                self._validation_label.setStyleSheet("color:#22c55e; font-size:7.5pt;")
            self.scene.mark_bad_edges(set())

    # -- file ops -----------------------------------------------------------

    def new_graph(self) -> None:
        if self._confirm_discard():
            self.graph = ExperimentGraph.default_cycle0d()
            self._file_path = None
            self.scene.set_graph(self.graph)
            self._seed_spin.setValue(int(self.graph.seed))
            self._trace_check.setChecked(bool(self.graph.collect_trace))
            self.scopes.log.set_text("New graph — Cycle0D demo loaded.", title="Log")
            self._dirty = False
            self._update_title()

    def load_preset(self, name: str) -> None:
        if not self._confirm_discard():
            return
        presets = ExperimentGraph.presets()
        g = presets.get(name)
        if not g:
            return
        self.graph = g
        self._file_path = None
        self.scene.set_graph(self.graph)
        self._seed_spin.setValue(int(self.graph.seed))
        self._trace_check.setChecked(bool(self.graph.collect_trace))
        self.scopes.log.set_text(f"Preset loaded: {name}", title="Log")
        self._dirty = False
        self._update_title()
        self._refresh_validation()

    def open_graph(self) -> None:
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open experiment", str(Path.home()), "Harness graph (*.harness.json *.json);;All (*)")
        if not path:
            return
        try:
            g = ExperimentGraph.load(path)
            # migration note: bump handling already in model.from_dict (schemaVersion)
            if g.schema_version < 2:
                g.schema_version = 2
            self.graph = g
            self._file_path = Path(path)
            self.scene.set_graph(self.graph)
            self._seed_spin.setValue(int(self.graph.seed))
            self._trace_check.setChecked(bool(self.graph.collect_trace))
            self.scopes.log.set_text(f"Loaded {path} (schemaVersion={g.schema_version})", title="Log")
            if int(g.schema_version) != int(g.to_dict().get("schemaVersion", 0)):
                self.scopes.log.append(f"Migrated graph to schemaVersion {g.schema_version}")
            self._dirty = False
            self._update_title()
            self._refresh_validation()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Open failed", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")

    def save_graph(self) -> None:
        if self._file_path is None:
            self.save_graph_as()
            return
        try:
            # sync scene positions into graph before save
            self._sync_scene_positions()
            self.graph.save(self._file_path)
            self._dirty = False
            self._update_title()
            self._status_label.setText(f"Saved {self._file_path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save failed", f"{type(exc).__name__}: {exc}")

    def save_graph_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save experiment", str(Path.home() / "experiment.harness.json"), "Harness graph (*.harness.json);;JSON (*.json);;All (*)")
        if not path:
            return
        try:
            self._sync_scene_positions()
            p = Path(path)
            self.graph.save(p)
            self._file_path = p
            self._dirty = False
            self._update_title()
            self._status_label.setText(f"Saved {p}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save failed", f"{type(exc).__name__}: {exc}")

    def _sync_scene_positions(self) -> None:
        for nid, item in self.scene._nodes.items():  # type: ignore[attr-defined]
            spec = self.graph.node_by_id(nid)
            if spec:
                spec.x = float(item.pos().x())
                spec.y = float(item.pos().y())

    def _confirm_discard(self) -> bool:
        if not self._dirty:
            return True
        box = QMessageBox(self)
        box.setWindowTitle("Unsaved changes")
        box.setText("You have unsaved changes — discard them?")
        box.setStandardButtons(QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(QMessageBox.StandardButton.Cancel)
        return box.exec() == QMessageBox.StandardButton.Discard

    def _update_title(self) -> None:
        name = self._file_path.name if self._file_path else self.graph.name
        dirty = " •" if self._dirty else ""
        self.setWindowTitle(f"harness — {name}{dirty}  —  Cooling-with-Heat Workbench")

    # -- palette / python ---------------------------------------------------

    def open_palette(self) -> None:
        patch = CommandPalette.get_patch(self)
        if not patch:
            return
        # apply to model then refresh scene
        assumptions = apply_patch(self.graph, patch)
        # full scene rebuild to reflect structural changes (new nodes/edges)
        self.scene.set_graph(self.graph)
        # select a mutated node if any
        if assumptions:
            self.scopes.log.append("Palette patch:\n" + "\n".join(f"  • {a}" for a in assumptions))
            self.scopes.log.append(f"Raw patch: {patch}")
        self._dirty = True
        self._refresh_validation()
        self.scopes.setCurrentWidget(self.scopes.log)

    def generate_python(self) -> None:
        code = self.graph.to_python()
        self.scopes.python.set_text(code, title="Python — reproducible script")
        self.scopes.setCurrentWidget(self.scopes.python)
        # also copy to clipboard
        from PyQt6.QtWidgets import QApplication  # noqa: WPS433

        QApplication.clipboard().setText(code)
        self._status_label.setText("Python script copied to clipboard — also in Python tab (Save…)")

    # -- run ---------------------------------------------------------------

    def run_graph(self) -> None:
        if self._running:
            return
        errs = [e for e in self.graph.validate() if e["code"] not in ("schedule_on_static",)]
        if errs:
            self.scopes.log.set_text("Cannot run — fix validation errors first:\n" + "\n".join(f"  • {e['message']}" for e in errs), title="Log")
            self.scopes.setCurrentWidget(self.scopes.log)
            return
        self._sync_scene_positions()
        # reflect seed/trace from toolbar
        self.graph.seed = int(self._seed_spin.value())
        self.graph.collect_trace = bool(self._trace_check.isChecked())
        self._running = True
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._status_label.setText("Running… (worker thread — JAX isolated)")
        self.scopes.log.set_text(f"Dispatching {self.graph.name}  seed={self.graph.seed}  collect_trace={self.graph.collect_trace}\n", title="Log")
        # mark nodes running
        for item in self.scene._nodes.values():  # type: ignore[attr-defined]
            item.set_status(QColor("#f59e0b"))
        # invoke worker via queued signal
        self._worker.request_cancel = False  # type: ignore[attr-defined]
        # Use singleShot to ensure queued
        from PyQt6.QtCore import QTimer  # noqa: WPS433

        QTimer.singleShot(0, lambda: self._worker.run(self.graph))

    def _on_stop(self) -> None:
        try:
            self._worker.request_cancel()  # type: ignore[attr-defined]
        except Exception:
            pass
        self._status_label.setText("Cancel requested…")
        self._stop_btn.setEnabled(False)

    @pyqtSlot(str)
    def _on_job_started(self, msg: str) -> None:
        self.scopes.log.append(msg)
        self._status_label.setText(msg)

    @pyqtSlot(dict)
    def _on_job_progress(self, info: dict) -> None:
        self.scopes.log.append(f"progress: {info}")
        self._status_label.setText(f"progress: {info}")

    @pyqtSlot(object)
    def _on_job_finished(self, payload: ExecutorResult) -> None:
        self._running = False
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        # reset node status to done
        for item in self.scene._nodes.values():  # type: ignore[attr-defined]
            item.set_status(QColor("#22c55e"))
        self.scopes.set_result(payload)
        # handle sweep payload routing (G3/G4)
        sweep_df = getattr(payload, "sweep_df", None)
        sweep_axis = getattr(payload, "sweep_axis", None)
        if sweep_df is not None:
            try:
                self.scopes.set_sweep_dataframe(sweep_df, title=f"Sweep — {sweep_axis} ({len(sweep_df)} pts)")
                self.scopes.log.append(f"\n— sweep ({sweep_axis}) — {len(sweep_df)} points — see Sweep tab (table + curve)")
                # also push to History as SCP envelope for quick view
                if "SCP_W_kg" in getattr(sweep_df, "columns", []):
                    try:
                        import pandas as pd  # noqa: WPS433
                        rows = sweep_df.to_dict(orient="records")  # type: ignore
                        self.scopes.history.set_history([{"objective": float(r.get("SCP_W_kg", 0))} for r in rows],
                                                        best_objective=float(sweep_df["SCP_W_kg"].max()))
                    except Exception:
                        pass
                self._status_label.setText(f"Sweep done ({sweep_axis}) in {payload.elapsed_s:.2f}s — {len(sweep_df)} pts")
                self.scopes.setCurrentWidget(self.scopes.sweep)
                # keep payload for compare (sweep counts as result for compare)
                self._prev_result = self._last_result
                self._last_result = payload
            except Exception as e:
                self.scopes.log.append(f"Sweep post-processing failed: {e}")
            self._refresh_validation()
            return
        # store for Compare (keep two deep)
        self._prev_result = self._last_result
        self._last_result = payload
        # log summary
        self.scopes.log.append("\n— result —\n" + payload.summary_text)
        self.scopes.log.append(f"\nElapsed {payload.elapsed_s:.2f}s")
        if payload.result is not None and getattr(payload.result, "best_design", None):
            self.scopes.log.append(f"best_design: {payload.result.best_design}")
            self.scopes.log.append(f"best_metrics: {payload.result.best_metrics}")
        else:
            self.scopes.log.append(f"metrics: {payload.metrics}")
        self._status_label.setText(f"Done in {payload.elapsed_s:.2f}s — Metrics / History / Python tabs updated")
        self.scopes.setCurrentWidget(self.scopes.metrics)
        # also enable validation green
        self._refresh_validation()

    @pyqtSlot(str)
    def _on_job_failed(self, msg: str) -> None:
        self._running = False
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        for item in self.scene._nodes.values():  # type: ignore[attr-defined]
            item.set_status(QColor("#ef4444"))
        self.scopes.log.set_text(msg, title="Log — FAILED")
        self.scopes.setCurrentWidget(self.scopes.log)
        self._status_label.setText("Failed — see Log tab")

    @pyqtSlot(str)
    def _on_job_log(self, line: str) -> None:
        self.scopes.log.append(line)

    # -- Ranking / Calibration quick actions --------------------------------

    def run_ranking(self, preset: str = "datacenter") -> None:
        """One-click ranking for the current material set (called from menu)."""
        self.scopes.log.append(f"Ranking preset={preset} — running harness.rank …")
        self.scopes.setCurrentWidget(self.scopes.log)
        try:
            import harness.rank as rank  # noqa: WPS433
            import harness.materials as mats  # noqa: WPS433

            mats_list = mats.load_anchors()[:6]  # demo subset; full sweep via worker for G2
            df = rank.sweep_materials(mats_list, profiles=[preset])
            self.scopes.set_ranking_dataframe(df, title=f"Ranking — {preset} (anchor subset)")
            self._status_label.setText(f"Ranking done — {len(df)} rows (anchor subset, see harness.rank for full T2 sweep)")
            self.scopes.log.append(f"Ranking table has {len(df)} rows, columns {list(df.columns)}")
        except Exception as exc:  # noqa: BLE001
            self.scopes.log.append(f"Ranking failed: {exc}\n{traceback.format_exc()}")

    def run_ranking_all(self) -> None:
        self.scopes.log.append("Ranking all 4 profiles (anchor subset) …")
        self.scopes.setCurrentWidget(self.scopes.log)
        try:
            import harness.rank as rank  # noqa: WPS433
            import harness.materials as mats  # noqa: WPS433

            mats_list = mats.load_anchors()[:6]
            df = rank.sweep_materials(mats_list, profiles=["cpu", "human", "vehicle", "datacenter"])
            self.scopes.set_ranking_dataframe(df, title="Ranking — all 4 profiles (anchor subset)")
            self._status_label.setText(f"Ranking done — {len(df)} rows across 4 profiles")
        except Exception as exc:  # noqa: BLE001
            self.scopes.log.append(f"Ranking failed: {exc}\n{traceback.format_exc()}")

    def run_calibration(self) -> None:
        self.scopes.log.append("Calibration — running harness.calibration.model_case_rows …")
        self.scopes.setCurrentWidget(self.scopes.log)
        try:
            import harness.calibration as cal  # noqa: WPS433

            cases = list(cal.BENCHMARK_CASES)[:6]  # demo subset
            rows = cal.model_case_rows(cases, design={}, k_by_rig=None, h_by_rig=None, hx_by_rig=None)
            import pandas as pd  # noqa: WPS433

            df = pd.DataFrame(rows)
            self.scopes.set_calibration_dataframe(df, title="Calibration — model vs exp (subset)")
            self._status_label.setText(f"Calibration done — {len(df)} rows")
            self.scopes.log.append(f"Calibration table: columns {list(df.columns)}")
        except Exception as exc:  # noqa: BLE001
            self.scopes.log.append(f"Calibration failed: {exc}\n{traceback.format_exc()}")

    def run_sweep(self) -> None:
        """Bed1D t_switch sweep via harness.control.switch_time_sweep (demo, now via SweepScope)."""
        self.scopes.log.append("Sweep — running Bed1D t_switch sweep …")
        self.scopes.setCurrentWidget(self.scopes.log)
        try:
            import harness.control as ctrl  # noqa: WPS433
            import numpy as np  # noqa: WPS433
            import pandas as pd  # noqa: WPS433

            # use demo problem (n_cells 8 for speed) — grid inside bounds (60–500)
            prob = ctrl.demo_problem(soft_switch=False, dt_phys_s=0.1, n_cycles=2, n_steps=20008)
            grid = np.linspace(80, 480, 8)
            rows = ctrl.switch_time_sweep(prob, grid)
            df = pd.DataFrame(rows)
            # new G3/G4: use SweepScope dataframe + plot
            self.scopes.set_sweep_dataframe(df, title="Sweep — Bed1D SCP/COP vs t_switch")
            # also show as history-like curve (objective = SCP)
            self.scopes.history.set_history([{"objective": r["SCP_W_kg"]} for r in rows], best_objective=max(r["SCP_W_kg"] for r in rows))
            self._status_label.setText(f"Sweep done — {len(df)} points (via direct ctrl demo)")
            self.scopes.log.append(f"Sweep df: {df.head().to_string(index=False)}")
            # store for Compare demo
            try:
                from harness.gui.executor import ExecutorResult  # noqa: WPS433
                fake = ExecutorResult(result=None, trace=None, metrics=dict(df.iloc[df["SCP_W_kg"].idxmax()].to_dict()),
                                      summary_text=df.to_string(index=False), python_code="# sweep demo", elapsed_s=0.0,
                                      sweep_df=df, sweep_axis="t_switch")
                self._prev_result = self._last_result
                self._last_result = fake
            except Exception:
                pass
        except Exception as exc:  # noqa: BLE001
            self.scopes.log.append(f"Sweep failed: {exc}\n{traceback.format_exc()}")

    def run_sweep_via_graph(self) -> None:
        """Insert a Sweep node (if missing) and Run via executor path (t_switch grid via graph)."""
        # add Sweep node if graph has none
        if not self.graph.has_sweep():
            # find physics position to place sweep nearby
            phys = self.graph.physics_node()
            x = (phys.x + 140) if phys else 540
            y = (phys.y) if phys else 140
            spec = self.graph.add_node("Sweep", x=float(x), y=float(y),
                                       data={"axis": "t_switch", "grid": [80, 180, 300, 480], "budget": 8,
                                             "grid_min": 80.0, "grid_max": 480.0, "steps": 6})
            # try to wire Physics problem -> Sweep, and Sweep result -> first Scope
            from harness.gui.scene import NodeItem  # noqa: WPS433
            item = NodeItem(spec)
            self.scene.addItem(item)
            self.scene._nodes[spec.id] = item  # type: ignore[attr-defined]
            # wire if possible
            phys_item = self.scene._nodes.get(phys.id) if phys else None
            if phys_item:
                try:
                    self.scene.add_edge(phys.id, "problem", spec.id, "problem")
                except Exception:
                    pass
            # find first Scope to wire result
            for n in self.graph.nodes:
                if n.type == "Scope":
                    try:
                        self.scene.add_edge(spec.id, "result", n.id, "result")
                        break
                    except Exception:
                        pass
            self._dirty = True
            self._refresh_validation()
            self.scopes.log.append(f"Added Sweep node {spec.id} (t_switch 80–480, 6 pts) — wiring to canvas. Press Run to execute via worker.")
            self.scopes.setCurrentWidget(self.scopes.log)
            return
        self.scopes.log.append("Graph already has Sweep node — running via worker (▶ Run will trigger Sweep path).")
        self.run_graph()

    def run_compare(self) -> None:
        """Compare last two successful results side-by-side (CompareScope)."""
        if self._last_result is None:
            self.scopes.log.append("Compare: no results yet — Run twice to populate history (each Run keeps last two).")
            QMessageBox.information(self, "Compare", "Run two experiments first — Compare will then show the last two results side-by-side.")
            return
        if self._prev_result is None:
            # synthesize a second run by re-using last? Instead show single result diff vs itself with note
            self.scopes.log.append("Compare: only one result in history — compare requires two runs. Showing last vs last (Δ should be 0). Use Run again after changing a parameter.")
            self.scopes.set_compare(self._last_result, self._last_result, title="Compare — single result (run again to diff)")
            return
        try:
            # prefer metrics dicts if OptimizeResult style, else sweep metrics
            a = self._prev_result
            b = self._last_result
            self.scopes.set_compare(a, b, title="Compare — previous vs last run")
            self.scopes.log.append("Compare table shows Δ (B-A) and Δ% per metric — see Compare tab.")
            self._status_label.setText("Compare — shown in Compare tab")
        except Exception as exc:  # noqa: BLE001
            self.scopes.log.append(f"Compare failed: {exc}\n{traceback.format_exc()}")

    def run_v3(self) -> None:
        """V3 oracle-limit check: Bed1D high-k vs Cycle0D (cpu, 600 s, bare bed)."""
        self.scopes.log.append("V3 oracle-limit — Bed1D (high k, thin, strong wall) vs Cycle0D (cpu, 600 s, hx=1) …")
        self.scopes.setCurrentWidget(self.scopes.log)
        try:
            from harness.envs.cycle0d import Cycle0D  # noqa: WPS433
            from harness.physics.bed1d import simulate_bed  # noqa: WPS433

            oracle = Cycle0D("anchor:Silica gel RD", "cpu").evaluate({"cycle_time_s": 600.0, "hx_mass_factor": 1.0})
            bed_out = simulate_bed(
                q_sat_kg_kg=0.35, q_st_j_kg=2.5e6, e_char_j_mol=4500.0, n_da=1.8,
                rho_s_kg_m3=600.0, c_s_j_kg_k=1000.0, c_pl_j_kg_k=4184.0,
                k_eff_w_m_k=0.3, h_wall_w_m2_k=2000.0, L_m=0.002, n_cells=16,
                hx_mass_factor=1.0, t_evap_c=18.0, t_cond_c=35.0,
                t_f_ads_c=35.0, t_f_des_c=75.0,
                k_ldf_s_1=5.0, t_ads_s=600.0, t_des_s=600.0, dt_s=0.015, n_cycles=4,
            )
            bed = {k: float(v) for k, v in bed_out["summary"].items()}
            cop_gap = abs(bed["COP"] - oracle["COP"]) / oracle["COP"] * 100 if oracle["COP"] else 0
            scp_gap = abs(bed["SCP_W_kg"] - oracle["SCP_W_kg"]) / oracle["SCP_W_kg"] * 100 if oracle["SCP_W_kg"] else 0
            ok = cop_gap < 2 and scp_gap < 2
            msg = (f"Oracle (cpu, 600 s, hx=1) COP {oracle['COP']:.4f} SCP {oracle['SCP_W_kg']:.1f}\n"
                   f"Bed1D  (k=5, h=2000, L=2 mm, n=16) COP {bed['COP']:.4f} SCP {bed['SCP_W_kg']:.1f}\n"
                   f"COP gap {cop_gap:.2f}%  SCP gap {scp_gap:.2f}%  →  {'PASS (<2%)' if ok else 'FAIL (≥2%)'}  (V3 gate <2%)")
            self.scopes.sweep.set_text(msg, title="V3 oracle-limit")
            self.scopes.setCurrentWidget(self.scopes.sweep)
            self._status_label.setText(f"V3 check — COP {cop_gap:.2f}% SCP {scp_gap:.2f}% {'PASS' if ok else 'FAIL'}")
            self.scopes.log.append(msg)
        except Exception as exc:  # noqa: BLE001
            self.scopes.log.append(f"V3 failed: {exc}\n{traceback.format_exc()}")
        except Exception as exc:  # noqa: BLE001
            self.scopes.log.append(f"Ranking failed: {exc}\n{traceback.format_exc()}")

    # -- close --------------------------------------------------------------

    def closeEvent(self, event: Any) -> None:  # type: ignore[override]
        if not self._confirm_discard():
            event.ignore()
            return
        # stop worker thread
        try:
            self._thread.quit()
            self._thread.wait(1500)
        except Exception:
            pass
        super().closeEvent(event)


def run() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("harness-gui")
    app.setOrganizationName("CoolingWithHeat")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()

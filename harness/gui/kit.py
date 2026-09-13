"""Shared GUI kit: dark theme, generic scope widgets, background job runner.

App-agnostic on purpose — the adsorbent-ml data explorer imports this module
too (``from harness.gui.kit import ...``), matching the one-directional
adsorbent-ml → harness dependency rule.

Threading contract (the fix over the parked workbench): the job callable is
dispatched into the worker thread via the thread's ``started`` signal
(queued connection) — never via ``QTimer.singleShot``, which runs on the
caller's thread. Heavy imports (jax, harness, pandas) belong inside the job
function, never at module import time.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

DARK_QSS = """
QMainWindow { background: #0f1115; }
QDockWidget { background: #0f1115; color: #e2e8f0; }
QDockWidget::title { background: #1b1e24; padding: 4px 8px; font-size: 8.5pt; color: #94a3b8; }
QToolBar { background: #0f1115; spacing: 6px; padding: 4px; border: none; }
QToolBar QLabel { color: #94a3b8; font-size: 8pt; }
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
QGroupBox { color: #94a3b8; border: 1px solid #1e222a; border-radius: 6px; margin-top: 8px; padding-top: 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; }
"""

FIG_BG = "#0f1115"
FIG_FG = "#e2e8f0"
FIG_AXIS = "#94a3b8"
FIG_SPINE = "#1e222a"
FIG_PANEL = "#1b1e24"
FIG_ACCENT = "#38bdf8"


def dark_figure(figsize: tuple[float, float] = (5.0, 3.0)) -> tuple[Figure, FigureCanvas]:
    """Dark-styled matplotlib Figure + canvas embedded in Qt."""
    fig = Figure(figsize=figsize, facecolor=FIG_BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(FIG_BG)
    return fig, FigureCanvas(fig)


def style_axis(ax) -> None:
    ax.set_facecolor(FIG_BG)
    ax.tick_params(colors=FIG_AXIS, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(FIG_SPINE)


def placeholder_text(ax, text: str) -> None:
    ax.clear()
    style_axis(ax)
    ax.text(0.5, 0.5, text, ha="center", va="center", color="#475569",
            fontsize=8, transform=ax.transAxes)
    ax.set_axis_off()


def _copy_table(table: QTableWidget) -> None:
    from PyQt6.QtWidgets import QApplication  # noqa: PLC0415

    headers = [table.horizontalHeaderItem(c).text() if table.horizontalHeaderItem(c)
               else f"col{c}" for c in range(table.columnCount())]
    rows = ["\t".join(headers)]
    for r in range(table.rowCount()):
        vals = [table.item(r, c).text() if table.item(r, c) else ""
                for c in range(table.columnCount())]
        rows.append("\t".join(vals))
    QApplication.clipboard().setText("\n".join(rows))


class MetricsScope(QWidget):
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.label = QLabel("No result yet — Run.")
        self.label.setStyleSheet("color:#94a3b8; font-size:8.5pt;")
        layout.addWidget(self.label)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["metric", "value"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table, 1)
        btn = QPushButton("Copy table")
        btn.clicked.connect(lambda: _copy_table(self.table))
        layout.addWidget(btn)

    def set_metrics(self, metrics: dict[str, float], extra: str = "") -> None:
        self.table.setRowCount(0)
        for k in sorted(metrics):
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(str(k)))
            v = metrics[k]
            txt = f"{v:.6g}" if isinstance(v, float) else str(v)
            self.table.setItem(r, 1, QTableWidgetItem(txt))
        self.label.setText(extra or f"{len(metrics)} metrics")


class HistoryScope(QWidget):
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.fig, self.canvas = dark_figure((5, 3))
        self.ax = self.fig.axes[0]
        layout.addWidget(self.canvas, 1)
        self.info = QLabel("No history yet.")
        self.info.setStyleSheet("color:#64748b; font-size:7.5pt;")
        layout.addWidget(self.info)
        self._placeholder()

    def _placeholder(self) -> None:
        placeholder_text(self.ax, "History will appear after an optimization run")
        self.fig.tight_layout()
        self.canvas.draw()

    def set_history(self, history: list[dict[str, Any]],
                    best_objective: float | None = None) -> None:
        if not history:
            self._placeholder()
            self.info.setText("Empty history")
            return
        objs = [float(h.get("objective", 0)) for h in history]
        xs = list(range(len(objs)))
        best, cur = [], float("-inf")
        for v in objs:
            cur = max(cur, v)
            best.append(cur)
        self.ax.clear()
        style_axis(self.ax)
        self.ax.plot(xs, objs, ".", color="#475569", ms=3, alpha=0.5, label="evals")
        self.ax.plot(xs, best, "-", color=FIG_ACCENT, lw=1.6, label="best-so-far")
        self.ax.set_xlabel("eval", color=FIG_AXIS, fontsize=7)
        self.ax.set_ylabel("objective", color=FIG_AXIS, fontsize=7)
        self.ax.legend(facecolor=FIG_PANEL, edgecolor="#2a2f3a", fontsize=7,
                       labelcolor=FIG_FG)
        self.ax.set_title("Optimization history", color=FIG_FG, fontsize=9)
        self.ax.set_axis_on()
        self.fig.tight_layout()
        self.canvas.draw()
        self.info.setText(
            f"{len(history)} evals  ·  best {best[-1]:.6g}"
            + (f"  (reported {best_objective:.6g})" if best_objective is not None else ""))

    def save_png(self, path: str) -> None:
        self.fig.savefig(path, dpi=180, facecolor=self.fig.get_facecolor())


class TextScope(QWidget):
    """Log / JSON / Python viewer with copy & save."""

    def __init__(self, title: str = "Log", parent: Any = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.label = QLabel(title)
        self.label.setStyleSheet("color:#e2e8f0; font-weight:600; font-size:9pt;")
        layout.addWidget(self.label)
        self.edit = QTextEdit()
        self.edit.setReadOnly(True)
        self.edit.setStyleSheet(
            "background:#111318; color:#cbd5e1; font-family: monospace; "
            "font-size:8pt; border:1px solid #1e222a; border-radius:6px;")
        layout.addWidget(self.edit, 1)
        row = QHBoxLayout()
        self.copy_btn = QPushButton("Copy")
        self.copy_btn.clicked.connect(self._copy)
        self.save_btn = QPushButton("Save…")
        self.save_btn.clicked.connect(self._save)
        row.addStretch()
        row.addWidget(self.copy_btn)
        row.addWidget(self.save_btn)
        layout.addLayout(row)
        self._save_filter = "Text (*.txt);;All (*)"
        self._save_default = "output.txt"

    def set_text(self, text: str, title: str | None = None) -> None:
        if title:
            self.label.setText(title)
        self.edit.setPlainText(text)

    def append(self, line: str) -> None:
        self.edit.append(line)

    def _copy(self) -> None:
        from PyQt6.QtWidgets import QApplication  # noqa: PLC0415

        QApplication.clipboard().setText(self.edit.toPlainText())

    def _save(self) -> None:
        from PyQt6.QtWidgets import QFileDialog  # noqa: PLC0415

        path, _ = QFileDialog.getSaveFileName(self, "Save", self._save_default,
                                              self._save_filter)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.edit.toPlainText())


class DataFrameScope(QWidget):
    """Sortable pandas table + raw preview + CSV export (generic)."""

    def __init__(self, title: str = "Table", parent: Any = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.label = QLabel(title)
        self.label.setStyleSheet("color:#e2e8f0; font-weight:600; font-size:9pt;")
        layout.addWidget(self.label)
        self.table = QTableWidget(0, 0)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setStretchLastSection(False)
        layout.addWidget(self.table, 2)
        self.edit = QTextEdit()
        self.edit.setReadOnly(True)
        self.edit.setStyleSheet(
            "background:#111318; color:#94a3b8; font-family: monospace; "
            "font-size:7.5pt; border:1px solid #1e222a; border-radius:6px;")
        self.edit.setMaximumHeight(110)
        layout.addWidget(self.edit, 1)
        row = QHBoxLayout()
        copy_btn = QPushButton("Copy table")
        copy_btn.clicked.connect(lambda: _copy_table(self.table))
        save_btn = QPushButton("Save CSV…")
        save_btn.clicked.connect(self._save_csv)
        row.addStretch()
        row.addWidget(copy_btn)
        row.addWidget(save_btn)
        layout.addLayout(row)
        self._df = None
        self._save_default = "table.csv"

    def set_dataframe(self, df: Any, title: str | None = None) -> None:
        if title:
            self.label.setText(title)
        self._df = df
        if df is None or (hasattr(df, "empty") and df.empty):
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            self.edit.setPlainText("(empty)")
            return
        try:
            import pandas as pd  # noqa: PLC0415

            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
        except Exception:
            pass
        cols = list(df.columns)
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels([str(c) for c in cols])
        self.table.setRowCount(len(df))
        for r, (_, row) in enumerate(df.iterrows()):
            for c, col in enumerate(cols):
                v = row[col]
                if isinstance(v, float):
                    txt = f"{v:.6g}"
                elif isinstance(v, bool):
                    txt = "yes" if v else "no"
                else:
                    txt = str(v)
                it = QTableWidgetItem(txt)
                if isinstance(v, (int, float)):
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                        | Qt.AlignmentFlag.AlignVCenter)
                if col in ("transport_provenance", "provenance") and "default" in str(v):
                    it.setForeground(Qt.GlobalColor.yellow)
                if col == "out_of_window" and v:
                    it.setForeground(Qt.GlobalColor.red)
                self.table.setItem(r, c, it)
        self.table.resizeColumnsToContents()
        try:
            self.edit.setPlainText(df.to_string(index=False, max_rows=20))
        except Exception:
            self.edit.setPlainText(str(df)[:4000])

    def _save_csv(self) -> None:
        from PyQt6.QtWidgets import QFileDialog  # noqa: PLC0415

        path, _ = QFileDialog.getSaveFileName(self, "Save CSV",
                                              self._save_default, "CSV (*.csv);;All (*)")
        if not path or self._df is None:
            return
        self._df.to_csv(path, index=False)

    def current_row(self) -> int:
        return self.table.currentRow()


class CompareScope(QWidget):
    """Side-by-side metrics diff (A vs B, Δ and Δ%)."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.label = QLabel("Compare — side-by-side diff")
        self.label.setStyleSheet("color:#e2e8f0; font-weight:600; font-size:9pt;")
        layout.addWidget(self.label)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["metric", "A", "B", "Δ (B-A)", "Δ%"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table, 2)
        self.edit = QTextEdit()
        self.edit.setReadOnly(True)
        self.edit.setStyleSheet(
            "background:#111318; color:#94a3b8; font-family: monospace; "
            "font-size:7.5pt; border:1px solid #1e222a; border-radius:6px;")
        self.edit.setMaximumHeight(120)
        layout.addWidget(self.edit, 1)
        row = QHBoxLayout()
        copy_btn = QPushButton("Copy table")
        copy_btn.clicked.connect(lambda: _copy_table(self.table))
        row.addStretch()
        row.addWidget(copy_btn)
        layout.addLayout(row)

    @staticmethod
    def _extract(m: Any) -> dict[str, float]:
        if m is None:
            return {}
        if isinstance(m, dict):
            return dict(m)
        for attr in ("best_metrics", "metrics"):
            v = getattr(m, attr, None)
            if v is not None:
                return dict(v)
        inner = getattr(m, "result", None)
        if inner is not None and getattr(inner, "best_metrics", None) is not None:
            return dict(inner.best_metrics)
        return {}

    def set_compare(self, a: Any, b: Any, title: str | None = None,
                    a_label: str = "A", b_label: str = "B") -> None:
        da, db = self._extract(a), self._extract(b)
        self.label.setText(title or f"Compare — {a_label} vs {b_label}")
        keys = sorted(set(da) | set(db))
        if not keys:
            self.edit.setPlainText("(no metrics to compare)")
            self.table.setRowCount(0)
            return
        self.table.setRowCount(0)
        for k in keys:
            va, vb = da.get(k, np.nan), db.get(k, np.nan)
            try:
                va_f = float(va) if va is not None else np.nan
                vb_f = float(vb) if vb is not None else np.nan
                delta = vb_f - va_f
                pct = (delta / abs(va_f) * 100.0
                       if np.isfinite(va_f) and va_f != 0 else np.nan)
            except (TypeError, ValueError):
                va_f = vb_f = delta = pct = np.nan
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(str(k)))
            self.table.setItem(r, 1, QTableWidgetItem(
                f"{va_f:.6g}" if np.isfinite(va_f) else str(va)))
            self.table.setItem(r, 2, QTableWidgetItem(
                f"{vb_f:.6g}" if np.isfinite(vb_f) else str(vb)))
            self.table.setItem(r, 3, QTableWidgetItem(
                f"{delta:.6g}" if np.isfinite(delta) else "—"))
            self.table.setItem(r, 4, QTableWidgetItem(
                f"{pct:.2f}%" if np.isfinite(pct) else "—"))
            for col in (1, 2, 3, 4):
                it = self.table.item(r, col)
                if it:
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                        | Qt.AlignmentFlag.AlignVCenter)
            if np.isfinite(delta):
                it = self.table.item(r, 3)
                if it and delta > 0:
                    it.setForeground(Qt.GlobalColor.green)
                elif it and delta < 0:
                    it.setForeground(Qt.GlobalColor.red)
        self.table.resizeColumnsToContents()


class Worker(QObject):
    """Runs one job callable inside its owner thread; heavy imports belong in the job."""

    started = pyqtSignal(str)
    progressed = pyqtSignal(dict)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, parent: Any | None = None) -> None:
        super().__init__(parent)
        self._fn: Callable[[Any], Any] | None = None
        self._label = ""
        self.cancel_requested = False

    def set_job(self, label: str, fn: Callable[[Any], Any]) -> None:
        self._label = label
        self._fn = fn

    def request_cancel(self) -> None:
        self.cancel_requested = True

    @pyqtSlot()
    def run(self) -> None:
        self.cancel_requested = False
        if self._fn is None:
            return
        self.started.emit(self._label)
        try:
            payload = self._fn(self)
        except Exception as exc:  # noqa: BLE001
            import traceback  # noqa: PLC0415

            self.failed.emit(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
            return
        self.finished.emit(payload)


class JobRunner(QObject):
    """Owns a worker thread; jobs are dispatched via the queued ``started``
    signal so the callable genuinely executes off the UI thread. Reuse one
    runner per window: ``start`` a new job any time (the previous one must
    have finished — a running JAX call is not interruptible)."""

    started = pyqtSignal(str)
    progressed = pyqtSignal(dict)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, parent: Any | None = None) -> None:
        super().__init__(parent)
        self._thread = QThread(self)
        self.worker = Worker()
        self.worker.moveToThread(self._thread)
        # queued connection: worker.run executes in the worker thread
        self._thread.started.connect(self.worker.run)
        self.worker.started.connect(self.started)
        self.worker.progressed.connect(self.progressed)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.log.connect(self.log)

    def busy(self) -> bool:
        return self._thread.isRunning()

    def start(self, label: str, fn: Callable[[Any], Any]) -> None:
        if self.busy():
            self.log.emit("a job is already running")
            return
        self.worker.set_job(label, fn)
        self._thread.start()

    def request_cancel(self) -> None:
        self.worker.request_cancel()

    def _on_finished(self, payload: object) -> None:
        self._thread.quit()
        self.finished.emit(payload)

    def _on_failed(self, message: str) -> None:
        self._thread.quit()
        self.failed.emit(message)

    def shutdown(self) -> None:
        self.worker.request_cancel()
        self._thread.quit()
        self._thread.wait(2000)


__all__ = [
    "CompareScope",
    "DataFrameScope",
    "DARK_QSS",
    "HistoryScope",
    "JobRunner",
    "MetricsScope",
    "TextScope",
    "Worker",
    "dark_figure",
    "placeholder_text",
    "style_axis",
]

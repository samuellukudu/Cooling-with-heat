"""Scope widgets — Metrics, History, Trace, Ranking, Calibration.

Each scope is a QWidget that can be embedded in the MainWindow's
ScopeTabs. They are fed by `ExecutorResult` and by direct rank/calibration
calls (no worker needed for small data).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# -- helpers ---------------------------------------------------------------


def _copy_table(table: QTableWidget) -> None:
    from PyQt6.QtWidgets import QApplication  # noqa: WPS433

    rows = []
    headers = [table.horizontalHeaderItem(c).text() if table.horizontalHeaderItem(c) else f"col{c}" for c in range(table.columnCount())]
    rows.append("\t".join(headers))
    for r in range(table.rowCount()):
        vals = []
        for c in range(table.columnCount()):
            it = table.item(r, c)
            vals.append(it.text() if it else "")
        rows.append("\t".join(vals))
    QApplication.clipboard().setText("\n".join(rows))


class MetricsScope(QWidget):
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.label = QLabel("No result yet — Run the graph.")
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
        self.fig = Figure(figsize=(5, 3), facecolor="#0f1115")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#0f1115")
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas, 1)
        self.info = QLabel("No history yet.")
        self.info.setStyleSheet("color:#64748b; font-size:7.5pt;")
        layout.addWidget(self.info)
        self._placeholder()

    def _placeholder(self) -> None:
        self.ax.clear()
        self.ax.set_facecolor("#0f1115")
        self.ax.text(0.5, 0.5, "History will appear after an optimization run",
                     ha="center", va="center", color="#475569", fontsize=8, transform=self.ax.transAxes)
        self.ax.set_axis_off()
        self.fig.tight_layout()
        self.canvas.draw()

    def set_history(self, history: list[dict[str, Any]], best_objective: float | None = None) -> None:
        if not history:
            self._placeholder()
            self.info.setText("Empty history")
            return
        objs = [float(h.get("objective", 0)) for h in history]
        xs = list(range(len(objs)))
        # best-so-far envelope
        best = []
        cur = float("-inf")
        for v in objs:
            cur = max(cur, v)
            best.append(cur)
        self.ax.clear()
        self.ax.set_facecolor("#0f1115")
        self.ax.plot(xs, objs, ".", color="#475569", ms=3, alpha=0.5, label="evals")
        self.ax.plot(xs, best, "-", color="#38bdf8", lw=1.6, label="best-so-far")
        self.ax.set_xlabel("eval", color="#94a3b8", fontsize=7)
        self.ax.set_ylabel("objective", color="#94a3b8", fontsize=7)
        self.ax.tick_params(colors="#94a3b8", labelsize=7)
        for spine in self.ax.spines.values():
            spine.set_color("#1e222a")
        self.ax.legend(facecolor="#1b1e24", edgecolor="#2a2f3a", fontsize=7, labelcolor="#e2e8f0")
        self.ax.set_title("Optimization history", color="#e2e8f0", fontsize=9)
        self.fig.tight_layout()
        self.canvas.draw()
        self.info.setText(f"{len(history)} evals  ·  best {best[-1]:.6g}" + (f"  (reported {best_objective:.6g})" if best_objective is not None else ""))

    def save_png(self, path: str) -> None:
        self.fig.savefig(path, dpi=180, facecolor=self.fig.get_facecolor())


class TraceScope(QWidget):
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.fig = Figure(figsize=(5, 3.2), facecolor="#0f1115")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#0f1115")
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas, 1)
        self.info = QLabel("No trace yet — enable Collect trace and Run.")
        self.info.setStyleSheet("color:#64748b; font-size:7.5pt;")
        layout.addWidget(self.info)
        self._placeholder()

    def _placeholder(self) -> None:
        self.ax.clear()
        self.ax.set_facecolor("#0f1115")
        self.ax.text(0.5, 0.5, "T(x,t) heatmap appears after a dynamic run\n(Bed1D / TwoBed with Collect trace)",
                     ha="center", va="center", color="#475569", fontsize=8, transform=self.ax.transAxes)
        self.ax.set_axis_off()
        self.fig.tight_layout()
        self.canvas.draw()

    def set_trace(self, trace: Any) -> None:
        if trace is None or not getattr(trace, "series", None):
            self._placeholder()
            self.info.setText("No trace — static problem or collection disabled.")
            return
        series = trace.series
        # Try to find a 2-D time×space field (future: full T(x,t) — not yet in v1)
        img_key = None
        for k, v in series.items():
            arr = np.asarray(v)
            if arr.ndim == 2 and arr.shape[0] > 4 and arr.shape[1] > 1:
                img_key = k
                break
        if img_key is not None:
            arr = np.asarray(series[img_key])
            self.ax.clear()
            self.ax.set_facecolor("#0f1115")
            im = self.ax.imshow(arr, aspect="auto", origin="lower", cmap="inferno", interpolation="nearest")
            self.ax.set_xlabel("space cell", color="#94a3b8", fontsize=7)
            self.ax.set_ylabel("time step", color="#94a3b8", fontsize=7)
            self.ax.tick_params(colors="#94a3b8", labelsize=7)
            for spine in self.ax.spines.values():
                spine.set_color("#1e222a")
            self.ax.set_title(f"Trace  {img_key}  [{arr.shape[0]} × {arr.shape[1]}]", color="#e2e8f0", fontsize=9)
            self.fig.colorbar(im, ax=self.ax, shrink=0.85).ax.tick_params(colors="#94a3b8", labelsize=6)
            self.fig.tight_layout()
            self.canvas.draw()
            self.info.setText(f"series {img_key}: {arr.shape}  ·  summary {trace.summary}")
            return
        # 1-D fallback: Bed1D / TwoBed time series with phase shading
        self.ax.clear()
        self.ax.set_facecolor("#0f1115")
        # detect TwoBed vs Bed1D
        is_twobed = any(k.startswith("A_") or k.startswith("B_") for k in series)
        # downsample for speed if very long
        def _arr(key: str) -> np.ndarray | None:
            if key not in series:
                return None
            a = np.asarray(series[key]).ravel()
            if a.size > 2000:
                step = max(1, a.size // 600)
                return a[::step]
            return a
        # phase shading helper
        def _shade(phase_arr: np.ndarray | None) -> None:
            if phase_arr is None or phase_arr.size < 4:
                return
            # phase 0=ads, 1=des — shade des phases
            x = np.arange(phase_arr.size)
            # simple: where phase >0.5
            des = phase_arr > 0.5
            # find transitions for vspan
            in_des = False
            start = 0
            for i, v in enumerate(des):
                if v and not in_des:
                    start = i
                    in_des = True
                elif not v and in_des:
                    self.ax.axvspan(start, i, color="#1e293b", alpha=0.35, zorder=0)
                    in_des = False
            if in_des:
                self.ax.axvspan(start, len(des), color="#1e293b", alpha=0.35, zorder=0)

        if is_twobed:
            # TwoBed: show A/B wall temps + q
            a_twall = _arr("A_t_wall_k")
            b_twall = _arr("B_t_wall_k")
            a_q = _arr("A_q_mean_kg_kg")
            b_q = _arr("B_q_mean_kg_kg")
            phase = _arr("sys_phase") if "sys_phase" in series else _arr("A_phase")
            _shade(phase)
            if a_twall is not None:
                self.ax.plot(a_twall, label="A T_wall", color="#f59e0b", lw=1.2)
            if b_twall is not None:
                self.ax.plot(b_twall, label="B T_wall", color="#06b6d4", lw=1.2)
            # second axis for uptake
            ax2 = self.ax.twinx()
            ax2.set_facecolor("#0f1115")
            if a_q is not None:
                ax2.plot(a_q, label="A q", color="#f59e0b", ls="--", lw=1.0, alpha=0.7)
            if b_q is not None:
                ax2.plot(b_q, label="B q", color="#06b6d4", ls="--", lw=1.0, alpha=0.7)
            ax2.tick_params(colors="#94a3b8", labelsize=6)
            for spine in ax2.spines.values():
                spine.set_color("#1e222a")
            # combine legends
            h1, l1 = self.ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            self.ax.legend(h1 + h2, l1 + l2, facecolor="#1b1e24", edgecolor="#2a2f3a", fontsize=6, labelcolor="#e2e8f0", loc="upper right")
            self.ax.set_title("TwoBed trace — wall temps (solid) & uptake (dashed) — shaded = des phase", color="#e2e8f0", fontsize=8)
        else:
            # Bed1D
            tw = _arr("t_wall_k")
            tb = _arr("t_bed_mean_k")
            q = _arr("q_mean_kg_kg")
            phase = _arr("phase")
            _shade(phase)
            if tw is not None:
                self.ax.plot(tw, label="T_wall", color="#f59e0b", lw=1.3)
            if tb is not None:
                self.ax.plot(tb, label="T_bed mean", color="#eab308", lw=1.1)
            ax2 = self.ax.twinx()
            ax2.set_facecolor("#0f1115")
            if q is not None:
                ax2.plot(q, label="q mean", color="#22c55e", lw=1.1)
                # q* for reference
                qstar = _arr("q_star_mean_kg_kg")
                if qstar is not None:
                    ax2.plot(qstar, label="q* mean", color="#22c55e", ls="--", lw=0.9, alpha=0.6)
            ax2.tick_params(colors="#94a3b8", labelsize=6)
            for spine in ax2.spines.values():
                spine.set_color("#1e222a")
            h1, l1 = self.ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            self.ax.legend(h1 + h2, l1 + l2, facecolor="#1b1e24", edgecolor="#2a2f3a", fontsize=6, labelcolor="#e2e8f0", loc="upper right")
            self.ax.set_title("Bed1D trace — temps (solid) & uptake (dashed) — shaded = des", color="#e2e8f0", fontsize=8)
        self.ax.set_xlabel("trace step (downsampled)", color="#94a3b8", fontsize=7)
        self.ax.tick_params(colors="#94a3b8", labelsize=7)
        for spine in self.ax.spines.values():
            spine.set_color("#1e222a")
        self.fig.tight_layout()
        self.canvas.draw()
        self.info.setText(f"{len(series)} series  ·  summary {trace.summary}")

    def save_png(self, path: str) -> None:
        self.fig.savefig(path, dpi=180, facecolor=self.fig.get_facecolor())


class TextScope(QWidget):
    """Simple log / JSON / Python viewer."""

    def __init__(self, title: str = "Log", parent: Any = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.label = QLabel(title)
        self.label.setStyleSheet("color:#e2e8f0; font-weight:600; font-size:9pt;")
        layout.addWidget(self.label)
        self.edit = QTextEdit()
        self.edit.setReadOnly(True)
        self.edit.setStyleSheet("background:#111318; color:#cbd5e1; font-family: monospace; font-size:8pt; border:1px solid #1e222a; border-radius:6px;")
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
        from PyQt6.QtWidgets import QApplication  # noqa: WPS433

        QApplication.clipboard().setText(self.edit.toPlainText())

    def _save(self) -> None:
        from PyQt6.QtWidgets import QFileDialog  # noqa: WPS433

        path, _ = QFileDialog.getSaveFileName(self, "Save", self._save_default, self._save_filter)
        if path:
            open(path, "w", encoding="utf-8").write(self.edit.toPlainText())


class DataFrameScope(QWidget):
    """Table + raw text for DataFrame results (ranking / calibration)."""

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
        self.edit.setStyleSheet("background:#111318; color:#94a3b8; font-family: monospace; font-size:7.5pt; border:1px solid #1e222a; border-radius:6px;")
        self.edit.setMaximumHeight(110)
        layout.addWidget(self.edit, 1)
        row = QHBoxLayout()
        self.copy_btn = QPushButton("Copy table")
        self.copy_btn.clicked.connect(lambda: _copy_table(self.table))
        self.save_btn = QPushButton("Save CSV…")
        self.save_btn.clicked.connect(self._save_csv)
        self.save_raw_btn = QPushButton("Save raw…")
        self.save_raw_btn.clicked.connect(self._save_raw)
        row.addStretch()
        row.addWidget(self.copy_btn)
        row.addWidget(self.save_btn)
        row.addWidget(self.save_raw_btn)
        layout.addLayout(row)
        self._df = None  # type: ignore
        self._save_filter = "CSV (*.csv);;All (*)"
        self._save_default = "table.csv"

    def set_dataframe(self, df: Any, title: str | None = None) -> None:
        """Populate from pandas DataFrame (or list-of-dicts fallback)."""
        if title:
            self.label.setText(title)
        self._df = df
        # handle empty
        if df is None or (hasattr(df, "empty") and df.empty):
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            self.edit.setPlainText("(empty)")
            return
        # normalize to DataFrame if needed
        try:
            import pandas as pd  # noqa: WPS433

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
                # format floats
                if isinstance(v, float):
                    txt = f"{v:.6g}"
                elif isinstance(v, bool):
                    txt = "yes" if v else "no"
                else:
                    txt = str(v)
                it = QTableWidgetItem(txt)
                # numeric alignment
                if isinstance(v, (int, float)):
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                # provenance / out-of-window highlighting
                if col in ("transport_provenance", "provenance") and "default" in str(v):
                    it.setForeground(Qt.GlobalColor.yellow)
                if col == "out_of_window" and v:
                    it.setForeground(Qt.GlobalColor.red)
                self.table.setItem(r, c, it)
        self.table.resizeColumnsToContents()
        # raw text preview
        try:
            self.edit.setPlainText(df.to_string(index=False, max_rows=20))
        except Exception:
            self.edit.setPlainText(str(df)[:4000])

    def set_text(self, text: str, title: str | None = None) -> None:
        if title:
            self.label.setText(title)
        self.edit.setPlainText(text)
        # also try to parse as csv-like?
        self.table.setRowCount(0)
        self.table.setColumnCount(0)

    def append(self, line: str) -> None:
        self.edit.append(line)

    def _save_csv(self) -> None:
        from PyQt6.QtWidgets import QFileDialog  # noqa: WPS433

        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", self._save_default, self._save_filter)
        if not path or self._df is None:
            return
        try:
            self._df.to_csv(path, index=False)  # type: ignore
        except Exception:
            open(path, "w", encoding="utf-8").write(self.edit.toPlainText())

    def _save_raw(self) -> None:
        from PyQt6.QtWidgets import QFileDialog  # noqa: WPS433

        path, _ = QFileDialog.getSaveFileName(self, "Save raw", "raw.txt", "Text (*.txt);;All (*)")
        if path:
            open(path, "w", encoding="utf-8").write(self.edit.toPlainText())


class CompareScope(QWidget):
    """Side-by-side OptimizeResult diff table (like harness.report.summary side-by-side)."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.label = QLabel("Compare — side-by-side OptimizeResult diff")
        self.label.setStyleSheet("color:#e2e8f0; font-weight:600; font-size:9pt;")
        layout.addWidget(self.label)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["metric", "A", "B", "Δ (B-A)", "Δ%"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSortingEnabled(True)
        layout.addWidget(self.table, 2)
        self.edit = QTextEdit()
        self.edit.setReadOnly(True)
        self.edit.setStyleSheet("background:#111318; color:#94a3b8; font-family: monospace; font-size:7.5pt; border:1px solid #1e222a; border-radius:6px;")
        self.edit.setMaximumHeight(120)
        layout.addWidget(self.edit, 1)
        row = QHBoxLayout()
        self.copy_btn = QPushButton("Copy table")
        self.copy_btn.clicked.connect(lambda: _copy_table(self.table))
        row.addStretch()
        row.addWidget(self.copy_btn)
        layout.addLayout(row)
        self._text_summary = ""

    def set_compare(self, a_metrics: dict[str, Any] | Any, b_metrics: dict[str, Any] | Any, title: str | None = None,
                    a_label: str = "A", b_label: str = "B") -> None:
        # Normalize: accept OptimizeResult, ExecutorResult, or dict
        def _extract(m):
            if m is None:
                return {}
            if isinstance(m, dict):
                return dict(m)
            # OptimizeResult
            if hasattr(m, "best_metrics") and getattr(m, "best_metrics") is not None:
                return dict(getattr(m, "best_metrics"))
            if hasattr(m, "metrics") and getattr(m, "metrics") is not None:
                return dict(getattr(m, "metrics"))
            if hasattr(m, "result") and getattr(m, "result") is not None:
                inner = getattr(m, "result")
                if hasattr(inner, "best_metrics"):
                    return dict(getattr(inner, "best_metrics"))
            return {}
        da = _extract(a_metrics)
        db = _extract(b_metrics)
        # also allow full result objects for richer info
        if title:
            self.label.setText(title)
        else:
            self.label.setText(f"Compare — {a_label} vs {b_label}")
        # build table over union of keys
        keys = sorted(set(da.keys()) | set(db.keys()))
        if not keys:
            self.edit.setPlainText("(no metrics to compare)")
            self.table.setRowCount(0)
            return
        self.table.setRowCount(0)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["metric", a_label, b_label, "Δ (B-A)", "Δ%"])
        # also keep human-readable summary similar to harness.report.summary side-by-side
        lines = [f"{'metric':<20s} {a_label:>14s} {b_label:>14s} {'Δ':>14s} {'Δ%':>10s}", "-"*74]
        for k in keys:
            va = da.get(k, float("nan"))
            vb = db.get(k, float("nan"))
            try:
                va_f = float(va) if va is not None else float("nan")
                vb_f = float(vb) if vb is not None else float("nan")
                delta = vb_f - va_f
                pct = (delta / abs(va_f) * 100.0) if va_f and not np.isnan(va_f) and va_f != 0 else float("nan")
            except Exception:
                va_f = vb_f = delta = pct = float("nan")
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(str(k)))
            self.table.setItem(r, 1, QTableWidgetItem(f"{va_f:.6g}" if not np.isnan(va_f) else str(va)))
            self.table.setItem(r, 2, QTableWidgetItem(f"{vb_f:.6g}" if not np.isnan(vb_f) else str(vb)))
            self.table.setItem(r, 3, QTableWidgetItem(f"{delta:.6g}" if not np.isnan(delta) else "—"))
            self.table.setItem(r, 4, QTableWidgetItem(f"{pct:.2f}%" if not np.isnan(pct) else "—"))
            for col in (1, 2, 3, 4):
                it = self.table.item(r, col)
                if it:
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            # color delta
            if not np.isnan(delta):
                it = self.table.item(r, 3)
                if it and delta > 0:
                    it.setForeground(Qt.GlobalColor.green)
                elif it and delta < 0:
                    it.setForeground(Qt.GlobalColor.red)
            lines.append(f"{k:<20s} {va_f:>14.6g} {vb_f:>14.6g} {delta:>14.6g} {pct:>9.2f}%" if not np.isnan(pct) else f"{k:<20s} {va_f:>14.6g} {vb_f:>14.6g} {delta:>14.6g} {'—':>10s}")
        self.table.resizeColumnsToContents()
        self._text_summary = "\n".join(lines)
        self.edit.setPlainText(self._text_summary)

    def set_compare_results(self, result_a: Any, result_b: Any, title: str | None = None) -> None:
        self.set_compare(result_a, result_b, title=title)

    def set_text(self, text: str, title: str | None = None) -> None:
        if title:
            self.label.setText(title)
        self.edit.setPlainText(text)
        self.table.setRowCount(0)

    def save_csv(self, path: str) -> None:
        # reuse table -> csv
        from pathlib import Path as _P  # noqa: WPS433
        rows = []
        headers = [self.table.horizontalHeaderItem(c).text() if self.table.horizontalHeaderItem(c) else f"col{c}" for c in range(self.table.columnCount())]
        rows.append("\t".join(headers))
        for r in range(self.table.rowCount()):
            vals = [self.table.item(r, c).text() if self.table.item(r, c) else "" for c in range(self.table.columnCount())]
            rows.append("\t".join(vals))
        _P(path).write_text("\n".join(rows), encoding="utf-8")


class SweepScope(QWidget):
    """Sweep curve: t_switch vs SCP/COP (reuses HistoryScope logic) + table."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.label = QLabel("Sweep — t_switch vs SCP/COP")
        self.label.setStyleSheet("color:#e2e8f0; font-weight:600; font-size:9pt;")
        layout.addWidget(self.label)
        self.fig = Figure(figsize=(5, 2.8), facecolor="#0f1115")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#0f1115")
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas, 2)
        self.table = QTableWidget(0, 0)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSortingEnabled(False)
        layout.addWidget(self.table, 2)
        self.info = QLabel("No sweep yet.")
        self.info.setStyleSheet("color:#64748b; font-size:7.5pt;")
        layout.addWidget(self.info)
        self._df = None  # type: ignore
        self._placeholder()

    def _placeholder(self) -> None:
        self.ax.clear()
        self.ax.set_facecolor("#0f1115")
        self.ax.text(0.5, 0.5, "Sweep curve will appear after a Sweep run\n(t_switch vs SCP/COP or material vs score)",
                     ha="center", va="center", color="#475569", fontsize=8, transform=self.ax.transAxes)
        self.ax.set_axis_off()
        self.fig.tight_layout()
        self.canvas.draw()

    def set_sweep_dataframe(self, df: Any, title: str | None = None) -> None:
        if title:
            self.label.setText(title)
        self._df = df
        # also alias for DataFrameScope compat
        self.set_dataframe(df, title=title)

    def set_dataframe(self, df: Any, title: str | None = None) -> None:
        if title:
            self.label.setText(title)
        self._df = df
        if df is None or (hasattr(df, "empty") and df.empty):
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            self._placeholder()
            self.info.setText("(empty sweep)")
            return
        try:
            import pandas as pd  # noqa: WPS433
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
        except Exception:
            pass
        self._df = df
        cols = list(df.columns)
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels([str(c) for c in cols])
        self.table.setRowCount(len(df))
        for r, (_, row) in enumerate(df.iterrows()):
            for c, col in enumerate(cols):
                v = row[col]
                txt = f"{v:.6g}" if isinstance(v, float) else str(v)
                it = QTableWidgetItem(txt)
                if isinstance(v, (int, float)):
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(r, c, it)
        self.table.resizeColumnsToContents()
        # plot
        self._plot(df)

    def _plot(self, df: Any) -> None:
        try:
            import pandas as pd  # noqa: WPS433
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
        except Exception:
            self._placeholder()
            return
        self.ax.clear()
        self.ax.set_facecolor("#0f1115")
        # t_switch sweep
        if "t_switch_s" in df.columns:
            x = df["t_switch_s"].to_numpy(dtype=float)
            # SCP and COP
            has_scp = "SCP_W_kg" in df.columns
            has_cop = "COP" in df.columns
            if has_scp:
                self.ax.plot(x, df["SCP_W_kg"].to_numpy(dtype=float), "-o", color="#38bdf8", ms=4, lw=1.6, label="SCP [W/kg]")
                self.ax.set_ylabel("SCP [W/kg]", color="#38bdf8", fontsize=7)
                self.ax.tick_params(colors="#94a3b8", labelsize=7)
            if has_cop:
                ax2 = self.ax.twinx()
                ax2.set_facecolor("#0f1115")
                ax2.plot(x, df["COP"].to_numpy(dtype=float), "-s", color="#f59e0b", ms=4, lw=1.4, label="COP")
                ax2.set_ylabel("COP", color="#f59e0b", fontsize=7)
                ax2.tick_params(colors="#f59e0b", labelsize=7)
                for spine in ax2.spines.values():
                    spine.set_color("#1e222a")
                h1, l1 = self.ax.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                self.ax.legend(h1+h2, l1+l2, facecolor="#1b1e24", edgecolor="#2a2f3a", fontsize=7, labelcolor="#e2e8f0")
            else:
                self.ax.legend(facecolor="#1b1e24", edgecolor="#2a2f3a", fontsize=7, labelcolor="#e2e8f0")
            self.ax.set_xlabel("t_switch [s]", color="#94a3b8", fontsize=7)
            for spine in self.ax.spines.values():
                spine.set_color("#1e222a")
            # best point annotation
            if has_scp and len(df):
                idx = int(df["SCP_W_kg"].idxmax())
                best_x = float(df.loc[idx, "t_switch_s"])
                best_y = float(df.loc[idx, "SCP_W_kg"])
                self.ax.annotate(f"best SCP {best_y:.1f} @ {best_x:.0f}s", xy=(best_x, best_y), xytext=(6, 10), textcoords="offset points",
                                 color="#e2e8f0", fontsize=7, ha="left",
                                 bbox=dict(boxstyle="round,pad=0.2", fc="#1b1e24", ec="#2a2f3a", alpha=0.9),
                                 arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=0.8))
            self.ax.set_title("Sweep — SCP/COP vs t_switch", color="#e2e8f0", fontsize=9)
        elif "material" in df.columns and "score" in df.columns:
            # ranking-style bar horizontal
            sub = df.sort_values("score", ascending=False).head(12)
            y = range(len(sub))
            self.ax.barh(list(y), sub["score"].to_numpy(dtype=float), color="#38bdf8", alpha=0.85)
            self.ax.set_yticks(list(y))
            self.ax.set_yticklabels([str(s)[:18] for s in sub["material"].tolist()], fontsize=6, color="#94a3b8")
            self.ax.set_xlabel("score (profile-weighted)", color="#94a3b8", fontsize=7)
            self.ax.invert_yaxis()
            for spine in self.ax.spines.values():
                spine.set_color("#1e222a")
            self.ax.tick_params(colors="#94a3b8", labelsize=7)
            self.ax.set_title("Sweep — material ranking (top 12)", color="#e2e8f0", fontsize=9)
        else:
            # generic: plot first numeric vs first column
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(num_cols) >= 2:
                self.ax.plot(df[num_cols[0]].to_numpy(dtype=float), df[num_cols[1]].to_numpy(dtype=float), "-o", color="#38bdf8", ms=4, lw=1.4)
                self.ax.set_xlabel(str(num_cols[0]), color="#94a3b8", fontsize=7)
                self.ax.set_ylabel(str(num_cols[1]), color="#94a3b8", fontsize=7)
                self.ax.tick_params(colors="#94a3b8", labelsize=7)
                for spine in self.ax.spines.values():
                    spine.set_color("#1e222a")
                self.ax.set_title("Sweep — generic sweep", color="#e2e8f0", fontsize=9)
            else:
                self._placeholder()
                return
        self.fig.tight_layout()
        self.canvas.draw()
        # info
        try:
            n = len(df)
            if "SCP_W_kg" in df.columns:
                best_scp = float(df["SCP_W_kg"].max())
                best_idx = int(df["SCP_W_kg"].idxmax())
                best_t = float(df.loc[best_idx, "t_switch_s"]) if "t_switch_s" in df.columns else 0
                self.info.setText(f"{n} sweep points  ·  best SCP {best_scp:.1f} W/kg @ t_switch {best_t:.0f}s")
            else:
                self.info.setText(f"{n} sweep points  ·  cols {list(df.columns)[:6]}")
        except Exception:
            self.info.setText(f"{len(df)} sweep points")

    def set_text(self, text: str, title: str | None = None) -> None:
        if title:
            self.label.setText(title)
        self.info.setText(text[:200])
        # table cleared when text only
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        # try to keep placeholder
        if not text.strip():
            self._placeholder()

    def set_sweep_text(self, text: str, title: str | None = None) -> None:
        self.set_text(text, title=title)

    def save_png(self, path: str) -> None:
        self.fig.savefig(path, dpi=180, facecolor=self.fig.get_facecolor())

    def save_csv(self, path: str) -> None:
        if self._df is not None:
            try:
                self._df.to_csv(path, index=False)  # type: ignore
                return
            except Exception:
                pass
        open(path, "w", encoding="utf-8").write(self.info.text())


class ScopeTabs(QTabWidget):
    """Container for all scope cards."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self.setDocumentMode(True)
        self.setTabPosition(QTabWidget.TabPosition.North)
        self.metrics = MetricsScope()
        self.history = HistoryScope()
        self.trace = TraceScope()
        self.log = TextScope("Log")
        self.python = TextScope("Python")
        self.python._save_filter = "Python (*.py);;All (*)"
        self.python._save_default = "experiment.py"
        self.ranking = DataFrameScope("Ranking")
        self.calibration = DataFrameScope("Calibration")
        self.compare = CompareScope()
        self.sweep = SweepScope()

        self.addTab(self.metrics, "Metrics")
        self.addTab(self.history, "History")
        self.addTab(self.trace, "Trace")
        self.addTab(self.log, "Log")
        self.addTab(self.python, "Python")
        self.addTab(self.ranking, "Ranking")
        self.addTab(self.calibration, "Calibration")
        self.addTab(self.compare, "Compare")
        self.addTab(self.sweep, "Sweep")

    def set_result(self, payload: Any) -> None:
        # metrics
        self.metrics.set_metrics(payload.metrics, extra=f"elapsed {payload.elapsed_s:.2f}s")
        # history
        hist = getattr(payload.result, "history", []) if payload.result is not None else []
        best = getattr(payload.result, "best_objective", None) if payload.result is not None else None
        if hist:
            self.history.set_history(hist, best_objective=best)  # type: ignore[arg-type]
        else:
            # fabricate single-point history from metrics
            self.history.set_history([], best_objective=best)
        # trace
        self.trace.set_trace(payload.trace)
        # python
        self.python.set_text(payload.python_code, title="Python — reproducible script (Generate Python)")
        # sweep in-place if payload carries sweep_df
        sweep_df = getattr(payload, "sweep_df", None)
        if sweep_df is not None:
            try:
                self.set_sweep_dataframe(sweep_df, title=f"Sweep — {len(sweep_df)} points")
            except Exception:
                pass
        # also handle sweep_axis for title
        if getattr(payload, "sweep_axis", None):
            self.sweep.label.setText(f"Sweep — {payload.sweep_axis}")

    def set_ranking_dataframe(self, df: Any, title: str | None = None) -> None:
        self.ranking.set_dataframe(df, title=title or "Ranking")
        self.setCurrentWidget(self.ranking)

    def set_ranking_text(self, text: str) -> None:
        # backward-compat for tests that use text
        self.ranking.set_text(text, title="Ranking")
        self.setCurrentWidget(self.ranking)

    def set_calibration_dataframe(self, df: Any, title: str | None = None) -> None:
        self.calibration.set_dataframe(df, title=title or "Calibration")
        self.setCurrentWidget(self.calibration)

    def set_compare(self, a: Any, b: Any, title: str | None = None, **kwargs) -> None:
        self.compare.set_compare(a, b, title=title, **kwargs)
        self.setCurrentWidget(self.compare)

    def set_compare_results(self, a: Any, b: Any, title: str | None = None) -> None:
        self.compare.set_compare_results(a, b, title=title)
        self.setCurrentWidget(self.compare)

    def set_sweep_dataframe(self, df: Any, title: str | None = None) -> None:
        self.sweep.set_sweep_dataframe(df, title=title or "Sweep")
        self.setCurrentWidget(self.sweep)

    def set_sweep_text(self, text: str, title: str | None = None) -> None:
        self.sweep.set_text(text, title=title or "Sweep")
        self.setCurrentWidget(self.sweep)

    # Compatibility: allow old sweeps that fed DataFrame to TextScope-style handling
    def set_sweep(self, df_or_text: Any, title: str | None = None) -> None:
        try:
            import pandas as pd  # noqa: WPS433
            if isinstance(df_or_text, pd.DataFrame) or (isinstance(df_or_text, list) and df_or_text and isinstance(df_or_text[0], dict)):
                self.set_sweep_dataframe(df_or_text, title=title)
                return
        except Exception:
            pass
        if isinstance(df_or_text, str):
            self.set_sweep_text(df_or_text, title=title)
        else:
            self.set_sweep_dataframe(df_or_text, title=title)

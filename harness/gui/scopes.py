"""Harness-domain scopes: episode TraceScope, SweepScope, and the tab set.

Generic widgets (tables, history, compare, log) live in ``kit`` — these two
know the cooling physics trace/sweep shapes (Bed1D / TwoBed series keys,
t_switch and material sweep axes).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from .kit import (
    CompareScope,
    DataFrameScope,
    HistoryScope,
    MetricsScope,
    TextScope,
    dark_figure,
    placeholder_text,
    style_axis,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: F401  (re-export for embedders)

FIG_FG = "#e2e8f0"
FIG_AXIS = "#94a3b8"
FIG_SPINE = "#1e222a"
FIG_PANEL = "#1b1e24"
FIG_ACCENT = "#38bdf8"
FIG_AMBER = "#f59e0b"


class TraceScope(QWidget):
    """Episode trace: 2-D field heatmap if present, else Bed1D/TwoBed series."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.fig, self.canvas = dark_figure((5, 3.2))
        self.ax = self.fig.axes[0]
        layout.addWidget(self.canvas, 1)
        self.info = QLabel("No trace yet — enable Collect trace and Run.")
        self.info.setStyleSheet("color:#64748b; font-size:7.5pt;")
        layout.addWidget(self.info)
        self._placeholder()

    def _placeholder(self) -> None:
        placeholder_text(self.ax, "Trace appears after a dynamic run\n(Bed1D / TwoBed with Collect trace)")
        self.fig.tight_layout()
        self.canvas.draw()

    def set_trace(self, trace: Any) -> None:
        if trace is None or not getattr(trace, "series", None):
            self._placeholder()
            self.info.setText("No trace — static problem or collection disabled.")
            return
        series = trace.series
        img_key = None
        for k, v in series.items():
            arr = np.asarray(v)
            if arr.ndim == 2 and arr.shape[0] > 4 and arr.shape[1] > 1:
                img_key = k
                break
        if img_key is not None:
            arr = np.asarray(series[img_key])
            self.ax.clear()
            style_axis(self.ax)
            im = self.ax.imshow(arr, aspect="auto", origin="lower",
                                cmap="inferno", interpolation="nearest")
            self.ax.set_xlabel("space cell", color=FIG_AXIS, fontsize=7)
            self.ax.set_ylabel("time step", color=FIG_AXIS, fontsize=7)
            self.ax.set_title(f"Trace  {img_key}  [{arr.shape[0]} × {arr.shape[1]}]",
                              color=FIG_FG, fontsize=9)
            self.fig.colorbar(im, ax=self.ax, shrink=0.85).ax.tick_params(
                colors=FIG_AXIS, labelsize=6)
            self.fig.tight_layout()
            self.canvas.draw()
            self.info.setText(f"series {img_key}: {arr.shape}  ·  summary {trace.summary}")
            return

        self.ax.clear()
        style_axis(self.ax)
        is_twobed = any(k.startswith("A_") or k.startswith("B_") for k in series)

        def arr(key: str) -> np.ndarray | None:
            if key not in series:
                return None
            a = np.asarray(series[key]).ravel()
            if a.size > 2000:
                return a[:: max(1, a.size // 600)]
            return a

        def shade(phase: np.ndarray | None) -> None:
            if phase is None or phase.size < 4:
                return
            des, in_des, start = phase > 0.5, False, 0
            for i, v in enumerate(des):
                if v and not in_des:
                    start, in_des = i, True
                elif not v and in_des:
                    self.ax.axvspan(start, i, color="#1e293b", alpha=0.35, zorder=0)
                    in_des = False
            if in_des:
                self.ax.axvspan(start, len(des), color="#1e293b", alpha=0.35, zorder=0)

        if is_twobed:
            a_t, b_t = arr("A_t_wall_k"), arr("B_t_wall_k")
            a_q, b_q = arr("A_q_mean_kg_kg"), arr("B_q_mean_kg_kg")
            shade(arr("sys_phase") if "sys_phase" in series else arr("A_phase"))
            if a_t is not None:
                self.ax.plot(a_t, label="A T_wall", color=FIG_AMBER, lw=1.2)
            if b_t is not None:
                self.ax.plot(b_t, label="B T_wall", color="#06b6d4", lw=1.2)
            ax2 = self.ax.twinx()
            ax2.set_facecolor("#0f1115")
            if a_q is not None:
                ax2.plot(a_q, label="A q", color=FIG_AMBER, ls="--", lw=1.0, alpha=0.7)
            if b_q is not None:
                ax2.plot(b_q, label="B q", color="#06b6d4", ls="--", lw=1.0, alpha=0.7)
            ax2.tick_params(colors=FIG_AXIS, labelsize=6)
            for spine in ax2.spines.values():
                spine.set_color(FIG_SPINE)
            h1, l1 = self.ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            self.ax.legend(h1 + h2, l1 + l2, facecolor=FIG_PANEL,
                           edgecolor="#2a2f3a", fontsize=6, labelcolor=FIG_FG,
                           loc="upper right")
            self.ax.set_title("TwoBed trace — wall temps (solid) & uptake (dashed) — shaded = des",
                              color=FIG_FG, fontsize=8)
        else:
            tw, tb = arr("t_wall_k"), arr("t_bed_mean_k")
            q, qstar = arr("q_mean_kg_kg"), arr("q_star_mean_kg_kg")
            shade(arr("phase"))
            if tw is not None:
                self.ax.plot(tw, label="T_wall", color=FIG_AMBER, lw=1.3)
            if tb is not None:
                self.ax.plot(tb, label="T_bed mean", color="#eab308", lw=1.1)
            ax2 = self.ax.twinx()
            ax2.set_facecolor("#0f1115")
            if q is not None:
                ax2.plot(q, label="q mean", color="#22c55e", lw=1.1)
                if qstar is not None:
                    ax2.plot(qstar, label="q* mean", color="#22c55e", ls="--",
                             lw=0.9, alpha=0.6)
            ax2.tick_params(colors=FIG_AXIS, labelsize=6)
            for spine in ax2.spines.values():
                spine.set_color(FIG_SPINE)
            h1, l1 = self.ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            self.ax.legend(h1 + h2, l1 + l2, facecolor=FIG_PANEL,
                           edgecolor="#2a2f3a", fontsize=6, labelcolor=FIG_FG,
                           loc="upper right")
            self.ax.set_title("Bed1D trace — temps (solid) & uptake (dashed) — shaded = des",
                              color=FIG_FG, fontsize=8)
        self.ax.set_xlabel("trace step (downsampled)", color=FIG_AXIS, fontsize=7)
        self.ax.tick_params(colors=FIG_AXIS, labelsize=7)
        self.ax.set_axis_on()
        self.fig.tight_layout()
        self.canvas.draw()
        self.info.setText(f"{len(series)} series  ·  summary {trace.summary}")

    def save_png(self, path: str) -> None:
        self.fig.savefig(path, dpi=180, facecolor=self.fig.get_facecolor())


class SweepScope(QWidget):
    """Sweep curve + table (t_switch vs SCP/COP, material ranking, generic)."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.label = QLabel("Sweep")
        self.label.setStyleSheet("color:#e2e8f0; font-weight:600; font-size:9pt;")
        layout.addWidget(self.label)
        self.fig, self.canvas = dark_figure((5, 2.8))
        self.ax = self.fig.axes[0]
        layout.addWidget(self.canvas, 2)
        self.table = QTableWidget(0, 0)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.table, 2)
        self.info = QLabel("No sweep yet.")
        self.info.setStyleSheet("color:#64748b; font-size:7.5pt;")
        layout.addWidget(self.info)
        self._df = None
        self._placeholder()

    def _placeholder(self) -> None:
        placeholder_text(self.ax, "Sweep curve appears after a Sweep run\n(t_switch vs SCP/COP or material vs score)")
        self.fig.tight_layout()
        self.canvas.draw()

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
            import pandas as pd  # noqa: PLC0415

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
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                        | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(r, c, it)
        self.table.resizeColumnsToContents()
        self._plot(df)

    def _plot(self, df: Any) -> None:
        try:
            import pandas as pd  # noqa: PLC0415

            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
        except Exception:
            self._placeholder()
            return
        self.ax.clear()
        style_axis(self.ax)
        if "t_switch_s" in df.columns:
            x = df["t_switch_s"].to_numpy(dtype=float)
            has_scp = "SCP_W_kg" in df.columns
            has_cop = "COP" in df.columns
            if has_scp:
                self.ax.plot(x, df["SCP_W_kg"].to_numpy(dtype=float), "-o",
                             color=FIG_ACCENT, ms=4, lw=1.6, label="SCP [W/kg]")
                self.ax.set_ylabel("SCP [W/kg]", color=FIG_ACCENT, fontsize=7)
            if has_cop:
                ax2 = self.ax.twinx()
                ax2.set_facecolor("#0f1115")
                ax2.plot(x, df["COP"].to_numpy(dtype=float), "-s",
                         color=FIG_AMBER, ms=4, lw=1.4, label="COP")
                ax2.set_ylabel("COP", color=FIG_AMBER, fontsize=7)
                ax2.tick_params(colors=FIG_AMBER, labelsize=7)
                for spine in ax2.spines.values():
                    spine.set_color(FIG_SPINE)
                h1, l1 = self.ax.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                self.ax.legend(h1 + h2, l1 + l2, facecolor=FIG_PANEL,
                               edgecolor="#2a2f3a", fontsize=7, labelcolor=FIG_FG)
            elif has_scp:
                self.ax.legend(facecolor=FIG_PANEL, edgecolor="#2a2f3a",
                               fontsize=7, labelcolor=FIG_FG)
            self.ax.set_xlabel("t_switch [s]", color=FIG_AXIS, fontsize=7)
            if has_scp and len(df):
                idx = int(df["SCP_W_kg"].idxmax())
                best_x, best_y = float(df.loc[idx, "t_switch_s"]), float(df.loc[idx, "SCP_W_kg"])
                self.ax.annotate(f"best SCP {best_y:.1f} @ {best_x:.0f}s",
                                 xy=(best_x, best_y), xytext=(6, 10),
                                 textcoords="offset points", color=FIG_FG,
                                 fontsize=7, ha="left",
                                 bbox={"boxstyle": "round,pad=0.2", "fc": FIG_PANEL,
                                       "ec": "#2a2f3a", "alpha": 0.9},
                                 arrowprops={"arrowstyle": "->", "color": FIG_ACCENT,
                                             "lw": 0.8})
            self.ax.set_title("Sweep — SCP/COP vs t_switch", color=FIG_FG, fontsize=9)
        elif "material" in df.columns and "score" in df.columns:
            sub = df.sort_values("score", ascending=False).head(12)
            y = range(len(sub))
            self.ax.barh(list(y), sub["score"].to_numpy(dtype=float),
                         color=FIG_ACCENT, alpha=0.85)
            self.ax.set_yticks(list(y))
            self.ax.set_yticklabels([str(s)[:18] for s in sub["material"].tolist()],
                                    fontsize=6, color=FIG_AXIS)
            self.ax.set_xlabel("score (profile-weighted)", color=FIG_AXIS, fontsize=7)
            self.ax.invert_yaxis()
            self.ax.set_title("Sweep — material ranking (top 12)", color=FIG_FG, fontsize=9)
        else:
            num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
            if len(num_cols) >= 2:
                self.ax.plot(df[num_cols[0]].to_numpy(dtype=float),
                             df[num_cols[1]].to_numpy(dtype=float), "-o",
                             color=FIG_ACCENT, ms=4, lw=1.4)
                self.ax.set_xlabel(str(num_cols[0]), color=FIG_AXIS, fontsize=7)
                self.ax.set_ylabel(str(num_cols[1]), color=FIG_AXIS, fontsize=7)
                self.ax.set_title("Sweep", color=FIG_FG, fontsize=9)
            else:
                self._placeholder()
                return
        self.ax.set_axis_on()
        self.fig.tight_layout()
        self.canvas.draw()
        try:
            self.info.setText(f"{len(df)} sweep points  ·  cols {list(df.columns)[:6]}")
        except Exception:
            self.info.setText("sweep done")

    def save_png(self, path: str) -> None:
        self.fig.savefig(path, dpi=180, facecolor=self.fig.get_facecolor())

    def save_csv(self, path: str) -> None:
        if self._df is not None:
            self._df.to_csv(path, index=False)


class ScopeTabs(QTabWidget):
    """Right-hand result tabs for the launcher."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self.setDocumentMode(True)
        self.metrics = MetricsScope()
        self.history = HistoryScope()
        self.trace = TraceScope()
        self.table = DataFrameScope("Table")
        self.compare = CompareScope()
        self.sweep = SweepScope()
        self.log = TextScope("Log")
        self.python = TextScope("Python")
        self.python._save_filter = "Python (*.py);;All (*)"
        self.python._save_default = "experiment.py"
        for w, name in ((self.metrics, "Metrics"), (self.history, "History"),
                        (self.trace, "Trace"), (self.table, "Table"),
                        (self.compare, "Compare"), (self.sweep, "Sweep"),
                        (self.log, "Log"), (self.python, "Python")):
            self.addTab(w, name)

    def set_result(self, payload: Any) -> None:
        self.metrics.set_metrics(payload.metrics, extra=f"elapsed {payload.elapsed_s:.2f}s")
        hist = getattr(payload.result, "history", []) if payload.result is not None else []
        best = getattr(payload.result, "best_objective", None) if payload.result is not None else None
        self.history.set_history(hist, best_objective=best)
        self.trace.set_trace(payload.trace)
        self.python.set_text(payload.python_code, title="Python — reproducible script")
        sweep_df = getattr(payload, "sweep_df", None)
        if sweep_df is not None:
            self.sweep.set_dataframe(sweep_df, title=f"Sweep — {payload.sweep_axis}")
            self.setCurrentWidget(self.sweep)

    def set_log(self, text: str) -> None:
        self.log.set_text(text)

    def append_log(self, line: str) -> None:
        self.log.append(line)

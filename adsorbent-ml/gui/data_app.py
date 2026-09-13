#!/usr/bin/env python
"""Adsorbent data explorer — PyQt6 app over the adsorbent-ml datasets.

Run:  python adsorbent-ml/gui/data_app.py        (from the repo root)

Six pages, all read-only over ``data_cache/`` via ``datasets.py``:
Overview (what's on disk), Materials (labels + features explorer),
Isotherms & fits (unit-free normalized curves + Q_st vs T), Stability
(MOFSimplify gate), Structures (inventory + OPTIMADE manifests), and
Rankings (on-demand harness COP sweep over the material table).

All widgets come from ``harness.gui.kit`` (shared dark theme/scopes); heavy
loads run on the kit's background worker. Data pages never import jax on the
UI thread — the Rankings page does it inside the worker.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
for _sub in ("data", "features", "eval", "models"):
    _p = str(REPO_ROOT / "adsorbent-ml" / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# VRAM hygiene BEFORE anything imports harness/jax (4 GB shared GPU):
# no XLA preallocation; freed device memory returns to the driver.
from harness.gpu import clear_caches, configure as gpu_configure  # noqa: E402

gpu_configure()

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from datasets import datasets
from harness.gui.kit import (
    DARK_QSS,
    DataFrameScope,
    FIG_AXIS,
    FIG_FG,
    JobRunner,
    dark_figure,
    style_axis,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg


def _parse_series(raw: str) -> np.ndarray:
    """ISODB parquet stores point lists as strings."""
    try:
        return np.asarray(ast.literal_eval(raw), dtype=float)
    except (ValueError, SyntaxError):
        return np.asarray([], dtype=float)


# -- pages -------------------------------------------------------------------


class OverviewPage(QWidget):
    def __init__(self, runner: JobRunner) -> None:
        super().__init__()
        self.runner = runner
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        self.table = DataFrameScope("Datasets on disk (data_cache/)")
        self.missing = TextIntro()
        lay.addWidget(self.table, 3)
        lay.addWidget(self.missing, 1)
        self.reload()

    def reload(self) -> None:
        status = datasets.status()
        rows = []
        for key, value in sorted(status.items()):
            if key in ("data_cache", "structures") or not isinstance(value, dict):
                continue
            n = value.get("n_entries") or value.get("rows") or value.get("n_rows")
            rows.append({
                "dataset": key,
                "entries": n if n is not None else "",
                "notes": (value.get("filter") or value.get("stage")
                          or value.get("provenance") or "")[:90],
            })
        self.table.set_dataframe(pd.DataFrame(rows), title="Datasets on disk (data_cache/)")
        missing = MISSING_BUILDS()
        self.missing.set_text(
            "Missing caches (build with):\n  " + "\n  ".join(missing)
            if missing else "All dataset caches present.")

    def set_status(self, _payload) -> None:  # worker-compat hook
        self.reload()


class TextIntro(QTextEdit):
    def __init__(self) -> None:
        super().__init__()
        self.setReadOnly(True)
        self.setStyleSheet(
            "background:#111318; color:#cbd5e1; font-family: monospace; "
            "font-size:8pt; border:1px solid #1e222a; border-radius:6px;")

    def set_text(self, text: str) -> None:
        self.setPlainText(text)


def MISSING_BUILDS() -> list[str]:
    checks = [
        (datasets.CACHE / "fits" / "da_params.csv", "python adsorbent-ml/data/fit_da.py"),
        (datasets.CACHE / "n2" / "labels.csv", "python adsorbent-ml/training/train_baseline.py"),
        (datasets.CACHE / "stability" / "mofsimplify_ssd_tsd.csv",
         "python adsorbent-ml/data/stability_export.py"),
        (datasets.CACHE / "optimade", "python adsorbent-ml/data/optimade_export.py --with-cifs ..."),
    ]
    return [cmd for path, cmd in checks if not Path(path).exists()]


class MaterialsPage(QWidget):
    def __init__(self, runner: JobRunner) -> None:
        super().__init__()
        self.runner = runner
        self.X: pd.DataFrame | None = None
        self.Y: pd.DataFrame | None = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        top = QHBoxLayout()
        self.family = QComboBox()
        self.family.addItem("(all families)")
        self.family.currentTextChanged.connect(self._apply_filter)
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("name contains …")
        self.filter_edit.textChanged.connect(self._apply_filter)
        top.addWidget(QLabel("family"))
        top.addWidget(self.family)
        top.addWidget(QLabel("name"))
        top.addWidget(self.filter_edit, 1)
        lay.addLayout(top)

        split = QSplitter(Qt.Orientation.Vertical)
        self.table = DataFrameScope("Material table (labels + coverage)")
        self.table.table.clicked.connect(self._on_row)
        self.detail = TextIntro()
        self.detail.setMaximumHeight(150)
        split.addWidget(self.table)
        split.addWidget(self.detail)
        lay.addWidget(split, 3)

        self.fig, self.canvas = dark_figure((4, 2.2))
        self.fig.axes[0].set_title("q_sat [kg/kg] distribution by family",
                                   color=FIG_FG, fontsize=9)
        lay.addWidget(self.canvas, 2)
        self.load()

    def load(self) -> None:
        try:
            X, Y, prov = datasets.material_table()
        except FileNotFoundError as exc:
            self.detail.set_text(str(exc))
            return
        self.X, self.Y = X, Y
        families = sorted(str(f) for f in Y["family"].unique())
        self.family.blockSignals(True)
        self.family.clear()
        self.family.addItem("(all families)")
        self.family.addItems(families)
        self.family.blockSignals(False)
        cov = prov["coverage"]
        self.detail.set_text(
            f"coverage: pores {cov['n_with_pores']} CoRE + {cov['n_with_iza_pores']} IZA + "
            f"{cov['n_with_qmof_pores']} QMOF of {cov['n_rows']} · "
            f"formulas {cov['n_with_formula']} · rows {cov['n_rows']}")
        self._apply_filter()
        self._plot()

    def _apply_filter(self) -> None:
        if self.Y is None or self.X is None:
            return
        fam = self.family.currentText()
        name = self.filter_edit.text().strip().lower()
        mask = pd.Series(True, index=self.Y.index)
        if fam != "(all families)":
            mask &= self.Y["family"] == fam
        if name:
            mask &= self.Y["name"].str.lower().str.contains(name, regex=False)
        cols = [c for c in ["name", "family", "source", "anchor", "n_isotherms",
                            "q_sat_kg_kg", "q_st_j_kg", "e_char_j_mol", "n_da"]
                if c in self.Y.columns]
        pore_cols = [c for c in ["pore_LCD", "pore_PLD", "pore_has_data",
                                 "pore_iza", "pore_qmof"] if c in self.X.columns]
        merged = self.Y.loc[mask, cols]
        if pore_cols:
            # anchor+fit rows can share a name — keep one pore row per material
            merged = merged.merge(
                self.X[["name"] + pore_cols].drop_duplicates("name"),
                on="name", how="left")
        self.table.set_dataframe(merged.reset_index(drop=True),
                                 title=f"Material table ({len(merged)} rows)")

    def _on_row(self, index) -> None:
        if self.Y is None:
            return
        name = self.table.table.item(index.row(), 0).text()
        row = self.Y[self.Y["name"] == name]
        if len(row):
            r = row.iloc[0].to_dict()
            self.detail.set_text("\n".join(
                f"{k}: {v}" for k, v in r.items()
                if k != "name" and pd.notna(v)))

    def _plot(self) -> None:
        ax = self.fig.axes[0]
        ax.clear()
        style_axis(ax)
        if self.Y is not None and "q_sat_kg_kg" in self.Y.columns:
            data = self.Y.dropna(subset=["q_sat_kg_kg"])
            fams = sorted(data["family"].unique())
            arrays = [data.loc[data["family"] == f, "q_sat_kg_kg"].to_numpy()
                      for f in fams]
            ax.hist(arrays, bins=24, stacked=True, label=fams,
                    color=["#38bdf8", "#f59e0b", "#22c55e", "#a78bfa",
                           "#f472b6", "#94a3b8", "#eab308"][:len(fams)])
            ax.set_xlabel("q_sat [kg/kg]", color=FIG_AXIS, fontsize=7)
            ax.legend(facecolor="#1b1e24", edgecolor="#2a2f3a", fontsize=6,
                      labelcolor=FIG_FG)
        self.fig.tight_layout()
        self.canvas.draw()


class IsothermsPage(QWidget):
    def __init__(self, runner: JobRunner) -> None:
        super().__init__()
        self.runner = runner
        self.fits: pd.DataFrame | None = None
        self.isodb: pd.DataFrame | None = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        top = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("filter materials (e.g. silica, 13X, CuBTC) …")
        self.listw = QListWidget()
        self.listw.currentTextChanged.connect(self._select)
        top.addWidget(self.listw, 1)
        side = QVBoxLayout()
        side.addWidget(self.search)
        self.info = TextIntro()
        side.addWidget(self.info, 1)
        top.addLayout(side, 1)
        lay.addLayout(top, 1)

        self.fig, self.canvas = dark_figure((6, 4))
        lay.addWidget(self.canvas, 3)
        self.load()

    def load(self) -> None:
        fits_path = datasets.CACHE / "fits" / "da_params.csv"
        if not fits_path.exists():
            self.info.set_text(str(FileNotFoundError(fits_path)))
            return
        self.fits = pd.read_csv(fits_path)
        iso_path = datasets.CACHE / "isodb" / "water_isotherms.parquet"
        if iso_path.exists():
            self.isodb = pd.read_parquet(iso_path, columns=[
                "filename", "temperature_K", "pressure", "uptake"])
        counts = (self.fits[self.fits["fit_flag"] == "ok"]
                  .groupby("name").size().sort_values(ascending=False))
        self.listw.addItems([str(n) for n in counts.index[:200]])

    def _select(self, name: str) -> None:
        if self.fits is None or not name:
            return
        rows = self.fits[(self.fits["name"] == name)
                         & (self.fits["fit_flag"] == "ok")]
        ax = self.fig.axes[0]
        ax.clear()
        style_axis(ax)
        qst_rows = rows.dropna(subset=["q_st_j_kg"]).sort_values("temperature_K")
        for _, r in rows.iterrows():
            T, qsat = float(r["temperature_K"]), float(r["q_sat_kg_kg"])
            E, n = float(r["e_char_j_mol"]), float(r["n_da"])
            psat = float(r["psat_pa"])
            if not all(np.isfinite(v) for v in (T, qsat, E, n, psat)) or qsat <= 0:
                continue
            p = np.logspace(np.log10(psat * 0.005), np.log10(psat * 1.05), 120)
            A = np.maximum(8.3144626 * T * np.log(psat / p), 0.0)
            q_norm = np.exp(-((A / E) ** n))
            ax.plot(p / psat, q_norm, "-", lw=1.3,
                    label=f"{T:.0f} K  rmse {float(r['fit_rmse']):.3f}")
            raw = self._raw_points(r)
            if raw is not None:
                ax.plot(raw[:, 0], raw[:, 1], "o", ms=4, alpha=0.7)
        ax.set_xlabel("p / p_sat", color=FIG_AXIS, fontsize=7)
        ax.set_ylabel("normalized uptake  q / q_sat", color=FIG_AXIS, fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.legend(facecolor="#1b1e24", edgecolor="#2a2f3a", fontsize=6,
                  labelcolor=FIG_FG)
        ax.set_title(f"{name} — Dubinin–Astakhov fits (unit-free)", color=FIG_FG, fontsize=9)
        self.fig.tight_layout()
        self.canvas.draw()
        lines = [f"{len(rows)} fitted isotherms"]
        if len(qst_rows):
            lines.append("Q_st [kJ/mol] vs T [K]: " + "; ".join(
                f"{float(r.temperature_K):.0f}→{float(r.q_st_j_kg) / 1e3:.1f}"
                for r in qst_rows.itertuples()))
        else:
            lines.append("no multi-T Q_st (single-temperature fits only)")
        self.info.set_text("\n".join(lines))

    def _raw_points(self, fit_row: pd.Series) -> np.ndarray | None:
        if self.isodb is None:
            return None
        hit = self.isodb[self.isodb["filename"] == fit_row["isotherm_id"]]
        if not len(hit):
            return None
        row = hit.iloc[0]
        p = _parse_series(row["pressure"]) * 1e5  # ISODB export is bar throughout
        q = _parse_series(row["uptake"])
        qsat_native, psat = float(fit_row["q_sat_native"]), float(fit_row["psat_pa"])
        if not len(p) or not len(q) or not np.isfinite(qsat_native) or qsat_native <= 0:
            return None
        keep = (p > 0) & (p < psat)
        return np.column_stack([p[keep] / psat, q[keep] / qsat_native])


class StabilityPage(QWidget):
    def __init__(self, runner: JobRunner) -> None:
        super().__init__()
        self.runner = runner
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        self.pass_table = DataFrameScope("Feasibility pass")
        self.fail_table = DataFrameScope("Fail (collapse / too fragile / unknown)")
        self.fig, self.canvas = dark_figure((6, 2))
        self.fig.axes[0].set_title("TGA decomposition onsets [°C]", color=FIG_FG, fontsize=9)
        lay.addWidget(self.pass_table, 2)
        lay.addWidget(self.fail_table, 2)
        lay.addWidget(self.canvas, 1)
        self.load()

    def load(self) -> None:
        try:
            ok, bad = datasets.stability_shortlist()
        except FileNotFoundError as exc:
            self.pass_table.set_text(str(exc))
            return
        self.pass_table.set_dataframe(ok, title=f"Feasibility pass ({len(ok)})")
        self.fail_table.set_dataframe(bad, title=f"Fail ({len(bad)})")
        ax = self.fig.axes[0]
        ax.clear()
        style_axis(ax)
        decomp = ok["thermal_decomp_c"].dropna()
        if len(decomp):
            ax.hist(decomp.to_numpy(dtype=float), bins=30, color="#38bdf8")
            ax.axvline(150.0, color="#f59e0b", lw=1.2, ls="--", label="gate 150 °C")
            ax.legend(facecolor="#1b1e24", edgecolor="#2a2f3a", fontsize=6,
                      labelcolor=FIG_FG)
        ax.set_xlabel("T_decomp [°C]", color=FIG_AXIS, fontsize=7)
        self.fig.tight_layout()
        self.canvas.draw()


class StructuresPage(QWidget):
    def __init__(self, runner: JobRunner) -> None:
        super().__init__()
        self.runner = runner
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        self.inv_table = DataFrameScope("Structure caches (CIFs on disk)")
        self.opt_table = DataFrameScope("OPTIMADE pulls (manifests)")
        lay.addWidget(self.inv_table, 1)
        lay.addWidget(self.opt_table, 1)
        self.load()

    def load(self) -> None:
        inv = datasets.structures_inventory()
        rows = [{"source": k, "n_cif": v.get("n_cif", ""), "n_entries": v.get("n_entries", ""),
                 "dir": v.get("dir", "")} for k, v in inv.items()]
        self.inv_table.set_dataframe(pd.DataFrame(rows), title="Structure caches (CIFs on disk)")
        opt_rows = [r for r in rows if r["source"].startswith("optimade/")]
        self.opt_table.set_dataframe(pd.DataFrame(opt_rows),
                                     title=f"OPTIMADE pulls ({len(opt_rows)})")


class RankingsPage(QWidget):
    def __init__(self, runner: JobRunner) -> None:
        super().__init__()
        self.runner = runner
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        top = QHBoxLayout()
        top.addWidget(QLabel("profile"))
        self.profile = QComboBox()
        self.profile.addItems(["datacenter", "cpu", "human", "vehicle"])
        top.addWidget(self.profile)
        self.full_check = QCheckBox("full fitted table")
        top.addWidget(self.full_check)
        self.go = QPushButton("Rank materials")
        self.go.clicked.connect(self.rank)
        top.addWidget(self.go)
        top.addStretch()
        lay.addLayout(top)
        self.sweep = _RankingView()
        lay.addWidget(self.sweep, 1)
        self.info = TextIntro()
        self.info.setMaximumHeight(60)
        self.info.set_text("Ranks the harness anchor materials by profile-weighted "
                           "COP/SCP score (Cycle0D oracle). Tick the box to include "
                           "the full fitted material table.")
        lay.addWidget(self.info)

    def rank(self) -> None:
        self.go.setEnabled(False)
        profile = self.profile.currentText()
        full = self.full_check.isChecked()
        self.runner.start(f"ranking ({profile})", lambda w: _run_ranking(w, profile, full))

    def on_finished(self, payload) -> None:
        self.go.setEnabled(True)
        if isinstance(payload, pd.DataFrame):
            self.sweep.set_dataframe(payload)


class _RankingView(QWidget):
    """Material-score bar + table (SweepScope-shaped, local to this app)."""

    def __init__(self) -> None:
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.fig, self.canvas = dark_figure((6, 2.6))
        self.ax = self.fig.axes[0]
        self.ax.set_title("material ranking (top 15)", color=FIG_FG, fontsize=9)
        lay.addWidget(self.canvas, 1)
        self.table = DataFrameScope("Ranking table")
        lay.addWidget(self.table, 1)

    def set_dataframe(self, df: pd.DataFrame) -> None:
        self.table.set_dataframe(df)
        ax = self.ax
        ax.clear()
        style_axis(ax)
        if len(df) and {"material", "score"} <= set(df.columns):
            sub = df.sort_values("score", ascending=False).head(15)
            y = range(len(sub))
            ax.barh(list(y), sub["score"].to_numpy(dtype=float),
                    color="#38bdf8", alpha=0.85)
            ax.set_yticks(list(y))
            ax.set_yticklabels([str(s)[:24] for s in sub["material"].tolist()],
                               fontsize=6, color=FIG_AXIS)
            ax.invert_yaxis()
        self.fig.tight_layout()
        self.canvas.draw()


# -- main window ---------------------------------------------------------------


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Adsorbent data explorer")
        self.resize(1280, 840)
        self.setStyleSheet(DARK_QSS)

        self.runner_thread = JobRunner(self)
        self.runner_thread.finished.connect(self._on_finished)
        self.runner_thread.failed.connect(self._on_failed)
        self.runner_thread.log.connect(lambda m: self._status.showMessage(m))

        self.nav = QListWidget()
        self.nav.setFixedWidth(170)
        for name in ("Overview", "Materials", "Isotherms & fits", "Stability",
                     "Structures", "Rankings"):
            self.nav.addItem(name)

        self.overview = OverviewPage(self.runner_thread)
        self.materials = MaterialsPage(self.runner_thread)
        self.isotherms = IsothermsPage(self.runner_thread)
        self.stability = StabilityPage(self.runner_thread)
        self.structures = StructuresPage(self.runner_thread)
        self.rankings = RankingsPage(self.runner_thread)

        self.stack = QStackedWidget()
        for page in (self.overview, self.materials, self.isotherms,
                     self.stability, self.structures, self.rankings):
            self.stack.addWidget(page)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.nav)
        splitter.addWidget(self.stack)
        splitter.setSizes([170, 1110])
        self.setCentralWidget(splitter)

        self._status = QStatusBar(self)
        self.setStatusBar(self._status)

    def _on_finished(self, payload) -> None:
        self.rankings.on_finished(payload)

    def _on_failed(self, message: str) -> None:
        self._status.showMessage(f"failed: {message.splitlines()[0][:160]}")
        self.rankings.go.setEnabled(True)


def _run_ranking(worker, profile: str, full: bool) -> pd.DataFrame:
    """Worker fn: sweep the material table through the Cycle0D oracle
    (vmapped batch kernel — one device call per profile)."""
    from harness import rank as rank_mod
    from harness.materials import load_anchors

    mats = list(load_anchors())
    if full:
        # the honestly-flagged sweep set: ok fits, physical q_sat, multi-T
        # Q_st — rows without Q_st cannot enter the equilibrium cycle
        mats += rank_mod.load_sweep_materials()
    worker.log.emit(f"sweeping {len(mats)} materials on {profile}")
    try:
        df = rank_mod.sweep_materials_batched(mats, profiles=[profile])
    except Exception:
        clear_caches()
        raise
    clear_caches()  # reclaim jitted-executable memory while the explorer stays open
    worker.log.emit(f"{len(df)} rows")
    return df



def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()

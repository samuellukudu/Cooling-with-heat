"""Palette dock — searchable, collapsible block list grouped by category."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDockWidget,
    QFrame,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QToolBox,
    QVBoxLayout,
    QWidget,
)

from .model import NODE_CATEGORIES, PHYSICS_KINDS, SCOPE_KINDS


BLOCKS: list[dict[str, Any]] = [
    # Sources
    {"type": "Material", "label": "Material", "icon": "◆", "tip": "MaterialParams row — anchor or fitted ISODB", "data": {"ref": "anchor:Silica gel RD"}},
    {"type": "Profile", "label": "Profile", "icon": "⬢", "tip": "Application profile — 4 presets + raw setpoints", "data": {"ref": "datacenter"}},
    # Physics — expand per kind
    {"type": "Physics", "label": "Cycle0D", "icon": "◈", "tip": "Equilibrium oracle (static)", "data": {"kind": "Cycle0D-v0"}},
    {"type": "Physics", "label": "Bed1D", "icon": "▣", "tip": "Transient 1-D slab — heat + LDF", "data": {"kind": "Bed1D-v0", "n_cells": 16}},
    {"type": "Physics", "label": "TwoBed", "icon": "▦", "tip": "Counter-phase pair + heat recovery", "data": {"kind": "TwoBed-v0"}},
    {"type": "Physics", "label": "TwoBedSchedule", "icon": "▤", "tip": "TwoBed under per-step source series", "data": {"kind": "TwoBedSchedule-v0"}},
    {"type": "Physics", "label": "ForcedConv", "icon": "≋", "tip": "2-D channel advection-diffusion cooling (T-A5 trial)", "data": {"kind": "ForcedConv-v0", "nx": 24, "ny": 8}},
    {"type": "Physics", "label": "Cloak2D", "icon": "◉", "tip": "Thermal-cloak kappa-field (T-A4 trial)", "data": {"kind": "Cloak2D-v0", "nx": 16, "ny": 16}},
    {"type": "Physics", "label": "Absorption", "icon": "♨", "tip": "Lumped single-effect absorption chiller (T-A3 trial: LiBr/H2O class)", "data": {"kind": "AbsorptionCycle-v0"}},
    {"type": "Physics", "label": "Thermoelectric", "icon": "⚡", "tip": "Lumped Peltier pair cooler (T-A2 trial)", "data": {"kind": "Thermoelectric-v0"}},
    {"type": "Physics", "label": "NaturalConv", "icon": "♒", "tip": "Buoyancy enclosure via Nu(Ra) correlations (T-A1 trial)", "data": {"kind": "NaturalConv-v0"}},
    {"type": "Physics", "label": "Telegrapher", "icon": "〜", "tip": "Thermal-wave cancellation control (T-E08 hardening)", "data": {"kind": "Telegrapher-v0"}},
    {"type": "Physics", "label": "TE-Segmented", "icon": "⚡", "tip": "Segmented TE leg grading (T-A2 extension)", "data": {"kind": "Thermoelectric1D-v0"}},
    {"type": "Physics", "label": "Boussinesq", "icon": "🌀", "tip": "Resolved RB cavity CFD (T-A1 extension)", "data": {"kind": "Boussinesq-v0"}},
    # Design & Objective
    {"type": "Objective", "label": "Objective  COP", "icon": "◎", "tip": "Weighted metric to maximize", "data": {"preset": "COP", "weights": {"COP": 1.0}}},
    {"type": "Objective", "label": "Objective  SCP", "icon": "◎", "tip": "Maximize SCP", "data": {"preset": "SCP_W_kg", "weights": {"SCP_W_kg": 1.0}}},
    {"type": "Objective", "label": "Objective  COP+SCP", "icon": "◎", "tip": "Profile-weighted score", "data": {"preset": "COP+SCP", "weights": {"COP": 0.5, "SCP_W_kg": 0.5}, "normalize": {"COP": [0.05, 0.85], "SCP_W_kg": [20.0, 1600.0]}}},
    # Sweep — Design & Objective meta-block
    {"type": "Sweep", "label": "Sweep  t_switch", "icon": "⇄", "tip": "Grid sweep over t_switch: SCP/COP vs switch time via Bed1DControls (uses harness.control.switch_time_sweep)", "data": {"axis": "t_switch", "grid": [80, 180, 300, 480], "budget": 8, "grid_min": 60.0, "grid_max": 600.0, "steps": 6}},
    {"type": "Sweep", "label": "Sweep  materials", "icon": "⇆", "tip": "Grid sweep over materials: ranking per profile via harness.rank.sweep_materials", "data": {"axis": "material", "grid": ["anchor:Silica gel RD", "anchor:Zeolite 13X (NaX)", "anchor:Aluminum fumarate"], "budget": 6, "material_grid": ["anchor:Silica gel RD", "anchor:Zeolite 13X (NaX)"]}},
    # Optimizers
    {"type": "Optimizer", "label": "Search (CMA-ES)", "icon": "⬡", "tip": "Derivative-free CMA-ES / TPE", "data": {"kind": "search", "method": "cmaes", "budget": 200}},
    {"type": "Optimizer", "label": "Search (TPE)", "icon": "⬡", "tip": "Optuna TPE", "data": {"kind": "search", "method": "tpe", "budget": 200}},
    {"type": "Optimizer", "label": "Grad (Adam)", "icon": "⬢", "tip": "JAX grad + optax Adam", "data": {"kind": "grad", "n_starts": 3, "n_steps": 200, "step_size": 0.05}},
    # Scopes
    {"type": "Scope", "label": "Metrics", "icon": "▭", "tip": "Episode metrics card — COP/SCP/Q etc.", "data": {"kind": "Metrics"}},
    {"type": "Scope", "label": "History", "icon": "〰", "tip": "Optimization convergence chart", "data": {"kind": "History"}},
    {"type": "Scope", "label": "Trace", "icon": "▥", "tip": "T(x,t) heatmap + sensors", "data": {"kind": "Trace"}},
    {"type": "Scope", "label": "Ranking", "icon": "☰", "tip": "T2 material ranking table", "data": {"kind": "Ranking"}},
    {"type": "Scope", "label": "Calibration", "icon": "▤", "tip": "Literature parity error table", "data": {"kind": "Calibration"}},
    {"type": "Scope", "label": "Compare", "icon": "⚖", "tip": "Side-by-side OptimizeResult diff — like harness/report.summary side-by-side", "data": {"kind": "Compare"}},
    {"type": "Scope", "label": "Sweep", "icon": "◿", "tip": "Sweep curve: t_switch vs SCP/COP (reuses History logic)", "data": {"kind": "Sweep"}},
]


class PaletteDock(QDockWidget):
    block_requested = pyqtSignal(dict)  # {"type":..., "data":..., "label":...}

    def __init__(self, parent: Any = None) -> None:
        super().__init__("Palette", parent)
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.TopDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.setObjectName("PaletteDock")
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        title = QLabel("Blocks  —  drag or double-click to add")
        title.setStyleSheet("color:#94a3b8; font-size:8.5pt;")
        layout.addWidget(title)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter — e.g.  material  bed1d  cma")
        self.search.textChanged.connect(self._filter)
        layout.addWidget(self.search)

        # Categorized toolbox — collapses to one visible section at a time,
        # drastically reducing initial clutter (22 → ~5 visible rows).
        self.toolbox = QToolBox()
        self.toolbox.setStyleSheet(
            "QToolBox::tab { background: #1b1e24; color: #94a3b8; border-radius: 4px; margin-top: 2px; }"
            "QToolBox::tab:selected { color: #e2e8f0; font-weight: 600; }"
        )
        self._lists: dict[str, QListWidget] = {}
        # Build one page per NODE_CATEGORIES entry in declared order
        for cat, types_in_cat in NODE_CATEGORIES.items():
            lst = QListWidget()
            lst.setDragEnabled(True)
            lst.setAlternatingRowColors(False)
            # compact row height so 5 categories still fit in dock
            lst.setSpacing(1)
            lst.itemDoubleClicked.connect(self._on_double)
            count = sum(1 for b in BLOCKS if b["type"] in types_in_cat)
            # keep physics page a bit taller (8 physics kinds)
            self._lists[cat] = lst
            self.toolbox.addItem(lst, f"{cat}  ({count})")
        # Default: Physics expanded, others collapsed (index 1 = Physics)
        try:
            phys_idx = list(NODE_CATEGORIES.keys()).index("Physics")
            self.toolbox.setCurrentIndex(phys_idx)
        except ValueError:
            self.toolbox.setCurrentIndex(0)
        layout.addWidget(self.toolbox, 1)

        hint = QLabel("Tip: connect Material/Profile → Physics → Objective → Optimizer → Scope.\nRight-drag a port to wire.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#64748b; font-size:7.5pt; background:#111318; border-radius:4px; padding:4px;")
        hint.setFrameStyle(QFrame.Shape.NoFrame)
        layout.addWidget(hint)

        self.setWidget(root)
        self._populate(BLOCKS)

    def _populate(self, blocks: list[dict[str, Any]]) -> None:
        # Bucket blocks by category so each toolbox page only shows its own.
        by_cat: dict[str, list[dict[str, Any]]] = {
            cat: [] for cat in NODE_CATEGORIES
        }
        for b in blocks:
            for cat, types in NODE_CATEGORIES.items():
                if b["type"] in types:
                    by_cat[cat].append(b)
                    break
        for cat, lst in self._lists.items():
            lst.clear()
            for b in by_cat.get(cat, []):
                it = QListWidgetItem(f"{b['icon']}  {b['label']}")
                it.setToolTip(b["tip"])
                it.setData(Qt.ItemDataRole.UserRole, b)
                lst.addItem(it)
        # If filtering left only one non-empty category, auto-expand it
        if len(blocks) != len(BLOCKS):
            for i, cat in enumerate(NODE_CATEGORIES.keys()):
                if by_cat[cat]:
                    self.toolbox.setCurrentIndex(i)
                    break
        # Keep backward compat: self.list alias to the currently visible list
        # (tests / external code may reference palette.list)
        try:
            self.list = self._lists[list(NODE_CATEGORIES.keys())[self.toolbox.currentIndex()]]  # type: ignore[attr-defined]
        except Exception:
            pass

    def _filter(self, text: str) -> None:
        q = text.strip().lower()
        if not q:
            self._populate(BLOCKS)
            return
        filtered = [b for b in BLOCKS if q in b["label"].lower() or q in b["type"].lower() or q in b["tip"].lower()]
        self._populate(filtered)

    def _on_double(self, item: QListWidgetItem) -> None:
        data = item.data(Qt.ItemDataRole.UserRole)
        if data:
            self.block_requested.emit(dict(data))

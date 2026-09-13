"""Inspector dock — contextual property sheet for the selected node.

Dynamic Physics page: shows only the kwargs relevant to the selected
``kind`` (see ``model.PHYSICS_KWARGS``) plus any extra keys in the
node's ``data``. This makes the sheet researcher-editable for every
1D/2D/3D ladder kind, not hardcoded to Bed1D.
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .model import DEFAULT_NODE_DATA, PHYSICS_DEFAULTS, PHYSICS_KINDS, PHYSICS_KWARGS, SCOPE_KINDS

# ---------------------------------------------------------------------------
# Field definitions for Physics dynamic form
# (key -> tuple describing widget). Unknown keys fall back to generic type.
# ---------------------------------------------------------------------------
_FIELD_DEFS: dict[str, tuple] = {
    "n_cells": ("int", 4, 256, 1, ""),
    "nx": ("int", 4, 256, 1, ""),
    "ny": ("int", 4, 256, 1, ""),
    "nz": ("int", 4, 256, 1, ""),
    "n_cycles": ("int", 1, 20, 1, ""),
    "n_steps": ("int", 10, 20000, 10, ""),
    "save_every": ("int", 1, 100, 1, ""),
    "dt": ("float", 1e-5, 0.5, 0.0001, " s", 6),
    "dt_phys_s": ("float", 0.0, 0.5, 0.0001, " s", 6),
    "t_end": ("float", 0.01, 200.0, 0.1, " s", 3),
    "cycle_time_s": ("float", 10.0, 7200.0, 10.0, " s", 1),
    "t_amb_c": ("float", -50.0, 100.0, 1.0, " °C", 1),
    "T_hot": ("float", -50.0, 300.0, 1.0, " °C", 1),
    "T_cold": ("float", -50.0, 300.0, 1.0, " °C", 1),
    "T_init": ("float", -50.0, 300.0, 1.0, " °C", 1),
    "T_core": ("float", -50.0, 300.0, 1.0, " °C", 1),
    "core_radius": ("float", 0.01, 5.0, 0.01, " m", 3),
    "L": ("float", 0.05, 10.0, 0.05, " m", 3),
    "Lx": ("float", 0.1, 10.0, 0.1, " m", 2),
    "Ly": ("float", 0.1, 10.0, 0.1, " m", 2),
    "L_m": ("float", 0.0005, 0.05, 0.0005, " m", 4),
    "r_in": ("float", 0.02, 2.0, 0.01, " m", 3),
    "r_out": ("float", 0.05, 2.0, 0.01, " m", 3),
    "bg": ("float", 0.0001, 1.0, 0.001, "", 4),
    "alpha": ("float", 0.0001, 2.0, 0.001, " m²/s", 5),
    "source_amplitude": ("float", 0.0, 10000.0, 10.0, "", 1),
    "hx_mass_factor": ("float", 1.0, 20.0, 0.05, "", 2),
    "k_eff_w_m_k": ("float", 0.01, 10.0, 0.05, " W/mK", 3),
    "h_wall_w_m2_k": ("float", 5.0, 10000.0, 50.0, " W/m²K", 1),
    "recovery_ua_w_m2_k": ("float", 0.0, 5000.0, 10.0, " W/m²K", 1),
    "dt_ctrl_s": ("float", 0.1, 200.0, 0.5, " s", 2),
    "ua_loss_W_K_per_kg": ("float", 0.0, 20.0, 0.1, "", 3),
    "tau": ("float", 0.001, 10.0, 0.01, " s", 4),
    "soft_switch": ("bool",),
    "mode": ("enum", ["slab", "core"]),
    "material_b": ("str_or_none",),
    "t_rec_s": ("float", 0.0, 200.0, 1.0, " s", 1),
}

_LABELS = {
    "n_cells": "n_cells", "nx": "nx", "ny": "ny", "nz": "nz",
    "n_cycles": "n_cycles", "n_steps": "n_steps", "save_every": "save_every",
    "dt": "dt", "dt_phys_s": "dt_phys_s", "t_end": "t_end",
    "L": "L", "L_m": "L_m (bed)", "Lx": "Lx", "Ly": "Ly",
    "r_in": "r_in", "r_out": "r_out", "bg": "bg", "alpha": "alpha",
    "hx_mass_factor": "hx_mass_factor", "k_eff_w_m_k": "k_eff [W/mK]",
    "h_wall_w_m2_k": "h_wall [W/m²K]", "recovery_ua_w_m2_k": "UA_rec [W/m²K]",
    "T_hot": "T_hot", "T_cold": "T_cold", "T_init": "T_init", "T_core": "T_core",
    "core_radius": "core_radius", "mode": "mode", "tau": "tau",
    "cycle_time_s": "cycle_time_s", "t_amb_c": "t_amb_c",
    "material_b": "material_b (2nd)", "soft_switch": "soft_switch",
    "dt_ctrl_s": "dt_ctrl_s", "source_amplitude": "source_amplitude",
    "ua_loss_W_K_per_kg": "ua_loss", "t_rec_s": "t_rec_s",
}


def _harness_material_names() -> list[str]:
    try:
        from harness.registry import REGISTRIES  # noqa: WPS433
        return sorted(REGISTRIES["materials"].names())
    except Exception:
        return ["anchor:Silica gel RD"]


def _harness_profile_names() -> list[str]:
    try:
        from harness.registry import REGISTRIES  # noqa: WPS433
        return sorted(REGISTRIES["profiles"].names())
    except Exception:
        return ["datacenter", "cpu", "human", "vehicle"]


class InspectorDock(QDockWidget):
    node_changed = pyqtSignal(str, dict)  # node_id, new_data
    delete_requested = pyqtSignal(str)

    def __init__(self, parent: Any = None) -> None:
        super().__init__("Inspector", parent)
        self.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.TopDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.setObjectName("InspectorDock")
        self._current_id: str | None = None
        self._current_type: str | None = None
        self._current_data: dict[str, Any] = {}
        self._updating = False
        # dynamic physics widgets: key -> widget
        self._phys_dynamic_widgets: dict[str, Any] = {}
        self._phys_form: QFormLayout | None = None

        root = QWidget()
        self._root_layout = QVBoxLayout(root)
        self._root_layout.setContentsMargins(6, 6, 6, 6)
        self._root_layout.setSpacing(6)

        self._header = QLabel("No selection")
        self._header.setStyleSheet("color:#e2e8f0; font-weight:600; font-size:10pt;")
        self._root_layout.addWidget(self._header)

        self._sub = QLabel("")
        self._sub.setStyleSheet("color:#94a3b8; font-size:8pt;")
        self._sub.setWordWrap(True)
        self._root_layout.addWidget(self._sub)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color:#1e222a;")
        self._root_layout.addWidget(line)

        # stacked per-type editors
        self._stack = QStackedWidget()
        self._root_layout.addWidget(self._stack, 1)

        self._empty = QLabel("Select a block on the canvas.\nDouble-click a card to edit. Right-click a card → Delete/Duplicate.")
        self._empty.setStyleSheet("color:#64748b; font-size:8.5pt;")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stack.addWidget(self._empty)

        # build per-type pages
        self._pages: dict[str, QWidget] = {}
        for t in ("Material", "Profile", "Physics", "Objective", "Optimizer", "Scope", "Sweep"):
            page = self._build_page(t)
            self._pages[t] = page
            self._stack.addWidget(page)

        # bottom actions
        btn_row = QHBoxLayout()
        self._del_btn = QPushButton("Delete block")
        self._del_btn.setStyleSheet("background:#1e293b; color:#f87171; padding:4px; border-radius:4px;")
        self._del_btn.clicked.connect(self._on_delete)
        btn_row.addWidget(self._del_btn, 1)
        self._dup_btn = QPushButton("Duplicate")
        self._dup_btn.setStyleSheet("background:#1e293b; color:#38bdf8; padding:4px; border-radius:4px;")
        self._dup_btn.clicked.connect(self._on_duplicate)
        btn_row.addWidget(self._dup_btn, 1)
        self._root_layout.addLayout(btn_row)

        self.setWidget(root)
        self._show_empty()

    # -- page builders ------------------------------------------------------

    def _build_page(self, node_type: str) -> QWidget:
        w = QWidget()
        # Use QVBoxLayout for pages that contain scroll areas; otherwise QFormLayout
        if node_type == "Physics":
            layout = QVBoxLayout(w)
            layout.setContentsMargins(2, 2, 2, 2)
            layout.setSpacing(6)
            # kind row (fixed) — keep in a form for label alignment
            top_form = QFormLayout()
            top_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
            top_form.setSpacing(6)
            self._phys_kind = QComboBox()
            self._phys_kind.addItems(PHYSICS_KINDS)
            self._phys_kind.currentTextChanged.connect(self._on_phys_kind_changed)
            top_form.addRow("kind", self._phys_kind)
            layout.addLayout(top_form)
            # scrollable dynamic fields
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setStyleSheet("QScrollArea { background: transparent; }")
            container = QWidget()
            self._phys_form = QFormLayout(container)
            self._phys_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
            self._phys_form.setContentsMargins(2, 2, 2, 2)
            self._phys_form.setSpacing(5)
            scroll.setWidget(container)
            layout.addWidget(scroll, 1)
            # raw data fallback expander
            raw_row = QHBoxLayout()
            self._phys_raw_edit = QLineEdit()
            self._phys_raw_edit.setPlaceholderText('Add field — e.g.  alpha=0.02  or  custom_key="hello"')
            self._phys_raw_edit.setStyleSheet("font-family: monospace; font-size: 7.5pt;")
            # trigger on Return
            self._phys_raw_edit.returnPressed.connect(self._on_phys_raw_add)
            raw_add = QPushButton("Add")
            raw_add.setFixedWidth(48)
            raw_add.clicked.connect(self._on_phys_raw_add)
            raw_row.addWidget(self._phys_raw_edit, 1)
            raw_row.addWidget(raw_add)
            layout.addLayout(raw_row)
            hint = QLabel("Tip: kind switch rebuilds the form. Raw “Add” lets you inject any research param (e.g. mode=core).")
            hint.setStyleSheet("color:#64748b; font-size:7pt;")
            hint.setWordWrap(True)
            layout.addWidget(hint)
            return w

        form = QFormLayout(w)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setContentsMargins(2, 2, 2, 2)
        form.setSpacing(6)

        if node_type == "Material":
            self._mat_combo = QComboBox()
            self._mat_combo.setEditable(True)
            self._mat_combo.currentTextChanged.connect(lambda v: self._commit({"ref": v.strip()}))
            # allow Enter to confirm free-typed refs (research custom materials)
            self._mat_combo.lineEdit().returnPressed.connect(lambda: self._commit({"ref": self._mat_combo.currentText().strip()}))
            form.addRow("ref", self._mat_combo)
            self._mat_info = QLabel("—")
            self._mat_info.setStyleSheet("color:#94a3b8; font-size:7.5pt;")
            self._mat_info.setWordWrap(True)
            form.addRow("", self._mat_info)
            # research hint
            hint = QLabel("Editable combo — type any ref (anchor:…, custom:…) and press Enter.")
            hint.setStyleSheet("color:#64748b; font-size:7pt;")
            hint.setWordWrap(True)
            form.addRow("", hint)

        elif node_type == "Profile":
            self._prof_combo = QComboBox()
            self._prof_combo.setEditable(True)
            self._prof_combo.currentTextChanged.connect(lambda v: self._commit({"ref": v.strip()}))
            self._prof_combo.lineEdit().returnPressed.connect(lambda: self._commit({"ref": self._prof_combo.currentText().strip()}))
            form.addRow("ref", self._prof_combo)
            self._prof_info = QLabel("—")
            self._prof_info.setStyleSheet("color:#94a3b8; font-size:7.5pt;")
            self._prof_info.setWordWrap(True)
            form.addRow("", self._prof_info)

        elif node_type == "Objective":
            self._obj_preset = QComboBox()
            self._obj_preset.addItems(["COP", "SCP_W_kg", "COP+SCP", "custom"])
            self._obj_preset.currentTextChanged.connect(self._on_obj_preset)
            form.addRow("preset", self._obj_preset)
            self._obj_w_cop = QDoubleSpinBox()
            self._obj_w_cop.setRange(-5.0, 5.0)
            self._obj_w_cop.setSingleStep(0.1)
            self._obj_w_cop.valueChanged.connect(self._on_obj_weights)
            form.addRow("w COP", self._obj_w_cop)
            self._obj_w_scp = QDoubleSpinBox()
            self._obj_w_scp.setRange(-5.0, 5.0)
            self._obj_w_scp.setSingleStep(0.1)
            self._obj_w_scp.valueChanged.connect(self._on_obj_weights)
            form.addRow("w SCP", self._obj_w_scp)
            self._obj_norm = QCheckBox("normalize (0.05–0.85 COP, 20–1600 SCP)")
            self._obj_norm.toggled.connect(self._on_obj_norm)
            form.addRow("", self._obj_norm)

        elif node_type == "Optimizer":
            self._opt_kind = QComboBox()
            self._opt_kind.addItems(["search", "grad"])
            self._opt_kind.currentTextChanged.connect(self._on_opt_kind)
            form.addRow("backend", self._opt_kind)
            self._opt_method = QComboBox()
            self._opt_method.addItems(["cmaes", "tpe"])
            self._opt_method.currentTextChanged.connect(lambda v: self._commit({"method": v}))
            form.addRow("method", self._opt_method)
            self._opt_budget = QSpinBox()
            self._opt_budget.setRange(10, 10000)
            self._opt_budget.setSingleStep(10)
            self._opt_budget.valueChanged.connect(lambda v: self._commit({"budget": int(v)}))
            form.addRow("budget", self._opt_budget)
            self._opt_starts = QSpinBox()
            self._opt_starts.setRange(1, 16)
            self._opt_starts.valueChanged.connect(lambda v: self._commit({"n_starts": int(v)}))
            form.addRow("n_starts (grad)", self._opt_starts)
            self._opt_steps = QSpinBox()
            self._opt_steps.setRange(10, 5000)
            self._opt_steps.setSingleStep(10)
            self._opt_steps.valueChanged.connect(lambda v: self._commit({"n_steps": int(v)}))
            form.addRow("n_steps (grad)", self._opt_steps)
            self._opt_lr = QDoubleSpinBox()
            self._opt_lr.setRange(1e-4, 1.0)
            self._opt_lr.setSingleStep(0.01)
            self._opt_lr.setDecimals(4)
            self._opt_lr.valueChanged.connect(lambda v: self._commit({"step_size": float(v)}))
            form.addRow("step_size", self._opt_lr)

        elif node_type == "Scope":
            self._scope_kind = QComboBox()
            self._scope_kind.addItems(SCOPE_KINDS)
            self._scope_kind.currentTextChanged.connect(lambda v: self._on_scope_kind(v))
            form.addRow("kind", self._scope_kind)
            self._scope_hint = QLabel("")
            self._scope_hint.setStyleSheet("color:#94a3b8; font-size:7.5pt;")
            self._scope_hint.setWordWrap(True)
            form.addRow("", self._scope_hint)

        elif node_type == "Sweep":
            self._sweep_axis = QComboBox()
            self._sweep_axis.addItems(["t_switch", "material"])
            self._sweep_axis.currentTextChanged.connect(self._on_sweep_axis)
            form.addRow("axis", self._sweep_axis)
            self._sweep_min = QDoubleSpinBox()
            self._sweep_min.setRange(10.0, 3600.0)
            self._sweep_min.setSingleStep(10.0)
            self._sweep_min.setSuffix(" s")
            self._sweep_min.valueChanged.connect(self._on_sweep_grid)
            form.addRow("grid min", self._sweep_min)
            self._sweep_max = QDoubleSpinBox()
            self._sweep_max.setRange(20.0, 7200.0)
            self._sweep_max.setSingleStep(10.0)
            self._sweep_max.setSuffix(" s")
            self._sweep_max.valueChanged.connect(self._on_sweep_grid)
            form.addRow("grid max", self._sweep_max)
            self._sweep_steps = QSpinBox()
            self._sweep_steps.setRange(2, 50)
            self._sweep_steps.valueChanged.connect(self._on_sweep_grid)
            form.addRow("steps", self._sweep_steps)
            self._sweep_budget = QSpinBox()
            self._sweep_budget.setRange(1, 200)
            self._sweep_budget.setSingleStep(1)
            self._sweep_budget.valueChanged.connect(lambda v: self._commit({"budget": int(v)}))
            form.addRow("budget", self._sweep_budget)
            self._sweep_mat_grid = QLineEdit()
            self._sweep_mat_grid.setPlaceholderText("anchor:Silica gel RD, anchor:zeolite 13X, ...")
            self._sweep_mat_grid.editingFinished.connect(self._on_sweep_mat_grid)
            form.addRow("materials", self._sweep_mat_grid)
            self._sweep_preview = QLabel("—")
            self._sweep_preview.setStyleSheet("color:#94a3b8; font-size:7.5pt;")
            self._sweep_preview.setWordWrap(True)
            form.addRow("preview", self._sweep_preview)

        return w

    # -- dynamic Physics helpers --------------------------------------------

    def _clear_phys_form(self) -> None:
        if self._phys_form is None:
            return
        while self._phys_form.rowCount() > 0:
            self._phys_form.removeRow(0)
        for w in list(self._phys_dynamic_widgets.values()):
            try:
                w.deleteLater()
            except Exception:
                pass
        self._phys_dynamic_widgets.clear()

    def _create_widget_for_key(self, key: str, value: Any):
        defn = _FIELD_DEFS.get(key)
        # enum
        if defn is not None and defn[0] == "enum":
            cb = QComboBox()
            cb.setEditable(False)
            opts = defn[1]
            cb.addItems([str(o) for o in opts])
            cur = str(value) if value is not None else str(opts[0])
            # block signals until caller sets up
            if cur in opts:
                cb.setCurrentText(cur)
            else:
                cb.setCurrentText(str(opts[0]))
            # capture key
            cb.currentTextChanged.connect(lambda v, k=key: self._commit({k: v}))
            return cb
        if defn is not None and defn[0] == "bool":
            cb = QCheckBox()
            cb.setChecked(bool(value))
            # label will be key itself
            cb.toggled.connect(lambda v, k=key: self._commit({k: bool(v)}))
            return cb
        if defn is not None and defn[0] == "str_or_none":
            le = QLineEdit()
            le.setPlaceholderText("none (empty) or material ref")
            le.setText("" if value is None else str(value))
            le.editingFinished.connect(lambda k=key, w=le: self._commit({k: (w.text().strip() or None)}))
            return le
        if isinstance(value, bool):
            cb = QCheckBox()
            cb.setChecked(bool(value))
            cb.toggled.connect(lambda v, k=key: self._commit({k: bool(v)}))
            return cb
        if isinstance(value, int) and not isinstance(value, bool):
            if defn is not None and defn[0] == "int":
                _, lo, hi, step, _ = defn
            else:
                lo, hi, step = 0, 100000, 1
            sb = QSpinBox()
            sb.setRange(int(lo), int(hi))
            sb.setSingleStep(int(step))
            # clamp
            v = int(value) if value is not None else int(lo)
            v = max(int(lo), min(int(hi), v))
            sb.setValue(v)
            sb.valueChanged.connect(lambda v, k=key: self._commit({k: int(v)}))
            return sb
        if isinstance(value, float) or (defn is not None and defn[0] == "float"):
            if defn is not None and defn[0] == "float":
                _, lo, hi, step, suffix, dec = defn
            else:
                lo, hi, step, suffix, dec = -1e6, 1e6, 0.01, "", 4
            dsb = QDoubleSpinBox()
            dsb.setRange(float(lo), float(hi))
            dsb.setSingleStep(float(step))
            dsb.setDecimals(int(dec))
            if suffix:
                dsb.setSuffix(suffix)
            v = float(value) if value is not None else float(lo)
            # clamp
            v = max(float(lo), min(float(hi), v))
            dsb.setValue(v)
            dsb.valueChanged.connect(lambda v, k=key: self._commit({k: float(v)}))
            return dsb
        # fallback string
        le = QLineEdit()
        le.setText(str(value) if value is not None else "")
        le.editingFinished.connect(lambda k=key, w=le: self._on_phys_lineedit_commit(k, w.text()))
        return le

    def _on_phys_lineedit_commit(self, key: str, text: str) -> None:
        if self._updating or self._current_id is None:
            return
        txt = text.strip()
        if txt == "":
            self.node_changed.emit(self._current_id, {key: None})
            return
        # try to infer type: int -> float -> str, or keep raw if quoted
        if (txt.startswith('"') and txt.endswith('"')) or (txt.startswith("'") and txt.endswith("'")):
            self.node_changed.emit(self._current_id, {key: txt[1:-1]})
            return
        try:
            if "." not in txt and "e" not in txt.lower():
                self.node_changed.emit(self._current_id, {key: int(txt)})
                return
        except Exception:
            pass
        try:
            self.node_changed.emit(self._current_id, {key: float(txt)})
            return
        except Exception:
            pass
        self.node_changed.emit(self._current_id, {key: txt})

    def _rebuild_physics_form(self, kind: str, data: dict[str, Any]) -> None:
        if self._phys_form is None:
            return
        self._clear_phys_form()
        allowed = set(PHYSICS_KWARGS.get(kind, set()))
        # For research-clutter control: only show allowed keys for this kind
        # plus truly custom keys (not part of any physics kind's known set).
        # Stale defaults from other kinds (e.g. L for Cycle0D) stay hidden.
        all_known = set().union(*PHYSICS_KWARGS.values()) if PHYSICS_KWARGS else set()
        all_known |= set(DEFAULT_NODE_DATA.get("Physics", {}).keys())
        custom_keys = {k for k in data.keys() if k != "kind" and k not in allowed and k not in all_known}
        # also keep allowed keys that have custom values in data (e.g. user set nx for ForcedConv)
        # but don't pull in stale keys that are known for other kinds.
        all_keys = set(allowed) | custom_keys
        # Stable order: allowed sorted, then custom sorted
        ordered = []
        for k in sorted(allowed):
            ordered.append(k)
        for k in sorted(custom_keys):
            ordered.append(k)
        # Fill defaults for missing allowed keys from per-kind defaults
        defaults = PHYSICS_DEFAULTS.get(kind, {})
        for key in ordered:
            val = data.get(key, defaults.get(key))
            # If still None and field is numeric, keep None for widget to show
            if val is None and key in ("dt_phys_s", "material_b"):
                # keep None
                pass
            elif val is None:
                # use a sensible default: 0 for numeric, "" for str
                defn = _FIELD_DEFS.get(key)
                if defn and defn[0] in ("int", "float"):
                    val = defn[1]  # lo
                else:
                    val = ""
            label = _LABELS.get(key, key)
            # bool needs special handling: checkbox label is key itself, form label empty?
            widget = self._create_widget_for_key(key, val)
            # for bool, show checkbox with label and no form label
            if isinstance(widget, QCheckBox):
                # For bool, put checkbox as field widget with label as key
                if key == "soft_switch":
                    widget.setText("soft_switch (substep-split)")
                    self._phys_form.addRow("", widget)
                else:
                    widget.setText(label)
                    self._phys_form.addRow("", widget)
            else:
                # add row with label and widget, plus small remove button for custom keys
                if key not in allowed:
                    # custom field — add with remove button
                    row_w = QWidget()
                    hl = QHBoxLayout(row_w)
                    hl.setContentsMargins(0, 0, 0, 0)
                    hl.addWidget(widget, 1)
                    rm = QPushButton("×")
                    rm.setFixedSize(18, 18)
                    rm.setStyleSheet("QPushButton { background: #1e293b; color: #f87171; border-radius: 9px; font-size: 9pt; padding: 0; }")
                    rm.setToolTip("Remove this field from node data")
                    rm.clicked.connect(lambda _=False, k=key: self._commit_remove_key(k))
                    hl.addWidget(rm)
                    self._phys_form.addRow(label, row_w)
                else:
                    self._phys_form.addRow(label, widget)
            self._phys_dynamic_widgets[key] = widget

    def _commit_remove_key(self, key: str) -> None:
        if self._current_id is None:
            return
        # Emit a patch that removes key by setting to None and letting model filter,
        # but model keeps None — we actually need to delete. For now set to None
        # and rebuild will still show it; better to emit a special signal. Instead,
        # we patch by sending a dict with key→None and also inform main window to
        # delete from graph node data directly via node_changed with sentinel.
        # Simplest: emit patch with key=None and let model handler delete if None?
        # We'll emit as usual and let app handle removal: app will do data.pop.
        # Use a sentinel: emit with value "__DELETE__" and handle in app.
        self.node_changed.emit(self._current_id, {key: None})
        # Also force rebuild on next show_node via data update; for immediate UI,
        # remove widget
        if key in self._phys_dynamic_widgets:
            # will be cleared on next rebuild; just remove row visually
            pass

    def _on_phys_kind_changed(self, kind: str) -> None:
        if self._updating or self._current_id is None:
            return
        # Commit kind first
        self.node_changed.emit(self._current_id, {"kind": kind})
        # Update local copy for immediate rebuild (research-fit: see new fields instantly)
        if self._current_data is not None:
            self._current_data = dict(self._current_data)
            self._current_data["kind"] = kind
            # ensure defaults for new kind are visible
            defaults = PHYSICS_DEFAULTS.get(kind, {})
            allowed = PHYSICS_KWARGS.get(kind, set())
            for k in allowed:
                if k not in self._current_data or self._current_data[k] is None:
                    if k in defaults:
                        self._current_data[k] = defaults[k]
                    elif k in DEFAULT_NODE_DATA.get("Physics", {}):
                        self._current_data[k] = DEFAULT_NODE_DATA["Physics"][k]
            self._rebuild_physics_form(kind, self._current_data)

    def _on_phys_raw_add(self) -> None:
        txt = self._phys_raw_edit.text().strip()
        if not txt or self._current_id is None:
            return
        # parse "key=value" or "key = value"
        if "=" not in txt:
            return
        k, vtxt = [s.strip() for s in txt.split("=", 1)]
        if not k:
            return
        # reuse lineedit commit logic
        self._on_phys_lineedit_commit(k, vtxt)
        self._phys_raw_edit.clear()
        # inject into current data for immediate rebuild
        try:
            # after commit, the node's data will be updated via main window;
            # for instant feedback, also update local and rebuild
            if self._current_data is not None:
                # infer value as before
                v = vtxt
                try:
                    if vtxt.startswith('"') and vtxt.endswith('"'):
                        v = vtxt[1:-1]
                    elif "." not in vtxt and "e" not in vtxt.lower():
                        v = int(vtxt)
                    else:
                        v = float(vtxt)
                except Exception:
                    pass
                self._current_data[k] = v
                kind = self._current_data.get("kind", "Cycle0D-v0")
                self._rebuild_physics_form(kind, self._current_data)
        except Exception:
            pass

    # -- public API ---------------------------------------------------------

    def show_node(self, node_id: str, node_type: str, data: dict[str, Any]) -> None:
        self._current_id = node_id
        self._current_type = node_type
        self._current_data = dict(data)
        self._updating = True
        try:
            self._header.setText(f"{node_type}  ·  {node_id[:10]}")
            self._sub.setText("")
            # populate per type
            if node_type == "Material":
                self._refresh_mat_combo()
                self._mat_combo.setCurrentText(str(data.get("ref", "")))
                self._mat_info.setText(self._material_detail(str(data.get("ref", ""))))
            elif node_type == "Profile":
                self._refresh_prof_combo()
                self._prof_combo.setCurrentText(str(data.get("ref", "")))
                self._prof_info.setText(self._profile_detail(str(data.get("ref", ""))))
            elif node_type == "Physics":
                self._phys_kind.blockSignals(True)
                self._phys_kind.setCurrentText(str(data.get("kind", "Cycle0D-v0")))
                self._phys_kind.blockSignals(False)
                self._rebuild_physics_form(str(data.get("kind", "Cycle0D-v0")), data)
            elif node_type == "Objective":
                preset = str(data.get("preset", "COP"))
                self._obj_preset.setCurrentText(preset if preset in ("COP", "SCP_W_kg", "COP+SCP", "custom") else "custom")
                w = data.get("weights", {})
                self._obj_w_cop.setValue(float(w.get("COP", 0.0)))
                self._obj_w_scp.setValue(float(w.get("SCP_W_kg", 0.0)))
                has_norm = bool(data.get("normalize"))
                self._obj_norm.setChecked(has_norm)
            elif node_type == "Optimizer":
                kind = str(data.get("kind", "search"))
                self._opt_kind.setCurrentText(kind if kind in ("search", "grad") else "search")
                self._opt_method.setCurrentText(str(data.get("method", "cmaes")))
                self._opt_budget.setValue(int(data.get("budget", 200)))
                self._opt_starts.setValue(int(data.get("n_starts", 3)))
                self._opt_steps.setValue(int(data.get("n_steps", 400)))
                self._opt_lr.setValue(float(data.get("step_size", 0.05)))
                self._on_opt_kind(kind)
            elif node_type == "Scope":
                self._scope_kind.setCurrentText(str(data.get("kind", "Metrics")))
                self._update_scope_hint(str(data.get("kind", "Metrics")))
            elif node_type == "Sweep":
                axis = str(data.get("axis", "t_switch"))
                self._sweep_axis.setCurrentText(axis if axis in ("t_switch", "material") else "t_switch")
                # grid handling: prefer grid_min/max/steps if present, else derive from grid
                grid = data.get("grid", [])
                gmin = data.get("grid_min", 60.0)
                gmax = data.get("grid_max", 600.0)
                steps = data.get("steps", len(grid) if grid else 6)
                # if grid is provided for t_switch, derive min/max from it when not explicitly stored
                if axis == "t_switch" and grid and isinstance(grid, (list, tuple)) and len(grid) >= 2:
                    try:
                        vals = [float(v) for v in grid]  # type: ignore
                        gmin = min(vals)
                        gmax = max(vals)
                        steps = len(vals)
                    except Exception:
                        pass
                self._sweep_min.setValue(float(gmin))
                self._sweep_max.setValue(float(gmax))
                self._sweep_steps.setValue(int(steps))
                self._sweep_budget.setValue(int(data.get("budget", 8)))
                # material grid
                mat_grid = data.get("material_grid", data.get("grid") if axis == "material" else "")
                if isinstance(mat_grid, (list, tuple)):
                    self._sweep_mat_grid.setText(", ".join(str(x) for x in mat_grid))
                else:
                    self._sweep_mat_grid.setText(str(mat_grid) if mat_grid else "")
                self._update_sweep_preview()
                self._update_sweep_enabled()
            self._stack.setCurrentWidget(self._pages[node_type])
        finally:
            self._updating = False

    def _show_empty(self) -> None:
        self._current_id = None
        self._current_type = None
        self._current_data = {}
        self._header.setText("No selection")
        self._sub.setText("Select a block on the canvas to edit its properties. Right-click a card for Delete/Duplicate.")
        self._stack.setCurrentWidget(self._empty)

    def clear_selection(self) -> None:
        self._show_empty()

    # -- commit helpers -----------------------------------------------------

    def _commit(self, patch: dict[str, Any]) -> None:
        if self._updating or self._current_id is None:
            return
        # keep local copy in sync for immediate rebuild
        if self._current_data is not None:
            self._current_data.update(patch)
        try:
            self.node_changed.emit(self._current_id, patch)
        except RuntimeError:
            pass

    def _on_delete(self) -> None:
        if self._current_id:
            self.delete_requested.emit(self._current_id)

    def _on_duplicate(self) -> None:
        if self._current_id and self._current_type:
            # duplicate via signal? Instead emit a patch that main window can handle
            # For now, main window will handle duplicate via separate signal; we reuse
            # node_changed with a sentinel. Better to add a new signal, but we can
            # directly call parent's handler if available.
            parent = self.parent()
            # Try to find MainWindow duplicate handler
            try:
                if hasattr(parent, "_on_duplicate_node"):
                    parent._on_duplicate_node(self._current_id)  # type: ignore[attr-defined]
                    return
            except Exception:
                pass
            # fallback: just emit delete+add via copy? No-op
            pass

    def _on_obj_preset(self, preset: str) -> None:
        if self._updating:
            return
        if preset == "COP":
            self._commit({"preset": preset, "weights": {"COP": 1.0}, "normalize": {}})
        elif preset == "SCP_W_kg":
            self._commit({"preset": preset, "weights": {"SCP_W_kg": 1.0}, "normalize": {}})
        elif preset == "COP+SCP":
            self._commit({"preset": preset, "weights": {"COP": 0.5, "SCP_W_kg": 0.5},
                          "normalize": {"COP": [0.05, 0.85], "SCP_W_kg": [20.0, 1600.0]}})
        else:
            self._commit({"preset": preset})

    def _on_obj_weights(self, _v: float) -> None:
        if self._updating:
            return
        self._commit({"weights": {"COP": float(self._obj_w_cop.value()), "SCP_W_kg": float(self._obj_w_scp.value())}})

    def _on_obj_norm(self, checked: bool) -> None:
        if self._updating:
            return
        if checked:
            self._commit({"normalize": {"COP": [0.05, 0.85], "SCP_W_kg": [20.0, 1600.0]}})
        else:
            self._commit({"normalize": {}})

    def _on_opt_kind(self, kind: str) -> None:
        # enable/disable method vs grad fields
        is_search = kind == "search"
        self._opt_method.setEnabled(is_search)
        self._opt_budget.setEnabled(is_search)
        self._opt_starts.setEnabled(not is_search)
        self._opt_steps.setEnabled(not is_search)
        self._opt_lr.setEnabled(not is_search)
        if not self._updating:
            self._commit({"kind": kind})

    def _on_scope_kind(self, kind: str) -> None:
        self._update_scope_hint(kind)
        if not self._updating:
            self._commit({"kind": kind})

    def _update_scope_hint(self, kind: str) -> None:
        hints = {
            "Metrics": "Single-run metrics table.",
            "History": "Optimization convergence (evals vs objective).",
            "Trace": "T(x,t) heatmap + sensors for dynamic runs.",
            "Ranking": "T2 material ranking table per profile.",
            "Calibration": "Literature parity error table.",
            "Compare": "Side-by-side diff of two OptimizeResults — shows Δ and Δ% per metric.",
            "Sweep": "Sweep curve table + plot: t_switch vs SCP/COP.",
            "System": "Generic system info.",
        }
        if hasattr(self, "_scope_hint"):
            self._scope_hint.setText(hints.get(kind, ""))

    def _on_sweep_axis(self, axis: str) -> None:
        self._update_sweep_enabled()
        self._update_sweep_preview()
        if not self._updating:
            self._commit({"axis": axis})

    def _on_sweep_grid(self, _v: float = 0) -> None:
        if self._updating:
            return
        gmin = float(self._sweep_min.value())
        gmax = float(self._sweep_max.value())
        steps = int(self._sweep_steps.value())
        try:
            import numpy as np  # noqa: WPS433
            grid = np.linspace(gmin, gmax, steps).tolist()
        except Exception:
            grid = [gmin, gmax]
        self._update_sweep_preview()
        self._commit({"grid_min": gmin, "grid_max": gmax, "steps": steps, "grid": grid})

    def _on_sweep_mat_grid(self) -> None:
        if self._updating:
            return
        txt = self._sweep_mat_grid.text().strip()
        if not txt:
            return
        items = [s.strip() for s in txt.split(",") if s.strip()]
        self._commit({"material_grid": items, "grid": items})

    def _update_sweep_preview(self) -> None:
        if not hasattr(self, "_sweep_preview"):
            return
        axis = self._sweep_axis.currentText() if hasattr(self, "_sweep_axis") else "t_switch"
        if axis == "t_switch":
            gmin = float(self._sweep_min.value()) if hasattr(self, "_sweep_min") else 60
            gmax = float(self._sweep_max.value()) if hasattr(self, "_sweep_max") else 600
            steps = int(self._sweep_steps.value()) if hasattr(self, "_sweep_steps") else 6
            txt = f"{steps} points  {gmin:g} → {gmax:g} s  (~{(gmax-gmin)/max(steps-1,1):.1f}s step)"
            self._sweep_preview.setText(txt)
        else:
            txt = self._sweep_mat_grid.text().strip() if hasattr(self, "_sweep_mat_grid") else ""
            n = len([s for s in txt.split(",") if s.strip()]) if txt else 0
            self._sweep_preview.setText(f"{n} materials — {txt[:80]}")

    def _update_sweep_enabled(self) -> None:
        axis = self._sweep_axis.currentText() if hasattr(self, "_sweep_axis") else "t_switch"
        is_t = axis == "t_switch"
        for w in (getattr(self, "_sweep_min", None), getattr(self, "_sweep_max", None), getattr(self, "_sweep_steps", None)):
            if w is not None:
                w.setEnabled(is_t)
        if hasattr(self, "_sweep_mat_grid"):
            self._sweep_mat_grid.setEnabled(not is_t)

    # -- registry helpers ---------------------------------------------------

    def _refresh_mat_combo(self) -> None:
        names = _harness_material_names()
        cur = self._mat_combo.currentText()
        self._mat_combo.blockSignals(True)
        self._mat_combo.clear()
        self._mat_combo.addItems(names)
        self._mat_combo.setCurrentText(cur)
        self._mat_combo.blockSignals(False)

    def _refresh_prof_combo(self) -> None:
        names = _harness_profile_names()
        cur = self._prof_combo.currentText()
        self._prof_combo.blockSignals(True)
        self._prof_combo.clear()
        self._prof_combo.addItems(names)
        self._prof_combo.setCurrentText(cur)
        self._prof_combo.blockSignals(False)

    def _material_detail(self, ref: str) -> str:
        try:
            from harness.materials import get_material  # noqa: WPS433
            m = get_material(ref)
            prov = getattr(m, "transport_provenance", "")
            return f"{m.key} · q_sat={m.q_sat_kg_kg:.3f}  Q_st={m.q_st_j_kg/1e6:.2f} MJ/kg  E={m.e_char_j_mol:.0f} n={m.n_da:.2f}  prov={prov}"
        except Exception as e:  # noqa: BLE001
            return f"unknown material — {e}  (tip: type a new ref and press Enter — it will be stored as custom)"

    def _profile_detail(self, ref: str) -> str:
        try:
            from harness.profiles import get_profile  # noqa: WPS433
            p = get_profile(ref)
            return f"{p.name}: evap {p.t_evap_c:g}°C  cond {p.t_cond_c:g}°C  des {p.t_des_c:g}°C  cycle {p.cycle_time_s:g}s"
        except Exception as e:  # noqa: BLE001
            return f"unknown profile — {e}"

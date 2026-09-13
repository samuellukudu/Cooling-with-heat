"""Wolfram-style command palette — QLineEdit + QCompleter + fuzzy parser."""

from __future__ import annotations

import re
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QCompleter, QDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout

from .model import ExperimentGraph, PHYSICS_KINDS


COMPLETIONS = [
    "material silica gel",
    "material 13x",
    "profile datacenter",
    "profile human",
    "profile vehicle",
    "profile cpu",
    "physics Cycle0D-v0",
    "physics Bed1D-v0",
    "physics TwoBed-v0",
    "physics ForcedConv-v0",
    "physics Cloak2D-v0",
    "physics AbsorptionCycle-v0",
    "physics Thermoelectric-v0",
    "physics NaturalConv-v0",
    "physics Telegrapher-v0",
    "physics Thermoelectric1D-v0",
    "physics Boussinesq-v0",
    "segmented leg grading",
    "peltier seebeck",
    "nusselt rayleigh",
    "absorption libr",
    "optimize for COP",
    "optimize for SCP",
    "rank datacenter",
    "calibrate",
    "search cmaes",
    "search tpe",
    "grad",
    "trace",
    "metrics",
    "history",
]


def parse_command(text: str) -> dict[str, Any]:
    """Very small regex/fuzzy parser — returns a graph patch dict.

    Patch keys: {"material": str, "profile": str, "physics": str,
                 "objective": str, "optimizer": str, "scope": str}
    Missing keys mean no change for that facet.
    """
    t = text.strip().lower()
    patch: dict[str, Any] = {}
    # material
    if "silica" in t or "rd" in t:
        patch["material"] = "anchor:Silica gel RD"
    elif "13x" in t or "13 x" in t:
        patch["material"] = "anchor:zeolite 13X"
    elif "mof" in t:
        # try to find a material name literally
        m = re.search(r"material\s+([a-z0-9 _\-\:]+)", t)
        if m:
            patch["material"] = m.group(1).strip()
    # profile
    for p in ("datacenter", "human", "vehicle", "cpu", "datacenter_dynamic"):
        if p in t:
            patch["profile"] = p
            break
    # physics
    if "segment" in t or ("graded" in t or "grading" in t):
        patch["physics"] = "Thermoelectric1D-v0"
    elif "boussinesq" in t or ("resolved" in t and "conv" in t) or "cavity" in t:
        patch["physics"] = "Boussinesq-v0"
    elif "thermoelec" in t or "peltier" in t or "seebeck" in t:
        patch["physics"] = "Thermoelectric-v0"
    elif "natural" in t and "conv" in t or "nusselt" in t or "rayleigh" in t or "boussinesq" in t:
        patch["physics"] = "NaturalConv-v0"
    elif "telegraph" in t or "second sound" in t or "cancellation" in t:
        patch["physics"] = "Telegrapher-v0"
    elif "absorp" in t or "libr" in t or "nh3" in t:
        patch["physics"] = "AbsorptionCycle-v0"
    elif "forcedconv" in t or "forced conv" in t or "channel" in t:
        patch["physics"] = "ForcedConv-v0"
    elif "cloak" in t:
        patch["physics"] = "Cloak2D-v0"
    elif "twobed" in t or "two bed" in t or "two-bed" in t:
        patch["physics"] = "TwoBed-v0"
    elif "bed1d" in t or "bed 1d" in t or "1d" in t:
        patch["physics"] = "Bed1D-v0"
    elif "cycle0d" in t or "cycle" in t or "oracle" in t:
        patch["physics"] = "Cycle0D-v0"
    # objective
    if "scp" in t and "cop" in t:
        patch["objective"] = "COP+SCP"
    elif "scp" in t:
        patch["objective"] = "SCP_W_kg"
    elif "cop" in t:
        patch["objective"] = "COP"
    # optimizer
    if "cma" in t or "cmaes" in t:
        patch["optimizer"] = "search:cmaes"
    elif "tpe" in t or "optuna" in t:
        patch["optimizer"] = "search:tpe"
    elif "grad" in t or "adam" in t:
        patch["optimizer"] = "grad"
    # scope
    if "trace" in t or "heatmap" in t:
        patch["scope"] = "Trace"
    elif "history" in t or "converg" in t:
        patch["scope"] = "History"
    elif "rank" in t:
        patch["scope"] = "Ranking"
    elif "calibr" in t:
        patch["scope"] = "Calibration"
    return patch


def apply_patch(graph: ExperimentGraph, patch: dict[str, Any]) -> list[str]:
    """Mutate graph in place from patch; return human-readable assumption lines."""
    assumptions: list[str] = []
    if "material" in patch:
        for n in graph.nodes:
            if n.type == "Material":
                n.data["ref"] = patch["material"]
                assumptions.append(f"Material → {patch['material']}")
                break
        else:
            m = graph.add_node("Material", x=40, y=80, data={"ref": patch["material"]})
            # wire to first physics if missing
            phys = graph.physics_node()
            if phys:
                # add edge if not already
                if not any(e.from_node == m.id for e in graph.edges):
                    graph.add_edge(m.id, "material", phys.id, "material")
            assumptions.append(f"Added Material {patch['material']}")
    if "profile" in patch:
        for n in graph.nodes:
            if n.type == "Profile":
                n.data["ref"] = patch["profile"]
                assumptions.append(f"Profile → {patch['profile']}")
                break
        else:
            p = graph.add_node("Profile", x=40, y=200, data={"ref": patch["profile"]})
            phys = graph.physics_node()
            if phys and not any(e.from_node == p.id for e in graph.edges):
                graph.add_edge(p.id, "profile", phys.id, "profile")
            assumptions.append(f"Added Profile {patch['profile']}")
    if "physics" in patch:
        phys = graph.physics_node()
        if phys:
            phys.data["kind"] = patch["physics"]
            assumptions.append(f"Physics → {patch['physics']}")
        else:
            graph.add_node("Physics", x=260, y=140, data={"kind": patch["physics"]})
            assumptions.append(f"Added Physics {patch['physics']}")
    if "objective" in patch:
        obj = graph.objective_node()
        preset = patch["objective"]
        weights = {"COP": 1.0} if preset == "COP" else {"SCP_W_kg": 1.0} if preset == "SCP_W_kg" else {"COP": 0.5, "SCP_W_kg": 0.5}
        norm = {"COP": [0.05, 0.85], "SCP_W_kg": [20.0, 1600.0]} if preset == "COP+SCP" else {}
        if obj:
            obj.data["preset"] = preset
            obj.data["weights"] = weights
            obj.data["normalize"] = norm
            assumptions.append(f"Objective → {preset}")
        else:
            graph.add_node("Objective", x=480, y=140, data={"preset": preset, "weights": weights, "normalize": norm})
            assumptions.append(f"Added Objective {preset}")
    if "optimizer" in patch:
        opt = graph.optimizer_node()
        val = patch["optimizer"]
        if val.startswith("search"):
            method = val.split(":")[1] if ":" in val else "cmaes"
            data = {"kind": "search", "method": method, "budget": 200}
        elif val == "grad":
            data = {"kind": "grad", "n_starts": 3, "n_steps": 200, "step_size": 0.05}
        else:
            data = {"kind": val}
        if opt:
            opt.data.update(data)
            assumptions.append(f"Optimizer → {val}")
        else:
            graph.add_node("Optimizer", x=680, y=140, data=data)
            assumptions.append(f"Added Optimizer {val}")
    if "scope" in patch:
        kind = patch["scope"]
        # ensure a scope of that kind exists; otherwise set first scope's kind
        found = any(n.type == "Scope" and n.data.get("kind") == kind for n in graph.nodes)
        if not found:
            for n in graph.nodes:
                if n.type == "Scope":
                    n.data["kind"] = kind
                    assumptions.append(f"Scope → {kind}")
                    break
            else:
                graph.add_node("Scope", x=880, y=140, data={"kind": kind})
                assumptions.append(f"Added Scope {kind}")
    return assumptions


class CommandPalette(QDialog):
    applied = pyqtSignal(dict)  # patch

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Command — Wolfram-style")
        self.setModal(True)
        self.setMinimumWidth(540)
        self.setStyleSheet("QDialog { background:#0f1115; }")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        hdr = QLabel("Type an instruction — e.g.  <b>optimize Bed1D silica gel RD for SCP under datacenter</b>")
        hdr.setStyleSheet("color:#94a3b8; font-size:8.5pt;")
        hdr.setWordWrap(True)
        layout.addWidget(hdr)

        self.edit = QLineEdit()
        self.edit.setPlaceholderText("optimize …  •  rank …  •  physics Bed1D …  •  material 13X …")
        self.edit.setStyleSheet("background:#1b1e24; color:#e2e8f0; border:1px solid #2a2f3a; border-radius:6px; padding:6px 8px;")
        completer = QCompleter(COMPLETIONS, self)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.edit.setCompleter(completer)
        self.edit.returnPressed.connect(self._on_accept)
        layout.addWidget(self.edit)

        self.preview = QLabel("Preview: —")
        self.preview.setStyleSheet("color:#64748b; font-size:8pt; background:#111318; border-radius:4px; padding:6px;")
        self.preview.setWordWrap(True)
        self.preview.setFrameStyle(QFrame.Shape.NoFrame)
        layout.addWidget(self.preview)

        self.edit.textChanged.connect(self._on_text_changed)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        apply = QPushButton("Apply")
        apply.setDefault(True)
        apply.clicked.connect(self._on_accept)
        apply.setStyleSheet("background:#3b82f6; color:white; padding:5px 14px; border-radius:6px;")
        btn_row.addWidget(cancel)
        btn_row.addWidget(apply)
        layout.addLayout(btn_row)

        self._patch: dict[str, Any] = {}

    def _on_text_changed(self, text: str) -> None:
        patch = parse_command(text)
        self._patch = patch
        if not patch:
            self.preview.setText("Preview: — (try:  optimize Bed1D for SCP,  material 13X,  physics TwoBed)")
        else:
            lines = [f"<b>{k}</b> → {v}" for k, v in patch.items()]
            self.preview.setText("Preview patch:<br/>" + "<br/>".join(lines))

    def _on_accept(self) -> None:
        if not self._patch:
            self._patch = parse_command(self.edit.text())
        if self._patch:
            self.applied.emit(dict(self._patch))
        self.accept()

    @staticmethod
    def get_patch(parent: Any = None) -> dict[str, Any] | None:
        dlg = CommandPalette(parent)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            return dlg._patch
        return None

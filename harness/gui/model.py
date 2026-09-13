"""Experiment graph model — toolkit-agnostic, JSON-serializable.

The model is the single source of truth; the QGraphicsScene is a
projection of it. It mirrors `harness/registry`, `harness/materials`,
`harness/profiles`, `harness/envs/base`, and `harness/backends` without
importing Qt.

Schema version 1: nodes + edges + seed + metadata. Every mutation goes
through the model so Save/Open and `to_python()` stay consistent.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2

# -- node type registry -----------------------------------------------------

NODE_TYPES = (
    "Material",
    "Profile",
    "Physics",      # subtype in data["kind"]: Cycle0D, Bed1D, TwoBed, TwoBedSchedule
    "Objective",
    "Optimizer",
    "Scope",        # subtype: Metrics, History, Trace, Ranking, Calibration, Compare, Sweep
    "Sweep",        # meta-block: grid sweep over t_switch or material
)

PHYSICS_KINDS = ("Cycle0D-v0", "Bed1D-v0", "TwoBed-v0", "TwoBedSchedule-v0", "Bed1DControls-v0", "ForcedConv-v0", "Cloak2D-v0", "AbsorptionCycle-v0", "Thermoelectric-v0", "NaturalConv-v0", "Telegrapher-v0", "Thermoelectric1D-v0", "Boussinesq-v0", "Heat1D-v0", "Heat2D-v0", "Heat3D-v0")
OPTIMIZER_KINDS = ("grad", "search", "search:cmaes", "search:tpe", "rl")
OBJECTIVE_PRESETS = ("COP", "SCP_W_kg", "COP+SCP", "custom")
SCOPE_KINDS = ("Metrics", "History", "Trace", "Ranking", "Calibration", "System", "Compare", "Sweep")
SWEEP_AXES = ("t_switch", "material")

# Category for palette grouping
NODE_CATEGORIES: dict[str, list[str]] = {
    "Sources": ["Material", "Profile"],
    "Physics": ["Physics"],
    "Design & Objective": ["Objective", "Sweep"],
    "Optimizers": ["Optimizer"],
    "Scopes": ["Scope"],
}

# Default data per node type (used when palette creates a node)
# Keep minimal — kind-specific defaults are injected by the Inspector on demand
# so the palette stays decluttered and the sheet only shows relevant fields.
DEFAULT_NODE_DATA: dict[str, dict[str, Any]] = {
    "Material": {"ref": "anchor:Silica gel RD"},
    "Profile": {"ref": "datacenter"},
    "Physics": {"kind": "Cycle0D-v0", "n_cells": 16, "dt_phys_s": None,
                "soft_switch": False, "n_cycles": 4, "hx_mass_factor": 1.35,
                "L_m": 0.002, "k_eff_w_m_k": 0.3, "h_wall_w_m2_k": 500.0,
                "recovery_ua_w_m2_k": 0.0, "t_rec_s": 0.0,
                "material_b": None},
    "Objective": {"preset": "COP", "weights": {"COP": 1.0},
                  "normalize": {}, "constraints": {}, "penalty_scale": 100.0},
    "Optimizer": {"kind": "search", "method": "cmaes", "budget": 300,
                  "n_starts": 3, "n_steps": 400, "step_size": 0.05, "sigma0": None},
    "Scope": {"kind": "Metrics"},
    "Sweep": {"axis": "t_switch", "grid": [60, 120, 240, 480, 600], "budget": 8,
              "grid_min": 60.0, "grid_max": 600.0, "steps": 6,
              "material_grid": ["anchor:Silica gel RD", "anchor:Zeolite 13X (NaX)"]},
}

# Per-kind defaults for fields not in the minimal node data — used by the
# Inspector to populate the form when the user switches kind (research-fit:
# every ladder kind has sensible starting values without polluting other kinds).
PHYSICS_DEFAULTS: dict[str, dict[str, Any]] = {
    "Cycle0D-v0": {"hx_mass_factor": 1.35},
    "Bed1D-v0": {"n_cells": 16, "dt_phys_s": None, "soft_switch": False, "n_cycles": 4, "hx_mass_factor": 1.35, "L_m": 0.002, "k_eff_w_m_k": 0.3, "h_wall_w_m2_k": 500.0},
    "Bed1DControls-v0": {"n_cells": 16, "dt_phys_s": None, "soft_switch": False, "n_cycles": 4, "hx_mass_factor": 1.35, "L_m": 0.002, "k_eff_w_m_k": 0.3, "h_wall_w_m2_k": 500.0},
    "TwoBed-v0": {"n_cells": 16, "dt_phys_s": None, "n_cycles": 4, "hx_mass_factor": 1.35, "L_m": 0.002, "k_eff_w_m_k": 0.3, "h_wall_w_m2_k": 500.0, "recovery_ua_w_m2_k": 0.0, "dt_ctrl_s": 5.0},
    "TwoBedSchedule-v0": {"n_cells": 16, "dt_phys_s": None, "n_cycles": 4, "hx_mass_factor": 1.35, "L_m": 0.002, "k_eff_w_m_k": 0.3, "h_wall_w_m2_k": 500.0, "recovery_ua_w_m2_k": 0.0, "dt_ctrl_s": 5.0},
    "ForcedConv-v0": {"nx": 24, "ny": 8, "dt": 0.002, "n_steps": 200, "alpha": 0.01, "Lx": 4.0, "Ly": 1.0, "source_amplitude": 500.0},
    "Cloak2D-v0": {"nx": 16, "ny": 16, "dt": 0.001, "n_steps": 60, "L": 1.0, "r_in": 0.20, "r_out": 0.40, "bg": 0.01},
    "AbsorptionCycle-v0": {"cycle_time_s": 300.0, "ua_loss_W_K_per_kg": 0.0, "t_amb_c": 20.0},
    "Thermoelectric-v0": {},
    "NaturalConv-v0": {"L_m": 0.1},
    "Telegrapher-v0": {"n_cells": 120, "dt": 0.001, "t_end": 0.25, "alpha": 1.0, "tau": 0.5},
    "Thermoelectric1D-v0": {"n_cells": 16, "dt": 0.001, "n_steps": 3000},
    "Boussinesq-v0": {"L_m": 0.03, "n_cells": 24, "t_end": 0.6},
    "Heat1D-v0": {"L": 1.0, "n_cells": 32, "dt": 0.001, "t_end": 2.0, "T_hot": 1.0, "T_cold": 0.0, "T_init": 0.0},
    "Heat2D-v0": {"L": 1.0, "nx": 16, "ny": 16, "dt": 0.001, "t_end": 5.0, "T_hot": 1.0, "T_cold": 0.0},
    "Heat3D-v0": {"L": 1.0, "nx": 12, "ny": 12, "nz": 12, "dt": 0.002, "t_end": 1.0, "save_every": 5, "T_hot": 1.0, "T_cold": 0.0, "mode": "slab", "core_radius": 0.2, "T_core": 1.0},
}

# Valid kwargs per physics kind (for executor/model filtering)
PHYSICS_KWARGS: dict[str, set[str]] = {
    "Cycle0D-v0": {"hx_mass_factor"},
    "Bed1D-v0": {"n_cells", "dt_phys_s", "soft_switch", "n_cycles", "hx_mass_factor",
                "L_m", "k_eff_w_m_k", "h_wall_w_m2_k"},
    "Bed1DControls-v0": {"n_cells", "dt_phys_s", "soft_switch", "n_cycles", "hx_mass_factor",
                        "L_m", "k_eff_w_m_k", "h_wall_w_m2_k"},
    "TwoBed-v0": {"n_cells", "dt_phys_s", "n_cycles", "hx_mass_factor", "recovery_ua_w_m2_k", "dt_ctrl_s",
                 "L_m", "k_eff_w_m_k", "h_wall_w_m2_k"},
    "TwoBedSchedule-v0": {"n_cells", "dt_phys_s", "n_cycles", "hx_mass_factor", "recovery_ua_w_m2_k", "dt_ctrl_s",
                         "L_m", "k_eff_w_m_k", "h_wall_w_m2_k"},
    "ForcedConv-v0": {"nx", "ny", "dt", "n_steps", "alpha", "Lx", "Ly", "source_amplitude"},
    "Cloak2D-v0": {"nx", "ny", "dt", "n_steps", "L", "r_in", "r_out", "bg"},
    "AbsorptionCycle-v0": {"cycle_time_s", "ua_loss_W_K_per_kg", "t_amb_c"},
    "Thermoelectric-v0": {},
    "NaturalConv-v0": {"L_m"},
    "Telegrapher-v0": {"n_cells", "dt", "t_end", "alpha", "tau"},
    "Thermoelectric1D-v0": {"n_cells", "dt", "n_steps"},
    "Boussinesq-v0": {"L_m", "n_cells", "t_end"},
    "Heat1D-v0": {"L", "n_cells", "dt", "t_end", "T_hot", "T_cold", "T_init"},
    "Heat2D-v0": {"L", "nx", "ny", "dt", "t_end", "T_hot", "T_cold"},
    "Heat3D-v0": {"L", "nx", "ny", "nz", "dt", "t_end", "save_every", "T_hot", "T_cold", "mode", "core_radius", "T_core"},
}

# Port typing — drives wire validation & palette drop-highlighting
# ``material_b`` is the composite-bed second material (TwoBed-v0 only; extra
# Material nodes may wire to this port, see ExperimentGraph.material_b_ref).
PORT_TYPES: dict[str, dict[str, list[str]]] = {
    "Material":  {"out": ["material", "material_b"]},
    "Profile":   {"out": ["profile"]},
    "Physics":   {"in": ["material", "material_b", "profile", "schedule"], "out": ["problem"]},
    "Objective": {"in": ["problem"], "out": ["objective"]},
    "Optimizer": {"in": ["objective"], "out": ["result"]},
    "Scope":     {"in": ["result", "trace", "problem"]},
    "Sweep":     {"in": ["problem", "objective", "material", "profile"], "out": ["result"]},
}

# -- dataclasses ------------------------------------------------------------


@dataclass
class NodeSpec:
    id: str
    type: str
    x: float = 0.0
    y: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)
    label: str = ""

    def __post_init__(self) -> None:
        if self.type not in NODE_TYPES:
            raise ValueError(f"unknown node type {self.type!r}")
        if not self.id:
            raise ValueError("node id must be non-empty")

    def display_label(self) -> str:
        if self.label:
            return self.label
        if self.type == "Physics":
            return self.data.get("kind", "Physics")
        if self.type == "Optimizer":
            return self.data.get("kind", "Optimizer")
        if self.type == "Scope":
            return self.data.get("kind", "Scope")
        if self.type == "Sweep":
            axis = self.data.get("axis", "sweep")
            return f"Sweep:{axis}"
        if self.type == "Material":
            return self.data.get("ref", "Material")
        if self.type == "Profile":
            return self.data.get("ref", "Profile")
        if self.type == "Objective":
            return self.data.get("preset", "Objective")
        return self.type


@dataclass
class EdgeSpec:
    id: str
    from_node: str
    from_port: str
    to_node: str
    to_port: str

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("edge id must be non-empty")


@dataclass
class ExperimentGraph:
    """Toolkit-agnostic experiment graph. Pure data, no Qt."""

    schema_version: int = SCHEMA_VERSION
    nodes: list[NodeSpec] = field(default_factory=list)
    edges: list[EdgeSpec] = field(default_factory=list)
    seed: int = 0
    collect_trace: bool = True
    name: str = "Untitled"
    description: str = ""

    # -- mutation helpers ---------------------------------------------------

    def add_node(self, node_type: str, x: float = 0, y: float = 0,
                 data: dict[str, Any] | None = None,
                 label: str = "") -> NodeSpec:
        nid = f"{node_type.lower()}_{uuid.uuid4().hex[:6]}"
        merged = dict(DEFAULT_NODE_DATA.get(node_type, {}))
        if data:
            merged.update(data)
        node = NodeSpec(id=nid, type=node_type, x=float(x), y=float(y),
                        data=merged, label=label)
        self.nodes.append(node)
        return node

    def remove_node(self, node_id: str) -> None:
        self.nodes = [n for n in self.nodes if n.id != node_id]
        self.edges = [e for e in self.edges if e.from_node != node_id and e.to_node != node_id]

    def add_edge(self, from_node: str, from_port: str, to_node: str, to_port: str) -> EdgeSpec:
        eid = f"e_{uuid.uuid4().hex[:6]}"
        edge = EdgeSpec(id=eid, from_node=from_node, from_port=from_port,
                        to_node=to_node, to_port=to_port)
        # prevent duplicate wire to same to_port on same node
        for e in self.edges:
            if e.to_node == to_node and e.to_port == to_port:
                # allow multiple? for now replace
                self.edges.remove(e)
                break
        self.edges.append(edge)
        return edge

    def remove_edge(self, edge_id: str) -> None:
        self.edges = [e for e in self.edges if e.id != edge_id]

    def node_by_id(self, nid: str) -> NodeSpec | None:
        for n in self.nodes:
            if n.id == nid:
                return n
        return None

    def edges_from(self, nid: str) -> list[EdgeSpec]:
        return [e for e in self.edges if e.from_node == nid]

    def edges_to(self, nid: str) -> list[EdgeSpec]:
        return [e for e in self.edges if e.to_node == nid]

    def incoming(self, nid: str, port: str) -> EdgeSpec | None:
        for e in self.edges:
            if e.to_node == nid and e.to_port == port:
                return e
        return None

    def outgoing(self, nid: str, port: str) -> list[EdgeSpec]:
        return [e for e in self.edges if e.from_node == nid and e.from_port == port]

    # -- validation ---------------------------------------------------------

    def validate(self) -> list[dict[str, str]]:
        """Return list of {node, code, message}. Empty means valid for run."""
        errors: list[dict[str, str]] = []
        ids = {n.id for n in self.nodes}
        # check edges reference existing nodes
        for e in self.edges:
            if e.from_node not in ids:
                errors.append({"node": e.id, "code": "bad_edge", "message": f"edge {e.id} from_node {e.from_node!r} not found"})
            if e.to_node not in ids:
                errors.append({"node": e.id, "code": "bad_edge", "message": f"edge {e.id} to_node {e.to_node!r} not found"})
        # domain rules
        physics_nodes = [n for n in self.nodes if n.type == "Physics"]
        if not physics_nodes:
            errors.append({"node": "*", "code": "no_physics", "message": "Graph needs at least one Physics block (Cycle0D/Bed1D/TwoBed)"})
        elif len(physics_nodes) > 1:
            # warn but not error for now — future multi-env compare
            pass
        for n in self.nodes:
            if n.type == "Physics":
                kind = n.data.get("kind", "")
                if kind not in PHYSICS_KINDS:
                    errors.append({"node": n.id, "code": "bad_physics_kind", "message": f"Unknown physics kind {kind!r}"})
                # Schedule semantics
                if kind == "Cycle0D-v0":
                    # schedule wire is warned
                    for e in self.edges_to(n.id):
                        if e.to_port == "schedule":
                            errors.append({"node": n.id, "code": "schedule_on_static", "message": "Cycle0D has no temporal structure — Schedule wire ignored (see harness/envs/cycle0d.py)"})
            if n.type == "Objective":
                w = n.data.get("weights", {})
                if not w:
                    errors.append({"node": n.id, "code": "no_weights", "message": "Objective needs at least one weight"})
            if n.type == "Optimizer":
                kind = n.data.get("kind", "")
                # check rl on static
                phys = self._resolved_physics_kind()
                if kind == "rl" and phys == "Cycle0D-v0":
                    errors.append({"node": n.id, "code": "rl_on_static", "message": "RL is not meaningful on Cycle0D (static problem) — use grad or search (DESIGN hon. rule)"})
            if n.type == "Material":
                ref = n.data.get("ref", "")
                if not ref:
                    errors.append({"node": n.id, "code": "no_material", "message": "Material block needs a registry ref"})
            if n.type == "Profile":
                ref = n.data.get("ref", "")
                if not ref:
                    errors.append({"node": n.id, "code": "no_profile", "message": "Profile block needs a registry ref"})
            if n.type == "Sweep":
                axis = n.data.get("axis", "t_switch")
                if axis not in ("t_switch", "material"):
                    errors.append({"node": n.id, "code": "bad_sweep_axis", "message": f"Sweep axis must be t_switch or material, got {axis!r}"})
                # grid validation
                grid = n.data.get("grid")
                if axis == "t_switch":
                    # expect numeric grid via grid or grid_min/max/steps
                    if grid is not None:
                        if not isinstance(grid, (list, tuple)) or len(grid) < 2:
                            errors.append({"node": n.id, "code": "bad_sweep_grid", "message": "t_switch Sweep needs grid with >=2 points"})
                        else:
                            try:
                                vals = [float(v) for v in grid]  # type: ignore
                                if any(v <= 0 for v in vals):
                                    errors.append({"node": n.id, "code": "bad_sweep_grid", "message": "t_switch grid values must be >0"})
                            except Exception:
                                errors.append({"node": n.id, "code": "bad_sweep_grid", "message": "t_switch grid must be numeric"})
                    else:
                        gmin = n.data.get("grid_min", 60)
                        gmax = n.data.get("grid_max", 600)
                        steps = n.data.get("steps", 6)
                        try:
                            if float(gmin) >= float(gmax) or int(steps) < 2:
                                errors.append({"node": n.id, "code": "bad_sweep_grid", "message": "Sweep needs grid_min < grid_max and steps >=2"})
                        except Exception:
                            errors.append({"node": n.id, "code": "bad_sweep_grid", "message": "Sweep grid_min/max/steps invalid"})
                else:  # material
                    mat_grid = n.data.get("grid", n.data.get("material_grid"))
                    if not mat_grid or not isinstance(mat_grid, (list, tuple)) or len(mat_grid) < 1:
                        errors.append({"node": n.id, "code": "bad_sweep_grid", "message": "material Sweep needs grid with >=1 material refs"})
            if n.type == "Scope":
                kind = n.data.get("kind", "")
                if kind not in SCOPE_KINDS:
                    errors.append({"node": n.id, "code": "bad_scope_kind", "message": f"Unknown scope kind {kind!r}"})
        # port type checks
        for e in self.edges:
            src = self.node_by_id(e.from_node)
            dst = self.node_by_id(e.to_node)
            if not src or not dst:
                continue
            src_out = PORT_TYPES.get(src.type, {}).get("out", [])
            dst_in = PORT_TYPES.get(dst.type, {}).get("in", [])
            if e.from_port not in src_out:
                errors.append({"node": e.id, "code": "bad_port", "message": f"{src.type} has no out port {e.from_port!r}"})
            if e.to_port not in dst_in:
                errors.append({"node": e.id, "code": "bad_port", "message": f"{dst.type} has no in port {e.to_port!r}"})
            # semantic port mapping
            if e.from_port == "material" and e.to_port != "material":
                errors.append({"node": e.id, "code": "port_mismatch", "message": "material out must wire to material in"})
            if e.from_port == "profile" and e.to_port != "profile":
                errors.append({"node": e.id, "code": "port_mismatch", "message": "profile out must wire to profile in"})
        return errors

    def _resolved_physics_kind(self) -> str | None:
        for n in self.nodes:
            if n.type == "Physics":
                return n.data.get("kind")
        return None

    # -- convenience extractors for executor --------------------------------

    def material_ref(self) -> str | None:
        for e in self.edges:
            src = self.node_by_id(e.from_node)
            if src and src.type == "Material" and e.to_port == "material":
                return src.data.get("ref")
        # fallback: first Material node even if not wired
        for n in self.nodes:
            if n.type == "Material":
                return n.data.get("ref")
        return None

    def material_b_ref(self) -> str | None:
        """Second material for composite ``TwoBed-v0`` (``material_b``)."""
        for e in self.edges:
            src = self.node_by_id(e.from_node)
            if src and src.type == "Material" and e.to_port == "material_b":
                return src.data.get("ref")
        # Fallback: Physics node's ``material_b`` data field when no wire
        phys = self.physics_node()
        if phys is not None:
            mb = phys.data.get("material_b")
            if isinstance(mb, str) and mb:
                return mb
        # Second Material node (unwired) as fallback for TwoBed demos
        mats = [n for n in self.nodes if n.type == "Material"]
        if len(mats) >= 2:
            # first is material_ref, second is material_b
            second = mats[1].data.get("ref")
            if second and second != self.material_ref():
                return second
        return None

    def profile_ref(self) -> str | None:
        for e in self.edges:
            src = self.node_by_id(e.from_node)
            if src and src.type == "Profile" and e.to_port == "profile":
                return src.data.get("ref")
        for n in self.nodes:
            if n.type == "Profile":
                return n.data.get("ref")
        return None

    def physics_node(self) -> NodeSpec | None:
        for n in self.nodes:
            if n.type == "Physics":
                return n
        return None

    def physics_kwargs(self) -> dict[str, Any]:
        n = self.physics_node()
        if not n:
            return {}
        kind = n.data.get("kind", "Cycle0D-v0")
        d = dict(n.data)
        d.pop("kind", None)
        if d.get("dt_phys_s") is None:
            d.pop("dt_phys_s", None)
        # material_b is handled separately (wired, not a struct kwarg)
        d.pop("material_b", None)
        allowed = PHYSICS_KWARGS.get(kind, set())
        # fallback: allow common if kind unknown
        if not allowed:
            allowed = {"n_cells", "dt_phys_s", "soft_switch", "n_cycles", "hx_mass_factor"}
        # also strip keys that are not in allowed but were in default data
        return {k: v for k, v in d.items() if k in allowed and v is not None}

    def objective_node(self) -> NodeSpec | None:
        for n in self.nodes:
            if n.type == "Objective":
                return n
        return None

    def objective_kwargs(self) -> dict[str, Any]:
        n = self.objective_node()
        if not n:
            return {"weights": {"COP": 1.0}, "normalize": {}, "constraints": {}, "penalty_scale": 100.0}
        return {"weights": dict(n.data.get("weights", {"COP": 1.0})),
                "normalize": dict(n.data.get("normalize", {})),
                "constraints": dict(n.data.get("constraints", {})),
                "penalty_scale": float(n.data.get("penalty_scale", 100.0))}

    def optimizer_node(self) -> NodeSpec | None:
        for n in self.nodes:
            if n.type == "Optimizer":
                return n
        return None

    def optimizer_kwargs(self) -> dict[str, Any]:
        n = self.optimizer_node()
        if not n:
            return {"kind": "search", "method": "cmaes", "budget": 100}
        d = dict(n.data)
        return d

    def sweep_node(self) -> NodeSpec | None:
        for n in self.nodes:
            if n.type == "Sweep":
                return n
        return None

    def sweep_kwargs(self) -> dict[str, Any]:
        n = self.sweep_node()
        if not n:
            return {}
        d = dict(n.data)
        # normalize grid: prefer explicit grid list, fallback to grid_min/max/steps
        if "grid" not in d or not d["grid"]:
            gmin = d.get("grid_min", 60.0)
            gmax = d.get("grid_max", 600.0)
            steps = int(d.get("steps", 6))
            try:
                import numpy as np  # noqa: WPS433
                d["grid"] = np.linspace(float(gmin), float(gmax), steps).tolist()
            except Exception:
                d["grid"] = [float(gmin), float(gmax)]
        # for material axis, also expose material_grid alias
        if d.get("axis") == "material" and "material_grid" in d and not d.get("grid"):
            d["grid"] = d["material_grid"]
        return d

    def has_sweep(self) -> bool:
        return any(n.type == "Sweep" for n in self.nodes)

    def backend_name(self) -> str:
        n = self.optimizer_node()
        if not n:
            return "search"
        kind = n.data.get("kind", "search")
        # registry only knows grad/search/rl; method is a kwarg
        if kind in ("grad", "search", "rl"):
            return kind
        # tolerate legacy "search:cmaes"
        if isinstance(kind, str) and kind.startswith("search"):
            return "search"
        return str(kind)

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "name": self.name,
            "description": self.description,
            "seed": self.seed,
            "collectTrace": self.collect_trace,
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentGraph":
        raw_version = int(d.get("schemaVersion", 1))
        # -- migration: v1 -> v2 (add Sweep node defaults, Compare scope kind) --
        # v1 files have no Sweep type; preserve all nodes verbatim, just bump version
        # If a Scope kind is unknown (e.g. future), keep it as-is and let validate warn.
        nodes_raw = d.get("nodes", [])
        # patch legacy: ensure Sweep nodes have required keys axis/grid/budget
        for nd in nodes_raw:
            if nd.get("type") == "Sweep":
                data = nd.get("data", {})
                data.setdefault("axis", "t_switch")
                data.setdefault("grid", [60, 120, 240, 480, 600])
                data.setdefault("budget", 8)
                data.setdefault("grid_min", 60.0)
                data.setdefault("grid_max", 600.0)
                data.setdefault("steps", 6)
        # also migrate SCOPE_KINDS: old System etc remain valid
        nodes = [NodeSpec(**nd) for nd in nodes_raw]
        edges_raw = d.get("edges", [])
        edges: list[EdgeSpec] = []
        for ed in edges_raw:
            # tolerate legacy keys from_node/from
            fn = ed.get("from_node", ed.get("from"))
            tn = ed.get("to_node", ed.get("to"))
            fp = ed.get("from_port", ed.get("fromPort", ""))
            tp = ed.get("to_port", ed.get("toPort", ""))
            edges.append(EdgeSpec(id=ed.get("id", f"e_{uuid.uuid4().hex[:6]}"),
                                   from_node=fn, from_port=fp, to_node=tn, to_port=tp))
        migrated_version = max(raw_version, SCHEMA_VERSION) if raw_version < SCHEMA_VERSION else raw_version
        # If incoming version is older, upgrade to current SCHEMA_VERSION
        effective_version = SCHEMA_VERSION if raw_version < SCHEMA_VERSION else raw_version
        return cls(
            schema_version=int(effective_version),
            nodes=nodes,
            edges=edges,
            seed=int(d.get("seed", 0)),
            collect_trace=bool(d.get("collectTrace", True)),
            name=str(d.get("name", "Untitled")),
            description=str(d.get("description", "")),
        )

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentGraph":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(d)

    # -- python code generation ---------------------------------------------

    def to_python(self) -> str:
        """Emit minimal reproducible harness script for this graph."""
        mat = self.material_ref() or "anchor:Silica gel RD"
        prof = self.profile_ref() or "datacenter"
        phys = self.physics_node()
        kind = phys.data.get("kind", "Cycle0D-v0") if phys else "Cycle0D-v0"
        phys_kw = self.physics_kwargs()
        obj_kw = self.objective_kwargs()
        opt_kw = self.optimizer_kwargs()
        sweep_kw = self.sweep_kwargs() if self.has_sweep() else {}
        backend = self.backend_name()
        # format phys kwargs as python dict entries
        def fmt(v: Any) -> str:
            if isinstance(v, str):
                return repr(v)
            if isinstance(v, bool):
                return repr(v)
            if v is None:
                return "None"
            return repr(v)
        phys_lines = ",\n    ".join(f"{k}={fmt(v)}" for k, v in phys_kw.items())
        phys_extra = f",\n    {phys_lines}" if phys_lines else ""
        # objective
        weights = obj_kw["weights"]
        normalize = obj_kw["normalize"]
        constraints = obj_kw["constraints"]
        penalty = obj_kw["penalty_scale"]
        # build objective code
        if len(weights) == 1 and not normalize and not constraints:
            metric = next(iter(weights))
            obj_code = f'harness.Objective.single({metric!r})'
        else:
            obj_code = (
                f'harness.Objective(\n'
                f'    weights={weights!r},\n'
                f'    normalize={normalize!r},\n'
                f'    constraints={constraints!r},\n'
                f'    penalty_scale={penalty!r},\n'
                f')'
            )
        # if Sweep present, emit sweep code instead of optimize
        if self.has_sweep():
            axis = sweep_kw.get("axis", "t_switch")
            grid = sweep_kw.get("grid", [60, 120, 240, 480])
            budget = sweep_kw.get("budget", 8)
            if axis == "t_switch":
                # Bed1DControls sweep via harness.control.switch_time_sweep
                return (
                    '"""Generated by harness GUI — Sweep (t_switch) reproducible experiment."""\n'
                    'import harness\n'
                    'import harness.control as ctrl\n'
                    'from harness.envs.bed1d import Bed1D, Bed1DControls\n'
                    'import numpy as np, pandas as pd\n'
                    '\n'
                    f'# Material & profile as wired on the canvas\n'
                    f'material = {mat!r}\n'
                    f'profile = {prof!r}\n'
                    f'grid = {grid!r}\n'
                    f'# Build Bed1D according to physics block (n_cells, n_cycles, etc.)\n'
                    f'bed = Bed1D(material=material, profile=profile, n_cells={phys_kw.get("n_cells", 8)}, '
                    f'n_cycles={phys_kw.get("n_cycles", 2)}, soft_switch={phys_kw.get("soft_switch", False)})\n'
                    f'prob = Bed1DControls(bed, n_cycles={phys_kw.get("n_cycles", 2)}, dt_phys_s=0.1, n_steps=20008)\n'
                    f'rows = ctrl.switch_time_sweep(prob, grid)\n'
                    f'df = pd.DataFrame(rows)\n'
                    f'print(df.to_string())\n'
                    f'print("best_scp:", df["SCP_W_kg"].max(), "best_cop:", df["COP"].max())\n'
                    f'# For ranking sweep (material axis) see harness.rank.sweep_materials\n'
                )
            else:
                # material axis -> harness.rank.sweep_materials
                return (
                    '"""Generated by harness GUI — Sweep (material) reproducible experiment."""\n'
                    'import harness\n'
                    'import harness.rank as rank\n'
                    'from harness.materials import get_material\n'
                    'import pandas as pd\n'
                    '\n'
                    f'grid_materials = {grid!r}\n'
                    f'profile = {prof!r}\n'
                    f'materials = [get_material(m) for m in grid_materials]\n'
                    f'df = rank.sweep_materials(materials, profiles=[profile])\n'
                    f'print(df.to_string())\n'
                    f'print(df.head())\n'
                )
        # optimizer kwargs for optimize call
        opt_call = ""
        if backend.startswith("search"):
            method = opt_kw.get("method", "cmaes")
            budget = opt_kw.get("budget", 300)
            opt_call = f'backend="{backend}", seed={self.seed}, budget={budget}, method={method!r}'
            # alternative signature: harness.optimize handles backend string
            opt_call = f'backend="{backend}", seed={self.seed}, budget={budget}'
        elif backend == "grad":
            n_starts = opt_kw.get("n_starts", 3)
            n_steps = opt_kw.get("n_steps", 400)
            step_size = opt_kw.get("step_size", 0.05)
            opt_call = f'backend="grad", seed={self.seed}, n_starts={n_starts}, n_steps={n_steps}, step_size={step_size}'
        else:
            opt_call = f'backend={backend!r}, seed={self.seed}'

        # Composite support (TwoBed pair, TE1D grading)
        mat_b = self.material_b_ref()
        mat_b_line = f'\n    material_b={mat_b!r},' if mat_b and kind in ("TwoBed-v0", "TwoBedSchedule-v0", "Thermoelectric1D-v0") else ""
        trace_line = "trace = env.rollout(collect_trace=True)" if self.collect_trace else "# trace disabled"
        return (
            '"""Generated by harness GUI — reproducible experiment."""\n'
            'import harness\n'
            'from harness.envs.base import Objective\n'
            '\n'
            f'# Material & profile resolved as wired on the canvas\n'
            f'env = harness.make(\n'
            f'    {kind!r},\n'
            f'    material={mat!r},{mat_b_line}\n'
            f'    profile={prof!r}{phys_extra},\n'
            f')\n'
            f'\n'
            f'objective = {obj_code}\n'
            f'\n'
            f'result = harness.optimize(env, objective, {opt_call})\n'
            f'print(harness.report.summary(result))\n'
            f'print("best_design:", result.best_design)\n'
            f'print("best_metrics:", result.best_metrics)\n'
            f'{trace_line}\n'
            f'if "trace" in dir():\n'
            f'    print("trace summary:", trace.summary if "trace" in locals() else "n/a")\n'
        )

    # -- factory helpers ----------------------------------------------------

    @classmethod
    def default_cycle0d(cls) -> "ExperimentGraph":
        g = cls(name="Cycle0D demo", description="Material → Profile → Cycle0D → Objective → Search → Metrics")
        m = g.add_node("Material", x=40, y=80, data={"ref": "anchor:Silica gel RD"})
        p = g.add_node("Profile", x=40, y=200, data={"ref": "datacenter"})
        phys = g.add_node("Physics", x=260, y=140, data={"kind": "Cycle0D-v0"})
        obj = g.add_node("Objective", x=480, y=140, data={"preset": "COP", "weights": {"COP": 1.0}})
        opt = g.add_node("Optimizer", x=680, y=140, data={"kind": "search", "method": "cmaes", "budget": 200})
        scope = g.add_node("Scope", x=880, y=140, data={"kind": "Metrics"})
        g.add_edge(m.id, "material", phys.id, "material")
        g.add_edge(p.id, "profile", phys.id, "profile")
        g.add_edge(phys.id, "problem", obj.id, "problem")
        g.add_edge(obj.id, "objective", opt.id, "objective")
        g.add_edge(opt.id, "result", scope.id, "result")
        return g

    @classmethod
    def default_bed1d(cls) -> "ExperimentGraph":
        g = cls(name="Bed1D demo", description="Material → Profile → Bed1D → Objective → Search → Metrics+Trace")
        m = g.add_node("Material", x=40, y=80, data={"ref": "anchor:Silica gel RD"})
        p = g.add_node("Profile", x=40, y=200, data={"ref": "datacenter"})
        phys = g.add_node("Physics", x=260, y=140, data={"kind": "Bed1D-v0", "n_cells": 16, "n_cycles": 4, "soft_switch": False})
        obj = g.add_node("Objective", x=480, y=140, data={"preset": "SCP_W_kg", "weights": {"SCP_W_kg": 1.0}})
        opt = g.add_node("Optimizer", x=680, y=140, data={"kind": "search", "method": "cmaes", "budget": 150})
        metrics = g.add_node("Scope", x=880, y=100, data={"kind": "Metrics"})
        trace = g.add_node("Scope", x=880, y=200, data={"kind": "Trace"})
        g.add_edge(m.id, "material", phys.id, "material")
        g.add_edge(p.id, "profile", phys.id, "profile")
        g.add_edge(phys.id, "problem", obj.id, "problem")
        g.add_edge(obj.id, "objective", opt.id, "objective")
        g.add_edge(opt.id, "result", metrics.id, "result")
        g.add_edge(opt.id, "result", trace.id, "result")
        return g

    @classmethod
    def presets(cls) -> dict[str, "ExperimentGraph"]:
        return {
            "Cycle0D demo": cls.default_cycle0d(),
            "Bed1D demo": cls.default_bed1d(),
        }

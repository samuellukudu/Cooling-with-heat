"""Thermoelectric1D-v0 — segmented leg as a static Problem (T-A2 extension).

Env = (1-D two-field leg physics) × (cold pair × hot pair) × (profile).
Discovery axes: **design** (drive current + segment fraction — grading
a Bi2Te3 cold stage under a PbTe hot stage beats either uniform leg)
and **materials** (pair assignment per segment). The lumped
``Thermoelectric-v0`` is the fast screen; this env answers the
spatially-resolved questions it cannot pose.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..physics import te_1d
from ..profiles import ApplicationProfile, get_profile
from ..registry import REGISTRIES
from .base import DesignSpace, EpisodeTrace, ProblemSpec
from .thermoelectric import _resolve_pair_name

DESIGN_KEYS = ("current_A", "seg_frac")
TE1D_ENV_METRIC_KEYS = te_1d.TE1D_METRIC_KEYS
TE1D_SCHEMA_VERSION = 1


class Thermoelectric1D:
    """Segmented Peltier leg over (cold pair × hot pair) × profile."""

    def __init__(
        self,
        material: Any = "Bi2Te3",
        material_b: Any | None = None,
        profile: "str | ApplicationProfile" = "cpu",
        *,
        n_cells: int = 16,
        dt: float = 0.001,
        n_steps: int = 3000,
    ):
        self.pair_cold, self.prov_cold = _resolve_pair_name(material)
        self.pair_hot, self.prov_hot = _resolve_pair_name(
            material_b if material_b is not None else material)
        self.material, self.material_b = material, material_b
        self.profile = get_profile(profile)
        self.Th_C = float(self.profile.t_cond_c)
        self.Tc_C = float(self.profile.t_evap_c)
        self.n_cells, self.dt, self.n_steps = int(n_cells), float(dt), int(n_steps)
        dx = te_1d.LEG_L_M / self.n_cells
        k_max = max(te_1d.leg_sigma_k(self.pair_cold)[2],
                    te_1d.leg_sigma_k(self.pair_hot)[2])
        if not te_1d.check_timestep(dx, k_max, self.dt):
            raise ValueError(f"TE1D dt={self.dt} violates CFL")
        self.spec = ProblemSpec(name="Thermoelectric1D-v0", kind="static",
                                metric_keys=TE1D_ENV_METRIC_KEYS,
                                schema_version=TE1D_SCHEMA_VERSION,
                                spatial_dim=1, time_resolved=True, grid_type="uniform_rect")
        self.design_space = DesignSpace(
            keys=DESIGN_KEYS,
            defaults={"current_A": 2.0, "seg_frac": 0.5},
            bounds={"current_A": (0.5, 8.0), "seg_frac": (0.15, 0.85)},
        )
        from ..physics.adapters import ModelSpec  # noqa: WPS433
        self.model_spec = ModelSpec(
            name="te_1d",
            state_vars=("T",),
            units={"current_A": "A", "Qc_W": "W"},
            dt_min_s=1e-6,
            dt_max_s=0.01,
            description="segmented TE leg (T-A2 extension)",
        )

    def metrics_jax(self, design: Mapping[str, Any] | None = None):
        merged = self.design_space.merge(design)
        out = te_1d.simulate_te_1d(
            Tc_C=self.Tc_C, Th_C=self.Th_C, current_A=merged["current_A"],
            pair_cold=self.pair_cold, pair_hot=self.pair_hot,
            seg_frac=merged["seg_frac"], n_cells=self.n_cells, dt=self.dt,
            n_steps=self.n_steps)
        return {k: out[k] for k in TE1D_ENV_METRIC_KEYS}

    def evaluate(self, design: Mapping[str, float] | None = None) -> dict[str, float]:
        return {k: float(v) for k, v in self.metrics_jax(design).items()}

    def rollout(self, design=None, controls=None, *, n_steps: int = 1) -> EpisodeTrace:
        del controls, n_steps
        merged = self.design_space.merge(design)
        out = te_1d.simulate_te_1d(
            Tc_C=self.Tc_C, Th_C=self.Th_C, current_A=merged["current_A"],
            pair_cold=self.pair_cold, pair_hot=self.pair_hot,
            seg_frac=merged["seg_frac"], n_cells=self.n_cells, dt=self.dt,
            n_steps=self.n_steps, collect_trace=True)
        summary = {k: float(out[k]) for k in TE1D_ENV_METRIC_KEYS}
        traj = out.get("trajectory")
        return EpisodeTrace(series={"T_trajectory": traj} if traj is not None else {},
                            summary=summary)


def build_te_1d(material: Any = "Bi2Te3", profile: "str | ApplicationProfile" = "cpu",
                material_b: Any | None = None, **kwargs) -> Thermoelectric1D:
    allowed = {"n_cells", "dt", "n_steps"}
    ctor = {k: v for k, v in kwargs.items() if k in allowed}
    return Thermoelectric1D(material=material, material_b=material_b,
                            profile=profile, **ctor)


REGISTRIES["envs"].register("Thermoelectric1D-v0", build_te_1d)

try:
    from ..physics.adapters import Fluxes as _Fluxes  # noqa: WPS433
    from ..physics.adapters import ModelSpec as _MS  # noqa: WPS433

    class _TE1DAdapter:
        def __init__(self):
            self.spec = _MS(name="te_1d", state_vars=("T",),
                            units={"Qc_W": "W"}, dt_min_s=1e-6, dt_max_s=0.01,
                            description="segmented TE adapter (T-A2)")

        def step(self, state, control, dt_s):
            return dict(state), _Fluxes()

        def metrics(self, trace):
            return dict(trace)

    try:
        REGISTRIES["models"].register("te_1d", _TE1DAdapter)
    except ValueError:
        pass
except Exception:
    pass

__all__ = ["Thermoelectric1D", "build_te_1d", "DESIGN_KEYS", "TE1D_ENV_METRIC_KEYS"]

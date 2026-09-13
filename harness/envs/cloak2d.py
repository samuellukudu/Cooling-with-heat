"""Cloak2D-v0 — thermal-cloak κ-field as a static harness Problem (trial T-A4).

Env = (inhomogeneous heat physics) × (raw setpoints). Discovery axis is
**design** (``kappa_inner``, ``kappa_ring`` ring params — the pixel field is
a pull-driven extension). No material/profile physics consumed; both are
accepted for ``harness.make()`` uniformity.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..physics import cloak2d as ck
from ..profiles import ApplicationProfile, get_profile
from ..registry import REGISTRIES
from .base import DesignSpace, EpisodeTrace, ProblemSpec

DESIGN_KEYS = ("kappa_inner", "kappa_ring")
CLOAK_METRIC_KEYS = ck.CLOAK_METRIC_KEYS
CLOAK_SCHEMA_VERSION = 1


class Cloak2D:
    def __init__(
        self,
        material: Any | None = None,
        profile: "str | ApplicationProfile" = "datacenter",
        *,
        L: float = ck.DEFAULT_L,
        nx: int = 16,
        ny: int = 16,
        dt: float = 0.001,
        n_steps: int = 60,
        r_in: float = ck.DEFAULT_R_IN,
        r_out: float = ck.DEFAULT_R_OUT,
        bg: float = ck.DEFAULT_BG,
    ):
        self.material = material
        self.profile = get_profile(profile)
        self.L = float(L)
        self.nx, self.ny = int(nx), int(ny)
        self.dt = float(dt)
        self.n_steps = int(n_steps)
        self.r_in, self.r_out = float(r_in), float(r_out)
        self.bg = float(bg)
        dx = self.L / self.nx
        dy = self.L / self.ny
        if not ck.check_timestep_cloak(dx, dy, 0.05, self.dt):
            raise ValueError(f"Cloak2D dt={self.dt} violates CFL at kappa_max=0.05")
        self.spec = ProblemSpec(name="Cloak2D-v0", kind="static",
                                metric_keys=CLOAK_METRIC_KEYS,
                                schema_version=CLOAK_SCHEMA_VERSION,
                                spatial_dim=2, time_resolved=True, grid_type="uniform_rect")
        self.design_space = DesignSpace(
            keys=DESIGN_KEYS,
            defaults={"kappa_inner": 0.002, "kappa_ring": 0.02},
            bounds={"kappa_inner": (0.001, 0.02), "kappa_ring": (0.005, 0.05)},
        )
        from ..physics.adapters import ModelSpec  # noqa: WPS433
        self.model_spec = ModelSpec(
            name="cloak2d",
            state_vars=("T", "kappa"),
            units={"T": "-", "kappa": "m2/s"},
            dt_min_s=1e-6,
            dt_max_s=0.01,
            description="2-D cloak heat (T-A4 trial; field-multiply kappa)",
        )

    def metrics_jax(self, design: Mapping[str, Any] | None = None):
        merged = self.design_space.merge(design)
        ki, kr = merged["kappa_inner"], merged["kappa_ring"]
        sim = ck.simulate_cloak(kappa_inner=ki, kappa_ring=kr, bg=self.bg, L=self.L,
                                nx=self.nx, ny=self.ny, dt=self.dt, n_steps=self.n_steps,
                                r_in=self.r_in, r_out=self.r_out)
        return ck.cloak_metrics(sim["final"], sim["kappa"], sim["X"], sim["Y"], sim["R"],
                                sim["dx"], sim["dy"], self.L, ki, kr,
                                r_in=self.r_in, r_out=self.r_out)

    def evaluate(self, design: Mapping[str, float] | None = None) -> dict[str, float]:
        return {k: float(v) for k, v in self.metrics_jax(design).items()}

    def rollout(self, design=None, controls=None, *, n_steps: int = 1) -> EpisodeTrace:
        del controls, n_steps
        merged = self.design_space.merge(design)
        ki, kr = merged["kappa_inner"], merged["kappa_ring"]
        sim = ck.simulate_cloak(kappa_inner=ki, kappa_ring=kr, bg=self.bg, L=self.L,
                                nx=self.nx, ny=self.ny, dt=self.dt, n_steps=self.n_steps,
                                r_in=self.r_in, r_out=self.r_out, collect_trace=True)
        summary = {k: float(v) for k, v in
                   ck.cloak_metrics(sim["final"], sim["kappa"], sim["X"], sim["Y"], sim["R"],
                                    sim["dx"], sim["dy"], self.L, ki, kr,
                                    r_in=self.r_in, r_out=self.r_out).items()}
        traj = sim.get("trajectory")
        series = {"T_trajectory": traj} if traj is not None else {}
        return EpisodeTrace(series=series, summary=summary)


def build_cloak2d(material: Any | None = None,
                  profile: "str | ApplicationProfile" = "datacenter",
                  **kwargs) -> Cloak2D:
    allowed = {"L", "nx", "ny", "dt", "n_steps", "r_in", "r_out", "bg"}
    ctor = {k: v for k, v in kwargs.items() if k in allowed}
    return Cloak2D(material=material, profile=profile, **ctor)


REGISTRIES["envs"].register("Cloak2D-v0", build_cloak2d)

try:
    from ..physics.adapters import ModelSpec as _MS  # noqa: WPS433

    class _CloakAdapter:
        def __init__(self):
            self.spec = _MS(name="cloak2d", state_vars=("T", "kappa"),
                            units={"T": "-"}, dt_min_s=1e-6, dt_max_s=0.01,
                            description="cloak adapter (T-A4)")

        def step(self, state, control, dt_s):
            return dict(state), __import__("harness.physics.adapters", fromlist=["Fluxes"]).Fluxes()

        def metrics(self, trace):
            return dict(trace)

    try:
        REGISTRIES["models"].register("cloak2d", _CloakAdapter)
    except ValueError:
        pass
except Exception:
    pass

__all__ = ["Cloak2D", "build_cloak2d", "DESIGN_KEYS", "CLOAK_METRIC_KEYS"]

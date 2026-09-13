"""Heat1D-v0 — 1D heat equation as a harness Problem (diffheat 1D ladder).

Env = (uniform ``Grid1D`` + ghost-cell BCs + 5-point Laplacian) × (scenario).
Discovery axes: **design** (diffusivity ``alpha``) and **materials**-as-data
(``alpha`` may be material-mapped later). Mirrors
``examples/01-1d-heat-equation/demo.py`` without importing ``diffheat``.
Time is the 4th dimension: the whole rollout is one ``jax.lax.scan``
(``harness.physics.heat1d`` + ``harness.solvers.scan``) and the
trajectory is time-first ``(n_steps+1, n_cells)``.
"""

from __future__ import annotations

from typing import Any, Mapping

import jax.numpy as jnp

from ..mesh.boundary import BoundaryCondition
from ..mesh.grid1d import Grid1D
from ..physics.heat1d import HeatEquation1D, simulate_heat_1d
from ..profiles import ApplicationProfile, get_profile
from ..registry import REGISTRIES
from .base import DesignSpace, EpisodeTrace, ProblemSpec

DESIGN_KEYS = ("alpha",)
HEAT1D_METRIC_KEYS = ("T_mean", "T_max", "steady_l2", "alpha")
HEAT1D_SCHEMA_VERSION = 1


class Heat1D:
    """1D slab heat problem (Dirichlet ends, explicit Euler in scan)."""

    def __init__(
        self,
        material: Any | None = None,
        profile: "str | ApplicationProfile" = "cpu",
        *,
        L: float = 1.0,
        n_cells: int = 32,
        dt: float = 0.001,
        t_end: float = 2.0,
        T_hot: float = 1.0,
        T_cold: float = 0.0,
        T_init: float = 0.0,
    ):
        self.material = material
        self.transport_provenance = "n/a"
        self.profile = get_profile(profile)
        self.L = float(L)
        self.n_cells = int(n_cells)
        self.dt = float(dt)
        self.t_end = float(t_end)
        self.T_hot = float(T_hot)
        self.T_cold = float(T_cold)
        self.T_init = float(T_init)
        self.grid = Grid1D.uniform(length=self.L, n_cells=self.n_cells)
        self.bc = BoundaryCondition(
            kind="dirichlet",
            value=jnp.array([self.T_hot, self.T_cold], dtype=jnp.float64),
        )
        # Steady profile at cell centers for the error metric (linear).
        self._steady = self.T_hot + (self.T_cold - self.T_hot) * (
            self.grid.centers / self.L
        )
        cfl = float(self.grid.dx_min ** 2) / (2.0 * 0.05)
        if self.dt > cfl:
            # Only fail for the worst-case bound, not the current alpha.
            import logging as _lg
            _lg.getLogger(__name__).debug("Heat1D dt=%g near CFL=%g", self.dt, cfl)
        self.spec = ProblemSpec(
            name="Heat1D-v0",
            kind="static",
            metric_keys=HEAT1D_METRIC_KEYS,
            schema_version=HEAT1D_SCHEMA_VERSION,
            spatial_dim=1,
            time_resolved=True,
            grid_type="uniform_rect",
        )
        self.design_space = DesignSpace(
            keys=DESIGN_KEYS,
            defaults={"alpha": 0.01},
            bounds={"alpha": (0.002, 0.05)},
        )
        from ..physics.adapters import ModelSpec  # noqa: WPS433
        self.model_spec = ModelSpec(
            name="heat1d",
            state_vars=("T",),
            units={"T": "degC", "alpha": "m2/s", "dt": "s"},
            dt_min_s=1e-6,
            dt_max_s=0.1,
            description="1-D heat equation (diffheat ladder, harness mesh)",
        )

    def _eqn(self, alpha) -> HeatEquation1D:
        return HeatEquation1D(grid=self.grid, bc=self.bc, alpha=alpha)

    def metrics_jax(self, design: Mapping[str, Any] | None = None):
        merged = self.design_space.merge(design)
        alpha = merged["alpha"]
        T0 = jnp.full((self.n_cells,), self.T_init, dtype=jnp.float64)
        traj = simulate_heat_1d(self._eqn(alpha), T0, (0.0, self.t_end), self.dt)
        final = traj[-1]
        t_mean = jnp.mean(final)
        t_max = jnp.max(final)
        l2 = jnp.mean((final - self._steady) ** 2)
        return {"T_mean": t_mean, "T_max": t_max, "steady_l2": l2, "alpha": alpha}

    def evaluate(self, design: Mapping[str, float] | None = None) -> dict[str, float]:
        # CFL guard at evaluation time (fail loudly, like forced_conv).
        merged = self.design_space.merge(design) if design else self.design_space.defaults
        alpha = float(merged["alpha"])
        cfl_limit = float(self.grid.dx_min ** 2) / (2.0 * alpha)
        if self.dt > cfl_limit:
            raise ValueError(
                f"Heat1D dt={self.dt} violates 1D CFL {cfl_limit:.3e} at alpha={alpha}"
            )
        return {k: float(v) for k, v in self.metrics_jax(design).items()}

    def rollout(self, design=None, controls=None, *, n_steps: int = 1) -> EpisodeTrace:
        del controls, n_steps
        merged = self.design_space.merge(design)
        alpha = merged["alpha"]
        T0 = jnp.full((self.n_cells,), self.T_init, dtype=jnp.float64)
        traj = simulate_heat_1d(self._eqn(alpha), T0, (0.0, self.t_end), self.dt)
        summary = {k: float(v) for k, v in self.metrics_jax(design).items()}
        return EpisodeTrace(series={"T_trajectory": traj}, summary=summary)


def build_heat1d(material: Any | None = None,
                 profile: "str | ApplicationProfile" = "cpu",
                 **kwargs) -> Heat1D:
    allowed = {"L", "n_cells", "dt", "t_end", "T_hot", "T_cold", "T_init"}
    ctor = {k: v for k, v in kwargs.items() if k in allowed}
    return Heat1D(material=material, profile=profile, **ctor)


REGISTRIES["envs"].register("Heat1D-v0", build_heat1d)

try:
    from ..physics.adapters import ModelSpec as _MS  # noqa: WPS433
    from ..physics.adapters import Fluxes as _Fluxes  # noqa: WPS433

    class _Heat1DAdapter:
        def __init__(self):
            self.spec = _MS(name="heat1d", state_vars=("T",),
                            units={"T": "degC"}, dt_min_s=1e-6, dt_max_s=0.1,
                            description="heat1d adapter (diffheat ladder)")

        def step(self, state, control, dt_s):
            return dict(state), _Fluxes()

        def metrics(self, trace):
            return dict(trace)

    try:
        REGISTRIES["models"].register("heat1d", _Heat1DAdapter)
    except ValueError:
        pass
except Exception:
    pass

__all__ = ["Heat1D", "build_heat1d", "DESIGN_KEYS", "HEAT1D_METRIC_KEYS"]

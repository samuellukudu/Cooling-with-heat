"""Heat2D-v0 — 2D heat equation as a harness Problem (diffheat 2D ladder).

Unit square ``[0,L]²``, time-resolved diffusion with Dirichlet
left/right and insulated top/bottom (matches ``examples/02`` + the
``Cloak2D`` BC pattern but with scalar diffusivity). Whole rollout is
one ``jax.lax.scan`` over ``harness.mesh.Grid2D`` +
``harness.operators.laplacian_2d`` with ghost-cell BCs; trajectory
is time-first ``(n_steps+1, nx, ny)`` (4D = 2 space + time).
"""

from __future__ import annotations

from typing import Any, Mapping

import jax.numpy as jnp

from ..mesh.boundary import BoundaryCondition2D
from ..mesh.grid2d import Grid2D
from ..physics.heat2d import HeatEquation2D, simulate_heat_2d
from ..profiles import ApplicationProfile, get_profile
from ..registry import REGISTRIES
from .base import DesignSpace, EpisodeTrace, ProblemSpec

DESIGN_KEYS = ("alpha",)
HEAT2D_METRIC_KEYS = ("T_mean", "T_max", "steady_l2", "alpha")
HEAT2D_SCHEMA_VERSION = 1


class Heat2D:
    """2D plate heat problem (scalar diffusivity, explicit Euler in scan)."""

    def __init__(
        self,
        material: Any | None = None,
        profile: "str | ApplicationProfile" = "cpu",
        *,
        L: float = 1.0,
        nx: int = 16,
        ny: int = 16,
        dt: float = 0.001,
        t_end: float = 5.0,
        T_hot: float = 1.0,
        T_cold: float = 0.0,
    ):
        self.material = material
        self.transport_provenance = "n/a"
        self.profile = get_profile(profile)
        self.L = float(L)
        self.nx, self.ny = int(nx), int(ny)
        self.dt = float(dt)
        self.t_end = float(t_end)
        self.T_hot = float(T_hot)
        self.T_cold = float(T_cold)
        self.grid = Grid2D.uniform(Lx=self.L, Ly=self.L, nx=self.nx, ny=self.ny)
        self.bc = BoundaryCondition2D(
            left={"kind": "dirichlet", "value": self.T_hot},
            right={"kind": "dirichlet", "value": self.T_cold},
            bottom={"kind": "neumann", "value": 0.0},
            top={"kind": "neumann", "value": 0.0},
        )
        # Linear steady state in x, uniform in y.
        self._steady = self.T_hot + (self.T_cold - self.T_hot) * (
            self.grid.X / self.L
        )
        self.spec = ProblemSpec(
            name="Heat2D-v0",
            kind="static",
            metric_keys=HEAT2D_METRIC_KEYS,
            schema_version=HEAT2D_SCHEMA_VERSION,
            spatial_dim=2,
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
            name="heat2d",
            state_vars=("T",),
            units={"T": "degC", "alpha": "m2/s"},
            dt_min_s=1e-6,
            dt_max_s=0.1,
            description="2-D heat equation (diffheat ladder, harness mesh)",
        )

    def _eqn(self, alpha) -> HeatEquation2D:
        return HeatEquation2D(grid=self.grid, bc=self.bc, alpha=alpha)

    def metrics_jax(self, design: Mapping[str, Any] | None = None):
        merged = self.design_space.merge(design)
        alpha = merged["alpha"]
        T0 = jnp.zeros((self.nx, self.ny), dtype=jnp.float64)
        traj = simulate_heat_2d(self._eqn(alpha), T0, (0.0, self.t_end), self.dt)
        final = traj[-1]
        t_mean = jnp.mean(final)
        t_max = jnp.max(final)
        l2 = jnp.mean((final - self._steady) ** 2)
        return {"T_mean": t_mean, "T_max": t_max, "steady_l2": l2, "alpha": alpha}

    def evaluate(self, design: Mapping[str, float] | None = None) -> dict[str, float]:
        merged = self.design_space.merge(design) if design else self.design_space.defaults
        alpha = float(merged["alpha"])
        dx_min = float(min(jnp.min(self.grid.dx), jnp.min(self.grid.dy)))
        cfl_limit = dx_min * dx_min / (4.0 * alpha)
        if self.dt > cfl_limit:
            raise ValueError(
                f"Heat2D dt={self.dt} violates 2D CFL {cfl_limit:.3e} at alpha={alpha}"
            )
        return {k: float(v) for k, v in self.metrics_jax(design).items()}

    def rollout(self, design=None, controls=None, *, n_steps: int = 1) -> EpisodeTrace:
        del controls, n_steps
        merged = self.design_space.merge(design)
        alpha = merged["alpha"]
        T0 = jnp.zeros((self.nx, self.ny), dtype=jnp.float64)
        traj = simulate_heat_2d(self._eqn(alpha), T0, (0.0, self.t_end), self.dt)
        summary = {k: float(v) for k, v in self.metrics_jax(design).items()}
        return EpisodeTrace(series={"T_trajectory": traj}, summary=summary)


def build_heat2d(material: Any | None = None,
                 profile: "str | ApplicationProfile" = "cpu",
                 **kwargs) -> Heat2D:
    allowed = {"L", "nx", "ny", "dt", "t_end", "T_hot", "T_cold"}
    ctor = {k: v for k, v in kwargs.items() if k in allowed}
    return Heat2D(material=material, profile=profile, **ctor)


REGISTRIES["envs"].register("Heat2D-v0", build_heat2d)

try:
    from ..physics.adapters import Fluxes as _Fluxes  # noqa: WPS433
    from ..physics.adapters import ModelSpec as _MS  # noqa: WPS433

    class _Heat2DAdapter:
        def __init__(self):
            self.spec = _MS(name="heat2d", state_vars=("T",),
                            units={"T": "degC"}, dt_min_s=1e-6, dt_max_s=0.1,
                            description="heat2d adapter (diffheat ladder)")

        def step(self, state, control, dt_s):
            return dict(state), _Fluxes()

        def metrics(self, trace):
            return dict(trace)

    try:
        REGISTRIES["models"].register("heat2d", _Heat2DAdapter)
    except ValueError:
        pass
except Exception:
    pass

__all__ = ["Heat2D", "build_heat2d", "DESIGN_KEYS", "HEAT2D_METRIC_KEYS"]

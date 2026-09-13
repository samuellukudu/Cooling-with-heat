"""Heat3D-v0 — 3D heat equation as a harness Problem (diffheat 3D ladder).

Cube ``[0,L]³``, slab heating (hot ``x=0`` wall, cold ``x=L`` wall,
insulated ``y/z`` faces — the extruded ``Heat1D/2D`` ladder) with
``save_every`` for memory. Also supports the ``examples/03`` hot-core
mode via ``core_radius``. Time-resolved diffusion over
``harness.mesh.Grid3D`` (``(nx, ny, nz)`` fields) and the 4th dim time:
trajectory is ``(n_saved+1, nx, ny, nz)`` time-first.
"""

from __future__ import annotations

from typing import Any, Mapping

import jax.numpy as jnp

from ..mesh.boundary import BoundaryCondition3D
from ..mesh.grid3d import Grid3D
from ..physics.heat3d import HeatEquation3D, simulate_heat_3d
from ..profiles import ApplicationProfile, get_profile
from ..registry import REGISTRIES
from .base import DesignSpace, EpisodeTrace, ProblemSpec

DESIGN_KEYS = ("alpha",)
HEAT3D_METRIC_KEYS = ("T_mean", "T_max", "steady_l2", "alpha")
HEAT3D_SCHEMA_VERSION = 1


class Heat3D:
    """3D cube heat problem (scalar diffusivity, save_every-aware scan)."""

    def __init__(
        self,
        material: Any | None = None,
        profile: "str | ApplicationProfile" = "cpu",
        *,
        L: float = 1.0,
        nx: int = 12,
        ny: int = 12,
        nz: int = 12,
        dt: float = 0.002,
        t_end: float = 1.0,
        save_every: int = 5,
        T_hot: float = 1.0,
        T_cold: float = 0.0,
        mode: str = "slab",
        core_radius: float = 0.2,
        T_core: float = 1.0,
    ):
        if mode not in ("slab", "core"):
            raise ValueError(f"Heat3D mode must be 'slab' or 'core', got {mode!r}")
        self.material = material
        self.transport_provenance = "n/a"
        self.profile = get_profile(profile)
        self.L = float(L)
        self.nx, self.ny, self.nz = int(nx), int(ny), int(nz)
        self.dt = float(dt)
        self.t_end = float(t_end)
        self.save_every = int(save_every)
        self.T_hot = float(T_hot)
        self.T_cold = float(T_cold)
        self.mode = str(mode)
        self.core_radius = float(core_radius)
        self.T_core = float(T_core)
        self.grid = Grid3D.uniform(
            Lx=self.L, Ly=self.L, Lz=self.L,
            nx=self.nx, ny=self.ny, nz=self.nz,
        )
        if self.mode == "slab":
            # Homologous ladder: hot left (x=0), cold right (x=L), y/z insulated.
            self.bc = BoundaryCondition3D(
                left={"kind": "dirichlet", "value": self.T_hot},
                right={"kind": "dirichlet", "value": self.T_cold},
                bottom={"kind": "neumann", "value": 0.0},
                top={"kind": "neumann", "value": 0.0},
                front={"kind": "neumann", "value": 0.0},
                back={"kind": "neumann", "value": 0.0},
            )
            self._steady = self.T_hot + (self.T_cold - self.T_hot) * (
                self.grid.X / self.L
            )
        else:
            cold = {"kind": "dirichlet", "value": 0.0}
            self.bc = BoundaryCondition3D(
                left=cold, right=cold, bottom=cold, top=cold, front=cold, back=cold,
            )
            self._steady = jnp.zeros((self.nx, self.ny, self.nz))
        self.spec = ProblemSpec(
            name="Heat3D-v0",
            kind="static",
            metric_keys=HEAT3D_METRIC_KEYS,
            schema_version=HEAT3D_SCHEMA_VERSION,
            spatial_dim=3,
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
            name="heat3d",
            state_vars=("T",),
            units={"T": "degC", "alpha": "m2/s"},
            dt_min_s=1e-6,
            dt_max_s=0.1,
            description="3-D heat equation (diffheat ladder, save_every)",
        )

    def _eqn(self, alpha) -> HeatEquation3D:
        return HeatEquation3D(grid=self.grid, bc=self.bc, alpha=alpha)

    def _initial(self):
        if self.mode == "core":
            X, Y, Z = self.grid.X, self.grid.Y, self.grid.Z
            r2 = (X - self.L / 2.0) ** 2 + (Y - self.L / 2.0) ** 2 + (Z - self.L / 2.0) ** 2
            return jnp.where(r2 <= self.core_radius ** 2, self.T_core, 0.0).astype(jnp.float64)
        return jnp.zeros((self.nx, self.ny, self.nz), dtype=jnp.float64)

    def metrics_jax(self, design: Mapping[str, Any] | None = None):
        merged = self.design_space.merge(design)
        alpha = merged["alpha"]
        T0 = self._initial()
        traj = simulate_heat_3d(
            self._eqn(alpha), T0, (0.0, self.t_end), self.dt,
            save_every=self.save_every,
        )
        final = traj[-1]
        t_mean = jnp.mean(final)
        t_max = jnp.max(final)
        # Steady error uses the slab steady; core mode steady is 0 (cooling to walls).
        l2 = jnp.mean((final - self._steady) ** 2)
        return {"T_mean": t_mean, "T_max": t_max, "steady_l2": l2, "alpha": alpha}

    def evaluate(self, design: Mapping[str, float] | None = None) -> dict[str, float]:
        merged = self.design_space.merge(design) if design else self.design_space.defaults
        alpha = float(merged["alpha"])
        dx_min = float(min(
            jnp.min(self.grid.dx), jnp.min(self.grid.dy), jnp.min(self.grid.dz)
        ))
        cfl_limit = dx_min * dx_min / (6.0 * alpha)
        if self.dt > cfl_limit:
            raise ValueError(
                f"Heat3D dt={self.dt} violates 3D CFL {cfl_limit:.3e} at alpha={alpha}"
            )
        return {k: float(v) for k, v in self.metrics_jax(design).items()}

    def rollout(self, design=None, controls=None, *, n_steps: int = 1) -> EpisodeTrace:
        del controls, n_steps
        merged = self.design_space.merge(design)
        alpha = merged["alpha"]
        T0 = self._initial()
        traj = simulate_heat_3d(
            self._eqn(alpha), T0, (0.0, self.t_end), self.dt,
            save_every=self.save_every,
        )
        summary = {k: float(v) for k, v in self.metrics_jax(design).items()}
        return EpisodeTrace(series={"T_trajectory": traj}, summary=summary)


def build_heat3d(material: Any | None = None,
                 profile: "str | ApplicationProfile" = "cpu",
                 **kwargs) -> Heat3D:
    allowed = {"L", "nx", "ny", "nz", "dt", "t_end", "save_every",
               "T_hot", "T_cold", "mode", "core_radius", "T_core"}
    ctor = {k: v for k, v in kwargs.items() if k in allowed}
    return Heat3D(material=material, profile=profile, **ctor)


REGISTRIES["envs"].register("Heat3D-v0", build_heat3d)

try:
    from ..physics.adapters import Fluxes as _Fluxes  # noqa: WPS433
    from ..physics.adapters import ModelSpec as _MS  # noqa: WPS433

    class _Heat3DAdapter:
        def __init__(self):
            self.spec = _MS(name="heat3d", state_vars=("T",),
                            units={"T": "degC"}, dt_min_s=1e-6, dt_max_s=0.1,
                            description="heat3d adapter (diffheat ladder)")

        def step(self, state, control, dt_s):
            return dict(state), _Fluxes()

        def metrics(self, trace):
            return dict(trace)

    try:
        REGISTRIES["models"].register("heat3d", _Heat3DAdapter)
    except ValueError:
        pass
except Exception:
    pass

__all__ = ["Heat3D", "build_heat3d", "DESIGN_KEYS", "HEAT3D_METRIC_KEYS"]

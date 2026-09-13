"""Dimensional experiments harness: 1D/2D/3D + 4D time suites (DESIGN §12).

Provides a uniform battery over the dimensional ladder — the same
``make``/``Objective``/``Backend`` loop, just filtered by
``ProblemSpec.spatial_dim`` / ``time_resolved`` / ``grid_type``:

- :func:`list_by_dim`  — registry filter by dimension;
- :func:`dim_battery`  — evaluate the same design axis across 1D/2D/3D
  (e.g. diffusivity ``alpha``) and return per-dim metrics;
- :func:`parity_vs_diffheat` — head-to-head harness vs frozen diffheat
  on the shared heat problem per dim (< 1e-4, same note as the T-A4
  parity test: float64 vs diffheat's device dtype).

All helpers are import-safe (no gym/backend required) and deterministic
at fixed grid sizes. The harness GUI and notebooks call them as the
R&D-lab harness for mesh-refinement, diffusivity-sweep and gradient/
backend comparisons.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

import harness
from harness.envs.base import Objective, ProblemSpec
from harness.registry import REGISTRIES

jnp  # keep linter honest

# Preferred demo names in ladder order (0D lumped → 1D → 2D → 3D).
LADDER_ENVS: tuple[str, ...] = (
    "Cycle0D-v0",
    "Heat1D-v0",
    "Heat2D-v0",
    "Heat3D-v0",
)

# Canonical ladder for heat (diffheat-mirrored): same alpha sweep works
# across 1/2/3D because the thermal problem is homologous; Boussinesq /
# forced_conv ladders have their own sweeps and stay separate.
HEAT_LADDER = ("Heat1D-v0", "Heat2D-v0", "Heat3D-v0")


def list_by_dim(
    spatial_dim: int | tuple[int, ...] | None = None,
    *,
    time_resolved: bool | None = None,
    grid_type: str | None = None,
) -> list[str]:
    """Registry names filtered by dimensional metadata.

    Args:
        spatial_dim: single dim or tuple (0/1/2/3), or None for all.
        time_resolved: if set, only envs with that 4D flag.
        grid_type: if set, only envs with that grid type.
    """
    wanted = None if spatial_dim is None else (
        (spatial_dim,) if isinstance(spatial_dim, int) else tuple(spatial_dim)
    )
    out: list[str] = []
    # Probe each registered env with default construction (cheap — grids ≤ 12³).
    # Envs that need mandatory schedules (TwoBedSchedule) are expected to
    # fail probe and are excluded from ladder scans that use them.
    for name in REGISTRIES["envs"].names():
        # Try to instantiate, but don't crash the whole listing if one
        # env needs extra args (e.g. TwoBedSchedule needs a schedule).
        try:
            prob = harness.make(name)
        except Exception:
            continue
        spec: ProblemSpec = getattr(prob, "spec", None)  # type: ignore[assignment]
        if spec is None:
            continue
        if wanted is not None and spec.spatial_dim not in wanted:
            continue
        if time_resolved is not None and spec.time_resolved != time_resolved:
            continue
        if grid_type is not None and spec.grid_type != grid_type:
            continue
        out.append(name)
    return sorted(out)


def dim_summary_table(
    spatial_dims: tuple[int, ...] = (0, 1, 2, 3),
) -> list[dict[str, Any]]:
    """One row per registered env: ``{name, spatial_dim, time_resolved, grid_type, kind}``."""
    rows: list[dict[str, Any]] = []
    for name in REGISTRIES["envs"].names():
        try:
            prob = harness.make(name)
        except Exception:
            continue
        spec = getattr(prob, "spec", None)
        if spec is None or spec.spatial_dim not in spatial_dims:
            continue
        rows.append(
            {
                "name": name,
                "spatial_dim": spec.spatial_dim,
                "time_resolved": spec.time_resolved,
                "grid_type": spec.grid_type,
                "kind": spec.kind,
                "metric_keys": spec.metric_keys,
            }
        )
    # Sort by (dim, name) to keep the ladder readable.
    rows.sort(key=lambda r: (r["spatial_dim"], r["name"]))
    return rows


def dim_battery(
    env_names: tuple[str, ...] = HEAT_LADDER,
    design: dict[str, float] | None = None,
    *,
    profile: str = "cpu",
    env_kwargs: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate ``design`` on each env in ``env_names`` and return per-env metrics.

    The harness equivalent of the diffheat ``examples/01-03`` sweep —
    same diffusivity across 1D/2D/3D shows diffusion-time scaling vs mesh
    cost under one API. Use ``env_kwargs[name]`` to pass per-env
    overrides (e.g. ``{"Heat3D-v0": {"nx": 8}}`` for a quick smoke run).
    """
    design = design or {"alpha": 0.01}
    env_kwargs = env_kwargs or {}
    rows: list[dict[str, Any]] = []
    for name in env_names:
        kwargs = dict(env_kwargs.get(name, {}))
        prob = harness.make(name, profile=profile, **kwargs)
        # Some probes (Heat3D) carry save_every; not all expose the same name.
        m = prob.evaluate(design)
        rows.append(
            {
                "env": name,
                "spatial_dim": prob.spec.spatial_dim,
                "grid_type": prob.spec.grid_type,
                "time_resolved": prob.spec.time_resolved,
                "metrics": m,
                "trajectory_shape": prob.rollout(design).series.get(
                    "T_trajectory", prob.rollout(design).series.get("trajectory", None)
                ).shape
                if prob.rollout(design).series
                else None,
            }
        )
    return rows


def grad_vs_fd(
    env_name: str,
    key: str = "alpha",
    eps: float = 1e-4,
    design: dict[str, float] | None = None,
) -> dict[str, float]:
    """V5-style gradient check on ``env_name``'s scalar metric derived from ``key``.

    Finite-difference vs ``jax.grad`` on ``T_mean`` (heat ladder) or the
    first metric. Returns ``{ad, fd, rel}``.
    """
    prob = harness.make(env_name)
    base = dict(design or {"alpha": 0.01})

    def score(x):
        m = prob.metrics_jax({key: x})
        # Heat ladder has T_mean; fall back to first metric.
        if "T_mean" in m:
            return m["T_mean"]
        first = prob.spec.metric_keys[0]
        return m[first]

    gjit = jax.jit(jax.grad(score))
    x0 = float(base[key])
    g_ad = float(gjit(jnp.asarray(x0)))
    f_lo = float(score(jnp.asarray(x0 - eps)))
    f_hi = float(score(jnp.asarray(x0 + eps)))
    g_fd = (f_hi - f_lo) / (2 * eps)
    denom = max(abs(g_fd), 1e-8)
    return {"ad": g_ad, "fd": g_fd, "rel": abs(g_ad - g_fd) / denom, "x0": x0}


def parity_vs_diffheat(
    spatial_dim: int,
    *,
    n_cells: int = 12,
    alpha: float = 0.01,
    dt: float = 0.002,
    n_steps: int = 20,
):
    """Harness vs diffheat trajectory parity for the shared heat problem.

    Only dims 1/2 are exercised here; 3D parity is covered by the dedicated
    ``test_heat*_parity`` tests which hold the save_every convention.

    Imported lazily via :mod:`importlib` so the harness stays free of a
    hard ``diffheat`` import (see :mod:`tests.harness.test_import_hygiene`).

    Returns ``(max_abs_err, harness_traj, ref_traj)``.
    """
    import importlib as _il  # noqa: WPS433

    dh = _il.import_module("diffheat")
    if spatial_dim == 1:
        from harness.mesh import Grid1D  # noqa: WPS433
        from harness.operators import laplacian_1d  # noqa: WPS433

        grid_h = Grid1D.uniform(length=1.0, n_cells=n_cells)
        DHGrid1D = getattr(dh, "Grid1D")
        grid_d = DHGrid1D.uniform(length=1.0, n_cells=n_cells)
        BC_h_cls = getattr(_il.import_module("harness.mesh.boundary"), "BoundaryCondition")
        bc_h = BC_h_cls(kind="dirichlet", value=jnp.array([1.0, 0.0]))
        bc_d = getattr(dh, "BoundaryCondition")(
            kind="dirichlet", value=jnp.array([1.0, 0.0])
        )
        _ = laplacian_1d(jnp.zeros(n_cells), grid_h)

        from harness.physics.heat1d import HeatEquation1D as HEq
        from harness.physics.heat1d import simulate_heat_1d

        eqn_h = HEq(grid=grid_h, bc=bc_h, alpha=alpha)
        eqn_d = getattr(dh, "HeatEquation1D")(grid=grid_d, bc=bc_d, alpha=alpha)
        T0_h = jnp.zeros(n_cells)
        T0_d = jnp.zeros(n_cells)
        traj_h = simulate_heat_1d(eqn_h, T0_h, (0.0, dt * n_steps), dt)
        traj_d = getattr(dh, "solve_heat_1d")(eqn_d, T0_d, (0.0, dt * n_steps), dt)
        err = float(jnp.max(jnp.abs(traj_h - traj_d)))
        return err, traj_h, traj_d

    if spatial_dim == 2:
        from harness.mesh import Grid2D  # noqa: WPS433

        L = 1.0
        grid_h = Grid2D.uniform(Lx=L, Ly=L, nx=n_cells, ny=n_cells)
        DHGrid2D = getattr(dh, "Grid2D")
        grid_d = DHGrid2D.uniform(Lx=L, Ly=L, nx=n_cells, ny=n_cells)
        bc_h = getattr(
            _il.import_module("harness.mesh.boundary"), "BoundaryCondition2D"
        )(
            left={"kind": "dirichlet", "value": 1.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "neumann", "value": 0.0},
            top={"kind": "neumann", "value": 0.0},
        )
        bc_d = getattr(dh, "BoundaryCondition2D")(
            left={"kind": "dirichlet", "value": 1.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "neumann", "value": 0.0},
            top={"kind": "neumann", "value": 0.0},
        )
        from harness.physics.heat2d import HeatEquation2D as HEq2
        from harness.physics.heat2d import simulate_heat_2d

        eqn_h = HEq2(grid=grid_h, bc=bc_h, alpha=alpha)
        eqn_d = getattr(dh, "HeatEquation2D")(grid=grid_d, bc=bc_d, alpha=alpha)
        T0_h = jnp.zeros((n_cells, n_cells))
        T0_d = jnp.zeros((n_cells, n_cells))
        traj_h = simulate_heat_2d(eqn_h, T0_h, (0.0, dt * n_steps), dt)
        traj_d = getattr(dh, "solve_heat_2d")(eqn_d, T0_d, (0.0, dt * n_steps), dt)
        err = float(jnp.max(jnp.abs(traj_h - traj_d)))
        return err, traj_h, traj_d

    raise ValueError(f"parity_vs_diffheat only supports 1/2, got {spatial_dim!r}")


__all__ = [
    "HEAT_LADDER",
    "LADDER_ENVS",
    "dim_battery",
    "dim_summary_table",
    "grad_vs_fd",
    "list_by_dim",
    "parity_vs_diffheat",
]

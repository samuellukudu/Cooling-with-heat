"""3D heat-equation problem + solver (DESIGN §4).

``dT/dt = alpha·∇²T + S(x, y, z, t)`` on a :class:`harness.mesh.Grid3D`
(``(nx, ny, nz)`` fields) with ghost-cell BCs. Mirrored from the frozen
``diffheat`` 3D hot path without importing it. ``save_every`` bounds
trajectory memory (the 3D rule: store every N-th frame + the initial).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import jax.numpy as jnp

from ..mesh.boundary import BoundaryCondition3D, apply_boundary_conditions_3d
from ..mesh.grid3d import Grid3D
from ..operators.laplacian import laplacian_3d
from ..solvers.scan import solve_3d
from ..solvers.stability import check_cfl_3d, max_timestep_3d

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HeatEquation3D:
    """Complete 3D heat problem definition.

    Args:
        grid: 3D spatial grid. bc: six-face boundary conditions.
        alpha: diffusivity (scalar or ``(nx, ny, nz)`` field).
        source: optional ``S(X, Y, Z, t)`` with ``(nx, ny, nz)`` coords.
    """

    grid: Grid3D
    bc: BoundaryCondition3D
    alpha: float | jnp.ndarray
    source: Optional[Callable] = None


def heat_rhs_3d(T, grid: Grid3D, t, eqn: HeatEquation3D):
    """Semi-discrete RHS ``dT/dt`` for :class:`HeatEquation3D`."""
    L_T, b = apply_boundary_conditions_3d(
        lambda x: laplacian_3d(x, grid), grid, eqn.bc, T
    )
    dT = eqn.alpha * (L_T + b)
    if eqn.source is not None:
        dT = dT + eqn.source(grid.X, grid.Y, grid.Z, t)
    return dT


def simulate_heat_3d(
    eqn: HeatEquation3D,
    T0: jnp.ndarray,
    t_span: tuple[float, float],
    dt: float,
    save_every: int = 1,
) -> jnp.ndarray:
    """Roll out the 3D heat equation (explicit Euler in scan).

    Returns ``(n_saved+1, nx, ny, nz)`` with ``n_saved = n_steps//save_every``.
    """
    try:
        if not check_cfl_3d(eqn.grid, eqn.alpha, float(dt)):
            _logger.warning(
                "dt=%g exceeds 3D CFL limit %g",
                float(dt), float(max_timestep_3d(eqn.grid, eqn.alpha)),
            )
    except Exception:
        pass

    def rhs_fn(T, grid, t, params):
        del params
        return heat_rhs_3d(T, grid, t, eqn)

    return solve_3d(rhs_fn, jnp.asarray(T0, dtype=jnp.float64),
                    eqn.grid, t_span, float(dt), save_every=save_every)


solve_heat_3d = simulate_heat_3d

__all__ = ["HeatEquation3D", "heat_rhs_3d", "simulate_heat_3d", "solve_heat_3d"]

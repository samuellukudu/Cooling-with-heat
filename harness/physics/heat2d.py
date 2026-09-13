"""2D heat-equation problem + solver (DESIGN §4).

``dT/dt = alpha·∇²T + S(x, y, t)`` on a :class:`harness.mesh.Grid2D`
(``(nx, ny)`` fields, ``indexing="ij"``) with ghost-cell BCs. Mirrored
from the frozen ``diffheat`` 2D hot path without importing it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import jax.numpy as jnp

from ..mesh.boundary import BoundaryCondition2D, apply_boundary_conditions_2d
from ..mesh.grid2d import Grid2D
from ..operators.laplacian import laplacian_2d
from ..solvers.scan import solve_2d
from ..solvers.stability import check_cfl_2d, max_timestep_2d

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HeatEquation2D:
    """Complete 2D heat problem definition.

    Args:
        grid: 2D spatial grid. bc: four-edge boundary conditions.
        alpha: diffusivity (scalar or ``(nx, ny)`` field).
        source: optional ``S(X, Y, t)`` with ``X, Y`` shape ``(nx, ny)``.
    """

    grid: Grid2D
    bc: BoundaryCondition2D
    alpha: float | jnp.ndarray
    source: Optional[Callable] = None


def heat_rhs_2d(T, grid: Grid2D, t, eqn: HeatEquation2D):
    """Semi-discrete RHS ``dT/dt`` for :class:`HeatEquation2D`."""
    L_T, b = apply_boundary_conditions_2d(
        lambda x: laplacian_2d(x, grid), grid, eqn.bc, T
    )
    dT = eqn.alpha * (L_T + b)
    if eqn.source is not None:
        dT = dT + eqn.source(grid.X, grid.Y, t)
    return dT


def simulate_heat_2d(
    eqn: HeatEquation2D,
    T0: jnp.ndarray,
    t_span: tuple[float, float],
    dt: float,
) -> jnp.ndarray:
    """Roll out the 2D heat equation (explicit Euler in scan).

    Returns ``(n_steps+1, nx, ny)`` trajectory, frame 0 = ``T0``.
    """
    try:
        if not check_cfl_2d(eqn.grid, eqn.alpha, float(dt)):
            _logger.warning(
                "dt=%g exceeds 2D CFL limit %g",
                float(dt), float(max_timestep_2d(eqn.grid, eqn.alpha)),
            )
    except Exception:
        pass

    def rhs_fn(T, grid, t, params):
        del params
        return heat_rhs_2d(T, grid, t, eqn)

    return solve_2d(rhs_fn, jnp.asarray(T0, dtype=jnp.float64),
                    eqn.grid, t_span, float(dt))


solve_heat_2d = simulate_heat_2d

__all__ = ["HeatEquation2D", "heat_rhs_2d", "simulate_heat_2d", "solve_heat_2d"]

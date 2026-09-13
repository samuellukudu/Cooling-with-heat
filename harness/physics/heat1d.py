"""1D heat-equation problem + solver (DESIGN §4).

``dT/dt = alpha·d²T/dx² + S(x, t)`` on a :class:`harness.mesh.Grid1D`
with ghost-cell BCs. Mirrored from the frozen ``diffheat`` 1D hot path
without importing it. Whole rollout is jittable/differentiable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import jax.numpy as jnp

from ..mesh.boundary import BoundaryCondition, apply_boundary_conditions_1d
from ..mesh.grid1d import Grid1D
from ..operators.laplacian import laplacian_1d
from ..solvers.scan import solve_1d
from ..solvers.stability import check_cfl_1d, max_timestep_1d

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HeatEquation1D:
    """Complete 1D heat problem definition.

    Args:
        grid: 1D spatial grid. bc: boundary conditions.
        alpha: thermal diffusivity (scalar or ``(n_cells,)`` field).
        source: optional ``S(x, t)`` with ``x`` shape ``(n_cells,)``.
    """

    grid: Grid1D
    bc: BoundaryCondition
    alpha: float | jnp.ndarray
    source: Optional[Callable[[jnp.ndarray, float], jnp.ndarray]] = None


def heat_rhs_1d(T, grid: Grid1D, t, eqn: HeatEquation1D):
    """Semi-discrete RHS ``dT/dt`` for :class:`HeatEquation1D`."""
    L_T, b = apply_boundary_conditions_1d(
        lambda x: laplacian_1d(x, grid), grid, eqn.bc, T
    )
    dT = eqn.alpha * (L_T + b)
    if eqn.source is not None:
        dT = dT + eqn.source(grid.centers, t)
    return dT


def simulate_heat_1d(
    eqn: HeatEquation1D,
    T0: jnp.ndarray,
    t_span: tuple[float, float],
    dt: float,
) -> jnp.ndarray:
    """Roll out the 1D heat equation (explicit Euler in scan).

    Returns ``(n_steps+1, n_cells)`` trajectory, frame 0 = ``T0``.
    Warns (does not raise) when ``dt`` violates the CFL bound.
    """
    try:
        if not check_cfl_1d(eqn.grid, eqn.alpha, float(dt)):
            _logger.warning(
                "dt=%g exceeds 1D CFL limit %g",
                float(dt), float(max_timestep_1d(eqn.grid, eqn.alpha)),
            )
    except Exception:
        pass

    def rhs_fn(T, grid, t, params):
        del params
        return heat_rhs_1d(T, grid, t, eqn)

    return solve_1d(rhs_fn, jnp.asarray(T0, dtype=jnp.float64),
                    eqn.grid, t_span, float(dt))


# Back-compat alias matching the diffheat solver name.
solve_heat_1d = simulate_heat_1d

__all__ = ["HeatEquation1D", "heat_rhs_1d", "simulate_heat_1d", "solve_heat_1d"]

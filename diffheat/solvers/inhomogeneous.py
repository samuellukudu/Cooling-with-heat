# diffheat/solvers/inhomogeneous.py
"""Convenience API for heat equation with inhomogeneous boundary conditions.

Implements the decomposition from Hancock §9:

    u = uE + v

where:
- uE is the steady-state (equilibrium) solution satisfying the
  inhomogeneous BCs,
- v is the transient solution with homogeneous BCs and modified
  initial condition v₀ = u₀ − uE.

This avoids needing to manually set up the split.
"""
import jax.numpy as jnp

from ..mesh.boundary import BoundaryCondition2D, BoundaryCondition3D
from ..physics.heat2d import HeatEquation2D
from ..physics.heat3d import HeatEquation3D
from .scan import solve_heat_2d, solve_heat_3d
from .steady_state import solve_steady_state_2d, solve_steady_state_3d


def _make_homogeneous_bc_2d(bc: BoundaryCondition2D) -> BoundaryCondition2D:
    """Return a copy of *bc* with all boundary values set to 0 (homogeneous)."""
    return BoundaryCondition2D(
        left={**bc.left, "value": 0.0},
        right={**bc.right, "value": 0.0},
        bottom={**bc.bottom, "value": 0.0},
        top={**bc.top, "value": 0.0},
    )


def _make_homogeneous_bc_3d(bc: BoundaryCondition3D) -> BoundaryCondition3D:
    """Return a copy of *bc* with all boundary values set to 0 (homogeneous)."""
    return BoundaryCondition3D(
        left={**bc.left, "value": 0.0},
        right={**bc.right, "value": 0.0},
        bottom={**bc.bottom, "value": 0.0},
        top={**bc.top, "value": 0.0},
        front={**bc.front, "value": 0.0},
        back={**bc.back, "value": 0.0},
    )


def solve_heat_inhomogeneous_2d(
    eqn: HeatEquation2D,
    T0: jnp.ndarray,
    t_span: tuple[float, float],
    dt: float,
) -> jnp.ndarray:
    """Solve the 2D heat equation with inhomogeneous BCs.

    Uses the decomposition  u = uE + v  (§9 of Hancock) to reduce
    the inhomogeneous problem to a homogeneous one:

    1. Compute equilibrium uE = steady-state solution of eqn.
    2. Form modified IC  v₀ = T0 − uE.
    3. Solve transient v with homogeneous BCs.
    4. Return  u(t) = v(t) + uE.

    Args:
        eqn: 2D heat equation description (may have inhomogeneous BCs).
        T0: (nx, ny) initial temperature field.
        t_span: (t_start, t_end) simulation time range.
        dt: Time step size.

    Returns:
        (n_steps+1, nx, ny) temperature trajectory.  First frame is T0.
    """
    # 1. Equilibrium with the given (possibly inhomogeneous) BCs
    uE = solve_steady_state_2d(eqn)

    # 2. Homogeneous-BC copy of the equation
    bc_homog = _make_homogeneous_bc_2d(eqn.bc)
    eqn_homog = HeatEquation2D(
        grid=eqn.grid, bc=bc_homog, alpha=eqn.alpha, source=eqn.source
    )

    # 3. Transient solve with modified IC
    v0 = T0 - uE
    v_traj = solve_heat_2d(eqn_homog, v0, t_span, dt)

    # 4. Reconstruct full solution
    return v_traj + uE[jnp.newaxis, :, :]


def solve_heat_inhomogeneous_3d(
    eqn: HeatEquation3D,
    T0: jnp.ndarray,
    t_span: tuple[float, float],
    dt: float,
    save_every: int = 1,
) -> jnp.ndarray:
    """Solve the 3D heat equation with inhomogeneous BCs.

    Uses the same  u = uE + v  decomposition as the 2D version.

    Args:
        eqn: 3D heat equation description (may have inhomogeneous BCs).
        T0: (nx, ny, nz) initial temperature field.
        t_span: (t_start, t_end) simulation time range.
        dt: Time step size.
        save_every: Save a frame every this many steps.

    Returns:
        (n_saved+1, nx, ny, nz) temperature trajectory.  First frame is T0.
    """
    # 1. Equilibrium with the given (possibly inhomogeneous) BCs
    uE = solve_steady_state_3d(eqn)

    # 2. Homogeneous-BC copy of the equation
    bc_homog = _make_homogeneous_bc_3d(eqn.bc)
    eqn_homog = HeatEquation3D(
        grid=eqn.grid, bc=bc_homog, alpha=eqn.alpha, source=eqn.source
    )

    # 3. Transient solve with modified IC
    v0 = T0 - uE
    v_traj = solve_heat_3d(eqn_homog, v0, t_span, dt, save_every=save_every)

    # 4. Reconstruct full solution
    return v_traj + uE[jnp.newaxis, :, :, :]

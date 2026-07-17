# diffheat/solvers/steady_state.py
"""Steady-state Poisson and Laplace solvers for 1D, 2D, and 3D."""
import jax
import jax.numpy as jnp

from ..mesh.boundary import (
    apply_boundary_conditions_1d,
    apply_boundary_conditions_2d,
    apply_boundary_conditions_3d,
)
from ..operators import laplacian_1d, laplacian_2d, laplacian_3d
from ..physics import HeatEquation1D, HeatEquation2D, HeatEquation3D


def solve_steady_state_1d(eqn: HeatEquation1D) -> jnp.ndarray:
    """Solve the 1D steady-state heat equation (Laplace/Poisson equation).

    Solves:
        -L_mod(T) = b_source + S / alpha
    where L_mod(T) is the boundary-modified Laplacian operator.

    Args:
        eqn: The 1D heat equation description.

    Returns:
        (n_cells,) steady-state temperature field.
    """
    grid = eqn.grid
    bc = eqn.bc
    alpha = eqn.alpha

    # 1. Compute dummy boundaries to get constant b_source
    dummy_T = jnp.zeros(grid.n_cells)
    _, b_source = apply_boundary_conditions_1d(
        lambda x: laplacian_1d(x, grid), grid, bc, dummy_T
    )

    # 2. Evaluate heat source
    if eqn.source is not None:
        S = eqn.source(grid.centers, 0.0)
        rhs = b_source + S / alpha
    else:
        rhs = b_source

    # 3. Define linear operator A(T) = -L_mod(T)
    def A_op(T):
        L_T_mod, _ = apply_boundary_conditions_1d(
            lambda x: laplacian_1d(x, grid), grid, bc, T
        )
        return -L_T_mod

    # 4. Solve system A(T) = rhs using Conjugate Gradient
    T_init = jnp.zeros(grid.n_cells)
    T_steady, _ = jax.scipy.sparse.linalg.cg(A_op, rhs, x0=T_init)
    return T_steady


def solve_steady_state_2d(eqn: HeatEquation2D) -> jnp.ndarray:
    """Solve the 2D steady-state heat equation (Laplace/Poisson equation).

    Solves:
        -L_mod(T) = b_source + S / alpha
    where L_mod(T) is the boundary-modified Laplacian operator.

    Args:
        eqn: The 2D heat equation description.

    Returns:
        (nx, ny) steady-state temperature field.
    """
    grid = eqn.grid
    bc = eqn.bc
    alpha = eqn.alpha

    # 1. Compute dummy boundaries to get constant b_source
    dummy_T = jnp.zeros((grid.nx, grid.ny))
    _, b_source = apply_boundary_conditions_2d(
        lambda x: laplacian_2d(x, grid), grid, bc, dummy_T
    )

    # 2. Evaluate heat source
    if eqn.source is not None:
        S = eqn.source(grid.X.T, grid.Y.T, 0.0)
        rhs = b_source + S / alpha
    else:
        rhs = b_source

    # 3. Define linear operator A(T) = -L_mod(T)
    def A_op(T):
        L_T_mod, _ = apply_boundary_conditions_2d(
            lambda x: laplacian_2d(x, grid), grid, bc, T
        )
        return -L_T_mod

    # 4. Solve system A(T) = rhs using Conjugate Gradient
    T_init = jnp.zeros((grid.nx, grid.ny))
    T_steady, _ = jax.scipy.sparse.linalg.cg(A_op, rhs, x0=T_init)
    return T_steady


def solve_steady_state_3d(eqn: HeatEquation3D) -> jnp.ndarray:
    """Solve the 3D steady-state heat equation (Laplace/Poisson equation).

    Solves:
        -L_mod(T) = b_source + S / alpha
    where L_mod(T) is the boundary-modified Laplacian operator.

    Args:
        eqn: The 3D heat equation description.

    Returns:
        (nx, ny, nz) steady-state temperature field.
    """
    grid = eqn.grid
    bc = eqn.bc
    alpha = eqn.alpha

    # 1. Compute dummy boundaries to get constant b_source
    dummy_T = jnp.zeros((grid.nx, grid.ny, grid.nz))
    _, b_source = apply_boundary_conditions_3d(
        lambda x: laplacian_3d(x, grid), grid, bc, dummy_T
    )

    # 2. Evaluate heat source
    if eqn.source is not None:
        S = eqn.source(grid.X, grid.Y, grid.Z, 0.0)
        rhs = b_source + S / alpha
    else:
        rhs = b_source

    # 3. Define linear operator A(T) = -L_mod(T)
    def A_op(T):
        L_T_mod, _ = apply_boundary_conditions_3d(
            lambda x: laplacian_3d(x, grid), grid, bc, T
        )
        return -L_T_mod

    # 4. Solve system A(T) = rhs using Conjugate Gradient
    T_init = jnp.zeros((grid.nx, grid.ny, grid.nz))
    T_steady, _ = jax.scipy.sparse.linalg.cg(A_op, rhs, x0=T_init)
    return T_steady

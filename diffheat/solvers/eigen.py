# diffheat/solvers/eigen.py
"""Rayleigh Quotient and eigenvalue calculation using JAX."""
import jax
import jax.numpy as jnp

from ..mesh.boundary import (
    BoundaryCondition,
    BoundaryCondition2D,
    BoundaryCondition3D,
    apply_boundary_conditions_1d,
    apply_boundary_conditions_2d,
    apply_boundary_conditions_3d,
)
from ..mesh.grid1d import Grid1D
from ..mesh.grid2d import Grid2D
from ..mesh.grid3d import Grid3D
from ..operators import laplacian_1d, laplacian_2d, laplacian_3d


def rayleigh_quotient_1d(v: jnp.ndarray, grid: Grid1D) -> jnp.ndarray:
    """Compute the Rayleigh Quotient of a 1D field v with zero Dirichlet BCs.

    R(v) = (v, -L v) / (v, v)

    Args:
        v: (n_cells,) field at cell centers.
        grid: The 1D grid.

    Returns:
        Scalar Rayleigh Quotient.
    """
    bc_zero = BoundaryCondition(kind="dirichlet", value=jnp.zeros(2))
    L_mod, _ = apply_boundary_conditions_1d(
        lambda x: laplacian_1d(x, grid), grid, bc_zero, v
    )

    dx = grid.dx
    num = jnp.sum(v * (-L_mod) * dx)
    den = jnp.sum(v * v * dx)
    return num / den


def rayleigh_quotient_2d(v: jnp.ndarray, grid: Grid2D) -> jnp.ndarray:
    """Compute the Rayleigh Quotient of a 2D field v with zero Dirichlet BCs.

    R(v) = (v, -L v) / (v, v)

    Args:
        v: (nx, ny) field at cell centers.
        grid: The 2D grid.

    Returns:
        Scalar Rayleigh Quotient.
    """
    bc_zero = BoundaryCondition2D(
        left={"kind": "dirichlet", "value": 0.0},
        right={"kind": "dirichlet", "value": 0.0},
        bottom={"kind": "dirichlet", "value": 0.0},
        top={"kind": "dirichlet", "value": 0.0},
    )
    L_mod, _ = apply_boundary_conditions_2d(
        lambda x: laplacian_2d(x, grid), grid, bc_zero, v
    )

    dA = grid.dx[:, jnp.newaxis] * grid.dy[jnp.newaxis, :]
    num = jnp.sum(v * (-L_mod) * dA)
    den = jnp.sum(v * v * dA)
    return num / den


def rayleigh_quotient_3d(v: jnp.ndarray, grid: Grid3D) -> jnp.ndarray:
    """Compute the Rayleigh Quotient of a 3D field v with zero Dirichlet BCs.

    R(v) = (v, -L v) / (v, v)

    Args:
        v: (nx, ny, nz) field at cell centers.
        grid: The 3D grid.

    Returns:
        Scalar Rayleigh Quotient.
    """
    bc_zero = BoundaryCondition3D(
        left={"kind": "dirichlet", "value": 0.0},
        right={"kind": "dirichlet", "value": 0.0},
        bottom={"kind": "dirichlet", "value": 0.0},
        top={"kind": "dirichlet", "value": 0.0},
        front={"kind": "dirichlet", "value": 0.0},
        back={"kind": "dirichlet", "value": 0.0},
    )
    L_mod, _ = apply_boundary_conditions_3d(
        lambda x: laplacian_3d(x, grid), grid, bc_zero, v
    )

    dV = (
        grid.dx[:, jnp.newaxis, jnp.newaxis]
        * grid.dy[jnp.newaxis, :, jnp.newaxis]
        * grid.dz[jnp.newaxis, jnp.newaxis, :]
    )
    num = jnp.sum(v * (-L_mod) * dV)
    den = jnp.sum(v * v * dV)
    return num / den


def find_first_eigenvalue_1d(
    grid: Grid1D, max_iter: int = 20
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Find the smallest eigenvalue and corresponding eigenfunction of the 1D Laplacian.

    Uses inverse power iteration:
        w_k = (-L_mod)^(-1) v_k
        v_{k+1} = w_k / ||w_k||

    Args:
        grid: The 1D grid.
        max_iter: Maximum number of power iterations.

    Returns:
        (eigenvalue, eigenfunction) where eigenfunction has shape (n_cells,).
    """
    v = jnp.ones(grid.n_cells)
    dx = grid.dx
    v = v / jnp.sqrt(jnp.sum(v**2 * dx))

    bc_zero = BoundaryCondition(kind="dirichlet", value=jnp.zeros(2))

    def A_op(T):
        L_T_mod, _ = apply_boundary_conditions_1d(
            lambda x: laplacian_1d(x, grid), grid, bc_zero, T
        )
        return -L_T_mod

    for _ in range(max_iter):
        w, _ = jax.scipy.sparse.linalg.cg(A_op, v, x0=v)
        norm = jnp.sqrt(jnp.sum(w**2 * dx))
        v = w / norm

    eigenvalue = rayleigh_quotient_1d(v, grid)
    return eigenvalue, v


def find_first_eigenvalue_2d(
    grid: Grid2D, max_iter: int = 20
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Find the smallest eigenvalue and corresponding eigenfunction of the 2D Laplacian.

    Uses inverse power iteration.

    Args:
        grid: The 2D grid.
        max_iter: Maximum number of power iterations.

    Returns:
        (eigenvalue, eigenfunction) where eigenfunction has shape (nx, ny).
    """
    v = jnp.ones((grid.nx, grid.ny))
    dA = grid.dx[:, jnp.newaxis] * grid.dy[jnp.newaxis, :]
    v = v / jnp.sqrt(jnp.sum(v**2 * dA))

    bc_zero = BoundaryCondition2D(
        left={"kind": "dirichlet", "value": 0.0},
        right={"kind": "dirichlet", "value": 0.0},
        bottom={"kind": "dirichlet", "value": 0.0},
        top={"kind": "dirichlet", "value": 0.0},
    )

    def A_op(T):
        L_T_mod, _ = apply_boundary_conditions_2d(
            lambda x: laplacian_2d(x, grid), grid, bc_zero, T
        )
        return -L_T_mod

    for _ in range(max_iter):
        w, _ = jax.scipy.sparse.linalg.cg(A_op, v, x0=v)
        norm = jnp.sqrt(jnp.sum(w**2 * dA))
        v = w / norm

    eigenvalue = rayleigh_quotient_2d(v, grid)
    return eigenvalue, v


def find_first_eigenvalue_3d(
    grid: Grid3D, max_iter: int = 20
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Find the smallest eigenvalue and corresponding eigenfunction of the 3D Laplacian.

    Uses inverse power iteration.

    Args:
        grid: The 3D grid.
        max_iter: Maximum number of power iterations.

    Returns:
        (eigenvalue, eigenfunction) where eigenfunction has shape (nx, ny, nz).
    """
    v = jnp.ones((grid.nx, grid.ny, grid.nz))
    dV = (
        grid.dx[:, jnp.newaxis, jnp.newaxis]
        * grid.dy[jnp.newaxis, :, jnp.newaxis]
        * grid.dz[jnp.newaxis, jnp.newaxis, :]
    )
    v = v / jnp.sqrt(jnp.sum(v**2 * dV))

    bc_zero = BoundaryCondition3D(
        left={"kind": "dirichlet", "value": 0.0},
        right={"kind": "dirichlet", "value": 0.0},
        bottom={"kind": "dirichlet", "value": 0.0},
        top={"kind": "dirichlet", "value": 0.0},
        front={"kind": "dirichlet", "value": 0.0},
        back={"kind": "dirichlet", "value": 0.0},
    )

    def A_op(T):
        L_T_mod, _ = apply_boundary_conditions_3d(
            lambda x: laplacian_3d(x, grid), grid, bc_zero, T
        )
        return -L_T_mod

    for _ in range(max_iter):
        w, _ = jax.scipy.sparse.linalg.cg(A_op, v, x0=v)
        norm = jnp.sqrt(jnp.sum(w**2 * dV))
        v = w / norm

    eigenvalue = rayleigh_quotient_3d(v, grid)
    return eigenvalue, v


# ---------------------------------------------------------------------------
# Rayleigh quotient upper bounds via trial functions
# ---------------------------------------------------------------------------


def rayleigh_upper_bounds_1d(
    grid: Grid1D,
    trial_fns: list,
) -> jnp.ndarray:
    """Compute Rayleigh quotient upper bounds for a set of trial functions (1D).

    The Rayleigh quotient R(v) = (v, -L v) / (v, v) provides an upper
    bound for the first eigenvalue λ₁ for any trial function v that
    satisfies the boundary conditions.  The minimum R(v) across trial
    functions gives the tightest upper bound.

    This is useful for testing candidate eigenfunctions and for
    variational approximations.

    Args:
        grid: The 1D grid.
        trial_fns: List of callables ``f(grid) -> v`` that each return
                   a (n_cells,) trial function satisfying zero Dirichlet BCs.

    Returns:
        (len(trial_fns),) array of Rayleigh quotients, one per trial function.
    """
    values = []
    for fn in trial_fns:
        v = fn(grid)
        rq = rayleigh_quotient_1d(v, grid)
        values.append(rq)
    return jnp.array(values)


def rayleigh_upper_bounds_2d(
    grid: Grid2D,
    trial_fns: list,
) -> jnp.ndarray:
    """Compute Rayleigh quotient upper bounds for a set of trial functions (2D).

    The Rayleigh quotient R(v) = (v, -L v) / (v, v) provides an upper
    bound for the first eigenvalue λ₁ for any trial function v that
    satisfies the boundary conditions.  The minimum R(v) across trial
    functions gives the tightest upper bound.

    This is useful for testing candidate eigenfunctions and for
    variational approximations on rectangular domains.

    Args:
        grid: The 2D grid.
        trial_fns: List of callables ``f(grid) -> v`` that each return
                   a (nx, ny) trial function satisfying zero Dirichlet BCs.

    Returns:
        (len(trial_fns),) array of Rayleigh quotients, one per trial function.
    """
    values = []
    for fn in trial_fns:
        v = fn(grid)
        rq = rayleigh_quotient_2d(v, grid)
        values.append(rq)
    return jnp.array(values)


def rayleigh_upper_bounds_3d(
    grid: Grid3D,
    trial_fns: list,
) -> jnp.ndarray:
    """Compute Rayleigh quotient upper bounds for a set of trial functions (3D).

    The Rayleigh quotient R(v) = (v, -L v) / (v, v) provides an upper
    bound for the first eigenvalue λ₁ for any trial function v that
    satisfies the boundary conditions.  The minimum R(v) across trial
    functions gives the tightest upper bound.

    Args:
        grid: The 3D grid.
        trial_fns: List of callables ``f(grid) -> v`` that each return
                   a (nx, ny, nz) trial function satisfying zero Dirichlet BCs.

    Returns:
        (len(trial_fns),) array of Rayleigh quotients, one per trial function.
    """
    values = []
    for fn in trial_fns:
        v = fn(grid)
        rq = rayleigh_quotient_3d(v, grid)
        values.append(rq)
    return jnp.array(values)

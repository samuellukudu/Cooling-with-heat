# diffheat/operators/laplacian.py
"""Discrete Laplacian operators for 1D and 2D."""
import jax.numpy as jnp

from ..mesh.grid1d import Grid1D
from ..mesh.grid2d import Grid2D
from ..utils import array


def laplacian_1d(T: jnp.ndarray, grid: Grid1D) -> jnp.ndarray:
    """Compute the 1D Laplacian d^2T/dx^2 using centered finite differences.

    Uses ``jnp.roll`` — boundaries are implicitly periodic; correct them
    with ``apply_boundary_conditions_1d``.

    Args:
        T: (N,) field at cell centers.
        grid: The 1D grid.

    Returns:
        (N,) Laplacian at cell centers.
    """
    dx2 = grid.dx * grid.dx  # (N,)
    d2T_dx2 = (jnp.roll(T, -1) + jnp.roll(T, 1) - 2.0 * T) / dx2
    return d2T_dx2


def make_laplacian(grid: Grid1D) -> jnp.ndarray:
    """Build the (N, N) tridiagonal Laplacian matrix using centered finite differences.

    Interior: [1, -2, 1] / dx^2
    Boundaries: left unmodified (apply_boundary_conditions handles them).

    Args:
        grid: The 1D grid.

    Returns:
        (N, N) Laplacian matrix.
    """
    n = grid.n_cells
    dx = grid.dx

    # Diagonal: -2 / dx^2
    diag = array(-2.0 / (dx * dx))

    # Off-diagonals: 1 / dx^2
    # NOTE: this formula is valid only for uniform grids.
    # Non-uniform grids require a different finite-difference stencil.
    dx2_left = dx[:-1] * dx[:-1]
    dx2_right = dx[1:] * dx[1:]
    off_diag = array(2.0 / (dx2_left + dx2_right))

    L = jnp.diag(diag) + jnp.diag(off_diag, k=1) + jnp.diag(off_diag, k=-1)
    return L


def laplacian_2d(T: jnp.ndarray, grid: Grid2D) -> jnp.ndarray:
    """Compute the 2D Laplacian nabla^2 T = partial^2 T/partial x^2 + partial^2 T/partial y^2.

    Uses centered finite differences with a 5-point stencil:
        (T[i+1,j] + T[i-1,j] - 2*T[i,j]) / dx^2 +
        (T[i,j+1] + T[i,j-1] - 2*T[i,j]) / dy^2

    Args:
        T: (nx, ny) temperature field at cell centers.
        grid: The 2D grid.

    Returns:
        (nx, ny) Laplacian at cell centers.
    """
    dx2 = grid.dx * grid.dx  # (nx,)
    dy2 = grid.dy * grid.dy  # (ny,)

    # partial^2 T/partial x^2: along axis 0
    d2T_dx2 = (jnp.roll(T, -1, axis=0) + jnp.roll(T, 1, axis=0) - 2.0 * T) / dx2[:, jnp.newaxis]

    # partial^2 T/partial y^2: along axis 1
    d2T_dy2 = (jnp.roll(T, -1, axis=1) + jnp.roll(T, 1, axis=1) - 2.0 * T) / dy2[jnp.newaxis, :]

    return d2T_dx2 + d2T_dy2

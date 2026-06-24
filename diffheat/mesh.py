# diffheat/mesh.py
"""Grid and boundary condition definitions for 1D heat equation."""
from dataclasses import dataclass

import jax.numpy as jnp

from .utils import array


@dataclass(frozen=True)
class BoundaryCondition:
    """Boundary condition for the 1D heat equation.

    Args:
        kind: "dirichlet" (fixed temperature) or "neumann" (fixed flux/gradient).
        value: (2,) array — [left_value, right_value].
               For Dirichlet: prescribed temperature at each boundary.
               For Neumann: prescribed dT/dx at each boundary (positive = into domain).
    """
    kind: str
    value: jnp.ndarray

    def __post_init__(self):
        if self.kind not in ("dirichlet", "neumann"):
            raise ValueError(f"Unknown boundary kind: {self.kind}")
        if self.value.shape != (2,):
            raise ValueError(f"Boundary value must have shape (2,), got {self.value.shape}")


@dataclass(frozen=True)
class Grid1D:
    """Uniform 1D grid for finite difference discretization.

    Attributes:
        x: (N+1,) cell interface positions [x0, x1, ..., xN]
        centers: (N,) cell center positions
        dx: (N,) cell widths (uniform = length / n_cells for each cell)
        length: total domain length
        n_cells: number of cells N
    """
    x: jnp.ndarray
    centers: jnp.ndarray
    dx: jnp.ndarray
    length: float
    n_cells: int

    @classmethod
    def uniform(cls, length: float, n_cells: int) -> "Grid1D":
        """Create a uniformly spaced 1D grid.

        Args:
            length: Domain length (0 to length).
            n_cells: Number of interior cells.

        Returns:
            Grid1D with uniform spacing.
        """
        if length <= 0:
            raise ValueError(f"length must be positive, got {length}")
        if n_cells < 2:
            raise ValueError(f"n_cells must be at least 2, got {n_cells}")

        cell_width = length / n_cells
        x = array(jnp.linspace(0.0, length, n_cells + 1))          # interfaces
        centers = array(0.5 * (x[:-1] + x[1:]))                     # cell centers
        dx = array(jnp.full(n_cells, cell_width))                   # cell widths

        return cls(x=x, centers=centers, dx=dx, length=length, n_cells=n_cells)

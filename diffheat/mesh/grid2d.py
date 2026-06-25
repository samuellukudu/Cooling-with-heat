# diffheat/mesh/grid2d.py
"""2D uniform rectangular grid for finite difference discretization."""
from dataclasses import dataclass

import jax.numpy as jnp

from ..utils import array


@dataclass(frozen=True)
class Grid2D:
    """Uniform rectangular 2D grid for finite difference discretization.

    Cell-centered storage: fields live at cell centers.
    X, Y meshgrids follow imshow's y-first convention: shape (ny, nx).

    Attributes:
        x: (nx+1,) interface positions in x-direction
        y: (ny+1,) interface positions in y-direction
        x_centers: (nx,) cell centers in x
        y_centers: (ny,) cell centers in y
        dx: (nx,) cell widths in x
        dy: (ny,) cell widths in y
        Lx: domain length in x
        Ly: domain length in y
        nx: number of cells in x
        ny: number of cells in y
        X: (ny, nx) 2D meshgrid of x-coordinates at cell centers
        Y: (ny, nx) 2D meshgrid of y-coordinates at cell centers
    """
    x: jnp.ndarray
    y: jnp.ndarray
    x_centers: jnp.ndarray
    y_centers: jnp.ndarray
    dx: jnp.ndarray
    dy: jnp.ndarray
    Lx: float
    Ly: float
    nx: int
    ny: int
    X: jnp.ndarray
    Y: jnp.ndarray

    @classmethod
    def uniform(cls, Lx: float, Ly: float, nx: int, ny: int) -> "Grid2D":
        """Create a uniformly spaced 2D grid.

        Args:
            Lx: Domain length in x (0 to Lx).
            Ly: Domain length in y (0 to Ly).
            nx: Number of cells in x-direction.
            ny: Number of cells in y-direction.

        Returns:
            Grid2D with uniform spacing in both directions.
        """
        if Lx <= 0:
            raise ValueError(f"Lx must be positive, got {Lx}")
        if Ly <= 0:
            raise ValueError(f"Ly must be positive, got {Ly}")
        if nx < 2:
            raise ValueError(f"nx must be at least 2, got {nx}")
        if ny < 2:
            raise ValueError(f"ny must be at least 2, got {ny}")

        dx_val = Lx / nx
        dy_val = Ly / ny

        x = array(jnp.linspace(0.0, Lx, nx + 1))
        y = array(jnp.linspace(0.0, Ly, ny + 1))
        x_centers = array(0.5 * (x[:-1] + x[1:]))
        y_centers = array(0.5 * (y[:-1] + y[1:]))
        dx = array(jnp.full(nx, dx_val))
        dy = array(jnp.full(ny, dy_val))
        X, Y = jnp.meshgrid(x_centers, y_centers)  # indexing='xy' default: X(j,i)=x[i], Y(j,i)=y[j]

        return cls(
            x=x, y=y,
            x_centers=x_centers, y_centers=y_centers,
            dx=dx, dy=dy,
            Lx=Lx, Ly=Ly,
            nx=nx, ny=ny,
            X=array(X), Y=array(Y),
        )

"""2D uniform rectangular grid (DESIGN §4).

Mirrored from the frozen ``diffheat`` 2D mesh without importing it.
Cell-centred storage: fields live at cell centers with shape
``(nx, ny)`` — x varies along axis 0, y along axis 1 (``indexing="ij"``),
matching the harness 2-D physics convention
(``forced_conv``, ``cloak2d``) rather than ``imshow`` order.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class Grid2D:
    """Uniform rectangular 2D grid.

    Attributes:
        x: (nx+1,) interfaces in x. y: (ny+1,) interfaces in y.
        x_centers: (nx,) cell centers in x. y_centers: (ny,) in y.
        dx: (nx,) widths in x. dy: (ny,) widths in y.
        Lx, Ly: domain lengths. nx, ny: cell counts.
        X, Y: (nx, ny) meshgrids at cell centers (indexing="ij").
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
        """Create a uniformly spaced 2D grid."""
        if Lx <= 0:
            raise ValueError(f"Lx must be positive, got {Lx}")
        if Ly <= 0:
            raise ValueError(f"Ly must be positive, got {Ly}")
        if nx < 2:
            raise ValueError(f"nx must be at least 2, got {nx}")
        if ny < 2:
            raise ValueError(f"ny must be at least 2, got {ny}")
        nx, ny = int(nx), int(ny)
        dx_val = float(Lx) / nx
        dy_val = float(Ly) / ny
        x = jnp.linspace(0.0, float(Lx), nx + 1, dtype=jnp.float64)
        y = jnp.linspace(0.0, float(Ly), ny + 1, dtype=jnp.float64)
        x_centers = 0.5 * (x[:-1] + x[1:])
        y_centers = 0.5 * (y[:-1] + y[1:])
        dx = jnp.full((nx,), dx_val, dtype=jnp.float64)
        dy = jnp.full((ny,), dy_val, dtype=jnp.float64)
        X, Y = jnp.meshgrid(x_centers, y_centers, indexing="ij")
        return cls(
            x=x, y=y,
            x_centers=x_centers, y_centers=y_centers,
            dx=dx, dy=dy,
            Lx=float(Lx), Ly=float(Ly),
            nx=nx, ny=ny,
            X=X.astype(jnp.float64), Y=Y.astype(jnp.float64),
        )

    @property
    def dx_min(self) -> float:
        return float(min(jnp.min(self.dx), jnp.min(self.dy)))

    def shape(self) -> tuple[int, ...]:
        """Field shape for this grid: ``(nx, ny)``."""
        return (self.nx, self.ny)


__all__ = ["Grid2D"]

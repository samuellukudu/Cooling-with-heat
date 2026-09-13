"""3D uniform rectangular grid (DESIGN §4).

Mirrored from the frozen ``diffheat`` 3D mesh without importing it.
Cell-centred storage: fields have shape ``(nx, ny, nz)`` — x along
axis 0, y along axis 1, z along axis 2 (``indexing="ij"``).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class Grid3D:
    """Uniform rectangular 3D grid.

    Attributes:
        x/y/z: (n+1,) interfaces. x_centers/y_centers/z_centers: (n,).
        dx/dy/dz: (n,) widths. Lx/Ly/Lz: lengths. nx/ny/nz: counts.
        X/Y/Z: (nx, ny, nz) meshgrids at cell centers.
    """

    x: jnp.ndarray
    y: jnp.ndarray
    z: jnp.ndarray
    x_centers: jnp.ndarray
    y_centers: jnp.ndarray
    z_centers: jnp.ndarray
    dx: jnp.ndarray
    dy: jnp.ndarray
    dz: jnp.ndarray
    Lx: float
    Ly: float
    Lz: float
    nx: int
    ny: int
    nz: int
    X: jnp.ndarray
    Y: jnp.ndarray
    Z: jnp.ndarray

    @classmethod
    def uniform(
        cls,
        Lx: float,
        Ly: float,
        Lz: float,
        nx: int,
        ny: int,
        nz: int,
    ) -> "Grid3D":
        """Create a uniformly spaced 3D grid."""
        for name, val in (("Lx", Lx), ("Ly", Ly), ("Lz", Lz)):
            if val <= 0:
                raise ValueError(f"{name} must be positive, got {val}")
        for name, val in (("nx", nx), ("ny", ny), ("nz", nz)):
            if val < 2:
                raise ValueError(f"{name} must be at least 2, got {val}")
        nx, ny, nz = int(nx), int(ny), int(nz)
        dx_val = float(Lx) / nx
        dy_val = float(Ly) / ny
        dz_val = float(Lz) / nz
        x = jnp.linspace(0.0, float(Lx), nx + 1, dtype=jnp.float64)
        y = jnp.linspace(0.0, float(Ly), ny + 1, dtype=jnp.float64)
        z = jnp.linspace(0.0, float(Lz), nz + 1, dtype=jnp.float64)
        x_centers = 0.5 * (x[:-1] + x[1:])
        y_centers = 0.5 * (y[:-1] + y[1:])
        z_centers = 0.5 * (z[:-1] + z[1:])
        dx = jnp.full((nx,), dx_val, dtype=jnp.float64)
        dy = jnp.full((ny,), dy_val, dtype=jnp.float64)
        dz = jnp.full((nz,), dz_val, dtype=jnp.float64)
        X, Y, Z = jnp.meshgrid(x_centers, y_centers, z_centers, indexing="ij")
        return cls(
            x=x, y=y, z=z,
            x_centers=x_centers, y_centers=y_centers, z_centers=z_centers,
            dx=dx, dy=dy, dz=dz,
            Lx=float(Lx), Ly=float(Ly), Lz=float(Lz),
            nx=nx, ny=ny, nz=nz,
            X=X.astype(jnp.float64),
            Y=Y.astype(jnp.float64),
            Z=Z.astype(jnp.float64),
        )

    @property
    def dx_min(self) -> float:
        return float(min(jnp.min(self.dx), jnp.min(self.dy), jnp.min(self.dz)))

    def shape(self) -> tuple[int, ...]:
        """Field shape for this grid: ``(nx, ny, nz)``."""
        return (self.nx, self.ny, self.nz)


__all__ = ["Grid3D"]

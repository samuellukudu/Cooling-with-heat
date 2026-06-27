# diffheat/mesh/grid3d.py
"""3D uniform rectangular grid for finite difference discretization."""
from dataclasses import dataclass

import jax.numpy as jnp

from ..utils import array


@dataclass(frozen=True)
class Grid3D:
    """Uniform rectangular 3D grid for finite difference discretization.

    Cell-centered storage: fields live at cell centers.
    Field arrays have shape ``(nx, ny, nz)`` — x varies along axis 0,
    y along axis 1, z along axis 2.

    Attributes:
        x: (nx+1,) interface positions in x-direction
        y: (ny+1,) interface positions in y-direction
        z: (nz+1,) interface positions in z-direction
        x_centers: (nx,) cell centers in x
        y_centers: (ny,) cell centers in y
        z_centers: (nz,) cell centers in z
        dx: (nx,) cell widths in x
        dy: (ny,) cell widths in y
        dz: (nz,) cell widths in z
        Lx: domain length in x
        Ly: domain length in y
        Lz: domain length in z
        nx: number of cells in x
        ny: number of cells in y
        nz: number of cells in z
        X: (nz, ny, nx) 3D meshgrid of x-coordinates at cell centers
        Y: (nz, ny, nx) 3D meshgrid of y-coordinates at cell centers
        Z: (nz, ny, nx) 3D meshgrid of z-coordinates at cell centers
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
        """Create a uniformly spaced 3D grid.

        Args:
            Lx: Domain length in x (0 to Lx).
            Ly: Domain length in y (0 to Ly).
            Lz: Domain length in z (0 to Lz).
            nx: Number of cells in x-direction.
            ny: Number of cells in y-direction.
            nz: Number of cells in z-direction.

        Returns:
            Grid3D with uniform spacing in all three directions.

        Raises:
            ValueError: If any domain length is non-positive or any cell
                count is less than 2.
        """
        for name, val in (("Lx", Lx), ("Ly", Ly), ("Lz", Lz)):
            if val <= 0:
                raise ValueError(f"{name} must be positive, got {val}")
        for name, val in (("nx", nx), ("ny", ny), ("nz", nz)):
            if val < 2:
                raise ValueError(f"{name} must be at least 2, got {val}")

        dx_val = Lx / nx
        dy_val = Ly / ny
        dz_val = Lz / nz

        x = array(jnp.linspace(0.0, Lx, nx + 1))
        y = array(jnp.linspace(0.0, Ly, ny + 1))
        z = array(jnp.linspace(0.0, Lz, nz + 1))
        x_centers = array(0.5 * (x[:-1] + x[1:]))
        y_centers = array(0.5 * (y[:-1] + y[1:]))
        z_centers = array(0.5 * (z[:-1] + z[1:]))
        dx = array(jnp.full(nx, dx_val))
        dy = array(jnp.full(ny, dy_val))
        dz = array(jnp.full(nz, dz_val))

        # meshgrid: indexing='ij' gives shapes (nx, ny, nz) — consistent with
        # the (nx, ny, nz) field convention used throughout diffheat.
        X, Y, Z = jnp.meshgrid(x_centers, y_centers, z_centers, indexing="ij")

        return cls(
            x=x, y=y, z=z,
            x_centers=x_centers, y_centers=y_centers, z_centers=z_centers,
            dx=dx, dy=dy, dz=dz,
            Lx=Lx, Ly=Ly, Lz=Lz,
            nx=nx, ny=ny, nz=nz,
            X=array(X), Y=array(Y), Z=array(Z),
        )

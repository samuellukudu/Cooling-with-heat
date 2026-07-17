# diffheat/mesh/circular.py
"""Polar grid for finite-difference and spectral methods on a disc.

The disc of radius *R* is discretised into *nr* radial cells and
*ntheta* angular cells.  Cell centres are stored in Cartesian (X, Y)
and polar (R_mesh, THETA_mesh) coordinates for convenience.

This grid is cell-centred: the first radial cell centre is at dr/2
so the r=0 singularity is never evaluated directly.
"""
from dataclasses import dataclass

import jax.numpy as jnp

from ..utils import array


@dataclass(frozen=True)
class PolarGrid:
    """Uniform polar grid on a disc of radius *R*.

    Cell-centred in both *r* and *θ*.

    Attributes:
        r: (nr+1,) radial interface positions.
        theta: (ntheta+1,) angular interface positions, covers [-π, π].
        r_centers: (nr,) radial cell centres.
        theta_centers: (ntheta,) angular cell centres.
        dr: (nr,) radial cell widths.
        dtheta: Angular cell width (scalar float).
        R: Disc radius.
        nr: Number of radial cells.
        ntheta: Number of angular cells.
        X: (nr, ntheta) Cartesian x-coordinates of cell centres.
        Y: (nr, ntheta) Cartesian y-coordinates of cell centres.
        R_mesh: (nr, ntheta) radial coordinates of cell centres.
        THETA_mesh: (nr, ntheta) angular coordinates of cell centres.
        area: (nr, ntheta) cell areas  r_i * dr_i * dtheta.
    """
    r: jnp.ndarray
    theta: jnp.ndarray
    r_centers: jnp.ndarray
    theta_centers: jnp.ndarray
    dr: jnp.ndarray
    dtheta: float
    R: float
    nr: int
    ntheta: int
    X: jnp.ndarray
    Y: jnp.ndarray
    R_mesh: jnp.ndarray
    THETA_mesh: jnp.ndarray
    area: jnp.ndarray

    @classmethod
    def uniform(cls, R: float, nr: int, ntheta: int) -> "PolarGrid":
        """Create a uniformly-spaced polar grid.

        Args:
            R: Disc radius (must be > 0).
            nr: Number of radial cells (≥ 2).
            ntheta: Number of angular cells (≥ 4, must be even for
                    symmetry about θ = 0).

        Returns:
            PolarGrid instance.
        """
        if R <= 0:
            raise ValueError(f"R must be positive, got {R}")
        if nr < 2:
            raise ValueError(f"nr must be at least 2, got {nr}")
        if ntheta < 4:
            raise ValueError(f"ntheta must be at least 4, got {ntheta}")

        dr_val = R / nr
        dtheta_val = 2.0 * jnp.pi / ntheta

        # Radial interfaces and centres
        r = array(jnp.linspace(0.0, R, nr + 1))
        r_centers = array(0.5 * (r[:-1] + r[1:]))
        dr = array(jnp.full(nr, dr_val))

        # Angular interfaces and centres — cover [-π, π]
        theta = array(jnp.linspace(-jnp.pi, jnp.pi, ntheta + 1))
        theta_centers = array(0.5 * (theta[:-1] + theta[1:]))

        # Meshgrids: (nr, ntheta) with ij-indexing (r varies along axis 0)
        R_mesh, THETA_mesh = jnp.meshgrid(r_centers, theta_centers, indexing="ij")
        X = array(R_mesh * jnp.cos(THETA_mesh))
        Y = array(R_mesh * jnp.sin(THETA_mesh))

        # Cell areas: r_i * dr_i * dtheta (Jacobian in polar coords)
        area = array(R_mesh * dr[:, jnp.newaxis] * dtheta_val)

        return cls(
            r=r,
            theta=theta,
            r_centers=r_centers,
            theta_centers=theta_centers,
            dr=dr,
            dtheta=float(dtheta_val),
            R=R,
            nr=nr,
            ntheta=ntheta,
            X=array(X),
            Y=array(Y),
            R_mesh=array(R_mesh),
            THETA_mesh=array(THETA_mesh),
            area=area,
        )

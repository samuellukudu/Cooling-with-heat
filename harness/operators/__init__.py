"""Discrete differential operators for finite-difference PDEs (DESIGN §4).

Shared 1D/2D/3D stencils over :mod:`harness.mesh` grids. Pure JAX —
no gym, no I/O, no env policy. Mirrored from the frozen ``diffheat``
operators without importing them.
"""

from .advection import advection_1d, advection_2d, advection_3d
from .divergence import divergence_2d, divergence_3d
from .gradient import (
    gradient_1d,
    gradient_2d,
    gradient_3d,
    gradient_x,
    gradient_x3d,
    gradient_y,
    gradient_y3d,
    gradient_z3d,
)
from .laplacian import laplacian_1d, laplacian_2d, laplacian_3d

__all__ = [
    "laplacian_1d",
    "laplacian_2d",
    "laplacian_3d",
    "gradient_1d",
    "gradient_x",
    "gradient_y",
    "gradient_2d",
    "gradient_x3d",
    "gradient_y3d",
    "gradient_z3d",
    "gradient_3d",
    "divergence_2d",
    "divergence_3d",
    "advection_1d",
    "advection_2d",
    "advection_3d",
]

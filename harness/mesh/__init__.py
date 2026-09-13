"""Uniform Cartesian meshes for 1D/2D/3D (``DESIGN.md`` §4).

Pure JAX dataclasses — no gym, no I/O, no env policy. Mirrored from the
frozen ``diffheat`` mesh without importing it, so the harness stays
self-contained and the frozen package stays frozen (parity is a test,
not an import).
"""

from .boundary import (
    BoundaryCondition,
    BoundaryCondition2D,
    BoundaryCondition3D,
    apply_boundary_conditions_1d,
    apply_boundary_conditions_2d,
    apply_boundary_conditions_3d,
)
from .grid1d import Grid1D
from .grid2d import Grid2D
from .grid3d import Grid3D

__all__ = [
    "Grid1D",
    "Grid2D",
    "Grid3D",
    "BoundaryCondition",
    "BoundaryCondition2D",
    "BoundaryCondition3D",
    "apply_boundary_conditions_1d",
    "apply_boundary_conditions_2d",
    "apply_boundary_conditions_3d",
]

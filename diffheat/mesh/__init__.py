# diffheat/mesh/__init__.py
"""Grid and boundary condition definitions."""
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

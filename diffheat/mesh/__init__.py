# diffheat/mesh/__init__.py
"""Grid and boundary condition definitions."""
from .boundary import (
    BoundaryCondition,
    BoundaryCondition2D,
    apply_boundary_conditions_1d,
    apply_boundary_conditions_2d,
)
from .grid1d import Grid1D
from .grid2d import Grid2D

__all__ = [
    "Grid1D",
    "Grid2D",
    "BoundaryCondition",
    "BoundaryCondition2D",
    "apply_boundary_conditions_1d",
    "apply_boundary_conditions_2d",
]

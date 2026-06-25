# diffheat/mesh/__init__.py
"""Grid and boundary condition definitions."""
from .boundary import BoundaryCondition
from .grid1d import Grid1D
from .grid2d import Grid2D

__all__ = ["Grid1D", "Grid2D", "BoundaryCondition"]

# diffheat/mesh/__init__.py
"""Grid and boundary condition definitions."""
from .boundary import BoundaryCondition
from .grid1d import Grid1D

__all__ = ["Grid1D", "BoundaryCondition"]

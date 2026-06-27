# diffheat/physics/__init__.py
"""Physical problem definitions."""
from ..operators.laplacian import make_laplacian
from .heat1d import HeatEquation1D, apply_boundary_conditions

__all__ = ["HeatEquation1D", "apply_boundary_conditions", "make_laplacian"]

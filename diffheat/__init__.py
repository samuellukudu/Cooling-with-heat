# diffheat/__init__.py
"""diffheat — Differentiable heat equation simulations with JAX."""
import logging

from .mesh import BoundaryCondition, BoundaryCondition2D, Grid1D, Grid2D
from .operators import make_laplacian
from .physics import HeatEquation1D, apply_boundary_conditions
from .solvers import check_cfl, check_cfl_2d, explicit_euler_step, solve_heat_1d, solve_2d
from .utils import array, get_default_dtype, get_device

_logger = logging.getLogger(__name__)
_logger.info(f"diffheat running on: {get_device()}")

__all__ = [
    "Grid1D",
    "Grid2D",
    "BoundaryCondition",
    "BoundaryCondition2D",
    "HeatEquation1D",
    "make_laplacian",
    "apply_boundary_conditions",
    "explicit_euler_step",
    "solve_heat_1d",
    "solve_2d",
    "check_cfl",
    "check_cfl_2d",
    "get_device",
    "get_default_dtype",
    "array",
]

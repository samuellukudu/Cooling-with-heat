# diffheat/__init__.py
"""diffheat — Differentiable heat equation simulations with JAX."""
import logging

from .mesh import (
    BoundaryCondition,
    BoundaryCondition2D,
    BoundaryCondition3D,
    Grid1D,
    Grid2D,
    Grid3D,
    apply_boundary_conditions_1d,
    apply_boundary_conditions_3d,
)
from .operators import (
    divergence_2d,
    divergence_3d,
    gradient_1d,
    gradient_2d,
    gradient_3d,
    gradient_x3d,
    gradient_y3d,
    gradient_z3d,
    laplacian_1d,
    laplacian_2d,
    laplacian_3d,
    make_laplacian,
)
from .physics import HeatEquation1D, apply_boundary_conditions
from .solvers import (
    check_cfl,
    check_cfl_2d,
    check_cfl_3d,
    explicit_euler_step,
    explicit_euler_step_1d,
    explicit_euler_step_3d,
    solve_1d,
    solve_2d,
    solve_3d,
    solve_heat_1d,
)
from .utils import array, get_default_dtype, get_device

_logger = logging.getLogger(__name__)
_logger.info(f"diffheat running on: {get_device()}")

__all__ = [
    # Mesh — 1D
    "Grid1D",
    "BoundaryCondition",
    "apply_boundary_conditions_1d",
    # Mesh — 2D
    "Grid2D",
    "BoundaryCondition2D",
    # Mesh — 3D
    "Grid3D",
    "BoundaryCondition3D",
    "apply_boundary_conditions_3d",
    # Operators — 1D
    "make_laplacian",
    "laplacian_1d",
    "gradient_1d",
    # Operators — 2D
    "laplacian_2d",
    "gradient_2d",
    "divergence_2d",
    # Operators — 3D
    "laplacian_3d",
    "gradient_3d",
    "gradient_x3d",
    "gradient_y3d",
    "gradient_z3d",
    "divergence_3d",
    # Physics
    "HeatEquation1D",
    "apply_boundary_conditions",
    # Solvers — 1D
    "explicit_euler_step",
    "explicit_euler_step_1d",
    "solve_heat_1d",
    "solve_1d",
    "check_cfl",
    # Solvers — 2D
    "explicit_euler_step_2d",
    "solve_2d",
    "check_cfl_2d",
    # Solvers — 3D
    "explicit_euler_step_3d",
    "solve_3d",
    "check_cfl_3d",
    # Utils
    "get_device",
    "get_default_dtype",
    "array",
]

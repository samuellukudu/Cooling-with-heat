# diffheat/solvers/__init__.py
"""Time integration solvers."""
from .explicit import (
    explicit_euler_step,
    explicit_euler_step_1d,
    explicit_euler_step_2d,
    explicit_euler_step_3d,
)
from .scan import solve_1d, solve_2d, solve_3d, solve_heat_1d
from .stability import check_cfl, check_cfl_2d, check_cfl_3d

__all__ = [
    "explicit_euler_step",
    "explicit_euler_step_1d",
    "explicit_euler_step_2d",
    "explicit_euler_step_3d",
    "solve_heat_1d",
    "solve_1d",
    "solve_2d",
    "solve_3d",
    "check_cfl",
    "check_cfl_2d",
    "check_cfl_3d",
]

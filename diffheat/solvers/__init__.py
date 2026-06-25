# diffheat/solvers/__init__.py
"""Time integration solvers."""
from .explicit import explicit_euler_step
from .scan import solve_heat_1d
from .stability import check_cfl

__all__ = ["explicit_euler_step", "solve_heat_1d", "check_cfl"]

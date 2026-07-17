# diffheat/solvers/__init__.py
"""Time integration solvers."""
from .eigen import (
    find_first_eigenvalue_1d,
    find_first_eigenvalue_2d,
    find_first_eigenvalue_3d,
    rayleigh_quotient_1d,
    rayleigh_quotient_2d,
    rayleigh_quotient_3d,
    rayleigh_upper_bounds_1d,
    rayleigh_upper_bounds_2d,
    rayleigh_upper_bounds_3d,
)
from .explicit import (
    explicit_euler_step,
    explicit_euler_step_1d,
    explicit_euler_step_2d,
    explicit_euler_step_3d,
)
from .inhomogeneous import (
    solve_heat_inhomogeneous_2d,
    solve_heat_inhomogeneous_3d,
)
from .scan import (
    solve_1d,
    solve_2d,
    solve_3d,
    solve_heat_1d,
    solve_heat_2d,
    solve_heat_3d,
)
from .stability import (
    check_cfl,
    check_cfl_2d,
    check_cfl_3d,
    check_cfl_wave_1d,
    check_cfl_wave_2d,
    check_cfl_wave_3d,
    check_cfl_telegrapher_1d,
    check_cfl_telegrapher_2d,
    check_cfl_telegrapher_3d,
)
from .steady_state import (
    solve_steady_state_1d,
    solve_steady_state_2d,
    solve_steady_state_3d,
)
from .wave import solve_wave_1d, solve_wave_2d, solve_wave_3d
from .polar import (
    solve_heat_disc_analytical,
    solve_steady_state_disc,
    solve_steady_state_cylinder_3d,
)
from .telegrapher import solve_telegrapher_1d, solve_telegrapher_2d, solve_telegrapher_3d

__all__ = [
    "explicit_euler_step",
    "explicit_euler_step_1d",
    "explicit_euler_step_2d",
    "explicit_euler_step_3d",
    "solve_heat_1d",
    "solve_heat_2d",
    "solve_heat_3d",
    "solve_1d",
    "solve_2d",
    "solve_3d",
    "solve_wave_1d",
    "solve_wave_2d",
    "solve_wave_3d",
    "solve_telegrapher_1d",
    "solve_telegrapher_2d",
    "solve_telegrapher_3d",
    "check_cfl",
    "check_cfl_2d",
    "check_cfl_3d",
    "check_cfl_wave_1d",
    "check_cfl_wave_2d",
    "check_cfl_wave_3d",
    "check_cfl_telegrapher_1d",
    "check_cfl_telegrapher_2d",
    "check_cfl_telegrapher_3d",
    "solve_steady_state_1d",
    "solve_steady_state_2d",
    "solve_steady_state_3d",
    "solve_heat_inhomogeneous_2d",
    "solve_heat_inhomogeneous_3d",
    "rayleigh_quotient_1d",
    "rayleigh_quotient_2d",
    "rayleigh_quotient_3d",
    "rayleigh_upper_bounds_1d",
    "rayleigh_upper_bounds_2d",
    "rayleigh_upper_bounds_3d",
    "find_first_eigenvalue_1d",
    "find_first_eigenvalue_2d",
    "find_first_eigenvalue_3d",
    "solve_heat_disc_analytical",
    "solve_steady_state_disc",
    "solve_steady_state_cylinder_3d",
]



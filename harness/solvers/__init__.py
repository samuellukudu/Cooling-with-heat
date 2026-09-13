"""Time-integration solvers: dim-dispatched scans + CFL guards (DESIGN §4).

Pure JAX — no gym, no I/O, no env policy. Equation-agnostic scan
machinery (:mod:`.scan`), explicit diffusion CFL checks (:mod:`.stability`)
and the uniform 4D time helpers (:mod:`.time`) share one import surface.
"""

from .scan import solve_1d, solve_2d, solve_3d, solve_dim
from .stability import (
    check_cfl,
    check_cfl_1d,
    check_cfl_2d,
    check_cfl_3d,
    max_timestep_1d,
    max_timestep_2d,
    max_timestep_3d,
    max_timestep_advection_diffusion_2d,
)
from .time import (
    MAX_TRACE_SAMPLES,
    n_steps_for,
    save_every_outer_steps,
    thin_trace,
    time_grid,
)

__all__ = [
    "MAX_TRACE_SAMPLES",
    "solve_1d",
    "solve_2d",
    "solve_3d",
    "solve_dim",
    "check_cfl",
    "check_cfl_1d",
    "check_cfl_2d",
    "check_cfl_3d",
    "max_timestep_1d",
    "max_timestep_2d",
    "max_timestep_3d",
    "max_timestep_advection_diffusion_2d",
    "n_steps_for",
    "save_every_outer_steps",
    "thin_trace",
    "time_grid",
]

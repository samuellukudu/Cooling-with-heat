# diffheat/solvers/advection_diffusion.py
"""Advection-diffusion equation solvers using explicit Euler + jax.lax.scan."""
import logging

import jax
import jax.numpy as jnp

from ..mesh.boundary import (
    apply_boundary_conditions_1d,
    apply_boundary_conditions_2d,
    apply_boundary_conditions_3d,
)
from ..operators.advection import advection_1d, advection_2d, advection_3d
from ..operators.laplacian import laplacian_1d, laplacian_2d, laplacian_3d
from ..physics.advection_diffusion import (
    AdvectionDiffusion1D,
    AdvectionDiffusion2D,
    AdvectionDiffusion3D,
)
from .scan import solve_1d, solve_2d, solve_3d
from .stability import (
    check_cfl_advection_diffusion_1d,
    check_cfl_advection_diffusion_2d,
    check_cfl_advection_diffusion_3d,
)

_logger = logging.getLogger(__name__)


def solve_advection_diffusion_1d(
    eqn: AdvectionDiffusion1D,
    T0: jnp.ndarray,
    t_span: tuple[float, float],
    dt: float,
) -> jnp.ndarray:
    """Solve the 1D advection-diffusion equation with explicit Euler.

    Args:
        eqn: Advection-diffusion problem definition.
        T0: (N,) initial temperature field.
        t_span: (t_start, t_end) simulation time range.
        dt: Time step size.

    Returns:
        (n_steps+1, N) temperature trajectory. First frame is T0.
    """
    t0, t_end = t_span
    n_steps = int((t_end - t0) / dt)
    if n_steps < 1:
        raise ValueError(f"t_span too short for dt={dt}: {t_span}")

    # CFL check
    try:
        u = eqn.velocity(eqn.grid.centers, t0)
        u_max = float(jnp.max(jnp.abs(u)))
        if not check_cfl_advection_diffusion_1d(eqn.grid, eqn.alpha, u_max, dt):
            alpha_max = float(jnp.max(jnp.asarray(eqn.alpha)))
            dx_min = float(jnp.min(eqn.grid.dx))
            dt_diff = dx_min**2 / (2.0 * alpha_max) if alpha_max > 0 else float("inf")
            dt_adv = dx_min / u_max if u_max > 0 else float("inf")
            _logger.warning(
                f"dt={dt:.2e} exceeds CFL limit (diff={dt_diff:.2e}, adv={dt_adv:.2e}). "
                f"Solution may be unstable."
            )
    except jax.errors.ConcretizationTypeError:
        pass

    def rhs_fn(T, grid, t, params):
        u = eqn.velocity(grid.centers, t)
        L_T, b_source = apply_boundary_conditions_1d(
            lambda x: laplacian_1d(x, grid), grid, eqn.bc, T
        )
        dT_dt = eqn.alpha * (L_T + b_source)
        dT_dt = dT_dt + advection_1d(T, u, grid.dx)
        if eqn.source is not None:
            dT_dt = dT_dt + eqn.source(grid.centers, t)
        return dT_dt

    return solve_1d(rhs_fn, T0, eqn.grid, t_span, dt)


def solve_advection_diffusion_2d(
    eqn: AdvectionDiffusion2D,
    T0: jnp.ndarray,
    t_span: tuple[float, float],
    dt: float,
) -> jnp.ndarray:
    """Solve the 2D advection-diffusion equation with explicit Euler.

    Args:
        eqn: Advection-diffusion problem definition.
        T0: (nx, ny) initial temperature field.
        t_span: (t_start, t_end) simulation time range.
        dt: Time step size.

    Returns:
        (n_steps+1, nx, ny) temperature trajectory. First frame is T0.
    """
    t0, t_end = t_span
    n_steps = int((t_end - t0) / dt)
    if n_steps < 1:
        raise ValueError(f"t_span too short for dt={dt}: {t_span}")

    # CFL check — uses initial velocity field as representative
    try:
        u_x, u_y = eqn.velocity(eqn.grid.X.T, eqn.grid.Y.T, t0)
        u_x_max = float(jnp.max(jnp.abs(u_x)))
        u_y_max = float(jnp.max(jnp.abs(u_y)))
        if not check_cfl_advection_diffusion_2d(eqn.grid, eqn.alpha, u_x_max, u_y_max, dt):
            alpha_max = float(jnp.max(jnp.asarray(eqn.alpha)))
            dx_min = float(jnp.min(eqn.grid.dx))
            dy_min = float(jnp.min(eqn.grid.dy))
            dt_diff = min(dx_min**2, dy_min**2) / (4.0 * alpha_max) if alpha_max > 0 else float("inf")
            dt_adv = 1.0 / (u_x_max / dx_min + u_y_max / dy_min) if (u_x_max > 0 or u_y_max > 0) else float("inf")
            _logger.warning(
                f"dt={dt:.2e} exceeds CFL limit (diff={dt_diff:.2e}, adv={dt_adv:.2e}). "
                f"Solution may be unstable."
            )
    except jax.errors.ConcretizationTypeError:
        pass

    def rhs_fn(T, grid, t, params):
        u_x, u_y = eqn.velocity(grid.X.T, grid.Y.T, t)
        L_T, b_source = apply_boundary_conditions_2d(
            lambda x: laplacian_2d(x, grid), grid, eqn.bc, T
        )
        dT_dt = eqn.alpha * (L_T + b_source)
        dT_dt = dT_dt + advection_2d(T, u_x, u_y, grid.dx, grid.dy)
        if eqn.source is not None:
            dT_dt = dT_dt + eqn.source(grid.X.T, grid.Y.T, t)
        return dT_dt

    return solve_2d(rhs_fn, T0, eqn.grid, t_span, dt)


def solve_advection_diffusion_3d(
    eqn: AdvectionDiffusion3D,
    T0: jnp.ndarray,
    t_span: tuple[float, float],
    dt: float,
    save_every: int = 1,
) -> jnp.ndarray:
    """Solve the 3D advection-diffusion equation with explicit Euler.

    Args:
        eqn: Advection-diffusion problem definition.
        T0: (nx, ny, nz) initial temperature field.
        t_span: (t_start, t_end) simulation time range.
        dt: Time step size.
        save_every: Save a frame every this many steps.

    Returns:
        (n_saved+1, nx, ny, nz) temperature trajectory. First frame is T0.
    """
    t0, t_end = t_span
    n_steps = int((t_end - t0) / dt)
    if n_steps < 1:
        raise ValueError(f"t_span too short for dt={dt}: {t_span}")

    # CFL check
    try:
        u_x, u_y, u_z = eqn.velocity(eqn.grid.X, eqn.grid.Y, eqn.grid.Z, t0)
        u_x_max = float(jnp.max(jnp.abs(u_x)))
        u_y_max = float(jnp.max(jnp.abs(u_y)))
        u_z_max = float(jnp.max(jnp.abs(u_z)))
        if not check_cfl_advection_diffusion_3d(eqn.grid, eqn.alpha, u_x_max, u_y_max, u_z_max, dt):
            alpha_max = float(jnp.max(jnp.asarray(eqn.alpha)))
            dx_min = float(jnp.min(eqn.grid.dx))
            dy_min = float(jnp.min(eqn.grid.dy))
            dz_min = float(jnp.min(eqn.grid.dz))
            dt_diff = min(dx_min**2, dy_min**2, dz_min**2) / (6.0 * alpha_max) if alpha_max > 0 else float("inf")
            u_sum = u_x_max / dx_min + u_y_max / dy_min + u_z_max / dz_min
            dt_adv = 1.0 / u_sum if u_sum > 0 else float("inf")
            _logger.warning(
                f"dt={dt:.2e} exceeds CFL limit (diff={dt_diff:.2e}, adv={dt_adv:.2e}). "
                f"Solution may be unstable."
            )
    except jax.errors.ConcretizationTypeError:
        pass

    def rhs_fn(T, grid, t, params):
        u_x, u_y, u_z = eqn.velocity(grid.X, grid.Y, grid.Z, t)
        L_T, b_source = apply_boundary_conditions_3d(
            lambda x: laplacian_3d(x, grid), grid, eqn.bc, T
        )
        dT_dt = eqn.alpha * (L_T + b_source)
        dT_dt = dT_dt + advection_3d(T, u_x, u_y, u_z, grid.dx, grid.dy, grid.dz)
        if eqn.source is not None:
            dT_dt = dT_dt + eqn.source(grid.X, grid.Y, grid.Z, t)
        return dT_dt

    return solve_3d(rhs_fn, T0, eqn.grid, t_span, dt, save_every=save_every)

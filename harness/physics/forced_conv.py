"""Forced-convection channel physics — 2-D advection-diffusion (DESIGN §7.2).

Mirrors the frozen ``diffheat`` advection-diffusion 2-D hot path
(``diffheat/operators/advection.py::advection_2d``,
``diffheat/operators/laplacian.py::laplacian_2d``,
``diffheat/mesh/boundary.py::apply_boundary_conditions_2d``,
``diffheat/solvers/advection_diffusion.py::solve_advection_diffusion_2d``)
without importing it (import-hygiene contract: ``harness.physics`` never
imports ``diffheat``). Parity is pinned by
``tests/harness/test_diffheat_trials.py::test_forced_conv_parity_vs_diffheat``.

Model (trial T-A5, ``docs/applications.md`` §5, ``examples/09``):
2-D channel ``[0,Lx]×[0,Ly]``, uniform horizontal flow ``u=(U,0)``,
cold Dirichlet inlet/outlet (left/right ``T=0``), insulated top/bottom
(Neumann 0), Gaussian chip source at ``(Lx/2, 0)``::

    dT/dt = alpha*nabla^2 T - U*dT/dx + S(x,y)

Discretization: cell-centred finite volumes, 5-point Laplacian via
``jnp.roll`` + ghost-cell BC correction (diffusion only — advection keeps
the raw upwind roll, exactly like diffheat), first-order upwind advection,
explicit Euler inside ``jax.lax.scan``. Whole rollout is jittable and
differentiable in ``inlet_velocity`` (tracer-safe: no ``float()`` on tracers
inside the hot path).

Units: SI-ish demo units (°C for T, m/s for U, s for t). The env boundary
converts nothing — metrics are reported directly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Default trial geometry/material/source (mirrors examples/09 demo).
DEFAULT_LX = 4.0
DEFAULT_LY = 1.0
DEFAULT_ALPHA = 0.01
DEFAULT_CHIP_X0 = 2.0  # Lx/2
DEFAULT_CHIP_Y0 = 0.0
DEFAULT_CHIP_WIDTH = 0.1
DEFAULT_SOURCE_AMP = 500.0

FORCED_CONV_METRIC_KEYS = (
    "T_max",
    "T_mean",
    "pumping_proxy",
    "peclet",
)


def max_timestep(dx: float, dy: float, alpha: float, u_max: float) -> float:
    """Combined advection-diffusion explicit limit (mirrors diffheat)."""
    alpha_f = float(alpha)
    dt_diff = min(dx * dx, dy * dy) / (4.0 * alpha_f) if alpha_f > 0 else float("inf")
    if u_max > 0:
        dt_adv = 1.0 / (u_max / dx)
    else:
        dt_adv = float("inf")
    return min(dt_diff, dt_adv)


def check_timestep(dx: float, dy: float, alpha: float, u_max: float, dt: float) -> bool:
    """True when ``dt`` satisfies the explicit CFL bound (fail loudly)."""
    return bool(dt <= max_timestep(dx, dy, alpha, u_max))


def laplacian_2d(T: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """5-point Laplacian via roll (periodic; corrected by BC step)."""
    d2x = (jnp.roll(T, -1, axis=0) + jnp.roll(T, 1, axis=0) - 2.0 * T) / (dx * dx)
    d2y = (jnp.roll(T, -1, axis=1) + jnp.roll(T, 1, axis=1) - 2.0 * T) / (dy * dy)
    return d2x + d2y


def advection_2d_upwind(T: jnp.ndarray, u_x: jnp.ndarray, u_y: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """First-order upwind ``-(u·grad T)`` (mirrors diffheat, incl. raw roll at BCs)."""
    T_fx = jnp.roll(T, -1, axis=0)
    T_bx = jnp.roll(T, 1, axis=0)
    adv_x = -u_x * jnp.where(u_x > 0, (T - T_bx) / dx, (T_fx - T) / dx)
    T_fy = jnp.roll(T, -1, axis=1)
    T_by = jnp.roll(T, 1, axis=1)
    adv_y = -u_y * jnp.where(u_y > 0, (T - T_by) / dy, (T_fy - T) / dy)
    return adv_x + adv_y


def apply_bc_2d(
    L_raw: jnp.ndarray,
    T: jnp.ndarray,
    dx: float,
    dy: float,
    bc_left: tuple[str, float] = ("dirichlet", 0.0),
    bc_right: tuple[str, float] = ("dirichlet", 0.0),
    bc_bottom: tuple[str, float] = ("neumann", 0.0),
    bc_top: tuple[str, float] = ("neumann", 0.0),
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Ghost-cell BC correction for the Laplacian (mirrors diffheat §boundary).

    Returns ``(L_T, b_source)`` with ``dT/dt = alpha*(L_T + b_source) + ...``.
    ``T`` shape is ``(nx, ny)``; left/right act on axis 0, bottom/top on axis 1.
    """
    nx, ny = T.shape
    L_T = L_raw
    b_source = jnp.zeros_like(T)
    # Left (i=0)
    kind, val = bc_left
    if kind == "dirichlet":
        incorrect = (T[nx - 1, :] + T[1, :] - 2.0 * T[0, :]) / (dx * dx)
        correct = (T[1, :] - 3.0 * T[0, :]) / (dx * dx)
        L_T = L_T.at[0, :].add(correct - incorrect)
        b_source = b_source.at[0, :].add(2.0 * val / (dx * dx))
    else:
        incorrect = (T[nx - 1, :] + T[1, :] - 2.0 * T[0, :]) / (dx * dx)
        correct = (T[1, :] - T[0, :]) / (dx * dx)
        L_T = L_T.at[0, :].add(correct - incorrect)
        b_source = b_source.at[0, :].add(-val / dx)
    # Right (i=nx-1)
    kind, val = bc_right
    if kind == "dirichlet":
        incorrect = (T[0, :] + T[nx - 2, :] - 2.0 * T[nx - 1, :]) / (dx * dx)
        correct = (T[nx - 2, :] - 3.0 * T[nx - 1, :]) / (dx * dx)
        L_T = L_T.at[nx - 1, :].add(correct - incorrect)
        b_source = b_source.at[nx - 1, :].add(2.0 * val / (dx * dx))
    else:
        incorrect = (T[0, :] + T[nx - 2, :] - 2.0 * T[nx - 1, :]) / (dx * dx)
        correct = (T[nx - 2, :] - T[nx - 1, :]) / (dx * dx)
        L_T = L_T.at[nx - 1, :].add(correct - incorrect)
        b_source = b_source.at[nx - 1, :].add(-val / dx)
    # Bottom (j=0)
    kind, val = bc_bottom
    if kind == "dirichlet":
        incorrect = (T[:, ny - 1] + T[:, 1] - 2.0 * T[:, 0]) / (dy * dy)
        correct = (T[:, 1] - 3.0 * T[:, 0]) / (dy * dy)
        L_T = L_T.at[:, 0].add(correct - incorrect)
        b_source = b_source.at[:, 0].add(2.0 * val / (dy * dy))
    else:
        incorrect = (T[:, ny - 1] + T[:, 1] - 2.0 * T[:, 0]) / (dy * dy)
        correct = (T[:, 1] - T[:, 0]) / (dy * dy)
        L_T = L_T.at[:, 0].add(correct - incorrect)
        b_source = b_source.at[:, 0].add(-val / dy)
    # Top (j=ny-1)
    kind, val = bc_top
    if kind == "dirichlet":
        incorrect = (T[:, 0] + T[:, ny - 2] - 2.0 * T[:, ny - 1]) / (dy * dy)
        correct = (T[:, ny - 2] - 3.0 * T[:, ny - 1]) / (dy * dy)
        L_T = L_T.at[:, ny - 1].add(correct - incorrect)
        b_source = b_source.at[:, ny - 1].add(2.0 * val / (dy * dy))
    else:
        incorrect = (T[:, 0] + T[:, ny - 2] - 2.0 * T[:, ny - 1]) / (dy * dy)
        correct = (T[:, ny - 2] - T[:, ny - 1]) / (dy * dy)
        L_T = L_T.at[:, ny - 1].add(correct - incorrect)
        b_source = b_source.at[:, ny - 1].add(-val / dy)
    return L_T, b_source


def chip_source_field(X: jnp.ndarray, Y: jnp.ndarray, amplitude: float = DEFAULT_SOURCE_AMP,
                      x0: float = DEFAULT_CHIP_X0, y0: float = DEFAULT_CHIP_Y0,
                      width: float = DEFAULT_CHIP_WIDTH) -> jnp.ndarray:
    """Gaussian chip source (mirrors examples/09). Tracer-safe in amplitude."""
    r2 = ((X - x0) ** 2 + (Y - y0) ** 2) / (2.0 * width * width)
    return amplitude * jnp.exp(-r2)


def make_grids(Lx: float, Ly: float, nx: int, ny: int):
    """Cell-centre grids ``X, Y`` of shape ``(nx, ny)`` (matches diffheat X.T)."""
    dx = Lx / nx
    dy = Ly / ny
    xc = (jnp.arange(nx, dtype=jnp.float64) + 0.5) * dx
    yc = (jnp.arange(ny, dtype=jnp.float64) + 0.5) * dy
    X, Y = jnp.meshgrid(xc, yc, indexing="ij")
    return X, Y, float(dx), float(dy)


def rhs_forced_conv(T, u_x, u_y, X, Y, *, alpha, dx, dy, source_amplitude):
    """Semi-discrete RHS (tracer-safe in ``u_x``/``source_amplitude``)."""
    L_raw = laplacian_2d(T, dx, dy)
    L_T, b = apply_bc_2d(L_raw, T, dx, dy)
    dT = alpha * (L_T + b) + advection_2d_upwind(T, u_x, u_y, dx, dy)
    dT = dT + chip_source_field(X, Y, amplitude=source_amplitude)
    return dT


def simulate_forced_conv(
    *,
    inlet_velocity,
    alpha: float = DEFAULT_ALPHA,
    Lx: float = DEFAULT_LX,
    Ly: float = DEFAULT_LY,
    nx: int = 24,
    ny: int = 8,
    dt: float = 0.002,
    n_steps: int = 20,
    source_amplitude: float = DEFAULT_SOURCE_AMP,
    t0_init=None,
    collect_trace: bool = False,
):
    """Roll out the channel (explicit Euler in ``jax.lax.scan``).

    ``inlet_velocity``/``source_amplitude`` may be JAX tracers (grad flows).
    ``nx/ny/n_steps`` are static. Returns ``{"final": T, "trajectory": ...}``.
    """
    nx, ny, n_steps = int(nx), int(ny), int(n_steps)
    X, Y, dx, dy = make_grids(float(Lx), float(Ly), nx, ny)
    if t0_init is None:
        T_init = jnp.zeros((nx, ny), dtype=jnp.float64)
    else:
        T_init = jnp.asarray(t0_init, dtype=jnp.float64)

    def step_fn(T, _):
        u_x = inlet_velocity * jnp.ones_like(T)
        u_y = jnp.zeros_like(T)
        dT = rhs_forced_conv(T, u_x, u_y, X, Y, alpha=alpha, dx=dx, dy=dy,
                             source_amplitude=source_amplitude)
        return T + dt * dT, None

    # CFL guard — skip during tracing (mirrors diffheat scan pattern).
    try:
        u_max = float(jnp.max(jnp.asarray(inlet_velocity * jnp.ones(()))))
        if not check_timestep(dx, dy, float(alpha), u_max, float(dt)):
            import logging as _logging  # noqa: WPS433
            _logging.getLogger(__name__).warning(
                "dt=%g exceeds forced-conv CFL (dx=%g dy=%g alpha=%g U=%g)",
                float(dt), dx, dy, float(alpha), u_max)
    except Exception:
        pass

    T_final, _ = jax.lax.scan(step_fn, T_init, None, length=n_steps)
    out: dict = {"final": T_final}
    if collect_trace:
        # Re-run with emission for diagnostics (small grids only).
        def step_emit(T, _):
            u_x = inlet_velocity * jnp.ones_like(T)
            u_y = jnp.zeros_like(T)
            dT = rhs_forced_conv(T, u_x, u_y, X, Y, alpha=alpha, dx=dx, dy=dy,
                                 source_amplitude=source_amplitude)
            Tn = T + dt * dT
            return Tn, Tn
        _, traj = jax.lax.scan(step_emit, T_init, None, length=n_steps)
        out["trajectory"] = jnp.concatenate([T_init[jnp.newaxis], traj], axis=0)
    return out


def summary_from_final(T_final, *, inlet_velocity, alpha=DEFAULT_ALPHA, Lx=DEFAULT_LX):
    """Episode metrics from the final field (all JAX scalars, grad-safe)."""
    t_max = jnp.max(T_final)
    t_mean = jnp.mean(T_final)
    pumping = inlet_velocity * inlet_velocity
    peclet = inlet_velocity * Lx / alpha
    return {"T_max": t_max, "T_mean": t_mean, "pumping_proxy": pumping, "peclet": peclet}


__all__ = [
    "FORCED_CONV_METRIC_KEYS",
    "advection_2d_upwind",
    "apply_bc_2d",
    "check_timestep",
    "chip_source_field",
    "laplacian_2d",
    "make_grids",
    "max_timestep",
    "rhs_forced_conv",
    "simulate_forced_conv",
    "summary_from_final",
]

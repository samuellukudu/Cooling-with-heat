# diffheat/analysis/validator.py
"""Validators for fundamental PDE properties.

These check whether a computed numerical solution satisfies key
analytical properties of Laplace / Poisson equations.

PDF §11 — Mean Value Property:
    For any harmonic function v (satisfying ∇²v = 0), the value at
    any point equals the average over any circle (2D) or sphere (3D)
    centred at that point.

PDF §12 — Maximum Principle:
    For Laplace's equation ∇²u = 0, the maximum and minimum of u
    occur on the boundary, never in the interior (unless u is constant).
"""
import jax.numpy as jnp

from ..mesh.grid2d import Grid2D


def check_mean_value_2d(
    u: jnp.ndarray,
    grid: Grid2D,
    center: tuple[float, float],
    radius: float,
    n_angles: int = 200,
) -> tuple[float, float, float]:
    """Check the Mean Value Property for a 2D harmonic function.

    For a Laplace solution ∇²u = 0, the property states:

        u(x₀, y₀) = (1 / 2π) ∫₀^{2π} u(x₀ + R cos θ, y₀ + R sin θ) dθ

    This function numerically evaluates u at *n_angles* equally spaced
    points on a circle of given *radius* and compares the average to
    the value at the centre.

    Uses bilinear interpolation to evaluate u at arbitrary points.

    Args:
        u: (nx, ny) scalar field (typically a Laplace solution).
        grid: The 2D grid.
        center: (x0, y0) centre point for the circle.
        radius: Radius of the evaluation circle.
        n_angles: Number of angular sample points (default 200).

    Returns:
        (u_center, boundary_mean, abs_error)
        where abs_error = |u_center - boundary_mean|.
    """
    x0, y0 = center
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, n_angles)

    x_pts = x0 + radius * jnp.cos(theta)
    y_pts = y0 + radius * jnp.sin(theta)

    # Bilinear interpolation at each sample point
    u_samples = _bilinear_interpolate(u, grid, x_pts, y_pts)
    boundary_mean = jnp.mean(u_samples)

    # Interpolate u at centre
    u_center = _bilinear_interpolate(
        u, grid,
        jnp.array([x0]),
        jnp.array([y0]),
    )[0]

    abs_error = jnp.abs(u_center - boundary_mean)
    return u_center, boundary_mean, abs_error


def check_maximum_principle_2d(
    u: jnp.ndarray,
    grid: Grid2D,
) -> tuple[float, float, float, float, bool]:
    """Check the Maximum Principle for a 2D Laplace solution.

    For ∇²u = 0, the maximum and minimum values of u must lie on the
    boundary. This function extracts boundary values and compares
    interior extrema against them.

    Args:
        u: (nx, ny) scalar field (typically a Laplace solution).
        grid: The 2D grid.

    Returns:
        (interior_max, boundary_max, interior_min, boundary_min, holds)
        where holds is True if both:
            boundary_min <= interior_min  and  interior_max <= boundary_max
        i.e., all interior values lie within the boundary range.
    """
    nx, ny = grid.nx, grid.ny

    # Boundary mask: cells adjacent to the domain edges
    boundary_mask = jnp.zeros((nx, ny), dtype=bool)
    boundary_mask = boundary_mask.at[0, :].set(True)
    boundary_mask = boundary_mask.at[nx - 1, :].set(True)
    boundary_mask = boundary_mask.at[:, 0].set(True)
    boundary_mask = boundary_mask.at[:, ny - 1].set(True)

    interior_mask = ~boundary_mask

    u_boundary = u[boundary_mask]
    u_interior = u[interior_mask]

    if u_interior.size == 0:
        # All cells are boundary cells — trivially holds
        bmax = float(jnp.max(u_boundary))
        bmin = float(jnp.min(u_boundary))
        return bmax, bmax, bmin, bmin, True

    boundary_max = float(jnp.max(u_boundary))
    boundary_min = float(jnp.min(u_boundary))
    interior_max = float(jnp.max(u_interior))
    interior_min = float(jnp.min(u_interior))

    holds = bool(boundary_min <= interior_min and interior_max <= boundary_max)
    return interior_max, boundary_max, interior_min, boundary_min, holds


def _bilinear_interpolate(
    u: jnp.ndarray,
    grid: Grid2D,
    x_query: jnp.ndarray,
    y_query: jnp.ndarray,
) -> jnp.ndarray:
    """Bilinear interpolation of field u at query points (x_query, y_query).

    Args:
        u: (nx, ny) field values at cell centres.
        grid: The 2D grid.
        x_query: (N,) x-coordinates to interpolate at.
        y_query: (N,) y-coordinates to interpolate at.

    Returns:
        (N,) interpolated values.  Points outside the domain are
        clamped to the nearest cell centre.
    """
    nx, ny = grid.nx, grid.ny
    Lx, Ly = grid.Lx, grid.Ly

    # Fractional indices in continuous [0, nx) and [0, ny)
    fx = x_query / Lx * nx - 0.5
    fy = y_query / Ly * ny - 0.5

    i0 = jnp.clip(jnp.floor(fx).astype(int), 0, nx - 1)
    i1 = jnp.clip(i0 + 1, 0, nx - 1)
    j0 = jnp.clip(jnp.floor(fy).astype(int), 0, ny - 1)
    j1 = jnp.clip(j0 + 1, 0, ny - 1)

    wx = fx - i0
    wy = fy - j0

    # Clamp weights
    wx = jnp.clip(wx, 0.0, 1.0)
    wy = jnp.clip(wy, 0.0, 1.0)

    u00 = u[i0, j0]
    u10 = u[i1, j0]
    u01 = u[i0, j1]
    u11 = u[i1, j1]

    u0 = u00 * (1.0 - wx) + u10 * wx
    u1 = u01 * (1.0 - wx) + u11 * wx
    return u0 * (1.0 - wy) + u1 * wy

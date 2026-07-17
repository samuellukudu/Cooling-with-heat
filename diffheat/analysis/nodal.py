# diffheat/analysis/nodal.py
"""Nodal line / nodal surface computation for eigenfunctions.

Nodal lines are the curves where an eigenfunction φ(x, y) = 0.
They partition the domain into regions of opposite sign and are
important for understanding mode shapes (PDF §14).

This module uses marching squares to trace zero-crossing contours
on a 2D Cartesian grid.
"""
import jax.numpy as jnp

from ..mesh.grid2d import Grid2D


def compute_nodal_lines_2d(
    phi: jnp.ndarray, grid: Grid2D
) -> list[tuple[jnp.ndarray, jnp.ndarray]]:
    """Compute nodal lines (zero-crossing contours) of a 2D eigenfunction.

    Uses marching squares on the *dual grid*: each dual cell is defined
    by four adjacent cell centres at corners (i,j), (i+1,j), (i,j+1),
    (i+1,j+1).  Sign changes on the four edges of each dual cell are
    detected and zero-crossing points are linearly interpolated, then
    connected to form contour segments.

    Args:
        phi: (nx, ny) eigenfunction field at cell centres.
        grid: The 2D grid.

    Returns:
        List of (x_coords, y_coords) tuples, each representing a contour
        segment (line between two zero-crossing points on dual-cell edges).
    """
    nx, ny = grid.nx, grid.ny

    # Convert to plain numeric arrays for the host-side loop.
    phi_np = jnp.asarray(phi)
    xc = jnp.asarray(grid.x_centers)
    yc = jnp.asarray(grid.y_centers)

    segments: list[tuple[jnp.ndarray, jnp.ndarray]] = []

    # Dual cells: corners are cell centres (i,j), (i+1,j), (i,j+1), (i+1,j+1).
    for i in range(nx - 1):
        for j in range(ny - 1):
            v00 = float(phi_np[i, j])
            v10 = float(phi_np[i + 1, j])
            v01 = float(phi_np[i, j + 1])
            v11 = float(phi_np[i + 1, j + 1])

            crossing_pts: list[tuple[float, float]] = []

            # --- bottom edge: (i,j) → (i+1,j) ---
            if v00 * v10 < 0:
                t = -v00 / (v10 - v00)
                crossing_pts.append(
                    (float(xc[i]) + t * (float(xc[i + 1]) - float(xc[i])),
                     float(yc[j]))
                )
            elif v00 == 0 and v10 != 0:
                crossing_pts.append((float(xc[i]), float(yc[j])))
            elif v10 == 0 and v00 != 0:
                crossing_pts.append((float(xc[i + 1]), float(yc[j])))

            # --- top edge: (i,j+1) → (i+1,j+1) ---
            if v01 * v11 < 0:
                t = -v01 / (v11 - v01)
                crossing_pts.append(
                    (float(xc[i]) + t * (float(xc[i + 1]) - float(xc[i])),
                     float(yc[j + 1]))
                )
            elif v01 == 0 and v11 != 0:
                crossing_pts.append((float(xc[i]), float(yc[j + 1])))
            elif v11 == 0 and v01 != 0:
                crossing_pts.append((float(xc[i + 1]), float(yc[j + 1])))

            # --- left edge: (i,j) → (i,j+1) ---
            if v00 * v01 < 0:
                t = -v00 / (v01 - v00)
                crossing_pts.append(
                    (float(xc[i]),
                     float(yc[j]) + t * (float(yc[j + 1]) - float(yc[j])))
                )
            elif v00 == 0 and v01 != 0:
                crossing_pts.append((float(xc[i]), float(yc[j])))
            elif v01 == 0 and v00 != 0:
                crossing_pts.append((float(xc[i]), float(yc[j + 1])))

            # --- right edge: (i+1,j) → (i+1,j+1) ---
            if v10 * v11 < 0:
                t = -v10 / (v11 - v10)
                crossing_pts.append(
                    (float(xc[i + 1]),
                     float(yc[j]) + t * (float(yc[j + 1]) - float(yc[j])))
                )
            elif v10 == 0 and v11 != 0:
                crossing_pts.append((float(xc[i + 1]), float(yc[j])))
            elif v11 == 0 and v10 != 0:
                crossing_pts.append((float(xc[i + 1]), float(yc[j + 1])))

            # Deduplicate near-coincident points (can happen at corners where
            # two edges both register the same zero).
            crossing_pts = _dedup_points(crossing_pts)

            # Connect pairs of crossing points.
            if len(crossing_pts) == 2:
                p0, p1 = crossing_pts
                segments.append(
                    (jnp.array([p0[0], p1[0]]), jnp.array([p0[1], p1[1]]))
                )
            elif len(crossing_pts) == 4:
                # Ambiguous (saddle) case — resolve using cell-centre value.
                # Connect so that the two segments separate positive corners
                # from negative corners.
                cx = 0.5 * (float(xc[i]) + float(xc[i + 1]))
                cy = 0.5 * (float(yc[j]) + float(yc[j + 1]))
                # Bilinear interpolant at centre
                v_center = 0.25 * (v00 + v10 + v01 + v11)

                # Pair points by proximity along the cell boundary.
                # Sort points by angle around the dual-cell centre.
                def _angle(p):
                    return float(jnp.arctan2(p[1] - cy, p[0] - cx))

                sorted_pts = sorted(crossing_pts, key=_angle)
                # Connect (0,1) and (2,3) — this separates the quadrants.
                p0, p1 = sorted_pts[0], sorted_pts[1]
                p2, p3 = sorted_pts[2], sorted_pts[3]
                # Choose pairing that respects sign: if v00 and v11 have the
                # same sign, connect (0,2) and (1,3); otherwise (0,1) and (2,3).
                # The standard marching-squares lookup table handles this, but
                # a simple heuristic based on the average value works well.
                if v00 * v_center > 0:
                    # Connect so that the positive region around v00 is separated
                    # from the negative region.
                    segments.append(
                        (jnp.array([sorted_pts[0][0], sorted_pts[1][0]]),
                         jnp.array([sorted_pts[0][1], sorted_pts[1][1]]))
                    )
                    segments.append(
                        (jnp.array([sorted_pts[2][0], sorted_pts[3][0]]),
                         jnp.array([sorted_pts[2][1], sorted_pts[3][1]]))
                    )
                else:
                    segments.append(
                        (jnp.array([sorted_pts[0][0], sorted_pts[3][0]]),
                         jnp.array([sorted_pts[0][1], sorted_pts[3][1]]))
                    )
                    segments.append(
                        (jnp.array([sorted_pts[1][0], sorted_pts[2][0]]),
                         jnp.array([sorted_pts[1][1], sorted_pts[2][1]]))
                    )
            # len == 0 or 1 or 3: degenerate — skip.

    return segments


def _dedup_points(
    pts: list[tuple[float, float]], tol: float = 1e-12
) -> list[tuple[float, float]]:
    """Remove near-duplicate points from a list."""
    if len(pts) <= 1:
        return pts
    out = []
    for p in pts:
        is_dup = False
        for q in out:
            if abs(p[0] - q[0]) < tol and abs(p[1] - q[1]) < tol:
                is_dup = True
                break
        if not is_dup:
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# Nodal lines for the disc (PDF §14.2)
# ---------------------------------------------------------------------------

def compute_nodal_lines_disc(
    m: int,
    n: int,
    kind: str = "cos",
    R: float = 1.0,
    nr: int = 200,
    ntheta: int = 400,
) -> list[tuple[jnp.ndarray, jnp.ndarray]]:
    """Compute nodal lines for an eigenfunction of the disc.

    The (m,n) eigenfunction on a disc of radius *R* is:

        v_{m,n}(r,θ) = J_m(r · j_{m,n} / R) · {cos(mθ)  if kind='cos'
                                                 {sin(mθ)  if kind='sin'}

    Nodal lines are (PDF §14.2):

    - **m = 0**: concentric circles at radii
      r_k = R · j_{0,k} / j_{0,n}   for k = 1, …, n-1,
      plus the boundary r = R.

    - **m ≥ 1**: the same concentric circles (from J_m zeros for k < n)
      **plus** diameters at angles
      θ_k = kπ/m   (for kind='sin') or θ_k = (k+½)π/m (for kind='cos')
      for k = 0, …, 2m-1.

    Args:
        m: Angular mode number.
        n: Radial mode number (1-based).
        kind: ``'cos'`` or ``'sin'`` — angular factor.
        R: Disc radius (default 1).
        nr: Radial resolution for circle sampling.
        ntheta: Angular resolution for circle/diameter sampling.

    Returns:
        List of ``(x_coords, y_coords)`` tuples, each representing a
        nodal curve (concentric circle or diameter segment).
    """
    from ..physics.bessel import bessel_j_zero

    j_mn = bessel_j_zero(m, n)
    lines: list[tuple[jnp.ndarray, jnp.ndarray]] = []

    # --- Concentric circles from interior zeros of J_m ---
    # For k < n, r_k = R * j_{m,k} / j_{m,n} is a nodal circle.
    for k in range(1, n):
        j_mk = bessel_j_zero(m, k)
        r_k = R * j_mk / j_mn
        theta_circle = jnp.linspace(-jnp.pi, jnp.pi, ntheta)
        x_circle = r_k * jnp.cos(theta_circle)
        y_circle = r_k * jnp.sin(theta_circle)
        lines.append((x_circle, y_circle))

    # --- Boundary circle ---
    theta_bd = jnp.linspace(-jnp.pi, jnp.pi, ntheta)
    lines.append((R * jnp.cos(theta_bd), R * jnp.sin(theta_bd)))

    # --- Diameters (m ≥ 1) ---
    if m >= 1:
        r_line = jnp.linspace(0.0, R, nr)
        if kind == "sin":
            # sin(mθ) = 0 → mθ = kπ → θ = kπ/m for k = 0, …, 2m-1
            for k in range(2 * m):
                theta_k = k * jnp.pi / m
                x_diam = r_line * jnp.cos(theta_k)
                y_diam = r_line * jnp.sin(theta_k)
                lines.append((x_diam, y_diam))
        else:  # kind == "cos"
            # cos(mθ) = 0 → mθ = (k+½)π → θ = (k+½)π/m
            for k in range(2 * m):
                theta_k = (k + 0.5) * jnp.pi / m
                x_diam = r_line * jnp.cos(theta_k)
                y_diam = r_line * jnp.sin(theta_k)
                lines.append((x_diam, y_diam))

    return lines

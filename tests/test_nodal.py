# tests/test_nodal.py
"""Tests for nodal line computation."""
import jax.numpy as jnp
import pytest

from diffheat import Grid2D, compute_nodal_lines_2d, compute_nodal_lines_disc


class TestNodalLines2D:
    def test_single_mode_x(self):
        """sin(2*pi*x) on [0,1]² has nodal line at x=0.5."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=40, ny=40)
        phi = jnp.sin(2.0 * jnp.pi * grid.X.T)

        segments = compute_nodal_lines_2d(phi, grid)
        assert len(segments) > 0

        # Nodal segments should be near x=0.5
        for sx, sy in segments:
            assert jnp.all(jnp.abs(sx - 0.5) < 0.1)

    def test_single_mode_y(self):
        """sin(2*pi*y) on [0,1]² has nodal line at y=0.5."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=40, ny=40)
        phi = jnp.sin(2.0 * jnp.pi * grid.Y.T)

        segments = compute_nodal_lines_2d(phi, grid)
        assert len(segments) > 0
        # Segments should be near y=0.5
        for sx, sy in segments:
            assert jnp.all(jnp.abs(sy - 0.5) < 0.1)

    def test_product_mode(self):
        """sin(pi*x)*sin(2*pi*y) has nodal line at y=0.5."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=40, ny=40)
        phi = jnp.sin(jnp.pi * grid.X.T) * jnp.sin(2.0 * jnp.pi * grid.Y.T)

        segments = compute_nodal_lines_2d(phi, grid)
        assert len(segments) > 0

        # There should be some segments near y=0.5
        found_y_mid = False
        for sx, sy in segments:
            if jnp.all(jnp.abs(sy - 0.5) < 0.1):
                found_y_mid = True
                break
        assert found_y_mid, "Expected nodal line near y=0.5"

    def test_no_crossing_positive_field(self):
        """A strictly positive field should yield no nodal segments."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=20, ny=20)
        phi = jnp.ones((20, 20)) + 0.1 * jnp.sin(grid.X.T)

        segments = compute_nodal_lines_2d(phi, grid)
        assert len(segments) == 0

    def test_empty_for_zero_crossing(self):
        """Even if some values are zero but no sign change, should handle."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=20, ny=20)
        phi = jnp.abs(jnp.sin(jnp.pi * grid.X.T))  # non-negative

        segments = compute_nodal_lines_2d(phi, grid)
        # No true sign changes (all >= 0)
        assert len(segments) == 0


class TestNodalLinesDisc:
    def test_m0_n1_only_boundary(self):
        """(0,1) mode has only the boundary circle as nodal line."""
        lines = compute_nodal_lines_disc(m=0, n=1, R=1.0)
        # One line: the boundary circle at r=R
        assert len(lines) == 1  # boundary only

    def test_m0_n2_concentric(self):
        """(0,2) mode: boundary + one interior concentric circle."""
        lines = compute_nodal_lines_disc(m=0, n=2, R=1.0)
        # Two lines: interior circle at r = j_{0,1}/j_{0,2} + boundary
        assert len(lines) == 2

    def test_m1_n1_has_diameters(self):
        """(1,1) cos mode: boundary + 2 diameters (θ = π/2, 3π/2)."""
        lines = compute_nodal_lines_disc(m=1, n=1, kind="cos", R=1.0)
        # boundary + 2 diameter lines
        assert len(lines) == 3  # 1 boundary + 2 diameters

    def test_m1_n1_sin_has_diameters(self):
        """(1,1) sin mode: boundary + 2 diameters (θ = 0, π)."""
        lines = compute_nodal_lines_disc(m=1, n=1, kind="sin", R=1.0)
        assert len(lines) == 3  # 1 boundary + 2 diameters

    def test_all_points_within_disc(self):
        """All nodal line points should lie within or on the disc."""
        lines = compute_nodal_lines_disc(m=2, n=3, kind="cos", R=2.0)
        for x, y in lines:
            r = jnp.sqrt(x ** 2 + y ** 2)
            assert jnp.all(r <= 2.0 + 1e-10)

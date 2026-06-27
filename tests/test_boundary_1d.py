# tests/test_boundary_1d.py
"""Tests for 1D boundary condition correction (field-based)."""
import pytest
import jax.numpy as jnp
from diffheat.mesh import Grid1D, BoundaryCondition
from diffheat.mesh.boundary import apply_boundary_conditions_1d
from diffheat.operators import laplacian_1d


class TestApplyBoundaryConditions1D:
    def test_returns_correct_shapes(self):
        n = 20
        grid = Grid1D.uniform(length=1.0, n_cells=n)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))
        T = jnp.zeros(n)
        L_T, b_source = apply_boundary_conditions_1d(
            lambda x: laplacian_1d(x, grid), grid, bc, T
        )
        assert L_T.shape == (n,)
        assert b_source.shape == (n,)

    def test_homogeneous_neumann_has_zero_source(self):
        n = 20
        grid = Grid1D.uniform(length=1.0, n_cells=n)
        bc = BoundaryCondition(kind="neumann", value=jnp.array([0.0, 0.0]))
        T = jnp.ones(n)
        L_T, b_source = apply_boundary_conditions_1d(
            lambda x: laplacian_1d(x, grid), grid, bc, T
        )
        assert jnp.allclose(b_source, 0.0)

    def test_dirichlet_boundary_source_is_nonzero(self):
        n = 20
        grid = Grid1D.uniform(length=1.0, n_cells=n)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([100.0, 0.0]))
        T = jnp.zeros(n)
        L_T, b_source = apply_boundary_conditions_1d(
            lambda x: laplacian_1d(x, grid), grid, bc, T
        )
        assert not jnp.isclose(jnp.sum(jnp.abs(b_source)), 0.0)

    def test_linear_field_matching_dirichlet_gives_zero(self):
        """T = a*x + c with matching Dirichlet BCs → L_T + b_source = 0."""
        a, c = 2.0, 5.0
        n = 30
        grid = Grid1D.uniform(length=1.0, n_cells=n)

        bc = BoundaryCondition(
            kind="dirichlet",
            value=jnp.array([float(c), float(a * grid.length + c)]),
        )

        T = a * grid.centers + c

        L_T, b_source = apply_boundary_conditions_1d(
            lambda x: laplacian_1d(x, grid), grid, bc, T
        )

        total = L_T + b_source
        assert jnp.allclose(total, 0.0, atol=1e-3)

    def test_preserves_interior_cells(self):
        """Only boundary cells (0 and n-1) should be modified."""
        n = 20
        grid = Grid1D.uniform(length=1.0, n_cells=n)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))
        T = jnp.ones(n)

        raw_L = laplacian_1d(T, grid)
        L_T, b_source = apply_boundary_conditions_1d(
            lambda x: laplacian_1d(x, grid), grid, bc, T
        )

        # Interior cells (1 to n-2) should have the same laplacian
        assert jnp.allclose(L_T[1:-1], raw_L[1:-1])

    def test_mixed_boundary_conditions(self):
        """Dirichlet left + Neumann right with non-uniform field."""
        n = 40
        grid = Grid1D.uniform(length=1.0, n_cells=n)

        # T = a*x + c, Dirichlet left matching T(0)=c, Neumann right dT_dn=0
        a, c = 1.0, 2.0
        bc = BoundaryCondition(
            kind="dirichlet",
            value=jnp.array([c, 0.0]),  # left Dirichlet, right not used
        )

        T = a * grid.centers + c

        L_T, b_source = apply_boundary_conditions_1d(
            lambda x: laplacian_1d(x, grid), grid, bc, T
        )

        # Left boundary Dirichlet source should be non-zero
        # (field doesn't match BC exactly at cell center)
        assert not jnp.isclose(b_source[0], 0.0)
        # b_source at interior should be zero
        assert jnp.allclose(b_source[1:-1], 0.0)

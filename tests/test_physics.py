# tests/test_physics.py
import pytest
import jax.numpy as jnp
from diffheat.mesh import Grid1D, BoundaryCondition
from diffheat.physics import make_laplacian, apply_boundary_conditions, HeatEquation1D


class TestMakeLaplacian:
    def test_returns_square_matrix(self):
        grid = Grid1D.uniform(length=1.0, n_cells=10)
        L = make_laplacian(grid)
        assert L.shape == (10, 10)

    def test_interior_rows_sum_to_zero(self):
        grid = Grid1D.uniform(length=1.0, n_cells=10)
        L = make_laplacian(grid)
        # Interior rows (not first or last) should sum to zero
        for i in range(1, grid.n_cells - 1):
            assert jnp.isclose(jnp.sum(L[i]), 0.0), f"Row {i} sum = {jnp.sum(L[i])}"

    def test_laplacian_of_linear_is_zero(self):
        """d^2(ax+b)/dx^2 = 0, so L @ (a*x + b) should be near zero in interior."""
        grid = Grid1D.uniform(length=1.0, n_cells=20)
        L = make_laplacian(grid)
        a, b = 2.0, 1.0
        T_linear = a * grid.centers + b
        result = L @ T_linear
        # Interior cells should be ~0 (boundary effects at edges)
        assert jnp.allclose(result[2:-2], 0.0, atol=1e-3)

    def test_uniform_grid_dx_squared_scaling(self):
        """Laplacian entries should scale as 1/dx^2."""
        grid_coarse = Grid1D.uniform(length=1.0, n_cells=10)
        grid_fine = Grid1D.uniform(length=1.0, n_cells=20)
        L_coarse = make_laplacian(grid_coarse)
        L_fine = make_laplacian(grid_fine)
        # Ratio of diagonal magnitudes should be ~(dx_fine/dx_coarse)^2 = 1/4
        ratio = jnp.abs(L_fine[5, 5]) / jnp.abs(L_coarse[2, 2])
        assert jnp.isclose(ratio, 4.0, rtol=0.01)


class TestApplyBoundaryConditions:
    def test_dirichlet_modifies_boundary_rows(self):
        grid = Grid1D.uniform(length=1.0, n_cells=10)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.5]))
        L = make_laplacian(grid)
        L_mod, b_source = apply_boundary_conditions(L, grid, bc)
        # Boundary source should be non-zero at edges for Dirichlet
        assert not jnp.isclose(b_source[0], 0.0)
        assert not jnp.isclose(b_source[-1], 0.0)

    def test_neumann_boundary_source_is_zero_for_homogeneous(self):
        grid = Grid1D.uniform(length=1.0, n_cells=10)
        bc = BoundaryCondition(kind="neumann", value=jnp.array([0.0, 0.0]))
        L = make_laplacian(grid)
        L_mod, b_source = apply_boundary_conditions(L, grid, bc)
        # Zero-flux Neumann: no boundary source term
        assert jnp.allclose(b_source, 0.0)

    def test_steady_state_dirichlet_is_linear(self):
        """Steady state with Dirichlet BCs T(0)=T_left, T(L)=T_right should be linear."""
        grid = Grid1D.uniform(length=1.0, n_cells=20)
        T_left, T_right = 1.0, 0.0
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([T_left, T_right]))
        L = make_laplacian(grid)
        L_mod, b_source = apply_boundary_conditions(L, grid, bc)
        # Solve steady state: L_mod @ T + b_source = 0
        T_steady = jnp.linalg.solve(L_mod, -b_source)
        # Should be linear from T_left to T_right
        expected = T_left + (T_right - T_left) * grid.centers / grid.length
        assert jnp.allclose(T_steady, expected, atol=1e-6)


class TestHeatEquation1D:
    def test_construction(self):
        grid = Grid1D.uniform(length=1.0, n_cells=10)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))
        eqn = HeatEquation1D(grid=grid, bc=bc, alpha=0.1)
        assert eqn.alpha == 0.1
        assert eqn.source is None

    def test_construction_with_source(self):
        grid = Grid1D.uniform(length=1.0, n_cells=10)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))

        def source(x, t):
            return jnp.sin(x)

        eqn = HeatEquation1D(grid=grid, bc=bc, alpha=0.1, source=source)
        assert eqn.source is source

    def test_frozen_dataclass(self):
        grid = Grid1D.uniform(length=1.0, n_cells=10)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))
        eqn = HeatEquation1D(grid=grid, bc=bc, alpha=0.1)
        with pytest.raises(Exception):
            eqn.alpha = 0.2

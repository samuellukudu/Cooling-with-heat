# tests/test_steady_state.py
"""Tests for steady-state heat equation (Laplace/Poisson) solvers."""
import jax
import jax.numpy as jnp
import pytest

from diffheat import (
    BoundaryCondition,
    BoundaryCondition2D,
    BoundaryCondition3D,
    Grid1D,
    Grid2D,
    Grid3D,
    HeatEquation1D,
    HeatEquation2D,
    HeatEquation3D,
    solve_steady_state_1d,
    solve_steady_state_2d,
    solve_steady_state_3d,
)


class TestSteadyState1D:
    def test_linear_solution_no_source(self):
        """1D with left=10.0, right=20.0 and zero source should be a straight line."""
        grid = Grid1D.uniform(length=1.0, n_cells=20)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([10.0, 20.0]))
        eqn = HeatEquation1D(grid=grid, bc=bc, alpha=1.0)

        T_steady = solve_steady_state_1d(eqn)
        T_expected = 10.0 + 10.0 * grid.centers

        # Boundaries might have slight FDM errors, but should match closely
        assert jnp.allclose(T_steady, T_expected, atol=1e-5)


class TestSteadyState2D:
    def test_linear_solution_no_source(self):
        """2D with left=0.0, right=1.0, and top/bottom insulated should give T(x,y) = x."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=10, ny=10)
        bc = BoundaryCondition2D(
            left={"kind": "dirichlet", "value": 0.0},
            right={"kind": "dirichlet", "value": 1.0},
            bottom={"kind": "neumann", "value": 0.0},
            top={"kind": "neumann", "value": 0.0},
        )
        eqn = HeatEquation2D(grid=grid, bc=bc, alpha=2.0)

        T_steady = solve_steady_state_2d(eqn)
        assert T_steady.shape == (10, 10)

        # Expected: T_expected[i, j] = grid.x_centers[i]
        X = grid.X.T
        assert jnp.allclose(T_steady, X, atol=1e-5)

    def test_gradient_wrt_bc(self):
        """Gradients of the steady-state solution flow w.r.t boundary values."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=8, ny=8)

        def loss_fn(val):
            bc = BoundaryCondition2D(
                left={"kind": "dirichlet", "value": val},
                right={"kind": "dirichlet", "value": 0.0},
                bottom={"kind": "neumann", "value": 0.0},
                top={"kind": "neumann", "value": 0.0},
            )
            eqn = HeatEquation2D(grid=grid, bc=bc, alpha=1.0)
            T_steady = solve_steady_state_2d(eqn)
            return jnp.mean(T_steady)

        grad = jax.grad(loss_fn)(10.0)
        # Increasing left BC should increase the mean temperature, so gradient > 0
        assert grad > 0.0


class TestSteadyState3D:
    def test_linear_solution_no_source(self):
        """3D Cube with left=0.0, right=1.0, other walls insulated: T(x,y,z) = x."""
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=6, ny=6, nz=6)
        bc = BoundaryCondition3D(
            left={"kind": "dirichlet", "value": 0.0},
            right={"kind": "dirichlet", "value": 1.0},
            bottom={"kind": "neumann", "value": 0.0},
            top={"kind": "neumann", "value": 0.0},
            front={"kind": "neumann", "value": 0.0},
            back={"kind": "neumann", "value": 0.0},
        )
        eqn = HeatEquation3D(grid=grid, bc=bc, alpha=0.5)

        T_steady = solve_steady_state_3d(eqn)
        assert T_steady.shape == (6, 6, 6)
        assert jnp.allclose(T_steady, grid.X, atol=1e-5)

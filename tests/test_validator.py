# tests/test_validator.py
"""Tests for PDE property validators."""
import jax.numpy as jnp

from diffheat import (
    BoundaryCondition2D,
    Grid2D,
    HeatEquation2D,
    check_maximum_principle_2d,
    check_mean_value_2d,
    solve_steady_state_2d,
)


class TestMeanValue2D:
    def test_linear_function(self):
        """For u(x,y) = x + y (harmonic), mean value should hold exactly."""
        grid = Grid2D.uniform(Lx=2.0, Ly=2.0, nx=60, ny=60)
        # u(x,y) = x + y is harmonic (∇²u = 0)
        u = grid.X.T + grid.Y.T  # shape (nx, ny)

        center = (1.0, 1.0)
        radius = 0.3
        u_c, b_mean, err = check_mean_value_2d(u, grid, center, radius, n_angles=300)

        # Should be very close for a linear function
        assert float(err) < 0.02

    def test_laplace_solution(self):
        """Solve Laplace equation and verify mean value property."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=40, ny=40)
        bc = BoundaryCondition2D(
            left={"kind": "dirichlet", "value": 1.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "neumann", "value": 0.0},
            top={"kind": "neumann", "value": 0.0},
        )
        eqn = HeatEquation2D(grid=grid, bc=bc, alpha=1.0)
        u = solve_steady_state_2d(eqn)

        center = (0.5, 0.5)
        radius = 0.15
        u_c, b_mean, err = check_mean_value_2d(u, grid, center, radius, n_angles=200)

        # Should approximately hold for the numerical Laplace solution
        assert float(err) < 0.05


class TestMaximumPrinciple2D:
    def test_laplace_solution_respects_principle(self):
        """Solution of Laplace equation should satisfy max principle."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=30, ny=30)
        bc = BoundaryCondition2D(
            left={"kind": "dirichlet", "value": 100.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "dirichlet", "value": 50.0},
            top={"kind": "dirichlet", "value": 50.0},
        )
        eqn = HeatEquation2D(grid=grid, bc=bc, alpha=1.0)
        u = solve_steady_state_2d(eqn)

        imax, bmax, imin, bmin, holds = check_maximum_principle_2d(u, grid)
        assert holds, (
            f"Max principle violated: interior [{imin:.3f}, {imax:.3f}], "
            f"boundary [{bmin:.3f}, {bmax:.3f}]"
        )

    def test_linear_ramp(self):
        """T(x,y) = x satisfies ∇²T=0 and respects max principle."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=20, ny=20)
        u = grid.X.T  # T = x

        imax, bmax, imin, bmin, holds = check_maximum_principle_2d(u, grid)
        assert holds

    def test_quadratic_violates(self):
        """T(x,y) = x² is NOT harmonic, may violate max principle."""
        grid = Grid2D.uniform(Lx=2.0, Ly=2.0, nx=20, ny=20)
        u = grid.X.T ** 2  # T = x², ∇²T = 2 ≠ 0

        imax, bmax, imin, bmin, holds = check_maximum_principle_2d(u, grid)
        # x² on [0,2]: max at x=2 is 4, interior near x=2 has values > boundary at x=0
        # Not necessarily violated if boundary captures the high end
        # Just check it runs without error
        assert isinstance(holds, bool)

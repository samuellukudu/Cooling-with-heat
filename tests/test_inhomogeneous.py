# tests/test_inhomogeneous.py
"""Tests for inhomogeneous BC convenience API."""
import jax
import jax.numpy as jnp
import pytest

from diffheat import (
    BoundaryCondition2D,
    BoundaryCondition3D,
    Grid2D,
    Grid3D,
    HeatEquation2D,
    HeatEquation3D,
    solve_heat_2d,
    solve_heat_3d,
    solve_heat_inhomogeneous_2d,
    solve_heat_inhomogeneous_3d,
    solve_steady_state_2d,
    solve_steady_state_3d,
)


class TestInhomogeneous2D:
    def test_equivalence_to_manual_decomposition(self):
        """The inhomogeneous solver should match manual u=uE+v decomposition."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=20, ny=20)
        alpha = 0.1

        # Inhomogeneous BCs: hot left, cold right
        bc = BoundaryCondition2D(
            left={"kind": "dirichlet", "value": 100.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "neumann", "value": 0.0},
            top={"kind": "neumann", "value": 0.0},
        )
        eqn = HeatEquation2D(grid=grid, bc=bc, alpha=alpha)

        T0 = jnp.zeros((20, 20))

        dt = 0.001
        t_span = (0.0, 0.01)
        traj_auto = solve_heat_inhomogeneous_2d(eqn, T0, t_span, dt)

        # Manual decomposition
        uE = solve_steady_state_2d(eqn)
        bc_homog = BoundaryCondition2D(
            left={"kind": "dirichlet", "value": 0.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "neumann", "value": 0.0},
            top={"kind": "neumann", "value": 0.0},
        )
        eqn_homog = HeatEquation2D(grid=grid, bc=bc_homog, alpha=alpha)
        v0 = T0 - uE
        v_traj = solve_heat_2d(eqn_homog, v0, t_span, dt)
        traj_manual = v_traj + uE[jnp.newaxis, :, :]

        assert jnp.allclose(traj_auto, traj_manual, atol=1e-10)

    def test_trajectory_shape(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=16, ny=16)
        bc = BoundaryCondition2D(
            left={"kind": "dirichlet", "value": 100.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "dirichlet", "value": 50.0},
            top={"kind": "dirichlet", "value": 50.0},
        )
        eqn = HeatEquation2D(grid=grid, bc=bc, alpha=0.1)
        T0 = jnp.zeros((16, 16))

        dt = 0.001
        traj = solve_heat_inhomogeneous_2d(eqn, T0, (0.0, 0.005), dt)
        # 5 steps + initial = 6 frames
        assert traj.shape == (6, 16, 16)
        assert jnp.allclose(traj[0], T0)

    def test_differentiable(self):
        """Gradient through inhomogeneous solver should work."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=12, ny=12)

        def loss(left_val):
            bc = BoundaryCondition2D(
                left={"kind": "dirichlet", "value": left_val},
                right={"kind": "dirichlet", "value": 0.0},
                bottom={"kind": "neumann", "value": 0.0},
                top={"kind": "neumann", "value": 0.0},
            )
            eqn = HeatEquation2D(grid=grid, bc=bc, alpha=0.1)
            T0 = jnp.zeros((12, 12))
            traj = solve_heat_inhomogeneous_2d(eqn, T0, (0.0, 0.002), 0.001)
            return jnp.mean(traj[-1])

        grad = jax.grad(loss)(100.0)
        assert float(grad) > 0  # Higher left BC => higher mean temperature


class TestInhomogeneous3D:
    def test_equivalence_to_manual(self):
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=8, ny=8, nz=8)
        alpha = 0.1

        bc = BoundaryCondition3D(
            left={"kind": "dirichlet", "value": 100.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "dirichlet", "value": 0.0},
            top={"kind": "dirichlet", "value": 0.0},
            front={"kind": "dirichlet", "value": 0.0},
            back={"kind": "dirichlet", "value": 0.0},
        )
        eqn = HeatEquation3D(grid=grid, bc=bc, alpha=alpha)
        T0 = jnp.zeros((8, 8, 8))

        dt = 0.005
        traj_auto = solve_heat_inhomogeneous_3d(eqn, T0, (0.0, 0.01), dt)

        # Manual decomposition
        uE = solve_steady_state_3d(eqn)
        bc_homog = BoundaryCondition3D(
            left={"kind": "dirichlet", "value": 0.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "dirichlet", "value": 0.0},
            top={"kind": "dirichlet", "value": 0.0},
            front={"kind": "dirichlet", "value": 0.0},
            back={"kind": "dirichlet", "value": 0.0},
        )
        eqn_homog = HeatEquation3D(grid=grid, bc=bc_homog, alpha=alpha)
        v0 = T0 - uE
        v_traj = solve_heat_3d(eqn_homog, v0, (0.0, 0.01), dt)
        traj_manual = v_traj + uE[jnp.newaxis, :, :, :]

        assert jnp.allclose(traj_auto, traj_manual, atol=1e-10)

    def test_trajectory_shape(self):
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=8, ny=8, nz=8)
        bc = BoundaryCondition3D(
            left={"kind": "dirichlet", "value": 100.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "dirichlet", "value": 0.0},
            top={"kind": "dirichlet", "value": 0.0},
            front={"kind": "dirichlet", "value": 0.0},
            back={"kind": "dirichlet", "value": 0.0},
        )
        eqn = HeatEquation3D(grid=grid, bc=bc, alpha=0.1)
        T0 = jnp.zeros((8, 8, 8))

        dt = 0.005
        traj = solve_heat_inhomogeneous_3d(eqn, T0, (0.0, 0.01), dt)
        # 2 steps + initial = 3 frames
        assert traj.shape == (3, 8, 8, 8)

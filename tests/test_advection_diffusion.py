# tests/test_advection_diffusion.py
"""Tests for advection-diffusion solvers and CFL conditions."""
import jax.numpy as jnp
import pytest
from diffheat.mesh import Grid1D, Grid2D, Grid3D


class TestCFLAdvectionDiffusion1D:
    @pytest.fixture
    def grid(self):
        return Grid1D.uniform(length=1.0, n_cells=50)

    def test_stable_dt_passes(self, grid):
        from diffheat.solvers.stability import check_cfl_advection_diffusion_1d
        alpha = 0.01
        u_max = 1.0
        dx_min = float(jnp.min(grid.dx))
        dt_diff = dx_min**2 / (2 * alpha)
        dt_adv = dx_min / u_max
        dt_max = min(dt_diff, dt_adv)
        assert check_cfl_advection_diffusion_1d(grid, alpha, u_max, 0.9 * dt_max)

    def test_unstable_diffusive_fails(self, grid):
        from diffheat.solvers.stability import check_cfl_advection_diffusion_1d
        alpha = 0.01
        u_max = 0.0  # pure diffusion
        dx_min = float(jnp.min(grid.dx))
        dt_limit = dx_min**2 / (2 * alpha)
        assert not check_cfl_advection_diffusion_1d(grid, alpha, u_max, 2.0 * dt_limit)

    def test_unstable_advective_fails(self, grid):
        from diffheat.solvers.stability import check_cfl_advection_diffusion_1d
        alpha = 0.0  # pure advection
        u_max = 2.0
        dx_min = float(jnp.min(grid.dx))
        dt_limit = dx_min / u_max
        if dt_limit > 0:
            assert not check_cfl_advection_diffusion_1d(grid, alpha, u_max, 2.0 * dt_limit)

    def test_zero_velocity_matches_heat_cfl(self, grid):
        from diffheat.solvers.stability import check_cfl_advection_diffusion_1d, check_cfl
        alpha = 0.01
        dt = 0.001
        assert check_cfl_advection_diffusion_1d(grid, alpha, 0.0, dt) == check_cfl(grid, alpha, dt)


class TestCFLAdvectionDiffusion2D:
    @pytest.fixture
    def grid(self):
        return Grid2D.uniform(Lx=1.0, Ly=1.0, nx=40, ny=40)

    def test_stable_dt_passes(self, grid):
        from diffheat.solvers.stability import check_cfl_advection_diffusion_2d
        alpha = 0.01
        u_x_max = 1.0
        u_y_max = 0.5
        dx_min = float(jnp.min(grid.dx))
        dy_min = float(jnp.min(grid.dy))
        dt_diff = min(dx_min**2, dy_min**2) / (4 * alpha)
        dt_adv = 1.0 / (u_x_max / dx_min + u_y_max / dy_min)
        dt_max = min(dt_diff, dt_adv)
        assert check_cfl_advection_diffusion_2d(grid, alpha, u_x_max, u_y_max, 0.9 * dt_max)

    def test_unstable_dt_fails(self, grid):
        from diffheat.solvers.stability import check_cfl_advection_diffusion_2d
        alpha = 0.01
        u_x_max = 10.0
        u_y_max = 10.0
        assert not check_cfl_advection_diffusion_2d(grid, alpha, u_x_max, u_y_max, 0.1)

    def test_zero_velocity_matches_heat_cfl(self, grid):
        from diffheat.solvers.stability import check_cfl_advection_diffusion_2d, check_cfl_2d
        alpha = 0.01
        dt = 0.001
        assert check_cfl_advection_diffusion_2d(grid, alpha, 0.0, 0.0, dt) == check_cfl_2d(grid, alpha, dt)


class TestCFLAdvectionDiffusion3D:
    @pytest.fixture
    def grid(self):
        return Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=20, ny=20, nz=20)

    def test_stable_dt_passes(self, grid):
        from diffheat.solvers.stability import check_cfl_advection_diffusion_3d
        alpha = 0.01
        u_max = 1.0
        dx_min = float(jnp.min(grid.dx))
        dy_min = float(jnp.min(grid.dy))
        dz_min = float(jnp.min(grid.dz))
        dt_diff = min(dx_min**2, dy_min**2, dz_min**2) / (6 * alpha)
        dt_adv = 1.0 / (u_max / dx_min + u_max / dy_min + u_max / dz_min)
        dt_max = min(dt_diff, dt_adv)
        assert check_cfl_advection_diffusion_3d(grid, alpha, u_max, u_max, u_max, 0.9 * dt_max)

    def test_zero_velocity_matches_heat_cfl(self, grid):
        from diffheat.solvers.stability import check_cfl_advection_diffusion_3d, check_cfl_3d
        alpha = 0.01
        dt = 0.0001
        assert check_cfl_advection_diffusion_3d(grid, alpha, 0.0, 0.0, 0.0, dt) == check_cfl_3d(grid, alpha, dt)


class TestAdvectionDiffusionPhysics:
    def test_1d_creation(self):
        from diffheat.physics.advection_diffusion import AdvectionDiffusion1D
        from diffheat.mesh import Grid1D, BoundaryCondition
        import jax.numpy as jnp

        grid = Grid1D.uniform(length=1.0, n_cells=50)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([100.0, 0.0]))

        def velocity(x, t):
            return jnp.ones_like(x)

        eqn = AdvectionDiffusion1D(grid=grid, bc=bc, alpha=0.01, velocity=velocity)
        assert eqn.alpha == 0.01
        assert eqn.grid is grid
        assert eqn.source is None

    def test_1d_negative_alpha_raises(self):
        from diffheat.physics.advection_diffusion import AdvectionDiffusion1D
        from diffheat.mesh import Grid1D, BoundaryCondition
        import jax.numpy as jnp

        grid = Grid1D.uniform(length=1.0, n_cells=50)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([100.0, 0.0]))

        def velocity(x, t):
            return jnp.ones_like(x)

        with pytest.raises(ValueError, match="alpha must be positive"):
            AdvectionDiffusion1D(grid=grid, bc=bc, alpha=-0.01, velocity=velocity)

    def test_1d_with_source(self):
        from diffheat.physics.advection_diffusion import AdvectionDiffusion1D
        from diffheat.mesh import Grid1D, BoundaryCondition
        import jax.numpy as jnp

        grid = Grid1D.uniform(length=1.0, n_cells=50)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([100.0, 0.0]))

        def velocity(x, t):
            return jnp.ones_like(x)

        def source(x, t):
            return jnp.exp(-((x - 0.5) ** 2) / 0.01)

        eqn = AdvectionDiffusion1D(grid=grid, bc=bc, alpha=0.01, velocity=velocity, source=source)
        assert eqn.source is not None

    def test_2d_creation(self):
        from diffheat.physics.advection_diffusion import AdvectionDiffusion2D
        from diffheat.mesh import Grid2D, BoundaryCondition2D
        import jax.numpy as jnp

        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=40, ny=40)
        bc = BoundaryCondition2D(
            left={"kind": "dirichlet", "value": 1.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "neumann", "value": 0.0},
            top={"kind": "neumann", "value": 0.0},
        )

        def velocity(X, Y, t):
            return jnp.ones_like(X), jnp.zeros_like(Y)

        eqn = AdvectionDiffusion2D(grid=grid, bc=bc, alpha=0.01, velocity=velocity)
        assert eqn.alpha == 0.01

    def test_3d_creation(self):
        from diffheat.physics.advection_diffusion import AdvectionDiffusion3D
        from diffheat.mesh import Grid3D, BoundaryCondition3D
        import jax.numpy as jnp

        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=20, ny=20, nz=20)
        bc = BoundaryCondition3D(
            left={"kind": "dirichlet", "value": 1.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "neumann", "value": 0.0},
            top={"kind": "neumann", "value": 0.0},
            front={"kind": "neumann", "value": 0.0},
            back={"kind": "neumann", "value": 0.0},
        )

        def velocity(X, Y, Z, t):
            return jnp.ones_like(X), jnp.zeros_like(Y), jnp.zeros_like(Z)

        eqn = AdvectionDiffusion3D(grid=grid, bc=bc, alpha=0.01, velocity=velocity)
        assert eqn.alpha == 0.01


class TestSolveAdvectionDiffusion1D:
    @pytest.fixture
    def grid(self):
        from diffheat.mesh import Grid1D
        return Grid1D.uniform(length=10.0, n_cells=200)

    def test_pure_diffusion_matches_heat_solver(self, grid):
        """With u=0, the advection-diffusion solver should match the heat solver."""
        import jax.numpy as jnp
        from diffheat import HeatEquation1D, solve_heat_1d, BoundaryCondition
        from diffheat.physics.advection_diffusion import AdvectionDiffusion1D
        from diffheat.solvers.advection_diffusion import solve_advection_diffusion_1d

        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([100.0, 0.0]))
        alpha = 0.01
        T0 = jnp.zeros(grid.n_cells)
        dt = 0.005
        t_span = (0.0, 0.5)

        def zero_velocity(x, t):
            return jnp.zeros_like(x)

        # Solve with advection-diffusion solver (u=0)
        eqn_adv = AdvectionDiffusion1D(grid=grid, bc=bc, alpha=alpha, velocity=zero_velocity)
        traj_adv = solve_advection_diffusion_1d(eqn_adv, T0, t_span, dt)

        # Solve with heat solver
        eqn_heat = HeatEquation1D(grid=grid, bc=bc, alpha=alpha)
        traj_heat = solve_heat_1d(eqn_heat, T0, t_span, dt)

        assert traj_adv.shape == traj_heat.shape
        assert jnp.allclose(traj_adv, traj_heat, atol=1e-6)

    def test_is_jax_differentiable(self, grid):
        """jax.grad should work through the full solve."""
        import jax.numpy as jnp
        from diffheat import BoundaryCondition
        from diffheat.physics.advection_diffusion import AdvectionDiffusion1D
        from diffheat.solvers.advection_diffusion import solve_advection_diffusion_1d
        import jax

        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([100.0, 0.0]))
        T0 = jnp.zeros(grid.n_cells)
        dt = 0.005
        t_span = (0.0, 0.1)

        def velocity(x, t):
            return jnp.ones_like(x)

        def final_mean_temp(alpha):
            eqn = AdvectionDiffusion1D(grid=grid, bc=bc, alpha=alpha, velocity=velocity)
            traj = solve_advection_diffusion_1d(eqn, T0, t_span, dt)
            return jnp.mean(traj[-1])

        grad_fn = jax.grad(final_mean_temp)
        sensitivity = grad_fn(0.01)
        assert sensitivity != 0.0


class TestSolveAdvectionDiffusion2D:
    @pytest.fixture
    def grid(self):
        from diffheat.mesh import Grid2D
        return Grid2D.uniform(Lx=2.0, Ly=1.0, nx=40, ny=20)

    def test_pure_diffusion_matches_heat_solver(self, grid):
        """With u=0 everywhere, result matches HeatEquation2D."""
        import jax.numpy as jnp
        from diffheat import HeatEquation2D, solve_heat_2d, BoundaryCondition2D
        from diffheat.physics.advection_diffusion import AdvectionDiffusion2D
        from diffheat.solvers.advection_diffusion import solve_advection_diffusion_2d

        bc = BoundaryCondition2D(
            left={"kind": "dirichlet", "value": 100.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "neumann", "value": 0.0},
            top={"kind": "neumann", "value": 0.0},
        )
        alpha = 0.01
        T0 = jnp.zeros((grid.nx, grid.ny))
        dt = 0.001
        t_span = (0.0, 0.1)

        def zero_velocity(X, Y, t):
            return jnp.zeros_like(X), jnp.zeros_like(Y)

        eqn_adv = AdvectionDiffusion2D(grid=grid, bc=bc, alpha=alpha, velocity=zero_velocity)
        traj_adv = solve_advection_diffusion_2d(eqn_adv, T0, t_span, dt)

        eqn_heat = HeatEquation2D(grid=grid, bc=bc, alpha=alpha)
        traj_heat = solve_heat_2d(eqn_heat, T0, t_span, dt)

        assert traj_adv.shape == traj_heat.shape
        assert jnp.allclose(traj_adv, traj_heat, atol=1e-6)

    def test_advection_bends_temperature_field(self, grid):
        """With horizontal flow, a hot spot should shift downstream over time."""
        import jax.numpy as jnp
        from diffheat import BoundaryCondition2D
        from diffheat.physics.advection_diffusion import AdvectionDiffusion2D
        from diffheat.solvers.advection_diffusion import solve_advection_diffusion_2d

        bc = BoundaryCondition2D(
            left={"kind": "dirichlet", "value": 0.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "neumann", "value": 0.0},
            top={"kind": "neumann", "value": 0.0},
        )
        alpha = 0.001  # small diffusion
        T0 = jnp.zeros((grid.nx, grid.ny))
        # Hot spot near left edge
        T0 = T0.at[5, grid.ny // 2].set(100.0)
        dt = 0.001
        t_span = (0.0, 0.2)

        def rightward_flow(X, Y, t):
            return 2.0 * jnp.ones_like(X), jnp.zeros_like(Y)

        eqn = AdvectionDiffusion2D(grid=grid, bc=bc, alpha=alpha, velocity=rightward_flow)
        traj = solve_advection_diffusion_2d(eqn, T0, t_span, dt)

        # Peak temperature at final time should be downstream of initial position
        final_T = traj[-1]
        peak_x_initial = 5
        peak_x_final = jnp.argmax(jnp.max(final_T, axis=1))
        assert peak_x_final > peak_x_initial, f"Expected peak to move right, but peak at x={peak_x_final}"

    def test_is_jax_differentiable(self, grid):
        import jax.numpy as jnp
        from diffheat import BoundaryCondition2D
        from diffheat.physics.advection_diffusion import AdvectionDiffusion2D
        from diffheat.solvers.advection_diffusion import solve_advection_diffusion_2d
        import jax

        bc = BoundaryCondition2D(
            left={"kind": "dirichlet", "value": 1.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "neumann", "value": 0.0},
            top={"kind": "neumann", "value": 0.0},
        )
        T0 = jnp.zeros((grid.nx, grid.ny))

        def velocity(X, Y, t):
            return jnp.ones_like(X), jnp.zeros_like(Y)

        def final_mean_temp(alpha):
            eqn = AdvectionDiffusion2D(grid=grid, bc=bc, alpha=alpha, velocity=velocity)
            traj = solve_advection_diffusion_2d(eqn, T0, (0.0, 0.05), 0.001)
            return jnp.mean(traj[-1])

        grad_fn = jax.grad(final_mean_temp)
        sensitivity = grad_fn(0.01)
        assert sensitivity != 0.0


class TestSolveAdvectionDiffusion3D:
    @pytest.fixture
    def grid(self):
        from diffheat.mesh import Grid3D
        return Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=20, ny=20, nz=20)

    def test_pure_diffusion_matches_heat_solver(self, grid):
        """With u=0 everywhere, result matches HeatEquation3D."""
        import jax.numpy as jnp
        from diffheat import HeatEquation3D, solve_heat_3d, BoundaryCondition3D
        from diffheat.physics.advection_diffusion import AdvectionDiffusion3D
        from diffheat.solvers.advection_diffusion import solve_advection_diffusion_3d

        bc = BoundaryCondition3D(
            left={"kind": "dirichlet", "value": 100.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "neumann", "value": 0.0},
            top={"kind": "neumann", "value": 0.0},
            front={"kind": "neumann", "value": 0.0},
            back={"kind": "neumann", "value": 0.0},
        )
        alpha = 0.01
        T0 = jnp.zeros((grid.nx, grid.ny, grid.nz))
        dt = 0.0005
        t_span = (0.0, 0.01)

        def zero_velocity(X, Y, Z, t):
            return jnp.zeros_like(X), jnp.zeros_like(Y), jnp.zeros_like(Z)

        eqn_adv = AdvectionDiffusion3D(grid=grid, bc=bc, alpha=alpha, velocity=zero_velocity)
        traj_adv = solve_advection_diffusion_3d(eqn_adv, T0, t_span, dt)

        eqn_heat = HeatEquation3D(grid=grid, bc=bc, alpha=alpha)
        traj_heat = solve_heat_3d(eqn_heat, T0, t_span, dt)

        assert traj_adv.shape == traj_heat.shape
        assert jnp.allclose(traj_adv, traj_heat, atol=1e-5)

    def test_is_jax_differentiable(self, grid):
        import jax.numpy as jnp
        from diffheat import BoundaryCondition3D
        from diffheat.physics.advection_diffusion import AdvectionDiffusion3D
        from diffheat.solvers.advection_diffusion import solve_advection_diffusion_3d
        import jax

        bc = BoundaryCondition3D(
            left={"kind": "dirichlet", "value": 1.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "neumann", "value": 0.0},
            top={"kind": "neumann", "value": 0.0},
            front={"kind": "neumann", "value": 0.0},
            back={"kind": "neumann", "value": 0.0},
        )
        T0 = jnp.zeros((grid.nx, grid.ny, grid.nz))

        def velocity(X, Y, Z, t):
            return jnp.ones_like(X), jnp.zeros_like(Y), jnp.zeros_like(Z)

        def final_mean_temp(alpha):
            eqn = AdvectionDiffusion3D(grid=grid, bc=bc, alpha=alpha, velocity=velocity)
            traj = solve_advection_diffusion_3d(eqn, T0, (0.0, 0.005), 0.0005)
            return jnp.mean(traj[-1])

        grad_fn = jax.grad(final_mean_temp)
        sensitivity = grad_fn(0.01)
        assert sensitivity != 0.0

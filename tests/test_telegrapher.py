# tests/test_telegrapher.py
"""Tests for Telegrapher (hyperbolic heat) equation solvers."""
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
    TelegrapherEquation1D,
    TelegrapherEquation2D,
    TelegrapherEquation3D,
    check_cfl_telegrapher_1d,
    check_cfl_telegrapher_2d,
    check_cfl_telegrapher_3d,
    solve_telegrapher_1d,
    solve_telegrapher_2d,
    solve_telegrapher_3d,
)


def _bc1d():
    return BoundaryCondition(kind="dirichlet", value=jnp.array([0.0, 0.0]))


def _bc2d():
    return BoundaryCondition2D(
        left={"kind": "dirichlet", "value": 0.0},
        right={"kind": "dirichlet", "value": 0.0},
        bottom={"kind": "dirichlet", "value": 0.0},
        top={"kind": "dirichlet", "value": 0.0},
    )


def _bc3d():
    return BoundaryCondition3D(
        left={"kind": "dirichlet", "value": 0.0},
        right={"kind": "dirichlet", "value": 0.0},
        bottom={"kind": "dirichlet", "value": 0.0},
        top={"kind": "dirichlet", "value": 0.0},
        front={"kind": "dirichlet", "value": 0.0},
        back={"kind": "dirichlet", "value": 0.0},
    )


# ---------------------------------------------------------------------------
# CFL condition tests
# ---------------------------------------------------------------------------


class TestCFLTelegrapher1D:
    def test_stable(self):
        grid = Grid1D.uniform(length=1.0, n_cells=50)
        assert check_cfl_telegrapher_1d(grid, alpha=1.0, tau=1.0, dt=0.001)

    def test_unstable(self):
        grid = Grid1D.uniform(length=1.0, n_cells=50)
        # c = sqrt(1/1) = 1, dx = 0.02, limit = dx/c = 0.02
        assert not check_cfl_telegrapher_1d(grid, alpha=1.0, tau=1.0, dt=0.1)

    def test_large_tau_loosens_cfl(self):
        """Larger tau → smaller wave speed → looser CFL."""
        grid = Grid1D.uniform(length=1.0, n_cells=50)
        # c = sqrt(1/100) = 0.1, dx = 0.02, limit = 0.02/0.1 = 0.2
        assert check_cfl_telegrapher_1d(grid, alpha=1.0, tau=100.0, dt=0.01)


class TestCFLTelegrapher2D:
    def test_stable(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=30, ny=30)
        assert check_cfl_telegrapher_2d(grid, alpha=1.0, tau=1.0, dt=0.005)

    def test_unstable(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=30, ny=30)
        # c=1, dx~0.033, limit=dx/(sqrt2*c)~0.023
        assert not check_cfl_telegrapher_2d(grid, alpha=1.0, tau=1.0, dt=0.1)


class TestCFLTelegrapher3D:
    def test_stable(self):
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=20, ny=20, nz=20)
        assert check_cfl_telegrapher_3d(grid, alpha=1.0, tau=1.0, dt=0.005)

    def test_unstable(self):
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=20, ny=20, nz=20)
        # c=1, dx=0.05, limit=dx/(sqrt3*c)~0.029
        assert not check_cfl_telegrapher_3d(grid, alpha=1.0, tau=1.0, dt=0.1)


# ---------------------------------------------------------------------------
# 1D solver tests
# ---------------------------------------------------------------------------


class TestTelegrapher1D:
    def test_shape(self):
        """Output trajectory should have correct shape."""
        grid = Grid1D.uniform(length=1.0, n_cells=40)
        eqn = TelegrapherEquation1D(grid=grid, bc=_bc1d(), alpha=1.0, tau=1.0)

        u0 = jnp.sin(jnp.pi * grid.centers)
        v0 = jnp.zeros_like(u0)
        traj = solve_telegrapher_1d(eqn, u0, v0, t_span=(0.0, 0.1), dt=0.001)

        n_steps = int(0.1 / 0.001) + 1
        assert traj.shape == (n_steps, 40)

    def test_no_nans(self):
        """Solution should not contain NaNs."""
        grid = Grid1D.uniform(length=1.0, n_cells=40)
        eqn = TelegrapherEquation1D(grid=grid, bc=_bc1d(), alpha=1.0, tau=1.0)

        u0 = jnp.sin(jnp.pi * grid.centers)
        v0 = jnp.zeros_like(u0)
        traj = solve_telegrapher_1d(eqn, u0, v0, t_span=(0.0, 0.1), dt=0.001)
        assert not jnp.any(jnp.isnan(traj))

    def test_zero_initial_velocity(self):
        """With v0=0 and no source, solution should start from rest and decay."""
        grid = Grid1D.uniform(length=1.0, n_cells=40)
        eqn = TelegrapherEquation1D(grid=grid, bc=_bc1d(), alpha=1.0, tau=0.1)

        u0 = jnp.sin(jnp.pi * grid.centers)
        v0 = jnp.zeros_like(u0)
        traj = solve_telegrapher_1d(eqn, u0, v0, t_span=(0.0, 0.1), dt=0.001)

        # With damping, amplitude should decrease
        max0 = float(jnp.max(jnp.abs(traj[0])))
        max_end = float(jnp.max(jnp.abs(traj[-1])))
        assert max_end < max0 * 1.01  # allow small numerical noise

    def test_large_tau_wave_like(self):
        """For large tau, the solution should exhibit wave-like oscillations."""
        grid = Grid1D.uniform(length=1.0, n_cells=60)
        # Large tau → small damping, wave-like behavior
        eqn = TelegrapherEquation1D(grid=grid, bc=_bc1d(), alpha=1.0, tau=10.0)

        u0 = jnp.sin(jnp.pi * grid.centers)
        v0 = jnp.zeros_like(u0)
        traj = solve_telegrapher_1d(eqn, u0, v0, t_span=(0.0, 1.0), dt=0.001)

        # Should still be well-behaved (not blow up)
        assert float(jnp.max(jnp.abs(traj))) < 10.0

    def test_differentiable_wrt_alpha(self):
        """Gradient w.r.t. alpha should be computable."""
        grid = Grid1D.uniform(length=1.0, n_cells=30)

        def loss(alpha):
            eqn = TelegrapherEquation1D(grid=grid, bc=_bc1d(), alpha=alpha, tau=1.0)
            u0 = jnp.sin(jnp.pi * grid.centers)
            v0 = jnp.zeros_like(u0)
            traj = solve_telegrapher_1d(eqn, u0, v0, t_span=(0.0, 0.02), dt=0.001)
            return jnp.sum(traj[-1])

        grad = jax.grad(loss)(1.0)
        assert jnp.isfinite(grad)

    def test_differentiable_wrt_tau(self):
        """Gradient w.r.t. tau should be computable."""
        grid = Grid1D.uniform(length=1.0, n_cells=30)

        def loss(tau):
            eqn = TelegrapherEquation1D(grid=grid, bc=_bc1d(), alpha=1.0, tau=tau)
            u0 = jnp.sin(jnp.pi * grid.centers)
            v0 = jnp.zeros_like(u0)
            traj = solve_telegrapher_1d(eqn, u0, v0, t_span=(0.0, 0.02), dt=0.001)
            return jnp.sum(traj[-1])

        grad = jax.grad(loss)(1.0)
        assert jnp.isfinite(grad)

    def test_boundary_dirichlet_zero(self):
        """With zero Dirichlet BCs, boundaries should stay near zero."""
        grid = Grid1D.uniform(length=1.0, n_cells=40)
        eqn = TelegrapherEquation1D(grid=grid, bc=_bc1d(), alpha=1.0, tau=1.0)

        u0 = jnp.sin(jnp.pi * grid.centers)
        v0 = jnp.zeros_like(u0)
        traj = solve_telegrapher_1d(eqn, u0, v0, t_span=(0.0, 0.05), dt=0.001)

        # Boundary cells should be close to zero (ghost-cell correction)
        assert float(jnp.max(jnp.abs(traj[:, 0]))) < 0.2
        assert float(jnp.max(jnp.abs(traj[:, -1]))) < 0.2

    def test_small_tau_approaches_heat(self):
        """For very small tau, the behaviour should approach heat diffusion."""
        grid = Grid1D.uniform(length=1.0, n_cells=50)
        # Small tau → strong damping, heat-like
        eqn = TelegrapherEquation1D(grid=grid, bc=_bc1d(), alpha=1.0, tau=0.001)

        u0 = jnp.sin(jnp.pi * grid.centers)
        v0 = jnp.zeros_like(u0)
        traj = solve_telegrapher_1d(eqn, u0, v0, t_span=(0.0, 0.1), dt=0.0001)

        # Monotonic decay of max amplitude (diffusive, no oscillations)
        max_vals = [float(jnp.max(jnp.abs(traj[i]))) for i in range(traj.shape[0])]
        for i in range(len(max_vals) - 1):
            assert max_vals[i] >= max_vals[i + 1] - 1e-10


# ---------------------------------------------------------------------------
# 2D solver tests
# ---------------------------------------------------------------------------


class TestTelegrapher2D:
    def test_shape(self):
        """Output trajectory should have correct shape."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=30, ny=30)
        eqn = TelegrapherEquation2D(grid=grid, bc=_bc2d(), alpha=1.0, tau=1.0)

        u0 = jnp.sin(jnp.pi * grid.X.T) * jnp.sin(jnp.pi * grid.Y.T)
        v0 = jnp.zeros_like(u0)
        traj = solve_telegrapher_2d(eqn, u0, v0, t_span=(0.0, 0.02), dt=0.001)

        n_steps = int(0.02 / 0.001) + 1
        assert traj.shape == (n_steps, 30, 30)

    def test_no_nans(self):
        """Solution should not contain NaNs."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=30, ny=30)
        eqn = TelegrapherEquation2D(grid=grid, bc=_bc2d(), alpha=1.0, tau=1.0)

        u0 = jnp.sin(jnp.pi * grid.X.T) * jnp.sin(jnp.pi * grid.Y.T)
        v0 = jnp.zeros_like(u0)
        traj = solve_telegrapher_2d(eqn, u0, v0, t_span=(0.0, 0.02), dt=0.001)
        assert not jnp.any(jnp.isnan(traj))

    def test_differentiable_wrt_alpha(self):
        """Gradient w.r.t. alpha should be computable."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=20, ny=20)

        def loss(alpha):
            eqn = TelegrapherEquation2D(grid=grid, bc=_bc2d(), alpha=alpha, tau=1.0)
            u0 = jnp.sin(jnp.pi * grid.X.T) * jnp.sin(jnp.pi * grid.Y.T)
            v0 = jnp.zeros_like(u0)
            traj = solve_telegrapher_2d(eqn, u0, v0, t_span=(0.0, 0.01), dt=0.001)
            return jnp.sum(traj[-1])

        grad = jax.grad(loss)(1.0)
        assert jnp.isfinite(grad)

    def test_large_tau_propagates(self):
        """For large tau, a localised pulse should spread as a wavefront."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=40, ny=40)
        # Large tau → underdamped wave propagation
        eqn = TelegrapherEquation2D(grid=grid, bc=_bc2d(), alpha=1.0, tau=10.0)

        # Gaussian pulse at centre
        X = grid.X.T
        Y = grid.Y.T
        u0 = jnp.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.01)
        v0 = jnp.zeros_like(u0)
        traj = solve_telegrapher_2d(eqn, u0, v0, t_span=(0.0, 0.05), dt=0.0005)

        # Peak should spread (max decrease, non-peak regions increase)
        max0 = float(jnp.max(traj[0]))
        max_end = float(jnp.max(traj[-1]))
        assert max_end < max0

    def test_small_tau_heat_like(self):
        """Small tau should approach diffusive (heat-like) behaviour."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=25, ny=25)
        eqn = TelegrapherEquation2D(grid=grid, bc=_bc2d(), alpha=1.0, tau=0.001)

        u0 = jnp.sin(jnp.pi * grid.X.T) * jnp.sin(jnp.pi * grid.Y.T)
        v0 = jnp.zeros_like(u0)
        traj = solve_telegrapher_2d(eqn, u0, v0, t_span=(0.0, 0.02), dt=0.0001)

        # Monotonic decay
        max_vals = [float(jnp.max(jnp.abs(traj[i]))) for i in range(traj.shape[0])]
        for i in range(len(max_vals) - 1):
            assert max_vals[i] >= max_vals[i + 1] - 1e-10


# ---------------------------------------------------------------------------
# 3D solver tests
# ---------------------------------------------------------------------------


class TestTelegrapher3D:
    def test_shape(self):
        """Output trajectory should have correct shape."""
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=16, ny=16, nz=16)
        eqn = TelegrapherEquation3D(grid=grid, bc=_bc3d(), alpha=1.0, tau=1.0)

        u0 = (
            jnp.sin(jnp.pi * grid.X)
            * jnp.sin(jnp.pi * grid.Y)
            * jnp.sin(jnp.pi * grid.Z)
        )
        v0 = jnp.zeros_like(u0)
        traj = solve_telegrapher_3d(eqn, u0, v0, t_span=(0.0, 0.01), dt=0.001)

        n_steps = int(0.01 / 0.001) + 1
        assert traj.shape == (n_steps, 16, 16, 16)

    def test_no_nans(self):
        """Solution should not contain NaNs."""
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=16, ny=16, nz=16)
        eqn = TelegrapherEquation3D(grid=grid, bc=_bc3d(), alpha=1.0, tau=1.0)

        u0 = (
            jnp.sin(jnp.pi * grid.X)
            * jnp.sin(jnp.pi * grid.Y)
            * jnp.sin(jnp.pi * grid.Z)
        )
        v0 = jnp.zeros_like(u0)
        traj = solve_telegrapher_3d(eqn, u0, v0, t_span=(0.0, 0.01), dt=0.001)
        assert not jnp.any(jnp.isnan(traj))

    def test_save_every(self):
        """save_every should reduce output frame count."""
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=12, ny=12, nz=12)
        eqn = TelegrapherEquation3D(grid=grid, bc=_bc3d(), alpha=1.0, tau=1.0)

        u0 = jnp.ones((12, 12, 12)) * 0.5
        v0 = jnp.zeros_like(u0)

        traj_all = solve_telegrapher_3d(
            eqn, u0, v0, t_span=(0.0, 0.01), dt=0.001, save_every=1
        )
        traj_skip = solve_telegrapher_3d(
            eqn, u0, v0, t_span=(0.0, 0.01), dt=0.001, save_every=2
        )

        # save_every=2 should have roughly half the frames (plus initial)
        assert traj_skip.shape[0] < traj_all.shape[0]

    def test_differentiable_wrt_alpha(self):
        """Gradient w.r.t. alpha should be computable."""
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=10, ny=10, nz=10)

        def loss(alpha):
            eqn = TelegrapherEquation3D(grid=grid, bc=_bc3d(), alpha=alpha, tau=1.0)
            u0 = (
                jnp.sin(jnp.pi * grid.X)
                * jnp.sin(jnp.pi * grid.Y)
                * jnp.sin(jnp.pi * grid.Z)
            )
            v0 = jnp.zeros_like(u0)
            traj = solve_telegrapher_3d(
                eqn, u0, v0, t_span=(0.0, 0.005), dt=0.001
            )
            return jnp.sum(traj[-1])

        grad = jax.grad(loss)(1.0)
        assert jnp.isfinite(grad)


# ---------------------------------------------------------------------------
# Physics definition validation
# ---------------------------------------------------------------------------


class TestTelegrapherPhysics:
    def test_negative_alpha_raises(self):
        grid = Grid1D.uniform(length=1.0, n_cells=10)
        with pytest.raises(ValueError):
            TelegrapherEquation1D(grid=grid, bc=_bc1d(), alpha=-1.0, tau=1.0)

    def test_negative_tau_raises(self):
        grid = Grid1D.uniform(length=1.0, n_cells=10)
        with pytest.raises(ValueError):
            TelegrapherEquation1D(grid=grid, bc=_bc1d(), alpha=1.0, tau=-0.5)

    def test_zero_tau_raises(self):
        """tau must be strictly positive (zero gives division by zero)."""
        grid = Grid1D.uniform(length=1.0, n_cells=10)
        with pytest.raises(ValueError):
            TelegrapherEquation1D(grid=grid, bc=_bc1d(), alpha=1.0, tau=0.0)

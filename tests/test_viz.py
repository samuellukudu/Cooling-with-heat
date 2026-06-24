"""Smoke tests for visualization module — no GUI rendering in CI."""
import numpy as np
import pytest
import jax.numpy as jnp
from diffheat.mesh import Grid1D, BoundaryCondition
from diffheat.physics import HeatEquation1D
from diffheat.solvers import solve_heat_1d


class TestVizImport:
    def test_can_import_viz_module(self):
        """viz module should be importable even without display."""
        from diffheat import viz
        assert hasattr(viz, "run_viewer")

    def test_run_viewer_exists(self):
        """run_viewer function should be callable (smoke test without display)."""
        from diffheat.viz import run_viewer
        assert callable(run_viewer)


class TestVizDataFlow:
    def test_trajectory_to_numpy_conversion(self):
        """Trajectory data should be convertible to numpy for Qt consumption."""
        grid = Grid1D.uniform(length=1.0, n_cells=10)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))
        eqn = HeatEquation1D(grid=grid, bc=bc, alpha=0.1)
        T0 = jnp.zeros(grid.n_cells)
        trajectory = solve_heat_1d(eqn, T0, (0.0, 0.01), dt=0.001)

        # Convert to numpy (what viz does internally)
        traj_np = np.asarray(jnp.asarray(trajectory))
        assert traj_np.shape == trajectory.shape
        assert not isinstance(traj_np, type(trajectory))  # should be plain ndarray

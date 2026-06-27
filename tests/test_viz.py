"""Smoke tests for visualization module — no GUI rendering in CI."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import jax.numpy as jnp
from PyQt6 import QtWidgets
from diffheat.mesh import Grid1D, BoundaryCondition, Grid2D
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


class TestViz2DImports:
    def test_can_import_heatmap2d(self):
        """Heatmap2DWidget should be importable."""
        from diffheat.viz.heatmap2d import Heatmap2DWidget
        assert hasattr(Heatmap2DWidget, "set_data")

    def test_can_import_viewer_window_2d(self):
        """ViewerWindow2D should be importable from viz module."""
        from diffheat.viz import ViewerWindow2D
        assert hasattr(ViewerWindow2D, "set_data")

    def test_run_viewer_2d_exists(self):
        """run_viewer_2d function should be callable."""
        from diffheat.viz import run_viewer_2d
        assert callable(run_viewer_2d)


class TestViz2DDataFlow:
    def test_trajectory_to_numpy_conversion_2d(self):
        """2D trajectory data should be convertible to numpy for Qt consumption."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=8, ny=6)
        trajectory = jnp.zeros((11, grid.nx, grid.ny))

        traj_np = np.asarray(jnp.asarray(trajectory))
        assert traj_np.shape == (11, grid.nx, grid.ny)
        assert not isinstance(traj_np, type(trajectory))

    def test_heatmap2d_widget_init(self):
        """Heatmap2DWidget can be instantiated and set_data called (no display)."""
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        from diffheat.viz.heatmap2d import Heatmap2DWidget

        widget = Heatmap2DWidget()
        assert widget.trajectory is None

        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=8, ny=6)
        trajectory = jnp.zeros((5, grid.nx, grid.ny))
        widget.set_data(trajectory, grid, dt=0.01)

        assert widget.trajectory is not None
        assert widget.trajectory.shape == (5, grid.nx, grid.ny)
        assert widget.times is not None
        assert len(widget.times) == 5
        assert widget.current_frame == 0

    def test_heatmap2d_set_frame(self):
        """set_frame should clamp to valid range without crashing."""
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        from diffheat.viz.heatmap2d import Heatmap2DWidget

        widget = Heatmap2DWidget()
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=8, ny=6)
        trajectory = jnp.zeros((5, grid.nx, grid.ny))
        widget.set_data(trajectory, grid, dt=0.01)

        widget.set_frame(2)
        assert widget.current_frame == 2

        widget.set_frame(-5)
        assert widget.current_frame == 0

        widget.set_frame(100)
        assert widget.current_frame == 4

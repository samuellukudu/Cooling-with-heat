# diffheat/viz/window.py
"""Main viewer window composing widgets and launching the Qt event loop."""
import sys

import jax.numpy as jnp
from PyQt6 import QtWidgets

from ..mesh.grid1d import Grid1D
from ..mesh.grid2d import Grid2D
from .controls import ControlPanel
from .heatmap1d import HeatmapWidget
from .heatmap2d import Heatmap2DWidget


class ViewerWindow(QtWidgets.QMainWindow):
    """Main window composing the heatmap, snapshot, and controls."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("diffheat — 1D Heat Equation Viewer")
        self.resize(1000, 700)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        self.heatmap = HeatmapWidget()
        self.controls = ControlPanel()

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.heatmap, stretch=1)
        layout.addWidget(self.controls)
        central.setLayout(layout)

        self.controls.frame_changed.connect(self.heatmap.set_frame)

    def set_data(self, trajectory: jnp.ndarray, grid: Grid1D, dt: float):
        """Load simulation data and prepare the viewer."""
        self.heatmap.set_data(trajectory, grid, dt)
        self.controls.set_n_frames(len(trajectory))
        self.controls.set_frame(0)


def run_viewer(
    trajectory: jnp.ndarray,
    grid: Grid1D,
    dt: float = 0.001,
) -> None:
    """Launch the PyQt6 viewer for a heat equation trajectory.

    Blocks until the user closes the window.

    Args:
        trajectory: (n_steps+1, N) temperature array from solve_heat_1d.
        grid: The Grid1D used for the simulation.
        dt: Time step size (for time axis labeling).
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = ViewerWindow()
    window.set_data(trajectory, grid, dt)
    window.show()
    app.exec()


class ViewerWindow2D(QtWidgets.QMainWindow):
    """Main window for 2D heat equation visualization."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("diffheat — 2D Heat Equation Viewer")
        self.resize(800, 800)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        self.heatmap = Heatmap2DWidget()
        self.controls = ControlPanel()

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.heatmap, stretch=1)
        layout.addWidget(self.controls)
        central.setLayout(layout)

        self.controls.frame_changed.connect(self.heatmap.set_frame)

    def set_data(self, trajectory: jnp.ndarray, grid: Grid2D, dt: float):
        """Load simulation data and prepare the viewer."""
        self.heatmap.set_data(trajectory, grid, dt)
        self.controls.set_n_frames(len(trajectory))
        self.controls.set_frame(0)


def run_viewer_2d(
    trajectory: jnp.ndarray,
    grid: Grid2D,
    dt: float = 0.001,
) -> None:
    """Launch the PyQt6 viewer for a 2D heat equation trajectory.

    Blocks until the user closes the window.

    Args:
        trajectory: (n_steps+1, nx, ny) temperature array from solve_2d.
        grid: The Grid2D used for the simulation.
        dt: Time step size (for time axis labeling).
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = ViewerWindow2D()
    window.set_data(trajectory, grid, dt)
    window.show()
    app.exec()

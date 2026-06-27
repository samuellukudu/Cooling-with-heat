"""2D heatmap widget for visualizing 2D temperature fields."""
import numpy as np
from PyQt6 import QtWidgets
import jax.numpy as jnp

from ..mesh.grid2d import Grid2D
from .canvas import MatplotlibCanvas


class Heatmap2DWidget(QtWidgets.QWidget):
    """2D heatmap imshow of the temperature field."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = MatplotlibCanvas(self, width=8, height=6)
        self.ax_main = self.canvas.fig.add_subplot(1, 1, 1)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.trajectory = None
        self.grid = None
        self.times = None
        self.current_frame = 0
        self._colorbar = None

    def set_data(self, trajectory: jnp.ndarray, grid: Grid2D, dt: float):
        """Load trajectory data for display.

        Args:
            trajectory: (n_steps+1, nx, ny) temperature array.
            grid: The Grid2D used for the simulation.
            dt: Time step size.
        """
        self.trajectory = np.asarray(jnp.asarray(trajectory))
        self.grid = grid
        self.times = np.arange(len(trajectory)) * dt
        self.current_frame = 0
        self._colorbar = None
        self._draw()

    def set_frame(self, frame_idx: int):
        """Display a specific frame."""
        if self.trajectory is None:
            return
        self.current_frame = max(0, min(frame_idx, len(self.trajectory) - 1))
        self._draw()

    def _draw(self):
        if self.trajectory is None or self.grid is None:
            return

        self.ax_main.clear()

        # Temperature field at current frame
        T_frame = self.trajectory[self.current_frame]

        # Use grid extents for the imshow axes
        x_edges = np.asarray(self.grid.x)
        y_edges = np.asarray(self.grid.y)
        extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

        im = self.ax_main.imshow(
            T_frame,
            aspect="equal",
            extent=extent,
            cmap="hot",
            origin="lower",
            interpolation="bilinear",
        )

        current_time = self.times[self.current_frame]
        self.ax_main.set_xlabel("x (m)")
        self.ax_main.set_ylabel("y (m)")
        self.ax_main.set_title(f"Temperature Field at t = {current_time:.3f} s")

        if self._colorbar is None:
            self._colorbar = self.canvas.fig.colorbar(im, ax=self.ax_main, label="Temperature (°C)")
        else:
            self._colorbar.update_normal(im)

        self.canvas.fig.tight_layout()
        self.canvas.draw()

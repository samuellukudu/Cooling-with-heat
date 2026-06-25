# diffheat/viz/heatmap1d.py
"""1D space-time heatmap widget."""
import numpy as np
from PyQt6 import QtWidgets
import jax.numpy as jnp

from ..mesh.grid1d import Grid1D
from .canvas import MatplotlibCanvas


class HeatmapWidget(QtWidgets.QWidget):
    """Space-time heatmap of the temperature field."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = MatplotlibCanvas(self, width=8, height=5)
        self.ax_heatmap = self.canvas.fig.add_subplot(2, 1, 1)
        self.ax_snapshot = self.canvas.fig.add_subplot(2, 1, 2)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.trajectory = None
        self.grid = None
        self.times = None
        self.current_frame = 0
        self._colorbar = None

    def set_data(self, trajectory: jnp.ndarray, grid: Grid1D, dt: float):
        """Load trajectory data for display.

        Args:
            trajectory: (n_steps+1, N) temperature array.
            grid: The Grid1D used for the simulation.
            dt: Time step size.
        """
        # Copy to CPU numpy for Qt thread safety
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

        self.ax_heatmap.clear()
        self.ax_snapshot.clear()

        x_centers = np.asarray(self.grid.centers)
        x_edges = np.asarray(self.grid.x)

        # Space-time heatmap
        extent = [x_edges[0], x_edges[-1], self.times[-1], self.times[0]]
        im = self.ax_heatmap.imshow(
            self.trajectory,
            aspect="auto",
            extent=extent,
            cmap="hot",
            origin="upper",
        )
        self.ax_heatmap.set_ylabel("Time")
        self.ax_heatmap.set_xlabel("Position")
        self.ax_heatmap.set_title(
            f"Temperature Field (t = {self.times[self.current_frame]:.3f})"
        )
        if self._colorbar is None:
            self._colorbar = self.canvas.fig.colorbar(im, ax=self.ax_heatmap, label="Temperature")
        else:
            self._colorbar.update_normal(im)

        # Current frame indicator line
        current_time = self.times[self.current_frame]
        self.ax_heatmap.axhline(current_time, color="cyan", linewidth=1, alpha=0.8)

        # Snapshot: T(x) at current frame
        self.ax_snapshot.plot(
            x_centers,
            self.trajectory[self.current_frame],
            "b-",
            linewidth=2,
        )
        self.ax_snapshot.set_xlabel("Position")
        self.ax_snapshot.set_ylabel("Temperature")
        self.ax_snapshot.set_title(f"Snapshot at t = {current_time:.3f}")
        self.ax_snapshot.set_ylim(
            self.trajectory.min() - 0.1,
            self.trajectory.max() + 0.1,
        )
        self.ax_snapshot.grid(True, alpha=0.3)

        # Boundary markers
        ylim = self.ax_snapshot.get_ylim()
        self.ax_snapshot.plot(
            [x_edges[0], x_edges[0]], ylim, "r--", linewidth=1, alpha=0.5
        )
        self.ax_snapshot.plot(
            [x_edges[-1], x_edges[-1]], ylim, "b--", linewidth=1, alpha=0.5
        )

        self.canvas.fig.tight_layout()
        self.canvas.draw()

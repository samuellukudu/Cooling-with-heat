# diffheat/viz.py
"""PyQt6-based visualization for 1D heat equation trajectories.

This is the ONLY module in diffheat that imports PyQt or matplotlib.
The core library remains headless and works without these dependencies.
"""
import sys

import jax.numpy as jnp
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6 import QtCore, QtWidgets

from .mesh import Grid1D


class MatplotlibCanvas(FigureCanvasQTAgg):
    """Matplotlib figure embedded in a Qt widget."""

    def __init__(self, parent=None, width=8, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)


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

        n_steps = len(self.trajectory)
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
        self.canvas.fig.colorbar(im, ax=self.ax_heatmap, label="Temperature")

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


class ControlPanel(QtWidgets.QWidget):
    """Play/pause, frame navigation, and parameter display."""

    frame_changed = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.n_frames = 0
        self.current_frame = 0
        self._playing = False
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._tick)

        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QHBoxLayout()

        # Play/Pause
        self.btn_play = QtWidgets.QPushButton("▶ Play")
        self.btn_play.clicked.connect(self._toggle_play)
        layout.addWidget(self.btn_play)

        # Step back
        self.btn_back = QtWidgets.QPushButton("◀ Step")
        self.btn_back.clicked.connect(self._step_back)
        layout.addWidget(self.btn_back)

        # Step forward
        self.btn_fwd = QtWidgets.QPushButton("Step ▶")
        self.btn_fwd.clicked.connect(self._step_forward)
        layout.addWidget(self.btn_fwd)

        # Reset
        self.btn_reset = QtWidgets.QPushButton("⏮ Reset")
        self.btn_reset.clicked.connect(self._reset)
        layout.addWidget(self.btn_reset)

        # Frame slider
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self._on_slider)
        layout.addWidget(self.slider)

        # Frame counter
        self.label_frame = QtWidgets.QLabel("0 / 0")
        layout.addWidget(self.label_frame)

        self.setLayout(layout)

    def set_n_frames(self, n: int):
        self.n_frames = n
        self.slider.setRange(0, n - 1)
        self.set_frame(0)

    def set_frame(self, frame_idx: int):
        self.current_frame = max(0, min(frame_idx, self.n_frames - 1))
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_frame)
        self.slider.blockSignals(False)
        self.label_frame.setText(f"{self.current_frame} / {self.n_frames - 1}")
        self.frame_changed.emit(self.current_frame)

    def _toggle_play(self):
        self._playing = not self._playing
        if self._playing:
            self.btn_play.setText("⏸ Pause")
            self._timer.start(50)  # ~20 fps
        else:
            self.btn_play.setText("▶ Play")
            self._timer.stop()

    def _step_forward(self):
        self._playing = False
        self.btn_play.setText("▶ Play")
        self._timer.stop()
        self.set_frame(self.current_frame + 1)

    def _step_back(self):
        self._playing = False
        self.btn_play.setText("▶ Play")
        self._timer.stop()
        self.set_frame(self.current_frame - 1)

    def _reset(self):
        self._playing = False
        self.btn_play.setText("▶ Play")
        self._timer.stop()
        self.set_frame(0)

    def _tick(self):
        if self.current_frame < self.n_frames - 1:
            self.set_frame(self.current_frame + 1)
        else:
            self._playing = False
            self.btn_play.setText("▶ Play")
            self._timer.stop()

    def _on_slider(self, value: int):
        self.current_frame = value
        self.label_frame.setText(f"{self.current_frame} / {self.n_frames - 1}")
        self.frame_changed.emit(self.current_frame)


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
    app = QtWidgets.QApplication(sys.argv)
    window = ViewerWindow()
    window.set_data(trajectory, grid, dt)
    window.show()
    app.exec()

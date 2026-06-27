"""3D slice heatmap widget for visualizing 3D temperature fields."""
import numpy as np
from PyQt6 import QtWidgets, QtCore
import jax.numpy as jnp

from ..mesh.grid3d import Grid3D
from .canvas import MatplotlibCanvas


class Heatmap3DWidget(QtWidgets.QWidget):
    """3D slice viewer displaying orthogonal XY, XZ, and YZ slices."""

    slice_x_changed = QtCore.pyqtSignal(int)
    slice_y_changed = QtCore.pyqtSignal(int)
    slice_z_changed = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 3 subplots: [XY Slice, XZ Slice, YZ Slice]
        self.canvas = MatplotlibCanvas(self, width=12, height=4)
        self.ax_xy = self.canvas.fig.add_subplot(1, 3, 1)
        self.ax_xz = self.canvas.fig.add_subplot(1, 3, 2)
        self.ax_yz = self.canvas.fig.add_subplot(1, 3, 3)

        # Slice indices
        self.idx_x = 0
        self.idx_y = 0
        self.idx_z = 0

        self.trajectory = None
        self.grid = None
        self.times = None
        self.current_frame = 0
        self.vmin = 0.0
        self.vmax = 100.0

        # Subplot image references for colorbar updates
        self.im_xy = None
        self.im_xz = None
        self.im_yz = None
        self.cbar_xy = None
        self.cbar_xz = None
        self.cbar_yz = None

        # Build UI layout
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(self.canvas, stretch=1)

        # Add slice controls layout
        controls_layout = QtWidgets.QGridLayout()
        
        # X-slice
        self.lbl_x = QtWidgets.QLabel("X Slice Index: 0")
        self.sld_x = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sld_x.valueChanged.connect(self.set_slice_x)
        controls_layout.addWidget(self.lbl_x, 0, 0)
        controls_layout.addWidget(self.sld_x, 0, 1)

        # Y-slice
        self.lbl_y = QtWidgets.QLabel("Y Slice Index: 0")
        self.sld_y = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sld_y.valueChanged.connect(self.set_slice_y)
        controls_layout.addWidget(self.lbl_y, 1, 0)
        controls_layout.addWidget(self.sld_y, 1, 1)

        # Z-slice
        self.lbl_z = QtWidgets.QLabel("Z Slice Index: 0")
        self.sld_z = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sld_z.valueChanged.connect(self.set_slice_z)
        controls_layout.addWidget(self.lbl_z, 2, 0)
        controls_layout.addWidget(self.sld_z, 2, 1)

        main_layout.addLayout(controls_layout)
        self.setLayout(main_layout)

    def set_data(self, trajectory: jnp.ndarray, grid: Grid3D, dt: float):
        """Load trajectory data for display.

        Args:
            trajectory: (n_steps+1, nx, ny, nz) temperature array.
            grid: The Grid3D used for the simulation.
            dt: Time step size.
        """
        self.trajectory = np.asarray(jnp.asarray(trajectory))
        self.grid = grid
        self.times = np.arange(len(trajectory)) * dt
        self.current_frame = 0

        # Global color limits across the entire trajectory
        self.vmin = float(self.trajectory.min())
        self.vmax = float(self.trajectory.max())
        if np.isclose(self.vmin, self.vmax):
            self.vmin -= 1.0
            self.vmax += 1.0

        # Reset default slice indices to middle of the domain
        self.idx_x = grid.nx // 2
        self.idx_y = grid.ny // 2
        self.idx_z = grid.nz // 2

        # Configure sliders
        self.sld_x.blockSignals(True)
        self.sld_x.setRange(0, grid.nx - 1)
        self.sld_x.setValue(self.idx_x)
        self.sld_x.blockSignals(False)
        self.lbl_x.setText(f"YZ Slice (X = {grid.x_centers[self.idx_x]:.3f} m)")

        self.sld_y.blockSignals(True)
        self.sld_y.setRange(0, grid.ny - 1)
        self.sld_y.setValue(self.idx_y)
        self.sld_y.blockSignals(False)
        self.lbl_y.setText(f"XZ Slice (Y = {grid.y_centers[self.idx_y]:.3f} m)")

        self.sld_z.blockSignals(True)
        self.sld_z.setRange(0, grid.nz - 1)
        self.sld_z.setValue(self.idx_z)
        self.sld_z.blockSignals(False)
        self.lbl_z.setText(f"XY Slice (Z = {grid.z_centers[self.idx_z]:.3f} m)")

        self.cbar_xy = None
        self.cbar_xz = None
        self.cbar_yz = None
        self._draw()

    def set_frame(self, frame_idx: int):
        """Display a specific frame."""
        if self.trajectory is None:
            return
        self.current_frame = max(0, min(frame_idx, len(self.trajectory) - 1))
        self._draw()

    def set_slice_x(self, idx: int):
        """Set the X slice index."""
        if self.grid is None:
            return
        self.idx_x = max(0, min(idx, self.grid.nx - 1))
        self.lbl_x.setText(f"YZ Slice (X = {self.grid.x_centers[self.idx_x]:.3f} m)")
        self.slice_x_changed.emit(self.idx_x)
        self._draw()

    def set_slice_y(self, idx: int):
        """Set the Y slice index."""
        if self.grid is None:
            return
        self.idx_y = max(0, min(idx, self.grid.ny - 1))
        self.lbl_y.setText(f"XZ Slice (Y = {self.grid.y_centers[self.idx_y]:.3f} m)")
        self.slice_y_changed.emit(self.idx_y)
        self._draw()

    def set_slice_z(self, idx: int):
        """Set the Z slice index."""
        if self.grid is None:
            return
        self.idx_z = max(0, min(idx, self.grid.nz - 1))
        self.lbl_z.setText(f"XY Slice (Z = {self.grid.z_centers[self.idx_z]:.3f} m)")
        self.slice_z_changed.emit(self.idx_z)
        self._draw()

    def _draw(self):
        if self.trajectory is None or self.grid is None:
            return

        self.ax_xy.clear()
        self.ax_xz.clear()
        self.ax_yz.clear()

        T_frame = self.trajectory[self.current_frame]

        x_edges = np.asarray(self.grid.x)
        y_edges = np.asarray(self.grid.y)
        z_edges = np.asarray(self.grid.z)

        # 1. XY Slice (columns=x, rows=y) -> T[:, :, idx_z].T
        extent_xy = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
        self.im_xy = self.ax_xy.imshow(
            T_frame[:, :, self.idx_z].T,
            aspect="equal",
            extent=extent_xy,
            cmap="hot",
            origin="lower",
            vmin=self.vmin,
            vmax=self.vmax,
            interpolation="bilinear",
        )
        self.ax_xy.set_xlabel("x (m)")
        self.ax_xy.set_ylabel("y (m)")
        self.ax_xy.set_title(f"XY Slice (Z = {self.grid.z_centers[self.idx_z]:.3f} m)")

        # 2. XZ Slice (columns=x, rows=z) -> T[:, idx_y, :].T
        extent_xz = [x_edges[0], x_edges[-1], z_edges[0], z_edges[-1]]
        self.im_xz = self.ax_xz.imshow(
            T_frame[:, self.idx_y, :].T,
            aspect="equal",
            extent=extent_xz,
            cmap="hot",
            origin="lower",
            vmin=self.vmin,
            vmax=self.vmax,
            interpolation="bilinear",
        )
        self.ax_xz.set_xlabel("x (m)")
        self.ax_xz.set_ylabel("z (m)")
        self.ax_xz.set_title(f"XZ Slice (Y = {self.grid.y_centers[self.idx_y]:.3f} m)")

        # 3. YZ Slice (columns=y, rows=z) -> T[self.idx_x, :, :].T
        extent_yz = [y_edges[0], y_edges[-1], z_edges[0], z_edges[-1]]
        self.im_yz = self.ax_yz.imshow(
            T_frame[self.idx_x, :, :].T,
            aspect="equal",
            extent=extent_yz,
            cmap="hot",
            origin="lower",
            vmin=self.vmin,
            vmax=self.vmax,
            interpolation="bilinear",
        )
        self.ax_yz.set_xlabel("y (m)")
        self.ax_yz.set_ylabel("z (m)")
        self.ax_yz.set_title(f"YZ Slice (X = {self.grid.x_centers[self.idx_x]:.3f} m)")

        # Handle colorbars
        if self.cbar_xy is None:
            self.cbar_xy = self.canvas.fig.colorbar(self.im_xy, ax=self.ax_xy, label="T (°C)")
        else:
            self.cbar_xy.update_normal(self.im_xy)

        if self.cbar_xz is None:
            self.cbar_xz = self.canvas.fig.colorbar(self.im_xz, ax=self.ax_xz, label="T (°C)")
        else:
            self.cbar_xz.update_normal(self.im_xz)

        if self.cbar_yz is None:
            self.cbar_yz = self.canvas.fig.colorbar(self.im_yz, ax=self.ax_yz, label="T (°C)")
        else:
            self.cbar_yz.update_normal(self.im_yz)

        current_time = self.times[self.current_frame]
        self.canvas.fig.suptitle(f"3D Slices at t = {current_time:.3f} s", fontsize=14, y=0.98)
        self.canvas.fig.tight_layout()
        self.canvas.draw()

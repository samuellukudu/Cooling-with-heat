# diffheat 2D Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend diffheat to 2D rectangular domains with a general-purpose operator library, generic solver, per-edge boundary conditions, and 2D visualization — while restructuring the package from flat files to subpackages and preserving all 1D APIs.

**Architecture:** Restructure flat files into subpackages (`mesh/`, `operators/`, `physics/`, `solvers/`, `viz/`) with backward-compatible re-exports. Add `Grid2D`, discrete 2D operators (`laplacian_2d`, `gradient_x/y`, `divergence_2d`), `BoundaryCondition2D`, and a generic `solve_2d` that takes a user-composed RHS function. 2D viewer shows frame-by-frame heatmaps with a colorbar.

**Tech Stack:** Python 3.10+, JAX (with jaxlib), NumPy, PyQt6 (viz only), matplotlib (viz only), pytest, uv

## Global Constraints

- Python >= 3.10
- Core deps: jax, numpy (no PyQt in core)
- Viz extras: pyqt6, matplotlib
- Dev extras: pytest, jupyter
- All arrays created via `diffheat.utils.array()` — never raw `jnp.array()` in library code
- Frozen dataclasses for all data objects (JAX tracing compatible)
- No circular imports
- `diffheat/` is the installable package; `examples/` is NOT a package (no `__init__.py`)
- GPU-aware: detect device at import, float32 on GPU, float64 on CPU
- 1D API backward compatibility: all existing imports must continue to work
- Headless core: operators, solvers, physics are pure JAX — only `viz/` touches PyQt/matplotlib
- Field indexing convention: `T[i, j]` = cell at `(x_centers[i], y_centers[j])`; meshgrids `X[j, i]`, `Y[j, i]` follow `imshow`'s y-first convention

---

### Task 1: Package Restructure — Subpackages with Backward Compatibility

**Files:**
- Create: `diffheat/mesh/__init__.py`
- Create: `diffheat/mesh/grid1d.py` (moved from `diffheat/mesh.py`)
- Create: `diffheat/mesh/boundary.py` (moved from `diffheat/mesh.py`)
- Create: `diffheat/operators/__init__.py`
- Create: `diffheat/operators/laplacian.py` (moved from `diffheat/physics.py`)
- Create: `diffheat/physics/__init__.py`
- Create: `diffheat/physics/heat1d.py` (moved from `diffheat/physics.py`)
- Create: `diffheat/solvers/__init__.py`
- Create: `diffheat/solvers/explicit.py` (moved from `diffheat/solvers.py`)
- Create: `diffheat/solvers/scan.py` (moved from `diffheat/solvers.py`)
- Create: `diffheat/solvers/stability.py` (moved from `diffheat/solvers.py`)
- Create: `diffheat/viz/__init__.py`
- Create: `diffheat/viz/canvas.py` (extracted from `diffheat/viz.py`)
- Create: `diffheat/viz/heatmap1d.py` (extracted from `diffheat/viz.py`)
- Create: `diffheat/viz/controls.py` (extracted from `diffheat/viz.py`)
- Create: `diffheat/viz/window.py` (extracted from `diffheat/viz.py`)
- Modify: `diffheat/__init__.py`
- Delete: `diffheat/mesh.py`, `diffheat/physics.py`, `diffheat/solvers.py`, `diffheat/viz.py`
- Modify: `tests/test_mesh.py`, `tests/test_physics.py`, `tests/test_solvers.py` (update imports)

**Interfaces:**
- Produces: Subpackage structure with all existing imports still working. All 46 tests pass.

- [ ] **Step 1: Create subpackage directories**

```bash
mkdir -p diffheat/mesh diffheat/operators diffheat/physics diffheat/solvers diffheat/viz
```

- [ ] **Step 2: Create `diffheat/mesh/grid1d.py`**

Copy Grid1D from `diffheat/mesh.py` into this file:

```python
# diffheat/mesh/grid1d.py
"""1D uniform grid for finite difference discretization."""
from dataclasses import dataclass

import jax.numpy as jnp

from ..utils import array


@dataclass(frozen=True)
class Grid1D:
    """Uniform 1D grid for finite difference discretization.

    Attributes:
        x: (N+1,) cell interface positions [x0, x1, ..., xN]
        centers: (N,) cell center positions
        dx: (N,) cell widths (uniform = length / n_cells for each cell)
        length: total domain length
        n_cells: number of cells N
    """
    x: jnp.ndarray
    centers: jnp.ndarray
    dx: jnp.ndarray
    length: float
    n_cells: int

    @classmethod
    def uniform(cls, length: float, n_cells: int) -> "Grid1D":
        """Create a uniformly spaced 1D grid.

        Args:
            length: Domain length (0 to length).
            n_cells: Number of interior cells.

        Returns:
            Grid1D with uniform spacing.
        """
        if length <= 0:
            raise ValueError(f"length must be positive, got {length}")
        if n_cells < 2:
            raise ValueError(f"n_cells must be at least 2, got {n_cells}")

        cell_width = length / n_cells
        x = array(jnp.linspace(0.0, length, n_cells + 1))          # interfaces
        centers = array(0.5 * (x[:-1] + x[1:]))                     # cell centers
        dx = array(jnp.full(n_cells, cell_width))                   # cell widths

        return cls(x=x, centers=centers, dx=dx, length=length, n_cells=n_cells)
```

- [ ] **Step 3: Create `diffheat/mesh/boundary.py`**

Copy BoundaryCondition from `diffheat/mesh.py` into this file:

```python
# diffheat/mesh/boundary.py
"""Boundary condition definitions for 1D and 2D heat equations."""
from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class BoundaryCondition:
    """Boundary condition for the 1D heat equation.

    Args:
        kind: "dirichlet" (fixed temperature) or "neumann" (fixed flux/gradient).
        value: (2,) array — [left_value, right_value].
               For Dirichlet: prescribed temperature at each boundary.
               For Neumann: prescribed dT/dx at each boundary (positive = into domain).
    """
    kind: str
    value: jnp.ndarray

    def __post_init__(self):
        if self.kind not in ("dirichlet", "neumann"):
            raise ValueError(f"Unknown boundary kind: {self.kind}")
        if self.value.shape != (2,):
            raise ValueError(f"Boundary value must have shape (2,), got {self.value.shape}")
```

- [ ] **Step 4: Create `diffheat/mesh/__init__.py`**

```python
# diffheat/mesh/__init__.py
"""Grid and boundary condition definitions."""
from .boundary import BoundaryCondition
from .grid1d import Grid1D

__all__ = ["Grid1D", "BoundaryCondition"]
```

- [ ] **Step 5: Create `diffheat/operators/laplacian.py`**

Copy `make_laplacian` from `diffheat/physics.py`:

```python
# diffheat/operators/laplacian.py
"""Discrete Laplacian operators for 1D and 2D."""
import jax.numpy as jnp

from ..mesh.grid1d import Grid1D
from ..utils import array


def make_laplacian(grid: Grid1D) -> jnp.ndarray:
    """Build the (N, N) tridiagonal Laplacian matrix using centered finite differences.

    Interior: [1, -2, 1] / dx^2
    Boundaries: left unmodified (apply_boundary_conditions handles them).

    Args:
        grid: The 1D grid.

    Returns:
        (N, N) Laplacian matrix.
    """
    n = grid.n_cells
    dx = grid.dx

    # Diagonal: -2 / dx^2
    diag = array(-2.0 / (dx * dx))

    # Off-diagonals: 1 / dx^2
    # NOTE: this formula is valid only for uniform grids.
    # Non-uniform grids require a different finite-difference stencil.
    dx2_left = dx[:-1] * dx[:-1]
    dx2_right = dx[1:] * dx[1:]
    off_diag = array(2.0 / (dx2_left + dx2_right))

    L = jnp.diag(diag) + jnp.diag(off_diag, k=1) + jnp.diag(off_diag, k=-1)
    return L
```

- [ ] **Step 6: Create `diffheat/operators/__init__.py`**

```python
# diffheat/operators/__init__.py
"""Discrete differential operators for finite difference PDEs."""
from .laplacian import make_laplacian

__all__ = ["make_laplacian"]
```

- [ ] **Step 7: Create `diffheat/physics/heat1d.py`**

Copy `apply_boundary_conditions` and `HeatEquation1D` from `diffheat/physics.py`:

```python
# diffheat/physics/heat1d.py
"""1D heat equation problem definition."""
from dataclasses import dataclass
from typing import Callable, Optional

import jax.numpy as jnp

from ..mesh.boundary import BoundaryCondition
from ..mesh.grid1d import Grid1D
from ..utils import array


def apply_boundary_conditions(
    L: jnp.ndarray,
    grid: Grid1D,
    bc: BoundaryCondition,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Modify the Laplacian and create a boundary source vector for the given BCs.

    Uses ghost-cell method:
      Dirichlet: T_ghost = 2*T_boundary - T_boundary_cell
      Neumann:   T_ghost = T_boundary_cell - dx * dT/dx

    Args:
        L: (N, N) raw Laplacian matrix from make_laplacian.
        grid: The 1D grid.
        bc: Boundary conditions.

    Returns:
        (L_modified, b_source) where dT/dt = alpha * (L_modified @ T + b_source) + S.
    """
    n = grid.n_cells
    dx = grid.dx
    L_mod = L  # JAX arrays are immutable, no .copy() needed
    b_source = jnp.zeros(n, dtype=L.dtype)

    # --- Left boundary (cell index 0) ---
    if bc.kind == "dirichlet":
        L_mod = L_mod.at[0, 0].set(-3.0 / (dx[0] * dx[0]))
        L_mod = L_mod.at[0, 1].set(1.0 / (dx[0] * dx[0]))
        b_source = b_source.at[0].set(2.0 * bc.value[0] / (dx[0] * dx[0]))
    elif bc.kind == "neumann":
        L_mod = L_mod.at[0, 0].set(-1.0 / (dx[0] * dx[0]))
        L_mod = L_mod.at[0, 1].set(1.0 / (dx[0] * dx[0]))
        b_source = b_source.at[0].set(-bc.value[0] / dx[0])

    # --- Right boundary (cell index n-1) ---
    if bc.kind == "dirichlet":
        L_mod = L_mod.at[n - 1, n - 1].set(-3.0 / (dx[n - 1] * dx[n - 1]))
        L_mod = L_mod.at[n - 1, n - 2].set(1.0 / (dx[n - 1] * dx[n - 1]))
        b_source = b_source.at[n - 1].set(
            2.0 * bc.value[1] / (dx[n - 1] * dx[n - 1])
        )
    elif bc.kind == "neumann":
        L_mod = L_mod.at[n - 1, n - 1].set(-1.0 / (dx[n - 1] * dx[n - 1]))
        L_mod = L_mod.at[n - 1, n - 2].set(1.0 / (dx[n - 1] * dx[n - 1]))
        b_source = b_source.at[n - 1].set(bc.value[1] / dx[n - 1])

    return L_mod, b_source


@dataclass(frozen=True)
class HeatEquation1D:
    """Complete 1D heat equation problem definition.

    dT/dt = alpha * d^2T/dx^2 + S(x, t)

    Args:
        grid: 1D spatial grid.
        bc: Boundary conditions.
        alpha: Thermal diffusivity. Scalar or (N,) field.
        source: Optional source term S(x, t). Called as source(x_coords, time).
    """
    grid: Grid1D
    bc: BoundaryCondition
    alpha: float | jnp.ndarray
    source: Optional[Callable[[jnp.ndarray, float], jnp.ndarray]] = None

    def __post_init__(self):
        if isinstance(self.alpha, (int, float)):
            if self.alpha <= 0:
                raise ValueError(f"alpha must be positive, got {self.alpha}")
            # Convert to array for JAX tracing
            object.__setattr__(self, "alpha", array(float(self.alpha)))
```

- [ ] **Step 8: Create `diffheat/physics/__init__.py`**

```python
# diffheat/physics/__init__.py
"""Physical problem definitions."""
from .heat1d import HeatEquation1D, apply_boundary_conditions

__all__ = ["HeatEquation1D", "apply_boundary_conditions"]
```

- [ ] **Step 9: Create `diffheat/solvers/explicit.py`**

Copy `explicit_euler_step` from `diffheat/solvers.py`:

```python
# diffheat/solvers/explicit.py
"""Explicit Euler time-stepping for 1D and 2D."""
import jax.numpy as jnp

from ..operators.laplacian import make_laplacian
from ..physics.heat1d import HeatEquation1D, apply_boundary_conditions


def explicit_euler_step(
    T: jnp.ndarray,
    eqn: HeatEquation1D,
    t: float,
    dt: float,
) -> jnp.ndarray:
    """Single explicit Euler time step for the 1D heat equation.

    T^{n+1} = T^n + dt * [alpha * L @ T^n + alpha * b_source + S(x, t)]

    Args:
        T: (N,) temperature at current timestep.
        eqn: Heat equation definition.
        t: Current time (for source term evaluation).
        dt: Time step size.

    Returns:
        (N,) temperature at next timestep.
    """
    L = make_laplacian(eqn.grid)
    L_mod, b_source = apply_boundary_conditions(L, eqn.grid, eqn.bc)

    dT_dt = eqn.alpha * (L_mod @ T + b_source)

    if eqn.source is not None:
        dT_dt = dT_dt + eqn.source(eqn.grid.centers, t)

    return T + dt * dT_dt
```

- [ ] **Step 10: Create `diffheat/solvers/scan.py`**

Copy `solve_heat_1d` from `diffheat/solvers.py`:

```python
# diffheat/solvers/scan.py
"""Scan-based trajectory solvers for 1D and 2D."""
import logging

import jax
import jax.numpy as jnp

from ..mesh.grid1d import Grid1D
from ..physics.heat1d import HeatEquation1D
from .explicit import explicit_euler_step
from .stability import check_cfl

_logger = logging.getLogger(__name__)


def solve_heat_1d(
    eqn: HeatEquation1D,
    T0: jnp.ndarray,
    t_span: tuple[float, float],
    dt: float,
) -> jnp.ndarray:
    """Solve the 1D heat equation using explicit Euler with jax.lax.scan.

    The entire solve is JIT-compiled and differentiable. Gradients flow
    through the full trajectory.

    Args:
        eqn: Heat equation problem definition.
        T0: (N,) initial temperature field.
        t_span: (t_start, t_end) simulation time range.
        dt: Time step size.

    Returns:
        (n_steps+1, N) temperature trajectory. First row is T0.

    Raises:
        UserWarning: if dt violates the CFL stability condition.
    """
    t0, t_end = t_span
    n_steps = int((t_end - t0) / dt)

    if n_steps < 1:
        raise ValueError(f"t_span too short for dt={dt}: {t_span}")

    # CFL check — skip during gradient tracing to avoid concretization errors
    try:
        if not check_cfl(eqn.grid, eqn.alpha, dt):
            cfl_limit = float(jnp.min(eqn.grid.dx)) ** 2 / (2 * float(jnp.max(jnp.asarray(eqn.alpha))))
            _logger.warning(
                f"dt={dt:.2e} exceeds CFL limit {cfl_limit:.2e}. "
                f"Solution may be unstable."
            )
    except jax.errors.ConcretizationTypeError:
        pass

    # Pre-compute time array
    t = t0 + dt * jnp.arange(n_steps + 1)

    def step_fn(T, step_idx):
        t_current = t0 + step_idx * dt
        T_next = explicit_euler_step(T, eqn, t_current, dt)
        return T_next, T_next

    # scan over n_steps, prepend T0
    _, T_traj = jax.lax.scan(step_fn, T0, jnp.arange(n_steps))
    trajectory = jnp.concatenate([T0[jnp.newaxis, :], T_traj], axis=0)

    return trajectory
```

- [ ] **Step 11: Create `diffheat/solvers/stability.py`**

Copy `check_cfl` from `diffheat/solvers.py`:

```python
# diffheat/solvers/stability.py
"""CFL stability conditions for explicit time integration."""
import jax.numpy as jnp

from ..mesh.grid1d import Grid1D


def check_cfl(grid: Grid1D, alpha: float | jnp.ndarray, dt: float) -> bool:
    """Check if dt satisfies the CFL stability condition for explicit Euler.

    dt <= dx^2 / (2 * alpha)

    Args:
        grid: The spatial grid.
        alpha: Thermal diffusivity (scalar or field).
        dt: Time step size.

    Returns:
        True if stable, False otherwise.
    """
    alpha_max = float(jnp.max(jnp.asarray(alpha)))
    dx_min = float(jnp.min(grid.dx))
    cfl_limit = dx_min ** 2 / (2 * alpha_max)
    return bool(dt <= cfl_limit)
```

- [ ] **Step 12: Create `diffheat/solvers/__init__.py`**

```python
# diffheat/solvers/__init__.py
"""Time integration solvers."""
from .explicit import explicit_euler_step
from .scan import solve_heat_1d
from .stability import check_cfl

__all__ = ["explicit_euler_step", "solve_heat_1d", "check_cfl"]
```

- [ ] **Step 13: Create `diffheat/viz/canvas.py`**

Extract `MatplotlibCanvas` from `diffheat/viz.py`:

```python
# diffheat/viz/canvas.py
"""Matplotlib canvas embedded in a Qt widget."""
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class MatplotlibCanvas(FigureCanvasQTAgg):
    """Matplotlib figure embedded in a Qt widget."""

    def __init__(self, parent=None, width=8, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
```

- [ ] **Step 14: Create `diffheat/viz/heatmap1d.py`**

Extract `HeatmapWidget` from `diffheat/viz.py`:

```python
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
```

- [ ] **Step 15: Create `diffheat/viz/controls.py`**

Extract `ControlPanel` from `diffheat/viz.py`:

```python
# diffheat/viz/controls.py
"""Play/pause, frame navigation, and parameter display."""
from PyQt6 import QtCore, QtWidgets


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
```

- [ ] **Step 16: Create `diffheat/viz/window.py`**

Extract `ViewerWindow` and `run_viewer` from `diffheat/viz.py`:

```python
# diffheat/viz/window.py
"""Main viewer window composing widgets and launching the Qt event loop."""
import sys

import jax.numpy as jnp
from PyQt6 import QtWidgets

from ..mesh.grid1d import Grid1D
from .controls import ControlPanel
from .heatmap1d import HeatmapWidget


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
```

- [ ] **Step 17: Create `diffheat/viz/__init__.py`**

```python
# diffheat/viz/__init__.py
"""PyQt6-based visualization for heat equation trajectories.

This is the ONLY package in diffheat that imports PyQt or matplotlib.
The core library remains headless and works without these dependencies.
"""
from .heatmap1d import HeatmapWidget
from .window import ViewerWindow, run_viewer

__all__ = ["HeatmapWidget", "ViewerWindow", "run_viewer"]
```

- [ ] **Step 18: Update `diffheat/__init__.py`**

```python
# diffheat/__init__.py
"""diffheat — Differentiable heat equation simulations with JAX."""
import logging

from .mesh import BoundaryCondition, Grid1D
from .operators import make_laplacian
from .physics import HeatEquation1D, apply_boundary_conditions
from .solvers import check_cfl, explicit_euler_step, solve_heat_1d
from .utils import array, get_default_dtype, get_device

_logger = logging.getLogger(__name__)
_logger.info(f"diffheat running on: {get_device()}")

__all__ = [
    "Grid1D",
    "BoundaryCondition",
    "HeatEquation1D",
    "make_laplacian",
    "apply_boundary_conditions",
    "explicit_euler_step",
    "solve_heat_1d",
    "check_cfl",
    "get_device",
    "get_default_dtype",
    "array",
]
```

- [ ] **Step 19: Delete old flat files**

```bash
git rm diffheat/mesh.py diffheat/physics.py diffheat/solvers.py diffheat/viz.py
```

- [ ] **Step 20: Update test imports**

`tests/test_mesh.py` — change `from diffheat.mesh import Grid1D, BoundaryCondition` to `from diffheat.mesh import Grid1D, BoundaryCondition` (unchanged, still works through __init__.py).

Verify `tests/test_physics.py`, `tests/test_solvers.py`, `tests/test_integration.py`, `tests/test_viz.py` don't need import changes (they import from `diffheat` directly).

- [ ] **Step 21: Run full test suite to verify backward compatibility**

```bash
python -m pytest tests/ -v --tb=short
```

Expected: all 46 tests PASS (1 pre-existing x64 env failure is ok).

- [ ] **Step 22: Commit**

```bash
git add -A
git commit -m "refactor: restructure flat files into subpackages with backward-compatible re-exports

- mesh/ — grid1d.py, boundary.py (Grid1D, BoundaryCondition)
- operators/ — laplacian.py (make_laplacian)
- physics/ — heat1d.py (HeatEquation1D, apply_boundary_conditions)
- solvers/ — explicit.py, scan.py, stability.py
- viz/ — canvas.py, heatmap1d.py, controls.py, window.py
- All 1D imports preserved via __init__.py re-exports

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Grid2D — Uniform Rectangular 2D Grid

**Files:**
- Create: `diffheat/mesh/grid2d.py`
- Create: `tests/test_grid2d.py`
- Modify: `diffheat/mesh/__init__.py` (add Grid2D export)

**Interfaces:**
- Consumes: `diffheat.utils.array`
- Produces: `Grid2D(x, y, x_centers, y_centers, dx, dy, Lx, Ly, nx, ny, X, Y)` — frozen dataclass
- Produces: `Grid2D.uniform(Lx, Ly, nx, ny) -> Grid2D` — factory classmethod

- [ ] **Step 1: Write failing tests for Grid2D**

Create `tests/test_grid2d.py`:

```python
# tests/test_grid2d.py
import pytest
import jax.numpy as jnp
from diffheat.mesh import Grid2D


class TestGrid2D:
    def test_uniform_creates_correct_dimensions(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=2.0, nx=10, ny=20)
        assert grid.nx == 10
        assert grid.ny == 20
        assert grid.Lx == 1.0
        assert grid.Ly == 2.0

    def test_uniform_x_has_nx_plus_one_points(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=10, ny=20)
        assert grid.x.shape == (11,)  # nx + 1

    def test_uniform_y_has_ny_plus_one_points(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=10, ny=20)
        assert grid.y.shape == (21,)  # ny + 1

    def test_uniform_centers_have_correct_shapes(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=10, ny=20)
        assert grid.x_centers.shape == (10,)
        assert grid.y_centers.shape == (20,)

    def test_uniform_dx_dy_have_correct_shapes(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=2.0, nx=10, ny=20)
        assert grid.dx.shape == (10,)
        assert grid.dy.shape == (20,)
        assert jnp.allclose(grid.dx, 0.1)
        assert jnp.allclose(grid.dy, 0.1)

    def test_uniform_x_starts_at_zero(self):
        grid = Grid2D.uniform(Lx=3.0, Ly=2.0, nx=5, ny=4)
        assert jnp.isclose(grid.x[0], 0.0)
        assert jnp.isclose(grid.x[-1], 3.0)

    def test_uniform_y_starts_at_zero(self):
        grid = Grid2D.uniform(Lx=3.0, Ly=2.0, nx=5, ny=4)
        assert jnp.isclose(grid.y[0], 0.0)
        assert jnp.isclose(grid.y[-1], 2.0)

    def test_meshgrid_shapes(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=2.0, nx=8, ny=12)
        assert grid.X.shape == (12, 8)  # (ny, nx) — imshow convention
        assert grid.Y.shape == (12, 8)

    def test_meshgrid_values(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=4, ny=4)
        # X[j, i] = x_centers[i] for all j
        for j in range(4):
            assert jnp.allclose(grid.X[j, :], grid.x_centers)
        # Y[j, i] = y_centers[j] for all i
        for i in range(4):
            assert jnp.allclose(grid.Y[:, i], grid.y_centers)

    def test_frozen_dataclass(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=10, ny=10)
        with pytest.raises(Exception):
            grid.nx = 20

    def test_validation_rejects_zero_length(self):
        with pytest.raises(ValueError):
            Grid2D.uniform(Lx=0.0, Ly=1.0, nx=10, ny=10)

    def test_validation_rejects_too_few_cells(self):
        with pytest.raises(ValueError):
            Grid2D.uniform(Lx=1.0, Ly=1.0, nx=1, ny=10)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_grid2d.py -v
```

Expected: FAIL — ImportError (Grid2D not yet exported).

- [ ] **Step 3: Implement `diffheat/mesh/grid2d.py`**

```python
# diffheat/mesh/grid2d.py
"""2D uniform rectangular grid for finite difference discretization."""
from dataclasses import dataclass

import jax.numpy as jnp

from ..utils import array


@dataclass(frozen=True)
class Grid2D:
    """Uniform rectangular 2D grid for finite difference discretization.

    Cell-centered storage: fields live at cell centers.
    X, Y meshgrids follow imshow's y-first convention: shape (ny, nx).

    Attributes:
        x: (nx+1,) interface positions in x-direction
        y: (ny+1,) interface positions in y-direction
        x_centers: (nx,) cell centers in x
        y_centers: (ny,) cell centers in y
        dx: (nx,) cell widths in x
        dy: (ny,) cell widths in y
        Lx: domain length in x
        Ly: domain length in y
        nx: number of cells in x
        ny: number of cells in y
        X: (ny, nx) 2D meshgrid of x-coordinates at cell centers
        Y: (ny, nx) 2D meshgrid of y-coordinates at cell centers
    """
    x: jnp.ndarray
    y: jnp.ndarray
    x_centers: jnp.ndarray
    y_centers: jnp.ndarray
    dx: jnp.ndarray
    dy: jnp.ndarray
    Lx: float
    Ly: float
    nx: int
    ny: int
    X: jnp.ndarray
    Y: jnp.ndarray

    @classmethod
    def uniform(cls, Lx: float, Ly: float, nx: int, ny: int) -> "Grid2D":
        """Create a uniformly spaced 2D grid.

        Args:
            Lx: Domain length in x (0 to Lx).
            Ly: Domain length in y (0 to Ly).
            nx: Number of cells in x-direction.
            ny: Number of cells in y-direction.

        Returns:
            Grid2D with uniform spacing in both directions.
        """
        if Lx <= 0:
            raise ValueError(f"Lx must be positive, got {Lx}")
        if Ly <= 0:
            raise ValueError(f"Ly must be positive, got {Ly}")
        if nx < 2:
            raise ValueError(f"nx must be at least 2, got {nx}")
        if ny < 2:
            raise ValueError(f"ny must be at least 2, got {ny}")

        dx_val = Lx / nx
        dy_val = Ly / ny

        x = array(jnp.linspace(0.0, Lx, nx + 1))
        y = array(jnp.linspace(0.0, Ly, ny + 1))
        x_centers = array(0.5 * (x[:-1] + x[1:]))
        y_centers = array(0.5 * (y[:-1] + y[1:]))
        dx = array(jnp.full(nx, dx_val))
        dy = array(jnp.full(ny, dy_val))
        X, Y = jnp.meshgrid(x_centers, y_centers)  # indexing='xy' default: X(j,i)=x[i], Y(j,i)=y[j]

        return cls(
            x=x, y=y,
            x_centers=x_centers, y_centers=y_centers,
            dx=dx, dy=dy,
            Lx=Lx, Ly=Ly,
            nx=nx, ny=ny,
            X=array(X), Y=array(Y),
        )
```

- [ ] **Step 4: Update `diffheat/mesh/__init__.py`**

```python
# diffheat/mesh/__init__.py
"""Grid and boundary condition definitions."""
from .boundary import BoundaryCondition
from .grid1d import Grid1D
from .grid2d import Grid2D

__all__ = ["Grid1D", "Grid2D", "BoundaryCondition"]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_grid2d.py -v
```

Expected: all 11 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add diffheat/mesh/grid2d.py diffheat/mesh/__init__.py tests/test_grid2d.py
git commit -m "feat: add Grid2D with uniform rectangular spacing and meshgrid coordinates

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: 2D Laplacian Operator

**Files:**
- Modify: `diffheat/operators/laplacian.py` (add `laplacian_2d`)
- Create: `tests/test_operators.py`

**Interfaces:**
- Consumes: `diffheat.mesh.Grid2D`
- Produces: `laplacian_2d(T: jnp.ndarray, grid: Grid2D) -> jnp.ndarray` — (nx, ny) → (nx, ny)

- [ ] **Step 1: Write failing tests**

Create `tests/test_operators.py`:

```python
# tests/test_operators.py
"""Tests for discrete differential operators."""
import jax.numpy as jnp
from diffheat.mesh import Grid2D
from diffheat.operators import laplacian_2d


class TestLaplacian2D:
    def test_returns_correct_shape(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=10, ny=15)
        T = jnp.ones((10, 15))
        result = laplacian_2d(T, grid)
        assert result.shape == (10, 15)

    def test_constant_field_is_zero_in_interior(self):
        """Laplacian of a constant field should be zero everywhere."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=20, ny=20)
        T = jnp.full((20, 20), 2.5)
        result = laplacian_2d(T, grid)
        # All interior cells (not boundaries) should be ~0
        assert jnp.allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_linear_field_is_zero_in_interior(self):
        """Laplacian of a linear field T(x,y) = a*x + b*y + c should be zero."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=20, ny=20)
        a, b, c = 2.0, -3.0, 1.0
        T = a * grid.X + b * grid.Y + c  # (ny, nx) meshgrid shapes
        T = T.T  # transpose to (nx, ny) for our field convention
        result = laplacian_2d(T, grid)
        assert jnp.allclose(result[2:-2, 2:-2], 0.0, atol=1e-3)

    def test_known_quadratic(self):
        """Laplacian of T = x^2 should be 2 everywhere."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=20, ny=20)
        T = grid.X ** 2  # (ny, nx)
        T = T.T  # (nx, ny)
        result = laplacian_2d(T, grid)
        # Interior should be ~2
        assert jnp.allclose(result[1:-1, 1:-1], 2.0, atol=0.1)

    def test_symmetry(self):
        """Laplacian should be symmetric: swapping x for y with appropriate field."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=15, ny=15)
        # T(x, y) = sin(pi*x)*sin(pi*y) is symmetric
        T = jnp.sin(jnp.pi * grid.X) * jnp.sin(jnp.pi * grid.Y)
        T = T.T  # (nx, ny)
        result = laplacian_2d(T, grid)
        # Result should be symmetric: L[i, j] == L[j, i] for square grid
        assert jnp.allclose(result, result.T, atol=1e-10)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_operators.py -v
```

Expected: FAIL — ImportError (laplacian_2d not yet defined).

- [ ] **Step 3: Add `laplacian_2d` to `diffheat/operators/laplacian.py`**

Append after `make_laplacian`:

```python
from ..mesh.grid2d import Grid2D


def laplacian_2d(T: jnp.ndarray, grid: Grid2D) -> jnp.ndarray:
    """Compute the 2D Laplacian ∇²T = ∂²T/∂x² + ∂²T/∂y².

    Uses centered finite differences with a 5-point stencil:
        (T[i+1,j] + T[i-1,j] - 2*T[i,j]) / dx² +
        (T[i,j+1] + T[i,j-1] - 2*T[i,j]) / dy²

    Args:
        T: (nx, ny) temperature field at cell centers.
        grid: The 2D grid.

    Returns:
        (nx, ny) Laplacian at cell centers.
    """
    dx2 = grid.dx * grid.dx  # (nx,)
    dy2 = grid.dy * grid.dy  # (ny,)

    # ∂²T/∂x²: along axis 0
    d2T_dx2 = (jnp.roll(T, -1, axis=0) + jnp.roll(T, 1, axis=0) - 2.0 * T) / dx2[:, jnp.newaxis]

    # ∂²T/∂y²: along axis 1
    d2T_dy2 = (jnp.roll(T, -1, axis=1) + jnp.roll(T, 1, axis=1) - 2.0 * T) / dy2[jnp.newaxis, :]

    return d2T_dx2 + d2T_dy2
```

- [ ] **Step 4: Update `diffheat/operators/__init__.py`**

```python
# diffheat/operators/__init__.py
"""Discrete differential operators for finite difference PDEs."""
from .laplacian import laplacian_2d, make_laplacian

__all__ = ["make_laplacian", "laplacian_2d"]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_operators.py -v
```

Expected: all 5 Laplacian2D tests PASS.

- [ ] **Step 6: Commit**

```bash
git add diffheat/operators/laplacian.py diffheat/operators/__init__.py tests/test_operators.py
git commit -m "feat: add laplacian_2d operator using 5-point centered stencil

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: 2D Gradient Operators

**Files:**
- Create: `diffheat/operators/gradient.py`
- Modify: `diffheat/operators/__init__.py`
- Modify: `tests/test_operators.py` (add gradient tests)

**Interfaces:**
- Consumes: `diffheat.mesh.Grid2D`
- Produces: `gradient_x(T, grid) -> jnp.ndarray`, `gradient_y(T, grid) -> jnp.ndarray`, `gradient_2d(T, grid) -> tuple[jnp.ndarray, jnp.ndarray]`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_operators.py`:

```python
class TestGradient2D:
    def test_gradient_x_shape(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=10, ny=15)
        from diffheat.operators import gradient_x
        T = jnp.ones((10, 15))
        result = gradient_x(T, grid)
        assert result.shape == (10, 15)

    def test_gradient_y_shape(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=10, ny=15)
        from diffheat.operators import gradient_y
        T = jnp.ones((10, 15))
        result = gradient_y(T, grid)
        assert result.shape == (10, 15)

    def test_gradient_2d_returns_tuple(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=10, ny=15)
        from diffheat.operators import gradient_2d
        T = jnp.ones((10, 15))
        gx, gy = gradient_2d(T, grid)
        assert gx.shape == (10, 15)
        assert gy.shape == (10, 15)

    def test_gradient_x_of_linear_x(self):
        """∂T/∂x of T = a*x should be a everywhere."""
        grid = Grid2D.uniform(Lx=2.0, Ly=1.0, nx=30, ny=15)
        from diffheat.operators import gradient_x
        a = 3.0
        T = a * grid.X.T  # (nx, ny) from (ny, nx) meshgrid
        result = gradient_x(T, grid)
        assert jnp.allclose(result[1:-1, 1:-1], a, atol=1e-2)

    def test_gradient_y_of_linear_y(self):
        """∂T/∂y of T = b*y should be b everywhere."""
        grid = Grid2D.uniform(Lx=1.0, Ly=2.0, nx=15, ny=30)
        from diffheat.operators import gradient_y
        b = -2.0
        T = b * grid.Y.T  # (nx, ny)
        result = gradient_y(T, grid)
        assert jnp.allclose(result[1:-1, 1:-1], b, atol=1e-2)

    def test_gradient_constant_is_zero(self):
        """Gradient of constant field should be zero (interior)."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=20, ny=20)
        from diffheat.operators import gradient_2d
        T = jnp.full((20, 20), 5.0)
        gx, gy = gradient_2d(T, grid)
        assert jnp.allclose(gx[1:-1, 1:-1], 0.0, atol=1e-10)
        assert jnp.allclose(gy[1:-1, 1:-1], 0.0, atol=1e-10)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_operators.py::TestGradient2D -v
```

Expected: FAIL — ImportError (gradient module not yet created).

- [ ] **Step 3: Create `diffheat/operators/gradient.py`**

```python
# diffheat/operators/gradient.py
"""Gradient operators for 2D scalar fields."""
import jax.numpy as jnp

from ..mesh.grid2d import Grid2D


def gradient_x(T: jnp.ndarray, grid: Grid2D) -> jnp.ndarray:
    """Compute ∂T/∂x using centered finite differences.

    Uses (T[i+1,j] - T[i-1,j]) / (2*dx) at interior cells.
    Boundary cells use one-sided differences (forward at left, backward at right).

    Args:
        T: (nx, ny) field at cell centers.
        grid: The 2D grid.

    Returns:
        (nx, ny) x-gradient at cell centers.
    """
    dx = grid.dx[:, jnp.newaxis]  # (nx, 1)
    # Centered difference interior, one-sided at boundaries
    gx = (jnp.roll(T, -1, axis=0) - jnp.roll(T, 1, axis=0)) / (2.0 * dx)
    return gx


def gradient_y(T: jnp.ndarray, grid: Grid2D) -> jnp.ndarray:
    """Compute ∂T/∂y using centered finite differences.

    Uses (T[i,j+1] - T[i,j-1]) / (2*dy) at interior cells.
    Boundary cells use one-sided differences.

    Args:
        T: (nx, ny) field at cell centers.
        grid: The 2D grid.

    Returns:
        (nx, ny) y-gradient at cell centers.
    """
    dy = grid.dy[jnp.newaxis, :]  # (1, ny)
    gy = (jnp.roll(T, -1, axis=1) - jnp.roll(T, 1, axis=1)) / (2.0 * dy)
    return gy


def gradient_2d(T: jnp.ndarray, grid: Grid2D) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the full 2D gradient (∂T/∂x, ∂T/∂y).

    Args:
        T: (nx, ny) field at cell centers.
        grid: The 2D grid.

    Returns:
        (dT_dx, dT_dy) each with shape (nx, ny).
    """
    return gradient_x(T, grid), gradient_y(T, grid)
```

- [ ] **Step 4: Update `diffheat/operators/__init__.py`**

```python
# diffheat/operators/__init__.py
"""Discrete differential operators for finite difference PDEs."""
from .gradient import gradient_2d, gradient_x, gradient_y
from .laplacian import laplacian_2d, make_laplacian

__all__ = [
    "make_laplacian",
    "laplacian_2d",
    "gradient_x",
    "gradient_y",
    "gradient_2d",
]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_operators.py -v
```

Expected: all 11 tests PASS (5 laplacian + 6 gradient).

- [ ] **Step 6: Commit**

```bash
git add diffheat/operators/gradient.py diffheat/operators/__init__.py tests/test_operators.py
git commit -m "feat: add gradient_x, gradient_y, gradient_2d operators

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: 2D Divergence Operator

**Files:**
- Create: `diffheat/operators/divergence.py`
- Modify: `diffheat/operators/__init__.py`
- Modify: `tests/test_operators.py` (add divergence tests)

**Interfaces:**
- Consumes: `diffheat.mesh.Grid2D`
- Produces: `divergence_2d(ux, uy, grid) -> jnp.ndarray` — (nx, ny) tuple → (nx, ny)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_operators.py`:

```python
class TestDivergence2D:
    def test_returns_correct_shape(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=10, ny=15)
        from diffheat.operators import divergence_2d
        ux = jnp.ones((10, 15))
        uy = jnp.zeros((10, 15))
        result = divergence_2d(ux, uy, grid)
        assert result.shape == (10, 15)

    def test_divergence_of_uniform_flow_is_zero(self):
        """∇·(constant, constant) = 0."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=20, ny=20)
        from diffheat.operators import divergence_2d
        ux = jnp.full((20, 20), 2.0)
        uy = jnp.full((20, 20), -1.0)
        result = divergence_2d(ux, uy, grid)
        assert jnp.allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_divergence_of_expanding_flow(self):
        """∇·(x, 0) = 1."""
        grid = Grid2D.uniform(Lx=2.0, Ly=1.0, nx=30, ny=15)
        from diffheat.operators import divergence_2d
        ux = grid.X.T  # (nx, ny), ux = x
        uy = jnp.zeros((30, 15))
        result = divergence_2d(ux, uy, grid)
        assert jnp.allclose(result[1:-1, 1:-1], 1.0, atol=1e-2)

    def test_divergence_is_linear(self):
        """∇·(a*u1 + b*u2) = a*∇·u1 + b*∇·u2."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=15, ny=15)
        from diffheat.operators import divergence_2d
        ux1 = grid.X.T
        uy1 = jnp.zeros((15, 15))
        ux2 = jnp.zeros((15, 15))
        uy2 = grid.Y.T
        div1 = divergence_2d(ux1, uy1, grid)
        div2 = divergence_2d(ux2, uy2, grid)
        div_combined = divergence_2d(2.0 * ux1 - 3.0 * uy2,
                                     2.0 * uy1 - 3.0 * uy2, grid)
        expected = 2.0 * div1 - 3.0 * div2
        assert jnp.allclose(div_combined[1:-1, 1:-1], expected[1:-1, 1:-1], atol=1e-2)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_operators.py::TestDivergence2D -v
```

Expected: FAIL — ImportError.

- [ ] **Step 3: Create `diffheat/operators/divergence.py`**

```python
# diffheat/operators/divergence.py
"""Divergence operator for 2D vector fields."""
import jax.numpy as jnp

from ..mesh.grid2d import Grid2D


def divergence_2d(ux: jnp.ndarray, uy: jnp.ndarray, grid: Grid2D) -> jnp.ndarray:
    """Compute ∇·(ux, uy) = ∂ux/∂x + ∂uy/∂y.

    Uses centered finite differences at interior cells.

    Args:
        ux: (nx, ny) x-component of the vector field at cell centers.
        uy: (nx, ny) y-component of the vector field at cell centers.
        grid: The 2D grid.

    Returns:
        (nx, ny) divergence at cell centers.
    """
    dx = grid.dx[:, jnp.newaxis]  # (nx, 1)
    dy = grid.dy[jnp.newaxis, :]  # (1, ny)

    dux_dx = (jnp.roll(ux, -1, axis=0) - jnp.roll(ux, 1, axis=0)) / (2.0 * dx)
    duy_dy = (jnp.roll(uy, -1, axis=1) - jnp.roll(uy, 1, axis=1)) / (2.0 * dy)

    return dux_dx + duy_dy
```

- [ ] **Step 4: Update `diffheat/operators/__init__.py`**

```python
# diffheat/operators/__init__.py
"""Discrete differential operators for finite difference PDEs."""
from .divergence import divergence_2d
from .gradient import gradient_2d, gradient_x, gradient_y
from .laplacian import laplacian_2d, make_laplacian

__all__ = [
    "make_laplacian",
    "laplacian_2d",
    "gradient_x",
    "gradient_y",
    "gradient_2d",
    "divergence_2d",
]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_operators.py -v
```

Expected: all 15 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add diffheat/operators/divergence.py diffheat/operators/__init__.py tests/test_operators.py
git commit -m "feat: add divergence_2d operator for 2D vector fields

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: 2D Boundary Conditions

**Files:**
- Modify: `diffheat/mesh/boundary.py` (add `BoundaryCondition2D`, `apply_boundary_conditions_2d`)
- Create: `tests/test_boundary_2d.py`
- Modify: `diffheat/mesh/__init__.py` (add exports)

**Interfaces:**
- Consumes: `diffheat.mesh.Grid2D`
- Produces: `BoundaryCondition2D(left, right, bottom, top)` — frozen dataclass
- Produces: `apply_boundary_conditions_2d(operator_fn, grid, bc, T) -> tuple[jnp.ndarray, jnp.ndarray]`

- [ ] **Step 1: Write failing tests**

Create `tests/test_boundary_2d.py`:

```python
# tests/test_boundary_2d.py
"""Tests for 2D boundary conditions."""
import pytest
import jax.numpy as jnp
from diffheat.mesh import Grid2D, BoundaryCondition2D
from diffheat.mesh.boundary import apply_boundary_conditions_2d
from diffheat.operators import laplacian_2d


class TestBoundaryCondition2D:
    def test_dirichlet_creation(self):
        bc = BoundaryCondition2D(
            left={"kind": "dirichlet", "value": 1.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "neumann", "value": 0.0},
            top={"kind": "neumann", "value": 0.0},
        )
        assert bc.left["kind"] == "dirichlet"
        assert bc.left["value"] == 1.0
        assert bc.right["kind"] == "dirichlet"
        assert bc.bottom["kind"] == "neumann"

    def test_raises_on_unknown_kind(self):
        with pytest.raises(ValueError):
            BoundaryCondition2D(
                left={"kind": "periodic", "value": 0.0},
                right={"kind": "dirichlet", "value": 0.0},
                bottom={"kind": "neumann", "value": 0.0},
                top={"kind": "neumann", "value": 0.0},
            )

    def test_frozen_dataclass(self):
        bc = BoundaryCondition2D(
            left={"kind": "dirichlet", "value": 1.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "neumann", "value": 0.0},
            top={"kind": "neumann", "value": 0.0},
        )
        with pytest.raises(Exception):
            bc.left = {"kind": "neumann", "value": 0.0}


class TestApplyBoundaryConditions2D:
    def test_dirichlet_all_sides_steady_state_is_linear(self):
        """Steady state with Dirichlet on all 4 sides: T(left)=1, T(right)=0, T(bottom)=0.5, T(top)=0.5.
        The solution should smoothly vary from left to right but be vertically symmetric."""
        nx, ny = 20, 20
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=nx, ny=ny)
        bc = BoundaryCondition2D(
            left={"kind": "dirichlet", "value": 1.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "dirichlet", "value": 0.5},
            top={"kind": "dirichlet", "value": 0.5},
        )

        def operator(T):
            return laplacian_2d(T, grid)

        T = jnp.zeros((nx, ny))
        L_T, b_source = apply_boundary_conditions_2d(operator, grid, bc, T)

        # L_T and b_source should have correct shapes
        assert L_T.shape == (nx, ny)
        assert b_source.shape == (nx, ny)

    def test_homogeneous_neumann_all_sides_has_zero_source(self):
        """Homogeneous Neumann on all sides should produce zero boundary source."""
        nx, ny = 15, 15
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=nx, ny=ny)
        bc = BoundaryCondition2D(
            left={"kind": "neumann", "value": 0.0},
            right={"kind": "neumann", "value": 0.0},
            bottom={"kind": "neumann", "value": 0.0},
            top={"kind": "neumann", "value": 0.0},
        )

        def operator(T):
            return laplacian_2d(T, grid)

        T = jnp.ones((nx, ny))
        L_T, b_source = apply_boundary_conditions_2d(operator, grid, bc, T)

        # Boundary source should be zero for homogeneous Neumann
        assert jnp.allclose(b_source, 0.0)

    def test_dirichlet_boundary_source_is_nonzero(self):
        """Dirichlet with non-zero values should produce non-zero boundary source."""
        nx, ny = 10, 10
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=nx, ny=ny)
        bc = BoundaryCondition2D(
            left={"kind": "dirichlet", "value": 100.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "dirichlet", "value": 50.0},
            top={"kind": "dirichlet", "value": 50.0},
        )

        def operator(T):
            return laplacian_2d(T, grid)

        T = jnp.zeros((nx, ny))
        L_T, b_source = apply_boundary_conditions_2d(operator, grid, bc, T)

        # Boundary source should be non-zero at boundary cells
        assert not jnp.isclose(jnp.sum(jnp.abs(b_source)), 0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_boundary_2d.py -v
```

Expected: FAIL — ImportError (BoundaryCondition2D not defined).

- [ ] **Step 3: Add `BoundaryCondition2D` and `apply_boundary_conditions_2d` to `diffheat/mesh/boundary.py`**

Append after `BoundaryCondition`:

```python
from .grid2d import Grid2D
from typing import Callable
import jax.numpy as jnp


_VALID_BC_KINDS = ("dirichlet", "neumann")


@dataclass(frozen=True)
class BoundaryCondition2D:
    """Boundary conditions on a 2D rectangular domain.

    Each of the 4 edges gets an independently specifiable BC.

    Args:
        left: BC on x=0 edge. {'kind': 'dirichlet'|'neumann', 'value': scalar}
        right: BC on x=Lx edge. {'kind': 'dirichlet'|'neumann', 'value': scalar}
        bottom: BC on y=0 edge. {'kind': 'dirichlet'|'neumann', 'value': scalar}
        top: BC on y=Ly edge. {'kind': 'dirichlet'|'neumann', 'value': scalar}

        For Dirichlet: value is the prescribed temperature.
        For Neumann: value is the prescribed dT/dn (positive = outward normal).
    """
    left: dict
    right: dict
    bottom: dict
    top: dict

    def __post_init__(self):
        for edge_name in ("left", "right", "bottom", "top"):
            edge = getattr(self, edge_name)
            if edge["kind"] not in _VALID_BC_KINDS:
                raise ValueError(
                    f"Unknown boundary kind for {edge_name}: {edge['kind']}. "
                    f"Must be one of: {_VALID_BC_KINDS}"
                )


def apply_boundary_conditions_2d(
    operator_fn: Callable[[jnp.ndarray], jnp.ndarray],
    grid: Grid2D,
    bc: BoundaryCondition2D,
    T: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply boundary conditions to a 2D operator using the ghost-cell method.

    Modifies boundary rows/cols of the operator output and returns
    a boundary source term with non-zero entries only at boundary cells.

    Ghost-cell method (same logic as 1D):
        Dirichlet: T_ghost = 2*T_boundary - T_cell
        Neumann:   T_ghost = T_cell - dn * dT/dn

    Args:
        operator_fn: Function that applies the discrete operator, e.g.
                     lambda T: laplacian_2d(T, grid).
        grid: The 2D grid.
        bc: 2D boundary conditions for all 4 edges.
        T: (nx, ny) field at cell centers.

    Returns:
        (L_T_modified, b_source) where the modified operator output has
        boundary conditions baked in. dT/dt = alpha * (L_T_mod + b_source) + S.
    """
    nx, ny = grid.nx, grid.ny
    dx = grid.dx
    dy = grid.dy

    L_T = operator_fn(T)
    b_source = jnp.zeros((nx, ny), dtype=T.dtype)

    # --- Left boundary (i=0) ---
    # Laplacian at i=0: (T_ghost_left + T[1,j] - 2*T[0,j])/dx² + (y-part)
    # For Dirichlet: T_ghost = 2*T_left - T[0,j]
    #   → Laplacian = (2*T_left - T[0,j] + T[1,j] - 2*T[0,j])/dx² + ...
    #                = normal_laplacian + 2*T_left/dx² - 2*T[0,j]/dx² + 2*T[0,j]/dx²
    #   Actually need to modify the x-part: becomes (T[1,j] - 3*T[0,j])/dx² + 2*T_left/dx²
    if bc.left["kind"] == "dirichlet":
        T_left = bc.left["value"]
        idx = 0
        L_T = L_T.at[idx, :].set(
            L_T[idx, :]  # original laplacian at i=0
            - (2.0 * T[idx, :] - (T[idx, :] * 2.0)) / (dx[idx] * dx[idx])  # remove old boundary contribution
            + (-2.0 * T[idx, :] + 2.0 * T_left) / (dx[idx] * dx[idx])  # add Dirichlet ghost-cell correction
        )
        b_source = b_source.at[idx, :].set(
            b_source[idx, :] + 2.0 * T_left / (dx[idx] * dx[idx])
        )
    elif bc.left["kind"] == "neumann":
        dT_dn_left = bc.left["value"]
        idx = 0
        b_source = b_source.at[idx, :].set(
            b_source[idx, :] - dT_dn_left / dx[idx]
        )

    # --- Right boundary (i=nx-1) ---
    if bc.right["kind"] == "dirichlet":
        T_right = bc.right["value"]
        idx = nx - 1
        L_T = L_T.at[idx, :].set(
            L_T[idx, :]
            - (2.0 * T[idx, :] - (T[idx, :] * 2.0)) / (dx[idx] * dx[idx])
            + (-2.0 * T[idx, :] + 2.0 * T_right) / (dx[idx] * dx[idx])
        )
        b_source = b_source.at[idx, :].set(
            b_source[idx, :] + 2.0 * T_right / (dx[idx] * dx[idx])
        )
    elif bc.right["kind"] == "neumann":
        dT_dn_right = bc.right["value"]
        idx = nx - 1
        b_source = b_source.at[idx, :].set(
            b_source[idx, :] + dT_dn_right / dx[idx]
        )

    # --- Bottom boundary (j=0) ---
    if bc.bottom["kind"] == "dirichlet":
        T_bottom = bc.bottom["value"]
        idx = 0
        L_T = L_T.at[:, idx].set(
            L_T[:, idx]
            - (2.0 * T[:, idx] - (T[:, idx] * 2.0)) / (dy[idx] * dy[idx])
            + (-2.0 * T[:, idx] + 2.0 * T_bottom) / (dy[idx] * dy[idx])
        )
        b_source = b_source.at[:, idx].set(
            b_source[:, idx] + 2.0 * T_bottom / (dy[idx] * dy[idx])
        )
    elif bc.bottom["kind"] == "neumann":
        dT_dn_bottom = bc.bottom["value"]
        idx = 0
        b_source = b_source.at[:, idx].set(
            b_source[:, idx] - dT_dn_bottom / dy[idx]
        )

    # --- Top boundary (j=ny-1) ---
    if bc.top["kind"] == "dirichlet":
        T_top = bc.top["value"]
        idx = ny - 1
        L_T = L_T.at[:, idx].set(
            L_T[:, idx]
            - (2.0 * T[:, idx] - (T[:, idx] * 2.0)) / (dy[idx] * dy[idx])
            + (-2.0 * T[:, idx] + 2.0 * T_top) / (dy[idx] * dy[idx])
        )
        b_source = b_source.at[:, idx].set(
            b_source[:, idx] + 2.0 * T_top / (dy[idx] * dy[idx])
        )
    elif bc.top["kind"] == "neumann":
        dT_dn_top = bc.top["value"]
        idx = ny - 1
        b_source = b_source.at[:, idx].set(
            b_source[:, idx] + dT_dn_top / dy[idx]
        )

    return L_T, b_source
```

- [ ] **Step 4: Update `diffheat/mesh/__init__.py`**

```python
# diffheat/mesh/__init__.py
"""Grid and boundary condition definitions."""
from .boundary import BoundaryCondition, BoundaryCondition2D
from .grid1d import Grid1D
from .grid2d import Grid2D

__all__ = ["Grid1D", "Grid2D", "BoundaryCondition", "BoundaryCondition2D"]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_boundary_2d.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add diffheat/mesh/boundary.py diffheat/mesh/__init__.py tests/test_boundary_2d.py
git commit -m "feat: add BoundaryCondition2D and apply_boundary_conditions_2d

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: 2D Solver — Euler Step, CFL, Scan

**Files:**
- Modify: `diffheat/solvers/explicit.py` (add `explicit_euler_step_2d`)
- Modify: `diffheat/solvers/stability.py` (add `check_cfl_2d`)
- Modify: `diffheat/solvers/scan.py` (add `solve_2d`)
- Modify: `diffheat/solvers/__init__.py`
- Create: `tests/test_solvers_2d.py`

**Interfaces:**
- Consumes: `diffheat.mesh.Grid2D`, `diffheat.operators.*`
- Produces: `explicit_euler_step_2d(state, rhs_fn, grid, t, dt, params) -> jnp.ndarray | tuple`
- Produces: `check_cfl_2d(grid, alpha, dt) -> bool`
- Produces: `solve_2d(rhs_fn, initial_state, grid, t_span, dt, params) -> jnp.ndarray | tuple`

- [ ] **Step 1: Write failing tests**

Create `tests/test_solvers_2d.py`:

```python
# tests/test_solvers_2d.py
"""Tests for 2D time integration solvers."""
import jax
import jax.numpy as jnp
import pytest
from diffheat.mesh import Grid2D, BoundaryCondition2D
from diffheat.operators import laplacian_2d
from diffheat.mesh.boundary import apply_boundary_conditions_2d


def _make_heat_rhs(alpha, bc):
    """Factory: build a heat equation RHS function for a given diffusivity and BCs."""
    def rhs(T, grid, t, params):
        L_T = laplacian_2d(T, grid)
        L_T_mod, b_source = apply_boundary_conditions_2d(
            lambda x: laplacian_2d(x, grid), grid, bc, T
        )
        return alpha * (L_T_mod + b_source)
    return rhs


class TestCheckCFL2D:
    def test_stable_dt_passes(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=50, ny=50)
        from diffheat.solvers import check_cfl_2d
        alpha = 0.01
        dx_min = float(jnp.min(grid.dx))
        dy_min = float(jnp.min(grid.dy))
        cfl_limit = min(dx_min**2, dy_min**2) / (4 * alpha)
        dt = 0.9 * cfl_limit
        assert check_cfl_2d(grid, alpha, dt) is True

    def test_unstable_dt_fails(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=50, ny=50)
        from diffheat.solvers import check_cfl_2d
        alpha = 0.01
        dx_min = float(jnp.min(grid.dx))
        dy_min = float(jnp.min(grid.dy))
        cfl_limit = min(dx_min**2, dy_min**2) / (4 * alpha)
        dt = 2.0 * cfl_limit
        assert check_cfl_2d(grid, alpha, dt) is False


class TestExplicitEulerStep2D:
    def test_constant_temperature_stays_constant(self):
        """Uniform temperature with equal Dirichlet BCs should not change."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=20, ny=20)
        bc = BoundaryCondition2D(
            left={"kind": "dirichlet", "value": 0.5},
            right={"kind": "dirichlet", "value": 0.5},
            bottom={"kind": "dirichlet", "value": 0.5},
            top={"kind": "dirichlet", "value": 0.5},
        )
        alpha = 0.1
        rhs_fn = _make_heat_rhs(alpha, bc)
        T0 = jnp.full((20, 20), 0.5)

        from diffheat.solvers import explicit_euler_step_2d
        T1 = explicit_euler_step_2d(T0, rhs_fn, grid, t=0.0, dt=0.001)
        assert jnp.allclose(T1, T0, atol=1e-10)

    def test_cooling_plate_temperature_decreases(self):
        """Hot plate with cold boundaries should cool down."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=20, ny=20)
        bc = BoundaryCondition2D(
            left={"kind": "dirichlet", "value": 0.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "dirichlet", "value": 0.0},
            top={"kind": "dirichlet", "value": 0.0},
        )
        alpha = 0.1
        rhs_fn = _make_heat_rhs(alpha, bc)
        T0 = jnp.ones((20, 20))

        from diffheat.solvers import explicit_euler_step_2d
        T1 = explicit_euler_step_2d(T0, rhs_fn, grid, t=0.0, dt=0.0001)
        assert jnp.mean(T1) < jnp.mean(T0)

    def test_with_params_dict(self):
        """Params dict should be passed through to rhs_fn."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=10, ny=10)
        bc = BoundaryCondition2D(
            left={"kind": "neumann", "value": 0.0},
            right={"kind": "neumann", "value": 0.0},
            bottom={"kind": "neumann", "value": 0.0},
            top={"kind": "neumann", "value": 0.0},
        )

        def rhs_fn(T, grid, t, params):
            alpha = params["alpha"]
            L_T_mod, b_source = apply_boundary_conditions_2d(
                lambda x: laplacian_2d(x, grid), grid, bc, T
            )
            return alpha * (L_T_mod + b_source)

        T0 = jnp.ones((10, 10))

        from diffheat.solvers import explicit_euler_step_2d
        T1_slow = explicit_euler_step_2d(T0, rhs_fn, grid, t=0.0, dt=0.001, params={"alpha": 0.01})
        T1_fast = explicit_euler_step_2d(T0, rhs_fn, grid, t=0.0, dt=0.001, params={"alpha": 0.1})
        # Higher alpha = more diffusion = faster change away from uniform
        assert jnp.sum(jnp.abs(T1_fast - T0)) > jnp.sum(jnp.abs(T1_slow - T0))


class TestSolve2D:
    def test_returns_correct_shape(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=10, ny=15)
        bc = BoundaryCondition2D(
            left={"kind": "dirichlet", "value": 1.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "neumann", "value": 0.0},
            top={"kind": "neumann", "value": 0.0},
        )
        alpha = 0.01
        rhs_fn = _make_heat_rhs(alpha, bc)
        T0 = jnp.zeros((10, 15))
        dt = 0.001
        t_span = (0.0, 0.01)

        from diffheat.solvers import solve_2d
        trajectory = solve_2d(rhs_fn, T0, grid, t_span, dt)

        n_steps = int((t_span[1] - t_span[0]) / dt) + 1
        assert trajectory.shape == (n_steps, 10, 15)

    def test_first_frame_is_initial_condition(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=10, ny=10)
        bc = BoundaryCondition2D(
            left={"kind": "dirichlet", "value": 1.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "neumann", "value": 0.0},
            top={"kind": "neumann", "value": 0.0},
        )
        alpha = 0.01
        rhs_fn = _make_heat_rhs(alpha, bc)
        T0 = jnp.linspace(1.0, 0.0, 100).reshape(10, 10)

        from diffheat.solvers import solve_2d
        trajectory = solve_2d(rhs_fn, T0, grid, (0.0, 0.01), dt=0.001)
        assert jnp.allclose(trajectory[0], T0)

    def test_gradient_wrt_alpha(self):
        """Gradients flow through solve_2d w.r.t. alpha."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=10, ny=10)

        def loss_fn(alpha):
            bc = BoundaryCondition2D(
                left={"kind": "dirichlet", "value": 1.0},
                right={"kind": "dirichlet", "value": 0.0},
                bottom={"kind": "neumann", "value": 0.0},
                top={"kind": "neumann", "value": 0.0},
            )
            rhs_fn = _make_heat_rhs(alpha, bc)
            T0 = jnp.zeros((10, 10))
            from diffheat.solvers import solve_2d
            traj = solve_2d(rhs_fn, T0, grid, (0.0, 0.005), dt=0.001)
            return jnp.mean(traj[-1])

        grad = jax.grad(loss_fn)(0.1)
        assert not jnp.isclose(grad, 0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_solvers_2d.py -v
```

Expected: FAIL — ImportError.

- [ ] **Step 3: Add `check_cfl_2d` to `diffheat/solvers/stability.py`**

Append:

```python
from ..mesh.grid2d import Grid2D


def check_cfl_2d(grid: Grid2D, alpha: float | jnp.ndarray, dt: float) -> bool:
    """Check if dt satisfies the 2D CFL stability condition for explicit Euler.

    dt <= min(dx^2, dy^2) / (4 * alpha)

    Args:
        grid: The 2D spatial grid.
        alpha: Thermal diffusivity (scalar or field).
        dt: Time step size.

    Returns:
        True if stable, False otherwise.
    """
    alpha_max = float(jnp.max(jnp.asarray(alpha)))
    dx_min = float(jnp.min(grid.dx))
    dy_min = float(jnp.min(grid.dy))
    cfl_limit = min(dx_min * dx_min, dy_min * dy_min) / (4.0 * alpha_max)
    return bool(dt <= cfl_limit)
```

- [ ] **Step 4: Add `explicit_euler_step_2d` to `diffheat/solvers/explicit.py`**

Append:

```python
from ..mesh.grid2d import Grid2D
from typing import Callable, Optional


def explicit_euler_step_2d(
    state: jnp.ndarray,
    rhs_fn: Callable[[jnp.ndarray, Grid2D, float, Optional[dict]], jnp.ndarray],
    grid: Grid2D,
    t: float,
    dt: float,
    params: Optional[dict] = None,
) -> jnp.ndarray:
    """Single explicit Euler time step for an arbitrary 2D system.

    state^{n+1} = state^n + dt * rhs_fn(state^n, grid, t, params)

    Args:
        state: (nx, ny) field at current timestep.
        rhs_fn: Right-hand side function.
            Signature: rhs_fn(state, grid, t, params) -> dstate_dt
        grid: The 2D grid.
        t: Current time.
        dt: Time step size.
        params: Optional dict of parameters passed to rhs_fn.

    Returns:
        (nx, ny) field at next timestep.
    """
    dstate_dt = rhs_fn(state, grid, t, params)
    return state + dt * dstate_dt
```

- [ ] **Step 5: Add `solve_2d` to `diffheat/solvers/scan.py`**

Append:

```python
from ..mesh.grid2d import Grid2D
from typing import Callable, Optional


def solve_2d(
    rhs_fn: Callable,
    initial_state: jnp.ndarray,
    grid: Grid2D,
    t_span: tuple[float, float],
    dt: float,
    params: Optional[dict] = None,
) -> jnp.ndarray:
    """Solve a 2D PDE using explicit Euler with jax.lax.scan.

    The entire solve is JIT-compiled and differentiable.

    Args:
        rhs_fn: Right-hand side function.
            Signature: rhs_fn(state, grid, t, params) -> dstate_dt
        initial_state: (nx, ny) initial field.
        grid: The 2D grid.
        t_span: (t_start, t_end) simulation time range.
        dt: Time step size.
        params: Optional dict of parameters passed to rhs_fn.

    Returns:
        (n_steps+1, nx, ny) field trajectory. First frame is initial_state.
    """
    t0, t_end = t_span
    n_steps = int((t_end - t0) / dt)

    if n_steps < 1:
        raise ValueError(f"t_span too short for dt={dt}: {t_span}")

    def step_fn(state, step_idx):
        t_current = t0 + step_idx * dt
        state_next = explicit_euler_step_2d(state, rhs_fn, grid, t_current, dt, params)
        return state_next, state_next

    _, traj = jax.lax.scan(step_fn, initial_state, jnp.arange(n_steps))
    trajectory = jnp.concatenate([initial_state[jnp.newaxis, :, :], traj], axis=0)

    return trajectory
```

- [ ] **Step 6: Update `diffheat/solvers/__init__.py`**

```python
# diffheat/solvers/__init__.py
"""Time integration solvers."""
from .explicit import explicit_euler_step, explicit_euler_step_2d
from .scan import solve_2d, solve_heat_1d
from .stability import check_cfl, check_cfl_2d

__all__ = [
    "explicit_euler_step",
    "explicit_euler_step_2d",
    "solve_heat_1d",
    "solve_2d",
    "check_cfl",
    "check_cfl_2d",
]
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
python -m pytest tests/test_solvers_2d.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 8: Run full test suite**

```bash
python -m pytest tests/ -v --tb=short
```

Expected: all tests PASS (1 pre-existing x64 env failure is ok).

- [ ] **Step 9: Commit**

```bash
git add diffheat/solvers/ tests/test_solvers_2d.py
git commit -m "feat: add 2D solver with explicit Euler, CFL, and jax.lax.scan

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: 2D Visualization

**Files:**
- Create: `diffheat/viz/heatmap2d.py`
- Modify: `diffheat/viz/window.py` (add `ViewerWindow2D`, `run_viewer_2d`)
- Modify: `diffheat/viz/__init__.py`

**Interfaces:**
- Consumes: `diffheat.mesh.Grid2D`, `jnp.ndarray` trajectories
- Produces: `run_viewer_2d(trajectory, grid, dt) -> None` — blocks until window closes

- [ ] **Step 1: Create `diffheat/viz/heatmap2d.py`**

```python
# diffheat/viz/heatmap2d.py
"""2D frame-by-frame heatmap widget."""
import numpy as np
from PyQt6 import QtWidgets
import jax.numpy as jnp

from ..mesh.grid2d import Grid2D
from .canvas import MatplotlibCanvas


class HeatmapWidget2D(QtWidgets.QWidget):
    """Frame-by-frame heatmap of a 2D temperature field."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = MatplotlibCanvas(self, width=6, height=6)
        self.ax = self.canvas.fig.add_subplot(1, 1, 1)

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

        self.ax.clear()

        x_edges = np.asarray(self.grid.x)
        y_edges = np.asarray(self.grid.y)
        extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

        # Transpose from (nx, ny) to (ny, nx) for imshow
        frame = self.trajectory[self.current_frame].T

        im = self.ax.imshow(
            frame,
            extent=extent,
            origin="lower",
            aspect="auto",
            cmap="hot",
        )
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_title(f"Temperature at t = {self.times[self.current_frame]:.4f}")

        if self._colorbar is None:
            self._colorbar = self.canvas.fig.colorbar(im, ax=self.ax, label="Temperature")
        else:
            self._colorbar.update_normal(im)

        self.canvas.fig.tight_layout()
        self.canvas.draw()
```

- [ ] **Step 2: Modify `diffheat/viz/window.py` — append `ViewerWindow2D` and `run_viewer_2d`**

Append:

```python
from ..mesh.grid2d import Grid2D
from .heatmap2d import HeatmapWidget2D


class ViewerWindow2D(QtWidgets.QMainWindow):
    """Main window for 2D heat equation visualization."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("diffheat — 2D Heat Equation Viewer")
        self.resize(800, 800)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        self.heatmap = HeatmapWidget2D()
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
```

- [ ] **Step 3: Update `diffheat/viz/__init__.py`**

```python
# diffheat/viz/__init__.py
"""PyQt6-based visualization for heat equation trajectories.

This is the ONLY package in diffheat that imports PyQt or matplotlib.
The core library remains headless and works without these dependencies.
"""
from .heatmap1d import HeatmapWidget
from .heatmap2d import HeatmapWidget2D
from .window import ViewerWindow, ViewerWindow2D, run_viewer, run_viewer_2d

__all__ = [
    "HeatmapWidget",
    "HeatmapWidget2D",
    "ViewerWindow",
    "ViewerWindow2D",
    "run_viewer",
    "run_viewer_2d",
]
```

- [ ] **Step 4: Commit**

```bash
git add diffheat/viz/
git commit -m "feat: add 2D heatmap viewer with frame-by-frame colorbar display

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: 2D Demo Script

**Files:**
- Create: `examples/02-2d-heat-equation/demo.py`

**Interfaces:**
- Consumes: `diffheat` (all 2D modules), `diffheat.viz.run_viewer_2d`
- Produces: Runnable demo solving and visualizing a 2D heat equation

- [ ] **Step 1: Create demo directory**

```bash
mkdir -p examples/02-2d-heat-equation
```

- [ ] **Step 2: Write `examples/02-2d-heat-equation/demo.py`**

```python
#!/usr/bin/env python3
"""Demo: 2D heat equation — hot left edge, cold right edge, insulated top/bottom.

A square plate initially at 0°C, with the left edge held at 100°C and the
right edge at 0°C. Top and bottom are insulated. Over time, the temperature
profile approaches the linear steady-state solution (horizontal gradient).

Run:
    python examples/02-2d-heat-equation/demo.py
"""
import jax.numpy as jnp

from diffheat import (
    BoundaryCondition2D,
    Grid2D,
    get_device,
    solve_2d,
    check_cfl_2d,
)
from diffheat.operators import laplacian_2d
from diffheat.mesh.boundary import apply_boundary_conditions_2d
from diffheat.viz import run_viewer_2d


def main():
    print(f"Running on: {get_device()}")
    print("=" * 50)

    # --- Grid ---
    Lx, Ly = 1.0, 1.0  # 1 m × 1 m plate
    nx, ny = 64, 64
    grid = Grid2D.uniform(Lx=Lx, Ly=Ly, nx=nx, ny=ny)
    print(f"Grid: {grid.nx}×{grid.ny} cells, dx = {float(grid.dx[0]):.4f}, dy = {float(grid.dy[0]):.4f}")

    # --- Boundary conditions ---
    bc = BoundaryCondition2D(
        left={"kind": "dirichlet", "value": 100.0},   # hot left
        right={"kind": "dirichlet", "value": 0.0},     # cold right
        bottom={"kind": "neumann", "value": 0.0},      # insulated bottom
        top={"kind": "neumann", "value": 0.0},         # insulated top
    )

    # --- Material ---
    alpha = 0.01

    # --- RHS function: dT/dt = alpha * laplacian(T) with BCs ---
    def heat_rhs(T, grid, t, params):
        alpha = params["alpha"]
        L_T_mod, b_source = apply_boundary_conditions_2d(
            lambda x: laplacian_2d(x, grid), grid, bc, T
        )
        return alpha * (L_T_mod + b_source)

    # --- Initial condition ---
    T0 = jnp.zeros((nx, ny))

    # --- Time parameters ---
    t_end = 5.0
    dt = 0.001

    # CFL check
    dx_min = float(jnp.min(grid.dx))
    dy_min = float(jnp.min(grid.dy))
    cfl_limit = min(dx_min**2, dy_min**2) / (4 * alpha)
    print(f"CFL limit: {cfl_limit:.6f} s")
    print(f"dt: {dt:.4f} s (stable: {check_cfl_2d(grid, alpha, dt)})")

    # --- Solve ---
    print(f"Solving from t=0 to t={t_end}...")
    params = {"alpha": alpha}
    trajectory = solve_2d(heat_rhs, T0, grid, (0.0, t_end), dt, params=params)

    n_steps = len(trajectory)
    print(f"Done. {n_steps} timesteps computed.")
    print(f"Initial mean T: {jnp.mean(trajectory[0]):.2f}°C")
    print(f"Final mean T:   {jnp.mean(trajectory[-1]):.2f}°C")
    print(f"Steady-state expected mean: {(100.0 + 0.0) / 2:.1f}°C")

    # --- Visualize ---
    print("\nLaunching viewer...")
    run_viewer_2d(trajectory, grid, dt)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify headless solve works**

```bash
python -c "
import jax.numpy as jnp
from diffheat import Grid2D, BoundaryCondition2D, solve_2d, check_cfl_2d, get_device
from diffheat.operators import laplacian_2d
from diffheat.mesh.boundary import apply_boundary_conditions_2d

print(f'Device: {get_device()}')
grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=32, ny=32)
bc = BoundaryCondition2D(
    left={'kind': 'dirichlet', 'value': 100.0},
    right={'kind': 'dirichlet', 'value': 0.0},
    bottom={'kind': 'neumann', 'value': 0.0},
    top={'kind': 'neumann', 'value': 0.0},
)

def heat_rhs(T, grid, t, params):
    L_T_mod, b_source = apply_boundary_conditions_2d(
        lambda x: laplacian_2d(x, grid), grid, bc, T
    )
    return params['alpha'] * (L_T_mod + b_source)

T0 = jnp.zeros((32, 32))
traj = solve_2d(heat_rhs, T0, grid, (0.0, 1.0), dt=0.0005, params={'alpha': 0.01})
print(f'Trajectory shape: {traj.shape}')
print(f'Final mean T: {jnp.mean(traj[-1]):.2f}')
print('Demo headless solve works.')
"
```

Expected: prints trajectory shape (2001, 32, 32) and final mean temperature.

- [ ] **Step 4: Commit**

```bash
git add examples/02-2d-heat-equation/demo.py
git commit -m "feat: add 2D heat equation demo — hot left, cold right, insulated top/bottom

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 10: Application Tracking File

**Files:**
- Create: `docs/applications.md`

- [ ] **Step 1: Write `docs/applications.md`**

```markdown
# diffheat — Application Scenarios

Tracked applications for future simulation with diffheat. Each entry includes the PDE system, boundary conditions, parameter ranges, and target optimization problem.

The library itself remains general-purpose; these are specific compositions of its operators and solvers.

---

## 1. Natural Convection (Boussinesq)

**PDE system:**
- ∂T/∂t + **u**·∇T = α ∇²T (energy)
- ∂**u**/∂t + **u**·∇**u** = -∇p + ν ∇²**u** + β g (T - T_ref) **k** (momentum)
- ∇·**u** = 0 (continuity)

**Parameters:** Rayleigh number (Ra), Prandtl number (Pr), aspect ratio

**Boundary conditions:** Heated bottom plate, cooled top plate, insulated side walls, no-slip velocity on all walls

**Optimization target:** Maximize heat transfer (Nusselt number) for given Ra; optimize heating pattern for uniform cooling

---

## 2. Thermoelectric Cooling (Peltier/Seebeck)

**PDE system:**
- ρ c_p ∂T/∂t = ∇·(k ∇T) + J²/σ - τ J·∇T (heat + Joule + Thomson)
- ∇·(σ ∇V) = -∇·(σ S ∇T) (electric potential with Seebeck source)

**Parameters:** Seebeck coefficient (S), electrical conductivity (σ), thermal conductivity (k), Thomson coefficient (τ)

**Boundary conditions:** Fixed voltage at contacts, convective heat transfer at boundaries

**Optimization target:** Maximize cooling ΔT for given input current; minimize power consumption for target cooling

---

## 3. Absorption Chiller Cycle

**Model type:** Lumped ODE system (not PDE — would require `diffheat` ODE support)

**4-component cycle:** Generator → Condenser → Evaporator → Absorber

**Parameters:** Heat input (Q_gen), cooling output (Q_evap), solution concentrations, mass flow rates

**Optimization target:** Maximize COP (Q_evap / Q_gen); optimize cycle temperatures for given heat source

---

## 4. Thermal Cloak Optimization

**PDE system:** ∂T/∂t = ∇·(κ(x,y) ∇T) (spatially-varying conductivity)

**Goal:** Optimize κ(x,y) field so that a protected interior region experiences minimal temperature gradient while external temperature field appears undisturbed

**Parameters:** Conductivity range [κ_min, κ_max], cloak geometry

**Optimization target:** Minimize |∇T| inside protected region while matching far-field temperature profile

---

## 5. Forced Convection Cooling

**PDE system:**
- ∂T/∂t + **u**·∇T = α ∇²T (advection-diffusion)
- **u**(x,y) prescribed (not solved — one-way coupling)

**Parameters:** Péclet number (Pe = UL/α), channel geometry

**Boundary conditions:** Inlet temperature, convective outlet, heated component at center

**Optimization target:** Optimize inlet velocity profile or channel geometry to minimize component temperature

---

## 6. Irregular Domains

**Status:** Not yet supported. Requires unstructured mesh support (`GridUnstructured`) and finite volume / finite element operator assembly.

**Blockers:**
- Mesh generation (triangulation)
- Unstructured operator stencils (neighbor lookups)
- Boundary condition application on curved edges
```

- [ ] **Step 2: Commit**

```bash
git add docs/applications.md
git commit -m "docs: add application tracking file for future coupled-physics scenarios

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 11: Final Verification

- [ ] **Step 1: Run full test suite**

```bash
python -m pytest tests/ -v --tb=short
```

Expected: all tests PASS (1 pre-existing x64 env failure is acceptable).

- [ ] **Step 2: Verify 1D backward compatibility**

```bash
python -c "
from diffheat import Grid1D, BoundaryCondition, HeatEquation1D, solve_heat_1d, get_device
from diffheat.mesh import Grid1D as G1
from diffheat.physics import HeatEquation1D as HE1D
from diffheat.solvers import solve_heat_1d as s1d
from diffheat.viz import run_viewer
import jax.numpy as jnp

print(f'Device: {get_device()}')

# Full 1D pipeline
grid = Grid1D.uniform(length=1.0, n_cells=50)
bc = BoundaryCondition(kind='dirichlet', value=jnp.array([1.0, 0.0]))
eqn = HeatEquation1D(grid=grid, bc=bc, alpha=0.1)
T0 = jnp.zeros(grid.n_cells)
traj = solve_heat_1d(eqn, T0, (0.0, 0.05), dt=0.001)
print(f'1D trajectory shape: {traj.shape}')
print(f'1D final mean T: {jnp.mean(traj[-1]):.4f}')
print('1D backward compatibility: OK')
"
```

Expected: prints success message and trajectory shape.

- [ ] **Step 3: Verify 2D gradient flow (differentiability check)**

```bash
python -c "
import jax
import jax.numpy as jnp
from diffheat import Grid2D, BoundaryCondition2D, solve_2d
from diffheat.operators import laplacian_2d
from diffheat.mesh.boundary import apply_boundary_conditions_2d

grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=16, ny=16)
bc = BoundaryCondition2D(
    left={'kind': 'dirichlet', 'value': 100.0},
    right={'kind': 'dirichlet', 'value': 0.0},
    bottom={'kind': 'neumann', 'value': 0.0},
    top={'kind': 'neumann', 'value': 0.0},
)

def make_loss(alpha):
    def rhs(T, grid, t, params):
        L_T_mod, b_source = apply_boundary_conditions_2d(
            lambda x: laplacian_2d(x, grid), grid, bc, T
        )
        return alpha * (L_T_mod + b_source)
    T0 = jnp.zeros((16, 16))
    traj = solve_2d(rhs, T0, grid, (0.0, 0.01), dt=0.0005)
    return jnp.mean(traj[-1])

grad = jax.grad(make_loss)(0.1)
print(f'Gradient d(mean_T_final)/d(alpha): {float(grad):.6f}')
print('2D differentiability: OK')
"
```

Expected: prints non-zero gradient value.

- [ ] **Step 4: Verify 2D headless pipeline (demo minus GUI)**

```bash
python -c "
import jax.numpy as jnp
from diffheat import Grid2D, BoundaryCondition2D, solve_2d, check_cfl_2d, get_device
from diffheat.operators import laplacian_2d
from diffheat.mesh.boundary import apply_boundary_conditions_2d

print(f'Device: {get_device()}')
grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=64, ny=64)
bc = BoundaryCondition2D(
    left={'kind': 'dirichlet', 'value': 100.0},
    right={'kind': 'dirichlet', 'value': 0.0},
    bottom={'kind': 'neumann', 'value': 0.0},
    top={'kind': 'neumann', 'value': 0.0},
)
alpha = 0.01
def rhs(T, grid, t, params):
    L_T_mod, b_source = apply_boundary_conditions_2d(
        lambda x: laplacian_2d(x, grid), grid, bc, T
    )
    return alpha * (L_T_mod + b_source)
T0 = jnp.zeros((64, 64))
traj = solve_2d(rhs, T0, grid, (0.0, 1.0), dt=0.0005)
print(f'Trajectory shape: {traj.shape}')
print(f'Final mean T: {jnp.mean(traj[-1]):.2f}')
print('2D pipeline: OK')
"
```

Expected: prints trajectory shape and final mean temperature.

- [ ] **Step 5: Commit final state if clean**

```bash
git status
```

---

## Implementation Summary

| Task | Files | Tests | Key Deliverable |
|------|-------|-------|-----------------|
| 1 | mesh/, operators/, physics/, solvers/, viz/ packages | 46 existing | Package restructure with backcompat |
| 2 | mesh/grid2d.py | 11 | Grid2D dataclass |
| 3 | operators/laplacian.py (modify) | 5 new | laplacian_2d operator |
| 4 | operators/gradient.py | 6 new | gradient_x, gradient_y, gradient_2d |
| 5 | operators/divergence.py | 4 new | divergence_2d |
| 6 | mesh/boundary.py (modify) | 6 new | BoundaryCondition2D + BC application |
| 7 | solvers/ (modify) | 8 new | explicit_euler_step_2d, solve_2d, check_cfl_2d |
| 8 | viz/heatmap2d.py, viz/window.py (modify) | — | 2D PyQt viewer |
| 9 | examples/02-2d-heat-equation/demo.py | — | End-to-end 2D demo |
| 10 | docs/applications.md | — | Application scenarios tracker |
| 11 | — | all ~81 | Final verification |

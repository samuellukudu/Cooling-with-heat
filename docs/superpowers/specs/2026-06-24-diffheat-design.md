# diffheat — Design Specification

**Date:** 2026-06-24  
**Status:** Approved  
**Goal:** A minimal JAX-based differentiable heat equation library with PyQt visualization, built from first principles. Start with the solver infrastructure; layer in physics optimization problems later.

---

## 1. Philosophy

- **First principles, not rushing.** Build the right abstractions before optimizing for features.
- **Infrastructure before application.** v0.1 delivers the differentiable solver + visualization harness. Specific physics optimization problems (thermal cloak, hot-driven cold spot, etc.) come after the core works.
- **Library vs. examples separation.** `diffheat/` is the installable package. `examples/` are standalone scripts that import it. Never mix them.
- **Headless core.** Physics and solvers are pure JAX — no GUI dependencies. `viz` is the only module that touches PyQt. The library must work from notebooks, scripts, and CI without a display.
- **GPU-aware from day one.** Device detection at import. All array creation routed through a single utility that respects the active backend.

---

## 2. Project Layout

```
Cooling-with-heat/
├── diffheat/                  # The library (pip-installable)
│   ├── __init__.py            # Public API surface, device detection
│   ├── mesh.py                # Grid1D, BoundaryCondition
│   ├── physics.py             # HeatEquation1D, make_laplacian
│   ├── solvers.py             # solve_heat_1d, steppers
│   ├── viz.py                 # PyQt viewer (HeatmapWidget, ViewerWindow)
│   └── utils.py               # Device helpers, array utilities
├── examples/                  # Case studies (NOT a package — no __init__.py)
│   └── 01-1d-heat-equation/
│       ├── demo.py            # Simplest forward solve + visualization
│       └── explore.ipynb      # Interactive exploration notebook
├── docs/
│   └── superpowers/
│       └── specs/             # Design documents
├── pyproject.toml             # uv/pip project config, dependencies
├── README.md
└── .gitignore
```

---

## 3. Package Architecture

### 3.1 Module Responsibilities

| Module    | Owns                                                    | Exports                              |
|-----------|---------------------------------------------------------|--------------------------------------|
| `mesh`    | 1D grid geometry, cell spacing, boundary masks          | `Grid1D`, `BoundaryCondition`        |
| `physics` | Heat equation operator (Laplacian), source terms, BCs   | `HeatEquation1D`, `make_laplacian`   |
| `solvers` | Time integration via `jax.lax.scan`, step functions     | `solve_heat_1d`, `Stepper`           |
| `viz`     | PyQt window, heatmap rendering, animation controls      | `run_viewer`, `ViewerWindow`         |
| `utils`   | Device detection, array creation, dtype management      | `get_device`, `array`, `get_default_dtype` |

**Rule:** No module imports PyQt except `viz`. Physics + solvers are pure JAX.

### 3.2 Dependency Direction

```
viz ──► solvers ──► physics ──► mesh ──► utils
  │        │           │          │
  └────────┴───────────┴──────────┘
                  │
              PyQt (optional extra)
```

No circular imports. Each module only depends on modules to its right.

---

## 4. Data Model

### 4.1 Grid1D

```python
@dataclass(frozen=True)
class Grid1D:
    x: jnp.ndarray        # (N+1,) cell interfaces [x0, x1, ..., xN]
    centers: jnp.ndarray  # (N,)   cell centers
    dx: jnp.ndarray       # (N,)   cell widths
    length: float         # total domain length
    n_cells: int          # N
```

Frozen dataclass — safe for JAX tracing. `x` stores interfaces for boundary placement; `centers` is where temperature degrees of freedom live. Factory method accepts `(length, n_cells, uniform=True)` for the common case, or a custom spacing array.

### 4.2 BoundaryCondition

```python
@dataclass(frozen=True)
class BoundaryCondition:
    kind: str             # "dirichlet", "neumann", "robin"
    value: jnp.ndarray    # (2,) left/right — can be JAX tracers
```

Both ends independently specifiable. `kind` can be a single string applied to both ends, or a tuple `("dirichlet", "neumann")`. The `value` array stores left and right boundary parameters and is differentiable — gradients flow through boundary temperatures.

### 4.3 HeatEquation1D

```python
@dataclass(frozen=True)
class HeatEquation1D:
    grid: Grid1D
    bc: BoundaryCondition
    alpha: jnp.ndarray    # (N,) or scalar — thermal diffusivity field
    source: Callable[[jnp.ndarray, float], jnp.ndarray] | None  # S(x, t)
```

Encapsulates the full problem definition. The Laplacian operator is constructed lazily (cached after first JIT compilation).

---

## 5. Solver Design

### 5.1 PDE Discretization

The 1D heat equation:

**∂T/∂t = α ∂²T/∂x² + S(x,t)**

Discretized with centered finite differences on the cell-centered grid:

**(∂²T/∂x²)ᵢ ≈ (Tᵢ₊₁ - 2Tᵢ + Tᵢ₋₁) / Δx²**

Assembled as a tridiagonal matrix-vector product (efficient, JIT-friendly).

### 5.2 Explicit Euler Stepper

```python
def explicit_euler_step(T: jnp.ndarray, eqn: HeatEquation1D, dt: float) -> jnp.ndarray:
    laplacian_T = make_laplacian(eqn.grid) @ T
    dT_dt = eqn.alpha * laplacian_T
    if eqn.source is not None:
        dT_dt += eqn.source(eqn.grid.centers, t)
    return T + dt * dT_dt
```

### 5.3 Scan-Based Time Loop

```python
def solve_heat_1d(
    eqn: HeatEquation1D,
    T0: jnp.ndarray,           # (N,) initial temperature
    t_span: tuple[float, float],
    dt: float,
) -> Trajectory:               # (n_steps+1, N) + metadata
```

Uses `jax.lax.scan` over the step function. The entire solve is a single JIT-table computation. Gradients flow through the full trajectory — you can compute ∂T_final/∂α, ∂T_final/∂bc_value, or ∂loss/∂T0.

**Stability:** CFL condition for explicit Euler is dt ≤ Δx²/(2α). The solver warns (but doesn't error) if the user's dt exceeds this.

### 5.4 Future Solver Extensions (not v0.1)
- Crank-Nicolson (implicit, tridiagonal solve per step)
- Adaptive timestepping
- 2D/3D operator assembly

---

## 6. Device & Precision Management

### 6.1 Device Detection (at import)

```python
# diffheat/utils.py
def get_device() -> str:
    """Return 'gpu', 'tpu', or 'cpu'."""
    return jax.default_backend()

def get_default_dtype() -> jnp.dtype:
    """float32 on GPU/TPU, float64 on CPU."""
    if get_device() == "cpu":
        return jnp.float64
    return jnp.float32
```

### 6.2 Centralized Array Creation

```python
def array(data, dtype=None) -> jnp.ndarray:
    """Create JAX array on default device with default dtype."""
    if dtype is None:
        dtype = get_default_dtype()
    return jnp.array(data, dtype=dtype)
```

Every array in the library goes through `utils.array`. No module calls `jnp.array` or `jax.device_put` directly. This centralizes device placement and makes mixed-precision or multi-device support straightforward to add later.

### 6.3 Import-Time Notification

```python
# diffheat/__init__.py
import logging
from .utils import get_device

_logger = logging.getLogger(__name__)
_logger.info(f"diffheat running on: {get_device()}")
```

Users see immediately what hardware is active. The solver also warns for mesh sizes that are too small to benefit from GPU acceleration.

---

## 7. Visualization (PyQt6)

### 7.1 Architecture

```
viz.py
├── MatplotlibCanvas        # FigureCanvasQTAgg — embeds matplotlib in Qt
├── HeatmapWidget           # Space-time heatmap of the temperature field
├── SnapshotPlot            # Line plot of temperature at current timestep
├── ControlPanel            # Play/pause, time slider, parameter knobs
└── ViewerWindow            # QMainWindow composing all widgets
```

### 7.2 v0.1 Views

- **Space-time heatmap:** x-axis = position, y-axis = time (scrolling down), color = temperature. Boundary values shown as colored edge markers.
- **Snapshot:** temperature profile T(x) at the current frame, overlaid as a line plot.
- **Controls:** play/pause toggle, step forward/back, reset to t=0, dt slider, alpha slider, current time display.

### 7.3 Integration

```python
from diffheat.viz import run_viewer

trajectory = solve_heat_1d(eqn, T0, t_span, dt)
run_viewer(trajectory, eqn.grid)  # blocks until window closes
```

`run_viewer` is the only entry point. It starts the Qt event loop and blocks. The viewer owns a copy of the trajectory data (moved to CPU via `numpy.array`), so the Qt thread never touches JAX arrays directly.

### 7.4 Future Visualization (not v0.1)
- Gradient overlay (∂L/∂T₀, ∂L/∂α rendered alongside temperature)
- Real-time parameter tweaking with live re-solve
- 2D/3D OpenGL rendering when we go to higher dimensions

---

## 8. Dependencies

### 8.1 pyproject.toml

```toml
[project]
name = "diffheat"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "jax",
    "numpy",
]

[project.optional-dependencies]
viz = ["pyqt6", "matplotlib"]
dev = ["pytest", "jupyter"]

[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"
```

### 8.2 uv Workflow

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[viz,dev]"
python examples/01-1d-heat-equation/demo.py
```

---

## 9. v0.1 Scope (What We're Building Now)

### In scope
- [x] `Grid1D` with uniform spacing
- [x] `BoundaryCondition` (Dirichlet, Neumann)
- [x] `HeatEquation1D` with constant alpha and optional source term
- [x] Explicit Euler solver with `jax.lax.scan`
- [x] GPU-aware device detection and array creation
- [x] PyQt6 space-time heatmap viewer with play/pause and slider controls
- [x] Example script: 1D rod with hot left end, cold right end — reaching steady state
- [x] Jupyter notebook for interactive exploration

### Out of scope (future)
- Non-uniform meshes
- Implicit solvers (Crank-Nicolson)
- 2D/3D meshes
- Boundary condition optimization problems
- Material property optimization
- Training data generation pipeline
- Multi-GPU support

---

## 10. Future Avenues (Tracked)

These are ideas generated during design that we want to explore after v0.1:

1. **Thermal cloak optimization** — optimize conductivity field to route heat around a protected region
2. **Hot-driven cold spot** — optimize boundary control to create a region colder than ambient
3. **Heat-to-work proxy** — maximize temperature drop across a material junction
4. **Absorption chiller (lumped ODE)** — 4-component thermodynamic cycle
5. **Thermoelectric (Peltier) PDE** — coupled heat + charge transport
6. **Training data generation** — parameter sweeps + neural surrogate models
7. **Material metamaterial design** — inverse design of thermal conductivity fields
8. **Adaptive mesh refinement** — non-uniform grids for sharp gradients

---

## 11. Implementation Order

1. `utils.py` — device detection, array helpers
2. `mesh.py` — Grid1D, BoundaryCondition
3. `physics.py` — Laplacian operator, HeatEquation1D
4. `solvers.py` — explicit Euler step, scan loop
5. `__init__.py` — public API
6. `viz.py` — PyQt viewer
7. `examples/01-1d-heat-equation/demo.py` — end-to-end demo
8. `pyproject.toml` — package configuration

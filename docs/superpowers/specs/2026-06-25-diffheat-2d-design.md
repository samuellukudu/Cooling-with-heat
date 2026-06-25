# diffheat 2D — Design Specification

**Date:** 2026-06-25
**Status:** In Review
**Goal:** Extend diffheat to 2D rectangular domains with a general-purpose operator library, generic solvers, and per-edge boundary conditions — enabling coupled multi-physics simulations while keeping 1D code untouched.

---

## 1. Philosophy

- **Library vs. applications.** The library provides operators, grids, boundary conditions, and solvers. Specific physics (Boussinesq, thermoelectric, absorption chiller) are *applications* built on top, tracked separately.
- **Operators are building blocks.** `laplacian_2d`, `gradient_x`, `gradient_y`, `divergence_2d` are pure JAX functions with no physics assumptions. Users compose them into arbitrary coupled PDE systems.
- **1D code is frozen.** `Grid1D`, `HeatEquation1D`, `solve_heat_1d` move into subpackages but their behavior and API are preserved.
- **Headless core.** All operator and solver code is pure JAX. Only `viz/` touches PyQt/matplotlib.
- **Differentiability is non-negotiable.** Every operator and solver must be end-to-end differentiable via JAX.

---

## 2. Package Restructure

Flat files → subpackages for scalability. The 1D code moves in but preserves API.

```
diffheat/
├── __init__.py              # Public API (1D + 2D exports)
├── utils.py                 # Device detection, array helpers (unchanged)
├── mesh/                    # Grids & boundary conditions
│   ├── __init__.py
│   ├── grid1d.py            # Grid1D (moved from mesh.py)
│   ├── grid2d.py            # Grid2D (new)
│   └── boundary.py          # BoundaryCondition (moved), BoundaryCondition2D (new)
├── operators/               # Discrete differential operators (NEW, pure JAX)
│   ├── __init__.py
│   ├── laplacian.py         # make_laplacian (moved), laplacian_2d (new)
│   ├── gradient.py          # gradient_x, gradient_y, gradient_2d (new)
│   └── divergence.py        # divergence_2d (new)
├── physics/                 # Problem definitions
│   ├── __init__.py
│   └── heat1d.py            # HeatEquation1D, apply_boundary_conditions (moved)
├── solvers/                 # Time integration
│   ├── __init__.py
│   ├── explicit.py          # explicit_euler_step (moved), explicit_euler_step_2d (new)
│   ├── scan.py              # solve_heat_1d (moved), solve_2d (new)
│   └── stability.py         # check_cfl (moved), check_cfl_2d (new)
└── viz/                     # Visualization
    ├── __init__.py
    ├── canvas.py            # MatplotlibCanvas (extracted)
    ├── heatmap1d.py         # HeatmapWidget (extracted)
    ├── heatmap2d.py         # HeatmapWidget2D (new)
    ├── controls.py          # ControlPanel (extracted)
    └── window.py            # ViewerWindow, ViewerWindow2D, run_viewer (extracted + new)
```

**Dependency direction (unchanged from 1D):**

```
viz ──► solvers ──► physics ──► operators ──► mesh ──► utils
  │        │           │            │           │
  └────────┴───────────┴────────────┴───────────┘
                      │
                  PyQt (optional extra)
```

---

## 3. Grid2D

### 3.1 Data Model

```python
@dataclass(frozen=True)
class Grid2D:
    x: jnp.ndarray       # (nx+1,) cell interface positions in x
    y: jnp.ndarray       # (ny+1,) cell interface positions in y
    x_centers: jnp.ndarray  # (nx,) cell centers in x
    y_centers: jnp.ndarray  # (ny,) cell centers in y
    dx: jnp.ndarray      # (nx,) cell widths in x
    dy: jnp.ndarray      # (ny,) cell widths in y
    Lx: float
    Ly: float
    nx: int
    ny: int
    X: jnp.ndarray       # (ny, nx) 2D meshgrid of x-coordinates
    Y: jnp.ndarray       # (ny, nx) 2D meshgrid of y-coordinates

    @classmethod
    def uniform(cls, Lx: float, Ly: float, nx: int, ny: int) -> "Grid2D": ...
```

### 3.2 Design Decisions

- Frozen dataclass — safe for JAX tracing
- `X, Y` precomputed meshgrids for convenience in user source terms
- Field indexing: `T[i, j]` corresponds to cell `(x_centers[i], y_centers[j])`. The meshgrids `X[j, i]` and `Y[j, i]` follow matplotlib's y-first convention for `imshow` compatibility.
- Uniform spacing only for v0.2. Non-uniform (adaptive) grids deferred.
- Factory method only: `Grid2D.uniform(Lx, Ly, nx, ny)`. Direct constructor available but discouraged.

---

## 4. Operators

### 4.1 Design Rules

- Every operator is a **pure function**: `(field, grid) -> result`
- No mutation, no side effects
- Boundaries are NOT handled inside operators — that's `mesh/boundary.py`'s job
- All array creation through `utils.array()`
- All operators are JIT-compatible and differentiable

### 4.2 Operator Functions

```python
# operators/laplacian.py
def laplacian_2d(T: jnp.ndarray, grid: Grid2D) -> jnp.ndarray:
    """∇²T = ∂²T/∂x² + ∂²T/∂y² using 5-point centered stencil.
    T: (nx, ny). Returns (nx, ny)."""

# operators/gradient.py
def gradient_x(T: jnp.ndarray, grid: Grid2D) -> jnp.ndarray:
    """∂T/∂x via centered differences. T: (nx, ny). Returns (nx, ny)."""

def gradient_y(T: jnp.ndarray, grid: Grid2D) -> jnp.ndarray:
    """∂T/∂y via centered differences. T: (nx, ny). Returns (nx, ny)."""

def gradient_2d(T: jnp.ndarray, grid: Grid2D) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Returns (∂T/∂x, ∂T/∂y)."""

# operators/divergence.py
def divergence_2d(ux: jnp.ndarray, uy: jnp.ndarray, grid: Grid2D) -> jnp.ndarray:
    """∇·(ux, uy) = ∂ux/∂x + ∂uy/∂y. Returns (nx, ny)."""
```

### 4.3 Laplacian Stencil

Standard 5-point stencil for interior cells:

```
(∂²T/∂x² + ∂²T/∂y²)ᵢⱼ ≈
    (Tᵢ₊₁ⱼ - 2Tᵢⱼ + Tᵢ₋₁ⱼ) / Δx² +
    (Tᵢⱼ₊₁ - 2Tᵢⱼ + Tᵢⱼ₋₁) / Δy²
```

Implemented as vectorized operations on 2D arrays — no explicit loops.

---

## 5. Boundary Conditions (2D)

### 5.1 Data Model

```python
@dataclass(frozen=True)
class BoundaryCondition2D:
    left: dict     # {'kind': 'dirichlet'|'neumann', 'value': scalar or 1D array}
    right: dict
    bottom: dict
    top: dict
```

Each of the 4 edges gets an independently specifiable BC:
- `kind`: `"dirichlet"` (fixed value) or `"neumann"` (fixed gradient)
- `value`: scalar (constant along edge) or 1D array (varies along edge)
- Per-edge independence enables mixed BCs: Dirichlet on left/right for temperature, Neumann on top/bottom for insulated walls, etc.

### 5.2 Ghost-Cell Method

```python
def apply_boundary_conditions_2d(
    operator_fn: Callable,  # e.g. partial(laplacian_2d, grid=grid)
    grid: Grid2D,
    bc: BoundaryCondition2D,
    T: jnp.ndarray,  # (nx, ny)
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Returns (operator @ T modified, boundary_source).
    
    Modifies boundary rows/cols of the discretized operator and
    returns a 2D source array with non-zero entries only at boundary cells.
    """
```

Same ghost-cell logic as 1D, applied per-edge to the 2D array. Each edge modifies its boundary row/column in the operator and contributes to the boundary source term.

---

## 6. 2D Solver

### 6.1 Generic Time-Stepper

```python
def explicit_euler_step_2d(
    state: jnp.ndarray | tuple[jnp.ndarray, ...],
    rhs_fn: Callable,
    grid: Grid2D,
    t: float,
    dt: float,
    params: dict | None = None,
) -> jnp.ndarray | tuple[jnp.ndarray, ...]:
    """Single explicit Euler step for arbitrary 2D system.
    
    state: single field (nx, ny) or tuple of fields for coupled systems.
    rhs_fn: callable(state, grid, t, params) -> dstate_dt
    params: arbitrary JAX-traceable parameters (alpha, Ra, Pr, etc.)
    """
```

### 6.2 Scan-Based Solve

```python
def solve_2d(
    rhs_fn: Callable,
    initial_state: jnp.ndarray | tuple[jnp.ndarray, ...],
    grid: Grid2D,
    t_span: tuple[float, float],
    dt: float,
    params: dict | None = None,
) -> jnp.ndarray | tuple[jnp.ndarray, ...]:
    """Full trajectory via jax.lax.scan. Differentiable end-to-end.
    
    Returns: (n_steps+1, nx, ny) or tuple of such arrays for multi-field.
    """
```

### 6.3 CFL Condition

```python
def check_cfl_2d(grid: Grid2D, alpha: float, dt: float) -> bool:
    """2D CFL: dt ≤ min(Δx², Δy²) / (4 * alpha)"""
```

### 6.4 Design Notes

- `rhs_fn` is the key abstraction — users compose it from the operator library
- `params` dict carries physical parameters (diffusivity, Rayleigh number, Prandtl number, etc.) — anything the RHS needs
- The solver knows nothing about what equation it's solving — it's a generic integrator
- For coupled systems, `state` is a tuple of fields and `rhs_fn` returns a matching tuple
- Same `jax.lax.scan` pattern as 1D, just with 2D arrays

---

## 7. Visualization (2D)

### 7.1 HeatmapWidget2D

- Frame-by-frame 2D color map (`imshow`) of the temperature field
- Colorbar showing temperature scale
- Uses `Grid2D.X, Grid2D.Y` for coordinate labeling
- Same play/pause/step/reset transport from 1D controld

### 7.2 ViewerWindow2D

- Composes HeatmapWidget2D + ControlPanel
- Accepts single-field trajectories: `(n_steps+1, nx, ny)`
- For coupled systems: shows one primary field (user chooses), with dropdown to switch fields

---

## 8. Examples

### 8.1 Directory

```
examples/
├── 01-1d-heat-equation/     # Existing (unchanged)
└── 02-2d-heat-equation/     # New
    ├── demo.py              # Simple 2D plate with hot left edge, cold right edge
    └── explore.ipynb        # Interactive 2D exploration notebook
```

### 8.2 Demo Scenario

2D plate (1m × 1m, 64×64 cells) with:
- Left edge held at 100°C (Dirichlet)
- Right edge held at 0°C (Dirichlet)
- Top/bottom insulated (Neumann, zero flux)
- Initial: 0°C everywhere
- Visualized as frame-by-frame heatmap reaching linear steady state

---

## 9. Application Tracking

Application scenarios are tracked in a separate markdown file, not in the library code. This keeps the library general and the applications as a roadmap.

**File:** `docs/applications.md`

Tracked scenarios (to be simulated later):
1. Natural convection (Boussinesq) — buoyancy-driven flow from heated plate
2. Thermoelectric cooling (Peltier/Seebeck) — coupled heat + charge transport
3. Absorption chiller — 4-component thermodynamic cycle (lumped ODE)
4. Thermal cloak optimization — optimize conductivity field to route heat around region
5. Forced convection cooling — velocity field + heat advection-diffusion

Each application entry includes: PDE system, boundary conditions, parameter ranges, and target optimization problem.

---

## 10. v0.2 Scope

### In Scope
- [ ] Package restructure: flat files → subpackages (preserving 1D API)
- [ ] `Grid2D` with uniform spacing
- [ ] Operator library: `laplacian_2d`, `gradient_x`, `gradient_y`, `gradient_2d`, `divergence_2d`
- [ ] `BoundaryCondition2D` with per-edge Dirichlet/Neumann
- [ ] `apply_boundary_conditions_2d` (ghost-cell method)
- [ ] `explicit_euler_step_2d`, `solve_2d`, `check_cfl_2d`
- [ ] 2D heatmap viewer (frame-by-frame with colorbar)
- [ ] Example: `02-2d-heat-equation/demo.py`
- [ ] Application tracking file: `docs/applications.md`

### Out of Scope (future)
- Non-uniform / irregular meshes
- Implicit solvers (Crank-Nicolson, ADI)
- Staggered grids for Navier-Stokes
- 3D operators
- Specific coupled-physics implementations (those are applications, not library)

---

## 11. Preserving 1D API

All existing 1D imports must continue to work:

```python
# These must still work after the restructure:
from diffheat import Grid1D, BoundaryCondition, HeatEquation1D
from diffheat import make_laplacian, apply_boundary_conditions
from diffheat import solve_heat_1d, check_cfl, explicit_euler_step
from diffheat import get_device, array, get_default_dtype
from diffheat.viz import run_viewer
```

Subpackage `__init__.py` files re-export everything to maintain backward compatibility.

---

## 12. Implementation Order

1. Package restructure (move 1D code, verify all tests pass)
2. `mesh/grid2d.py` — Grid2D
3. `operators/laplacian.py` — laplacian_2d
4. `operators/gradient.py` — gradient_x, gradient_y, gradient_2d
5. `operators/divergence.py` — divergence_2d
6. `mesh/boundary.py` — BoundaryCondition2D + apply_boundary_conditions_2d
7. `solvers/explicit.py` — explicit_euler_step_2d
8. `solvers/stability.py` + `solvers/scan.py` — check_cfl_2d, solve_2d
9. `viz/heatmap2d.py` + `viz/window.py` — 2D viewer
10. `examples/02-2d-heat-equation/demo.py` — end-to-end demo
11. `docs/applications.md` — application tracking file
12. Final verification (full test suite + demo run)

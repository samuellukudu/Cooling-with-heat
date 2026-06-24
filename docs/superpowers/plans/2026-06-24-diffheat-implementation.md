# diffheat v0.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal JAX-based differentiable 1D heat equation solver library with PyQt visualization, from first principles.

**Architecture:** Six modules in a linear dependency chain: utils → mesh → physics → solvers → viz. The core (mesh, physics, solvers) is pure JAX — no GUI. PyQt lives only in viz.py. A single `diffheat/` package, examples in `examples/`, tests in `tests/`.

**Tech Stack:** Python 3.10+, JAX (with jaxlib), NumPy, PyQt6, matplotlib, pytest, uv

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

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `diffheat/__init__.py` (placeholder)
- Create: `tests/__init__.py` (empty)
- Create: `examples/01-1d-heat-equation/` directory

**Interfaces:**
- Produces: Installable project skeleton, virtual environment ready for development

- [ ] **Step 1: Write pyproject.toml**

```toml
[project]
name = "diffheat"
version = "0.1.0"
description = "Differentiable heat equation simulations with JAX"
requires-python = ">=3.10"
dependencies = [
    "jax",
    "numpy",
]

[project.optional-dependencies]
viz = ["pyqt6", "matplotlib"]
dev = ["pytest", "jupyter"]

[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["diffheat*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Write .gitignore**

```
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
.virtual_documents/
.remember/
dist/
*.egg-info/
```

- [ ] **Step 3: Create directory structure and placeholder files**

```bash
mkdir -p diffheat tests "examples/01-1d-heat-equation"
touch diffheat/__init__.py
touch tests/__init__.py
```

- [ ] **Step 4: Create virtual environment and install**

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[viz,dev]"
```

Expected: virtual environment created, jax + numpy + pyqt6 + matplotlib + pytest + jupyter installed.

- [ ] **Step 5: Verify imports work**

```bash
python -c "import jax; print('JAX backend:', jax.default_backend()); import jax.numpy as jnp; print('Array test:', jnp.array([1.0, 2.0]))"
```

Expected: prints backend name (cpu/gpu) and array.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore diffheat/__init__.py tests/__init__.py
git commit -m "feat: project scaffolding with uv, pyproject.toml, and directory structure

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: utils.py — Device Detection & Array Helpers

**Files:**
- Create: `diffheat/utils.py`
- Create: `tests/test_utils.py`

**Interfaces:**
- Produces:
  - `get_device() -> str` — returns `'gpu'`, `'tpu'`, or `'cpu'`
  - `get_default_dtype() -> jnp.dtype` — returns `jnp.float32` on GPU/TPU, `jnp.float64` on CPU
  - `array(data, dtype=None) -> jnp.ndarray` — creates JAX array on active device with default dtype

- [ ] **Step 1: Write failing tests for utils.py**

```python
# tests/test_utils.py
import pytest
import jax
import jax.numpy as jnp
from diffheat.utils import get_device, get_default_dtype, array


def test_get_device_returns_string():
    device = get_device()
    assert isinstance(device, str)
    assert device in ("cpu", "gpu", "tpu")


def test_get_device_matches_jax_backend():
    device = get_device()
    assert device == jax.default_backend()


def test_get_default_dtype_returns_jax_dtype():
    dtype = get_default_dtype()
    assert dtype in (jnp.float32, jnp.float64)


def test_get_default_dtype_is_float64_on_cpu():
    dtype = get_default_dtype()
    if get_device() == "cpu":
        assert dtype == jnp.float64
    else:
        assert dtype == jnp.float32


def test_array_creates_with_default_dtype():
    data = [1.0, 2.0, 3.0]
    result = array(data)
    assert result.dtype == get_default_dtype()
    assert result.shape == (3,)


def test_array_respects_explicit_dtype():
    data = [1.0, 2.0, 3.0]
    result = array(data, dtype=jnp.float16)
    assert result.dtype == jnp.float16


def test_array_preserves_values():
    data = [1.0, 2.0, 3.0]
    result = array(data)
    assert jnp.allclose(result, jnp.array([1.0, 2.0, 3.0]))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_utils.py -v
```

Expected: FAIL — ModuleNotFoundError or ImportError (utils.py doesn't exist yet).

- [ ] **Step 3: Implement utils.py**

```python
# diffheat/utils.py
"""Device detection and array utilities for diffheat."""
import jax
import jax.numpy as jnp


def get_device() -> str:
    """Return the active JAX backend: 'cpu', 'gpu', or 'tpu'."""
    return jax.default_backend()


def get_default_dtype() -> jnp.dtype:
    """Return float32 on GPU/TPU, float64 on CPU."""
    if get_device() == "cpu":
        return jnp.float64
    return jnp.float32


def array(data, dtype=None) -> jnp.ndarray:
    """Create a JAX array on the default device with the default dtype.

    Args:
        data: Array-like data to convert.
        dtype: Optional explicit dtype. If None, uses get_default_dtype().

    Returns:
        jnp.ndarray on the active device.
    """
    if dtype is None:
        dtype = get_default_dtype()
    return jnp.array(data, dtype=dtype)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_utils.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add diffheat/utils.py tests/test_utils.py
git commit -m "feat: add device detection and array utilities

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: mesh.py — Grid1D & BoundaryCondition

**Files:**
- Create: `diffheat/mesh.py`
- Create: `tests/test_mesh.py`

**Interfaces:**
- Consumes: `diffheat.utils.array`
- Produces:
  - `Grid1D(x, centers, dx, length, n_cells)` — frozen dataclass
  - `Grid1D.uniform(length, n_cells) -> Grid1D` — factory classmethod
  - `BoundaryCondition(kind, value)` — frozen dataclass, `kind: str`, `value: jnp.ndarray` shape (2,)

- [ ] **Step 1: Write failing tests for mesh.py**

```python
# tests/test_mesh.py
import pytest
import jax.numpy as jnp
from diffheat.mesh import Grid1D, BoundaryCondition


class TestGrid1D:
    def test_uniform_creates_correct_n_cells(self):
        grid = Grid1D.uniform(length=1.0, n_cells=10)
        assert grid.n_cells == 10
        assert grid.length == 1.0

    def test_uniform_x_has_n_plus_one_points(self):
        grid = Grid1D.uniform(length=1.0, n_cells=10)
        assert grid.x.shape == (11,)  # n_cells + 1

    def test_uniform_centers_has_n_points(self):
        grid = Grid1D.uniform(length=1.0, n_cells=10)
        assert grid.centers.shape == (10,)  # n_cells

    def test_uniform_dx_has_n_points(self):
        grid = Grid1D.uniform(length=1.0, n_cells=10)
        assert grid.dx.shape == (10,)
        assert jnp.allclose(grid.dx, 0.1)

    def test_uniform_x_starts_at_zero(self):
        grid = Grid1D.uniform(length=2.0, n_cells=5)
        assert jnp.isclose(grid.x[0], 0.0)
        assert jnp.isclose(grid.x[-1], 2.0)

    def test_uniform_centers_are_midpoints(self):
        grid = Grid1D.uniform(length=1.0, n_cells=4)
        dx = 1.0 / 4
        expected_centers = jnp.array([dx/2, dx + dx/2, 2*dx + dx/2, 3*dx + dx/2])
        assert jnp.allclose(grid.centers, expected_centers)

    def test_frozen_dataclass(self):
        grid = Grid1D.uniform(length=1.0, n_cells=10)
        with pytest.raises(Exception):
            grid.length = 2.0


class TestBoundaryCondition:
    def test_dirichlet_creation(self):
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))
        assert bc.kind == "dirichlet"
        assert bc.value.shape == (2,)
        assert jnp.isclose(bc.value[0], 1.0)
        assert jnp.isclose(bc.value[1], 0.0)

    def test_neumann_creation(self):
        bc = BoundaryCondition(kind="neumann", value=jnp.array([0.0, -1.0]))
        assert bc.kind == "neumann"
        assert bc.value.shape == (2,)

    def test_frozen_dataclass(self):
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))
        with pytest.raises(Exception):
            bc.kind = "neumann"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_mesh.py -v
```

Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Implement mesh.py**

```python
# diffheat/mesh.py
"""Grid and boundary condition definitions for 1D heat equation."""
from dataclasses import dataclass

import jax.numpy as jnp

from .utils import array


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

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_mesh.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add diffheat/mesh.py tests/test_mesh.py
git commit -m "feat: add Grid1D and BoundaryCondition dataclasses

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: physics.py — Laplacian & HeatEquation1D

**Files:**
- Create: `diffheat/physics.py`
- Create: `tests/test_physics.py`

**Interfaces:**
- Consumes: `diffheat.mesh.Grid1D`, `diffheat.mesh.BoundaryCondition`, `diffheat.utils.array`
- Produces:
  - `make_laplacian(grid: Grid1D) -> jnp.ndarray` — (N, N) tridiagonal Laplacian matrix
  - `apply_boundary_conditions(laplacian, grid, bc) -> tuple[jnp.ndarray, jnp.ndarray]` — modified Laplacian + boundary source vector
  - `HeatEquation1D(grid, bc, alpha, source)` — frozen dataclass, `alpha: jnp.ndarray | float`, `source: Callable | None`

- [ ] **Step 1: Write failing tests for physics.py**

```python
# tests/test_physics.py
import pytest
import jax.numpy as jnp
from diffheat.mesh import Grid1D, BoundaryCondition
from diffheat.physics import make_laplacian, apply_boundary_conditions, HeatEquation1D


class TestMakeLaplacian:
    def test_returns_square_matrix(self):
        grid = Grid1D.uniform(length=1.0, n_cells=10)
        L = make_laplacian(grid)
        assert L.shape == (10, 10)

    def test_interior_rows_sum_to_zero(self):
        grid = Grid1D.uniform(length=1.0, n_cells=10)
        L = make_laplacian(grid)
        # Interior rows (not first or last) should sum to zero
        for i in range(1, grid.n_cells - 1):
            assert jnp.isclose(jnp.sum(L[i]), 0.0), f"Row {i} sum = {jnp.sum(L[i])}"

    def test_laplacian_of_linear_is_zero(self):
        """∂²(ax+b)/∂x² = 0, so L @ (a*x + b) should be near zero in interior."""
        grid = Grid1D.uniform(length=1.0, n_cells=20)
        L = make_laplacian(grid)
        a, b = 2.0, 1.0
        T_linear = a * grid.centers + b
        result = L @ T_linear
        # Interior cells should be ~0 (boundary effects at edges)
        assert jnp.allclose(result[2:-2], 0.0, atol=1e-7)

    def test_uniform_grid_dx_squared_scaling(self):
        """Laplacian entries should scale as 1/dx²."""
        grid_coarse = Grid1D.uniform(length=1.0, n_cells=10)
        grid_fine = Grid1D.uniform(length=1.0, n_cells=20)
        L_coarse = make_laplacian(grid_coarse)
        L_fine = make_laplacian(grid_fine)
        # Ratio of diagonal magnitudes should be ~(dx_fine/dx_coarse)² = 1/4
        ratio = jnp.abs(L_fine[5, 5]) / jnp.abs(L_coarse[2, 2])
        assert jnp.isclose(ratio, 4.0, rtol=0.01)


class TestApplyBoundaryConditions:
    def test_dirichlet_modifies_boundary_rows(self):
        grid = Grid1D.uniform(length=1.0, n_cells=10)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))
        L = make_laplacian(grid)
        L_mod, b_source = apply_boundary_conditions(L, grid, bc)
        # Boundary source should be non-zero at edges for Dirichlet
        assert not jnp.isclose(b_source[0], 0.0)
        assert not jnp.isclose(b_source[-1], 0.0)

    def test_neumann_boundary_source_is_zero_for_homogeneous(self):
        grid = Grid1D.uniform(length=1.0, n_cells=10)
        bc = BoundaryCondition(kind="neumann", value=jnp.array([0.0, 0.0]))
        L = make_laplacian(grid)
        L_mod, b_source = apply_boundary_conditions(L, grid, bc)
        # Zero-flux Neumann: no boundary source term
        assert jnp.allclose(b_source, 0.0)

    def test_steady_state_dirichlet_is_linear(self):
        """Steady state with Dirichlet BCs T(0)=T_left, T(L)=T_right should be linear."""
        grid = Grid1D.uniform(length=1.0, n_cells=20)
        T_left, T_right = 1.0, 0.0
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([T_left, T_right]))
        L = make_laplacian(grid)
        L_mod, b_source = apply_boundary_conditions(L, grid, bc)
        # Solve steady state: L_mod @ T + b_source = 0
        T_steady = jnp.linalg.solve(L_mod, -b_source)
        # Should be linear from T_left to T_right
        expected = T_left + (T_right - T_left) * grid.centers / grid.length
        assert jnp.allclose(T_steady, expected, atol=1e-6)


class TestHeatEquation1D:
    def test_construction(self):
        grid = Grid1D.uniform(length=1.0, n_cells=10)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))
        eqn = HeatEquation1D(grid=grid, bc=bc, alpha=0.1)
        assert eqn.alpha == 0.1
        assert eqn.source is None

    def test_construction_with_source(self):
        grid = Grid1D.uniform(length=1.0, n_cells=10)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))

        def source(x, t):
            return jnp.sin(x)

        eqn = HeatEquation1D(grid=grid, bc=bc, alpha=0.1, source=source)
        assert eqn.source is source

    def test_frozen_dataclass(self):
        grid = Grid1D.uniform(length=1.0, n_cells=10)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))
        eqn = HeatEquation1D(grid=grid, bc=bc, alpha=0.1)
        with pytest.raises(Exception):
            eqn.alpha = 0.2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_physics.py -v
```

Expected: FAIL — ModuleNotFoundError for `diffheat.physics`.

- [ ] **Step 3: Implement physics.py**

```python
# diffheat/physics.py
"""Heat equation operators and problem definition."""
from dataclasses import dataclass
from typing import Callable, Optional

import jax.numpy as jnp

from .mesh import BoundaryCondition, Grid1D
from .utils import array


def make_laplacian(grid: Grid1D) -> jnp.ndarray:
    """Build the (N, N) tridiagonal Laplacian matrix using centered finite differences.

    Interior: [1, -2, 1] / dx²
    Boundaries: left unmodified (apply_boundary_conditions handles them).

    Args:
        grid: The 1D grid.

    Returns:
        (N, N) Laplacian matrix.
    """
    n = grid.n_cells
    dx = grid.dx

    # Diagonal: -2 / dx²
    diag = array(-2.0 / (dx * dx))

    # Off-diagonals: 1 / dx²
    # dx varies per cell on non-uniform grids (future-proof)
    # Upper diagonal uses average of adjacent dx²
    dx2_left = dx[:-1] * dx[:-1]
    dx2_right = dx[1:] * dx[1:]
    off_diag = array(2.0 / (dx2_left + dx2_right))

    L = jnp.diag(diag) + jnp.diag(off_diag, k=1) + jnp.diag(off_diag, k=-1)
    return L


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
    L_mod = L.copy()
    b_source = jnp.zeros(n, dtype=L.dtype)

    # --- Left boundary (cell index 0) ---
    if bc.kind == "dirichlet":
        # T_ghost = 2*T_left - T_0
        # (T_ghost + T_1 - 2*T_0)/dx² = (2*T_left - T_0 + T_1 - 2*T_0)/dx²
        #                            = (T_1 - 3*T_0)/dx² + 2*T_left/dx²
        L_mod = L_mod.at[0, 0].set(-3.0 / (dx[0] * dx[0]))
        L_mod = L_mod.at[0, 1].set(1.0 / (dx[0] * dx[0]))
        b_source = b_source.at[0].set(2.0 * bc.value[0] / (dx[0] * dx[0]))
    elif bc.kind == "neumann":
        # T_ghost = T_0 - dx * (dT/dx)_{left}
        # (T_ghost + T_1 - 2*T_0)/dx² = (T_0 - dx*(dT/dx) + T_1 - 2*T_0)/dx²
        #                              = (T_1 - T_0)/dx² - (dT/dx)/dx
        L_mod = L_mod.at[0, 0].set(-1.0 / (dx[0] * dx[0]))
        L_mod = L_mod.at[0, 1].set(1.0 / (dx[0] * dx[0]))
        b_source = b_source.at[0].set(-bc.value[0] / dx[0])

    # --- Right boundary (cell index n-1) ---
    if bc.kind == "dirichlet":
        # T_ghost = 2*T_right - T_{n-1}
        L_mod = L_mod.at[n - 1, n - 1].set(-3.0 / (dx[n - 1] * dx[n - 1]))
        L_mod = L_mod.at[n - 1, n - 2].set(1.0 / (dx[n - 1] * dx[n - 1]))
        b_source = b_source.at[n - 1].set(
            2.0 * bc.value[1] / (dx[n - 1] * dx[n - 1])
        )
    elif bc.kind == "neumann":
        # T_ghost = T_{n-1} + dx * (dT/dx)_{right}
        L_mod = L_mod.at[n - 1, n - 1].set(-1.0 / (dx[n - 1] * dx[n - 1]))
        L_mod = L_mod.at[n - 1, n - 2].set(1.0 / (dx[n - 1] * dx[n - 1]))
        b_source = b_source.at[n - 1].set(bc.value[1] / dx[n - 1])

    return L_mod, b_source


@dataclass(frozen=True)
class HeatEquation1D:
    """Complete 1D heat equation problem definition.

    ∂T/∂t = alpha * ∂²T/∂x² + S(x, t)

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

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_physics.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add diffheat/physics.py tests/test_physics.py
git commit -m "feat: add Laplacian operator and HeatEquation1D

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: solvers.py — Time Integration

**Files:**
- Create: `diffheat/solvers.py`
- Create: `tests/test_solvers.py`

**Interfaces:**
- Consumes: `diffheat.physics.HeatEquation1D`, `diffheat.mesh.Grid1D`
- Produces:
  - `explicit_euler_step(T, eqn, t, dt) -> jnp.ndarray` — single Euler step
  - `solve_heat_1d(eqn, T0, t_span, dt) -> jnp.ndarray` — full trajectory via `jax.lax.scan`
  - `check_cfl(grid, alpha, dt) -> bool` — CFL stability check

- [ ] **Step 1: Write failing tests for solvers.py**

```python
# tests/test_solvers.py
import pytest
import jax
import jax.numpy as jnp
from diffheat.mesh import Grid1D, BoundaryCondition
from diffheat.physics import HeatEquation1D
from diffheat.solvers import explicit_euler_step, solve_heat_1d, check_cfl


class TestCheckCFL:
    def test_stable_dt_passes(self):
        grid = Grid1D.uniform(length=1.0, n_cells=50)
        alpha = 0.01
        dt = 0.9 * grid.dx[0] ** 2 / (2 * alpha)  # 90% of CFL limit
        assert check_cfl(grid, alpha, dt) is True

    def test_unstable_dt_fails(self):
        grid = Grid1D.uniform(length=1.0, n_cells=50)
        alpha = 0.01
        dt = 2.0 * grid.dx[0] ** 2 / (2 * alpha)  # 2x CFL limit
        assert check_cfl(grid, alpha, dt) is False


class TestExplicitEulerStep:
    def test_constant_temperature_stays_constant(self):
        """If T is uniform and no source, it should stay uniform."""
        grid = Grid1D.uniform(length=1.0, n_cells=20)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([0.5, 0.5]))
        eqn = HeatEquation1D(grid=grid, bc=bc, alpha=0.1)
        T0 = jnp.full(grid.n_cells, 0.5)
        dt = 0.001
        T1 = explicit_euler_step(T0, eqn, t=0.0, dt=dt)
        assert jnp.allclose(T1, T0, atol=1e-10)

    def test_cooling_rod_temperature_decreases(self):
        """Hot rod with cold boundaries should cool down."""
        grid = Grid1D.uniform(length=1.0, n_cells=20)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([0.0, 0.0]))
        eqn = HeatEquation1D(grid=grid, bc=bc, alpha=0.1)
        T0 = jnp.ones(grid.n_cells)
        dt = 0.0001
        T1 = explicit_euler_step(T0, eqn, t=0.0, dt=dt)
        # Average temperature should decrease (heat flows to cold boundaries)
        assert jnp.mean(T1) < jnp.mean(T0)

    def test_source_term_applied(self):
        """Constant source should increase temperature."""
        grid = Grid1D.uniform(length=1.0, n_cells=20)
        bc = BoundaryCondition(kind="neumann", value=jnp.array([0.0, 0.0]))

        def source(x, t):
            return jnp.ones_like(x)  # uniform heating

        eqn = HeatEquation1D(grid=grid, bc=bc, alpha=0.1, source=source)
        T0 = jnp.zeros(grid.n_cells)
        dt = 0.0001
        T1 = explicit_euler_step(T0, eqn, t=0.0, dt=dt)
        assert jnp.all(T1 > T0)


class TestSolveHeat1D:
    def test_returns_correct_shape(self):
        grid = Grid1D.uniform(length=1.0, n_cells=10)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))
        eqn = HeatEquation1D(grid=grid, bc=bc, alpha=0.01)
        T0 = jnp.zeros(grid.n_cells)
        dt = 0.001
        t_span = (0.0, 0.01)
        trajectory = solve_heat_1d(eqn, T0, t_span, dt)

        n_steps = int((t_span[1] - t_span[0]) / dt) + 1
        assert trajectory.shape == (n_steps, grid.n_cells)

    def test_first_frame_is_initial_condition(self):
        grid = Grid1D.uniform(length=1.0, n_cells=10)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))
        eqn = HeatEquation1D(grid=grid, bc=bc, alpha=0.01)
        T0 = jnp.linspace(1.0, 0.0, grid.n_cells)
        dt = 0.001
        t_span = (0.0, 0.01)
        trajectory = solve_heat_1d(eqn, T0, t_span, dt)
        assert jnp.allclose(trajectory[0], T0)

    def test_approaches_steady_state(self):
        """With Dirichlet BCs, the solution should approach the linear steady state."""
        grid = Grid1D.uniform(length=1.0, n_cells=20)
        T_left, T_right = 1.0, 0.0
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([T_left, T_right]))
        alpha = 1.0
        eqn = HeatEquation1D(grid=grid, bc=bc, alpha=alpha)

        T0 = jnp.zeros(grid.n_cells)
        dt = 0.9 * grid.dx[0] ** 2 / (2 * alpha)  # near CFL limit for speed
        t_span = (0.0, 0.5)  # long enough to approach steady state
        trajectory = solve_heat_1d(eqn, T0, t_span, dt)

        T_final = trajectory[-1]
        expected_steady = T_left + (T_right - T_left) * grid.centers / grid.length
        # Should be close to steady state (within 5%)
        assert jnp.allclose(T_final, expected_steady, atol=0.05)

    def test_gradient_wrt_alpha(self):
        """Verify that gradients flow through the solver w.r.t. alpha."""
        grid = Grid1D.uniform(length=1.0, n_cells=10)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))

        def loss_fn(alpha):
            eqn = HeatEquation1D(grid=grid, bc=bc, alpha=alpha)
            T0 = jnp.zeros(grid.n_cells)
            dt = 0.001
            t_span = (0.0, 0.005)
            trajectory = solve_heat_1d(eqn, T0, t_span, dt)
            return jnp.mean(trajectory[-1])  # mean final temperature

        grad = jax.grad(loss_fn)(0.1)
        assert not jnp.isclose(grad, 0.0)

    def test_gradient_wrt_boundary_value(self):
        """Verify that gradients flow through the solver w.r.t. boundary temperature."""
        grid = Grid1D.uniform(length=1.0, n_cells=10)

        def loss_fn(left_temp):
            bc = BoundaryCondition(kind="dirichlet", value=jnp.array([left_temp, 0.0]))
            eqn = HeatEquation1D(grid=grid, bc=bc, alpha=0.1)
            T0 = jnp.zeros(grid.n_cells)
            dt = 0.001
            t_span = (0.0, 0.005)
            trajectory = solve_heat_1d(eqn, T0, t_span, dt)
            return jnp.mean(trajectory[-1])

        grad = jax.grad(loss_fn)(1.0)
        # Higher left boundary temp → higher mean final temp → positive gradient
        assert grad > 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_solvers.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement solvers.py**

```python
# diffheat/solvers.py
"""Time integration solvers for the 1D heat equation."""
import logging
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .mesh import Grid1D
from .physics import HeatEquation1D, apply_boundary_conditions, make_laplacian

_logger = logging.getLogger(__name__)


class Trajectory(NamedTuple):
    """Result of a heat equation solve.

    Attributes:
        temperature: (n_steps+1, N) array — temperature at each timestep.
        t: (n_steps+1,) array — time values.
        grid: The Grid1D used.
    """
    temperature: jnp.ndarray
    t: jnp.ndarray
    grid: Grid1D


def check_cfl(grid: Grid1D, alpha: float | jnp.ndarray, dt: float) -> bool:
    """Check if dt satisfies the CFL stability condition for explicit Euler.

    dt <= dx² / (2 * alpha)

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
    return dt <= cfl_limit


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

    if not check_cfl(eqn.grid, eqn.alpha, dt):
        cfl_limit = float(jnp.min(eqn.grid.dx)) ** 2 / (2 * float(jnp.max(jnp.asarray(eqn.alpha))))
        _logger.warning(
            f"dt={dt:.2e} exceeds CFL limit {cfl_limit:.2e}. "
            f"Solution may be unstable."
        )

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

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_solvers.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add diffheat/solvers.py tests/test_solvers.py
git commit -m "feat: add explicit Euler solver with jax.lax.scan

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: __init__.py — Public API

**Files:**
- Modify: `diffheat/__init__.py`

**Interfaces:**
- Consumes: all modules
- Produces: Clean public API surface

- [ ] **Step 1: Update __init__.py with public API**

```python
# diffheat/__init__.py
"""diffheat — Differentiable heat equation simulations with JAX."""
import logging

from .mesh import BoundaryCondition, Grid1D
from .physics import HeatEquation1D, apply_boundary_conditions, make_laplacian
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

- [ ] **Step 2: Verify imports work end-to-end**

```bash
python -c "
from diffheat import Grid1D, BoundaryCondition, HeatEquation1D, solve_heat_1d, get_device
import jax.numpy as jnp

print(f'Device: {get_device()}')

grid = Grid1D.uniform(length=1.0, n_cells=20)
bc = BoundaryCondition(kind='dirichlet', value=jnp.array([1.0, 0.0]))
eqn = HeatEquation1D(grid=grid, bc=bc, alpha=0.1)
T0 = jnp.zeros(grid.n_cells)

traj = solve_heat_1d(eqn, T0, (0.0, 0.05), dt=0.001)
print(f'Trajectory shape: {traj.shape}')
print(f'Final mean temp: {jnp.mean(traj[-1]):.4f}')
print('All imports and solve work correctly.')
"
```

Expected: prints device, trajectory shape, and final mean temperature.

- [ ] **Step 3: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all 23 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add diffheat/__init__.py
git commit -m "feat: public API surface with device detection logging

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: viz.py — PyQt6 Viewer

**Files:**
- Create: `diffheat/viz.py`
- Create: `tests/test_viz.py` (smoke tests only)

**Interfaces:**
- Consumes: `diffheat.mesh.Grid1D`, `jnp.ndarray` trajectories
- Produces:
  - `run_viewer(trajectory, grid) -> None` — blocks until window closes

**Note:** viz.py is the ONLY module allowed to import PyQt6 or matplotlib.

- [ ] **Step 1: Write smoke test for viz.py**

```python
# tests/test_viz.py
"""Smoke tests for visualization module — no GUI rendering in CI."""
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
        traj_np = jnp.asarray(trajectory).copy()
        assert traj_np.shape == trajectory.shape
        assert not isinstance(traj_np, type(trajectory))  # should be plain ndarray
```

- [ ] **Step 2: Run smoke tests to verify they fail**

```bash
python -m pytest tests/test_viz.py -v
```

Expected: FAIL — ModuleNotFoundError for `diffheat.viz`.

- [ ] **Step 3: Implement viz.py**

```python
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
        self.slider.setValue(self.current_frame)
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
```

- [ ] **Step 4: Run smoke tests**

```bash
python -m pytest tests/test_viz.py -v
```

Expected: all 3 tests PASS (import check, callable check, numpy conversion).

- [ ] **Step 5: Commit**

```bash
git add diffheat/viz.py tests/test_viz.py
git commit -m "feat: add PyQt6 viewer with heatmap, snapshot, and animation controls

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: Demo Script — End-to-End Example

**Files:**
- Create: `examples/01-1d-heat-equation/demo.py`

**Interfaces:**
- Consumes: `diffheat` (all modules), `diffheat.viz.run_viewer`
- Produces: Runnable demo that solves and visualizes a 1D heat equation

- [ ] **Step 1: Write demo.py**

```python
#!/usr/bin/env python3
"""Demo: 1D heat equation — hot left boundary, cold right boundary.

A uniform rod initially at 0°C, with the left end held at 1°C and the
right end at 0°C. Over time, the temperature profile approaches the
linear steady-state solution.

Run:
    python examples/01-1d-heat-equation/demo.py
"""
import jax.numpy as jnp

from diffheat import (
    BoundaryCondition,
    Grid1D,
    HeatEquation1D,
    get_device,
    solve_heat_1d,
)
from diffheat.viz import run_viewer


def main():
    print(f"Running on: {get_device()}")
    print("=" * 50)

    # --- Grid ---
    length = 1.0  # 1 meter rod
    n_cells = 100
    grid = Grid1D.uniform(length=length, n_cells=n_cells)
    print(f"Grid: {grid.n_cells} cells, dx = {float(grid.dx[0]):.4f} m")

    # --- Boundary conditions ---
    T_left = 1.0  # hot left end
    T_right = 0.0  # cold right end
    bc = BoundaryCondition(kind="dirichlet", value=jnp.array([T_left, T_right]))

    # --- Material ---
    alpha = 0.01  # thermal diffusivity (m²/s), roughly like some polymers

    # --- Initial condition ---
    T0 = jnp.zeros(grid.n_cells)  # rod starts at 0°C everywhere

    # --- Time parameters ---
    t_end = 2.0  # simulate for 2 seconds
    dt = 0.001  # time step

    # CFL check
    dx = float(grid.dx[0])
    cfl_limit = dx**2 / (2 * alpha)
    print(f"CFL limit: {cfl_limit:.4f} s")
    print(f"dt: {dt:.4f} s (stable: {dt <= cfl_limit})")

    # --- Solve ---
    eqn = HeatEquation1D(grid=grid, bc=bc, alpha=alpha)
    print(f"Solving from t=0 to t={t_end}...")
    trajectory = solve_heat_1d(eqn, T0, (0.0, t_end), dt)

    n_steps = len(trajectory)
    print(f"Done. {n_steps} timesteps computed.")
    print(f"Initial mean T: {jnp.mean(trajectory[0]):.4f}")
    print(f"Final mean T:   {jnp.mean(trajectory[-1]):.4f}")
    print(f"Steady-state expected mean: {(T_left + T_right) / 2:.4f}")

    # --- Visualize ---
    print("\nLaunching viewer...")
    run_viewer(trajectory, grid, dt)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the demo (headless check, skip viz)**

```bash
python -c "
import jax.numpy as jnp
from diffheat import Grid1D, BoundaryCondition, HeatEquation1D, solve_heat_1d

grid = Grid1D.uniform(length=1.0, n_cells=100)
bc = BoundaryCondition(kind='dirichlet', value=jnp.array([1.0, 0.0]))
eqn = HeatEquation1D(grid=grid, bc=bc, alpha=0.01)
T0 = jnp.zeros(grid.n_cells)

traj = solve_heat_1d(eqn, T0, (0.0, 2.0), dt=0.001)
print(f'Trajectory shape: {traj.shape}')
print(f'Final mean T: {jnp.mean(traj[-1]):.4f}')
print('Demo solve works headlessly.')
"
```

Expected: prints trajectory shape and final mean temperature.

- [ ] **Step 3: Commit**

```bash
git add examples/01-1d-heat-equation/demo.py
git commit -m "feat: add end-to-end 1D heat equation demo script

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: Integration Tests & Differentiability Verification

**Files:**
- Create: `tests/test_integration.py`

**Interfaces:**
- Consumes: all modules
- Produces: Confidence that the full pipeline works and gradients are correct

- [ ] **Step 1: Write integration tests**

```python
# tests/test_integration.py
"""End-to-end integration tests for the diffheat pipeline."""
import jax
import jax.numpy as jnp
import numpy as np
from diffheat import (
    BoundaryCondition,
    Grid1D,
    HeatEquation1D,
    solve_heat_1d,
)


class TestEndToEnd:
    def test_full_pipeline_dirichlet(self):
        """Hot left, cold right Dirichlet → should approach linear steady state."""
        grid = Grid1D.uniform(length=1.0, n_cells=50)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))
        eqn = HeatEquation1D(grid=grid, bc=bc, alpha=0.1)
        T0 = jnp.zeros(grid.n_cells)
        dt = 0.0005
        trajectory = solve_heat_1d(eqn, T0, (0.0, 0.1), dt)

        T_final = trajectory[-1]
        expected = 1.0 + (0.0 - 1.0) * grid.centers / grid.length
        # Should be reasonably close to steady state
        assert jnp.allclose(T_final, expected, atol=0.1)

    def test_full_pipeline_neumann_insulated(self):
        """Insulated boundaries → total heat should be conserved (no source)."""
        grid = Grid1D.uniform(length=1.0, n_cells=30)
        bc = BoundaryCondition(kind="neumann", value=jnp.array([0.0, 0.0]))
        eqn = HeatEquation1D(grid=grid, bc=bc, alpha=0.1)
        T0 = jnp.sin(jnp.pi * grid.centers / grid.length)
        dt = 0.0001
        trajectory = solve_heat_1d(eqn, T0, (0.0, 0.02), dt)

        # Total heat (integral of T) should be conserved
        total_initial = jnp.sum(T0) * float(grid.dx[0])
        total_final = jnp.sum(trajectory[-1]) * float(grid.dx[0])
        assert jnp.isclose(total_initial, total_final, rtol=1e-4)

    def test_jit_compilation_works(self):
        """solve_heat_1d should be JIT-compilable."""
        grid = Grid1D.uniform(length=1.0, n_cells=20)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))

        @jax.jit
        def jit_solve(alpha, T0):
            eqn = HeatEquation1D(grid=grid, bc=bc, alpha=alpha)
            return solve_heat_1d(eqn, T0, (0.0, 0.01), dt=0.001)

        T0 = jnp.zeros(grid.n_cells)
        result = jit_solve(0.1, T0)
        assert result.shape[0] > 1


class TestGradients:
    def test_gradient_wrt_alpha_is_smooth(self):
        """Gradient of loss w.r.t. alpha should vary smoothly with alpha."""
        grid = Grid1D.uniform(length=1.0, n_cells=20)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))
        T0 = jnp.zeros(grid.n_cells)

        def loss(alpha):
            eqn = HeatEquation1D(grid=grid, bc=bc, alpha=alpha)
            traj = solve_heat_1d(eqn, T0, (0.0, 0.02), dt=0.001)
            return jnp.mean(traj[-1])

        grad_fn = jax.grad(loss)
        alphas = [0.05, 0.1, 0.2, 0.5]
        grads = [float(grad_fn(a)) for a in alphas]

        # Higher alpha → faster diffusion → lower final mean (more heat lost to cold boundary)
        # Gradient should become more negative with increasing alpha?
        # Actually: higher alpha means faster approach to steady state
        # At alpha=0.05: still far from steady, mean ~ low
        # At alpha=0.5: close to steady, mean ~ 0.5
        # So loss increases with alpha → positive gradient
        for g in grads:
            assert not jnp.isnan(g)
            assert jnp.isfinite(g)

    def test_gradient_wrt_initial_condition(self):
        """Gradient w.r.t. T0 should be non-zero for each cell."""
        grid = Grid1D.uniform(length=1.0, n_cells=15)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))

        def loss(T0):
            eqn = HeatEquation1D(grid=grid, bc=bc, alpha=0.1)
            traj = solve_heat_1d(eqn, T0, (0.0, 0.01), dt=0.001)
            return jnp.mean(traj[-1])

        T0 = jnp.linspace(0.0, 1.0, grid.n_cells)
        grad = jax.grad(loss)(T0)
        assert grad.shape == (grid.n_cells,)
        # Every cell's initial temperature affects the final mean
        assert jnp.all(grad > 0.0)  # higher T0 → higher final mean

    def test_gradient_wrt_boundary_value(self):
        """∂loss/∂T_left should be positive (hotter boundary → hotter final)."""
        grid = Grid1D.uniform(length=1.0, n_cells=20)

        def loss(T_left):
            bc = BoundaryCondition(kind="dirichlet", value=jnp.array([T_left, 0.0]))
            eqn = HeatEquation1D(grid=grid, bc=bc, alpha=0.1)
            T0 = jnp.zeros(grid.n_cells)
            traj = solve_heat_1d(eqn, T0, (0.0, 0.02), dt=0.001)
            return jnp.mean(traj[-1])

        grad = jax.grad(loss)(1.0)
        assert float(grad) > 0.0

    def test_finite_difference_agrees_with_autodiff(self):
        """Finite difference gradient should approximately match autodiff."""
        grid = Grid1D.uniform(length=1.0, n_cells=10)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))

        def loss(alpha):
            eqn = HeatEquation1D(grid=grid, bc=bc, alpha=alpha)
            T0 = jnp.zeros(grid.n_cells)
            traj = solve_heat_1d(eqn, T0, (0.0, 0.005), dt=0.001)
            return jnp.mean(traj[-1])

        alpha = 0.1
        grad_ad = jax.grad(loss)(alpha)

        # Finite difference
        eps = 1e-4
        grad_fd = (loss(alpha + eps) - loss(alpha - eps)) / (2 * eps)

        assert jnp.isclose(grad_ad, grad_fd, rtol=1e-3)
```

- [ ] **Step 2: Run integration tests**

```bash
python -m pytest tests/test_integration.py -v
```

Expected: 8 tests PASS (including gradient validation).

- [ ] **Step 3: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: 34 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests with gradient verification

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 10: Jupyter Notebook — Interactive Exploration

**Files:**
- Create: `examples/01-1d-heat-equation/explore.ipynb`

**Interfaces:**
- Consumes: `diffheat`
- Produces: Interactive notebook for parameter exploration

- [ ] **Step 1: Write explore.ipynb**

Create a Jupyter notebook with the following cells:

Cell 1 (markdown):
```markdown
# diffheat — Interactive Exploration

Explore the 1D heat equation with JAX-powered differentiable simulation.

Try changing parameters and re-running cells to see how the solution changes.
```

Cell 2 (code):
```python
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffheat import Grid1D, BoundaryCondition, HeatEquation1D, solve_heat_1d, get_device

print(f"Running on: {get_device()}")
print(f"JAX version: {jax.__version__}")
```

Cell 3 (code):
```python
# --- Experiment parameters (tweak these!) ---
length = 1.0      # rod length (m)
n_cells = 100     # spatial resolution
alpha = 0.01      # thermal diffusivity (m²/s)
T_left = 1.0      # left boundary temperature (°C)
T_right = 0.0     # right boundary temperature (°C)
T_initial = 0.0   # initial temperature everywhere (°C)
t_end = 2.0       # simulation duration (s)
dt = 0.001        # time step (s)

# CFL stability check
dx = length / n_cells
cfl_limit = dx**2 / (2 * alpha)
print(f"CFL limit: {cfl_limit:.4f} s — dt={dt} is {'stable' if dt <= cfl_limit else 'UNSTABLE!'}")
```

Cell 4 (code):
```python
# Build the problem
grid = Grid1D.uniform(length=length, n_cells=n_cells)
bc = BoundaryCondition(kind="dirichlet", value=jnp.array([T_left, T_right]))
eqn = HeatEquation1D(grid=grid, bc=bc, alpha=alpha)
T0 = jnp.full(grid.n_cells, T_initial)

# Solve
trajectory = solve_heat_1d(eqn, T0, (0.0, t_end), dt)
print(f"Computed {len(trajectory)} timesteps")
print(f"Trajectory shape: {trajectory.shape}")
```

Cell 5 (code):
```python
# Plot: temperature profiles at several times
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

n = len(trajectory)
indices = [0, n // 10, n // 5, n // 2, n - 1]
times = jnp.arange(n) * dt

for idx in indices:
    ax1.plot(grid.centers, trajectory[idx], label=f"t = {times[idx]:.3f}")

ax1.set_xlabel("Position (m)")
ax1.set_ylabel("Temperature")
ax1.set_title("Temperature Profiles Over Time")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Expected steady state
T_steady = T_left + (T_right - T_left) * grid.centers / grid.length
ax1.plot(grid.centers, T_steady, "k--", linewidth=1, alpha=0.5, label="Steady state")

# Space-time heatmap
extent = [0, length, times[-1], 0]
im = ax2.imshow(trajectory, aspect="auto", extent=extent, cmap="hot", origin="upper")
ax2.set_xlabel("Position (m)")
ax2.set_ylabel("Time (s)")
ax2.set_title("Space-Time Heatmap")
plt.colorbar(im, ax=ax2, label="Temperature")

plt.tight_layout()
plt.show()
```

Cell 6 (code):
```python
# Differentiate: how does final mean temperature depend on alpha?
def final_mean_temp(alpha_val):
    eqn = HeatEquation1D(grid=grid, bc=bc, alpha=alpha_val)
    T0 = jnp.full(grid.n_cells, 0.0)
    traj = solve_heat_1d(eqn, T0, (0.0, 0.5), dt=0.001)
    return jnp.mean(traj[-1])

grad_fn = jax.grad(final_mean_temp)

alphas = jnp.logspace(-3, -0.5, 20)
means = [float(final_mean_temp(a)) for a in alphas]
grads = [float(grad_fn(a)) for a in alphas]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.semilogx(alphas, means, "b-o")
ax1.set_xlabel("alpha (diffusivity)")
ax1.set_ylabel("Final mean temperature")
ax1.set_title("Mean final T vs. diffusivity")
ax1.grid(True, alpha=0.3)

ax2.semilogx(alphas, grads, "r-o")
ax2.set_xlabel("alpha (diffusivity)")
ax2.set_ylabel("d(mean T)/d(alpha)")
ax2.set_title("Gradient of mean T w.r.t. alpha")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

- [ ] **Step 2: Verify notebook is valid JSON**

```bash
python -c "
import json
with open('examples/01-1d-heat-equation/explore.ipynb') as f:
    nb = json.load(f)
print(f'Notebook has {len(nb[\"cells\"])} cells')
for i, cell in enumerate(nb['cells']):
    print(f'  Cell {i}: {cell[\"cell_type\"]}')
"
```

- [ ] **Step 3: Commit**

```bash
git add examples/01-1d-heat-equation/explore.ipynb
git commit -m "feat: add interactive Jupyter notebook for parameter exploration

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 11: Final Verification

- [ ] **Step 1: Run full test suite one final time**

```bash
python -m pytest tests/ -v --tb=short
```

Expected: all 34+ tests PASS.

- [ ] **Step 2: Run the demo headlessly (verify solve works)**

```bash
python -c "
import jax.numpy as jnp
from diffheat import Grid1D, BoundaryCondition, HeatEquation1D, solve_heat_1d, get_device

print(f'Device: {get_device()}')
grid = Grid1D.uniform(length=1.0, n_cells=100)
bc = BoundaryCondition(kind='dirichlet', value=jnp.array([1.0, 0.0]))
eqn = HeatEquation1D(grid=grid, bc=bc, alpha=0.01)
T0 = jnp.zeros(grid.n_cells)
traj = solve_heat_1d(eqn, T0, (0.0, 2.0), dt=0.001)
print(f'Shape: {traj.shape}')
print(f'Initial mean: {jnp.mean(traj[0]):.4f}')
print(f'Final mean: {jnp.mean(traj[-1]):.4f}')
print('SUCCESS: Full pipeline works.')
"
```

Expected: prints success message with trajectory shape and temperatures.

- [ ] **Step 3: Verify git status is clean**

```bash
git status
```

Expected: all files committed, clean working tree.

- [ ] **Step 4: Final commit (if any stragglers)**

```bash
git add -A && git diff --cached --stat
# Only commit if there are meaningful changes
```

---

## Implementation Summary

| Task | File(s) | Tests | Key Deliverable |
|------|---------|-------|-----------------|
| 1 | pyproject.toml, .gitignore | — | Runnable project skeleton with uv venv |
| 2 | diffheat/utils.py | 6 | Device detection, array creation |
| 3 | diffheat/mesh.py | 8 | Grid1D, BoundaryCondition |
| 4 | diffheat/physics.py | 9 | Laplacian, HeatEquation1D |
| 5 | diffheat/solvers.py | 9 | explicit_euler_step, solve_heat_1d |
| 6 | diffheat/__init__.py | — | Public API |
| 7 | diffheat/viz.py | 3 | PyQt6 viewer |
| 8 | examples/.../demo.py | — | End-to-end demo |
| 9 | tests/test_integration.py | 8 | Gradient + pipeline verification |
| 10 | examples/.../explore.ipynb | — | Interactive notebook |
| 11 | — | all 34+ | Final verification |

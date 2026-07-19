# Forced Convection (Advection-Diffusion) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add advection-diffusion PDE support (forced convection) to diffheat in 1D, 2D, and 3D with upwind-stabilized advection operators, combined CFL conditions, and end-to-end JAX differentiability.

**Architecture:** Three new modules following the established HeatEquation→WaveEquation→TelegrapherEquation pattern: `operators/advection.py` (upwind advection operators), `physics/advection_diffusion.py` (AdvectionDiffusion dataclasses), `solvers/advection_diffusion.py` (jax.lax.scan-based solvers). CFL functions added to existing `stability.py`. No changes to any existing operator, physics, or solver code.

**Tech Stack:** JAX, jax.numpy, Python 3.10+, pytest

## Global Constraints

- All operators must be pure JAX functions — no classes, no state, JIT-compatible
- End-to-end differentiability: `jax.grad` must flow through every component
- Upwind scheme is first-order (higher-order schemes are out of scope)
- Velocity field is prescribed (callable) — no Navier-Stokes coupling
- CFL functions return `bool` — consistent with existing `check_cfl`, `check_cfl_wave_*`, `check_cfl_telegrapher_*`
- `alpha` validated > 0 in `__post_init__`

---

### Task 1: Advection Operators (1D)

**Files:**
- Create: `diffheat/operators/advection.py`
- Create: `tests/test_advection_operators.py`

**Interfaces:**
- Consumes: `jax.numpy` (standard)
- Produces: `advection_1d(T: jnp.ndarray, u: jnp.ndarray, dx: float | jnp.ndarray) -> jnp.ndarray`

- [ ] **Step 1: Write failing tests for advection_1d**

Create `tests/test_advection_operators.py`:

```python
# tests/test_advection_operators.py
"""Tests for advection operators."""
import jax
import jax.numpy as jnp
import pytest


class TestAdvection1D:
    @pytest.fixture
    def grid_1d(self):
        """100 cells, length 10.0, dx = 0.1."""
        from diffheat.mesh import Grid1D
        return Grid1D.uniform(length=10.0, n_cells=100)

    def test_zero_velocity_returns_zero(self, grid_1d):
        """advection_1d with u=0 everywhere should return zeros."""
        from diffheat.operators.advection import advection_1d

        T = jnp.sin(2 * jnp.pi * grid_1d.centers / grid_1d.length)
        u = jnp.zeros_like(T)
        result = advection_1d(T, u, grid_1d.dx)
        assert result.shape == T.shape
        assert jnp.allclose(result, 0.0)

    def test_constant_temperature_returns_zero(self, grid_1d):
        """Advection of constant field should be zero regardless of velocity."""
        from diffheat.operators.advection import advection_1d

        T = jnp.full(grid_1d.n_cells, 5.0)
        u = jnp.full(grid_1d.n_cells, 2.0)
        result = advection_1d(T, u, grid_1d.dx)
        # interior should be zero; boundaries may differ due to roll wrap-around
        assert jnp.allclose(result[1:-1], 0.0, atol=1e-10)

    def test_positive_velocity_uses_backward_difference(self, grid_1d):
        """With u > 0 everywhere, advection uses backward difference."""
        from diffheat.operators.advection import advection_1d

        # Linear T = a*x => dT/dx = a everywhere
        a = 3.0
        T = a * grid_1d.centers
        # With u = 2.0 everywhere: advection = -u * dT/dx = -2.0 * a
        u = jnp.full(grid_1d.n_cells, 2.0)
        result = advection_1d(T, u, grid_1d.dx)
        expected = -2.0 * a
        # First cell uses forward difference (one-sided), skip it
        assert jnp.allclose(result[1:], expected, atol=0.1)

    def test_negative_velocity_uses_forward_difference(self, grid_1d):
        """With u < 0 everywhere, advection uses forward difference."""
        from diffheat.operators.advection import advection_1d

        a = 3.0
        T = a * grid_1d.centers
        u = jnp.full(grid_1d.n_cells, -2.0)
        result = advection_1d(T, u, grid_1d.dx)
        expected = 2.0 * a  # -(-2.0) * a = 2.0 * a
        # Last cell uses backward difference (one-sided), skip it
        assert jnp.allclose(result[:-1], expected, atol=0.1)

    def test_gaussian_translation(self, grid_1d):
        """A Gaussian pulse advected at constant u should translate."""
        from diffheat.operators.advection import advection_1d

        dx = float(jnp.mean(grid_1d.dx))
        x = grid_1d.centers
        sigma = 0.5
        x0 = 5.0
        # Gaussian centered at x0
        T = jnp.exp(-((x - x0) ** 2) / (2 * sigma**2))
        u = jnp.full_like(T, 1.0)  # uniform flow to the right

        # After one explicit Euler step: T_new = T - dt * u * dT/dx
        dt = 0.01
        adv = advection_1d(T, u, grid_1d.dx)
        T_new = T + dt * adv

        # Exact: Gaussian centered at x0 + u*dt
        T_exact = jnp.exp(-((x - (x0 + 1.0 * dt)) ** 2) / (2 * sigma**2))
        # Check that peak moved right (interior only, skip boundaries)
        assert jnp.max(T_new[10:-10]) > jnp.max(T[10:-10]) * 0.9
        # Peak should be near new position
        peak_idx = jnp.argmax(T_new[10:-10]) + 10
        expected_peak_idx = jnp.argmax(T_exact)
        assert abs(peak_idx - expected_peak_idx) <= 3  # within 3 cells

    def test_is_jax_differentiable(self, grid_1d):
        """jax.grad should work through advection_1d."""
        from diffheat.operators.advection import advection_1d

        T = jnp.sin(grid_1d.centers)
        u = jnp.ones_like(T)
        dx = float(jnp.mean(grid_1d.dx))

        def sum_adv(T):
            return jnp.sum(advection_1d(T, u, dx))

        grad = jax.grad(sum_adv)(T)
        assert grad.shape == T.shape
        assert not jnp.allclose(grad, 0.0)  # gradient should be non-zero
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/samu2505/ENTERPRISE/Cooling-with-heat && uv run pytest tests/test_advection_operators.py::TestAdvection1D -v
```
Expected: All fail with `ModuleNotFoundError: No module named 'diffheat.operators.advection'`

- [ ] **Step 3: Write advection_1d implementation**

Create `diffheat/operators/advection.py`:

```python
# diffheat/operators/advection.py
"""Advection operators using first-order upwind finite differences."""
import jax.numpy as jnp


def advection_1d(T: jnp.ndarray, u: jnp.ndarray, dx: float | jnp.ndarray) -> jnp.ndarray:
    """Compute -u * dT/dx using first-order upwind finite differences.

    Uses backward difference where u > 0 (flow left-to-right),
    forward difference where u <= 0 (flow right-to-left).

    Args:
        T: (N,) temperature field at cell centers.
        u: (N,) velocity field at cell centers.
        dx: Cell width. Scalar or (N,) array.

    Returns:
        (N,) advection term -u * dT/dx at cell centers.
    """
    T_forward = jnp.roll(T, -1)   # T[i+1]
    T_backward = jnp.roll(T, 1)   # T[i-1]

    forward_diff = (T_forward - T) / dx
    backward_diff = (T - T_backward) / dx

    # Upwind: use backward diff where u > 0, forward diff elsewhere
    return -u * jnp.where(u > 0, backward_diff, forward_diff)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/samu2505/ENTERPRISE/Cooling-with-heat && uv run pytest tests/test_advection_operators.py::TestAdvection1D -v
```
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add diffheat/operators/advection.py tests/test_advection_operators.py
git commit -m "feat: add advection_1d operator with first-order upwind scheme

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Advection Operators (2D)

**Files:**
- Modify: `diffheat/operators/advection.py`
- Modify: `tests/test_advection_operators.py`

**Interfaces:**
- Consumes: `advection_1d` (Task 1)
- Produces: `advection_2d(T: jnp.ndarray, u_x: jnp.ndarray, u_y: jnp.ndarray, dx: float | jnp.ndarray, dy: float | jnp.ndarray) -> jnp.ndarray`

- [ ] **Step 1: Write failing tests for advection_2d**

Append to `tests/test_advection_operators.py`:

```python
class TestAdvection2D:
    @pytest.fixture
    def grid_2d(self):
        """40x30 grid on [0,2]x[0,1]."""
        from diffheat.mesh import Grid2D
        return Grid2D.uniform(Lx=2.0, Ly=1.0, nx=40, ny=30)

    def test_returns_correct_shape(self, grid_2d):
        from diffheat.operators.advection import advection_2d
        T = jnp.ones((grid_2d.nx, grid_2d.ny))
        u_x = jnp.zeros_like(T)
        u_y = jnp.zeros_like(T)
        result = advection_2d(T, u_x, u_y, grid_2d.dx, grid_2d.dy)
        assert result.shape == (grid_2d.nx, grid_2d.ny)

    def test_zero_velocity_returns_zero(self, grid_2d):
        from diffheat.operators.advection import advection_2d
        T = jnp.sin(jnp.pi * jnp.arange(grid_2d.nx)[:, None] / grid_2d.nx)
        u_x = jnp.zeros_like(T)
        u_y = jnp.zeros_like(T)
        result = advection_2d(T, u_x, u_y, grid_2d.dx, grid_2d.dy)
        assert jnp.allclose(result, 0.0)

    def test_uniform_x_flow(self, grid_2d):
        """With u_x > 0, u_y = 0: result should match 1D advection row-by-row."""
        from diffheat.operators.advection import advection_2d

        # T varies only in x: T = a * x
        a = 2.0
        T = a * jnp.arange(grid_2d.nx, dtype=jnp.float64)[:, None] * jnp.ones(grid_2d.ny)[None, :]
        u_x = jnp.full_like(T, 1.5)
        u_y = jnp.zeros_like(T)
        dx = float(jnp.mean(grid_2d.dx))

        result = advection_2d(T, u_x, u_y, grid_2d.dx, grid_2d.dy)
        expected = -1.5 * a  # -u_x * dT/dx
        # interior x, all y
        assert jnp.allclose(result[1:-1, :], expected, atol=0.15)

    def test_uniform_y_flow(self, grid_2d):
        """With u_x = 0, u_y > 0: advection acts only in y."""
        from diffheat.operators.advection import advection_2d

        # T varies only in y: T = b * y
        b = 2.0
        T = b * jnp.ones(grid_2d.nx)[:, None] * jnp.arange(grid_2d.ny, dtype=jnp.float64)[None, :]
        u_x = jnp.zeros_like(T)
        u_y = jnp.full_like(T, 3.0)
        dy = float(jnp.mean(grid_2d.dy))

        result = advection_2d(T, u_x, u_y, grid_2d.dx, grid_2d.dy)
        expected = -3.0 * b
        # all x, interior y
        assert jnp.allclose(result[:, 1:-1], expected, atol=0.15)

    def test_is_jax_differentiable(self, grid_2d):
        from diffheat.operators.advection import advection_2d
        T = jnp.sin(jnp.pi * jnp.arange(grid_2d.nx)[:, None] / grid_2d.nx)
        u_x = jnp.ones_like(T)
        u_y = jnp.zeros_like(T)

        def sum_adv(T):
            return jnp.sum(advection_2d(T, u_x, u_y, grid_2d.dx, grid_2d.dy))

        grad = jax.grad(sum_adv)(T)
        assert grad.shape == T.shape
        assert not jnp.allclose(grad, 0.0)
```

- [ ] **Step 2: Run new tests to verify they fail**

```bash
cd /home/samu2505/ENTERPRISE/Cooling-with-heat && uv run pytest tests/test_advection_operators.py::TestAdvection2D -v
```
Expected: All fail with `ImportError: cannot import name 'advection_2d'`

- [ ] **Step 3: Write advection_2d implementation**

Append to `diffheat/operators/advection.py`:

```python
def advection_2d(
    T: jnp.ndarray,
    u_x: jnp.ndarray,
    u_y: jnp.ndarray,
    dx: float | jnp.ndarray,
    dy: float | jnp.ndarray,
) -> jnp.ndarray:
    """Compute -(u_x * dT/dx + u_y * dT/dy) using first-order upwind.

    Upwinding is applied independently in x and y.
    Uses backward difference where velocity component > 0,
    forward difference where velocity component <= 0.

    Args:
        T: (nx, ny) temperature field at cell centers.
        u_x: (nx, ny) x-velocity field.
        u_y: (nx, ny) y-velocity field.
        dx: x cell width. Scalar or (nx,) array.
        dy: y cell width. Scalar or (ny,) array.

    Returns:
        (nx, ny) advection term at cell centers.
    """
    # --- x-direction upwind ---
    # Broadcast dx to (nx, 1) for dividing (nx, ny) arrays
    _dx = dx[:, jnp.newaxis] if dx.ndim == 1 else dx
    T_forward_x = jnp.roll(T, -1, axis=0)
    T_backward_x = jnp.roll(T, 1, axis=0)
    forward_diff_x = (T_forward_x - T) / _dx
    backward_diff_x = (T - T_backward_x) / _dx
    adv_x = -u_x * jnp.where(u_x > 0, backward_diff_x, forward_diff_x)

    # --- y-direction upwind ---
    # Broadcast dy to (1, ny) for dividing (nx, ny) arrays
    _dy = dy[jnp.newaxis, :] if dy.ndim == 1 else dy
    T_forward_y = jnp.roll(T, -1, axis=1)
    T_backward_y = jnp.roll(T, 1, axis=1)
    forward_diff_y = (T_forward_y - T) / _dy
    backward_diff_y = (T - T_backward_y) / _dy
    adv_y = -u_y * jnp.where(u_y > 0, backward_diff_y, forward_diff_y)

    return adv_x + adv_y
```

- [ ] **Step 4: Run all advection tests**

```bash
cd /home/samu2505/ENTERPRISE/Cooling-with-heat && uv run pytest tests/test_advection_operators.py -v
```
Expected: All 11 tests PASS (6 from 1D + 5 from 2D)

- [ ] **Step 5: Commit**

```bash
git add diffheat/operators/advection.py tests/test_advection_operators.py
git commit -m "feat: add advection_2d operator with first-order upwind scheme

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Advection Operators (3D)

**Files:**
- Modify: `diffheat/operators/advection.py`
- Modify: `tests/test_advection_operators.py`

**Interfaces:**
- Consumes: `advection_1d`, `advection_2d` (Tasks 1-2)
- Produces: `advection_3d(T: jnp.ndarray, u_x: jnp.ndarray, u_y: jnp.ndarray, u_z: jnp.ndarray, dx: float | jnp.ndarray, dy: float | jnp.ndarray, dz: float | jnp.ndarray) -> jnp.ndarray`

- [ ] **Step 1: Write failing tests for advection_3d**

Append to `tests/test_advection_operators.py`:

```python
class TestAdvection3D:
    @pytest.fixture
    def grid_3d(self):
        """20x15x10 grid on [1,1,1]."""
        from diffheat.mesh import Grid3D
        return Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=20, ny=15, nz=10)

    def test_returns_correct_shape(self, grid_3d):
        from diffheat.operators.advection import advection_3d
        T = jnp.ones((grid_3d.nx, grid_3d.ny, grid_3d.nz))
        u_x = jnp.zeros_like(T)
        u_y = jnp.zeros_like(T)
        u_z = jnp.zeros_like(T)
        result = advection_3d(T, u_x, u_y, u_z, grid_3d.dx, grid_3d.dy, grid_3d.dz)
        assert result.shape == (grid_3d.nx, grid_3d.ny, grid_3d.nz)

    def test_zero_velocity_returns_zero(self, grid_3d):
        from diffheat.operators.advection import advection_3d
        T = jnp.ones((grid_3d.nx, grid_3d.ny, grid_3d.nz))
        u_x = jnp.zeros_like(T)
        u_y = jnp.zeros_like(T)
        u_z = jnp.zeros_like(T)
        result = advection_3d(T, u_x, u_y, u_z, grid_3d.dx, grid_3d.dy, grid_3d.dz)
        assert jnp.allclose(result, 0.0)

    def test_uniform_x_flow(self, grid_3d):
        """With u_x > 0, u_y = u_z = 0: result matches 1D per y-z slice."""
        from diffheat.operators.advection import advection_3d

        a = 2.0
        T = a * jnp.arange(grid_3d.nx, dtype=jnp.float64)[:, None, None] * jnp.ones((1, grid_3d.ny, grid_3d.nz))
        u_x = jnp.full_like(T, 1.5)
        u_y = jnp.zeros_like(T)
        u_z = jnp.zeros_like(T)
        dx = float(jnp.mean(grid_3d.dx))

        result = advection_3d(T, u_x, u_y, u_z, grid_3d.dx, grid_3d.dy, grid_3d.dz)
        expected = -1.5 * a
        assert jnp.allclose(result[1:-1, :, :], expected, atol=0.15)

    def test_is_jax_differentiable(self, grid_3d):
        from diffheat.operators.advection import advection_3d
        T = jnp.ones((grid_3d.nx, grid_3d.ny, grid_3d.nz))
        u_x = jnp.ones_like(T)
        u_y = jnp.zeros_like(T)
        u_z = jnp.zeros_like(T)

        def sum_adv(T):
            return jnp.sum(advection_3d(T, u_x, u_y, u_z, grid_3d.dx, grid_3d.dy, grid_3d.dz))

        grad = jax.grad(sum_adv)(T)
        assert grad.shape == T.shape
```

- [ ] **Step 2: Run new tests to verify they fail**

```bash
cd /home/samu2505/ENTERPRISE/Cooling-with-heat && uv run pytest tests/test_advection_operators.py::TestAdvection3D -v
```
Expected: All fail with `ImportError: cannot import name 'advection_3d'`

- [ ] **Step 3: Write advection_3d implementation**

Append to `diffheat/operators/advection.py`:

```python
def advection_3d(
    T: jnp.ndarray,
    u_x: jnp.ndarray,
    u_y: jnp.ndarray,
    u_z: jnp.ndarray,
    dx: float | jnp.ndarray,
    dy: float | jnp.ndarray,
    dz: float | jnp.ndarray,
) -> jnp.ndarray:
    """Compute -(u_x*dT/dx + u_y*dT/dy + u_z*dT/dz) using first-order upwind.

    Upwinding is applied independently in x, y, and z.

    Args:
        T: (nx, ny, nz) temperature field at cell centers.
        u_x: (nx, ny, nz) x-velocity field.
        u_y: (nx, ny, nz) y-velocity field.
        u_z: (nx, ny, nz) z-velocity field.
        dx: x cell width. Scalar or (nx,) array.
        dy: y cell width. Scalar or (ny,) array.
        dz: z cell width. Scalar or (nz,) array.

    Returns:
        (nx, ny, nz) advection term at cell centers.
    """
    # --- x-direction upwind ---
    _dx = dx[:, jnp.newaxis, jnp.newaxis] if isinstance(dx, jnp.ndarray) and dx.ndim >= 1 else dx
    T_forward_x = jnp.roll(T, -1, axis=0)
    T_backward_x = jnp.roll(T, 1, axis=0)
    forward_diff_x = (T_forward_x - T) / _dx
    backward_diff_x = (T - T_backward_x) / _dx
    adv_x = -u_x * jnp.where(u_x > 0, backward_diff_x, forward_diff_x)

    # --- y-direction upwind ---
    _dy = dy[jnp.newaxis, :, jnp.newaxis] if isinstance(dy, jnp.ndarray) and dy.ndim >= 1 else dy
    T_forward_y = jnp.roll(T, -1, axis=1)
    T_backward_y = jnp.roll(T, 1, axis=1)
    forward_diff_y = (T_forward_y - T) / _dy
    backward_diff_y = (T - T_backward_y) / _dy
    adv_y = -u_y * jnp.where(u_y > 0, backward_diff_y, forward_diff_y)

    # --- z-direction upwind ---
    _dz = dz[jnp.newaxis, jnp.newaxis, :] if isinstance(dz, jnp.ndarray) and dz.ndim >= 1 else dz
    T_forward_z = jnp.roll(T, -1, axis=2)
    T_backward_z = jnp.roll(T, 1, axis=2)
    forward_diff_z = (T_forward_z - T) / _dz
    backward_diff_z = (T - T_backward_z) / _dz
    adv_z = -u_z * jnp.where(u_z > 0, backward_diff_z, forward_diff_z)

    return adv_x + adv_y + adv_z
```

- [ ] **Step 4: Run all advection tests**

```bash
cd /home/samu2505/ENTERPRISE/Cooling-with-heat && uv run pytest tests/test_advection_operators.py -v
```
Expected: All 15 tests PASS (6 1D + 5 2D + 4 3D)

- [ ] **Step 5: Commit**

```bash
git add diffheat/operators/advection.py tests/test_advection_operators.py
git commit -m "feat: add advection_3d operator with first-order upwind scheme

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Export Advection Operators

**Files:**
- Modify: `diffheat/operators/__init__.py`

**Interfaces:**
- Consumes: `advection_1d`, `advection_2d`, `advection_3d` (Tasks 1-3)
- Produces: Public imports from `diffheat.operators`

- [ ] **Step 1: Add exports to operators __init__.py**

Edit `diffheat/operators/__init__.py`:

```python
# diffheat/operators/__init__.py
"""Discrete differential operators for finite difference PDEs."""
from .advection import advection_1d, advection_2d, advection_3d
from .divergence import divergence_2d, divergence_3d
from .gradient import (
    gradient_1d,
    gradient_2d,
    gradient_3d,
    gradient_x,
    gradient_x3d,
    gradient_y,
    gradient_y3d,
    gradient_z3d,
)
from .laplacian import laplacian_1d, laplacian_2d, laplacian_3d, make_laplacian

__all__ = [
    "make_laplacian",
    "laplacian_1d",
    "laplacian_2d",
    "laplacian_3d",
    "gradient_1d",
    "gradient_x",
    "gradient_y",
    "gradient_2d",
    "gradient_x3d",
    "gradient_y3d",
    "gradient_z3d",
    "gradient_3d",
    "divergence_2d",
    "divergence_3d",
    "advection_1d",
    "advection_2d",
    "advection_3d",
]
```

- [ ] **Step 2: Verify imports work**

```bash
cd /home/samu2505/ENTERPRISE/Cooling-with-heat && uv run python -c "from diffheat.operators import advection_1d, advection_2d, advection_3d; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add diffheat/operators/__init__.py
git commit -m "feat: export advection operators from diffheat.operators

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: CFL Conditions for Advection-Diffusion

**Files:**
- Modify: `diffheat/solvers/stability.py`
- Create: `tests/test_advection_diffusion.py`

**Interfaces:**
- Consumes: `Grid1D`, `Grid2D`, `Grid3D` (existing mesh), `check_cfl`, `check_cfl_2d`, `check_cfl_3d` (existing stability)
- Produces: `check_cfl_advection_diffusion_1d`, `check_cfl_advection_diffusion_2d`, `check_cfl_advection_diffusion_3d`

- [ ] **Step 1: Write failing CFL tests**

Create `tests/test_advection_diffusion.py`:

```python
# tests/test_advection_diffusion.py
"""Tests for advection-diffusion solvers and CFL conditions."""
import jax.numpy as jnp
import pytest
from diffheat.mesh import Grid1D, Grid2D, Grid3D


class TestCFLAdvectionDiffusion1D:
    @pytest.fixture
    def grid(self):
        return Grid1D.uniform(length=1.0, n_cells=50)

    def test_stable_dt_passes(self, grid):
        from diffheat.solvers.stability import check_cfl_advection_diffusion_1d
        alpha = 0.01
        u_max = 1.0
        dx_min = float(jnp.min(grid.dx))
        dt_diff = dx_min**2 / (2 * alpha)
        dt_adv = dx_min / u_max
        dt_max = min(dt_diff, dt_adv)
        assert check_cfl_advection_diffusion_1d(grid, alpha, u_max, 0.9 * dt_max)

    def test_unstable_diffusive_fails(self, grid):
        from diffheat.solvers.stability import check_cfl_advection_diffusion_1d
        alpha = 0.01
        u_max = 0.0  # pure diffusion
        dx_min = float(jnp.min(grid.dx))
        dt_limit = dx_min**2 / (2 * alpha)
        assert not check_cfl_advection_diffusion_1d(grid, alpha, u_max, 2.0 * dt_limit)

    def test_unstable_advective_fails(self, grid):
        from diffheat.solvers.stability import check_cfl_advection_diffusion_1d
        alpha = 0.0  # pure advection
        u_max = 2.0
        dx_min = float(jnp.min(grid.dx))
        dt_limit = dx_min / u_max
        # When alpha=0, diffusion limit is infinite; advection binds
        # Actually check_cfl_advection_diffusion handles alpha=0 via a large dt_diff
        if dt_limit > 0:
            assert not check_cfl_advection_diffusion_1d(grid, alpha, u_max, 2.0 * dt_limit)

    def test_zero_velocity_matches_heat_cfl(self, grid):
        from diffheat.solvers.stability import check_cfl_advection_diffusion_1d, check_cfl
        alpha = 0.01
        dt = 0.001
        assert check_cfl_advection_diffusion_1d(grid, alpha, 0.0, dt) == check_cfl(grid, alpha, dt)


class TestCFLAdvectionDiffusion2D:
    @pytest.fixture
    def grid(self):
        return Grid2D.uniform(Lx=1.0, Ly=1.0, nx=40, ny=40)

    def test_stable_dt_passes(self, grid):
        from diffheat.solvers.stability import check_cfl_advection_diffusion_2d
        alpha = 0.01
        u_x_max = 1.0
        u_y_max = 0.5
        dx_min = float(jnp.min(grid.dx))
        dy_min = float(jnp.min(grid.dy))
        dt_diff = min(dx_min**2, dy_min**2) / (4 * alpha)
        dt_adv = 1.0 / (u_x_max / dx_min + u_y_max / dy_min)
        dt_max = min(dt_diff, dt_adv)
        assert check_cfl_advection_diffusion_2d(grid, alpha, u_x_max, u_y_max, 0.9 * dt_max)

    def test_unstable_dt_fails(self, grid):
        from diffheat.solvers.stability import check_cfl_advection_diffusion_2d
        alpha = 0.01
        u_x_max = 10.0
        u_y_max = 10.0
        assert not check_cfl_advection_diffusion_2d(grid, alpha, u_x_max, u_y_max, 0.1)

    def test_zero_velocity_matches_heat_cfl(self, grid):
        from diffheat.solvers.stability import check_cfl_advection_diffusion_2d, check_cfl_2d
        alpha = 0.01
        dt = 0.001
        assert check_cfl_advection_diffusion_2d(grid, alpha, 0.0, 0.0, dt) == check_cfl_2d(grid, alpha, dt)


class TestCFLAdvectionDiffusion3D:
    @pytest.fixture
    def grid(self):
        return Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=20, ny=20, nz=20)

    def test_stable_dt_passes(self, grid):
        from diffheat.solvers.stability import check_cfl_advection_diffusion_3d
        alpha = 0.01
        u_max = 1.0
        dx_min = float(jnp.min(grid.dx))
        dy_min = float(jnp.min(grid.dy))
        dz_min = float(jnp.min(grid.dz))
        dt_diff = min(dx_min**2, dy_min**2, dz_min**2) / (6 * alpha)
        dt_adv = 1.0 / (u_max / dx_min + u_max / dy_min + u_max / dz_min)
        dt_max = min(dt_diff, dt_adv)
        assert check_cfl_advection_diffusion_3d(grid, alpha, u_max, u_max, u_max, 0.9 * dt_max)

    def test_zero_velocity_matches_heat_cfl(self, grid):
        from diffheat.solvers.stability import check_cfl_advection_diffusion_3d, check_cfl_3d
        alpha = 0.01
        dt = 0.0001
        assert check_cfl_advection_diffusion_3d(grid, alpha, 0.0, 0.0, 0.0, dt) == check_cfl_3d(grid, alpha, dt)
```

- [ ] **Step 2: Run CFL tests to verify they fail**

```bash
cd /home/samu2505/ENTERPRISE/Cooling-with-heat && uv run pytest tests/test_advection_diffusion.py::TestCFL -v
```
Expected: All fail with `ImportError`

- [ ] **Step 3: Write CFL functions**

Append to `diffheat/solvers/stability.py`:

```python
# ---------------------------------------------------------------------------
# Advection-diffusion CFL conditions
# ---------------------------------------------------------------------------

def check_cfl_advection_diffusion_1d(
    grid: Grid1D, alpha: float | jnp.ndarray, u_max: float, dt: float
) -> bool:
    """Check if dt satisfies the combined advection-diffusion CFL condition.

    dt <= min(dx^2 / (2*alpha), dx / |u|_max)

    Args:
        grid: The 1D spatial grid.
        alpha: Thermal diffusivity.
        u_max: Maximum absolute velocity |u|_max.
        dt: Time step size.

    Returns:
        True if stable, False otherwise.
    """
    alpha_max = float(jnp.max(jnp.asarray(alpha)))
    dx_min = float(jnp.min(grid.dx))

    dt_diff = dx_min ** 2 / (2.0 * alpha_max) if alpha_max > 0 else float("inf")
    dt_adv = dx_min / u_max if u_max > 0 else float("inf")
    dt_max = min(dt_diff, dt_adv)
    return bool(dt <= dt_max)


def check_cfl_advection_diffusion_2d(
    grid: Grid2D,
    alpha: float | jnp.ndarray,
    u_x_max: float,
    u_y_max: float,
    dt: float,
) -> bool:
    """Check if dt satisfies the combined 2D advection-diffusion CFL condition.

    dt <= min(min(dx^2, dy^2) / (4*alpha), 1 / (|u_x|_max/dx + |u_y|_max/dy))

    Args:
        grid: The 2D spatial grid.
        alpha: Thermal diffusivity.
        u_x_max: Maximum absolute x-velocity.
        u_y_max: Maximum absolute y-velocity.
        dt: Time step size.

    Returns:
        True if stable, False otherwise.
    """
    alpha_max = float(jnp.max(jnp.asarray(alpha)))
    dx_min = float(jnp.min(grid.dx))
    dy_min = float(jnp.min(grid.dy))

    dt_diff = min(dx_min**2, dy_min**2) / (4.0 * alpha_max) if alpha_max > 0 else float("inf")

    if u_x_max > 0 or u_y_max > 0:
        dt_adv = 1.0 / (u_x_max / dx_min + u_y_max / dy_min)
    else:
        dt_adv = float("inf")

    dt_max = min(dt_diff, dt_adv)
    return bool(dt <= dt_max)


def check_cfl_advection_diffusion_3d(
    grid: Grid3D,
    alpha: float | jnp.ndarray,
    u_x_max: float,
    u_y_max: float,
    u_z_max: float,
    dt: float,
) -> bool:
    """Check if dt satisfies the combined 3D advection-diffusion CFL condition.

    dt <= min(min(dx^2, dy^2, dz^2) / (6*alpha),
              1 / (|u_x|_max/dx + |u_y|_max/dy + |u_z|_max/dz))

    Args:
        grid: The 3D spatial grid.
        alpha: Thermal diffusivity.
        u_x_max: Maximum absolute x-velocity.
        u_y_max: Maximum absolute y-velocity.
        u_z_max: Maximum absolute z-velocity.
        dt: Time step size.

    Returns:
        True if stable, False otherwise.
    """
    alpha_max = float(jnp.max(jnp.asarray(alpha)))
    dx_min = float(jnp.min(grid.dx))
    dy_min = float(jnp.min(grid.dy))
    dz_min = float(jnp.min(grid.dz))

    dt_diff = min(dx_min**2, dy_min**2, dz_min**2) / (6.0 * alpha_max) if alpha_max > 0 else float("inf")

    if u_x_max > 0 or u_y_max > 0 or u_z_max > 0:
        dt_adv = 1.0 / (u_x_max / dx_min + u_y_max / dy_min + u_z_max / dz_min)
    else:
        dt_adv = float("inf")

    dt_max = min(dt_diff, dt_adv)
    return bool(dt <= dt_max)
```

- [ ] **Step 4: Run CFL tests to verify they pass**

```bash
cd /home/samu2505/ENTERPRISE/Cooling-with-heat && uv run pytest tests/test_advection_diffusion.py -v -k "CFL"
```
Expected: 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add diffheat/solvers/stability.py tests/test_advection_diffusion.py
git commit -m "feat: add combined advection-diffusion CFL conditions for 1D/2D/3D

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: AdvectionDiffusion Physics Dataclasses

**Files:**
- Create: `diffheat/physics/advection_diffusion.py`

**Interfaces:**
- Consumes: `Grid1D`, `Grid2D`, `Grid3D`, `BoundaryCondition`, `BoundaryCondition2D`, `BoundaryCondition3D` (existing mesh)
- Produces: `AdvectionDiffusion1D`, `AdvectionDiffusion2D`, `AdvectionDiffusion3D`

- [ ] **Step 1: Write failing physics tests**

Append to `tests/test_advection_diffusion.py`:

```python
class TestAdvectionDiffusionPhysics:
    def test_1d_creation(self):
        from diffheat.physics.advection_diffusion import AdvectionDiffusion1D
        from diffheat.mesh import Grid1D, BoundaryCondition
        import jax.numpy as jnp

        grid = Grid1D.uniform(length=1.0, n_cells=50)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([100.0, 0.0]))

        def velocity(x, t):
            return jnp.ones_like(x)

        eqn = AdvectionDiffusion1D(grid=grid, bc=bc, alpha=0.01, velocity=velocity)
        assert eqn.alpha == 0.01
        assert eqn.grid is grid
        assert eqn.source is None

    def test_1d_negative_alpha_raises(self):
        from diffheat.physics.advection_diffusion import AdvectionDiffusion1D
        from diffheat.mesh import Grid1D, BoundaryCondition
        import jax.numpy as jnp

        grid = Grid1D.uniform(length=1.0, n_cells=50)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([100.0, 0.0]))

        def velocity(x, t):
            return jnp.ones_like(x)

        with pytest.raises(ValueError, match="alpha must be positive"):
            AdvectionDiffusion1D(grid=grid, bc=bc, alpha=-0.01, velocity=velocity)

    def test_1d_with_source(self):
        from diffheat.physics.advection_diffusion import AdvectionDiffusion1D
        from diffheat.mesh import Grid1D, BoundaryCondition
        import jax.numpy as jnp

        grid = Grid1D.uniform(length=1.0, n_cells=50)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([100.0, 0.0]))

        def velocity(x, t):
            return jnp.ones_like(x)

        def source(x, t):
            return jnp.exp(-((x - 0.5) ** 2) / 0.01)

        eqn = AdvectionDiffusion1D(grid=grid, bc=bc, alpha=0.01, velocity=velocity, source=source)
        assert eqn.source is not None

    def test_2d_creation(self):
        from diffheat.physics.advection_diffusion import AdvectionDiffusion2D
        from diffheat.mesh import Grid2D, BoundaryCondition2D
        import jax.numpy as jnp

        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=40, ny=40)
        bc = BoundaryCondition2D(
            left={"kind": "dirichlet", "value": 1.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "neumann", "value": 0.0},
            top={"kind": "neumann", "value": 0.0},
        )

        def velocity(X, Y, t):
            return jnp.ones_like(X), jnp.zeros_like(Y)

        eqn = AdvectionDiffusion2D(grid=grid, bc=bc, alpha=0.01, velocity=velocity)
        assert eqn.alpha == 0.01

    def test_3d_creation(self):
        from diffheat.physics.advection_diffusion import AdvectionDiffusion3D
        from diffheat.mesh import Grid3D, BoundaryCondition3D
        import jax.numpy as jnp

        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=20, ny=20, nz=20)
        bc = BoundaryCondition3D(
            xmin={"kind": "dirichlet", "value": 1.0},
            xmax={"kind": "dirichlet", "value": 0.0},
            ymin={"kind": "neumann", "value": 0.0},
            ymax={"kind": "neumann", "value": 0.0},
            zmin={"kind": "neumann", "value": 0.0},
            zmax={"kind": "neumann", "value": 0.0},
        )

        def velocity(X, Y, Z, t):
            return jnp.ones_like(X), jnp.zeros_like(Y), jnp.zeros_like(Z)

        eqn = AdvectionDiffusion3D(grid=grid, bc=bc, alpha=0.01, velocity=velocity)
        assert eqn.alpha == 0.01
```

- [ ] **Step 2: Run physics tests to verify they fail**

```bash
cd /home/samu2505/ENTERPRISE/Cooling-with-heat && uv run pytest tests/test_advection_diffusion.py::TestAdvectionDiffusionPhysics -v
```
Expected: All fail with `ModuleNotFoundError`

- [ ] **Step 3: Write AdvectionDiffusion dataclasses**

Create `diffheat/physics/advection_diffusion.py`:

```python
# diffheat/physics/advection_diffusion.py
"""Advection-diffusion equation problem definitions."""
from dataclasses import dataclass
from typing import Callable, Optional

import jax.numpy as jnp

from ..mesh.boundary import BoundaryCondition, BoundaryCondition2D, BoundaryCondition3D
from ..mesh.grid1d import Grid1D
from ..mesh.grid2d import Grid2D
from ..mesh.grid3d import Grid3D
from ..utils import array


@dataclass(frozen=True)
class AdvectionDiffusion1D:
    """Complete 1D advection-diffusion equation problem definition.

    dT/dt = alpha * d^2T/dx^2 - u * dT/dx + S(x, t)

    Args:
        grid: 1D spatial grid.
        bc: Boundary conditions.
        alpha: Thermal diffusivity. Scalar or (N,) field.
        velocity: Velocity field u(x, t). Called as velocity(x_centers, t).
                  Returns (N,) velocity array.
        source: Optional source term S(x, t).
    """
    grid: Grid1D
    bc: BoundaryCondition
    alpha: float | jnp.ndarray
    velocity: Callable[[jnp.ndarray, float], jnp.ndarray]
    source: Optional[Callable[[jnp.ndarray, float], jnp.ndarray]] = None

    def __post_init__(self):
        if isinstance(self.alpha, (int, float)):
            if self.alpha <= 0:
                raise ValueError(f"alpha must be positive, got {self.alpha}")
            object.__setattr__(self, "alpha", array(float(self.alpha)))


@dataclass(frozen=True)
class AdvectionDiffusion2D:
    """Complete 2D advection-diffusion equation problem definition.

    dT/dt = alpha * nabla^2 T - u_x*dT/dx - u_y*dT/dy + S(x, y, t)

    Args:
        grid: 2D spatial grid.
        bc: Boundary conditions.
        alpha: Thermal diffusivity. Scalar or (nx, ny) field.
        velocity: Velocity field (u_x, u_y). Called as velocity(X, Y, t)
                  where X, Y are (nx, ny) arrays matching T.
                  Returns ((nx, ny), (nx, ny)) tuple.
        source: Optional source term S(x, y, t).
    """
    grid: Grid2D
    bc: BoundaryCondition2D
    alpha: float | jnp.ndarray
    velocity: Callable[[jnp.ndarray, jnp.ndarray, float], tuple[jnp.ndarray, jnp.ndarray]]
    source: Optional[Callable[[jnp.ndarray, jnp.ndarray, float], jnp.ndarray]] = None

    def __post_init__(self):
        if isinstance(self.alpha, (int, float)):
            if self.alpha <= 0:
                raise ValueError(f"alpha must be positive, got {self.alpha}")
            object.__setattr__(self, "alpha", array(float(self.alpha)))


@dataclass(frozen=True)
class AdvectionDiffusion3D:
    """Complete 3D advection-diffusion equation problem definition.

    dT/dt = alpha * nabla^2 T - u_x*dT/dx - u_y*dT/dy - u_z*dT/dz + S(x, y, z, t)

    Args:
        grid: 3D spatial grid.
        bc: Boundary conditions.
        alpha: Thermal diffusivity. Scalar or (nx, ny, nz) field.
        velocity: Velocity field (u_x, u_y, u_z). Called as velocity(X, Y, Z, t)
                  where X, Y, Z are (nx, ny, nz) arrays matching T.
                  Returns ((nx, ny, nz), (nx, ny, nz), (nx, ny, nz)) tuple.
        source: Optional source term S(x, y, z, t).
    """
    grid: Grid3D
    bc: BoundaryCondition3D
    alpha: float | jnp.ndarray
    velocity: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, float],
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ]
    source: Optional[Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, float], jnp.ndarray]] = None

    def __post_init__(self):
        if isinstance(self.alpha, (int, float)):
            if self.alpha <= 0:
                raise ValueError(f"alpha must be positive, got {self.alpha}")
            object.__setattr__(self, "alpha", array(float(self.alpha)))
```

- [ ] **Step 4: Run physics tests to verify they pass**

```bash
cd /home/samu2505/ENTERPRISE/Cooling-with-heat && uv run pytest tests/test_advection_diffusion.py::TestAdvectionDiffusionPhysics -v
```
Expected: 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add diffheat/physics/advection_diffusion.py tests/test_advection_diffusion.py
git commit -m "feat: add AdvectionDiffusion1D/2D/3D physics dataclasses

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: Solver Functions

**Files:**
- Create: `diffheat/solvers/advection_diffusion.py`

**Interfaces:**
- Consumes: `AdvectionDiffusion1D/2D/3D` (Task 6), `advection_1d/2d/3d` (Tasks 1-3), `laplacian_1d/2d/3d` (existing), `explicit_euler_step_1d/2d/3d` (existing), `check_cfl_advection_diffusion_1d/2d/3d` (Task 5), `apply_boundary_conditions_1d/2d/3d` (existing), `solve_1d/2d/3d` (existing)
- Produces: `solve_advection_diffusion_1d`, `solve_advection_diffusion_2d`, `solve_advection_diffusion_3d`

- [ ] **Step 1: Write failing solver tests**

Append to `tests/test_advection_diffusion.py`:

```python
class TestSolveAdvectionDiffusion1D:
    @pytest.fixture
    def grid(self):
        from diffheat.mesh import Grid1D
        return Grid1D.uniform(length=10.0, n_cells=200)

    def test_pure_diffusion_matches_heat_solver(self, grid):
        """With u=0, the advection-diffusion solver should match the heat solver."""
        import jax.numpy as jnp
        from diffheat import HeatEquation1D, solve_heat_1d, BoundaryCondition
        from diffheat.physics.advection_diffusion import AdvectionDiffusion1D
        from diffheat.solvers.advection_diffusion import solve_advection_diffusion_1d

        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([100.0, 0.0]))
        alpha = 0.01
        T0 = jnp.zeros(grid.n_cells)
        dt = 0.005
        t_span = (0.0, 0.5)

        def zero_velocity(x, t):
            return jnp.zeros_like(x)

        # Solve with advection-diffusion solver (u=0)
        eqn_adv = AdvectionDiffusion1D(grid=grid, bc=bc, alpha=alpha, velocity=zero_velocity)
        traj_adv = solve_advection_diffusion_1d(eqn_adv, T0, t_span, dt)

        # Solve with heat solver
        eqn_heat = HeatEquation1D(grid=grid, bc=bc, alpha=alpha)
        traj_heat = solve_heat_1d(eqn_heat, T0, t_span, dt)

        assert traj_adv.shape == traj_heat.shape
        assert jnp.allclose(traj_adv, traj_heat, atol=1e-6)

    def test_is_jax_differentiable(self, grid):
        """jax.grad should work through the full solve."""
        import jax.numpy as jnp
        from diffheat import BoundaryCondition
        from diffheat.physics.advection_diffusion import AdvectionDiffusion1D
        from diffheat.solvers.advection_diffusion import solve_advection_diffusion_1d
        import jax

        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([100.0, 0.0]))
        T0 = jnp.zeros(grid.n_cells)
        dt = 0.005
        t_span = (0.0, 0.1)

        def velocity(x, t):
            return jnp.ones_like(x)

        def final_mean_temp(alpha):
            eqn = AdvectionDiffusion1D(grid=grid, bc=bc, alpha=alpha, velocity=velocity)
            traj = solve_advection_diffusion_1d(eqn, T0, t_span, dt)
            return jnp.mean(traj[-1])

        grad_fn = jax.grad(final_mean_temp)
        sensitivity = grad_fn(0.01)
        assert sensitivity != 0.0


class TestSolveAdvectionDiffusion2D:
    @pytest.fixture
    def grid(self):
        from diffheat.mesh import Grid2D
        return Grid2D.uniform(Lx=2.0, Ly=1.0, nx=40, ny=20)

    def test_pure_diffusion_matches_heat_solver(self, grid):
        """With u=0 everywhere, result matches HeatEquation2D."""
        import jax.numpy as jnp
        from diffheat import HeatEquation2D, solve_heat_2d, BoundaryCondition2D
        from diffheat.physics.advection_diffusion import AdvectionDiffusion2D
        from diffheat.solvers.advection_diffusion import solve_advection_diffusion_2d

        bc = BoundaryCondition2D(
            left={"kind": "dirichlet", "value": 100.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "neumann", "value": 0.0},
            top={"kind": "neumann", "value": 0.0},
        )
        alpha = 0.01
        T0 = jnp.zeros((grid.nx, grid.ny))
        dt = 0.001
        t_span = (0.0, 0.1)

        def zero_velocity(X, Y, t):
            return jnp.zeros_like(X), jnp.zeros_like(Y)

        eqn_adv = AdvectionDiffusion2D(grid=grid, bc=bc, alpha=alpha, velocity=zero_velocity)
        traj_adv = solve_advection_diffusion_2d(eqn_adv, T0, t_span, dt)

        eqn_heat = HeatEquation2D(grid=grid, bc=bc, alpha=alpha)
        traj_heat = solve_heat_2d(eqn_heat, T0, t_span, dt)

        assert traj_adv.shape == traj_heat.shape
        assert jnp.allclose(traj_adv, traj_heat, atol=1e-6)

    def test_advection_bends_temperature_field(self, grid):
        """With horizontal flow, a hot spot should shift downstream over time."""
        import jax.numpy as jnp
        from diffheat import BoundaryCondition2D
        from diffheat.physics.advection_diffusion import AdvectionDiffusion2D
        from diffheat.solvers.advection_diffusion import solve_advection_diffusion_2d

        bc = BoundaryCondition2D(
            left={"kind": "dirichlet", "value": 0.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "neumann", "value": 0.0},
            top={"kind": "neumann", "value": 0.0},
        )
        alpha = 0.001  # small diffusion
        T0 = jnp.zeros((grid.nx, grid.ny))
        # Hot spot near left edge
        T0 = T0.at[5, grid.ny // 2].set(100.0)
        dt = 0.001
        t_span = (0.0, 0.2)

        def rightward_flow(X, Y, t):
            return 2.0 * jnp.ones_like(X), jnp.zeros_like(Y)

        eqn = AdvectionDiffusion2D(grid=grid, bc=bc, alpha=alpha, velocity=rightward_flow)
        traj = solve_advection_diffusion_2d(eqn, T0, t_span, dt)

        # Peak temperature at final time should be downstream of initial position
        final_T = traj[-1]
        peak_x_initial = 5
        peak_x_final = jnp.argmax(jnp.max(final_T, axis=1))
        assert peak_x_final > peak_x_initial, f"Expected peak to move right, but peak at x={peak_x_final}"

    def test_is_jax_differentiable(self, grid):
        import jax.numpy as jnp
        from diffheat import BoundaryCondition2D
        from diffheat.physics.advection_diffusion import AdvectionDiffusion2D
        from diffheat.solvers.advection_diffusion import solve_advection_diffusion_2d
        import jax

        bc = BoundaryCondition2D(
            left={"kind": "dirichlet", "value": 1.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "neumann", "value": 0.0},
            top={"kind": "neumann", "value": 0.0},
        )
        T0 = jnp.zeros((grid.nx, grid.ny))

        def velocity(X, Y, t):
            return jnp.ones_like(X), jnp.zeros_like(Y)

        def final_mean_temp(alpha):
            eqn = AdvectionDiffusion2D(grid=grid, bc=bc, alpha=alpha, velocity=velocity)
            traj = solve_advection_diffusion_2d(eqn, T0, (0.0, 0.05), 0.001)
            return jnp.mean(traj[-1])

        grad_fn = jax.grad(final_mean_temp)
        sensitivity = grad_fn(0.01)
        assert sensitivity != 0.0
```

- [ ] **Step 2: Run solver tests to verify they fail**

```bash
cd /home/samu2505/ENTERPRISE/Cooling-with-heat && uv run pytest tests/test_advection_diffusion.py::TestSolveAdvectionDiffusion -v
```
Expected: All fail with `ModuleNotFoundError`

- [ ] **Step 3: Write solver implementations**

Create `diffheat/solvers/advection_diffusion.py`:

```python
# diffheat/solvers/advection_diffusion.py
"""Advection-diffusion equation solvers using explicit Euler + jax.lax.scan."""
import logging

import jax
import jax.numpy as jnp

from ..mesh.boundary import (
    apply_boundary_conditions_1d,
    apply_boundary_conditions_2d,
    apply_boundary_conditions_3d,
)
from ..operators.advection import advection_1d, advection_2d, advection_3d
from ..operators.laplacian import laplacian_1d, laplacian_2d, laplacian_3d
from ..physics.advection_diffusion import (
    AdvectionDiffusion1D,
    AdvectionDiffusion2D,
    AdvectionDiffusion3D,
)
from .scan import solve_1d, solve_2d, solve_3d
from .stability import (
    check_cfl_advection_diffusion_1d,
    check_cfl_advection_diffusion_2d,
    check_cfl_advection_diffusion_3d,
)

_logger = logging.getLogger(__name__)


def solve_advection_diffusion_1d(
    eqn: AdvectionDiffusion1D,
    T0: jnp.ndarray,
    t_span: tuple[float, float],
    dt: float,
) -> jnp.ndarray:
    """Solve the 1D advection-diffusion equation with explicit Euler.

    Args:
        eqn: Advection-diffusion problem definition.
        T0: (N,) initial temperature field.
        t_span: (t_start, t_end) simulation time range.
        dt: Time step size.

    Returns:
        (n_steps+1, N) temperature trajectory. First frame is T0.
    """
    t0, t_end = t_span
    n_steps = int((t_end - t0) / dt)
    if n_steps < 1:
        raise ValueError(f"t_span too short for dt={dt}: {t_span}")

    # CFL check
    try:
        u = eqn.velocity(eqn.grid.centers, t0)
        u_max = float(jnp.max(jnp.abs(u)))
        if not check_cfl_advection_diffusion_1d(eqn.grid, eqn.alpha, u_max, dt):
            alpha_max = float(jnp.max(jnp.asarray(eqn.alpha)))
            dx_min = float(jnp.min(eqn.grid.dx))
            dt_diff = dx_min**2 / (2.0 * alpha_max) if alpha_max > 0 else float("inf")
            dt_adv = dx_min / u_max if u_max > 0 else float("inf")
            _logger.warning(
                f"dt={dt:.2e} exceeds CFL limit (diff={dt_diff:.2e}, adv={dt_adv:.2e}). "
                f"Solution may be unstable."
            )
    except jax.errors.ConcretizationTypeError:
        pass

    def rhs_fn(T, grid, t, params):
        u = eqn.velocity(grid.centers, t)
        L_T, b_source = apply_boundary_conditions_1d(
            lambda x: laplacian_1d(x, grid), grid, eqn.bc, T
        )
        dT_dt = eqn.alpha * (L_T + b_source)
        dT_dt = dT_dt + advection_1d(T, u, grid.dx)
        if eqn.source is not None:
            dT_dt = dT_dt + eqn.source(grid.centers, t)
        return dT_dt

    return solve_1d(rhs_fn, T0, eqn.grid, t_span, dt)


def solve_advection_diffusion_2d(
    eqn: AdvectionDiffusion2D,
    T0: jnp.ndarray,
    t_span: tuple[float, float],
    dt: float,
) -> jnp.ndarray:
    """Solve the 2D advection-diffusion equation with explicit Euler.

    Args:
        eqn: Advection-diffusion problem definition.
        T0: (nx, ny) initial temperature field.
        t_span: (t_start, t_end) simulation time range.
        dt: Time step size.

    Returns:
        (n_steps+1, nx, ny) temperature trajectory. First frame is T0.
    """
    t0, t_end = t_span
    n_steps = int((t_end - t0) / dt)
    if n_steps < 1:
        raise ValueError(f"t_span too short for dt={dt}: {t_span}")

    # CFL check — uses initial velocity field as representative
    try:
        u_x, u_y = eqn.velocity(eqn.grid.X.T, eqn.grid.Y.T, t0)
        u_x_max = float(jnp.max(jnp.abs(u_x)))
        u_y_max = float(jnp.max(jnp.abs(u_y)))
        if not check_cfl_advection_diffusion_2d(eqn.grid, eqn.alpha, u_x_max, u_y_max, dt):
            alpha_max = float(jnp.max(jnp.asarray(eqn.alpha)))
            dx_min = float(jnp.min(eqn.grid.dx))
            dy_min = float(jnp.min(eqn.grid.dy))
            dt_diff = min(dx_min**2, dy_min**2) / (4.0 * alpha_max) if alpha_max > 0 else float("inf")
            dt_adv = 1.0 / (u_x_max / dx_min + u_y_max / dy_min) if (u_x_max > 0 or u_y_max > 0) else float("inf")
            _logger.warning(
                f"dt={dt:.2e} exceeds CFL limit (diff={dt_diff:.2e}, adv={dt_adv:.2e}). "
                f"Solution may be unstable."
            )
    except jax.errors.ConcretizationTypeError:
        pass

    def rhs_fn(T, grid, t, params):
        u_x, u_y = eqn.velocity(grid.X.T, grid.Y.T, t)
        L_T, b_source = apply_boundary_conditions_2d(
            lambda x: laplacian_2d(x, grid), grid, eqn.bc, T
        )
        dT_dt = eqn.alpha * (L_T + b_source)
        dT_dt = dT_dt + advection_2d(T, u_x, u_y, grid.dx, grid.dy)
        if eqn.source is not None:
            dT_dt = dT_dt + eqn.source(grid.X.T, grid.Y.T, t)
        return dT_dt

    return solve_2d(rhs_fn, T0, eqn.grid, t_span, dt)


def solve_advection_diffusion_3d(
    eqn: AdvectionDiffusion3D,
    T0: jnp.ndarray,
    t_span: tuple[float, float],
    dt: float,
    save_every: int = 1,
) -> jnp.ndarray:
    """Solve the 3D advection-diffusion equation with explicit Euler.

    Args:
        eqn: Advection-diffusion problem definition.
        T0: (nx, ny, nz) initial temperature field.
        t_span: (t_start, t_end) simulation time range.
        dt: Time step size.
        save_every: Save a frame every this many steps.

    Returns:
        (n_saved+1, nx, ny, nz) temperature trajectory. First frame is T0.
    """
    t0, t_end = t_span
    n_steps = int((t_end - t0) / dt)
    if n_steps < 1:
        raise ValueError(f"t_span too short for dt={dt}: {t_span}")

    # CFL check
    try:
        u_x, u_y, u_z = eqn.velocity(eqn.grid.X, eqn.grid.Y, eqn.grid.Z, t0)
        u_x_max = float(jnp.max(jnp.abs(u_x)))
        u_y_max = float(jnp.max(jnp.abs(u_y)))
        u_z_max = float(jnp.max(jnp.abs(u_z)))
        if not check_cfl_advection_diffusion_3d(eqn.grid, eqn.alpha, u_x_max, u_y_max, u_z_max, dt):
            alpha_max = float(jnp.max(jnp.asarray(eqn.alpha)))
            dx_min = float(jnp.min(eqn.grid.dx))
            dy_min = float(jnp.min(eqn.grid.dy))
            dz_min = float(jnp.min(eqn.grid.dz))
            dt_diff = min(dx_min**2, dy_min**2, dz_min**2) / (6.0 * alpha_max) if alpha_max > 0 else float("inf")
            u_sum = u_x_max / dx_min + u_y_max / dy_min + u_z_max / dz_min
            dt_adv = 1.0 / u_sum if u_sum > 0 else float("inf")
            _logger.warning(
                f"dt={dt:.2e} exceeds CFL limit (diff={dt_diff:.2e}, adv={dt_adv:.2e}). "
                f"Solution may be unstable."
            )
    except jax.errors.ConcretizationTypeError:
        pass

    def rhs_fn(T, grid, t, params):
        u_x, u_y, u_z = eqn.velocity(grid.X, grid.Y, grid.Z, t)
        L_T, b_source = apply_boundary_conditions_3d(
            lambda x: laplacian_3d(x, grid), grid, eqn.bc, T
        )
        dT_dt = eqn.alpha * (L_T + b_source)
        dT_dt = dT_dt + advection_3d(T, u_x, u_y, u_z, grid.dx, grid.dy, grid.dz)
        if eqn.source is not None:
            dT_dt = dT_dt + eqn.source(grid.X, grid.Y, grid.Z, t)
        return dT_dt

    return solve_3d(rhs_fn, T0, eqn.grid, t_span, dt, save_every=save_every)
```

- [ ] **Step 4: Run all advection-diffusion tests**

```bash
cd /home/samu2505/ENTERPRISE/Cooling-with-heat && uv run pytest tests/test_advection_diffusion.py -v
```
Expected: All tests PASS (8 CFL + 6 physics + 5 solver = 19 tests)

- [ ] **Step 5: Verify existing tests still pass**

```bash
cd /home/samu2505/ENTERPRISE/Cooling-with-heat && uv run pytest tests/ -v --ignore=tests/test_advection_operators.py --ignore=tests/test_advection_diffusion.py
```
Expected: All existing tests PASS (no regressions)

- [ ] **Step 6: Commit**

```bash
git add diffheat/solvers/advection_diffusion.py tests/test_advection_diffusion.py
git commit -m "feat: add solve_advection_diffusion_1d/2d/3d solvers

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: Export New Public API

**Files:**
- Modify: `diffheat/physics/__init__.py`
- Modify: `diffheat/solvers/__init__.py`
- Modify: `diffheat/__init__.py`

**Interfaces:**
- Consumes: All new classes and functions from Tasks 1-7
- Produces: Public API accessible from `diffheat` and subpackages

- [ ] **Step 1: Update physics __init__.py**

Edit `diffheat/physics/__init__.py`:

```python
# diffheat/physics/__init__.py
"""Physical problem definitions."""
from ..operators.laplacian import make_laplacian
from .advection_diffusion import (
    AdvectionDiffusion1D,
    AdvectionDiffusion2D,
    AdvectionDiffusion3D,
)
from .bessel import (
    bessel_j_zero,
    eigenfunction_disc,
    eigenfunction_norm,
    eigenvalue_disc,
)
from .heat1d import HeatEquation1D, apply_boundary_conditions
from .heat2d import HeatEquation2D
from .heat3d import HeatEquation3D
from .wave import WaveEquation1D, WaveEquation2D, WaveEquation3D
from .telegrapher import TelegrapherEquation1D, TelegrapherEquation2D, TelegrapherEquation3D

__all__ = [
    "HeatEquation1D",
    "HeatEquation2D",
    "HeatEquation3D",
    "WaveEquation1D",
    "WaveEquation2D",
    "WaveEquation3D",
    "TelegrapherEquation1D",
    "TelegrapherEquation2D",
    "TelegrapherEquation3D",
    "AdvectionDiffusion1D",
    "AdvectionDiffusion2D",
    "AdvectionDiffusion3D",
    "apply_boundary_conditions",
    "make_laplacian",
    "bessel_j_zero",
    "eigenvalue_disc",
    "eigenfunction_disc",
    "eigenfunction_norm",
]
```

- [ ] **Step 2: Update solvers __init__.py**

Edit `diffheat/solvers/__init__.py`:

```python
# diffheat/solvers/__init__.py
"""Time integration solvers."""
from .advection_diffusion import (
    solve_advection_diffusion_1d,
    solve_advection_diffusion_2d,
    solve_advection_diffusion_3d,
)
from .eigen import (
    find_first_eigenvalue_1d,
    find_first_eigenvalue_2d,
    find_first_eigenvalue_3d,
    rayleigh_quotient_1d,
    rayleigh_quotient_2d,
    rayleigh_quotient_3d,
    rayleigh_upper_bounds_1d,
    rayleigh_upper_bounds_2d,
    rayleigh_upper_bounds_3d,
)
from .explicit import (
    explicit_euler_step,
    explicit_euler_step_1d,
    explicit_euler_step_2d,
    explicit_euler_step_3d,
)
from .inhomogeneous import (
    solve_heat_inhomogeneous_2d,
    solve_heat_inhomogeneous_3d,
)
from .scan import (
    solve_1d,
    solve_2d,
    solve_3d,
    solve_heat_1d,
    solve_heat_2d,
    solve_heat_3d,
)
from .stability import (
    check_cfl,
    check_cfl_2d,
    check_cfl_3d,
    check_cfl_wave_1d,
    check_cfl_wave_2d,
    check_cfl_wave_3d,
    check_cfl_telegrapher_1d,
    check_cfl_telegrapher_2d,
    check_cfl_telegrapher_3d,
    check_cfl_advection_diffusion_1d,
    check_cfl_advection_diffusion_2d,
    check_cfl_advection_diffusion_3d,
)
from .steady_state import (
    solve_steady_state_1d,
    solve_steady_state_2d,
    solve_steady_state_3d,
)
from .wave import solve_wave_1d, solve_wave_2d, solve_wave_3d
from .polar import (
    solve_heat_disc_analytical,
    solve_steady_state_disc,
    solve_steady_state_cylinder_3d,
)
from .telegrapher import solve_telegrapher_1d, solve_telegrapher_2d, solve_telegrapher_3d

__all__ = [
    "explicit_euler_step",
    "explicit_euler_step_1d",
    "explicit_euler_step_2d",
    "explicit_euler_step_3d",
    "solve_heat_1d",
    "solve_heat_2d",
    "solve_heat_3d",
    "solve_1d",
    "solve_2d",
    "solve_3d",
    "solve_wave_1d",
    "solve_wave_2d",
    "solve_wave_3d",
    "solve_telegrapher_1d",
    "solve_telegrapher_2d",
    "solve_telegrapher_3d",
    "solve_advection_diffusion_1d",
    "solve_advection_diffusion_2d",
    "solve_advection_diffusion_3d",
    "check_cfl",
    "check_cfl_2d",
    "check_cfl_3d",
    "check_cfl_wave_1d",
    "check_cfl_wave_2d",
    "check_cfl_wave_3d",
    "check_cfl_telegrapher_1d",
    "check_cfl_telegrapher_2d",
    "check_cfl_telegrapher_3d",
    "check_cfl_advection_diffusion_1d",
    "check_cfl_advection_diffusion_2d",
    "check_cfl_advection_diffusion_3d",
    "solve_steady_state_1d",
    "solve_steady_state_2d",
    "solve_steady_state_3d",
    "solve_heat_inhomogeneous_2d",
    "solve_heat_inhomogeneous_3d",
    "rayleigh_quotient_1d",
    "rayleigh_quotient_2d",
    "rayleigh_quotient_3d",
    "rayleigh_upper_bounds_1d",
    "rayleigh_upper_bounds_2d",
    "rayleigh_upper_bounds_3d",
    "find_first_eigenvalue_1d",
    "find_first_eigenvalue_2d",
    "find_first_eigenvalue_3d",
    "solve_heat_disc_analytical",
    "solve_steady_state_disc",
    "solve_steady_state_cylinder_3d",
]
```

- [ ] **Step 3a: Update top-level __init__.py — physics imports**

Edit `diffheat/__init__.py` lines 36-51. Replace:

```python
from .physics import (
    HeatEquation1D,
    HeatEquation2D,
    HeatEquation3D,
    WaveEquation1D,
    WaveEquation2D,
    WaveEquation3D,
    TelegrapherEquation1D,
    TelegrapherEquation2D,
    TelegrapherEquation3D,
    apply_boundary_conditions,
    bessel_j_zero,
    eigenfunction_disc,
    eigenfunction_norm,
    eigenvalue_disc,
)
```

With:

```python
from .physics import (
    AdvectionDiffusion1D,
    AdvectionDiffusion2D,
    AdvectionDiffusion3D,
    HeatEquation1D,
    HeatEquation2D,
    HeatEquation3D,
    WaveEquation1D,
    WaveEquation2D,
    WaveEquation3D,
    TelegrapherEquation1D,
    TelegrapherEquation2D,
    TelegrapherEquation3D,
    apply_boundary_conditions,
    bessel_j_zero,
    eigenfunction_disc,
    eigenfunction_norm,
    eigenvalue_disc,
)
```

- [ ] **Step 3b: Update top-level __init__.py — operators imports**

Edit `diffheat/__init__.py` lines 16-29. Replace:

```python
from .operators import (
    divergence_2d,
    divergence_3d,
    gradient_1d,
    gradient_2d,
    gradient_3d,
    gradient_x3d,
    gradient_y3d,
    gradient_z3d,
    laplacian_1d,
    laplacian_2d,
    laplacian_3d,
    make_laplacian,
)
```

With:

```python
from .operators import (
    advection_1d,
    advection_2d,
    advection_3d,
    divergence_2d,
    divergence_3d,
    gradient_1d,
    gradient_2d,
    gradient_3d,
    gradient_x3d,
    gradient_y3d,
    gradient_z3d,
    laplacian_1d,
    laplacian_2d,
    laplacian_3d,
    make_laplacian,
)
```

- [ ] **Step 3c: Update top-level __init__.py — solvers imports**

Edit `diffheat/__init__.py` lines 52-95. Replace the solvers import block with:

```python
from .solvers import (
    check_cfl,
    check_cfl_2d,
    check_cfl_3d,
    check_cfl_advection_diffusion_1d,
    check_cfl_advection_diffusion_2d,
    check_cfl_advection_diffusion_3d,
    check_cfl_telegrapher_1d,
    check_cfl_telegrapher_2d,
    check_cfl_telegrapher_3d,
    check_cfl_wave_1d,
    check_cfl_wave_2d,
    check_cfl_wave_3d,
    explicit_euler_step,
    explicit_euler_step_1d,
    explicit_euler_step_2d,
    explicit_euler_step_3d,
    find_first_eigenvalue_1d,
    find_first_eigenvalue_2d,
    find_first_eigenvalue_3d,
    rayleigh_quotient_1d,
    rayleigh_quotient_2d,
    rayleigh_quotient_3d,
    rayleigh_upper_bounds_1d,
    rayleigh_upper_bounds_2d,
    rayleigh_upper_bounds_3d,
    solve_1d,
    solve_2d,
    solve_3d,
    solve_advection_diffusion_1d,
    solve_advection_diffusion_2d,
    solve_advection_diffusion_3d,
    solve_heat_1d,
    solve_heat_2d,
    solve_heat_3d,
    solve_heat_disc_analytical,
    solve_heat_inhomogeneous_2d,
    solve_heat_inhomogeneous_3d,
    solve_steady_state_1d,
    solve_steady_state_2d,
    solve_steady_state_3d,
    solve_steady_state_cylinder_3d,
    solve_steady_state_disc,
    solve_telegrapher_1d,
    solve_telegrapher_2d,
    solve_telegrapher_3d,
    solve_wave_1d,
    solve_wave_2d,
    solve_wave_3d,
)
```

- [ ] **Step 3d: Update top-level __init__.py — __all__ list**

Edit `diffheat/__init__.py`. Add the following entries to the `__all__` list (insert alphabetically within each section):

After `# Operators — 1D` section add:
```python
    # Operators — Advection
    "advection_1d",
    "advection_2d",
    "advection_3d",
```

After `# Physics — Heat` section add:
```python
    # Physics — Advection-Diffusion
    "AdvectionDiffusion1D",
    "AdvectionDiffusion2D",
    "AdvectionDiffusion3D",
```

After `# Solvers` sections add:
```python
    # Solvers — Advection-Diffusion
    "check_cfl_advection_diffusion_1d",
    "check_cfl_advection_diffusion_2d",
    "check_cfl_advection_diffusion_3d",
    "solve_advection_diffusion_1d",
    "solve_advection_diffusion_2d",
    "solve_advection_diffusion_3d",
```

- [ ] **Step 4: Verify full import chain**

```bash
cd /home/samu2505/ENTERPRISE/Cooling-with-heat && uv run python -c "
from diffheat import (
    AdvectionDiffusion1D,
    AdvectionDiffusion2D,
    AdvectionDiffusion3D,
    advection_1d, advection_2d, advection_3d,
    solve_advection_diffusion_1d,
    solve_advection_diffusion_2d,
    solve_advection_diffusion_3d,
    check_cfl_advection_diffusion_1d,
    check_cfl_advection_diffusion_2d,
    check_cfl_advection_diffusion_3d,
)
print('All imports successful')
"
```
Expected: `All imports successful`

- [ ] **Step 5: Run full test suite**

```bash
cd /home/samu2505/ENTERPRISE/Cooling-with-heat && uv run pytest tests/ -v
```
Expected: All tests PASS (existing + new)

- [ ] **Step 6: Commit**

```bash
git add diffheat/physics/__init__.py diffheat/solvers/__init__.py diffheat/__init__.py
git commit -m "feat: export advection-diffusion API from diffheat and subpackages

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: Demo — Example 09 Forced Convection

**Files:**
- Create: `examples/09-forced-convection/demo.py`

**Interfaces:**
- Consumes: All new advection-diffusion API (Task 8), `run_viewer_2d` (existing viz)
- Produces: Runnable demo script

- [ ] **Step 1: Write demo script**

Create `examples/09-forced-convection/demo.py`:

```python
#!/usr/bin/env python3
"""Demo: 2D forced convection — channel flow over a heated component.

A 2D channel with uniform inlet flow from the left at T_cold. A heated chip
sits on the bottom wall at center. The thermal plume bends in the flow direction,
illustrating the transition from diffusion-dominated to advection-dominated
heat transfer.

Controls:
    - Adjust flow speed to see the Peclet number effect
    - Watch the thermal plume bend downstream

Run:
    python examples/09-forced-convection/demo.py
"""
import jax.numpy as jnp

from diffheat import (
    AdvectionDiffusion2D,
    BoundaryCondition2D,
    Grid2D,
    check_cfl_advection_diffusion_2d,
    get_device,
    solve_advection_diffusion_2d,
)
from diffheat.viz import run_viewer_2d


def main():
    print(f"Running on: {get_device()}")
    print("=" * 50)

    # --- Channel geometry ---
    Lx, Ly = 4.0, 1.0  # 4:1 aspect ratio channel
    nx, ny = 120, 30
    grid = Grid2D.uniform(Lx=Lx, Ly=Ly, nx=nx, ny=ny)
    dx = float(grid.dx[0])
    dy = float(grid.dy[0])
    print(f"Grid: {nx}×{ny} cells, dx = {dx:.4f}, dy = {dy:.4f}")

    # --- Material ---
    alpha = 0.01  # thermal diffusivity

    # --- Flow ---
    inlet_velocity = 1.0  # m/s, uniform inlet from left
    peclet = inlet_velocity * Lx / alpha
    print(f"Inlet velocity: {inlet_velocity} m/s")
    print(f"Peclet number (Pe = UL/alpha): {peclet:.1f}")

    # --- Boundary conditions ---
    bc = BoundaryCondition2D(
        left={"kind": "dirichlet", "value": 0.0},     # cold inlet
        right={"kind": "dirichlet", "value": 0.0},     # cold outlet (convective)
        bottom={"kind": "neumann", "value": 0.0},      # insulated bottom
        top={"kind": "neumann", "value": 0.0},         # insulated top
    )

    # --- Velocity field: uniform horizontal flow ---
    def channel_flow(X, Y, t):
        """Uniform horizontal flow through the channel."""
        u_x = inlet_velocity * jnp.ones_like(X)
        u_y = jnp.zeros_like(Y)
        return u_x, u_y

    # --- Heated chip source at bottom center ---
    # Gaussian heat source centered at (Lx/2, 0)
    chip_x0 = Lx / 2
    chip_y0 = 0.0
    chip_width = 0.1

    def chip_source(X, Y, t):
        """Gaussian heat source — heated component on bottom wall."""
        r2 = ((X - chip_x0) ** 2 + (Y - chip_y0) ** 2) / (2 * chip_width ** 2)
        return 500.0 * jnp.exp(-r2)

    eqn = AdvectionDiffusion2D(
        grid=grid,
        bc=bc,
        alpha=alpha,
        velocity=channel_flow,
        source=chip_source,
    )

    # --- Initial condition ---
    T0 = jnp.zeros((nx, ny))

    # --- Time parameters ---
    t_end = 2.0
    dt = 0.002

    # CFL check
    u_x_max = inlet_velocity
    u_y_max = 0.0
    stable = check_cfl_advection_diffusion_2d(grid, alpha, u_x_max, u_y_max, dt)
    print(f"dt: {dt:.4f} s (stable: {stable})")

    # --- Solve ---
    print(f"Solving from t=0 to t={t_end}...")
    trajectory = solve_advection_diffusion_2d(eqn, T0, (0.0, t_end), dt)

    n_steps = len(trajectory)
    print(f"Done. {n_steps} timesteps computed.")
    print(f"Initial max T: {float(jnp.max(trajectory[0])):.2f}°C")
    print(f"Final max T:   {float(jnp.max(trajectory[-1])):.2f}°C")
    print(f"Pe = {peclet:.1f} ({'advection-dominated' if peclet > 10 else 'diffusion-dominated' if peclet < 1 else 'mixed'})")

    # --- Visualize ---
    print("\nLaunching viewer...")
    run_viewer_2d(trajectory, grid, dt)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify demo runs (headless — just the solve, no viz)**

```bash
cd /home/samu2505/ENTERPRISE/Cooling-with-heat && uv run python -c "
import jax.numpy as jnp
from diffheat import (
    AdvectionDiffusion2D, BoundaryCondition2D, Grid2D,
    solve_advection_diffusion_2d, check_cfl_advection_diffusion_2d,
)

grid = Grid2D.uniform(Lx=4.0, Ly=1.0, nx=60, ny=15)
bc = BoundaryCondition2D(
    left={'kind': 'dirichlet', 'value': 0.0},
    right={'kind': 'dirichlet', 'value': 0.0},
    bottom={'kind': 'neumann', 'value': 0.0},
    top={'kind': 'neumann', 'value': 0.0},
)

def flow(X, Y, t):
    return jnp.ones_like(X), jnp.zeros_like(Y)

def source(X, Y, t):
    r2 = ((X - 2.0)**2 + Y**2) / (2 * 0.1**2)
    return 500.0 * jnp.exp(-r2)

eqn = AdvectionDiffusion2D(grid=grid, bc=bc, alpha=0.01, velocity=flow, source=source)
T0 = jnp.zeros((60, 15))
traj = solve_advection_diffusion_2d(eqn, T0, (0.0, 0.5), 0.002)
print(f'Trajectory shape: {traj.shape}')
print(f'Final max T: {float(jnp.max(traj[-1])):.2f}')
# Peak should be downstream of x=2.0 (chip center)
peak_x = jnp.argmax(jnp.max(traj[-1], axis=1))
print(f'Peak at x-index: {peak_x} (initial chip at x-index: {30})')
assert peak_x > 30, f'Expected peak downstream of chip, got x-index {peak_x}'
print('OK: thermal plume bends downstream')
"
```
Expected: `OK: thermal plume bends downstream`

- [ ] **Step 3: Commit**

```bash
git add examples/09-forced-convection/demo.py
git commit -m "feat: add forced convection demo (example 09)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 10: Final Verification

- [ ] **Step 1: Run full test suite**

```bash
cd /home/samu2505/ENTERPRISE/Cooling-with-heat && uv run pytest tests/ -v
```
Expected: All tests PASS with no regressions.

- [ ] **Step 2: Verify differentiability end-to-end**

```bash
cd /home/samu2505/ENTERPRISE/Cooling-with-heat && uv run python -c "
import jax
import jax.numpy as jnp
from diffheat import (
    AdvectionDiffusion2D, BoundaryCondition2D, Grid2D,
    solve_advection_diffusion_2d,
)

grid = Grid2D.uniform(Lx=2.0, Ly=1.0, nx=30, ny=15)
bc = BoundaryCondition2D(
    left={'kind': 'dirichlet', 'value': 1.0},
    right={'kind': 'dirichlet', 'value': 0.0},
    bottom={'kind': 'neumann', 'value': 0.0},
    top={'kind': 'neumann', 'value': 0.0},
)

def flow(X, Y, t):
    return jnp.ones_like(X), jnp.zeros_like(Y)

T0 = jnp.zeros((30, 15))

def loss(alpha):
    eqn = AdvectionDiffusion2D(grid=grid, bc=bc, alpha=alpha, velocity=flow)
    traj = solve_advection_diffusion_2d(eqn, T0, (0.0, 0.1), 0.002)
    return jnp.mean(traj[-1])

grad = jax.grad(loss)(0.01)
print(f'd(loss)/d(alpha) at alpha=0.01: {float(grad):.6f}')
assert grad != 0.0, 'Gradient should be non-zero'
print('OK: end-to-end differentiability verified')
"
```
Expected: `OK: end-to-end differentiability verified`

- [ ] **Step 3: Run existing examples to verify no regressions**

```bash
cd /home/samu2505/ENTERPRISE/Cooling-with-heat && uv run python examples/02-2d-heat-equation/demo.py 2>&1 | head -5
```
Expected: Normal output, no import errors.

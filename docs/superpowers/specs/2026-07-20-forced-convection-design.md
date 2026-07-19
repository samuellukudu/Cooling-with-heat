# Forced Convection (Advection-Diffusion) — Design Specification

**Date:** 2026-07-20
**Status:** In Review
**Goal:** Add forced convection (advection-diffusion) PDE support to diffheat in 1D, 2D, and 3D, following the established pattern of HeatEquation, WaveEquation, and TelegrapherEquation — with upwind-stabilized advection operators, combined CFL conditions, and end-to-end JAX differentiability.

---

## 1. Philosophy

- **Follow existing patterns.** `AdvectionDiffusion1D/2D/3D` mirror `HeatEquation1D/2D/3D` exactly. Every new PDE gets its own physics module, operator, solver, and tests.
- **Upwind by default.** Central differences for advection are unconditionally unstable for advection-dominated flows. The advection operator always uses first-order upwinding. Higher-order flux limiter schemes (TVD, WENO) are out of scope for v0.1.
- **Prescribed velocity.** The velocity field `u(x, t)` is a user-provided callable — no Navier-Stokes coupling yet. This keeps the scope tight and the solver independently verifiable.
- **No existing-file modifications** except `stability.py` (new CFL functions) and `__init__.py` files (new exports).
- **Differentiability is non-negotiable.** `jax.grad` must flow through the advection operator, the combined RHS, and the full trajectory.

---

## 2. PDE

```
∂T/∂t = α ∇²T - u·∇T + S
```

- α: thermal diffusivity (scalar or field)
- u: prescribed velocity field (callable: (X, t) → u or (X, Y, t) → (u_x, u_y) or (X, Y, Z, t) → (u_x, u_y, u_z))
- S: optional source term

Dimensions:
- 1D: `∂T/∂t = α ∂²T/∂x² - u ∂T/∂x + S`
- 2D: `∂T/∂t = α ∇²T - u_x ∂T/∂x - u_y ∂T/∂y + S`
- 3D: `∂T/∂t = α ∇²T - u_x ∂T/∂x - u_y ∂T/∂y - u_z ∂T/∂z + S`

---

## 3. File Layout

New files only (no modifications to existing code except stability.py and __init__.py exports):

```
diffheat/
├── operators/
│   ├── advection.py           # NEW: advection_1d, advection_2d, advection_3d
│   └── __init__.py            # MODIFIED: export advection operators
├── physics/
│   ├── advection_diffusion.py # NEW: AdvectionDiffusion1D/2D/3D
│   └── __init__.py            # MODIFIED: export new classes
├── solvers/
│   ├── advection_diffusion.py # NEW: solve_advection_diffusion_1d/2d/3d
│   ├── stability.py           # MODIFIED: check_cfl_advection_diffusion_1d/2d/3d
│   └── __init__.py            # MODIFIED: export new solvers + CFL
└── tests/
    ├── test_advection_operators.py     # NEW
    └── test_advection_diffusion.py     # NEW
examples/
└── 09-forced-convection/       # NEW
    └── demo.py
```

**Dependency chain:**

```
solvers/advection_diffusion.py
  → physics/advection_diffusion.py
    → operators/advection.py
    → operators/laplacian.py (existing)
  → solvers/explicit.py (existing euler_step)
  → solvers/stability.py
```

---

## 4. Advection Operator Design

### 4.1 Upwind Scheme

For each cell `i`, the advection term `-u ∂T/∂x` is discretized as:

```
if u_{i} > 0:
    advection[i] = -u[i] * (T[i] - T[i-1]) / dx    # backward difference
else:
    advection[i] = -u[i] * (T[i+1] - T[i]) / dx    # forward difference
```

In 2D, upwinding is applied independently in x and y:
```
advection[i,j] = -u_x[i,j] * (upwind_x) - u_y[i,j] * (upwind_y)
```

Same pattern extends to 3D. This guarantees stability for pure advection at any CFL ≤ 1.

### 4.2 Function Signatures

```python
def advection_1d(T: jnp.ndarray, u: jnp.ndarray, dx: float) -> jnp.ndarray:
    """Compute -u * dT/dx using first-order upwind."""

def advection_2d(T: jnp.ndarray, u_x: jnp.ndarray, u_y: jnp.ndarray,
                 dx: float, dy: float) -> jnp.ndarray:
    """Compute -(u_x * dT/dx + u_y * dT/dy) using first-order upwind."""

def advection_3d(T: jnp.ndarray, u_x: jnp.ndarray, u_y: jnp.ndarray,
                 u_z: jnp.ndarray, dx: float, dy: float, dz: float) -> jnp.ndarray:
    """Compute -(u_x * dT/dx + u_y * dT/dy + u_z * dT/dz) using first-order upwind."""
```

All operators are pure JAX functions — no classes, no state, fully JIT-compatible.

---

## 5. AdvectionDiffusion Dataclass

Follows the exact pattern of `HeatEquation1D/2D/3D`:

```python
@dataclass(frozen=True)
class AdvectionDiffusion1D:
    grid: Grid1D
    bc: BoundaryCondition
    alpha: float | jnp.ndarray
    velocity: Callable[[jnp.ndarray, float], jnp.ndarray]
    # velocity(x, t) -> u(x)
    source: Optional[Callable[[jnp.ndarray, float], jnp.ndarray]] = None

@dataclass(frozen=True)
class AdvectionDiffusion2D:
    grid: Grid2D
    bc: BoundaryCondition2D
    alpha: float | jnp.ndarray
    velocity: Callable[[jnp.ndarray, jnp.ndarray, float], tuple[jnp.ndarray, jnp.ndarray]]
    # velocity(X, Y, t) -> (u_x, u_y)
    source: Optional[Callable[[jnp.ndarray, jnp.ndarray, float], jnp.ndarray]] = None

@dataclass(frozen=True)
class AdvectionDiffusion3D:
    grid: Grid3D
    bc: BoundaryCondition3D
    alpha: float | jnp.ndarray
    velocity: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, float],
                        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]
    # velocity(X, Y, Z, t) -> (u_x, u_y, u_z)
    source: Optional[Callable] = None
```

`__post_init__` validates alpha > 0 and converts scalar alpha to JAX array (matching existing pattern).

Velocity is a callable to support:
- Steady flow: `velocity = lambda X, Y, t: (u_const_x, u_const_y)`
- Time-varying flow: `velocity = lambda X, Y, t: (u_x * sin(t), u_y)`
- Spatially-varying: arbitrary functions of X, Y

---

## 6. CFL Condition

The combined stability limit is the minimum of the diffusive and advective constraints:

**1D:**
```
dt_diff = dx² / (2 * alpha)
dt_adv  = dx / max(|u|)
dt_max  = min(dt_diff, dt_adv)
```

**2D:**
```
dt_diff = min(dx², dy²) / (4 * alpha)    -- matches existing check_cfl_2d
dt_adv  = 1 / (max(|u_x|)/dx + max(|u_y|)/dy)
dt_max  = min(dt_diff, dt_adv)
```

**3D:** same pattern with 1/dz² and max(|u_z|)/dz terms.

The CFL checker returns `bool` — consistent with the existing `check_cfl`, `check_cfl_wave_*`, and `check_cfl_telegrapher_*` functions.

```python
def check_cfl_advection_diffusion_2d(grid, alpha, u_x, u_y, dt) -> bool:
    """Returns True if dt satisfies both diffusive and advective CFL constraints."""
```

---

## 7. Solver

Follows the existing `solve_heat_*` pattern — uses `jax.lax.scan` over explicit Euler steps:

```python
def solve_advection_diffusion_2d(eqn: AdvectionDiffusion2D, T0, t_span, dt):
    """Solve with explicit Euler. Returns (trajectory, steps)."""
```

Each step:
1. Compute `d2T = laplacian_2d(T, dx, dy)`
2. Evaluate `u_x, u_y = eqn.velocity(grid.X, grid.Y, t)`
3. Compute `adv = advection_2d(T, u_x, u_y, dx, dy)`
4. Compute source if present
5. `dT_dt = alpha * d2T + adv + source`
6. `T_next = T + dt * dT_dt`
7. Apply boundary conditions

CFL check runs before the scan; a warning is logged if unstable (but execution proceeds — matching existing behavior).

---

## 8. Testing Strategy

### 8.1 Operator Tests (`test_advection_operators.py`)

| Test | Description |
|------|-------------|
| Pure translation 1D | Gaussian pulse at constant u; exact solution is translation. Verify L2 error. |
| Pure translation 2D | Gaussian pulse in uniform flow; verify L2 error and direction. |
| Upwind vs central 1D | Show central differences diverge for advection-dominated case, upwind stays stable. |
| Sign handling 1D | Verify forward vs backward difference selection based on u sign at each cell. |
| Zero velocity | advection_* returns zeros when u = 0 everywhere. |
| JAX differentiable | `jax.grad` through advection_2d with respect to T. |
| 3D consistency | Pure translation in each axis direction independently. |

### 8.2 Physics + Solver Tests (`test_advection_diffusion.py`)

| Test | Description |
|------|-------------|
| Pure diffusion limit | u = 0: result matches HeatEquation solve exactly. |
| Pure advection limit | alpha ≈ 0: Gaussian translates without spreading (within upwind numerical diffusion). |
| Steady-state balance | Constant u, α: solve to steady state, verify dT/dt ≈ 0 everywhere. |
| Manufactured solution | T_exact(x,t) = sin(x - u*t) * exp(-α*t) with matching source term. Verify convergence. |
| CFL rejection | dt above combined limit → `check_cfl` returns False. |
| CFL acceptance | dt below combined limit → `check_cfl` returns True. |
| Differentiability | `jax.grad(final_mean_temp)(alpha)` works and gives nonzero result. |
| Boundary conditions | Dirichlet BCs correctly enforced at each timestep. |
| 1D/2D/3D parity | Same physical problem in all dimensions gives consistent results. |

### 8.3 Integration Tests

| Test | Description |
|------|-------------|
| Channel flow | 2D channel with parabolic inlet profile over heated component; verify thermal plume bends downstream. |
| End-to-end grad | Optimize inlet velocity to minimize max temperature (gradient-based). |

---

## 9. Demo: Example 09 — Forced Convection

**Scenario:** 2D channel (aspect ratio 4:1), uniform inlet flow from left at T_cold, heated chip on bottom wall at center, convective outlet at right, insulated top/bottom elsewhere.

**Visualization:** Side-by-side heatmaps showing temperature field evolving over time. The thermal plume should visibly bend in the flow direction. Controls for adjusting flow speed to show transition from diffusion-dominated to advection-dominated regimes.

**Péclet number annotation:** Display Pe = UL/α to show which regime is active.

---

## 10. Out of Scope (v0.2+)

- Navier-Stokes coupling (Boussinesq natural convection)
- Higher-order advection schemes (TVD, WENO, MUSCL)
- SUPG/streamline diffusion stabilization
- Unstructured mesh advection
- Compressible flow effects
- Temperature-dependent velocity (two-way coupling)

# diffheat — Differentiable Heat Equation Simulations with JAX

`diffheat` is a differentiable 1D heat-equation solver that is fully compatible with
JAX's automatic differentiation.  Because gradients flow through every timestep,
you can **optimise material properties, initial conditions, boundary values, and
source terms** with standard gradient-descent tools — making it useful for inverse
design, parameter identification, and "cooling-with-heat" explorations where
thermal behaviour needs to be tuned rather than just simulated.

## What can it do?

| Capability | Detail |
|---|---|
| **Forward simulation** | 1D heat equation with Dirichlet or Neumann boundary conditions, spatially-varying diffusivity, and user-defined source terms. |
| **End-to-end differentiability** | The full trajectory is traced through JAX — `jax.grad`, `jax.jit`, and `jax.vmap` all work out of the box. |
| **Stability checks** | Built-in CFL condition checking so you don't accidentally run unstable explicit-Euler integrations. |
| **Interactive visualisation** | Optional PyQt6 + Matplotlib viewer with play/pause, stepping, and a space-time heatmap. |
| **Headless core** | The solver and physics modules have zero GUI dependencies — safe for HPC, CI, and containerised workloads. |

### Architecture

```
diffheat/
├── __init__.py     # Public API surface
├── utils.py        # Device detection, array helpers
├── mesh.py         # Grid1D, BoundaryCondition
├── physics.py      # Laplacian operator, BC application, HeatEquation1D
├── solvers.py      # Explicit Euler, CFL check, trajectory solve (jax.lax.scan)
└── viz.py          # PyQt6 viewer (optional, only module with GUI deps)
```

## Installation

### Prerequisites

- **Python ≥ 3.10**

Pick one of the environment managers below.  All three approaches work; `uv`
is the fastest and what we use day-to-day, while `conda` is convenient when
you also need Jupyter and GPU-accelerated JAX in the same environment.

---

### Option A: uv (recommended)

[`uv`](https://docs.astral.sh/uv/) is a fast Python package and project
manager written in Rust — it replaces `pip`, `venv`, and `pip-tools` with a
single tool.

```bash
# Install uv if you don't have it yet
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and enter the project
git clone <repo-url> diffheat && cd diffheat

# uv creates a virtualenv, installs JAX + diffheat + all extras in one step
uv sync --extra viz --extra dev
```

`uv sync` reads `pyproject.toml` and pins dependencies in `uv.lock` for
reproducible installs.  To add a new dependency later:

```bash
uv add <package>
```

**JAX hardware backend with uv:**  uv respects the same `[cpu]` / `[cuda12]`
extras that pip does.  If you need to pin a specific JAX variant, add it
explicitly:

```bash
uv add "jax[cuda12]"
```

Run any script inside the managed environment with:

```bash
uv run python examples/01-1d-heat-equation/demo.py
uv run pytest
uv run jupyter notebook examples/01-1d-heat-equation/explore.ipynb
```

---

### Option B: Conda

Conda is a good choice when you need Jupyter and GPU-accelerated JAX in a
single environment, or when you already work inside the conda ecosystem.

```bash
# Create a fresh environment with Python 3.12
conda create -n diffheat python=3.12 -y
conda activate diffheat

# Install JAX for your hardware
pip install "jax[cpu]"        # CPU-only
# pip install "jax[cuda12]"   # GPU (CUDA 12)

# Install diffheat in editable mode with all extras
pip install -e ".[viz,dev]"
```

To make the kernel available in Jupyter:

```bash
conda activate diffheat
python -m ipykernel install --user --name diffheat --display-name "Python (diffheat)"
```

Now you can select the *Python (diffheat)* kernel when launching notebooks:

```bash
jupyter notebook examples/01-1d-heat-equation/explore.ipynb
```

> **Tip:**  If you use Mamba as a drop-in faster conda, replace `conda` with
> `mamba` in the commands above — everything else is identical.

---

### Option C: pip + venv (standard)

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows

# JAX for your hardware
pip install "jax[cpu]"

# diffheat with all extras
pip install -e ".[viz,dev]"
```

---

### Dependency groups

| Extra | What it installs | When you need it |
|---|---|---|
| *(none)* | `jax`, `numpy` | Headless solves, CI, HPC |
| `viz` | `pyqt6`, `matplotlib` | Interactive viewer (`run_viewer`) |
| `dev` | `pytest`, `jupyter` | Running tests, notebooks |

## Quickstart

```python
import jax.numpy as jnp
from diffheat import (
    BoundaryCondition,
    Grid1D,
    HeatEquation1D,
    get_device,
    solve_heat_1d,
)

print(f"Running on: {get_device()}")

# ---- 1 m rod, 100 cells ----
grid = Grid1D.uniform(length=1.0, n_cells=100)

# ---- Left end at 100°C, right end at 0°C (Dirichlet) ----
bc = BoundaryCondition(kind="dirichlet", value=jnp.array([100.0, 0.0]))

# ---- Thermal diffusivity (m²/s) ----
alpha = 0.01

# ---- Initial condition: 0°C everywhere ----
T0 = jnp.zeros(grid.n_cells)

# ---- Solve: 5 seconds with dt = 0.001 ----
eqn = HeatEquation1D(grid=grid, bc=bc, alpha=alpha)
trajectory = solve_heat_1d(eqn, T0, t_span=(0.0, 5.0), dt=0.001)

print(f"Shape: {trajectory.shape}")   # (5001, 100)
print(f"Final mean temperature: {jnp.mean(trajectory[-1]):.2f}°C")
```

### Visualise the result

```python
from diffheat.viz import run_viewer

run_viewer(trajectory, grid, dt=0.001)
```

![viewer screenshot — space-time heatmap + snapshot + transport controls]

### Differentiate through the solver

Because the solver is pure JAX, you can compute gradients with respect to
any input:

```python
import jax

def final_temperature(alpha):
    """Mean temperature at the end of the simulation, as a function of alpha."""
    eqn = HeatEquation1D(grid=grid, bc=bc, alpha=alpha)
    traj = solve_heat_1d(eqn, T0, t_span=(0.0, 5.0), dt=0.001)
    return jnp.mean(traj[-1])

# d(mean_T) / d(alpha)
grad_fn = jax.grad(final_temperature)
sensitivity = grad_fn(0.01)
print(f"∂T̄/∂α at α=0.01: {sensitivity:.4f}")
```

### Adding a source term

```python
# Heat generated at the centre of the rod decays over time
def gaussian_source(x, t):
    return 50.0 * jnp.exp(-((x - 0.5) ** 2) / 0.01) * jnp.exp(-t)

eqn = HeatEquation1D(grid=grid, bc=bc, alpha=alpha, source=gaussian_source)
trajectory = solve_heat_1d(eqn, T0, t_span=(0.0, 5.0), dt=0.001)
```

## Running the demo

```bash
python examples/01-1d-heat-equation/demo.py
```

This simulates a 1 m rod with the left boundary held at 1°C and the right at
0°C, then opens the interactive viewer so you can scrub through the temperature
evolution.

## Interactive exploration

```bash
jupyter notebook examples/01-1d-heat-equation/explore.ipynb
```

The notebook walks through grid setup, CFL stability, boundary condition
effects, and gradient verification — useful for building intuition before
tackling inverse problems.

## Running tests

```bash
pytest
```

## Why "diffheat"?

The name is a double reference: **differentiable heat**, and also a nod toward
the thermal management problems this library was built to explore — using
controlled heat flow (diffusion) to solve engineering cooling challenges.

## License

[Add your license here]

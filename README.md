# Cooling-with-heat — deep learning for heat-driven cooling

Train neural networks (physics-informed and RL) to solve, control, and design
**heat-driven adsorption cooling** — air conditioning powered by waste heat
instead of electricity.

> **Status (2026-09): simplified to a DL focus.** The project was reduced to
> two code efforts — `harness/` (physics + gym environments + optimizers) and
> `adsorbent-ml/` (API-fed data pipeline + models). The frozen `diffheat`
> solver library, a generic PDE "trial zoo", and the legacy `Materials/`
> screening effort were removed from the tree (the GUI lives parked in
> [`attic/`](attic)). See [`ROADMAP.md`](ROADMAP.md) for what's next.

## Layout

| Path | Role |
|---|---|
| [`harness/`](harness) | **Physics + environments + optimization.** Pure-JAX cooling physics (equilibrium cycle oracle, 1-D adsorber bed with LDF kinetics + Dubinin–Astakhov uptake, two-bed system with heat recovery, LiBr/H₂O absorption chiller), gymnasium-compatible environments, literature calibration, and three optimization backends: gradient (optax), search (CMA-ES/TPE), and **RL (PPO via stable-baselines3)**. |
| [`adsorbent-ml/`](adsorbent-ml) | **Data + models.** API-fed acquisition (NIST ISODB, CoRE MOF, QMOF, IZA-SC, OPTIMADE, Materials Project, MOFSimplify), Dubinin–Astakhov isotherm fitting, feature engineering, the **bed PINN** (JAX physics-informed neural operator surrogate of Bed1D), a tabular baseline, training CLIs, and eval (COP-ranked retrieval, PINN field metrics). |
| `attic/` | Parked, restorable work (the PyQt6 GUI workbench). |
| `data_cache/` | All downloaded/derived datasets (gitignored; every build writes a `manifest.json`). |

```
materials (q_sat, Q_st, D–A params)
        │  adsorbent-ml: predict from structure/composition
        ▼
harness physics ──► COP / SCP per application profile
        ▲
        │  harness RL/grad/search backends: cycle design + control
```

**The model proposes; the simulator disposes.** ML predictions are always
judged by the trusted JAX physics (cycle COP for ranking, Bed1D fields for the
PINN), never by held-out ML metrics alone.

## Quickstart

```bash
uv sync --group dev                 # jax[cpu] + harness + tooling
JAX_PLATFORMS=cpu uv run pytest     # everything green before you start
```

Optional GPU JAX (the physics is float64 and the dev GPU is a shared 4 GB
laptop card, so memory hygiene is automatic via `harness.gpu`: no XLA
preallocation, freed VRAM returns to the driver, OOMs fail with a remedy,
jitted caches are dropped after each GUI job):

```bash
uv sync --group dev --group gpu     # adds the nvidia CUDA-12 wheels
python -m harness.gpu               # sanity: prints jax devices
```

Optimize a cycle with the RL backend:

```python
import harness

env = harness.make("Bed1D-v0", material="anchor:Silica gel RD", profile="datacenter")
result = harness.optimize(env, harness.Objective.single("COP"), backend="grad")
print(result.best_metrics["COP"])
# backend="rl" (PPO) / backend="search" (CMA-ES, TPE) on the same problem
```

Train the bed PINN surrogate:

```bash
cd adsorbent-ml
uv run ../.venv/bin/python training/train_pinn.py --steps 2000   # see --help
```

Rebuild datasets from source APIs (each stage caches + writes a manifest):

```bash
cd adsorbent-ml/data
# isodb clone → water isotherms; CoRE/QMOF/IZA/OPTIMADE exporters — see ACQUISITION.md
```

## Docs

- [`ROADMAP.md`](ROADMAP.md) — DL-focused milestones and status.
- [`harness/DESIGN.md`](harness/DESIGN.md) — physics, env specs, backends, milestones.
- [`harness/benchmarks.md`](harness/benchmarks.md) — V4 literature calibration (Uyun 2009, Sztekler 2021).
- [`adsorbent-ml/README.md`](adsorbent-ml/README.md) + [`adsorbent-ml/data/ACQUISITION.md`](adsorbent-ml/data/ACQUISITION.md) — data sources, APIs, per-source status.

## Provenance notes

- `harness.physics.cycle0d` / `harness.physics.thermo` are exact JAX mirrors of
  the original NumPy oracle (`tests/harness/reference/cooling_physics.py`),
  pinned by 1,000+ parity cases at < 1e-12.
- `adsorbent-ml/data/mp_screen.py` is vendored from the archived legacy
  screening effort; the full `Materials/` tree lives outside the repo
  (`~/ENTERPRISE/_archive/Cooling-with-heat-Materials`).

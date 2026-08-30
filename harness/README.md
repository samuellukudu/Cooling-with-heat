# harness

> **A differentiable optimization harness for heat-driven cooling, with
> gym-compatible environments.**
> RL is one of three optimization backends — not the identity of the project.
> That framing keeps us honest about which problems need which solver, and
> makes the project legible to people who (rightly) raise an eyebrow at
> "we used RL" on a 3-parameter physics function.

## What it is

One loop that connects the physics (`Materials/cooling_physics.py` lineage),
the materials database (`adsorbent-ml/` fitted isotherms), and the four
application profiles (`cpu`, `human`, `vehicle`, `datacenter`) — and runs
experiments on it:

```
env = (physics model) × (material) × (application profile)
          │  jittable JAX rollout  +  gym-compatible API
          ├── grad backend      (jax.grad through the episode)
          ├── search backend    (CMA-ES / Optuna)
          └── rl backend        (PPO, optional extra — only for schedule
                                 control under time-varying profiles)
                ⇒  optimal algorithms (schedules) · materials (shortlists) ·
                   designs (geometry / operating parameters)
```

Three environments, one oracle:

| env | physics | gives you |
|---|---|---|
| `Cycle0D-v0` | equilibrium cycle (port of `simulate_adsorption_cycle`) | the oracle & test bench |
| `Bed1D-v0` | transient 1-D adsorber bed: heat equation + LDF kinetics + Dubinin–Astakhov | control & design optimization with real dynamics |
| `TwoBed-v0` | two beds + valves + heat recovery + time-varying source profiles | the schedule problem where sequence actually matters |

## Status

**H0–H1.5 implemented (2026-08).** In place: the package skeleton; the
`Cycle0D-v0` oracle with exact parity against `Materials/cooling_physics.py`
(V1 < 1e-12); the `grad` and `search` backends with the V7 acceptance test;
the **dynamic 1-D bed** (`physics/bed1d.py` — RK4-in-scan, exact-exponential
LDF substep, ghost-cell wall BC) with its V2 conservation gates; the
`Bed1D-v0` env (§5.2 obs/action/reward) passing `gymnasium.check_env`; the
**V3 oracle-limit keystone** — the dynamic bed reproduces the frozen
equilibrium oracle within 2 % (COP gap 1.55 %, SCP 0.15 %); **V4 literature
calibration** (`calibration.py` + [`benchmarks.md`](benchmarks.md): two
open-access experimental rigs, standard points calibrated, cycle-time ↔ SCP
trend reproduced, the T_hs-trend gap attributed to vapour-side dynamics);
and **H1.5 control experiments** (`Bed1DControls`, V5 gradient gate,
[`control_notebook.ipynb`](control_notebook.ipynb), the Open Question 3
decision: hard-valve gradients are blind — soft switching + the
search-on-switch-times hybrid is the recipe). Import-linter contracts
enforced. Next: **H2** (two-bed system, datacenter profile, ISODB sweep) —
see [`DESIGN.md`](DESIGN.md) §12.

## Target API

```python
import harness

env = harness.make("Bed1D-v0",
                   material="anchors:Silica gel RD",
                   profile="datacenter")

result = harness.optimize(env, backend="grad",
                          design={"bed_thickness_m": 2e-4,
                                  "switch_time_s": 300.0})
result.metrics   # {"COP": ..., "SCP_W_kg": ...}
result.trace     # per-step diagnostics
```

## Extension rule

New capability = **one adapter + one data contract**, never a core change:
new physics via `SimulatorAdapter` (external simulators — Modelica, TESPy, …),
new optimizers via the `Backend` protocol, new materials via rows in a
parameter table, new applications via registered profiles. See `DESIGN.md` §7.

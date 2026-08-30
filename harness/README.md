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

**H0 implemented (2026-08).** The package skeleton, the `Cycle0D-v0` oracle
with exact parity against `Materials/cooling_physics.py` (V1 < 1e-12), the
`grad` and `search` backends, and the V7 acceptance test (harness optimum
reproduces the legacy screen's best point) are in place — 1,333 tests green,
import-linter contracts enforced. Next: **H1, the dynamic 1-D bed** — the
spec is in [`DESIGN.md`](DESIGN.md) §4.2 and §12.

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

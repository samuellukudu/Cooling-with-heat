# harness

> **A differentiable optimization harness for heat-driven cooling, with
> gym-compatible environments.**
> Three optimization backends — gradient, search, and RL — against one
> trusted physics core. That framing keeps us honest about which problems
> need which solver, and makes the project legible to people who (rightly)
> raise an eyebrow at "we used RL" on a 3-parameter physics function.

## What it is

One loop that connects the physics (the vendored V1 oracle lineage,
[`tests/harness/reference/cooling_physics.py`](../tests/harness/reference/cooling_physics.py)),
the materials database (`adsorbent-ml/` fitted isotherms), and any operating
scenario — a preset application profile (`cpu`, `human`, `vehicle`,
`datacenter`) or raw setpoints/schedules passed directly — and runs
experiments on it:

```
env = (physics model) × (material) × (operating scenario)
          │  jittable JAX rollout  +  gym-compatible API
          ├── grad backend      (jax.grad through the episode)
          ├── search backend    (CMA-ES / Optuna)
          └── rl backend        (PPO, optional extra — for schedule
                                 control under time-varying profiles)
                ⇒  optimal algorithms (schedules) · materials (shortlists) ·
                   designs (geometry / operating parameters)
```

The posture is an **R&D lab for heat-driven cooling**: pose an idea, configure
an experiment against any scenario, get honest numbers back. Application
profiles are preset scenarios for cross-tool comparability, not the frame.

Environments and the oracle:

| env | physics | gives you |
|---|---|---|
| `Cycle0D-v0` | equilibrium cycle (port of `simulate_adsorption_cycle`) | the oracle & test bench |
| `Bed1D-v0` | transient 1-D adsorber bed: heat equation + LDF kinetics + Dubinin–Astakhov | control & design optimization with real dynamics; the PINN's physics oracle |
| `TwoBed-v0` | counter-phase bed pair + request-bit valves + film-disconnect heat recovery | the system question: recovery, composite beds, duty continuity |
| `TwoBedSchedule-v0` | TwoBed under a per-step source series with policy parameters | schedule experiments: gating, source-following, recovery windows — the RL surface |
| `AbsorptionCycle-v0` | lumped single-effect absorption chiller, 3-T ideal × pair effectiveness | design + materials discovery: generator T, working-pair shortlists |

## Status

**H0–H2.3 implemented (2026-08); simplified to the cooling core (2026-09).**
In place: the package skeleton; the `Cycle0D-v0` oracle with exact parity
against the vendored `cooling_physics.py` reference (V1 < 1e-12); the `grad`
and `search` backends with the V7 acceptance test; the **dynamic 1-D bed**
(`physics/bed1d.py` — RK4-in-scan, exact-exponential LDF substep, ghost-cell
wall BC) with its V2 conservation gates; the `Bed1D-v0` env passing
`gymnasium.check_env`; the **V3 oracle-limit keystone** — the dynamic bed
reproduces the frozen equilibrium oracle within 2 % (COP gap 1.55 %, SCP
0.15 %); **V4 literature calibration** (`calibration.py` +
[`benchmarks.md`](benchmarks.md): two open-access experimental rigs, standard
points calibrated, cycle-time ↔ SCP trend reproduced, the T_hs-trend gap
attributed to vapour-side dynamics); **H1.5 control experiments**
(`Bed1DControls`, V5 gradient gate,
[`control_notebook.ipynb`](control_notebook.ipynb), the Open Question 3
decision: hard-valve gradients are blind — soft switching + the
search-on-switch-times hybrid is the recipe); the **two-bed system**
(`physics/system.py` + `TwoBed-v0`, V6: heat recovery raises COP
monotonically with exactly-conserving books, duty continuity, equilibrium
limit); the **schedule gate** (`TwoBedSchedule-v0`: optimized schedules beat
the fixed schedule ≈ 7 % under a varying source); and the **T2 reference
rankings** ([`rank.py`](rank.py) +
[`ranking_notebook.ipynb`](ranking_notebook.ipynb): 13X bottom-third on the
datacenter profile, as screened).

The 2026-09 simplification removed the generic PDE trial environments and
their finite-difference scaffolding (they carried no cooling content and no
PINN machinery) and parked the PyQt6 GUI in
[`attic/`](../attic/README.md). Deep learning lives next door:
the bed PINN surrogate (`adsorbent-ml/models/bed_pinn.py`) is trained against
`physics/bed1d.py` as its oracle, and the RL backend targets
`TwoBedSchedule-v0` — see [`../ROADMAP.md`](../ROADMAP.md).

## Target API

```python
import harness

env = harness.make("Bed1D-v0",
                   material="anchor:Silica gel RD",
                   profile="datacenter")

result = harness.optimize(env, backend="grad",
                          design={"bed_thickness_m": 2e-4,
                                  "switch_time_s": 300.0})
result.metrics   # {"COP": ..., "SCP_W_kg": ...}
result.trace     # per-step diagnostics
```

## GUI — RL & simulation lab

`harness/gui/` is a PyQt6 launcher over the environments (no build step, no
canvas): pick an env from the registry, set material/profile/kwargs in forms
generated from the factory signatures and the problem's design space, choose
evaluate / optimize (grad · search · rl) / sweep (t_switch or material), and
run it on a background thread (queued dispatch — jitted runs aren't
interruptible mid-flight; sweeps are, between points). Scopes show metrics,
optimization history, episode traces (Bed1D/TwoBed series with desorption
shading), sweep curves, A/B compare, the log, and a generated reproducible
Python script.

```bash
uv sync --extra gui          # PyQt6 + matplotlib
harness-gui                  # or: python -m harness.gui
```

The old node-canvas workbench this launcher replaces stays parked in
[`../attic/`](../attic/README.md). The adsorbent-ml side has its own data
explorer built on the same widget kit (`harness.gui.kit`).

## Extension rule

New capability = **one data contract**, never a core change: new optimizers
via the `Backend` protocol, new materials via rows in a parameter table, new
applications via registered profiles. See `DESIGN.md` §7.

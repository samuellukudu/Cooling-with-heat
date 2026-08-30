# harness — Design Document

> **A differentiable optimization harness for heat-driven cooling, with
> gym-compatible environments.**
> RL is one of three optimization backends — not the identity of the project.
>
> Status: **H0 + H1 implemented (2026-08)** — package skeleton, Cycle0D
> oracle (V1 parity < 1e-12), grad + search backends, V7 acceptance green;
> the dynamic 1-D bed (Bed1D-v0) with V3 oracle limit, V4 literature
> calibration and V5 control gradients green; H2 (the two-bed system) in
> progress. See §12 for the milestone ladder.

**Posture: an R&D lab, not an application product.** The harness exists so
ideas about heat-driven cooling can be posed as cheap, honest experiments —
"does heat recovery pay?", "which physics approximation matters?", "which
material property drives system COP?" — against any operating scenario.
The four application profiles are **preset scenarios** for comparability
with the legacy screen, never the organizing frame: every experiment can
pass raw setpoints, geometry, and schedules directly, and no milestone is
scoped as a feature for one application.

This document specifies the `harness/` package: what it is, its module
layout, the physics of each environment, the environment interfaces
(state / action / reward), the optimization backends, the plugin contract
for future extensions, the validation ladder, and the packaging.

Companion documents:

- [`../ROADMAP.md`](../ROADMAP.md) — project strategy (adsorbent-ML stage ladder).
- [`../Materials/cooling_physics.py`](../Materials/cooling_physics.py) — canonical
  equilibrium cycle model; the harness is its differentiable, environment-shaped
  evolution, not a replacement.
- [`../adsorbent-ml/`](../adsorbent-ml/) — supplies the material parameters the
  harness consumes (fitted Dubinin–Astakhov isotherms).

---

## 1. Purpose

`harness` turns the pieces this project already has into one reusable loop:

```
                 ┌──────────────────────────────────────────────────────┐
                 │                      harness.make(...)               │
                 │   env = (physics model) × (material) × (profile)     │
                 └──────────────┬───────────────────────────────────────┘
                                │  jittable rollout  +  gym API
        ┌───────────────────────┼──────────────────────────┐
        ▼                       ▼                          ▼
   grad backend            search backend              rl backend
   (jax.grad, optax)       (CMA-ES / Optuna)           (PPO via Gymnasium)
        │                       │                          │
        └───────────────────────┴──────────────────────────┘
                                │  OptimizeResult
                                ▼
        optimal algorithms (control schedules) · materials (shortlists) ·
        designs (geometry/operating parameters)
```

One environment, three tunable axes, three kinds of output:

| Axis | What varies | Optimizer output |
|---|---|---|
| **Control** | switching schedule, fluid temperatures, valve states, source routing | **algorithms** — control policies / schedules |
| **Material** | `q_sat`, `Q_st`, D–A `E`, `n` (+ transport props) from the adsorbent-ml database | **materials** — ranked shortlists per application profile |
| **Design** | bed thickness, fin pitch, HX mass, cycle time, heat-recovery policy | **designs** — geometry and operating parameters |

The naming discipline, restated so it survives contact with enthusiasm:
this is an **optimization harness**. `rl` is a backend you attach to the two
or three problems that actually need sequential decision-making — not the
project's identity, and never a default dependency.

### Why a harness unifies this repository

- `diffheat/` (frozen) proved the pattern: differentiable thermal simulation
  in JAX. The harness is where that pattern pays off at system level.
- `Materials/` has the physics (`cooling_physics.py`) and the four
  application profiles (`cpu`, `human`, `vehicle`, `datacenter`).
- `adsorbent-ml/` is building the material parameter database
  (ISODB / CoRE / QMOF / IZA exporters done; `fit_da.py` next). Its ROADMAP
  task **T2 — "system-level ranker"** *is* an environment evaluation; the
  harness is T2 made interactive and reusable, and later becomes the reward
  function for active learning (T3) and guided generation (Stage 4).

### Non-goals

- **Not a general HVAC simulator.** Scope is adsorption/thermochemical
  cooling with water refrigerant, per `Materials/APPLICATIONS.md`.
- **Not an RL framework.** We call Gymnasium; we do not reimplement PPO.
- **Not a replacement for `cooling_physics.py`.** The NumPy implementation
  stays canonical for the equilibrium model; the harness must reproduce it
  exactly (validation item V1) and diverge from it only by adding dynamics.
- **No generative/inverse crystal design here.** That stays deferred in the
  adsorbent-ml roadmap (GeoField lesson). The harness only *scores* candidates.

---

## 2. Package layout

Exactly **two subfolders** (`physics/`, `envs/`); everything else is a flat
module. Tests live in the existing root `tests/harness/` (no third folder).

```
harness/
├── README.md            # tagline, status, target API — points here
├── DESIGN.md            # this document
├── pyproject.toml       # package `harness`; uv workspace member of the root
├── __init__.py          # public API: make, register, optimize, rollout
├── physics/             # pure JAX solvers — no gym, no I/O, no env policy
│   ├── thermo.py        # Psat (Magnus/Antoine 2-branch), h_fg (Watson), D–A uptake
│   ├── cycle0d.py       # equilibrium cycle model (Env-0 oracle)
│   ├── bed1d.py         # dynamic 1D adsorber bed (Env-1 solver)
│   └── system.py        # two-bed plumbing, heat recovery, source schedules (Env-2 solver)
├── envs/                # problem layer: state/action/reward wiring
│   ├── base.py          # protocols, ProblemSpec, DesignSpace, EpisodeTrace, Objective
│   ├── cycle0d.py       # Cycle0D-v0      (static problem)
│   ├── bed1d.py         # Bed1D-v0        (dynamic env)
│   └── two_bed.py       # TwoBed-v0       (dynamic env)
├── registry.py          # five registries + make() + lazy entry-point loading
├── backends.py          # Backend protocol + grad / search / rl implementations
├── materials.py         # MaterialParams, loaders (fit_da schema), builtin anchors
├── profiles.py          # ApplicationProfile (steady setpoints + optional schedules)
└── report.py            # OptimizeResult / EpisodeTrace → summary tables, JSONL
```

Dependency rules (enforced by import-linter in CI):

- `physics/` imports only `jax`, `numpy`, and `harness.physics.thermo`.
  It knows nothing about materials, profiles, or gym.
- `envs/` imports `physics/`, `materials`, `profiles`; never `backends`.
- `backends.py` imports `envs.base` (protocols only) — backends are consumers
  of problems, not providers.
- Nothing in `harness/` imports `diffheat` or `Materials/`. The physics in
  `physics/thermo.py` is a *mirrored copy* of the correlations in
  `cooling_physics.py` (they are ~50 lines; see V1). This keeps the frozen
  package frozen and the harness self-contained; the parity test is the
  guarantee that "mirrored" means "identical".

---

## 3. Core abstractions

Everything is duck-typed through `typing.Protocol` so external simulators and
third-party plugins can participate without inheriting from us.

```python
# envs/base.py (sketch)

@dataclass(frozen=True)
class ProblemSpec:
    name: str                 # e.g. "Bed1D-v0"
    kind: Literal["static", "dynamic"]
    obs_spec: tuple[Sensor, ...]     # name, unit, lo, hi
    action_spec: ActionSpec          # continuous Box / discrete set, documented
    metric_keys: tuple[str, ...]     # what evaluate()/episode info returns

class Problem(Protocol):
    """Anything a Backend can optimize. Static: no temporal structure."""
    spec: ProblemSpec
    def evaluate(self, design: dict[str, Array]) -> dict[str, float]: ...
    def rollout(self, design, controls, *, n_steps: int) -> EpisodeTrace: ...

class DynamicProblem(Problem, Protocol):
    """Gym-compatible temporal problem."""
    def reset(self, *, seed: int | None = None) -> Observation: ...
    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict]: ...

class Backend(Protocol):
    name: str
    def solve(self, problem: Problem, objective: Objective,
              budget: Budget) -> OptimizeResult: ...
```

```python
# top-level API (harness/__init__.py)

env  = harness.make("Bed1D-v0",
                    material="anchors:Silica gel RD",
                    profile="datacenter")
res  = harness.optimize(env, backend="grad",
                        design={"bed_thickness_m": 2e-4,
                                "switch_time_s": 300.0})
# res.metrics  -> {"COP": ..., "SCP_W_kg": ...}
# res.trace    -> per-step series for diagnostics
# res.best     -> design dict + metrics

harness.register_env / register_backend / register_material / register_profile
```

### Conventions

- **Units.** SI inside `physics/` (Kelvin, Pa, J, s, W). Environment
  boundaries expose °C and kW to match `Materials` profiles and literature,
  and convert at the edge. Every `Sensor` and design parameter declares its
  unit in its spec — no bare floats crossing module boundaries.
- **Metrics vs reward.** Metrics (`COP`, `SCP_W_kg`, …) are *reported*.
  Reward is the *optimization handle* (shaped, possibly penalized) and is
  never presented as a physical quantity. `info` always carries the metrics.
- **Determinism.** Same seed ⇒ same trajectory, and the jitted rollout must
  equal the eager rollout (tested). Stochastic profiles (Env-2, H2+) draw
  from a seeded key that is part of the problem configuration, not global
  state.
- **Episode definition.** One episode = N full adsorption/desorption cycles
  (default 4). Metrics are computed over the last N−1 cycles so start-up
  transients don't dominate; `EpisodeTrace` records everything.

---

## 4. Physics models

All correlations are shared in `physics/thermo.py` and **mirrored exactly**
from `Materials/cooling_physics.py`:

- Saturation pressure of water: two-branch Magnus (≤100 °C) / NIST-Antoine
  (>100 °C), error < 0.3 % vs IAPWS-IF97.
- Latent heat: Watson correlation (exponent 0.321), anchored to
  2 256 400 J/kg at 100 °C, error < 0.5 %.
- Dubinin–Astakhov equilibrium uptake with independent characteristic energy:
  `q*(T,P) = q_sat · exp(−(A/E)^n)`, Polanyi potential
  `A = R·T·ln(P_sat(T)/P)`.

CoolProp's role stays what commit `e6ca51f` gave it: **reference, not hot
path**. An optional `harness[cool]` extra runs property parity checks against
CoolProp in tests; the JAX hot path uses the in-house correlations.

### 4.1 Env-0 — `Cycle0D` (equilibrium oracle, static problem)

A faithful JAX port of `simulate_adsorption_cycle`: D–A uptake at the two
bed states, sensible + desorption heat with the HX mass factor, COP and SCP.
Not a dynamic env — a static `Problem` (it has no meaningful `step`; faking
dynamics where there are none is exactly what the naming discipline forbids).
The RL backend wraps static problems as single-step envs if anyone insists.

**Role:** unit-test oracle (V1), backend shakedown target, and the fast
pre-filter when sweeping a whole materials database.

### 4.2 Env-1 — `Bed1D` (dynamic single adsorber)

The first real environment: a 1-D adsorber coating/slab on a heat-exchanger
wall, with transient heat transfer and finite-rate adsorption. This is the
model that can represent the two barriers `APPLICATIONS.md` names (thermal
conductivity → transient bed heat-up; cycle behaviour), and the natural
evolution of the frozen `diffheat` 1-D solver.

**Geometry v1 (coated fin, wall-normal slab).** `x ∈ [0, L]`, `x = 0` at the
HX wall (fluid side), `x = L` adiabatic (vapor-side symmetry). A packed-bed
variant (1-D along flow) is a registered alternative model, not a code fork.

**State.** Bed temperature `T(x,t)` [K], uptake `q(x,t)` [kg/kg], lumped bed
vapor pressure `P(t)` set by the valve, phase ∈ {ads, des}.

**Governing equations.**

```
bed energy:   (c_s + c_pl·q)·ρ_s · ∂T/∂t = ∂/∂x( k_eff · ∂T/∂x ) + ρ_s·q̇·Q_st
uptake (LDF): q̇(x,t) = k_LDF · ( q*(T(x,t), P(t)) − q(x,t) )
isotherm:     q*(T,P) = q_sat · exp( −(A/E)^n ),  A = R·T·ln( P_sat(T)/P )
wall BC:      −k_eff·∂T/∂x |_{x=0} = h·( T_f(t) − T(0,t) )      [convective]
              ∂T/∂x |_{x=L} = 0                                  [adiabatic]
valve:        ads → P = P_sat(T_evap);  des → P = P_sat(T_cond)
```

Source sign (fixed at implementation, H1.1): adsorption (q̇ > 0) *heats*
the bed. With the adsorbed-phase enthalpy convention `h_a = h_g − Q_st`
this form is the exact bed energy balance — the vapour-enthalpy and
storage cross-terms cancel identically — which is what makes the
desorption-phase wall-flux integral reduce to the `Cycle0D` heat-input
accounting in the equilibrium limit (V3).

`Q_st` = isosteric heat [J/kg water]; `c_s, ρ_s, k_eff` = solid effective
properties; `h, T_f(t)` = film coefficient and heat-transfer-fluid
temperature (the main design/control knobs); `k_LDF` = kinetic coefficient
[1/s], calibrated per material class (H1) with literature priors.

**Known approximation (documented, revisited in H2).** The isosteric
pre-heating/pre-cooling phase is instantaneous in v1 (valve flips, `P` jumps,
`q` frozen). A v2 option adds a lumped vapor mass balance
(`dP/dt` from vapor inventory) to represent the isosteric swing explicitly.

**Energy accounting.** Cooling delivered during adsorption:
`Q̇_cool(t) = −ṁ_vap(t) · h_fg(T_evap)` with
`ṁ_vap = m_s · d(mean q)/dt` (positive when vapor flows *from* evaporator to
bed). Heat input during desorption: integral of the wall-side flux plus
sensible terms — the cycle-level `COP` and `SCP` follow
(`SCP = Q_cool,cycle / (t_cycle · m_s)`). Implemented at H1.4: the wall
flux is credited to the bed only (`hx = 1` is the bare bed, the V3
oracle correspondence); `hx_mass_factor > 1` adds, once per cycle, the
desorption-phase sensible swing of an external metal/water mass lumped
at the wall — `(hx−1)·c_eff·(T_des,end − T_des,start)` — pure
accounting (the trajectory is untouched), which is what makes `hx` a
*rig* property: V4 fits it per experimental rig, not per material.

**Numerics.**

- Method of lines, N = 32–128 cells; explicit RK4 time stepping inside
  `jax.lax.scan` (whole episode = one scan: `jit`-able, `vmap`-able).
- **Stiffness control.** LDF uptake is integrated with an exact exponential
  sub-update per step given frozen `q*`:
  `q_{n+1} = q* + (q_n − q*)·exp(−k_LDF·Δt)`. This removes the `k·Δt < 2`
  explicit-Euler bound of the uptake ODE itself (any `k·Δt` is stable in
  `q`). The remaining limit is the two-way adsorption-heat ↔ temperature
  coupling of the split scheme: linearising the one-step map gives roughly
  `|Q_st/cap · dq*/dT|·(1 − e^{−k_LDF·Δt}) ≲ 2`, i.e. `k_LDF·Δt ≲ 0.25`
  for typical parameters. The equilibrium limit V3 exercises is therefore
  τ_kin = 1/k_LDF ≪ cycle time (k-convergence asserted in the V3 test),
  not literally k_LDF → ∞.
- **Swing self-limiting (observed at H1.2/1.3).** During a swing the
  adsorption heat adds `ρ_s·Q_st·|dq*/dT|` (~10×) to the effective heat
  capacity, stretching the swing time constant to `τ ≈ L²/α_eff` with
  `α_eff = k_eff/cap_eff`. Endpoint completion therefore needs
  `t_phase ≫ τ_eff` (strong `h`, thin bed, or long phases) — the dynamic
  system's COP can be non-monotone in `q_sat` under transients, and SCP
  comparisons across cycle times must complete swings to be meaningful.
- Conduction step respects the diffheat CFL rule of thumb
  `Δt ≤ Δx² / (2·α_eff)`; `physics/bed1d.py` exposes a `check_timestep`
  helper (mirroring `diffheat.check_cfl`) so backend sweeps fail loudly.
- Gradients flow through everything (`jax.grad` on any continuous design or
  control); discrete valve flips make the episode metric a *staircase* in
  the switch times — flat between physics substeps, so the only smooth
  gradient term is the trivial metric/duration scaling (H1 finding). The
  soft-switch option (`soft_switch=True`) splits boundary substeps into
  two part-steps, each integrated with its own phase conditions over its
  exact fraction — episode metrics become continuous in the switch time
  at full physical strength, with no blended-pressure burst (the D–A is
  exponentially pressure-sensitive). Verified by V5.

### 4.3 Env-2 — `TwoBed` (system with switching, heat recovery, profiles)

Two `Bed1D` instances (possibly different materials — the basis for
composite-bed questions), four valves, optional mass/heat recovery between
the beds, and a heat-transfer-fluid loop driven by an `ApplicationProfile`
or by raw per-step series — the lab posture means the dynamic sources are
*data* (per-step temperature/request arrays), not named applications.

**Implemented (H2.1).** Counter-phase pair in one `jax.lax.scan`
(`physics/system.py`): system phase σ ∈ {0, 1} records which bed adsorbs;
both beds always flip together (phase durations are role-keyed per bed, so
asymmetric ads/des durations stay consistent). Phases end at their declared
duration or earlier on a **request bit** — the valve-level hook schedule
policies (H2.2) use to gate desorption on a time-varying source — subject
to a minimum dwell. Heat recovery: during the first `t_rec` seconds of
each phase the beds are disconnected from their fluids (the recovery loop
owns the circuits) and exchange heat through a lumped `UA_rec` between
their mean temperatures, applied as exactly antisymmetric per-cell sources
(conserves `U_A + U_B` to machine precision, free in the `Q_in` books).
The hot bed is whichever just finished desorbing, so the transfer always
runs in the productive direction; the v1 machine keeps its vapor valves on
phase connections during the window (no vapor-pressure dynamics — the
Open Question 2 lumped inventory is the extension head).

Finding worth keeping: recovery raises COP only when the swings complete
with slack (the exchanged heat then offsets source draw at unchanged
swing); in transient-bound regimes the film-off window delays the swings
and `recovery_gain` goes negative — a throughput-vs-efficiency trade-off
that schedule optimization should pick the operating regime for.

This is the environment where sequential decision-making is real: the action
is *when* to switch and *how* to route heat under a time-varying boundary —
the regime where gradient schedules, CMA-ES, and RL are all meaningful and
comparable.

---

## 5. Environment interface specs

Concrete state / observation / action / reward definitions. Observations are
sensor-like subsets (what a controller could actually measure), not full
fields; full fields stay available in `EpisodeTrace` for diagnostics.

### 5.1 `Cycle0D-v0` — static

| | |
|---|---|
| Design params | `q_sat` [kg/kg], `Q_st` [J/kg], `E` [J/mol], `n` [–], `hx_mass_factor` [–], `cycle_time_s` [s] |
| Metrics | `COP`, `SCP_W_kg`, `delta_q`, `q_ads`, `q_des`, `P_evap_kPa`, `P_cond_kPa`, `h_fg_MJ_kg` |

### 5.2 `Bed1D-v0` — dynamic

| | spec |
|---|---|
| Hidden state | `T(x)`, `q(x)` (N cells), `P`, phase, phase timer |
| Observation (float32 vector) | `T_wall`, `T_bed_mean`, `T_bed_max`, `q_mean`, `q_star_mean`, `P/P_sat(T_evap)`, `time_in_phase/t_switch`, `T_f`, cumulative `Q_cool`, cumulative `Q_in` |
| Action (v0 control) | continuous: desorption fluid temperature `T_f,des` ∈ allowed band; switch time `t_switch` (or per-step discrete `{connect_ads, connect_des}` — both exposed, one is fixed per experiment) |
| Reward (per step) | `r_t = ΔQ_cool_t − λ·ΔQ_in_t − μ·violation_t` — `λ` (default 0 → maximize SCP; `λ = 1/COP_target` ≈ heat-cost penalty), `μ` setpoint-violation penalty. Metrics ride in `info` |
| Episode | 4 cycles (metrics over last 3); `t_cycle = t_switch,ads + t_switch,des` from profile or action |
| Design params (not observed, optimized) | `L` (bed thickness), `k_eff`, `h`, `hx_mass_factor`, material params |

### 5.3 `TwoBed-v0` — dynamic

| | spec |
|---|---|
| State | two `Bed1D` states + valve states + source/load loop temperatures |
| Observation | per-bed sensor vectors + `T_source(t)`, `T_chw`, load demand, time |
| Action | switching schedule per bed (times or per-step valve bits), source allocation fraction, heat-recovery on/off |
| Reward | `met_load_fraction − λ_energy·(heat input + pump proxy) − μ·unmet_setpoint` |
| Metrics | `COP`, `SCP_W_kg`, `unmet_load_frac`, `recovery_gain` (COP with vs without heat recovery) |
| Stochasticity | optional seeded noise on source temperature and load (H2+) |

---

## 6. Optimization backends

Three built-ins behind one protocol. Each `solve()` returns the same
`OptimizeResult` (best design, metrics, budget spent, trace of evaluations),
so results are comparable across backends and plottable by `report.py`.

| Backend | Method | Engine | Right problem class |
|---|---|---|---|
| `grad` | value-and-gradient ascent (optax; multi-start) | `jax.grad` through the scan rollout | smooth continuous design/control params |
| `search` | CMA-ES + Optuna TPE; constraints via penalty | `cma`, `optuna` | mixed/structured design spaces, cheap evals, no gradients needed |
| `rl` | PPO over gym API (episode reward) | Gymnasium → `stable-baselines3` (behind `harness[rl]` extra, torch included) | schedule/control under time-varying or stochastic profiles |

**The honesty rule.** Match the backend to the problem class:

- The equilibrium model is solvable *exactly* by grid search and gradients —
  running RL on it must fail to beat `search`, and the harness should say so
  (a benchmark run in CI asserts `search` ≥ `rl` on `Cycle0D-v0`).
- RL is justified only when the action is a *schedule* under time-varying or
  stochastic boundaries — i.e. `TwoBed-v0` with profiles, and later
  uncertainty-aware variants.
- **Mandatory floor:** every RL result is reported against the best fixed
  schedule found by `search` (and the profile's naive schedule). RL that
  can't beat a thermostat is a negative result and gets reported as one.
  (Same philosophy as the ROADMAP's "GBDT is the floor GNNs must beat".)

Gradient-through-discrete-switching caveat: valve flips make the objective
piecewise-smooth. `grad` handles switch *times* (subgradients) and the
soft-switch option; if a H1 experiment shows pathological landscapes, the
fallback is `search` on switch times + `grad` on everything else. Decide
with data at H1, not by preference.

---

## 7. Extension & plugin contract

The user-facing rule, borrowed from the ROADMAP's GeoField lesson
("one head + one labeler + one verifier"): **new capability = one adapter +
one data contract — never a core change.**

### 7.1 Registries

Five registries live in `registry.py` (a neutral module, so `materials` and
`profiles` can register without importing the env layer), all with the same
shape (`register_x(name, factory)`, case-sensitive names, collision raises):

`harness.models` · `harness.envs` · `harness.backends` ·
`harness.materials` · `harness.profiles`

Third-party packages plug in via entry points (imported lazily on `make`):

```toml
[project.entry-points."harness.envs"]
MyBed-v0 = "my_pkg.envs:MyBed"
[project.entry-points."harness.backends"]
nsga2 = "my_pkg.backends:NSGA2Backend"
```

### 7.2 Pluggable physics — `SimulatorAdapter`

The physics behind an environment is itself pluggable, which is the answer to
"integrate other simulation tools we might adopt". An external solver joins
by implementing one protocol — no env or backend changes:

```python
class SimulatorAdapter(Protocol):
    spec: ModelSpec                      # state columns, units, dt bounds
    def step(self, state, control, dt) -> tuple[state, Fluxes]: ...
    def metrics(self, trace) -> dict[str, float]: ...   # episode summary
```

Planned/anticipated adapters (each is a *future* milestone item, not H0–H1):

- **OpenModelica / Modelica Buildings-AixLib**: system-level dynamics and
  building envelopes; adapter drives a Modelica bed/chiller model over a
  socket/FMU bridge. Same `TwoBed-v0` env wiring, slower rollout.
- **TESPy**: steady cycle design with real-fluid properties; adapter for
  `Cycle0D`-class problems with real refrigerant properties.
- **CoolProp**: property *reference* in tests (already the project's stated
  bounded role).
- **RASPA / ML-potential GCMC**: not a runtime adapter — it upstreams into
  adsorbent-ml's L2 labels, which arrive here as better-fitted material rows.

### 7.3 Materials as data, profiles as data

- A **material** is a row in a parameter table, loaded by `materials.py` —
  adding one requires zero code. Schema in §8. Built-ins ship from
  `adsorbent-ml/data/anchors.csv` (13 curated commercial adsorbents).
- A **profile** is a registered `ApplicationProfile`. Built-ins mirror
  `Materials/heat_cooling_screen.py` **with the same four keys**
  (`cpu`, `human`, `vehicle`, `datacenter`) and the same field names for the
  steady setpoints, extended with optional `source_schedule` / `load_schedule`
  for the dynamic envs. Same key = same physical scenario across subprojects.

---

## 8. Data contracts

### 8.1 `MaterialParams` (consumed from adsorbent-ml)

`fit_da.py` (planned in adsorbent-ml) writes the per-isotherm D–A fits; this
is the schema the harness loads. Parquet (preferred) or CSV:

| column | unit | notes |
|---|---|---|
| `material_id`, `name`, `source` | — | `source` ∈ {isodb, core_mof, qmof, iza, anchor} |
| `isotherm_id` | — | provenance into ISODB |
| `q_sat_kg_kg` | kg/kg | |
| `q_st_j_kg` | J/kg | isosteric heat |
| `e_char_j_mol`, `n_da` | J/mol, – | D–A parameters |
| `t_range_c` | °C | fit validity window; env warns outside it |
| `fit_rmse`, `n_points` | — | fit quality; rankers may filter |
| `k_ldf_s_1` | 1/s | optional; class-default if absent |
| `rho_kg_m3`, `cp_j_kg_k`, `k_eff_w_m_k` | SI | optional; **provenance-flagged defaults** if absent (see below) |

Honesty note that must survive into any result table: equilibrium parameters
come from data; transport properties (`k_eff`, `h`) currently come from
class-default literature values, flagged `provenance="default"`. Ranking
*within* a fixed transport assumption is meaningful; absolute SCP across
materials with defaulted transport is not yet. Closing that gap (per-material
`k_eff` from mofdscribe descriptors / literature) is the natural H3 extension
head — and it attacks the very barrier (`APPLICATIONS.md` §thermal
conductivity) that motivated this project.

### 8.2 `ApplicationProfile`

| field | unit | notes |
|---|---|---|
| `name`, `description` | — | |
| `t_evap_c`, `t_cond_c`, `t_des_c` | °C | steady setpoints (dynamic envs use them as bands/nominals) |
| `cycle_time_s` | s | nominal half-cycle |
| `cop_weight`, `scp_weight`, … | – | inherited verbatim from `heat_cooling_screen.py` for cross-tool comparability |
| `source_schedule(t)`, `load_schedule(t)` | °C, kW | optional; None ⇒ steady env |
| `t_fluid_min_c`, `t_fluid_max_c` | °C | actuator band for control actions |

---

## 9. Validation ladder

Each rung is an automated test (root `tests/harness/`). A rung blocks the
milestone that needs it; nothing downstream may hand-wave past a red rung.

| # | Test | Criterion |
|---|---|---|
| V1 | Property parity: `physics/thermo.py` vs `Materials/cooling_physics.py` | max rel. error < 1e-12 on a dense T-grid (identical formulas); optional CoolProp check < 0.5 % |
| V2 | Conservation | per-step energy + adsorbate mass residual < 1e-6 (relative, jitted) |
| V3 | **Oracle limit**: `Bed1D` with `k_LDF → ∞`, long cycle, thin bed → `Cycle0D` | COP/SCP match within 2 % |
| V4 | Literature: silica-gel/water LDF benchmarks (Sakoda–Suzuki-type model; published chiller data, e.g. the Saha–Koyama series) | COP within ~15 % after calibrating `k_LDF`, `h`, UA within literature ranges; correct trend in cycle-time ↔ SCP sweep — ✅ implemented: `calibration.py` + `benchmarks.md`; standard points ±1 %/+22 %→refit; cycle-time trend green |
| V5 | Gradients: `jax.grad` vs central finite differences on smooth controls | rel. error < 1e-3 at ≥ 10 probe points — ✅ implemented: 10 probes on the soft-switched control problem, max rel. err < 1e-3 (`test_v5_control_gradients.py`) |
| V6 | System sanity: `TwoBed` heat recovery ≥ no-recovery COP; cooling duty continuous across bed switchover | monotone improvements; no duty gaps > 1 dt — ✅ implemented: recovery raises COP monotonically in `UA_rec`/`t_rec` at unchanged swing; duty continuity made precise for the v1 valve model — the flip burst (present in the single bed too) may make the per-step duty nonpositive for a few τ_kin, so the gate bounds the nonpositive stretch by the burst timescale (never the phase duration) and requires net-positive cooling over any one-phase window (`test_v6_two_bed_system.py`) |
| V7 | Backend cross-check: `grad` and CMA-ES agree on the `Cycle0D-v0` optimum; reproduce `pareto_target_window`'s best point | same optimum within tolerance |
| V8 | RL floor: PPO on deterministic `Cycle0D-v0` (single-step wrapper) must NOT beat `search` | codifies the honesty rule |

V3 is the keystone: it ties the dynamic model to the canonical equilibrium
model by construction, so "the harness agrees with the screening tool" is a
test, not a claim. Implemented (H1.3): the limit is taken as τ_kin ≪ cycle
time with k- and dt-convergence asserted; the residual ~1.5 % COP gap is
the documented trajectory-convention difference (the oracle books the
desorption sensible heat at the frozen adsorption uptake `q_ads·c_pl·ΔT`
— exact for its instantaneous-isosteric trajectory — while the dynamic bed
integrates `c_pl·∫q*(T, P_cond) dT` along the coupled trajectory; the
cooling side and SCP match exactly).

---

## 10. Testing strategy

- Location: root `tests/harness/` (existing pytest config applies).
- Physics: V1–V3, V5 as unit tests (CPU JAX, fast; small N).
- Env API: Gymnasium `check_env` for dynamic envs; obs/action spec round-trip;
  determinism (jitted == eager; same seed twice).
- Backends: smoke runs at tiny budgets on `Cycle0D-v0`; V7/V8; `OptimizeResult`
  schema stability (schema version constant).
- Registry/entry points: register → make round-trip, collision error, lazy
  plugin import.
- CI posture: everything above runs CPU-only in < 5 min; GPU and `rl` extra
  are opt-in local runs, never required for a green build.

---

## 11. Packaging & dependencies

`harness/pyproject.toml` is a **uv workspace member** of the root project
(root gains `[tool.uv.workspace] members = ["harness"]`); `uv sync` continues
to manage one lockfile. Package name: `harness` (matches the folder; the
README tagline carries the identity — renaming the distribution later is a
two-line change if the generic name ever hurts).

```toml
dependencies = ["jax[cpu]", "numpy", "gymnasium", "cma", "optuna"]
[project.optional-dependencies]
rl     = ["stable-baselines3", "torch"]      # never a default dependency
cool   = ["CoolProp"]                        # property reference checks only
viz    = ["matplotlib"]
dev    = ["pytest", "import-linter", "hypothesis"]
```

`jax[cuda12]` stays driven by the root project's existing pattern; harness
code is device-agnostic. Python ≥ 3.10, matching the root.

---

## 12. Milestones

Each milestone has acceptance criteria; a milestone without green criteria is
not done.

### H0 — Skeleton + oracle — ✅ done 2026-08

- Package skeleton, `core/base` protocols, registries, `make()`.
- `physics/thermo.py` + `physics/cycle0d.py` (JAX port) with V1 green.
- `envs/cycle0d.py` static problem; `backends.py` with `grad` + `search`.
- `materials.py` + `profiles.py` with the four built-in profiles and
  `anchors.csv` shipped as built-in materials.
- **Accept:** `harness.optimize(make("Cycle0D-v0", material="anchor:Silica gel RD", profile="datacenter"), backend="grad")` reproduces the
  `pareto_target_window` best point (V7); import-linter rules green.

### H1 — The dynamic bed (≈ 1–2 weeks) — *the risky part*

Ordered so each task unblocks the next; every task ends with its gate green.

- **H1.0 — Shared labels artifact (adsorbent-ml, do first) — ✅ done 2026-08.**
  `adsorbent-ml/data/fit_da.py` fits D–A parameters (`q_sat`, `E`, `n`) to
  the 1,221 ISODB water isotherms and writes the §8.1 schema to
  `data_cache/fits/da_params.csv` with a QC report (fit-RMSE distribution;
  flag S-shaped isotherms — the MIL-101 class — where D–A fits poorly).
  One artifact, two consumers: Stage-1 training labels *and* the harness
  material database. *Gate met: 386 usable fits, median nRMSE 0.028;
  24 adsorbents with multi-T Q_st; flag/unit/q_sat-physical breakdown in
  `data_cache/fits/da_params_qc.md`; the loader side consumed it via
  `harness.load_materials_csv` (553 rows, §8.1).*

  **Reusability contract** (library + thin CLI — a general fitter applied
  1,221 times, not a one-off training script):
  - `fit_isotherm_da(p, q, T, psat_fn, ...) -> DAFit` — pure function, one
    isotherm, **adsorbate-agnostic**: the saturation-pressure callable is
    injected (water's Magnus correlation is only the CLI's default). No
    ISODB parsing inside — plain arrays in — so CoRE/QMOF computed
    isotherms and future GCMC (L2) labels reuse it unchanged.
  - Diagnostics are first-class return values (`rmse`,
    `flag ∈ {ok, poor_fit, S_shaped, insufficient_points}`, `n_points`),
    never side prints — they fill the §8.1 columns and drive the QC gate.
  - `isosteric_heat(isotherms) -> float | None` — Clausius–Clapeyron
    across ≥ 2 temperatures at matched uptake; `None` + flagged for
    single-T adsorbents; coverage statistics go to the QC report.
  - `fit_all(rows) -> DataFrame` batch helper; the CLI
    (`--mirror --out --qc-report`) stays thin: parse → call library →
    write.
  - Unit tests: synthetic noisy D–A curves recover their true parameters;
    flags fire on synthetic Type-V curves.
  *Gate: fits exported; median RMSE and flag count documented; and the
  fitter demonstrably importable/reusable as a library (e.g. from a
  notebook) without going through the CLI.*
- **H1.1 — Bed physics core.** ✅ done 2026-08. `physics/bed1d.py`: 1-D bed
  energy equation with the adsorption-heat source (sign fixed — see §4.2),
  LDF uptake with the exact-exponential substep, D–A isotherm from
  `thermo.py`, ghost-cell convective wall BC + adiabatic far BC; RK4 inside
  `jax.lax.scan`; `check_timestep` stability helper.
  *Gate (V2 + unit) — green: pure-conduction limit matches the analytic
  semi-infinite slab (max profile error 0.29 % of the 50 K step); uptake
  matches its exponential solution at any k·Δt (incl. k·Δt = 500);
  per-step energy identity and semi-discrete telescoping hold to ~1e-10
  relative; RK4 order 4 verified against expm of the semi-discrete system.*
- **H1.2 — Bed env + gym wrapper.** ✅ done 2026-08. `envs/bed1d.py` with
  the §5.2 observation/action/reward spec (registered `Bed1D-v0` in both
  the harness and Gymnasium registries); jitted whole-episode rollout for
  the grad backend (gradients flow through the scan); Gymnasium wrapper
  (`Bed1DGymEnv`) passing `check_env`. *Gate — green: check_env clean;
  metrics schema versioned (`BED1D_SCHEMA_VERSION = 1`); rollout and
  stepwise paths agree to 1e-6.*
- **H1.3 — Oracle limit (V3, keystone).** ✅ done 2026-08. Equilibrium
  limit (τ_kin = 0.2 s vs 600 s phases, h = 2000 W/m²K, 2 mm bed) →
  `Cycle0D` on COP and SCP. *Gate — green: COP gap 1.55 %, SCP gap 0.15 %
  (< 2 %); k_LDF- and dt-convergence < 0.5 %. The residual COP gap is the
  documented isosteric-convention difference (§9).*
- **H1.4 — Literature calibration (V4).** ✅ done 2026-08. Calibrated
  against two open-access experimental rigs (Uyun+2009 silica gel RD;
  Sztekler+2021 KD gel), one standard point each, sweeps as predictions.
  *Gate — green: standard points COP +0.9 %/−0.7 % (Uyun 70 °C); cycle-
  time ↔ SCP trend reproduced (rise–peak–fall, modelled peak 500–600 s
  vs measured 500 s); full table, mapping and findings in
  `harness/benchmarks.md`. Documented v1-scope gap: the heat-source-
  temperature *trend* needs the lumped vapour inventory (Open Question 2,
  answered below).*
- **H1.5 — First control experiments.** ✅ done 2026-08. `Bed1DControls`
  exposes the §5.2 controls to the backends; V5 gradient-vs-FD gate green
  (10 probes, < 1e-3, soft switch); headline plot + optimization in
  `harness/control_notebook.ipynb` (Adam +31 % over defaults; CMA-ES to
  the grid optimum). *Gate — green; the Open Question 3 decision is
  recorded in §13.3.*

**H1 accept (cumulative):** V3 within 2 %; benchmark table exists; the
notebook reproduces the headline plot.

### H2 — System + materials sweep

- **H2.1 — Two-bed physics.** ✅ done 2026-08. `physics/system.py` +
  `envs/two_bed.py` with the §5.3 spec (v0 continuous action =
  `(t_switch, T_f,des, t_rec)`; per-step valve bits live at the physics
  level as request bits for schedule policies; load model deferred).
  Counter-phase pair, role-keyed asymmetric phase durations, request
  bits with minimum dwell, film-disconnect heat recovery. *Gate — green
  (V6): recovery raises COP monotonically in `UA_rec`/`t_rec` (+2 %
  … +8 % over the swept range, 600 s phases, unchanged swing) with the
  source draw dropping by exactly the exchanged heat; no duty gaps
  beyond the documented flip-burst timescale; exchange conserves
  `U_A + U_B` to 1e-9; system COP/SCP match the `Cycle0D` oracle within
  2 % in the equilibrium limit. Documented trade-off: in transient-bound
  regimes (short phases, slow kinetics) recovery goes negative —
  see §4.3.* Also in this change: `bed1d.step_bed`/`step_episode` gained
  `extra_source_w_m2` / `wall_film_scale` (the recovery plumbing) with
  the single-bed paths numerically untouched (V2/V3/V5 still green).
- **H2.2 — Time-varying sources + schedule policies** *(recast from
  "datacenter profile, dynamic" per the R&D-lab posture: the milestone is
  the experiment family, the 45–70 °C coolant loop is one preset
  instance)*. Source temperature and switching policies as per-step
  series; schedule optimization via `search`/`grad`. *Gate: schedule
  optimization beats the nominal fixed schedule on the datacenter loop
  scenario.*
- **H2.3 — Materials sweep = the T2 ranker.** The fitted ISODB table
  (H1.0) through `Cycle0D-v0` (all materials) and `Bed1D-v0` (top
  candidates) per profile → ranked shortlists. This produces the
  brute-force reference rankings that the adsorbent-ml Stage-2 surrogate's
  **top-k hit rate** is scored against — where the harness and the ML
  ladder meet. *Gate: shortlist sanity vs anchors (zeolite 13X ranks
  poorly on the 60 °C-regeneration datacenter profile — the known
  screening result); ranking notebook in `harness/`.*
- **H2.4 — rl backend (optional, pull-driven).** PPO on stochastic
  `TwoBed-v0` behind the `rl` extra, always reported against the mandatory
  floor (best fixed schedule from `search`; V8 posture). Only started if
  H2.2 shows schedules matter enough to be worth amortizing into a policy.

### H3 — Integration & optional backends (deferred, pull-driven)

- One external `SimulatorAdapter` end-to-end (OpenModelica-FMU or TESPy —
  whichever the first concrete experiment needs; don't build both speculatively).
- `rl` extra with PPO on stochastic `TwoBed-v0`, reported against the
  mandatory floor (V8 posture).
- Per-material transport-property heads (`k_eff`) — closes the honesty gap
  in §8.1; two-bed composite-material experiments.

---

## 13. Open questions

1. **`k_LDF` priors per material class** — ✅ answered by V4 (2026-08):
   chiller-level data identify the *bed-level* effective kinetics when the
   bed is kinetics-limited (k_LDF ≈ 2.5e-4 s⁻¹ for packed RD, 1.5e-3 s⁻¹
   for finer KD gel — consistent with r⁻² grain scaling) and are
   uninformative in the heat-transfer-limited regime. Grain-level priors
   cannot be recovered from cycle data; ISODB kinetic isobars remain the
   route to grain-level values if ever needed. See `benchmarks.md`.
2. **Isosteric phase modeling** — ✅ answered by V4 (2026-08): the v1
   fixed-reservoir-pressure approximation calibrates *point* performance
   (±1 %) and reproduces cycle-time trends, but the heat-source-
   temperature *trend* is far too weak (measured SCP ~triples 65→80 °C,
   model +11 %) — the lumped vapour-inventory option (H2) is needed for
   T_hs-trend fidelity. See `benchmarks.md` finding 2.
3. **Gradient-through-switching** — ✅ decided by the H1.5 experiment
   (2026-08): hard-valve subgradients do **not** exist in any useful sense
   (the metric is flat between substeps; `jax.grad` returns only the blind
   metric/duration term and ascent runs the switch time to a bound — the
   no-go). Soft switching (substep split) gives V5-verified true gradients,
   but fixed-lr Adam drifts along the ridge near the optimum. **Recipe:
   `search` on switch times + `grad` on smooth controls** (CMA-ES reached
   the grid optimum where Adam plateaued ~6 % short), or multi-start grad.
   See `harness/control_notebook.ipynb` and the V5 tests.
4. **Stochastic profile spec** — which uncertainties matter for the datacenter
   profile (source temperature variance? load steps?) and what sets their
   magnitudes. Needs one source citation before H2 RL experiments.
5. **GPU** — irrelevant until materials sweeps get large; the ROADMAP's open
   GPU question stays open and unblocking.

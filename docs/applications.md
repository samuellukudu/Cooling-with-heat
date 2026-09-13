# Application scenarios

The project's "applications" are **preset scenarios** that make results
comparable across subprojects — never the frame (the harness is an R&D lab;
new scenarios are new data rows, not new code). There are two families:

- **Part I — adsorption-cooling application profiles** (active): the
  setpoint/weight presets in `harness/profiles.py` that drive every harness
  physics run — Cycle0D rankings, Bed1D episodes, TwoBed schedules.
- **Part II — diffheat PDE application scenarios** (frozen reference): the
  application zoo documented for the removed `diffheat` library (deleted at
  `f36e929`, recoverable from git history). Kept because they remain the
  design reference for PDE-level scenarios that may return inside the ML
  pipeline — §3 (absorption chiller) is the likeliest candidate, and
  `adsorbent-ml/data/ACQUISITION.md` maps its moisture-transport work to §1.

---

## Part I — Adsorption-cooling application profiles

Registry: `REGISTRIES["profiles"]` in
[`harness/profiles.py`](../harness/profiles.py). Built-ins mirror the legacy
screen (`Materials/heat_cooling_screen.py::APPLICATIONS`, archived) with
the **same four keys** — `cpu`, `human`, `vehicle`, `datacenter` — and
identical setpoints and weights, so the same key means the same physical
scenario across tools (DESIGN §7.3/§8.2).

### Posture: presets, not the frame

Profiles are data. Any env accepts a raw `ApplicationProfile(...)` instance
in place of a registry key, and H2+ envs additionally accept time schedules
(`t [s] → °C / kW` plain callables) — experiments pass their own directly.
The four built-ins exist so rankings, calibrations, and papers are talking
about the same numbers, not to scope features to particular products.

### Anatomy of a profile

| field | unit | role in the physics |
|---|---|---|
| `t_evap_c` | °C | evaporator saturation temperature → `p_evap`, the adsorption-stage vapor pressure, and latent heat `h_fg(t_evap)` |
| `t_cond_c` | °C | condenser saturation temperature → `p_cond` (desorption back-pressure) **and** the adsorption-stage bed temperature `t_ads = t_cond` |
| `t_des_c` | °C | regeneration (heat-source) temperature the bed must reach to desorb |
| `cycle_time_s` | s | half-cycle (per-leg) time — divides cooling power: `SCP = q_cool / cycle_time` |
| `cop_weight`, `scp_weight` | – | objective weights (renormalized over the pair) for the profile-weighted score |
| `stability_weight`, `conductivity_weight`, `density_weight`, `pareto_weight` | – | legacy screening weights carried **verbatim for cross-tool comparability — not consumed by harness physics** |
| `t_fluid_min_c` / `t_fluid_max_c` | °C | heat-transfer-fluid actuator band for the dynamic envs (Bed1D, TwoBed) |
| `source_schedule`, `load_schedule` | – | optional `t [s] → °C / kW` time-varying drives (H2+); `None` for steady use |

Equilibrium cycle per profile (`harness.physics.cycle0d.simulate_cycle`,
water refrigerant, D–A isotherm):

```
q_ads = uptake(t_ads = t_cond, p_sat(t_evap))     # bed cooled to t_cond, evaporator pressure
q_des = uptake(t_des,           p_sat(t_cond))    # bed regenerated, condenser back-pressure
Δq    = max(0, q_ads − q_des)                     # working capacity
q_cool = Δq · h_fg(t_evap)
COP    = q_cool / (hx · (sensible + Δq · Q_st))   # hx = hx_mass_factor (metal/inventory penalty)
SCP    = q_cool / cycle_time_s                    # specific cooling power, W per kg adsorbent
```

The profile-weighted score min-max normalizes with the canonical ranges
carried from the legacy screen — COP ∈ (0.05, 0.85), SCP ∈ (20, 1600) W/kg
(`Cycle0D.CANONICAL_NORMALIZATION`) — then takes the weight-normalized sum.
`rank` = 1 is best within one profile; ranks are never compared across
profiles.

### The four built-ins

| | `cpu` | `human` | `vehicle` | `datacenter` |
|---|---|---|---|---|
| scenario | CPU / electronics cold-plate assist | human thermal comfort / HVAC | vehicle waste-heat cooling | data-center waste-heat cooling |
| `t_evap_c` | 18 | 10 | 7 | 16 |
| `t_cond_c` | 35 | 35 | 45 | 35 |
| `t_des_c` | 75 | 80 | **120** | **60** |
| `cycle_time_s` | **120** | 600 | 180 | 300 |
| `cop_weight` | 0.20 | **0.40** | 0.20 | 0.35 |
| `scp_weight` | **0.45** | 0.20 | **0.40** | 0.30 |
| fluid band | 60–85 °C | 75–95 °C | 90–130 °C | 45–70 °C |

**`cpu` — cold-plate assist.** The power-density case: tightest cycle
(120 s half-cycle) and the highest SCP weight, so it punishes anything with
slow kinetics or poor bed heat transfer. Practical use is rack/cold-plate
scale rather than inside a chip package (the notes field says this; the
physics can't). The H2.2 schedule gate ran here: TwoBed on the 60–85 °C
band with a compressed 30-min load period — optimizing the switch schedule
against the fixed nominal bought **COP +17 % / SCP +9 %**, while stacking
extra sinusoids on the source *lost* cooling (film disconnect between bed
and fluid — the gate's key negative result).

**`human` — HVAC / solar thermal.** The COP-weighted case (0.40): buildings
care about efficiency over power density, and the 80 °C regeneration is
reachable by flat-plate solar or HVAC waste heat. Longest cycle (600 s).
Notes flag the intended pairing: solar-thermal/waste-heat adsorption
chillers on water refrigerant.

**`vehicle` — exhaust-grade waste heat.** Hottest regeneration (120 °C) and
second-highest SCP weight; compactness, vibration, and fast cycling
dominate in reality. In the current fitted table this profile is the
**out-of-window honesty lever**: most isotherms were measured well below
120 °C, so top rows carry `out_of_window = True` — the ranking is an
extrapolation there by construction, flagged rather than hidden
(`rank.sweep_materials` column `out_of_window`: regeneration setpoint above
the fit window's top, or condensation below its bottom).

**`datacenter` — the hard low-grade case.** Only 60 °C of regeneration from
a warm-water coolant loop — deliberately the most constrained profile. This
is where H2.3's headline landed: zeolite 13X — the textbook adsorbent —
sits near the bottom of the table (rank 19/21 in the current sweep below)
because its D–A working capacity collapses at 60 °C regeneration. The
profile also has the dynamic variant below, and the `datacenter` coolant
loop is the motivating scenario for the TwoBed schedule work (H2.2+).

**`datacenter_dynamic` — same setpoints, time-varying source.** A
registered fifth profile whose `source_schedule` is the 45–70 °C
`datacenter_coolant_loop()` sinusoid on a 4 h period (IT load following).
The H2.2 gate compresses the same shape to 30 min for CI speed. Schedules
are plain data — pass your own callable to the env instead of registering
a new profile when an experiment needs a different drive.

Fluid bands derive from each profile's regeneration temperature and its
stated heat-source range in the notes (datacenter loops 45–70 °C; solar/HVAC
collectors 75–95 °C; exhaust-grade heat up to ~130 °C; cpu above its 75 °C
setpoint).

### Reference rankings (T2, current fitted table)

Every honestly-flagged adsorbent in the D–A fit export through Cycle0D per
profile (`rank.load_sweep_materials()` + `rank.sweep_materials`; 21
adsorbents pass the filter as of 2026-09). Top-5 with equilibrium COP/SCP —
equilibrium numbers, no transport caveat; the Bed1D refinement
(`rank.refine_with_bed1d`, `workers=` fan-out) is the next, dynamic, stage:

**`cpu`** — 1. MIL-160 (0.938, COP 0.69, SCP 2351, out-of-window) · 2. MIL-100 (0.920, COP 0.64, SCP 4663) · 3. CuBTC (0.914, COP 0.63, SCP 3913) · 4. Silica Gel (0.887) · 5. Zn₂C₁₄N₂O₈H₄ (0.861, out-of-window). Zeolite 13X: rank 17/21.

**`human`** — 1. MIL-160 (0.733, COP 0.76, out-of-window) · 2. CuBTC (0.587) · 3. MIL-100 (0.540, out-of-window) · 4. Sorbonorit 4 AC (0.440) · 5. Silica Gel (0.420). 13X: rank 14/21.

**`vehicle`** — 1. MIL-160 (0.965, out-of-window) · 2. CuBTC (0.884, out-of-window) · 3. MOF-74-Ni (0.821, out-of-window) · 4. Silica Gel (0.398, out-of-window) · 5. Grade 03 Silica Gel (0.336, out-of-window). 13X: rank 9/21. Read as: *nothing measured so far covers 120 °C regeneration — a measurement gap, not a winner list.*

**`datacenter`** — 1. MIL-100 (0.495, COP 0.57, SCP 523) · 2. CuBTC (0.352) · 3. Xtrusorb oxidized (0.332, out-of-window) · 4. Silica Gel (0.317) · 5. Sorbonorit 4 AC (0.303). 13X: rank 19/21 — the H2.3 finding in one number.

Regenerate: `JAX_PLATFORMS=cpu uv run python -c "import harness; from harness import rank; mats = rank.load_sweep_materials(); df = rank.sweep_materials(mats); print(rank.shortlist(df, 'datacenter', k=10))"` (~60 ms; `sweep_materials_batched` is the vmapped equivalent).

### Adding your own application

One data row, no core change (the extension rule):

```python
from harness.profiles import ApplicationProfile
from harness.envs.bed1d import Bed1D

battery = ApplicationProfile(
    name="battery", description="Battery pack thermal buffer",
    t_evap_c=20.0, t_cond_c=40.0, t_des_c=85.0, cycle_time_s=240.0,
    cop_weight=0.25, scp_weight=0.35,
    t_fluid_min_c=65.0, t_fluid_max_c=95.0,
    notes="Passive buffering; weight SCP over COP for compactness.")
env = Bed1D("anchor:Silica gel RD", battery)   # instances bypass the registry
```

or `harness.register_profile("battery", lambda: battery)` to make it appear
in the GUIs' pickers. The GUIs list registry keys but accept raw values
(lab posture); the data explorer's rankings page offers the four built-ins
as combos.

Related calibration note: the V4 literature calibration
([`harness/benchmarks.md`](../harness/benchmarks.md)) is **per physical
rig** (Uyun 2009, Sztekler 2021), not per profile — rigs validate the
transport physics, profiles define scenarios; don't conflate the two axes.

---

## Part II — Diffheat PDE application scenarios (frozen reference)

> **Status: frozen.** The `diffheat` library was removed from the repo at
> `f36e929` (2026-09 simplification; recoverable from git history at
> `diffheat/`). This section is the kept design reference: each entry is a
> composition of PDE operators, boundary conditions, parameter ranges, and
> a target optimization problem. §3 is the scenario most likely to return
> inside the ML pipeline (bed-level modeling behind the surrogate);
> `adsorbent-ml/data/ACQUISITION.md` maps its moisture-transport work to §1.

### 1. Natural Convection (Boussinesq)

**PDE system:**
- ∂T/∂t + **u**·∇T = α ∇²T (energy)
- ∂**u**/∂t + **u**·∇**u** = −∇p + ν ∇²**u** + β g (T − T_ref) **k** (momentum)
- ∇·**u** = 0 (continuity)

**Parameters:** Rayleigh number (Ra), Prandtl number (Pr), aspect ratio.

**Boundary conditions:** heated bottom plate, cooled top plate, insulated
side walls, no-slip velocity on all walls.

**Optimization target:** maximize heat transfer (Nusselt number) for given
Ra; optimize heating pattern for uniform cooling.

### 2. Thermoelectric Cooling (Peltier/Seebeck)

**PDE system:**
- ρ c_p ∂T/∂t = ∇·(k ∇T) + J²/σ − τ J·∇T (heat + Joule + Thomson)
- ∇·(σ ∇V) = −∇·(σ S ∇T) (electric potential with Seebeck source)

**Parameters:** Seebeck coefficient (S), electrical conductivity (σ),
thermal conductivity (k), Thomson coefficient (τ).

**Boundary conditions:** fixed voltage at contacts, convective heat
transfer at boundaries.

**Optimization target:** maximize cooling ΔT for given input current;
minimize power consumption for target cooling.

### 3. Absorption Chiller Cycle

**Model type:** lumped ODE system (not PDE — would require `diffheat` ODE
support).

**4-component cycle:** generator → condenser → evaporator → absorber.

**Parameters:** heat input (Q_gen), cooling output (Q_evap), solution
concentrations, mass flow rates.

**Optimization target:** maximize COP (Q_evap / Q_gen); optimize cycle
temperatures for given heat source.

*Bridge to Part I:* the harness's Cycle0D is the adsorption-clade cousin of
this scenario (same generator/condenser/evaporator roles, solid sorption
instead of a liquid solution) — exactly why §3 is the first Part-II
scenario to revisit if PDE-level work resumes.

### 4. Thermal Cloak Optimization

**PDE system:** ∂T/∂t = ∇·(κ(x,y) ∇T) (spatially-varying conductivity).

**Goal:** optimize the κ(x,y) field so a protected interior region
experiences minimal temperature gradient while the external temperature
field appears undisturbed.

**Parameters:** conductivity range [κ_min, κ_max], cloak geometry.

**Optimization target:** minimize |∇T| inside the protected region while
matching the far-field temperature profile.

### 5. Forced Convection Cooling

**PDE system:**
- ∂T/∂t + **u**·∇T = α ∇²T (advection-diffusion)
- **u**(x,y) prescribed (not solved — one-way coupling)

**Parameters:** Péclet number (Pe = UL/α), channel geometry.

**Boundary conditions:** inlet temperature, convective outlet, heated
component at center.

**Optimization target:** optimize inlet velocity profile or channel
geometry to minimize component temperature.

### 6. Irregular Domains

**Status:** not supported by the removed library (required unstructured
mesh support — `GridUnstructured` — and finite volume / finite element
operator assembly). Listed for completeness as a capability gap, not an
application. Blockers were: mesh generation (triangulation), unstructured
operator stencils (neighbor lookups), boundary-condition application on
curved edges.

---

*Part I data: `harness/profiles.py` (setpoints/weights verbatim from the
archived `Materials/heat_cooling_screen.py`); physics:
`harness/physics/cycle0d.py` + `harness/physics/thermo.py`; rankings
regenerated 2026-09-13 from the H1.0 fit export
(`data_cache/fits/da_params.csv`, 21 usable adsorbents). Part II carried
over from the pre-simplification version of this file.*

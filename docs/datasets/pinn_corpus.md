# PINN corpus — synthetic fields + real-condition rows

**Role:** training inputs for the bed PINN (N1): infinite cheap synthetic
phase fields for the curriculum base + scarce real-condition rows for
physics-residual collocation and hardening.

## Source

- Builder: `adsorbent-ml/data/corpus.py` on the trusted solver
  (`harness.physics.bed1d.step_bed` — exact-exponential LDF + RK4 conduction).
- Synthetic: single-phase `T(x,t)` / `q(x,t)` rollouts with FIXED fluid T
  and vapour P (no valve logic — the PINN scope boundary); `dt=None`
  auto-derives from the conduction CFL bound (the `Bed1D` env convention).
- Real: ISODB-derived D–A fits + 13 anchors as *conditioning rows* (no `T(x,t)`
  fields — never reported as field accuracy). Single-T rows without `Q_st`
  are SKIPPED for N1 (the energy equation needs it; imputing would launder
  a guess into a physics label). They remain valid N2 targets.
- Transport where fits carry none: class defaults (`rho_s` 800 kg/m³,
  `c_s` 1000 J/kg/K, `k_eff` 0.2 W/m/K, `k_ldf` 2.5e-4 1/s) — flagged via
  `transport_provenance`, never silent (DESIGN §8.1 honesty rule).

## Build (in-memory; no cache artifact)

```python
from corpus import preset_configs, build_corpus, load_real_conditions
data, manifest = build_corpus(preset_configs(), n_data_per_config=800)
conds, report = load_real_conditions()
```

Preset sweep: 2 materials × 2 thicknesses (mm-scale so CFL dt stays
tractable); kinetics τ ~ 100 s, horizons ~40–100 s — the coupled transient
the PINN must learn.

## Splits

- `family_split()`: whole families to train/val — never random
  (random splits leak near-duplicate frameworks).

## Feeds

- `models/bed_pinn.py` coordinate PINN (`x_hat, t_hat, cond[15] → T, q`).
- `training/train_pinn.py` curriculum (data warm-up → physics ramp →
  real-condition hardening); `eval/pinn_metrics.py` acceptance gates.

## Validation

| gate | ok | issues |
|---|---|---|
| `fits` | ✅ | — |
| `labels` | ✅ | — |
| `matched` | ✅ | — |
| `stability` | ✅ | — |
| `refrigerants` | ✅ | — |

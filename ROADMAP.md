# Cooling-with-Heat — Roadmap

> **Simplification (2026-09).** The project is now scoped to one question:
> **how far can deep learning (PINN surrogates + RL control) go on heat-driven
> adsorption cooling?** Everything not serving that was removed from the tree:
> the frozen `diffheat` library, a generic PDE trial zoo, and the legacy
> `Materials/` screening effort (archived at
> `~/ENTERPRISE/_archive/Cooling-with-heat-Materials`; its parity oracle and
> MP query layer were vendored back in). The PyQt6 GUI is parked in `attic/`.
>
> Active development: `harness/` (physics, gym envs, grad/search/RL backends)
> and `adsorbent-ml/` (API-fed data pipeline, bed PINN, baselines, eval).

## Where We Are (2026-09)

Two tracks share one artifact — the trusted JAX physics:

- **harness** — H0–H2.3 done: Cycle0D oracle with V1 parity < 1e-12; the
  dynamic 1-D bed (V3 oracle limit, V4 literature calibration, V5 control
  gradients); the counter-phase two-bed system with heat recovery (V6) and
  schedule optimization beating the fixed schedule (H2.2); T2 reference
  rankings over the fitted material table (H2.3 — 13X bottom-third on the
  datacenter profile, as screened).
- **adsorbent-ml** — data exports done (ISODB: 1,221 pure-water isotherms →
  386 usable D–A fits, 24 adsorbents with multi-temperature Q_st; CoRE MOF
  12k structures; QMOF 20k DFT rows; IZA CIFs + pore table; OPTIMADE pulls
  with CIFs; MOFSimplify stability tables), the N2 tabular baseline trained
  + COP-ranked, and the **N1 bed PINN built end-to-end** (synthetic corpus
  from the harness solver, curriculum training, field metrics).
- **GUIs (2026-09)** — two PyQt6 apps: `harness/gui` (RL & simulation
  launcher over the env registry: evaluate / optimize / sweep with scopes)
  and `adsorbent-ml/gui` (dataset explorer over `data_cache/`). The old
  node-canvas workbench is parked in `attic/`.

## North Star

Deep learning on both axes of the cooling-with-heat problem:

1. **PINN surrogates of the physics** — a physics-informed neural operator
   for the adsorber bed (and later the two-bed system) that evaluates
   trajectories orders of magnitude faster than the finite-volume solver,
   conditioned on material + geometry + schedule, accurate enough to trust
   for screening and control.
2. **RL for cycle control** — policies that run the two-bed machine
   (valve/source scheduling under time-varying heat input) beyond what
   hand-tuned schedules achieve.

And the materials axis feeding both: predict adsorption thermodynamics
(`q_sat`, `Q_st`, D–A `E`, `n`) from crystal structure, so thousands of
candidates can be ranked by system-level performance without simulations.

Design principle: **the model proposes; the simulator disposes.** ML
predictions are always converted to system metrics by the trusted physics
(`harness.physics`), never judged by held-out ML metrics alone.

## Problem Framing

| Task | Input → Output | Feeds |
|---|---|---|
| **T-PINN — Bed surrogate** | (x̂, t̂, material/geometry/schedule) → (T, x) fields with PDE residuals | Fast screening, RL reward shaping |
| **T-RL — Cycle control** | State history → valve/source schedule | TwoBed under time-varying/stochastic heat input |
| **T1 — Forward surrogate** | Crystal structure → `q_sat`, `Q_st`, D–A params | T2 |
| **T2 — System-level ranker** | Predicted props → COP / SCP via the harness cycle oracle per profile | Candidate shortlists |
| **T3 — Active learning** (later) | Model uncertainty → next candidates for expensive evaluation | Label growth |

Generative inverse design stays **out of scope** until a validated forward
surrogate exists (GeoField lesson — see below).

## Data Strategy

Labels are the bottleneck, not models. Sources are pulled via their APIs and
cached with manifests — see
[`adsorbent-ml/data/ACQUISITION.md`](adsorbent-ml/data/ACQUISITION.md) for
per-source status.

- **L1 — Literature/computed datasets** (current): experimental water-isotherm
  curves (NIST ISODB), computed structures/properties (CoRE MOF, QMOF, IZA,
  OPTIMADE providers), stability flags (MOFSimplify). D–A parameters are
  fitted per isotherm so targets stay consistent with the cycle model's
  inputs.
- **L2 — Self-generated labels** (deferred): GCMC via RASPA or ML-potential
  adsorption sims, spent only on candidates active learning selects.
- Synthetic fields for the PINN come from the harness Bed1D solver itself
  (`adsorbent-ml/data/corpus.py`) — free, exact, and on-policy.

Splits are by chemistry family / node type, never random — random splits leak
near-duplicate frameworks.

## Technology Stack (JAX-centered)

| Layer | Choice |
|---|---|
| Envs / physics / RL | `harness` (gymnasium + stable-baselines3 PPO; `rl` extra) |
| PINN + training | pure JAX + `optax` (`adsorbent-ml/models/bed_pinn.py`, `training/train_pinn.py`) |
| Structures / query | `pymatgen` + `mp-api` (optional extras, used by the exporters) |
| Tabular featurization | Magpie-lite composition + pore descriptors (mofdscribe-compatible CSVs) |
| Pretrained foundation models | CHGNet / M3GNet (`matgl`), MACE-MP — offline relaxers/embedders, not reimplemented |
| NN library (Stage-2 GNN) | `equinox`; small hand-rolled crystal-graph message passing |
| Baselines | `scikit-learn` GBDT — the mandatory floor GNNs must beat |
| HPO / tracking | `optuna`; wandb or TensorBoard (not yet wired) |

## Milestone Ladder (DL focus)

Each stage gates the next.

1. **PINN hardening (now)** — train the bed PINN to real-condition accuracy:
   corpus from calibrated materials (V4), curriculum stage C
   (real-condition collocation), rel-L2 + residual gates from
   `eval/pinn_metrics.py`. Gate: matches the Bed1D solver within the V3
   oracle-limit tolerance.
2. **RL on TwoBed (H2.4/H3)** — PPO with a schedule action space on
   `TwoBedSchedule-v0` under time-varying/stochastic source profiles;
   baseline = H2.2 optimized fixed schedules. Gate: beats the schedule
   gate's ≈7% margin honestly (burst-tolerant duty accounting).
3. **Data coverage lift (parallel)** — persist the OPTIMADE bulk MOF pull,
   MOFSimplify stability tables, wire the IZA pore table into features;
   re-run Stage-1 baseline; matched-coverage is the gate (50/156 → majority).
4. **Crystal-graph GNN (Stage 2)** — multi-head shared-latent surrogate,
   log-space heads; two-stage training (structural props first, adsorption
   heads second). Gate: beats the refreshed tabular floor and improves
   top-k hit rate vs the H2.3 reference rankings.
5. **Uncertainty + closed loop (Stage 3)** — deep ensembles drive shortlists
   for expensive L2 evaluation; shortlist enrichment vs random is the gate.

## Evaluation Protocol

1. PINN level: rel-L2 field errors, PDE/BC/IC residual norms,
   autodiff-vs-FD agreement, super-resolution error (`adsorbent-ml/eval/pinn_metrics.py`).
2. RL level: COP/SCP of learned schedules vs optimized fixed schedules on
   identical duty envelopes.
3. Property level: MAE/RMSE on `q_sat`, `Q_st`, D–A `E`; Spearman ρ vs true ranking.
4. System level: predictions through the cycle oracle per profile; **top-k
   hit rate** vs brute-force ranked lists is the business metric.
5. Always reported alongside the tabular baseline.

## Reference Architecture: GeoField

[connorkapoor/geofield-bracket](https://github.com/connorkapoor/geofield-bracket)
is the strongest available template for the materials end state ("learning
where rules are weak, rules where they are exact, simulation where trust
matters"). Lessons adopted:

- **Free-form generation fails** on reconstruction-trained latents (blobs);
  valid designs are isolated islands. If guided generation ever happens it
  will be parameter-space generation over an exact parametric family +
  verifier loop.
- **Log-space heads are non-negotiable** for wide-range physical quantities.
- **Calibrate datasets into the decision-relevant regime.**
- **Feed explicit engineering features** (Polanyi potential
  `A = RT·ln(Psat/P)`, pore-limiting diameter, accessible volume,
  regeneration ΔT) rather than making networks rediscover them.
- **Extension contract:** new physics = one head + one labeler + one
  verifier, never backbone changes.

Caution: GeoField is PyTorch and AGPL-3.0 — borrow patterns, not code.

## Target Repository Layout

```
Cooling-with-heat/
├── harness/             # physics + gym envs + grad/search/RL backends
│                        #   (see harness/DESIGN.md) — T2 ranking loop,
│                        #   PINN oracle, RL reward source
├── adsorbent-ml/        # data (API exporters), features, models (bed PINN,
│                        #   tabular baseline), training, eval
├── attic/               # parked work (PyQt6 GUI) — restorable, not maintained
├── tests/               # tests/harness, tests/adsorbent_ml (+ vendored V1
│                        #   parity oracle in tests/harness/reference)
└── data_cache/          # all datasets, gitignored, manifest-per-build
```

## Open Decisions

- [x] Which L1 dataset first? → **Resolved**: NIST ISODB water isotherms +
      CoRE MOF + QMOF + IZA/anchors — status in
      [`adsorbent-ml/data/ACQUISITION.md`](adsorbent-ml/data/ACQUISITION.md).
- [x] Port `cooling_physics.py` to JAX early? → **Resolved 2026-08**: yes —
      harness H0 (oracle parity < 1e-12).
- [ ] GPU availability for PINN scaling + GNN sweeps (current tracks run on CPU).
- [ ] RL action space: discrete valve schedule vs continuous source modulation first?
- [ ] Shared-latent multi-head surrogate vs independent per-property models as Stage-2 default?

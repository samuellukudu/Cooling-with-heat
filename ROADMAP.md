# Cooling-with-Heat — Roadmap

> **Direction change (2026-08).** This project started as *diffheat*, a
> hand-built differentiable simulation library. It has pivoted: we now focus
> on **training machine-learning models** for heat-driven cooling material
> discovery, using mature libraries and APIs instead of building physics
> infrastructure from scratch.
>
> - `diffheat/` is **frozen** — kept as a working reference, no new features.
> - `Materials/` screening experiments are **kept as data tooling** — their
>   query layer exports candidates, their cycle simulator evaluates them.
> - Active development happens in adsorbent-ML work (see stage ladder below).

## North Star

Train surrogates that predict adsorption thermodynamics (`q_sat`, `Q_st`,
Dubinin–Astakhov parameters) directly from crystal structure, so thousands of
candidate materials can be ranked by system-level performance (COP / SCP per
application profile) without running expensive simulations for each one.

Design principle: **the model predicts material properties; trusted physics
(`Materials/cooling_physics.py`) converts them to system metrics.**
The model proposes; the simulator disposes.

## Problem Framing

| Task | Input → Output | Feeds |
|---|---|---|
| **T1 — Forward surrogate** | Crystal structure → `q_sat`, `Q_st`, D–A params (`E`, `n`) | T2 |
| **T2 — System-level ranker** | Predicted props → COP / SCP via `simulate_adsorption_cycle()` per application profile | Candidate shortlists |
| **T3 — Active learning** | Model uncertainty → select next candidates for expensive evaluation | Label growth |

Generative inverse design is explicitly **out of scope** until a validated
forward surrogate exists (see GeoField lessons below).

## Data Strategy

Labels are the bottleneck, not models. Three tiers:

- **L0 — Proxy labels** (current heuristics in `heat_cooling_screen.py`):
  pipeline smoke tests only. A model trained on these merely echoes the
  heuristics — never report its accuracy as scientific results.
- **L1 — Literature/computed datasets**: published computed water-isotherm
  sets on framework databases (CoRE MOF, hypothetical MOFs, zeotypes) and
  experimental curves (ISODB/NIST). Dubinin–Astakhov parameters are fitted per
  isotherm so targets stay consistent with the cycle model's inputs.
- **L2 — Self-generated labels** (GCMC via RASPA or ML-potential-accelerated
  adsorption sims): deferred until active learning justifies spending compute
  on selected candidates only.

Splits are by chemistry family / node type, never random — random splits leak
near-duplicate frameworks.

## Technology Stack (JAX-centered)

| Layer | Choice |
|---|---|
| Structures / query | `pymatgen` + `mp-api` (already in use) |
| Tabular featurization | `matminer`, `mofdscribe` (MOF/pore descriptors), Zeo++-style pore geometry |
| Pretrained foundation models | CHGNet / M3GNet (`matgl`), MACE-MP — used **offline** as relaxers/embedding generators, not reimplemented |
| NN library in JAX | `equinox` (or flax.nnx); small hand-rolled crystal-graph message passing (jraph is archived) |
| Optimizers / checkpoints | `optax`, `orbax-checkpoint` |
| Baselines | `scikit-learn` (+ optional XGBoost) — GBDT on matminer features is the mandatory floor GNNs must beat |
| HPO / tracking | `optuna`; wandb or TensorBoard |

## Stage Ladder

Each stage gates the next.

0. **Data plumbing** — export MP candidates once to parquet + structures;
   featurize; reproducible dataset build. *(proxy labels OK here)*
1. **Tabular baseline** — GBDT/RF on featurized structures → `q_sat`, `Q_st`
   on real L1 labels. Honest family-split CV error establishes the floor.
2. **Crystal-graph GNN in JAX** — multi-head shared-latent surrogate.
   Log-space heads for quantities spanning decades. Two-stage training:
   structural props first (abundant), adsorption heads second (scarce, low LR).
3. **Uncertainty + closed loop** — deep ensembles drive shortlists for
   expensive L2 evaluation; shortlist enrichment vs random is the gate.
4. **Guided generation** (deferred) — JAX-port of the cycle sim, gradient-based
   guidance, parameter-space generation over a parametric framework family.

## Evaluation Protocol

1. Property level: MAE/RMSE on `q_sat`, `Q_st`, D–A `E`; Spearman ρ vs true ranking.
2. System level: predictions through the cycle sim per app profile; **top-k hit
   rate** vs brute-force ranked lists is the business metric.
3. Calibration: uncertainty intervals must cover held-out errors (needed for Stage 3).
4. Always reported alongside the Stage-1 tabular baseline.

## Reference Architecture: GeoField

[connorkapoor/geofield-bracket](https://github.com/connorkapoor/geofield-bracket)
is the strongest available template for our end state ("learning where rules
are weak, rules where they are exact, simulation where trust matters").
Lessons adopted:

- **Free-form generation fails** on reconstruction-trained latents (blobs);
  valid designs are isolated islands. If Phase 3 ever happens it will be
  parameter-space generation over an exact parametric family + verifier loop.
- **Log-space heads are non-negotiable** for wide-range physical quantities.
- **Calibrate datasets into the decision-relevant regime** (their analog:
  load cases spanning 30–95% yield utilization).
- **Feed explicit engineering features** (Polanyi potential `A = RT·ln(Psat/P)`,
  pore-limiting diameter, accessible volume, regeneration ΔT) rather than
  making networks rediscover them.
- **Extension contract:** new physics = one head + one labeler + one verifier,
  never backbone changes.

Caution: GeoField is PyTorch and AGPL-3.0 — borrow patterns, not code.

## Target Repository Layout

```
Cooling-with-heat/
├── diffheat/            # FROZEN — reference library
├── Materials/           # legacy screening + data-export tooling
├── docs/
└── adsorbent-ml/        # NEW home for ML work
    ├── data/            # mp_export.py, isotherms.py, fit_da.py
    ├── features/        # matminer/mofdscribe wrappers → numpy
    ├── models/          # baseline.py (sklearn), crystal_gnn.py (equinox)
    ├── training/        # train.py (optax loop, optuna), track.py
    └── eval/            # metrics.py, rank.py (predictions → COP/SCP tables)
```

Environment: extend the root `pyproject.toml` (uv-managed) with an `ml` extra
rather than maintaining a second venv.

## Open Decisions

- [x] Which L1 dataset first? → **Resolved**: NIST ISODB water isotherms +
      CoRE MOF + QMOF + IZA/anchors, in that order — concrete plan in
      [`adsorbent-ml/data/ACQUISITION.md`](adsorbent-ml/data/ACQUISITION.md).
- [ ] GPU availability for Stage 2+ (Stages 0–1 run fine on CPU)?
- [ ] Port `cooling_physics.py` to JAX early (gradient diagnostics) or wait for Stage 3?
- [ ] Parametric framework generator (GeoField-style family) early vs database-screening-first?
- [ ] Shared-latent multi-head surrogate vs independent per-property models as Stage-2 default?

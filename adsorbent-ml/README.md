# adsorbent-ml

ML pipeline for heat-driven cooling adsorbent discovery — see
[`../ROADMAP.md`](../ROADMAP.md) for the full strategy.

Layout (stages fill in progressively):

```
data/       dataset builders: mp_export.py (Stage 0); isotherms.py, fit_da.py
            -> see data/ACQUISITION.md for the concrete external-data plan
            federated structures: optimade_export.py (§5, one client ~40 DBs)
            working fluids: refrigerants.py (§7, water/methanol/ethanol/ammonia)
            feasibility: stability.py (§6, MOFSimplify/TGA label ingest)
features/   matminer/mofdscribe featurization wrappers -> numpy        (Stage 1)
models/     sklearn baseline; equinox crystal-graph GNN               (Stages 1-2)
training/   optax training loop, optuna HPO, experiment tracking      (Stage 2)
eval/       metrics (MAE, Spearman, top-k hit rate); COP/SCP ranking  (all stages)
```

## Stage 0 — data export

`data/mp_export.py` queries Materials Project **once** via the vendored query
layer in `data/mp_screen.py` (from the archived legacy screen) and caches
everything locally. Needs `mp_api` + `pymatgen` at call time and
`MP_API_KEY` in `adsorbent-ml/.env` or the environment:

```bash
# smoke test (a few queries, a few structures)
python data/mp_export.py \
    --apps datacenter --max-generated-chemsys 3 --limit-per-system 5 \
    --with-structures --structure-limit 5

# full export for one application profile (~1400 chemsys queries)
python data/mp_export.py --apps human
```

Output lands in `data_cache/mp/` (gitignored):

- `candidates.parquet` — one row per unique material
- `structures/*.cif` — optional, resumable downloads
- `manifest.json` — args, counts, durations

Re-running resumes structure downloads and never re-downloads existing CIFs.

## Neural differentiable simulators (N1 PINN track)

Standard: `https://acceleratedunderstanding.com/` — neural nets *are* the
differentiable simulators (direct one-shot rollouts, resolution-invariant,
broad conditioned models, directional feedback via autodiff, known physics
as the scoring signal). `diffheat/` + `harness/` are data generators and
verifiers, not the optimization path.

| Module | Role |
|---|---|
| `data/corpus.py` | Unified corpus: single-phase `T(x,t)`/`q(x,t)` fields from the trusted `harness.physics.bed1d.step_bed` (CFL auto-dt per the `Bed1D` env convention) + real-condition rows from anchors/fits (single-T rows without `Q_st` skipped, never imputed) + leak-free `family_split` |
| `models/bed_pinn.py` | Coordinate-based PINN `(x_hat, t_hat, cond[15]) → (T, q)`: bed-energy + LDF residuals via autodiff, convective/adiabatic BCs, IC. Pure JAX, no new deps |
| `training/train_pinn.py` | Curriculum CLI: data warm-up → physics ramp → real-condition hardening; cosine-decay Adam, `.npz` checkpoints |
| `eval/pinn_metrics.py` | Acceptance: rel-L2 fields, residual norms, grad-vs-FD agreement (V5-style), super-resolution error, `Q_cool` proxy |

```bash
python3 adsorbent-ml/training/train_pinn.py --steps 2000 --n-data 800
python3 -m pytest tests/adsorbent_ml/test_bed_pinn.py -q
```

Device note: everything is `jax`-device-agnostic (CPU now; GPU lights up
with a CUDA jaxlib — the RTX on this machine needs `jax[cuda12]`, a
separate install decision).

## N2-v1 — structure→property floor (Stage 1 baseline)

Data moat first, model second. One row per material, leak-safe by family:

```
data/labels.py            143 ISODB (median-aggregated) + 13 anchors; log-space
                          targets; Q_st multi-T subset only, never imputed
data/match_structures.py  IZA rules + self-verifying CoRE substrings
                          → matched.parquet + unmatched_names.csv (famous UKEBOD
                          lesson locked as a test: short substrings rejected)
features/composition.py   dependency-free formula parser + Magpie-lite stats +
                          curated class-formula table (no pymatgen here)
features/descriptors.py   pores (CoRE exemplar) + family one-hots + composition
                          → 30 numeric cols, NaN where honestly unknown
models/tabular_baseline.py  HistGradientBoosting per target, GroupKFold OOF,
                          dummy-relative skill in every report
eval/rank_eval.py         OOF predictions → cycle COP per profile → Spearman +
                          top-k hit rate (the business metric)
training/train_baseline.py  thin CLI → data_cache/n2/ (labels, matched,
                          bundle, report)
```

```bash
python3 adsorbent-ml/training/train_baseline.py --out data_cache/n2
```

**Honest floor (2026-09 refresh, 156 materials, grouped OOF):** matching
covers 50/156 (22 IZA + 19 Ongari-CSD + 7 CSD-fuzzy + 2 CoRE-substring; DOI
bridge implemented, 0 hits on this cache — water papers ≠ structure papers
here); pores now on 44/156 rows — 19 CoRE + 22 IZA-SC framework-table LCD/PLD
(`pore_iza=1`, wired into `features/descriptors.py`) + 3 QMOF fallbacks —
pore-consistency gate live (0 exclusions). Skill vs dummy, grouped OOF:
q_sat +0.12, Q_st +0.21 (both turned positive with the IZA pores); E −0.55,
n −0.16 still negative. rank@datacenter top-10 hit 0.30, rank@human 0.20.
The remaining bottleneck is the 106 unmatched materials — OPTIMADE pulls
with CIF conversion are cached (`data_cache/optimade/`, 2,221 entries) and
feed the next matching round. The GNN (Stage 2, equinox
crystal-graph, shared latent + log heads, MOFid-topology splits) starts when
matched coverage makes it honest.

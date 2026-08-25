# adsorbent-ml

ML pipeline for heat-driven cooling adsorbent discovery — see
[`../ROADMAP.md`](../ROADMAP.md) for the full strategy.

Layout (stages fill in progressively):

```
data/       dataset builders: mp_export.py (Stage 0), isotherms.py, fit_da.py
features/   matminer/mofdscribe featurization wrappers -> numpy        (Stage 1)
models/     sklearn baseline; equinox crystal-graph GNN               (Stages 1-2)
training/   optax training loop, optuna HPO, experiment tracking      (Stage 2)
eval/       metrics (MAE, Spearman, top-k hit rate); COP/SCP ranking  (all stages)
```

## Stage 0 — data export

`data/mp_export.py` queries Materials Project **once** via the battle-tested
layer in `../Materials/heat_cooling_screen.py` and caches everything locally:

```bash
# smoke test (a few queries, a few structures)
../../Materials/.venv/bin/python data/mp_export.py \
    --apps datacenter --max-generated-chemsys 3 --limit-per-system 5 \
    --with-structures --structure-limit 5

# full export for one application profile (~1400 chemsys queries)
../../Materials/.venv/bin/python data/mp_export.py --apps human
```

Output lands in `data_cache/mp/` (gitignored):

- `candidates.parquet` — one row per unique material
- `structures/*.cif` — optional, resumable downloads
- `manifest.json` — args, counts, durations

Re-running resumes structure downloads and never re-downloads existing CIFs.

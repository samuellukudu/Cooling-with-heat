# Dataset cards

Generated from live `data_cache/` manifests — do not hand-edit numbers here.
Regenerate after any rebuild:

```bash
python adsorbent-ml/data/cards.py
```

## Lineage

```
ISODB raw mirror → normalized isotherms → D–A fits → N2 labels ─┬─→ features → baseline → rank_eval
structures (CoRE/QMOF/IZA/OPTIMADE/MP) ──→ matching ─────────────┘        (top-k hit rate)
stability (MOFSimplify) ──→ feasibility gate ──→ screening shortlists
refrigerants (Antoine/Watson) ──→ validator + working-pair expansion
harness Bed1D solver ──→ PINN corpus (synthetic fields + real conditions)
```

## Cards

- [`isodb.md`](isodb.md) — ISODB — experimental water isotherms
- [`fits.md`](fits.md) — Fits — D–A parameters per isotherm
- [`labels.md`](labels.md) — Labels — N2 per-material training table
- [`structures.md`](structures.md) — Structures — frameworks behind the labels
- [`stability.md`](stability.md) — Stability — feasibility gate before ranking
- [`refrigerants.md`](refrigerants.md) — Refrigerants — working fluids beyond water
- [`pinn_corpus.md`](pinn_corpus.md) — PINN corpus — synthetic fields + real rows

## Validation snapshot (generated at card-build time)

| gate | ok | issues |
|---|---|---|
| `fits` | ✅ | — |
| `labels` | ✅ | — |
| `matched` | ✅ | — |
| `stability` | ✅ | — |
| `refrigerants` | ✅ | — |

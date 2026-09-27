# Labels — N2 per-material training table

**Role:** one row per material: the supervised table the tabular baseline
and (gated) the Stage-2 GNN train against.

## Source

- Builder: `adsorbent-ml/data/labels.py` — usable ISODB fits
  (`fit_flag == ok`, physical `q_sat`, mass-normalized) median-aggregated
  across isotherms; spread kept as `q_sat_logstd` (honest disagreement proxy).
- 13 commercial anchors (`adsorbent-ml/data/anchors.csv`) appended as gold
  rows (`anchor=True`): silica gel RD, 13X/NaA, AlPO-18/SAPO-34, MIL-101(Cr),
  UiO-66, Mg-MOF-74, CAU-10-H, aluminum fumarate, silicalite, carbon, LiCl/silica.
- Targets are log-space (`log_q_sat`, `log_Q_st`, `log_E`; `n_da` linear);
  `log_Q_st` on the multi-T subset only — single-T rows carry no isosteric
  heat and are never imputed.
- Families: v1 keyword rules (zeolite/silica/carbon/MOF/COF/composite/other),
  coarse but leak-safe for GroupKFold; node/topology families arrive Stage 2.

## Build

```bash
python adsorbent-ml/training/train_baseline.py --out data_cache/n2   # writes labels.csv + manifest
python adsorbent-ml/data/match_structures.py                          # writes matched.parquet + unmatched_names.csv
```

## Counts (from `data_cache/n2/manifest.json` + live probe)

| key | value |
|---|---|
| `n_materials` | 156 |
| `n_isodb` | 143 |
| `n_anchors` | 13 |
| `n_with_Q_st` | 34 |
| `family:carbon` | 42 |
| `family:cof` | 1 |
| `family:composite` | 1 |
| `family:mof` | 49 |
| `family:other` | 33 |
| `family:silica` | 9 |
| `family:zeolite` | 21 |

Matching: **50/156** (32%); by kind `{'none': 106, 'iza': 22, 'csd': 19, 'csd-fuzzy': 7, 'core': 2}`.

Name→structure matching (cheapest first): IZA trade-name rules → verified
CoRE substrings (≥1 filename hit required; <4-char substrings rejected —
the UKEBOD lesson) → anchor join → Ongari CSD tables → QMOF join →
DOI bridge. Remainder → `unmatched_names.csv` curation queue.

## Schema (`labels.csv`)

| column | notes |
|---|---|---|
| `material_id, name, source, anchor, family` | identity + split key |
| `q_sat_kg_kg, q_sat_logstd, n_isotherms` | median + spread + support |
| `q_st_j_kg` | NaN when single-T (never imputed) |
| `e_char_j_mol, n_da` | D–A medians |
| `log_q_sat, log_Q_st, log_E` | training targets |

## Feeds

- `features/descriptors.py` → 30-col matrix (pores + family + composition).
- `models/tabular_baseline.py` → HistGradientBoosting per target, GroupKFold OOF.
- `eval/rank_eval.py` → OOF predictions through the cycle oracle (top-k hit rate).

## Validation

| gate | ok | issues |
|---|---|---|
| `fits` | ✅ | — |
| `labels` | ✅ | — |
| `matched` | ✅ | — |
| `stability` | ✅ | — |
| `refrigerants` | ✅ | — |

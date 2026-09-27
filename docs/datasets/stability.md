# Stability — feasibility gate before ranking

**Role:** N2 must not propose adsorbents that collapse under water cycling
or decompose below regeneration temperature. This gate runs BEFORE
system ranking — never as a training target.

## Source / license / access

- MOFSimplify (MIT Kulik lab, Nandy et al. Sci. Data 2022), Zenodo record 5737968
  (`SciData.zip`): `full_SSD_data.csv` (text-mined solvent-removal stability) +
  `full_TSD_data.csv` (TGA decomposition onsets). Manual Zenodo download,
  then normalized by the exporter; raw zip cached under
  `data_cache/stability/_raw/`.

## Build

```bash
python adsorbent-ml/data/stability_export.py          # download + build (cached under _raw/)
python adsorbent-ml/data/stability_export.py --local # rebuild from an existing download
```

## Counts (manifest + live filter probe)

| key | value |
|---|---|
| `rows` | 4148 |
| `n_with_solvent_label` | 2179 |
| `n_with_decomp_c` | 3132 |
| `n_pass_default_gate_150C` | 1293 |
| `n_fail_default_gate_150C` | 2855 |

Default gate (`min_decomp_c=150 °C`, unknown water fails): **1293 pass / 2855 fail** of 4148.

## Schema (`mofsimplify_ssd_tsd.csv`)

| column | notes |
|---|---|---|
| `mof_name` | CSD refcode (one row per refcode, merged SSD+TSD) |
| `water_stable` | intentionally EMPTY — the dataset measures desolvation, not hydrolysis |
| `solvent_removal_stable` | `true/false` text-mined desolvation label |
| `thermal_decomp_c` | TGA onset [°C], empty ⇒ None |
| `source, confidence` | DOI + `text-mined` / `tga-extracted` |

Loader (`stability.py`): alias-tolerant headers; `water_stable` accepts
`true/false/1/0/yes/no/stable/unstable`; rows without a name are skipped (counted).

## Gate semantics (`feasibility_filter`)

- Unknown water stability FAILS by default (`require_water_stable=True`) —
  unknown ≠ stable. Exception: solvent-removal-stable counts as conservative
  desolvation evidence; an explicit `water_stable=false` never passes.
- Missing TGA never fails alone (absence of data is not decomposition).
- Thermal gate: `thermal_decomp_c >= min_decomp_c` (default 150 °C).

## Feeds

- `datasets.stability_shortlist()` → `(pass, fail)` DataFrames for N2 screening.
- GUI data explorer → Stability tab (gate + TGA histogram).

## Validation

| gate | ok | issues |
|---|---|---|
| `fits` | ✅ | — |
| `labels` | ✅ | — |
| `matched` | ✅ | — |
| `stability` | ✅ | — |
| `refrigerants` | ✅ | — |

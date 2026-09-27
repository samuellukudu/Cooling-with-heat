# ISODB — experimental water isotherms

**Role:** L1 experimental adsorption labels — the raw material every
fit, label, and match downstream consumes.

## Source / license / access

- NIST ISODB bulk mirror: `git clone https://github.com/NIST-ISODB/isodb-library`
  → `data_cache/isodb/mirror/` (one JSON per isotherm + `DOI_mapping.csv`).
- Targeted refreshes via the JSON/XML/CSV REST APIs at <https://adsorption.nist.gov>.
- License: source records carry their own citations (see `doi` column);
  the export preserves provenance per row.

## Build

```bash
python adsorbent-ml/data/nist_isodb.py   # → water_isotherms.parquet + adsorbent_index.csv + manifest.json
```

## Counts (from `data_cache/isodb/manifest.json`)

| key | value |
|---|---|
| `total` | 39824 |
| `pure_water` | 1221 |
| `mixed_water` | 1234 |
| `unique_adsorbents` | 557 |
| `rows_in_temp_window` | 1122 |

## Schema (`water_isotherms.parquet`)

| column | notes |
|---|---|
| `isotherm_id, doi, material_name, matched_refcode?` | provenance |
| `T_K, pressure[], uptake[], uptake_unit, original_unit` | originals preserved |
| `temperature` window | 280–380 K application window |

Unit normalization to kg/kg vs Pa happens in `fit_da.py`, not here.

## Gotchas

- Filename casing varies (`isotherm2.json` / `Isotherm4.json`); some files
  are misspelled (`isothem10.json`) — the exporter walks `Library/<doi>/*.json`
  and skips the three registry directories, never filters by filename.
- Uptake units are heterogeneous (mmol/g, cm³(STP)/g, molecules/unitcell,
  g/g, mg/g, wt% …); `molecules/unitcell` needs framework cell composition.
- Material names are non-standard ("Cu-BTC", "MOF-199", "MIL-53(Al)…");
  see the `labels` card for the matching strategy.

## Feeds

- `fits` card (D–A parameters) ← pure-water rows in the 280–380 K window.
- `labels` card matching queue (`unmatched_names.csv`) ← adsorbent index.

## Validation

| gate | ok | issues |
|---|---|---|
| `fits` | ✅ | — |
| `labels` | ✅ | — |
| `matched` | ✅ | — |
| `stability` | ✅ | — |
| `refrigerants` | ✅ | — |

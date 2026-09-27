# Fits — Dubinin–Astakhov parameters per isotherm

**Role:** parameter estimation (not model training): one independent,
deterministic fit per isotherm. Output is the label set Stage-1/2
surrogates train against AND the parameter source `harness.physics` consumes.

## Source

- Library: `adsorbent-ml/data/fit_da.py` — adsorbate- and unit-agnostic
  (`fit_isotherm_da`, `isosteric_heat`, `fit_all`); the CLI is thin.
- Physics: `q = q_sat · exp(−(A/E)^n)`, `A = R·T·ln(Psat(T)/P)`, `R = 8.314 J/(mol·K)`
  — identical constants to `harness.physics.thermo` by contract.
- `Q_st` via Clausius–Clapeyron on fitted curves at common uptake levels;
  `None` when < 2 usable temperatures ≥ 5 K apart (never imputed).

## Build

```bash
python adsorbent-ml/data/fit_da.py   # → data_cache/fits/da_params.csv + da_params_qc.md
```

## Counts (live probe of `da_params.csv`)

| key | value |
|---|---|
| `isotherm_rows` | 1221 |
| `adsorbents` | 557 |
| `adsorbents_with_Q_st` | 24 |
| `flag:insufficient_points` | 400 |
| `flag:ok` | 386 |
| `flag:poor_fit` | 336 |
| `flag:out_of_window` | 99 |

QC report: `data_cache/fits/da_params_qc.md` (4980 bytes)

## Schema (DESIGN §8.1 subset)

| column | unit | notes |
|---|---|---|
| `material_id, name, source, isotherm_id, doi` | — | provenance |
| `q_sat_kg_kg` | kg/kg | mass-normalized only where convertible |
| `q_st_j_kg` | J/kg | multi-T subset only |
| `e_char_j_mol, n_da` | J/mol, – | D–A parameters (unit-independent) |
| `t_range_c` | °C | fit validity window; env warns outside it |
| `fit_rmse, n_points, fit_flag, fit_detail, psat_pa` | — | diagnostics are first-class |

`fit_flag ∈ {ok, poor_fit, S_shaped, insufficient_points, out_of_window}`;
`q_sat_physical` flags mass-normalized q_sat outside 1e-4–2.5 kg/kg
(mislabeled source units — the fit itself stays valid).

## Gotchas

- Points at/above Psat are excluded (condensed phase, outside the model).
- S-shaped (Type-V, MIL-101 class) isotherms fail D–A by design — flagged, kept.
- Single-T adsorbents carry no `Q_st`; downstream N1 conditioning skips them.

## Feeds

- `labels` card ← usable (`ok` + physical + mass-normalized) fits, median-aggregated.
- `harness.materials.load_materials_csv` ← same CSV (§8.1 schema).

## Validation

| gate | ok | issues |
|---|---|---|
| `fits` | ✅ | — |
| `labels` | ✅ | — |
| `matched` | ✅ | — |
| `stability` | ✅ | — |
| `refrigerants` | ✅ | — |

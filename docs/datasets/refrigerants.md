# Refrigerants — working fluids beyond water

**Role:** trusted thermodynamic reference OUTSIDE the differentiable path
(same bounded role as CoolProp): working-pair expansion + validator.
Never imported by `harness.physics` or the PINN.

## Source

- Library: `adsorbent-ml/data/refrigerants.py` (stdlib + numpy).
- Antoine Psat (`log10(P/mmHg) = A − B/(T/°C + C)`, NIST WebBook-consistent
  constants) + Watson latent heat anchored to the tabulated `h_fg` (exact
  at the anchor by construction).

## Fluids

| fluid | Antoine window [°C] | Tb [°C] | ASHRAE 34 | GWP100 |
|---|---|---|---|---|
| `ammonia` | -60–60 | -33.3 | B2L | 0 |
| `ethanol` | 0–80 | 78.4 | — | ~0 (indicative) |
| `methanol` | -20–80 | 64.7 | B2* | ~3 (indicative) |
| `water` | 1–100 | 100 | — | 0 |

Spot check (live): methanol Psat(35 °C) = 28.0 kPa (~28 kPa); ammonia = 1317 kPa (~1350 kPa).

Water branch agrees with `harness.physics.thermo` to < 3 % over 5–95 °C
(both approximate IAPWS-IF97); boiling points at 1 atm within 2 %.

## Fail-loudly contract

- Out-of-window calls RAISE (no silent extrapolation into the desorption band):
  water 1–100 °C, methanol −20–80 °C, ethanol 0–80 °C, ammonia −60–60 °C.
- Unknown fluids raise `KeyError`. Extending a range via CoolProp/REFPROP
  is an explicit follow-up (see ACQUISITION §7).

## Feeds

- Validator upgrades (dense-grid reference calls).
- `rank.py` working-pair shortlists; N1 conditioning rows for non-water
  phases (P ranges differ 10–100× across fluids).
- `datasets.refrigerants_table()` exposes this card's numbers as a DataFrame.

## Validation

| gate | ok | issues |
|---|---|---|
| `fits` | ✅ | — |
| `labels` | ✅ | — |
| `matched` | ✅ | — |
| `stability` | ✅ | — |
| `refrigerants` | ✅ | — |

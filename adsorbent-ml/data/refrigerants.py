#!/usr/bin/env python
"""Working-fluid property library: saturation pressure + latent heat.

Companion to ``adsorbent-ml/data/ACQUISITION.md`` §7 (WebBook refrigerants).
Bounded role, same as CoolProp (ACQUISITION "Adopted" section): a trusted
reference OUTSIDE the differentiable path — never imported by
``harness.physics`` or the PINN. Its two jobs:

1. Working-pair expansion: evaluate adsorption cycles with methanol /
   ethanol / ammonia instead of water-only (methanol Psat(35 °C) ≈ 28 kPa
   vs water ≈ 5.6 kPa changes every pressure ratio in the cycle).
2. Validator: water branch must agree with ``harness.physics.thermo`` to a
   few percent over 0–100 °C (both approximate IAPWS-IF97).

Correlations (Antoine, ``log10(P/mmHg) = A − B/(T/°C + C)``):

- Constants are standard Antoine compilations consistent with the NIST
  Chemistry WebBook. Each fluid declares its validity window and the
  library FAILS LOUDLY outside it (no silent extrapolation into the
  desorption band — extending a fluid's range via CoolProp/REFPROP is an
  explicit follow-up, see ACQUISITION §7).
- Latent heat: Watson correlation anchored to the tabulated ``h_fg`` at a
  reference temperature (exact at the anchor by construction).

Pure stdlib + numpy. Importable without the CLI (fit_da precedent).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

# Antoine constants (P in mmHg, T in °C) + validity window + Watson anchors.
# Spot checks encoded in tests/adsorbent_ml/test_refrigerants.py:
#   methanol Psat(35 °C) ≈ 28 kPa, ammonia Psat(35 °C) ≈ 1350 kPa
#   (ACQUISITION.md §7), water/ethanol boil at 100/78.4 °C (760 mmHg).
REFRIGERANTS: dict[str, dict] = {
    "water": {
        "antoine": {"A": 8.07131, "B": 1730.63, "C": 233.426},
        "t_min_c": 1.0, "t_max_c": 100.0,
        "t_crit_k": 647.1, "h_fg_ref_j_kg": 2_256_400.0, "t_ref_k": 373.15,
        "watson_exp": 0.321, "molar_mass_kg_mol": 0.01801528,
        "tb_c": 100.0, "ashrae34": "—", "gwp100": 0,
    },
    "methanol": {
        "antoine": {"A": 8.08097, "B": 1582.271, "C": 239.726},
        "t_min_c": -20.0, "t_max_c": 80.0,
        "t_crit_k": 512.6, "h_fg_ref_j_kg": 1_100_000.0, "t_ref_k": 337.85,
        "watson_exp": 0.38, "molar_mass_kg_mol": 0.03204,
        "tb_c": 64.7, "ashrae34": "B2*", "gwp100": "~3 (indicative)",
    },
    "ethanol": {
        "antoine": {"A": 8.20417, "B": 1642.89, "C": 230.3},
        "t_min_c": 0.0, "t_max_c": 80.0,
        "t_crit_k": 514.0, "h_fg_ref_j_kg": 855_000.0, "t_ref_k": 351.5,
        "watson_exp": 0.38, "molar_mass_kg_mol": 0.04607,
        "tb_c": 78.4, "ashrae34": "—", "gwp100": "~0 (indicative)",
    },
    "ammonia": {
        "antoine": {"A": 7.36050, "B": 926.132, "C": 240.17},
        "t_min_c": -60.0, "t_max_c": 60.0,
        "t_crit_k": 405.4, "h_fg_ref_j_kg": 1_369_000.0, "t_ref_k": 239.85,
        "watson_exp": 0.38, "molar_mass_kg_mol": 0.017031,
        "tb_c": -33.3, "ashrae34": "B2L", "gwp100": 0,
    },
}
MMHG_TO_PA = 133.322


def list_refrigerants() -> list[str]:
    """Names with tabulated parameters."""
    return sorted(REFRIGERANTS)


def _check_range(name: str, t_c: float) -> dict:
    spec = REFRIGERANTS[name]  # KeyError for unknown fluids: fail loudly
    if not (spec["t_min_c"] <= t_c <= spec["t_max_c"]):
        raise ValueError(
            f"{name}: T={t_c:g} °C outside Antoine window "
            f"[{spec['t_min_c']:g}, {spec['t_max_c']:g}] °C — refusing to "
            "extrapolate (extend via CoolProp/REFPROP, ACQUISITION §7)")
    return spec


def psat_pa(refrigerant: str, t_k: float | np.ndarray) -> float | np.ndarray:
    """Saturation pressure [Pa]. Scalar or numpy array in, same out."""
    spec = REFRIGERANTS[refrigerant]
    t_c = np.asarray(t_k, dtype=float) - 273.15
    if np.any((t_c < spec["t_min_c"]) | (t_c > spec["t_max_c"])):
        worst = float(np.min(t_c) if np.any(t_c < spec["t_min_c"]) else np.max(t_c))
        _check_range(refrigerant, worst)
    a = spec["antoine"]
    mmhg = 10.0 ** (a["A"] - a["B"] / (t_c + a["C"]))
    pa = mmhg * MMHG_TO_PA
    return float(pa) if np.ndim(pa) == 0 else pa


def h_fg_j_kg(refrigerant: str, t_k: float | np.ndarray) -> float | np.ndarray:
    """Latent heat [J/kg] via Watson, anchored to the tabulated value."""
    spec = REFRIGERANTS[refrigerant]
    t = np.asarray(t_k, dtype=float)
    t_ref, t_crit = spec["t_ref_k"], spec["t_crit_k"]
    if np.any((t <= 0.0) | (t >= t_crit)):
        raise ValueError(f"{refrigerant}: T out of (0, Tc={t_crit:g} K)")
    val = spec["h_fg_ref_j_kg"] * ((t_crit - t) / (t_crit - t_ref)) ** spec["watson_exp"]
    return float(val) if np.ndim(val) == 0 else val


def main() -> None:
    ap = argparse.ArgumentParser(description="Spot-check working-fluid properties.")
    ap.add_argument("--fluid", default="methanol", choices=list_refrigerants())
    ap.add_argument("--t-c", type=float, nargs="+", default=[35.0])
    args = ap.parse_args()
    for t_c in args.t_c:
        print(f"{args.fluid} @ {t_c:g} °C: Psat={psat_pa(args.fluid, t_c + 273.15) / 1e3:.2f} kPa, "
              f"h_fg={h_fg_j_kg(args.fluid, t_c + 273.15) / 1e3:.1f} kJ/kg")


if __name__ == "__main__":
    main()

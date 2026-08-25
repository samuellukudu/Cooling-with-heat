"""mp_query_validator.py — Thermodynamic cross-check and MP API query validator.

Runs two kinds of validation:

1. **Physics bench** — compares computed saturation pressures and cycle
   metrics against IAPWS-IF97 spot values and published silica-gel /
   zeolite-13X benchmarks.

2. **MP API query bench** — for each of the 4 applications, fires a small
   representative Materials-Project query (1 chemical system per app) and
   reports what the MP database actually returns so you can sanity-check
   the search-criteria filters.

Usage
-----
    python mp_query_validator.py               # physics bench only (no API key needed)
    python mp_query_validator.py --mp          # also run the MP API query bench
    python mp_query_validator.py --mp --all    # query every chemsys for every app
"""

import argparse
import math
import sys
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Physics imports (must be importable without mp_api)
# ---------------------------------------------------------------------------
try:
    from cooling_physics import (
        simulate_adsorption_cycle,
        water_sat_pressure_pa,
        water_h_fg_j_kg,
    )
except ImportError as exc:
    sys.exit(f"Could not import cooling_physics: {exc}")

from env_utils import get_mp_api_key

try:
    from mp_api.client import MPRester
    HAS_MP = True
except ImportError:
    HAS_MP = False

from heat_cooling_screen import (
    APPLICATIONS,
    SEARCH_CRITERIA,
    TOXIC_OR_COSTLY_ELEMENTS,
    generate_chemsys,
)

# ---------------------------------------------------------------------------
# IAPWS-IF97 reference saturation pressures (Pa)
# Source: IAPWS-IF97 (Wagner & Kruse, 2008)
# ---------------------------------------------------------------------------
IAPWS_SPOT_VALUES: List[Dict] = [
    {"t_c":   0.0, "p_pa_ref":    611.7},
    {"t_c":  10.0, "p_pa_ref":   1227.9},
    {"t_c":  35.0, "p_pa_ref":   5626.7},
    {"t_c":  60.0, "p_pa_ref":  19940.0},
    {"t_c":  80.0, "p_pa_ref":  47390.0},
    {"t_c": 100.0, "p_pa_ref": 101325.0},
    {"t_c": 120.0, "p_pa_ref": 198480.0},
]

# ---------------------------------------------------------------------------
# Literature benchmarks for COP / SCP
# Sources:
#   Silica gel / water (RD-type): Rezk et al. (2012), Appl. Therm. Eng. 39, 156.
#   Zeolite 13X / water:           Aristov (2013), J. Chem. Eng. Data 58, 2787.
# ---------------------------------------------------------------------------
CYCLE_BENCHMARKS: List[Dict] = [
    {
        # Rezk et al. (2012) App. Therm. Eng. 39, 156-165 — silica gel RD-type.
        # SCP range: adsorbent-mass basis (no HX mass), no bed heat recovery.
        "label": "Silica gel (RD) — human/HVAC",
        "q_sat": 0.35,
        "q_st": 2.50e6,
        "app": "human",
        "e_char_j_mol": 4500.0,   # Aristov (2002): E ~ 4.4 kJ/mol for RD silica gel
        "cop_published_min": 0.40,
        "cop_published_max": 0.70,
        "scp_published_min": 100.0,
        "scp_published_max": 550.0,
    },
    {
        # Low-grade heat case — silica gel, T_des = 60 °C.
        # Published COP: Ng et al. (2006) Int. J. Refrig. 29, 34.
        "label": "Silica gel (RD) — datacenter (low-grade)",
        "q_sat": 0.35,
        "q_st": 2.50e6,
        "app": "datacenter",
        "e_char_j_mol": 4500.0,
        "cop_published_min": 0.25,
        "cop_published_max": 0.60,
        "scp_published_min": 60.0,
        "scp_published_max": 400.0,
    },
    {
        # Zeolite 13X / water, vehicle cycle.
        # Ng et al. (2001) Appl. Therm. Eng. 21, 1735.
        # COP reduced by hx_mass_factor vs. adiabatic published value.
        "label": "Zeolite 13X — vehicle (120 °C)",
        "q_sat": 0.28,
        "q_st": 3.50e6,
        "app": "vehicle",
        "e_char_j_mol": 14000.0,  # E ~ 13-16 kJ/mol (Ng et al. 2001)
        "cop_published_min": 0.25,
        "cop_published_max": 0.65,
        "scp_published_min": 200.0,
        "scp_published_max": 1100.0,
    },
    {
        # AlPO-18 / SAPO analogue, CPU fast cycle.
        # Henninger et al. (2012) Prog. Colloid. Polym. Sci. 139.
        "label": "AlPO-18 analogue — cpu (fast cycle)",
        "q_sat": 0.30,
        "q_st": 2.80e6,
        "app": "cpu",
        "e_char_j_mol": 8000.0,   # SAPO/AlPO: E ~ 7-9 kJ/mol
        "cop_published_min": 0.25,
        "cop_published_max": 0.60,
        "scp_published_min": 500.0,
        "scp_published_max": 2500.0,
    },
]

# One representative chemical system per application for the MP query bench
MP_PROBE_SYSTEMS: Dict[str, str] = {
    "cpu":        "Al-O-Si",
    "human":      "Al-O-Si",
    "vehicle":    "Al-O-Ti",
    "datacenter": "Al-Ca-O",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pass_fail(ok: bool) -> str:
    return "✓ PASS" if ok else "✗ FAIL"


def _pct_err(computed: float, reference: float) -> float:
    if reference == 0.0:
        return float("nan")
    return 100.0 * (computed - reference) / reference


# ---------------------------------------------------------------------------
# Bench 1 — Saturation pressure
# ---------------------------------------------------------------------------

def bench_saturation_pressure() -> bool:
    print("\n" + "=" * 70)
    print("BENCH 1 — Water saturation pressure vs. IAPWS-IF97")
    print("=" * 70)
    print(f"{'T (°C)':>8}  {'Computed (Pa)':>14}  {'IAPWS (Pa)':>12}  {'Error %':>8}  {'Result':>8}")
    print("-" * 70)
    all_pass = True
    for row in IAPWS_SPOT_VALUES:
        t_k = row["t_c"] + 273.15
        p_computed = water_sat_pressure_pa(t_k)
        p_ref = row["p_pa_ref"]
        err = _pct_err(p_computed, p_ref)
        ok = abs(err) < 2.0  # accept < 2 % for a first-principles screener
        all_pass = all_pass and ok
        print(f"{row['t_c']:>8.1f}  {p_computed:>14.2f}  {p_ref:>12.1f}  {err:>7.2f}%  {_pass_fail(ok):>8}")
    print()
    return all_pass


# ---------------------------------------------------------------------------
# Bench 2 — Latent heat
# ---------------------------------------------------------------------------

def bench_latent_heat() -> bool:
    print("=" * 70)
    print("BENCH 2 — Water latent heat vs. IAPWS-IF97 spot values")
    print("=" * 70)
    # IAPWS-IF97 h_fg values [J/kg]
    iapws_hfg = [
        (0.0,   2_500_900.0),
        (10.0,  2_477_200.0),
        (35.0,  2_418_400.0),
        (60.0,  2_358_500.0),
        (80.0,  2_308_000.0),
        (100.0, 2_256_400.0),
        (120.0, 2_202_600.0),
    ]
    print(f"{'T (°C)':>8}  {'Computed (J/kg)':>16}  {'IAPWS (J/kg)':>14}  {'Error %':>8}  {'Result':>8}")
    print("-" * 70)
    all_pass = True
    for t_c, h_ref in iapws_hfg:
        h_comp = water_h_fg_j_kg(t_c + 273.15)
        err = _pct_err(h_comp, h_ref)
        ok = abs(err) < 1.0
        all_pass = all_pass and ok
        print(f"{t_c:>8.1f}  {h_comp:>16.1f}  {h_ref:>14.1f}  {err:>7.2f}%  {_pass_fail(ok):>8}")
    print()
    return all_pass


# ---------------------------------------------------------------------------
# Bench 3 — COP / SCP cycle benchmarks
# ---------------------------------------------------------------------------

def bench_cycle() -> bool:
    print("=" * 70)
    print("BENCH 3 — COP / SCP vs. published literature")
    print("=" * 70)
    all_pass = True
    for bm in CYCLE_BENCHMARKS:
        profile = APPLICATIONS[bm["app"]]
        cycle = simulate_adsorption_cycle(
            q_sat=bm["q_sat"],
            q_st=bm["q_st"],
            t_evap_c=profile.t_evap_c,
            t_cond_c=profile.t_cond_c,
            t_des_c=profile.t_des_c,
            cycle_time_sec=profile.cycle_time_sec,
            e_char_j_mol=bm["e_char_j_mol"],
        )
        cop_ok  = bm["cop_published_min"] <= cycle["COP"] <= bm["cop_published_max"]
        scp_ok  = bm["scp_published_min"] <= cycle["SCP_W_kg"] <= bm["scp_published_max"]
        ok = cop_ok and scp_ok
        all_pass = all_pass and ok

        status = _pass_fail(ok)
        cop_range  = f"{bm['cop_published_min']:.2f}–{bm['cop_published_max']:.2f}"
        scp_range  = f"{bm['scp_published_min']:.0f}–{bm['scp_published_max']:.0f} W/kg"

        print(f"\n  {bm['label']}")
        print(f"    App:      {profile.name}  (T_evap={profile.t_evap_c}°C, T_cond={profile.t_cond_c}°C, T_des={profile.t_des_c}°C)")
        print(f"    Inputs:   q_sat={bm['q_sat']} kg/kg,  Q_st={bm['q_st']/1e6:.2f} MJ/kg")
        print(f"    Computed: COP={cycle['COP']:.3f}  (lit. {cop_range})  {_pass_fail(cop_ok)}")
        print(f"              SCP={cycle['SCP_W_kg']:.1f} W/kg  (lit. {scp_range})  {_pass_fail(scp_ok)}")
        print(f"              Δq={cycle['delta_q']:.3f} kg/kg,  h_fg={cycle['h_fg_MJ_kg']:.3f} MJ/kg")
        print(f"              P_evap={cycle['P_evap_kPa']:.2f} kPa,  P_cond={cycle['P_cond_kPa']:.2f} kPa")
        print(f"    Result:   {status}")
    print()
    return all_pass


# ---------------------------------------------------------------------------
# Bench 4 — MP API probe query
# ---------------------------------------------------------------------------

def bench_mp_query(api_key: Optional[str], all_systems: bool = False) -> None:
    if not HAS_MP:
        print("  [SKIP] mp_api not installed.")
        return
    if not api_key:
        print("  [SKIP] No MP_API_KEY found in .env or environment.")
        return

    fields = [
        "material_id", "formula_pretty", "density",
        "energy_above_hull", "band_gap", "elements",
    ]

    print("=" * 70)
    print("BENCH 4 — Materials Project API probe queries")
    print("=" * 70)

    with MPRester(api_key) as mpr:
        for app_key, app_name in [
            ("cpu", "CPU / electronics"),
            ("human", "Human HVAC"),
            ("vehicle", "Vehicle waste-heat"),
            ("datacenter", "Data-center"),
        ]:
            criteria = SEARCH_CRITERIA[app_key]
            profile  = APPLICATIONS[app_key]
            if all_systems:
                systems = generate_chemsys(criteria)
            else:
                systems = [MP_PROBE_SYSTEMS[app_key]]

            print(f"\n  [{app_key.upper()}] {app_name}")
            print(f"    T_evap={profile.t_evap_c}°C  T_cond={profile.t_cond_c}°C  T_des={profile.t_des_c}°C")
            print(f"    Filters: density={criteria.density}, Ehull={criteria.energy_above_hull}, "
                  f"band_gap>={criteria.band_gap_min} eV")
            print(f"    Chemical systems queried: {systems}")

            all_docs = []
            for system in systems:
                docs = mpr.materials.summary.search(
                    chemsys=system,
                    density=criteria.density,
                    energy_above_hull=criteria.energy_above_hull,
                    band_gap=(criteria.band_gap_min, 100.0),
                    exclude_elements=list(TOXIC_OR_COSTLY_ELEMENTS),
                    fields=fields,
                    num_chunks=1,
                    chunk_size=50,
                )
                all_docs.extend(docs)

            if not all_docs:
                print("    No results returned.")
                continue

            densities    = [d.density for d in all_docs if d.density]
            band_gaps    = [d.band_gap for d in all_docs if d.band_gap is not None]
            hulls        = [d.energy_above_hull for d in all_docs if d.energy_above_hull is not None]

            print(f"    Results: {len(all_docs)} materials")
            if densities:
                print(f"    Density:   {min(densities):.2f} – {max(densities):.2f} g/cm³  "
                      f"(mean {sum(densities)/len(densities):.2f})")
            if band_gaps:
                print(f"    Band gap:  {min(band_gaps):.2f} – {max(band_gaps):.2f} eV  "
                      f"(mean {sum(band_gaps)/len(band_gaps):.2f})")
            if hulls:
                print(f"    E_hull:    {min(hulls):.4f} – {max(hulls):.4f} eV/atom")

            print(f"    Top 5 by stability:")
            sorted_docs = sorted(all_docs, key=lambda d: (d.energy_above_hull or 1.0))
            for doc in sorted_docs[:5]:
                print(f"      {doc.material_id:<14} {doc.formula_pretty:<14} "
                      f"rho={doc.density:.2f}  Ehull={doc.energy_above_hull:.4f}  "
                      f"Eg={doc.band_gap:.2f} eV")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Thermodynamic cross-check and MP query validator.")
    parser.add_argument("--mp",  action="store_true", help="Run Materials Project API probe queries.")
    parser.add_argument("--all", action="store_true", help="Query all generated chemsys (slow); default: one probe system per app.")
    parser.add_argument("--api-key", default=get_mp_api_key(), help="Materials Project API key.")
    args = parser.parse_args()

    results = []
    results.append(("Saturation pressure", bench_saturation_pressure()))
    results.append(("Latent heat",         bench_latent_heat()))
    results.append(("COP / SCP cycles",    bench_cycle()))

    if args.mp:
        print("\n" + "=" * 70)
        print("MP API QUERY VALIDATION")
        print("=" * 70)
        bench_mp_query(args.api_key, all_systems=args.all)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_ok = True
    for name, ok in results:
        print(f"  {_pass_fail(ok)}  {name}")
        all_ok = all_ok and ok
    print()
    if all_ok:
        print("All physics benches PASSED.")
    else:
        print("One or more physics benches FAILED — review output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

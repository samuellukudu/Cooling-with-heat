"""T2 ranker — brute-force system-level material rankings (DESIGN §12 H2.3).

The adsorbent-ml Stage-2 surrogate's **top-k hit rate** is scored against
the rankings this module produces: every material in the fitted ISODB
table (H1.0) plus the curated anchors goes through ``Cycle0D-v0`` per
application profile; top candidates are then refined through the dynamic
``Bed1D-v0``. These are the *reference* rankings — the slow, honest
baseline the surrogate must hit at a fraction of the cost.

Honesty rules carried through every table:

- Equilibrium-only sweep (Cycle0D) uses no transport properties — clean
  with respect to the §8.1 transport caveat. The dynamic refinement
  (Bed1D) uses class-default ``k_ldf``/``k_eff`` for fitted rows, so its
  SCP column carries ``transport_provenance="default"`` and is *not*
  absolute across materials; within-material *trends* (cycle time,
  regeneration temperature) and the COP ordering under a fixed transport
  assumption are the meaningful outputs.
- The sweep set is the honestly-flagged subset of ``da_params.csv``:
  fits flagged ``ok``, physically-plausible ``q_sat``, and a multi-T
  isosteric heat. Everything else is excluded up front, not silently.
- Single-isotherm fits are aggregated per adsorbent (median parameters);
  the isotherm fit window is reported so out-of-window operation is
  visible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .envs.base import Objective, objective_value
from .envs.bed1d import Bed1D
from .envs.cycle0d import CANONICAL_NORMALIZATION, Cycle0D
from .materials import MaterialParams, get_material
from .profiles import ApplicationProfile, get_profile

# Default source: the H1.0 D–A fit export (fit_da.py → §8.1 schema),
# resolved relative to the repository root (mirrors materials.ANCHORS_PATH).
DEFAULT_FITS_PATH = Path(__file__).resolve().parents[1] / "data_cache" / "fits" / "da_params.csv"


def profile_objective(profile: "str | ApplicationProfile") -> Objective:
    """The profile-weighted normalized score over COP and SCP (the legacy
    screen's weights, renormalized over the two consumed metrics, with the
    canonical min-max ranges)."""
    prof = get_profile(profile)
    w = prof.cop_weight + prof.scp_weight
    return Objective(
        weights={"COP": prof.cop_weight / w, "SCP_W_kg": prof.scp_weight / w},
        normalize=dict(CANONICAL_NORMALIZATION),
    )


def load_sweep_materials(path: "str | Path | None" = None) -> list[MaterialParams]:
    """The honestly-flagged sweep set from the fit_da export: one
    ``MaterialParams`` per adsorbent (median D–A parameters across its
    usable isotherm fits, the shared multi-T ``q_st``).

    Rows are dropped up front when the fit is flagged anything but ``ok``,
    ``q_sat`` is not physical, or no multi-temperature ``q_st`` exists —
    the excluded counts are printed so the sweep's denominator is always
    visible.
    """
    path = Path(path) if path is not None else DEFAULT_FITS_PATH
    df = pd.read_csv(path)
    loadable = df[df["q_sat_kg_kg"].notna() & df["e_char_j_mol"].notna() & df["n_da"].notna()]
    usable = loadable[
        loadable["fit_flag"].eq("ok")
        & loadable["q_sat_physical"].eq(True)
        & loadable["q_st_j_kg"].notna()
    ]
    n_dropped_flag = len(loadable) - (loadable["fit_flag"].eq("ok")).sum()
    n_dropped_unphysical = int((~loadable["q_sat_physical"].eq(True)).sum())
    print(
        f"[harness.rank] {path}: {len(df)} fits → {len(loadable)} loadable → "
        f"{len(usable)} usable rows, {usable['name'].nunique()} adsorbents "
        f"(dropped: {n_dropped_flag} flagged, {n_dropped_unphysical} unphysical q_sat)"
    )

    materials: list[MaterialParams] = []
    for name, group in usable.groupby("name"):
        t_lo = min(float(t.split("-")[0]) for t in group["t_range_c"])
        t_hi = max(float(t.split("-")[-1]) for t in group["t_range_c"])
        materials.append(
            MaterialParams(
                material_id=str(group["material_id"].iloc[0]),
                name=str(name),
                source=str(group["source"].iloc[0]),
                q_sat_kg_kg=float(group["q_sat_kg_kg"].median()),
                q_st_j_kg=float(group["q_st_j_kg"].iloc[0]),
                e_char_j_mol=float(group["e_char_j_mol"].median()),
                n_da=float(group["n_da"].median()),
                t_range_c=(t_lo, t_hi),
                fit_rmse=float(group["fit_rmse"].median()),
                n_points=int(group["n_points"].median()),
                notes=f"{len(group)} usable isotherm fit(s)",
            )
        )
    materials.sort(key=lambda m: m.name)
    return materials


def sweep_materials(materials: "Iterable[str | MaterialParams]",
                    profiles: "Iterable[str | ApplicationProfile]" = (
                        "cpu", "human", "vehicle", "datacenter"),
                    ) -> pd.DataFrame:
    """Every material through ``Cycle0D-v0`` on every profile; returns the
    long-format ranked table (one row per material × profile).

    Columns: ``profile``, ``material``, ``source``, ``COP``, ``SCP_W_kg``,
    ``delta_q``, ``q_ads``, ``q_des``, ``score`` (profile-weighted
    normalized), ``rank`` (1 = best within the profile), ``t_window_c``
    (isotherm fit window), ``out_of_window`` (regeneration or condensation
    setpoint outside it).
    """
    rows: list[dict] = []
    for profile in profiles:
        prof = get_profile(profile)
        objective = profile_objective(prof)
        entries = list(materials)
        for entry in entries:
            mat = get_material(entry)
            problem = Cycle0D(mat, prof)
            metrics = problem.evaluate()
            score = float(objective_value(objective, metrics))
            t_lo, t_hi = mat.t_range_c
            rows.append({
                "profile": prof.name,
                "material": mat.name,
                "source": mat.source,
                "q_sat_kg_kg": mat.q_sat_kg_kg,
                "Q_st_MJ_kg": mat.q_st_j_kg / 1e6,
                "e_char_j_mol": mat.e_char_j_mol,
                "n_da": mat.n_da,
                "COP": metrics["COP"],
                "SCP_W_kg": metrics["SCP_W_kg"],
                "delta_q": metrics["delta_q"],
                "q_ads": metrics["q_ads"],
                "q_des": metrics["q_des"],
                "score": score,
                "t_window_c": f"{t_lo:g}-{t_hi:g}",
                "out_of_window": bool(prof.t_des_c > t_hi or prof.t_cond_c < t_lo),
            })
        prof_rows = [r for r in rows if r["profile"] == prof.name]
        for rank, r in enumerate(sorted(prof_rows, key=lambda r: -r["score"]), start=1):
            r["rank"] = rank
    return pd.DataFrame(rows)


def refine_with_bed1d(ranked: pd.DataFrame, profile: "str | ApplicationProfile",
                      k: int = 5, *, n_cells: int = 16,
                      dt_phys_s: float | None = 0.015) -> pd.DataFrame:
    """Refine the top-``k`` of one profile's equilibrium ranking through the
    dynamic ``Bed1D-v0`` (4-cycle episodes, default bed geometry).

    Fitted rows carry no transport data — the class defaults are applied
    via ``MaterialParams.with_transport_defaults`` and the resulting rows
    are flagged ``transport_provenance="default"``: within the fixed
    transport assumption the *ordering* is meaningful, absolute SCP across
    materials is not (§8.1 honesty note).
    """
    prof = get_profile(profile)
    top = ranked[ranked["profile"] == prof.name].nsmallest(k, "rank")
    rows: list[dict] = []
    for _, r in top.iterrows():
        # Rebuild the material from the sweep row itself (aggregated
        # parameters) — no registry round-trip, works for anchors and
        # fitted rows alike.
        mat = MaterialParams(
            name=str(r["material"]), source=str(r["source"]),
            q_sat_kg_kg=float(r["q_sat_kg_kg"]),
            q_st_j_kg=float(r["Q_st_MJ_kg"]) * 1e6,
            e_char_j_mol=float(r["e_char_j_mol"]),
            n_da=float(r["n_da"]),
        )
        mat = mat.with_transport_defaults(rho_kg_m3=600.0, cp_j_kg_k=1000.0,
                                          k_eff_w_m_k=0.3)
        env = Bed1D(mat, prof, n_cells=n_cells, dt_phys_s=dt_phys_s)
        metrics = env.evaluate()
        rows.append({
            "profile": prof.name,
            "material": mat.name,
            "rank_cycle0d": int(r["rank"]),
            "COP": metrics["COP"],
            "SCP_W_kg": metrics["SCP_W_kg"],
            "delta_q": metrics["delta_q"],
            "COP_cycle0d": r["COP"],
            "SCP_cycle0d": r["SCP_W_kg"],
            "transport_provenance": mat.transport_provenance,
        })
    return pd.DataFrame(rows)


def shortlist(ranked: pd.DataFrame, profile: "str | ApplicationProfile",
              k: int = 10) -> pd.DataFrame:
    """The top-``k`` shortlist for one profile (the T2 reference ranking)."""
    prof = get_profile(profile)
    return ranked[ranked["profile"] == prof.name].nsmallest(k, "rank")


__all__ = [
    "DEFAULT_FITS_PATH",
    "profile_objective",
    "load_sweep_materials",
    "sweep_materials",
    "refine_with_bed1d",
    "shortlist",
]

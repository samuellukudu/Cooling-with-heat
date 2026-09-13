"""System-level eval: predicted properties → COP ranking (the business metric).

ROADMAP evaluation protocol, level 2: predictions ride through the trusted
cycle model per application profile, and the **top-k hit rate vs the
true-parameter ranking** is the score GNNs must beat alongside the GBDT.
Property-level MAE lives in the baseline report; this module answers "does
a better q_sat actually buy a better shortlist?".

Profiles are read from the harness registry at runtime (no hardcoded
setpoints to drift). Rows missing any of (q_sat, Q_st, E, n) are skipped —
Q_st-free rows cannot serve the cycle physics (same honesty rule as the
corpus loader).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

EVAL_PROFILES = ("datacenter", "human")


def cycle_cop(q_sat: float, q_st: float, e_char: float, n_da: float,
              profile_name: str, *, hx_mass_factor: float = 1.35) -> float:
    """Equilibrium COP via the harness oracle (JAX, CPU-fast)."""
    from harness.physics.cycle0d import simulate_cycle  # noqa: PLC0415
    from harness.profiles import get_profile  # noqa: PLC0415

    prof = get_profile(profile_name)
    out = simulate_cycle(q_sat, q_st, prof.t_evap_c, prof.t_cond_c,
                         prof.t_des_c, prof.cycle_time_s,
                         e_char_j_mol=e_char, n_heterogeneity=n_da,
                         hx_mass_factor=hx_mass_factor)
    return float(out["COP"])


def rank_materials(params: pd.DataFrame, profile_name: str) -> pd.Series:
    """COP per row of a ``{q_sat_kg_kg, q_st_j_kg, e_char_j_mol, n_da}`` table."""
    return params.apply(lambda r: cycle_cop(r["q_sat_kg_kg"], r["q_st_j_kg"],
                                            r["e_char_j_mol"], r["n_da"],
                                            profile_name), axis=1)


def topk_hit_rate(true_rank: pd.Series, pred_rank: pd.Series, k: int) -> float:
    """Fraction of the true top-k also in the predicted top-k."""
    k = min(k, len(true_rank))
    return float(len(set(true_rank.nsmallest(k).index)
                     & set(pred_rank.nsmallest(k).index)) / k)


def system_rank_eval(true_params: pd.DataFrame, pred_params: pd.DataFrame, *,
                     profiles: tuple[str, ...] = EVAL_PROFILES,
                     k_values: tuple[int, ...] = (5, 10)) -> dict:
    """Rank agreement per profile. Both frames indexed alike, natural scale."""
    report: dict = {}
    for prof in profiles:
        cop_true = rank_materials(true_params, prof)
        cop_pred = rank_materials(pred_params, prof)
        rho = float(spearmanr(cop_true, cop_pred).statistic)
        rank_true, rank_pred = cop_true.rank(ascending=False), cop_pred.rank(ascending=False)
        entry: dict = {"spearman_cop": rho if np.isfinite(rho) else 0.0,
                       "n": int(len(true_params))}
        for k in k_values:
            if k <= len(true_params):
                entry[f"top{k}_hit_rate"] = topk_hit_rate(rank_true, rank_pred, k)
        report[prof] = entry
    return report


__all__ = ["EVAL_PROFILES", "cycle_cop", "rank_materials", "system_rank_eval",
           "topk_hit_rate"]

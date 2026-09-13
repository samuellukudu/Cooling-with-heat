"""Tests for the GBDT floor + system rank eval."""

import numpy as np
import pandas as pd

from rank_eval import cycle_cop, system_rank_eval
from tabular_baseline import (BaselineConfig, cross_validate, fit_full,
                              load_bundle, predict, save_bundle)


def _synthetic(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(6)})
    labels = pd.DataFrame({
        "log_q_sat": X["f0"] * 0.8 + rng.normal(0, 0.05, n),
        "log_Q_st": X["f1"] * 0.5 + rng.normal(0, 0.05, n),
        "q_st_j_kg": np.exp(X["f1"] * 0.5),
        "log_E": X["f2"] * 0.6 + rng.normal(0, 0.05, n),
        "n_da": 1.8 + 0.1 * X["f3"],
        "family": np.repeat(["a", "b", "c", "d"], n // 4)})
    return X, labels


def test_grouped_cv_recovers_signal_and_roundtrips(tmp_path):
    X, labels = _synthetic()
    feats = [c for c in X.columns]
    cfg = BaselineConfig(max_iter=100, n_splits=2, random_state=0)
    oof, report = cross_validate(X, labels, feats, cfg)
    assert report["log_q_sat"]["spearman"] > 0.9
    assert report["n_da"]["spearman"] > 0.7
    bundle = fit_full(X, labels, feats, cfg)
    save_bundle(bundle, tmp_path / "b.pkl")
    pred = load_bundle(tmp_path / "b.pkl")
    out = predict(pred, X)
    assert set(out.columns) == {"q_sat", "Q_st", "E", "n_da"}
    assert (out["q_sat"] > 0).all() and (out["Q_st"] > 0).all()


def test_rank_eval_perfect_and_regeneration_physics():
    # 13X-like (strong binding) vs silica-like: datacenter's 60 °C regen
    # starves 13X; human's 80 °C recovers it — the H2.3 qualitative result.
    silica = {"q_sat_kg_kg": 0.35, "q_st_j_kg": 2.5e6,
              "e_char_j_mol": 4500.0, "n_da": 1.8}
    x13 = {"q_sat_kg_kg": 0.28, "q_st_j_kg": 3.5e6,
           "e_char_j_mol": 14000.0, "n_da": 1.8}
    def cop_of(p, profile):
        return cycle_cop(p["q_sat_kg_kg"], p["q_st_j_kg"], p["e_char_j_mol"],
                         p["n_da"], profile)

    dc_sil, dc_x = cop_of(silica, "datacenter"), cop_of(x13, "datacenter")
    hu_sil, hu_x = cop_of(silica, "human"), cop_of(x13, "human")
    assert np.isfinite([dc_sil, dc_x, hu_sil, hu_x]).all()
    assert hu_x / hu_sil > dc_x / dc_sil  # regen-temperature-driven ranking

    true = pd.DataFrame([silica, x13,
                         {"q_sat_kg_kg": 0.30, "q_st_j_kg": 2.8e6,
                          "e_char_j_mol": 8000.0, "n_da": 1.8}])
    rep = system_rank_eval(true, true.copy(), k_values=(2,))
    assert rep["datacenter"]["spearman_cop"] == 1.0
    assert rep["datacenter"]["top2_hit_rate"] == 1.0

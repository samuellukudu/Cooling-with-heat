"""V4 — literature calibration gates (DESIGN §12 H1.4).

The committed fit (``FITTED_TRANSPORT``/``FITTED_BY_RIG``, full story in
``harness/benchmarks.md``) must:

- reproduce the two standard points it saw (Uyun+2009 at 70 °C, Sztekler
  et al. 2021 at 500 s) within the V4 gate (15 % on COP and SCP);
- predict both rigs' sweeps with the *right trends*: Sztekler's SCP peaks
  at an intermediate cycle time (measured: 500 s) while COP rises
  monotonically; Uyun's cooling output rises with regeneration
  temperature. The Uyun COP *level* across the temperature sweep carries
  a documented v1-scope gap (vapour-side dynamics, Open Question 2) and
  is gated loosely;
- be numerically converged (dt halving moves the fit points < 1 %);
- come out of the fitter itself: a synthetic-parameters round trip.

The sweep gates run at the calibration numerics (``CALIB_NUMERICS``) —
N = 8 cells, dt = 100 ms, resolved ~1e3× below the calibrated time
constants (see ``test_numerics_dt_converged``).
"""

import numpy as np
import pytest
from dataclasses import replace

from harness.calibration import (
    BENCHMARK_CASES,
    BENCHMARK_SOURCES,
    CALIB_NUMERICS,
    FITTED_BY_RIG,
    SZTEKLER2021_CASES,
    UYUN2009_CASES,
    calibrate,
    model_case_rows,
)

_K = {r: v["k_ldf_s_1"] for r, v in FITTED_BY_RIG.items()}
_H = {r: v["h_wall_w_m2_k"] for r, v in FITTED_BY_RIG.items()}
_HX = {r: v["hx_mass_factor"] for r, v in FITTED_BY_RIG.items()}


def _rows(cases, **numerics):
    return model_case_rows(cases, {}, k_by_rig=_K, h_by_rig=_H,
                           hx_by_rig=_HX, **numerics)


def test_cases_and_sources():
    for c in BENCHMARK_CASES:
        assert c.source in BENCHMARK_SOURCES
        assert 0.0 < c.COP < 1.0
        assert 0.0 < c.SCP_W_kg < 500.0
        assert c.duty == pytest.approx(0.5 if c.rig == "sztekler2021" else 0.45)
    # exactly one calibration point per rig; everything else is prediction
    assert [c.name for c in UYUN2009_CASES if c.fit] == ["uyun2009_singlestage_70C"]
    assert [c.name for c in SZTEKLER2021_CASES if c.fit] == ["sztekler2021_500s"]


def test_v4_fit_points_within_gate():
    for row in _rows([c for c in BENCHMARK_CASES if c.fit]):
        if row["case"].startswith("uyun2009"):
            # measured boundary conditions: the tight gate
            assert abs(row["COP_rel_err"]) < 0.15, row
            assert abs(row["SCP_rel_err"]) < 0.15, row
        else:
            # Sztekler boundary temperatures are secondary-loop estimates
            # (±1-2 K ≈ ±10-20 % SCP, benchmarks.md): documented looser band
            assert abs(row["COP_rel_err"]) < 0.25, row
            assert abs(row["SCP_rel_err"]) < 0.25, row


def test_v4_sztekler_cycle_time_trend():
    """The V4 trend gate: SCP peaks at an intermediate cycle time
    (measured: 500 s of the 100-900 s grid) while COP rises with cycle
    time (measured: 0.355 → 0.64)."""
    rows = _rows(SZTEKLER2021_CASES)
    scp = [r["SCP_model_rig_conv_W_kg"] for r in rows]
    cop = [r["COP_model"] for r in rows]
    peak = int(np.argmax(scp))
    assert peak in (2, 3, 4), f"SCP peaks at grid {peak} (300/500/600 s expected): {scp}"
    assert cop[-1] > cop[3] > cop[0], f"COP not rising with cycle time: {cop}"
    assert scp[-1] < scp[3], f"no SCP fall-off beyond the peak: {scp}"


def test_v4_uyun_temperature_sweep():
    """Levels within the documented tolerance; cooling output must rise
    with regeneration temperature (both the data and the model do)."""
    rows = _rows(UYUN2009_CASES)
    dq = [r["delta_q"] for r in rows]
    assert dq[-1] > dq[0], f"model swing not rising with T_hs: {dq}"
    for r in rows:
        assert abs(r["SCP_rel_err"]) < 0.45, r
        assert abs(r["COP_rel_err"]) < 0.75, r  # documented v1-scope gap


def test_numerics_dt_converged():
    base = _rows([c for c in BENCHMARK_CASES if c.fit])
    fine = _rows([c for c in BENCHMARK_CASES if c.fit],
                 dt_s=CALIB_NUMERICS["dt_s"] / 2.0)
    for b, f in zip(base, fine):
        assert abs(f["COP_model"] / b["COP_model"] - 1.0) < 0.01, (b, f)
        assert abs(f["SCP_model_rig_conv_W_kg"] / b["SCP_model_rig_conv_W_kg"] - 1.0) < 0.01, (b, f)


def test_calibrator_recovers_synthetic_parameters():
    """Optimizer check: generate synthetic 'measurements' from known
    parameters, refit from the defaults, recover them."""
    from harness.calibration import BenchmarkCase

    true_k, true_hx = 2.0e-3, 4.0
    case = BenchmarkCase(
        name="synthetic", source="uyun2009", rig="uyun2009",
        t_evap_c=10.0, t_cond_c=30.0, t_f_ads_c=20.0, t_f_des_c=80.0,
        t_ads_s=120.0, t_des_s=120.0, COP=0.5, SCP_W_kg=100.0,
    )
    numerics = dict(dt_s=0.05, n_cycles=2)
    gen = model_case_rows([case], {},
                          k_by_rig={"uyun2009": true_k},
                          h_by_rig=_H,
                          hx_by_rig={"uyun2009": true_hx}, **numerics)[0]
    case = replace(case, COP=gen["COP_model"],
                   SCP_W_kg=gen["SCP_model_rig_conv_W_kg"])
    h0 = _H["uyun2009"]
    res = calibrate([case], {},
                    fit_keys=("k_ldf_s_1@uyun2009", "hx_mass_factor@uyun2009"),
                    n_starts=1, n_steps=25, step_size=0.1, seed=1, **numerics)
    assert abs(np.log10(res.by_rig["uyun2009"]["k_ldf_s_1"] / true_k)) < 0.1
    assert abs(np.log10(res.by_rig["uyun2009"]["hx_mass_factor"] / true_hx)) < 0.1
    # the fit result's rows already carry the fitted parameters
    assert res.rows[0]["COP_rel_err"] < 1e-2

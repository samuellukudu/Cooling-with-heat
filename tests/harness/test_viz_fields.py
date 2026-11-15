"""Field capture + hydrogym-style rendering (viz layer).

Physics contract: ``collect_fields=True`` is capture-only — summaries are
bit-identical to the plain path and snapshots thin to ≤ 512 frames.
Rendering contract: frames are ``(H, W, 3)`` uint8; gym ``render()``
needs a prior ``reset()``.
"""

import numpy as np
import pytest

import harness
from harness import viz
from harness.envs.bed1d import Bed1DGymEnv
from harness.envs.two_bed import TwoBedGymEnv
from harness.physics import bed1d
from harness.physics import system as system_phys

BED_KW = dict(
    q_sat_kg_kg=0.3, q_st_j_kg=2.5e6, e_char_j_mol=5000.0, n_da=2.0,
    k_ldf_s_1=0.05, rho_s_kg_m3=600.0, c_s_j_kg_k=900.0, c_pl_j_kg_k=4180.0,
    k_eff_w_m_k=0.3, h_wall_w_m2_k=500.0, L_m=0.002, n_cells=8,
    hx_mass_factor=1.35, t_evap_c=18.0, t_cond_c=35.0, t_f_ads_c=35.0,
    t_f_des_c=75.0, t_ads_s=120.0, t_des_s=120.0, dt_s=0.05, n_cycles=2,
)


def test_bed1d_fields_bit_identical_and_thinned():
    a = bed1d.simulate_bed(**BED_KW)
    b = bed1d.simulate_bed(**BED_KW, collect_trace=True, collect_fields=True)
    assert a["fields"] is None if "fields" in a else True
    for k in a["summary"]:
        assert float(a["summary"][k]) == float(b["summary"][k]), k
    f = b["fields"]
    assert f["T_K"].shape == f["q_kg_kg"].shape
    assert f["T_K"].shape[0] <= 512 and f["T_K"].shape[1] == 8
    t = np.asarray(f["t_s"])
    assert bool((t[1:] > t[:-1]).all())
    x = np.asarray(f["x_m"])
    assert x[0] > 0 and x[-1] < BED_KW["L_m"]


def test_twobed_fields_bit_identical():
    kw = {k: v for k, v in BED_KW.items()
          if k not in ("t_f_ads_c", "t_ads_s", "t_des_s", "n_cycles")}
    a = system_phys.simulate_two_bed(
        bed_a=dict(kw), bed_b=dict(kw), t_evap_c=18.0, t_cond_c=35.0,
        t_f_ads_c=35.0, t_f_des_c=75.0, t_ads_s=120.0, t_des_s=120.0,
        dt_s=0.05, n_cycles=2)
    b = system_phys.simulate_two_bed(
        bed_a=dict(kw), bed_b=dict(kw), t_evap_c=18.0, t_cond_c=35.0,
        t_f_ads_c=35.0, t_f_des_c=75.0, t_ads_s=120.0, t_des_s=120.0,
        dt_s=0.05, n_cycles=2, collect_fields=True)
    for k in a["summary"]:
        assert float(a["summary"][k]) == float(b["summary"][k]), k
    f = b["fields"]
    assert f["A_T_K"].shape == f["B_T_K"].shape
    assert f["A_T_K"].shape[0] <= 512


def test_rollout_carries_fields():
    bed = harness.make("Bed1D-v0", material="anchor:Silica gel RD",
                       profile="cpu")
    tr = bed.rollout(collect_fields=True)
    assert tr.fields is not None and tr.fields["T_K"].ndim == 2
    tr_plain = bed.rollout()
    assert tr_plain.fields is None
    assert tr.summary["COP"] == pytest.approx(tr_plain.summary["COP"])


def test_viz_frames_are_rgb():
    rng = np.random.default_rng(0)
    f = viz.bed_frame(np.linspace(300, 350, 8), np.linspace(0.2, 0.05, 8),
                      np.linspace(0.0001, 0.0019, 8), t_s=1.0)
    assert f.dtype == np.uint8 and f.ndim == 3 and f.shape[2] == 3
    assert len(np.unique(f.reshape(-1, 3), axis=0)) > 100  # not blank
    h = viz.field_heatmap(rng.normal(320, 10, (60, 8)),
                          np.linspace(0, 300, 60),
                          np.linspace(0.0001, 0.0019, 8))
    assert h.dtype == np.uint8 and h.ndim == 3
    frames = viz.episode_frames(
        {"T_K": np.full((9, 8), 320.0), "q_kg_kg": np.full((9, 8), 0.1),
         "t_s": np.arange(9.0), "x_m": np.linspace(0.0001, 0.0019, 8)},
        stride=2)
    assert len(frames) == 5 and all(f.shape == frames[0].shape for f in frames)


def test_gym_render_rgb_array():
    env = Bed1DGymEnv(material="anchor:Silica gel RD", profile="cpu",
                      n_cells=8)
    with pytest.raises(RuntimeError, match="reset"):
        env.render()
    env.reset()
    env.step(np.array([75.0, 300.0], dtype=np.float32))
    frame = env.render()
    assert frame.dtype == np.uint8 and frame.ndim == 3
    tb = TwoBedGymEnv(material="anchor:Silica gel RD", profile="cpu",
                      n_cells=8, n_cycles=1)
    tb.reset()
    tb.step(np.array([300.0, 75.0, 0.0], dtype=np.float32))
    assert tb.render().ndim == 3

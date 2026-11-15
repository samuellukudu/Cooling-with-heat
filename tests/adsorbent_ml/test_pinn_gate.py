"""Gate-runner smoke: the N1 acceptance pass runs end-to-end on a tiny model.

The smoke model (width-16/depth-2, ~30 steps) will NOT pass the numeric
gates — the test asserts the report is complete, finite and self-consistent,
not that an untrained net passes. The real gate verdict comes from
``pinn_gate.py`` on the trained checkpoint (data_cache/pinn/gate_report.json).
"""

import json

import jax
import numpy as np
import pytest

from corpus import build_corpus, preset_configs
from pinn_gate import GATES, run_gate
from train_pinn import TrainConfig, make_batches, save_params, train

jax.config.update("jax_enable_x64", False)


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory):
    data, _ = build_corpus(preset_configs(), n_data_per_config=100,
                           seed=0, n_cells=8, dt_s=None, n_steps=100)
    cfg = TrainConfig(width=16, depth=2, steps=20, ramp_steps=10,
                      n_colloc=32, n_bc=16, n_ic=16, seed=0)
    batches = make_batches(data, np.asarray(data["conds"]), cfg, seed=1)
    params, _ = train(batches, cfg, log_every=0)
    out = tmp_path_factory.mktemp("pinn") / "tiny.npz"
    save_params(out, params)
    return out


def test_gate_report_complete_and_finite(tiny_ckpt):
    report = run_gate(tiny_ckpt, n_real_rows=2, n_steps=50)
    assert set(report) == {"ckpt", "created_unix", "scope", "field_interp",
                           "superres", "residual_honesty", "grad_agreement",
                           "cooling_proxy", "gates", "overall_pass"}
    assert len(report["field_interp"]) == len(preset_configs())  # 1 gate config per sweep preset
    for err in report["field_interp"].values():
        assert np.isfinite(err["rel_l2_T"]) and err["rel_l2_T"] > 0.0
        assert np.isfinite(err["rel_l2_q"])
    assert set(report["gates"]) == set(GATES) | {"cooling_sign"}
    for g in report["gates"].values():
        assert isinstance(g["pass"], bool)
    assert isinstance(report["overall_pass"], bool)
    # residual honesty on real rows ran and produced finite label-free numbers
    assert report["residual_honesty"]["n_conditions"] > 0
    assert np.isfinite(report["residual_honesty"]["mean_abs_r_energy"])
    assert np.isfinite(report["residual_honesty"]["mean_abs_r_ldf"])
    # the report serializes (it is written as gate_report.json by the CLI)
    json.dumps(report)


def test_gate_ckpt_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_gate(tmp_path / "nope.npz", n_real_rows=1, n_steps=20)


def test_direction_presets_pick_real_phases():
    """Pin the round-1 cooling-proxy bug: [1] was a silica DES preset."""
    from pinn_gate import _direction_presets

    des, ads = _direction_presets()
    assert des["t_f_c"] > des["t_init_c"] and ads["t_f_c"] < ads["t_init_c"]
    # the adsorption proxy must be the zeolite (cools), not a silica des preset
    assert ads["name"].startswith("13X")
    assert des["name"].startswith("silica")
    assert des["q_init_kg_kg"] is None and ads["q_init_kg_kg"] is None

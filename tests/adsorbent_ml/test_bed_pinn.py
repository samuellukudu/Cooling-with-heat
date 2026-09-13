"""Acceptance tests for the Bed1D PINN (N1) + corpus.

Fast by design (CPU, tiny nets, small rollouts): training smoke uses
width-16/depth-2 and 30 steps; the corpus rollout is 8 cells × 20 steps.
Mirrors the harness honesty rules — V5-style gradient agreement, V2-style
boundedness, resolution-invariance as a test rather than a claim.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from bed_pinn import (COND_DIM, Q_MAX_KG_KG, T_HALF_K, T_MID_K,
                      bc_residuals, boundary_loss, decode_condition,
                      encode_condition, ic_residuals, init_params,
                      initial_loss, physics_loss, pinn_residuals,
                      predict_Tq, predict_batch, total_loss)
from corpus import (build_corpus, family_split, preset_configs,
                    rollout_kwargs, rollout_phase_fields)
from pinn_metrics import field_errors, grad_agreement
from train_pinn import TrainConfig, make_batches, train

jax.config.update("jax_enable_x64", False)


@pytest.fixture
def psat_fn():
    from harness.physics.thermo import water_sat_pressure_pa

    return water_sat_pressure_pa


@pytest.fixture
def cond_vec():
    cond = {"q_sat_kg_kg": 0.35, "q_st_j_kg": 2.5e6, "e_char_j_mol": 4500.0,
            "n_da": 1.8, "k_ldf_s_1": 2.5e-4, "rho_s_kg_m3": 800.0,
            "c_s_j_kg_k": 1000.0, "k_eff_w_m_k": 0.2, "h_wall_w_m2_k": 200.0,
            "L_m": 2e-4, "t_f_c": 80.0, "p_pa": 7380.0, "t_init_c": 30.0,
            "q_init_frac": 0.5, "t_phase_s": 60.0}
    return jnp.asarray(encode_condition(cond))


def test_cond_encode_decode_roundtrip():
    vals = {"q_sat_kg_kg": 0.35, "q_st_j_kg": 2.5e6, "e_char_j_mol": 4500.0,
            "n_da": 1.8, "k_ldf_s_1": 2.5e-4, "rho_s_kg_m3": 800.0,
            "c_s_j_kg_k": 1000.0, "k_eff_w_m_k": 0.2, "h_wall_w_m2_k": 200.0,
            "L_m": 2e-4, "t_f_c": 80.0, "p_pa": 7380.0, "t_init_c": 30.0,
            "q_init_frac": 0.5, "t_phase_s": 60.0}
    back = decode_condition(encode_condition(vals))
    assert len(back) >= COND_DIM
    for k, v in vals.items():
        assert float(back[k]) == pytest.approx(v, rel=1e-5)


def test_apply_shapes_bounds_resolution_invariance(cond_vec):
    params = init_params(jax.random.PRNGKey(0), width=16, depth=2)
    for n in (5, 37):  # same weights, arbitrary query counts
        x = jnp.linspace(0, 1, n)
        t = jnp.linspace(0, 1, n)
        T, q = predict_batch(params, x, t, cond_vec)
        assert T.shape == (n,) and q.shape == (n,)
        assert bool(jnp.all((T >= T_MID_K - T_HALF_K) & (T <= T_MID_K + T_HALF_K)))
        assert bool(jnp.all((q >= 0.0) & (q <= Q_MAX_KG_KG)))
    T0, _ = predict_Tq(params, jnp.asarray(0.3), jnp.asarray(0.7), cond_vec)
    assert T0.shape == ()


def test_residuals_finite_and_grads_flow(cond_vec, psat_fn):
    params = init_params(jax.random.PRNGKey(1), width=16, depth=2)
    x = jnp.asarray(0.4)
    t = jnp.asarray(0.6)
    res = pinn_residuals(params, x, t, cond_vec, psat_fn)
    for k in ("r_energy", "r_ldf"):
        assert jnp.isfinite(res[k]), k
    bc = bc_residuals(params, t, cond_vec)
    ic = ic_residuals(params, x, cond_vec)
    for d in (bc, ic):
        for k, v in d.items():
            if k.startswith("r_"):
                assert jnp.isfinite(v), k
    batch = {"cond": cond_vec,
             "x_data": jnp.linspace(0, 1, 8), "t_data": jnp.linspace(0, 1, 8),
             "uT_data": jnp.zeros(8), "uq_data": jnp.full(8, 0.1),
             "x_colloc": jnp.linspace(0, 1, 8), "t_colloc": jnp.linspace(0, 1, 8),
             "t_bc": jnp.linspace(0, 1, 4), "x_ic": jnp.linspace(0, 1, 4)}
    w = {"data": 1.0, "pde": 1.0, "bc": 1.0, "ic": 1.0}
    (loss, aux), grads = jax.value_and_grad(total_loss, has_aux=True)(
        params, batch, w, psat_fn)
    assert jnp.isfinite(loss)
    leaves = jax.tree_util.tree_leaves(grads)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
    assert set(aux) == {"data", "pde", "pde_energy", "pde_ldf", "bc", "ic", "total"}


def test_corpus_rollout_shapes_and_physical_bounds():
    cfg = dict(preset_configs()[0])
    fields = rollout_phase_fields(n_cells=8, dt_s=None, n_steps=200,
                                  **rollout_kwargs(cfg))
    assert fields["T"].shape == (201, 8)
    assert fields["q"].shape == (201, 8)
    assert np.all(np.isfinite(fields["T"])) and np.all(np.isfinite(fields["q"]))
    # Bounded by the driving temperatures (no spurious heating at init scale)
    assert fields["T"].min() >= 290.0 and fields["T"].max() <= 365.0
    assert fields["q"].min() >= 0.0
    assert fields["q"].max() <= cfg["q_sat_kg_kg"] * 1.01


def test_training_smoke_loss_decreases():
    data, manifest = build_corpus(preset_configs()[:1], n_data_per_config=200,
                                  seed=0, n_cells=8, dt_s=None, n_steps=200)
    assert manifest["n_configs"] == 1 and manifest["n_points"] == 200
    cfg = TrainConfig(width=16, depth=2, steps=30, n_colloc=64, n_bc=32,
                      n_ic=32, ramp_steps=10, seed=0)
    batches = make_batches(data, np.asarray(data["conds"]), cfg, seed=1)
    _, history = train(batches, cfg, log_every=0)
    assert history[-1]["total"] < history[0]["total"]
    assert np.isfinite(history[-1]["total"])


def test_directional_feedback_grad_vs_fd(cond_vec):
    params = init_params(jax.random.PRNGKey(2), width=16, depth=2)

    def mean_final_T(cond):
        x = jnp.linspace(0, 1, 9)
        t = jnp.ones_like(x)
        T, _ = predict_batch(params, x, t, cond)
        return jnp.mean(T)

    rep = grad_agreement(mean_final_T, cond_vec, wrt=8, eps=1e-2)
    assert np.isfinite(rep["grad_ad"]) and abs(rep["grad_ad"]) > 0.0
    assert rep["rel_err"] < 5e-2


def test_family_split_leak_free():
    fams = np.array(["a", "a", "b", "c", "c", "c"])
    tr, va = family_split(fams, val_fraction=0.34, seed=0)
    assert set(fams[tr]).isdisjoint(set(fams[va]))
    assert tr.sum() + va.sum() == len(fams)


def test_field_errors_runs(cond_vec):
    params = init_params(jax.random.PRNGKey(3), width=16, depth=2)
    x = np.linspace(0, 1, 10).astype(np.float32)
    t = np.linspace(0, 1, 10).astype(np.float32)
    T_pred, q_pred = predict_batch(params, jnp.asarray(x), jnp.asarray(t), cond_vec)
    err = field_errors(lambda a, b, c: predict_batch(params, jnp.asarray(a),
                                                     jnp.asarray(b), c),
                       x, t, cond_vec,
                       np.asarray(T_pred) + 1.0, np.asarray(q_pred))
    assert err["rel_l2_T"] > 0.0 and np.isfinite(err["mae_T_K"])

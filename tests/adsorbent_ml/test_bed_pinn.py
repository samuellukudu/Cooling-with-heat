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

from bed_pinn import (COND_DIM, Q_MAX_KG_KG, S_LOG_MAX, T_HALF_K, T_MID_K,
                      bc_residuals, boundary_loss, decode_condition,
                      encode_condition, ic_residuals, init_params,
                      initial_loss, physics_loss, pinn_residuals,
                      predict_Tq, predict_batch, total_loss)
from corpus import (OP_ENVELOPE, build_corpus, family_split, preset_configs,
                    rollout_kwargs, rollout_phase_fields)
from pinn_metrics import field_errors, grad_agreement
from train_pinn import TrainConfig, make_batches, train

jax.config.update("jax_enable_x64", False)


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
        # structured log-space head: q = q_eq · exp(s) ≥ 0 by construction;
        # the trained 0..q_sat bound comes from data/IC, not the raw net
        assert bool(jnp.all(q >= 0.0))
    T0, _ = predict_Tq(params, jnp.asarray(0.3), jnp.asarray(0.7), cond_vec)
    assert T0.shape == ()


def test_psat_matches_harness():
    """The uptake head's internal psat must equal the harness's (two-branch)."""
    from harness.physics.thermo import water_sat_pressure_pa

    from bed_pinn import _water_psat_pa

    for t_k in (270.0, 300.0, 340.0, 372.0, 373.15, 380.0, 420.0, 450.0):
        mine = float(_water_psat_pa(jnp.asarray(t_k)))
        ref = float(water_sat_pressure_pa(jnp.asarray(t_k)))
        assert mine == pytest.approx(ref, rel=1e-5), t_k


def test_structured_uptake_head_decomposition(cond_vec):
    """q decomposes as q_eq(T_net, p) + f — the exponential is never learned."""
    from bed_pinn import q_equilibrium

    params = init_params(jax.random.PRNGKey(3), width=16, depth=2)
    x = jnp.linspace(0.05, 0.95, 7)
    t = jnp.linspace(0.0, 0.9, 7)
    T, q = predict_batch(params, x, t, cond_vec)
    phys = decode_condition(cond_vec)
    q_star = q_equilibrium(T, phys["p_pa"], phys["q_sat_kg_kg"],
                           phys["e_char_j_mol"], phys["n_da"])
    # log-space correction: q = q* · exp(s), s ∈ [−S, S], and q ≥ 0 always
    ratio = q / q_star
    assert bool(jnp.all((ratio >= jnp.exp(-S_LOG_MAX) - 1e-4)
                        & (ratio <= jnp.exp(S_LOG_MAX) + 1e-4)))
    assert bool(jnp.all(q >= 0.0))
    # q responds to pressure through the ANALYTIC term: raising p at the same
    # predicted T must raise the equilibrium part (Polanyi potential shrinks)
    p_hi = phys["p_pa"] * 3.0
    from bed_pinn import COND_KEYS

    phys_raw = decode_condition(cond_vec)
    vals = {k: float(phys_raw[k]) for k in COND_KEYS}
    vals["p_pa"] = float(p_hi)
    T2, q2 = predict_batch(params, x, t, encode_condition(vals))
    q_star_hi = q_equilibrium(T2, vals["p_pa"], phys["q_sat_kg_kg"],
                              phys["e_char_j_mol"], phys["n_da"])
    # equilibrium stroke direction: higher p -> higher q* at the same T
    assert float(jnp.mean(q_star_hi)) > float(jnp.mean(q_star))


def test_residuals_finite_and_grads_flow(cond_vec):
    params = init_params(jax.random.PRNGKey(1), width=16, depth=2)
    x = jnp.asarray(0.4)
    t = jnp.asarray(0.6)
    res = pinn_residuals(params, x, t, cond_vec)
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
        params, batch, w)
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


def test_preset_sweep_covers_operating_envelope():
    """Round-2 op-point diversity: the corpus hull defines the N1 gate's test.

    Round 1 trained one operating point per family and field interpolation
    failed the gate (rel-L2 T 0.102 vs 0.05). The sweep must now cover the
    shared OP_ENVELOPE with both phase directions, explicit partial charges,
    and the round-1 anchors preserved at indices 0/1 (tests + checkpoints).
    """
    presets = preset_configs()
    assert len(presets) == 96  # 2 families × 24 op points (6 anchors + 18 LHS) × 2
    # round-1 anchors preserved (ordering + names pinned by earlier work)
    assert presets[0]["name"] == "silica-RD-des-L0.002"
    assert presets[1]["name"] == "silica-RD-des-L0.005"
    assert "13X-ads-L0.002" in {p["name"] for p in presets}
    by_fam = {"silica": [p for p in presets if p["family"] == "silica"],
              "zeolite": [p for p in presets if p["family"] == "zeolite"]}
    for fam, plist in by_fam.items():
        assert len(plist) == 48
        for p in plist:
            for key in ("t_f_c", "t_init_c", "p_pa"):
                lo, hi = OP_ENVELOPE[key]
                assert lo <= p[key] <= hi, (p["name"], key)
        des = [p for p in plist if p["t_f_c"] > p["t_init_c"]]
        ads = [p for p in plist if p["t_f_c"] < p["t_init_c"]]
        # each family sweeps its own direction with margin (no near-iso points)
        if fam == "silica":
            assert len(des) == 48 and not ads
            assert all(p["t_f_c"] >= p["t_init_c"] + 5.0 for p in des)
        else:
            assert len(ads) == 48 and not des
            assert all(p["t_init_c"] >= p["t_f_c"] + 5.0 for p in ads)
        # operating-point diversity on the axes the gate interpolates
        assert len({p["t_init_c"] for p in plist}) >= 5
        assert len({round(p["p_pa"]) for p in plist}) >= 5
        assert any(p["q_init_kg_kg"] is not None for p in plist)
    # every sweep point rolls out finitely (cheap: 60 steps)
    for p in presets[::4]:
        f = rollout_phase_fields(n_cells=8, dt_s=None, n_steps=60,
                                 **rollout_kwargs(p))
        assert np.all(np.isfinite(f["T"])) and np.all(np.isfinite(f["q"]))


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


# ---------------------------------------------------------------------------
# Stage C: real-condition hardening (physics-only batches from real rows)
# ---------------------------------------------------------------------------

def test_stage_c_batches_real_params_no_labels():
    from corpus import load_real_conditions
    from train_pinn import stage_c_batches

    real_conds, report = load_real_conditions(max_rows=3)
    assert report["n_conditions"] >= 1
    cfg = TrainConfig(steps=0, steps_c=0, samples_per_row=2,
                      n_colloc=32, n_bc=16, n_ic=16, seed=0)
    batches, sc = stage_c_batches(real_conds, cfg, seed=1)
    assert len(batches) == 2 * report["n_conditions"] == sc["n_batches"]
    for b in batches:
        assert b["cond"].shape == (COND_DIM,)
        # physics-only: no field labels anywhere in the batch
        assert "x_data" not in b and "uT_data" not in b
        for k in ("x_colloc", "t_colloc", "t_bc", "x_ic"):
            assert jnp.isfinite(b[k]).all(), k
        phys = decode_condition(b["cond"])
        # the sampled operating point must sit inside the declared envelope
        # (defaults = corpus.OP_ENVELOPE, shared with the synthetic sweep)
        assert cfg.op_t_f_c[0] <= float(phys["t_f_c"]) <= cfg.op_t_f_c[1]
        assert cfg.op_p_pa[0] <= float(phys["p_pa"]) <= cfg.op_p_pa[1]
        assert cfg.op_t_init_c[0] <= float(phys["t_init_c"]) <= cfg.op_t_init_c[1]
        assert 0.05 <= float(phys["q_init_frac"]) <= 1.0
    assert np.isfinite(sc["n_clipped_values"])


def test_stage_c_training_smoke():
    from corpus import load_real_conditions, preset_configs
    from train_pinn import stage_c_batches

    data, _ = build_corpus(preset_configs()[:1], n_data_per_config=100,
                           seed=0, n_cells=8, dt_s=None, n_steps=100)
    real_conds, _ = load_real_conditions(max_rows=2)
    cfg = TrainConfig(width=16, depth=2, steps=10, steps_c=5, ramp_steps=5,
                      n_colloc=32, n_bc=16, n_ic=16, samples_per_row=1, seed=0)
    batches = make_batches(data, np.asarray(data["conds"]), cfg, seed=1)
    real_batches, _ = stage_c_batches(real_conds, cfg, seed=2)
    params, history = train(batches, cfg, log_every=0, real_batches=real_batches)
    stages = [h.get("stage") for h in history]
    assert stages[:cfg.steps] == ["AB"] * cfg.steps
    assert stages[cfg.steps:] == ["C"] * cfg.steps_c
    for h in history[cfg.steps:]:
        assert np.isfinite(h["total"]) and np.isfinite(h["real_pde"])
    leaves = jax.tree_util.tree_leaves(params)
    assert all(bool(jnp.all(jnp.isfinite(p))) for p in leaves)

"""H1.1 gates (DESIGN §12): V2 conservation + analytic limits of the
dynamic 1-D bed physics (``harness.physics.bed1d``).

- Semi-infinite slab: pure-conduction limit (k_LDF = 0) against the
  analytic convective-BC solution (erfc form).
- LDF uptake: the exact-exponential substep reproduces the analytic
  exponential for any k·Δt — including stiff values where explicit Euler
  diverges.
- Per-step energy identity: ``Σ cap·ΔT·Δx = Φ_wall + ρ_s·Q_st·Σ Δq·Δx``
  to machine precision (the RK4 wall-flux quadrature is the same one the
  T-update uses), and the semi-discrete conduction operator telescopes
  exactly to the wall flux.
- RK4 temporal order 4 on the semi-discrete system.
- ``check_timestep`` mirrors ``diffheat.check_cfl``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import linalg as scipy_linalg
from scipy.special import erfc

from harness.physics import bed1d
from harness.physics.thermo import da_uptake

# One shared parameter set (silica-gel-like bed) for the operator tests.
PARAMS = dict(
    dx=None,  # filled per test (geometry varies)
    q_sat_kg_kg=0.35,
    q_st_j_kg=2.5e6,
    e_char_j_mol=4500.0,
    n_da=1.8,
    k_ldf_s_1=5.0,
    rho_s_kg_m3=600.0,
    c_s_j_kg_k=1000.0,
    c_pl_j_kg_k=4184.0,
    k_eff_w_m_k=0.3,
    h_wall_w_m2_k=500.0,
)


def _scan_steps(T0, q0, t_f_k, p_pa, dt_s, n_steps, params):
    """Run ``step_bed`` n times via scan; return the final fields."""

    def body(carry, _):
        T, q = carry
        T_new, q_new, _ = bed1d.step_bed(T, q, t_f_k, p_pa, dt_s, **params)
        return (T_new, q_new), None

    (T_f, q_f), _ = jax.lax.scan(body, (T0, q0), None, length=n_steps)
    return T_f, q_f


# --------------------------------------------------------------------------
# V2 gate 1: pure-conduction limit vs the analytic semi-infinite slab
# --------------------------------------------------------------------------


def test_semi_infinite_slab_conduction():
    # Dummy solid with cap = rho*cs = 1e6 -> alpha = 1e-6 m²/s; rho_s must
    # match the cap used in the analytic diffusivity (q = 0 throughout).
    params = dict(PARAMS, k_ldf_s_1=0.0, rho_s_kg_m3=1000.0)
    L_m, n_cells = 0.05, 400
    params["dx"] = L_m / n_cells
    params["k_eff_w_m_k"] = 1.0
    params["h_wall_w_m2_k"] = 50.0

    cap = params["rho_s_kg_m3"] * params["c_s_j_kg_k"]
    alpha = params["k_eff_w_m_k"] / cap
    t_i, t_f_k, h = 300.0, 350.0, params["h_wall_w_m2_k"]

    dt_s = 0.5 * bed1d.max_timestep(L_m, n_cells, params["k_eff_w_m_k"], cap)
    n_steps = int(round(30.0 / dt_s))
    assert bed1d.check_timestep(L_m, n_cells, params["k_eff_w_m_k"], cap, dt_s)

    T0 = jnp.full((n_cells,), t_i)
    T_final, _ = _scan_steps(T0, jnp.zeros((n_cells,)), t_f_k, 1.0e5, dt_s,
                             n_steps, params)

    t = n_steps * dt_s
    eta = (np.arange(n_cells) + 0.5) * params["dx"] / (2.0 * np.sqrt(alpha * t))
    root = h * np.sqrt(alpha * t) / params["k_eff_w_m_k"]
    analytic = t_i + (t_f_k - t_i) * (
        erfc(eta)
        - np.exp(h * (np.arange(n_cells) + 0.5) * params["dx"]
                 / params["k_eff_w_m_k"] + (root ** 2))
        * erfc(eta + root)
    )
    err = np.abs(np.asarray(T_final) - analytic)
    assert err.max() / (t_f_k - t_i) < 0.01, f"max profile error {err.max():.3f} K"
    assert np.sqrt(np.mean(err ** 2)) / (t_f_k - t_i) < 0.005
    # Far face must be untouched (semi-infinite validity of the comparison).
    assert abs(float(T_final[-1]) - t_i) < 0.01


# --------------------------------------------------------------------------
# V2 gate 2: LDF uptake matches its exponential solution at any k·dt
# --------------------------------------------------------------------------


def test_uptake_exact_exponential_stiff():
    # No conduction (k_eff = h = 0), no adsorption heat (Q_st = 0): the
    # temperature stays exactly constant, so q* is constant and the exact
    # exponential substep must compose to the analytic solution.
    params = dict(PARAMS, k_eff_w_m_k=0.0, h_wall_w_m2_k=0.0, q_st_j_kg=0.0)
    n_cells = 8
    params["dx"] = 2.0e-3 / n_cells
    T0 = jnp.full((n_cells,), 310.0)
    q_star = float(
        da_uptake(310.0, 1500.0, PARAMS["q_sat_kg_kg"],
                  PARAMS["e_char_j_mol"], PARAMS["n_da"])
    )

    k, dt_s = 2.0, 5.0  # k·dt = 10: explicit Euler would need k·dt < 2
    params = dict(params, k_ldf_s_1=k)
    q0 = jnp.full((n_cells,), 0.02)
    n_steps = 20
    _, q_final = _scan_steps(T0, q0, 310.0, 1500.0, dt_s, n_steps, params)
    analytic = q_star + (0.02 - q_star) * np.exp(-k * dt_s * n_steps)
    assert np.abs(np.asarray(q_final) - analytic).max() < 1e-12

    # Single mega-step (k·dt = 500): still exact, still finite.
    T1, q1, _ = bed1d.step_bed(T0, q0, 310.0, 1500.0, 5.0,
                               **dict(params, k_ldf_s_1=100.0))
    assert np.abs(np.asarray(q1) - q_star).max() < 1e-15
    assert np.allclose(np.asarray(T1), 310.0)


# --------------------------------------------------------------------------
# V2 gate 3: per-step energy identity + semi-discrete telescoping
# --------------------------------------------------------------------------


def test_step_energy_conservation():
    params = dict(PARAMS)
    n_cells, L_m = 24, 2.0e-3
    params["dx"] = L_m / n_cells
    x = (np.arange(n_cells) + 0.5) * params["dx"]
    T0 = jnp.asarray(320.0 + 15.0 * np.sin(np.pi * x / L_m))
    q0 = jnp.asarray(0.10 + 0.05 * np.cos(2.0 * np.pi * x / L_m))

    dt_s = 0.008  # below the conduction CFL (~0.0106 s here); k·dt = 0.04
    T_new, q_new, info = bed1d.step_bed(T0, q0, 340.0, 3000.0, dt_s, **params)

    cap_mid = PARAMS["rho_s_kg_m3"] * (
        PARAMS["c_s_j_kg_k"] + PARAMS["c_pl_j_kg_k"] * 0.5 * (q0 + q_new)
    )
    stored = float(jnp.sum(cap_mid * (T_new - T0)) * params["dx"])
    flux_plus_source = (
        float(info["wall_flux_integral"]) + float(info["adsorption_heat"])
    )
    scale = max(1.0, abs(flux_plus_source))
    assert abs(stored - flux_plus_source) <= 1e-10 * scale, (
        f"energy identity residual {abs(stored - flux_plus_source):.3e} "
        f"(scale {scale:.3e})"
    )


def test_rhs_energy_conservation_semidiscrete():
    params = dict(PARAMS)
    n_cells, L_m = 24, 2.0e-3
    params["dx"] = L_m / n_cells
    x = (np.arange(n_cells) + 0.5) * params["dx"]
    T0 = jnp.asarray(320.0 + 15.0 * np.sin(np.pi * x / L_m))
    q0 = jnp.asarray(0.10 + 0.05 * np.cos(2.0 * np.pi * x / L_m))

    d_t, q_dot, wall = bed1d.bed_rhs(T0, q0, 340.0, 3000.0, **params)
    # sum(div) = wall flux exactly; div enters dT as
    # dT = (div + rho*qdot*Qst*dx)/(cap*dx)  =>  sum(div) = sum(cap*dT*dx)
    #      - rho*Qst*sum(qdot)*dx
    cap = PARAMS["rho_s_kg_m3"] * (PARAMS["c_s_j_kg_k"] + PARAMS["c_pl_j_kg_k"] * q0)
    lhs = float(jnp.sum(cap * d_t) * params["dx"]
                - PARAMS["rho_s_kg_m3"] * PARAMS["q_st_j_kg"]
                * jnp.sum(q_dot) * params["dx"])
    scale = max(1.0, abs(wall))
    assert abs(lhs - float(wall)) <= 1e-10 * scale


# --------------------------------------------------------------------------
# RK4 temporal order + timestep helper
# --------------------------------------------------------------------------


def test_rk4_fourth_order():
    params = dict(PARAMS, k_ldf_s_1=0.0, h_wall_w_m2_k=0.0)  # pure diffusion
    n_cells, L_m = 32, 2.0e-3
    params["dx"] = L_m / n_cells
    x = (np.arange(n_cells) + 0.5) * params["dx"]
    T0 = 320.0 + 15.0 * np.sin(np.pi * x / L_m)
    alpha = PARAMS["k_eff_w_m_k"] / (PARAMS["rho_s_kg_m3"] * PARAMS["c_s_j_kg_k"])

    # Reference: the exact semi-discrete solution expm(t·A)·T0 (linear
    # system, constant coefficients) — self-convergence at these error
    # levels would drown in float noise.
    dx2 = params["dx"] ** 2
    a = alpha / dx2
    A = np.zeros((n_cells, n_cells))
    idx = np.arange(n_cells)
    A[idx, idx] = -2.0 * a
    A[idx[:-1], idx[:-1] + 1] = a
    A[idx[1:], idx[1:] - 1] = a
    A[0, 0] = -a   # adiabatic wall (h = 0): ghost = mirror
    A[-1, -1] = -a  # adiabatic far face
    ref = scipy_linalg.expm(1.0 * A) @ T0

    def run(dt_s):
        n = int(round(1.0 / dt_s))
        T_f, _ = _scan_steps(jnp.asarray(T0), jnp.zeros(n_cells), 320.0,
                             1.0e5, dt_s, n, params)
        return np.asarray(T_f)

    # dt = 1/256 sits at the conduction CFL (dx²/2α = 1/256 s here) — the
    # temporal error is then well above the float-noise floor that swamps
    # it at finer dt. Above the CFL the scheme diverges, which is exactly
    # what check_timestep guards.
    e1 = np.linalg.norm(run(1.0 / 256) - ref)
    e2 = np.linalg.norm(run(1.0 / 512) - ref)
    ratio = e1 / e2
    assert 10.0 < ratio < 22.0, f"RK4 order check: error ratio {ratio:.2f} (want ~16)"


def test_check_timestep():
    L_m, n_cells, k_eff = 2.0e-3, 32, 0.3
    rho_cp = 600.0 * (1000.0 + 4184.0 * 0.35)
    limit = bed1d.max_timestep(L_m, n_cells, k_eff, rho_cp)
    dx = L_m / n_cells
    assert limit == pytest.approx(dx ** 2 / (2.0 * k_eff / rho_cp))
    assert bed1d.check_timestep(L_m, n_cells, k_eff, rho_cp, limit)
    assert not bed1d.check_timestep(L_m, n_cells, k_eff, rho_cp, limit * 1.01)


# --------------------------------------------------------------------------
# hx metal-mass accounting (V4 rig mapping): pure bookkeeping — hx must not
# touch the trajectory, and hx = 1 must reproduce the bare-bed heat input.
# --------------------------------------------------------------------------


def test_hx_metal_mass_accounting():
    base = dict(
        q_sat_kg_kg=0.35, q_st_j_kg=2.5e6, e_char_j_mol=4500.0, n_da=1.8,
        k_ldf_s_1=5.0,
        rho_s_kg_m3=600.0, c_s_j_kg_k=1000.0, c_pl_j_kg_k=4184.0,
        k_eff_w_m_k=0.3, h_wall_w_m2_k=300.0, L_m=0.002, n_cells=16,
        t_evap_c=10.0, t_cond_c=30.0, t_f_ads_c=25.0, t_f_des_c=70.0,
        t_ads_s=300.0, t_des_s=300.0, dt_s=0.02, n_cycles=3,
    )

    def run(hx):
        out = bed1d.simulate_bed(**base, hx_mass_factor=hx)
        return {k: float(v) for k, v in out["summary"].items()}

    bare, metal = run(1.0), run(2.0)
    # The metal block is accounting-only: the cooling side is untouched.
    assert metal["Q_cool_J_kg"] == bare["Q_cool_J_kg"]
    assert metal["delta_q"] == bare["delta_q"]
    # It only adds heat input, proportional to (hx - 1).
    assert metal["Q_in_J_kg"] > bare["Q_in_J_kg"]
    mid = run(1.5)
    assert bare["Q_in_J_kg"] < mid["Q_in_J_kg"] < metal["Q_in_J_kg"]
    # hx = 1 reproduces the bare-bed (pre-V4) accounting exactly.
    expected_bare_cop = bare["Q_cool_J_kg"] / bare["Q_in_J_kg"]
    assert bare["COP"] == pytest.approx(expected_bare_cop)

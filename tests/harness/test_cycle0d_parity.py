"""V1 extension (DESIGN §9): the JAX oracle reproduces
``cooling_physics.simulate_adsorption_cycle`` exactly across the parameter
box, including the degenerate branches; jit/eager agreement; gradients flow.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

import cooling_physics
from harness.physics import cycle0d

PROFILES = {  # (t_evap_c, t_cond_c, t_des_c) per heat_cooling_screen.APPLICATIONS
    "cpu": (18.0, 35.0, 75.0),
    "human": (10.0, 35.0, 80.0),
    "vehicle": (7.0, 45.0, 120.0),
    "datacenter": (16.0, 35.0, 60.0),
}

Q_SATS = (0.08, 0.35, 0.90)
Q_STS = (2.3e6, 3.2e6, 4.1e6)
E_CHARS = (3500.0, 4500.0, 14000.0)
N_HETS = (1.3, 1.8, 3.5)
HX_FACTORS = (1.0, 1.35)
CYCLES = (120.0, 600.0)

# 3*3*4*3*3*2*2 = 1296 cases over the full box; the branch structures
# (p >= p_sat, delta_q = 0, q_in = 0, cycle_time = 0) are covered by the
# dedicated cases below.
CASES = [
    (q, qst, *temps, cyc, e, n, hx)
    for q in Q_SATS
    for qst in Q_STS
    for temps in PROFILES.values()
    for cyc in CYCLES
    for e in E_CHARS
    for n in N_HETS
    for hx in HX_FACTORS
]

_jitted = jax.jit(cycle0d.simulate_cycle)


def _canonical(**kwargs):
    return cooling_physics.simulate_adsorption_cycle(**kwargs)


@pytest.mark.parametrize("case", CASES)
def test_oracle_parity(case):
    q, qst, t_evap, t_cond, t_des, cyc, e, n, hx = case
    ref = _canonical(
        q_sat=q, q_st=qst, t_evap_c=t_evap, t_cond_c=t_cond, t_des_c=t_des,
        cycle_time_sec=cyc, e_char_j_mol=e, n_heterogeneity=n, hx_mass_factor=hx,
    )
    got = {
        k: float(v)
        for k, v in _jitted(
            jnp.float64(q), jnp.float64(qst), jnp.float64(t_evap), jnp.float64(t_cond),
            jnp.float64(t_des), jnp.float64(cyc), jnp.float64(e), jnp.float64(n), jnp.float64(hx),
        ).items()
    }
    for key, ref_value in ref.items():
        np.testing.assert_allclose(
            got[key], ref_value, rtol=1e-12, atol=1e-15,
            err_msg=f"metric {key!r} diverged for case {case}",
        )


@pytest.mark.parametrize("cycle_time", [0.0, -5.0])
def test_zero_cycle_time_branch(cycle_time):
    ref = _canonical(q_sat=0.35, q_st=2.5e6, t_evap_c=16.0, t_cond_c=35.0, t_des_c=60.0, cycle_time_sec=cycle_time)
    got = {k: float(v) for k, v in cycle0d.simulate_cycle(
        jnp.float64(0.35), jnp.float64(2.5e6), jnp.float64(16.0), jnp.float64(35.0),
        jnp.float64(60.0), jnp.float64(cycle_time)).items()}
    assert got["SCP_W_kg"] == ref["SCP_W_kg"] == 0.0
    assert got["COP"] == pytest.approx(ref["COP"], rel=1e-12)


def test_no_uptake_delta_branch():
    """t_des == t_ads (== t_cond): delta_q = 0 ⇒ COP = 0, matching canonical."""
    ref = _canonical(q_sat=0.35, q_st=2.5e6, t_evap_c=16.0, t_cond_c=35.0, t_des_c=35.0, cycle_time_sec=300.0)
    got = {k: float(v) for k, v in cycle0d.simulate_cycle(
        jnp.float64(0.35), jnp.float64(2.5e6), jnp.float64(16.0), jnp.float64(35.0),
        jnp.float64(35.0), jnp.float64(300.0)).items()}
    assert got["delta_q"] == ref["delta_q"] == 0.0
    assert got["COP"] == ref["COP"] == 0.0


def test_jit_equals_eager():
    args = tuple(jnp.float64(v) for v in (0.45, 2.8e6, 7.0, 35.0, 80.0, 600.0, 4500.0, 1.8, 1.35))
    eager = cycle0d.simulate_cycle(*args)
    jitted = _jitted(*args)
    for key in eager:
        np.testing.assert_allclose(float(jitted[key]), float(eager[key]), rtol=1e-15, atol=0)


def test_gradients_flow():
    q = jnp.float64(0.45)

    def cop_of(q_sat):
        return cycle0d.simulate_cycle(q_sat, jnp.float64(2.8e6), jnp.float64(16.0), jnp.float64(35.0), jnp.float64(60.0), jnp.float64(300.0))["COP"]

    grad = jax.grad(cop_of)(q)
    assert np.isfinite(float(grad))
    assert float(grad) != 0.0

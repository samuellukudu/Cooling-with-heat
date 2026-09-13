"""Telegrapher-cancellation trial on the harness (T-E08 hardening).

Gates (mirroring ``examples/08`` demo 2, whose hand-coded gradient descent
is exactly what the backends must rediscover):
- Parity: harness 1-D telegrapher mirrors diffheat
  ``solve_telegrapher_1d`` (< 1e-4; same dtype note as T-A5/T-A4).
- Wave CFL guard fails loudly.
- Symmetric A = −1 cancels > 20 % (the demo's success bar; measured 64 %).
- Search finds the interior optimum (A ≈ −0.95, off the naive −1 guess)
  and beats the −0.5 default; grad flows with the right sign there.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import pytest

import harness
from harness.envs.base import Objective, objective_value
from harness.registry import REGISTRIES


def test_telegrapher_registered():
    assert "Telegrapher-v0" in REGISTRIES["envs"].names()
    assert "telegrapher_cancel" in REGISTRIES["models"].names()


def test_telegrapher_protocol_and_metrics():
    prob = harness.make("Telegrapher-v0", profile="cpu")
    from harness.envs.base import validate_problem
    validate_problem(prob)
    m = prob.evaluate()
    assert set(m) >= {"sensor_mse", "sensor_peak", "cancellation_pct", "secondary_amplitude"}
    assert all(v == v and abs(v) != float("inf") for v in m.values()), m


def test_unknown_design_key_raises():
    prob = harness.make("Telegrapher-v0", profile="cpu")
    with pytest.raises(KeyError):
        prob.evaluate({"tau": 1.0})  # structural, not design


def test_cfl_guard_fails_loudly():
    from harness.physics.telegrapher import check_timestep_1d
    assert check_timestep_1d(1.0 / 120, 1.0, 0.5, 0.001) is True
    assert check_timestep_1d(1.0 / 120, 1.0, 0.5, 1.0) is False
    with pytest.raises(ValueError):
        harness.make("Telegrapher-v0", profile="cpu", dt=1.0)


def _diffheat_traj(n=48, dt=0.001, t_end=0.1, A=-0.7):
    from diffheat import BoundaryCondition, Grid1D, TelegrapherEquation1D
    from diffheat import solve_telegrapher_1d
    grid = Grid1D.uniform(length=1.0, n_cells=n)
    bc = BoundaryCondition(kind="dirichlet", value=jnp.array([0.0, 0.0]))
    eqn = TelegrapherEquation1D(grid=grid, bc=bc, alpha=1.0, tau=0.5)
    u0 = (jnp.exp(-((grid.centers - 0.3) ** 2) / 0.0005)
          + A * jnp.exp(-((grid.centers - 0.7) ** 2) / 0.0005))
    # float64 ICs: diffheat grids are float32 on GPU while harness runs
    # float64 (x64); without this, diffheat's own scan carry mixes dtypes
    # (frozen-package GPU fragility — harness side is float64-clean).
    u0 = jnp.asarray(u0, dtype=jnp.float64)
    v0 = jnp.zeros(n, dtype=jnp.float64)
    return solve_telegrapher_1d(eqn, u0, v0, (0.0, t_end), dt)


def test_parity_vs_diffheat():
    from harness.physics import telegrapher as tg
    n, dt, t_end, A = 48, 0.001, 0.1, -0.7
    ref = _diffheat_traj(n, dt, t_end, A)
    dx = 1.0 / n
    xc = (jnp.arange(n, dtype=jnp.float64) + 0.5) * dx
    u0 = tg.gaussian(xc, 0.3) + A * tg.gaussian(xc, 0.7)
    sim = tg.simulate_telegrapher_1d(u0=u0, n_cells=n, dt=dt,
                                     n_steps=int(round(t_end / dt)),
                                     collect_trace=True)
    traj = sim["trajectory"]
    assert traj.shape == ref.shape
    # Same dtype-discipline note as the T-A5/T-A4 parity tests.
    assert float(jnp.max(jnp.abs(traj - ref))) < 1e-4


def test_symmetric_cancellation_beats_demo_bar():
    prob = harness.make("Telegrapher-v0", profile="cpu")
    m = prob.evaluate({"secondary_amplitude": -1.0})
    assert m["cancellation_pct"] > 20.0  # the demo's success bar


def test_search_discovers_off_symmetric_optimum():
    prob = harness.make("Telegrapher-v0", profile="cpu")
    obj = Objective(weights={"sensor_mse": -1.0})
    naive = float(objective_value(obj, prob.evaluate()))
    res = harness.optimize(prob, obj, backend="search", method="tpe", seed=0, budget=30)
    assert res.best_objective > naive
    # Damping/boundaries shift the optimum off the naive −1 guess —
    # the harness finds it; the test only pins the neighbourhood.
    assert -1.3 < res.best_design["secondary_amplitude"] < -0.6
    assert prob.evaluate(res.best_design)["cancellation_pct"] > 40.0


def test_grad_flows_with_right_sign():
    prob = harness.make("Telegrapher-v0", profile="cpu")
    f = lambda a: prob.metrics_jax({"secondary_amplitude": a})["sensor_mse"]
    g_ad = float(jax.grad(f)(-0.5))
    assert g_ad == g_ad and abs(g_ad) > 0
    assert g_ad > 0  # at −0.5, increasing A (toward 0) worsens mse


def test_adapter_model_satisfies_protocol():
    from harness.physics.adapters import SimulatorAdapter
    inst = REGISTRIES["models"].resolve("telegrapher_cancel")()
    assert isinstance(inst, SimulatorAdapter)

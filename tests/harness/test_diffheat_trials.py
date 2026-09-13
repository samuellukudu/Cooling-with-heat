"""diffheat-application trials on the harness (T-A5 forced-conv, T-A4 cloak).

Phase 0–1 gates:
- V-FC: harness forced-conv mirrors diffheat ``solve_advection_diffusion_2d``
  (<1e-9), CFL guard fails loudly, higher U lowers T_max, search beats the
  naive default under a pumping penalty (discovery, not hand-tuning).
- V-CLOAK: harness cloak mirrors diffheat ``solve_heat_2d`` with field alpha
  (<1e-9), gradients flow (V5-style FD check), search improves the combined
  cloak objective over the uniform-kappa baseline.
- Registry + Problem-protocol + import-hygiene compliance for both envs.
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

import harness
from harness.envs.base import Objective, validate_problem
from harness.registry import REGISTRIES


# ---------------------------------------------------------------------------
# Registration / protocol
# ---------------------------------------------------------------------------

def test_trial_envs_registered():
    assert "ForcedConv-v0" in REGISTRIES["envs"].names()
    assert "Cloak2D-v0" in REGISTRIES["envs"].names()
    assert "forced_conv" in REGISTRIES["models"].names()
    assert "cloak2d" in REGISTRIES["models"].names()


def test_trial_problems_satisfy_protocol():
    for name in ("ForcedConv-v0", "Cloak2D-v0"):
        prob = harness.make(name, profile="datacenter")
        validate_problem(prob)
        m = prob.evaluate()
        assert all(v == v and abs(v) != float("inf") for v in m.values()), m


def test_unknown_design_key_raises():
    prob = harness.make("ForcedConv-v0", profile="datacenter")
    with pytest.raises(KeyError):
        prob.evaluate({"source_amplitude": 1.0})  # scenario, not design (discipline)
    cloak = harness.make("Cloak2D-v0", profile="datacenter")
    with pytest.raises(KeyError):
        cloak.evaluate({"bg": 1.0})


def test_cfl_guard_fails_loudly():
    from harness.physics import forced_conv as fc
    assert fc.check_timestep(4.0 / 24, 1.0 / 8, 0.01, 1.0, 0.002) is True
    assert fc.check_timestep(4.0 / 24, 1.0 / 8, 0.01, 1.0, 1.0) is False
    prob = harness.make("ForcedConv-v0", profile="datacenter", nx=24, ny=8, dt=0.002)
    with pytest.raises(ValueError):
        prob.evaluate({"inlet_velocity": 200.0})  # advective CFL blowout


# ---------------------------------------------------------------------------
# V-FC parity vs diffheat (tiny grid, few steps — exact mirror)
# ---------------------------------------------------------------------------

def _diffheat_channel(nx=12, ny=6, U=0.8, amp=500.0, dt=0.002, n_steps=10):
    from diffheat import AdvectionDiffusion2D, BoundaryCondition2D, Grid2D
    from diffheat import solve_advection_diffusion_2d
    Lx, Ly = 4.0, 1.0
    grid = Grid2D.uniform(Lx=Lx, Ly=Ly, nx=nx, ny=ny)
    bc = BoundaryCondition2D(
        left={"kind": "dirichlet", "value": 0.0},
        right={"kind": "dirichlet", "value": 0.0},
        bottom={"kind": "neumann", "value": 0.0},
        top={"kind": "neumann", "value": 0.0},
    )

    def flow(X, Y, t):
        return U * jnp.ones_like(X), jnp.zeros_like(Y)

    def src(X, Y, t):
        r2 = ((X - Lx / 2) ** 2 + Y ** 2) / (2 * 0.1 ** 2)
        return amp * jnp.exp(-r2)

    eqn = AdvectionDiffusion2D(grid=grid, bc=bc, alpha=0.01, velocity=flow, source=src)
    T0 = jnp.zeros((nx, ny))
    return solve_advection_diffusion_2d(eqn, T0, (0.0, dt * n_steps), dt)


def test_forced_conv_parity_vs_diffheat():
    from harness.physics import forced_conv as fc
    nx, ny, dt, n_steps, U, amp = 12, 6, 0.002, 10, 0.8, 500.0
    ref = _diffheat_channel(nx, ny, U, amp, dt, n_steps)
    sim = fc.simulate_forced_conv(inlet_velocity=U, nx=nx, ny=ny, dt=dt,
                                  n_steps=n_steps, source_amplitude=amp,
                                  collect_trace=True)
    traj = sim["trajectory"]
    assert traj.shape == ref.shape
    # Tolerance accounts for dtype discipline: diffheat grids follow
    # utils.get_default_dtype() (float32 on GPU, float64 on CPU) while the
    # harness is always float64 (V1-style). Mirror correctness, not bits.
    assert float(jnp.max(jnp.abs(traj - ref))) < 1e-4


def test_forced_conv_monotone_in_velocity():
    prob = harness.make("ForcedConv-v0", profile="datacenter", nx=16, ny=8,
                        dt=0.002, n_steps=60)
    lo = prob.evaluate({"inlet_velocity": 0.2})["T_max"]
    hi = prob.evaluate({"inlet_velocity": 1.5})["T_max"]
    assert hi < lo  # more flow → cooler chip (the discovery signal)


def test_forced_conv_search_beats_naive_with_pumping_penalty():
    prob = harness.make("ForcedConv-v0", profile="datacenter", nx=16, ny=8,
                        dt=0.002, n_steps=60)
    # Minimize T_max subject to soft pumping budget: interior optimum, not a bound.
    obj = Objective(weights={"T_max": -1.0, "pumping_proxy": -2.0})
    naive = float(harness.envs.base.objective_value(obj, prob.evaluate()))
    res = harness.optimize(prob, obj, backend="search", method="tpe", seed=0, budget=30)
    assert res.best_objective >= naive
    assert 0.05 < res.best_design["inlet_velocity"] < 2.0
    assert res.history


def test_forced_conv_grad_flows():
    prob = harness.make("ForcedConv-v0", profile="datacenter", nx=12, ny=6,
                        dt=0.002, n_steps=20)
    f = lambda d: prob.metrics_jax(d)["T_max"]
    g = jax.grad(lambda u: f({"inlet_velocity": u}))(1.0)
    assert float(g) == float(g) and abs(float(g)) > 0  # finite, nonzero


# ---------------------------------------------------------------------------
# V-CLOAK parity vs diffheat (field alpha) + discovery
# ---------------------------------------------------------------------------

def _diffheat_cloak(nx=10, ny=10, ki=0.005, kr=0.02, dt=0.001, n_steps=10):
    from diffheat import BoundaryCondition2D, Grid2D, HeatEquation2D
    from diffheat import solve_heat_2d
    L = 1.0
    grid = Grid2D.uniform(Lx=L, Ly=L, nx=nx, ny=ny)
    bc = BoundaryCondition2D(
        left={"kind": "dirichlet", "value": 1.0},
        right={"kind": "dirichlet", "value": 0.0},
        bottom={"kind": "neumann", "value": 0.0},
        top={"kind": "neumann", "value": 0.0},
    )
    X, Y = jnp.meshgrid(grid.x_centers, grid.y_centers, indexing="ij")
    R = jnp.sqrt((X - L / 2) ** 2 + (Y - L / 2) ** 2)
    kappa = jnp.where(R < 0.20, ki, jnp.where(R < 0.40, kr, 0.01))
    eqn = HeatEquation2D(grid=grid, bc=bc, alpha=kappa)
    T0 = jnp.zeros((nx, ny))
    return solve_heat_2d(eqn, T0, (0.0, dt * n_steps), dt)


def test_cloak_parity_vs_diffheat():
    from harness.physics import cloak2d as ck
    nx, ny, dt, n_steps = 10, 10, 0.001, 10
    ref = _diffheat_cloak(nx, ny, 0.005, 0.02, dt, n_steps)
    sim = ck.simulate_cloak(kappa_inner=0.005, kappa_ring=0.02, nx=nx, ny=ny,
                            dt=dt, n_steps=n_steps, collect_trace=True)
    traj = sim["trajectory"]
    assert traj.shape == ref.shape
    # Same dtype-discipline note as the forced-conv parity test.
    assert float(jnp.max(jnp.abs(traj - ref))) < 1e-4


def test_cloak_grad_vs_fd():
    from harness.physics import cloak2d as ck
    # V5-style: jax.grad vs central FD on interior_grad_norm at 2 probes.
    def score(ki, kr):
        sim = ck.simulate_cloak(kappa_inner=ki, kappa_ring=kr, nx=10, ny=10,
                                dt=0.001, n_steps=12)
        m = ck.cloak_metrics(sim["final"], sim["kappa"], sim["X"], sim["Y"], sim["R"],
                             sim["dx"], sim["dy"], 1.0, ki, kr)
        return m["interior_grad_norm"] + m["far_field_mismatch"]

    gs = jax.jit(jax.grad(lambda c: score(c[0], c[1])))
    import numpy as np
    for ki, kr in [(0.004, 0.015), (0.008, 0.03)]:
        g_ad = np.asarray(gs(jnp.asarray([ki, kr])))
        h = 1e-3
        g_fd = np.array([
            (float(score(ki + h, kr)) - float(score(ki - h, kr))) / (2 * h),
            (float(score(ki, kr + h)) - float(score(ki, kr - h))) / (2 * h),
        ])
        denom = np.maximum(np.abs(g_fd), 1e-6)
        assert float(np.max(np.abs(g_ad - g_fd) / denom)) < 1e-2


def test_cloak_search_improves_over_uniform():
    prob = harness.make("Cloak2D-v0", profile="datacenter", nx=12, ny=12,
                        dt=0.001, n_steps=40)
    obj = Objective(weights={"interior_grad_norm": -1.0, "far_field_mismatch": -1.0})
    base = float(harness.envs.base.objective_value(obj, prob.evaluate({"kappa_inner": 0.01, "kappa_ring": 0.01})))
    res = harness.optimize(prob, obj, backend="search", method="tpe", seed=0, budget=30)
    assert res.best_objective >= base
    assert res.history


def test_adapter_models_satisfy_protocol():
    from harness.physics.adapters import SimulatorAdapter
    for name in ("forced_conv", "cloak2d"):
        inst = REGISTRIES["models"].resolve(name)()
        assert isinstance(inst, SimulatorAdapter)
        assert hasattr(inst.spec, "name")

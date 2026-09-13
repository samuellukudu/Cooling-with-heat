"""Dimensional ladder: 1D/2D/3D + 4D time harness (diffheat parity).

Validates that the harness mesh/operators/solvers mirror the frozen
diffheat ladder and that every registered env carries explicit
dimensional metadata (``spatial_dim``/``time_resolved``/``grid_type``)
so experiments can be filtered by dimension.
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

import harness
from harness.envs.base import Objective, validate_problem
from harness.registry import REGISTRIES


# ---------------------------------------------------------------------------
# Spec dimension tagging
# ---------------------------------------------------------------------------

EXPECTED_DIMS = {
    # 0D lumped
    "Cycle0D-v0": (0, False, "none"),
    "AbsorptionCycle-v0": (0, False, "none"),
    "Thermoelectric-v0": (0, False, "none"),
    "NaturalConv-v0": (0, False, "none"),
    "MockCycle-v0": (0, False, "none"),
    # 1D
    "Heat1D-v0": (1, True, "uniform_rect"),
    "Bed1D-v0": (1, True, "uniform_rect"),
    "Telegrapher-v0": (1, True, "uniform_rect"),
    "Thermoelectric1D-v0": (1, True, "uniform_rect"),
    "TwoBed-v0": (1, True, "uniform_rect"),
    # 2D
    "Heat2D-v0": (2, True, "uniform_rect"),
    "ForcedConv-v0": (2, True, "uniform_rect"),
    "Cloak2D-v0": (2, True, "uniform_rect"),
    "Boussinesq-v0": (2, True, "mac"),
    # 3D
    "Heat3D-v0": (3, True, "uniform_rect"),
}


def test_spec_dim_metadata():
    for name, (sd, tr, gt) in EXPECTED_DIMS.items():
        prob = harness.make(name)
        assert prob.spec.spatial_dim == sd, name
        assert prob.spec.time_resolved == tr, name
        assert prob.spec.grid_type == gt, name
        validate_problem(prob)


def test_list_by_dim_filter():
    from harness.experiments import list_by_dim

    assert set(list_by_dim(0)) >= {"Cycle0D-v0", "AbsorptionCycle-v0"}
    assert set(list_by_dim(1)) >= {"Heat1D-v0", "Bed1D-v0", "Telegrapher-v0"}
    assert set(list_by_dim(2)) >= {"Heat2D-v0", "ForcedConv-v0", "Cloak2D-v0"}
    assert "Heat3D-v0" in list_by_dim(3)
    assert not set(list_by_dim(1)) & set(list_by_dim(3))
    # time-resolved ladder includes all PDE envs
    assert "Heat1D-v0" in list_by_dim(time_resolved=True)
    assert "Cycle0D-v0" not in list_by_dim(time_resolved=True)
    # grid type
    assert "Boussinesq-v0" in list_by_dim(2, grid_type="mac")
    assert "Boussinesq-v0" not in list_by_dim(2, grid_type="uniform_rect")


# ---------------------------------------------------------------------------
# Mesh parity harness vs diffheat (Grid1D/2D/3D)
# ---------------------------------------------------------------------------

def test_mesh_parity_vs_diffheat():
    from harness.mesh import Grid1D, Grid2D, Grid3D
    from diffheat import Grid1D as DG1, Grid2D as DG2, Grid3D as DG3

    g = Grid1D.uniform(1.0, 10)
    dg = DG1.uniform(1.0, 10)
    assert float(jnp.max(jnp.abs(g.centers - dg.centers))) < 1e-12
    assert float(jnp.max(jnp.abs(g.dx - dg.dx))) < 1e-12
    assert g.shape() == (10,)

    g2 = Grid2D.uniform(1.0, 2.0, 8, 6)
    dg2 = DG2.uniform(1.0, 2.0, 8, 6)
    # Harness uses (nx,ny) ij, diffheat uses (ny,nx) xy — shapes differ but
    # spacings match and X extents agree after transpose.
    assert float(jnp.max(jnp.abs(g2.dx - dg2.dx))) < 1e-12
    assert float(jnp.max(jnp.abs(g2.dy - dg2.dy))) < 1e-12
    assert g2.shape() == (8, 6)

    g3 = Grid3D.uniform(1.0, 1.0, 1.0, 6, 6, 6)
    dg3 = DG3.uniform(1.0, 1.0, 1.0, 6, 6, 6)
    assert float(jnp.max(jnp.abs(g3.dx - dg3.dx))) < 1e-12
    assert g3.shape() == (6, 6, 6)


# ---------------------------------------------------------------------------
# Operator parity (Laplacian 1D/2D/3D)
# ---------------------------------------------------------------------------

def test_laplacian_parity_vs_diffheat():
    from harness.mesh import Grid1D, Grid2D, Grid3D
    from harness.operators import laplacian_1d, laplacian_2d, laplacian_3d
    from diffheat import Grid1D as DG1, Grid2D as DG2, Grid3D as DG3
    from diffheat.operators import laplacian_1d as dl1, laplacian_2d as dl2, laplacian_3d as dl3

    g = Grid1D.uniform(1.0, 12)
    dg = DG1.uniform(1.0, 12)
    T = jnp.array([float(i) * 0.3 for i in range(12)])
    assert float(jnp.max(jnp.abs(laplacian_1d(T, g) - dl1(T, dg)))) < 1e-12

    g2 = Grid2D.uniform(1.0, 1.0, 8, 6)
    dg2 = DG2.uniform(1.0, 1.0, 8, 6)
    T2 = jnp.ones((8, 6))
    assert float(jnp.max(jnp.abs(laplacian_2d(T2, g2) - dl2(T2, dg2)))) < 1e-12

    g3 = Grid3D.uniform(1.0, 1.0, 1.0, 4, 4, 4)
    dg3 = DG3.uniform(1.0, 1.0, 1.0, 4, 4, 4)
    T3 = jnp.ones((4, 4, 4))
    assert float(jnp.max(jnp.abs(laplacian_3d(T3, g3) - dl3(T3, dg3)))) < 1e-12


# ---------------------------------------------------------------------------
# Heat solve parity 1D/2D/3D + 4D time contract
# ---------------------------------------------------------------------------

def _harness_heat1d_traj(n_cells=10, alpha=0.01, dt=0.001, n_steps=12):
    from harness.mesh import Grid1D
    from harness.mesh.boundary import BoundaryCondition
    from harness.physics.heat1d import HeatEquation1D, simulate_heat_1d

    grid = Grid1D.uniform(1.0, n_cells)
    bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))
    eqn = HeatEquation1D(grid=grid, bc=bc, alpha=alpha)
    T0 = jnp.zeros(n_cells)
    return simulate_heat_1d(eqn, T0, (0.0, dt * n_steps), dt)


def _diffheat_heat1d_traj(n_cells=10, alpha=0.01, dt=0.001, n_steps=12):
    from diffheat import BoundaryCondition, Grid1D, HeatEquation1D, solve_heat_1d

    grid = Grid1D.uniform(1.0, n_cells)
    bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))
    eqn = HeatEquation1D(grid=grid, bc=bc, alpha=alpha)
    T0 = jnp.zeros(n_cells)
    return solve_heat_1d(eqn, T0, (0.0, dt * n_steps), dt)


def test_heat1d_parity_and_time_shape():
    h = _harness_heat1d_traj()
    d = _diffheat_heat1d_traj()
    assert h.shape == d.shape == (13, 10)  # n_steps+1, n_cells
    assert float(jnp.max(jnp.abs(h - d))) < 1e-12
    # 4D contract: time-first, frame 0 = initial
    assert float(jnp.max(jnp.abs(h[0]))) < 1e-12


def _harness_heat2d_traj(n=10, alpha=0.01, dt=0.001, n_steps=12):
    from harness.mesh import Grid2D
    from harness.mesh.boundary import BoundaryCondition2D
    from harness.physics.heat2d import HeatEquation2D, simulate_heat_2d

    grid = Grid2D.uniform(1.0, 1.0, n, n)
    bc = BoundaryCondition2D(
        left={"kind": "dirichlet", "value": 1.0},
        right={"kind": "dirichlet", "value": 0.0},
        bottom={"kind": "neumann", "value": 0.0},
        top={"kind": "neumann", "value": 0.0},
    )
    eqn = HeatEquation2D(grid=grid, bc=bc, alpha=alpha)
    T0 = jnp.zeros((n, n))
    return simulate_heat_2d(eqn, T0, (0.0, dt * n_steps), dt)


def _diffheat_heat2d_traj(n=10, alpha=0.01, dt=0.001, n_steps=12):
    from diffheat import BoundaryCondition2D, Grid2D, HeatEquation2D, solve_heat_2d

    grid = Grid2D.uniform(1.0, 1.0, n, n)
    bc = BoundaryCondition2D(
        left={"kind": "dirichlet", "value": 1.0},
        right={"kind": "dirichlet", "value": 0.0},
        bottom={"kind": "neumann", "value": 0.0},
        top={"kind": "neumann", "value": 0.0},
    )
    eqn = HeatEquation2D(grid=grid, bc=bc, alpha=alpha)
    T0 = jnp.zeros((n, n))
    return solve_heat_2d(eqn, T0, (0.0, dt * n_steps), dt)


def test_heat2d_parity_and_time_shape():
    h = _harness_heat2d_traj()
    d = _diffheat_heat2d_traj()
    assert h.shape == d.shape == (13, 10, 10)
    assert float(jnp.max(jnp.abs(h - d))) < 1e-12


def test_heat3d_parity_and_save_every():
    from harness.mesh import Grid3D
    from harness.mesh.boundary import BoundaryCondition3D
    from harness.physics.heat3d import HeatEquation3D, simulate_heat_3d
    from diffheat import BoundaryCondition3D as DBC3, Grid3D as DG3, HeatEquation3D as DHE3, solve_heat_3d

    n, dt, n_steps = 6, 0.002, 20
    save_every = 5
    grid_h = Grid3D.uniform(1.0, 1.0, 1.0, n, n, n)
    grid_d = DG3.uniform(1.0, 1.0, 1.0, n, n, n)
    cold = {"kind": "dirichlet", "value": 0.0}
    bc_h = BoundaryCondition3D(left=cold, right=cold, bottom=cold, top=cold, front=cold, back=cold)
    bc_d = DBC3(left=cold, right=cold, bottom=cold, top=cold, front=cold, back=cold)
    alpha = 0.01
    eqn_h = HeatEquation3D(grid=grid_h, bc=bc_h, alpha=alpha)
    eqn_d = DHE3(grid=grid_d, bc=bc_d, alpha=alpha)
    X, Y, Z = grid_h.X, grid_h.Y, grid_h.Z
    T0_h = jnp.where((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2 <= 0.2**2, 1.0, 0.0)
    Xd, Yd, Zd = grid_d.X, grid_d.Y, grid_d.Z
    T0_d = jnp.where((Xd - 0.5) ** 2 + (Yd - 0.5) ** 2 + (Zd - 0.5) ** 2 <= 0.2**2, 1.0, 0.0)

    h = simulate_heat_3d(eqn_h, T0_h, (0.0, dt * n_steps), dt, save_every=save_every)
    d = solve_heat_3d(eqn_d, T0_d, (0.0, dt * n_steps), dt, save_every=save_every)
    assert h.shape == d.shape == (n_steps // save_every + 1, n, n, n)
    assert float(jnp.max(jnp.abs(h - d))) < 1e-12

    # Time-first, frame 0 = T0
    assert float(jnp.max(jnp.abs(h[0] - T0_h))) < 1e-12
    # save_every=1 reproduces every step
    h_all = simulate_heat_3d(eqn_h, T0_h, (0.0, dt * n_steps), dt, save_every=1)
    assert h_all.shape[0] == n_steps + 1


# ---------------------------------------------------------------------------
# Gradient + experiment harness smoke
# ---------------------------------------------------------------------------

def test_heat_envs_grad_flow_and_optimize():
    for name in ("Heat1D-v0", "Heat2D-v0", "Heat3D-v0"):
        prob = harness.make(name)
        # V5-style: grad vs FD on T_mean
        from harness.experiments import grad_vs_fd

        g = grad_vs_fd(name, eps=1e-4)
        assert g["rel"] < 1e-3, f"{name} grad rel {g['rel']}"

        # Backend smoke: search improves over naive with same objective
        obj = Objective(weights={"T_mean": 1.0})
        naive = float(harness.envs.base.objective_value(obj, prob.evaluate({"alpha": 0.01})))
        res = harness.optimize(prob, obj, backend="search", method="tpe", seed=0, budget=20)
        assert res.best_objective >= naive
        assert "alpha" in res.best_design


def test_dim_battery_heat_ladder():
    from harness.experiments import dim_battery

    rows = dim_battery(
        env_kwargs={
            "Heat1D-v0": {"n_cells": 10, "t_end": 0.5, "dt": 0.002},
            "Heat2D-v0": {"nx": 6, "ny": 6, "t_end": 0.5, "dt": 0.002},
            "Heat3D-v0": {"nx": 4, "ny": 4, "nz": 4, "t_end": 0.4, "dt": 0.004, "save_every": 2},
        },
        design={"alpha": 0.012},
    )
    assert len(rows) == 3
    for r in rows:
        assert r["spatial_dim"] in (1, 2, 3)
        assert r["time_resolved"] is True
        assert "T_mean" in r["metrics"]
        assert r["trajectory_shape"][0] > 1  # time dimension present (4D)

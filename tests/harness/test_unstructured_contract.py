"""Unstructured-mesh contract (T-A6 capability gap, docs/applications.md §6).

Full FVM/FEM on curved domains is out of scope — this pins the harness
side of that future: ``ModelSpec`` carries unstructured topology opaquely
(``mesh="unstructured"`` + ``topology``), so experiments version it while
physics stays pluggable. The mock adapter below (non-uniform 1-D steady
conduction, exact linear profile) proves the contract end to end:
protocol compliance, registry round-trip, flux-balance honesty.
"""

import jax.numpy as jnp
import pytest

from harness.physics.adapters import ModelSpec, SimulatorAdapter
from harness.registry import REGISTRIES


class MockUnstructuredAdapter:
    """Non-uniform 1-D rod, steady conduction, Dirichlet ends (in-test mock).

    Nodes need not be uniform — that IS the point: topology rides in
    ``spec.topology`` and the stepwise flux balance
    ``k·(T[i+1]−T[i])/dx[i] = const`` holds to machine precision for the
    linear profile on ANY node distribution.
    """

    def __init__(self, nodes=(0.0, 0.13, 0.47, 0.66, 1.0), k=1.0):
        self.nodes = tuple(float(x) for x in nodes)
        self.k = float(k)
        n = len(self.nodes)
        self.spec = ModelSpec(
            name="mock_unstructured",
            state_vars=("T",),
            units={"T": "degC", "x": "m"},
            description="in-test unstructured conduction mock (T-A6 contract)",
            mesh="unstructured",
            topology={
                "nodes": [(x,) for x in self.nodes],
                "connectivity": [(i, i + 1) for i in range(n - 1)],
                "boundary_sets": {"left": [0], "right": [n - 1]},
                "operator": "fv1d-nonuniform",
            },
        )

    def step(self, state, control, dt_s):
        from harness.physics.adapters import Fluxes  # noqa: WPS433
        T0 = float(control.get("T_left", 0.0))
        T1 = float(control.get("T_right", 1.0))
        xs = jnp.asarray(self.nodes)
        T = T0 + (T1 - T0) * (xs - xs[0]) / (xs[-1] - xs[0])
        dx = jnp.diff(xs)
        fluxes = self.k * jnp.diff(T) / dx
        resid = jnp.max(jnp.abs(fluxes[1:] - fluxes[:-1]))
        next_state = {"T": T, "flux_residual": resid}
        return next_state, Fluxes(extra={"residual": float(resid)})

    def metrics(self, trace):
        T = jnp.asarray(trace["T"])
        return {"T_mid": float(T[len(T) // 2]), "flux_residual": float(trace["flux_residual"])}


def test_spec_defaults_keep_back_compat():
    from harness.physics.adapters import MockSimulatorAdapter
    assert MockSimulatorAdapter().spec.mesh == "structured"
    assert dict(MockSimulatorAdapter().spec.topology) == {}


def test_mock_satisfies_protocol_and_balances():
    mock = MockUnstructuredAdapter()
    assert isinstance(mock, SimulatorAdapter)
    assert mock.spec.mesh == "unstructured"
    assert mock.spec.topology["operator"] == "fv1d-nonuniform"
    state, _ = mock.step({}, {"T_left": 0.0, "T_right": 1.0}, 0.01)
    assert state["flux_residual"] < 1e-12  # exact on any node distribution
    m = mock.metrics(state)
    assert m["T_mid"] == pytest.approx(0.47, rel=1e-9)  # linear profile at x=0.47


def test_registry_round_trip_and_cleanup():
    REGISTRIES["models"].register("mock_unstructured_ta6", MockUnstructuredAdapter)
    try:
        assert "mock_unstructured_ta6" in REGISTRIES["models"].names()
        inst = REGISTRIES["models"].resolve("mock_unstructured_ta6")()
        assert isinstance(inst, SimulatorAdapter)
        assert inst.spec.topology["boundary_sets"] == {"left": [0], "right": [4]}
    finally:
        del REGISTRIES["models"]._factories["mock_unstructured_ta6"]
        REGISTRIES["models"]._entry_points_scanned = False

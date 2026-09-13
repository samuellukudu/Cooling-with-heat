"""Adapter-demo envs — pluggable physics behind the Problem protocol (DESIGN §7.2).

``MockCycle-v0`` demonstrates that the physics behind an environment is
pluggable via :class:`harness.physics.adapters.SimulatorAdapter`: the env
itself (spec / design_space / evaluate / rollout) stays the same shape that
backends expect, but its ``evaluate`` delegates to the adapter's ``step``
+ ``metrics`` instead of calling ``harness.physics.cycle0d`` directly.

``harness.models`` is the adapter registry (``REGISTRIES["models"]``) — see
``harness/registry.py``. This module registers:

- ``mock_cycle0d`` → :class:`harness.physics.adapters.MockSimulatorAdapter`
- ``MockCycle-v0`` → :class:`AdapterCycle` (a ``Problem`` that uses an adapter)

Both registrations are import-time (like the other built-ins), so
``harness.make("MockCycle-v0", ...)`` and ``harness.REGISTRIES["models"]``
work without extra setup. The physics in ``harness/physics/adapters.py``
stays pure (no registry/env imports); this shim is what wires it into the
higher layers, which is why it lives in ``harness/envs`` and is allowed to
import the registries.

The demo env is intentionally trivial (equilibrium cycle via a single
adapter step) — it exists to prove the *plumbing*, not to replace the
real ``TwoBed-v0`` dynamics. A future FMU/OpenModelica adapter would swap
the ``MockSimulatorAdapter`` for a real external solver without touching the
env or backend code.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..materials import MaterialParams, get_material
from ..physics.adapters import MockSimulatorAdapter
from ..profiles import ApplicationProfile, get_profile
from ..registry import REGISTRIES
from .base import DesignSpace, EpisodeTrace, ProblemSpec

DESIGN_KEYS = (
    "q_sat_kg_kg",
    "Q_st_j_kg",
    "e_char_j_mol",
    "n_da",
    "cycle_time_s",
    "hx_mass_factor",
)

ADAPTER_METRIC_KEYS = (
    "COP",
    "SCP_W_kg",
    "delta_q",
    "q_ads",
    "q_des",
    "P_evap_kPa",
    "P_cond_kPa",
    "h_fg_MJ_kg",
)


class AdapterCycle:
    """Equilibrium cycle *via* a :class:`SimulatorAdapter`.

    This satisfies the harness ``Problem`` protocol (``spec``,
    ``design_space``, ``evaluate``, ``metrics_jax``, ``rollout``) and is
    optimizable by ``grad``/``search``/``rl`` unchanged. The only difference
    from :class:`harness.envs.cycle0d.Cycle0D` is that the physics is
    delegated to ``self.adapter`` (``MockSimulatorAdapter`` by default).
    """

    def __init__(
        self,
        material: "str | MaterialParams" = "anchor:Silica gel RD",
        profile: "str | ApplicationProfile" = "cpu",
        adapter: Any | None = None,
        hx_mass_factor: float = 1.35,
    ):
        self.material = get_material(material)
        self.profile = get_profile(profile)
        # Adapter may be instance, factory, or registry key
        if adapter is None:
            self.adapter = MockSimulatorAdapter(
                t_evap_c=self.profile.t_evap_c,
                t_cond_c=self.profile.t_cond_c,
                cycle_time_s=self.profile.cycle_time_s,
                hx_mass_factor=float(hx_mass_factor),
            )
        elif isinstance(adapter, str):
            # Resolve via models registry
            factory = REGISTRIES["models"].resolve(adapter)
            self.adapter = factory()
        elif callable(adapter) and not hasattr(adapter, "step"):
            # Factory callable
            self.adapter = adapter()
        else:
            self.adapter = adapter  # type: ignore[assignment]
        self.spec = ProblemSpec(
            name="MockCycle-v0",
            kind="static",
            metric_keys=ADAPTER_METRIC_KEYS,
            spatial_dim=0, time_resolved=False, grid_type="none",
        )
        self.design_space = DesignSpace(
            keys=DESIGN_KEYS,
            defaults={
                "q_sat_kg_kg": float(self.material.q_sat_kg_kg),
                "Q_st_j_kg": float(self.material.q_st_j_kg),
                "e_char_j_mol": float(self.material.e_char_j_mol),
                "n_da": float(self.material.n_da),
                "cycle_time_s": float(self.profile.cycle_time_s),
                "hx_mass_factor": float(hx_mass_factor),
            },
            bounds={
                "q_sat_kg_kg": (0.08, 0.90),
                "Q_st_j_kg": (2.30e6, 4.10e6),
                "e_char_j_mol": (2000.0, 20000.0),
                "n_da": (1.0, 4.0),
                "cycle_time_s": (30.0, 3600.0),
                "hx_mass_factor": (1.0, 2.0),
            },
        )
        # Expose the adapter's spec for introspection (mirrors the
        # SimulatorAdapter contract: env can forward model metadata).
        self.model_spec = getattr(self.adapter, "spec", None)

    def metrics_jax(self, design: Mapping[str, Any] | None = None) -> dict[str, Any]:
        merged = self.design_space.merge(design)
        state: dict[str, Any] = {}
        control = {
            "q_sat_kg_kg": merged["q_sat_kg_kg"],
            "Q_st_j_kg": merged["Q_st_j_kg"],
            "e_char_j_mol": merged["e_char_j_mol"],
            "n_da": merged["n_da"],
            "t_des_c": self.profile.t_des_c,
            "cycle_time_s": merged["cycle_time_s"],
            "hx_mass_factor": merged["hx_mass_factor"],
        }
        next_state, _fluxes = self.adapter.step(state, control, dt_s=float(merged["cycle_time_s"]))
        # Adapter's metrics() expects a trace; the mock expects the next_state dict
        return self.adapter.metrics(next_state)  # type: ignore[return-value]

    def evaluate(self, design: Mapping[str, float] | None = None) -> dict[str, float]:
        return {k: float(v) for k, v in self.metrics_jax(design).items()}

    def rollout(self, design=None, controls=None, *, n_steps: int = 1) -> EpisodeTrace:
        # Static: same as evaluate, no time series
        del controls, n_steps
        return EpisodeTrace(summary=self.evaluate(design))


class AdapterCycleGym:
    """Gym wrapper for AdapterCycle (exposed at module top-level for import)."""

    pass  # placeholder, defined fully below if gymnasium available


def build_adapter_cycle(
    material: "str | MaterialParams" = "anchor:Silica gel RD",
    profile: "str | ApplicationProfile" = "cpu",
    adapter: Any | None = None,
    hx_mass_factor: float = 1.35,
    **kwargs,
) -> AdapterCycle:
    # ``kwargs`` may carry extra design overrides (e.g. from GUI)
    extra_design = {k: float(v) for k, v in kwargs.items() if k in DESIGN_KEYS}
    inst = AdapterCycle(material, profile, adapter=adapter, hx_mass_factor=hx_mass_factor)
    if extra_design:
        # Overlay onto the defaults without changing the declared keys
        merged = inst.design_space.merge(extra_design)
        inst.design_space = DesignSpace(keys=DESIGN_KEYS, defaults=merged, bounds=inst.design_space.bounds)
    return inst


# Define the real Gym wrapper if gymnasium is available (kept import-safe)
try:
    import gymnasium as _gym  # noqa: WPS433
    import numpy as _np  # noqa: WPS433

    class AdapterCycleGym(_gym.Env):  # type: ignore[no-redef]
        metadata = {"render_modes": []}

        def __init__(self, **kw):
            self._prob = build_adapter_cycle(**kw)

        @property
        def problem(self):
            return self._prob

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return _np.zeros((1,), dtype=_np.float32), {}

        def step(self, action):
            keys = self._prob.design_space.keys
            lo, hi = self._prob.design_space.bounds_for(keys)
            a = _np.clip(_np.asarray(action).ravel(), 0.0, 1.0)
            if a.size != len(keys):
                a = _np.resize(a, len(keys))
            real = _np.asarray(lo) + a * (_np.asarray(hi) - _np.asarray(lo))
            design = {k: float(v) for k, v in zip(keys, real)}
            metrics = self._prob.evaluate(design)
            return _np.zeros((1,), dtype=_np.float32), float(metrics.get("COP", 0.0)), True, False, {"metrics": metrics, "design": design}

        @property
        def observation_space(self):
            import gymnasium as g  # noqa: WPS433

            return g.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=_np.float32)

        @property
        def action_space(self):
            import gymnasium as g  # noqa: WPS433

            lo, hi = self._prob.design_space.bounds_for(self._prob.design_space.keys)
            return g.spaces.Box(low=_np.asarray(lo, dtype=_np.float32), high=_np.asarray(hi, dtype=_np.float32), dtype=_np.float32)

except Exception:  # pragma: no cover

    class AdapterCycleGym:  # type: ignore[no-redef]
        pass


# -- registration -----------------------------------------------------------

def _register_adapter_demo() -> None:
    # SimulatorAdapter registry (harness.models)
    try:
        REGISTRIES["models"].register("mock_cycle0d", lambda: MockSimulatorAdapter())
    except ValueError:
        # Already registered (re-import)
        pass
    # Env registry
    try:
        REGISTRIES["envs"].register("MockCycle-v0", build_adapter_cycle)
    except ValueError:
        pass
    # Gymnasium registration
    try:
        import gymnasium as gym  # noqa: WPS433

        # Only register if the wrapper is a real gym.Env subclass
        if isinstance(AdapterCycleGym, type) and issubclass(AdapterCycleGym, gym.Env):
            try:
                gym.register("MockCycle-v0", entry_point="harness.envs.adapter:AdapterCycleGym")
            except Exception:
                pass
    except Exception:
        pass


_register_adapter_demo()

__all__ = ["AdapterCycle", "build_adapter_cycle"]

"""Telegrapher-v0 — active thermal-wave cancellation as a static Problem.

Hardening trial T-E08 (``examples/08`` demo 2): a primary Gaussian heat
pulse fires left of a centre sensor; the **algorithm** to discover is the
secondary (cancelling) pulse amplitude on the symmetric right. The demo
hand-codes gradient descent for this; here ``grad``/``search``/``rl``
(competing honestly — static problem, so ``search`` is the floor)
discover it through the harness.

Setup mirrors the demo: ``L = 1``, ``kappa = 1``, ``tau = 0.5``
(``v = sqrt(2)``), sensor at the centre, pulses ±0.2 away, Dirichlet 0
ends. Metric ``sensor_mse`` is the demo loss; ``cancellation_pct`` is its
>20 % success bar, measured against the primary-only baseline precomputed
eagerly at construction (static — grad-safe).
"""

from __future__ import annotations

from typing import Any, Mapping

import jax.numpy as jnp

from ..physics import telegrapher as tg
from ..profiles import ApplicationProfile, get_profile
from ..registry import REGISTRIES
from .base import DesignSpace, EpisodeTrace, ProblemSpec

DESIGN_KEYS = ("secondary_amplitude",)
TELEGRAPHER_ENV_METRIC_KEYS = tg.TELEGRAPHER_METRIC_KEYS
TELEGRAPHER_SCHEMA_VERSION = 1

SENSOR_X = 0.5
PULSE_DIST = 0.2


class TelegrapherCancel:
    """Two-pulse cancellation problem (static: one rollout per amplitude)."""

    def __init__(
        self,
        material: Any | None = None,
        profile: "str | ApplicationProfile" = "cpu",
        *,
        n_cells: int = 120,
        dt: float = 0.001,
        t_end: float = 0.25,
        alpha: float = tg.DEFAULT_ALPHA,
        tau: float = tg.DEFAULT_TAU,
    ):
        self.material = material
        self.transport_provenance = "n/a"
        self.profile = get_profile(profile)
        self.n_cells = int(n_cells)
        self.dt = float(dt)
        self.n_steps = int(round(float(t_end) / self.dt))
        self.alpha = float(alpha)
        self.tau = float(tau)
        dx = 1.0 / self.n_cells
        if not tg.check_timestep_1d(dx, self.alpha, self.tau, self.dt):
            raise ValueError(f"Telegrapher dt={self.dt} violates wave CFL")
        self.sensor_idx = self.n_cells // 2
        self.spec = ProblemSpec(name="Telegrapher-v0", kind="static",
                                metric_keys=TELEGRAPHER_ENV_METRIC_KEYS,
                                schema_version=TELEGRAPHER_SCHEMA_VERSION,
                                spatial_dim=1, time_resolved=True, grid_type="uniform_rect")
        self.design_space = DesignSpace(
            keys=DESIGN_KEYS,
            defaults={"secondary_amplitude": -0.5},
            bounds={"secondary_amplitude": (-2.0, 0.0)},
        )
        from ..physics.adapters import ModelSpec  # noqa: WPS433
        self.model_spec = ModelSpec(
            name="telegrapher_cancel",
            state_vars=("u",),
            units={"u": "degC", "dt": "s"},
            dt_min_s=1e-6,
            dt_max_s=0.01,
            description="thermal-wave cancellation (T-E08 hardening)",
        )
        # Primary-only baseline (static floats — eager, once).
        base_traj = self._rollout(0.0)
        self._baseline_peak = float(jnp.max(jnp.abs(base_traj[:, self.sensor_idx])))

    def _u0(self, secondary_amplitude) -> jnp.ndarray:
        dx = 1.0 / self.n_cells
        xc = (jnp.arange(self.n_cells, dtype=jnp.float64) + 0.5) * dx
        primary = tg.gaussian(xc, SENSOR_X - PULSE_DIST)
        secondary = tg.gaussian(xc, SENSOR_X + PULSE_DIST)
        return primary + secondary_amplitude * secondary

    def _rollout(self, secondary_amplitude):
        sim = tg.simulate_telegrapher_1d(
            u0=self._u0(secondary_amplitude), alpha=self.alpha, tau=self.tau,
            n_cells=self.n_cells, dt=self.dt, n_steps=self.n_steps,
            collect_trace=True)
        return sim["trajectory"]

    def metrics_jax(self, design: Mapping[str, Any] | None = None):
        merged = self.design_space.merge(design)
        A = merged["secondary_amplitude"]
        traj = self._rollout(A)
        sensor = traj[:, self.sensor_idx]
        mse = jnp.mean(sensor ** 2)
        peak = jnp.max(jnp.abs(sensor))
        cancel = (self._baseline_peak - peak) / max(self._baseline_peak, 1e-12) * 100.0
        return {"sensor_mse": mse, "sensor_peak": peak,
                "cancellation_pct": cancel, "secondary_amplitude": A}

    def evaluate(self, design: Mapping[str, float] | None = None) -> dict[str, float]:
        return {k: float(v) for k, v in self.metrics_jax(design).items()}

    def rollout(self, design=None, controls=None, *, n_steps: int = 1) -> EpisodeTrace:
        del controls, n_steps
        merged = self.design_space.merge(design)
        traj = self._rollout(merged["secondary_amplitude"])
        summary = {k: float(v) for k, v in self.metrics_jax(design).items()}
        return EpisodeTrace(series={"u_trajectory": traj}, summary=summary)


def build_telegrapher(material: Any | None = None,
                      profile: "str | ApplicationProfile" = "cpu",
                      **kwargs) -> TelegrapherCancel:
    allowed = {"n_cells", "dt", "t_end", "alpha", "tau"}
    ctor = {k: v for k, v in kwargs.items() if k in allowed}
    return TelegrapherCancel(material=material, profile=profile, **ctor)


REGISTRIES["envs"].register("Telegrapher-v0", build_telegrapher)

try:
    from ..physics.adapters import Fluxes as _Fluxes  # noqa: WPS433
    from ..physics.adapters import ModelSpec as _MS  # noqa: WPS433

    class _TelegrapherAdapter:
        def __init__(self):
            self.spec = _MS(name="telegrapher_cancel", state_vars=("u",),
                            units={"u": "degC"}, dt_min_s=1e-6, dt_max_s=0.01,
                            description="telegrapher adapter (T-E08)")

        def step(self, state, control, dt_s):
            return dict(state), _Fluxes()

        def metrics(self, trace):
            return dict(trace)

    try:
        REGISTRIES["models"].register("telegrapher_cancel", _TelegrapherAdapter)
    except ValueError:
        pass
except Exception:
    pass

__all__ = ["TelegrapherCancel", "build_telegrapher", "DESIGN_KEYS", "TELEGRAPHER_ENV_METRIC_KEYS"]

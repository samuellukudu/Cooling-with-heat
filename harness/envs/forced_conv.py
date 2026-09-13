"""ForcedConv-v0 — channel cooling as a static harness Problem (trial T-A5).

Env = (advection-diffusion physics) × (profile scenario). Material axis is
absent (``transport_provenance="n/a"``) — the discovery axes are **design**
(``inlet_velocity``) and, later, **algorithms** (``U(t)`` schedules).

Discovery discipline: ``source_amplitude`` (heater power) is a fixed
scenario parameter, NOT an optimizable key — otherwise the optimizer would
trivially turn the heater down. ``inlet_velocity`` bounds ``[0.05, 2.0]``
mirror ``examples/09`` (Pe ~ 20–800 at ``alpha=0.01``).
"""

from __future__ import annotations

from typing import Any, Mapping

from ..physics import forced_conv as fc
from ..profiles import ApplicationProfile, get_profile
from ..registry import REGISTRIES
from .base import DesignSpace, EpisodeTrace, ProblemSpec

DESIGN_KEYS = ("inlet_velocity",)
FORCED_CONV_METRIC_KEYS = fc.FORCED_CONV_METRIC_KEYS
FORCEDCONV_SCHEMA_VERSION = 1


class ForcedConv:
    """2-D channel cooling problem (static: one rollout per design)."""

    def __init__(
        self,
        material: Any | None = None,
        profile: "str | ApplicationProfile" = "datacenter",
        *,
        alpha: float = fc.DEFAULT_ALPHA,
        Lx: float = fc.DEFAULT_LX,
        Ly: float = fc.DEFAULT_LY,
        nx: int = 24,
        ny: int = 8,
        dt: float = 0.002,
        n_steps: int = 200,
        source_amplitude: float = fc.DEFAULT_SOURCE_AMP,
    ):
        # Material accepted-and-ignored for harness.make() uniformity.
        self.material = material
        self.transport_provenance = "n/a"
        self.profile = get_profile(profile)
        self.alpha = float(alpha)
        self.Lx, self.Ly = float(Lx), float(Ly)
        self.nx, self.ny = int(nx), int(ny)
        self.dt = float(dt)
        self.n_steps = int(n_steps)
        self.source_amplitude = float(source_amplitude)
        dx = self.Lx / self.nx
        dy = self.Ly / self.ny
        if not fc.check_timestep(dx, dy, self.alpha, 2.0, self.dt):
            raise ValueError(
                f"ForcedConv dt={self.dt} violates CFL at U_max=2.0 "
                f"(dx={dx:.4g} dy={dy:.4g} alpha={self.alpha}); "
                f"limit={fc.max_timestep(dx, dy, self.alpha, 2.0):.3e}")
        self.spec = ProblemSpec(name="ForcedConv-v0", kind="static",
                                metric_keys=FORCED_CONV_METRIC_KEYS,
                                schema_version=FORCEDCONV_SCHEMA_VERSION,
                                spatial_dim=2, time_resolved=True, grid_type="uniform_rect")
        self.design_space = DesignSpace(
            keys=DESIGN_KEYS,
            defaults={"inlet_velocity": 1.0},
            bounds={"inlet_velocity": (0.05, 2.0)},
        )
        # Adapter spec for introspection (SimulatorAdapter contract §7.2).
        from ..physics.adapters import ModelSpec  # noqa: WPS433 (env may import physics)
        self.model_spec = ModelSpec(
            name="forced_conv",
            state_vars=("T",),
            units={"T": "degC", "inlet_velocity": "m/s", "dt": "s"},
            dt_min_s=1e-5,
            dt_max_s=0.01,
            description="2-D channel advection-diffusion (T-A5 trial)",
        )

    def metrics_jax(self, design: Mapping[str, Any] | None = None):
        merged = self.design_space.merge(design)
        U = merged["inlet_velocity"]
        sim = fc.simulate_forced_conv(
            inlet_velocity=U, alpha=self.alpha, Lx=self.Lx, Ly=self.Ly,
            nx=self.nx, ny=self.ny, dt=self.dt, n_steps=self.n_steps,
            source_amplitude=self.source_amplitude)
        return fc.summary_from_final(sim["final"], inlet_velocity=U,
                                     alpha=self.alpha, Lx=self.Lx)

    def evaluate(self, design: Mapping[str, float] | None = None) -> dict[str, float]:
        merged = self.design_space.merge(design)
        U = float(merged["inlet_velocity"])
        dx = self.Lx / self.nx
        dy = self.Ly / self.ny
        if not fc.check_timestep(dx, dy, self.alpha, U, self.dt):
            raise ValueError(f"dt={self.dt} violates CFL at U={U}")
        return {k: float(v) for k, v in self.metrics_jax(design).items()}

    def rollout(self, design=None, controls=None, *, n_steps: int = 1) -> EpisodeTrace:
        del controls, n_steps
        merged = self.design_space.merge(design)
        U = merged["inlet_velocity"]
        sim = fc.simulate_forced_conv(
            inlet_velocity=U, alpha=self.alpha, Lx=self.Lx, Ly=self.Ly,
            nx=self.nx, ny=self.ny, dt=self.dt, n_steps=self.n_steps,
            source_amplitude=self.source_amplitude, collect_trace=True)
        summary = {k: float(v) for k, v in
                   fc.summary_from_final(sim["final"], inlet_velocity=U,
                                         alpha=self.alpha, Lx=self.Lx).items()}
        traj = sim.get("trajectory")
        series = {"T_trajectory": traj} if traj is not None else {}
        return EpisodeTrace(series=series, summary=summary)


def build_forced_conv(material: Any | None = None,
                      profile: "str | ApplicationProfile" = "datacenter",
                      **kwargs) -> ForcedConv:
    # Filter GUI/legacy kwargs to constructor params.
    allowed = {"alpha", "Lx", "Ly", "nx", "ny", "dt", "n_steps", "source_amplitude"}
    ctor = {k: v for k, v in kwargs.items() if k in allowed}
    # Drop design overrides passed as structural kwargs (they flow via evaluate).
    return ForcedConv(material=material, profile=profile, **ctor)


REGISTRIES["envs"].register("ForcedConv-v0", build_forced_conv)

# SimulatorAdapter registration (DESIGN §7.2 — models registry).
try:
    from ..physics.adapters import Fluxes as _Fluxes  # noqa: WPS433
    from ..physics.adapters import ModelSpec as _MS  # noqa: WPS433

    class _ForcedConvAdapter:
        """Thin adapter wrapper (non-grad path; grad uses env.metrics_jax)."""

        def __init__(self):
            self.spec = _MS(name="forced_conv", state_vars=("T",),
                            units={"T": "degC"}, dt_min_s=1e-5, dt_max_s=0.01,
                            description="forced-conv adapter (T-A5)")

        def step(self, state, control, dt_s):
            import jax.numpy as _jnp  # noqa: WPS433
            U = float(control.get("inlet_velocity", 1.0))
            sim = fc.simulate_forced_conv(inlet_velocity=U, n_steps=10, dt=float(dt_s),
                                          nx=12, ny=6)
            return {"T_max": float(_jnp.max(sim["final"]))}, _Fluxes()

        def metrics(self, trace):
            return dict(trace)

    try:
        REGISTRIES["models"].register("forced_conv", _ForcedConvAdapter)
    except ValueError:
        pass
except Exception:
    pass

__all__ = ["ForcedConv", "build_forced_conv", "DESIGN_KEYS", "FORCED_CONV_METRIC_KEYS"]

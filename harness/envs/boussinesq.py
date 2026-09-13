"""Boussinesq-v0 — resolved RB cavity as a static Problem (T-A1 extension).

Env = (resolved MAC physics) × (fluid row) × (profile scenario), sharing
the correlation level's design language: design ``delta_T_K`` with fluid
``material`` and structural gap ``L_m`` map to ``Ra`` internally
(``Ra = group·ΔT·L³``). The correlation ``NaturalConv-v0`` is the fast
screen; this env answers the resolved-flow questions it cannot pose.
Validated envelope (``in_validated`` flag): ``N ≥ 24``, ``Ra ≤ 3e5``.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..physics import boussinesq as bq
from ..physics.natural_conv import resolve_fluid
from ..profiles import ApplicationProfile, get_profile
from ..registry import REGISTRIES
from .base import DesignSpace, EpisodeTrace, ProblemSpec

DESIGN_KEYS = ("delta_T_K",)
BOUSSINESQ_ENV_METRIC_KEYS = bq.BOUSSINESQ_METRIC_KEYS
BOUSSINESQ_SCHEMA_VERSION = 1


def _resolve_fluid_name(material: Any) -> tuple[str, str]:
    if isinstance(material, str):
        try:
            f = resolve_fluid(material.strip())
            return f["name"], "fluid"
        except KeyError:
            from ..physics.natural_conv import DEFAULT_FLUID  # noqa: WPS433
            return DEFAULT_FLUID, "anchor-fallback"
    if material is None:
        from ..physics.natural_conv import DEFAULT_FLUID  # noqa: WPS433
        return DEFAULT_FLUID, "default"
    from ..physics.natural_conv import DEFAULT_FLUID  # noqa: WPS433
    return DEFAULT_FLUID, "anchor-fallback"


class Boussinesq:
    """Resolved RB enclosure over fluid × scenario."""

    def __init__(
        self,
        material: Any = "air",
        profile: "str | ApplicationProfile" = "human",
        *,
        L_m: float = 0.03,
        n_cells: int = 24,
        t_end: float = 0.6,
    ):
        """Defaults keep air in the validated laminar envelope across the
        ΔT bounds (Ra(30 K, 3 cm) ≈ 7e4)."""
        self.fluid_name, self.fluid_provenance = _resolve_fluid_name(material)
        self.material = material
        self.profile = get_profile(profile)
        self.fluid = resolve_fluid(self.fluid_name)
        self.L_m = float(L_m)
        self.n_cells = int(n_cells)
        self.t_end = float(t_end)
        dx = 1.0 / self.n_cells
        lo, hi = 1.0, 30.0
        ra_hi = self.fluid["ra_group"] * hi * self.L_m ** 3
        _, self._dt = bq.cfl_params(ra_hi, float(self.fluid["Pr"]), dx)
        self._ra_hi = float(ra_hi)
        # JIT-cache the rollout keyed on shapes (Bed1D._advance_jit pattern):
        # Ra arrives as an array argument, so per-design evaluations and
        # grad steps reuse one compilation instead of retracing per value.
        # Only array metrics cross the jit boundary (dx/dt ints stay out).
        import jax  # noqa: WPS433
        _Pr, _n, _dt, _te, _rahi = (float(self.fluid["Pr"]), self.n_cells,
                                    self._dt, self.t_end, self._ra_hi)

        def _run(Ra_arr):
            out = bq.simulate_boussinesq(
                Ra=Ra_arr, Pr=_Pr, n_cells=_n, dt=_dt, t_end=_te,
                Ra_hi=_rahi)
            return out["Nu"], out["Nu_bot"], out["Nu_top"], out["u_max"]

        self._metrics_jit = jax.jit(_run)
        self.spec = ProblemSpec(name="Boussinesq-v0", kind="static",
                                metric_keys=BOUSSINESQ_ENV_METRIC_KEYS,
                                schema_version=BOUSSINESQ_SCHEMA_VERSION,
                                spatial_dim=2, time_resolved=True, grid_type="mac")
        self.design_space = DesignSpace(
            keys=DESIGN_KEYS,
            defaults={"delta_T_K": 10.0},
            bounds={"delta_T_K": (lo, hi)},
        )
        from ..physics.adapters import ModelSpec  # noqa: WPS433
        self.model_spec = ModelSpec(
            name="boussinesq",
            state_vars=("u", "v", "p", "T"),
            units={"delta_T_K": "K", "Nu": "-"},
            dt_min_s=1e-9,
            dt_max_s=1.0,
            description="resolved RB cavity (T-A1 extension)",
        )

    def _ra_of(self, delta_T_K) -> Any:
        return self.fluid["ra_group"] * delta_T_K * self.L_m ** 3

    def metrics_jax(self, design: Mapping[str, Any] | None = None):
        import jax.numpy as jnp  # noqa: WPS433
        merged = self.design_space.merge(design)
        dT = merged["delta_T_K"]
        Ra = self._ra_of(dT)
        out = self._metrics_jit(jnp.asarray(Ra))
        Nu, Nu_bot, Nu_top, u_max = out
        validated = ((Ra <= bq.RA_VALID_HI) & (self.n_cells >= bq.N_VALID_LO))
        return {"Nu": Nu, "Nu_bot": Nu_bot,
                "Nu_top": Nu_top, "u_max": u_max,
                "Ra": Ra, "delta_T_K": dT,
                "in_validated": jnp_validated(validated)}

    def evaluate(self, design: Mapping[str, float] | None = None) -> dict[str, float]:
        return {k: float(v) for k, v in self.metrics_jax(design).items()}

    def rollout(self, design=None, controls=None, *, n_steps: int = 1) -> EpisodeTrace:
        del controls, n_steps
        merged = self.design_space.merge(design)
        dT = merged["delta_T_K"]
        Ra = self._ra_of(dT)
        out = bq.simulate_boussinesq(
            Ra=Ra, Pr=float(self.fluid["Pr"]), n_cells=self.n_cells,
            dt=self._dt, t_end=self.t_end,
            Ra_hi=self.fluid["ra_group"] * 30.0 * self.L_m ** 3,
            collect_trace=True)
        summary = {k: float(v) for k, v in self.metrics_jax(design).items()}
        traj = out.get("trajectory")
        return EpisodeTrace(series={"T_trajectory": traj} if traj is not None else {},
                            summary=summary)


def jnp_validated(flag) -> Any:
    """0/1 float from a (possibly traced) boolean."""
    import jax.numpy as jnp  # noqa: WPS433
    return jnp.where(flag, 1.0, 0.0)


def build_boussinesq(material: Any = "air",
                     profile: "str | ApplicationProfile" = "human",
                     **kwargs) -> Boussinesq:
    allowed = {"L_m", "n_cells", "t_end"}
    ctor = {k: v for k, v in kwargs.items() if k in allowed}
    return Boussinesq(material=material, profile=profile, **ctor)


REGISTRIES["envs"].register("Boussinesq-v0", build_boussinesq)

try:
    from ..physics.adapters import Fluxes as _Fluxes  # noqa: WPS433
    from ..physics.adapters import ModelSpec as _MS  # noqa: WPS433

    class _BoussinesqAdapter:
        def __init__(self):
            self.spec = _MS(name="boussinesq", state_vars=("u", "v", "p", "T"),
                            units={"Nu": "-"}, dt_min_s=1e-9, dt_max_s=1.0,
                            description="boussinesq adapter (T-A1)")

        def step(self, state, control, dt_s):
            return dict(state), _Fluxes()

        def metrics(self, trace):
            return dict(trace)

    try:
        REGISTRIES["models"].register("boussinesq", _BoussinesqAdapter)
    except ValueError:
        pass
except Exception:
    pass

__all__ = ["Boussinesq", "build_boussinesq", "DESIGN_KEYS", "BOUSSINESQ_ENV_METRIC_KEYS"]

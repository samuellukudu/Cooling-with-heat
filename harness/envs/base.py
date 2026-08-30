"""Problem protocol, specs, and shared objective math (``DESIGN.md`` §3).

Everything is duck-typed through ``typing.Protocol`` so external simulators
and third-party plugins can participate without inheriting from us. This
module holds only structure and shared math — no backend logic (import
direction rules, root pyproject).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import jax.numpy as jnp

from ..registry import REGISTRIES

register_env = REGISTRIES["envs"].register


@dataclass(frozen=True)
class Sensor:
    """One observation channel: name, unit, and expected range."""

    name: str
    unit: str
    lo: float
    hi: float


@dataclass(frozen=True)
class ActionSpec:
    """Action space description; ``kind`` ∈ {"none", "continuous", "discrete"}."""

    kind: str = "none"
    names: tuple[str, ...] = ()
    lo: tuple[float, ...] | None = None
    hi: tuple[float, ...] | None = None
    choices: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ProblemSpec:
    name: str
    kind: str  # "static" | "dynamic"
    obs_spec: tuple[Sensor, ...] = ()
    action_spec: ActionSpec = ActionSpec()
    metric_keys: tuple[str, ...] = ()
    schema_version: int = 1


@dataclass(frozen=True)
class DesignSpace:
    """Optimizable design parameters: keys, defaults, optional bounds.

    ``defaults`` is the complete parameter set (every key the physics
    needs, with values); ``keys`` declares the subset backends may vary —
    often equal, but a screen-mirroring experiment may fix some parameters
    by declaring a two-key space over the full default set (see the V7
    test). Bounds default to ``default ± max(1, |default|)`` when a backend
    needs finite values and none are declared; prefer declaring them.
    """

    keys: tuple[str, ...]
    defaults: Mapping[str, float]
    bounds: Mapping[str, tuple[float, float]] = field(default_factory=dict)

    def merge(self, design: Mapping[str, float] | None = None) -> dict[str, float]:
        """All defaults overlaid with ``design``; unknown keys raise.

        Values pass through unconverted so JAX tracers survive — the grad
        backend calls this inside ``jax.jit`` with symbolic inputs.
        """
        merged = dict(self.defaults)
        if design:
            unknown = set(design) - set(self.defaults)
            if unknown:
                raise KeyError(f"unknown design keys {sorted(unknown)}; valid: {sorted(self.defaults)}")
            merged.update(design)
        return merged

    def bounds_for(self, keys) -> tuple[list[float], list[float]]:
        lo, hi = [], []
        for k in keys:
            b = self.bounds.get(k)
            if b is None:
                d = float(self.defaults[k])
                span = max(1.0, abs(d))
                b = (d - span, d + span)
            lo.append(float(b[0]))
            hi.append(float(b[1]))
        return lo, hi


@dataclass
class EpisodeTrace:
    """Per-step series plus episode-level summary metrics.

    Static problems return an empty ``series``; dynamic envs record every
    documented channel. ``summary`` always carries the ``ProblemSpec``
    metric keys so downstream code never special-cases problem kinds.
    """

    series: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Objective:
    """What a backend maximizes: a weighted (optionally normalized) score
    over problem metrics, with optional soft constraint penalties.

    ``weights`` maps metric keys to weights (maximize semantics — use a
    negative weight to minimize). ``normalize`` optionally maps metric keys
    to ``(lo, hi)`` min-max ranges applied before weighting. ``constraints``
    maps metric keys to ``(op, threshold)`` with ``op`` ∈ {">=", "<="};
    violations subtract ``penalty_scale × violation`` from the score (soft
    penalty — report metrics, don't trust the penalty as a hard guarantee).
    """

    weights: Mapping[str, float]
    normalize: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    constraints: Mapping[str, tuple[str, float]] = field(default_factory=dict)
    penalty_scale: float = 100.0

    @classmethod
    def single(cls, metric: str, *, normalize: tuple[float, float] | None = None) -> "Objective":
        norm = {metric: normalize} if normalize else {}
        return cls(weights={metric: 1.0}, normalize=norm)

    def snapshot(self) -> dict[str, Any]:
        return {
            "weights": dict(self.weights),
            "normalize": {k: list(v) for k, v in self.normalize.items()},
            "constraints": {k: list(v) for k, v in self.constraints.items()},
            "penalty_scale": self.penalty_scale,
        }


def objective_value(objective: Objective, metrics: Mapping[str, Any]):
    """Weighted normalized score + constraint penalties; works on floats
    and JAX scalars alike (backends share this exact function)."""
    total = 0.0
    for key, w in objective.weights.items():
        if key not in metrics:
            raise KeyError(f"objective weight refers to unknown metric {key!r}; metrics: {sorted(metrics)}")
        v = metrics[key]
        if key in objective.normalize:
            lo, hi = objective.normalize[key]
            v = jnp.clip((v - lo) / (hi - lo), 0.0, 1.0)
        total = total + w * v
    for key, (op, threshold) in objective.constraints.items():
        v = metrics[key]
        if op == ">=":
            violation = jnp.maximum(threshold - v, 0.0)
        elif op == "<=":
            violation = jnp.maximum(v - threshold, 0.0)
        else:
            raise ValueError(f"constraint operator must be '>=' or '<=', got {op!r}")
        total = total - objective.penalty_scale * violation
    return total


class Problem(Protocol):
    """Anything a backend can optimize (DESIGN §3).

    Static problems implement ``evaluate``/``metrics_jax`` and a trivial
    ``rollout``; dynamic problems additionally implement the gym-style
    ``reset``/``step`` pair (H1+).
    """

    spec: ProblemSpec
    design_space: DesignSpace

    def evaluate(self, design: Mapping[str, float] | None = None) -> dict[str, float]: ...

    def metrics_jax(self, design: Mapping[str, Any]) -> dict[str, Any]: ...

    def rollout(self, design=None, controls=None, *, n_steps: int = 1) -> EpisodeTrace: ...


def validate_problem(problem: Any) -> None:
    """Fail with an actionable message if ``problem`` cannot be optimized."""
    missing = [a for a in ("spec", "design_space", "evaluate", "metrics_jax", "rollout") if not hasattr(problem, a)]
    if missing:
        raise TypeError(
            f"{type(problem).__name__!r} does not satisfy the harness Problem protocol; "
            f"missing members: {missing} (see harness/envs/base.py)"
        )

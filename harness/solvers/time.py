"""4D time utilities: time grids, thinning, trajectory contracts (DESIGN §4-5).

The 4th dimension (time) is uniform across the dimensional ladder:

- trajectories are time-first: ``(n_steps+1, *spatial)`` with frame 0
  the initial condition;
- ``save_every`` subsamples 3D (and large 2D) rollouts to bound memory;
- series channels are thinned to ≤ 2048 samples for diagnostics;
- ``n_steps`` is always static (scan length), so every rollout stays
  jittable and differentiable.
"""

from __future__ import annotations

import jax.numpy as jnp

#: Max diagnostic samples kept per channel (Bed1D ``simulate_bed`` rule).
MAX_TRACE_SAMPLES = 2048


def time_grid(t_span: tuple[float, float], dt: float) -> jnp.ndarray:
    """Node times ``[t0 .. t0+n*dt]`` for ``n = (t1-t0)/dt`` steps."""
    t0, t1 = t_span
    n_steps = int((float(t1) - float(t0)) / float(dt))
    if n_steps < 1:
        raise ValueError(f"t_span too short for dt={dt}: {t_span}")
    return jnp.asarray(t0) + jnp.arange(n_steps + 1) * float(dt)


def n_steps_for(t_span: tuple[float, float], dt: float) -> int:
    """Number of physics steps for ``t_span`` at ``dt``."""
    t0, t1 = t_span
    n_steps = int((float(t1) - float(t0)) / float(dt))
    if n_steps < 1:
        raise ValueError(f"t_span too short for dt={dt}: {t_span}")
    return n_steps


def thin_trace(ys: jnp.ndarray, max_samples: int = MAX_TRACE_SAMPLES) -> jnp.ndarray:
    """Stride-thin a ``(n_steps, ...)`` trace to ``≤ max_samples`` rows."""
    n = int(ys.shape[0])
    stride = max(1, n // int(max_samples))
    return ys[::stride]


def save_every_outer_steps(n_steps: int, save_every: int) -> int:
    """Number of outer (saved) steps for ``n_steps`` with ``save_every``."""
    if save_every < 1:
        raise ValueError(f"save_every must be >= 1, got {save_every}")
    return int(n_steps) // int(save_every)


__all__ = [
    "MAX_TRACE_SAMPLES",
    "time_grid",
    "n_steps_for",
    "thin_trace",
    "save_every_outer_steps",
]

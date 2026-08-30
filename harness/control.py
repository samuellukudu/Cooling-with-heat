"""First control experiments on ``Bed1D-v0`` (DESIGN §12 H1.5).

The reusable experiment logic behind the control notebook and the V5
gradient gate:

- :func:`switch_time_sweep` — COP/SCP of the steady periodic response as
  a function of the switch time (the headline plot's data);
- :func:`optimize_controls` — the ``grad`` backend on the control-space
  problem (:class:`harness.envs.bed1d.Bed1DControls`);
- :func:`gradient_check` — the V5 gate: ``jax.grad`` vs central finite
  differences at probe points.

Soft switching (``soft_switch=True``) is the valve-crossfade
gradient-diagnostic of DESIGN §4.2: with the physical hard valve the
flip times quantize to physics substeps, so the episode metric is a
staircase in the switch time and ``∂SCP/∂t_switch`` collapses to the
trivial ``−SCP/t`` denominator term — blind to the physics and actively
misleading for optimization. Soft switching gives the boundary substep
its exact fractional valve duty (pressure and fluid temperature blend
across one substep), making the metric continuous in the switch time
with the true physical slope. Open Question 3 is decided from exactly
this comparison (DESIGN §13.3 records the decision).
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .backends import optimize
from .envs.base import Objective
from .envs.bed1d import Bed1D, Bed1DControls

# A documented demo design for the control experiments: anchor RD silica
# gel isotherm with mid-span effective transport (tau_kin ~ 100 s), fast
# wall coupling, cpu profile. Fast enough that the SCP-vs-switch-time
# optimum sits inside the §5.2 action window, slow enough to be
# swing-limited (the interesting regime).
DEMO_DESIGN = {
    "k_ldf_s_1": 0.01,
    "h_wall_w_m2_k": 500.0,
}


def demo_problem(t_switch_bounds: "tuple[float, float] | None" = None,
                 soft_switch: bool = False,
                 dt_phys_s: float = 0.1,
                 n_cycles: int = 2,
                 n_steps: "int | None" = None) -> Bed1DControls:
    """The control-space problem at the demo design and numerics.

    ``n_steps`` overrides the default horizon (sized from the switch-time
    upper bound) — useful to shorten episodes in tests; the metrics come
    from the last completed cycles either way.
    """
    bed = Bed1D(design=dict(DEMO_DESIGN), n_cells=8,
                dt_phys_s=dt_phys_s, soft_switch=soft_switch,
                n_cycles=max(n_cycles, 2))
    return Bed1DControls(bed, t_switch_bounds=t_switch_bounds,
                         n_cycles=n_cycles, dt_phys_s=dt_phys_s,
                         n_steps=n_steps)


def switch_time_sweep(problem: Bed1DControls,
                      t_switch_grid: Sequence[float]) -> list[dict[str, float]]:
    """Steady-periodic COP/SCP at each switch time (T_f,des at default).

    Returns one row per grid point with ``t_switch_s``, ``COP``,
    ``SCP_W_kg``, ``delta_q``.
    """
    rows = []
    for t in t_switch_grid:
        m = problem.evaluate({"t_switch_s": float(t)})
        rows.append({"t_switch_s": float(t), "COP": m["COP"],
                     "SCP_W_kg": m["SCP_W_kg"], "delta_q": m["delta_q"]})
    return rows


def optimize_controls(problem: Bed1DControls, objective: Objective, **kwargs):
    """Run a backend on the control-space problem (thin wrapper over
    :func:`harness.backends.optimize`; keyword pass-through)."""
    return optimize(problem, objective, **kwargs)


def gradient_check(problem: Bed1DControls, probes: Sequence[tuple[float, float]],
                   *, h_switch: float = 0.01, h_t_f: float = 0.05,
                   metric: str = "SCP_W_kg") -> list[dict[str, float]]:
    """V5 gate data: ``jax.grad`` vs central finite differences of
    ``metric`` at each probe point ``(t_switch_s, t_f_des_c)``.

    ``h_switch`` must stay below the physics substep for the hard-switch
    comparison (the metric is flat between substeps); for soft switching
    any small step works.
    """
    def score(c):
        return problem.metrics_jax({"t_switch_s": c[0], "t_f_des_c": c[1]})[metric]

    score_j = jax.jit(score)
    grad_j = jax.jit(jax.grad(score))

    def metric_of(t_switch, t_f_des):
        return float(score_j(jnp.asarray([t_switch, t_f_des], dtype=jnp.float64)))

    rows = []
    for t_switch, t_f_des in probes:
        g_ad = np.asarray(grad_j(jnp.asarray([t_switch, t_f_des], dtype=jnp.float64)))
        g_fd = np.array([
            (metric_of(t_switch + h_switch, t_f_des) - metric_of(t_switch - h_switch, t_f_des)) / (2.0 * h_switch),
            (metric_of(t_switch, t_f_des + h_t_f) - metric_of(t_switch, t_f_des - h_t_f)) / (2.0 * h_t_f),
        ])
        denom = np.maximum(np.abs(g_fd), 1e-6)
        rows.append({
            "t_switch_s": float(t_switch), "t_f_des_c": float(t_f_des),
            "grad_ad": g_ad.tolist(), "grad_fd": g_fd.tolist(),
            "rel_err": float(np.max(np.abs(g_ad - g_fd) / denom)),
        })
    return rows

"""PINN acceptance metrics — the verifier side of the loop.

The harness stays the judge (its V-gates are reused, not rewritten):

- field accuracy: relative L2 on ``T`` / ``q`` vs held-out solver rollouts
  (family-split: no material family leaks train→test).
- physics honesty: mean PDE / BC / IC residual norms on test conditions —
  the AU "known laws score proposals" signal, reported even where no data
  exists (real-condition rows).
- directional-feedback agreement: ``jax.grad`` through the surrogate vs
  central finite differences on a scalar design objective (mirrors V5:
  rel. error must be small, else the "improvement direction" is a lie).
- super-resolution: net trained on coarse-sampled fields vs fine solver
  rollout — the resolution-invariance claim, measured.

System-level top-k hit rate (ROADMAP business metric) arrives with N2
(structure→property); N1 reports the COP-proxy below so the coupling point
is exercised early: ``Q_cool = h_fg · Δq̄`` from the predicted uptake swing
over the adsorption window.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def rel_l2(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    denom = float(np.sqrt(np.mean(true ** 2)))
    return float(np.sqrt(np.mean((pred - true) ** 2)) / (denom + 1e-18))


def field_errors(predict_fn, x_hat: np.ndarray, t_hat: np.ndarray,
                 cond: jax.Array, T_true: np.ndarray,
                 q_true: np.ndarray) -> dict[str, float]:
    """Relative-L2 field errors of one condition's test points."""
    T_pred, q_pred = predict_fn(x_hat, t_hat, cond)
    return {"rel_l2_T": rel_l2(np.asarray(T_pred), T_true),
            "rel_l2_q": rel_l2(np.asarray(q_pred), q_true),
            "mae_T_K": float(np.mean(np.abs(np.asarray(T_pred) - T_true))),
            "mae_q": float(np.mean(np.abs(np.asarray(q_pred) - q_true)))}


def residual_summary(residual_fn, x_hat: np.ndarray, t_hat: np.ndarray,
                     cond: jax.Array, psat_fn) -> dict[str, float]:
    """Mean |residual| of energy/LDF over query points (no labels needed)."""
    import sys  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
    from bed_pinn import pinn_residuals  # noqa: PLC0415

    res = jax.vmap(lambda xh, th: pinn_residuals(
        residual_fn, xh, th, cond, psat_fn))(jnp.asarray(x_hat), jnp.asarray(t_hat))
    return {"mean_abs_r_energy": float(jnp.mean(jnp.abs(res["r_energy"]))),
            "mean_abs_r_ldf": float(jnp.mean(jnp.abs(res["r_ldf"])))}


def grad_agreement(objective_fn, cond: jax.Array, *, wrt: int = 8,
                   eps: float = 1e-3) -> dict[str, float]:
    """Autodiff vs central-FD agreement of ``dObjective/dCond[wrt]]``.

    ``objective_fn(cond) -> scalar`` must be a pure function of the condition
    vector (e.g. mean predicted final-T). Mirrors harness V5 (rel. err < 1e-3
    on smooth controls); the gate here is < 5e-2 on the normalized cond axis.
    """
    g_ad = float(jax.grad(objective_fn)(cond)[wrt])
    e = np.zeros(cond.shape[0], dtype=np.float32)
    e[wrt] = eps
    f_hi = float(objective_fn(cond + jnp.asarray(e)))
    f_lo = float(objective_fn(cond - jnp.asarray(e)))
    g_fd = (f_hi - f_lo) / (2.0 * eps)
    denom = max(abs(g_fd), 1e-6)
    return {"grad_ad": g_ad, "grad_fd": g_fd,
            "rel_err": float(abs(g_ad - g_fd) / denom)}


def super_resolution_error(predict_fn, fields_fine: dict[str, np.ndarray],
                           cond: jax.Array) -> dict[str, float]:
    """Surrogate (trained coarse) vs FINE solver rollout — resolution claim."""
    n_steps = fields_fine["n_steps"]
    n_cells = fields_fine["n_cells"]
    xs = (np.arange(n_cells, dtype=np.float32) + 0.5) / n_cells
    xg, tg = np.meshgrid(xs, np.linspace(0, 1, n_steps + 1).astype(np.float32))
    T_pred, q_pred = predict_fn(xg.ravel(), tg.ravel(), cond)
    T_pred = np.asarray(T_pred).reshape(n_steps + 1, n_cells)
    q_pred = np.asarray(q_pred).reshape(n_steps + 1, n_cells)
    return {"superres_rel_l2_T": rel_l2(T_pred, fields_fine["T"]),
            "superres_rel_l2_q": rel_l2(q_pred, fields_fine["q"])}


def cooling_proxy(q_pred: np.ndarray, *, h_fg_j_kg: float) -> dict[str, float]:
    """``Q_cool = h_fg · Δq̄`` from the predicted uptake swing (bare-bed proxy).

    ``q_pred`` shape ``(n_time, n_cells)`` over the phase; the swing is
    mean-uptake last-minus-first. Positive on adsorption, negative on
    desorption — the sign is the check, not just the magnitude.
    """
    q_pred = np.asarray(q_pred, dtype=np.float64)
    dq = float(np.mean(q_pred[-1]) - np.mean(q_pred[0]))
    return {"delta_q_mean": dq, "Q_cool_J_kg": float(h_fg_j_kg * dq)}


__all__ = ["cooling_proxy", "field_errors", "grad_agreement", "rel_l2",
           "residual_summary", "super_resolution_error"]

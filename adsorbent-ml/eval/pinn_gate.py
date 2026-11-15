"""PINN hardening gate — the N1 acceptance pass (ROADMAP milestone 1).

Loads a trained checkpoint and judges it against the Bed1D oracle:

- **field interpolation** (rel-L2 on ``T``/``q``): fresh single-phase solver
  rollouts at *unseen operating points and thickness* for the trained
  families — the "matches the Bed1D solver" gate. Same-family by design: the
  real-condition rows carry no field labels, so cross-family FIELD accuracy
  is not claimed anywhere (honesty rule, corpus.py docstring).
- **super-resolution**: fine-grid oracle rollout (16 cells × 800 steps) vs
  the coarse-trained surrogate — the resolution-invariance claim, measured.
- **physics honesty on real rows**: mean PDE residual norms on
  ``load_real_conditions`` rows at sampled operating points — the label-free
  signal the stage-C curriculum trains on.
- **directional feedback**: autodiff vs central-FD on ``d(mean final T)/d
  t_f`` (mirrors V5; gate < 5e-2 on the normalized cond axis).
- **cooling proxy sign**: predicted uptake swing positive on adsorption,
  negative on desorption (``Q_cool = h_fg · Δq̄``).

Emits ``gate_report.json`` (per-gate value/limit/pass + overall) and prints
a table. Exit code 0 only when the overall gate passes (CI-friendly).

Gate limits (encoded here so the gate is runnable, not aspirational):
rel-L2 T ≤ 0.05 · rel-L2 q ≤ 0.10 · superres T ≤ 0.10 / q ≤ 0.15 ·
mean |r_energy|, |r_ldf| ≤ 0.5 (normalized) · grad rel-err < 5e-2 ·
cooling sign correct both directions.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "models"))
sys.path.insert(0, str(_HERE.parent / "data"))
sys.path.insert(0, str(_HERE.parent / "training"))
sys.path.insert(0, str(_HERE))

from bed_pinn import (COND_DIM, decode_condition, encode_condition,  # noqa: E402
                      predict_batch)
from corpus import (OP_ENVELOPE, config_to_cond, load_real_conditions,  # noqa: E402
                    preset_configs, rollout_kwargs, rollout_phase_fields)
from pinn_metrics import (cooling_proxy, field_errors, grad_agreement,  # noqa: E402
                          rel_l2, residual_summary, super_resolution_error)
from train_pinn import TrainConfig, load_params, stage_c_batches  # noqa: E402

# gate: name -> limit
GATES = {
    "rel_l2_T": 0.05,
    "rel_l2_q": 0.10,
    "superres_rel_l2_T": 0.10,
    "superres_rel_l2_q": 0.15,
    "mean_abs_r_energy": 0.5,
    "mean_abs_r_ldf": 0.5,
    "grad_rel_err": 0.05,
}


def _interp_configs(seed: int = 0) -> list[dict]:
    """Unseen operating points + unseen thickness for the trained families.

    Phase direction is preserved: desorption heats (t_f > t_init), adsorption
    cools. Thickness 3.5e-3 sits strictly between the trained 2e-3 / 5e-3.
    Perturbations are clamped into ``OP_ENVELOPE`` — the corpus hull — so the
    test stays *interpolation*, and scaled to the dense sweep (±(2–8) °C,
    p ×exp(±0.2)): with ~10 °C neighbor spacing the round-2 ranges (±15 °C)
    crossed trained points and broke phase directions. A final direction
    projection keeps the 8 °C phase margin the sweep itself enforces.
    """
    rng = np.random.default_rng(seed)
    out = []
    margin = 8.0
    for preset in preset_configs():
        cfg = dict(preset, L_m=3.5e-3, q_init_kg_kg=None,
                   name=f"{preset['name']}-gate")
        des = preset["t_f_c"] > preset["t_init_c"]
        if des:
            cfg["t_f_c"] = preset["t_f_c"] + float(rng.uniform(-2.0, 8.0))
            cfg["t_init_c"] = preset["t_init_c"] + float(rng.uniform(-2.0, 6.0))
        else:
            cfg["t_f_c"] = preset["t_f_c"] + float(rng.uniform(-6.0, 2.0))
            cfg["t_init_c"] = preset["t_init_c"] + float(rng.uniform(-2.0, 6.0))
        cfg["t_f_c"] = float(np.clip(cfg["t_f_c"], *OP_ENVELOPE["t_f_c"]))
        cfg["t_init_c"] = float(np.clip(cfg["t_init_c"], *OP_ENVELOPE["t_init_c"]))
        if des:
            cfg["t_f_c"] = max(cfg["t_f_c"], cfg["t_init_c"] + margin)
        else:
            cfg["t_init_c"] = max(cfg["t_init_c"], cfg["t_f_c"] + margin)
        cfg["p_pa"] = float(np.clip(
            preset["p_pa"] * np.exp(rng.uniform(-0.2, 0.2)), *OP_ENVELOPE["p_pa"]))
        out.append(cfg)
    return out


def _direction_presets() -> tuple[dict, dict]:
    """(desorption, adsorption) anchor presets selected by phase direction.

    Direction-based, not index-based: the round-1 gate indexed
    ``preset_configs()[1]`` for the adsorption proxy — but silica presets
    come first, so the "13X cools" test actually ran on a silica DESORPTION
    rollout and reported a false FAIL (both proxy values were correctly
    negative for desorption; 2026-11-15). Equilibrium-charge anchors only
    (``q_init_kg_kg is None``) keep the round-1 semantics.
    """
    presets = preset_configs()
    des = next(c for c in presets
               if c["t_f_c"] > c["t_init_c"] and c["q_init_kg_kg"] is None)
    ads = next(c for c in presets
               if c["t_f_c"] < c["t_init_c"] and c["q_init_kg_kg"] is None)
    return des, ads


def run_gate(ckpt: Path, *, n_real_rows: int = 8, seed: int = 0,
             n_steps: int = 400) -> dict:
    """Full gate pass. Returns the report dict (also JSON-serializable)."""
    params = load_params(ckpt)
    predict_fn = lambda a, b, c: predict_batch(params, jnp.asarray(a),  # noqa: E731
                                               jnp.asarray(b), c)

    # --- field interpolation on unseen operating points -------------------
    field = {}
    for cfg in _interp_configs(seed):
        fields = rollout_phase_fields(n_cells=8, dt_s=None, n_steps=n_steps,
                                      **rollout_kwargs(cfg))
        cond_d = config_to_cond(cfg, fields["t_phase_s"])
        cond_v = jnp.asarray(encode_condition(cond_d))
        n_cells, n_steps_ = fields["n_cells"], fields["n_steps"]
        xs = ((np.arange(n_cells) + 0.5) / n_cells).astype(np.float32)
        ts = np.linspace(0.0, 1.0, n_steps_ + 1).astype(np.float32)
        xg, tg = np.meshgrid(xs, ts)
        err = field_errors(predict_fn, xg.ravel(), tg.ravel(), cond_v,
                           fields["T"].ravel(), fields["q"].ravel(),
                           q_denom_floor=0.25 * cfg["q_sat_kg_kg"])
        field[cfg["name"]] = err

    # --- super-resolution: fine oracle rollout, coarse-trained surrogate ---
    preset = preset_configs()[0]
    fine = rollout_phase_fields(n_cells=16, dt_s=None, n_steps=800,
                                **rollout_kwargs(preset))
    cond_v = jnp.asarray(encode_condition(config_to_cond(preset, fine["t_phase_s"])))
    superres = super_resolution_error(predict_fn, fine, cond_v)

    # --- physics honesty on real rows (no labels exist) --------------------
    real_conds, real_report = load_real_conditions(max_rows=n_real_rows)
    cfg_small = TrainConfig(steps=0, steps_c=0, n_colloc=256, n_bc=64, n_ic=64)
    real_batches, sc_report = stage_c_batches(real_conds, cfg_small, seed=seed)
    res_rows = [residual_summary(params, b["x_colloc"], b["t_colloc"],
                                 b["cond"]) for b in real_batches]
    honesty = {"mean_abs_r_energy": float(np.mean(
                   [r["mean_abs_r_energy"] for r in res_rows])),
               "mean_abs_r_ldf": float(np.mean(
                   [r["mean_abs_r_ldf"] for r in res_rows])),
               "n_conditions": len(real_batches),
               "real_report": {k: v for k, v in real_report.items()
                               if k != "transport_defaults"}}

    # --- directional feedback (d final-T / d t_f) --------------------------
    probe = preset_configs()[0]
    cond_d = config_to_cond(probe, 120.0)
    cond_v = jnp.asarray(encode_condition(cond_d))

    def mean_final_T(cond):
        x = jnp.linspace(0.05, 0.95, 9)
        T, _ = predict_batch(params, x, jnp.ones_like(x), cond)
        return jnp.mean(T)

    grad = grad_agreement(mean_final_T, cond_v, wrt=10, eps=1e-2)

    # --- cooling-proxy sign (adsorption +, desorption −) -------------------
    def q_field(cfg):
        fields = rollout_phase_fields(n_cells=8, dt_s=None, n_steps=n_steps,
                                      **rollout_kwargs(cfg))
        cv = jnp.asarray(encode_condition(config_to_cond(cfg, fields["t_phase_s"])))
        xs = ((np.arange(fields["n_cells"]) + 0.5) / fields["n_cells"]).astype(np.float32)
        ts = np.linspace(0.0, 1.0, fields["n_steps"] + 1).astype(np.float32)
        xg, tg = np.meshgrid(xs, ts)
        _, q_pred = predict_fn(xg.ravel(), tg.ravel(), cv)
        return np.asarray(q_pred).reshape(fields["n_steps"] + 1, fields["n_cells"])

    h_fg = 2.45e6
    des_preset, ads_preset = _direction_presets()
    ads = cooling_proxy(q_field(ads_preset), h_fg_j_kg=h_fg)  # bed cools → q rises
    des = cooling_proxy(q_field(des_preset), h_fg_j_kg=h_fg)  # bed heats → q falls
    proxy = {"adsorption_preset": ads_preset["name"],
             "desorption_preset": des_preset["name"],
             "adsorption_delta_q": ads["delta_q_mean"],
             "desorption_delta_q": des["delta_q_mean"]}

    # --- gates --------------------------------------------------------------
    vals = {
        "rel_l2_T": max(v["rel_l2_T"] for v in field.values()),
        "rel_l2_q": max(v["rel_l2_q"] for v in field.values()),
        "superres_rel_l2_T": superres["superres_rel_l2_T"],
        "superres_rel_l2_q": superres["superres_rel_l2_q"],
        "mean_abs_r_energy": honesty["mean_abs_r_energy"],
        "mean_abs_r_ldf": honesty["mean_abs_r_ldf"],
        "grad_rel_err": grad["rel_err"],
    }
    gates = {k: {"value": round(v, 6), "limit": lim, "pass": bool(v <= lim)}
             for k, (v, lim) in ((k, (vals[k], GATES[k])) for k in GATES)}
    gates["cooling_sign"] = {
        "value": [proxy["adsorption_delta_q"], proxy["desorption_delta_q"]],
        "limit": "ads > 0, des < 0",
        "pass": bool(proxy["adsorption_delta_q"] > 0.0
                     and proxy["desorption_delta_q"] < 0.0)}
    report = {
        "ckpt": str(ckpt), "created_unix": time.time(),
        "scope": ("single-phase interpolation on trained families at unseen "
                  "operating points/thickness; real-condition FIELD accuracy "
                  "is not claimed (no labels) — residual honesty only; q "
                  "relative error floored at 0.25·q_sat per condition "
                  "(well-posedness on near-dry beds)"),
        "field_interp": field, "superres": superres,
        "residual_honesty": honesty, "grad_agreement": grad,
        "cooling_proxy": proxy, "gates": gates,
        "overall_pass": bool(all(g["pass"] for g in gates.values())),
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="PINN N1 acceptance gate.")
    ap.add_argument("--ckpt", type=Path, default=Path("data_cache/pinn/bed_pinn.npz"))
    ap.add_argument("--out", type=Path, default=Path("data_cache/pinn/gate_report.json"))
    ap.add_argument("--n-real-rows", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    report = run_gate(args.ckpt, n_real_rows=args.n_real_rows, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print(f"\nPINN gate — {args.ckpt}")
    print(f"{'gate':<22}{'value':>12}{'limit':>10}  pass")
    for k, g in report["gates"].items():
        v = g["value"]
        vs = (f"[{v[0]:+.4f}, {v[1]:+.4f}]" if isinstance(v, list)
              else f"{v:12.4f}")
        print(f"{k:<22}{vs:>12}{str(g['limit']):>10}  "
              f"{'PASS' if g['pass'] else 'FAIL'}")
    print(f"\noverall: {'PASS' if report['overall_pass'] else 'FAIL'}"
          f"  (report: {args.out})")
    if not report["overall_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

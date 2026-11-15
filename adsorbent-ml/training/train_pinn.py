"""PINN training loop: synthetic pretrain + physics-ramp curriculum.

Curriculum (cheap → expensive, AU-style):

- Phase A (data warm-up): ``w_pde = 0`` — the net learns the trajectory
  manifold from synthetic solver fields (milliseconds per label).
- Phase B (physics ramp): ``w_pde`` ramps 0 → target — PDE/BC/IC residuals
  take over where data is sparse; the governing equations become the signal
  (the AU self-improvement loop in miniature: known laws score proposals).
- Phase C (real-condition hardening): collocation on ``load_real_conditions``
  rows (fitted/anchor physics parameters WITHOUT field labels — pure
  physics loss + IC/BC). Data loss stays on synthetic.

Full-batch Adam with cosine decay; the step is ``jit``-ed. Checkpoints are
plain ``.npz`` (no new dependencies). CPU now, GPU unchanged with CUDA jaxlib.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))  # harness (solver oracle) + data builders
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from bed_pinn import (COND_SPEC, Q_MAX_KG_KG, T_HALF_K, T_MID_K, boundary_loss,
                      encode_condition, ic_residuals, init_params,
                      initial_loss, physics_loss, total_loss)
from corpus import OP_ENVELOPE


@dataclass
class TrainConfig:
    width: int = 64
    depth: int = 3
    steps: int = 2000
    lr: float = 3e-3
    w_data: float = 1.0
    w_pde: float = 1.0
    w_bc: float = 1.0
    w_ic: float = 1.0
    ramp_steps: int = 500  # phase-B physics ramp length
    n_colloc: int = 512
    n_bc: int = 128
    n_ic: int = 128
    seed: int = 0
    # Stage C (real-condition hardening): paired steps — one synthetic batch
    # (data signal stays on) + one real-row physics-only batch per iteration.
    steps_c: int = 500
    lr_c: float = 3e-4
    samples_per_row: int = 2
    # Operating-point envelope for real rows: the same hull the synthetic
    # sweep (corpus.OP_ENVELOPE) trains, so stage C physics and stage A/B
    # data see one consistent condition manifold. Inside COND_SPEC bounds;
    # spans the application profiles' fluid bands and condenser/evaporator
    # pressures. t_init reaches 90 °C because the adsorption phase starts
    # from a freshly regenerated (hot) bed.
    op_t_f_c: tuple[float, float] = OP_ENVELOPE["t_f_c"]
    op_p_pa: tuple[float, float] = OP_ENVELOPE["p_pa"]
    op_t_init_c: tuple[float, float] = OP_ENVELOPE["t_init_c"]
    op_q_init_frac: tuple[float, float] = (0.05, 1.0)
    op_t_phase_s: tuple[float, float] = (60.0, 600.0)
    # Thickness sampled within the synthetic training envelope. The rows'
    # L_m=2e-4 placeholder sits ~10× below the corpus slabs (2–5 mm) — out
    # there the conduction residual is unreachable and stage C degrades the
    # net (seen 2026-11-15: real_pde ~8e4, final total 3.4 vs 1.4e-3).
    op_L_m: tuple[float, float] = (2e-3, 5e-3)


def save_params(path: Path, params: dict) -> None:
    np.savez(path, **{k: np.asarray(v) for k, v in params.items()})


def load_params(path: Path) -> dict:
    z = np.load(path)
    return {k: jnp.asarray(z[k]) for k in z.files}


def _to_normalized_targets(data: dict) -> tuple[jax.Array, jax.Array]:
    uT = (jnp.asarray(data["T"]) - T_MID_K) / T_HALF_K
    uq = jnp.asarray(data["q"]) / Q_MAX_KG_KG
    return uT.astype(jnp.float32), uq.astype(jnp.float32)


def _physics_total_loss(params, batch, weights):
    """Physics-only objective (no data term): PDE + BC + IC on one condition.

    Real-condition rows carry NO field labels, so stage C must never touch
    ``data_loss`` — an empty-array data term would mean-reduce to NaN.
    """
    phys = physics_loss(params, batch["x_colloc"], batch["t_colloc"],
                        batch["cond"])
    l_bc = boundary_loss(params, batch["t_bc"], batch["cond"])
    l_ic = initial_loss(params, batch["x_ic"], batch["cond"])
    total = (weights["pde"] * phys["total"] + weights["bc"] * l_bc
             + weights["ic"] * l_ic)
    aux = {"pde": phys["total"], "pde_energy": phys["energy"],
           "pde_ldf": phys["ldf"], "bc": l_bc, "ic": l_ic, "total": total}
    return total, aux


def make_batches(data: dict, cond_vecs: np.ndarray, cfg: TrainConfig,
                 seed: int) -> list[dict]:
    """One batch per config: its data points + fresh collocation/BC/IC sets."""
    rng = np.random.default_rng(seed)
    x = np.asarray(data["x_hat"], dtype=np.float32)
    t = np.asarray(data["t_hat"], dtype=np.float32)
    uT_all, uq_all = _to_normalized_targets(data)
    uT_all = np.asarray(uT_all, dtype=np.float32)
    uq_all = np.asarray(uq_all, dtype=np.float32)
    idx = np.asarray(data["cond_idx"])
    batches = []
    for ci in range(cond_vecs.shape[0]):
        m = idx == ci
        batches.append({
            "cond": jnp.asarray(cond_vecs[ci]),
            "x_data": jnp.asarray(x[m]), "t_data": jnp.asarray(t[m]),
            "uT_data": jnp.asarray(uT_all[m]), "uq_data": jnp.asarray(uq_all[m]),
            "x_colloc": jnp.asarray(rng.uniform(0, 1, cfg.n_colloc).astype(np.float32)),
            "t_colloc": jnp.asarray(rng.uniform(0, 1, cfg.n_colloc).astype(np.float32)),
            "t_bc": jnp.asarray(rng.uniform(0, 1, cfg.n_bc).astype(np.float32)),
            "x_ic": jnp.asarray(rng.uniform(0, 1, cfg.n_ic).astype(np.float32)),
        })
    return batches


def stage_c_batches(real_conds: list[dict], cfg: TrainConfig,
                    seed: int) -> tuple[list[dict], dict]:
    """Physics-only batches from real rows at sampled operating points.

    Each real row (real D–A parameters + Q_st, honest transport defaults) is
    re-sampled ``samples_per_row`` times across the operating envelope: the
    material identity is real, the operating point is synthetic-but-realistic
    (documented ranges covering the four application profiles). Field labels
    do not exist for these rows — the batches carry collocation/BC/IC sets
    only. Condition values outside the COND_SPEC training envelope are
    clipped (counted in the report, never silently).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
    from corpus import load_real_conditions  # noqa: PLC0415

    if not real_conds:
        real_conds, report = load_real_conditions()
    else:
        report = {"n_conditions": len(real_conds), "provenance": "caller-supplied"}
    rng = np.random.default_rng(seed)
    batches, n_clipped, n_L_overridden = [], 0, 0
    for row in real_conds:
        for _ in range(cfg.samples_per_row):
            cond = dict(row)
            if "L_m" in cond and not (cfg.op_L_m[0] <= cond["L_m"] <= cfg.op_L_m[1]):
                n_L_overridden += 1
            cond["L_m"] = float(np.exp(rng.uniform(
                np.log(cfg.op_L_m[0]), np.log(cfg.op_L_m[1]))))
            cond["t_f_c"] = rng.uniform(*cfg.op_t_f_c)
            cond["p_pa"] = float(np.exp(rng.uniform(
                np.log(cfg.op_p_pa[0]), np.log(cfg.op_p_pa[1]))))
            cond["t_init_c"] = rng.uniform(*cfg.op_t_init_c)
            cond["t_phase_s"] = float(np.exp(rng.uniform(
                np.log(cfg.op_t_phase_s[0]), np.log(cfg.op_t_phase_s[1]))))
            # Initial charge: absolute fraction of q_sat, dry (regenerated)
            # through saturated — both adsorption and desorption transients.
            cond["q_init_frac"] = rng.uniform(*cfg.op_q_init_frac)
            # Clip into the COND_SPEC envelope (honesty: counted, not silent).
            for key, lo, hi, _log in COND_SPEC:
                if key in cond and cond[key] is not None:
                    v = float(np.clip(float(cond[key]), lo, hi))
                    if abs(v - float(cond[key])) > 1e-12:
                        n_clipped += 1
                    cond[key] = v
            batches.append({
                "cond": jnp.asarray(encode_condition(cond)),
                "x_colloc": jnp.asarray(rng.uniform(0, 1, cfg.n_colloc).astype(np.float32)),
                "t_colloc": jnp.asarray(rng.uniform(0, 1, cfg.n_colloc).astype(np.float32)),
                "t_bc": jnp.asarray(rng.uniform(0, 1, cfg.n_bc).astype(np.float32)),
                "x_ic": jnp.asarray(rng.uniform(0, 1, cfg.n_ic).astype(np.float32)),
            })
    report = dict(report, n_batches=len(batches), n_clipped_values=n_clipped,
                  n_L_overridden=n_L_overridden,
                  op_envelope={"t_f_c": cfg.op_t_f_c, "p_pa": cfg.op_p_pa,
                               "t_init_c": cfg.op_t_init_c,
                               "q_init_frac": cfg.op_q_init_frac,
                               "t_phase_s": cfg.op_t_phase_s,
                               "L_m": cfg.op_L_m})
    return batches, report


def train(batches: list[dict], cfg: TrainConfig, *, log_every: int = 200,
          real_batches: list[dict] | None = None
          ) -> tuple[dict, list[dict]]:
    """Full training run (A→B ramp, then optional stage C). Returns ``(params, history)``.

    Stage C pairs each iteration: one synthetic batch (data signal stays on —
    the docstring contract) + one real-condition physics-only batch. History
    entries carry ``stage`` ("AB" | "C").
    """
    key = jax.random.PRNGKey(cfg.seed)
    params = init_params(key, width=cfg.width, depth=cfg.depth)
    schedule = optax.cosine_decay_schedule(cfg.lr, cfg.steps, alpha=0.05)
    opt = optax.adam(schedule)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, batch, w_pde):
        weights = {"data": cfg.w_data, "pde": w_pde,
                   "bc": cfg.w_bc, "ic": cfg.w_ic}
        (loss, aux), grads = jax.value_and_grad(total_loss, has_aux=True)(
            params, batch, weights)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, aux

    @jax.jit
    def step_real(params, opt_state, batch, w_pde):
        weights = {"pde": w_pde, "bc": cfg.w_bc, "ic": cfg.w_ic}
        (loss, aux), grads = jax.value_and_grad(_physics_total_loss, has_aux=True)(
            params, batch, weights)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, aux

    history: list[dict] = []
    for it in range(cfg.steps):
        ramp = min(1.0, it / max(1, cfg.ramp_steps))
        w_pde = cfg.w_pde * ramp  # phase A→B: physics fades in
        agg: dict[str, float] = {}
        for batch in batches:
            params, opt_state, aux = step(params, opt_state, batch,
                                          jnp.asarray(w_pde, dtype=jnp.float32))
            for k, v in aux.items():
                agg[k] = agg.get(k, 0.0) + float(v) / len(batches)
        agg["step"] = it
        agg["stage"] = "AB"
        agg["w_pde"] = float(w_pde)
        history.append(agg)
        if log_every and (it % log_every == 0 or it == cfg.steps - 1):
            print(f"[{it:5d}/{cfg.steps}] total={agg['total']:.3e} "
                  f"data={agg['data']:.3e} pde={agg['pde']:.3e} "
                  f"bc={agg['bc']:.3e} ic={agg['ic']:.3e}", flush=True)

    # ---- Stage C: real-condition hardening (physics-only on real rows) ----
    if real_batches and cfg.steps_c > 0:
        schedule_c = optax.cosine_decay_schedule(cfg.lr_c, cfg.steps_c, alpha=0.05)
        opt_c = optax.adam(schedule_c)
        opt_state_c = opt_c.init(params)

        @jax.jit
        def step_c_synth(params, opt_state, batch, w_pde):
            weights = {"data": cfg.w_data, "pde": w_pde,
                       "bc": cfg.w_bc, "ic": cfg.w_ic}
            (loss, aux), grads = jax.value_and_grad(total_loss, has_aux=True)(
                params, batch, weights)
            updates, opt_state = opt_c.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, aux

        @jax.jit
        def step_c_real(params, opt_state, batch, w_pde):
            weights = {"pde": w_pde, "bc": cfg.w_bc, "ic": cfg.w_ic}
            (loss, aux), grads = jax.value_and_grad(_physics_total_loss, has_aux=True)(
                params, batch, weights)
            updates, opt_state = opt_c.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, aux

        for it in range(cfg.steps_c):
            w_pde = jnp.asarray(cfg.w_pde, dtype=jnp.float32)
            agg = {}
            synth = batches[it % len(batches)]
            params, opt_state_c, aux_s = step_c_synth(params, opt_state_c, synth, w_pde)
            for k, v in aux_s.items():
                agg[k] = float(v)
            real = real_batches[it % len(real_batches)]
            params, opt_state_c, aux_r = step_c_real(params, opt_state_c, real, w_pde)
            for k, v in aux_r.items():
                agg[f"real_{k}"] = float(v)
            agg["step"] = cfg.steps + it
            agg["stage"] = "C"
            agg["w_pde"] = float(w_pde)
            history.append(agg)
            if log_every and (it % log_every == 0 or it == cfg.steps_c - 1):
                print(f"[C {it:4d}/{cfg.steps_c}] total={agg['total']:.3e} "
                      f"real_pde={agg['real_pde']:.3e} real_bc={agg['real_bc']:.3e} "
                      f"real_ic={agg['real_ic']:.3e}", flush=True)
    return params, history


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the Bed1D PINN (thin CLI).")
    ap.add_argument("--out", type=Path, default=Path("data_cache/pinn/bed_pinn.npz"))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--steps-c", type=int, default=500,
                    help="stage-C iterations (real-condition hardening); 0 disables")
    ap.add_argument("--max-real-rows", type=int, default=64,
                    help="cap on real-condition rows loaded for stage C")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--n-data", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import sys  # noqa: PLC0415

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from corpus import build_corpus, load_real_conditions, preset_configs  # noqa: PLC0415

    t0 = time.time()
    data, manifest = build_corpus(preset_configs(), n_data_per_config=args.n_data,
                                  seed=args.seed)
    real_conds, real_report = load_real_conditions(max_rows=args.max_real_rows)
    cfg = TrainConfig(width=args.width, depth=args.depth, steps=args.steps,
                      steps_c=args.steps_c, seed=args.seed)
    batches = make_batches(data, np.asarray(data["conds"]), cfg, seed=args.seed + 1)
    real_batches, sc_report = stage_c_batches(real_conds, cfg, seed=args.seed + 2)
    print(f"stage C: {len(real_batches)} physics batches from "
          f"{real_report['n_conditions']} real rows "
          f"({sc_report['n_clipped_values']} values clipped to the envelope)")
    params, history = train(batches, cfg, real_batches=real_batches)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_params(args.out, params)
    stages = {h.get("stage") for h in history}
    (args.out.parent / "history.json").write_text(json.dumps(
        {"config": asdict(cfg), "manifest": {k: v for k, v in manifest.items()
                                             if k != "cond_dicts"},
         "real_conditions": real_report, "stage_c": sc_report,
         "stages": sorted(stages),
         "seconds": time.time() - t0, "final": history[-1]}))
    print(f"saved {args.out}  stages={sorted(stages)}  "
          f"final total={history[-1]['total']:.3e}")


if __name__ == "__main__":
    main()

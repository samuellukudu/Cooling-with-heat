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
from bed_pinn import (Q_MAX_KG_KG, T_HALF_K, T_MID_K, init_params,
                      total_loss)


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


def save_params(path: Path, params: dict) -> None:
    np.savez(path, **{k: np.asarray(v) for k, v in params.items()})


def load_params(path: Path) -> dict:
    z = np.load(path)
    return {k: jnp.asarray(z[k]) for k in z.files}


def _to_normalized_targets(data: dict) -> tuple[jax.Array, jax.Array]:
    uT = (jnp.asarray(data["T"]) - T_MID_K) / T_HALF_K
    uq = jnp.asarray(data["q"]) / Q_MAX_KG_KG
    return uT.astype(jnp.float32), uq.astype(jnp.float32)


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


def train(batches: list[dict], cfg: TrainConfig, *, log_every: int = 200,
          psat_fn=None) -> tuple[dict, list[dict]]:
    """Full training run. Returns ``(params, history)``."""
    if psat_fn is None:
        from harness.physics.thermo import water_sat_pressure_pa  # noqa: PLC0415

        psat_fn = water_sat_pressure_pa
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
            params, batch, weights, psat_fn)
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
        agg["w_pde"] = float(w_pde)
        history.append(agg)
        if log_every and (it % log_every == 0 or it == cfg.steps - 1):
            print(f"[{it:5d}/{cfg.steps}] total={agg['total']:.3e} "
                  f"data={agg['data']:.3e} pde={agg['pde']:.3e} "
                  f"bc={agg['bc']:.3e} ic={agg['ic']:.3e}", flush=True)
    return params, history


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the Bed1D PINN (thin CLI).")
    ap.add_argument("--out", type=Path, default=Path("data_cache/pinn/bed_pinn.npz"))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--n-data", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import sys  # noqa: PLC0415

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from corpus import build_corpus, preset_configs  # noqa: PLC0415

    t0 = time.time()
    data, manifest = build_corpus(preset_configs(), n_data_per_config=args.n_data,
                                  seed=args.seed)
    cfg = TrainConfig(width=args.width, depth=args.depth, steps=args.steps,
                      seed=args.seed)
    batches = make_batches(data, np.asarray(data["conds"]), cfg, seed=args.seed + 1)
    params, history = train(batches, cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_params(args.out, params)
    (args.out.parent / "history.json").write_text(json.dumps(
        {"config": asdict(cfg), "manifest": {k: v for k, v in manifest.items()
                                             if k != "cond_dicts"},
         "seconds": time.time() - t0, "final": history[-1]}))
    print(f"saved {args.out}  final total={history[-1]['total']:.3e}")


if __name__ == "__main__":
    main()

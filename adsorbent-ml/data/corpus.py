"""Unified training corpus: synthetic phase fields + real-label conditions.

Two tiers (labels are the bottleneck, not models — ROADMAP data strategy):

- **SYNTHETIC (infinite, cheap)**: single-phase ``T(x,t)`` / ``q(x,t)`` fields
  rolled out with the trusted solver (``harness.physics.bed1d.step_bed`` —
  exact-exponential LDF + RK4 conduction, the V2-gated numerics). This is the
  curriculum base: pretrain the PINN here, where labels cost milliseconds.
- **REAL (scarce, decision-relevant)**: ISODB-derived D–A fits
  (``data_cache/fits/da_params.csv``) + the 13 commercial anchors. These do
  not carry ``T(x,t)`` fields, so they feed N1 as *conditioning rows* (physics
  parameters to simulate + physics-residual collocation) and N2
  (structure→property) as training labels. Never reported as field accuracy.

``diffheat`` conduction trajectories are the natural *earlier* curriculum rung
(pure ``∂T/∂t = α∇²T`` before adsorption coupling); the schema below accepts
them unchanged — a future exporter only.

Every builder returns plain numpy arrays + a manifest dict (counts, bounds,
provenance, timestamp). Splits are by family/company, never random
(ROADMAP: random splits leak near-duplicate frameworks).
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
FITS_CSV = REPO_ROOT / "data_cache" / "fits" / "da_params.csv"

# Class-default transport where fits carry none (flagged, never silent —
# harness/DESIGN.md §8.1 honesty rule; closes via the per-material k_eff head).
TRANSPORT_DEFAULTS = {
    "rho_s_kg_m3": 800.0,
    "c_s_j_kg_k": 1000.0,
    "k_eff_w_m_k": 0.2,
    "k_ldf_s_1": 2.5e-4,
}


# Keys in a preset/condition config that are metadata, not solver inputs.
_META_KEYS = frozenset({"name", "family", "transport_provenance"})


def rollout_kwargs(config: dict) -> dict:
    """Strip metadata keys so a config dict can splat into rollout fns."""
    return {k: v for k, v in config.items() if k not in _META_KEYS}


def rollout_phase_fields(*, q_sat_kg_kg: float, q_st_j_kg: float,
                         e_char_j_mol: float, n_da: float, k_ldf_s_1: float,
                         rho_s_kg_m3: float, c_s_j_kg_k: float,
                         c_pl_j_kg_k: float, k_eff_w_m_k: float,
                         h_wall_w_m2_k: float, L_m: float, n_cells: int,
                         t_f_c: float, p_pa: float, t_init_c: float,
                         q_init_kg_kg: float | None, dt_s: float | None,
                         n_steps: int) -> dict[str, np.ndarray]:
    """One single-phase rollout with FIXED fluid T and vapour P.

    Uses ``step_bed`` directly (no valve logic — the PINN scope boundary).
    ``dt_s=None`` auto-derives the step from the conduction CFL bound with
    the dry capacity — the exact ``Bed1D`` env convention
    (``harness/envs/bed1d.py``: ``dt = dx²·ρ·c_s/(2·k_eff)``, 0.9 margined).
    Returns ``T`` / ``q`` histories shaped ``(n_steps+1, n_cells)`` plus the
    grid metadata needed to form normalized query points.
    """
    from harness.physics.bed1d import check_timestep, step_bed
    from harness.physics.thermo import da_uptake

    dx = L_m / n_cells
    if dt_s is None:
        dt_s = 0.9 * dx * dx * rho_s_kg_m3 * c_s_j_kg_k / (2.0 * k_eff_w_m_k)
    rho_cp = rho_s_kg_m3 * (c_s_j_kg_k + c_pl_j_kg_k * q_sat_kg_kg * 0.5)
    if not check_timestep(L_m, n_cells, k_eff_w_m_k, rho_cp, dt_s):
        raise ValueError(
            f"dt_s={dt_s:g} violates the conduction CFL bound for "
            f"L={L_m:g}, n_cells={n_cells}, k_eff={k_eff_w_m_k:g} "
            "(see diffheat.check_cfl / bed1d.check_timestep)")

    T = jnp.full((n_cells,), t_init_c + 273.15)
    q = (jnp.full((n_cells,), q_init_kg_kg) if q_init_kg_kg is not None
         else da_uptake(T, p_pa, q_sat_kg_kg, e_char_j_mol, n_da))
    t_f_k = t_f_c + 273.15
    kw = dict(dx=dx, q_sat_kg_kg=q_sat_kg_kg, q_st_j_kg=q_st_j_kg,
              e_char_j_mol=e_char_j_mol, n_da=n_da, k_ldf_s_1=k_ldf_s_1,
              rho_s_kg_m3=rho_s_kg_m3, c_s_j_kg_k=c_s_j_kg_k,
              c_pl_j_kg_k=c_pl_j_kg_k, k_eff_w_m_k=k_eff_w_m_k,
              h_wall_w_m2_k=h_wall_w_m2_k)

    def body(carry, _):
        T_c, q_c = carry
        T_n, q_n, _ = step_bed(T_c, q_c, t_f_k, p_pa, dt_s, **kw)
        return (T_n, q_n), (T_n, q_n)

    (_, _), (T_hist, q_hist) = jax.lax.scan(body, (T, q), None, length=n_steps)
    T_hist = np.asarray(jnp.concatenate([T[None, :], T_hist]))
    q_hist = np.asarray(jnp.concatenate([q[None, :], q_hist]))
    return {"T": T_hist, "q": q_hist, "dx": dx, "dt_s": dt_s,
            "n_steps": n_steps, "n_cells": n_cells,
            "t_phase_s": dt_s * n_steps, "L_m": L_m}


def sample_query_points(rng: np.random.Generator, fields: dict[str, np.ndarray],
                        n_points: int) -> dict[str, np.ndarray]:
    """Uniform ``(x_hat, t_hat)`` samples with ground-truth ``(T, q)``."""
    n_steps = fields["n_steps"]
    n_cells = fields["n_cells"]
    x_hat = rng.uniform(0.0, 1.0, size=n_points).astype(np.float32)
    t_hat = rng.uniform(0.0, 1.0, size=n_points).astype(np.float32)
    ix = np.clip((x_hat * n_cells).astype(int), 0, n_cells - 1)
    it = np.clip((t_hat * n_steps).astype(int), 0, n_steps)
    return {"x_hat": x_hat, "t_hat": t_hat,
            "T": fields["T"][it, ix].astype(np.float32),
            "q": fields["q"][it, ix].astype(np.float32)}


def config_to_cond(config: dict[str, float], t_phase_s: float) -> dict[str, float]:
    """Rollout config -> PINN condition dict (adds phase horizon + q fraction)."""
    cond = dict(config)
    cond["t_phase_s"] = t_phase_s
    q_sat = config["q_sat_kg_kg"]
    q_init = config.get("q_init_kg_kg")
    if q_init is None:
        from harness.physics.thermo import da_uptake  # noqa: PLC0415

        q_init = float(da_uptake(jnp.asarray(config["t_init_c"] + 273.15),
                                 jnp.asarray(config["p_pa"]), q_sat,
                                 config["e_char_j_mol"], config["n_da"]))
    cond["q_init_frac"] = float(np.clip(q_init / q_sat, 0.0, 1.2))
    return cond


def build_corpus(configs: list[dict], *, n_data_per_config: int, seed: int = 0,
                 n_cells: int = 8, dt_s: float | None = None,
                 n_steps: int = 400) -> tuple[dict[str, np.ndarray], dict]:
    """Roll out ``configs`` and sample query points. Returns ``(data, manifest)``.

    ``data`` stacks all configs: ``x_hat, t_hat, T, q, cond_idx`` + the
    ``conds`` matrix (``n_configs × COND_DIM``, encoded ``[-1,1]``) and the
    raw ``cond_dicts`` for decoding. One config = one broad-model example.
    """
    import sys  # noqa: PLC0415

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
    from bed_pinn import encode_condition  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    xs, ts, Ts, qs, idx = [], [], [], [], []
    cond_vecs, cond_dicts, families = [], [], []
    for ci, cfg in enumerate(configs):
        fields = rollout_phase_fields(n_cells=n_cells, dt_s=dt_s,
                                      n_steps=n_steps, **rollout_kwargs(cfg))
        pts = sample_query_points(rng, fields, n_data_per_config)
        xs.append(pts["x_hat"])
        ts.append(pts["t_hat"])
        Ts.append(pts["T"])
        qs.append(pts["q"])
        idx.append(np.full(n_data_per_config, ci, dtype=np.int32))
        cond = config_to_cond(cfg, fields["t_phase_s"])
        cond_dicts.append(cond)
        cond_vecs.append(np.asarray(encode_condition(cond), dtype=np.float32))
        families.append(cfg.get("family", cfg.get("name", f"config-{ci}")))
    data = {"x_hat": np.concatenate(xs), "t_hat": np.concatenate(ts),
            "T": np.concatenate(Ts), "q": np.concatenate(qs),
            "cond_idx": np.concatenate(idx),
            "conds": np.stack(cond_vecs), "families": np.array(families)}
    manifest = {"n_configs": len(configs), "n_points": int(data["x_hat"].size),
                "n_data_per_config": n_data_per_config, "n_cells": n_cells,
                "dt_s": dt_s, "n_steps": n_steps, "seed": seed,
                "provenance": "harness.physics.bed1d.step_bed single-phase rollouts",
                "created_unix": time.time(), "cond_dicts": cond_dicts}
    return data, manifest


def preset_configs() -> list[dict]:
    """Small anchor-based sweep: 2 materials × 2 thicknesses × 1 phase each.

    Geometries are mm-scale so the CFL-derived dt keeps rollouts tractable;
    kinetics (τ ~ 100 s) and phase horizons (~40–100 s) cover the coupled
    thermal + uptake transient the PINN must learn. The desorption case heats
    a saturated bed; the adsorption case cools a hot bed (strong transients
    in opposite directions).
    """
    base = dict(e_char_j_mol=4500.0, n_da=1.8, k_ldf_s_1=1e-2,
                rho_s_kg_m3=600.0, c_s_j_kg_k=1000.0, c_pl_j_kg_k=4184.0,
                k_eff_w_m_k=0.2, h_wall_w_m2_k=200.0, q_init_kg_kg=None)
    silica_des = dict(base, name="silica-RD-des", family="silica",
                      q_sat_kg_kg=0.35, q_st_j_kg=2.5e6,
                      t_init_c=30.0, t_f_c=80.0, p_pa=7380.0)
    x13_ads = dict(base, name="13X-ads", family="zeolite",
                   q_sat_kg_kg=0.28, q_st_j_kg=3.5e6, e_char_j_mol=14000.0,
                   t_init_c=80.0, t_f_c=30.0, p_pa=1700.0)
    out = []
    for cfg in (silica_des, x13_ads):
        for L_m in (2e-3, 5e-3):
            c = dict(cfg, L_m=L_m, name=f"{cfg['name']}-L{L_m:g}")
            out.append(c)
    return out


def load_real_conditions(*, fits_csv: Path = FITS_CSV,
                         max_rows: int | None = None) -> tuple[list[dict], dict]:
    """Fitted + anchor rows -> PINN condition dicts (with transport provenance).

    Honesty: rows missing ``q_st`` (single-T isotherms) are SKIPPED for N1
    conditioning — the energy equation needs ``Q_st``; imputing it would
    launder a guess into a physics label. They remain valid N2 targets.
    """
    from harness.materials import load_anchors, load_materials_csv  # noqa: PLC0415

    skipped_no_qst = 0
    conds: list[dict] = []
    for m in list(load_anchors()) + load_materials_csv(fits_csv):
        if m.q_st_j_kg is None:
            skipped_no_qst += 1
            continue
        rho = m.rho_kg_m3 if m.rho_kg_m3 is not None else TRANSPORT_DEFAULTS["rho_s_kg_m3"]
        c_s = m.cp_j_kg_k if m.cp_j_kg_k is not None else TRANSPORT_DEFAULTS["c_s_j_kg_k"]
        k_eff = (m.k_eff_w_m_k if m.k_eff_w_m_k is not None
                 else TRANSPORT_DEFAULTS["k_eff_w_m_k"])
        conds.append({
            "name": m.name, "family": m.material_class or m.source,
            "q_sat_kg_kg": m.q_sat_kg_kg, "q_st_j_kg": m.q_st_j_kg,
            "e_char_j_mol": m.e_char_j_mol, "n_da": m.n_da,
            "k_ldf_s_1": m.k_ldf_s_1 or TRANSPORT_DEFAULTS["k_ldf_s_1"],
            "rho_s_kg_m3": rho, "c_s_j_kg_k": c_s, "c_pl_j_kg_k": 4184.0,
            "k_eff_w_m_k": k_eff, "h_wall_w_m2_k": 200.0,
            "L_m": 2e-4, "t_f_c": 80.0, "t_init_c": 30.0,
            "p_pa": 7380.0, "q_init_kg_kg": None,
            "transport_provenance": m.transport_provenance or "default",
        })
        if max_rows is not None and len(conds) >= max_rows:
            break
    report = {"n_conditions": len(conds),
              "skipped_single_T_no_qst": skipped_no_qst,
              "transport_defaults": TRANSPORT_DEFAULTS,
              "provenance": f"anchors + {fits_csv}"}
    return conds, report


def family_split(families: np.ndarray, *, val_fraction: float = 0.25,
                 seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Split whole families into train/val (no family leaks across the split)."""
    rng = np.random.default_rng(seed)
    uniq = sorted(set(families.tolist()))
    n_val = max(1, int(round(len(uniq) * val_fraction)))
    val_fams = set(rng.choice(uniq, size=n_val, replace=False).tolist())
    is_val = np.array([f in val_fams for f in families])
    return ~is_val, is_val


__all__ = ["TRANSPORT_DEFAULTS", "build_corpus", "config_to_cond", "family_split",
           "load_real_conditions", "preset_configs", "rollout_kwargs",
           "rollout_phase_fields", "sample_query_points"]

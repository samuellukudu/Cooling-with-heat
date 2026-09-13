"""Bed1D physics-informed neural operator (single-phase, coordinate-based).

AU-standard mapping (https://acceleratedunderstanding.com/):

- **Direct 4D, one-shot**: the net maps ``(x_hat, t_hat, cond) -> (T, q)``
  for any query point in the phase slab. A full trajectory is one ``vmap``,
  never an autoregressive rollout — errors cannot compound step to step.
- **Resolution-invariant**: inputs are continuous coordinates, not grid
  vectors. Train on coarse samples, query at any resolution
  (super-resolution is tested, not claimed).
- **Broad, not narrow**: ONE weight set conditioned on material / transport /
  geometry / boundary / initial state. Narrow per-material nets are the
  baseline this must beat, not the design.
- **Directional feedback**: every output is differentiable w.r.t. every
  conditioning input via ``jax.grad`` — the surrogate itself gives the
  improvement direction (``dObjective/dDesign``), verified against finite
  differences in the same way harness V5 verifies solver gradients.
- **Physics as signal, not just data**: bed-energy + LDF residuals, wall/far
  boundary conditions and the initial state are first-class loss terms, so
  training is not bottlenecked on expensive labels.

Scope (deliberate): a SINGLE adsorption/desorption phase with fixed fluid
temperature and vapour pressure. Valve flips are discrete events whose
gradients are blind (harness Open Question 3 decision) — they stay in the
trusted solver. A full episode composes one-shot phase predictions with
harness switching logic, exactly like the ``search``-on-switch-times recipe.

Physics (harness/DESIGN.md §4.2, ``harness.physics.bed1d``)::

    cap(T,q) * dT/dt = k_eff * d2T/dx2 + rho_s * dq/dt * Q_st
    dq/dt = k_LDF * (q*(T,P) - q),   q* = Dubinin-Astakhov
    wall (x=0):  -k_eff * dT/dx = h * (T_f - T)
    far  (x=L):  dT/dx = 0
    IC: T(x,0) = T_init, q(x,0) = q_init

Pure JAX + optax (Adam lives in the training module). No equinox/flax
dependency: the parameter tree is a plain dict, the whole model is
``jit``/``vmap``/``grad`` compatible, and it runs on CPU now / GPU
unchanged once a CUDA jaxlib is installed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Normalization bounds (module constants = the contract)
# ---------------------------------------------------------------------------

T_MID_K = 360.0
T_HALF_K = 90.0  # network output covers [270, 450] K
Q_MAX_KG_KG = 1.5  # network q output covers [0, 1.5] kg/kg
C_PL_J_KG_K = 4184.0  # adsorbed-phase specific heat (matches thermo.CP_LIQUID)

# Conditioning vector. (key, lo, hi, log-scale). Decoded values are physical;
# the network always sees [-1, 1]. Log-scale follows the GeoField lesson for
# quantities spanning decades (ROADMAP.md reference architecture).
COND_SPEC: tuple[tuple[str, float, float, bool], ...] = (
    ("q_sat_kg_kg", 0.05, 1.5, True),
    ("q_st_j_kg", 0.8e6, 4.0e6, False),
    ("e_char_j_mol", 2000.0, 16000.0, True),
    ("n_da", 1.0, 4.0, False),
    ("k_ldf_s_1", 1e-4, 1e-2, True),
    ("rho_s_kg_m3", 200.0, 2000.0, False),
    ("c_s_j_kg_k", 500.0, 1500.0, False),
    ("k_eff_w_m_k", 0.05, 1.0, True),
    ("h_wall_w_m2_k", 10.0, 2000.0, True),
    ("L_m", 1e-4, 5e-3, True),
    ("t_f_c", 10.0, 150.0, False),
    ("p_pa", 500.0, 50000.0, True),
    ("t_init_c", 10.0, 150.0, False),
    ("q_init_frac", 0.0, 1.2, False),  # q_init = frac * q_sat (bounded by physics)
    ("t_phase_s", 30.0, 3600.0, True),
)
COND_KEYS = tuple(k for k, _, _, _ in COND_SPEC)
COND_DIM = len(COND_KEYS)


def encode_condition(values: dict[str, float]) -> jax.Array:
    """Physical dict -> ``[-1, 1]`` vector (differentiable)."""
    out = []
    for key, lo, hi, log in COND_SPEC:
        v = jnp.asarray(values[key], dtype=jnp.float32)
        if log:
            u = (jnp.log10(v) - jnp.log10(lo)) / (jnp.log10(hi) - jnp.log10(lo))
        else:
            u = (v - lo) / (hi - lo)
        out.append(2.0 * u - 1.0)
    return jnp.stack(out).astype(jnp.float32)


def decode_condition(cond: jax.Array) -> dict[str, jax.Array]:
    """``[-1, 1]`` vector -> physical dict (differentiable — grads flow)."""
    out: dict[str, jax.Array] = {}
    for i, (key, lo, hi, log) in enumerate(COND_SPEC):
        u = (cond[i] + 1.0) / 2.0
        if log:
            out[key] = 10.0 ** (jnp.log10(lo) + u * (jnp.log10(hi) - jnp.log10(lo)))
        else:
            out[key] = lo + u * (hi - lo)
    # Physical derived quantities (kept inside decode so every consumer agrees)
    out["t_f_k"] = out["t_f_c"] + 273.15
    out["t_init_k"] = out["t_init_c"] + 273.15
    out["q_init_kg_kg"] = out["q_init_frac"] * out["q_sat_kg_kg"]
    return out


# ---------------------------------------------------------------------------
# Model: shared tanh trunk + two heads (smooth activations: PDE grads need C²)
# ---------------------------------------------------------------------------

Params = dict[str, jax.Array]


def init_params(key: jax.Array, *, width: int = 64, depth: int = 3) -> Params:
    """Xavier-initialized MLP trunk + linear T/q heads."""
    keys = jax.random.split(key, 2 * depth + 2)
    params: Params = {}
    fan_in = 2 + COND_DIM
    for d in range(depth):
        std = jnp.sqrt(2.0 / (fan_in + width))
        params[f"W{d}"] = jax.random.normal(keys[d], (fan_in, width)) * std
        params[f"b{d}"] = jnp.zeros((width,))
        fan_in = width
    std = jnp.sqrt(2.0 / (width + 1))
    params["WT"] = jax.random.normal(keys[-2], (width, 1)) * std
    params["bT"] = jnp.zeros((1,))
    params["Wq"] = jax.random.normal(keys[-1], (width, 1)) * std
    params["bq"] = jnp.zeros((1,))
    return params


def _trunk(params: Params, z: jax.Array) -> jax.Array:
    depth = (len(params) - 4) // 2
    h = z
    for d in range(depth):
        h = jnp.tanh(h @ params[f"W{d}"] + params[f"b{d}"])
    return h


def predict_Tq(params: Params, x_hat: jax.Array, t_hat: jax.Array,
               cond: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Single query point -> ``(T [K], q [kg/kg])``. Pure; vmap over points."""
    z = jnp.concatenate([jnp.reshape(x_hat, (1,)), jnp.reshape(t_hat, (1,)), cond])
    h = _trunk(params, z)
    T = T_MID_K + jnp.tanh((h @ params["WT"] + params["bT"])[0]) * T_HALF_K
    q = jax.nn.sigmoid((h @ params["Wq"] + params["bq"])[0]) * Q_MAX_KG_KG
    return T, q


def predict_normalized(params: Params, x_hat: jax.Array, t_hat: jax.Array,
                       cond: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Single query point -> ``(u_T in [-1,1], u_q in [0,1])`` (training units)."""
    T, q = predict_Tq(params, x_hat, t_hat, cond)
    return (T - T_MID_K) / T_HALF_K, q / Q_MAX_KG_KG


predict_batch = jax.vmap(predict_Tq, in_axes=(None, 0, 0, None))
predict_normalized_batch = jax.vmap(predict_normalized, in_axes=(None, 0, 0, None))


# ---------------------------------------------------------------------------
# Physics residuals (autodiff through the surrogate = the PINN core)
# ---------------------------------------------------------------------------

def _da_uptake(T: jax.Array, p: jax.Array, q_sat: jax.Array,
               e_char: jax.Array, n: jax.Array,
               psat_fn) -> jax.Array:
    from harness.physics.thermo import GAS_CONSTANT  # local: keeps module import-light

    p_sat = psat_fn(T)
    ratio = jnp.maximum(p_sat / p, 1.0)
    potential = GAS_CONSTANT * T * jnp.log(ratio)
    return q_sat * jnp.exp(-((potential / e_char) ** n))


def pinn_residuals(params: Params, x_hat: jax.Array, t_hat: jax.Array,
                   cond: jax.Array, psat_fn) -> dict[str, jax.Array]:
    """Normalized PDE residuals at one query point (interior of the slab).

    Returns raw fields plus ``r_energy`` / ``r_ldf`` normalized to ~O(1) so a
    single physics weight applies across materials. All terms differentiate
    w.r.t. ``cond`` (decode is differentiable) — this is the directional
    feedback path.
    """
    phys = decode_condition(cond)
    L = phys["L_m"]
    t_phase = phys["t_phase_s"]
    rho = phys["rho_s_kg_m3"]
    c_s = phys["c_s_j_kg_k"]
    k_eff = phys["k_eff_w_m_k"]
    q_st = phys["q_st_j_kg"]
    k_ldf = phys["k_ldf_s_1"]
    P = phys["p_pa"]

    def T_of(xt: jax.Array) -> jax.Array:
        T, _ = predict_Tq(params, xt[0], xt[1], cond)
        return T

    def q_of(xt: jax.Array) -> jax.Array:
        _, q = predict_Tq(params, xt[0], xt[1], cond)
        return q

    xt = jnp.stack([x_hat, t_hat])
    T, q = predict_Tq(params, x_hat, t_hat, cond)
    dT = jax.grad(T_of)(xt)  # [dT/dx_hat, dT/dt_hat]
    dq_dt_hat = jax.grad(q_of)(xt)[1]
    d2T_dx_hat2 = jax.grad(lambda z: jax.grad(T_of)(z)[0])(xt)[0]

    dT_dt = dT[1] / t_phase
    dT_dx = dT[0] / L
    d2T_dx2 = d2T_dx_hat2 / (L * L)
    dq_dt = dq_dt_hat / t_phase

    q_star = _da_uptake(T, P, phys["q_sat_kg_kg"], phys["e_char_j_mol"], phys["n_da"], psat_fn)
    cap = rho * (c_s + C_PL_J_KG_K * q)

    energy = cap * dT_dt - k_eff * d2T_dx2 - rho * dq_dt * q_st
    cap_ref = rho * (c_s + C_PL_J_KG_K * phys["q_sat_kg_kg"] * 0.5)
    r_energy = energy / (cap_ref * T_HALF_K / t_phase + 1e-12)

    ldf = dq_dt - k_ldf * (q_star - q)
    r_ldf = ldf / (Q_MAX_KG_KG / t_phase + 1e-12)
    return {"T": T, "q": q, "q_star": q_star, "dT_dx": dT_dx,
            "r_energy": r_energy, "r_ldf": r_ldf}


def bc_residuals(params: Params, t_hat: jax.Array, cond: jax.Array) -> dict[str, jax.Array]:
    """Normalized wall (convective) + far (adiabatic) boundary residuals."""
    phys = decode_condition(cond)
    L = phys["L_m"]
    k_eff = phys["k_eff_w_m_k"]
    h = phys["h_wall_w_m2_k"]
    T_f = phys["t_f_k"]

    def T_of(xt: jax.Array) -> jax.Array:
        T, _ = predict_Tq(params, xt[0], xt[1], cond)
        return T

    T_wall, _ = predict_Tq(params, jnp.asarray(0.0), t_hat, cond)
    T_far, _ = predict_Tq(params, jnp.asarray(1.0), t_hat, cond)
    dT_wall = jax.grad(T_of)(jnp.stack([jnp.asarray(0.0), t_hat]))[0] / L
    dT_far = jax.grad(T_of)(jnp.stack([jnp.asarray(1.0), t_hat]))[0] / L

    flux_scale = h * T_HALF_K + k_eff * T_HALF_K / L + 1e-12
    r_wall = (-k_eff * dT_wall - h * (T_f - T_wall)) / flux_scale
    r_far = dT_far / (T_HALF_K / L + 1e-12)
    return {"T_wall": T_wall, "T_far": T_far, "r_wall": r_wall, "r_far": r_far}


def ic_residuals(params: Params, x_hat: jax.Array, cond: jax.Array) -> dict[str, jax.Array]:
    """Normalized initial-state residuals at ``t_hat = 0``."""
    phys = decode_condition(cond)
    T0, q0 = predict_Tq(params, x_hat, jnp.asarray(0.0), cond)
    r_T = (T0 - phys["t_init_k"]) / T_HALF_K
    r_q = (q0 - phys["q_init_kg_kg"]) / Q_MAX_KG_KG
    return {"r_T": r_T, "r_q": r_q}


# ---------------------------------------------------------------------------
# Losses (all terms ~O(1); weights ramp in the training curriculum)
# ---------------------------------------------------------------------------

def data_loss(params: Params, x_hat: jax.Array, t_hat: jax.Array,
              cond: jax.Array, uT_true: jax.Array, uq_true: jax.Array) -> jax.Array:
    uT, uq = predict_normalized_batch(params, x_hat, t_hat, cond)
    return jnp.mean((uT - uT_true) ** 2 + (uq - uq_true) ** 2)


def physics_loss(params: Params, x_hat: jax.Array, t_hat: jax.Array,
                 cond: jax.Array, psat_fn) -> dict[str, jax.Array]:
    res = jax.vmap(lambda xh, th: pinn_residuals(params, xh, th, cond, psat_fn))(x_hat, t_hat)
    l_energy = jnp.mean(res["r_energy"] ** 2)
    l_ldf = jnp.mean(res["r_ldf"] ** 2)
    return {"energy": l_energy, "ldf": l_ldf, "total": l_energy + l_ldf}


def boundary_loss(params: Params, t_hat: jax.Array, cond: jax.Array) -> jax.Array:
    res = jax.vmap(lambda th: bc_residuals(params, th, cond))(t_hat)
    return jnp.mean(res["r_wall"] ** 2 + res["r_far"] ** 2)


def initial_loss(params: Params, x_hat: jax.Array, cond: jax.Array) -> jax.Array:
    res = jax.vmap(lambda xh: ic_residuals(params, xh, cond))(x_hat)
    return jnp.mean(res["r_T"] ** 2 + res["r_q"] ** 2)


def total_loss(params: Params, batch: dict[str, jax.Array], weights: dict[str, float],
               psat_fn) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Joint loss. ``batch`` holds one condition's point sets; per-config
    training loops over configs (each config = one broad-model example)."""
    cond = batch["cond"]
    l_data = data_loss(params, batch["x_data"], batch["t_data"], cond,
                       batch["uT_data"], batch["uq_data"])
    phys = physics_loss(params, batch["x_colloc"], batch["t_colloc"], cond, psat_fn)
    l_bc = boundary_loss(params, batch["t_bc"], cond)
    l_ic = initial_loss(params, batch["x_ic"], cond)
    total = (weights["data"] * l_data + weights["pde"] * phys["total"]
             + weights["bc"] * l_bc + weights["ic"] * l_ic)
    aux = {"data": l_data, "pde": phys["total"], "pde_energy": phys["energy"],
           "pde_ldf": phys["ldf"], "bc": l_bc, "ic": l_ic, "total": total}
    return total, aux


__all__ = [
    "COND_DIM", "COND_KEYS", "COND_SPEC", "C_PL_J_KG_K",
    "Q_MAX_KG_KG", "T_HALF_K", "T_MID_K",
    "bc_residuals", "boundary_loss", "data_loss", "decode_condition",
    "encode_condition", "ic_residuals", "initial_loss", "init_params",
    "physics_loss", "pinn_residuals", "predict_Tq", "predict_batch",
    "predict_normalized", "predict_normalized_batch", "total_loss",
]

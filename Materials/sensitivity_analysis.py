"""Sensitivity analysis for the adsorption cooling cycle.

Local elasticity and global (Monte-Carlo SRC) sensitivity of COP / SCP to the
two material-level design variables, ``q_sat`` and ``Q_st``.

Physics comes exclusively from ``cooling_physics.simulate_adsorption_cycle``
(the calibrated Magnus/Antoine + Watson model). This module contains no
duplicate thermodynamics.
"""

import numpy as np

from cooling_physics import simulate_adsorption_cycle

# Baseline candidate: moderate capacity, standard water-adsorption heat.
Q_SAT_NOMINAL = 0.45      # kg_water / kg_adsorbent
Q_ST_NOMINAL = 2.8e6      # J / kg_water (~1.24x latent heat of water)

# Standard HVAC operating point.
OPERATING_POINT = {
    "t_evap_c": 7.0,
    "t_cond_c": 35.0,
    "t_des_c": 80.0,
    "cycle_time_sec": 600.0,
}


# ==========================================
# 1. LOCAL SENSITIVITY (ELASTICITY)
# ==========================================
def calculate_local_elasticity(
    q_sat_base: float,
    Q_st_base: float,
    epsilon: float = 1e-4,
    e_char_j_mol: float = 4500.0,
) -> dict:
    """Dimensionless elasticity at a baseline operating point.

    Elasticity = (% change in output) / (% change in input).
    """
    def evaluate(q_sat: float, q_st: float) -> dict:
        return simulate_adsorption_cycle(
            q_sat=q_sat,
            q_st=q_st,
            e_char_j_mol=e_char_j_mol,
            **OPERATING_POINT,
        )

    base_res = evaluate(q_sat_base, Q_st_base)
    base_cop, base_scp = base_res["COP"], base_res["SCP_W_kg"]

    res_q = evaluate(q_sat_base * (1.0 + epsilon), Q_st_base)
    res_Q = evaluate(q_sat_base, Q_st_base * (1.0 + epsilon))

    return {
        "COP": {
            "q_sat": ((res_q["COP"] - base_cop) / base_cop) / epsilon,
            "Q_st": ((res_Q["COP"] - base_cop) / base_cop) / epsilon,
        },
        "SCP": {
            "q_sat": ((res_q["SCP_W_kg"] - base_scp) / base_scp) / epsilon,
            "Q_st": ((res_Q["SCP_W_kg"] - base_scp) / base_scp) / epsilon,
        },
    }


# ==========================================
# 2. GLOBAL SENSITIVITY (MONTE CARLO & SRC)
# ==========================================
def perform_global_sensitivity(
    n_samples: int = 1000,
    seed: int = 42,
    e_char_j_mol: float = 4500.0,
) -> dict:
    """Monte-Carlo sweep over the material design space.

    Returns Standardized Regression Coefficients (SRC): the fraction of
    output variance explained by each input across the sampled population.
    Seeded RNG keeps runs reproducible.
    """
    rng = np.random.default_rng(seed)

    # Physical ranges representing potential materials.
    # q_sat: 0.1-0.8 kg/kg (zeolite/MOF capacities).
    # Q_st:  2.3e6-3.5e6 J/kg (water interaction energies).
    q_sat_samples = rng.uniform(0.1, 0.8, n_samples)
    Q_st_samples = rng.uniform(2.3e6, 3.5e6, n_samples)

    cops = np.zeros(n_samples)
    scps = np.zeros(n_samples)
    for i in range(n_samples):
        res = simulate_adsorption_cycle(
            q_sat=float(q_sat_samples[i]),
            q_st=float(Q_st_samples[i]),
            e_char_j_mol=e_char_j_mol,
            **OPERATING_POINT,
        )
        cops[i] = res["COP"]
        scps[i] = res["SCP_W_kg"]

    X = np.column_stack([q_sat_samples, Q_st_samples])
    X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)

    def compute_src(y: np.ndarray) -> np.ndarray:
        y_std = y.std()
        if y_std == 0:
            return np.array([0.0, 0.0])
        y_scaled = (y - y.mean()) / y_std
        # Least-squares beta = (X^T X)^-1 X^T y on standardized variables.
        return np.linalg.inv(X_scaled.T @ X_scaled) @ X_scaled.T @ y_scaled

    src_cop = compute_src(cops)
    src_scp = compute_src(scps)

    return {
        "COP": {"q_sat": src_cop[0], "Q_st": src_cop[1]},
        "SCP": {"q_sat": src_scp[0], "Q_st": src_scp[1]},
    }


# ==========================================
# 3. REPORTING
# ==========================================
if __name__ == "__main__":
    print("--- NOMINAL EVALUATION POINT ---")
    print(f"Capacity (q_sat): {Q_SAT_NOMINAL} kg/kg | "
          f"Adsorption Heat (Q_st): {Q_ST_NOMINAL/1e6:.2f} MJ/kg")
    nominal = simulate_adsorption_cycle(Q_SAT_NOMINAL, Q_ST_NOMINAL, **OPERATING_POINT)
    print(f"Baseline COP: {nominal['COP']:.3f} | "
          f"Baseline SCP: {nominal['SCP_W_kg']:.2f} W/kg\n")

    local_sens = calculate_local_elasticity(Q_SAT_NOMINAL, Q_ST_NOMINAL)
    print("--- 1. LOCAL SENSITIVITY (ELASTICITY) ---")
    print("Interpretation: % change in metric per 1% change in material property.")
    print(f"  COP sensitivity to q_sat: {local_sens['COP']['q_sat']:+.4f}")
    print(f"  COP sensitivity to Q_st : {local_sens['COP']['Q_st']:+.4f}")
    print(f"  SCP sensitivity to q_sat: {local_sens['SCP']['q_sat']:+.4f}")
    print(f"  SCP sensitivity to Q_st : {local_sens['SCP']['Q_st']:+.4f}\n")

    global_sens = perform_global_sensitivity(n_samples=5000)
    print("--- 2. GLOBAL SENSITIVITY (STANDARDIZED REGRESSION COEFFICIENTS) ---")
    print("Interpretation: Relative driver of variance across a population of materials.")
    print(f"  COP Variance driven by q_sat: {global_sens['COP']['q_sat']:.4f}")
    print(f"  COP Variance driven by Q_st : {global_sens['COP']['Q_st']:.4f}")
    print(f"  SCP Variance driven by q_sat: {global_sens['SCP']['q_sat']:.4f}")
    print(f"  SCP Variance driven by Q_st : {global_sens['SCP']['Q_st']:.4f}")

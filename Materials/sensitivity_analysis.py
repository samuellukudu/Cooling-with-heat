import math
import numpy as np

# ==========================================
# 1. CORE THERMODYNAMIC MODEL
# ==========================================
def simulate_adsorption_cycle(q_sat: float, Q_st: float, 
                              T_evap_C: float = 7.0, 
                              T_cond_C: float = 35.0, 
                              T_des_C: float = 80.0, 
                              cycle_time_sec: float = 600.0) -> dict:
    """
    Evaluates system performance for a specified set of material properties.
    Inputs:
        q_sat: Max adsorption capacity (kg_refrigerant / kg_adsorbent)
        Q_st: Isosteric heat of adsorption (J / kg_refrigerant)
    """
    T_evap = T_evap_C + 273.15
    T_cond = T_cond_C + 273.15
    T_des = T_des_C + 273.15
    T_ads = T_cond  # Adsorption bed is cooled by the condenser loop sink
    
    R = 8.314  # J/(mol*K)
    M_refrigerant = 0.018015  # kg/mol (Water)
    C_p_adsorbent = 1000.0  # J/(kg*K)
    C_p_liquid = 4184.0  # J/(kg*K)
    
    # Vapor pressure approximation (Clausius-Clapeyron for water)
    def get_sat_pressure(T_k):
        delta_H_vap = 40700.0  # J/mol
        P0 = 101325.0  # Pa
        T0 = 373.15  # K
        return P0 * math.exp(-(delta_H_vap / R) * (1.0 / T_k - 1.0 / T0))
        
    P_evap = get_sat_pressure(T_evap)
    P_cond = get_sat_pressure(T_cond)
    
    # Dubinin-Astakhov isotherm model
    E_adsorption = 14000.0  # J/mol (Characteristic energy)
    n_heterogeneity = 1.8
    
    def calculate_uptake(T_bed, P_system):
        P_sat = get_sat_pressure(T_bed)
        if P_system >= P_sat:
            return q_sat
        A_pot = R * T_bed * math.log(P_sat / P_system)
        return q_sat * math.exp(- (A_pot / E_adsorption) ** n_heterogeneity)
        
    q_ads = calculate_uptake(T_ads, P_evap)
    q_des = calculate_uptake(T_des, P_cond)
    delta_q = max(0.0, q_ads - q_des)
    
    h_fg = 40700.0 / M_refrigerant  # Latent heat of water (~2.26 MJ/kg)
    Q_cool = delta_q * h_fg
    
    Q_sensible = (C_p_adsorbent + q_ads * C_p_liquid) * (T_des - T_ads)
    Q_desorption = delta_q * Q_st
    Q_in = Q_sensible + Q_desorption
    
    cop = Q_cool / Q_in if Q_in > 0 else 0.0
    scp = Q_cool / cycle_time_sec if cycle_time_sec > 0 else 0.0
    
    return {"COP": cop, "SCP": scp}

# ==========================================
# 2. LOCAL SENSITIVITY (ELASTICITY)
# ==========================================
def calculate_local_elasticity(q_sat_base: float, Q_st_base: float, epsilon: float = 1e-4):
    """
    Computes dimensionless elasticity (local sensitivity) at a baseline operating point.
    Elasticity = (% Change in Output) / (% Change in Input)
    """
    # Evaluate at baseline
    base_res = simulate_adsorption_cycle(q_sat_base, Q_st_base)
    base_cop, base_scp = base_res["COP"], base_res["SCP"]
    
    # 1. Perturb q_sat
    q_sat_perturbed = q_sat_base * (1.0 + epsilon)
    res_q = simulate_adsorption_cycle(q_sat_perturbed, Q_st_base)
    
    elasticity_cop_q = ((res_q["COP"] - base_cop) / base_cop) / epsilon
    elasticity_scp_q = ((res_q["SCP"] - base_scp) / base_scp) / epsilon
    
    # 2. Perturb Q_st
    Q_st_perturbed = Q_st_base * (1.0 + epsilon)
    res_Q = simulate_adsorption_cycle(q_sat_base, Q_st_perturbed)
    
    elasticity_cop_Q = ((res_Q["COP"] - base_cop) / base_cop) / epsilon
    elasticity_scp_Q = ((res_Q["SCP"] - base_scp) / base_scp) / epsilon
    
    return {
        "COP": {"q_sat": elasticity_cop_q, "Q_st": elasticity_cop_Q},
        "SCP": {"q_sat": elasticity_scp_q, "Q_st": elasticity_scp_Q}
    }

# ==========================================
# 3. GLOBAL SENSITIVITY (MONTE CARLO & SRC)
# ==========================================
def perform_global_sensitivity(n_samples: int = 1000):
    """
    Performs a Monte Carlo sweep across the material design space 
    and computes Standardized Regression Coefficients (SRC) to index global sensitivity.
    """
    # Define broad physical ranges representing potential materials
    # q_sat: 0.1 to 0.8 kg/kg (corresponds to various zeolite/MOF capacities)
    q_sat_samples = np.random.uniform(0.1, 0.8, n_samples)
    # Q_st: 2.3e6 J/kg to 3.5e6 J/kg (range of interaction energy with water)
    Q_st_samples = np.random.uniform(2.3e6, 3.5e6, n_samples)
    
    cops = np.zeros(n_samples)
    scps = np.zeros(n_samples)
    
    for i in range(n_samples):
        res = simulate_adsorption_cycle(q_sat_samples[i], Q_st_samples[i])
        cops[i] = res["COP"]
        scps[i] = res["SCP"]
        
    # Standardize inputs (X) and outputs (Y) to calculate dimensionless SRCs
    X = np.column_stack([q_sat_samples, Q_st_samples])
    X_mean, X_std = np.mean(X, axis=0), np.std(X, axis=0)
    X_scaled = (X - X_mean) / X_std
    
    def compute_src(Y):
        Y_mean, Y_std = np.mean(Y), np.std(Y)
        if Y_std == 0:
            return np.array([0.0, 0.0])
        Y_scaled = (Y - Y_mean) / Y_std
        # Linear regression solver: beta = (X^T * X)^-1 * X^T * Y
        beta = np.linalg.inv(X_scaled.T @ X_scaled) @ X_scaled.T @ Y_scaled
        return beta
        
    src_cop = compute_src(cops)
    src_scp = compute_src(scps)
    
    return {
        "COP": {"q_sat": src_cop[0], "Q_st": src_cop[1]},
        "SCP": {"q_sat": src_scp[0], "Q_st": src_scp[1]}
    }

# ==========================================
# 4. REPORTING THE RESULTS
# ==========================================
if __name__ == "__main__":
    # Baseline candidate: Moderate capacity, standard water-adsorption heat
    Q_SAT_NOMINAL = 0.45      # kg_water / kg_adsorbent
    Q_ST_NOMINAL = 2.8e6      # J / kg_water (~1.24x latent heat of water)
    
    print(f"--- NOMINAL EVALUATION POINT ---")
    print(f"Capacity (q_sat): {Q_SAT_NOMINAL} kg/kg | Adsorption Heat (Q_st): {Q_ST_NOMINAL/1e6:.2f} MJ/kg")
    nominal_perf = simulate_adsorption_cycle(Q_SAT_NOMINAL, Q_ST_NOMINAL)
    print(f"Baseline COP: {nominal_perf['COP']:.3f} | Baseline SCP: {nominal_perf['SCP']:.2f} W/kg\n")
    
    # 1. Run Local Sensitivity
    local_sens = calculate_local_elasticity(Q_SAT_NOMINAL, Q_ST_NOMINAL)
    print("--- 1. LOCAL SENSITIVITY (ELASTICITY) ---")
    print("Interpretation: % change in metric per 1% change in material property.")
    print(f"  COP sensitivity to q_sat: {local_sens['COP']['q_sat']:+.4f}")
    print(f"  COP sensitivity to Q_st : {local_sens['COP']['Q_st']:+.4f}")
    print(f"  SCP sensitivity to q_sat: {local_sens['SCP']['q_sat']:+.4f}")
    print(f"  SCP sensitivity to Q_st : {local_sens['SCP']['Q_st']:+.4f}\n")
    
    # 2. Run Global Sensitivity
    global_sens = perform_global_sensitivity(n_samples=5000)
    print("--- 2. GLOBAL SENSITIVITY (STANDARDIZED REGRESSION COEFFICIENTS) ---")
    print("Interpretation: Relative driver of variance across a population of materials.")
    print(f"  COP Variance driven by q_sat: {global_sens['COP']['q_sat']:.4f}")
    print(f"  COP Variance driven by Q_st : {global_sens['COP']['Q_st']:.4f}")
    print(f"  SCP Variance driven by q_sat: {global_sens['SCP']['q_sat']:.4f}")
    print(f"  SCP Variance driven by Q_st : {global_sens['SCP']['Q_st']:.4f}")
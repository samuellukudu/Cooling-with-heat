import torch

def differentiable_hvac_cycle(q_sat: torch.Tensor, Q_st: torch.Tensor) -> torch.Tensor:
    """
    A PyTorch-based, fully differentiable thermodynamic adsorption cycle simulator.
    Keeps track of computational graphs for backpropagation.
    """
    # Operating conditions (Constants)
    T_evap = torch.tensor(7.0 + 273.15)
    T_cond = torch.tensor(35.0 + 273.15)
    T_des = torch.tensor(80.0 + 273.15)
    T_ads = T_cond
    
    R = 8.314
    M_refrigerant = 0.018015
    C_p_adsorbent = 1000.0
    C_p_liquid = 4184.0
    
    # Differentiable vapor pressure calculation
    def get_sat_pressure(T_k):
        delta_H_vap = 40700.0
        P0 = 101325.0
        T0 = 373.15
        return P0 * torch.exp(-(delta_H_vap / R) * (1.0 / T_k - 1.0 / T0))
        
    P_evap = get_sat_pressure(T_evap)
    P_cond = get_sat_pressure(T_cond)
    
    # Dubinin-Astakhov isotherm model
    E_adsorption = 14000.0
    n_heterogeneity = 1.8
    
    def calculate_uptake(T_bed, P_system):
        P_sat = get_sat_pressure(T_bed)
        # Smooth approximation of conditional indexing to maintain clean gradients
        A_pot = R * T_bed * torch.log(P_sat / P_system)
        return q_sat * torch.exp(- (A_pot / E_adsorption) ** n_heterogeneity)
        
    q_ads = calculate_uptake(T_ads, P_evap)
    q_des = calculate_uptake(T_des, P_cond)
    
    # Smooth ReLU approximation to handle physical boundary where delta_q > 0
    delta_q = torch.clamp(q_ads - q_des, min=0.0)
    
    h_fg = 40700.0 / M_refrigerant
    Q_cool = delta_q * h_fg
    
    Q_sensible = (C_p_adsorbent + q_ads * C_p_liquid) * (T_des - T_ads)
    Q_desorption = delta_q * Q_st
    Q_in = Q_sensible + Q_desorption
    
    cop = Q_cool / Q_in
    return cop

# --- Demonstration of Gradient Backpropagation ---
if __name__ == "__main__":
    # We define our material properties as tensors that require gradients
    # This simulates receiving predicted values from a Graph Neural Network
    q_sat_pred = torch.tensor(0.45, requires_grad=True)
    Q_st_pred = torch.tensor(2.8e6, requires_grad=True)
    
    # Run the differentiable forward simulation
    cop = differentiable_hvac_cycle(q_sat_pred, Q_st_pred)
    print(f"Calculated COP: {cop.item():.4f}")
    
    # Backpropagate the system-level objective (COP)
    cop.backward()
    
    # Now we have the exact mathematical gradients!
    print("\n--- ANALYTICAL SYSTEM GRADIENTS ---")
    print(f"d(COP) / d(q_sat) : {q_sat_pred.grad.item():+.4f}")
    print(f"d(COP) / d(Q_st)  : {Q_st_pred.grad.item():.4e}")
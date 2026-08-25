> **Status (2026-08).** Reference material for the eventual gradient-guided
> phase — the premise of building our own differentiable simulation stack is
> de-emphasized in the current plan. The PyTorch cycle demo below
> (`differentiable_hvac_cycle`) would be ported to JAX (matching the project
> stack) only if that phase starts. See [`../ROADMAP.md`](../ROADMAP.md).

There is a profound and highly active connection between **generative inverse design** and **differentiable simulations**. 

In conventional materials science, numerical simulations (like Grand Canonical Monte Carlo to simulate gas adsorption, or thermodynamic solvers to calculate COP) are treated as "black boxes". If you want to know how changing a pore's diameter affects the cycle COP, you have to run a new simulation from scratch and estimate the sensitivity using finite differences. This makes exploring millions of hypothetical structures incredibly slow and limits you to derivative-free optimization methods (like Reinforcement Learning or Genetic Algorithms), which scale poorly in high-dimensional spaces.

A **differentiable simulation** rewrites the physics equations using Automatic Differentiation (AD) frameworks like **JAX, PyTorch, or TensorFlow**. Because every step of the physics is mathematically tracked, you can calculate the exact, analytical gradient of your system-level output (like COP) with respect to your microstructural inputs (like atomic coordinates or pore volumes) in a single backward pass.

---

### The Connection: End-to-End Gradient Descent

By combining a differentiable machine learning model (a GNN) with a differentiable physics simulation, you create an end-to-end differentiable pipeline. 

Using the chain rule of calculus, you can compute:

$$\frac{\partial (\text{HVAC COP})}{\partial (\text{Atomic Coordinates})} = \frac{\partial (\text{HVAC COP})}{\partial (q_{sat}, Q_{st})} \times \frac{\partial (q_{sat}, Q_{st})}{\partial (\text{Atomic Coordinates})}$$

This means you do not have to guess or run slow search loops. You can use standard gradient descent (like the Adam optimizer) to **physically "nudge" the atoms in your crystal lattice** in the exact vector direction that increases your thermodynamic COP.

This is a highly active area of academic and industrial research. For instance, researchers recently published *"End-to-End Differentiability and Tensor Processing Unit Computing to Accelerate Materials' Inverse Design,"* where they reformulated a lattice Density Functional Theory (DFT) gas sorption simulator as a differentiable program. By doing so, they trained generative AI models directly against the physical simulation, allowing the AI to generate the exact porous matrices required for highly specific adsorption curves.

---

### Making the HVAC Loop Differentiable (Concept)

To illustrate how easy it is to bridge this gap, we can translate our previously written thermodynamic cycle simulator from NumPy/math into **PyTorch**. This instantly transforms the standard cycle equations into a differentiable simulator, giving us direct access to system gradients.

```python
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
```

### Why This is Revolutionary for Your Workflow

If you construct your pipeline this way, your generative AI doesn't have to guess what coordinates to try next through trial and error. 

Instead, the generative model receives a clear vector map:
* It reads `d(COP) / d(q_sat) = +1.57`, telling it: *"To improve the HVAC system, alter the generated coordinates to make the pores larger so they hold more gas."*
* It reads `d(COP) / d(Q_st) = -1.97e-7`, telling it: *"Simultaneously, swap out highly polar elements in the framework to weaken the binding affinity so we can desorb the gas with less heat."*

This turns inverse design into a guided optimization problem where you literally carve out porous networks using the gradients of your thermodynamic cycle.
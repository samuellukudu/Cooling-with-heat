import math
import warnings
from pymatgen.core import Structure, Lattice

from env_utils import get_mp_api_key

# Suppress pymatgen warnings for cleaner terminal output
warnings.filterwarnings("ignore")

try:
    from mp_api.client import MPRester
except ImportError:
    MPRester = None

# ==========================================
# 1. DATA ACQUISITION LAYER
# ==========================================
def get_material_structure(api_key: str, material_id: str = "mp-22526") -> Structure:
    """
    Fetches a crystal structure from the Materials Project API.
    If the API key is empty or connections fail, it falls back to a mock structure
    so the script remains runnable out of the box.
    """
    if api_key and MPRester is None:
        print("MP_API_KEY found, but mp_api is not installed in this Python environment.")
        print("Using a mock structure. Try: .venv/bin/python main.py")
    elif api_key and api_key != "YOUR_API_KEY_HERE":
        try:
            print(f"Retrieving structure for {material_id} from Materials Project...")
            with MPRester(api_key) as mpr:
                # Retrieve the structure using the modern mp-api route
                structure = mpr.get_structure_by_material_id(material_id)
                print(f"Successfully retrieved structure. Composition: {structure.composition.reduced_formula}")
                return structure
        except Exception as e:
            print(f"Materials Project API lookup failed: {e}")
    
    print("API Key not configured or connection failed. Falling back to a mock local structure...")
    # Mock a standard cubic structure resembling a porous framework (e.g., cubic perovsite/oxide matrix)
    lattice = Lattice.cubic(5.4)
    return Structure(
        lattice, 
        ["Al", "O", "O", "O"], 
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]
    )

# ==========================================
# 2. FEATURE EXTRACTION & SURROGATE ML PREDICTOR
# ==========================================
def featurize_structure(structure: Structure) -> dict:
    """
    Extracts geometric and compositional descriptors from the pymatgen structure object.
    These descriptors mimic the input representation for Graph Neural Networks or ML models.
    """
    density = structure.density  # g/cm3
    volume_per_atom = structure.volume / len(structure)  # Å³ per atom
    
    # Calculate average electronegativity of all sites to represent chemical binding profiles
    electronegativities = []
    for site in structure:
        try:
            electronegativities.append(site.specie.X)
        except Exception:
            electronegativities.append(2.0)  # Standard fallback electronegativity value
            
    avg_electronegativity = sum(electronegativities) / len(electronegativities)
    
    return {
        "density": density,
        "volume_per_atom": volume_per_atom,
        "avg_electronegativity": avg_electronegativity
    }

def predict_adsorption_properties(features: dict) -> dict:
    """
    Surrogate function mapping crystal structures to adsorption isotherm parameters.
    In a fully realized pipeline, this step is replaced by a trained GNN
    (e.g., predicting Dubinin-Astakhov isotherm limits from crystal structures).
    """
    vol_per_atom = features["volume_per_atom"]
    avg_en = features["avg_electronegativity"]
    density = features["density"]
    
    # Sigmoid mapping to ensure the predicted maximum uptake (q_sat) behaves physically
    # (Porous structures with high volume-per-atom generally hold more refrigerant gas)
    q_sat = 0.8 * (1.0 / (1.0 + math.exp(-0.1 * (vol_per_atom - 20.0))))  # kg_refrigerant / kg_adsorbent
    
    # Isosteric Heat of Adsorption (Q_st) based on average chemical electronegativity
    Q_st = 2.4e6 + (avg_en * 2.5e5) - (density * 1.2e5)  # J/kg_refrigerant
    
    # Impose physical bounds
    return {
        "q_sat": max(0.05, min(q_sat, 1.5)),
        "Q_st": max(2.3e6, min(Q_st, 4.5e6))
    }

# ==========================================
# 3. THERMODYNAMIC CYCLE SIMULATOR (HVAC COUPLING)
# ==========================================
def simulate_hvac_cycle(q_sat: float, Q_st: float, 
                         T_evap_C: float = 7.0, 
                         T_cond_C: float = 35.0, 
                         T_des_C: float = 80.0, 
                         cycle_time_sec: float = 600.0) -> dict:
    """
    Models a thermodynamic adsorption cooling cycle using predicted adsorbent parameters.
    Calculates operational pressures, loading limits, cycle COP, and Specific Cooling Power.
    """
    # Convert temperatures to Kelvin
    T_evap = T_evap_C + 273.15
    T_cond = T_cond_C + 273.15
    T_des = T_des_C + 273.15
    T_ads = T_cond_C + 273.15 # Adsorption occurs at cooling water temperature (condenser loop sink)
    
    # Physical Constants
    R = 8.314  # J/(mol*K)
    M_refrigerant = 0.018015  # kg/mol (Water)
    C_p_adsorbent = 1000.0  # J/(kg*K)
    C_p_liquid = 4184.0  # J/(kg*K)
    
    # Vapor pressure approximation of water using Clausius-Clapeyron relation
    def get_sat_pressure(T_kelvin: float) -> float:
        delta_H_vap = 40700.0  # J/mol (latent heat of water)
        P0 = 101325.0  # Pa (atmospheric pressure at boiling point)
        T0 = 373.15  # K
        return P0 * math.exp(-(delta_H_vap / R) * (1.0 / T_kelvin - 1.0 / T0))
        
    P_evap = get_sat_pressure(T_evap)
    P_cond = get_sat_pressure(T_cond)
    
    # Dubinin-Astakhov (D-A) model parameters
    E_adsorption = 14000.0  # J/mol (Characteristic energy of adsorption)
    n_heterogeneity = 1.8  
    
    def calculate_uptake(T_bed: float, P_system: float) -> float:
        P_sat = get_sat_pressure(T_bed)
        if P_system >= P_sat:
            return q_sat
        # D-A equation: q = q_sat * exp(-(A / E)^n) where A = R * T * ln(P_sat / P)
        A_pot = R * T_bed * math.log(P_sat / P_system)
        return q_sat * math.exp(- (A_pot / E_adsorption) ** n_heterogeneity)
        
    # State 1: End of adsorption (low temp, low pressure)
    q_ads = calculate_uptake(T_ads, P_evap)
    
    # State 2: End of desorption (high temp, high pressure)
    q_des = calculate_uptake(T_des, P_cond)
    
    delta_q = max(0.0, q_ads - q_des)
    
    # Latent heat of vaporization of water refrigerant (J/kg)
    h_fg = 40700.0 / M_refrigerant  # ~2.26 MJ/kg
    
    # Cooling energy generated per cycle (J / kg of adsorbent)
    Q_cool = delta_q * h_fg
    
    # Thermal energy input required for regeneration (J / kg of adsorbent)
    Q_sensible = (C_p_adsorbent + q_ads * C_p_liquid) * (T_des - T_ads)
    Q_desorption = delta_q * Q_st
    Q_in = Q_sensible + Q_desorption
    
    # Performance metrics
    cop = Q_cool / Q_in if Q_in > 0 else 0.0
    scp = Q_cool / cycle_time_sec if cycle_time_sec > 0 else 0.0
    
    return {
        "P_evap_kPa": P_evap / 1000.0,
        "P_cond_kPa": P_cond / 1000.0,
        "q_ads": q_ads,
        "q_des": q_des,
        "delta_q": delta_q,
        "Q_cool_kJ_kg": Q_cool / 1000.0,
        "Q_in_kJ_kg": Q_in / 1000.0,
        "COP": cop,
        "SCP_W_kg": scp
    }

# ==========================================
# 4. EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    # Set MP_API_KEY in .env or your shell to query Materials Project.
    # Without it, the script uses a fallback mock structure.
    MP_API_KEY = get_mp_api_key() or ""
    
    # 1. Fetch materials data
    struct = get_material_structure(MP_API_KEY, material_id="mp-22526")
    
    # 2. Extract structural features (representing computational filtering)
    features = featurize_structure(struct)
    print("\n--- EXTRACTED ATOMIC FEATURES ---")
    for key, val in features.items():
        print(f"{key}: {val:.4f}")
        
    # 3. Regress properties using the ML proxy
    predicted_props = predict_adsorption_properties(features)
    print("\n--- PREDICTED MATERIAL PROPERTIES ---")
    print(f"Max Capacity (q_sat): {predicted_props['q_sat']:.4f} kg_H2O / kg_adsorbent")
    print(f"Isosteric Heat (Q_st): {predicted_props['Q_st'] / 1e6:.4f} MJ / kg_H2O")
    
    # 4. Evaluate in a simulated dynamic HVAC loop
    # Standard operating points: T_evap = 7°C, T_cond = 35°C, T_desorption = 80°C
    results = simulate_hvac_cycle(
        q_sat=predicted_props["q_sat"], 
        Q_st=predicted_props["Q_st"],
        T_evap_C=7.0,
        T_cond_C=35.0,
        T_des_C=80.0,
        cycle_time_sec=600.0
    )
    
    print("\n--- SYSTEM-LEVEL THERMODYNAMIC PERFORMANCE ---")
    print(f"Operating Pressures: Evaporator = {results['P_evap_kPa']:.2f} kPa, Condenser = {results['P_cond_kPa']:.2f} kPa")
    print(f"Adsorbent Working Uptake (q_ads): {results['q_ads']:.4f} kg_ref/kg_ads")
    print(f"Adsorbent Residual Uptake (q_des): {results['q_des']:.4f} kg_ref/kg_ads")
    print(f"Effective Mass Flow (delta_q): {results['delta_q']:.4f} kg_ref/kg_ads")
    print(f"Cycle Cooling Output: {results['Q_cool_kJ_kg']:.2f} kJ/kg")
    print(f"Cycle Thermal Input: {results['Q_in_kJ_kg']:.2f} kJ/kg")
    print(f"Coefficient of Performance (COP): {results['COP']:.3f}")
    print(f"Specific Cooling Power (SCP): {results['SCP_W_kg']:.2f} W/kg")

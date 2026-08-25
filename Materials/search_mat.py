from env_utils import get_mp_api_key

try:
    from mp_api.client import MPRester
except ImportError:
    MPRester = None

def search_adsorption_candidates(api_key: str):
    """
    Queries the Materials Project for open-framework (low density) 
    Aluminophosphates (Al-P-O) and Silicates (Si-O) that are stable.
    """
    # 1. Define our search filters
    max_density = 1.8          # g/cm3 (dense minerals are usually > 2.5)
    max_energy_above_hull = 0.08  # eV/atom (ensures thermodynamic viability)
    
    # We will search in two highly relevant chemistries for adsorption cooling:
    target_chemical_systems = ["Al-P-O", "Si-O"]
    
    all_candidates = []
    
    # Check if the API key is provided and valid
    if api_key and MPRester is None:
        print("MP_API_KEY found, but mp_api is not installed in this Python environment.")
        print("Using local mock candidates. Try: .venv/bin/python search_mat.py\n")
    elif api_key and api_key != "YOUR_MATERIALS_PROJECT_API_KEY":
        try:
            print("Connecting to Materials Project API...")
            with MPRester(api_key) as mpr:
                for chemsys in target_chemical_systems:
                    print(f"Searching chemical system: {chemsys} ...")
                    
                    # Search using the materials summary endpoint
                    results = mpr.materials.summary.search(
                        chemsys=chemsys,
                        density=(0.1, max_density),
                        energy_above_hull=(0.0, max_energy_above_hull),
                        fields=["material_id", "formula_pretty", "density", "energy_above_hull", "volume"]
                    )
                    
                    for doc in results:
                        all_candidates.append({
                            "material_id": str(doc.material_id),
                            "formula": doc.formula_pretty,
                            "density": doc.density,
                            "energy_above_hull": doc.energy_above_hull,
                            "volume": doc.volume,
                            "chemical_system": chemsys
                        })
            
            print(f"Query complete. Found {len(all_candidates)} candidates matching criteria.")
            return all_candidates
            
        except Exception as e:
            print(f"\n[API Warning] Connection failed: {e}")
            print("Processing with mock data for demonstration purposes.\n")
            
    # 2. Fallback Mock Data (Simulates what the API returns)
    print("Generating local database candidates (Simulation)...")
    return [
        {"material_id": "mp-555823", "formula": "SiO2", "density": 1.54, "energy_above_hull": 0.02, "volume": 128.5, "chemical_system": "Si-O"},
        {"material_id": "mp-110234", "formula": "AlPO4", "density": 1.68, "energy_above_hull": 0.04, "volume": 242.1, "chemical_system": "Al-P-O"},
        {"material_id": "mp-753901", "formula": "SiO2", "density": 1.41, "energy_above_hull": 0.05, "volume": 139.8, "chemical_system": "Si-O"},
        {"material_id": "mp-221054", "formula": "AlP3O10", "density": 1.72, "energy_above_hull": 0.01, "volume": 310.2, "chemical_system": "Al-P-O"},
    ]

# --- Run the Search ---
# Set MP_API_KEY in .env or your shell to query Materials Project.
# Without it, the script uses local demonstration data.
API_KEY = get_mp_api_key() or ""
candidates = search_adsorption_candidates(API_KEY)

# Display the sorted list of potential candidates (sorting by lowest density first)
print("\n--- POROUS CANDIDATES IDENTIFIED ---")
print(f"{'ID':<15} | {'Formula':<10} | {'Density (g/cm³)':<16} | {'E_hull (eV/atom)':<17} | {'System'}")
print("-" * 75)
for c in sorted(candidates, key=lambda x: x["density"]):
    print(f"{c['material_id']:<15} | {c['formula']:<10} | {c['density']:<16.2f} | {c['energy_above_hull']:<17.3f} | {c['chemical_system']}")

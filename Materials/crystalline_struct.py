import numpy as np
import warnings
from pymatgen.core import Structure, Lattice

# Suppress pymatgen warnings for clean output
warnings.filterwarnings("ignore")

# ==========================================
# 1. NODE FEATURE GENERATION
# ==========================================
def extract_node_features(structure: Structure) -> np.ndarray:
    """
    Constructs a feature matrix for the crystal nodes (atoms).
    Each row corresponds to an atom in the unit cell.
    """
    node_features = []
    
    for site in structure:
        # We extract standard elemental properties available via pymatgen
        element = site.specie
        
        # Simple feature vector containing:
        # [Atomic Number, Atomic Mass, Electronegativity, Covalent Radius]
        features = [
            float(element.Z),
            float(element.atomic_mass),
            float(element.X if hasattr(element, "X") else 2.0), # fallback electronegativity
            float(element.average_ionic_radius if element.average_ionic_radius else 1.0) # fallback radius
        ]
        node_features.append(features)
        
    return np.array(node_features, dtype=np.float32)

# ==========================================
# 2. EDGE FEATURE GENERATION (GAUSSIAN EXPANSION)
# ==========================================
def expand_distances_rbf(distances: np.ndarray, 
                           min_dist: float = 0.0, 
                           max_dist: float = 6.0, 
                           step: float = 0.5, 
                           width: float = 0.5) -> np.ndarray:
    """
    Expands continuous bond distances into high-dimensional vectors 
    using Gaussian Radial Basis Functions (RBF). This is the standard 
    representation in models like CGCNN, MEGNet, and ALIGNN.
    """
    # Create centers for the Gaussian filters (e.g., [0.0, 0.5, 1.0, ..., 5.5, 6.0])
    centers = np.arange(min_dist, max_dist + step, step)
    
    # Broadcast subtraction: (num_edges, 1) - (1, num_centers)
    diff = distances[:, np.newaxis] - centers[np.newaxis, :]
    
    # Apply Gaussian function: exp(-(d - mu)^2 / (2 * sigma^2))
    rbf_features = np.exp(-(diff ** 2) / (2 * (width ** 2)))
    
    return rbf_features

# ==========================================
# 3. CRYSTAL TO GRAPH PIPELINE
# ==========================================
def structure_to_graph(structure: Structure, cutoff_radius: float = 5.0) -> dict:
    """
    Converts a pymatgen Structure object (loaded from a CIF) into 
    matrices suitable for Graph Neural Networks.
    """
    # A. Node Features
    X = extract_node_features(structure)
    
    # B. Edge Index & Distances under Periodic Boundary Conditions (PBC)
    # get_neighbor_list queries all periodic images within the cutoff radius.
    # It returns four arrays: center node index, neighbor node index, 
    # periodic image offsets, and actual physical distances in Angstroms.
    center_indices, neighbor_indices, images, distances = structure.get_neighbor_list(
        r=cutoff_radius, 
        numerical_tol=1e-5,
        exclude_self=True
    )
    
    # Formulate edge index as (2, Num_Edges) array (standard for PyTorch Geometric)
    edge_index = np.vstack([center_indices, neighbor_indices])
    
    # C. Edge Features (Expand raw 1D distances using Gaussian RBF)
    edge_attr = expand_distances_rbf(distances)
    
    return {
        "x": X,                # Node feature matrix [Num_Atoms, Num_Node_Features]
        "edge_index": edge_index,  # Adjacency connections [2, Num_Edges]
        "edge_attr": edge_attr,    # Edge feature matrix [Num_Edges, Num_RBF_Features]
        "raw_distances": distances # Real-space distances in Angstroms
    }

# ==========================================
# 4. TESTING THE GRAPH FEATURIZER
# ==========================================
if __name__ == "__main__":
    # In a real environment, you would load your CIF file directly:
    # structure = Structure.from_file("path_to_structure.cif")
    
    # For this demonstration, we'll construct a simple, periodic quartz lattice (SiO2)
    lattice = Lattice.cubic(4.5)
    silica_structure = Structure(
        lattice, 
        ["Si", "O", "O"], 
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5]]
    )
    
    # Convert the periodic structure to graph arrays
    cutoff = 5.0 # Angstroms
    graph_data = structure_to_graph(silica_structure, cutoff_radius=cutoff)
    
    print("--- GRAPH FEATURIZATION COMPLETE ---")
    print(f"Structure parsed with: {len(silica_structure)} atoms in the unit cell.")
    print(f"Cutoff Radius applied: {cutoff} Å\n")
    
    print("1. NODE FEATURE MATRIX (x):")
    print(f"Shape: {graph_data['x'].shape} -> (Atoms, Node Features)")
    print("First atom features [Z, mass, electronegativity, radius]:")
    print(f"  {graph_data['x'][0]}\n")
    
    print("2. EDGE INDEX (edge_index):")
    print(f"Shape: {graph_data['edge_index'].shape} -> (Source/Target pairs, Num Edges)")
    print("First 5 edge connections (Source -> Target):")
    print(f"  {graph_data['edge_index'][:, :5]}\n")
    
    print("3. EDGE ATTRIBUTE MATRIX (edge_attr - via Gaussian RBF):")
    print(f"Shape: {graph_data['edge_attr'].shape} -> (Edges, Expanded RBF dimensions)")
    print("First edge distance raw value: ")
    print(f"  {graph_data['raw_distances'][0]:.4f} Å")
    print("First edge distance RBF representation: ")
    print(f"  {graph_data['edge_attr'][0]}")
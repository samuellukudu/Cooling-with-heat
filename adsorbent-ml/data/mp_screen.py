"""Materials Project query layer (Stage-0 export support).

Vendored from the archived ``Materials/heat_cooling_screen.py`` screening
effort (2026-09): only the proven API query layer is kept — application
search criteria, chemical-system generation, and candidate fetching. The
scoring/ranking half (which depended on ``cooling_physics``) stayed behind;
adsorbent-ml judges candidates with the harness physics instead.

Requires ``mp_api`` (+ ``pymatgen``) at call time — an optional, heavy,
imported lazily so the rest of the data pipeline never pays for it.
"""

import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

try:
    from mp_api.client import MPRester
except ImportError:
    MPRester = None


@dataclass(frozen=True)
class SearchCriteria:
    label: str
    required_elements: Sequence[str]
    allowed_elements: Sequence[str]
    density: tuple[float, float]
    energy_above_hull: tuple[float, float]
    num_elements: tuple[int, int]
    rationale: str
    band_gap_min: float = 0.1  # eV — excludes metals; adsorbers are insulators/semiconductors


SEARCH_CRITERIA: Dict[str, SearchCriteria] = {
    "cpu": SearchCriteria(
        label="fast electronic thermal management",
        required_elements=("O",),
        allowed_elements=("Al", "Si", "Ti", "Zr", "Mg", "Zn", "B", "C", "Cu", "Fe", "Co", "Mn", "H", "N"),
        density=(0.1, 4.8),
        energy_above_hull=(0.0, 0.15),
        num_elements=(2, 5),
        band_gap_min=0.5,
        rationale="fast cycling rewards stable oxides/MOFs with conductive framework elements and moderate density",
    ),
    "human": SearchCriteria(
        label="low-cost HVAC adsorption",
        required_elements=("O",),
        allowed_elements=("Al", "Si", "P", "Mg", "Ca", "Zn", "Ti", "Zr", "Cu", "Fe", "C", "H", "N"),
        density=(0.1, 3.5),
        energy_above_hull=(0.0, 0.15),
        num_elements=(2, 5),
        band_gap_min=0.1,
        rationale="comfort cooling favors hydrophilic zeotypes and MOFs; allowing H/N/C and higher Ehull captures porous frameworks",
    ),
    "vehicle": SearchCriteria(
        label="rugged waste-heat cooling",
        required_elements=("O",),
        allowed_elements=("Al", "Si", "Ti", "Zr", "Mg", "Zn", "Ca", "B", "C", "Fe", "Mn", "H", "N"),
        density=(0.1, 4.8),
        energy_above_hull=(0.0, 0.12),
        num_elements=(2, 5),
        band_gap_min=0.5,
        rationale="vehicle systems can use hotter heat, so durability and compact cycling matter, but porous architectures are still needed",
    ),
    "datacenter": SearchCriteria(
        label="low-grade liquid-loop waste heat",
        required_elements=("O",),
        allowed_elements=("Al", "Si", "P", "Mg", "Ca", "Zn", "Cu", "Fe", "C", "H", "N"),
        density=(0.1, 3.0),
        energy_above_hull=(0.0, 0.15),
        num_elements=(2, 5),
        band_gap_min=0.1,
        rationale="data-center heat is low temperature, emphasizing ultra-porous MOFs and soft-frameworks that require H/N and high Ehull bounds",
    ),
}

TOXIC_OR_COSTLY_ELEMENTS = {"Cd", "Hg", "Pb", "As", "Tl", "Be", "U", "Th", "Re", "Os", "Ir", "Pt", "Au"}


def generate_chemsys(criteria: SearchCriteria) -> List[str]:
    required = tuple(dict.fromkeys(criteria.required_elements))
    optional = [el for el in criteria.allowed_elements if el not in required]
    min_elements, max_elements = criteria.num_elements
    systems = []

    for size in range(max(min_elements, len(required)), max_elements + 1):
        optional_count = size - len(required)
        for combo in itertools.combinations(optional, optional_count):
            systems.append("-".join(sorted((*required, *combo))))

    return systems


def fetch_candidates(
    api_key: Optional[str],
    search_criteria: Iterable[SearchCriteria],
    chemsys_override: Optional[Iterable[str]],
    limit_per_system: int,
    max_generated_chemsys: Optional[int],
) -> List[Dict]:
    if not api_key:
        raise RuntimeError("MP_API_KEY not found in .env or environment.")
    if not MPRester:
        raise RuntimeError("mp_api is not installed in this environment.")

    fields = [
        "material_id",
        "formula_pretty",
        "density",
        "energy_above_hull",
        "formation_energy_per_atom",
        "volume",
        "nsites",
        "elements",
        "band_gap",
    ]
    candidates_by_id: Dict[str, Dict] = {}

    with MPRester(api_key) as mpr:
        for criteria in search_criteria:
            systems = list(chemsys_override or generate_chemsys(criteria))
            if max_generated_chemsys is not None and chemsys_override is None:
                systems = systems[:max_generated_chemsys]
            print(f"Searching {criteria.label}: {criteria.rationale}")
            print(
                f"  MP filters: density={criteria.density}, "
                f"energy_above_hull={criteria.energy_above_hull}, "
                f"band_gap>={criteria.band_gap_min} eV, "
                f"num_elements={criteria.num_elements}, chemsys={len(systems)} systems"
            )

            for system in systems:
                docs = mpr.materials.summary.search(
                    chemsys=system,
                    density=criteria.density,
                    energy_above_hull=criteria.energy_above_hull,
                    band_gap=(criteria.band_gap_min, 100.0),
                    exclude_elements=list(TOXIC_OR_COSTLY_ELEMENTS),
                    fields=fields,
                    num_chunks=1,
                    chunk_size=limit_per_system,
                )
                for doc in docs:
                    material_id = str(doc.material_id)
                    row = candidates_by_id.setdefault(
                        material_id,
                        {
                            "material_id": material_id,
                            "formula": doc.formula_pretty,
                            "density": doc.density,
                            "energy_above_hull": doc.energy_above_hull,
                            "formation_energy_per_atom": doc.formation_energy_per_atom,
                            "volume": doc.volume,
                            "nsites": doc.nsites,
                            "elements": [str(el) for el in doc.elements],
                            "band_gap": doc.band_gap,
                            "chemical_system": system,
                            "search_labels": [],
                        },
                    )
                    row["search_labels"].append(criteria.label)

    return list(candidates_by_id.values())


def search_criteria_for_apps(app_keys: Iterable[str]) -> List[SearchCriteria]:
    criteria_by_label: Dict[str, SearchCriteria] = {}
    for app_key in app_keys:
        criteria = SEARCH_CRITERIA[app_key]
        criteria_by_label[criteria.label] = criteria
    return list(criteria_by_label.values())

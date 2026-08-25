import argparse
import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from cooling_physics import (
    clamp,
    normalize,
    pareto_closeness_score,
    pareto_target_window,
    simulate_adsorption_cycle,
)
from env_utils import get_mp_api_key

try:
    from mp_api.client import MPRester
except ImportError:
    MPRester = None


@dataclass(frozen=True)
class ApplicationProfile:
    name: str
    t_evap_c: float
    t_cond_c: float
    t_des_c: float
    cycle_time_sec: float
    cop_weight: float
    scp_weight: float
    stability_weight: float
    conductivity_weight: float
    density_weight: float
    pareto_weight: float
    notes: str


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


APPLICATIONS: Dict[str, ApplicationProfile] = {
    "cpu": ApplicationProfile(
        name="CPU / electronics cold plate assist",
        t_evap_c=18.0,
        t_cond_c=35.0,
        t_des_c=75.0,
        cycle_time_sec=120.0,
        cop_weight=0.20,
        scp_weight=0.45,
        stability_weight=0.15,
        conductivity_weight=0.15,
        density_weight=0.05,
        pareto_weight=0.15,
        notes="Needs very high specific cooling power and excellent bed heat transfer; practical use is more likely at rack/cold-plate scale than inside a chip package.",
    ),
    "human": ApplicationProfile(
        name="Human thermal comfort / HVAC",
        t_evap_c=10.0,
        t_cond_c=35.0,
        t_des_c=80.0,
        cycle_time_sec=600.0,
        cop_weight=0.40,
        scp_weight=0.20,
        stability_weight=0.20,
        conductivity_weight=0.05,
        density_weight=0.15,
        pareto_weight=0.15,
        notes="Best fit for solar thermal or waste-heat regenerated adsorption chillers using water refrigerant.",
    ),
    "vehicle": ApplicationProfile(
        name="Vehicle waste-heat cooling",
        t_evap_c=7.0,
        t_cond_c=45.0,
        t_des_c=120.0,
        cycle_time_sec=180.0,
        cop_weight=0.20,
        scp_weight=0.40,
        stability_weight=0.20,
        conductivity_weight=0.15,
        density_weight=0.05,
        pareto_weight=0.15,
        notes="Can exploit hotter exhaust/coolant heat, but compactness, vibration tolerance, and fast cycling dominate.",
    ),
    "datacenter": ApplicationProfile(
        name="Data-center waste-heat cooling",
        t_evap_c=16.0,
        t_cond_c=35.0,
        t_des_c=60.0,
        cycle_time_sec=300.0,
        cop_weight=0.35,
        scp_weight=0.30,
        stability_weight=0.20,
        conductivity_weight=0.10,
        density_weight=0.05,
        pareto_weight=0.20,
        notes="Hard low-grade heat case: candidates must regenerate from warm liquid loops around 45-70 C.",
    ),
}


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

CONDUCTIVE_ELEMENTS = {"Al", "B", "C", "Cu", "Zn", "Ti", "Zr", "Si", "Mg"}
HYDROPHILIC_ELEMENTS = {"Al", "P", "Si", "Ti", "Zr", "Mg", "Ca", "Zn", "O"}
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


def material_elements(candidate: Dict) -> List[str]:
    elements = candidate.get("elements") or []
    return [str(e) for e in elements]


def estimate_adsorption_properties(candidate: Dict) -> Dict[str, float]:
    density = float(candidate.get("density") or 3.0)
    volume = float(candidate.get("volume") or 0.0)
    nsites = int(candidate.get("nsites") or 1)
    volume_per_atom = volume / max(nsites, 1)
    elements = set(material_elements(candidate))

    low_density_bonus = normalize(3.0 - density, 0.0, 2.2)
    open_framework_bonus = normalize(volume_per_atom, 12.0, 38.0)
    hydrophilic_fraction = (
        len(elements & HYDROPHILIC_ELEMENTS) / max(len(elements), 1)
    )

    q_sat = 0.08 + 0.72 * (
        0.45 * low_density_bonus + 0.40 * open_framework_bonus + 0.15 * hydrophilic_fraction
    )
    q_sat = clamp(q_sat, 0.05, 0.85)

    q_st = 2.35e6 + 0.95e6 * hydrophilic_fraction + 0.35e6 * low_density_bonus
    if {"Al", "P", "O"}.issubset(elements) or {"Al", "Si", "O"}.issubset(elements):
        q_st += 0.20e6
    q_st = clamp(q_st, 2.3e6, 4.1e6)

    conductivity_proxy = 0.25 + 0.45 * (
        len(elements & CONDUCTIVE_ELEMENTS) / max(len(elements), 1)
    ) + 0.30 * normalize(density, 1.0, 4.5)
    toxicity_penalty = 0.25 if elements & TOXIC_OR_COSTLY_ELEMENTS else 0.0

    return {
        "q_sat": q_sat,
        "Q_st": q_st,
        "conductivity_proxy": clamp(conductivity_proxy),
        "toxicity_penalty": toxicity_penalty,
        "open_framework_proxy": clamp(0.55 * low_density_bonus + 0.45 * open_framework_bonus),
    }


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
        raise RuntimeError("mp_api is not installed. Try: .venv/bin/python heat_cooling_screen.py")

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


def discover_target_property_window(profile: ApplicationProfile) -> Dict[str, float]:
    return pareto_target_window(
        t_evap_c=profile.t_evap_c,
        t_cond_c=profile.t_cond_c,
        t_des_c=profile.t_des_c,
        cycle_time_sec=profile.cycle_time_sec,
        cop_weight=profile.cop_weight,
        scp_weight=profile.scp_weight,
    )


def score_candidate(
    candidate: Dict,
    profile: ApplicationProfile,
    target: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    props = estimate_adsorption_properties(candidate)
    cycle = simulate_adsorption_cycle(
        q_sat=props["q_sat"],
        q_st=props["Q_st"],
        t_evap_c=profile.t_evap_c,
        t_cond_c=profile.t_cond_c,
        t_des_c=profile.t_des_c,
        cycle_time_sec=profile.cycle_time_sec,
    )

    density = float(candidate.get("density") or 5.0)
    hull = float(candidate.get("energy_above_hull") or 0.2)
    stability_score = 1.0 - normalize(hull, 0.0, 0.08)
    density_score = 1.0 - normalize(density, 1.0, 4.0)
    cop_score = normalize(cycle["COP"], 0.05, 0.85)
    scp_score = normalize(cycle["SCP_W_kg"], 20.0, 1600.0)
    pareto_score = pareto_closeness_score(props["q_sat"], props["Q_st"], target)
    non_toxic_score = 1.0 - props["toxicity_penalty"]

    score = (
        profile.cop_weight * cop_score
        + profile.scp_weight * scp_score
        + profile.stability_weight * stability_score
        + profile.conductivity_weight * props["conductivity_proxy"]
        + profile.density_weight * density_score
        + profile.pareto_weight * pareto_score
        - props["toxicity_penalty"]
    )

    return {
        **props,
        **cycle,
        "score": max(0.0, score),
        "stability_score": stability_score,
        "density_score": density_score,
        "pareto_score": pareto_score,
        "non_toxic_score": non_toxic_score,
    }


def rank_for_application(
    candidates: List[Dict],
    profile: ApplicationProfile,
    target: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    ranked = []
    for candidate in candidates:
        metrics = score_candidate(candidate, profile, target)
        ranked.append({**candidate, **metrics})
    return sorted(ranked, key=lambda row: row["score"], reverse=True)


def find_materials_pareto_frontier(scored_candidates: List[Dict]) -> List[Dict]:
    objectives = [
        "pareto_score",
        "stability_score",
        "conductivity_proxy",
        "open_framework_proxy",
        "density_score",
        "non_toxic_score",
    ]
    frontier = []

    for candidate in scored_candidates:
        dominated = False
        for other in scored_candidates:
            if other is candidate:
                continue

            equal_or_better = all(other[key] >= candidate[key] for key in objectives)
            strictly_better = any(other[key] > candidate[key] for key in objectives)
            if equal_or_better and strictly_better:
                dominated = True
                break

        if not dominated:
            frontier.append(candidate)

    return sorted(frontier, key=lambda row: row["score"], reverse=True)


def print_candidate_table(rows: List[Dict], top: int) -> None:
    print(
        f"{'Rank':<5} {'ID':<14} {'Formula':<12} {'Score':<7} {'COP':<6} "
        f"{'SCP W/kg':<9} {'dq kg/kg':<9} {'q_sat':<7} {'Q_st MJ/kg':<10} {'Pscore':<7} {'rho':<5} {'Ehull':<6}"
    )
    print("-" * 122)
    for idx, row in enumerate(rows[:top], start=1):
        print(
            f"{idx:<5} {row['material_id']:<14} {row['formula']:<12} "
            f"{row['score']:<7.3f} {row['COP']:<6.3f} {row['SCP_W_kg']:<9.1f} "
            f"{row['delta_q']:<9.3f} {row['q_sat']:<7.3f} {row['Q_st'] / 1e6:<10.2f} "
            f"{row['pareto_score']:<7.3f} {float(row['density']):<5.2f} {float(row['energy_above_hull']):<6.3f}"
        )


def print_materials_frontier_table(rows: List[Dict], top: int) -> None:
    print(
        f"{'Rank':<5} {'ID':<14} {'Formula':<12} {'Pscore':<7} {'Stable':<7} "
        f"{'Cond':<6} {'Open':<6} {'Dense':<6} {'NonTox':<7} {'Score':<7}"
    )
    print("-" * 88)
    for idx, row in enumerate(rows[:top], start=1):
        print(
            f"{idx:<5} {row['material_id']:<14} {row['formula']:<12} "
            f"{row['pareto_score']:<7.3f} {row['stability_score']:<7.3f} "
            f"{row['conductivity_proxy']:<6.3f} {row['open_framework_proxy']:<6.3f} "
            f"{row['density_score']:<6.3f} {row['non_toxic_score']:<7.3f} {row['score']:<7.3f}"
        )


def print_rankings(candidates: List[Dict], app_keys: Iterable[str], top: int, frontier_top: int) -> None:
    for app_key in app_keys:
        profile = APPLICATIONS[app_key]
        target = discover_target_property_window(profile)
        ranked = rank_for_application(candidates, profile, target)
        materials_frontier = find_materials_pareto_frontier(ranked)
        print(f"\n=== {profile.name.upper()} ===")
        print(profile.notes)
        print(
            "Pareto target window: "
            f"q_sat={target['q_sat_min']:.2f}-{target['q_sat_max']:.2f} kg/kg, "
            f"Q_st={target['Q_st_min'] / 1e6:.2f}-{target['Q_st_max'] / 1e6:.2f} MJ/kg, "
            f"frontier={target['frontier_size']}/{target['sampled_points']}, "
            f"window_points={target['window_points']}"
        )
        print("\nWeighted ranking:")
        print_candidate_table(ranked, top)
        print(f"\nMaterials Pareto frontier ({len(materials_frontier)} non-dominated candidates):")
        print_materials_frontier_table(materials_frontier, frontier_top)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank Materials Project candidates for heat-driven cooling applications."
    )
    parser.add_argument(
        "--api-key",
        default=get_mp_api_key(),
        help="Materials Project API key. Defaults to MP_API_KEY from .env or the environment.",
    )
    parser.add_argument(
        "--apps",
        nargs="+",
        choices=sorted(APPLICATIONS),
        default=sorted(APPLICATIONS),
        help="Application profiles to rank.",
    )
    parser.add_argument(
        "--chemsys",
        nargs="+",
        default=None,
        help="Optional manual chemical systems. If omitted, systems are generated from application search criteria.",
    )
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--frontier-top", type=int, default=10)
    parser.add_argument("--limit-per-system", type=int, default=200)
    parser.add_argument(
        "--max-generated-chemsys",
        type=int,
        default=None,
        help="Cap generated application-derived chemical systems for quick exploratory runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    candidates = fetch_candidates(
        api_key=args.api_key,
        search_criteria=search_criteria_for_apps(args.apps),
        chemsys_override=args.chemsys,
        limit_per_system=args.limit_per_system,
        max_generated_chemsys=args.max_generated_chemsys,
    )
    print(f"Loaded {len(candidates)} candidate materials.")
    print_rankings(candidates, args.apps, args.top, args.frontier_top)


if __name__ == "__main__":
    main()

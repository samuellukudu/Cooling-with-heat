"""Composition features without pymatgen/matminer (this env has neither).

Two formula sources, both curated or parsed — never guessed:

1. **Parsed**: ISODB names that ARE formulas (``C115.5H202N14O43Zn4``).
   Tokenizer handles decimals; parentheses/solvent adducts fail loud (None).
2. **Class table** (``CLASS_FORMULAS``): standard-material stoichiometries
   with confidence tiers (zeolite unit cells, SiO2, carbons, anchor MOFs
   pre-expanded — no parens for the parser to choke on).

Descriptors are Magpie-lite: stoichiometry-weighted mean/std/min/max over
(mass, Pauling EN, covalent radius) + n_elements + metal fraction. Pore
geometry (LCD/PLD/ASA/AV) joins from CoRE in ``descriptors.py`` — explicit
engineering features per the GeoField lesson, not rediscovered ones.
"""

from __future__ import annotations

import re

# Element: (atomic mass, Pauling EN, covalent radius pm). Common
# porous-material elements; absent elements fail loudly in featurize().
ELEMENTS: dict[str, tuple[float, float, float]] = {
    "H": (1.008, 2.20, 31), "B": (10.81, 2.04, 84),
    "C": (12.011, 2.55, 76), "N": (14.007, 3.04, 71),
    "O": (15.999, 3.44, 66), "F": (18.998, 3.98, 57),
    "Na": (22.990, 0.93, 166), "Mg": (24.305, 1.31, 141),
    "Al": (26.982, 1.61, 121), "Si": (28.085, 1.90, 111),
    "P": (30.974, 2.19, 107), "S": (32.06, 2.58, 105),
    "Cl": (35.45, 3.16, 102), "K": (39.098, 0.82, 203),
    "Ca": (40.078, 1.00, 176), "Cr": (51.996, 1.66, 139),
    "Mn": (54.938, 1.55, 150), "Fe": (55.845, 1.83, 132),
    "Co": (58.933, 1.88, 126), "Ni": (58.693, 1.91, 124),
    "Cu": (63.546, 1.90, 132), "Zn": (65.38, 1.65, 122),
    "Br": (79.904, 2.96, 120), "Zr": (91.224, 1.33, 175),
    "Cd": (112.411, 1.69, 144), "Eu": (151.964, 1.20, 198),
    "Tb": (158.925, 1.10, 194), "Dy": (162.50, 1.22, 192),
    "Ho": (164.930, 1.23, 192), "Er": (167.259, 1.24, 189),
    "Tm": (168.934, 1.25, 190), "W": (183.84, 2.36, 162),
    "Li": (6.94, 0.98, 128),
}
METALS = frozenset({"Li", "Na", "Mg", "Al", "K", "Ca", "Cr", "Mn", "Fe",
                    "Co", "Ni", "Cu", "Zn", "Zr", "Cd", "Eu", "Tb", "Dy",
                    "Ho", "Er", "Tm", "W"})

_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")


def parse_formula(formula: str) -> dict[str, float]:
    """``'C6H12O6' -> {'C': 6, ...}``. Full-string match or ValueError."""
    s = formula.strip().replace("·", "").replace(" ", "")
    if not s or _TOKEN.sub("", s) != "":
        raise ValueError(f"cannot parse formula {formula!r} "
                         "(parens/adducts unsupported — curate pre-expanded)")
    counts: dict[str, float] = {}
    for el, num in _TOKEN.findall(s):
        if el not in ELEMENTS:
            raise ValueError(f"element {el!r} not in property table")
        counts[el] = counts.get(el, 0.0) + (float(num) if num else 1.0)
    if not counts:
        raise ValueError(f"cannot parse formula {formula!r}")
    return counts


def looks_like_formula(name: str) -> bool:
    """Names that are bare element-count strings (C…H… with digits).

    Every element token must be in the property table — 'M1' or 'AOP-CH2'
    may match the shape but are not compositions.
    """
    s = name.strip()
    if not (re.fullmatch(r"(?:[A-Z][a-z]?\d*\.?\d*)+", s) and any(c.isdigit() for c in s)):
        return False
    return all(el in ELEMENTS for el, _ in _TOKEN.findall(s))


# (name-substring, formula, confidence). Ordered: anchors/exact first.
CLASS_FORMULAS: tuple[tuple[str, str, str], ...] = (
    ("13x", "Na86Al86Si106O384", "high"),
    ("zeolite y", "Na56Al56Si136O384", "medium"),
    ("naa", "Na12Al12Si12O48", "high"),
    ("zeolite a", "Na12Al12Si12O48", "high"),
    ("zeolite 4a", "Na12Al12Si12O48", "high"),
    ("zeolite 5a", "Ca4.5Na3Al12Si12O48", "medium"),
    ("zsm", "SiO2", "medium"), ("silicalite", "SiO2", "high"),
    ("hisiv", "SiO2", "medium"), ("mfi", "SiO2", "medium"),
    ("chabazite", "Ca2Al4Si8O24", "medium"), ("sapo-34", "SiAl11P12O48", "low"),
    ("alpo-18", "AlPO4", "medium"),
    ("silica", "SiO2", "high"), ("xtrusorb", "SiO2", "medium"),
    ("cab-", "SiO2", "medium"),
    ("zif-8", "ZnC8H10N4", "high"),
    ("uio-66", "Zr6C48H28O32", "high"),
    ("mg-mof-74", "Mg2C8H2O6", "high"), ("mof-74-ni", "Ni2C8H2O6", "high"),
    ("zn-mof-74", "Zn2C8H2O6", "high"),
    ("mil-101", "Cr3C24H16O15F", "medium"),
    ("mil-100", "Cr3C18H10O15F", "low"),
    ("cau-10", "AlC8H5O5", "medium"),
    ("aluminum fumarate", "AlC4H3O5", "medium"),
)


def formula_for(name: str, family: str) -> tuple[str | None, str]:
    """(formula, provenance ∈ {parsed, class, none}). Carbons fall back to C."""
    if looks_like_formula(name):
        return name.strip(), "parsed"
    n = name.lower()
    for key, formula, _conf in CLASS_FORMULAS:
        if key in n:
            return formula, "class"
    if family == "carbon":
        return "C", "class"
    return None, "none"


STAT_PROPS = ("mass", "en", "radius")


def featurize_composition(counts: dict[str, float]) -> dict[str, float]:
    """Magpie-lite stats; stoichiometry-weighted. All finite by construction."""
    import numpy as np  # noqa: PLC0415

    total = sum(counts.values())
    frac = {el: c / total for el, c in counts.items()}
    out: dict[str, float] = {"n_elements": float(len(counts)),
                             "metal_frac": sum(f for el, f in frac.items() if el in METALS)}
    for i, prop in enumerate(STAT_PROPS):
        vals = np.array([ELEMENTS[el][i] for el in counts])
        w = np.array([frac[el] for el in counts])
        mean = float(np.sum(vals * w))
        out[f"mean_{prop}"] = mean
        out[f"std_{prop}"] = float(np.sqrt(np.sum(w * (vals - mean) ** 2)))
        out[f"min_{prop}"] = float(vals.min())
        out[f"max_{prop}"] = float(vals.max())
    return out


COMPOSITION_COLUMNS = (["n_elements", "metal_frac"]
                       + [f"{s}_{p}" for p in STAT_PROPS for s in ("mean", "std", "min", "max")])


__all__ = ["CLASS_FORMULAS", "COMPOSITION_COLUMNS", "ELEMENTS", "METALS",
           "featurize_composition", "formula_for", "looks_like_formula",
           "parse_formula"]

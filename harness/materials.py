"""Material parameters as data, not code (``DESIGN.md`` §7.3, §8.1).

A material is a ``MaterialParams`` row. Built-ins ship from the curated
anchor table in ``adsorbent-ml/data/anchors.csv`` (13 commercial adsorbents,
source tag ``anchor``); fitted database rows arrive through
``load_materials_csv`` using the §8.1 schema that ``adsorbent-ml/fit_da.py``
writes. Adding a material never requires code.

Honesty rule (§8.1): equilibrium parameters come from data; transport
properties may be class-defaults — always report ``transport_provenance``
alongside any SCP that depends on them.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

from .registry import REGISTRIES

# Contract with adsorbent-ml/fit_da.py (DESIGN §8.1). Columns marked
# optional may be absent; absent transport properties stay None rather than
# being silently filled with defaults.
REQUIRED_COLUMNS = (
    "material_id",
    "name",
    "source",
    "q_sat_kg_kg",
    "q_st_j_kg",
    "e_char_j_mol",
    "n_da",
)
OPTIONAL_FLOAT_COLUMNS = (
    "k_ldf_s_1",
    "rho_kg_m3",
    "cp_j_kg_k",
    "k_eff_w_m_k",
    "fit_rmse",
)
ANCHORS_PATH = Path(__file__).resolve().parents[1] / "adsorbent-ml" / "data" / "anchors.csv"


@dataclass(frozen=True)
class MaterialParams:
    name: str
    source: str  # isodb | core_mof | qmof | iza | anchor | custom
    q_sat_kg_kg: float
    q_st_j_kg: float | None  # None when the source data has single-T coverage
    e_char_j_mol: float
    n_da: float
    material_id: str = ""
    t_range_c: tuple[float, float] = (0.0, 200.0)
    material_class: str = ""
    k_ldf_s_1: float | None = None
    rho_kg_m3: float | None = None
    cp_j_kg_k: float | None = None
    k_eff_w_m_k: float | None = None
    fit_rmse: float | None = None
    n_points: int | None = None
    confidence: str = ""
    notes: str = ""
    transport_provenance: str = "none"

    @property
    def key(self) -> str:
        """Registry key: ``{source}:{name}`` (DESIGN §7.3)."""
        return f"{self.source}:{self.name}"

    def with_transport_defaults(
        self,
        *,
        rho_kg_m3: float,
        cp_j_kg_k: float,
        k_eff_w_m_k: float,
        provenance: str = "default",
    ) -> "MaterialParams":
        """Fill missing transport properties with class defaults — never
        silently: the returned row is flagged via ``transport_provenance``."""
        return replace(
            self,
            rho_kg_m3=self.rho_kg_m3 if self.rho_kg_m3 is not None else rho_kg_m3,
            cp_j_kg_k=self.cp_j_kg_k if self.cp_j_kg_k is not None else cp_j_kg_k,
            k_eff_w_m_k=self.k_eff_w_m_k if self.k_eff_w_m_k is not None else k_eff_w_m_k,
            transport_provenance=provenance,
        )


def _parse_t_range(raw: str) -> tuple[float, float]:
    parts = str(raw).replace("–", "-").split("-")
    if len(parts) != 2:
        raise ValueError(f"cannot parse temperature range {raw!r} as 'lo-hi'")
    return float(parts[0]), float(parts[1])


def _float_or_none(row: Mapping[str, str], key: str) -> float | None:
    raw = (row.get(key) or "").strip()
    return float(raw) if raw else None


def load_anchors(path: str | Path = ANCHORS_PATH) -> list[MaterialParams]:
    """Load the curated commercial-adsorbent anchor table.

    The anchor CSV is curated by hand (not a fit_da export): it carries
    ``q_st_MJ_kg`` and a ``class`` column, and its ``source`` column holds
    literature citations rather than a database tag — rows get source
    ``anchor``.
    """
    rows: list[MaterialParams] = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                MaterialParams(
                    name=row["name"].strip(),
                    source="anchor",
                    material_class=row.get("class", "").strip(),
                    q_sat_kg_kg=float(row["q_sat_kg_kg"]),
                    q_st_j_kg=float(row["q_st_MJ_kg"]) * 1e6,
                    e_char_j_mol=float(row["e_char_J_mol"]),
                    n_da=float(row["n_da"]),
                    t_range_c=_parse_t_range(row["t_range_C"]),
                    confidence=row.get("confidence", "").strip(),
                    notes=row.get("notes", "").strip(),
                )
            )
    return rows


def load_materials_csv(path: str | Path) -> list[MaterialParams]:
    """Load fitted adsorbent rows in the §8.1 ``fit_da.py`` output schema.

    Column headers in ``REQUIRED_COLUMNS`` must exist; *values* may be
    empty for optional quantities. Rows missing ``q_sat_kg_kg``,
    ``e_char_j_mol`` or ``n_da`` (e.g. flagged fits) cannot serve the cycle
    physics and are skipped; ``q_st_j_kg`` is optional (empty ⇒ None —
    single-temperature isotherms carry no isosteric heat) and must be
    checked by consumers.
    """
    skipped = 0
    rows_out: list[MaterialParams] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing required column(s) {missing} (DESIGN §8.1 schema)")
        for row in reader:
            q_sat = _float_or_none(row, "q_sat_kg_kg")
            e_char = _float_or_none(row, "e_char_j_mol")
            n_da = _float_or_none(row, "n_da")
            if q_sat is None or e_char is None or n_da is None:
                skipped += 1
                continue
            n_points_raw = (row.get("n_points") or "").strip()
            rows_out.append(
                MaterialParams(
                    material_id=row["material_id"].strip(),
                    name=row["name"].strip(),
                    source=row["source"].strip(),
                    q_sat_kg_kg=q_sat,
                    q_st_j_kg=_float_or_none(row, "q_st_j_kg"),
                    e_char_j_mol=e_char,
                    n_da=n_da,
                    t_range_c=_parse_t_range(row["t_range_c"]) if (row.get("t_range_c") or "").strip() else (0.0, 200.0),
                    k_ldf_s_1=_float_or_none(row, "k_ldf_s_1"),
                    rho_kg_m3=_float_or_none(row, "rho_kg_m3"),
                    cp_j_kg_k=_float_or_none(row, "cp_j_kg_k"),
                    k_eff_w_m_k=_float_or_none(row, "k_eff_w_m_k"),
                    fit_rmse=_float_or_none(row, "fit_rmse"),
                    n_points=int(float(n_points_raw)) if n_points_raw else None,
                    transport_provenance="fit" if (row.get("k_eff_w_m_k") or "").strip() else "none",
                )
            )
    if skipped:
        print(f"[harness.materials] {path}: skipped {skipped} row(s) missing q_sat/e_char/n_da")
    return rows_out


def register_materials(materials: list[MaterialParams], *, overwrite: bool = False) -> None:
    for params in materials:
        REGISTRIES["materials"].register(params.key, lambda p=params: p, overwrite=overwrite)


def get_material(key_or_params: "str | MaterialParams") -> MaterialParams:
    """Resolve a registry key (``{source}:{name}``) or pass an instance through."""
    if isinstance(key_or_params, MaterialParams):
        return key_or_params
    factory = REGISTRIES["materials"].resolve(str(key_or_params))
    value = factory()
    if not isinstance(value, MaterialParams):
        raise TypeError(f"harness.materials factory for {key_or_params!r} returned {type(value).__name__}")
    return value


def _register_builtins() -> None:
    try:
        anchors = load_anchors()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"built-in anchor table not found at {ANCHORS_PATH}; the harness "
            "package expects the repository layout (adsorbent-ml/data/anchors.csv)"
        ) from exc
    register_materials(anchors)


_register_builtins()

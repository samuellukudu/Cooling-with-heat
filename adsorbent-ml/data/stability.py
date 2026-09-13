"""Stability-label ingest: water/thermal feasibility filter for candidates.

Companion to ``adsorbent-ml/data/ACQUISITION.md`` §6 (MOFSimplify + TGA
compilations). Role: N2 must not propose adsorbents that collapse under
water cycling or decompose below regeneration temperature — these labels
are the gate BEFORE system-level ranking, not a training target.

Schema (one row per material, CSV built by ``stability_export.py`` from the
MOFSimplify Zenodo record, or hand-curated per ACQUISITION §6):

    mof_name, water_stable, solvent_removal_stable, thermal_decomp_c, source, confidence

- ``water_stable``: ``true/false`` (accepts 1/0/yes/no); empty ⇒ None
  (unknown — caller decides, never silently treated as stable).
- ``solvent_removal_stable``: MOFSimplify's text-mined label (does the MOF
  survive desolvation). Conservative proxy for water-cycling service:
  unknown water + solvent-stable counts as stable, an explicit
  ``water_stable=false`` never does.
- ``thermal_decomp_c``: TGA decomposition onset [°C]; empty ⇒ None.
- Header aliases accepted for hand-curated tables (documented below).

Pure stdlib. Importable without network (fit_da precedent).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

NAME_KEYS = ("mof_name", "name", "mof", "refcode")
WATER_KEYS = ("water_stable", "water_stability", "hydrolytically_stable")
SOLVENT_KEYS = ("solvent_removal_stable", "solvent_removal_stability",
                "solvent_stable")
DECOMP_KEYS = ("thermal_decomp_c", "decomposition_T_C", "decomp_c", "tga_decomp_c")
SOURCE_KEYS = ("source", "doi", "reference")
CONF_KEYS = ("confidence",)


def _pick(row: dict, keys: tuple[str, ...]) -> str:
    for k in keys:
        if k in row and (row[k] or "").strip():
            return row[k].strip()
    return ""


def _parse_bool(raw: str) -> bool | None:
    if not raw:
        return None
    v = raw.strip().lower()
    if v in ("true", "1", "yes", "stable"):
        return True
    if v in ("false", "0", "no", "unstable", "collapsed"):
        return False
    raise ValueError(f"cannot parse water-stability flag {raw!r} "
                     "(expected true/false/1/0/yes/no/stable/unstable)")


@dataclass(frozen=True)
class StabilityRecord:
    name: str
    water_stable: bool | None
    thermal_decomp_c: float | None
    solvent_removal_stable: bool | None = None
    source: str = ""
    confidence: str = ""


def load_stability_csv(path: str | Path) -> list[StabilityRecord]:
    """Load a stability table; rows without a name are skipped (counted)."""
    out: list[StabilityRecord] = []
    skipped = 0
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = set(reader.fieldnames or [])
        if not fields & set(NAME_KEYS):
            raise ValueError(f"{path}: no name column {NAME_KEYS} in {sorted(fields)}")
        for row in reader:
            name = _pick(row, NAME_KEYS)
            if not name:
                skipped += 1
                continue
            decomp_raw = _pick(row, DECOMP_KEYS)
            out.append(StabilityRecord(
                name=name,
                water_stable=_parse_bool(_pick(row, WATER_KEYS)),
                thermal_decomp_c=float(decomp_raw) if decomp_raw else None,
                solvent_removal_stable=_parse_bool(_pick(row, SOLVENT_KEYS)),
                source=_pick(row, SOURCE_KEYS),
                confidence=_pick(row, CONF_KEYS)))
    if skipped:
        print(f"[stability] {path}: skipped {skipped} nameless row(s)")
    return out


def feasibility_filter(records: list[StabilityRecord], *,
                       require_water_stable: bool = True,
                       min_decomp_c: float = 150.0) -> tuple[list[StabilityRecord],
                                                            list[StabilityRecord]]:
    """Split ``(pass, fail)``. Unknown water stability FAILS when
    ``require_water_stable`` — unless the MOFSimplify solvent-removal label
    says the MOF survives desolvation (conservative evidence for water
    cycling; an explicit ``water_stable=false`` never passes).
    ``thermal_decomp_c=None`` never fails alone (absence of TGA data is not
    decomposition)."""
    ok, bad = [], []
    for r in records:
        if not require_water_stable:
            water_ok = r.water_stable is not False
        elif r.water_stable is True:
            water_ok = True
        elif r.water_stable is False:
            water_ok = False
        else:
            water_ok = r.solvent_removal_stable is True
        decomp_ok = r.thermal_decomp_c is None or r.thermal_decomp_c >= min_decomp_c
        (ok if (water_ok and decomp_ok) else bad).append(r)
    return ok, bad


__all__ = ["StabilityRecord", "feasibility_filter", "load_stability_csv"]

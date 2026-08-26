#!/usr/bin/env python
"""Export the pure-water isotherm subset of the NIST ISODB git mirror.

Scans ``data_cache/isodb/mirror`` (clone of NIST-ISODB/isodb-library),
selects single-adsorbate water isotherms, and writes:

    <out>/water_isotherms.parquet   one row per isotherm (arrays preserved,
                                    original units — normalization happens
                                    downstream in fit_da.py)
    <out>/adsorbent_index.csv       every water adsorbent name/hashkey,
                                    ranked by isotherm count (input to the
                                    name->structure matching task)
    <out>/manifest.json             counts, filters, durations

Usage:
    python nist_isodb.py                       # defaults below
    python nist_isodb.py --mirror path/to/isodb-library --out path/to/out
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MIRROR = REPO_ROOT / "data_cache" / "isodb" / "mirror"
DEFAULT_OUT = REPO_ROOT / "data_cache" / "isodb"

# Standard InChIKey of water as used by ISODB records.
WATER_INCHIKEY = "XLYOFNOQVPJJNP-UHFFFAOYSA-N"
WATER_NAME = "water"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mirror", type=Path, default=DEFAULT_MIRROR,
                   help="Path to the cloned isodb-library repo.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="Output directory.")
    p.add_argument("--t-min", type=float, default=280.0,
                   help="Application temperature window lower bound [K].")
    p.add_argument("--t-max", type=float, default=380.0,
                   help="Application temperature window upper bound [K].")
    return p.parse_args()


def iter_isotherm_files(mirror: Path):
    """Yield *.json isotherm files under Library/, case-insensitively.

    The mirror mixes naming conventions (`...isotherm2.json`,
    `...Isotherm4.json`, even misspelled `...isothem10.json`); filename-based
    filters silently drop records. Match on directory layout instead, but
    skip the registry subdirectories (citation/material/adsorbate records,
    not isotherms).
    """
    library = mirror / "Library"
    skip = {"Bibliography", "Adsorbents", "Adsorbates"}
    for path in sorted(library.glob("*/*.json")):
        if path.parent.name not in skip:
            yield path


def scan(mirror: Path, t_min: float, t_max: float) -> dict:
    """Single pass over the mirror; returns summary + pure-water rows."""
    rows = []
    stats = {
        "total": 0, "unreadable": 0, "pure_water": 0, "mixed_water": 0,
        "not_water": 0,
    }
    adsorbents = Counter()   # hashkey -> count

    for path in iter_isotherm_files(mirror):
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception:
            stats["unreadable"] += 1
            continue

        stats["total"] += 1
        adsorbates = d.get("adsorbates") or []
        keys = {a.get("InChIKey", "").upper() for a in adsorbates}
        names = {str(a.get("name", "")).strip().lower() for a in adsorbates}
        is_water = WATER_INCHIKEY in keys or WATER_NAME in names

        if not is_water:
            stats["not_water"] += 1
            continue

        hk = (d.get("adsorbent") or {}).get("hashkey", "")
        if len(adsorbates) > 1:
            stats["mixed_water"] += 1
            continue

        stats["pure_water"] += 1
        adsorbents[hk] += 1

        points = d.get("isotherm_data") or []
        pressures, uptakes = [], []
        for pt in points:
            try:
                pressures.append(float(pt["pressure"]))
                uptakes.append(float(pt["total_adsorption"]))
            except (KeyError, TypeError, ValueError):
                continue

        temp = d.get("temperature")
        rows.append({
            "filename": d.get("filename") or path.stem,
            "doi": d.get("DOI", ""),
            "adsorbent_name": (d.get("adsorbent") or {}).get("name", ""),
            "adsorbent_hashkey": hk,
            "temperature_K": float(temp) if temp is not None else None,
            "in_temp_window": bool(temp is not None and t_min <= float(temp) <= t_max),
            "n_points": len(pressures),
            "pressure": pressures,
            "uptake": uptakes,
            "pressure_units": d.get("pressureUnits", ""),
            "uptake_units": d.get("adsorptionUnits", ""),
            "isotherm_type": d.get("isotherm_type", ""),
            "source_path": str(path.relative_to(mirror)),
        })

    return {"rows": rows, "stats": stats, "adsorbents": adsorbents}


def main() -> None:
    cli = parse_args()

    if not (cli.mirror / "Library").exists():
        raise SystemExit(
            f"Mirror not found at {cli.mirror}/Library.\n"
            f"Clone it first:\n  git clone --depth 1 "
            f"https://github.com/NIST-ISODB/isodb-library.git {cli.mirror}"
        )

    t0 = time.time()
    result = scan(cli.mirror, cli.t_min, cli.t_max)
    rows, stats, adsorbents = result["rows"], result["stats"], result["adsorbents"]

    import pandas as pd

    cli.out.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    table_path = cli.out / "water_isotherms.parquet"
    try:
        df.to_parquet(table_path, index=False)
    except Exception as exc:
        print(f"[warn] parquet failed ({exc}); writing CSV instead.")
        table_path = cli.out / "water_isotherms.csv"
        df.to_csv(table_path, index=False)

    idx = (
        pd.DataFrame(
            [(hk, cnt) for hk, cnt in adsorbents.most_common()],
            columns=["hashkey", "n_isotherms"],
        )
    )
    # attach display name (most common name seen for each hashkey)
    name_by_hk = {}
    for r in rows:
        name_by_hk.setdefault(r["adsorbent_hashkey"], r["adsorbent_name"])
    idx.insert(1, "name", idx["hashkey"].map(name_by_hk))
    idx_path = cli.out / "adsorbent_index.csv"
    idx.to_csv(idx_path, index=False)

    manifest = {
        "stage": "nist_isodb water-subset v1",
        "mirror": str(cli.mirror),
        "filters": {
            "adsorbate": "pure Water (single adsorbate)",
            "temp_window_K": [cli.t_min, cli.t_max],
        },
        **stats,
        "unique_adsorbents": len(adsorbents),
        "rows_in_temp_window": int(df["in_temp_window"].sum()) if len(df) else 0,
        "outputs": {"table": table_path.name, "index": idx_path.name},
        "duration_sec": round(time.time() - t0, 1),
    }
    (cli.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"scanned            : {stats['total']} isotherms "
          f"({stats['unreadable']} unreadable)")
    print(f"pure-water isotherms: {stats['pure_water']} "
          f"(in {cli.t_min}-{cli.t_max} K window: {manifest['rows_in_temp_window']})")
    print(f"unique adsorbents   : {len(adsorbents)}")
    print(f"wrote {table_path}")
    print(f"wrote {idx_path}")
    print(f"wrote {cli.out / 'manifest.json'}")

    if len(idx):
        print("\ntop unmatched-candidate names (by isotherm count):")
        for _, r in idx.head(10).iterrows():
            print(f"  {r['n_isotherms']:>4}  {r['name'][:55]}")


if __name__ == "__main__":
    main()

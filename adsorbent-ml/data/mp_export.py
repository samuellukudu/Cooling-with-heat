#!/usr/bin/env python
"""Stage-0 dataset export: query Materials Project once, cache locally forever.

Wraps the proven query layer in ``Materials/heat_cooling_screen.py`` (chemsys
generation, application search criteria, dedup) and writes everything needed
for ML work into a local cache directory:

    <out>/
      candidates.parquet   one row per material (falls back to .csv)
      structures/<mpid>.cif  optional relaxed-converged structures (opt-in, capped, resumable)
      manifest.json          args, counts, durations — every build is reproducible

Re-running skips nothing silently: metadata is rebuilt from the same query,
structure downloads resume (existing CIFs are kept), and the manifest records
what happened. The goal is to never re-fire the full chemsys sweep again.

Usage (smoke test):
    python mp_export.py --apps datacenter --max-generated-chemsys 3 \
        --limit-per-system 5 --structure-limit 5

Full export of one application:
    python mp_export.py --apps human
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MATERIALS_DIR = REPO_ROOT / "Materials"
sys.path.insert(0, str(MATERIALS_DIR))

from env_utils import get_mp_api_key, load_dotenv  # noqa: E402

try:
    from mp_api.client import MPRester
except ImportError:
    MPRester = None

from heat_cooling_screen import (  # noqa: E402
    fetch_candidates,
    search_criteria_for_apps,
)

DEFAULT_OUT = REPO_ROOT / "data_cache" / "mp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Materials Project screening candidates to a local ML dataset cache.",
    )
    parser.add_argument(
        "--apps",
        nargs="+",
        choices=["cpu", "human", "vehicle", "datacenter"],
        default=["datacenter"],
        help="Application profiles whose generated chemical systems are queried.",
    )
    parser.add_argument(
        "--chemsys",
        nargs="+",
        default=None,
        help="Manual chemical-system override (skips app-derived generation).",
    )
    parser.add_argument("--limit-per-system", type=int, default=200)
    parser.add_argument(
        "--max-generated-chemsys",
        type=int,
        default=None,
        help="Cap on generated systems per profile (smoke tests).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Cache directory (default: {DEFAULT_OUT}).",
    )
    parser.add_argument(
        "--with-structures",
        action="store_true",
        help="Also download crystal structures as CIF files.",
    )
    parser.add_argument(
        "--structure-limit",
        type=int,
        default=0,
        help="Max structures to download this run (0 = no cap). Existing CIFs are never re-downloaded.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="MP API key; defaults to MP_API_KEY from Materials/.env or environment.",
    )
    return parser.parse_args()


def rows_from_candidates(candidates: list[dict]) -> list[dict]:
    """Flatten API-shaped candidate dicts into flat table rows."""
    rows = []
    for cand in candidates:
        nsites = int(cand.get("nsites") or 0)
        volume = float(cand.get("volume") or 0.0)
        elements = sorted(cand.get("elements") or [])
        rows.append(
            {
                "material_id": cand["material_id"],
                "formula": cand["formula"],
                "chemical_system": cand.get("chemical_system", ""),
                "elements": " ".join(elements),
                "n_elements": len(elements),
                "density": cand.get("density"),
                "energy_above_hull": cand.get("energy_above_hull"),
                "formation_energy_per_atom": cand.get("formation_energy_per_atom"),
                "band_gap": cand.get("band_gap"),
                "volume": volume,
                "nsites": nsites,
                "volume_per_atom": volume / nsites if nsites else None,
                "search_labels": ";".join(sorted(set(cand.get("search_labels", [])))),
            }
        )
    return rows


def write_table(rows: list[dict], out_dir: Path) -> Path:
    import pandas as pd

    df = pd.DataFrame(rows).sort_values("material_id").reset_index(drop=True)
    parquet_path = out_dir / "candidates.parquet"
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception as exc:  # pragma: no cover - depends on optional deps
        print(f"  [warn] parquet failed ({exc}); writing CSV instead.")
        csv_path = out_dir / "candidates.csv"
        df.to_csv(csv_path, index=False)
        return csv_path


def download_structures(
    api_key: str,
    material_ids: list[str],
    structure_dir: Path,
    limit: int,
) -> dict:
    """Download CIFs with resume support; returns download statistics."""
    from pymatgen.io.cif import CifWriter

    pending = [
        mid for mid in material_ids if not (structure_dir / f"{mid}.cif").exists()
    ]
    if limit and len(pending) > limit:
        print(f"  Capping structure downloads at {limit} of {len(pending)} remaining.")
        pending = pending[:limit]

    already = len(material_ids) - len(pending)
    downloaded, failed = 0, []
    t0 = time.time()

    if pending:
        with MPRester(api_key) as mpr:
            for i, mid in enumerate(pending, start=1):
                try:
                    struct = mpr.get_structure_by_material_id(mid)
                    CifWriter(struct).write_file(str(structure_dir / f"{mid}.cif"))
                    downloaded += 1
                except Exception as exc:
                    failed.append({"material_id": mid, "error": str(exc)})
                if i % 25 == 0 or i == len(pending):
                    rate = i / max(time.time() - t0, 1e-9)
                    print(f"  structures {i}/{len(pending)} ({rate:.1f}/s)")

    return {
        "requested_total": len(material_ids),
        "already_cached": already,
        "downloaded_this_run": downloaded,
        "failed": failed,
        "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    args = parse_args()
    load_dotenv(str(MATERIALS_DIR / ".env"))
    api_key = args.api_key or get_mp_api_key()
    if not api_key:
        sys.exit("MP_API_KEY not found (set it in Materials/.env or pass --api-key).")
    if MPRester is None:
        sys.exit("mp_api is not installed in this environment.")

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    run_started = time.time()

    print(f"Fetching candidates for apps: {', '.join(args.apps)}")
    candidates = fetch_candidates(
        api_key=api_key,
        search_criteria=search_criteria_for_apps(args.apps),
        chemsys_override=args.chemsys,
        limit_per_system=args.limit_per_system,
        max_generated_chemsys=args.max_generated_chemsys,
    )
    print(f"Fetched {len(candidates)} unique candidate materials.")

    table_path = write_table(rows_from_candidates(candidates), out_dir)
    print(f"Wrote {table_path}")

    structure_stats = None
    if args.with_structures or args.structure_limit:
        structure_dir = out_dir / "structures"
        structure_dir.mkdir(exist_ok=True)
        material_ids = [c["material_id"] for c in candidates]
        structure_stats = download_structures(
            api_key, material_ids, structure_dir, args.structure_limit,
        )
        cached_now = sum(1 for _ in structure_dir.glob("*.cif"))
        print(
            f"Structures on disk: {cached_now} "
            f"(downloaded now: {structure_stats['downloaded_this_run']}, "
            f"failed: {len(structure_stats['failed'])})"
        )

    manifest = {
        "stage": "mp_export v1",
        "args": {k: str(v) for k, v in vars(args).items()},
        "n_candidates": len(candidates),
        "table": table_path.name,
        "structures": structure_stats,
        "duration_sec": round(time.time() - run_started, 1),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {manifest_path}")
    print("Done.")


if __name__ == "__main__":
    main()

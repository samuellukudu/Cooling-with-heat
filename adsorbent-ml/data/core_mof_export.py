#!/usr/bin/env python
"""Export the CoRE MOF 2019 database (source #2 of ACQUISITION.md).

Uses the official `CoRE-MOF` pip package and writes:

    <out>/properties.parquet     pore metrics (LCD, PLD, ASA, AV), metals,
                                 OMS flags, DOIs — one row per structure
    <out>/structures/<name>.cif  original CIF bytes, resumable downloads
    <out>/manifest.json          counts, dataset, duration

Resumable: existing CIFs are never re-extracted, so an interrupted run
continues where it stopped.

Usage:
    python core_mof_export.py                       # full export (~12k)
    python core_mof_export.py --limit 50            # smoke test
    python core_mof_export.py --dataset 2019-FSR
"""

import argparse
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "data_cache" / "core_mof"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", default="2019-ASR",
                   choices=["2014", "2019-ASR", "2019-FSR"],
                   help="CoRE MOF subset (default: 2019-ASR = all solvent removed).")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--limit", type=int, default=0,
                   help="Cap on structures exported this run (0 = all).")
    p.add_argument("--no-structures", action="store_true",
                   help="Export the property table only.")
    return p.parse_args()


def bulk_extract_structures(dataset: str, names: list[str], sdir: Path) -> tuple[int, int]:
    """Fast path: stream the package's bundled tar.xz once.

    Falls back gracefully if the archive cannot be located; returns
    (copied, missing).
    """
    import tarfile
    import CoRE_MOF as CoRE_MOF_pkg

    pkg_dir = Path(CoRE_MOF_pkg.__file__).resolve().parent
    tar_path = pkg_dir / "data" / f"{dataset}.tar.xz"
    if not tar_path.exists():
        return 0, len(names)

    wanted = {f"{n}.cif" for n in names}
    copied = 0
    with tarfile.open(tar_path, mode="r:xz") as tf:
        for member in tf:
            name = Path(member.name).name
            if name not in wanted:
                continue
            dest = sdir / name
            if dest.exists():
                continue
            src = tf.extractfile(member)
            dest.write_bytes(src.read())
            copied += 1
    missing = len([n for n in names if not (sdir / f"{n}.cif").exists()])
    return copied, missing


def main() -> None:
    args = parse_args()
    import CoRE_MOF

    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    names = list(CoRE_MOF.list_structures(args.dataset))
    print(f"{args.dataset}: {len(names)} structures")

    # ---- property table -------------------------------------------------
    props = CoRE_MOF.get_properties(args.dataset)
    unnamed = [c for c in props.columns if str(c).startswith("Unnamed")]
    props = props.drop(columns=unnamed)
    table_path = args.out / "properties.parquet"
    try:
        props.to_parquet(table_path, index=False)
    except Exception as exc:
        print(f"[warn] parquet failed ({exc}); writing CSV instead.")
        table_path = args.out / "properties.csv"
        props.to_csv(table_path, index=False)
    print(f"wrote {table_path} ({props.shape[0]} rows x {props.shape[1]} cols)")

    # ---- structures -----------------------------------------------------
    struct_stats = None
    if not args.no_structures:
        sdir = args.out / "structures"
        sdir.mkdir(exist_ok=True)

        todo = [n for n in names if not (sdir / f"{n}.cif").exists()]
        already = len(names) - len(todo)
        failed = []

        if args.limit:
            # Smoke-test path: per-structure API is slow (~1/s) but respects
            # an exact cap without extracting the whole archive.
            todo = todo[: args.limit]
            import CoRE_MOF
            for name in todo:
                try:
                    with CoRE_MOF.get_CIF_structure_file(args.dataset, name) as src:
                        (sdir / f"{name}.cif").write_bytes(Path(src).read_bytes())
                except Exception as exc:
                    failed.append({"name": name, "error": str(exc)})
            copied = len(todo) - len(failed)
        else:
            # Fast path: one-pass bulk extraction straight from the bundled
            # archive (the per-structure API re-scans the tar.xz every call,
            # ~1/s vs thousands/s here).
            copied, _ = bulk_extract_structures(args.dataset, names, sdir)

        struct_stats = {
            "total_in_dataset": len(names),
            "already_cached": already,
            "copied_this_run": copied,
            "failed": failed,
        }
        cached_now = len(list(sdir.glob("*.cif")))
        print(f"structures on disk: {cached_now} "
              f"(now: {copied}, previously: {already}, failed: {len(failed)})")

    manifest = {
        "stage": "core_mof_export v1",
        "dataset": args.dataset,
        "package_versions_note": "CoRE-MOF pip pkg 2019.1; data CC BY 4.0",
        "n_structures": len(names),
        "property_columns": int(props.shape[1]),
        "structures": struct_stats,
        "outputs": {
            "properties": table_path.name,
            "structures_dir": None if args.no_structures else "structures/",
        },
        "duration_sec": round(time.time() - t0, 1),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()

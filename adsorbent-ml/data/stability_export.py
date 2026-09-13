#!/usr/bin/env python
"""Build the normalized stability table from the MOFSimplify Zenodo record.

Source: Zenodo record 5737968 — "MOFSimplify: Machine Learning Models with
Extracted Stability Data of Three Thousand Metal-Organic Frameworks"
(Nandy et al., Sci. Data 2022). Downloads ``SciData.zip`` once (cached under
``data_cache/stability/_raw/``), then emits one row per CSD refcode merging:

- ``full_SSD_data.csv``  → solvent-removal stability (text-mined label)
- ``full_TSD_data.csv``  → TGA decomposition onset [°C]

Output: ``data_cache/stability/mofsimplify_ssd_tsd.csv`` in the schema the
``stability.py`` loader consumes (``mof_name, water_stable,
solvent_removal_stable, thermal_decomp_c, source, confidence``) plus
``manifest.json``. ``water_stable`` is left empty on purpose — the dataset
measures desolvation stability, not hydrolysis; the loader's filter treats
solvent-removal-stable as conservative evidence for water service.

Usage:
    python stability_export.py            # download if needed + build
    python stability_export.py --local    # build from an existing download
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / ".." / ".." / "data_cache" / "stability" / "_raw"
ZENODO_URL = ("https://zenodo.org/api/records/5737968/files/"
              "SciData.zip/content")

SSD_REL = "SciData/separate_files/solvent_removal_stability/full_SSD_data.csv"
TSD_REL = "SciData/separate_files/thermal_stability/full_TSD_data.csv"


def _ensure_zip(path: Path) -> Path:
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    print(f"Downloading {ZENODO_URL} ...")
    import urllib.request

    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(ZENODO_URL, headers={
        "User-Agent": "cooling-with-heat/stability_export"})
    with urllib.request.urlopen(req, timeout=300) as resp, open(path, "wb") as fh:
        fh.write(resp.read())
    print(f"  saved {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def _read_rows(zh: zipfile.ZipFile, member: str) -> list[dict]:
    with zh.open(member) as fh:
        text = fh.read().decode("utf-8-sig")  # files carry a BOM
    return list(csv.DictReader(text.splitlines()))


def build(zip_path: Path, out_csv: Path) -> dict:
    merged: dict[str, dict] = {}
    with zipfile.ZipFile(zip_path) as zh:
        for row in _read_rows(zh, SSD_REL):
            ref = (row.get("refcode") or "").strip()
            if not ref:
                continue
            rec = merged.setdefault(ref, {"mof_name": ref})
            rec["solvent_removal_stable"] = (
                (row.get("assigned_solvent_removal_stability") or "").strip())
            rec["source"] = (row.get("doi") or "").strip()
            rec["confidence"] = "text-mined"
        for row in _read_rows(zh, TSD_REL):
            ref = (row.get("refcode") or "").strip()
            if not ref:
                continue
            rec = merged.setdefault(ref, {"mof_name": ref})
            rec["thermal_decomp_c"] = (
                row.get("assigned_T_decomp (°C)") or "").strip()
            rec["source"] = rec.get("source") or (row.get("doi") or "").strip()
            rec["confidence"] = rec.get("confidence") or "tga-extracted"

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["mof_name", "water_stable", "solvent_removal_stable",
              "thermal_decomp_c", "source", "confidence"]
    n_solvent = n_decomp = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for ref in sorted(merged):
            rec = merged[ref]
            solvent = rec.get("solvent_removal_stable", "")
            if solvent:
                n_solvent += 1
            if rec.get("thermal_decomp_c"):
                n_decomp += 1
            writer.writerow({
                "mof_name": rec["mof_name"],
                "water_stable": "",  # honest: dataset measures desolvation, not hydrolysis
                "solvent_removal_stable": {"1": "true", "0": "false"}.get(solvent, solvent),
                "thermal_decomp_c": rec.get("thermal_decomp_c", ""),
                "source": rec.get("source", ""),
                "confidence": rec.get("confidence", ""),
            })
    return {
        "stage": "stability_export v1 (MOFSimplify Zenodo 5737968)",
        "rows": len(merged),
        "n_with_solvent_label": n_solvent,
        "n_with_decomp_c": n_decomp,
        "table": out_csv.name,
        "created_unix": time.time(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--local", action="store_true",
                    help="reuse an existing download; never hit the network")
    args = ap.parse_args()

    zip_path = RAW_DIR / "SciData.zip"
    if not args.local:
        _ensure_zip(zip_path)
    if not zip_path.exists():
        sys.exit(f"{zip_path} missing (run without --local to download).")

    out_csv = DATA_DIR / ".." / ".." / "data_cache" / "stability" / "mofsimplify_ssd_tsd.csv"
    manifest = build(zip_path.resolve(), out_csv.resolve())
    manifest_path = out_csv.parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {out_csv.resolve()}: {manifest['rows']} rows "
          f"({manifest['n_with_solvent_label']} solvent labels, "
          f"{manifest['n_with_decomp_c']} TGA onsets)")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()

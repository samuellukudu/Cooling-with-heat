#!/usr/bin/env python
"""Export the QMOF Database (source #3 of ACQUISITION.md).

Queries the Figshare API for article 13147324 (QMOF Database,
Rosen et al., 20k+ MOFs with periodic-DFT properties incl. partial
charges), downloads the archive(s), and extracts tabular properties:

    <out>/qmof_database.zip          kept archive (392 MB)
    <out>/properties/*.csv           tabulated DFT property tables
    <out>/manifest.json              provenance

Geometry archives can be extracted on demand; pass --extract-all to unpack
everything (several GB uncompressed).

Usage:
    python qmof_export.py                    # download + list contents
    python qmof_export.py --extract-all      # also unpack every archive
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

FIGSHARE_ARTICLE = "13147324"
API_URL = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE}"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "data_cache" / "qmof"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--extract-all", action="store_true",
                   help="Unpack downloaded zips fully (large on disk).")
    p.add_argument("--skip-download", action="store_true",
                   help="Reuse existing zips in <out>.")
    return p.parse_args()


def figshare_files() -> list[dict]:
    with urllib.request.urlopen(API_URL, timeout=60) as r:
        meta = json.load(r)
    return [
        {"name": f["name"], "url": f["download_url"], "size": f["size"]}
        for f in meta.get("files", [])
        if f["name"].endswith(".zip")
    ]


def download(url: str, dest: Path) -> None:
    t0 = time.time()
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                mb = done / 1e6
                print(f"\r  {dest.name}: {mb:7.1f} / {total/1e6:.1f} MB "
                      f"({done/max(time.time()-t0,1e-9)/1e6:.1f} MB/s)",
                      end="", flush=True)
    print()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    files = figshare_files()
    print(f"Figshare article {FIGSHARE_ARTICLE}: {len(files)} zip file(s)")

    results = []
    for spec in files:
        dest = args.out / spec["name"]
        if args.skip_download and dest.exists():
            print(f"kept existing {dest.name} ({dest.stat().st_size/1e6:.0f} MB)")
        elif not dest.exists() or dest.stat().st_size != spec["size"]:
            print(f"downloading {spec['name']} ({spec['size']/1e6:.0f} MB)")
            download(spec["url"], dest)

        import zipfile
        with zipfile.ZipFile(dest) as zf:
            names = zf.namelist()
            csv_members = [n for n in names if n.lower().endswith(".csv")]
            results.append({
                "archive": spec["name"],
                "members": len(names),
                "csv_tables": csv_members,
                "extracted": False,
            })
            print(f"  {spec['name']}: {len(names)} members, "
                  f"{len(csv_members)} CSV tables")

            # Always pull out the small tabular property tables.
            prop_dir = args.out / "properties"
            prop_dir.mkdir(exist_ok=True)
            for member in csv_members:
                target = prop_dir / Path(member).name
                if not target.exists():
                    target.write_bytes(zf.read(member))

            # Extract useful nested zips (e.g. QMOF's relaxed_structures.zip
            # with DFT-optimized CIFs); skip anything huge.
            for member in zf.namelist():
                if not member.lower().endswith(".zip"):
                    continue
                info = zf.getinfo(member)
                if info.file_size > 600e6:
                    print(f"  skipping large nested archive {member}")
                    continue
                nested_dir = args.out / Path(member).stem
                if not nested_dir.exists():
                    print(f"  extracting nested {Path(member).name}")
                    zf.extract(member, args.out)
                    with zipfile.ZipFile(args.out / member) as nzf:
                        nzf.extractall(nested_dir)
                    results[-1]["nested_extracted"] = str(nested_dir)

            if args.extract_all:
                zf.extractall(args.out / "extracted")
                results[-1]["extracted"] = True

    prop_dir = args.out / "properties"
    n_tables = len(list(prop_dir.glob("*.csv"))) if prop_dir.exists() else 0
    print(f"property tables available in {prop_dir}: {n_tables}")

    manifest = {
        "stage": "qmof_export v1",
        "figshare_article": FIGSHARE_ARTICLE,
        "citation": "Rosen et al., Matter 4, 1578-1597 (2021); "
                    "doi 10.6084/m9.figshare.13147324",
        "archives": results,
        "property_tables_extracted": n_tables,
        "duration_sec": round(time.time() - t0, 1),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()

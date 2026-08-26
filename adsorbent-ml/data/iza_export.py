#!/usr/bin/env python
"""Export IZA zeolite framework CIFs (source #4a of ACQUISITION.md).

Scrapes the framework-type code list from the IZA-SC database index
(https://america.iza-structure.org/IZA-SC/ftc_table.php) and downloads each
framework's reference CIF from https://america.iza-structure.org/IZA-SC/cif/<CODE>.cif

    <out>/frameworks/<CODE>.cif   ~250 approved frameworks
    <out>/frameworks.csv          code + provenance per framework
    <out>/manifest.json           counts/duration; resumable by design

Be polite: a small delay between requests is applied and existing files are
never re-downloaded.
"""

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "data_cache" / "iza"
BASE = "https://america.iza-structure.org/IZA-SC"
TABLE_URL = f"{BASE}/ftc_table.php"
CODE_RE = re.compile(r'CodeTable "><a  href ="framework\.php\?ID=\d+">([A-Z]{3})')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--delay", type=float, default=0.3,
                   help="Seconds between requests (be polite).")
    return p.parse_args()


def fetch(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "cooling-with-heat research/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def main() -> None:
    args = parse_args()
    fdir = args.out / "frameworks"
    fdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    html = fetch(TABLE_URL).decode("utf-8", errors="replace")
    codes = sorted(set(CODE_RE.findall(html)))
    if len(codes) < 200:
        raise SystemExit(f"Only {len(codes)} framework codes found — "
                         f"the IZA page layout may have changed.")
    print(f"{len(codes)} approved framework codes")

    downloaded = already = failed = 0
    for i, code in enumerate(codes, start=1):
        dest = fdir / f"{code}.cif"
        if dest.exists():
            already += 1
            continue
        try:
            data = fetch(f"{BASE}/cif/{code}.cif")
            if not data.startswith(b"data_"):
                raise ValueError("response does not look like a CIF")
            dest.write_bytes(data)
            downloaded += 1
        except Exception as exc:
            failed += 1
            print(f"  [fail] {code}: {exc}")
        time.sleep(args.delay)
        if i % 50 == 0:
            print(f"  {i}/{len(codes)}")

    import pandas as pd
    df = pd.DataFrame({
        "framework_code": codes,
        "source_url": [f"{BASE}/cif/{c}.cif" for c in codes],
        "cif_file": [f"frameworks/{c}.cif" for c in codes],
    })
    csv_path = args.out / "frameworks.csv"
    df.to_csv(csv_path, index=False)

    manifest = {
        "stage": "iza_export v1",
        "base_url": BASE,
        "n_frameworks": len(codes),
        "downloaded_this_run": downloaded,
        "already_cached": already,
        "failed": failed,
        "duration_sec": round(time.time() - t0, 1),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\ndownloaded: {downloaded}, cached: {already}, failed: {failed}")
    print(f"wrote {csv_path} and manifest")


if __name__ == "__main__":
    main()

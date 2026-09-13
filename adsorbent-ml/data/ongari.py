"""Ongari et al. name→refcode evidence (vendored, MIT — see
``data_cache/ongari/ATTRIBUTION.md``).

``step-09.yml`` / ``step-10.yml`` map NIST-ISODB adsorbent names to CSD
refcodes (``{name: {REFCODE: sghash}}``). This module parses them into one
``(adsorbent_name, refcode, source)`` table. Fetch once (network), parse
offline afterwards::

    python3 adsorbent-ml/data/ongari.py --fetch   # vendor into data_cache
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data_cache" / "ongari"
BASE_URL = ("https://raw.githubusercontent.com/danieleongari/"
            "matching_isodb_csd/master/data")
TABLES = ("step-09.yml", "step-10.yml")


def fetch_tables(out_dir: Path = OUT_DIR) -> list[Path]:
    """Download the vendored mapping tables (one-time, network)."""
    import urllib.request  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in TABLES:
        dest = out_dir / name
        if not dest.exists():
            urllib.request.urlretrieve(f"{BASE_URL}/{name}", dest)
        paths.append(dest)
    if not (out_dir / "ATTRIBUTION.md").exists():
        raise FileNotFoundError(
            f"{out_dir}/ATTRIBUTION.md missing — vendoring requires the "
            "MIT attribution (see ACQUISITION data checklist)")
    return paths


def load_ongari_map(out_dir: Path = OUT_DIR) -> pd.DataFrame:
    """All ``(adsorbent_name, refcode, source)`` rows from both tables."""
    rows = []
    for name in TABLES:
        with open(out_dir / name, encoding="utf-8") as fh:
            table = yaml.safe_load(fh) or {}
        for adsorbent, refs in table.items():
            for refcode in (refs or {}):
                rows.append({"adsorbent_name": str(adsorbent),
                             "refcode": str(refcode).upper(), "source": name})
    df = pd.DataFrame(rows).drop_duplicates()
    return df.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Vendor Ongari mapping tables.")
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    if args.fetch:
        fetch_tables()
    df = load_ongari_map()
    print(f"rows={len(df)} adsorbents={df['adsorbent_name'].nunique()} "
          f"refcodes={df['refcode'].nunique()}")


if __name__ == "__main__":
    main()

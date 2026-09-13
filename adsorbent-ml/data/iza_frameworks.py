#!/usr/bin/env python
"""Scrape IZA-SC framework data: LCD/PLD, density, void fraction per code.

Companion to ``iza_export.py`` (which fetches CIFs): the framework pages
(``framework.php?ID=<n>``) publish geometric pore data the CIFs alone don't
give us without Zeo++:

- maximum sphere diameter that can be **included** (= LCD [Å]);
- that can **diffuse** along a/b/c (directional PLD [Å] — we take the min);
- framework density FD_Si [T/1000 Å³], accessible volume [%], channel
  dimensionality, essential rings, natural tiling.

Polite (delay, resume by skipping codes already in the CSV), stdlib-only
like ``iza_export.py``. Output ``data_cache/iza/framework_data.csv`` —
the pore-block source for IZA-matched rows (``pore_iza=1``).

    python3 adsorbent-ml/data/iza_frameworks.py --out data_cache/iza
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "data_cache" / "iza"
BASE = "https://america.iza-structure.org/IZA-SC"
TABLE_URL = f"{BASE}/ftc_table.php"
ID_RE = re.compile(r'framework\.php\?ID=(\d+)">([A-Z]{3})')

COLUMNS = ["code", "fd_si", "lcd_included_a", "pld_diffuse_a",
           "pld_diffuse_b", "pld_diffuse_c", "pld_min_a",
           "accessible_vol_pct", "channel_dim", "page_id"]


def fetch(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "cooling-with-heat research/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def page_text(html: bytes) -> str:
    raw = html.decode("utf-8", errors="replace")
    raw = re.sub(r"&[a-zA-Z]+;", " ", raw)  # &nbsp; &Aring; &middot; …
    text = re.sub(r"<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", text).strip()


def _num(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def parse_framework_page(html: bytes, code: str, page_id: int) -> dict:
    """Parse one framework page into a ``COLUMNS`` row (None where absent)."""
    t = page_text(html)
    diffuse: list[str] = []
    anchor = t.find("that can diffuse")
    if anchor >= 0:
        for _, val in re.findall(r"([abc])\s*:\s*([\d.]+)", t[anchor:anchor + 160]):
            diffuse.append(val)
    channel = re.search(r"Channel dimensionality:[^.]*?(\d)-dimensional", t)
    diffuse_f = [float(v) for v in diffuse]
    return {
        "code": code,
        "fd_si": _num(r"Framework density \(FD Si \)\s*:\s*([\d.]+)", t),
        "lcd_included_a": _num(r"that can be included\s+([\d.]+)", t),
        "pld_diffuse_a": diffuse_f[0] if len(diffuse_f) > 0 else None,
        "pld_diffuse_b": diffuse_f[1] if len(diffuse_f) > 1 else None,
        "pld_diffuse_c": diffuse_f[2] if len(diffuse_f) > 2 else None,
        "pld_min_a": min(diffuse_f) if diffuse_f else None,
        "accessible_vol_pct": _num(r"Accessible volume\s*:\s*([\d.]+)", t),
        "channel_dim": int(channel.group(1)) if channel else None,
        "page_id": page_id,
    }


def load_existing(out: Path) -> dict[str, dict]:
    csv_path = out / "framework_data.csv"
    if not csv_path.exists():
        return {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        return {r["code"]: r for r in csv.DictReader(fh)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Scrape IZA-SC framework pore data.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--delay", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=0, help="0 = all codes")
    args = ap.parse_args()

    pairs = sorted(set(ID_RE.findall(fetch(TABLE_URL).decode("utf-8", errors="replace"))),
                   key=lambda p: p[1])
    if len(pairs) < 200:
        raise SystemExit(f"Only {len(pairs)} code-ID pairs — page layout changed?")
    if args.limit:
        pairs = pairs[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    have = load_existing(args.out)
    rows = list(have.values())
    n_new = n_fail = 0
    for i, (pid, code) in enumerate(pairs, start=1):
        if code in have:
            continue
        try:
            rows.append(parse_framework_page(fetch(f"{BASE}/framework.php?ID={pid}"),
                                             code, int(pid)))
            n_new += 1
        except Exception as exc:
            n_fail += 1
            print(f"  [fail] {code}: {exc}")
        time.sleep(args.delay)
        if i % 50 == 0:
            print(f"  {i}/{len(pairs)}")
    rows.sort(key=lambda r: r["code"])
    with open(args.out / "framework_data.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    (args.out / "framework_data_manifest.json").write_text(json.dumps(
        {"n_codes": len(rows), "n_new": n_new, "n_fail": n_fail,
         "provenance": "IZA-SC framework pages (LCD/PLD/FD/void)",
         "created_unix": time.time()}, indent=2))
    coverage = sum(1 for r in rows if r["lcd_included_a"] not in (None, ""))
    print(f"codes={len(rows)} new={n_new} fail={n_fail} lcd_coverage={coverage}")


if __name__ == "__main__":
    main()

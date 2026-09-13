#!/usr/bin/env python
"""OPTIMADE federated structure client: one exporter for many databases.

Companion to ``adsorbent-ml/data/ACQUISITION.md`` §5. OPTIMADE is a REST
standard (JSON:API) spoken by Materials Project, OQMD, NOMAD, Materials
Cloud, COD and ~25 others — this module replaces N per-database exporters
with one filtered query plus a discovery step:

1. **Discovery** (the actual OPTIMADE protocol): fetch the providers index
   (``/v1/links``); each entry is either a queryable database (answers
   ``/v1/info`` with ``data.type == "info"``) or another index (answers
   ``/v1/links`` — recurse one level). Verified live against
   ``providers.optimade.org`` (29 links, 2026-09).
2. **Query**: ``GET {base}/v1/structures?filter=...`` with selected
   ``response_fields``, following ``links.next`` pagination to ``max_entries``.
3. **Cache**: raw JSON entries → ``structures.jsonl`` + ``manifest.json``
   (filter, counts, provider, timestamp). CIF conversion is OUT OF SCOPE
   here (no pymatgen in this env) — the Materials env owns
   JSON → CIF/relaxation per ROADMAP's layering.

Pure ``requests`` (no optimade-python-tools dependency). All HTTP goes
through an injectable ``get`` callable so tests run offline.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX = "https://providers.optimade.org/v1/links"
DEFAULT_OUT = REPO_ROOT / "data_cache" / "optimade"

# Cheap MOF-ish default: C+O bearing, small cells (smoke queries stay fast).
DEFAULT_FILTER = 'elements HAS "C" AND elements HAS "O" AND nsites<50'
DEFAULT_FIELDS = ("id", "chemical_formula_descriptive", "elements",
                  "nsites", "lattice_vectors", "cartesian_site_positions",
                  "species_at_sites", "last_modified")


def _default_get(url: str, *, timeout: int = 10):
    if requests is None:  # pragma: no cover
        raise RuntimeError("requests is required for live OPTIMADE queries")
    resp = requests.get(url, timeout=timeout,
                        headers={"User-Agent": "cooling-with-heat/optimade_export"})
    return resp


def _is_queryable(info: dict) -> bool:
    """True only for databases serving structures — index meta-databases
    also answer ``/info`` with ``data.type == "info"`` but advertise just
    ``["info", "links"]`` endpoints (seen live on the COD static index)."""
    if not isinstance(info.get("data"), dict):
        return False
    if info["data"].get("type") != "info":
        return False
    endpoints = (info["data"].get("attributes", {}) or {}).get("available_endpoints") or []
    return "structures" in endpoints


def _json_or_raise(url: str, get: Callable) -> dict:
    resp = get(url)
    status = getattr(resp, "status_code", 200)
    if status != 200:
        raise RuntimeError(f"OPTIMADE GET {url} -> HTTP {status} (fail loudly, "
                           "never silently partial)")
    payload = resp.json() if hasattr(resp, "json") else resp
    if not isinstance(payload, dict) or "data" not in payload:
        raise ValueError(f"OPTIMADE GET {url}: not a JSON:API document (no 'data')")
    return payload


def resolve_queryable_bases(index_url: str = DEFAULT_INDEX,
                            get: Callable | None = None) -> list[dict]:
    """Index links -> ``[{provider, base_url}]`` of directly queryable DBs."""
    get = get or _default_get
    doc = _json_or_raise(index_url, get)
    entries = doc["data"] if isinstance(doc["data"], list) else [doc["data"]]
    out: list[dict] = []
    seen: set[str] = set()
    candidates: list[tuple[str, str]] = []
    for e in entries:
        attrs = e.get("attributes", {}) if isinstance(e, dict) else {}
        base = (attrs.get("base_url") or "").rstrip("/")
        if base:
            candidates.append((attrs.get("name", base), base))
    queue = list(candidates) + [(n, b + "/v1") for n, b in candidates
                                if not b.rstrip("/").endswith("/v1")]
    for name, base in queue:
        if base in seen:
            continue
        seen.add(base)
        try:
            info = _json_or_raise(base + "/info", get)
        except Exception:
            # Dead/slow/uncooperative provider (live lesson: MPDD hangs past
            # timeout): still try /links below; never kill discovery. The
            # query path stays strict.
            info = None
        if info is not None and _is_queryable(info):
            out.append({"provider": name, "base_url": base})
            continue
        # Otherwise (index wearing an /info doc, non-structures database, or
        # failed /info): recurse into /links.
        try:
            links = _json_or_raise(base + "/links", get)
        except Exception:
            continue
        kids = links["data"] if isinstance(links["data"], list) else [links["data"]]
        for k in kids:
            ka = k.get("attributes", {}) if isinstance(k, dict) else {}
            kb = (ka.get("base_url") or "").rstrip("/")
            if kb and kb not in seen:
                queue.append((ka.get("name", kb), kb))
    return out


def query_structures(base_url: str, optimade_filter: str = DEFAULT_FILTER, *,
                     response_fields: tuple[str, ...] = DEFAULT_FIELDS,
                     page_limit: int = 20, max_entries: int = 200,
                     get: Callable | None = None):
    """Yield raw structure entries, following ``links.next`` pagination."""
    from urllib.parse import quote  # noqa: PLC0415

    get = get or _default_get
    fields = ",".join(response_fields)
    url = (f"{base_url}/structures?filter={quote(optimade_filter)}"
           f"&response_fields={quote(fields)}"
           f"&page_limit={page_limit}&response_format=json")
    n = 0
    while url and n < max_entries:
        doc = _json_or_raise(url, get)
        data = doc["data"] if isinstance(doc["data"], list) else [doc["data"]]
        for entry in data:
            if n >= max_entries:
                return
            yield entry
            n += 1
        links = doc.get("links", {}) or {}
        url = links.get("next")
        if url is not None and not isinstance(url, str):
            url = (url or {}).get("href")


def entry_to_cif(entry: dict) -> str:
    """Minimal CIF from an OPTIMADE structure entry (no pymatgen needed).

    Fractional coordinates via ``lattice⁻¹ · cartesian``; occupancies from
    species concentrations (partial occupancies preserved, not rounded).
    Raises ``ValueError`` when lattice/positions/species are absent —
    never writes a corrupt CIF.
    """
    import numpy as np  # noqa: PLC0415

    attrs = entry.get("attributes", {})
    missing = [k for k in ("lattice_vectors", "cartesian_site_positions",
                           "species_at_sites") if attrs.get(k) is None]
    if missing:
        raise ValueError(f"entry {entry.get('id')}: missing {missing}, "
                         "refusing CIF conversion")
    lattice = np.asarray(attrs["lattice_vectors"], dtype=float)
    if lattice.shape != (3, 3):
        raise ValueError(f"entry {entry.get('id')}: lattice shape {lattice.shape}")
    a, b, c = (float(np.linalg.norm(v)) for v in lattice)
    cos_a = float(np.dot(lattice[1], lattice[2]) / (b * c))
    cos_b = float(np.dot(lattice[0], lattice[2]) / (a * c))
    cos_g = float(np.dot(lattice[0], lattice[1]) / (a * b))
    import math  # noqa: PLC0415

    alpha, beta, gamma = (math.degrees(math.acos(np.clip(x, -1.0, 1.0)))
                          for x in (cos_a, cos_b, cos_g))
    species = {s.get("name"): s for s in attrs.get("species", [])}
    inv = np.linalg.inv(lattice)
    lines = [f"data_{entry.get('id', 'optimade')}",
             "_symmetry_space_group_name_H-M 'P 1'",
             f"_cell_length_a {a:.6f}", f"_cell_length_b {b:.6f}",
             f"_cell_length_c {c:.6f}", f"_cell_angle_alpha {alpha:.4f}",
             f"_cell_angle_beta {beta:.4f}", f"_cell_angle_gamma {gamma:.4f}",
             "loop_", "_atom_site_label", "_atom_site_type_symbol",
             "_atom_site_fract_x", "_atom_site_fract_y",
             "_atom_site_fract_z", "_atom_site_occupancy"]
    for i, (sp_name, cart) in enumerate(zip(attrs["species_at_sites"],
                                            attrs["cartesian_site_positions"])):
        spec = species.get(sp_name, {})
        symbols = spec.get("chemical_symbols", [sp_name])
        conc = spec.get("concentration", [1.0])
        frac = inv @ np.asarray(cart, dtype=float)
        for sym, occ in zip(symbols, conc):
            lines.append(f"{sym}{i} {sym} {frac[0]:.6f} {frac[1]:.6f} "
                         f"{frac[2]:.6f} {float(occ):.4f}")
    return "\n".join(lines) + "\n"


def formula_similarity(target_counts: dict[str, float],
                       cand_counts: dict[str, float]) -> float:
    """H-free cosine similarity of count vectors (1.0 = identical metals/C/N/O).

    Hydrogen is excluded (solvation varies); transition/post-transition
    metals must match exactly or the score is 0 — metals define the MOF.
    Both dicts are element→count mappings (see features/composition).
    """
    import math  # noqa: PLC0415

    metals = {"Li", "Na", "Mg", "Al", "K", "Ca", "Cr", "Mn", "Fe", "Co",
              "Ni", "Cu", "Zn", "Zr", "Cd", "Eu", "Tb", "Dy", "Ho", "Er",
              "Tm", "W", "Ba", "Co", "In", "Sc", "V", "Ti", "Hf", "Pb"}
    tm = {e for e in target_counts if e in metals}
    cm = {e for e in cand_counts if e in metals}
    if tm != cm:
        return 0.0
    keys = [e for e in set(target_counts) | set(cand_counts) if e != "H"]
    if not keys:
        return 0.0
    dot = sum(target_counts.get(k, 0.0) * cand_counts.get(k, 0.0) for k in keys)
    nt = math.sqrt(sum(v * v for k, v in target_counts.items() if k != "H"))
    nc = math.sqrt(sum(v * v for k, v in cand_counts.items() if k != "H"))
    return dot / (nt * nc) if nt > 0 and nc > 0 else 0.0


def export_structures(provider_substr: str, optimade_filter: str = DEFAULT_FILTER, *,
                      max_entries: int = 200, page_limit: int = 20,
                      out_dir: Path = DEFAULT_OUT,
                      index_url: str = DEFAULT_INDEX,
                      get: Callable | None = None) -> dict:
    """End-to-end: discover -> match provider -> query -> cache. Returns manifest."""
    get = get or _default_get
    bases = resolve_queryable_bases(index_url, get)
    match = [b for b in bases if provider_substr.lower() in b["provider"].lower()]
    if not match:
        known = ", ".join(sorted(b["provider"] for b in bases)) or "(none resolved)"
        raise ValueError(f"no OPTIMADE provider matches {provider_substr!r}. "
                         f"Resolved: {known}")
    base = match[0]
    entries = list(query_structures(base["base_url"], optimade_filter,
                                    page_limit=page_limit,
                                    max_entries=max_entries, get=get))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "structures.jsonl", "w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
    manifest = {"provider": base["provider"], "base_url": base["base_url"],
                "filter": optimade_filter, "n_entries": len(entries),
                "capped": len(entries) >= max_entries,
                "provenance": "OPTIMADE v1 federated query (raw entries, no CIF conversion)",
                "created_unix": time.time()}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Federated OPTIMADE structure export.")
    ap.add_argument("--provider", default="Materials Cloud",
                    help="substring matched against provider names")
    ap.add_argument("--base-url", default=None,
                    help="queryable base URL (skips discovery, e.g. from --list-providers)")
    ap.add_argument("--filter", default=DEFAULT_FILTER)
    ap.add_argument("--max-entries", type=int, default=200)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--list-providers", action="store_true")
    args = ap.parse_args()
    if args.list_providers:
        for b in resolve_queryable_bases():
            print(f"{b['provider']}  {b['base_url']}")
        return
    if args.base_url:
        entries = list(query_structures(args.base_url, args.filter,
                                        max_entries=args.max_entries))
        args.out.mkdir(parents=True, exist_ok=True)
        with open(args.out / "structures.jsonl", "w", encoding="utf-8") as fh:
            for e in entries:
                fh.write(json.dumps(e) + "\n")
        manifest = {"provider": args.base_url, "base_url": args.base_url,
                    "filter": args.filter, "n_entries": len(entries),
                    "capped": len(entries) >= args.max_entries,
                    "provenance": "OPTIMADE v1 direct-base query",
                    "created_unix": time.time()}
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"{args.base_url}: {len(entries)} entries -> {args.out}")
        return
    manifest = export_structures(args.provider, args.filter,
                                 max_entries=args.max_entries, out_dir=args.out)
    print(f"{manifest['provider']}: {manifest['n_entries']} entries -> {args.out}")


if __name__ == "__main__":
    main()

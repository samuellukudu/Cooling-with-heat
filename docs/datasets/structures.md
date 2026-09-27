# Structures — frameworks behind the labels

**Role:** crystal structures + computed properties for featurization
(Stage-1 pores/composition) and Stage-2 crystal-graph inputs; also the
match targets for ISODB names and the L2 (GCMC) label-generator enabler.

## Sources / licenses / access

| source | what | access |
|---|---|---|
| CoRE MOF 2019-ASR | 12k solvent-removed experimental MOFs + 28-col porosity/DDEC table | pip `CoRE-MOF` bulk tar.xz (MIT code / CC BY 4.0 data) |
| QMOF | 20k DFT MOFs × 94 cols (gaps, energies, DDEC6/CM5 charges) + optimized CIFs | Figshare doi 10.6084/m9.figshare.13147324 |
| IZA-SC | ~250 approved zeolite frameworks, free CIFs | `iza_export.py` live scrape (EFN/EWE publish none) |
| OPTIMADE | one client, ~40 structure DBs (MC3D/MC2D/pyrene-MOFs/CURATED-COFs, …) | `optimade_export.py` filtered queries |
| Materials Project | chemsys candidates + CIFs (vendored query layer) | `mp_export.py` (needs `MP_API_KEY`) |

## Build

```bash
python adsorbent-ml/data/core_mof_export.py
python adsorbent-ml/data/qmof_export.py
python adsorbent-ml/data/iza_export.py
python adsorbent-ml/data/optimade_export.py --list-providers
python adsorbent-ml/data/mp_export.py --apps datacenter --max-generated-chemsys 3 --limit-per-system 5
```

## Counts (manifests + CIF globs)

| key | value |
|---|---|
| `core_mof:n_structures` | 12020 |
| `core_mof:property_columns` | 28 |
| `core_mof:dataset` | 2019-ASR |
| `qmof:figshare_article` | 13147324 |
| `iza:n_frameworks` | 250 |
| `core_mof:cif_on_disk` | 12020 |
| `qmof:cif_on_disk` | 0 |
| `iza:cif_on_disk` | 248 |
| `mp:cif_on_disk` | 14 |
| `optimade/curated-cofs:entries` | 2000 |
| `optimade/pyrene-mofs:entries` | 221 |

## Gotchas

- CoRE "~14k" counts include non-public CSD entries; the public ASR pull is 12,020.
- MOF cells are big: OPTIMADE queries need `nsites<2000`, not the `<50` default.
- OPTIMADE index DBs answer `/info` with only `["info","links"]` endpoints —
  filter by `available_endpoints` containing `structures`.
- QMOF↔CoRE join key is formula/composition (imperfect overlap, documented).

## Feeds

- Pore descriptors (LCD/PLD/AV) + family/topology splits → `labels` card.
- QMOF partial charges → future L2 (RASPA GCMC) on active-learning shortlists only.

## Validation

| gate | ok | issues |
|---|---|---|
| `fits` | ✅ | — |
| `labels` | ✅ | — |
| `matched` | ✅ | — |
| `stability` | ✅ | — |
| `refrigerants` | ✅ | — |

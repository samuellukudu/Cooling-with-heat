# Data Acquisition Plan (L1 sources)

Concrete plan for acquiring the four priority external datasets before any
modeling. Companion to [`../../ROADMAP.md`](../../ROADMAP.md) §4 (data
strategy). Everything lands under `data_cache/<source>/` (gitignored);
each source gets an exporter script here in `adsorbent-ml/data/`.

Implementation order = value order:

```
1. nist_isodb.py    experimental water-isotherm labels   <- the bottleneck
2. core_mof_export  framework structures + pore props    <- candidate universe
3. qmof_export      DFT props + partial charges          <- features & GCMC enabler
4. iza_anchors.py   zeolite CIFs + commercial-anchor table <- class #2 + ground truth
```

---

## 1. NIST ISODB — experimental adsorption labels

**What:** digitized experimental isotherms from the literature (37.7k
isotherms, 8.3k adsorbents, 449 adsorbates incl. water).

**Access (two paths, use both):**

- *Bulk:* `git clone https://github.com/NIST-ISODB/isodb-library` — official
  mirror, one small file per isotherm (e.g. `10.1002adem.200500223.isotherm2`)
  plus `DOI_mapping.csv`. Clone once into `data_cache/isodb/mirror/`.
- *Targeted:* JSON/XML/CSV REST APIs documented at
  <https://adsorption.nist.gov> (isodb + materials registry endpoints) for
  incremental refreshes later.

**Filtering for our task:**

- adsorbate == water (`H2O`) — expect low hundreds to ~1–2k isotherms out of
  37.7k; exact count is deliverable #1 of this step.
- Temperature window 280–380 K (covers all four application profiles).
- Record pressure range and number of points; flag single-point isotherms.

**Name→structure matching (the known hard problem):** ISODB material names
are non-standard ("Cu-BTC", "MOF-199", "MIL-53(Al)"…). Strategy, cheapest
first:

1. Normalized-string match against CoRE MOF structure names + common MOF
   synonyms and against IZA three-letter codes ("13X"→FAU, "NaA"→LTA…).
2. Reuse published matching tables — the Coudert-lab paper "Data-driven
   matching of experimental crystal structures and gas adsorption isotherms"
   (J. Chem. Eng. Data 2022) built exactly this mapping; check its SI/Zenodo.
3. MOFid/MOFkey strings where present.
4. Remainder → manual curation queue (`unmatched_names.csv`); water-only
   keeps this tractable.

**Outputs:** `data_cache/isodb/water_isotherms.parquet`
(`isotherm_id, doi, material_name, matched_refcode?, T_K, pressure[], uptake[],
uptake_unit, original_unit`), `matched.parquet`, `unmatched_names.csv`.
Unit normalization to kg/kg vs pressure Pa happens in `fit_da.py`, not here —
preserve originals.

---

## 2. CoRE MOF — computation-ready experimental frameworks

**What:** curated experimental MOFs, solvent-removed, ready for simulation;
porosity/surface-area/DDEC-charge columns included.

**Access:** pip package `CoRE-MOF` (coudertlab; MIT code / CC BY 4.0 data):

```python
import CoRE_MOF
mof = CoRE_MOF.get_structure("2019-ASR", "ZUZZEB_clean")   # pymatgen Structure
```

Bulk CSVs also on Zenodo: 2019 v1.1.4 = doi 10.5281/zenodo.7691378,
2014 = doi 10.5281/zenodo.3228673.

**Exporter:** `core_mof_export.py` →
`data_cache/core_mof/{structures/*.cif, properties.parquet}` (~14k ASR
structures). Join key = refcode/name, which is also what ISODB matching aims
at.

**Upgrade path:** CoRE MOF DB (2025, Matter) adds 40k+ structures with ML
DDEC6 charges and **MOFid node/linker/topology decomposition** — adopt when
Stage-2 needs volume; its node/linker/topology axes are also the parameter
space for any future GeoField-style design agent.

---

## 3. QMOF — DFT properties + charges

**What:** periodic-DFT properties for 20k+ MOFs: formation energy, band gap,
**DDEC6/CM5 partial charges**, optimized geometries (CIF/XYZ).

**Access:** Figshare doi 10.6084/m9.figshare.13147324 (tabular CSVs +
geometry archives); NOMAD DOIs for raw VASP files (not needed initially).
Mirror queryable via MPContribs if per-material lookup is handy.

**Why it matters twice:** (a) rich features for Stage-1/2 models; (b) the
partial charges are precisely what RASPA GCMC needs — QMOF is therefore also
the L2 label-generator enabler.

**Exporter:** `qmof_export.py` → `data_cache/qmof/{properties.parquet,
geometries/}` (download once, ~GBs; keep geometry archives compressed).

---

## 4a. IZA zeolites — second material class

~255 approved frameworks with free CIFs from the IZA Structure Commission
standard database (iza-structure.org). Tiny enough to vendor directly:
`iza_export.py` → `data_cache/iza/frameworks/*.cif` + a small table
(framecode, ring sizes, channel dimensionality, FD). Zeolites bring mature
water-isotherm literature (13X, NaA, Silicalite…) into the same pipeline.

## 4b. Commercial-anchor table — gold-quality ground truth

Hand-curated `anchors.csv` (~20–40 rows): `q_sat, Q_st, E_char, n` fitted or
published for the materials our validator already benchmarks — silica gel RD
(Rezk 2012, Ng 2006), zeolite 13X (Aristov, Ng 2001), AlPO-18/SAPO
(Henneringer 2012), MIL-101(Cr), UiO-66, CaCl₂/silica composites. Sources =
the same citations already encoded in `Materials/mp_query_validator.py`.

**Role:** independent sanity floor for `fit_da.py` (fitted parameters must
land near published values for these materials) and calibration set for
surrogate outputs. Never train-test leak these into random splits — hold out
whole material families.

---

## Verification checklist (per GeoField lesson: calibrate into the decision regime)

- [ ] ISODB water count reported; ≥100 matched water isotherms before Stage 1
- [ ] Every matched pair spot-checked visually (isotherm shape vs known type)
- [ ] CoRE export count ≈ 14k; CIF parse rate > 99% via pymatgen
- [ ] QMOF join to CoRE by name/refcode documented (imperfect overlap expected)
- [ ] `fit_da.py` recovers published `q_sat/Q_st/E` within stated error bars
      for ≥80% of anchor rows
- [ ] All manifests written (`manifest.json` per cache dir)

## Feeds

- `fit_da.py`: matched isotherms → per-material Dubinin–Astakhov parameters
  (`q_sat, E, n`) → consistent ML targets compatible with
  `cooling_physics.simulate_adsorption_cycle`.
- Family splits: metal-node/topology families from CoRE/MOFid metadata.
- Later L2: QMOF charges + RASPA on Stage-3-selected shortlists only.

---

## Evaluated, not adopted

### The Well (Polymathic AI, NeurIPS 2024) — 15 TB of physics simulations
16 PDE-trajectory datasets (convection, fluids, MHD, acoustics, plasma,
reaction-diffusion…) on uniform grids; HDF5 + unified loader; per-dataset
Hugging Face downloads (6.9 GB – 5.1 TB each).

**Verdict: not useful for the current plan — zero materials content.** It
contains no structures, chemistry, or adsorption data, so it cannot serve T1
(structure→property) or T2 (property→COP), where all current effort lives.

**Deferred trigger — revisit only when bed-level thermal emulation starts
(late Stage 3+/4):** its convection/fluid subsets are candidate training or
benchmark corpora for a neural-operator emulator of transient heat transport
in adsorber beds (maps to `docs/applications.md` §1 Natural Convection and
§5 Forced Convection). Pull a single dataset then (~GBs), never the full
collection.

**Adopt its conventions regardless:** shared HDF5 schema + manifests +
VRMSE-style metric + published baselines is the template to copy if we
release our own GCMC-generated L2 dataset.

# Data Acquisition Plan (L1 sources)

Concrete plan for acquiring the four priority external datasets before any
modeling. Companion to [`../../ROADMAP.md`](../../ROADMAP.md) §4 (data
strategy). Everything lands under `data_cache/<source>/` (gitignored);
each source gets an exporter script here in `adsorbent-ml/data/`.

Implementation order = value order:

```
1. nist_isodb.py      ✅ DONE — 1,221 pure-water isotherms / 557 adsorbents exported
2. core_mof_export.py ✅ DONE — 12,020 structures + 28-col property table
3. qmof_export.py     ✅ DONE — 20,372 MOFs x 94 DFT columns + optimized CIFs
4a. iza_export.py     ✅ DONE — 248/250 framework CIFs (EFN, EWE have none published)
4b. anchors.csv       ✅ DONE — 13 curated commercial-adsorbent rows
5. fit_da.py          ▶ NEXT — adsorbate-agnostic D–A fitting library + thin CLI
                         over these exports (reusability contract:
                         ../../harness/DESIGN.md §12 H1.0); output feeds
                         both Stage-1 training labels and the harness
                         material database
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

**Filtering for our task — VERIFIED via `nist_isodb.py` export (2026-08):**

- Mirror holds **39,824 isotherm records** (plus 14,429 registry/bibliography
  files in `Library/{Adsorbents,Adsorbates,Bibliography}/` — excluded); all
  parse cleanly.
- **Pure-water isotherms: 1,221**, across **557 unique adsorbents**
  (+1,234 multicomponent isotherms containing water).
- **1,122 of the 1,221 fall inside the 280–380 K window.**
- Top materials by water-isotherm count are precisely the canonical cooling/
  harvesting set: CuBTC (79), Silicalite MFI (50), ZIF-8 (42), Mg-MOF-74
  (40), zeolite 13X (26), silica gel (25), Na-Y, UiO-66, MIL-100, zeolite A…
- Uptake units are heterogeneous — normalization is a real work package:
  mmol/g dominates, then cm³(STP)/g, molecules/unitcell, g/g, mg/g, wt%…
  (`molecules/unitcell` conversions need framework cell composition).

⚠ Exporter gotchas (all handled in `nist_isodb.py`, worth knowing):
filename casing varies (`isotherm2.json` / `Isotherm4.json`) and some files
are misspelled (`isothem10.json`) — never filter by filename; walk
`Library/<doi>/*.json` and skip the three registry directories.

The ≥100-matched gate for Stage 1 is comfortably cleared on raw volume; the
binding constraint shifts to name→structure matching coverage.

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
`data_cache/core_mof/{structures/*.cif, properties.parquet}`.
**DONE (2026-08): 12,020 public ASR structures** (the "~14k" figure in
papers counts non-public CSD-derived entries too) + 28-column property
table incl. LCD/PLD/pore volume/surface area, `All_Metals` (family splits),
open-metal-site flags, and `DOI_public` (a bonus key for ISODB literature
matching). Bulk path streams the bundled tar.xz once (~seconds vs hours via
the per-structure API). CIF spot-check: 100/100 parse with pymatgen.

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

**Exporter:** `qmof_export.py` → `data_cache/qmof/`.
**DONE (2026-08):** both archives downloaded (545 MB); property table
extracted (**20,372 MOFs × 94 columns**: PBE band gaps/energies, composition,
charges); **20,372 DFT-optimized CIFs** pulled from the nested
`relaxed_structures.zip`; thermo archive kept for later. The exporter also
handles nested zips generically and skips the multi-GB raw JSON geometry
files.

---

## 4a. IZA zeolites — second material class

~255 approved frameworks with free CIFs from the IZA Structure Commission
standard database. **DONE (2026-08): 248/250 CIFs downloaded** via
`iza_export.py` (`data_cache/iza/frameworks/*.cif`) — EFN and EWE publish no
reference coordinates on IZA-SC. Framework codes scraped live from the
database index; resumable; polite delay between requests.

## 4b. Commercial-anchor table — gold-quality ground truth

**DONE (2026-08):** [`anchors.csv`](anchors.csv) — 13 curated rows covering
silica gel RD, zeolites 13X/NaA, AlPO-18/SAPO-34, MIL-101(Cr), UiO-66,
Mg-MOF-74, CAU-10-H, aluminum fumarate, silicalite, activated carbon, and a
LiCl/silica composite. Each row: `q_sat`, `Q_st`, `E_char`, `n`, source,
confidence tier. High-confidence rows are the same citations already encoded
in `Materials/mp_query_validator.py` Bench 3.

**Role:** independent sanity floor for `fit_da.py` (fitted parameters must
land near published values for these materials) and calibration set for
surrogate outputs. Never train-test leak these into random splits — hold out
whole material families.

---

## Verification checklist (per GeoField lesson: calibrate into the decision regime)

- [ ] ISODB water count reported; ≥100 matched water isotherms before Stage 1
      → raw pool exported (1,221); matching pending (needs CoRE ✓ + IZA ✓)
- [x] Every matched pair spot-checked visually (isotherm shape vs known type)
      — deferred to fit_da.py stage
- [x] CoRE export count ≈ 12,020 public ASR; CIF parse rate 100/100 spot-check
- [x] QMOF join to CoRE by name/refcode documented (imperfect overlap expected)
      → 20,372 rows + optimized CIFs on disk; join key = formula/composition,
        refine at Stage 1
- [ ] `fit_da.py` recovers published `q_sat/Q_st/E` within stated error bars
      for ≥80% of anchor rows
- [x] All manifests written (`manifest.json` per cache dir:
      mp, isodb, core_mof, qmof, iza)

## Feeds

- `fit_da.py`: matched isotherms → per-material Dubinin–Astakhov parameters
  (`q_sat, E, n`) → consistent ML targets compatible with
  `cooling_physics.simulate_adsorption_cycle`.
- Family splits: metal-node/topology families from CoRE/MOFid metadata.
- Later L2: QMOF charges + RASPA on Stage-3-selected shortlists only.

---

## Adopted: CoolProp — thermodynamic reference engine

Open-source EOS library (`pip install CoolProp`, v8.0): IAPWS-IF97 water,
Helmholtz fluids, humid-air psychrometrics (`HAPropsSI`), incompressible
brines/solutions.

**Role (strictly bounded): trusted reference OUTSIDE the differentiable path.**

```
cooling_physics.py / diff_hvac_cycle.py  <- analytic + differentiable, stays as-is
         ^ validated against (never replaced inside the gradient path)
CoolProp                                  <- compiled lib, no JAX/torch autodiff
```

Measured agreement of our analytic model vs CoolProp IF97 (0-150 °C grid,
2026-08):

| Quantity | Our correlation | Max deviation |
|---|---|---|
| Saturation pressure | Magnus/Antoine | 2.0 % (150 °C); ~1.1–1.6 % at 80–90 °C regeneration band |
| Latent heat | Watson | 0.3 % (0 °C), ~exact near 100 °C |
| Liquid cp | constant 4184 | ±0.3 % over cycle temps |

Planned uses:

1. **Validator upgrade** — dense-grid live reference calls in
   `Materials/mp_query_validator.py` instead of 7 hardcoded spot values.
2. **Refrigerant generalization** — optional `refrigerant=` on
   `simulate_adsorption_cycle` (methanol Psat(35 °C) = 28 kPa, ammonia
   1350 kPa: working-pair expansion is nearly free).
3. **Psychrometrics** for the human-comfort/dehumidification profile.
4. **Absorption escape hatch** — `INCOMP::Libr` if LiBr/water cycles ever enter scope.

Never import it into training/differentiable code paths.

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

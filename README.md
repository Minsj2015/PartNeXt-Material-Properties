# PartNeXt Material & Simulation-Ready Physical Properties

Per-part **material class** and **simulation-ready physical properties** (ν, E, ρ) for
**333,152 parts** across **23,221 textured 3D models** of
[PartNeXt](https://huggingface.co/datasets/AuWang/PartNeXt) (NeurIPS D&B 2025).

**📦 Data lives on Hugging Face:**
**https://huggingface.co/datasets/minsj1225/PartNeXt-Material-Properties**

This repository holds the tooling — the 190 MB annotation table stays on the Hub.

PartNeXt itself ships no material or physical-property labels; these annotations are new.

---

## Quick start

```bash
pip install -U pip
pip install huggingface_hub datasets pandas pyarrow

hf download minsj1225/PartNeXt-Material-Properties \
    --repo-type dataset --local-dir ./partnext_mat
cd partnext_mat

python3 join_with_partnext.py verify      # prove the join against the official masks
```

```
part_id       333,152 / 333,152 resolve to a masks key  (100.0000%)
type_id       mismatches 0 / 23,221
JOIN IS EXACT
```

## How a row maps onto a PartNeXt asset

The primary key is **`(model_id, part_id)`**:

| Column | Links to |
|---|---|
| `model_id` | a row of `AuWang/PartNeXt`, and the GLB basename |
| `part_id` | **a key of that row's `masks` dict** (= the `maskId` in its `hierarchyList`) |
| `type_id` | the mesh directory, so the file is `glbs/{type_id}/{model_id}.glb` in `AuWang/PartNeXt_mesh` |

`masks[part_id] = {mesh_id: [face_indices]}` gives the exact triangles.
`type_id` locates the **file**; `part_id` locates the **part inside it**.

> Read all three as strings — `type_id` carries leading zeros (`"000-029"`) and `part_id`
> is a string key upstream. Type inference breaks the join silently.

```python
import pandas as pd, json
from datasets import load_dataset

ann = pd.read_csv("PartNeXt_material_annotations.csv",
                  dtype={"model_id": str, "type_id": str, "part_id": str})
idx = ann.set_index(["model_id", "part_id"])          # build once, then O(1)

pnx = load_dataset("AuWang/PartNeXt", split="train")
for rec in pnx:
    for part_id, per_mesh in json.loads(rec["masks"]).items():
        try:
            a = idx.loc[(rec["model_id"], part_id)]
        except KeyError:
            continue                                   # not annotated
        # a["material_class"], a["density_kg_m3"],
        # a["youngs_modulus_GPa"], a["poissons_ratio"], a["sim_body_type"] …
```

## Scripts

| File | What it does |
|---|---|
| `join_with_partnext.py` | `verify` / `table` / `export` / `show` |
| `download_partnext_meshes.py` | fetches exactly the 23,221 annotated GLBs (55.10 GB), checksum-verified resume |
| `test_scripts.py` | 50 checks over a 10-model edge-case subset, incl. face indices vs real GLB geometry |
| `manifest_models.csv` | per model: `glb_relpath`, `glb_bytes`, `glb_sha256`, part count, solver plan |
| `buckets_properties.json` | 84 property buckets with per-quantity value, P25/P75 and source ids |
| `material_corpus_sources.csv` | the 82 corpus sources with citation, DOI/URL, license, redistribution clearance |
| `pipeline_config.json` | every fusion weight and threshold used for this release |

```bash
python3 join_with_partnext.py table  --out joined.parquet   # 1 row/part + glb_path + n_faces
python3 join_with_partnext.py export --out sim_json/        # 1 JSON/model, E already in Pa
python3 download_partnext_meshes.py --out ./partnext_mesh --check
python3 test_scripts.py
```

## Coverage

| | |
|---|---|
| Parts annotated | **333,152 / 350,145** (95.15%) |
| Models | **23,221 / 23,519** |
| Every `part_id` resolves to an official `masks` key | **333,152 / 333,152 = 100.0000%** |
| All three properties present | 329,289 (98.84%); 3,863 abstain — 3,701 sanitary whiteware with no traceable grade, 162 in the withheld `cfrp` bucket |
| Objects with **no** property at all | **624** (1,510 parts) — every part abstained, so these models cannot be loaded into a simulator as-is |

16,993 PartNeXt parts carry no annotation: 9,844 are geometrically degenerate (≤2 faces),
2,711 sit in 298 models with no GLB upstream, 12 have no `hierarchyList` node, and
**4,426 are a genuine pipeline gap** — named parts with real geometry, concentrated in
keyboards and laptops. A missing `(model_id, part_id)` always means *not annotated*,
never a failed join.

## Simulation-readiness notes

`sim_body_type` splits parts by stiffness: `rigid` (E ≥ 1 GPa), `semi_rigid` (0.1–1 GPa),
`deformable` (1–100 MPa), `soft` (< 1 MPa). Only the non-rigid ones belong in a deformable
solver; `E` and `nu` never enter a rigid-body solve.

**Read `sim_caveat` before routing a part.** 46,907 parts (14.1%) carry one of three warnings.

- *Dimensional basis (41,367 parts).* Woven-textile `E` (1.2e-4 GPa) is a cloth **membrane
  modulus** — physically N/m, not Pa — and `rho` (307 kg/m³) is a bulk packing density.
  These belong in a cloth or shell solver and are **not valid for volumetric FEM/MPM**.
  Foam `E` is an effective compressive modulus and `nu` is apparent, not the cell-wall value.
- *Withheld bucket (162 parts).* The `cfrp` bucket supplies no values: its density rested on a
  single record carrying an epoxy-matrix density (1200 kg/m³) rather than a laminate density
  (1550–1600 kg/m³), and no corpus source supplies one. These parts keep
  `material_class = composite` but have null properties.
- *Near-incompressibility (5,378 parts, `nu >= 0.45`).* The bulk modulus reaches roughly 50x
  the shear modulus, where low-order tetrahedral FEM volumetrically locks. Use a corotational
  MPM with enough substeps — about 250 at `dt = 1e-2` on a 15.6 mm grid — or an FEM
  stable-neo-Hookean model. Size the substep count from `sim_p_wave_speed_m_s`.

Three buckets carry `interval_kind = reference_range`: their value and interval come from a
published handbook range because the 82-source corpus holds no grade for that material. Their
`{rho,E,nu}_provenance` is `reference_value` and their `source_ids` is empty — claiming
corpus support for a number the corpus does not contain would be false. Each bucket's
`basis_note` in `buckets_properties.json` names the reference.

| Bucket | Parts | rho | E | nu | What it covers |
|---|---:|---:|---:|---:|---|
| `polyurethane_wheel` | 325 | 1180 | 40 MPa | 0.48 | cast PU caster and skateboard wheels, Shore A 85–95 |
| `flexible_pu_foam__upholstery` | 169 | 40 | 40 kPa | 0.30 | open-cell flexible upholstery foam — mattresses, cushions, ear pads |
| `cable_jacket__plasticized_pvc` | 92 | 1350 | 25 MPa | 0.40 | plasticized PVC cable and wire jacket compound |
| `copper_alloy` | 5,213 | 8600 | 128 GPa | 0.27 | copper and copper-alloy parts. All eight corpus records give 8600 kg/m³ and nu spans 0.27–0.33; pure copper is 8960 at nu 0.34, so the bucket is named for the family |

42% of objects mix rigid and soft bodies — a sofa's steel frame (203 GPa) against its
upholstery (1.2e-4 GPa) spans 1.7 × 10⁶. That is physically real, but one explicit solver is
throttled by the stiffest part's CFL. Follow `sim_object_solver_plan`.

---

## Agreement with external physical-property datasets

Evaluated against [PhysX-Mobility](https://huggingface.co/datasets/Caoza/PhysX-Mobility) and
UniPhys-Bench. The only link is the part name; their material labels and property values
are never fed to the pipeline and are used solely for scoring. Objects are split in half
and only the held-out half is reported.

Conditioned on `(object_category, part_name)` — the way these labels are produced:

| | |
|---|---|
| Material top-1 (held-out, n=2,935) | **89.7%** micro · **64.9%** macro · 75.2% majority baseline |
| ρ, where the class is correct | median 1.15× off, **100% within 2×** |
| E, where the class is correct | median 1.02× off, **96% within 2×** |
| ν, where the class is correct | median \|Δν\| = **0.014** |

**Look up with the object category.** Ignoring `object_category` and matching on part name
alone drops material top-1 to **64.5%** — generic names like `Door`, `Lid`, `Frame` and
`Lens` take their material from the host object.

Known weaknesses: macro accuracy is 64.9% because minority classes are hard — `rubber`
recall is near zero (`Button` and `Wheel` are silicone/rubber in appliance datasets but
plastic across PartNeXt), and wood `E` lands within 2× only 67% of the time because the
species is not visually determinable.

## License

The tooling in this repository and the annotation *layer* — part-to-material assignment,
bucket resolution, solver metadata — are CC BY 4.0 (see `LICENSE`). PartNeXt meshes and part
masks are CC BY 4.0 by their authors
([AuWang/PartNeXt](https://huggingface.co/datasets/AuWang/PartNeXt),
[AuWang/PartNeXt_mesh](https://huggingface.co/datasets/AuWang/PartNeXt_mesh)).

**The property values are not uniformly licensed.** Each inherits the licence of the corpus
sources it was measured from, and those are not mutually compatible. A row blends several
sources, so it cannot be assigned to a single-licence file; instead every row of the
annotation table on the Hub carries a **`license_tier`** column giving the strictest licence
that applies to it:

| `license_tier` | parts | share | what it means |
|---|---:|---:|---|
| `PERMISSIVE` | 85,763 | 25.7 % | CC0 / CC BY / MIT — attribution only |
| `PUBLIC_DOMAIN` | 92,825 | 27.9 % | US Government works — no restriction |
| `COPYLEFT` | 96,733 | 29.0 % | CC BY-SA 4.0 (CIRAD) and ODbL (SciGlass, Soft Robotics Materials Database) — **share-alike; those two are also incompatible with each other** |
| `NONCOMMERCIAL_OK` | 41,177 | 12.4 % | all `fabric` parts. The ARCSim cloth material database is released for **non-profit use only** — **not usable commercially** |
| `SPEC_DATA_CITED` | 12,205 | 3.7 % | manufacturer datasheet — published specification data, cited not relicensed |
| `NO_SOURCE` | 4,449 | 1.3 % | abstained rows, no property values |

**183,037 parts (54.9 %)** are `PERMISSIVE`, `PUBLIC_DOMAIN` or `NO_SOURCE` and carry no
share-alike or non-commercial obligation.

### Attribution you are required to give

Most of the corpus is under licences that make attribution a condition of use. If your work
uses parts in the corresponding tier, cite the source:

| Source | Rows | Licence | Citation |
|---|---:|---|---|
| Global Wood Density Database v.2 | 67,026 | CC BY 4.0 | Fischer, F.J., Chave, J., Zanne, A.E. et al. (2026), GWDD v.2 v2.2, Zenodo [10.5281/zenodo.20815517](https://doi.org/10.5281/zenodo.20815517); article: New Phytologist [10.1111/nph.70860](https://doi.org/10.1111/nph.70860) |
| ARCSim cloth material database | 41,177 | non-profit use, citation required | Wang, H., Ramamoorthi, R., O'Brien, J.F. (2011), *Data-Driven Elastic Models for Cloth*, ACM TOG 30(4), 71:1–11 |
| CIRAD wood density database | 67,026 | CC BY-SA 4.0 | Vieilledent et al., [github.com/ghislainv/wood-density-Cirad](https://github.com/ghislainv/wood-density-Cirad) |
| SciGlass | 28,791 | ODC-ODbL | SciGlass database, EPAM Systems release |
| Soft Robotics Materials Database | 916 | ODC-ODbL 1.0 | Marechal, L. et al. (2021), Soft Robotics 8(3), 284–297, [10.1089/soro.2019.0115](https://doi.org/10.1089/soro.2019.0115) |
| StressEng | 262,628 | CC BY 4.0 | Kumar, Kabra & Cole (2024), *Scientific Data* 11:1273 |
| USDA FPL Wood Handbook, MIL-HDBK-5J, NIST | 103,451+ | US Government works | no restriction |

All 82 sources with citation, DOI, licence and clearance are in
`material_corpus_sources.csv`; `{rho,E,nu}_source_ids` says which ones a given row used.

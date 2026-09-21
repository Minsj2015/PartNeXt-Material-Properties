#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Join PartNeXt-Material-Properties onto the official AuWang/PartNeXt assets.

Every annotated part is linked to (a) the GLB file it lives in and (b) the exact
triangle faces it occupies inside that GLB.

    model_id   -> AuWang/PartNeXt row, and the GLB basename
    type_id    -> the bucket directory:  glbs/{type_id}/{model_id}.glb
    part_id    -> a key of the official `masks` dict for that model
                  masks[part_id] = {mesh_id: [face_indices]}

Usage
-----
    pip install datasets pandas

    # 1) verify the join is exact (no downloads needed)
    python3 join_with_partnext.py verify

    # 2) flat table: one row per part, with mesh path + face counts
    python3 join_with_partnext.py table --out joined.parquet

    # 3) per-model simulation JSON (faces + material + properties)
    python3 join_with_partnext.py export --out sim_json/ --limit 100

    # 4) one model, human readable
    python3 join_with_partnext.py show --model-id 298ed9ae6f9b41dd8ee0c0c5b412f343
"""
import argparse, json, os, re, sys

HERE      = os.path.dirname(os.path.abspath(__file__))
ANNO_REPO = "minsj1225/PartNeXt-Material-Properties"
ANNO_NAME = "PartNeXt_material_annotations.csv"
ANNO_LOCAL = os.path.join(HERE, ANNO_NAME)
ANNO_HUB   = f"hf://datasets/{ANNO_REPO}/{ANNO_NAME}"
PNX_REPO  = "AuWang/PartNeXt"        # part masks + hierarchy
MESH_REPO = "AuWang/PartNeXt_mesh"   # GLB geometry

# part_id and the masks keys are strings upstream; type_id has leading zeros
# ("000-029"). Read both as str or the join silently loses rows.
DTYPES = {"model_id": str, "type_id": str, "part_id": str}


def default_annotations():
    """Prefer the copy sitting next to this script; fall back to streaming the Hub."""
    return ANNO_LOCAL if os.path.exists(ANNO_LOCAL) else ANNO_HUB


def load_annotations(path=None):
    require("pandas")
    import pandas as pd
    src = path or default_annotations()
    if src.startswith("hf://"):
        print(f"reading {src}\n  (185 MB over the network — download the file and run from "
              f"its directory to skip this)", file=sys.stderr)
        try:
            import fsspec, huggingface_hub  # noqa: F401
        except ImportError:
            sys.exit("reading hf:// paths needs fsspec and huggingface_hub:\n"
                     "    pip install fsspec huggingface_hub\n"
                     f"or point --annotations at a local {ANNO_NAME}")
    df = pd.read_csv(src, dtype=DTYPES)
    df["part_id"] = df["part_id"].astype(str)
    return df


def require(*mods):
    """Fail with something actionable instead of a ModuleNotFoundError traceback."""
    missing = []
    for m in mods:
        try:
            __import__(m)
        except ImportError:
            missing.append(m)
    if missing:
        sys.exit(f"missing dependenc{'y' if len(missing) == 1 else 'ies'}: "
                 f"{', '.join(missing)}\n    pip install {' '.join(missing)}")


def load_partnext(columns=("model_id", "type_id")):
    """Metadata only. `masks` is deliberately excluded: parsing all 350k masks at
    once costs several GB of Python objects. Use iter_masks() to stream them."""
    require("datasets")
    from datasets import load_dataset
    ds = load_dataset(PNX_REPO, split="train")
    return ds.select_columns(list(columns)).to_pandas(), ds


def mask_keys(masks_json):
    """The part_id keys of a masks blob, without parsing the face lists."""
    depth, out = 0, []
    for m in re.finditer(r'[{}]|"(\d+)"\s*:\s*\{', masks_json):
        tok = m.group(0)
        if m.group(1) is not None:
            if depth == 1:
                out.append(m.group(1))
            depth += 1
        elif tok == "{":
            depth += 1
        else:
            depth -= 1
    return out


def normalise(masks_d):
    """4 parts upstream store a lone face index as a bare int instead of [int].
    Normalise every entry to {mesh_id: [face, ...]} so callers can assume one shape."""
    for pid, per_mesh in masks_d.items():
        if isinstance(per_mesh, dict):
            for mesh_id, faces in per_mesh.items():
                if isinstance(faces, int):
                    per_mesh[mesh_id] = [faces]
    return masks_d


def iter_masks(ds, wanted=None, batch=256):
    """Yield (model_id, masks_dict) one model at a time, already normalised."""
    for i in range(0, len(ds), batch):
        chunk = ds.select(range(i, min(i + batch, len(ds)))).to_dict()
        for mid, blob in zip(chunk["model_id"], chunk["masks"]):
            if wanted is None or mid in wanted:
                yield mid, normalise(json.loads(blob))


def iter_mask_keys(ds, batch=1024):
    """Yield (model_id, [part_id, ...]) without materialising face lists."""
    for i in range(0, len(ds), batch):
        chunk = ds.select(range(i, min(i + batch, len(ds)))).to_dict()
        for mid, blob in zip(chunk["model_id"], chunk["masks"]):
            yield mid, mask_keys(blob)


def attach_paths(ann, meta):
    """Add glb_path and the upstream type_id check. No mask data involved."""
    m = ann.merge(meta.rename(columns={"type_id": "type_id_upstream"}),
                  on="model_id", how="left", validate="many_to_one")
    m["glb_path"] = "glbs/" + m["type_id"] + "/" + m["model_id"] + ".glb"
    return m


def cmd_verify(a):
    ann = load_annotations(a.annotations)
    meta, ds = load_partnext()
    print(f"annotations   {len(ann):,} parts / {ann['model_id'].nunique():,} models")
    print(f"AuWang/PartNeXt {len(meta):,} models")

    A, U = set(ann["model_id"]), set(meta["model_id"])
    print(f"\nmodel_id      ours-not-upstream {len(A - U)}   upstream-not-annotated {len(U - A):,}")

    tid = ann[["model_id", "type_id"]].drop_duplicates().merge(
        meta, on="model_id", suffixes=("_ours", "_up"))
    n_tid_bad = int((tid["type_id_ours"] != tid["type_id_up"]).sum())
    print(f"type_id       mismatches {n_tid_bad} / {len(tid):,}")

    ours_by_model = ann.groupby("model_id", sort=False)["part_id"].apply(set).to_dict()
    tot = hit = up_parts = all_parts = 0
    bad_models = []
    for mid, keys in iter_mask_keys(ds):
        ks = set(keys)
        all_parts += len(ks)
        ours = ours_by_model.get(mid)
        if ours is None:
            continue
        up_parts += len(ks)
        tot += len(ours); hit += len(ours & ks)
        if ours - ks:
            bad_models.append(mid)
    print(f"part_id       {hit:,} / {tot:,} resolve to a masks key  ({hit / tot * 100:.4f}%)")
    print(f"              models with an unresolvable part_id: {len(bad_models)}")
    print(f"\ncoverage      {tot:,} / {up_parts:,} parts of the annotated models ({tot / up_parts * 100:.1f}%)")
    print(f"              {tot:,} / {all_parts:,} parts of all PartNeXt       ({tot / all_parts * 100:.1f}%)")
    ok = (hit == tot) and not (A - U) and n_tid_bad == 0
    print("\n" + ("JOIN IS EXACT" if ok else "JOIN HAS GAPS — see above"))
    return 0 if ok else 1


def cmd_table(a):
    ann = load_annotations(a.annotations)
    meta, ds = load_partnext()
    m = attach_paths(ann, meta)
    nf = {}
    wanted = set(ann["model_id"])
    for mid, md in iter_masks(ds, wanted):
        for pid, per_mesh in md.items():
            nf[(mid, pid)] = sum(len(v) for v in per_mesh.values())
    m["n_faces"] = [nf.get((a_, b_), 0) for a_, b_ in zip(m["model_id"], m["part_id"])]
    out = m
    if a.out.endswith(".parquet"):
        require("pyarrow")
        out.to_parquet(a.out, index=False)
    else:
        out.to_csv(a.out, index=False)
    print(f"{len(out):,} rows -> {a.out}")
    print(f"  parts with an empty mask: {int((m['n_faces'] == 0).sum())}")
    print(f"  total faces covered: {int(m['n_faces'].sum()):,}")


def cmd_export(a):
    ann = load_annotations(a.annotations)
    meta, ds = load_partnext()
    m = attach_paths(ann, meta)
    os.makedirs(a.out, exist_ok=True)
    wanted = set(ann["model_id"])
    groups = {mid: g for mid, g in m.groupby("model_id", sort=False)}
    n = 0
    for mid, md in iter_masks(ds, wanted):
        g = groups.get(mid)
        if g is None:
            continue
        r0 = g.iloc[0]
        doc = {
            "model_id": mid,
            "type_id": r0["type_id"],
            "object_category": r0["object_category"],
            "glb": {"repo": MESH_REPO, "path": r0["glb_path"]},
            "solver_plan": r0.get("sim_object_solver_plan"),
            "stiffness_ratio": _num(r0.get("sim_object_stiffness_ratio")),
            "parts": [],
        }
        for _, r in g.iterrows():
            doc["parts"].append({
                "part_id": r["part_id"],
                "part_name": r["part_name"],
                "part_path": None if _isnan(r["part_path"]) else r["part_path"],
                "faces": md.get(r["part_id"]),            # {mesh_id: [face indices]}
                "material_class": r["material_class"],
                "bucket": None if _isnan(r["bucket"]) else r["bucket"],
                "confidence": _num(r["confidence"]),
                "density_kg_m3": _num(r["density_kg_m3"]),
                "youngs_modulus_Pa": (None if _isnan(r["youngs_modulus_GPa"])
                                      else float(r["youngs_modulus_GPa"]) * 1e9),
                "poissons_ratio": _num(r["poissons_ratio"]),
                "body_type": None if _isnan(r["sim_body_type"]) else r["sim_body_type"],
                "caveat": None if _isnan(r["sim_caveat"]) else r["sim_caveat"],
            })
        with open(os.path.join(a.out, f"{mid}.json"), "w") as f:
            json.dump(doc, f, ensure_ascii=False)
        n += 1
        if a.limit and n >= a.limit:
            break
    print(f"{n:,} model JSON files -> {a.out}/")
    print("  youngs_modulus is in Pa (SI) for direct solver input; density in kg/m^3")


def cmd_show(a):
    ann = load_annotations(a.annotations)
    ann = ann[ann["model_id"] == a.model_id]
    if ann.empty:
        sys.exit(f"{a.model_id} is not in the annotations")
    meta, ds = load_partnext()
    m = attach_paths(ann, meta)
    md = dict(iter_masks(ds, {a.model_id})).get(a.model_id, {})
    m["n_faces"] = [sum(len(v) for v in md.get(p, {}).values()) for p in m["part_id"]]
    r0 = m.iloc[0]
    print(f"model {a.model_id}   category {r0['object_category']}")
    print(f"GLB   {MESH_REPO}/{r0['glb_path']}")
    print(f"plan  {r0['sim_object_solver_plan']}   stiffness ratio {r0['sim_object_stiffness_ratio']}")
    print()
    hdr = f"{'part_id':>7} {'part_name':<26} {'material':<10} {'bucket':<22} {'rho':>7} {'E[GPa]':>10} {'nu':>5} {'faces':>7}"
    print(hdr); print("-" * len(hdr))
    for _, r in m.iterrows():
        bucket = "-" if _isnan(r["bucket"]) else str(r["bucket"])[:22]
        print(f"{r['part_id']:>7} {str(r['part_name'])[:26]:<26} {str(r['material_class']):<10} "
              f"{bucket:<22} {_f(r['density_kg_m3'],7,0)} {_f(r['youngs_modulus_GPa'],10,5)} "
              f"{_f(r['poissons_ratio'],5,2)} {r['n_faces']:>7,}")


def _isnan(v):
    return v is None or (isinstance(v, float) and v != v)


def _num(v):
    return None if _isnan(v) else float(v)


def _f(v, w, p):
    return " " * (w - 1) + "-" if _isnan(v) else f"{v:>{w}.{p}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", default=None,
                    help=f"path to {ANNO_NAME} (default: the copy next to this script, "
                         f"else stream it from the Hub)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify", help="check every part_id against the official masks")
    p = sub.add_parser("table", help="flat joined table"); p.add_argument("--out", default="joined.parquet")
    p = sub.add_parser("export", help="per-model simulation JSON")
    p.add_argument("--out", default="sim_json"); p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("show", help="print one model"); p.add_argument("--model-id", required=True)
    a = ap.parse_args()
    return {"verify": cmd_verify, "table": cmd_table, "export": cmd_export, "show": cmd_show}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main() or 0)

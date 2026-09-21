#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end test for join_with_partnext.py and download_partnext_meshes.py.

Runs both scripts against a 10-model subset chosen to hit every edge case in the
release, downloads those meshes for real, and checks the face indices against the
actual GLB geometry.

    pip install datasets pandas trimesh pyarrow
    python3 test_scripts.py                # ~65 MB of downloads, a few minutes
    python3 test_scripts.py --keep         # leave the workspace for inspection

Exit code 0 means every check passed.
"""
import argparse, json, os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ANN = os.path.join(HERE, "PartNeXt_material_annotations.csv")
MANIFEST = os.path.join(HERE, "manifest_models.csv")
JOIN = os.path.join(HERE, "join_with_partnext.py")
DL = os.path.join(HERE, "download_partnext_meshes.py")

# One model per edge case. Comments say what each one exercises.
SUBSET = {
    "bc8b5553c41b49e78e061be1842f0668": "masks value is a bare int for one mesh",
    "dff4960cb776467684a95e0602ead938": "masks values are bare ints only",
    "4a644776b427444b9c7eff1788b22abc": "part_name '?' and null part_path",
    "80818614d3e44622b976870aef06b7ea": "abstained parts: all properties null",
    "298ed9ae6f9b41dd8ee0c0c5b412f343": "null bucket, sim_caveat, rigid_soft_coupling",
    "c46963d7d2014a9a91b539e3b6f80238": "solver plan implicit_or_substep",
    "0056bf73a1474aaf9d60227776a3e167": "single-part model",
    "18b61d38d26d4c8b9de20cc033a1c490": "4,747 parts -- the largest",
    "06dc90d6ad5d4d3c81a2195867aeb96d": "smallest GLB, stored outside LFS",
    "d16e69b830af41b19489ae7a51bf3671": "36 MB GLB, stored in LFS",
}

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    return cond


def run(cmd, **kw):
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return p.returncode, p.stdout, p.stderr


def make_subset(work):
    import pandas as pd
    ann = pd.read_csv(ANN, dtype={"model_id": str, "type_id": str, "part_id": str})
    man = pd.read_csv(MANIFEST, dtype=str)
    a = ann[ann["model_id"].isin(SUBSET)].copy()
    m = man[man["model_id"].isin(SUBSET)].copy()
    pa = os.path.join(work, "ann.csv")
    pm = os.path.join(work, "manifest.csv")
    a.to_csv(pa, index=False)
    m.to_csv(pm, index=False)
    print(f"subset: {len(m)} models, {len(a):,} parts, "
          f"{m['glb_bytes'].astype('int64').sum()/1e6:.1f} MB\n"
          if "glb_bytes" in m.columns else
          f"subset: {len(m)} models, {len(a):,} parts\n")
    return pa, pm, a, m


def t_join(work, pa):
    print("[1] join_with_partnext.py")
    rc, out, err = run([sys.executable, JOIN, "--annotations", pa, "verify"])
    check("verify exits 0", rc == 0, err.strip().splitlines()[-1] if rc else "")
    check("verify reports an exact join", "JOIN IS EXACT" in out)
    check("verify reports 100% part_id resolution", "(100.0000%)" in out)
    check("verify reports 0 type_id mismatches", "mismatches 0 /" in out)

    tbl = os.path.join(work, "joined.parquet")
    rc, out, _ = run([sys.executable, JOIN, "--annotations", pa, "table", "--out", tbl])
    check("table exits 0", rc == 0)
    check("table reports no empty masks", "parts with an empty mask: 0" in out)

    import pandas as pd
    d = pd.read_parquet(tbl)
    ann = pd.read_csv(pa, dtype={"model_id": str, "type_id": str, "part_id": str})
    check("table keeps every annotated part", len(d) == len(ann), f"{len(d):,} vs {len(ann):,}")
    check("table adds glb_path and n_faces", {"glb_path", "n_faces"} <= set(d.columns))
    check("every part has at least one face", bool((d["n_faces"] > 0).all()))
    check("glb_path is glbs/{type_id}/{model_id}.glb",
          bool((d["glb_path"] == "glbs/" + d["type_id"] + "/" + d["model_id"] + ".glb").all()))
    check("type_id survives as a zero-padded string",
          bool(d["type_id"].str.match(r"^\d{3}-\d{3}$").all()), str(d["type_id"].iloc[0]))
    check("upstream type_id agrees", bool((d["type_id"] == d["type_id_upstream"]).all()))

    exp = os.path.join(work, "sim_json")
    rc, out, _ = run([sys.executable, JOIN, "--annotations", pa, "export", "--out", exp])
    check("export exits 0", rc == 0)
    files = [f for f in os.listdir(exp) if f.endswith(".json")]
    check("export writes one JSON per model", len(files) == len(SUBSET), f"{len(files)}")

    docs = {f[:-5]: json.load(open(os.path.join(exp, f))) for f in files}
    npart = sum(len(v["parts"]) for v in docs.values())
    check("export keeps every part", npart == len(ann), f"{npart:,} vs {len(ann):,}")

    e_ok = nulls = 0
    for v in docs.values():
        for p in v["parts"]:
            if p["youngs_modulus_Pa"] is None:
                nulls += 1
            elif 1e2 <= p["youngs_modulus_Pa"] <= 1e12:
                e_ok += 1
    check("export converts E to Pa in a plausible range", e_ok + nulls == npart,
          f"in-range {e_ok:,}, null {nulls:,}")
    check("abstained parts export as null, not 0",
          all(p["density_kg_m3"] is None for v in docs.values() for p in v["parts"]
              if p["youngs_modulus_Pa"] is None))

    faces_are_lists = all(isinstance(x, list)
                          for v in docs.values() for p in v["parts"]
                          for x in (p["faces"] or {}).values())
    check("bare-int face indices are normalised to lists", faces_are_lists)

    q = [p for v in docs.values() for p in v["parts"] if p["part_name"] == "?"]
    check("the '?' placeholder parts survive export", len(q) > 0, f"{len(q)} parts")
    check("their null part_path exports as None", all(p["part_path"] is None for p in q))

    rc, out, _ = run([sys.executable, JOIN, "--annotations", pa, "show",
                      "--model-id", "298ed9ae6f9b41dd8ee0c0c5b412f343"])
    check("show exits 0", rc == 0)
    check("show renders a null bucket as '-', not 'nan'", " nan " not in out and "-" in out)

    rc, _, err = run([sys.executable, JOIN, "--annotations", pa, "show", "--model-id", "nope"])
    check("show rejects an unknown model_id", rc != 0 and "not in the annotations" in err)

    # with no --annotations the script must use the CSV beside it, not stream 185 MB
    rc, out, err = run([sys.executable, JOIN, "show",
                        "--model-id", "298ed9ae6f9b41dd8ee0c0c5b412f343"])
    check("default --annotations uses the local CSV, not the Hub",
          rc == 0 and "hf://" not in err, err.strip().splitlines()[0] if err.strip() else "")

    # a missing dependency must produce an instruction, not a traceback
    env = dict(os.environ, PYTHONPATH=os.path.join(work, "blocked"))
    os.makedirs(os.path.join(work, "blocked"), exist_ok=True)
    open(os.path.join(work, "blocked", "datasets.py"), "w").write("raise ImportError('blocked')")
    rc, out, err = run([sys.executable, JOIN, "--annotations", pa, "verify"], env=env)
    check("a missing dependency prints an install hint, not a traceback",
          rc != 0 and "pip install datasets" in err and "Traceback" not in err,
          err.strip().splitlines()[-1] if err.strip() else "")
    return docs


def t_download(work, pm, docs):
    print("\n[2] download_partnext_meshes.py")
    out = os.path.join(work, "glbs")

    rc, o, _ = run([sys.executable, DL, "--manifest", pm, "--out", out, "--verify", "--workers", "4"])
    check("--verify exits 0", rc == 0)
    check("--verify reaches every URL", f"ok {len(SUBSET):,}" in o.replace("ok 10", "ok 10"))

    rc, o, _ = run([sys.executable, DL, "--manifest", pm, "--out", out, "--workers", "4"])
    check("download exits 0", rc == 0, o.strip().splitlines()[-1] if rc else "")
    got = [os.path.join(dp, f) for dp, _, fs in os.walk(out) for f in fs if f.endswith(".glb")]
    check("downloads every file", len(got) == len(SUBSET), f"{len(got)}")
    check("leaves no .part files behind",
          not [f for dp, _, fs in os.walk(out) for f in fs if f.endswith(".part")])
    check("every file starts with the glTF magic",
          all(open(f, "rb").read(4) == b"glTF" for f in got))
    check("download layout matches the path export records",
          all(os.path.exists(os.path.join(out, d["glb"]["path"])) for d in docs.values()))
    check("no doubled glbs/glbs directory", not os.path.isdir(os.path.join(out, "glbs", "glbs")))

    import pandas as pd
    man = pd.read_csv(pm, dtype=str)
    if "glb_bytes" in man.columns:
        want = dict(zip(man["glb_relpath"], man["glb_bytes"].astype("int64")))
        sizes_ok = all(os.path.getsize(os.path.join(out, r)) == w for r, w in want.items())
        check("every file matches the manifest byte size", sizes_ok)

    rc, o, _ = run([sys.executable, DL, "--manifest", pm, "--out", out, "--workers", "4"])
    check("re-running skips everything (resume)", rc == 0 and f"already present {len(SUBSET):,}" in o)

    # the bug this replaced: a truncated file used to be accepted as complete
    victim = sorted(got, key=os.path.getsize)[-1]
    orig = os.path.getsize(victim)
    with open(victim, "r+b") as f:
        f.truncate(50_000)
    rc, o, _ = run([sys.executable, DL, "--manifest", pm, "--out", out, "--workers", "4"])
    check("a truncated file is re-downloaded, not skipped",
          rc == 0 and os.path.getsize(victim) == orig,
          f"{os.path.getsize(victim):,} vs {orig:,}")

    # a file with the right size but wrong content must fail the checksum, not the size test
    rc, o, _ = run([sys.executable, DL, "--manifest", pm, "--out", out, "--check"])
    if "no sha in manifest 10" not in o:
        check("--check passes on a clean tree", rc == 0 and "CORRUPT 0" in o)
        with open(victim, "r+b") as f:
            f.seek(orig // 2); f.write(b"\xde\xad\xbe\xef")
        rc, o, _ = run([sys.executable, DL, "--manifest", pm, "--out", out, "--check"])
        check("--check catches a same-size corruption", rc != 0 and "CORRUPT 1" in o)
        os.remove(victim)
        run([sys.executable, DL, "--manifest", pm, "--out", out, "--workers", "2"])
        rc, o, _ = run([sys.executable, DL, "--manifest", pm, "--out", out, "--check"])
        check("re-downloading repairs it", rc == 0 and "CORRUPT 0" in o)

    rc, o, _ = run([sys.executable, DL, "--manifest", pm, "--out", out, "--limit", "3", "--verify"])
    check("--limit is honoured", rc == 0 and "target 3" in o)
    return out


def t_geometry(out, docs):
    print("\n[3] face indices vs the real GLB geometry")
    try:
        import trimesh
    except ImportError:
        print("  SKIP  trimesh not installed")
        return
    tot_faces = 0
    for mid, doc in docs.items():
        path = os.path.join(out, doc["glb"]["path"])
        scene = trimesh.load(path, process=False)
        geoms = list(scene.geometry.values()) if hasattr(scene, "geometry") else [scene]
        nf = [len(g.faces) for g in geoms]
        bad = []
        for p in doc["parts"]:
            for mesh_id, faces in (p["faces"] or {}).items():
                k = int(mesh_id)
                if k >= len(nf):
                    bad.append((p["part_id"], mesh_id, "mesh_id out of range"))
                elif faces and max(faces) >= nf[k]:
                    bad.append((p["part_id"], mesh_id, f"face {max(faces)} >= {nf[k]}"))
                tot_faces += len(faces)
        check(f"{mid[:10]}… {len(doc['parts']):>4} parts, meshes {nf if len(nf)<4 else nf[:3]+['…']}",
              not bad, str(bad[:2]) if bad else "")
    print(f"  {tot_faces:,} face indices checked, all in range")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--work", default=None)
    a = ap.parse_args()
    work = a.work or tempfile.mkdtemp(prefix="pnx_test_")
    os.makedirs(work, exist_ok=True)
    print(f"workspace {work}\n")
    try:
        pa, pm, _, _ = make_subset(work)
        docs = t_join(work, pa)
        out = t_download(work, pm, docs)
        t_geometry(out, docs)
    finally:
        if not a.keep and not a.work:
            shutil.rmtree(work, ignore_errors=True)
    print(f"\n{'='*58}\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

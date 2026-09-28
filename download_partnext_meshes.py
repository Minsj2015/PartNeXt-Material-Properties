#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download exactly the PartNeXt GLB meshes annotated in PartNeXt_material_annotations.csv.

    python3 download_partnext_meshes.py --out ./partnext_mesh --token hf_xxx [--workers 6]
    python3 download_partnext_meshes.py --out ./partnext_mesh --verify   # HEAD only, no bodies
    python3 download_partnext_meshes.py --out ./partnext_mesh --check    # re-hash what is on disk

Files land at <out>/glbs/{type_id}/{model_id}.glb, mirroring AuWang/PartNeXt_mesh, which is
also exactly the `glb` path that join_with_partnext.py export writes. Do not pass --out ./glbs
or you get glbs/glbs/....

23,221 files, 55.1 GB. Driven by manifest_models.csv, which carries the expected byte
size and sha256 of every file, so an interrupted run resumes correctly instead of
leaving a truncated file behind.

Integrity
  * each file downloads to <name>.part and is renamed only after its size matches the
    manifest -- an interrupted run therefore never leaves a file that looks complete;
  * --check re-hashes files already on disk against the manifest sha256;
  * anonymous requests get rate-limited (HTTP 429), so pass --token (a free account is
    enough: https://huggingface.co/settings/tokens) or set HF_TOKEN.

Mesh data is AuWang/PartNeXt_mesh (CC BY 4.0); this script only fetches it.
"""
import argparse, csv, hashlib, os, queue, subprocess, sys, threading, time

REPO = "AuWang/PartNeXt_mesh"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
GLTF_MAGIC = b"glTF"


def rows(manifest):
    with open(manifest, newline="") as f:
        for r in csv.DictReader(f):
            yield (r["model_id"], r["type_id"], r["glb_relpath"],
                   int(r.get("glb_bytes") or 0), (r.get("glb_sha256") or "").strip())


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(buf), b""):
            h.update(b)
    return h.hexdigest()


def fetch(url, dst, token, head=False, timeout=300):
    cmd = ["curl", "-sSL", "--max-time", str(timeout), "-w", "%{http_code} %{size_download}"]
    cmd += ["-I", "-o", os.devnull] if head else ["-o", dst]
    if token:
        cmd += ["-H", f"Authorization: Bearer {token}"]
    cmd.append(url)
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = p.stdout.strip().split()
    return (out[0] if out else "000"), (int(out[1]) if len(out) > 1 and out[1].isdigit() else 0)


def complete(path, want_bytes):
    """A file on disk counts as done only if it matches the manifest size and is a GLB.
    Without the size check a truncated file from an interrupted run is silently kept."""
    if not os.path.exists(path):
        return False
    size = os.path.getsize(path)
    if want_bytes:
        if size != want_bytes:
            return False
    elif size <= 1000:
        return False
    with open(path, "rb") as f:
        return f.read(4) == GLTF_MAGIC


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(here, "manifest_models.csv"))
    ap.add_argument("--out", default="./partnext_mesh",
                    help="download root; files go to <out>/glbs/{type_id}/{model_id}.glb")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--verify", action="store_true", help="HEAD check every URL, download nothing")
    ap.add_argument("--check", action="store_true", help="re-hash files already on disk, download nothing")
    ap.add_argument("--limit", type=int, default=0, help="only the first N entries (debugging)")
    a = ap.parse_args()

    todo = list(rows(a.manifest))
    if a.limit:
        todo = todo[:a.limit]
    have_sha = sum(1 for t in todo if t[4])

    if a.check:
        return recheck(todo, a.out)
    if not a.token:
        print("[warn] no --token: HuggingFace rate-limits anonymous requests (HTTP 429).",
              file=sys.stderr)
    if not have_sha:
        print("[warn] manifest has no glb_sha256 column; integrity checks limited to size.",
              file=sys.stderr)

    os.makedirs(a.out, exist_ok=True)
    q = queue.Queue()
    for t in todo:
        q.put(t)
    lock = threading.Lock()
    stat = {"ok": 0, "skip": 0, "fail": 0, "bytes": 0, "429": 0, "badsize": 0}
    failed = []

    def worker():
        delay = 0.0
        while True:
            try:
                mid, tid, rel, want, _ = q.get_nowait()
            except queue.Empty:
                return
            dst = os.path.join(a.out, rel)
            url = f"{BASE}/{rel}"

            if a.verify:
                code, _ = fetch(url, dst, a.token, head=True)
                with lock:
                    if code == "200":
                        stat["ok"] += 1
                    else:
                        stat["fail"] += 1
                        failed.append(f"{mid},{tid},{code}")
                q.task_done()
                continue

            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if complete(dst, want):
                with lock:
                    stat["skip"] += 1
                q.task_done()
                continue

            part = dst + ".part"
            code = "000"
            for attempt in range(5):
                if delay:
                    time.sleep(delay)
                code, size = fetch(url, part, a.token)
                if code == "429":
                    with lock:
                        stat["429"] += 1
                    delay = min(max(delay * 2, 2.0), 60.0)
                    continue
                if code == "200" and complete(part, want):
                    os.replace(part, dst)          # atomic: never a half file at dst
                    with lock:
                        stat["ok"] += 1
                        stat["bytes"] += size
                    delay = max(delay * 0.5, 0.0)
                    break
                if code == "200":
                    with lock:
                        stat["badsize"] += 1
                time.sleep(1.0 + attempt)
            else:
                with lock:
                    stat["fail"] += 1
                    failed.append(f"{mid},{tid},{code}")
            if os.path.exists(part):
                try:
                    os.remove(part)
                except OSError:
                    pass

            with lock:
                n = stat["ok"] + stat["skip"] + stat["fail"]
                if n % 200 == 0:
                    print(f"  {n}/{len(todo)}  ok={stat['ok']} skip={stat['skip']} "
                          f"fail={stat['fail']} 429={stat['429']} {stat['bytes']/1e9:.2f}GB",
                          flush=True)
            q.task_done()

    ts = [threading.Thread(target=worker, daemon=True) for _ in range(a.workers)]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    print(f"\n{'verify' if a.verify else 'download'} finished   target {len(todo):,}")
    if not a.verify:
        print(f"  meshes at {os.path.abspath(a.out)}/glbs/<type_id>/<model_id>.glb")
    print(f"  ok {stat['ok']:,}   already present {stat['skip']:,}   failed {stat['fail']:,}   "
          f"rate-limited retries {stat['429']:,}")
    if stat["badsize"]:
        print(f"  size mismatches retried: {stat['badsize']:,}")
    if not a.verify:
        print(f"  {stat['bytes']/1e9:.2f} GB in {time.time()-t0:.0f}s")
        if have_sha:
            print(f"  run --check to re-hash against the manifest sha256")
    if failed:
        p = os.path.join(a.out, "failed.txt")
        os.makedirs(a.out, exist_ok=True)
        open(p, "w").write("\n".join(failed) + "\n")
        print(f"  failures -> {p}  (re-running resumes)")
        return 1
    return 0


def recheck(todo, out):
    miss = bad = ok = nosha = 0
    broken = []
    for i, (mid, tid, rel, want, want_sha) in enumerate(todo, 1):
        p = os.path.join(out, rel)
        if not os.path.exists(p):
            miss += 1
            continue
        if not want_sha:
            nosha += 1
            continue
        if sha256(p) == want_sha:
            ok += 1
        else:
            bad += 1
            broken.append(rel)
        if i % 500 == 0:
            print(f"  {i}/{len(todo)}  ok={ok} bad={bad} missing={miss}", flush=True)
    print(f"\nchecksum  ok {ok:,}   CORRUPT {bad:,}   missing {miss:,}   no sha in manifest {nosha:,}")
    if broken:
        p = os.path.join(out, "corrupt.txt")
        open(p, "w").write("\n".join(broken) + "\n")
        print(f"  corrupt files -> {p}  (delete them and re-run to refetch)")
    return 1 if (bad or miss) else 0


if __name__ == "__main__":
    sys.exit(main())

"""Fetch chunked NIH ChestX-ray14 shards for a lung nodule/mass model.

Why this dataset and this framing:

  * Chest radiography cannot diagnose cancer. What it can do is flag a
    pulmonary NODULE or MASS, which is the finding that sends a patient for a
    CT. So the target here is "suspicious nodule or mass", not "cancer" - the
    same triage logic as the TB model, where the X-ray flags and a
    confirmatory test decides.
  * The 320px mirror is stored as ~64-190 MB parquet shards, so we can pull
    only as many as we need instead of the 42 GB original.
  * Normal and abnormal films come from the same hospital, so a difference we
    measure is pathology rather than data source.

Shards are fetched in parallel, smallest first, and the script stops once it
has enough positives.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import pathlib
import time

import pyarrow.parquet as pq
import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "nih"
REPO = "arudaev/chest-xray-14-320"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main/data"
RAW = f"https://huggingface.co/datasets/{REPO}/resolve/main/"

TARGET_LABELS = {"Mass", "Nodule"}


def list_shards() -> list[tuple[str, int]]:
    r = requests.get(API, params={"limit": 200}, timeout=40)
    r.raise_for_status()
    out = []
    for it in r.json():
        if it.get("type") != "file" or not it["path"].endswith(".parquet"):
            continue
        size = it.get("size") or (it.get("lfs") or {}).get("size") or 0
        out.append((it["path"], int(size)))
    return sorted(out, key=lambda t: t[1])       # smallest first


def grab(path: str) -> tuple[str, bool, int]:
    dest = DEST / pathlib.Path(path).name
    DEST.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return path, True, dest.stat().st_size
    try:
        with requests.get(RAW + path, stream=True, timeout=300) as r:
            if r.status_code != 200:
                return path, False, 0
            tmp = dest.with_suffix(".part")
            with tmp.open("wb") as fh:
                for blk in r.iter_content(1 << 20):
                    fh.write(blk)
            tmp.replace(dest)
    except Exception:  # noqa: BLE001
        return path, False, 0
    return path, True, dest.stat().st_size


def count_positives() -> collections.Counter:
    c = collections.Counter()
    for f in sorted(DEST.glob("*.parquet")):
        try:
            labels = pq.ParquetFile(f).read(columns=["labels"]).to_pydict()["labels"]
        except Exception:  # noqa: BLE001
            continue
        for lab in labels:
            parts = {x.strip() for x in str(lab).split("|") if x.strip()}
            if parts & TARGET_LABELS:
                c["positive"] += 1
            elif parts == {"No Finding"}:
                c["normal"] += 1
            c["total"] += 1
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-positives", type=int, default=700)
    ap.add_argument("--max-shards", type=int, default=22)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    shards = list_shards()
    total_mb = sum(s for _, s in shards) / 1e6
    print(f"{len(shards)} shards available, {total_mb:.0f} MB total "
          f"(the 42 GB original, at 320px)")

    have = {p.name for p in DEST.glob("*.parquet")}
    todo = [p for p, _ in shards if pathlib.Path(p).name not in have]

    t0 = time.time()
    fetched_mb = 0.0
    for batch_start in range(0, min(len(todo), args.max_shards), args.workers):
        batch = todo[batch_start:batch_start + args.workers]
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            for path, ok, size in ex.map(grab, batch):
                fetched_mb += size / 1e6
                print(f"  {'+' if ok else '!'} {pathlib.Path(path).name} "
                      f"{size / 1e6:.0f} MB", flush=True)
        c = count_positives()
        print(f"  -> {c['positive']} nodule/mass positives, {c['normal']} "
              f"normals, {c['total']} rows "
              f"({fetched_mb:.0f} MB in {time.time() - t0:.0f}s)", flush=True)
        if c["positive"] >= args.min_positives:
            print("  enough positives; stopping early")
            break

    c = count_positives()
    n_shards = len(list(DEST.glob("*.parquet")))
    disk = sum(p.stat().st_size for p in DEST.glob("*.parquet")) / 1e6
    print(f"\n{n_shards} shards on disk, {disk:.0f} MB")
    print(f"nodule/mass positives : {c['positive']}")
    print(f"No Finding normals    : {c['normal']}")
    print(f"rows scanned          : {c['total']}")


if __name__ == "__main__":
    main()

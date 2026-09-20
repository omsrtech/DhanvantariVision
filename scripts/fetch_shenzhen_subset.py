"""Fetch a balanced subset of the Shenzhen TB set as individual radiographs.

Why a subset. The full archive is 3.6 GB and the available bandwidth during
this build was ~0.2-1 MB/s on a single stream, i.e. hours. The mirror also
exposes each radiograph as its own file, so pulling a balanced sample over many
parallel connections gets usable cross-site data in minutes instead.

What this buys, and it is the important part: Shenzhen is a different country,
different equipment and a different population from Montgomery. Having both
enables a **train-on-Shenzhen / test-on-Montgomery** evaluation, which is the
only honest way to tell whether the model learned tuberculosis or learned one
hospital's scanner.

Labels are in the filename (CHNCXR_0327_1.png -> 1 = TB, 0 = normal), so a
balanced sample can be drawn without downloading anything first.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import pathlib
import random
import sys
import threading
import time

import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "raw" / "shenzhen_png"
REPO = "Famatsu123/montgomery-shenzhen-tuberculosis-cxr"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main/ChinaSet_AllFiles/CXR_png"
RAW = f"https://huggingface.co/datasets/{REPO}/resolve/main/"

_lock = threading.Lock()
_done = {"n": 0, "bytes": 0}


def list_files() -> list[str]:
    r = requests.get(API, params={"limit": 1000}, timeout=40)
    r.raise_for_status()
    return sorted(i["path"] for i in r.json()
                  if i.get("type") == "file" and i["path"].endswith(".png"))


def label_of(path: str) -> int | None:
    stem = pathlib.Path(path).stem
    try:
        return int(stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return None


def grab(path: str, total: int) -> bool:
    dest = DEST / pathlib.Path(path).name
    if dest.exists() and dest.stat().st_size > 10_000:
        with _lock:
            _done["n"] += 1
        return True
    try:
        with requests.get(RAW + path, stream=True, timeout=180) as r:
            if r.status_code != 200:
                return False
            tmp = dest.with_suffix(".part")
            n = 0
            with tmp.open("wb") as fh:
                for blk in r.iter_content(1 << 18):
                    fh.write(blk)
                    n += len(blk)
            tmp.replace(dest)
    except Exception as e:  # noqa: BLE001
        print(f"  ! {pathlib.Path(path).name}: {type(e).__name__}", file=sys.stderr)
        return False

    with _lock:
        _done["n"] += 1
        _done["bytes"] += n
        if _done["n"] % 20 == 0:
            print(f"  {_done['n']}/{total}  {_done['bytes'] / 1e6:.0f} MB",
                  flush=True)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=110,
                    help="radiographs to fetch per class")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    files = list_files()
    by_label: dict[int, list[str]] = {0: [], 1: []}
    for f in files:
        lab = label_of(f)
        if lab in (0, 1):
            by_label[lab].append(f)
    print(f"archive lists {len(files)} radiographs "
          f"(normal={len(by_label[0])}, TB={len(by_label[1])})")

    rng = random.Random(args.seed)
    picked = []
    for lab in (0, 1):
        pool = by_label[lab][:]
        rng.shuffle(pool)
        picked += pool[:args.per_class]
    rng.shuffle(picked)
    print(f"fetching {len(picked)} files with {args.workers} workers "
          f"-> {DEST}")

    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(lambda p: grab(p, len(picked)), picked))

    el = time.time() - t0
    have = sorted(DEST.glob("*.png"))
    print(f"\n{sum(results)}/{len(picked)} ok in {el:.0f}s "
          f"({_done['bytes'] / 1e6 / max(el, 1):.1f} MB/s aggregate)")
    print(f"on disk: {len(have)} radiographs, "
          f"{sum(p.stat().st_size for p in have) / 1e6:.0f} MB")
    n1 = sum(1 for p in have if label_of(p.name) == 1)
    print(f"  normal={len(have) - n1}  TB={n1}")


if __name__ == "__main__":
    main()

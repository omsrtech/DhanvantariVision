"""Download the NIH/NLM tuberculosis chest X-ray datasets and ImageNet weights.

Two public sets released by the U.S. National Library of Medicine, the standard
open benchmark for TB screening from chest radiographs:

  * Montgomery County, Maryland  - 138 radiographs   (~602 MB)
  * Shenzhen, China              - 662 radiographs   (~3.68 GB)

Labels are encoded in the filename: the digit after the final underscore is 0
for normal and 1 for a TB-consistent abnormality (e.g. CHNCXR_0327_1.png).

Source choice matters for build time. The canonical host
(openi.nlm.nih.gov) served at ~0.45 MB/s during development, which puts the
Shenzhen archive at over two hours. The HuggingFace mirror carries the same
NLM zips and serves an order of magnitude faster, so it is the primary here
with the NLM original kept as fallback.

Downloads run concurrently, resume on restart, and are size-verified.
"""
from __future__ import annotations

import concurrent.futures as cf
import pathlib
import sys
import threading
import time

import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MODELS = ROOT / "models"

HF = ("https://huggingface.co/datasets/Famatsu123/"
      "montgomery-shenzhen-tuberculosis-cxr/resolve/main/")
NLM = "https://openi.nlm.nih.gov/imgs/collections/"

# name -> (primary url, fallback url or None, destination dir)
TARGETS = {
    "MontgomerySet.zip": (HF + "MontgomerySet.zip",
                          NLM + "NLM-MontgomeryCXRSet.zip", RAW),
    "ChinaSet_AllFiles.zip": (HF + "ChinaSet_AllFiles.zip",
                              NLM + "ChinaSet_AllFiles.zip", RAW),
    "resnet18-f37072fd.pth": (
        "https://download.pytorch.org/models/resnet18-f37072fd.pth", None, MODELS),
}

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def remote_size(url: str) -> int:
    try:
        r = requests.head(url, timeout=30, allow_redirects=True)
        return int(r.headers.get("content-length", 0))
    except Exception:
        return 0


def fetch(name: str, url: str, dest_dir: pathlib.Path) -> bool:
    dest = dest_dir / name
    dest_dir.mkdir(parents=True, exist_ok=True)
    total = remote_size(url)
    have = dest.stat().st_size if dest.exists() else 0

    if total and have == total:
        log(f"  = {name} already complete ({total / 1e6:.0f} MB)")
        return True
    if total and have > total:
        have = 0

    headers = {"Range": f"bytes={have}-"} if have else {}
    t0, mark = time.time(), have
    try:
        with requests.get(url, headers=headers, stream=True, timeout=120) as r:
            if r.status_code not in (200, 206):
                log(f"  ! {name}: HTTP {r.status_code}")
                return False
            with dest.open("ab" if have else "wb") as fh:
                for blk in r.iter_content(1 << 20):
                    fh.write(blk)
                    have += len(blk)
                    if have - mark >= 100 << 20:      # every 100 MB
                        el = max(time.time() - t0, 1e-3)
                        pct = f"{100 * have / total:.0f}%" if total else "?"
                        log(f"    {name}: {have / 1e6:>6.0f} MB {pct:>5} "
                            f"({(have - (dest.stat().st_size - have + have)) or 0:.0f}"
                            f"{'':0s} {(have - mark) / el / 1e6:.1f} MB/s)")
                        mark = have
    except Exception as e:  # noqa: BLE001
        log(f"  ! {name}: {type(e).__name__}: {e}")
        return False

    got = dest.stat().st_size
    ok = (not total) or got == total
    log(f"  {'+' if ok else '!'} {name} {got / 1e6:.0f} MB in "
        f"{time.time() - t0:.0f}s")
    return ok


def fetch_with_fallback(name: str, primary: str, fallback: str | None,
                        dest: pathlib.Path) -> tuple[str, bool]:
    if fetch(name, primary, dest):
        return name, True
    if fallback:
        log(f"  ~ {name}: primary failed, trying NLM fallback (slower)")
        return name, fetch(name, fallback, dest)
    return name, False


def main() -> int:
    t0 = time.time()
    log(f"fetching {len(TARGETS)} files concurrently")
    results = {}
    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        futs = [ex.submit(fetch_with_fallback, n, p, f, d)
                for n, (p, f, d) in TARGETS.items()]
        for fut in cf.as_completed(futs):
            name, ok = fut.result()
            results[name] = ok

    bad = [n for n, ok in results.items() if not ok]
    log(f"\nfinished in {time.time() - t0:.0f}s; failures: {bad or 'none'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Assemble a demo set of radiographs the model has never been trained on.

Three groups, chosen to show what the model can and cannot do:

  held_out_tb / held_out_normal
      Films from our own test split. The model never saw these during
      training, and their true labels are known, so they demonstrate the
      screener working as intended.

  external_pneumonia
      Chest radiographs with PNEUMONIA, from a different public dataset
      entirely. The model has never seen pneumonia in any form - it was
      trained only on "TB-consistent" versus "normal". These exist to show
      the honest failure mode: a binary TB screener has no category for other
      pathology, so it must force pneumonia into one of its two buckets.

That third group is the important one. It is the difference between "this
detects TB" and "this detects the kind of abnormality it was trained on".
"""
from __future__ import annotations

import concurrent.futures as cf
import pathlib
import random
import shutil
import sys

import pandas as pd
import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
DEMO = ROOT / "data" / "demo"

# A different dataset, used ONLY as out-of-distribution demo material - never
# for training. Its own classes come from mismatched sources, which is exactly
# why it is unfit for training but fine for showing OOD behaviour.
EXT_REPO = "realsudarshan/chest-xray-tb-pneumonia"
EXT_API = f"https://huggingface.co/api/datasets/{EXT_REPO}/tree/main/test/PNEUMONIA"
EXT_RAW = f"https://huggingface.co/datasets/{EXT_REPO}/resolve/main/"


def copy_held_out(n_each: int = 4) -> None:
    labels = pd.read_csv(PROC / "labels.csv")
    preds_p = PROC / "predictions.csv"
    test = labels[labels.split == "test"]
    rng = random.Random(11)

    for label, name in [(1, "held_out_tb"), (0, "held_out_normal")]:
        out = DEMO / name
        out.mkdir(parents=True, exist_ok=True)
        ids = test[test.label == label].image_id.tolist()
        rng.shuffle(ids)
        # Copy the ORIGINAL radiographs, not data/processed/img512 - those are
        # already CLAHE-normalised, and Screener.preprocess() would apply CLAHE
        # a second time, scoring a different image than the model trained on.
        raw_dir = ROOT / "data" / "raw" / "shenzhen_png"
        copied = 0
        for image_id in ids:
            src = raw_dir / f"{image_id}.png"
            if src.exists():
                shutil.copy2(src, out / f"{image_id}.png")
                copied += 1
            if copied >= n_each:
                break
        print(f"  {name}: {len(list(out.glob('*.png')))} films")
    if preds_p.exists():
        print("  (true labels and the model's own scores for these are in "
              "predictions.csv)")


def fetch_external(n: int = 6) -> None:
    out = DEMO / "external_pneumonia"
    out.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(EXT_API, params={"limit": 1000}, timeout=40)
        r.raise_for_status()
        files = [i["path"] for i in r.json()
                 if i.get("type") == "file" and i["path"].endswith(".png")]
    except Exception as e:  # noqa: BLE001
        print(f"  ! could not list external set: {type(e).__name__}: {e}",
              file=sys.stderr)
        return

    rng = random.Random(5)
    rng.shuffle(files)
    picked = files[:n]

    def grab(path: str) -> bool:
        dest = out / pathlib.Path(path).name
        if dest.exists() and dest.stat().st_size > 5_000:
            return True
        try:
            resp = requests.get(EXT_RAW + path, timeout=120)
            if resp.status_code != 200:
                return False
            dest.write_bytes(resp.content)
            return True
        except Exception:  # noqa: BLE001
            return False

    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        ok = sum(ex.map(grab, picked))
    print(f"  external_pneumonia: {ok}/{len(picked)} films "
          f"({len(list(out.glob('*.png')))} on disk)")


def main() -> None:
    DEMO.mkdir(parents=True, exist_ok=True)
    print("assembling demo radiographs the model has not been trained on:")
    copy_held_out()
    fetch_external()
    total = len(list(DEMO.rglob("*.png")))
    print(f"\n{total} films in {DEMO}")
    for d in sorted(DEMO.iterdir()):
        if d.is_dir():
            print(f"  {d.name}/  {len(list(d.glob('*.png')))}")


if __name__ == "__main__":
    main()

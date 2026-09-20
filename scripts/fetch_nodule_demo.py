"""Assemble demo films for the lung nodule / mass condition.

Selection is by the model's own score on the HELD-OUT test split, so these are
films it never trained on:

  nodule_positive   labelled Mass or Nodule, correctly flagged (highest scores)
  nodule_normal     labelled No Finding, correctly cleared (lowest scores)
  nodule_missed     labelled Mass or Nodule but scored low - the miss

Two things this script is careful about:

1. **It extracts the ORIGINAL image bytes from the parquet shards**, not the
   files in data/processed_nodule/img512. Those are already CLAHE-normalised,
   and Screener.preprocess() applies CLAHE again - which silently scores a
   different image than the model trained on. That bug bit us once already.

2. **"Nodule or mass" is not "cancer".** A chest radiograph cannot diagnose
   malignancy; a nodule or mass is the finding that warrants a CT. The folder
   names and the UI say nodule/mass for that reason.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[1]
SHARDS = ROOT / "data" / "nih"
PROC = ROOT / "data" / "processed_nodule"
DEMO = ROOT / "data" / "demo"


def build_index() -> dict[str, tuple[pathlib.Path, int]]:
    """image_id -> (shard, row index), so originals can be pulled on demand."""
    idx: dict[str, tuple[pathlib.Path, int]] = {}
    for f in sorted(SHARDS.glob("*.parquet")):
        names = pq.ParquetFile(f).read(columns=["filename"]).to_pydict()["filename"]
        for i, fn in enumerate(names):
            idx.setdefault(pathlib.Path(str(fn)).stem, (f, i))
    return idx


def extract(idx, image_id: str, dest: pathlib.Path) -> bool:
    hit = idx.get(image_id)
    if hit is None:
        return False
    shard, row = hit
    blob = pq.ParquetFile(shard).read(columns=["image"]).to_pydict()["image"][row]
    raw = blob["bytes"] if isinstance(blob, dict) else blob
    if not raw:
        return False
    dest.write_bytes(raw)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()

    preds = pd.read_csv(PROC / "predictions.csv")
    test = preds[preds.split == "test"]
    if test.empty:
        raise SystemExit("no test-split predictions; run predict_all.py first")

    pos = test[test.label == 1].sort_values("probability", ascending=False)
    neg = test[test.label == 0].sort_values("probability")

    groups = {
        "nodule_positive": pos.head(args.n),
        "nodule_normal": neg.head(args.n),
        "nodule_missed": pos.tail(2),
    }

    print("building index over the parquet shards...")
    idx = build_index()
    print(f"  {len(idx)} films indexed\n")

    for name, rows in groups.items():
        out = DEMO / name
        out.mkdir(parents=True, exist_ok=True)
        for old in out.glob("*.png"):
            old.unlink()
        ok = 0
        for r in rows.itertuples():
            if extract(idx, r.image_id, out / f"{r.image_id}.png"):
                ok += 1
                print(f"  {name:<16} {r.image_id:<16} "
                      f"model score {r.probability:.3f}  "
                      f"(truth: {'nodule/mass' if r.label == 1 else 'No Finding'})")
        print(f"  -> {name}: {ok} films\n")

    total = sum(len(list((DEMO / g).glob('*.png'))) for g in groups)
    print(f"{total} demo films in {DEMO}")
    print("\nNOTE: these are nodule/mass findings, NOT confirmed cancer. A "
          "chest radiograph cannot diagnose malignancy - a nodule or mass is "
          "the finding that warrants a CT.")


if __name__ == "__main__":
    main()

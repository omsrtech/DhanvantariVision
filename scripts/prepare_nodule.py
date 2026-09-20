"""Build the lung nodule/mass dataset from the chunked NIH shards.

Target: pulmonary NODULE or MASS versus No Finding. Framed as a triage target,
not a diagnosis - a chest radiograph cannot diagnose cancer, but a nodule or
mass is the finding that sends a patient for a CT. Same logic as the TB model:
the X-ray flags, a confirmatory test decides.

One methodological point that matters more than anything else here:

  **The split is by PATIENT, not by image.** NIH filenames encode the patient
  (`00000013_005.png` is patient 13, study 5), and the same patient often
  appears in several films. An image-level split puts the same patient's
  radiographs in both train and test, which leaks and inflates every metric.
  Grouping by patient is the difference between a real number and a fake one.

Output mirrors the TB layout so the same trainer can consume it:
  data/processed_nodule/img512/<id>.png
  data/processed_nodule/labels.csv   image_id, source, patient, label, split
"""
from __future__ import annotations

import argparse
import collections
import pathlib

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[1]
SHARDS = ROOT / "data" / "nih"
PROC = ROOT / "data" / "processed_nodule"
IMG = PROC / "img512"
SIZE = 512
SEED = 4242
TARGET = {"Mass", "Nodule"}


def normalise(img: np.ndarray) -> np.ndarray:
    """Identical preprocessing to the TB pipeline, so the two are comparable."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
    return cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def patient_of(filename: str) -> str:
    """`00000013_005.png` -> patient `00000013`."""
    stem = pathlib.Path(str(filename)).stem
    return stem.split("_")[0] if "_" in stem else stem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--normals-per-positive", type=float, default=2.0)
    args = ap.parse_args()

    files = sorted(SHARDS.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no shards in {SHARDS} - run fetch_nodule_shards.py")
    IMG.mkdir(parents=True, exist_ok=True)

    pos, neg = [], []
    for f in files:
        t = pq.ParquetFile(f).read().to_pydict()
        for i, lab in enumerate(t["labels"]):
            parts = {x.strip() for x in str(lab).split("|") if x.strip()}
            fn = t["filename"][i] if "filename" in t else f"{f.stem}_{i}"
            rec = (f.name, i, str(fn))
            if parts & TARGET:
                pos.append(rec)
            elif parts == {"No Finding"}:
                neg.append(rec)
    print(f"{len(files)} shards: {len(pos)} nodule/mass, {len(neg)} No Finding")

    rng = np.random.default_rng(SEED)
    n_neg = min(len(neg), int(len(pos) * args.normals_per_positive))
    keep_neg = [neg[i] for i in rng.permutation(len(neg))[:n_neg]]
    keep = [(r, 1) for r in pos] + [(r, 0) for r in keep_neg]
    print(f"using {len(pos)} positives and {n_neg} normals "
          f"({len(keep)} films)")

    # decode and write images, grouped by shard to avoid re-reading
    by_shard: dict[str, list] = collections.defaultdict(list)
    for (shard, idx, fn), label in keep:
        by_shard[shard].append((idx, fn, label))

    rows = []
    for shard, items in by_shard.items():
        t = pq.ParquetFile(SHARDS / shard).read(columns=["image"]).to_pydict()
        for idx, fn, label in items:
            image_id = pathlib.Path(fn).stem or f"{shard}_{idx}"
            out = IMG / f"{image_id}.png"
            if not out.exists():
                blob = t["image"][idx]
                raw = blob["bytes"] if isinstance(blob, dict) else blob
                img = cv2.imdecode(np.frombuffer(raw, np.uint8),
                                   cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                cv2.imwrite(str(out), normalise(img))
            rows.append({"image_id": image_id, "source": "nih",
                         "patient": patient_of(fn), "label": int(label)})
        print(f"  {shard}: {len(items)} films decoded", flush=True)

    df = pd.DataFrame(rows).drop_duplicates(subset="image_id")

    # ---- PATIENT-level split (see module docstring) ----------------------
    patients = df.groupby("patient").label.max()          # patient is positive
    pats = np.array(patients.index)                        # if any film is
    labs = patients.to_numpy()
    df["split"] = "train"
    for lab in (0, 1):
        p = pats[labs == lab]
        p = p[rng.permutation(len(p))]
        n_test = max(1, int(round(0.18 * len(p))))
        n_val = max(1, int(round(0.14 * len(p))))
        df.loc[df.patient.isin(p[:n_test]), "split"] = "test"
        df.loc[df.patient.isin(p[n_test:n_test + n_val]), "split"] = "val"

    PROC.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROC / "labels.csv", index=False)

    print(f"\ntotal films: {len(df)}   unique patients: {df.patient.nunique()}")
    print("\nsplit x label (0 = No Finding, 1 = nodule/mass):")
    print(df.pivot_table(index="split", columns="label", values="image_id",
                         aggfunc="count", fill_value=0).to_string())
    print("\npatients per split (no patient appears in two splits):")
    print(df.groupby("split").patient.nunique().to_string())
    overlap = sum(
        1 for _, g in df.groupby("patient") if g.split.nunique() > 1)
    print(f"patients spanning splits: {overlap}  (must be 0)")
    print(f"\nwrote {PROC / 'labels.csv'} and "
          f"{len(list(IMG.glob('*.png')))} images")


if __name__ == "__main__":
    main()

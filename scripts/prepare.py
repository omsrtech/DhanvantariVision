"""Extract the TB chest X-ray zips, normalise the images, and build the splits.

Output:
  data/processed/img512/<image_id>.png   8-bit grayscale, 512x512
  data/processed/labels.csv              image_id, source, label, split, cross_site

Labels come from the filename: the digit after the final underscore is 0 for
normal and 1 for a TB-consistent abnormality (e.g. CHNCXR_0327_1.png).

Two splits are produced, deliberately:

  `split`       stratified random 70/15/15 train/val/test over the pooled data.
                The optimistic, conventional number.

  `cross_site`  train on Shenzhen, test on Montgomery. Different country,
                different equipment, different population. This is the honest
                measure of whether the model learned tuberculosis or learned
                one hospital's scanner, and it is normally much lower.
"""
from __future__ import annotations

import pathlib
import re
import zipfile

import cv2
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
IMG_DIR = PROC / "img512"
SIZE = 512
SEED = 1337

ZIPS = {
    "montgomery": RAW / "MontgomerySet.zip",
    "shenzhen": RAW / "ChinaSet_AllFiles.zip",
}
LABEL_RE = re.compile(r"_(\d)\.png$", re.IGNORECASE)


def is_cxr(name: str) -> bool:
    """Keep chest radiographs; drop the manual lung-segmentation masks."""
    low = name.lower()
    if not low.endswith(".png"):
        return False
    if "mask" in low or "__macosx" in low:
        return False
    return "cxr_png" in low or "/cxr/" in low


def normalise(img: np.ndarray) -> np.ndarray:
    """Grayscale, CLAHE for local contrast, square resize.

    CLAHE is standard for chest radiographs: it lifts local contrast in the
    lung fields without blowing out the mediastinum, and it partially
    normalises exposure differences between the two sites.
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
    return cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


PNG_DIRS = {"shenzhen": RAW / "shenzhen_png"}


def ingest_dir(source: str, d: pathlib.Path, rows: list) -> None:
    """Ingest loose radiographs fetched as individual files."""
    files = sorted(d.glob("*.png"))
    if not files:
        return
    print(f"  {source}: {len(files)} radiographs in {d.name}/")
    for f in files:
        m = LABEL_RE.search(f.name)
        if not m:
            continue
        label = int(m.group(1))
        if label not in (0, 1):
            continue
        out = IMG_DIR / f"{f.stem}.png"
        if not out.exists():
            img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"    ! undecodable: {f.name}")
                continue
            cv2.imwrite(str(out), normalise(img))
        rows.append({"image_id": f.stem, "source": source, "label": label})


def extract() -> pd.DataFrame:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for source, d in PNG_DIRS.items():
        if d.exists():
            ingest_dir(source, d, rows)
    for source, zpath in ZIPS.items():
        if not zpath.exists():
            print(f"  ! missing {zpath.name} - skipping {source}")
            continue
        if not zipfile.is_zipfile(zpath):
            # Still downloading, or truncated: skip rather than crash, so the
            # pipeline can run on whichever sets have fully arrived.
            print(f"  ! {zpath.name} is incomplete ("
                  f"{zpath.stat().st_size / 1e6:.0f} MB so far) - skipping {source}")
            continue
        with zipfile.ZipFile(zpath) as zf:
            names = [n for n in zf.namelist() if is_cxr(n)]
            print(f"  {source}: {len(names)} radiographs in archive")
            for n in names:
                m = LABEL_RE.search(pathlib.Path(n).name)
                if not m:
                    continue
                label = int(m.group(1))
                if label not in (0, 1):
                    continue
                image_id = pathlib.Path(n).stem
                out = IMG_DIR / f"{image_id}.png"
                if not out.exists():
                    buf = np.frombuffer(zf.read(n), dtype=np.uint8)
                    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        print(f"    ! undecodable: {n}")
                        continue
                    cv2.imwrite(str(out), normalise(img))
                rows.append({"image_id": image_id, "source": source,
                             "label": label})
    return pd.DataFrame(rows)


def add_splits(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    # pooled stratified 70/15/15
    df["split"] = "train"
    for (src, lab), grp in df.groupby(["source", "label"]):
        idx = np.array(grp.index, dtype=int)
        rng.shuffle(idx)
        n = len(idx)
        n_test = max(1, int(round(0.15 * n)))
        n_val = max(1, int(round(0.15 * n)))
        df.loc[idx[:n_test], "split"] = "test"
        df.loc[idx[n_test:n_test + n_val], "split"] = "val"

    # cross-site: train on Shenzhen, hold out Montgomery entirely
    df["cross_site"] = np.where(df["source"] == "montgomery", "test", "train")
    # carve a validation slice out of the Shenzhen training data
    sh = np.array(df.index[df["source"] == "shenzhen"], dtype=int)
    rng.shuffle(sh)
    df.loc[sh[:max(1, int(0.12 * len(sh)))], "cross_site"] = "val"
    return df


def main() -> None:
    df = extract()
    if df.empty:
        raise SystemExit("no images extracted - has scripts/fetch_data.py finished?")
    df = df.drop_duplicates(subset="image_id").reset_index(drop=True)
    df = add_splits(df)
    PROC.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROC / "labels.csv", index=False)

    print(f"\ntotal images: {len(df)}")
    print("\nby source and label (0=normal, 1=TB):")
    print(df.pivot_table(index="source", columns="label", values="image_id",
                         aggfunc="count", fill_value=0).to_string())
    print("\npooled split:")
    print(df.pivot_table(index="split", columns="label", values="image_id",
                         aggfunc="count", fill_value=0).to_string())
    print("\ncross-site split (train Shenzhen -> test Montgomery):")
    print(df.pivot_table(index="cross_site", columns="label",
                         values="image_id", aggfunc="count",
                         fill_value=0).to_string())
    print(f"\nwrote {PROC / 'labels.csv'} and {len(list(IMG_DIR.glob('*.png')))} "
          f"images to {IMG_DIR}")


if __name__ == "__main__":
    main()

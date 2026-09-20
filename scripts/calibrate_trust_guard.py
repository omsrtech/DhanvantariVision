"""Calibrate an out-of-distribution guard for uploaded films.

Motivation: a chest radiograph pulled off the open web scored 0.07 while its
Grad-CAM lit up both shoulders - regions containing no lung at all. The score
was meaningless, but nothing in the UI said so.

Two cheap signals, both measured on our own training films so the thresholds
are data-derived rather than guessed:

  off_lung_saliency
      Fraction of Grad-CAM saliency falling OUTSIDE the lung box (the zone
      grid spans y 0.12-0.92). Attention on shoulders, neck, abdomen or the
      image border means the score is not being driven by lung parenchyma.

  histogram_distance
      Chi-square distance between the film's intensity histogram (after our
      own preprocessing) and the mean training histogram. Catches films with
      unusual windowing, prior contrast enhancement, inversion or heavy
      compression - all common in web images.

Writes reports/trust_guard.json with the 95th/99th percentiles of each signal
across in-distribution films, which src/infer.py then uses as its thresholds.
"""
from __future__ import annotations

import json
import pathlib
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import data as D                         # noqa: E402
from gradcam import GradCAM              # noqa: E402
from infer import Screener               # noqa: E402

OUT = ROOT / "reports" / "trust_guard.json"
LUNG_Y = (0.12, 0.92)          # the zone grid's vertical extent
N_PER = 160


def off_lung_fraction(cam: np.ndarray) -> float:
    h = cam.shape[0]
    y0, y1 = int(LUNG_Y[0] * h), int(LUNG_Y[1] * h)
    total = float(cam.sum())
    if total <= 1e-8:
        return 0.0                       # empty map: nothing to mis-attend
    inside = float(cam[y0:y1, :].sum())
    return 1.0 - inside / total


def hist_of(img: np.ndarray) -> np.ndarray:
    h = cv2.calcHist([img], [0], None, [32], [0, 256]).ravel()
    return h / max(h.sum(), 1.0)


def chi2(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.sum((a - b) ** 2 / (a + b + 1e-10)))


def main() -> None:
    rng = np.random.default_rng(0)
    rows = []
    ref_hists = []

    for data_dir, tag in [("processed", ""), ("processed_nodule", "_nodule")]:
        lp = ROOT / "data" / data_dir / "labels.csv"
        if not lp.exists():
            continue
        scr = Screener(split="split", tag=tag)
        cam = GradCAM(scr.model, scr.device)
        df = pd.read_csv(lp)
        pick = df.iloc[rng.permutation(len(df))[:N_PER]]
        print(f"{data_dir}: sampling {len(pick)} films")
        for r in pick.itertuples():
            img = cv2.imread(str(ROOT / "data" / data_dir / "img512" /
                                 f"{r.image_id}.png"), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            ref_hists.append(hist_of(img))
            _, heat = cam(D.to_tensor(img).unsqueeze(0))
            rows.append({"set": data_dir, "off_lung": off_lung_fraction(heat)})

    d = pd.DataFrame(rows)
    mean_hist = np.mean(np.stack(ref_hists), axis=0)
    dists = [chi2(h, mean_hist) for h in ref_hists]

    report = {
        "n_films": int(len(d)),
        "off_lung_saliency": {
            "mean": float(d.off_lung.mean()),
            "p50": float(d.off_lung.quantile(0.50)),
            "p95": float(d.off_lung.quantile(0.95)),
            "p99": float(d.off_lung.quantile(0.99)),
            "max": float(d.off_lung.max()),
        },
        "histogram_distance": {
            "mean": float(np.mean(dists)),
            "p50": float(np.percentile(dists, 50)),
            "p95": float(np.percentile(dists, 95)),
            "p99": float(np.percentile(dists, 99)),
            "max": float(np.max(dists)),
        },
        "reference_histogram": mean_hist.tolist(),
        "lung_box_y": list(LUNG_Y),
        "note": ("Thresholds are the 95th percentile across in-distribution "
                 "films. Exceeding one means the film, or where the model "
                 "looked on it, is unlike anything in training."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))

    print("\nOFF-LUNG SALIENCY across in-distribution films")
    for k in ("mean", "p50", "p95", "p99", "max"):
        print(f"  {k:>5}: {report['off_lung_saliency'][k]:.3f}")
    print("\nHISTOGRAM DISTANCE to the training mean")
    for k in ("mean", "p50", "p95", "p99", "max"):
        print(f"  {k:>5}: {report['histogram_distance'][k]:.4f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

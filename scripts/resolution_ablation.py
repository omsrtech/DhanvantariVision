"""Is the NIH failure the site, or the resolution?

The NIH mirror stores 320px images. Our training films were 512px derived from
~3000px originals. So the NIH test confounds two things: a different hospital,
and a lossier image pipeline.

This isolates the second. It takes our OWN held-out test films - where the
model scores AUROC 0.945 - degrades them through the NIH pipeline
(512 -> 320 -> back up to 512), and re-scores. If performance collapses, the
resolution is the culprit and the NIH result says little about site transfer.
If it holds, the model genuinely fails on NIH films.
"""
from __future__ import annotations

import json
import pathlib
import sys

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from infer import Screener  # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = ROOT / "reports" / "resolution_ablation.json"


def boot(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    pt = roc_auc_score(y, p)
    st = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            st.append(roc_auc_score(y[i], p[i]))
    lo, hi = np.percentile(st, [2.5, 97.5])
    return float(pt), float(lo), float(hi)


def main() -> None:
    df = pd.read_csv(PROC / "labels.csv")
    test = df[df.split == "test"]
    scr = Screener()
    print(f"held-out test films: {len(test)}  (device {scr.device})\n")

    rows = []
    for r in test.itertuples():
        img = cv2.imread(str(PROC / "img512" / f"{r.image_id}.png"),
                         cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        native = scr.score_array(img, already_preprocessed=True).probability

        # simulate the NIH pipeline: down to 320, back up to 512
        small = cv2.resize(img, (320, 320), interpolation=cv2.INTER_AREA)
        degraded_img = cv2.resize(small, (512, 512), interpolation=cv2.INTER_CUBIC)
        degraded = scr.score_array(degraded_img,
                                   already_preprocessed=True).probability

        rows.append({"image_id": r.image_id, "label": int(r.label),
                     "native": native, "degraded": degraded})

    d = pd.DataFrame(rows)
    y = d.label.to_numpy()
    a_n, l_n, h_n = boot(y, d.native.to_numpy())
    a_d, l_d, h_d = boot(y, d.degraded.to_numpy())

    print(f"AUROC at native 512px      : {a_n:.4f}  [{l_n:.4f}, {h_n:.4f}]")
    print(f"AUROC after 320px round-trip: {a_d:.4f}  [{l_d:.4f}, {h_d:.4f}]")
    print(f"drop attributable to resolution: {a_n - a_d:.4f}")

    shift = float((d.degraded - d.native).mean())
    print(f"\nmean score shift from degradation: {shift:+.4f}")
    print(f"  on normals: {float((d[d.label==0].degraded - d[d.label==0].native).mean()):+.4f}")
    print(f"  on TB:      {float((d[d.label==1].degraded - d[d.label==1].native).mean()):+.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "n": int(len(d)),
        "auroc_native_512": {"auroc": a_n, "ci95": [l_n, h_n]},
        "auroc_320_roundtrip": {"auroc": a_d, "ci95": [l_d, h_d]},
        "auroc_drop": a_n - a_d,
        "mean_score_shift": shift,
    }, indent=2))

    print("\n" + "=" * 70)
    if a_n - a_d > 0.15:
        print("VERDICT: resolution explains much of the NIH failure. The NIH")
        print("result is largely a preprocessing artifact, not proof that the")
        print("model cannot transfer between hospitals. Re-test on full-")
        print("resolution NIH images before concluding anything about the site.")
    elif a_n - a_d > 0.05:
        print("VERDICT: resolution costs something but not everything. The NIH")
        print("failure is partly real site shift, partly pipeline mismatch.")
    else:
        print("VERDICT: resolution is not the problem. The model tolerates 320px")
        print("fine, so the NIH failure is genuine site/population shift.")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

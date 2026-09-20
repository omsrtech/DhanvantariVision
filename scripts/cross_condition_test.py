"""Does the nodule model detect nodules, or just "not-normal"?

A tuberculosis film scored 0.91 on the nodule/mass model, which has never seen
TB. Two very different explanations:

  DISEASE CONFUSION  the model fires on abnormal lungs generally, so TB
                     (which genuinely produces nodular opacities, cavities and
                     mass-like consolidation) lands in its positive class.
                     Prediction: TB-positive films score HIGH, TB-normal films
                     score LOW.

  DOMAIN SHIFT       the model simply cannot cope with films from a different
                     hospital, and pushes everything upward.
                     Prediction: BOTH TB-positive and TB-normal films score
                     high, and the labels barely separate.

AUROC on the TB labels settles it: well above 0.5 means the nodule model is
tracking real pathology across a disease it never trained on; near 0.5 means
its scores on these films are noise.

Runs both models over both datasets, so all four combinations are measured.
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

import data as D                     # noqa: E402
from gradcam import GradCAM          # noqa: E402
from infer import Screener           # noqa: E402

OUT = ROOT / "reports" / "cross_condition.json"
N = 260


def boot(y, p, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    pt = roc_auc_score(y, p)
    st = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            st.append(roc_auc_score(y[i], p[i]))
    lo, hi = np.percentile(st, [2.5, 97.5])
    return float(pt), float(lo), float(hi)


def score_set(scr, cam, data_dir: str, n: int, seed: int = 0):
    df = pd.read_csv(ROOT / "data" / data_dir / "labels.csv")
    rng = np.random.default_rng(seed)
    pick = df.iloc[rng.permutation(len(df))[:n]]
    ys, ps = [], []
    for r in pick.itertuples():
        img = cv2.imread(str(ROOT / "data" / data_dir / "img512" /
                             f"{r.image_id}.png"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        prob, _ = cam(D.to_tensor(img).unsqueeze(0))
        ys.append(int(r.label))
        ps.append(float(prob))
    return np.array(ys), np.array(ps)


def main() -> None:
    models = {}
    for label, tag in [("TB model", ""), ("nodule model", "_nodule")]:
        scr = Screener(split="split", tag=tag)
        models[label] = (scr, GradCAM(scr.model, scr.device), scr.threshold)

    report = {}
    for mlabel, (scr, cam, thr) in models.items():
        for dlabel, ddir in [("TB films (Montgomery+Shenzhen)", "processed"),
                             ("NIH films (nodule/mass set)", "processed_nodule")]:
            y, p = score_set(scr, cam, ddir, N)
            auc, lo, hi = boot(y, p)
            key = f"{mlabel} on {dlabel}"
            report[key] = {
                "n": int(len(y)),
                "auroc_on_that_set_labels": auc, "ci95": [lo, hi],
                "mean_score_positive": float(p[y == 1].mean()),
                "mean_score_negative": float(p[y == 0].mean()),
                "flag_rate_positive": float((p[y == 1] >= thr).mean()),
                "flag_rate_negative": float((p[y == 0] >= thr).mean()),
                "threshold": float(thr),
            }
            r = report[key]
            print(f"\n{key}  (n={r['n']}, threshold {thr:.3f})")
            print(f"  AUROC on that set's labels : {auc:.3f} "
                  f"[{lo:.3f}, {hi:.3f}]")
            print(f"  mean score, positives      : {r['mean_score_positive']:.3f}"
                  f"   flagged {r['flag_rate_positive']:.0%}")
            print(f"  mean score, negatives      : {r['mean_score_negative']:.3f}"
                  f"   flagged {r['flag_rate_negative']:.0%}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))

    k = "nodule model on TB films (Montgomery+Shenzhen)"
    r = report[k]
    print("\n" + "=" * 72)
    print("VERDICT on the nodule model applied to TB films:")
    if r["auroc_on_that_set_labels"] >= 0.70:
        print("  DISEASE CONFUSION, not noise. It separates TB from normal at")
        print(f"  AUROC {r['auroc_on_that_set_labels']:.3f} despite never having seen TB.")
        print("  It is tracking abnormal lung, and TB genuinely produces")
        print("  nodular and mass-like opacities, so TB lands in its positive")
        print("  class. The high score on a TB film is correct behaviour for")
        print("  what it is: a 'resembles my positives' detector.")
    elif r["auroc_on_that_set_labels"] >= 0.60:
        print("  PARTLY real signal, partly shift. Some separation survives,")
        print("  but scores are inflated overall.")
    else:
        print("  DOMAIN SHIFT. Scores on these films carry little information;")
        print("  the model cannot be applied across sites at all.")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

"""Quantify what transfers across sites and what does not.

The cross-site run produced a striking split: AUROC held up (0.945 pooled ->
0.899 cross-site) while specificity collapsed (0.885 -> 0.225). Those two facts
are not in tension, and the distinction is the most useful thing this project
found:

  * AUROC is threshold-free. It measures whether the model RANKS radiographs
    correctly, and ranking transfers between sites reasonably well.
  * Sensitivity and specificity are measured at a fixed threshold, so they
    depend on the model being CALIBRATED. Calibration is a property of the
    score distribution, and that distribution shifts when the scanner,
    exposure and population change.

Practical consequence: a screening threshold tuned at one hospital is not
transportable to another, even when the underlying model is still
discriminating well. This script measures the shift and shows how much of the
loss a small local recalibration sample recovers.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
OUT = ROOT / "reports" / "calibration.json"


def load(split: str) -> tuple[np.ndarray, np.ndarray, dict]:
    d = json.loads((MODELS / f"metrics_{split}.json").read_text())
    y = np.array(d["predictions"]["y"])
    p = np.array(d["predictions"]["p"], dtype=float)
    return y, p, d


def thr_for_sens(y, p, target=0.90) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(tpr >= target)[0]
    return float(thr[ok[0]]) if len(ok) else float(thr[np.argmax(tpr)])


def metrics_at(y, p, t) -> dict:
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    d = lambda a, b: float(a / b) if b else float("nan")  # noqa: E731
    return {"threshold": float(t), "sensitivity": d(tp, tp + fn),
            "specificity": d(tn, tn + fp), "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def main() -> None:
    yc, pc, dc = load("cross_site")
    ys, ps, ds = load("split")

    imported = dc["test"]["operating_point"]["threshold"]
    report = {
        "pooled": {
            "auroc": ds["test"]["auroc"], "ci": ds["test"]["auroc_ci95"],
            "at_operating_point": {
                k: ds["test"]["operating_point"][k]
                for k in ("threshold", "sensitivity", "specificity")},
        },
        "cross_site": {
            "auroc": dc["test"]["auroc"], "ci": dc["test"]["auroc_ci95"],
            "threshold_imported_from_source_site": metrics_at(yc, pc, imported),
        },
    }

    # --- score distribution shift -----------------------------------------
    report["score_shift"] = {
        "pooled_mean_score_normal": float(ps[ys == 0].mean()),
        "crosssite_mean_score_normal": float(pc[yc == 0].mean()),
        "pooled_mean_score_tb": float(ps[ys == 1].mean()),
        "crosssite_mean_score_tb": float(pc[yc == 1].mean()),
    }

    # --- what a locally-recalibrated threshold would give ------------------
    rng = np.random.default_rng(0)
    recal = {}
    for n_local in (20, 40, 80):
        sens, spec = [], []
        for _ in range(300):
            idx = rng.choice(len(yc), size=min(n_local, len(yc)), replace=False)
            if len(np.unique(yc[idx])) < 2:
                continue
            t = thr_for_sens(yc[idx], pc[idx], 0.90)
            held = np.setdiff1d(np.arange(len(yc)), idx)
            m = metrics_at(yc[held], pc[held], t)
            sens.append(m["sensitivity"]); spec.append(m["specificity"])
        recal[f"local_sample_{n_local}"] = {
            "median_sensitivity": float(np.nanmedian(sens)),
            "median_specificity": float(np.nanmedian(spec)),
            "n_trials": len(sens),
        }
    report["local_recalibration"] = recal

    # oracle: best achievable on the target site
    t_oracle = thr_for_sens(yc, pc, 0.90)
    report["cross_site"]["threshold_refit_on_target_site"] = metrics_at(yc, pc, t_oracle)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))

    # --- print -------------------------------------------------------------
    print("=" * 68)
    print("WHAT TRANSFERS ACROSS SITES, AND WHAT DOES NOT")
    print("=" * 68)
    pl, cs = report["pooled"], report["cross_site"]
    print(f"\nRANKING (threshold-free) -- transfers:")
    print(f"  pooled      AUROC {pl['auroc']:.4f}  CI [{pl['ci'][0]:.3f}, {pl['ci'][1]:.3f}]")
    print(f"  cross-site  AUROC {cs['auroc']:.4f}  CI [{cs['ci'][0]:.3f}, {cs['ci'][1]:.3f}]")
    print(f"  -> AUROC drop: {pl['auroc'] - cs['auroc']:.4f}")

    a = pl["at_operating_point"]; b = cs["threshold_imported_from_source_site"]
    print(f"\nCALIBRATION (fixed threshold) -- does NOT transfer:")
    print(f"  pooled      thr {a['threshold']:.4f}  sens {a['sensitivity']:.3f}  "
          f"spec {a['specificity']:.3f}")
    print(f"  cross-site  thr {b['threshold']:.4f}  sens {b['sensitivity']:.3f}  "
          f"spec {b['specificity']:.3f}   <-- threshold imported from source site")
    print(f"  -> specificity collapse: {a['specificity']:.3f} -> {b['specificity']:.3f} "
          f"({b['fp']} false positives out of {b['fp'] + b['tn']} normals)")

    s = report["score_shift"]
    print(f"\nWHY: the score distribution moved")
    print(f"  mean score on NORMAL films:  pooled {s['pooled_mean_score_normal']:.3f}"
          f"  ->  new site {s['crosssite_mean_score_normal']:.3f}")
    print(f"  mean score on TB films:      pooled {s['pooled_mean_score_tb']:.3f}"
          f"  ->  new site {s['crosssite_mean_score_tb']:.3f}")

    o = cs["threshold_refit_on_target_site"]
    print(f"\nTHE FIX: refit the threshold on local data")
    print(f"  refit on target site   thr {o['threshold']:.4f}  "
          f"sens {o['sensitivity']:.3f}  spec {o['specificity']:.3f}")
    for k, v in report["local_recalibration"].items():
        n = k.rsplit("_", 1)[1]
        print(f"  {n:>3} local films        median sens {v['median_sensitivity']:.3f}  "
              f"median spec {v['median_specificity']:.3f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

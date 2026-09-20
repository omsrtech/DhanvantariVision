"""Rewrite the `decision` column in cached predictions using the current rule.

Only the label changes - probabilities and Grad-CAM overlays are untouched, so
this is instant and needs no GPU. Run it after changing the decision logic
instead of re-running predict_all.py.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

DATASETS = [("processed", ""), ("processed_nodule", "_nodule")]


def decide(p: float, t_low: float, t_high: float, thr: float) -> str:
    positive = p >= thr
    confident = (p < t_low) or (p >= t_high)
    if positive and confident:
        return "screen positive - refer for confirmatory testing"
    if positive:
        return "screen positive, low confidence - human read required"
    if confident:
        return "screen negative"
    return "screen negative, low confidence - human read required"


for data_dir, tag in DATASETS:
    pred_p = ROOT / "data" / data_dir / "predictions.csv"
    met_p = ROOT / "models" / f"metrics_split{tag}.json"
    if not pred_p.exists() or not met_p.exists():
        print(f"  - skipping {data_dir} (missing files)")
        continue
    m = json.loads(met_p.read_text())
    op, band = m["test"]["operating_point"], m["test"]["abstention"]
    thr = float(op["threshold"])
    lo = float(band.get("t_low", thr))
    hi = float(band.get("t_high", thr))

    df = pd.read_csv(pred_p)
    df["decision"] = df.probability.map(lambda p: decide(p, lo, hi, thr))
    df.to_csv(pred_p, index=False)

    print(f"\n{data_dir}:  threshold {thr:.3f}   band [{lo:.3f}, {hi:.3f}]")
    print(df.decision.value_counts().to_string())
    t = df[df.split == "test"]
    if len(t):
        print(f"  held-out test (n={len(t)}):")
        print("   " + t.decision.value_counts().to_string().replace("\n", "\n   "))

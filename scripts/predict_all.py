"""Score every radiograph, compute Grad-CAM, and cache the results.

Writes:
  data/processed/predictions.csv   image_id, label, split, probability,
                                   decision, top zones
  data/processed/cam/<id>.png      Grad-CAM overlay, ready to display

Pre-computing means the app is pure file reading: instant, and it cannot fail
mid-demo because of a GPU or model-loading problem. The model is still loaded
live for newly uploaded images.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import data as D                       # noqa: E402
from gradcam import GradCAM, overlay, zone_attention   # noqa: E402
from model import build_model          # noqa: E402

PROC = ROOT / "data" / "processed"
CAM_DIR = PROC / "cam"
MODELS = ROOT / "models"


def decide(p: float, t_low: float, t_high: float, thr: float) -> str:
    """Direction from the operating threshold, confidence from the band."""
    positive = p >= thr
    confident = (p < t_low) or (p >= t_high)
    if positive and confident:
        return "screen positive - refer for confirmatory testing"
    if positive:
        return "screen positive, low confidence - human read required"
    if confident:
        return "screen negative"
    return "screen negative, low confidence - human read required"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_montgomery")
    ap.add_argument("--split", default="split")
    ap.add_argument("--data-dir", default="processed")
    args = ap.parse_args()

    global PROC, CAM_DIR
    PROC = ROOT / "data" / args.data_dir
    CAM_DIR = PROC / "cam"

    ckpt_path = MODELS / f"tbscope_{args.split}{args.tag}.pt"
    metrics_path = MODELS / f"metrics_{args.split}{args.tag}.json"
    if not ckpt_path.exists():
        raise SystemExit(f"no checkpoint at {ckpt_path} - run scripts/train.py")

    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    op = metrics.get("test", {}).get("operating_point", {})
    band = metrics.get("test", {}).get("abstention", {})
    thr = float(op.get("threshold", 0.5))
    t_low = float(band.get("t_low", thr))
    t_high = float(band.get("t_high", thr))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(None, num_classes=1).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
    model.eval()
    cam = GradCAM(model, device)
    print(f"device={device}  threshold={thr:.4f}  band=[{t_low:.4f},{t_high:.4f}]")

    df = pd.read_csv(PROC / "labels.csv")
    CAM_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, r in enumerate(df.itertuples(), 1):
        img = cv2.imread(str(PROC / "img512" / f"{r.image_id}.png"),
                         cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  ! missing image {r.image_id}")
            continue
        x = D.to_tensor(img).unsqueeze(0)
        prob, heat = cam(x)
        zones = zone_attention(heat)

        # overlay drawn on the 512px image for a crisp display
        big = cv2.resize(heat, (img.shape[1], img.shape[0]),
                         interpolation=cv2.INTER_CUBIC)
        ov = overlay(img, big)
        cv2.imwrite(str(CAM_DIR / f"{r.image_id}.png"),
                    cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))

        rows.append({
            "image_id": r.image_id, "label": int(r.label), "source": r.source,
            "split": r.split, "probability": round(float(prob), 5),
            "decision": decide(prob, t_low, t_high, thr),
            "zone1": zones[0][0], "zone1_frac": round(zones[0][1], 4),
            "zone2": zones[1][0], "zone2_frac": round(zones[1][1], 4),
            "zone3": zones[2][0], "zone3_frac": round(zones[2][1], 4),
        })
        if i % 25 == 0:
            print(f"  {i}/{len(df)}")

    out = pd.DataFrame(rows).sort_values("probability", ascending=False)
    out.to_csv(PROC / "predictions.csv", index=False)

    print(f"\nwrote {PROC / 'predictions.csv'} ({len(out)} rows)")
    print(f"overlays in {CAM_DIR}")
    print("\ndecision breakdown:")
    print(out.decision.value_counts().to_string())
    print("\nmean probability by true label:")
    print(out.groupby("label").probability.agg(["count", "mean"]).to_string())


if __name__ == "__main__":
    main()

"""Does the model detect tuberculosis, or just "abnormal chest"?

Runs the TB screener over NIH ChestX-ray14 - a third, completely independent
site (NIH Clinical Center, USA) with 14 labelled pathologies plus "No Finding".

Two questions, both unanswerable with our own data:

1. **Specificity at a third site.** Our normals so far came from Montgomery and
   Shenzhen. NIH "No Finding" films are a fresh negative population, so the
   flag rate on them is an independent specificity estimate.

2. **Is it TB-specific?** The model has never seen effusion, mass, nodule,
   cardiomegaly, pneumothorax or oedema. If it flags them at close to the rate
   it flags TB, then what it really detects is "this chest looks abnormal",
   and the product should be described that way.

Why this dataset and not an easier one: within ChestX-ray14 the normal and
abnormal films come from the SAME hospital and scanner, so any difference
measured is pathology rather than data source. A dataset whose classes come
from different sources would produce an impressive and meaningless result.
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import pathlib
import sys

import numpy as np
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2                                  # noqa: E402
from infer import Screener                  # noqa: E402

SHARD = ROOT / "data" / "nih" / "test-00000.parquet"
OUT = ROOT / "reports" / "other_pathology.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=60,
                    help="max films to score per label group")
    args = ap.parse_args()

    if not SHARD.exists():
        raise SystemExit(f"missing {SHARD} - download the NIH shard first")

    tbl = pq.ParquetFile(SHARD).read().to_pydict()
    n = len(tbl["labels"])
    print(f"shard rows: {n}")

    # NIH labels are pipe-separated multi-label strings, e.g. "Effusion|Mass"
    groups: dict[str, list[int]] = collections.defaultdict(list)
    for i, lab in enumerate(tbl["labels"]):
        parts = [p.strip() for p in str(lab).split("|") if p.strip()]
        if parts == ["No Finding"]:
            groups["No Finding"].append(i)
        elif len(parts) == 1:
            # single-pathology films only, so the attribution is unambiguous
            groups[parts[0]].append(i)

    print("\navailable single-label groups in this shard:")
    for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"  {k:<22} {len(v)}")

    scr = Screener()
    print(f"\nthreshold = {scr.threshold:.4f}  (device {scr.device})\n")

    rng = np.random.default_rng(0)
    results = {}
    for group, idxs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(idxs) < 12:
            continue
        pick = rng.permutation(idxs)[:args.per_class]
        probs = []
        for i in pick:
            blob = tbl["image"][int(i)]
            raw = blob["bytes"] if isinstance(blob, dict) else blob
            arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            probs.append(scr.score_array(img).probability)
        if not probs:
            continue
        probs = np.array(probs)
        results[group] = {
            "n": int(len(probs)),
            "mean_prob": float(probs.mean()),
            "median_prob": float(np.median(probs)),
            "flag_rate": float((probs >= scr.threshold).mean()),
        }
        print(f"  {group:<22} n={len(probs):>3}  mean={probs.mean():.3f}  "
              f"flagged={100 * (probs >= scr.threshold).mean():>5.1f}%")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"threshold": scr.threshold, "source": "NIH ChestX-ray14 (320px mirror)",
         "groups": results}, indent=2))

    # ---------------- interpretation ----------------
    nf = results.get("No Finding")
    print("\n" + "=" * 72)
    if nf:
        print(f"SPECIFICITY AT A THIRD SITE (NIH 'No Finding'): "
              f"{100 * (1 - nf['flag_rate']):.1f}% "
              f"({nf['n']} films, flag rate {100 * nf['flag_rate']:.1f}%)")
    path_only = {k: v for k, v in results.items() if k != "No Finding"}
    if path_only and nf:
        mean_path_flag = float(np.mean([v["flag_rate"] for v in path_only.values()]))
        print(f"MEAN FLAG RATE ON NON-TB PATHOLOGY: {100 * mean_path_flag:.1f}% "
              f"across {len(path_only)} conditions the model never saw")
        print(f"  vs {100 * nf['flag_rate']:.1f}% on normal films from the same hospital")
        ratio = mean_path_flag / max(nf["flag_rate"], 1e-9)
        print(f"  ratio: {ratio:.1f}x")
        print()
        if ratio > 2.0:
            print("READ: the model separates abnormal from normal at a new site,")
            print("but it fires on pathology it was never trained on. It is an")
            print("ABNORMALITY detector, not a TB-specific one.")
        else:
            print("READ: little separation between unseen pathology and normal,")
            print("so the signal is more TB-specific than feared - or the model")
            print("simply does not transfer to this site.")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

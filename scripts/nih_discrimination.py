"""Calibration failure, or discrimination failure? They are not the same.

The flag rate on NIH normals was 45%, versus ~12% at the model's own sites.
Two very different explanations produce that:

  * CALIBRATION failure - the model still RANKS films correctly, but the
    threshold imported from another site sits in the wrong place. Fixable with
    a handful of local films. This is what we already measured going from
    Shenzhen to Montgomery.

  * DISCRIMINATION failure - the model can no longer tell abnormal from normal
    at this site at all. Not fixable by recalibration; the model simply does
    not transfer.

AUROC distinguishes them, because it is threshold-free. AUROC near 0.5 means
discrimination is gone. AUROC well above 0.5 with a bad flag rate means the
ranking survived and only the threshold is wrong.

Also reports what an optimal NIH-local threshold would recover, for the same
reason.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

import cv2
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from infer import Screener  # noqa: E402

SHARD = ROOT / "data" / "nih" / "test-00000.parquet"
OUT = ROOT / "reports" / "nih_discrimination.json"
PER_GROUP = 120


def boot_auroc(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    pt = roc_auc_score(y, p)
    st = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            st.append(roc_auc_score(y[i], p[i]))
    lo, hi = np.percentile(st, [2.5, 97.5]) if st else (np.nan, np.nan)
    return float(pt), float(lo), float(hi)


def main() -> None:
    tbl = pq.ParquetFile(SHARD).read().to_pydict()
    groups = collections.defaultdict(list)
    for i, lab in enumerate(tbl["labels"]):
        parts = [x.strip() for x in str(lab).split("|") if x.strip()]
        if parts == ["No Finding"]:
            groups["No Finding"].append(i)
        elif len(parts) == 1:
            groups[parts[0]].append(i)

    scr = Screener()
    rng = np.random.default_rng(1)

    def score(idxs):
        out = []
        for i in rng.permutation(idxs)[:PER_GROUP]:
            blob = tbl["image"][int(i)]
            raw = blob["bytes"] if isinstance(blob, dict) else blob
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
            if img is not None:
                out.append(scr.score_array(img).probability)
        return np.array(out)

    print(f"scoring NIH films (threshold in use = {scr.threshold:.4f})\n")
    normal = score(groups["No Finding"])
    print(f"  No Finding      n={len(normal):>3}  mean={normal.mean():.3f}")

    path_probs, per_cond = [], {}
    for cond, idxs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if cond == "No Finding" or len(idxs) < 12:
            continue
        pr = score(idxs)
        if not len(pr):
            continue
        per_cond[cond] = pr
        path_probs.append(pr)
        print(f"  {cond:<15} n={len(pr):>3}  mean={pr.mean():.3f}")

    allpath = np.concatenate(path_probs)
    y = np.r_[np.zeros(len(normal)), np.ones(len(allpath))]
    p = np.r_[normal, allpath]

    auc, lo, hi = boot_auroc(y, p)
    print("\n" + "=" * 70)
    print("ABNORMAL (any of 10 unseen conditions) vs NORMAL, at NIH:")
    print(f"  AUROC {auc:.4f}   95% CI [{lo:.4f}, {hi:.4f}]"
          f"   (n={len(y)}: {len(normal)} normal, {len(allpath)} abnormal)")

    # what an NIH-local threshold would give
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(tpr >= 0.90)[0]
    t_local = float(thr[ok[0]]) if len(ok) else float(thr[np.argmax(tpr)])
    pred = (p >= t_local).astype(int)
    sens = float(((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1))
    spec = float(((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1))
    imported_spec = float((normal < scr.threshold).mean())

    print(f"\n  specificity with the IMPORTED threshold ({scr.threshold:.3f}): "
          f"{imported_spec:.3f}")
    print(f"  with an NIH-LOCAL threshold ({t_local:.3f}): "
          f"sens {sens:.3f}, spec {spec:.3f}")

    per_auc = {}
    print("\nper-condition AUROC vs NIH normals:")
    for cond, pr in per_cond.items():
        yy = np.r_[np.zeros(len(normal)), np.ones(len(pr))]
        pp = np.r_[normal, pr]
        a, l, h = boot_auroc(yy, pp, n=800)
        per_auc[cond] = {"auroc": a, "ci": [l, h], "n": int(len(pr))}
        print(f"  {cond:<15} AUROC {a:.3f}  [{l:.3f}, {h:.3f}]  n={len(pr)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "site": "NIH ChestX-ray14 (third site, 320px mirror)",
        "imported_threshold": scr.threshold,
        "abnormal_vs_normal": {"auroc": auc, "ci95": [lo, hi],
                               "n_normal": int(len(normal)),
                               "n_abnormal": int(len(allpath))},
        "specificity_imported_threshold": imported_spec,
        "local_threshold": {"threshold": t_local, "sensitivity": sens,
                            "specificity": spec},
        "per_condition_auroc": per_auc,
    }, indent=2))

    print("\n" + "=" * 70)
    if auc < 0.60:
        print("VERDICT: DISCRIMINATION failure. The ranking itself is broken at")
        print("this site - recalibration cannot fix it. The model does not")
        print("transfer to these films.")
    elif imported_spec < 0.70:
        print("VERDICT: CALIBRATION failure, not discrimination. The model still")
        print("ranks films meaningfully at this site; the imported threshold is")
        print("simply in the wrong place. Same conclusion as Shenzhen->Montgomery.")
    else:
        print("VERDICT: transfers acceptably at this site.")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

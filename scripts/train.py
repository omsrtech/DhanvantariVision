"""Fine-tune ResNet-18 for TB screening on chest radiographs, on GPU.

Screening, not diagnosis. That framing drives the evaluation:

* **AUROC with a bootstrap confidence interval.** With a few hundred test
  radiographs a point estimate is close to meaningless; the interval is the
  honest number and it is usually uncomfortably wide.
* **A sensitivity-first operating point.** The threshold is chosen on the
  validation set to hit a target sensitivity (default 90%), then applied
  unchanged to the test set. Missing a TB case is far worse than an
  unnecessary follow-up, so specificity is what we spend.
* **An abstention band.** Scores in the uncertain middle are routed to a human
  rather than called either way. We report how many cases that is, because a
  screening tool that quietly guesses on ambiguous films is dangerous.

Usage:
    python scripts/train.py --split split        # pooled, optimistic
    python scripts/train.py --split cross_site   # train Shenzhen, test Montgomery
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import data as D          # noqa: E402
from model import build_model  # noqa: E402

PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"


def set_seed(s: int) -> None:
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logit = model(x)
        ps.append(torch.sigmoid(logit.float()).cpu().numpy().ravel())
        ys.append(y.numpy().ravel())
    return np.concatenate(ys), np.concatenate(ps)


def bootstrap_auroc(y: np.ndarray, p: np.ndarray, n: int = 2000,
                    seed: int = 0) -> tuple[float, float, float]:
    """Point estimate and percentile 95% CI, resampling cases."""
    rng = np.random.default_rng(seed)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    point = roc_auc_score(y, p)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        stats.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5]) if stats else (np.nan, np.nan)
    return float(point), float(lo), float(hi)


def threshold_for_sensitivity(y: np.ndarray, p: np.ndarray,
                              target: float) -> float:
    """Lowest-specificity-cost threshold that still reaches `target` recall."""
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(tpr >= target)[0]
    if len(ok) == 0:
        return float(thr[np.argmax(tpr)])
    return float(thr[ok[0]])


def operating_metrics(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    d = lambda a, b: float(a / b) if b else float("nan")  # noqa: E731
    return {"threshold": float(t), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sensitivity": d(tp, tp + fn), "specificity": d(tn, tn + fp),
            "ppv": d(tp, tp + fp), "npv": d(tn, tn + fn),
            "referral_rate": d(tp + fp, len(y))}


def abstention_band(y_val, p_val, y_te, p_te, conf: float = 0.98) -> dict:
    """Route the uncertain middle to a human; report the cost and the benefit."""
    t_low = threshold_for_sensitivity(y_val, p_val, conf)   # below -> confident normal
    fpr, tpr, thr = roc_curve(y_val, p_val)
    spec = 1 - fpr
    ok = np.where(spec >= conf)[0]
    t_high = float(thr[ok[-1]]) if len(ok) else float(np.max(p_val))
    if t_high < t_low:
        t_low, t_high = t_high, t_low

    decided = (p_te < t_low) | (p_te >= t_high)
    if decided.sum() == 0:
        return {"t_low": t_low, "t_high": t_high, "abstain_rate": 1.0}
    pred = (p_te[decided] >= t_high).astype(int)
    acc = float((pred == y_te[decided]).mean())
    return {"t_low": float(t_low), "t_high": float(t_high),
            "abstain_rate": float(1 - decided.mean()),
            "accuracy_when_decided": acc,
            "n_decided": int(decided.sum()), "n_abstained": int((~decided).sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="split", choices=["split", "cross_site"])
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--target-sens", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--data-dir", default="processed",
                    help="subdirectory of data/ holding labels.csv + img512")
    args = ap.parse_args()

    global PROC
    PROC = ROOT / "data" / args.data_dir

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} "
          f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")

    tr, va, te, info = D.make_loaders(PROC / "labels.csv", PROC / "img512",
                                      split_col=args.split, batch=args.batch)
    print(f"split '{args.split}': train={info['n_train']} (pos {info['train_pos']}) "
          f"val={info['n_val']} (pos {info['val_pos']}) "
          f"test={info['n_test']} (pos {info['test_pos']})")
    print(f"  train sources={info['train_sources']} test sources={info['test_sources']}")

    model = build_model(MODELS / "resnet18-f37072fd.pth", num_classes=1).to(device)

    pos = max(info["train_pos"], 1)
    neg = max(info["n_train"] - info["train_pos"], 1)
    pos_weight = torch.tensor([neg / pos], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best = {"val_auroc": -1.0, "epoch": -1}
    ckpt = MODELS / f"tbscope_{args.split}{args.tag}.pt"
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for x, y in tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss)
            nb += 1
        sched.step()

        yv, pv = predict(model, va, device)
        vauc = roc_auc_score(yv, pv) if len(np.unique(yv)) > 1 else float("nan")
        print(f"  epoch {ep:>2}/{args.epochs}  loss={tot / max(nb, 1):.4f}  "
              f"val_auroc={vauc:.4f}  ({time.time() - t0:.0f}s)")
        if vauc == vauc and vauc > best["val_auroc"]:
            best = {"val_auroc": float(vauc), "epoch": ep}
            torch.save({"state_dict": model.state_dict(), "split": args.split,
                        "epoch": ep, "val_auroc": float(vauc)}, ckpt)

    # ---- final evaluation with the best checkpoint --------------------------
    model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
    yv, pv = predict(model, va, device)
    yt, pt = predict(model, te, device)

    auc, lo, hi = bootstrap_auroc(yt, pt, seed=args.seed)
    thr = threshold_for_sensitivity(yv, pv, args.target_sens)
    opm = operating_metrics(yt, pt, thr)
    band = abstention_band(yv, pv, yt, pt)

    out = {
        "split": args.split,
        "data": info,
        "training": {"epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                     "best_epoch": best["epoch"], "best_val_auroc": best["val_auroc"],
                     "seconds": round(time.time() - t0, 1),
                     "device": str(device),
                     "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
        "test": {
            "auroc": auc, "auroc_ci95": [lo, hi],
            "average_precision": float(average_precision_score(yt, pt))
            if len(np.unique(yt)) > 1 else None,
            "operating_point": {"target_sensitivity": args.target_sens, **opm},
            "abstention": band,
        },
        "predictions": {"y": yt.astype(int).tolist(), "p": [round(float(v), 5) for v in pt]},
    }
    mpath = MODELS / f"metrics_{args.split}{args.tag}.json"
    mpath.write_text(json.dumps(out, indent=2))

    print(f"\n=== test results ({args.split}) ===")
    print(f"  AUROC            {auc:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"  operating point  threshold={opm['threshold']:.4f} "
          f"(chosen on val for >={args.target_sens:.0%} sensitivity)")
    print(f"  sensitivity      {opm['sensitivity']:.4f}   "
          f"specificity {opm['specificity']:.4f}")
    print(f"  PPV / NPV        {opm['ppv']:.4f} / {opm['npv']:.4f}")
    print(f"  confusion        tp={opm['tp']} fp={opm['fp']} tn={opm['tn']} fn={opm['fn']}")
    if "accuracy_when_decided" in band:
        print(f"  abstention       {band['abstain_rate']:.1%} of cases routed to a "
              f"human; accuracy on the rest {band['accuracy_when_decided']:.4f}")
    print(f"  saved            {ckpt.name}, {mpath.name}")


if __name__ == "__main__":
    main()

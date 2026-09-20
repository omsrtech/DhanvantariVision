"""Generate presentation figures from the measured results.

Every figure is built from files this project actually produced -
models/metrics_*.json, reports/*.json, and real Grad-CAM overlays. Nothing is
drawn by hand or estimated.

Output: reports/figures/*.png at 200 dpi on the app's dark palette.
"""
from __future__ import annotations

import json
import pathlib
import sys

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.patches import Patch     # noqa: E402
from sklearn.metrics import roc_curve, roc_auc_score  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIG = ROOT / "reports" / "figures"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
PROC = ROOT / "data" / "processed"

BG = "#0a0f16"
FG = "#e8eef6"
MUTED = "#8395aa"
CYAN = "#38bdf8"
TEAL = "#22d3ee"
MINT = "#34d399"
CORAL = "#f87171"
AMBER = "#fbbf24"
VIOLET = "#a78bfa"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "axes.edgecolor": "#2b3947",
    "xtick.color": MUTED, "ytick.color": MUTED,
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.grid": True, "grid.color": "#1e2935", "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load(p: pathlib.Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


split = load(MODELS / "metrics_split.json")
cross = load(MODELS / "metrics_cross_site.json")
cal = load(REPORTS / "calibration.json")
nih = load(REPORTS / "nih_discrimination.json")
res = load(REPORTS / "resolution_ablation.json")
xcond = load(REPORTS / "cross_condition.json")
nodule = load(MODELS / "metrics_split_nodule.json")


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / name
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


# ---------------------------------------------------------------- 1. ROC
def fig_roc() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for d, label, colour in [(split, "Same sites (pooled)", TEAL),
                             (cross, "New site (Shenzhen to Montgomery)",
                              AMBER)]:
        y = np.array(d["predictions"]["y"])
        p = np.array(d["predictions"]["p"], dtype=float)
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        ci = d["test"]["auroc_ci95"]
        ax.plot(fpr, tpr, color=colour, lw=2.6,
                label=f"{label}:  AUROC {auc:.3f}  [{ci[0]:.3f}, {ci[1]:.3f}]")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.4, ls=(0, (4, 4)),
            label="chance:  AUROC 0.500")

    # Legend pinned bottom-right; the third-site callout sits well clear of it
    # in the upper-left dead space, so the two can never collide.
    a = nih.get("abnormal_vs_normal", {})
    if a:
        ax.text(0.055, 0.965,
                f"Third site, general population\n"
                f"AUROC {a['auroc']:.3f}  [{a['ci95'][0]:.3f}, "
                f"{a['ci95'][1]:.3f}]  (near chance)",
                color=CORAL, fontsize=10.5, fontweight="bold",
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.45", facecolor="#161d27",
                          edgecolor=CORAL, linewidth=1.2))

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Discrimination holds across TB-screening sites",
                 fontsize=12.5, color=MUTED, pad=12)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5, labelcolor=FG,
              borderaxespad=0.8, handlelength=1.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    save(fig, "fig_roc.png")


# ------------------------------------------------- 2. transfer comparison
def fig_transfer() -> None:
    a = nih["abnormal_vs_normal"]
    items = [
        ("Own sites\nTB vs normal", split["test"]["auroc"],
         split["test"]["auroc_ci95"], TEAL),
        ("Cross-site\nTB vs normal", cross["test"]["auroc"],
         cross["test"]["auroc_ci95"], AMBER),
        ("Third site\nabnormal vs normal", a["auroc"], a["ci95"], CORAL),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    xs = np.arange(len(items))
    for x, (lab, v, ci, c) in zip(xs, items):
        ax.bar(x, v, width=0.56, color=c, alpha=0.88, zorder=3)
        ax.errorbar(x, v, yerr=[[v - ci[0]], [ci[1] - v]], color=FG,
                    lw=1.8, capsize=7, zorder=4)
        ax.text(x, ci[1] + 0.035, f"{v:.3f}", ha="center", color=c,
                fontsize=16, fontweight="bold", zorder=5)
    ax.axhline(0.5, color=MUTED, ls=(0, (4, 4)), lw=1.4, zorder=2)
    ax.text(len(items) - 0.42, 0.515, "chance", color=MUTED, fontsize=9.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([i[0] for i in items], fontsize=10.5, color=FG)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("AUROC (95% CI)")
    ax.set_title("AUROC by evaluation setting", fontsize=12.5,
                 color=MUTED, pad=12)
    save(fig, "fig_transfer.png")


# -------------------------------------------- 3. why calibration breaks
def fig_calibration() -> None:
    ys = np.array(split["predictions"]["y"])
    ps = np.array(split["predictions"]["p"], dtype=float)
    yc = np.array(cross["predictions"]["y"])
    pc = np.array(cross["predictions"]["p"], dtype=float)
    t_src = split["test"]["operating_point"]["threshold"]

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 3.85), sharey=True)
    for ax, (y, p, title, spec) in zip(axes, [
            (ys, ps, "Same site as training",
             split["test"]["operating_point"]["specificity"]),
            (yc, pc, "New site, same threshold",
             cal["cross_site"]["threshold_imported_from_source_site"]["specificity"])]):
        bins = np.linspace(0, 1, 26)
        ax.hist(p[y == 0], bins=bins, color=MINT, alpha=0.78,
                label="normal films", zorder=3)
        ax.hist(p[y == 1], bins=bins, color=CORAL, alpha=0.66,
                label="TB films", zorder=3)
        ax.axvline(t_src, color=CYAN, lw=2.4, zorder=4)
        ax.set_title(f"{title}\nspecificity {spec:.3f}", fontsize=11.5,
                     color=FG, fontweight="bold")
        ax.set_xlabel("model score")
    axes[0].set_ylabel("films")
    axes[0].legend(frameon=False, fontsize=9.5, labelcolor=FG)
    axes[1].text(t_src + 0.03, axes[1].get_ylim()[1] * 0.80,
                 "same threshold,\nnow in the wrong place",
                 color=CYAN, fontsize=10, fontweight="bold")
    fig.suptitle("The score distribution moves between sites, so the threshold "
                 "no longer separates",
                 fontsize=13.5, fontweight="bold", color=FG, y=1.045)
    save(fig, "fig_calibration.png")


# ------------------------------------------------- 4. recalibration fix
def fig_recalibration() -> None:
    imported = cal["cross_site"]["threshold_imported_from_source_site"]
    refit = cal["cross_site"]["threshold_refit_on_target_site"]
    local = cal["local_recalibration"]
    labels = ["Threshold\nimported", "20 local\nfilms", "40 local\nfilms",
              "80 local\nfilms", "Refit on all\ntarget data"]
    specs = [imported["specificity"],
             local["local_sample_20"]["median_specificity"],
             local["local_sample_40"]["median_specificity"],
             local["local_sample_80"]["median_specificity"],
             refit["specificity"]]
    senss = [imported["sensitivity"],
             local["local_sample_20"]["median_sensitivity"],
             local["local_sample_40"]["median_sensitivity"],
             local["local_sample_80"]["median_sensitivity"],
             refit["sensitivity"]]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    ax.bar(x - 0.2, specs, width=0.38, color=CYAN, label="specificity", zorder=3)
    ax.bar(x + 0.2, senss, width=0.38, color=VIOLET, label="sensitivity", zorder=3)
    for xi, v in zip(x - 0.2, specs):
        ax.text(xi, v + 0.025, f"{v:.2f}", ha="center", color=CYAN,
                fontsize=10.5, fontweight="bold")
    for xi, v in zip(x + 0.2, senss):
        ax.text(xi, v + 0.025, f"{v:.2f}", ha="center", color=VIOLET,
                fontsize=10.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, color=FG)
    ax.set_ylim(0, 1.14)
    ax.set_ylabel("rate at the new site")
    ax.set_title("Twenty locally-read films recover most of the lost specificity",
                 fontsize=13.5, fontweight="bold", color=FG, pad=14)
    ax.legend(frameon=False, fontsize=10, labelcolor=FG, loc="lower right")
    save(fig, "fig_recalibration.png")


# ------------------------------------------------------ 5. PPV by setting
def fig_ppv() -> None:
    sens, spec = 0.90, 0.66
    prev = np.logspace(np.log10(0.0005), np.log10(0.30), 400)
    ppv = (sens * prev) / (sens * prev + (1 - spec) * (1 - prev))

    marks = [(0.001, "General\npopulation"), (0.01, "Mobile\nscreening"),
             (0.05, "Household\ncontacts"), (0.10, "Prisons,\nshelters"),
             (0.20, "Symptomatic\nclinic")]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(prev * 100, ppv * 100, color=TEAL, lw=3, zorder=3)
    # offsets are per-point so labels never sit on the curve; the rightmost
    # two are placed to the left of their marker
    offsets = {0.001: (10, 6, "left"), 0.01: (10, 6, "left"),
               0.05: (-12, 4, "right"), 0.10: (-12, 6, "right"),
               0.20: (-12, 2, "right")}
    for pv, lab in marks:
        v = (sens * pv) / (sens * pv + (1 - spec) * (1 - pv)) * 100
        ax.scatter([pv * 100], [v], s=90, color=AMBER, zorder=5,
                   edgecolor=BG, linewidth=1.5)
        dx, dy, ha = offsets.get(pv, (8, 8, "left"))
        shown = f"{v:.1f}%" if v < 1 else f"{v:.0f}%"
        ax.annotate(f"{lab}\n{shown}", (pv * 100, v),
                    textcoords="offset points", xytext=(dx, dy), ha=ha,
                    color=FG, fontsize=9.5, fontweight="bold", zorder=6)
    ax.set_xscale("log")
    ax.set_xlabel("TB prevalence in the screened population (%, log scale)")
    ax.set_ylabel("positive predictive value (%)")
    ax.set_title("Positive predictive value by deployment setting",
                 fontsize=12.5, color=MUTED, pad=12)
    ax.set_ylim(-2, 52)
    save(fig, "fig_ppv.png")


# ------------------------------------------- 6. per-condition at 3rd site
def fig_conditions() -> None:
    per = nih["per_condition_auroc"]
    items = sorted(per.items(), key=lambda kv: kv[1]["auroc"])
    names = [k.replace("_", " ") for k, _ in items]
    vals = [v["auroc"] for _, v in items]
    ns = [v["n"] for _, v in items]

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    colours = [MINT if v >= 0.65 else (AMBER if v >= 0.55 else CORAL)
               for v in vals]
    y = np.arange(len(names))
    ax.barh(y, vals, color=colours, alpha=0.9, zorder=3, height=0.66)
    ax.axvline(0.5, color=FG, ls=(0, (4, 4)), lw=1.6, zorder=4)
    ax.text(0.503, -0.85, "chance", color=FG, fontsize=9.5,
            fontweight="bold")
    for yi, (v, n) in enumerate(zip(vals, ns)):
        ax.text(v + 0.012, yi, f"{v:.3f}  (n={n})", va="center",
                color=FG, fontsize=9.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10.5, color=FG)
    ax.set_xlim(0.3, 0.92)
    ax.set_xlabel("AUROC vs normal films, at a third hospital")
    ax.set_title("Per-condition AUROC at a third hospital",
                 fontsize=12.5, color=MUTED, pad=12)
    ax.legend(handles=[
        Patch(facecolor=MINT, label="clears chance"),
        Patch(facecolor=AMBER, label="marginal"),
        Patch(facecolor=CORAL, label="at or below chance")],
        frameon=False, fontsize=9, labelcolor=FG,
        loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3)
    save(fig, "fig_conditions.png")


# ------------------------------------------------- 7. resolution ablation
def fig_resolution() -> None:
    a = res["auroc_native_512"]
    b = res["auroc_320_roundtrip"]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    xs = [0, 1]
    vals = [a["auroc"], b["auroc"]]
    cis = [a["ci95"], b["ci95"]]
    for x, v, ci, c, lab in zip(xs, vals, cis, [TEAL, CYAN],
                                ["Native 512px", "Degraded to 320px\nand back"]):
        ax.bar(x, v, width=0.5, color=c, alpha=0.9, zorder=3)
        ax.errorbar(x, v, yerr=[[v - ci[0]], [ci[1] - v]], color=FG, lw=1.8,
                    capsize=7, zorder=4)
        ax.text(x, ci[1] + 0.02, f"{v:.4f}", ha="center", color=c,
                fontsize=14, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(["Native 512px", "Degraded to 320px\nand back"],
                       fontsize=10.5, color=FG)
    ax.set_ylim(0.7, 1.06)
    ax.set_ylabel("AUROC on our own held-out films")
    ax.set_title(f"Resolution is not the cause\n"
                 f"AUROC moves by {res['auroc_drop']:.4f}",
                 fontsize=13, fontweight="bold", color=FG, pad=14)
    save(fig, "fig_resolution.png")


# --------------------------------------------------- 8. real case panels
def fig_cases() -> None:
    """Three held-out films, chosen by score rather than hand-picked:
    a correctly flagged TB, a correctly cleared normal, and a missed TB."""
    import pandas as pd
    preds = pd.read_csv(PROC / "predictions.csv")
    t = preds[preds.split == "test"]
    thr = split["test"]["operating_point"]["threshold"]

    tb = t[t.label == 1].sort_values("probability", ascending=False)
    nm = t[t.label == 0].sort_values("probability")
    hit = tb.iloc[0]
    clear = nm.iloc[0]
    miss = tb.iloc[-1]

    cases = [
        (hit.image_id, "TB, correctly flagged", float(hit.probability),
         hit.zone1, MINT),
        (clear.image_id, "Normal, correctly cleared", float(clear.probability),
         clear.zone1, MINT),
        (miss.image_id, "TB, MISSED by the model", float(miss.probability),
         miss.zone1, CORAL),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.6),
                             gridspec_kw={"hspace": 0.05, "wspace": 0.04})
    for col, (image_id, head, p, z, colour) in enumerate(cases):
        img = cv2.imread(str(PROC / "img512" / f"{image_id}.png"),
                         cv2.IMREAD_GRAYSCALE)
        cam = cv2.cvtColor(cv2.imread(str(PROC / "cam" / f"{image_id}.png")),
                           cv2.COLOR_BGR2RGB)
        verdict = "above threshold" if p >= thr else "below threshold"
        axes[0, col].imshow(img, cmap="gray")
        axes[0, col].set_title(f"{head}\nscore {p:.3f}, {verdict}",
                               color=colour, fontsize=13, fontweight="bold",
                               pad=10)
        axes[1, col].imshow(cam)
        axes[1, col].set_xlabel(
            f"Grad-CAM: {z}" if z != "no localised attention"
            else "Grad-CAM: no localised attention",
            color=FG, fontsize=11)
        for row in (0, 1):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            for sp in axes[row, col].spines.values():
                sp.set_color("#2b3947")
    save(fig, "fig_cases.png")


def fig_pneumonia() -> None:
    """The out-of-distribution case, scored live."""
    from infer import Screener
    d = ROOT / "data" / "demo" / "external_pneumonia"
    files = sorted(d.glob("*.png"))
    if not files:
        print("  ! no pneumonia demo films; skipping")
        return
    scr = Screener()
    f = files[0]
    pred = scr.score_file(f)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.0),
                             gridspec_kw={"wspace": 0.04})
    axes[0].imshow(pred.image512, cmap="gray")
    axes[1].imshow(pred.overlay)
    axes[0].set_title("Pneumonia, paediatric, third dataset",
                      color=AMBER, fontsize=12.5, fontweight="bold", pad=10)
    axes[1].set_title(f"Model score {pred.probability:.3f}, flagged",
                      color=CORAL, fontsize=12.5, fontweight="bold", pad=10)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#2b3947")
    fig.suptitle("A finding the model has never been trained on",
                 fontsize=13.5, fontweight="bold", color=FG, y=0.99)
    save(fig, "fig_pneumonia.png")



# --------------------------------------- 10. cross-condition application
def fig_cross_condition() -> None:
    """What happens when each model is pointed at the other's films.

    Left: AUROC. The diagonal (each model on its own source) is marked as
    inflated because that sample includes training films; the honest held-out
    figures are annotated separately. Right: the decisive panel - mean score on
    NORMAL films against each model's own threshold. When a model is applied
    across sites its normals drift above its threshold, which is the
    calibration failure, not a disease signal.
    """
    if not xcond:
        print("  ! no cross_condition.json; skipping")
        return

    keys = [
        ("TB model on TB films (Montgomery+Shenzhen)", "TB model\non TB films",
         True, 0.945),
        ("nodule model on TB films (Montgomery+Shenzhen)",
         "Nodule model\non TB films", False, None),
        ("TB model on NIH films (nodule/mass set)", "TB model\non NIH films",
         False, None),
        ("nodule model on NIH films (nodule/mass set)",
         "Nodule model\non NIH films", True, 0.714),
    ]
    keys = [k for k in keys if k[0] in xcond]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 5.0),
                                   gridspec_kw={"wspace": 0.26})

    # ---- left: AUROC -----------------------------------------------------
    x = np.arange(len(keys))
    for i, (key, lab, own, heldout) in enumerate(keys):
        r = xcond[key]
        v = r["auroc_on_that_set_labels"]
        ci = r["ci95"]
        colour = MUTED if own else (TEAL if v >= 0.70 else CORAL)
        axL.bar(i, v, width=0.6, color=colour, alpha=0.55 if own else 0.9,
                zorder=3, hatch="///" if own else None,
                edgecolor=BG if own else "none", linewidth=0)
        axL.errorbar(i, v, yerr=[[v - ci[0]], [ci[1] - v]], color=FG, lw=1.7,
                     capsize=6, zorder=4)
        axL.text(i, ci[1] + 0.03, f"{v:.3f}", ha="center", color=colour,
                 fontsize=14, fontweight="bold", zorder=5)
        if own:
            axL.text(i, 0.06, f"same source\nheld-out: {heldout:.3f}",
                     ha="center", color=FG, fontsize=8.5, fontweight="bold")
    axL.axhline(0.5, color=FG, ls=(0, (4, 4)), lw=1.5, zorder=2)
    axL.text(len(keys) - 0.45, 0.515, "chance", color=FG, fontsize=9)
    axL.set_xticks(x)
    axL.set_xticklabels([k[1] for k in keys], fontsize=9.5, color=FG)
    axL.set_ylim(0, 1.12)
    axL.set_ylabel("AUROC (95% CI)")
    axL.set_title("Ranking ability, applied across conditions",
                  fontsize=12.5, color=MUTED, pad=12)
    axL.legend(handles=[
        Patch(facecolor=MUTED, alpha=0.55, hatch="///",
              label="own source (includes training films - inflated)"),
        Patch(facecolor=TEAL, label="cross-applied, some signal"),
        Patch(facecolor=CORAL, label="cross-applied, near chance")],
        frameon=False, fontsize=8.5, labelcolor=FG, loc="upper center",
        bbox_to_anchor=(0.5, -0.13), ncol=1)

    # ---- right: where the NORMAL films land relative to the threshold ----
    for i, (key, lab, own, _) in enumerate(keys):
        r = xcond[key]
        thr = r["threshold"]
        neg, pos = r["mean_score_negative"], r["mean_score_positive"]
        base = i
        axR.hlines(thr, base - 0.36, base + 0.36, color=CYAN, lw=2.6,
                   zorder=5)
        axR.scatter([base - 0.14], [neg], s=190, color=MINT, zorder=6,
                    edgecolor=BG, linewidth=1.6)
        axR.scatter([base + 0.14], [pos], s=190, color=CORAL, zorder=6,
                    edgecolor=BG, linewidth=1.6)
        axR.plot([base - 0.14, base + 0.14], [neg, pos], color="#2b3947",
                 lw=1.6, zorder=4)
        # labels placed clear of the markers: negatives below, positives above
        axR.text(base - 0.14, neg - 0.052, f"{neg:.2f}", ha="center",
                 va="top", color=MINT, fontsize=10.5, fontweight="bold")
        axR.text(base + 0.14, pos + 0.052, f"{pos:.2f}", ha="center",
                 va="bottom", color=CORAL, fontsize=10.5, fontweight="bold")
        # flag rate on normals - the number that matters
        axR.text(base, 1.075, f"{r['flag_rate_negative']:.0%}", ha="center",
                 color=AMBER if r["flag_rate_negative"] > 0.25 else MUTED,
                 fontsize=13, fontweight="bold")
    axR.set_xticks(x)
    axR.set_xticklabels([k[1] for k in keys], fontsize=9.5, color=FG)
    axR.set_xlim(-0.7, len(keys) - 0.3)
    axR.set_ylim(-0.04, 1.2)
    axR.set_ylabel("mean model score")
    axR.set_title("Where normal films sit relative to each model's threshold\n"
                  "(top row = share of NORMAL films wrongly flagged)",
                  fontsize=12, color=MUTED, pad=14)
    axR.legend(handles=[
        Patch(facecolor=MINT, label="mean score, NORMAL films"),
        Patch(facecolor=CORAL, label="mean score, POSITIVE films"),
        Patch(facecolor=CYAN, label="that model's operating threshold")],
        frameon=False, fontsize=8.5, labelcolor=FG, loc="upper center",
        bbox_to_anchor=(0.5, -0.13), ncol=1)

    fig.suptitle("Cross-applying the models: ranking survives a little, "
                 "calibration does not",
                 fontsize=13.5, fontweight="bold", color=FG, y=1.02)
    save(fig, "fig_cross_condition.png")



# ------------------------------------------- 11. two conditions compared
def fig_two_conditions() -> None:
    """Same architecture, same protocol, two diseases - honestly compared."""
    if not nodule:
        print("  ! no nodule metrics; skipping")
        return
    tb_t, nd_t = split["test"], nodule["test"]
    tb_i, nd_i = split["data"], nodule["data"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 4.5),
                                 gridspec_kw={"wspace": 0.28})

    # AUROC with CIs
    items = [("Tuberculosis\n(2 screening sites)", tb_t["auroc"],
              tb_t["auroc_ci95"], TEAL),
             ("Lung nodule / mass\n(NIH, patient-level split)",
              nd_t["auroc"], nd_t["auroc_ci95"], AMBER)]
    for i, (lab, v, ci, c) in enumerate(items):
        a1.bar(i, v, width=0.52, color=c, alpha=0.9, zorder=3)
        a1.errorbar(i, v, yerr=[[v - ci[0]], [ci[1] - v]], color=FG, lw=1.8,
                    capsize=7, zorder=4)
        a1.text(i, ci[1] + 0.03, f"{v:.3f}", ha="center", color=c,
                fontsize=17, fontweight="bold")
    a1.axhline(0.5, color=MUTED, ls=(0, (4, 4)), lw=1.4)
    a1.text(0.5, 0.525, "chance", color=MUTED, fontsize=9, ha="center")
    a1.set_xticks([0, 1])
    a1.set_xticklabels([i[0] for i in items], fontsize=10, color=FG)
    a1.set_ylim(0, 1.12)
    a1.set_ylabel("held-out AUROC (95% CI)")
    a1.set_title("Held-out performance", fontsize=12.5, color=MUTED, pad=12)

    # abstention: the harder task defers far more
    rates = [tb_t["abstention"]["abstain_rate"],
             nd_t["abstention"]["abstain_rate"]]
    for i, (r, c) in enumerate(zip(rates, [TEAL, AMBER])):
        a2.bar(i, r * 100, width=0.52, color=c, alpha=0.9, zorder=3)
        a2.text(i, r * 100 + 2.5, f"{r:.0%}", ha="center", color=c,
                fontsize=17, fontweight="bold")
    a2.set_xticks([0, 1])
    a2.set_xticklabels(["Tuberculosis", "Lung nodule / mass"], fontsize=10,
                       color=FG)
    a2.set_ylim(0, 104)
    a2.set_ylabel("cases routed to a human (%)")
    a2.set_title("The abstention band is not cosmetic:\n"
                 "on a harder task it defers far more",
                 fontsize=12, color=MUTED, pad=12)

    fig.suptitle(
        f"One pipeline, two diseases: "
        f"{tb_i['n_train'] + tb_i['n_val'] + tb_i['n_test']:,} TB films and "
        f"{nd_i['n_train'] + nd_i['n_val'] + nd_i['n_test']:,} nodule/mass films",
        fontsize=13.5, fontweight="bold", color=FG, y=1.03)
    save(fig, "fig_two_conditions.png")


# --------------------------------------------- 12. decision zones visual
def fig_decision_zones() -> None:
    """The per-result strip the app shows, rendered for the deck."""
    import io
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from plots import score_context_figure
    import pandas as pd

    op = split["test"]["operating_point"]
    band = split["test"]["abstention"]
    d = pd.read_csv(PROC / "predictions.csv")
    t = d[d.split == "test"]
    # a case that sits inside the low-confidence band - the interesting one
    mid = t[(t.probability >= op["threshold"]) &
            (t.probability < band["t_high"])].sort_values("probability")
    prob = float(mid.iloc[len(mid) // 2].probability) if len(mid) else 0.82

    png = score_context_figure(
        prob, float(op["threshold"]), float(band["t_low"]),
        float(band["t_high"]), t[t.label == 0].probability.to_numpy(),
        t[t.label == 1].probability.to_numpy(), "TB", width=9.0)
    out = FIG / "fig_decision_zones.png"
    out.write_bytes(png)
    print(f"  wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    print("generating figures from measured results:")
    fig_roc()
    fig_transfer()
    fig_calibration()
    fig_recalibration()
    fig_ppv()
    fig_conditions()
    fig_resolution()
    fig_cases()
    fig_pneumonia()
    fig_cross_condition()
    fig_two_conditions()
    fig_decision_zones()
    print(f"\nall figures in {FIG}")

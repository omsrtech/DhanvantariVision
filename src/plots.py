"""Per-result figure: where one score sits in the decision space.

Shown under every analysed radiograph. It answers, in one picture, the question
the numbers alone kept raising: why did THIS score get THIS label?

Top strip  the four decision zones with their real boundaries, labelled inline,
           and this film's score marked on it.
Histogram  held-out scores for films with known labels, so you can see how
           unusual this score is, not just which side of a line it fell on.

Zone colours match the UI chips exactly - if they drifted, the picture would
contradict the label printed beside it.
"""
from __future__ import annotations

import io

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

BG = "#0a1018"
FG = "#e8eef6"
MUTED = "#8395aa"
CYAN = "#38bdf8"
MINT = "#34d399"
CORAL = "#f87171"
AMBER = "#fbbf24"

Z_CONF_NEG = "#34d399"
Z_LOW = "#fbbf24"
Z_CONF_POS = "#f87171"


def decision_zones(thr: float, t_low: float, t_high: float):
    """The four zones, as (start, end, colour, short label, alpha)."""
    lo = min(t_low, thr)
    hi = max(t_high, thr)
    return [
        (0.0, lo, Z_CONF_NEG, "confident\nnegative", 0.34),
        (lo, thr, Z_LOW, "low-confidence\nnegative", 0.20),
        (thr, hi, Z_LOW, "low-confidence\npositive", 0.34),
        (hi, 1.0, Z_CONF_POS, "confident\npositive", 0.34),
    ]


def score_context_figure(prob: float, thr: float, t_low: float, t_high: float,
                         neg_scores=None, pos_scores=None,
                         positive_label: str = "positive",
                         width: float = 8.4) -> bytes:
    """PNG bytes for the per-result decision strip."""
    fig, (ax, axh) = plt.subplots(
        2, 1, figsize=(width, 3.5), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.25], "hspace": 0.42})
    fig.patch.set_facecolor(BG)
    for a in (ax, axh):
        a.set_facecolor(BG)

    # ---- zone strip -----------------------------------------------------
    for x0, x1, colour, label, alpha in decision_zones(thr, t_low, t_high):
        if x1 - x0 <= 1e-6:
            continue
        ax.axvspan(x0, x1, color=colour, alpha=alpha, lw=0)
        # inline label only if the zone is wide enough to hold it
        if x1 - x0 > 0.13:
            ax.text((x0 + x1) / 2, 0.40, label, ha="center", va="center",
                    color=FG, fontsize=8.2, fontweight="bold", linespacing=1.25)

    for v, col, lw in [(t_low, MUTED, 1.3), (thr, CYAN, 2.2),
                       (t_high, MUTED, 1.3)]:
        ax.axvline(v, color=col, lw=lw, ls="-" if col == CYAN else (0, (3, 3)),
                   zorder=4)
        ax.text(v, 1.06, f"{v:.3f}", ha="center", va="bottom", color=col,
                fontsize=8.5, fontweight="bold")
    ax.text(thr, 1.30, "operating threshold", ha="center", va="bottom",
            color=CYAN, fontsize=8.5, fontweight="bold")

    # this film's score, labelled inside the strip so nothing is clipped
    ax.scatter([prob], [0.80], s=330, marker="v", color=FG, zorder=6,
               edgecolor=BG, linewidth=1.5)
    ha = "left" if prob < 0.72 else "right"
    ax.text(prob + (0.02 if ha == "left" else -0.02), 0.80,
            f"this film  {prob:.3f}", ha=ha, va="center", color=FG,
            fontsize=10.5, fontweight="bold", zorder=7)

    ax.set_ylim(0, 1.05)
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(axis="x", length=0, labelbottom=False)

    # ---- distribution of known-label scores ------------------------------
    bins = np.linspace(0, 1, 41)
    drew = False
    if neg_scores is not None and len(neg_scores):
        axh.hist(neg_scores, bins=bins, color=MINT, alpha=0.78,
                 label=f"held-out normal (n={len(neg_scores)})", zorder=3)
        drew = True
    if pos_scores is not None and len(pos_scores):
        axh.hist(pos_scores, bins=bins, color=CORAL, alpha=0.62,
                 label=f"held-out {positive_label} (n={len(pos_scores)})",
                 zorder=3)
        drew = True
    axh.axvline(thr, color=CYAN, lw=1.6, zorder=5)
    axh.axvline(prob, color=FG, lw=2.2, zorder=6)
    axh.set_xlim(0, 1)
    axh.set_xlabel("model score", color=FG, fontsize=10)
    axh.set_ylabel("films", color=MUTED, fontsize=9)
    axh.tick_params(colors=MUTED, labelsize=9)
    axh.grid(color="#1e2935", lw=0.8)
    axh.set_axisbelow(True)
    for name, sp in axh.spines.items():
        sp.set_color("#2b3947")
        if name in ("top", "right"):
            sp.set_visible(False)
    if drew:
        axh.legend(frameon=False, fontsize=8.5, labelcolor=FG,
                   loc="upper center", ncol=2)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight",
                pad_inches=0.18, facecolor=BG)
    plt.close(fig)
    return buf.getvalue()

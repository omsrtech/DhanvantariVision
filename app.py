"""Dhanvantari Vision - tuberculosis chest X-ray screening triage.

Two-stage pipeline:
  Stage 1  ResNet-18 fine-tuned on GPU -> calibrated TB probability + Grad-CAM
  Stage 2  Groq LLM -> referral note grounded strictly in Stage 1's numbers

The use case is worklist triage where radiologists are scarce: reorder the
queue so likely-positive films are read first. It is a screening aid, not a
diagnostic device.

Run:  streamlit run app.py
"""
from __future__ import annotations

import html
import json
import pathlib
import sys

import pandas as pd
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import report as RP           # noqa: E402
from infer import Screener    # noqa: E402
from plots import score_context_figure  # noqa: E402

PROC = ROOT / "data" / "processed"
CONDITIONS = {
    "Tuberculosis": {
        "tag": "", "data_dir": "processed", "positive": "TB-consistent",
        "finding": "tuberculosis",
        "confirm": "microbiological testing (sputum smear, culture or "
                   "molecular assay)",
        "dataset": "NLM Montgomery County + Shenzhen",
        "band": "upper",
        "reference_pattern":
            "post-primary pulmonary tuberculosis characteristically favours "
            "the UPPER zones, so an upper-zone finding is concordant and a "
            "mid/lower or diffuse finding is discordant",
        "blurb": "Trained on 800 radiographs from two dedicated TB-screening "
                 "sites. Cross-site validated.",
        "demo_groups": {
            "held_out_tb": ("Held-out TB", "real TB, never seen in training"),
            "held_out_normal": ("Held-out normal",
                                "real normal, never seen in training"),
            "external_pneumonia": ("Pneumonia (different dataset)",
                                   "pathology this model was never trained on"),
        },
        "truths": {
            "held_out_tb": "TB (from the dataset)",
            "held_out_normal": "normal (from the dataset)",
            "external_pneumonia":
                "PNEUMONIA - a finding this model has never been trained on",
        },
    },
    "Lung nodule / mass": {
        "tag": "_nodule", "data_dir": "processed_nodule",
        "positive": "nodule- or mass-consistent",
        "finding": "a pulmonary nodule or mass",
        "confirm": "CT of the chest, which is the imaging standard for "
                   "characterising a nodule",
        "dataset": "NIH ChestX-ray14 (Mass + Nodule vs No Finding)",
        "band": "",
        "reference_pattern":
            "a pulmonary nodule or mass has NO characteristic anatomical "
            "distribution, so the zone carries no concordance information",
        "blurb": "Trained on 2,214 films from 1,843 patients, split BY "
                 "PATIENT so no patient appears in two splits. A much harder "
                 "task than TB.",
        "demo_groups": {
            "nodule_positive": ("Held-out nodule / mass",
                                "correctly flagged, never seen in training"),
            "nodule_normal": ("Held-out normal",
                              "correctly cleared, never seen in training"),
            "nodule_missed": ("Missed nodule / mass",
                              "real finding the model scored low"),
        },
        "truths": {
            "nodule_positive": "nodule or mass (from the dataset)",
            "nodule_normal": "No Finding (from the dataset)",
            "nodule_missed":
                "nodule or mass - MISSED by the model, a false negative",
        },
    },
}

RP_OOD = ("binary TB screener: trained only on TB-consistent versus normal radiographs, so it has no category for other pathology and cannot identify a cause")
MODELS = ROOT / "models"

st.set_page_config(page_title="Dhanvantari Vision", page_icon="🫁", layout="wide",
                   initial_sidebar_state="expanded")

CSS = """
<style>
  @keyframes tsRise { from {opacity:0; transform:translateY(8px)} }
  @keyframes tsFill { from {width:0} }
  @keyframes tsScan { 0% {top:-10%} 100% {top:110%} }
  #MainMenu, footer, header[data-testid="stHeader"] {visibility:hidden;}

  .stApp { background:
      radial-gradient(1100px 600px at 10% -10%, #101f33 0%, rgba(5,8,12,0) 60%),
      radial-gradient(900px 520px at 90% -5%, #08262b 0%, rgba(5,8,12,0) 58%),
      #05080c; }
  .block-container { padding-top:1.1rem; max-width:1560px; }

  .ts-hero { position:relative; overflow:hidden; border-radius:16px;
      border:1px solid rgba(56,189,248,.22); padding:1.1rem 1.4rem;
      margin-bottom:1rem;
      background:linear-gradient(115deg, rgba(7,16,26,.95), rgba(8,30,36,.75));
      box-shadow:0 0 40px rgba(56,189,248,.08); }
  .ts-hero::after { content:""; position:absolute; left:0; right:0; height:18%;
      background:linear-gradient(180deg, transparent, rgba(56,189,248,.07),
      transparent); animation: tsScan 5.5s linear infinite; pointer-events:none; }
  .ts-title { font-size:2.5rem; font-weight:900; letter-spacing:-.04em; margin:0;
      background:linear-gradient(96deg,#38bdf8 0%,#22d3ee 45%,#a78bfa 100%);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
  .ts-kick { color:#22d3ee; font-size:.66rem; font-weight:800;
      letter-spacing:.3em; text-transform:uppercase; margin:0 0 .25rem; }
  .ts-tag { color:#8fa3ba; font-size:.95rem; margin:.35rem 0 0; }

  .ts-card { position:relative; background:rgba(10,16,24,.84);
      border:1px solid rgba(140,180,220,.12); border-radius:13px;
      padding:.95rem 1.1rem; margin-bottom:.8rem; animation: tsRise .4s ease both;
      box-shadow:0 10px 30px rgba(0,0,0,.4); }
  .ts-card::before, .ts-card::after { content:""; position:absolute;
      width:12px; height:12px; border-color:rgba(56,189,248,.5); }
  .ts-card::before { top:-1px; left:-1px; border-top:2px solid;
      border-left:2px solid; border-top-left-radius:13px; }
  .ts-card::after { bottom:-1px; right:-1px; border-bottom:2px solid;
      border-right:2px solid; border-bottom-right-radius:13px; }
  .ts-card h4 { margin:0 0 .2rem; font-size:.7rem; letter-spacing:.15em;
      text-transform:uppercase; color:#7cc7dd; font-weight:800; }
  .ts-card .sub { color:#8395aa; font-size:.82rem; margin:.1rem 0 .7rem;
      line-height:1.45; }

  .ts-bar { height:9px; border-radius:5px; background:rgba(255,255,255,.06);
      overflow:hidden; }
  .ts-bar > span { display:block; height:100%; border-radius:5px;
      animation: tsFill .7s cubic-bezier(.2,.9,.25,1) both; }
  .ts-big { font-family:"SFMono-Regular",Consolas,monospace; font-weight:800;
      font-size:2.3rem; line-height:1; }
  .ts-mono { font-family:"SFMono-Regular",Consolas,monospace; font-size:.84rem; }
  .ts-meta { color:#7d90a6; font-size:.76rem; line-height:1.5; }
  .ts-hdr { color:#6d8299; font-size:.66rem; letter-spacing:.12em;
      text-transform:uppercase; font-weight:700; white-space:nowrap; }

  .ts-chip { display:inline-block; font-size:.63rem; font-weight:800;
      letter-spacing:.07em; text-transform:uppercase; padding:.2rem .5rem;
      border-radius:999px; white-space:nowrap; }
  .ts-chip.pos { background:rgba(248,113,113,.14); color:#f87171;
      border:1px solid rgba(248,113,113,.45); }
  .ts-chip.neg { background:rgba(52,211,153,.12); color:#34d399;
      border:1px solid rgba(52,211,153,.4); }
  .ts-chip.unc { background:rgba(251,191,36,.13); color:#fbbf24;
      border:1px solid rgba(251,191,36,.45); }
  .ts-chip.info { background:rgba(56,189,248,.12); color:#38bdf8;
      border:1px solid rgba(56,189,248,.4); }
  .ts-chip.dim { background:rgba(255,255,255,.05); color:#8395aa;
      border:1px solid rgba(255,255,255,.12); }

  .ts-stat { display:flex; justify-content:space-between; gap:1rem;
      padding:.3rem 0; border-bottom:1px dashed rgba(255,255,255,.07);
      font-size:.83rem; }
  .ts-stat span:first-child { color:#8395aa; }
  .ts-stat span:last-child { color:#e8eef6; font-weight:700;
      font-family:"SFMono-Regular",Consolas,monospace; }

  .ts-note { background:linear-gradient(180deg,rgba(167,139,250,.10),
      rgba(167,139,250,.03)); border:1px solid rgba(167,139,250,.3);
      border-left:3px solid #a78bfa; }
  .ts-note p.body { margin:.45rem 0 0; color:#e8eef6; font-size:.95rem;
      line-height:1.62; }
  .ts-prov { color:#b4a4f5; font-size:.66rem; letter-spacing:.13em;
      text-transform:uppercase; font-weight:800; }
  .ts-byline { margin:.6rem 0 0; padding-top:.5rem;
      border-top:1px solid rgba(167,139,250,.22);
      color:#9db2c8; font-size:.74rem; letter-spacing:.02em; }
  .ts-byline b { color:#c9b6ff; font-weight:800; letter-spacing:.06em; }
  .ts-byline a { color:#38bdf8; text-decoration:none; }
  .ts-credits h3 { margin:0 0 .2rem; font-size:1.65rem; font-weight:900;
      letter-spacing:-.02em;
      background:linear-gradient(96deg,#38bdf8 0%,#22d3ee 45%,#a78bfa 100%);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
  .ts-team { display:flex; flex-wrap:wrap; gap:.45rem; margin:.7rem 0 0; }
  .ts-team span { background:rgba(56,189,248,.10);
      border:1px solid rgba(56,189,248,.35); color:#e8eef6;
      border-radius:999px; padding:.3rem .8rem; font-size:.92rem;
      font-weight:700; }

  .ts-warn { background:rgba(251,191,36,.07);
      border:1px solid rgba(251,191,36,.3); border-radius:9px;
      padding:.7rem .9rem; color:#ffefc9; font-size:.86rem; line-height:1.55; }
  .ts-danger { background:rgba(248,113,113,.07);
      border:1px solid rgba(248,113,113,.32); border-radius:9px;
      padding:.7rem .9rem; color:#ffe0e0; font-size:.86rem; line-height:1.55; }

  div.stButton > button { background:rgba(255,255,255,.035);
      border:1px solid rgba(140,180,220,.16); color:#c6d3e2; font-size:.78rem;
      padding:.2rem .55rem; border-radius:8px; }
  div.stButton > button:hover { border-color:#38bdf8; color:#38bdf8;
      background:rgba(56,189,248,.08); }
  .stTabs [data-baseweb="tab"] { font-size:.88rem; font-weight:700;
      color:#8395aa; }
  .stTabs [aria-selected="true"] { color:#38bdf8 !important; }

  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation:none !important; }
  }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# Sign-in gate and queue removed: this build runs locally and open, so no
# identity check or admission queue is applied. src/gate_ui.py,
# src/gatekeeper.py and src/policy.py are left in the tree but unused.


# ---------------------------------------------------------------- loading
@st.cache_data
def load_predictions(data_dir: str):
    p = ROOT / "data" / data_dir / "predictions.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


@st.cache_data
def load_transfer():
    out = {}
    for key, name in [("nih", "nih_discrimination.json"),
                      ("res", "resolution_ablation.json"),
                      ("path", "other_pathology.json")]:
        p = ROOT / "reports" / name
        if p.exists():
            out[key] = json.loads(p.read_text())
    return out


@st.cache_data
def load_calibration():
    p = ROOT / "reports" / "calibration.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data
def load_metrics(split: str, tag: str):
    p = MODELS / f"metrics_{split}{tag}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


@st.cache_data(show_spinner=False)
def score_context_png(prob: float, thr: float, t_low: float, t_high: float,
                      data_dir: str, positive_label: str):
    """Decision strip for one result. Cached on the values it draws."""
    neg = pos = None
    p = ROOT / "data" / data_dir / "predictions.csv"
    if p.exists():
        df = pd.read_csv(p)
        t = df[df.split == "test"]
        if len(t):
            neg = t[t.label == 0].probability.to_numpy()
            pos = t[t.label == 1].probability.to_numpy()
    return score_context_figure(prob, thr, t_low, t_high, neg, pos,
                                positive_label)


def card(title: str, sub: str = "", cls: str = "") -> str:
    s = f'<p class="sub">{sub}</p>' if sub else ""
    return f'<div class="ts-card {cls}"><h4>{title}</h4>{s}'


def decision_state(p: float, thr: float, t_low: float, t_high: float) -> str:
    """One source of truth for direction + confidence."""
    positive = p >= thr
    confident = (p < t_low) or (p >= t_high)
    if positive and confident:
        return "confident_positive"
    if positive:
        return "low_positive"
    if confident:
        return "confident_negative"
    return "low_negative"


STATE_COLOUR = {
    "confident_positive": "#f87171",   # red
    "low_positive": "#fbbf24",         # amber
    "low_negative": "#fbbf24",         # amber
    "confident_negative": "#34d399",   # green
}


def prob_color(p: float, thr: float, t_low: float | None = None,
               t_high: float | None = None) -> str:
    """Colour of the score, derived from the SAME rule as the chip.

    Previously this used hardcoded cutoffs (0.75, then the threshold), so the
    number could be red while the chip said "review", or green while the chip
    said "human read required". Deriving both from decision_state keeps them
    in agreement for any model and any threshold.
    """
    if t_low is None or t_high is None:
        return "#f87171" if p >= thr else "#34d399"
    return STATE_COLOUR[decision_state(p, thr, t_low, t_high)]


def bar(p: float, thr: float, t_low=None, t_high=None) -> str:
    c = prob_color(p, thr, t_low, t_high)
    return (f'<div class="ts-bar"><span style="width:{p * 100:.0f}%;'
            f'background:linear-gradient(90deg,{c}55,{c});'
            f'box-shadow:0 0 10px {c}88"></span></div>')


def chip_for(decision: str) -> str:
    """Four states, not two - a low-confidence negative still needs a reader."""
    low = "low confidence" in decision
    positive = decision.startswith("screen positive")
    if positive and not low:
        return '<span class="ts-chip pos">refer</span>'
    if positive:
        return '<span class="ts-chip unc">review &middot; likely pos</span>'
    if low:
        return '<span class="ts-chip unc">review &middot; likely neg</span>'
    return '<span class="ts-chip neg">clear</span>'


SPLIT = "split"
_available = [k for k, v in CONDITIONS.items()
              if (MODELS / f"metrics_split{v['tag']}.json").exists()]
if not _available:
    st.error("No trained model found. Run scripts/train.py first.")
    st.stop()

with st.sidebar:
    st.markdown('<p class="ts-hdr" style="margin-bottom:.2rem">'
                'Condition</p>', unsafe_allow_html=True)
    CONDITION = st.radio("Condition", _available, label_visibility="collapsed")
    st.caption(CONDITIONS[CONDITION]["blurb"])
    st.markdown("---")

COND = CONDITIONS[CONDITION]
TAG = COND["tag"]
PROC = ROOT / "data" / COND["data_dir"]
preds = load_predictions(COND["data_dir"])
metrics = load_metrics(SPLIT, TAG)

st.markdown(
    '<div class="ts-hero"><p class="ts-kick">Chest radiograph screening triage</p>'
    '<div class="ts-title">Dhanvantari Vision</div>'
    f'<p class="ts-tag">A trained model reorders the reading queue so the '
    f'films most likely to show {COND["finding"]} are read first, then '
    f'explains itself using only its own numbers. Screening aid, not a '
    f'diagnostic device.</p></div>',
    unsafe_allow_html=True)

if preds is None or not metrics:
    st.error("No predictions found. Run `scripts/train.py` then "
             "`scripts/predict_all.py` first.")
    st.stop()

test = metrics.get("test", {})
op = test.get("operating_point", {})
band = test.get("abstention", {})
info = metrics.get("data", {})
train_meta = metrics.get("training", {})
THR = float(op.get("threshold", 0.5))
T_LOW = float(band.get("t_low", THR))
T_HIGH = float(band.get("t_high", THR))

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(card("Cohort filter",
                     "The worklist below is sorted by model probability, "
                     "highest first — the triage order."),
                unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    only_test = st.toggle(
        "Held-out cases only", value=True,
        help="ON is the honest view: the model never saw these during "
             "training. OFF includes training images, where scores look "
             "far better than real performance.")
    dec_filter = st.multiselect(
        "Show decisions",
        ["refer", "needs human read", "clear"],
        default=["refer", "needs human read", "clear"])
    use_llm = st.toggle("AI referral notes (Groq)", value=True)
    _key_ok, _key_msg = RP.key_status()
    st.caption(("AI notes ON - " if _key_ok else "AI notes OFF - ") + _key_msg)
    if st.button("About / credits", use_container_width=True):
        _credits_dialog()

    st.markdown(card("Model"), unsafe_allow_html=True)
    st.markdown("".join(
        f'<div class="ts-stat"><span>{k}</span><span>{v}</span></div>'
        for k, v in [
            ("Backbone", "ResNet-18 (ImageNet)"),
            ("Trained on", f"{info.get('n_train', '?')} radiographs"),
            ("Device", (train_meta.get("gpu") or "cpu").replace("NVIDIA ", "")),
            ("Train time", f"{train_meta.get('seconds', '?')} s"),
            ("Test AUROC", f"{test.get('auroc', 0):.3f}"),
        ]), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

df = preds.copy()
if only_test:
    df = df[df.split == "test"]


def dec_key(d: str) -> str:
    """Bucket for filtering and the queue metrics.

    Must check confidence BEFORE direction: a low-confidence positive is not a
    referral, it is a case a human has to read. Checking `startswith` first
    counted every low-confidence call as a decided one.
    """
    if "low confidence" in d:
        return "needs human read"
    if d.startswith("screen positive"):
        return "refer"
    return "clear"


df["dec_key"] = df.decision.map(dec_key)
df = df[df.dec_key.isin(dec_filter)].sort_values("probability", ascending=False)

if "sel" not in st.session_state or st.session_state.sel not in set(df.image_id):
    st.session_state.sel = df.image_id.iloc[0] if len(df) else None

# ---------------------------------------------------------------- credits
TEAM = ["Om", "Aadidev", "Subramanian", "Parthiv"]


@st.dialog("Dhanvantari Vision")
def _credits_dialog() -> None:
    st.markdown(
        '<div class="ts-credits">'
        '<h3>Made by Team Akatsuki</h3>'
        '<p style="color:#8fa3ba;font-size:.9rem;margin:.15rem 0 0">'
        'Chest radiograph screening triage &middot; a research prototype</p>'
        '<div class="ts-team">'
        + "".join(f"<span>{n}</span>" for n in TEAM) +
        '</div>'
        '<p style="color:#9db2c8;font-size:.82rem;margin:1rem 0 0;'
        'padding-top:.7rem;border-top:1px solid rgba(255,255,255,.10)">'
        'Engineered by <b style="color:#c9b6ff">OMSR TECHNOLOGIES</b><br>'
        '<a href="https://omsrtech.com" target="_blank" '
        'style="color:#38bdf8;text-decoration:none">omsrtech.com</a></p>'
        '<p style="color:#f0a0a0;font-size:.76rem;margin:.9rem 0 0">'
        'Research prototype on public NLM/NIH datasets. Not a medical device, '
        'no clinical validation. Not for decisions about any real patient.</p>'
        '</div>', unsafe_allow_html=True)
    if st.button("Enter", use_container_width=True):
        st.session_state.credits_seen = True
        st.rerun()


if not st.session_state.get("credits_seen"):
    st.session_state.credits_seen = True
    _credits_dialog()


tab_work, tab_scan, tab_model, tab_about = st.tabs(
    ["🩻  Triage Worklist", "🔬  Analyse a Scan",
     "📊  Model Report Card", "ℹ️  What This Is"])

# ================================================================ worklist
with tab_work:
    if not len(df):
        st.info("No cases match the current filters.")
    else:
        k = st.columns(4)
        k[0].metric("Cases in queue", len(df))
        k[1].metric("Auto-refer (confident)",
                    int((df.dec_key == "refer").sum()))
        k[2].metric("Need a human read",
                    int((df.dec_key == "needs human read").sum()))
        k[3].metric("Operating threshold", f"{THR:.3f}")

        left, right = st.columns([1.05, 1], gap="large")

        with left:
            st.markdown(card(
                "Reading queue — highest risk first",
                "In a real clinic this is the whole product: the same films, "
                "read in a better order. Click ▸ to open a case."),
                unsafe_allow_html=True)

            COLS = [1.5, 1.6, 0.95, 1.5, 0.85, 0.4]
            head = st.columns(COLS)
            for c, t in zip(head, ["Case", "Probability", "Decision",
                                   "Model attended to", "Truth", ""]):
                c.markdown(f'<span class="ts-hdr">{t}</span>',
                           unsafe_allow_html=True)

            for r in df.head(25).itertuples():
                c = st.columns(COLS)
                is_sel = r.image_id == st.session_state.sel
                c[0].markdown(
                    f'<span class="ts-mono" style="color:'
                    f'{"#38bdf8" if is_sel else "#c6d3e2"}">'
                    f'{"◆ " if is_sel else ""}{r.image_id}</span>',
                    unsafe_allow_html=True)
                c[1].markdown(
                    f'<div style="display:flex;align-items:center;gap:.4rem">'
                    f'<span class="ts-mono" style="color:'
                    f'{prob_color(r.probability, THR, T_LOW, T_HIGH)};font-weight:800">'
                    f'{r.probability:.3f}</span>'
                    f'<div style="flex:1">{bar(r.probability, THR, T_LOW, T_HIGH)}</div></div>',
                    unsafe_allow_html=True)
                c[2].markdown(chip_for(r.decision), unsafe_allow_html=True)
                zone_txt = (f'{r.zone1} ({r.zone1_frac:.0%})'
                            if r.zone1 != "no localised attention"
                            else '<i>no localised attention</i>')
                c[3].markdown(f'<span class="ts-meta">{zone_txt}</span>',
                              unsafe_allow_html=True)
                c[4].markdown(
                    '<span class="ts-chip pos">TB</span>' if r.label == 1
                    else '<span class="ts-chip neg">normal</span>',
                    unsafe_allow_html=True)
                with c[5]:
                    if st.button("▸", key=f"sel_{r.image_id}",
                                 use_container_width=True):
                        st.session_state.sel = r.image_id
                        st.rerun()
            st.markdown(
                '<p class="ts-meta">The <b>Truth</b> column exists because '
                'this is a demo on a labelled research set. A deployed tool '
                'would not have it — that is the entire point of the '
                'probability column.</p></div>', unsafe_allow_html=True)

        # ---------------------------------------------------------- detail
        with right:
            sel = df[df.image_id == st.session_state.sel]
            if not len(sel):
                st.info("Select a case.")
            else:
                r = sel.iloc[0]
                st.markdown(card(f"Case {r.image_id}"), unsafe_allow_html=True)
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:1rem">'
                    f'<div class="ts-big" style="color:'
                    f'{prob_color(r.probability, THR, T_LOW, T_HIGH)}">{r.probability:.2f}</div>'
                    f'<div style="flex:1">{chip_for(r.decision)}'
                    f'{bar(r.probability, THR, T_LOW, T_HIGH)}'
                    f'<p class="ts-meta" style="margin:.3rem 0 0">'
                    f'TB-consistent abnormality probability · threshold '
                    f'{THR:.3f}</p></div></div></div>',
                    unsafe_allow_html=True)

                st.image(
                    score_context_png(round(float(r.probability), 5),
                                      round(float(THR), 5),
                                      round(T_LOW, 5), round(T_HIGH, 5),
                                      COND["data_dir"], COND["positive"]),
                    use_container_width=True)

                ic1, ic2 = st.columns(2)
                img_p = PROC / "img512" / f"{r.image_id}.png"
                cam_p = PROC / "cam" / f"{r.image_id}.png"
                if img_p.exists():
                    ic1.image(str(img_p), caption="radiograph (CLAHE normalised)",
                              use_container_width=True)
                if cam_p.exists():
                    ic2.image(str(cam_p), caption="Grad-CAM — where the model looked",
                              use_container_width=True)

                # ------- Stage 2
                ev = {
                    "probability": float(r.probability),
                    "threshold": THR,
                    "t_low": float(band.get("t_low", THR)),
                    "t_high": float(band.get("t_high", THR)),
                    "target_sensitivity": op.get("target_sensitivity", 0.9),
                    "decision": r.decision,
                    "zones": [(r.zone1, float(r.zone1_frac)),
                              (r.zone2, float(r.zone2_frac)),
                              (r.zone3, float(r.zone3_frac))],
                    "auroc": test.get("auroc"),
                    "auroc_ci": test.get("auroc_ci95"),
                    "sensitivity": op.get("sensitivity"),
                    "specificity": op.get("specificity"),
                    "target_finding": COND["positive"] + " abnormality",
                    "confirm": COND["confirm"],
                    "reference_pattern": COND["reference_pattern"],
                    "reference_pattern_band": COND.get("band", ""),
                    "target_finding": COND["positive"] + " abnormality",
            "confirm": COND["confirm"],
            "reference_pattern": COND["reference_pattern"],
            "reference_pattern_band": COND.get("band", ""),
            "train_description":
                        f"{info.get('n_train', '?')} radiographs, "
                        f"{', '.join(info.get('train_sources', [])) or 'unknown'} set",
                    "caveat": ("single-site training set of only "
                               f"{info.get('n_train', '?')} radiographs; "
                               "cross-site performance is not yet verified"),
                }

                @st.cache_data(show_spinner="Generating grounded referral note…")
                def get_note(image_id: str, ev_json: str, llm: bool):
                    return RP.generate_report(json.loads(ev_json), use_llm=llm).__dict__

                note = get_note(r.image_id, json.dumps(ev), use_llm)
                st.markdown(
                    f'<div class="ts-card ts-note"><h4>Clinical note &mdash; what the model found</h4>'
                    f'<span class="ts-prov">'
                    f'{"AI-generated · grounded" if note["mode"] == "llm" else "Template · grounded"}'
                    f'</span><p class="body">{html.escape(note["text"])}</p>'
                    f'<p class="ts-meta" style="margin-top:.55rem">'
                    f'{html.escape(note["label"])}</p>'
                    f'<p class="ts-byline">Report generated by <b>OMSR TECHNOLOGIES</b> &middot; <a href="https://omsrtech.com" target="_blank">omsrtech.com</a></p></div>',
                    unsafe_allow_html=True)
                with st.expander("Show the exact evidence the model was given"):
                    st.code(RP.build_evidence(ev), language="text")

                st.markdown(card(
                    "Attention by anatomical zone",
                    "Radiological convention: the patient's right lung appears "
                    "on the left of the image."), unsafe_allow_html=True)
                if r.zone1 == "no localised attention":
                    st.markdown(
                        '<div class="ts-warn">The saliency map is empty for '
                        'this case. Grad-CAM highlights evidence <i>for</i> '
                        'tuberculosis, so a confidently negative prediction '
                        'legitimately produces nothing to show — this is not a '
                        'positive finding of normality, and it is why the '
                        'heatmap panel above is flat.</div>',
                        unsafe_allow_html=True)
                else:
                    for z, f in [(r.zone1, r.zone1_frac), (r.zone2, r.zone2_frac),
                                 (r.zone3, r.zone3_frac)]:
                        st.markdown(
                            f'<div style="display:flex;align-items:center;'
                            f'gap:.6rem;margin:.25rem 0"><span class="ts-meta" '
                            f'style="width:150px;color:#c6d3e2">{z}</span>'
                            f'<div style="flex:1">{bar(min(f * 2.5, 1.0), 0)}</div>'
                            f'<span class="ts-mono" style="width:44px;'
                            f'text-align:right">{f:.0%}</span></div>',
                            unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)




# ================================================================ analyse
@st.cache_resource
def get_screener(tag: str):
    return Screener(split="split", tag=tag)


with tab_scan:
    st.markdown(card(
        "Run the model on a radiograph",
        "Upload a chest X-ray, or pick one of the bundled films the model was "
        "never trained on. Inference and Grad-CAM run live on the GPU."),
        unsafe_allow_html=True)
    st.markdown(
        f'<div class="ts-danger" style="margin-top:0"><b>Read this before '
        f'interpreting any result below.</b> This model answers exactly one '
        f'question: does this film resemble the {COND["positive"]} radiographs '
        f'it was trained on, or the normal ones? It has <b>no category</b> for '
        f'any other finding, so it <b>cannot tell you a cause</b> - any '
        f'unfamiliar abnormality is forced into one of its two buckets. For '
        f'the TB model we measured this directly: six pneumonia films from a '
        f'different dataset scored a mean of 0.821 and five of six were '
        f'flagged positive. Treat a high score as "a human should look at this '
        f'film", never as a diagnosis of {COND["finding"]}.</div>',
        unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    DEMO = ROOT / "data" / "demo"
    GROUPS = COND.get("demo_groups", {})
    TRUTHS = COND.get("truths", {})

    st.session_state.setdefault("scan_path", None)
    st.session_state.setdefault("scan_bytes", None)

    up = st.file_uploader("Upload a chest radiograph (PNG or JPEG)",
                          type=["png", "jpg", "jpeg"])
    if up is not None:
        st.session_state.scan_bytes = up.getvalue()
        st.session_state.scan_path = "upload:" + up.name

    st.markdown('<p class="ts-meta">...or pick a bundled film:</p>',
                unsafe_allow_html=True)
    for key, (title, sub) in GROUPS.items():
        d = DEMO / key
        files = sorted(d.glob("*.png")) if d.exists() else []
        if not files:
            continue
        st.markdown('<p class="ts-hdr" style="margin-top:.4rem">'
                    + title + " - " + sub + "</p>", unsafe_allow_html=True)
        cols = st.columns(min(len(files), 6))
        for col, f in zip(cols, files[:6]):
            with col:
                if st.button(f.stem[:14], key="demo_" + f.stem,
                             use_container_width=True):
                    st.session_state.scan_path = str(f)
                    st.session_state.scan_bytes = None
                    st.rerun()

    target = st.session_state.scan_path
    pred = None
    known = None
    origin = ""
    if target:
        try:
            scr = get_screener(TAG)
            if st.session_state.scan_bytes is not None:
                pred = scr.score_bytes(st.session_state.scan_bytes)
                origin = target.split("upload:", 1)[1]
            else:
                pred = scr.score_file(target)
                origin = pathlib.Path(target).name
                group = pathlib.Path(target).parent.name
                known = TRUTHS.get(group)
        except Exception as e:  # noqa: BLE001
            st.error("Could not score that image: "
                     + type(e).__name__ + ": " + str(e))

    if pred is not None:
        st.markdown(card("Result - " + origin), unsafe_allow_html=True)
        st.markdown(
            '<div style="display:flex;align-items:center;gap:1rem">'
            '<div class="ts-big" style="color:'
            + prob_color(pred.probability, pred.threshold, T_LOW, T_HIGH) + '">'
            + f"{pred.probability:.2f}" + '</div><div style="flex:1">'
            + chip_for(pred.decision)
            + bar(pred.probability, pred.threshold, T_LOW, T_HIGH)
            + '<p class="ts-meta" style="margin:.3rem 0 0">resemblance to the '
            + COND["positive"] + ' films this model was trained on '
              '&middot; threshold '
            + f"{pred.threshold:.3f}" + '</p></div></div>',
            unsafe_allow_html=True)
        if known:
            st.markdown('<p class="ts-meta" style="margin-top:.5rem">'
                        '<b>Known truth for this film:</b> ' + known + '</p>',
                        unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.image(
            score_context_png(round(float(pred.probability), 5),
                              round(float(pred.threshold), 5),
                              round(T_LOW, 5), round(T_HIGH, 5),
                              COND["data_dir"], COND["positive"]),
            use_container_width=True)

        tr = pred.trust or {}
        if tr.get("available") and not tr.get("trustworthy", True):
            st.markdown(
                '<div class="ts-danger"><b>Do not read this score as a '
                'result.</b> This film does not resemble the data the model '
                'was trained on, so its output is not meaningful here:<br>'
                + "".join(f"&bull; {r}<br>" for r in tr["reasons"])
                + 'A chest radiograph from an unknown source, a different '
                  'scanner, or a web image that has already been processed is '
                  'outside this model\'s validated range. We measured this '
                  'failure directly: on a third hospital\'s films the TB '
                  'model separated abnormal from normal at AUROC 0.579, near '
                  'chance.</div>', unsafe_allow_html=True)
        elif tr.get("available"):
            st.markdown(
                f'<p class="ts-meta">Trust check passed: '
                f'{tr["off_lung_saliency"]:.0%} of attention outside the lung '
                f'fields (limit {tr["off_lung_threshold"]:.0%}), image profile '
                f'distance {tr["histogram_distance"]:.3f} (limit '
                f'{tr["histogram_threshold"]:.3f}).</p>',
                unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        if pred.image512 is not None:
            c1.image(pred.image512, caption="preprocessed (CLAHE, 512px)",
                     use_container_width=True, clamp=True)
        if pred.overlay is not None:
            c2.image(pred.overlay,
                     caption="Grad-CAM - where the model looked",
                     use_container_width=True)

        ev = {
            "probability": pred.probability, "threshold": pred.threshold,
            "t_low": float(band.get("t_low", pred.threshold)),
            "t_high": float(band.get("t_high", pred.threshold)),
            "target_sensitivity": op.get("target_sensitivity", 0.9),
            "decision": pred.decision, "zones": pred.zones,
            "auroc": test.get("auroc"), "auroc_ci": test.get("auroc_ci95"),
            "sensitivity": op.get("sensitivity"),
            "specificity": op.get("specificity"),
            "target_finding": COND["positive"] + " abnormality",
            "confirm": COND["confirm"],
            "reference_pattern": COND["reference_pattern"],
            "reference_pattern_band": COND.get("band", ""),
            "train_description":
                str(info.get("n_train", "?")) + " radiographs from the NLM "
                "Montgomery and Shenzhen sets",
            "trust_warning": "; ".join((pred.trust or {}).get("reasons", [])),
            "caveat": RP_OOD + f" Current target: {COND['positive']} "
                       f"findings; confirmation is {COND['confirm']}.",
        }
        note = RP.generate_report(ev, use_llm=use_llm)
        st.markdown(
            '<div class="ts-card ts-note"><h4>Clinical note &mdash; what the model found</h4>'
            '<span class="ts-prov">'
            + ("AI-generated &middot; grounded" if note.mode == "llm"
               else "Template &middot; grounded")
            + '</span><p class="body">' + html.escape(note.text) + '</p>'
            '<p class="ts-meta" style="margin-top:.55rem">'
            + html.escape(note.label) + '</p>'
            + '<p class="ts-byline">Report generated by <b>OMSR TECHNOLOGIES</b> &middot; <a href="https://omsrtech.com" target="_blank">omsrtech.com</a></p>' + '</div>',
            unsafe_allow_html=True)
        with st.expander("Show the exact evidence the model was given"):
            st.code(RP.build_evidence(ev), language="text")

        zones = [z for z in pred.zones if z[0] != "no localised attention"]
        st.markdown(card("Attention by zone"), unsafe_allow_html=True)
        if not zones:
            st.markdown(
                '<div class="ts-warn">Saliency map is empty - Grad-CAM '
                'highlights evidence <i>for</i> TB, so a confident negative '
                'has nothing to show.</div>', unsafe_allow_html=True)
        else:
            for z, fr in zones[:3]:
                st.markdown(
                    '<div style="display:flex;align-items:center;gap:.6rem;'
                    'margin:.25rem 0"><span class="ts-meta" '
                    'style="width:150px;color:#c6d3e2">' + z + '</span>'
                    '<div style="flex:1">' + bar(min(fr * 2.5, 1.0), 0)
                    + '</div><span class="ts-mono" style="width:44px;'
                      'text-align:right">' + f"{fr:.0%}" + '</span></div>',
                    unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)




# ================================================================ metrics
with tab_model:
    m1, m2 = st.columns(2, gap="large")
    with m1:
        st.markdown(card(
            "Stage 1 — trained model",
            "ResNet-18 written explicitly and initialised from official "
            "ImageNet weights (loaded strict, no torchvision dependency), "
            "fine-tuned on GPU."), unsafe_allow_html=True)
        st.markdown("".join(
            f'<div class="ts-stat"><span>{k}</span><span>{v}</span></div>'
            for k, v in [
                ("Dataset", "NLM Montgomery + Shenzhen CXR sets"),
                ("Total radiographs", info.get("n_train", 0) + info.get("n_val", 0)
                 + info.get("n_test", 0)),
                ("Train / val / test", f"{info.get('n_train')} / "
                                       f"{info.get('n_val')} / {info.get('n_test')}"),
                ("TB positives in test", info.get("test_pos")),
                ("GPU", (train_meta.get("gpu") or "cpu")),
                ("Training time", f"{train_meta.get('seconds')} s "
                                  f"({train_meta.get('epochs')} epochs)"),
                ("Best epoch", train_meta.get("best_epoch")),
            ]), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(card("Held-out test performance"), unsafe_allow_html=True)
        ci = test.get("auroc_ci95") or [float("nan")] * 2
        st.markdown("".join(
            f'<div class="ts-stat"><span>{k}</span><span>{v}</span></div>'
            for k, v in [
                ("AUROC", f"{test.get('auroc', 0):.3f}"),
                ("AUROC 95% CI", f"[{ci[0]:.3f}, {ci[1]:.3f}]"),
                ("Sensitivity", f"{op.get('sensitivity', 0):.3f}"),
                ("Specificity", f"{op.get('specificity', 0):.3f}"),
                ("PPV / NPV", f"{op.get('ppv', 0):.3f} / {op.get('npv', 0):.3f}"),
                ("Confusion (tp/fp/tn/fn)",
                 f"{op.get('tp')}/{op.get('fp')}/{op.get('tn')}/{op.get('fn')}"),
                ("Abstention rate", f"{band.get('abstain_rate', 0):.1%}"),
                ("Accuracy when decided",
                 f"{band.get('accuracy_when_decided', 0):.3f}"),
            ]), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with m2:
        cal = load_calibration()
        if cal:
            pl, cs = cal["pooled"], cal["cross_site"]
            imported = cs["threshold_imported_from_source_site"]
            refit = cs["threshold_refit_on_target_site"]
            local20 = cal["local_recalibration"]["local_sample_20"]
            st.markdown(card(
                "Cross-site test — the number that actually matters",
                "Trained on Shenzhen (China), tested on Montgomery County "
                "(USA). Different scanner, different population, zero overlap."),
                unsafe_allow_html=True)
            st.markdown("".join(
                f'<div class="ts-stat"><span>{k}</span><span>{v}</span></div>'
                for k, v in [
                    ("AUROC — same site", f"{pl['auroc']:.3f}"),
                    ("AUROC — new site", f"{cs['auroc']:.3f}"),
                    ("Specificity — same site",
                     f"{pl['at_operating_point']['specificity']:.3f}"),
                    ("Specificity — new site, imported threshold",
                     f"{imported['specificity']:.3f}"),
                    ("False positives at the new site",
                     f"{imported['fp']} of {imported['fp'] + imported['tn']} normals"),
                ]), unsafe_allow_html=True)
            st.markdown(
                f'<div class="ts-danger" style="margin-top:.6rem">'
                f'<b>Ranking transfers. Calibration does not.</b> AUROC barely '
                f'moved ({pl["auroc"]:.3f} to {cs["auroc"]:.3f}) because it is '
                f'threshold-free — the model still <i>orders</i> radiographs '
                f'correctly at a new site. But the decision threshold tuned on '
                f'the source site produced specificity '
                f'{imported["specificity"]:.3f} at the new one: '
                f'{imported["fp"]} of {imported["fp"] + imported["tn"]} healthy '
                f'people would have been sent for unnecessary testing. The mean '
                f'score on normal films nearly doubled '
                f'({cal["score_shift"]["pooled_mean_score_normal"]:.3f} to '
                f'{cal["score_shift"]["crosssite_mean_score_normal"]:.3f}) — the '
                f'score distribution moved, so the threshold no longer sits '
                f'where it was calibrated to sit.</div>',
                unsafe_allow_html=True)
            st.markdown(
                f'<div class="ts-warn" style="margin-top:.6rem">'
                f'<b>The fix, and it is cheap.</b> Refitting the threshold on '
                f'target-site data restores specificity to '
                f'{refit["specificity"]:.3f} at {refit["sensitivity"]:.3f} '
                f'sensitivity. Just <b>20 locally-read radiographs</b> recover '
                f'most of it (median specificity '
                f'{local20["median_specificity"]:.3f} at sensitivity '
                f'{local20["median_sensitivity"]:.3f}, over '
                f'{local20["n_trials"]} resamples). So the deployment rule is: '
                f'ship the model, but calibrate the threshold on a small local '
                f'sample before trusting any decision it makes.</div></div>',
                unsafe_allow_html=True)

        tr = load_transfer()
        if tr.get("nih"):
            nih = tr["nih"]
            av = nih["abnormal_vs_normal"]
            res = tr.get("res", {})
            st.markdown(card(
                "Limits of transfer — where it stops working",
                "Tested on NIH ChestX-ray14: a third, independent hospital with "
                "14 labelled pathologies and a general (not TB-screening) "
                "population."), unsafe_allow_html=True)
            st.markdown("".join(
                f'<div class="ts-stat"><span>{k}</span><span>{v}</span></div>'
                for k, v in [
                    ("AUROC — own sites (TB vs normal)", "0.945"),
                    ("AUROC — cross-site (TB vs normal)", "0.899"),
                    ("AUROC — third site (abnormal vs normal)",
                     f"{av['auroc']:.3f}"),
                    ("95% CI at third site",
                     f"[{av['ci95'][0]:.3f}, {av['ci95'][1]:.3f}]"),
                    ("Normal films wrongly flagged there",
                     f"{100 * (1 - nih['specificity_imported_threshold']):.0f}%"),
                ]), unsafe_allow_html=True)
            st.markdown(
                f'<div class="ts-danger" style="margin-top:.6rem">'
                f'<b>This is a discrimination failure, not a calibration one.</b> '
                f'At NIH the model scores AUROC {av["auroc"]:.3f} separating any '
                f'abnormality from normal — near chance, and several individual '
                f'conditions fall below 0.5 (Nodule 0.404, Atelectasis 0.449). '
                f'No threshold fixes that: the ranking itself does not hold in a '
                f'general hospital population. The operational rule is that this '
                f'tool is only valid in a TB-screening population resembling its '
                f'training data.</div>', unsafe_allow_html=True)
            if res:
                st.markdown(
                    f'<div class="ts-warn" style="margin-top:.6rem">'
                    f'<b>Confound excluded.</b> The NIH mirror stores 320px '
                    f'images while training used 512px, so we degraded our own '
                    f'held-out films through the same pipeline '
                    f'(512&rarr;320&rarr;512) and re-scored: AUROC moved by only '
                    f'{res.get("auroc_drop", 0):.4f} '
                    f'({res["auroc_native_512"]["auroc"]:.3f} to '
                    f'{res["auroc_320_roundtrip"]["auroc"]:.3f}). Resolution is '
                    f'not the cause — the failure is genuine site and population '
                    f'shift.</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="ts-warn" style="margin-top:.6rem">'
                '<b>It is more TB-specific than it first appeared.</b> Because it '
                'does <i>not</i> detect general abnormality, an earlier read of '
                'ours — that it is simply an "abnormal chest" detector — was '
                'wrong. The only conditions it flags well above chance are '
                'Pneumothorax (0.726) and Fibrosis (0.713), and fibrotic '
                'scarring is a recognised TB sequela, so the signal it learned '
                'looks TB-adjacent rather than generic.</div></div>',
                unsafe_allow_html=True)

        st.markdown(card("Where this model is weak — stated plainly"),
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="ts-danger"><b>The confidence interval is the real '
            f'result.</b> AUROC {test.get("auroc", 0):.3f} sounds strong, but '
            f'the 95% interval runs [{ci[0]:.3f}, {ci[1]:.3f}] because the '
            f'test set is only {info.get("n_test", 0)} radiographs with '
            f'{info.get("test_pos", 0)} TB positives. Any single number from a '
            f'test set this small should be treated as a smoke test that the '
            f'pipeline works, not as evidence of clinical performance.</div>',
            unsafe_allow_html=True)
        st.markdown(
            '<div class="ts-warn" style="margin-top:.6rem">'
            '<b>Operating point is site-specific.</b> The pooled numbers above '
            'mix two sites, so they flatter the model. The cross-site panel is '
            'the honest read, and its conclusion is that no fixed threshold '
            'should be shipped to a new clinic without local recalibration.'
            '</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="ts-warn" style="margin-top:.6rem">'
            '<b>Shortcut learning is the classic failure.</b> Chest X-ray '
            'models have been shown to "detect" disease by spotting treatment '
            'devices or a hospital\'s burned-in text markers. The Grad-CAM '
            'panel on every case exists so this can be checked rather than '
            'assumed — if attention sits on image corners or text rather than '
            'lung parenchyma, the score is not trustworthy.</div>',
            unsafe_allow_html=True)
        st.markdown(
            '<div class="ts-warn" style="margin-top:.6rem">'
            '<b>Labels are image-level.</b> The dataset marks a radiograph as '
            'TB-consistent or normal. It carries no lesion boundaries, so '
            'Grad-CAM cannot be validated against ground-truth '
            'localisation — it indicates model attention only.</div></div>',
            unsafe_allow_html=True)

        st.markdown(card(
            "Stage 2 — grounded referral note",
            "Retrieval, then generation. The language model never sees the "
            "radiograph."), unsafe_allow_html=True)
        st.markdown(
            '<p class="ts-meta" style="font-size:.86rem">A language model '
            'cannot read a chest X-ray, so asking it to interpret one would be '
            'fabrication. Instead it receives a fixed evidence block — the '
            'probability, the operating point in use, the Grad-CAM zone '
            'distribution, and the model\'s own held-out performance — and is '
            'instructed to use only those numbers, to describe attention as '
            'model behaviour rather than a finding, to never state that the '
            'patient does or does not have tuberculosis, and to name '
            'microbiological testing as the diagnostic standard. Every case '
            'shows the exact block it was given.</p>'
            f'<div class="ts-stat"><span>Provider</span><span>Groq</span></div>'
            f'<div class="ts-stat"><span>Model</span>'
            f'<span>{RP.MODEL}</span></div>'
            f'<div class="ts-stat"><span>Key present</span>'
            f'<span>{RP.has_key()}</span></div>'
            '<p class="ts-meta" style="margin-top:.5rem">With no key, a '
            'template fills the same structure from the same evidence and the '
            'UI labels it as such.</p></div>', unsafe_allow_html=True)

# ================================================================ about
with tab_about:
    a1, a2 = st.columns([1.1, 1], gap="large")
    with a1:
        st.markdown(card(
            "The problem this addresses",
            "Not accuracy. Access."), unsafe_allow_html=True)
        st.markdown(
            '<p class="ts-meta" style="font-size:.9rem">Tuberculosis remains '
            'one of the world\'s deadliest infectious diseases, and chest '
            'radiography is the frontline screening tool. The bottleneck is '
            'not film — it is readers. A clinic can acquire far more '
            'radiographs in a day than a scarce radiologist can report, so '
            'films are read in the order they arrive and a patient with '
            'advanced disease can sit at position 63 in the queue for hours.'
            '</p>'
            '<p class="ts-meta" style="font-size:.9rem">This tool does not try '
            'to replace the reader. It reorders the queue so the films most '
            'likely to show tuberculosis are read first, and routes genuinely '
            'ambiguous cases to a human instead of guessing. The World Health '
            'Organization endorses computer-aided detection as a TB screening '
            'aid, which is the category this sits in.</p></div>',
            unsafe_allow_html=True)

        st.markdown(card("Why the threshold is set where it is"),
                    unsafe_allow_html=True)
        st.markdown(
            f'<p class="ts-meta" style="font-size:.9rem">The decision '
            f'threshold ({THR:.3f}) was chosen on the <b>validation</b> set to '
            f'reach {op.get("target_sensitivity", 0.9):.0%} sensitivity, then '
            f'applied unchanged to the test set. That is deliberate: for '
            f'screening, a missed case is far more costly than an unnecessary '
            f'follow-up test, so specificity is what gets spent. Tuning the '
            f'threshold on the test set would make the numbers look better and '
            f'mean nothing.</p></div>', unsafe_allow_html=True)

    with a2:
        st.markdown(card("What is real and what is not"),
                    unsafe_allow_html=True)
        st.markdown("".join(
            f'<div class="ts-stat"><span>{k}</span><span>{v}</span></div>'
            for k, v in [
                ("Radiographs", "real (NLM Montgomery)"),
                ("Labels", "real, from the dataset"),
                ("Model", "really trained, on this GPU"),
                ("Metrics", "real, held-out"),
                ("Grad-CAM", "real model gradients"),
                ("Referral note", "LLM, constrained to the evidence"),
                ("Cross-site validation", "done — 2 sites + 1 external"),
                ("Transfer limits", "measured — see report card"),
                ("Clinical validation", "NONE"),
                ("Regulatory clearance", "NONE"),
            ]), unsafe_allow_html=True)
        st.markdown(
            '<div class="ts-danger" style="margin-top:.7rem">This is a '
            'hackathon prototype built on a public research dataset. It is '
            'not a medical device, has no clinical validation, and must not be '
            'used to make decisions about any real patient.</div></div>',
            unsafe_allow_html=True)

st.markdown(
    '<p class="ts-meta" style="text-align:center;margin-top:1rem">'
    'TBScope · Stage 1 ResNet-18 fine-tuned locally on GPU · Stage 2 referral '
    'note grounded in that model\'s output · research prototype, '
    '<b>not for clinical use</b>.</p>', unsafe_allow_html=True)

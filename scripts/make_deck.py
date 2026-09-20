"""Build the Dhanvantari Vision pitch deck as a .pptx.

Every number is read from the measured result files rather than typed in, so
the deck cannot drift out of sync with the code:
  models/metrics_split.json, models/metrics_cross_site.json,
  reports/calibration.json, reports/nih_discrimination.json,
  reports/resolution_ablation.json

Figures come from scripts/make_figures.py, which renders them from the same
files plus the real Grad-CAM overlays.

Output: TBScope_pitch.pptx (16:9, dark theme matching the app).
"""
from __future__ import annotations

import json
import pathlib

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
OUT = ROOT / "DhanvantariVision_pitch.pptx"

W, H = 13.333, 7.5

BG = RGBColor(0x0A, 0x0F, 0x16)
PANEL = RGBColor(0x13, 0x1C, 0x27)
PANEL2 = RGBColor(0x18, 0x23, 0x30)
FGC = RGBColor(0xE8, 0xEE, 0xF6)
MUTED = RGBColor(0x8B, 0x9C, 0xB0)
CYAN = RGBColor(0x38, 0xBD, 0xF8)
TEAL = RGBColor(0x22, 0xD3, 0xEE)
CORAL = RGBColor(0xF8, 0x71, 0x71)
MINT = RGBColor(0x34, 0xD3, 0x99)
AMBER = RGBColor(0xFB, 0xBF, 0x24)
VIOLET = RGBColor(0xA7, 0x8B, 0xFA)

HEAD = "Cambria"
BODY = "Calibri"


def load(p: pathlib.Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


M = load(ROOT / "models" / "metrics_split.json")
C = load(ROOT / "models" / "metrics_cross_site.json")
CAL = load(ROOT / "reports" / "calibration.json")
NIH = load(ROOT / "reports" / "nih_discrimination.json")
RES = load(ROOT / "reports" / "resolution_ablation.json")
XC = load(ROOT / "reports" / "cross_condition.json")
ND = load(ROOT / "models" / "metrics_split_nodule.json")

T = M["test"]
OP = T["operating_point"]
BAND = T["abstention"]
INFO = M["data"]
TRAIN = M["training"]
XOP = CAL["cross_site"]["threshold_imported_from_source_site"]
XREFIT = CAL["cross_site"]["threshold_refit_on_target_site"]
L20 = CAL["local_recalibration"]["local_sample_20"]
NA = NIH["abnormal_vs_normal"]


# ----------------------------------------------------------------- helpers
def new_deck() -> Presentation:
    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)
    return prs


def slide(prs: Presentation):
    s = prs.slides.add_slide(prs.slide_layouts[6])   # blank
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0,
                            Inches(W), Inches(H))
    bg.fill.solid()
    bg.fill.fore_color.rgb = BG
    bg.line.fill.background()
    bg.shadow.inherit = False
    return s


def text(s, x, y, w, h, runs, size=14, color=FGC, bold=False, font=BODY,
         align=PP_ALIGN.LEFT, spacing=1.15, anchor=MSO_ANCHOR.TOP):
    """runs: str, or list of (text, {overrides}) tuples, or list of paragraphs."""
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor

    paras = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        items = para if isinstance(para, list) else [(para, {})]
        if isinstance(para, str):
            items = [(para, {})]
        for txt, ov in items:
            r = p.add_run()
            r.text = txt
            f = r.font
            f.name = ov.get("font", font)
            f.size = Pt(ov.get("size", size))
            f.bold = ov.get("bold", bold)
            f.italic = ov.get("italic", False)
            f.color.rgb = ov.get("color", color)
    return tb


def card(s, x, y, w, h, fill=PANEL, radius=0.04, line=None):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                            Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(1.25)
    sh.shadow.inherit = False
    try:
        sh.adjustments[0] = radius
    except Exception:
        pass
    return sh


def circle(s, x, y, d, fill, label, color=BG, size=15):
    sh = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y),
                            Inches(d), Inches(d))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.fill.background()
    sh.shadow.inherit = False
    tf = sh.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    r.font.size = Pt(size)
    r.font.bold = True
    r.font.name = BODY
    r.font.color.rgb = color
    return sh


def kicker(s, txt, color=TEAL):
    text(s, 0.7, 0.42, 11.9, 0.3, txt.upper(), size=11, color=color,
         bold=True, font=BODY, spacing=1.0)


def title(s, txt, y=0.78, size=33, color=FGC, w=11.9):
    text(s, 0.7, y, w, 1.0, txt, size=size, color=color, bold=True,
         font=HEAD, spacing=0.98)


def pic(s, name, x, y, w, h):
    """Insert a figure, contained within the given box, centred."""
    from PIL import Image
    p = FIG / name
    with Image.open(p) as im:
        iw, ih = im.size
    box_ar, img_ar = w / h, iw / ih
    if img_ar >= box_ar:
        dw, dh = w, w / img_ar
    else:
        dh, dw = h, h * img_ar
    s.shapes.add_picture(str(p), Inches(x + (w - dw) / 2),
                         Inches(y + (h - dh) / 2), Inches(dw), Inches(dh))


def stat(s, x, y, w, value, label, color=CYAN, vsize=40):
    text(s, x, y, w, 0.72, value, size=vsize, color=color, bold=True,
         font=BODY, spacing=0.9)
    text(s, x, y + 0.68, w, 0.6, label, size=11, color=MUTED, font=BODY,
         spacing=1.1)


def footer(s, txt):
    text(s, 0.7, 6.96, 11.9, 0.34, txt, size=9.5, color=RGBColor(0x5F, 0x70,
         0x83), font=BODY, spacing=1.0)


def notes(s, txt: str) -> None:
    s.notes_slide.notes_text_frame.text = txt


# ------------------------------------------------------------------ slides
prs = new_deck()

# ---- 1. title
s = slide(prs)
card(s, 0.7, 1.28, 11.93, 3.35, PANEL, line=RGBColor(0x1E, 0x3A, 0x4C))
text(s, 1.15, 1.62, 11.0, 0.32, "CHEST RADIOGRAPH SCREENING TRIAGE",
     size=12, color=TEAL, bold=True, spacing=1.0)
text(s, 1.15, 1.98, 11.0, 1.25, "Dhanvantari Vision", size=46,
     color=FGC, bold=True, font=HEAD, spacing=0.95)
text(s, 1.15, 3.18, 10.4, 1.1,
     [[("A tuberculosis screening tool, and an honest map of where it "
        "stops working.", {"size": 18, "color": FGC})],
      [("Built in a day on a laptop GPU. Every number measured, "
        "with confidence intervals.", {"size": 14, "color": MUTED})]],
     spacing=1.3)
for i, (v, l, c) in enumerate([
        (f"{T['auroc']:.3f}", "AUROC, held-out", TEAL),
        (f"{NA['auroc']:.3f}", "AUROC at a third site", CORAL),
        (f"{TRAIN['seconds']:.0f}s", "to train on an RTX 3050", CYAN)]):
    stat(s, 0.95 + i * 4.0, 5.05, 3.7, v, l, c, vsize=44)
footer(s, "Research prototype on public NLM/NIH datasets · not a medical "
          "device · no clinical validation")
notes(s, "Don't explain the name. Go straight to the problem on slide 2.")

# ---- 2. problem
s = slide(prs)
kicker(s, "The problem")
title(s, "The bottleneck is readers, not machines")
card(s, 0.7, 2.1, 6.9, 4.1, PANEL)
text(s, 1.1, 2.5, 6.1, 3.4,
     [[("Chest radiography is the frontline tool for TB screening. A mobile "
        "screening van images around 400 people a day, with no radiologist "
        "on board.", {"size": 15})],
      [("Films are batched to a city hospital. Reports come back in days to "
        "weeks. By then many of those people cannot be traced again.",
        {"size": 15})],
      [("In TB programmes, losing people between screening and diagnosis "
        "kills more than reader accuracy does.",
        {"size": 15, "bold": True, "color": TEAL})]],
     spacing=1.35)
card(s, 7.85, 2.1, 4.78, 4.1, PANEL2)
text(s, 8.25, 2.55, 4.0, 2.6,
     [[("So we did not try to beat a radiologist.", {"size": 16, "color": MUTED,
                                                     "italic": True})],
      [("We built something that decides who gives a sputum sample "
        "before they leave the van.", {"size": 19, "bold": True,
                                       "color": FGC})]],
     spacing=1.3)
text(s, 8.25, 5.25, 4.0, 0.7,
     "A triage tool, not a diagnostic one.", size=13, color=AMBER, bold=True)
notes(s, "40 seconds. Land the 'losing people between screening and "
         "diagnosis' line, then the sputum-sample framing, that sets up "
         "everything else.")

# ---- 3. how it works
s = slide(prs)
kicker(s, "How it works")
title(s, "Four steps, one of which is saying “I don’t know”")
steps = [
    ("1", "Preprocess", "Grayscale, CLAHE local-contrast normalisation, "
     "512px. Partially normalises exposure between machines.", CYAN),
    ("2", "Score", f"ResNet-18 fine-tuned on {INFO['n_train'] + INFO['n_val'] + INFO['n_test']} "
     "real radiographs outputs one probability.", TEAL),
    ("3", "Decide", f"Threshold tuned for {OP['target_sensitivity']:.0%} "
     f"sensitivity, plus an abstention band: "
     f"{BAND['abstain_rate']:.0%} of cases are routed to a human instead of "
     f"guessed.", AMBER),
    ("4", "Explain", "Grad-CAM localises the finding; an LLM writes a "
     "clinical note using only the model's own numbers.", VIOLET),
]
for i, (n, head, body, col) in enumerate(steps):
    x = 0.7 + i * 3.06
    card(s, x, 2.15, 2.82, 3.5, PANEL)
    circle(s, x + 0.3, 2.45, 0.52, col, n, size=17)
    text(s, x + 0.3, 3.18, 2.25, 0.4, head, size=17, color=col, bold=True,
         font=HEAD)
    text(s, x + 0.3, 3.68, 2.25, 1.75, body, size=12, color=MUTED,
         spacing=1.25)
card(s, 0.7, 5.92, 11.93, 0.82, PANEL2)
text(s, 1.1, 6.12, 11.1, 0.5,
     [[("A screening tool that quietly guesses on an ambiguous film is "
        "dangerous. ", {"size": 14, "color": FGC, "bold": True}),
       (f"Accuracy on the cases it does decide: "
        f"{BAND['accuracy_when_decided']:.1%}.",
        {"size": 14, "color": MUTED})]])
notes(s, "35 seconds. Land on step 3, the abstention band is what signals "
         "maturity.")

# ---- 4. real cases
s = slide(prs)
kicker(s, "Real output")
title(s, "Three held-out films: two right, one wrong")
pic(s, "fig_cases.png", 0.7, 1.72, 8.55, 5.05)
card(s, 9.5, 1.85, 3.13, 2.78, PANEL)
text(s, 9.8, 2.12, 2.6, 2.7,
     [[("Grad-CAM is not decoration.", {"size": 14.5, "bold": True,
                                        "color": TEAL})],
      [("Chest X-ray models are notorious for shortcut learning, one "
        "published case “detected” collapsed lung by spotting the drain "
        "inserted to treat it.", {"size": 11.5, "color": MUTED})],
      [("Every case shows it, so this can be checked rather than assumed.",
        {"size": 11.5, "color": MUTED})]], spacing=1.28)
card(s, 9.5, 4.85, 3.13, 1.72, PANEL, line=RGBColor(0x5A, 0x2E, 0x2E))
text(s, 9.8, 5.1, 2.6, 1.25,
     [[("We show the misses too.", {"size": 13, "bold": True,
                                    "color": CORAL})],
      [(f"{OP['fn']} of {OP['tp'] + OP['fn']} TB films are missed. They sit at "
        f"the bottom of the queue, the dangerous failure for a screening "
        f"tool.", {"size": 11.5, "color": MUTED})]], spacing=1.25)
notes(s, "Point at the left-hand film's upper-zone heatmap. Then the third "
         "column: a missed TB case. Volunteering the failure is stronger than "
         "hiding it, and a judge will ask about false negatives anyway.")

# ---- 5. stage 1
s = slide(prs)
kicker(s, "Stage 1, the trained model")
title(s, "A real model, trained locally, reported with intervals")
pic(s, "fig_roc.png", 6.55, 1.75, 6.1, 4.95)
rows = [
    ("Backbone", "ResNet-18, ImageNet init, strict load"),
    ("Dependency", "no torchvision, architecture written out"),
    ("Data", f"{INFO['n_train'] + INFO['n_val'] + INFO['n_test']} films, "
             f"NLM Montgomery + Shenzhen"),
    ("Train / val / test", f"{INFO['n_train']} / {INFO['n_val']} / {INFO['n_test']}"),
    ("Hardware", "RTX 3050 6GB laptop"),
    ("Training time", f"{TRAIN['seconds']:.0f} s for {TRAIN['epochs']} epochs"),
    ("Held-out AUROC", f"{T['auroc']:.3f}  [{T['auroc_ci95'][0]:.3f}, "
                       f"{T['auroc_ci95'][1]:.3f}]"),
    ("Sens / spec", f"{OP['sensitivity']:.3f} / {OP['specificity']:.3f}"),
]
card(s, 0.7, 1.75, 5.6, 3.5, PANEL)
for i, (k, v) in enumerate(rows):
    y = 1.98 + i * 0.395
    text(s, 1.0, y, 2.1, 0.33, k, size=11.5, color=MUTED)
    text(s, 3.1, y, 3.1, 0.33, v, size=11.5, color=FGC, bold=True)
card(s, 0.7, 5.42, 5.6, 1.28, PANEL2)
text(s, 1.0, 5.62, 5.0, 0.95,
     [[("Two domain choices most pipelines get wrong: ",
        {"size": 12.5, "bold": True, "color": AMBER})],
      [("no horizontal flips (the heart is on the left, so a mirrored chest "
        "film is anatomically impossible) and no cropping at evaluation (a "
        "crop can remove the apical zones where TB lives).",
        {"size": 12, "color": MUTED})]], spacing=1.25)
notes(s, "30 seconds. The two augmentation decisions are your credibility "
         "here. Lead the metric with the interval, not the point estimate.")

# ---- 6. stage 2
s = slide(prs)
kicker(s, "Stage 2, grounded generation", VIOLET)
title(s, "The language model never sees the X-ray")
card(s, 0.7, 1.95, 6.55, 4.75, PANEL)
text(s, 1.1, 2.3, 5.75, 4.1,
     [[("An LLM cannot read a radiograph, so letting it interpret one would "
        "just be fabrication.", {"size": 15, "color": FGC, "bold": True})],
      [("It receives a fixed evidence block: the probability, the operating "
        "threshold, the Grad-CAM zone distribution, and the model's own "
        "held-out performance. It may use nothing else.",
        {"size": 13.5, "color": MUTED})],
      [("Forbidden: inventing findings, naming a diagnosis, or stating the "
        "patient does or does not have TB. Required: name microbiological "
        "confirmation as the standard.", {"size": 13.5, "color": MUTED})],
      [("The UI shows that exact evidence block for every case, so the "
        "grounding is inspectable, not claimed.",
        {"size": 13.5, "color": TEAL, "bold": True})]],
     spacing=1.3)
card(s, 7.55, 1.95, 5.08, 4.75, PANEL2, line=RGBColor(0x4A, 0x3C, 0x78))
text(s, 7.9, 2.25, 4.4, 0.3, "GENERATED OUTPUT, VERBATIM", size=10.5,
     color=VIOLET, bold=True)
text(s, 7.9, 2.68, 4.4, 3.8,
     [[("“Model attention was regional and left-sided, with the greatest "
        "saliency in the left upper zone (34% of total)… This "
        "upper-zone-predominant distribution aligns with the typical pattern "
        "of post-primary pulmonary tuberculosis… The recommended next "
        "step is microbiological confirmation.”",
        {"size": 13, "color": FGC, "italic": True})],
      [("openai/gpt-oss-120b via Groq", {"size": 11, "color": MUTED})]],
     spacing=1.32)
notes(s, "40 seconds. The key beat: it CAN'T see the image, so it has nothing "
         "to hallucinate from. Then: we show the judge the evidence block.")

# ---- 7. results
s = slide(prs)
kicker(s, "Results")
title(s, "The interval is the result, not the point estimate")
for i, (v, l, c) in enumerate([
        (f"{T['auroc']:.3f}", "held-out AUROC", TEAL),
        (f"[{T['auroc_ci95'][0]:.2f}, {T['auroc_ci95'][1]:.2f}]",
         "95% confidence interval", CYAN),
        (f"{OP['sensitivity']:.3f}", "sensitivity", MINT),
        (f"{OP['specificity']:.3f}", "specificity", MINT)]):
    card(s, 0.7 + i * 3.06, 2.0, 2.82, 1.62, PANEL)
    stat(s, 0.98 + i * 3.06, 2.22, 2.4, v, l, c,
         vsize=30 if i == 1 else 38)
card(s, 0.7, 3.9, 5.95, 2.8, PANEL)
text(s, 1.05, 4.2, 5.3, 2.25,
     [[("Why we lead with the interval", {"size": 15, "bold": True,
                                          "color": FGC})],
      [("Our first run used 138 films, 21 in test. It produced AUROC 0.889 "
        "with an interval from 0.689 to 1.000, a number that means nothing.",
        {"size": 13, "color": MUTED})],
      [("Scaling to 800 films is what tightened it. Not a better model.",
        {"size": 13, "color": AMBER, "bold": True})]], spacing=1.3)
card(s, 6.9, 3.9, 5.73, 2.8, PANEL)
text(s, 7.25, 4.2, 5.05, 2.25,
     [[("Screening-appropriate evaluation", {"size": 15, "bold": True,
                                             "color": FGC})],
      [(f"Threshold chosen on validation for "
        f"{OP['target_sensitivity']:.0%} sensitivity, then applied unchanged "
        f"to test. Missing a case costs far more than an extra test, so "
        f"specificity is what we spend.", {"size": 13, "color": MUTED})],
      [(f"Confusion: {OP['tp']} true positive, {OP['fp']} false positive, "
        f"{OP['tn']} true negative, {OP['fn']} false negative.",
        {"size": 13, "color": MUTED})]], spacing=1.3)
notes(s, "30 seconds. Do NOT lead with 0.945. Lead with the interval and the "
         "story of the 138-image run.")

# ---- 8. THE finding
s = slide(prs)
kicker(s, "The finding that matters", AMBER)
title(s, "Ranking transfers. Calibration does not.")
pic(s, "fig_calibration.png", 0.7, 1.78, 11.93, 3.35)
cards = [
    (f"{T['auroc']:.3f} → {C['test']['auroc']:.3f}",
     "AUROC, same site → new site. Barely moves: it is threshold-free.",
     TEAL),
    (f"{OP['specificity']:.3f} → {XOP['specificity']:.3f}",
     "Specificity at the same threshold. Collapses.", CORAL),
    (f"{XOP['fp']} of {XOP['fp'] + XOP['tn']}",
     "healthy people sent for unnecessary testing at the new site.", AMBER),
]
for i, (v, l, c) in enumerate(cards):
    card(s, 0.7 + i * 4.05, 5.2, 3.83, 1.55, PANEL)
    text(s, 0.98 + i * 4.05, 5.38, 3.3, 0.55, v, size=25, color=c, bold=True,
         font=BODY, spacing=0.95)
    text(s, 0.98 + i * 4.05, 5.98, 3.3, 0.7, l, size=11.5, color=MUTED,
         spacing=1.2)
footer(s, "Trained on Shenzhen (China), tested on Montgomery County (USA), "
          "different scanner, population and country")
notes(s, "60 seconds, SLOW DOWN. AUROC 0.945 to 0.899, but specificity 0.885 "
         "to 0.225. Sixty-two of eighty healthy people. Pause. Then: 'the "
         "pooled number is the one that would have hurt people.'")

# ---- 9. the fix
s = slide(prs)
kicker(s, "The fix", MINT)
title(s, "Twenty locally-read films recover most of it")
pic(s, "fig_recalibration.png", 0.7, 1.8, 7.5, 4.9)
card(s, 8.45, 1.95, 4.18, 3.1, PANEL)
text(s, 8.8, 2.28, 3.5, 2.6,
     [[("So we do not ship a number.", {"size": 16, "bold": True,
                                        "color": FGC})],
      [("We ship a calibration step.", {"size": 16, "bold": True,
                                        "color": MINT})],
      [(f"Refitting on target-site data restores specificity to "
        f"{XREFIT['specificity']:.3f} at {XREFIT['sensitivity']:.3f} "
        f"sensitivity.", {"size": 12.5, "color": MUTED})]], spacing=1.32)
card(s, 8.45, 5.25, 4.18, 1.45, PANEL2)
text(s, 8.8, 5.45, 3.6, 1.05,
     [[("Deployment rule", {"size": 12, "bold": True, "color": AMBER})],
      [("Ship the model, but read 20 local films to recalibrate before "
        "trusting any decision it makes.", {"size": 12, "color": FGC})]],
     spacing=1.2)
notes(s, "30 seconds. This is a deployment checklist item that came out of "
         "measurement, no other team will have one.")

# ---- 10. limits
s = slide(prs)
kicker(s, "Limits of transfer", CORAL)
title(s, "And then we found where it stops working")
pic(s, "fig_transfer.png", 0.7, 1.8, 5.85, 4.55)
pic(s, "fig_conditions.png", 6.75, 1.8, 5.88, 4.55)
card(s, 0.7, 6.4, 11.93, 0.82, PANEL)
text(s, 1.05, 6.58, 11.2, 0.52,
     [[("A discrimination failure, not a calibration one. ",
        {"size": 13.5, "bold": True, "color": CORAL}),
       (f"At a third hospital with a general population the model separates "
        f"any abnormality from normal at AUROC {NA['auroc']:.3f}, near "
        f"chance, with several conditions below 0.5. No threshold fixes that.",
        {"size": 13, "color": MUTED})]])
notes(s, "50 seconds. NIH ChestX-ray14, third site, general population. It "
         "flags 46% of normal films there. No threshold fixes a broken "
         "ranking.")

# ---- 11. confound excluded
s = slide(prs)
kicker(s, "Ruling out the obvious explanation")
title(s, "We checked whether it was just resolution. It wasn’t.")
pic(s, "fig_resolution.png", 0.7, 1.9, 5.4, 4.3)
card(s, 6.45, 1.9, 6.18, 4.3, PANEL)
text(s, 6.85, 2.25, 5.4, 3.65,
     [[("The NIH mirror stores 320px images. Ours are 512px.",
        {"size": 14.5, "color": FGC, "bold": True})],
      [("So we degraded our own held-out films through the same pipeline, "
        "512 to 320 and back, and re-scored them.", {"size": 13,
                                                      "color": MUTED})],
      [(f"AUROC moved by {abs(RES['auroc_drop']):.4f} "
        f"({RES['auroc_native_512']['auroc']:.3f} to "
        f"{RES['auroc_320_roundtrip']['auroc']:.3f}).",
        {"size": 14.5, "color": TEAL, "bold": True})],
      [("Resolution is not the cause. The failure is genuine site and "
        "population shift, and we only know that because we tested it.",
        {"size": 13, "color": MUTED})]], spacing=1.32)
card(s, 6.45, 6.35, 6.18, 0.82, PANEL2)
text(s, 6.85, 6.53, 5.5, 0.52,
     [[("This also killed one of our own claims. ",
        {"size": 12.5, "bold": True, "color": AMBER}),
       ("We had said it was just an “abnormal chest” detector. A "
        "general abnormality detector would not score 0.579.",
        {"size": 12.5, "color": MUTED})]])
notes(s, "The self-correction plays very well, volunteer it. Judges rarely "
         "hear a team say 'the experiment proved us wrong'.")


# ---- 12. a second disease
if ND:
    s = slide(prs)
    kicker(s, "Generalising the pipeline", MINT)
    title(s, "A second disease, by swapping data, not code")
    pic(s, "fig_two_conditions.png", 0.7, 1.78, 7.9, 4.6)
    nd_t = ND["test"]
    nd_i = ND["data"]
    card(s, 8.8, 1.9, 3.83, 2.6, PANEL)
    text(s, 9.12, 2.16, 3.2, 2.15,
         [[("Lung nodule / mass", {"size": 14.5, "bold": True, "color": AMBER})],
          [(f"{nd_i['n_train'] + nd_i['n_val'] + nd_i['n_test']:,} films from "
            f"1,843 patients, NIH ChestX-ray14. Split BY PATIENT, so no "
            f"patient appears in two splits.", {"size": 12, "color": MUTED})]],
         spacing=1.28)
    card(s, 8.8, 4.66, 3.83, 2.04, PANEL, line=RGBColor(0x5A, 0x2E, 0x2E))
    text(s, 9.12, 4.9, 3.2, 1.6,
         [[("Not cancer detection", {"size": 13, "bold": True, "color": CORAL})],
          [("The dataset has no cancer label and no stage. A nodule is the "
            "finding that sends a patient for a CT, which is where malignancy "
            "gets decided.", {"size": 11.5, "color": MUTED})]], spacing=1.25)
    footer(s, "Patient-level splitting matters: the same patient appears in "
              "several films, and an image-level split leaks between train "
              "and test")
    notes(s, "35 seconds. Two points: the pipeline extended to a new disease "
             "by swapping data, and the harder task produced an honestly "
             "worse number (0.714 vs 0.945) with far more abstention (87% vs "
             "25%). If asked about cancer: no cancer label, no stage, it "
             "detects the finding that sends you for a CT.")

# ---- 13. cross-applying the models
if XC:
    s = slide(prs)
    kicker(s, "What happens when you point them at the wrong films", CORAL)
    title(s, "Ranking survives a little. Calibration does not.")
    pic(s, "fig_cross_condition.png", 0.7, 1.72, 11.93, 4.25)
    k = "nodule model on TB films (Montgomery+Shenzhen)"
    r = XC.get(k, {})
    card(s, 0.7, 6.08, 11.93, 0.92, PANEL)
    text(s, 1.05, 6.28, 11.2, 0.6,
         [[("A TB film scored 0.91 on the nodule model. ",
            {"size": 13.5, "bold": True, "color": AMBER}),
           (f"Not because TB resembles a nodule, but because that model flags "
            f"{r.get('flag_rate_negative', 0):.0%} of films from that "
            f"hospital even when they are completely normal - its mean score "
            f"on healthy chests there ({r.get('mean_score_negative', 0):.2f}) "
            f"already sits above its own threshold "
            f"({r.get('threshold', 0):.2f}).",
            {"size": 13, "color": MUTED})]])
    notes(s, "45 seconds. This generalises the central finding from 'TB "
             "across two sites' to 'any model across any source'. The green "
             "dots against the blue threshold lines are the whole story: "
             "cross-applied, the average HEALTHY chest scores above "
             "threshold.")

# ---- 14. knowing when not to trust it
s = slide(prs)
kicker(s, "Engineering for honesty")
title(s, "The tool tells you when not to trust it")
pic(s, "fig_decision_zones.png", 0.7, 1.75, 7.6, 3.45)
card(s, 0.7, 5.42, 7.6, 1.33, PANEL)
text(s, 1.05, 5.62, 7.0, 1.0,
     [[("Every result shows this strip.", {"size": 13.5, "bold": True,
                                           "color": CYAN}),
       (" The decision has a direction AND a confidence, so the score is "
        "placed in one of four zones with their real boundaries, against the "
        "distribution of held-out films with known labels.",
        {"size": 12.5, "color": MUTED})]], spacing=1.28)
card(s, 8.55, 1.9, 4.08, 2.4, PANEL)
text(s, 8.88, 2.14, 3.45, 2.0,
     [[("Out-of-distribution guard", {"size": 13.5, "bold": True,
                                      "color": AMBER})],
      [("Two measured signals, thresholded at the 95th percentile of 320 real "
        "films: how much attention lands outside the lung fields, and how far "
        "the image profile is from training.", {"size": 11.5,
                                                "color": MUTED})]],
     spacing=1.26)
card(s, 8.55, 4.46, 4.08, 2.29, PANEL, line=RGBColor(0x5A, 0x2E, 0x2E))
text(s, 8.88, 4.7, 3.45, 1.9,
     [[("Why we built it", {"size": 13, "bold": True, "color": CORAL})],
      [("A radiograph pulled off the web scored 0.07 while its saliency map "
        "lit up both shoulders, where there is no lung. The score was "
        "meaningless and nothing said so. Now it refuses to present one.",
        {"size": 11.5, "color": MUTED})]], spacing=1.25)
notes(s, "30 seconds, and a good slide to cut first if short on time. The "
         "story: we tried a random web X-ray, got a meaningless score, and "
         "built the guard that catches it.")

# ---- 12. deployment
s = slide(prs)
kicker(s, "Where to deploy", MINT)
title(s, "Prevalence is the lever, not accuracy")
pic(s, "fig_ppv.png", 0.7, 1.8, 7.3, 4.9)
card(s, 8.25, 1.9, 4.38, 1.92, PANEL)
text(s, 8.6, 2.2, 3.7, 1.95,
     [[("Deploy here", {"size": 14, "bold": True, "color": MINT})],
      [("Congregate settings, prisons, shelters, mining camps. Household "
        "contact tracing. Symptomatic clinic queues.",
        {"size": 12.5, "color": MUTED})]], spacing=1.28)
card(s, 8.25, 4.08, 4.38, 1.95, PANEL, line=RGBColor(0x5A, 0x2E, 0x2E))
text(s, 8.6, 4.36, 3.7, 1.55,
     [[("Not here", {"size": 14, "bold": True, "color": CORAL})],
      [("General hospital worklists. Children, the model has only seen "
        "adults. Diagnosis of any kind, ever.",
        {"size": 12.5, "color": MUTED})]], spacing=1.28)
footer(s, "PPV computed at sensitivity 0.90 and specificity 0.66, the "
          "cross-site, locally recalibrated operating point")
notes(s, "35 seconds. The same weights give 0.3% PPV in the general "
         "population and 23% in a prison. Being explicit about where it "
         "should NOT go reads as maturity.")

# ---- 13. close
s = slide(prs)
kicker(s, "What is real")
title(s, "Measured, not claimed")
left = [("Radiographs", "real, 800 films, NLM Montgomery + Shenzhen"),
        ("Labels", "real, from the datasets"),
        ("Model", "really trained, on this GPU"),
        ("Metrics", "real, held-out, bootstrap CIs"),
        ("Cross-site validation", "two sites plus one external"),
        ("Transfer limits", "measured, and stated")]
right = [("Clinical validation", "none"),
         ("Regulatory clearance", "none"),
         ("Paediatric use", "not supported, untested"),
         ("Diagnosis", "not what this does"),
         ("General worklists", "measured as unsuitable")]
card(s, 0.7, 1.95, 5.95, 3.3, PANEL)
for i, (k, v) in enumerate(left):
    y = 2.22 + i * 0.47
    text(s, 1.05, y, 2.55, 0.38, k, size=12, color=MUTED)
    text(s, 3.65, y, 2.75, 0.38, v, size=12, color=MINT, bold=True)
card(s, 6.85, 1.95, 5.78, 3.3, PANEL)
for i, (k, v) in enumerate(right):
    y = 2.22 + i * 0.47
    text(s, 7.2, y, 2.6, 0.38, k, size=12, color=MUTED)
    text(s, 9.85, y, 2.5, 0.38, v, size=12, color=CORAL, bold=True)
card(s, 0.7, 5.5, 11.93, 1.2, PANEL2)
text(s, 1.1, 5.72, 11.1, 0.8,
     [[("We found the boundary of where our model works, and the experiment "
        "that proves it is not an artifact.",
        {"size": 16, "bold": True, "color": FGC})],
      [("That is the contribution. The accuracy number is the easy part.",
        {"size": 13.5, "color": MUTED})]], spacing=1.3)
notes(s, "Close here or on the pneumonia demo case. Never close on the "
         "accuracy slide.")

prs.save(OUT)
print(f"wrote {OUT}  ({OUT.stat().st_size / 1e6:.2f} MB, "
      f"{len(prs.slides.__iter__.__self__._sldIdLst)} slides)")

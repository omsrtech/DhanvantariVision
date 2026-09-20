# 🫁 Dhanvantari Vision

**Tuberculosis chest X-ray screening triage: a CNN trained locally on GPU, plus a referral note grounded strictly in that model's own output.**

The use case is not accuracy — it's **access**. Chest radiography is the frontline TB screening tool, and the bottleneck is readers, not films. A clinic can acquire far more radiographs in a day than a scarce radiologist can report, so films get read in arrival order and a patient with advanced disease can sit at position 63 in the queue. Dhanvantari Vision reorders the queue so the films most likely to show TB are read first, and routes genuinely ambiguous cases to a human instead of guessing.

```bash
# train (GPU) and serve
python scripts/fetch_data.py
python scripts/fetch_shenzhen_subset.py --per-class 340 --workers 14
python scripts/prepare.py
python scripts/train.py --split split          # pooled
python scripts/train.py --split cross_site     # train Shenzhen -> test Montgomery
python scripts/calibration_analysis.py
python scripts/predict_all.py --tag "" --split split
streamlit run app.py                            # → http://localhost:8502
```

Training wants a CUDA GPU (this was trained on an RTX 3050, 6 GB). Running the app does not - it scores on CPU in about a second a film.

---

## Run it (5 minutes, no training, no GPU)

The radiographs are not redistributed and the weights are too big to commit,
so there are two steps between a clone and a running app.

```bash
git clone https://github.com/omsrtech/DhanvantariVision.git
cd DhanvantariVision
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python scripts/fetch_models.py    # ~90 MB, from this repo's Releases
python scripts/fetch_data.py      # rebuilds the radiographs - see DATA.md
python scripts/prepare.py

streamlit run app.py
```

**Optional - nicer written notes.** Every result already comes with a clinical
note built from the model's own numbers. Copy
`API_KEY.example.txt` to `API_KEY.txt`, paste a free
[Groq](https://console.groq.com/keys) key into it, and an LLM rewrites those notes in natural clinical prose instead. It changes the
*wording* only - never the score, never the decision. The sidebar says which
mode is live.

### What ships in this repo

| | |
|---|---|
| `models/tbscope_split.pt` | TB model (ResNet-18, 800 radiographs) |
| `models/tbscope_split_nodule.pt` | lung nodule / mass model (2,214 radiographs) |
| `data/processed*/*.csv` | derived tables: probability, decision, Grad-CAM zones - no pixels |
| `reports/` | every metric and figure quoted below |
| `scripts/` | fetch -> prepare -> train -> evaluate, end to end |

**No radiographs are in this repository.** They belong to the NLM and the NIH
Clinical Center, and the NLM asks that its TB sets not be redistributed. See
**[DATA.md](DATA.md)** for provenance, citations, and how to get them.

---

## The headline result

Two evaluations of the same model, and the gap between them is the whole point.

| | Pooled (same sites) | **Cross-site** (train Shenzhen → test Montgomery) |
|---|---|---|
| AUROC | 0.945 `[0.900, 0.981]` | **0.899** `[0.840, 0.947]` |
| Sensitivity | 0.831 | 0.983 |
| Specificity | 0.885 | **0.225** |
| False positives | 7 / 61 normals | **62 / 80 normals** |

**Ranking transfers. Calibration does not.**

AUROC barely moved (0.945 → 0.899) because it is threshold-free — the model still *orders* radiographs correctly at a completely new site. But the decision threshold tuned on the source site produced **specificity 0.225** at the target site: 62 of 80 healthy people would have been referred for unnecessary testing.

The cause is measurable. The mean score on normal films nearly doubled across sites (0.122 → 0.224). The score distribution moved, so a fixed threshold no longer sits where it was calibrated to sit.

**The fix is cheap, and that's the useful engineering conclusion.** Refitting the threshold on target-site data restores specificity to 0.688 at 0.914 sensitivity. Just **20 locally-read radiographs** recover most of it (median specificity 0.664 at sensitivity 0.897, over 300 resamples). So the deployment rule is: ship the model, but calibrate the threshold on a small local sample before trusting any decision it makes.

Most imaging demos report only the pooled number. The pooled number is the one that would have got people hurt.

---

## The two-stage pipeline

### Stage 1 — trained model

* **Architecture:** ResNet-18 written out explicitly in `src/model.py`, initialised from the official ImageNet weights and loaded with `strict=True`. That strict load is the correctness check — any name or shape mismatch raises instead of silently training from scratch. **No torchvision dependency.**
* **Data:** 800 radiographs — NLM Montgomery County (138) + NLM Shenzhen (662). Labels are encoded in the filenames (`CHNCXR_0327_1.png` → 1 = TB). Both classes come from the same scanner at each site.
* **Preprocessing:** grayscale, CLAHE for local contrast (standard for chest radiographs; partially normalises exposure differences between sites), resized to 512² and stored.
* **Training:** fine-tuned all layers, AdamW, cosine schedule, AMP, `pos_weight` for class balance. **14 epochs in 100 seconds** on an RTX 3050 6GB.

Two augmentation decisions that are domain-specific and easy to get wrong:

* **No horizontal flips.** It's the default in most vision pipelines and it's wrong here: the heart sits on the left and the liver on the right, so a mirrored radiograph is anatomically impossible and teaches the model that laterality carries no information.
* **No cropping at evaluation.** Random-resized-crop is standard on ImageNet, but a crop can remove the apical or costophrenic regions where TB findings concentrate. Training uses mild scale jitter; evaluation resizes the whole film so no lung field is discarded.

### Stage 2 — grounded referral note

The language model **never sees the radiograph** and is never asked for a diagnosis. A language model cannot read a chest X-ray, so letting it "interpret" one would be fabrication. What it can legitimately do is turn a screening result into the note a clinician needs to act on, while carrying the uncertainty forward.

It receives a fixed evidence block — the calibrated probability, the operating point in use, the Grad-CAM zone distribution, and the model's own held-out performance — and is instructed to use only those numbers, to describe attention as *model behaviour* rather than a finding, never to state that the patient does or does not have TB, and to name microbiological testing as the diagnostic standard. **Every case in the UI shows the exact block it was given.**

* Provider: **Groq**, `openai/gpt-oss-120b` (fallback `qwen/qwen3.8-27b`).
* Key read from a gitignored `.env` as `GROQ_API_KEY`.
* With no key, a template fills the same structure from the same evidence and the UI labels it as such. The note is never fabricated from nothing.

---

## Screening-appropriate evaluation

* **Bootstrap confidence intervals.** With a few hundred test films a point estimate means little. On the first run (138 images, 21 in test) AUROC was 0.889 with a CI of `[0.689, 1.000]` — an interval so wide it was uninformative. Going to 800 images tightened it to `[0.900, 0.981]`. The interval, not the point estimate, is the result.
* **Sensitivity-first operating point.** The threshold is chosen on the *validation* set to reach a target sensitivity (default 90%), then applied unchanged to test. Missing a TB case is far worse than an unnecessary follow-up, so specificity is what gets spent. Tuning the threshold on test would look better and mean nothing.
* **Abstention band.** Scores in the uncertain middle are routed to a human rather than called either way. On the pooled split that is 25% of cases, with 94.4% accuracy on the rest. A screening tool that quietly guesses on ambiguous films is dangerous.

---

## Did it learn tuberculosis, or an artifact?

Chest X-ray models are notorious for shortcut learning — "detecting" pneumothorax by spotting the chest drain placed to treat it, or separating classes by a hospital's burned-in text markers. Grad-CAM (`src/gradcam.py`) exists so this can be checked rather than assumed, and every case in the UI shows it.

Measured on the 800-image model, held-out cases only:

| | Top attention in an **upper** zone |
|---|---|
| TB-positive | **58.6%** (34/58) |
| Normal | 29.0% (9/31) |

Post-primary TB is characteristically an upper-lobe disease, and the model was given only image-level labels — no lesion boundaries, nothing telling it where to look. TB cases are about twice as likely as normals to draw upper-zone attention, and attention is concentrated (44.8% of saliency in one zone versus 16.7% for uniform). Upper zone is the single most common band for TB cases (34 of 58, versus 17 mid and 7 lower).

**Reported honestly:** this is a tendency, not a signature. See *Limits of transfer* below for the experiments that bound what this signal actually is. An earlier version of this analysis on a 96-image model showed 89.5%, which did not survive training on more data and evaluating only on held-out films — a small-sample artifact. Note also that the dataset has no lesion boundaries, so Grad-CAM **cannot** be validated against ground-truth localisation. It indicates model attention only.

### A bug this analysis caught

46 of 138 cases (and 268 of 800) produce an **all-zero Grad-CAM**, and the UI was initially printing "right upper zone (0%)" for them — implying attention where there was none. The cause is legitimate: Grad-CAM is computed on the positive-class logit and ReLU-clipped, so a confidently *negative* prediction has no evidence *for* TB to localise. The UI now says "no localised attention" and explains why, rather than fabricating a ranking out of zeros.

---

## Limits of transfer — where it stops working

Three experiments, each ruling out an explanation for the last. This is the
most important section of the project.

| Test | Task | AUROC |
|---|---|---|
| Own sites, pooled | TB vs normal | **0.945** `[0.900, 0.981]` |
| Cross-site, Shenzhen → Montgomery | TB vs normal | **0.899** `[0.840, 0.947]` |
| Third site, NIH ChestX-ray14 | any abnormality vs normal | **0.579** `[0.520, 0.640]` |

**1. Within TB screening, it transfers.** Trained on Shenzhen and tested on
Montgomery — different country, scanner and population — ranking held up
(0.899). Only the threshold moved, and 20 local films fix that.

**2. On a general hospital population, it collapses.** Tested against NIH
ChestX-ray14 (a third site, 14 labelled pathologies, general population) the
model separates *any abnormality* from normal at 0.579 — near chance. Several
individual conditions fall below 0.5: Nodule 0.404, Atelectasis 0.449,
Pleural Thickening 0.457, Emphysema 0.472. It flags **46% of normal films**
there. This is a *discrimination* failure, not a calibration one, so no
threshold adjustment repairs it.

**3. Resolution is not the cause.** The NIH mirror stores 320px images while
training used 512px, so we degraded our own held-out films through the same
pipeline (512 → 320 → 512) and re-scored. AUROC moved by **0.0014**
(0.9247 → 0.9233). The confound is excluded; the NIH failure is genuine site
and population shift.

### What that means

**It is more TB-specific than it first appeared.** An earlier read of ours —
based on six paediatric pneumonia films scoring a mean of 0.821 — was that the
model is simply an "abnormal chest" detector. The NIH result refutes that: a
general abnormality detector would not score 0.579 at abnormal-vs-normal. Those
paediatric films were almost certainly flagged for being out-of-distribution
(children, third source) rather than for pneumonia specifically.

The only conditions the model flags well above chance are **Pneumothorax
(0.726)** and **Fibrosis (0.713)** — and fibrotic scarring is a recognised TB
sequela, so what it learned looks TB-adjacent rather than generic.

### The operational rule this produces

> **Valid only in a TB-screening population resembling the training data.
> Pointed at a general hospital worklist, it is near-useless.**

That sharpens the deployment scope rather than shrinking it: congregate-setting
screening (prisons, shelters, mining camps) and household contact tracing are
exactly TB-screening populations, and they carry the high prevalence that makes
the positive predictive value workable.

A corollary worth stating: a "normal gate" for general radiology backlogs — an
attractive-sounding extension of this model — is **not supported** by these
results and should not be claimed.

Reproduce with:

```bash
python scripts/test_other_pathology.py     # flag rate per unseen condition
python scripts/nih_discrimination.py       # calibration vs discrimination
python scripts/resolution_ablation.py      # excludes the resolution confound
```

Results land in `reports/other_pathology.json`, `reports/nih_discrimination.json`
and `reports/resolution_ablation.json`.

---

## What's real and what isn't

| Component | Status |
|---|---|
| Radiographs | **Real** — NLM Montgomery + Shenzhen, 800 films |
| Labels | **Real**, from the datasets |
| Model | **Really trained**, locally, on an RTX 3050 |
| Metrics | **Real**, held-out, with bootstrap CIs |
| Cross-site evaluation | **Real** — train Shenzhen, test Montgomery |
| Grad-CAM | **Real** model gradients |
| Referral note | **LLM-generated**, constrained to the evidence block |
| Transfer limits | **Measured** - see *Limits of transfer* |
| Clinical validation | **NONE** |
| Regulatory clearance | **NONE** |

A dataset we deliberately **rejected**: a 3,300-image pre-resized set whose TB images came from `mendeley_TB` and whose normal images came from `jtiptj` — two different source datasets. A model trained on it would learn to identify the *source*, not tuberculosis, and would score ~99% while being clinically worthless. Both classes in the NLM sets come from the same scanner at the same site.

> This is a hackathon prototype built on public research datasets. It is not a medical device, has no clinical validation, and must not be used to make decisions about any real patient.

---

## Project layout

```
app.py                            Streamlit triage UI (worklist, case detail, report card)
src/model.py                      ResNet-18, explicit, loads ImageNet weights strict
src/data.py                       dataset + CXR-appropriate augmentation
src/gradcam.py                    Grad-CAM + anatomical zone attention
src/report.py                     Stage 2: evidence block, Groq call, template fallback
scripts/fetch_data.py             NLM zips (HF mirror, NLM fallback) + ImageNet weights
scripts/fetch_shenzhen_subset.py  parallel per-file fetch of the Shenzhen set
scripts/prepare.py                extract, CLAHE, 512px, pooled + cross-site splits
scripts/train.py                  GPU training + screening-appropriate evaluation
scripts/calibration_analysis.py   what transfers across sites and what doesn't
scripts/predict_all.py            batch inference + cached Grad-CAM overlays
models/                           checkpoints + metrics_*.json
reports/calibration.json          cross-site calibration findings
```

### A note on bandwidth

The NLM host served at ~0.45 MB/s (2.6 hours for the archives) and a single stream from the HF mirror degraded to 0.2 MB/s. Fetching the 662 Shenzhen radiographs as **individual files over 14 parallel connections** ran at **12.9 MB/s** — 3.77 GB in 196 seconds. If you're re-running this on a throttled connection, use `fetch_shenzhen_subset.py`, not the archive.

---

## Sources

* Jaeger S. et al. *Two public chest X-ray datasets for computer-aided screening of pulmonary diseases* — the Montgomery County and Shenzhen sets, U.S. National Library of Medicine.
* WHO endorses computer-aided detection (CAD) for TB screening from chest radiography, which is the product category this prototype sits in.

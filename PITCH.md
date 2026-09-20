# Dhanvantari Vision - presentation pack

Everything needed to build and deliver the pitch. Part 1 is a prompt to paste
into an AI slide generator. Part 2 is your speaker notes. Part 3 is judge Q&A prep.

**Every number in here is measured, not estimated.** Sources: `models/metrics_split.json`,
`models/metrics_cross_site.json`, `reports/calibration.json`,
`reports/nih_discrimination.json`, `reports/resolution_ablation.json`.

---

# PART 1 — Prompt for generating the slide deck

Paste the block below into Gamma, Claude, ChatGPT, Copilot, Beautiful.ai or
similar. It is written so the tool cannot invent numbers.

---

**PROMPT STARTS HERE**

Create a 10-slide technical presentation for a hackathon project called
**Dhanvantari Vision**. Audience: hackathon judges — technical, some with ML background,
some clinical. Tone: confident, precise, evidence-led. Not salesy. Do not add
statistics, claims or citations that are not in this brief — every number below
is measured and there are no others.

Visual style: dark background (near-black #05080c), cyan and teal accents
(#38bdf8, #22d3ee), one clear idea per slide, large numbers, minimal text.
Medical-instrument aesthetic, not corporate. Use tables for the metric
comparisons rather than prose.

**Slide 1 - Title**
Dhanvantari Vision. "A tuberculosis chest X-ray screening triage tool — and an honest map
of where it stops working." Subtitle: built in one day on an RTX 3050 laptop GPU.

**Slide 2 — The problem is access, not accuracy**
Chest radiography is the frontline TB screening tool. The bottleneck is not
machines, it is readers. A mobile screening van images ~400 people a day with no
radiologist on site; films are batched to a city hospital and results return in
days to weeks. In TB programmes, loss to follow-up between screening and
diagnosis is a bigger killer than reader accuracy. Key line: "We are not trying
to beat a radiologist. We are trying to decide who gives a sputum sample before
they leave the van."

**Slide 3 — What it does**
Four steps: (1) preprocess — grayscale, CLAHE contrast normalisation, 512px;
(2) score — ResNet-18 fine-tuned on 800 real radiographs outputs one
probability; (3) decide — threshold tuned for 90% sensitivity, with a
deliberate abstention band where it says "a human must read this" instead of
guessing; (4) explain — Grad-CAM localises the finding, then an LLM writes a
clinical note using only the model's own numbers. Note that 25% of cases land
in the abstention band, with 94.4% accuracy on the remainder.

**Slide 4 — Stage 1: the trained model**
ResNet-18 written out explicitly and initialised from official ImageNet weights
loaded with strict=True (no torchvision dependency). Data: 800 radiographs from
the NLM Montgomery County (138) and Shenzhen (662) sets. Trained 14 epochs in
100 seconds on an RTX 3050 6GB. Two domain-specific decisions worth calling
out: no horizontal flips (the heart is on the left, so a mirrored chest X-ray is
anatomically impossible) and no cropping at evaluation (a crop can remove the
apical zones where TB concentrates).

**Slide 5 — Stage 2: the LLM is constrained, not creative**
The language model never sees the radiograph. It receives a fixed evidence
block — probability, operating threshold, Grad-CAM zone distribution, and the
model's own held-out performance — and may use only those numbers. It is
forbidden from inventing radiological findings, from stating the patient has or
does not have TB, and must name microbiological testing as the diagnostic
standard. The UI shows the exact evidence block for every case. Include this
real generated output as a quote: "The screening model assigned a TB-consistent
abnormality probability of 0.989, well above the operating threshold of 0.597...
Model attention was regional and left-sided, with the greatest saliency in the
left upper zone (34% of total)... This upper-zone-predominant distribution
aligns with the typical pattern of post-primary pulmonary tuberculosis."

**Slide 6 — Results, with confidence intervals**
Table: Pooled held-out AUROC 0.945 (95% CI 0.900-0.981); sensitivity 0.831;
specificity 0.885; confusion tp=49 fp=7 tn=54 fn=10. Make the point that the
interval is the result, not the point estimate: an earlier 138-image run gave
AUROC 0.889 with a CI of 0.689-1.000, which was uninformative. More data
tightened it.

**Slide 7 — The finding that matters: ranking transfers, calibration does not**
This is the centrepiece slide. Table comparing same-site vs cross-site (trained
on Shenzhen, tested on Montgomery — different country, scanner, population):
AUROC 0.945 vs 0.899, but specificity 0.885 vs 0.225. The threshold imported
from the source site sent 62 of 80 healthy people for unnecessary testing.
Cause: the mean score on normal films nearly doubled across sites (0.122 to
0.224), so the threshold no longer sits where it was calibrated. AUROC is
threshold-free, which is why it barely moved.

**Slide 8 — The fix is cheap and specific**
Refitting the threshold on target-site data restores specificity to 0.688 at
0.914 sensitivity. Twenty locally-read radiographs recover most of it (median
specificity 0.664 at sensitivity 0.897, over 300 resamples). Deployment rule:
"Ship the model, but read 20 local films to recalibrate before trusting any
decision it makes."

**Slide 9 — Where it stops working, and the experiment that proves it**
Three experiments, each ruling out an explanation. Table: own sites TB vs normal
AUROC 0.945; cross-site TB vs normal 0.899; third site (NIH ChestX-ray14,
general hospital population, any abnormality vs normal) 0.579 with CI
0.520-0.640 — near chance, with individual conditions below 0.5 (Nodule 0.404,
Atelectasis 0.449). It flags 46% of normal films there. This is a
discrimination failure, not calibration, so no threshold repairs it. Confound
excluded: the NIH mirror is 320px versus our 512px, so we degraded our own
held-out films through the same pipeline and AUROC moved by 0.0014
(0.9247 to 0.9233). Conclusion: valid only in a TB-screening population
resembling the training data.

**Slide 10 — Where to deploy, with arithmetic**
Positive predictive value depends on prevalence, so the same model is roughly
ten times more useful in a prison than in a general population. Table of PPV by
setting (sens 0.90, spec 0.66): general population 0.1% prevalence -> PPV 0.3%;
mobile screening 1% -> 2.6%; household contacts 5% -> 12%; prisons and shelters
10% -> 23%; symptomatic clinic 20% -> 40%. Recommended deployment:
congregate-setting screening and household contact tracing. Explicitly not for:
general hospital worklists, paediatrics (never trained on children), or
diagnosis of any kind. Close with the disclaimer that this is a prototype on
public research data with no clinical validation and no regulatory clearance.

**PROMPT ENDS HERE**

---

# PART 2 — Speaker notes

Target: 5 minutes plus demo. Bold lines are worth saying close to verbatim.

### Slide 1 — Title (10s)
Don't explain the name. Go straight to the problem.

### Slide 2 — The problem (40s)
> "Chest X-ray is the frontline tool for TB screening. The bottleneck isn't
> machines — it's readers. A mobile screening van images four hundred people a
> day with no radiologist on board. Those films get batched to a city hospital
> and the report comes back days or weeks later, by which time a lot of those
> people cannot be traced again. **In TB programmes, losing people between
> screening and diagnosis kills more than reader accuracy does.**"

Then the framing sentence that sets up everything:
> "So we didn't try to beat a radiologist. We built something that decides who
> gives a sputum sample before they leave the van."

### Slide 3 — What it does (35s)
Walk the four steps quickly. Land on the abstention band, because it is the
part that signals maturity:
> "It has a third answer. Twenty-five percent of cases fall in a band where it
> says 'a human must read this' instead of guessing. A screening tool that
> quietly guesses on an ambiguous film is dangerous."

### Slide 4 — The model (30s)
The two augmentation decisions are your credibility on this slide:
> "Two things most pipelines get wrong here. We don't horizontally flip —
> that's the default augmentation in vision, but the heart is on the left, so a
> mirrored chest X-ray is anatomically impossible. And we don't crop at
> evaluation, because a crop can cut off the apical zones where TB actually
> lives."

If asked about the GPU: 14 epochs, 100 seconds, laptop RTX 3050.

### Slide 5 — Stage 2 (40s)
The key beat:
> "The language model never sees the X-ray. It can't — an LLM cannot read a
> radiograph, so letting it interpret one would just be fabrication. What it
> gets is a fixed evidence block: the probability, the threshold, where
> Grad-CAM says the model looked, and the model's own held-out performance.
> It's instructed to use only those numbers. **And we show the judge that exact
> block for every case, so the grounding is inspectable rather than claimed.**"

### Slide 6 — Results (30s)
Do not lead with 0.945. Lead with the interval:
> "AUROC 0.945, and the interval is 0.900 to 0.981. The interval is the result.
> Our first run on 138 images gave 0.889 with an interval from 0.69 to 1.00 —
> which is a number that means nothing. More data is what fixed that, not a
> better model."

### Slide 7 — The centrepiece (60s) — SLOW DOWN HERE
> "Then we did the test most imaging demos skip. We trained only on the
> Shenzhen films and tested only on Montgomery — different country, different
> scanner, different population.
>
> AUROC barely moved: 0.945 to 0.899. But specificity went from 0.885 to
> **0.225**. Sixty-two of eighty healthy people would have been sent for
> unnecessary testing.
>
> **Ranking transfers. Calibration does not.** AUROC is threshold-free, so it
> survives — the model still orders films correctly somewhere it's never seen.
> But sensitivity and specificity are measured at a fixed threshold, and the
> score distribution moved: mean score on normal films nearly doubled. The
> threshold no longer sits where we calibrated it."

Pause. Then:
> "The pooled number is the one that would have hurt people."

### Slide 8 — The fix (30s)
> "Twenty locally-read films recover most of the specificity. So we don't ship a
> number — we ship a calibration step. That's a deployment checklist item that
> came out of measurement."

### Slide 9 — Limits (50s)
This slide wins technical judges. Deliver it as a chain of experiments:
> "Then we pushed further and tested on NIH ChestX-ray14 — a third hospital, a
> general population, fourteen labelled pathologies. There it separates any
> abnormality from normal at 0.579. Near chance. Some conditions are below 0.5.
> It flags forty-six percent of normal films.
>
> That's a discrimination failure, not a calibration one — no threshold fixes
> it. And before we concluded that, we checked the obvious confound: the NIH
> mirror is 320 pixels, ours is 512. So we degraded our own held-out films
> through the same pipeline and re-scored. AUROC moved by 0.0014. **Resolution
> isn't the cause. The failure is real.**
>
> So the rule is: this is valid in a TB-screening population that resembles its
> training data, and near-useless on a general hospital worklist."

If you have time, add the self-correction — it plays very well:
> "This also killed one of our own earlier claims. We'd said it was just an
> 'abnormal chest' detector based on six pneumonia films. A general abnormality
> detector wouldn't score 0.579 on abnormal-versus-normal. We were wrong, and
> the experiment is what told us."

### Slide 10 — Deployment (35s)
> "Positive predictive value depends on prevalence, not on our model. The same
> weights give a 0.3% PPV in a general population and 23% in a prison. So we're
> explicit: congregate settings — prisons, shelters, mining camps — and
> household contact tracing. Not general hospital worklists. Not children, which
> we measured. And not diagnosis, ever."

### Demo (60-90s) — run these three in order
1. **Worklist** — "120 held-out films, sorted by probability. This ordering is
   the whole product."
2. **`CHNCXR_0601_1`** (held-out TB) — 0.99, refer. Point at the Grad-CAM on the
   left upper zone and the clinical note below it. "That note is generated from
   the numbers above it, and here's the exact evidence block."
3. **`jtiptj_train_person1016`** (pneumonia, paediatric) — 0.94, human read.
   "This is pneumonia in a child. The model has no such category. This is why
   there's an abstention band and why the note never says the word
   tuberculosis."

Close on slide 9 or the third demo case, never on the accuracy slide.

### Timing
| Section | Time |
|---|---|
| Slides 1-3 | 1:25 |
| Slides 4-6 | 1:40 |
| Slides 7-8 | 1:30 |
| Slides 9-10 | 1:25 |
| Demo | 1:30 |
| **Total** | **~7:30** (trim slides 4 and 6 for a 5-minute slot) |

---

# PART 3 — Judge Q&A prep

**"Why not just use an existing CAD product?"**
Commercial TB CAD exists and is WHO-endorsed; we're not claiming to beat it.
What we built is the honest evaluation around it — the cross-site calibration
finding and the transfer boundary — which is what determines whether any of
these tools works in a new clinic.

**"Only 800 images. Isn't that far too few?"**
Yes, and that's why every number has a confidence interval. Our first 138-image
run gave an interval of 0.69 to 1.00, which we reported as uninformative rather
than quoting 0.889. The pipeline scales; the honesty is the contribution.

**"How do you know it isn't shortcut learning?"**
We checked rather than assumed. Grad-CAM is on every case. On held-out films,
TB cases draw upper-zone attention 58.6% of the time versus 29.0% for normals,
with 44.8% of saliency concentrated in one zone against 16.7% for uniform.
Post-primary TB is an upper-lobe disease. It's a tendency, not proof — and we
found and fixed a real bug in that analysis where 268 of 800 cases had empty
saliency maps being displayed as though they had a zone.

**"Is the LLM hallucinating the report?"**
It cannot see the image, so it has nothing to hallucinate from. It gets a fixed
evidence block, and the UI shows that block for every case. Without an API key
it degrades to a template filled from the same evidence and labels itself as
such.

**"What about false negatives?"**
Ten in the held-out set. Two of the earliest ones scored 0.083 and 0.027 — they
would sit at the very bottom of the triage queue, which is the dangerous
failure mode for a screening tool and more important than the AUROC.

**"Did you use vision models?"**
Deliberately not. Our Groq account has two vision-capable models and we tested
that they accept a radiograph. We didn't use them, because a general-purpose
VLM is not a radiologist and would confidently narrate findings it cannot
reliably see. Putting that next to a validated CNN would undermine the CNN.

**"What would you do next, with a week?"**
Validate on full-resolution NIH images to confirm the transfer boundary;
fine-tune on a small target-site sample to see whether discrimination recovers;
and get a multi-pathology validation set so the abnormal-versus-normal question
can be answered properly rather than inferred.

**Two things never to say:** that it diagnoses tuberculosis, or that it is
ready for patients.

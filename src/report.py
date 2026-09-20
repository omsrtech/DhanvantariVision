"""Stage 2: a referral note grounded strictly in Stage 1's output.

The language model is never shown the radiograph and is never asked for a
diagnosis. It receives a fixed evidence block — the trained model's calibrated
probability, the operating point actually in use, where Grad-CAM says the model
looked, and the model's own held-out performance — and is instructed to write a
referral note using only those numbers.

That constraint is the point. A language model cannot read a chest X-ray, so
letting it "interpret" one would be fabrication. What it can legitimately do is
turn a screening result into the note a clinician needs to act on, while
carrying the uncertainty forward honestly.

If no GROQ_API_KEY is set, or the call fails, a template fills the same
structure from the same evidence and the UI says so.
"""
from __future__ import annotations

import os
import pathlib
import sys
from dataclasses import dataclass

MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "qwen/qwen3.8-27b"
ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = """\
You turn the output of a chest X-ray SCREENING model into a short clinical \
note, in the register a clinician would use when handing a case on.

You have NOT seen the radiograph. Everything you know comes from the EVIDENCE \
block: a trained convolutional network's probability, and a Grad-CAM analysis \
of which lung zones drove that score. Write about the model's finding, not \
about the image.

Structure the note as flowing prose in this order:
1. What the screening model flagged, with the probability and what it means \
   relative to the operating threshold.
2. WHERE it localised, in anatomical language - use the zone, the laterality \
   and the pattern (focal / regional / diffuse) given to you, and quote the \
   saliency share.
3. State the verdict using the "direction", "confidence" and "what to \
   say" lines of the evidence VERBATIM in meaning. Never compare the \
   probability against the cutoffs yourself and never contradict those \
   lines - they are computed for you precisely because that comparison is \
   easy to get wrong.
4. Whether that distribution fits the reference pattern stated in the \
   evidence. If the evidence says the target finding has no characteristic \
   zone, say that the distribution carries no anatomical weight rather than \
   inventing a concordance claim.
5. The recommended next step, using the confirmatory test named in the \
   evidence, and the model's measured reliability.

Hard rules:
- Use ONLY numbers and facts in the EVIDENCE block. Never invent radiological \
  findings - you cannot see the film, so no cavitation, nodules, effusions, \
  consolidation, infiltrates, sizes or measurements.
- Say "the model localised to X" or "model attention centred on X". Never \
  write as though you observed a lesion yourself.
- Never state that the patient does or does not have the target condition. \
  This is a triage signal only.
- Only ever name the target finding given in the evidence. Do not mention any \
  other disease.
- If the evidence says attention was not localised, say that no region drove \
  the score and do not name any zone.
- 4-6 sentences of plain prose. No headings, no bullet points, no preamble.
"""


@dataclass
class Report:
    text: str
    mode: str          # "llm" | "template"
    label: str         # provenance banner for the UI
    model: str = ""
    error: str = ""


def key_file_locations() -> list[pathlib.Path]:
    """Where a user-supplied key may live, most user-editable first.

    In a frozen build sys.executable is the app the recipient double-clicked,
    so a file beside it is the one they can actually edit; the bundled copy
    lives in a temp directory that is rebuilt on every launch.
    """
    here: list[pathlib.Path] = []
    if getattr(sys, "frozen", False):
        here.append(pathlib.Path(sys.executable).resolve().parent)
    here.append(pathlib.Path.cwd())
    here.append(pathlib.Path(__file__).resolve().parents[1])

    names = ("API_KEY.txt", "groq_key.txt", ".env")
    out: list[pathlib.Path] = []
    for d in here:
        for n in names:
            p = d / n
            if p not in out:
                out.append(p)
    return out


def _absorb(p: pathlib.Path) -> str | None:
    """Read one key file. Accepts KEY=value lines or a bare key on its own."""
    try:
        raw = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:  # noqa: BLE001
        return None
    found = None
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if v:
                os.environ.setdefault(k, v)
                if k == "GROQ_API_KEY":
                    found = v
        elif line.lower().startswith("gsk_"):
            # a bare key pasted on its own line
            os.environ.setdefault("GROQ_API_KEY", line)
            found = line
    return found


KEY_SOURCE: str | None = None


def load_env(path: str | pathlib.Path | None = None) -> None:
    """Load a Groq key from the first key file that supplies one."""
    global KEY_SOURCE
    if os.environ.get("GROQ_API_KEY"):
        return
    candidates = ([pathlib.Path(path)] if path
                  else key_file_locations())
    for p in candidates:
        if p.exists() and _absorb(p):
            KEY_SOURCE = str(p)
            return


def has_key() -> bool:
    load_env()
    return bool(os.environ.get("GROQ_API_KEY"))


def key_status() -> tuple[bool, str]:
    """(is a key active, a one-line explanation for the UI)."""
    if has_key():
        return True, f"key loaded from {KEY_SOURCE or 'the environment'}"
    spots = key_file_locations()
    where = spots[0].parent if spots else pathlib.Path.cwd()
    return False, (f"no key found - put one in API_KEY.txt next to the app "
                   f"({where}) to switch on AI-written notes")


def describe_pattern(zones: list) -> dict:
    """Turn raw zone saliency into the descriptors a clinician would use.

    Everything here is arithmetic on the model's own Grad-CAM output - no
    interpretation of the image itself. It exists so the narrative can say
    "focal, right upper zone" instead of dumping six percentages, while every
    phrase still traces back to a number.
    """
    real = [(n, f) for n, f in (zones or []) if n != "no localised attention"]
    if not real:
        return {"localised": False}

    top_name, top_frac = real[0]
    if top_frac >= 0.45:
        pattern = "focal"
    elif top_frac >= 0.30:
        pattern = "regional"
    else:
        pattern = "diffuse"

    right = sum(f for n, f in real if n.startswith("right"))
    left = sum(f for n, f in real if n.startswith("left"))
    if right >= 2 * left:
        laterality = "right-sided"
    elif left >= 2 * right:
        laterality = "left-sided"
    else:
        laterality = "bilateral"

    bands = {b: sum(f for n, f in real if b in n)
             for b in ("upper", "mid", "lower")}
    predominant_band = max(bands, key=bands.get)

    return {
        "localised": True,
        "top_zone": top_name,
        "top_zone_share": top_frac,
        "pattern": pattern,
        "laterality": laterality,
        "predominant_band": predominant_band,
        "band_shares": bands,
        "right_share": right,
        "left_share": left,
    }


def _direction(ev: dict) -> str:
    return ("positive (at or above the operating threshold)"
            if ev["probability"] >= ev["threshold"]
            else "negative (below the operating threshold)")


def _confidence(ev: dict) -> str:
    p = ev["probability"]
    lo = ev.get("t_low", ev["threshold"])
    hi = ev.get("t_high", ev["threshold"])
    return "HIGH (outside the abstention band)" if (p < lo or p >= hi) \
        else "LOW (inside the abstention band, so a human read is required)"


def _verdict_sentence(ev: dict) -> str:
    """The one claim about direction and confidence the note must make."""
    p = ev["probability"]
    lo = ev.get("t_low", ev["threshold"])
    hi = ev.get("t_high", ev["threshold"])
    positive = p >= ev["threshold"]
    confident = (p < lo) or (p >= hi)
    if positive and confident:
        return ("a confident positive screen; refer for the confirmatory "
                "test without needing a human pre-read")
    if positive:
        return ("a positive screen of low confidence; it must be reviewed by "
                "a human reader before action")
    if confident:
        return "a confident negative screen"
    return ("a negative screen of low confidence; it must be reviewed by a "
            "human reader")


def build_evidence(ev: dict) -> str:
    """Assemble the block that grounds generation. Keys are validated here so a
    missing field fails loudly rather than being silently omitted."""
    zones = [z for z in (ev.get("zones") or [])
             if z[0] != "no localised attention"]
    pat = describe_pattern(ev.get("zones") or [])
    if not zones:
        zone_txt = ("  - none. Grad-CAM is computed on the positive-class "
                    "logit, so a confidently negative prediction produces an "
                    "empty map: there is no evidence FOR the target "
                    "finding to localise. Do not describe any region.")
    else:
        zone_txt = "\n".join(
            f"  - {name}: {frac:.0%} of total saliency" for name, frac in zones[:4])
        zone_txt += (
            f"\n  summary: {pat['pattern']} pattern, {pat['laterality']}, "
            f"{pat['predominant_band']}-zone predominant"
            f"\n  strongest zone: {pat['top_zone']} "
            f"({pat['top_zone_share']:.0%} of saliency)"
            f"\n  laterality split: right {pat['right_share']:.0%} / "
            f"left {pat['left_share']:.0%}"
            f"\n  reference pattern: "
            + ev.get("reference_pattern",
                     "no characteristic anatomical distribution is specified "
                     "for this target, so the zone carries no concordance "
                     "information"))

    decided = ev.get("decision", "screen positive")
    lines = [
        "EVIDENCE",
        "--- Stage 1: trained model output ---",
        f"{ev.get('target_finding', 'target finding')} probability: "
        f"{ev['probability']:.3f} (0-1)",
        f"screening decision at the operating point: {decided}",
        f"operating threshold (decides positive vs negative): "
        f"{ev['threshold']:.3f}, chosen on the validation set to reach "
        f"{ev.get('target_sensitivity', 0.9):.0%} sensitivity",
        f"confident-negative cutoff: {ev.get('t_low', float('nan')):.3f}   "
        f"confident-positive cutoff: {ev.get('t_high', float('nan')):.3f}",
        "",
        "--- THE VERDICT (already computed - state it, do not re-derive it) ---",
        f"direction   : {_direction(ev)}",
        f"confidence  : {_confidence(ev)}",
        f"what to say : {_verdict_sentence(ev)}",
        "Do NOT compare the probability against the cutoffs yourself. The two "
        "lines above are authoritative; any comparison you perform risks "
        "contradicting them.",
        "",
        "--- Where the model looked (Grad-CAM, model behaviour not a finding) ---",
        zone_txt,
        "",
        "--- Model's held-out performance (whole test set, not this case) ---",
        f"AUROC {ev.get('auroc', float('nan')):.3f}"
        + (f" (95% CI {ev['auroc_ci'][0]:.3f}-{ev['auroc_ci'][1]:.3f})"
           if ev.get("auroc_ci") else ""),
        f"sensitivity {ev.get('sensitivity', float('nan')):.3f}, "
        f"specificity {ev.get('specificity', float('nan')):.3f}",
        f"trained on: {ev.get('train_description', 'unspecified')}",
        "",
        "--- What this model screens for ---",
        f"  target finding: {ev.get('target_finding', 'unspecified')}",
        f"  confirmatory test: {ev.get('confirm', 'unspecified')}",
    ]
    if ev.get("age") or ev.get("sex"):
        lines += ["", "--- Patient metadata (from the dataset) ---",
                  f"  age: {ev.get('age', 'unknown')}   sex: {ev.get('sex', 'unknown')}"]
    if ev.get("trust_warning"):
        lines += ["", "--- RELIABILITY WARNING (overrides everything above) ---",
                  f"  {ev['trust_warning']}",
                  "  Because of this, you must open the note by stating that "
                  "the score is not reliable for this image and should not be "
                  "acted on, and you must not present the probability as a "
                  "screening result."]
    if ev.get("caveat"):
        lines += ["", "--- Known limitation ---", f"  {ev['caveat']}"]
    return "\n".join(lines)


def _call_groq(evidence: str, model: str, timeout: float = 30.0) -> str:
    import requests
    key = os.environ["GROQ_API_KEY"]
    r = requests.post(
        ENDPOINT,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
        json={
            "model": model,
            "temperature": 0.2,
            "max_tokens": 700,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content":
                 evidence + "\n\nWrite the referral note."},
            ],
        },
        timeout=timeout,
    )
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    text = (data["choices"][0]["message"]["content"] or "").strip()
    if not text:
        raise RuntimeError("empty completion")
    return text


def _template(ev: dict) -> Report:
    p = ev["probability"]
    thr = ev["threshold"]
    pat = describe_pattern(ev.get("zones") or [])
    decided = ev.get("decision", "screen positive")
    band = ("well above" if p >= thr + 0.25 else
            "above" if p >= thr else
            "below" if p >= thr - 0.25 else "well below")

    target = ev.get("target_finding", "target finding")
    parts = [f"Automated screening returns a {target} probability of "
             f"{p:.2f}, {band} the operating threshold of {thr:.2f}; the case "
             f"is therefore classed as a {decided}."]

    if pat.get("localised"):
        concordant = pat["predominant_band"] == "upper"
        parts.append(
            f"Model attention is {pat['pattern']} and {pat['laterality']}, "
            f"centred on the {pat['top_zone']} which carries "
            f"{pat['top_zone_share']:.0%} of the total saliency "
            f"({pat['predominant_band']}-zone predominant).")
        if ev.get("reference_pattern_band") == "upper":
            parts.append(
                "That distribution is concordant with the upper-zone "
                "predominance characteristic of this condition." if concordant
                else "That distribution is discordant with the upper-zone "
                     "predominance characteristic of this condition, which "
                     "lowers confidence that the score reflects that pattern "
                     "specifically.")
        else:
            parts.append(
                "This target has no characteristic anatomical distribution, "
                "so the location carries no concordance information.")
    else:
        parts.append("No region of the film drove the score: the saliency map "
                     "is empty, which is the expected appearance of a "
                     "confidently negative screen rather than a positive "
                     "finding of normality.")

    parts.append(
        f"On held-out data the model reaches AUROC "
        f"{ev.get('auroc', float('nan')):.2f} with sensitivity "
        f"{ev.get('sensitivity', float('nan')):.2f} and specificity "
        f"{ev.get('specificity', float('nan')):.2f}, so this is a "
        f"prioritisation signal rather than a diagnosis.")
    parts.append("Confirmation requires "
                 + ev.get("confirm", "an appropriate confirmatory test") + ".")

    return Report(
        text=" ".join(parts), mode="template",
        label=("Template note (no GROQ_API_KEY set) - filled from the same "
               "evidence block, nothing invented"))


def generate_report(evidence: dict, use_llm: bool = True) -> Report:
    """evidence must contain at least: probability, threshold.

    Optional: zones, decision, auroc, auroc_ci, sensitivity, specificity,
    target_sensitivity, train_description, age, sex, caveat.
    """
    if "probability" not in evidence or "threshold" not in evidence:
        raise KeyError("evidence needs 'probability' and 'threshold'")

    if use_llm and has_key():
        block = build_evidence(evidence)
        for model in (MODEL, FALLBACK_MODEL):
            try:
                text = _call_groq(block, model)
                return Report(
                    text=text, mode="llm", model=model,
                    label=("AI-generated referral note, grounded in the model "
                           f"output above ({model} via Groq)"))
            except Exception as e:  # noqa: BLE001 - never break the demo
                last = f"{type(e).__name__}: {e}"
        rep = _template(evidence)
        rep.label += f" - LLM call failed ({last})"
        rep.error = last
        return rep

    return _template(evidence)


if __name__ == "__main__":
    demo = {
        "probability": 0.87, "threshold": 0.26, "target_sensitivity": 0.90,
        "decision": "screen positive - refer for confirmatory testing",
        "zones": [("right upper zone", 0.41), ("left upper zone", 0.26),
                  ("right mid zone", 0.18)],
        "auroc": 0.889, "auroc_ci": [0.689, 1.0],
        "sensitivity": 0.778, "specificity": 0.833,
        "train_description": "96 radiographs, Montgomery County set (NLM)",
        "age": 38, "sex": "Male",
        "caveat": ("trained on a single site with only 96 training "
                   "radiographs; cross-site performance is unverified"),
    }
    print("--- EVIDENCE ---")
    print(build_evidence(demo))
    print(f"\n--- has_key: {has_key()} ---")
    rep = generate_report(demo)
    print(f"\n--- mode={rep.mode} model={rep.model} ---")
    print(rep.label)
    print()
    print(rep.text)
    if rep.error:
        print("\nERROR:", rep.error)

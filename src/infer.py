"""Single-image inference: the path used by the app's upload tab.

Takes any chest radiograph file, applies the same preprocessing the model was
trained with (grayscale, CLAHE, 512px, then the 224px normalised tensor), and
returns the probability, the Grad-CAM overlay, and zone attention.

One caveat is enforced here rather than left to the caller: this model is a
BINARY TB screener. It was trained on "TB-consistent abnormality" versus
"normal" and has no category for any other pathology. Given a radiograph
showing pneumonia, a tumour or an effusion it cannot say so - it can only
report how much the film resembles the TB films it was trained on. The
`out_of_distribution_warning` field carries that forward so no UI can quietly
drop it.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import json

import cv2
import numpy as np
import torch

import data as D
from gradcam import GradCAM, overlay, zone_attention
from model import build_model

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

OOD_WARNING = (
    "This model is a binary TB screener trained on TB-consistent versus normal "
    "radiographs from two sites. It has no category for pneumonia, tumours, "
    "effusions, COVID or any other finding, and it cannot identify a cause. "
    "Any abnormality it has not been trained on will still be forced into one "
    "of its two buckets."
)


_GUARD_PATH = ROOT / "reports" / "trust_guard.json"


def _load_guard() -> dict:
    if _GUARD_PATH.exists():
        try:
            return json.loads(_GUARD_PATH.read_text())
        except Exception:  # noqa: BLE001
            pass
    return {}


GUARD = _load_guard()


def _hist(img: np.ndarray) -> np.ndarray:
    h = cv2.calcHist([img], [0], None, [32], [0, 256]).ravel()
    return h / max(h.sum(), 1.0)


def assess_trust(img512: np.ndarray, cam: np.ndarray) -> dict:
    """Is this film, and where the model looked on it, like the training data?

    Two measured signals, thresholded at the 95th percentile of
    in-distribution films (see scripts/calibrate_trust_guard.py):

      off_lung_saliency  saliency falling outside the lung box. High means the
                         score was driven by shoulders, neck, abdomen or the
                         image border rather than lung.
      histogram_distance chi-square distance to the mean training histogram.
                         High means unusual windowing, prior contrast
                         enhancement, inversion or heavy compression.
    """
    if not GUARD:
        return {"available": False}

    y0f, y1f = GUARD.get("lung_box_y", [0.12, 0.92])
    h = cam.shape[0]
    total = float(cam.sum())
    off = 0.0 if total <= 1e-8 else 1.0 - float(
        cam[int(y0f * h):int(y1f * h), :].sum()) / total

    ref = np.asarray(GUARD["reference_histogram"], dtype=float)
    cur = _hist(img512)
    dist = float(0.5 * np.sum((cur - ref) ** 2 / (cur + ref + 1e-10)))

    off_thr = float(GUARD["off_lung_saliency"]["p95"])
    dist_thr = float(GUARD["histogram_distance"]["p95"])
    reasons = []
    if off > off_thr:
        reasons.append(
            f"{off:.0%} of the model's attention fell OUTSIDE the lung fields "
            f"(typical films: {GUARD['off_lung_saliency']['mean']:.0%}, "
            f"95th percentile {off_thr:.0%}). The score was not driven by lung "
            f"tissue.")
    if dist > dist_thr:
        reasons.append(
            f"the image's intensity profile is unlike the training films "
            f"(distance {dist:.3f} vs 95th percentile {dist_thr:.3f}) - "
            f"different windowing, prior contrast processing, inversion or "
            f"compression.")

    return {"available": True, "off_lung_saliency": off,
            "off_lung_threshold": off_thr, "histogram_distance": dist,
            "histogram_threshold": dist_thr,
            "trustworthy": not reasons, "reasons": reasons}


@dataclass
class Prediction:
    probability: float
    decision: str
    threshold: float
    zones: list = field(default_factory=list)
    image512: np.ndarray | None = None
    overlay: np.ndarray | None = None
    trust: dict = field(default_factory=dict)
    out_of_distribution_warning: str = OOD_WARNING


class Screener:
    """Loads a checkpoint once and scores images on demand."""

    def __init__(self, split: str = "split", tag: str = "",
                 device: torch.device | None = None):
        ckpt = MODELS / f"tbscope_{split}{tag}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"no checkpoint at {ckpt}")
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(None, num_classes=1).to(self.device)
        self.model.load_state_dict(
            torch.load(ckpt, map_location=self.device)["state_dict"])
        self.model.eval()
        self.cam = GradCAM(self.model, self.device)

        import json
        mp = MODELS / f"metrics_{split}{tag}.json"
        m = json.loads(mp.read_text()) if mp.exists() else {}
        op = m.get("test", {}).get("operating_point", {})
        band = m.get("test", {}).get("abstention", {})
        self.threshold = float(op.get("threshold", 0.5))
        self.t_low = float(band.get("t_low", self.threshold))
        self.t_high = float(band.get("t_high", self.threshold))
        self.metrics = m

    # -- preprocessing identical to training -------------------------------
    @staticmethod
    def preprocess(img: np.ndarray, size: int = 512) -> np.ndarray:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255,
                                cv2.NORM_MINMAX).astype(np.uint8)
        img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    def decide(self, p: float) -> str:
        """Direction (from the operating threshold) plus confidence (from the
        abstention band). These are separate questions: a score can be above
        the screening threshold yet not confident enough to auto-refer."""
        positive = p >= self.threshold
        confident = (p < self.t_low) or (p >= self.t_high)
        if positive and confident:
            return "screen positive - refer for confirmatory testing"
        if positive:
            return "screen positive, low confidence - human read required"
        if confident:
            return "screen negative"
        return "screen negative, low confidence - human read required"

    def score_array(self, raw: np.ndarray,
                    already_preprocessed: bool = False) -> Prediction:
        if already_preprocessed:
            img512 = raw if raw.ndim == 2 else cv2.cvtColor(
                raw, cv2.COLOR_BGR2GRAY)
            if img512.shape[:2] != (512, 512):
                img512 = cv2.resize(img512, (512, 512),
                                    interpolation=cv2.INTER_AREA)
        else:
            img512 = self.preprocess(raw)
        x = D.to_tensor(img512).unsqueeze(0)
        prob, heat = self.cam(x)
        zones = zone_attention(heat)
        big = cv2.resize(heat, (img512.shape[1], img512.shape[0]),
                         interpolation=cv2.INTER_CUBIC)
        return Prediction(
            probability=float(prob), decision=self.decide(prob),
            threshold=self.threshold, zones=zones,
            image512=img512, overlay=overlay(img512, big),
            trust=assess_trust(img512, heat))

    def score_file(self, path: str | pathlib.Path,
                   preprocessed: bool = False) -> Prediction:
        """Score an image file.

        `preprocessed=True` means the file has ALREADY been through the
        training preprocessing (e.g. it came from data/processed/img512), so
        CLAHE must not be applied a second time. Applying it twice changes the
        image the model sees and shifts the score substantially - it was a real
        bug in the bundled demo films.
        """
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"could not read image: {path}")
        return self.score_array(img, already_preprocessed=preprocessed)

    def score_bytes(self, blob: bytes) -> Prediction:
        arr = np.frombuffer(blob, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("could not decode uploaded image")
        return self.score_array(img)

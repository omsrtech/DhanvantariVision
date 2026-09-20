"""Grad-CAM saliency for the TB screening model.

This exists for a specific reason, not for decoration. Chest X-ray classifiers
are notorious for shortcut learning: models have been shown to "detect"
pneumothorax by locating the chest drain placed to treat it, or to separate
classes using a hospital's burned-in text markers. A saliency map is how you
check whether the model is looking at lung parenchyma or at an artefact, and
it is the first thing a radiologist will ask to see.

Grad-CAM weights the final convolutional feature maps by the gradient of the
output logit with respect to those maps, giving a coarse (7x7 for a 224px
input) map of which regions drove the score. It is a diagnostic of the model,
not a segmentation of disease.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch


class GradCAM:
    """Grad-CAM on the final residual block of a ResNet."""

    def __init__(self, model, device: torch.device | None = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.eval()
        self._acts: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None
        # layer4 is the last conv block; its output is the standard CAM tap.
        target = self.model.layer4
        target.register_forward_hook(self._save_acts)
        target.register_full_backward_hook(self._save_grads)

    def _save_acts(self, _m, _i, out):
        self._acts = out.detach()

    def _save_grads(self, _m, _gi, gout):
        self._grads = gout[0].detach()

    def __call__(self, x: torch.Tensor) -> tuple[float, np.ndarray]:
        """x: (1,3,H,W) normalised tensor -> (probability, HxW map in [0,1])."""
        x = x.to(self.device)
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # Grad-CAM needs gradients, so autocast/no_grad must stay off here.
        self.model.zero_grad(set_to_none=True)
        logit = self.model(x)
        prob = float(torch.sigmoid(logit.float()).item())
        logit.sum().backward()

        if self._acts is None or self._grads is None:
            raise RuntimeError("hooks did not fire; is layer4 present?")

        # channel weights = spatially averaged gradients
        w = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self._acts).sum(dim=1, keepdim=True))
        cam = cam[0, 0].cpu().numpy()

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = cv2.resize(cam, (x.shape[-1], x.shape[-2]),
                         interpolation=cv2.INTER_CUBIC)
        return prob, np.clip(cam, 0.0, 1.0)


def overlay(gray: np.ndarray, cam: np.ndarray, alpha: float = 0.38) -> np.ndarray:
    """Blend a heatmap over a grayscale radiograph -> RGB uint8."""
    if gray.ndim == 2:
        base = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        base = gray
    if cam.shape[:2] != base.shape[:2]:
        cam = cv2.resize(cam, (base.shape[1], base.shape[0]))
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return np.clip((1 - alpha) * base + alpha * heat, 0, 255).astype(np.uint8)


# Anatomical zones of a frontal chest radiograph, as fractions of the image.
# Coarse by design: enough to say "right upper zone", which is where
# post-primary TB most often sits, without pretending to segment anatomy.
ZONES = {
    "right upper zone": (0.00, 0.50, 0.12, 0.40),
    "left upper zone": (0.50, 1.00, 0.12, 0.40),
    "right mid zone": (0.00, 0.50, 0.40, 0.65),
    "left mid zone": (0.50, 1.00, 0.40, 0.65),
    "right lower zone": (0.00, 0.50, 0.65, 0.92),
    "left lower zone": (0.50, 1.00, 0.65, 0.92),
}


NO_ATTENTION = "no localised attention"


def zone_attention(cam: np.ndarray) -> list[tuple[str, float]]:
    """Mean saliency per anatomical zone, descending.

    Note the radiological convention: the patient's RIGHT lung appears on the
    LEFT of a frontal image, which is why the x-ranges above look mirrored.

    Grad-CAM is computed on the positive-class logit and ReLU-clipped, so for a
    confidently NEGATIVE prediction the map is legitimately all zeros: there is
    no evidence *for* TB to localise. In that case every zone is reported as
    `NO_ATTENTION` rather than fabricating a ranking out of zeros, which would
    imply the model attended somewhere when it did not.
    """
    h, w = cam.shape[:2]
    out = []
    for name, (x0, x1, y0, y1) in ZONES.items():
        patch = cam[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
        out.append((name, float(patch.mean()) if patch.size else 0.0))

    total = sum(v for _, v in out)
    if total <= 1e-8:
        return [(NO_ATTENTION, 0.0) for _ in out]
    return sorted(((n, v / total) for n, v in out), key=lambda t: -t[1])

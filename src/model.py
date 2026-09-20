"""ResNet-18 defined explicitly, so the official ImageNet weights load without
torchvision as a dependency.

Module and parameter names mirror torchvision's `resnet18` exactly, which is
what lets `load_state_dict(..., strict=True)` succeed against
`resnet18-f37072fd.pth`. That strict load is the correctness check: if any name
or shape were wrong, it would raise rather than silently train from scratch.

Pretrained initialisation matters here. With only ~800 chest radiographs,
training from random weights badly underperforms fine-tuning from ImageNet.
"""
from __future__ import annotations

import pathlib

import torch
import torch.nn as nn


def conv3x3(cin: int, cout: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, cin: int, cout: int, stride: int = 1,
                 downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = conv3x3(cin, cout, stride)
        self.bn1 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(cout, cout)
        self.bn2 = nn.BatchNorm2d(cout)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes))
        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        layers += [BasicBlock(planes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Everything up to the final conv block - the Grad-CAM tap point."""
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)          # (B, 512, H/32, W/32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)


def build_model(weights_path: str | pathlib.Path | None = None,
                num_classes: int = 1) -> ResNet18:
    """ImageNet-pretrained ResNet-18 with a fresh `num_classes`-way head.

    num_classes=1 gives a single logit for binary screening, used with
    BCEWithLogitsLoss.
    """
    model = ResNet18(num_classes=1000)

    if weights_path is not None:
        p = pathlib.Path(weights_path)
        if not p.exists():
            raise FileNotFoundError(f"pretrained weights not found: {p}")
        state = torch.load(p, map_location="cpu")
        # strict=True: any name/shape mismatch is a hard error, not a silent
        # fall back to random initialisation.
        model.load_state_dict(state, strict=True)

    model.fc = nn.Linear(512, num_classes)
    nn.init.normal_(model.fc.weight, std=0.01)
    nn.init.zeros_(model.fc.bias)
    return model


if __name__ == "__main__":
    ROOT = pathlib.Path(__file__).resolve().parents[1]
    w = ROOT / "models" / "resnet18-f37072fd.pth"
    m = build_model(w if w.exists() else None)
    n = sum(p.numel() for p in m.parameters())
    print(f"parameters      : {n:,}")
    print(f"pretrained load : {'strict OK' if w.exists() else 'weights not yet downloaded'}")
    x = torch.randn(2, 3, 224, 224)
    print(f"forward         : {tuple(x.shape)} -> {tuple(m(x).shape)}")
    print(f"feature map     : {tuple(m.features(x).shape)}")

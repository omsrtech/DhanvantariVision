"""Dataset and augmentation for chest radiographs.

Two domain-specific choices worth stating, because both are easy to get wrong:

1. **No horizontal flipping.** It is the default augmentation in most vision
   pipelines and it is wrong for chest X-rays: the heart sits on the left and
   the liver on the right, so a mirrored radiograph is anatomically impossible
   and teaches the model that laterality carries no information. Situs inversus
   would be silently normalised away.

2. **No cropping at evaluation.** Random-resized-crop is standard on ImageNet,
   but a crop can remove the apical or costophrenic regions where TB findings
   concentrate. Training uses mild scale jitter; evaluation resizes the whole
   radiograph so no lung field is ever discarded.

Images are replicated to three channels and normalised with ImageNet statistics
because the backbone is ImageNet-pretrained.
"""
from __future__ import annotations

import pathlib

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT = 224


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mild, anatomy-preserving augmentation on a uint8 square image."""
    h, w = img.shape[:2]

    # rotation +/- 8 deg and translation +/- 4%, about the image centre
    ang = float(rng.uniform(-8, 8))
    tx, ty = rng.uniform(-0.04, 0.04, size=2) * w
    scale = float(rng.uniform(0.92, 1.08))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    # exposure jitter: radiographs vary a lot in penetration between sites
    alpha = float(rng.uniform(0.90, 1.10))     # contrast
    beta = float(rng.uniform(-12, 12))         # brightness
    img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return img


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """uint8 HxW -> normalised 3xINPUTxINPUT float tensor."""
    if img.shape[0] != INPUT or img.shape[1] != INPUT:
        img = cv2.resize(img, (INPUT, INPUT), interpolation=cv2.INTER_AREA)
    x = img.astype(np.float32) / 255.0
    x = np.stack([x, x, x], axis=-1)           # grayscale -> 3 channels
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class CXRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: pathlib.Path,
                 train: bool = False, seed: int = 0):
        self.df = df.reset_index(drop=True)
        self.img_dir = pathlib.Path(img_dir)
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        path = self.img_dir / f"{row.image_id}.png"
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        if self.train:
            img = augment(img, self.rng)
        return to_tensor(img), torch.tensor([float(row.label)])


def load_split(labels_csv: pathlib.Path, split_col: str = "split"):
    """Return (train_df, val_df, test_df) for the named split column."""
    df = pd.read_csv(labels_csv)
    if split_col not in df.columns:
        raise KeyError(f"{split_col!r} not in {list(df.columns)}")
    return (df[df[split_col] == "train"], df[df[split_col] == "val"],
            df[df[split_col] == "test"])


def make_loaders(labels_csv: pathlib.Path, img_dir: pathlib.Path,
                 split_col: str = "split", batch: int = 32,
                 workers: int = 0) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    tr, va, te = load_split(labels_csv, split_col)
    mk = lambda d, t: DataLoader(  # noqa: E731
        CXRDataset(d, img_dir, train=t), batch_size=batch, shuffle=t,
        num_workers=workers, pin_memory=True, drop_last=False)
    info = {
        "split_col": split_col,
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "train_pos": int(tr.label.sum()), "val_pos": int(va.label.sum()),
        "test_pos": int(te.label.sum()),
        "test_sources": sorted(te.source.unique().tolist()) if len(te) else [],
        "train_sources": sorted(tr.source.unique().tolist()) if len(tr) else [],
    }
    return mk(tr, True), mk(va, False), mk(te, False), info

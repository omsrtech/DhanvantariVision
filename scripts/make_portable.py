"""Assemble a trimmed payload for the standalone build.

The full project is ~1.7 GB of radiographs, because every training film has a
512px copy and a Grad-CAM overlay. A shareable build does not need them: the
app shows held-out cases by default, so only the test split, the demo films and
two checkpoints have to travel.

Written to build/payload/, which the PyInstaller spec then bundles.
"""
from __future__ import annotations

import pathlib
import shutil

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "payload"

KEEP_MODELS = [
    "tbscope_split.pt", "tbscope_split_nodule.pt",
    "metrics_split.json", "metrics_split_nodule.json",
]
KEEP_REPORTS = [
    "trust_guard.json", "calibration.json", "nih_discrimination.json",
    "resolution_ablation.json", "cross_condition.json", "other_pathology.json",
]
SRC_MODULES = ["data.py", "gradcam.py", "infer.py", "model.py", "plots.py",
               "report.py"]          # gate_ui/gatekeeper/policy are unused now


def copy(src: pathlib.Path, dst: pathlib.Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return src.stat().st_size


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    total = 0

    # ---- code -----------------------------------------------------------
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    # The payload ships only held-out films, so the toggle that would reveal
    # training images must not be offered - it would point at files that are
    # deliberately absent.
    app = app.replace(
        '''    only_test = st.toggle(
        "Held-out cases only", value=True,
        help="ON is the honest view: the model never saw these during "
             "training. OFF includes training images, where scores look "
             "far better than real performance.")''',
        '''    only_test = True   # portable build ships held-out films only
    st.caption("Showing held-out cases only: films the model never saw "
               "during training.")''')
    (OUT / "app.py").write_text(app, encoding="utf-8")
    total += len(app.encode())

    for m in SRC_MODULES:
        total += copy(ROOT / "src" / m, OUT / "src" / m)

    cfg = ROOT / ".streamlit" / "config.toml"
    if cfg.exists():
        total += copy(cfg, OUT / ".streamlit" / "config.toml")

    # ---- models and reports --------------------------------------------
    for m in KEEP_MODELS:
        p = ROOT / "models" / m
        if p.exists():
            total += copy(p, OUT / "models" / m)
    for r in KEEP_REPORTS:
        p = ROOT / "reports" / r
        if p.exists():
            total += copy(p, OUT / "reports" / r)

    # ---- held-out films only -------------------------------------------
    for data_dir in ("processed", "processed_nodule"):
        src = ROOT / "data" / data_dir
        if not (src / "predictions.csv").exists():
            continue
        total += copy(src / "predictions.csv", OUT / "data" / data_dir / "predictions.csv")
        if (src / "labels.csv").exists():
            total += copy(src / "labels.csv", OUT / "data" / data_dir / "labels.csv")

        df = pd.read_csv(src / "predictions.csv")
        keep = df[df.split == "test"].image_id.tolist()
        n = 0
        for image_id in keep:
            for sub in ("img512", "cam"):
                f = src / sub / f"{image_id}.png"
                if f.exists():
                    total += copy(f, OUT / "data" / data_dir / sub / f"{image_id}.png")
                    n += 1
        print(f"  {data_dir}: {len(keep)} held-out films, {n} images")

    # ---- demo films (scored live, so originals must travel) -------------
    demo = ROOT / "data" / "demo"
    if demo.exists():
        n = 0
        for f in demo.rglob("*.png"):
            total += copy(f, OUT / "data" / "demo" / f.relative_to(demo))
            n += 1
        print(f"  demo: {n} films")

    files = sum(1 for _ in OUT.rglob("*") if _.is_file())
    print(f"\npayload: {files} files, {total / 1e6:.0f} MB -> {OUT}")


if __name__ == "__main__":
    main()

"""Assemble a folder that can be dragged straight into a GitHub repo.

What goes in: all the code, the two trained checkpoints, every measured result,
the figures, the deck, and a small set of demo radiographs downscaled so the
app is usable the moment someone clones it.

What stays out: the ~1.7 GB of processed radiographs and parquet shards, the
build and dist directories, the virtualenvs, and anything holding a secret.
Scripts to regenerate the excluded data ship with it, so the repo stays small
without becoming unreproducible.

GitHub rejects any file over 100 MB and warns above 50 MB; every file here is
checked against that before it is copied.
"""
from __future__ import annotations

import pathlib
import shutil

import cv2

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "github_release"

HARD_LIMIT = 100 * 1024 * 1024
WARN_LIMIT = 50 * 1024 * 1024

# (source, destination) - directories are copied wholesale
CODE = [
    "app.py", "launcher.py", "requirements.txt", "run.cmd",
    "README.md", "PITCH.md", "READ_ME_FIRST.txt",
    "DhanvantariVision.spec", "DhanvantariVision_pitch.pptx",
]
SRC_MODULES = ["data.py", "gradcam.py", "infer.py", "model.py", "plots.py",
               "report.py"]
SCRIPTS = [
    "fetch_data.py", "fetch_shenzhen_subset.py", "fetch_nodule_shards.py",
    "fetch_demo_xrays.py", "fetch_nodule_demo.py", "prepare.py",
    "prepare_nodule.py", "train.py", "predict_all.py",
    "calibration_analysis.py", "nih_discrimination.py",
    "resolution_ablation.py", "test_other_pathology.py",
    "cross_condition_test.py", "calibrate_trust_guard.py",
    "relabel_decisions.py", "make_figures.py", "make_deck.py",
    "make_portable.py", "package_release.py", "make_github_folder.py",
]
# Only the two checkpoints the app actually loads.
MODELS = ["tbscope_split.pt", "tbscope_split_nodule.pt",
          "metrics_split.json", "metrics_split_nodule.json",
          "metrics_cross_site.json"]

GITIGNORE = """\
# data: regenerate with the scripts in scripts/ (see README)
data/raw/
data/nih/
data/hf/

# the held-out films and Grad-CAMs that ship here are deliberately committed;
# the training and validation images are not, so re-running prepare.py must
# not sweep them in
data/processed/img512/*
data/processed/cam/*
data/processed_nodule/img512/*
data/processed_nodule/cam/*
!data/*/img512/.gitkeep

# secrets - never commit a real key
.env
API_KEY.txt
.streamlit/secrets.toml
.streamlit/secrets.toml.disabled

# build artefacts
build/
dist/
build_venv/
.venv/
*.spec.bak
__pycache__/
*.pyc

# large extra checkpoints
models/tbscope_cross_site.pt
models/tbscope_split_montgomery.pt
models/resnet18-f37072fd.pth
"""


def copy(src: pathlib.Path, dst: pathlib.Path) -> int:
    if not src.exists():
        return 0
    size = src.stat().st_size
    if size > HARD_LIMIT:
        print(f"  SKIP (over GitHub's 100 MB limit): {src.name} "
              f"{size / 1e6:.0f} MB")
        return 0
    if size > WARN_LIMIT:
        print(f"  large ({size / 1e6:.0f} MB, GitHub will warn): {src.name}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return size


# How many held-out films to ship per condition. TB gets all 120; the nodule
# set has 415 and would triple the repo, so it gets a slice stratified by
# decision state, which is what the worklist is filtered and sorted by.
WORKLIST_CAP = {"processed": 999, "processed_nodule": 130}
FILM_PX = 448


def _shrink(src: pathlib.Path, dst: pathlib.Path, px: int) -> int:
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        return 0
    h, w = img.shape[:2]
    if max(h, w) > px:
        s = px / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)),
                         interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return dst.stat().st_size


def ship_worklist(out: pathlib.Path) -> int:
    """Ship the held-out films, their Grad-CAMs, and a matching predictions
    table.

    The training and validation rows are dropped: the worklist is a
    demonstration of performance on unseen data, and shipping the images the
    model was fitted on would be both misleading and three times the size.
    The predictions table is trimmed to exactly the films included, so no row
    in the app points at a missing image.
    """
    import pandas as pd

    total = 0
    for d, cap in WORKLIST_CAP.items():
        src = ROOT / "data" / d
        pred = src / "predictions.csv"
        if not pred.exists():
            continue
        df = pd.read_csv(pred)
        test = df[df["split"] == "test"].copy()
        if len(test) > cap:
            # Take the same proportion out of every decision state, highest
            # scores first, so the interesting cases survive the cut. Done
            # with a within-group rank rather than groupby.apply, which in
            # pandas 2 consumes the grouping column and would silently drop
            # `decision` from the shipped table.
            frac = cap / len(test)
            test = test.sort_values("probability", ascending=False)
            rank = test.groupby("decision").cumcount()
            quota = test["decision"].map(
                test["decision"].value_counts()
                    .map(lambda n: max(1, round(n * frac))))
            test = test[rank < quota]
        kept = 0
        for iid in test["image_id"]:
            a = _shrink(src / "img512" / f"{iid}.png",
                        out / "data" / d / "img512" / f"{iid}.png", FILM_PX)
            b = _shrink(src / "cam" / f"{iid}.png",
                        out / "data" / d / "cam" / f"{iid}.png", FILM_PX)
            total += a + b
            kept += 1

        dst = out / "data" / d / "predictions.csv"
        dst.parent.mkdir(parents=True, exist_ok=True)
        test.to_csv(dst, index=False)
        total += dst.stat().st_size
        total += copy(src / "labels.csv", out / "data" / d / "labels.csv")
        print(f"  {d}: {kept} held-out films (of {len(df)} total rows)")
    return total


QUICKSTART = r"""
---

## Run it (5 minutes, no training, no GPU)

The trained weights and the held-out radiographs are committed, so the app
works straight after a clone.

```bash
git clone <this-repo>
cd DhanvantariVision
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
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
| `data/processed*/` | the **held-out** films and Grad-CAMs only, 448 px |
| `data/demo/` | 24 unseen radiographs to drop into *Analyse a Scan* |
| `reports/` | every metric and figure quoted below |
| `scripts/` | fetch -> prepare -> train -> evaluate, end to end |

The training and validation images are *not* committed - that is ~1.7 GB, and
the point of the worklist is unseen data. `scripts/fetch_*.py` and
`scripts/prepare.py` rebuild them from the public sources.

---
"""


def patch_readme(out: pathlib.Path) -> None:
    """Make the README true for someone who just cloned this.

    The working copy's README is written for the machine it was built on: it
    names a hard-coded CUDA interpreter and assumes the full data tree is
    present. Neither holds after a clone, so both are replaced with a
    quickstart that matches what is actually in the repo.
    """
    import re
    p = out / "README.md"
    if not p.exists():
        return
    s = p.read_text(encoding="utf-8")
    s = re.sub(r"Run everything with the CUDA interpreter: `[^`]*`\.",
               "Training wants a CUDA GPU (this was trained on an RTX 3050, "
               "6 GB). Running the app does not - it scores on CPU in about "
               "a second a film.", s, count=1)
    marker = "\n---\n\n## The headline result"
    if marker in s:
        s = s.replace(marker, QUICKSTART + "\n## The headline result", 1)
    p.write_text(s, encoding="utf-8")



def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    total = 0

    for name in CODE:
        total += copy(ROOT / name, OUT / name)
    for m in SRC_MODULES:
        total += copy(ROOT / "src" / m, OUT / "src" / m)
    for s in SCRIPTS:
        total += copy(ROOT / "scripts" / s, OUT / "scripts" / s)
    for m in MODELS:
        total += copy(ROOT / "models" / m, OUT / "models" / m)

    # The key template is committed under a different name: API_KEY.txt
    # itself is gitignored, so nobody can push a real key by accident.
    total += copy(ROOT / "API_KEY.txt", OUT / "API_KEY.example.txt")

    for r in (ROOT / "reports").glob("*.json"):
        total += copy(r, OUT / "reports" / r.name)
    for f in (ROOT / "reports" / "figures").glob("*.png"):
        total += copy(f, OUT / "reports" / "figures" / f.name)

    cfg = ROOT / ".streamlit" / "config.toml"
    total += copy(cfg, OUT / ".streamlit" / "config.toml")

    total += ship_worklist(OUT)

    # demo radiographs, downscaled: the originals are ~2 MB each, and 512px is
    # what the model sees anyway
    n_demo = 0
    demo = ROOT / "data" / "demo"
    if demo.exists():
        for f in demo.rglob("*.png"):
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            h, w = img.shape[:2]
            if max(h, w) > 1024:
                scale = 1024 / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_AREA)
            dst = OUT / "data" / "demo" / f.relative_to(demo)
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), img)
            total += dst.stat().st_size
            n_demo += 1

    (OUT / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    patch_readme(OUT)

    files = [f for f in OUT.rglob("*") if f.is_file()]
    biggest = sorted(files, key=lambda f: -f.stat().st_size)[:5]
    print(f"\n{len(files)} files, {total / 1e6:.0f} MB -> {OUT}")
    print(f"demo radiographs downscaled: {n_demo}")
    print("\nlargest files:")
    for f in biggest:
        print(f"  {f.stat().st_size / 1e6:6.1f} MB  {f.relative_to(OUT)}")

    leaks = []
    for f in files:
        if f.suffix in (".py", ".txt", ".md", ".toml", ".json", ".cmd"):
            try:
                t = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            import re
            if re.search(r"gsk_[A-Za-z0-9]{20,}", t):
                leaks.append(f)
    print(f"\nsecret scan: {'LEAK -> ' + str(leaks) if leaks else 'clean'}")


if __name__ == "__main__":
    main()

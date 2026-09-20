"""Download the trained weights from this repo's GitHub Release.

The two checkpoints are ~45 MB each. They are attached to a release rather
than committed, so cloning the repo stays fast and the history does not carry
90 MB of binary that changes wholesale every time the model is retrained.

    python scripts/fetch_models.py

Set RELEASE_URL below to your own repo before publishing.
"""
from __future__ import annotations

import os
import pathlib
import sys
import urllib.request

# e.g. https://github.com/<you>/DhanvantariVision/releases/download/v1.0
RELEASE_URL = os.environ.get(
    "DV_RELEASE_URL",
    "https://github.com/OWNER/REPO/releases/download/v1.0",
)

MODELS = ["tbscope_split.pt", "tbscope_split_nodule.pt"]
DEST = pathlib.Path(__file__).resolve().parents[1] / "models"


def _progress(done: int, block: int, total: int) -> None:
    if total <= 0:
        return
    pct = min(100, 100 * done * block // total)
    print(f"\r  {pct:3d}%", end="", flush=True)


def main() -> int:
    if "OWNER/REPO" in RELEASE_URL:
        print("Set DV_RELEASE_URL (or edit RELEASE_URL in this file) to point "
              "at your release.", file=sys.stderr)
        return 1

    DEST.mkdir(parents=True, exist_ok=True)
    for name in MODELS:
        out = DEST / name
        if out.exists():
            print(f"{name}: already here ({out.stat().st_size / 1e6:.0f} MB)")
            continue
        url = f"{RELEASE_URL}/{name}"
        print(f"{name}: downloading from {url}")
        tmp = out.with_suffix(".part")
        try:
            urllib.request.urlretrieve(url, tmp, _progress)
        except Exception as exc:                          # noqa: BLE001
            tmp.unlink(missing_ok=True)
            print(f"\n  failed: {exc}", file=sys.stderr)
            return 1
        # Rename only once the download finished, so an interrupted run never
        # leaves a truncated file that torch.load would fail on cryptically.
        tmp.replace(out)
        print(f"\r  done ({out.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

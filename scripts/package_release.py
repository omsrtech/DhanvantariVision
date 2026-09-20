"""Zip the built app into something shareable.

Produces dist/DhanvantariVision-windows.zip containing the one-folder build
plus the read-me. One-folder rather than one-file so it starts in seconds
instead of re-extracting a gigabyte on every launch.
"""
from __future__ import annotations

import pathlib
import shutil
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD = ROOT / "dist" / "DhanvantariVision"
OUT = ROOT / "dist" / "DhanvantariVision-windows.zip"


def main() -> None:
    if not BUILD.exists():
        raise SystemExit(f"nothing to package: {BUILD} missing")

    # These two sit NEXT TO the exe, not inside the bundle: the bundle is
    # extracted to a temp directory on every launch, so anything the user is
    # meant to edit has to live beside the executable instead.
    for name in ("READ_ME_FIRST.txt", "API_KEY.txt"):
        src = ROOT / name
        if src.exists():
            shutil.copy2(src, BUILD / name)

    files = [f for f in BUILD.rglob("*") if f.is_file()]
    raw = sum(f.stat().st_size for f in files)
    print(f"packing {len(files)} files, {raw / 1e6:.0f} MB uncompressed...")

    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in files:
            z.write(f, pathlib.Path("DhanvantariVision") /
                    f.relative_to(BUILD))

    size = OUT.stat().st_size
    print(f"\nwrote {OUT}")
    print(f"  {size / 1e6:.0f} MB zipped  ({100 * size / raw:.0f}% of raw)")
    print("\nShare this zip. The recipient unzips it and double-clicks "
          "DhanvantariVision.exe - no Python, no install, no internet.")


if __name__ == "__main__":
    main()

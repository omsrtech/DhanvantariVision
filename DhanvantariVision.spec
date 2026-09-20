# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the standalone Dhanvantari Vision build.

One-folder, not one-file: a one-file build re-extracts ~1 GB to a temp
directory on every launch, which takes a minute and looks broken. One-folder
starts in seconds and zips just as well for sharing.
"""
import pathlib

from PyInstaller.utils.hooks import collect_all, collect_data_files, copy_metadata

ROOT = pathlib.Path(SPECPATH)
PAYLOAD = ROOT / "build" / "payload"

datas = []
binaries = []
hiddenimports = []

# The app itself: payload contents land at the root of the extraction dir, so
# app.py's `ROOT = Path(__file__).parent` resolves to the bundle.
for f in PAYLOAD.rglob("*"):
    if f.is_file():
        datas.append((str(f), str(f.parent.relative_to(PAYLOAD))))

# Streamlit ships its frontend build and a config schema as package data, and
# checks its own version through importlib.metadata at import time.
for pkg in ("streamlit",):
    a, b, c = collect_all(pkg)
    datas += a
    binaries += b
    hiddenimports += c

for pkg in ("streamlit", "altair", "pandas", "numpy", "torch", "sklearn",
            "scikit-learn", "matplotlib", "PIL", "pyarrow", "packaging",
            "requests", "tornado", "click", "rich", "pydeck", "watchdog",
            "opencv-python-headless", "protobuf", "tenacity", "toml",
            "typing_extensions", "gitpython", "jsonschema", "blinker",
            "cachetools", "narwhals"):
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass

datas += collect_data_files("matplotlib")
datas += collect_data_files("sklearn")

hiddenimports += [
    "streamlit.web.bootstrap", "streamlit.runtime.scriptrunner.magic_funcs",
    "sklearn.metrics", "sklearn.utils._typedefs", "sklearn.utils._heap",
    "sklearn.utils._sorting", "sklearn.utils._vector_sentinel",
    "matplotlib.backends.backend_agg", "PIL._tkinter_finder",
    "pandas._libs.tslibs.base", "cv2",
]

a = Analysis(
    ["launcher.py"],
    pathex=[str(ROOT), str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Trim weight that a screening demo never touches.
    excludes=["tkinter", "PyQt5", "PyQt6", "PySide2", "PySide6", "wx",
              "notebook", "jupyter", "IPython", "pytest", "torchvision",
              "torchaudio", "tensorboard", "sympy.plotting", "scipy.io.matlab"],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DhanvantariVision",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,          # keep the console: it shows the URL and any error
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="DhanvantariVision",
)

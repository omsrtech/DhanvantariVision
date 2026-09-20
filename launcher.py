"""Entry point for the standalone Dhanvantari Vision build.

Starts the Streamlit server in-process and opens a browser at it, so the user
double-clicks one file and gets the app. No Python install required.

Three things this has to get right under PyInstaller:

  * Paths. When frozen, the payload is extracted to sys._MEIPASS, and app.py
    derives every data path from its own location, so the app is launched from
    there rather than from the working directory.
  * The file watcher. Streamlit's module watcher walks sys.modules and trips
    over torch's custom class registry, so it is switched off - pointless in a
    frozen build anyway, since nothing on disk can change.
  * A free port. The default 8501 is often taken; we find an open one rather
    than failing with a stack trace in front of a teacher.
"""
from __future__ import annotations

import os
import pathlib
import socket
import sys
import threading
import time
import webbrowser


def base_dir() -> pathlib.Path:
    """Where the payload lives: the PyInstaller temp dir, or this folder."""
    if getattr(sys, "frozen", False):
        return pathlib.Path(getattr(sys, "_MEIPASS", pathlib.Path.cwd()))
    return pathlib.Path(__file__).resolve().parent / "build" / "payload"


def free_port(preferred: int = 8501) -> int:
    for port in (preferred, 8502, 8503, 8510, 8520):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    with socket.socket() as s:            # let the OS choose
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def open_when_ready(url: str, port: int, timeout: float = 90.0) -> None:
    """Poll until the server accepts connections, then open the browser."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                webbrowser.open(url)
                return
        time.sleep(0.4)


def _fix_streamlit_static() -> None:
    """Point Streamlit at its bundled frontend.

    Streamlit finds its HTML/JS with
    `dirname(streamlit/file_util.py) + "/static"`. Under PyInstaller that
    module lives in the compiled archive, so the computed path does not
    resolve and the server mounts no static route at all - it answers
    /_stcore/health fine and returns 404 for every page, which looks like the
    app is broken rather than misconfigured.

    Rather than fight the packager, locate the bundled frontend and override
    the lookup.
    """
    if not getattr(sys, "frozen", False):
        return
    try:
        import streamlit.file_util as file_util
    except Exception:                                   # noqa: BLE001
        return

    meipass = pathlib.Path(getattr(sys, "_MEIPASS", ""))
    candidates = [
        meipass / "streamlit" / "static",
        pathlib.Path(sys.executable).parent / "_internal" / "streamlit" / "static",
    ]
    for c in candidates:
        if (c / "index.html").is_file():
            file_util.get_static_dir = lambda _p=str(c): _p    # type: ignore[assignment]
            return
    print("WARNING: Streamlit frontend not found; the page may not load.",
          file=sys.stderr)


def main() -> int:
    base = base_dir()
    app = base / "app.py"
    if not app.exists():
        print(f"ERROR: app.py not found at {app}", file=sys.stderr)
        input("Press Enter to close...")
        return 1

    port = free_port()
    url = f"http://localhost:{port}"

    # Config via environment: read by Streamlit before anything else loads.
    os.environ.setdefault("STREAMLIT_SERVER_PORT", str(port))
    os.environ.setdefault("STREAMLIT_SERVER_ADDRESS", "127.0.0.1")
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    # torch's class registry breaks Streamlit's module watcher; nothing can
    # change on disk in a frozen build, so turn the watcher off entirely.
    os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
    os.environ.setdefault("STREAMLIT_GLOBAL_DEVELOPMENT_MODE", "false")
    os.environ.setdefault("STREAMLIT_CONFIG_DIR", str(base / ".streamlit"))

    sys.path.insert(0, str(base))
    sys.path.insert(0, str(base / "src"))

    _fix_streamlit_static()

    print("=" * 62)
    print("  Dhanvantari Vision - chest radiograph screening triage")
    print("  Made by Team Akatsuki | OMSR Technologies | omsrtech.com")
    print("=" * 62)
    print(f"\n  Starting... the app will open at {url}")
    print("  Close this window to stop the app.\n")

    threading.Thread(target=open_when_ready, args=(url, port),
                     daemon=True).start()

    from streamlit.web import bootstrap
    bootstrap.run(str(app), False, [], {})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:                        # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"\nThe app failed to start: {exc}")
        input("Press Enter to close...")
        raise SystemExit(1)

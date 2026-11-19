"""NUScope-TOMO desktop app: the Studio in its own window.

    nuscope-tomo app [project_dir]        # or the installed "NUScope-TOMO" app

The Studio server runs inside the app on 127.0.0.1, on a free port, and the
window is the system web view (WebKit on macOS, Edge WebView2 on Windows). With
no project given, a start page offers recent projects and a folder picker;
File > Open Project switches at any time. Without pywebview installed, the
Studio opens in the default browser instead.

Created by Roberto dos Reis (@rmsreis), NUANCE Center, Northwestern University.
"""
from __future__ import annotations

import html
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from . import __version__

HOME = Path(os.environ.get("NUSCOPE_TOMO_HOME", Path.home() / ".nuscope-tomo")).expanduser()
CONFIG = HOME / "app.json"
MAX_RECENT = 8


# ------------------------------------------------------------------ recent projects
def recent() -> list[str]:
    """Recent project folders that still exist, newest first."""
    try:
        items = json.loads(CONFIG.read_text()).get("recent", [])
    except (OSError, ValueError):
        return []
    return [p for p in items if isinstance(p, str) and Path(p).is_dir()][:MAX_RECENT]


def remember(project: str | Path) -> None:
    p = str(Path(project).resolve())
    items = [p, *[q for q in recent() if q != p]][:MAX_RECENT]
    HOME.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps({"recent": items}, indent=2))


# ------------------------------------------------------------------ the Studio server
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Studio:
    """One Studio server at a time, in a background thread."""

    def __init__(self):
        self.server = None
        self.thread: Optional[threading.Thread] = None
        self.project: Optional[Path] = None
        self.url: Optional[str] = None

    def open(self, project: str | Path, timeout: float = 30) -> str:
        import uvicorn

        from .studio.server import create_app

        project = Path(project).expanduser().resolve()
        if not project.is_dir():
            raise FileNotFoundError(f"Not a folder: {project}")
        self.close()
        port = free_port()
        config = uvicorn.Config(create_app(project), host="127.0.0.1", port=port, log_level="warning",
                                log_config=None)          # no console in a windowed app
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, name="studio", daemon=True)
        self.thread.start()
        end = time.time() + timeout
        while not self.server.started:
            if not self.thread.is_alive() or time.time() > end:
                raise RuntimeError("The Studio server did not start; see ~/.nuscope-tomo/app.log")
            time.sleep(0.05)
        self.project, self.url = project, f"http://127.0.0.1:{port}/"
        remember(project)
        return self.url

    def close(self) -> None:
        if self.server is not None:
            self.server.should_exit = True
            if self.thread is not None:
                self.thread.join(timeout=10)
        self.server = self.thread = self.url = None


# ------------------------------------------------------------------ the start page
def start_page(items: list[str], error: str = "") -> str:
    rows = "".join(
        f'<button class="recent" onclick="pywebview.api.open({html.escape(json.dumps(p), quote=True)})">'
        f'<b>{html.escape(Path(p).name)}</b><small>{html.escape(p)}</small></button>' for p in items) \
        or '<p class="quiet">No recent projects yet.</p>'
    err = f'<p class="error">{html.escape(error)}</p>' if error else ""
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>NUScope-TOMO</title><style>
:root {{ color-scheme: dark; }}
body {{ margin: 0; height: 100vh; display: grid; place-items: center; background: #0d1017; color: #e6e8ee;
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
main {{ width: min(560px, 90vw); }}
h1 {{ font-size: 26px; margin: 0; }} h1 span {{ color: #a78bfa; }}
.sub {{ color: #8a93a6; margin: 4px 0 28px; }}
.primary {{ all: unset; cursor: pointer; background: #0e7490; color: #fff; padding: 10px 18px; border-radius: 8px;
  font-weight: 600; }} .primary:hover {{ background: #0f86a8; }}
h2 {{ font-size: 12px; letter-spacing: .08em; text-transform: uppercase; color: #8a93a6; margin: 30px 0 8px; }}
.recent {{ all: unset; cursor: pointer; display: block; box-sizing: border-box; width: 100%; padding: 9px 12px;
  border-radius: 8px; border: 1px solid #1f2533; margin-bottom: 6px; }}
.recent:hover {{ background: #161b26; }} .recent b {{ display: block; }}
.recent small {{ color: #8a93a6; display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.quiet {{ color: #8a93a6; }} .error {{ color: #f87171; }}
footer {{ margin-top: 34px; color: #5b6478; font-size: 12px; }}
</style></head><body><main>
<h1>NUScope-TOMO <span>Studio</span></h1>
<p class="sub">FIB-SEM tomography: reconstruction, segmentation, quantification and EDS chemistry · v{__version__}</p>
<button class="primary" onclick="pywebview.api.choose()">Open project folder…</button>
{err}<h2>Recent</h2>{rows}
<footer>Created by Roberto dos Reis (@rmsreis) · NUANCE Center, Northwestern University</footer>
</main></body></html>"""


def loading_page(project: str | Path) -> str:
    name = html.escape(Path(project).name or str(project))
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
:root {{ color-scheme: dark; }} body {{ margin: 0; height: 100vh; display: grid; place-items: center; background: #0d1017;
  color: #e6e8ee; font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; text-align: center; }}
.ring {{ width: 34px; height: 34px; margin: 0 auto 16px; border: 3px solid #1f2533; border-top-color: #22d3ee;
  border-radius: 50%; animation: spin 0.9s linear infinite; }} @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
small {{ color: #8a93a6; display: block; margin-top: 6px; }}
</style></head><body><div><div class="ring"></div>Opening <b>{name}</b>…
<small>Indexing volumes and tables. The first time, macOS may ask to allow access to this folder.</small></div></body></html>"""


class Api:
    """Called from the start page and the menu."""

    def __init__(self, studio: Studio):
        self.studio = studio
        self.window = None

    def open(self, project: str) -> None:
        self.window.load_html(loading_page(project))
        try:
            url = self.studio.open(project)
        except Exception as e:                    # back to the start page, with the reason
            self.window.load_html(start_page(recent(), f"Could not open {project}: {e}"))
            return
        self.window.set_title(f"NUScope-TOMO · {Path(project).name}")
        self.window.load_url(url)

    def choose(self) -> None:
        import webview
        dialog = getattr(webview, "FileDialog", None)
        kind = dialog.FOLDER if dialog else webview.FOLDER_DIALOG
        picked = self.window.create_file_dialog(kind)
        if picked:
            self.open(picked[0] if isinstance(picked, (list, tuple)) else picked)

    def home(self) -> None:
        self.studio.close()
        self.window.set_title("NUScope-TOMO")
        self.window.load_html(start_page(recent()))


# ------------------------------------------------------------------ entry point
def _quiet_streams() -> None:
    """A windowed app has no console: send output to ~/.nuscope-tomo/app.log."""
    if sys.stdout is None or sys.stderr is None or getattr(sys, "frozen", False):
        HOME.mkdir(parents=True, exist_ok=True)
        log = open(HOME / "app.log", "a", buffering=1, encoding="utf-8")
        sys.stdout = sys.stdout or log
        sys.stderr = log


def main(argv: Optional[list[str]] = None) -> int:
    import multiprocessing
    multiprocessing.freeze_support()
    _quiet_streams()
    args = [a for a in (sys.argv[1:] if argv is None else argv) if not a.startswith("-psn_")]  # macOS Finder adds -psn_
    if args and args[0] == "--self-test":
        return self_test(args[1] if len(args) > 1 else None)
    project = args[0] if args else None
    studio = Studio()
    try:
        import webview
    except ImportError:
        return _browser(studio, project)

    api = Api(studio)
    # The window comes up at once; the project opens in the background, so a large folder (or a
    # macOS folder-access prompt) never leaves the app without a window.
    window = webview.create_window("NUScope-TOMO", html=loading_page(project) if project else start_page(recent()),
                                   js_api=api, width=1440, height=920, min_size=(960, 640))
    api.window = window

    def boot():
        if project:
            api.open(project)
    menu = []
    try:
        from webview.menu import Menu, MenuAction, MenuSeparator
        menu = [Menu("File", [MenuAction("Open Project…", api.choose), MenuAction("Start Page", api.home),
                              MenuSeparator(), MenuAction("Reload", lambda: window.evaluate_js("location.reload()"))])]
    except Exception:
        pass
    try:
        webview.start(boot, menu=menu) if menu else webview.start(boot)
    finally:
        studio.close()
    return 0


def self_test(project: Optional[str] = None) -> int:
    """Start the bundled Studio without a window, call its API, read a slice. For builds and CI.

    Prints one JSON line and returns 0 when everything answered.
    """
    import tempfile
    import urllib.request

    import numpy as np

    from .core.volume import Volume, VoxelSize

    if project is None:
        project = Path(tempfile.mkdtemp(prefix="nuscope-tomo-selftest-"))
        Volume(np.arange(4 * 16 * 16, dtype=np.uint16).reshape(4, 16, 16), VoxelSize(20, 10, 10)).to_zarr(project / "raw.zarr")
    studio = Studio()
    report: dict = {"version": __version__, "frozen": bool(getattr(sys, "frozen", False))}
    try:
        url = studio.open(project)
        get = lambda path: urllib.request.urlopen(url + path, timeout=30).read()  # noqa: E731
        info = json.loads(get("api/project"))
        ops = json.loads(get("api/ops"))
        vid = next(iter(info.get("volumes") or {}), None) if isinstance(info.get("volumes"), dict) else \
            (info.get("volumes") or [None])[0]
        vid = vid.get("id") if isinstance(vid, dict) else vid
        png = get(f"api/volume/{vid}/slice?axis=0&index=0") if vid else b""
        report.update(ok=bool(ops) and (not vid or png[:4] == b"\x89PNG"), project=info.get("name"),
                      ops=len(ops), volume=vid, slice_bytes=len(png), index=len(get("").decode()))
    except Exception as e:
        report.update(ok=False, error=f"{type(e).__name__}: {e}")
    finally:
        studio.close()
    line = json.dumps(report)
    print(line)
    (HOME / "self-test.json").parent.mkdir(parents=True, exist_ok=True)
    (HOME / "self-test.json").write_text(line)
    return 0 if report.get("ok") else 1


def _browser(studio: Studio, project: Optional[str]) -> int:
    import webbrowser
    project = project or (recent() or ["."])[0]
    url = studio.open(project)
    print(f"NUScope-TOMO Studio → {url}   (project: {studio.project})\nInstall pywebview for a window of its own. Ctrl+C stops.")
    webbrowser.open(url)
    try:
        while studio.thread and studio.thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        studio.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

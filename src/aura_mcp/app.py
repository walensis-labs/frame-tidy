"""Frame Tidy: the desktop launcher.

Starts the local review page on a free port and shows it in its own window
(pywebview), or in the default browser when no window is available. Closing
the window stops the server, and the phone listener if it was turned on.

Order matters: the data folder is put in ``AURA_MCP_HOME`` *before* anything
imports ``aura_mcp.server``, which reads it once at import time.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path

APP_NAME = "Frame Tidy"
HOST = "127.0.0.1"


def default_data_dir(platform: str, env: Mapping[str, str], home: Path) -> Path:
    """Where the app keeps its photos and settings."""
    if env.get("AURA_MCP_HOME"):
        return Path(env["AURA_MCP_HOME"]).expanduser()
    if platform == "darwin":
        return home / "Library" / "Application Support" / APP_NAME
    if platform.startswith("win"):
        appdata = env.get("APPDATA")
        base = Path(appdata) if appdata else home / "AppData" / "Roaming"
        return base / APP_NAME
    return home / ".local" / "share" / APP_NAME


def free_port(preferred: int = 0) -> int:
    """A port nothing is listening on (the preferred one if it is free)."""
    for candidate in (preferred, 0) if preferred else (0,):
        with socket.socket() as s:
            try:
                s.bind((HOST, candidate))
            except OSError:
                continue
            return s.getsockname()[1]
    raise OSError("no free port")


def configure_environment() -> Path:
    """Choose and create the data folder and export it. Call before importing the server."""
    root = default_data_dir(sys.platform, os.environ, Path.home())
    root.mkdir(parents=True, exist_ok=True)
    os.environ["AURA_MCP_HOME"] = str(root)
    return root


def _quiet_streams(root: Path) -> None:
    """A double-clicked app has no terminal; send stray output to a log file."""
    if sys.stdout is None or sys.stderr is None:
        log = open(root / "frame-tidy.log", "a", buffering=1)  # noqa: SIM115
        sys.stdout = sys.stdout or log
        sys.stderr = sys.stderr or log


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=f"{APP_NAME} desktop app")
    parser.add_argument("--port", type=int, default=0, help="port to use (default: any free one)")
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="serve only and print the address (for testing); stop with Ctrl-C",
    )
    parser.add_argument("--browser", action="store_true", help="use the default browser, not a window")
    args = parser.parse_args(argv)

    root = configure_environment()
    _quiet_streams(root)
    _upgrade_data(root, window=not (args.no_window or args.browser))

    import uvicorn

    from aura_mcp.review import Review  # after AURA_MCP_HOME is set

    port = free_port(args.port)
    url = f"http://{HOST}:{port}/"
    review = Review(root)
    server = uvicorn.Server(
        uvicorn.Config(review.app(), host=HOST, port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    if not server.started:
        raise SystemExit("The page server did not start.")
    print(f"Serving {url} (data in {root})", flush=True)

    try:
        if args.no_window:
            _wait(thread)
        else:
            shown = False if args.browser else _show_window(url)
            if not shown:
                import webbrowser

                webbrowser.open(url)
                _wait(thread)
    finally:
        # "Use on my phone" opens a second listener on the wifi: it closes
        # with the app, before the page server itself.
        review.stop_lan()
        # A removal or put-back under way finishes the photo it is on first.
        review.finish_work()
        server.should_exit = True
        thread.join(timeout=10)


def _upgrade_data(root: Path, window: bool) -> None:
    """Bring the data folder to this version's format, or stop with a plain reason."""
    from aura_mcp import migrations

    try:
        result = migrations.upgrade(root)
    except migrations.UpgradeFailed as exc:
        print(str(exc), file=sys.stderr, flush=True)
        if window:
            _show_problem(str(exc))
        raise SystemExit(1) from exc
    if result["backup"] is not None:
        print(f"Updated saved data to format {result['to']}; "
              f"the old files are in {result['backup']}", flush=True)


def _show_problem(text: str) -> None:
    """A window that says why the app cannot start (there is no terminal to read)."""
    import html

    try:
        import webview

        body = (
            "<body style='font:15px -apple-system,system-ui,sans-serif;padding:28px;line-height:1.5'>"
            f"<h2 style='margin-top:0'>{APP_NAME} can't open your data</h2>"
            f"<p>{html.escape(text)}</p></body>"
        )
        webview.create_window(APP_NAME, html=body, width=560, height=300)
        webview.start()
    except Exception:  # no GUI available: the message was printed already
        pass


def _wait(thread: threading.Thread) -> None:
    try:
        while thread.is_alive():
            thread.join(0.5)
    except KeyboardInterrupt:
        pass


def _show_window(url: str) -> bool:
    """Open our own window and block until it is closed. False if that is not possible."""
    try:
        import webview

        webview.create_window(APP_NAME, url, width=1280, height=860, min_size=(800, 600))
        webview.start()
        return True
    except Exception as exc:  # no GUI toolkit available, headless session, ...
        print(f"Could not open a window ({exc}); using the browser.", file=sys.stderr)
        return False


if __name__ == "__main__":
    main()

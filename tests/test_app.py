"""Pure logic of the desktop launcher: no window, no network."""

import os
import socket
import sys
from pathlib import Path

from aura_mcp import app


def test_data_dir_macos():
    d = app.default_data_dir("darwin", {}, Path("/Users/x"))
    assert d == Path("/Users/x/Library/Application Support/Frame Tidy")


def test_data_dir_windows_uses_appdata():
    d = app.default_data_dir("win32", {"APPDATA": "/appdata"}, Path("/h"))
    assert d == Path("/appdata") / "Frame Tidy"


def test_data_dir_windows_without_appdata_falls_back():
    d = app.default_data_dir("win32", {}, Path("/h"))
    assert d == Path("/h/AppData/Roaming/Frame Tidy")


def test_data_dir_linux():
    d = app.default_data_dir("linux", {}, Path("/home/x"))
    assert d == Path("/home/x/.local/share/Frame Tidy")


def test_override_wins_on_every_platform():
    for plat in ("darwin", "win32", "linux"):
        d = app.default_data_dir(plat, {"AURA_MCP_HOME": "/tmp/mine"}, Path("/h"))
        assert d == Path("/tmp/mine")


def test_free_port_is_bindable_and_in_range():
    port = app.free_port()
    assert 1024 < port < 65536
    with socket.socket() as s:
        s.bind(("127.0.0.1", port))


def test_free_port_prefers_requested_when_free():
    free = app.free_port()
    assert app.free_port(free) == free


def test_free_port_skips_busy_requested():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        busy = s.getsockname()[1]
        assert app.free_port(busy) != busy


def test_configure_sets_home_before_server_is_imported(tmp_path, monkeypatch):
    monkeypatch.delenv("AURA_MCP_HOME", raising=False)
    monkeypatch.setattr(app, "default_data_dir", lambda *a, **k: tmp_path / "data")
    for name in ("aura_mcp.server", "aura_mcp.review"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    root = app.configure_environment()
    assert root == tmp_path / "data" and root.is_dir()
    assert os.environ["AURA_MCP_HOME"] == str(root)
    assert "aura_mcp.server" not in sys.modules
    assert "aura_mcp.review" not in sys.modules


def test_the_app_will_not_open_data_from_a_newer_version(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AURA_MCP_HOME", str(tmp_path))
    (tmp_path / "people").mkdir()
    (tmp_path / "data-version.json").write_text('{"version": 999}')
    started = []
    monkeypatch.setattr(app, "free_port", lambda *a: started.append("server") or 0)
    try:
        app.main(["--no-window"])
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("expected the app to stop")
    assert started == []  # stopped before the page server
    assert "newer version of Frame Tidy" in capsys.readouterr().err


def test_the_app_records_the_data_format_of_a_new_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_MCP_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(app, "free_port", lambda *a: (_ for _ in ()).throw(SystemExit(0)))
    try:
        app.main(["--no-window"])
    except SystemExit:
        pass
    assert (tmp_path / "data" / "data-version.json").is_file()

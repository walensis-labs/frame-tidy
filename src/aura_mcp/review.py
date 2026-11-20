"""Local review page: sign in, get the photos, pick a person, remove their photos.

Run ``uv run aura-review`` and open the printed address. It talks to Aura in
three places only, each started by the user on the page: signing in, getting
a frame's photos, and removing photos after the confirmation dialog. What you
do there is also written to places the MCP tools read:

    selections/{frame_id}.json    photos you ticked (selections/all.json in "All frames")
    references/                   face crops you saved as "this is the person"
    people/                       everyone found in the photos, and their names

It listens on 127.0.0.1 only, so the photos are not visible to other machines,
until the user presses "Use on my phone": that opens a second listener on this
computer's wifi address which answers only a phone that opened the one-time
pairing link, and only for the routes in ``LAN_ALLOWED``.
"""

from __future__ import annotations

import argparse
import asyncio
import html
import ipaddress
import json
import os
import re
import secrets
import socket
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.routing import Match, Route

from . import people, people_index
from . import server as store
from .app import APP_NAME
from .aura_client import AuraAuthError, AuraError
from .face_matcher import MODEL_DOWNLOAD_BYTES, FaceMatcher, FaceMatcherError
from . import image_search
from .image_search import ImageSearcher, ImageSearchError

THUMB_SHORT_SIDE = 360
PAGE = Path(__file__).with_name("review.html")


def _fail(status: int, message: str, detail: str | None = None) -> JSONResponse:
    body = {"error": message}
    if detail:
        body["detail"] = detail  # the technical text, shown folded away on the page
    return JSONResponse(body, status_code=status)


def _explained(status: int, exc: BaseException) -> JSONResponse:
    plain = store.explain_error(exc)
    return _fail(status, plain["message"], plain["detail"])


MODEL_NEEDED = "The face model is needed first. Open Frames to download it."

# ------------------------------------------------------------ update notice

RELEASES_URL = "https://api.github.com/repos/walensis-labs/frame-tidy/releases/latest"
RELEASES_PAGE = "https://github.com/walensis-labs/frame-tidy/releases"
UPDATE_CHECK_ENV = "AURA_NO_UPDATE_CHECK"  # set to anything to never look
UPDATE_EVERY = 24 * 60 * 60  # seconds; a failed look is not retried sooner either


def app_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("aura-mcp")
    except PackageNotFoundError:
        return "0.0.0"


def latest_release_tag(transport: Any = None) -> str | None:
    """The newest release's tag on GitHub. Sends nothing but the request itself:
    no account, no cookies, no identifiers. Raises on any failure."""
    import httpx

    with httpx.Client(
        transport=transport,
        timeout=5.0,
        follow_redirects=True,
        headers={"User-Agent": "frame-tidy-update-check", "Accept": "application/vnd.github+json"},
    ) as http:
        response = http.get(RELEASES_URL)
        response.raise_for_status()
        tag = response.json().get("tag_name")
    return str(tag) if tag else None


def _version_key(text: str | None) -> tuple[int, ...] | None:
    parts = (text or "").strip().lstrip("vV").split(".")
    try:
        return tuple(int(part) for part in parts)
    except ValueError:
        return None


# What the page may do to Aura. Deleting from the whole account is left to
# the MCP tool: it has never been run against the real service.
PAGE_SCOPES = ("frame", "all_frames")

# The pretend frame that shows every loaded frame at once. It is handled here
# and never becomes a folder: nothing is read from or written to cache/all,
# flagged/all or removed/all, and no real frame may be loaded under this id.
ALL = "all"
ALL_NAME = "All frames"

# The names this computer answers to. A request addressed to any other name
# reached here through a website whose address was re-pointed at 127.0.0.1
# (DNS rebinding), and would otherwise pass for a same-origin request.
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}


def _is_local(host: str | None) -> bool:
    name = (host or "").strip().lower()
    if name.startswith("["):  # [::1]:8766
        name = name[: name.find("]") + 1]
    else:
        name = name.split(":", 1)[0]
    return name in LOCAL_HOSTS


# ------------------------------------------------------- phone on the wifi
#
# A request is from "the wifi" when it arrived on any listener other than the
# one on 127.0.0.1 (``scope["server"]`` is the address of the socket that
# accepted it; nothing the client sends can change it). Such a request is
# answered only when all of these hold, else 403:
#   - the phone listener is on, and the request came in on exactly its address;
#   - its Host header is exactly "<that address>:<port>" (DNS rebinding);
#   - it carries a pairing cookie handed out by /pair (checked in constant time),
#     from the same device address that paired. This does not stop someone who
#     can take over that address on the same network; it only raises the bar;
#   - its method and route are in LAN_ALLOWED. Anything not listed, including
#     every route added later, is refused until someone adds it here.

PHONE_COOKIE = "frame_tidy_phone"
PAIR_TTL = 10 * 60  # seconds a pairing link stays usable
LOOPBACK_SERVERS = {"127.0.0.1", "::1"}

LAN_ALLOWED = frozenset({
    ("GET", "/"),
    ("GET", "/api/frames"),
    ("GET", "/api/frames/{frame_id}/photos"),
    ("PUT", "/api/frames/{frame_id}/selection"),
    ("GET", "/api/frames/{frame_id}/photos/{asset_id}/faces"),
    ("POST", "/api/frames/{frame_id}/feedback"),
    ("POST", "/api/frames/{frame_id}/search"),
    ("POST", "/api/frames/{frame_id}/search-text"),
    ("GET", "/api/sync/status"),
    ("GET", "/api/setup"),
    ("GET", "/api/frames/{frame_id}/removed"),
    ("GET", "/removed-thumb/{frame_id}/{asset_id}.jpg"),
    ("POST", "/api/restore"),
    ("GET", "/api/restore/status"),
    ("GET", "/api/version"),
    ("POST", "/api/frames/{frame_id}/removal/preview"),
    ("POST", "/api/removal"),
    ("GET", "/api/removal/status"),
    ("GET", "/api/people"),
    ("POST", "/api/people/build"),
    ("GET", "/api/people/status"),
    ("POST", "/api/people/merge"),
    ("POST", "/api/people/unmerge"),
    ("POST", "/api/people/not-same"),
    ("GET", "/api/people/{person_id}/suggestions"),
    ("PATCH", "/api/people/{person_id}"),
    ("GET", "/people/{person_id}/avatar.jpg"),
    ("GET", "/api/people/{person_id}/faces"),
    ("POST", "/api/people/move-faces"),
    ("GET", "/face/{asset_id}/{index:int}.jpg"),
    ("GET", "/api/face/{asset_id}/{index:int}"),
    ("POST", "/api/references"),
    ("DELETE", "/api/references/{ref_id}"),
    ("GET", "/reference/{ref_id}.jpg"),
    ("GET", "/thumb/{frame_id}/{asset_id}.jpg"),
    ("GET", "/photo/{frame_id}/{asset_id}.jpg"),
})

NO_NETWORK = (
    "This computer does not seem to be on a home network right now. "
    "Connect it to your wifi and try again."
)
NOT_PAIRED = "This device is not connected. Scan the code shown on the computer again."
COMPUTER_ONLY = "This can only be done on the computer."


# The private IPv4 ranges home and office networks use (RFC 1918). Nothing
# else: not link-local, not carrier-grade NAT, not test or reserved ranges.
HOME_NETWORKS = tuple(
    ipaddress.ip_network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)


def usable_lan_ip(ip: str) -> bool:
    """An IPv4 address on a home or office network (never a public one)."""
    try:
        address = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return address.version == 4 and any(address in net for net in HOME_NETWORKS)


def find_lan_address() -> str | None:
    """This computer's address on the local network, or None.

    Asks the system which of its addresses it would use to reach an outside
    address. "Connecting" a UDP socket only picks a route: no packet is sent.
    The address used (192.0.2.1) is reserved for documentation and leads
    nowhere even if something were sent.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))
            ip = probe.getsockname()[0]
    except OSError:
        return None
    return ip if usable_lan_ip(ip) else None


def _client_ip(request: Request) -> str | None:
    """The address of the device that sent the request (from the socket)."""
    client = request.scope.get("client")
    return str(client[0]) if client else None


# What a phone is never shown: where things are kept on this computer. The
# phone only needs to know that a copy exists; where it is, is shown on the
# computer.
PATH_FIELDS = frozenset({"local_path", "path", "flagged_path", "local_copy", "cache_dir", "flagged_dir"})
ON_THE_COMPUTER = "the place shown on the computer"
_PATH_IN_TEXT = re.compile(r"(?<![\w.:/])(?:/|[A-Za-z]:\\)(?:[^\s'\"<>]*[^\s'\"<>.,;:)])?")


def hide_paths(value: Any, roots: list[str], key: str | None = None) -> Any:
    """``value`` with every file-system path taken out, for a phone.

    Fields that hold a path become ``ON_THE_COMPUTER`` (or stay empty); in any
    other text, the data folder (whose name may hold spaces) and then any
    absolute path is replaced by the same words.
    """
    if isinstance(value, dict):
        return {k: hide_paths(v, roots, k) for k, v in value.items()}
    if isinstance(value, list):
        return [hide_paths(v, roots) for v in value]
    if isinstance(value, str):
        if key in PATH_FIELDS:
            return ON_THE_COMPUTER if value else value
        for root in roots:  # a folder name with spaces in it becomes one word first
            value = value.replace(root, "/data-folder")
        return _PATH_IN_TEXT.sub(ON_THE_COMPUTER, value)
    return value


class PhoneError(Exception):
    def __init__(self, message: str, detail: str | None = None):
        super().__init__(message)
        self.message, self.detail = message, detail


def _lan_allowed(routes: list[Any], scope: dict[str, Any]) -> bool:
    """Whether the route this request would reach is in LAN_ALLOWED.

    Picks the route the way the router does (the first full match). No match,
    or a match not listed, is a no.
    """
    method = "GET" if scope.get("method") == "HEAD" else scope.get("method")
    for route in routes:
        try:
            match, _ = route.matches(scope)
        except Exception:  # noqa: BLE001 - anything odd is a no
            return False
        if match == Match.FULL:
            return (method, getattr(route, "path", None)) in LAN_ALLOWED
    return False


def qr_svg(text: str) -> str:
    """A QR code for ``text`` as inline SVG, drawn here (nothing is sent anywhere)."""
    import segno

    return segno.make(text, error="m", micro=False).svg_inline(
        scale=1, border=2, dark="#000", light="#fff", omitsize=True,
    )


# Shown once on the phone after pairing; it moves on to the page itself. The
# move is made by this page, not a redirect, so the browser counts it as the
# same site and sends the new (SameSite=Strict) cookie.
PAIRED_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer"><title>Connected</title></head>
<body style="font: 17px/1.5 -apple-system, system-ui, sans-serif; padding: 24px">
<p>This phone is connected. Opening the page…</p>
<p><a href="/">Open it</a></p>
<script>location.replace('/');</script>
</body></html>
"""


class LanListener:
    """The second server, on the wifi address, for the same page."""

    def __init__(self, app: Any, host: str, port: int = 0):
        import uvicorn

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self._sock.bind((host, port))
            self._sock.listen(64)
        except OSError:
            self._sock.close()
            raise
        self.port = self._sock.getsockname()[1]
        # Requests are never logged: the pairing link holds its secret in the
        # address. proxy_headers off: nothing is in front of this server.
        config = uvicorn.Config(
            app, log_level="warning", access_log=False, lifespan="off",
            proxy_headers=False, timeout_graceful_shutdown=3,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run, kwargs={"sockets": [self._sock]}, daemon=True,
        )
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started and self._thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not self._server.started:
            self.stop()
            raise OSError("the listener did not start")

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(10)
        self._sock.close()


class Job:
    """One long piece of work at a time, run off the event loop and polled.

    The page starts it, then asks for ``snapshot()`` until the state is
    "done" (with the work's result) or "error" (with its message).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {"state": "idle", "done": 0, "total": 0, "message": ""}
        self.thread: threading.Thread | None = None
        # Set when the app is closing: work that checks it stops at the end of
        # the photo it is on (see wind_down).
        self.stop = threading.Event()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def _set(self, **fields: Any) -> None:
        with self._lock:
            self._state = {**self._state, **fields}

    @property
    def running(self) -> bool:
        return self.snapshot()["state"] == "running"

    def start(self, work: Callable[[store.Progress], Awaitable[dict[str, Any]]]) -> bool:
        """Run ``work(progress)``; False if the last run has not finished."""
        with self._lock:
            if self._state["state"] == "running":
                return False
            self._state = {
                "state": "running", "done": 0, "total": 0, "message": "Starting",
                "started_at": time.time(),
            }  # replaces the last run's result and detail
            self.stop.clear()

        async def progress(done: float, total: float | None, message: str | None) -> None:
            self._set(done=int(done), total=int(total or 0), message=message or "")

        def run() -> None:
            try:
                result = asyncio.run(work(progress))
            except Exception as exc:  # noqa: BLE001 - shown to the user on the page
                plain = store.explain_error(exc)
                self._set(state="error", message=plain["message"], detail=plain["detail"] or None)
            else:
                self._set(state="done", result=result)

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        return True

    def wind_down(self, timeout: float) -> bool:
        """Ask the work to stop after the photo it is on, and wait for it.

        Returns False if it was still running when ``timeout`` ran out.
        """
        thread = self.thread
        if thread is None or not thread.is_alive():
            return True
        self.stop.set()
        thread.join(timeout)
        return not thread.is_alive()


# How long closing the app waits for a removal or a put-back to finish the
# photo it is on. Generous: one photo's original can take a while to fetch.
WIND_DOWN_SECONDS = 300


class Review:
    def __init__(
        self,
        root: Path,
        matcher: FaceMatcher | None = None,
        client_factory: Callable[..., Any] | None = None,
        update_check: Callable[[], str | None] | None = None,
        lan_address: Callable[[], str | None] | None = None,
        lan_listener: Callable[[Any, str, int], Any] | None = None,
        searcher: ImageSearcher | None = None,
    ):
        self.root = Path(root)
        # The page never fetches the face model by itself: only after the
        # user agrees (see api_model_download).
        self.matcher = matcher or FaceMatcher(download=False)
        # Nor the photo search model (see api_search_model_download).
        self.searcher = searcher or ImageSearcher(download=False)
        # A new Aura client per job: each job runs in its own event loop, and
        # a client's connections cannot be carried from one loop to the next.
        # Called with no arguments it uses the saved sign-in; with an email
        # and password, that login.
        self.client_factory = client_factory or (
            lambda email=None, password=None: store.new_client(self.root, email, password)
        )
        self.removal = Job()
        self.sync = Job()
        self.restore = Job()
        self.model_job = Job()
        self.search_model_job = Job()
        self.update_check = update_check or latest_release_tag
        self._update_lock = threading.Lock()
        self._update_thread: threading.Thread | None = None
        # Loading photos, finding people and removing each rewrite a frame's
        # index and face cache from a copy taken at the start, so only one of
        # them may run at a time.
        self._start_lock = threading.Lock()
        # The removal the user is looking at in the confirmation dialog.
        self._plan: dict[str, Any] | None = None
        self._thumb_lock = threading.Lock()
        # The "Find people" job: one at a time, polled by the page.
        self._build_lock = threading.Lock()
        self._build: dict[str, Any] = {"state": "idle", "done": 0, "total": 0, "message": ""}
        self._build_thread: threading.Thread | None = None
        # Using the page from a phone: off until the user turns it on.
        self.find_lan_address = lan_address or find_lan_address
        self.lan_listener = lan_listener or LanListener
        self._lan_lock = threading.Lock()
        self._lan: dict[str, Any] | None = None
        self._asgi: Starlette | None = None

    # ------------------------------------------------------------- data

    def frames(self) -> list[dict[str, Any]]:
        """The loaded frames, led by "All frames" when there are two or more."""
        frames = self._loaded()
        if len(frames) > 1:
            return [{"id": ALL, "name": ALL_NAME, "photos": len(self._holders())}, *frames]
        return frames

    def _loaded(self) -> list[dict[str, Any]]:
        """The real frames with photos here, in a fixed order (by id)."""
        cache = self.root / "cache"
        names = store._read_json(cache / "frames.json", {})
        frames = []
        for folder in sorted(p for p in cache.iterdir() if p.is_dir()) if cache.is_dir() else []:
            if folder.name == ALL:
                continue  # reserved; never a real frame
            count = sum(1 for _ in folder.glob("*.jpg"))
            if count:
                frames.append(
                    {"id": folder.name, "name": names.get(folder.name) or folder.name, "photos": count}
                )
        return frames

    def _loaded_ids(self) -> list[str]:
        return [frame["id"] for frame in self._loaded()]

    def _holders(self) -> dict[str, list[str]]:
        """``{asset_id: the loaded frames holding it}``, frames in loaded order.

        The same photo has the same id on every frame, so this is how "All
        frames" shows each photo once and knows where its file is.
        """
        held: dict[str, list[str]] = {}
        for frame in self._loaded_ids():
            for path in store.cache_dir(self.root, frame).glob("*.jpg"):
                held.setdefault(path.stem, []).append(frame)
        return held

    def _present(self, frame_id: str) -> set[str]:
        """The photos that can be shown and ticked in this view."""
        if frame_id == ALL:
            return set(self._holders())
        return {p.stem for p in store.cache_dir(self.root, frame_id).glob("*.jpg")}

    def _source(self, frame_id: str, asset_id: str) -> tuple[str, Path] | None:
        """The real frame and file for a photo; in "all", the first frame holding it."""
        for frame in self._loaded_ids() if frame_id == ALL else [frame_id]:
            path = store.cache_dir(self.root, frame) / f"{asset_id}.jpg"
            if path.is_file():
                return frame, path
        return None

    def photos(self, frame_id: str) -> dict[str, Any]:
        if frame_id == ALL:
            return self._photos_all()
        folder = store.cache_dir(self.root, frame_id)
        index = store._read_json(folder / "index.json", None)
        flagged = self.flagged(frame_id)
        scores = {m["asset_id"]: m["similarity"] for m in flagged["matches"]}
        answers = people.labels(self.root, flagged["person"], frame_id) if flagged["person"] else {}
        files = {p.stem: p for p in folder.glob("*.jpg")}
        photos = []
        for asset_id, path in files.items():
            taken = (index or {}).get(asset_id, {}).get("taken_at")
            photos.append(
                {
                    "id": asset_id,
                    "taken_at": taken,
                    # Without an index (a sync still running) fall back to
                    # download order, which follows Aura's newest-first listing.
                    "_order": taken or "",
                    "_mtime": path.stat().st_mtime,
                    "similarity": scores.get(asset_id),
                    "label": answers.get(asset_id) if asset_id in scores else None,
                }
            )
        if index:
            photos.sort(key=lambda p: p["_order"], reverse=True)
        else:
            photos.sort(key=lambda p: p["_mtime"])
        for photo in photos:
            del photo["_order"], photo["_mtime"]
        return {
            "frame_id": frame_id,
            "photos": photos,
            "selection": [a for a in self.selection(frame_id) if a in files],
            "sync_complete": index is not None,
            "flagged": {
                "person": flagged["person"],
                "person_name": flagged["person_name"],
                "similarity_threshold": flagged["similarity_threshold"],
                "review_mode": flagged["review_mode"],
            },
        }

    def _photos_all(self) -> dict[str, Any]:
        """Every loaded frame's photos, one entry per photo, each with its frames."""
        flagged, per_frame = self._flagged_all()
        person = flagged["person"]
        held = self._holders()
        answers = {
            frame: people.labels(self.root, person, frame) for frame in per_frame
        } if person else {}
        taken: dict[str, str | None] = {}
        complete = True
        for frame in self._loaded_ids():
            index = store._read_json(store.cache_dir(self.root, frame) / "index.json", None)
            if not isinstance(index, dict):
                complete = False
                continue
            for asset_id, entry in index.items():
                if taken.get(asset_id) is None and asset_id in held:
                    taken[asset_id] = (entry or {}).get("taken_at")
        photos = []
        for asset_id, frames in held.items():
            score = max(
                (m[asset_id]["similarity"] for m in per_frame.values() if asset_id in m),
                default=None,
            )
            label = None
            if score is not None and person:
                said = [answers[f].get(asset_id) for f in per_frame if asset_id in per_frame[f]]
                # Answers are written on every frame at once; should two frames
                # ever disagree, "not them" wins, as it keeps the photo.
                label = people.NO if people.NO in said else next((a for a in said if a), None)
            photos.append({
                "id": asset_id,
                "taken_at": taken.get(asset_id),
                "frames": frames,
                "similarity": score,
                "label": label,
            })
        photos.sort(key=lambda p: (p["taken_at"] or "", p["id"]), reverse=True)
        present = set(held)
        return {
            "frame_id": ALL,
            "photos": photos,
            "selection": [a for a in self.selection(ALL) if a in present],
            "sync_complete": complete,
            "flagged": {key: flagged[key] for key in (
                "person", "person_name", "similarity_threshold", "review_mode")},
        }

    def _flagged_all(self) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
        """The last person search seen across the frames, and its matches per frame.

        Opening someone in "All frames" searches every frame, so their last
        searches agree. Should they not (a frame searched on its own since),
        the first frame's person is shown and only frames searched for that
        person are merged.
        """
        found = [(frame, self.flagged(frame)) for frame in self._loaded_ids()]
        searched = [(frame, data) for frame, data in found if data["matches"]]
        lead = next((data for _, data in found if data["person"]), None)
        lead = lead or (searched[0][1] if searched else None)
        if lead is None:
            return self.flagged(None), {}
        per_frame = {
            frame: {m["asset_id"]: m for m in data["matches"]}
            for frame, data in found
            if data["person"] == lead["person"] and data["matches"]
        }
        best: dict[str, dict[str, Any]] = {}
        for matches in per_frame.values():
            for asset_id, match in matches.items():
                if asset_id not in best or match["similarity"] > best[asset_id]["similarity"]:
                    best[asset_id] = match
        return {**lead, "matches": sorted(
            best.values(), key=lambda m: m["similarity"], reverse=True)}, per_frame

    def flagged(self, frame_id: str | None) -> dict[str, Any]:
        """The last search on this frame: who it was for and what it matched."""
        if frame_id == ALL:
            return self._flagged_all()[0]
        data = {} if frame_id is None else store._read_json(store.flagged_dir(self.root, frame_id) / "matches.json", {})
        if isinstance(data, list):  # written by an older version
            data = {"matches": data}
        return {
            "person": data.get("person"),
            "person_name": data.get("person_name"),
            "similarity_threshold": data.get("similarity_threshold"),
            "review_mode": "opt_out" if data.get("review_mode") == "opt_out" else "opt_in",
            "matches": data.get("matches", []),
        }

    def selection(self, frame_id: str) -> list[str]:
        # "All frames" keeps its own selection, in selections/all.json.
        return store._read_json(store.selection_path(self.root, frame_id), {}).get("asset_ids", [])

    def references(self) -> list[dict[str, Any]]:
        return people.references(self.root)

    def _photo_path(self, request: Request) -> tuple[str, Path] | None:
        """The real frame and file asked for; see _source."""
        try:
            frame_id = store._check_id(request.path_params["frame_id"], "frame_id")
            asset_id = store._check_id(request.path_params["asset_id"], "asset_id")
        except ValueError:
            return None
        return self._source(frame_id, asset_id)

    # --------------------------------------------------------- handlers

    def page(self, request: Request) -> Response:
        return HTMLResponse(PAGE.read_text().replace("{{APP_NAME}}", html.escape(APP_NAME)))

    def api_frames(self, request: Request) -> Response:
        return JSONResponse({"frames": self.frames(), "references": self.references()})

    def api_photos(self, request: Request) -> Response:
        try:
            frame_id = store._check_id(request.path_params["frame_id"], "frame_id")
        except ValueError as exc:
            return _fail(400, str(exc))
        if not (self._loaded() if frame_id == ALL else store.cache_dir(self.root, frame_id).is_dir()):
            return _fail(404, "No cached photos for that frame.")
        return JSONResponse(self.photos(frame_id))

    async def api_set_selection(self, request: Request) -> Response:
        try:
            frame_id = store._check_id(request.path_params["frame_id"], "frame_id")
            asset_ids = (await request.json())["asset_ids"]
            assert isinstance(asset_ids, list)
        except (ValueError, KeyError, AssertionError, TypeError):
            return _fail(400, "Expected {\"asset_ids\": [...]}.")
        return JSONResponse({"selected": len(self._save_selection(frame_id, asset_ids))})

    def _save_selection(self, frame_id: str, asset_ids: list[Any], **record: Any) -> list[str]:
        """Save what is ticked. ``record`` replaces the pre-selection notes
        (see _preselect); without it they are carried over unchanged."""
        path = store.selection_path(self.root, frame_id)
        # Only photos that are really in this frame's cache can be selected.
        present = self._present(frame_id)
        valid = [
            a for a in dict.fromkeys(str(a) for a in asset_ids)
            if store._SAFE_ID.match(a) and a in present
        ]
        notes = {
            key: value
            for key, value in store._read_json(path, {}).items()
            if key in ("preselected", "unticked")
        }
        store._write_json(
            path,
            {
                **notes,
                **record,
                "frame_id": frame_id,
                "asset_ids": valid,
                "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
        )
        return valid

    def _preselect(self, frame_id: str, person: str, matches: list[str]) -> list[str]:
        """Tick a person's matches for removal, as an opt-out review starts.

        The selection file notes which photos were ticked this way and for
        whom, so that:
        - opening someone else takes the last person's ticks away again
          (photos picked by hand stay);
        - a photo the user unticked by hand is not ticked again the next time
          that person is opened or searched for.
        """
        saved = store._read_json(store.selection_path(self.root, frame_id), {})
        current = self.selection(frame_id)
        before = saved.get("preselected") or {}
        before_ids = set(before.get("asset_ids") or [])
        unticked = {k: list(v) for k, v in (saved.get("unticked") or {}).items()}
        left = before_ids - set(current)
        # Some of the last pre-selection unticked: remember which. All of it
        # gone is "Clear selection", which says nothing about single photos.
        if before.get("person") and left and left != before_ids:
            mine = set(unticked.get(before["person"], [])) | left
            unticked[before["person"]] = sorted(mine)
        skip = set(unticked.get(person, []))
        chosen = [a for a in matches if a not in skip]
        by_hand = [a for a in current if a not in before_ids]
        return self._save_selection(
            frame_id,
            [*by_hand, *chosen],
            preselected={"person": person, "asset_ids": chosen},
            unticked=unticked,
        )

    def thumb(self, request: Request) -> Response:
        found = self._photo_path(request)
        if found is None:
            return Response(status_code=404)
        frame_id, source = found
        dest = store.thumbs_dir(self.root, frame_id) / source.name
        if not dest.is_file():
            import cv2

            # Decode at quarter size: far faster than a full 12 MP decode.
            image = cv2.imread(str(source), cv2.IMREAD_REDUCED_COLOR_4)
            if image is None:
                return Response(status_code=415)
            scale = THUMB_SHORT_SIDE / min(image.shape[:2])
            if scale < 1:
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            with self._thumb_lock:
                dest.parent.mkdir(parents=True, exist_ok=True)
            # OpenCV picks the encoder from the extension, so keep ".jpg".
            tmp = dest.with_name(f"{dest.stem}.{uuid.uuid4().hex[:8]}.part.jpg")
            cv2.imwrite(str(tmp), image, [cv2.IMWRITE_JPEG_QUALITY, 82])
            tmp.replace(dest)
        return FileResponse(dest, headers={"Cache-Control": "max-age=86400"})

    def photo(self, request: Request) -> Response:
        found = self._photo_path(request)
        return FileResponse(found[1]) if found else Response(status_code=404)

    def api_faces(self, request: Request) -> Response:
        found = self._photo_path(request)
        if found is None:
            return _fail(404, "Photo not found.")
        try:
            return JSONResponse({"faces": self.matcher.face_boxes(found[1])})
        except FaceMatcherError as exc:
            return _fail(422, str(exc))

    async def api_add_reference(self, request: Request) -> Response:
        try:
            body = await request.json()
            frame_id = store._check_id(body["frame_id"], "frame_id")
            asset_id = store._check_id(body["asset_id"], "asset_id")
            box = {k: float(body["box"][k]) for k in ("x", "y", "w", "h")}
            name = str(body.get("name") or "").strip()[:60] or "Person"
        except (ValueError, KeyError, TypeError):
            return _fail(400, "Expected frame_id, asset_id, box and name.")
        found = self._source(frame_id, asset_id)
        if found is None:
            return _fail(404, "Photo not found.")
        frame_id, source = found  # in "all", the frame that holds the photo
        # Face detection is slow and blocking; keep it off the event loop.
        return await run_in_threadpool(
            self._save_reference, source, frame_id, asset_id, box, name
        )

    def _save_reference(
        self, source: Path, frame_id: str, asset_id: str, box: dict[str, float], name: str
    ) -> Response:
        ref_id = f"{people.slug(name)}-{uuid.uuid4().hex[:8]}"
        dest = store.references_dir(self.root) / f"{ref_id}.jpg"
        try:
            if not self._crop_reference(source, box, dest):
                dest.unlink(missing_ok=True)
                return _fail(422, "That face is too small or unclear to use. Try another photo.")
        except FaceMatcherError as exc:
            dest.unlink(missing_ok=True)
            return _fail(422, str(exc))

        entry = {
            "id": ref_id,
            "name": name,
            "path": str(dest),
            "person": people.slug(name),
            "frame_id": frame_id,
            "asset_id": asset_id,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        store._write_json(
            store.references_dir(self.root) / "references.json", self.references() + [entry]
        )
        return JSONResponse(entry)

    async def api_feedback(self, request: Request) -> Response:
        """Record "yes, that's them" / "no" for photos from the last search."""
        try:
            frame_id = store._check_id(request.path_params["frame_id"], "frame_id")
            body = await request.json()
            asset_ids = [str(a) for a in body["asset_ids"]]
            label = body["label"]
        except (ValueError, KeyError, TypeError):
            return _fail(400, "Expected asset_ids and label.")
        if frame_id == ALL:
            return self._feedback_all(asset_ids, label)
        flagged = self.flagged(frame_id)
        if not flagged["person"]:
            return _fail(409, "Search for a saved person first, then answer about the matches.")
        matched = {m["asset_id"]: m for m in flagged["matches"]}
        unknown = [a for a in asset_ids if a not in matched]
        if unknown:
            return _fail(404, "Those photos are not in the current flagged list.")
        # The answer is about the face that matched, so store that face.
        embeddings = store._load_embeddings(store.cache_dir(self.root, frame_id) / "embeddings.npz")
        faces = {}
        for asset_id in asset_ids:
            rows = embeddings.get(asset_id)
            index = matched[asset_id].get("face_index", 0)
            faces[asset_id] = rows[index] if rows is not None and index < len(rows) else None
        try:
            counts = people.set_labels(self.root, flagged["person"], frame_id, faces, label)
        except ValueError as exc:
            return _fail(400, str(exc))
        return JSONResponse(counts)

    def _feedback_all(self, asset_ids: list[str], label: Any) -> Response:
        """An answer given in "All frames" holds on every frame the photo is on."""
        flagged, per_frame = self._flagged_all()
        person = flagged["person"]
        if not person:
            return _fail(409, "Search for a saved person first, then answer about the matches.")
        if any(not any(a in m for m in per_frame.values()) for a in asset_ids):
            return _fail(404, "Those photos are not in the current flagged list.")
        held = self._holders()
        embeddings: dict[str, dict[str, Any]] = {}
        by_frame: dict[str, dict[str, Any]] = {}
        for asset_id in asset_ids:
            # The face that matched, from the frame where it matched best.
            source = max(
                (f for f in per_frame if asset_id in per_frame[f]),
                key=lambda f: per_frame[f][asset_id]["similarity"],
            )
            if source not in embeddings:
                embeddings[source] = store._load_embeddings(
                    store.cache_dir(self.root, source) / "embeddings.npz"
                )
            rows = embeddings[source].get(asset_id)
            index = per_frame[source][asset_id].get("face_index", 0)
            face = rows[index] if rows is not None and index < len(rows) else None
            if face is None and label is not None:
                return _fail(400, f"No face is known for photo {asset_id}.")
            for frame in dict.fromkeys([*held.get(asset_id, []), source]):
                by_frame.setdefault(frame, {})[asset_id] = face
        try:
            for frame, faces in by_frame.items():
                people.set_labels(self.root, person, frame, faces, label)
        except ValueError as exc:
            return _fail(400, str(exc))
        # Counted by photo, not by frame: one photo on two frames is one answer.
        entries = people.feedback(self.root).get(people.slug(person), {})
        said: dict[str, set[str]] = {people.YES: set(), people.NO: set()}
        for key, entry in entries.items():
            said.setdefault(entry["label"], set()).add(key.split("/", 1)[-1])
        return JSONResponse({"confirmed": len(said[people.YES]), "rejected": len(said[people.NO])})

    async def _search_all(self, person: str, threshold: float) -> dict[str, Any]:
        """Search every loaded frame for one person and merge what was found."""
        frames = self._loaded_ids()
        if not frames:
            raise ValueError("No photos are loaded yet.")
        results = [
            await store.find_person_in_frame_impl(
                self.matcher, self.root, frame, "", threshold, None, person
            )
            for frame in frames
        ]
        matched = list(dict.fromkeys(m["asset_id"] for r in results for m in r["matches"]))
        unreadable = {a for r in results for a in r["unreadable_asset_ids"]}
        return {
            "matches": [{"asset_id": a} for a in matched],
            "match_count": len(matched),
            "photos_scanned": len(set(self._holders()) - unreadable),
            "examples_used": results[-1]["examples_used"],
            "review_mode": results[0]["review_mode"],
        }

    async def api_search(self, request: Request) -> Response:
        """Run the search again from the page, using the answers given so far."""
        try:
            frame_id = store._check_id(request.path_params["frame_id"], "frame_id")
            body = await request.json()
        except (ValueError, TypeError):
            return _fail(400, "Bad request.")
        last = self.flagged(frame_id)
        person = str(body.get("person") or last["person"] or "")
        if not person:
            return _fail(409, "Pick a person to search for.")
        # The last cut-off is only carried over for the same person: one
        # lowered while looking for someone else must not widen this search.
        inherited = last["similarity_threshold"] if person == last["person"] else None
        try:
            threshold = float(body.get("similarity_threshold") or inherited or 0.45)
            if frame_id == ALL:
                result = await self._search_all(person, threshold)
            else:
                result = await store.find_person_in_frame_impl(
                    self.matcher, self.root, frame_id, "", threshold, None, person
                )
        except (ValueError, FaceMatcherError) as exc:
            return _fail(400, str(exc))
        selected = self.selection(frame_id)
        if result["review_mode"] == "opt_out":
            # Someone from the People view: their matches start out selected
            # for removal. Never for a reference person, whose matches each
            # need a "yes".
            selected = self._preselect(
                frame_id,
                self.flagged(frame_id)["person"],
                [m["asset_id"] for m in result["matches"]],
            )
        return JSONResponse(
            {
                "match_count": result["match_count"],
                "photos_scanned": result["photos_scanned"],
                "examples_used": result["examples_used"],
                "similarity_threshold": threshold,
                "review_mode": result["review_mode"],
                "selected": len(selected),
            }
        )

    async def api_search_text(self, request: Request) -> Response:
        """Open-vocabulary search: find photos matching a description (CLIP)."""
        try:
            frame_id = store._check_id(request.path_params["frame_id"], "frame_id")
            body = await request.json()
            query = str(body.get("query") or "").strip()
        except (ValueError, TypeError):
            return _fail(400, "Bad request.")
        if not query:
            return _fail(400, "Type something to search for.")
        try:
            threshold = float(body.get("similarity_threshold", store.CLIP_DEFAULT_THRESHOLD))
            top_k = int(body.get("top_k") or 300)
        except (TypeError, ValueError):
            threshold, top_k = store.CLIP_DEFAULT_THRESHOLD, 300
        frames = self._loaded_ids() if frame_id == ALL else [frame_id]
        if not frames:
            return _fail(404, "No cached photos for that frame.")
        if not self.searcher.model_present():
            # The page asks before the model is fetched (see api_search_model_download).
            return JSONResponse({"error": image_search.MODEL_NEEDED, "needs": "search_model"},
                                status_code=409)
        try:
            searcher = self.searcher
            results = [
                await store.find_photos_by_description_impl(
                    searcher, self.root, frame, query,
                    similarity_threshold=threshold, top_k=top_k,
                )
                for frame in frames
            ]
        except (ValueError, ImageSearchError) as exc:
            return _fail(400, str(exc))
        result = results[0]
        if frame_id == ALL:
            # The same photo on two frames: keep its best score.
            best: dict[str, dict[str, Any]] = {}
            for found in results:
                for match in found["matches"]:
                    held = best.get(match["asset_id"])
                    if held is None or match["similarity"] > held["similarity"]:
                        best[match["asset_id"]] = match
            matches = sorted(best.values(), key=lambda m: m["similarity"], reverse=True)
            if top_k > 0:
                matches = matches[:top_k]
            result = {**result, "matches": matches, "match_count": len(matches)}
        return JSONResponse({
            "query": result["query"],
            "match_count": result["match_count"],
            "similarity_threshold": result["similarity_threshold"],
            "matches": result["matches"],
        })

    def _crop_reference(self, source: Path, box: dict[str, float], dest: Path) -> bool:
        """Write a crop in which the clicked face is the one the tools will use.

        A reference image is read by taking its largest face, so a neighbour
        caught in the margin must not be bigger than the person clicked. Try
        margins from generous to tight and keep the first crop whose largest
        face is the clicked one, preferring a crop with nobody else in it.
        """
        usable = None
        for pad in (0.6, 0.35, 0.15, 1.2):
            cx, cy = self.matcher.save_face_crop(source, box, dest, pad=pad)
            faces = self.matcher.faces_in(dest)
            if not faces:
                continue
            x1, y1, x2, y2 = faces[0]
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                if len(faces) == 1:
                    return True
                usable = usable or pad
        if usable is None:
            return False
        self.matcher.save_face_crop(source, box, dest, pad=usable)
        return True

    def api_delete_reference(self, request: Request) -> Response:
        ref_id = request.path_params["ref_id"]
        kept = [r for r in self.references() if r["id"] != ref_id]
        for entry in self.references():
            if entry["id"] == ref_id:
                Path(entry["path"]).unlink(missing_ok=True)
        store._write_json(store.references_dir(self.root) / "references.json", kept)
        return JSONResponse({"references": kept})

    def reference_image(self, request: Request) -> Response:
        ref_id = request.path_params["ref_id"]
        for entry in self.references():
            if entry["id"] == ref_id:
                return FileResponse(entry["path"])
        return Response(status_code=404)

    def _busy(self) -> str | None:
        """What is running now, in words, or None."""
        if self.removal.running:
            return "A removal is running."
        if self.sync.running:
            return "Photos are still being loaded."
        if self.restore.running:
            return "Photos are being put back."
        if self.model_job.running:
            return "The face model is still downloading."
        if self.search_model_job.running:
            return "The photo search model is still downloading."
        with self._build_lock:
            if self._build["state"] == "running":
                return "Still looking for people."
        return None

    # ---------------------------------------------------------- account

    def api_account(self, request: Request) -> Response:
        return JSONResponse({"signed_in": store.signed_in(self.root)})

    async def api_sign_in(self, request: Request) -> Response:
        """Sign in to Aura and keep the token it issues. The password is sent
        to Aura and nowhere else, and is not written down."""
        try:
            body = await request.json()
            email, password = str(body["email"]).strip(), str(body["password"])
            if not email or not password:
                raise ValueError
        except (ValueError, KeyError, TypeError):
            return _fail(400, "Enter your Aura email and password.")
        client = self.client_factory(email, password)
        try:
            await client.login()
            store.save_session(self.root, client.session)
        except AuraAuthError as exc:
            return _fail(401, "Aura did not accept that email and password.", str(exc))
        except AuraError as exc:
            return _explained(502, exc)
        finally:
            await client.aclose()
        return JSONResponse({"signed_in": True})

    def api_sign_out(self, request: Request) -> Response:
        store.clear_session(self.root)
        return JSONResponse({"signed_in": store.signed_in(self.root)})

    async def api_account_frames(self, request: Request) -> Response:
        """The frames on the Aura account, and how much of each is loaded here."""
        client = self.client_factory()
        try:
            frames = await client.list_frames()
        except AuraAuthError as exc:
            return _explained(401, exc)
        except AuraError as exc:
            return _explained(502, exc)
        finally:
            await client.aclose()
        store.save_frame_names(self.root, frames)
        listed = []
        for frame in frames:
            folder = store.cache_dir(self.root, str(frame["id"]))
            listed.append({
                "id": str(frame["id"]),
                "name": frame.get("name") or str(frame["id"]),
                "asset_count": frame.get("asset_count"),
                "loaded": sum(1 for _ in folder.glob("*.jpg")) if folder.is_dir() else 0,
                "synced": (folder / "index.json").is_file(),
            })
        return JSONResponse({"frames": listed})

    async def api_sync(self, request: Request) -> Response:
        """Get the photos of the chosen frames, then work out who is in them."""
        try:
            wanted = (await request.json())["frame_ids"]
            if not isinstance(wanted, list) or not wanted:
                raise ValueError
            wanted = [store._check_id(f, "frame_id") for f in wanted]
            if ALL in wanted:
                raise ValueError("reserved frame id")
        except (ValueError, KeyError, TypeError):
            return _fail(400, "Choose at least one frame.")

        async def work(progress: store.Progress) -> dict[str, Any]:
            client = self.client_factory()
            try:
                frames = await client.list_frames()
                store.save_frame_names(self.root, frames)
                # Faces are read as photos arrive, and people grouped along the way.
                return await store.load_frames_impl(
                    client, self.matcher, self.root, wanted, progress
                )
            finally:
                await client.aclose()

        if not self.matcher.model_present():
            return _fail(409, MODEL_NEEDED)
        with self._start_lock:
            busy = self._busy()
            if busy or not self.sync.start(work):
                return _fail(409, f"{busy or 'Already getting photos.'} Wait for it to finish.")
        return JSONResponse({"state": "running"}, status_code=202)

    def api_sync_status(self, request: Request) -> Response:
        return JSONResponse(self.sync.snapshot())

    # ---------------------------------------------------------- removal

    async def api_removal_preview(self, request: Request) -> Response:
        """What a removal would do. Never removes anything.

        Without ``asset_ids`` it previews the photos ticked on this frame.
        The answer carries a token: confirming means handing that token back,
        so what is removed is exactly what this preview listed.
        """
        try:
            frame_id = store._check_id(request.path_params["frame_id"], "frame_id")
            body = await request.json()
            if not isinstance(body, dict) or body.get("scope") not in PAGE_SCOPES:
                raise ValueError("scope must be 'frame' or 'all_frames'.")
            scope, asset_ids = body["scope"], body.get("asset_ids")
            if asset_ids is None:
                asset_ids = self.selection(frame_id)
            if not isinstance(asset_ids, list):
                raise ValueError("asset_ids must be a list.")
            if frame_id == ALL:
                preview, groups = await self._preview_all(asset_ids)
                scope = "all_frames"
            else:
                preview = await store.delete_assets_impl(None, self.root, frame_id, asset_ids, scope)
                groups = None
        except (ValueError, TypeError) as exc:
            return _fail(400, str(exc))
        token = secrets.token_urlsafe(16)
        # One plan at a time: a newer preview replaces the one on screen.
        self._plan = {"token": token, "frame_id": frame_id, "scope": scope}
        if groups is None:
            self._plan["frames"] = {
                entry["asset_id"]: entry["frames"] for entry in preview["would_delete"]
            }
        else:
            # "All frames": per home frame, exactly what its part of the
            # preview listed, {asset_id: frames}. Nothing else is ever run.
            self._plan["groups"] = groups
        return JSONResponse({**preview, "frame_id": frame_id, "token": token})

    async def _preview_all(
        self, asset_ids: list[Any]
    ) -> tuple[dict[str, Any], list[tuple[str, dict[str, list[str]]]]]:
        """Preview removing photos from every frame they are on, from "All frames".

        delete_assets_impl works from one frame whose list holds each photo,
        so the photos are grouped by the first loaded frame that has both the
        photo and a finished list, each group previewed with scope
        "all_frames", and the previews merged into one. Returns the preview
        and, per group, what it listed: the removal runs exactly that.
        """
        held = self._holders()
        synced = store.synced_frames(self.root)
        requested = list(dict.fromkeys(str(a).strip() for a in asset_ids))
        homes: dict[str, list[str]] = {}
        unknown: list[str] = []
        for asset_id in requested:
            home = next((f for f in held.get(asset_id, []) if f in synced), None)
            if home is None:
                unknown.append(asset_id)  # not here, or its frames still loading
            else:
                homes.setdefault(home, []).append(asset_id)
        previews = [
            (home, await store.delete_assets_impl(None, self.root, home, ids, "all_frames"))
            for home, ids in homes.items()
        ]

        would_delete = [entry for _, p in previews for entry in p["would_delete"]]
        order = {frame: i for i, frame in enumerate(self._loaded_ids())}
        counts: dict[str, int] = {}
        for entry in would_delete:
            for frame in entry["frames"]:
                counts[frame] = counts.get(frame, 0) + 1
        names = store.frame_names(self.root)
        merged: dict[str, Any] = {
            "status": "confirmation_required",
            "scope": "all_frames",
            "effect": store.SCOPE_EFFECT["all_frames"],
            "deleted_count": 0,
            "would_delete_count": len(would_delete),
            "would_delete": would_delete,
            "frame_removal_count": sum(counts.values()),
            "frames_affected": [
                {"id": frame, "name": names.get(frame) or frame, "count": counts[frame]}
                for frame in sorted(counts, key=lambda f: (order.get(f, len(order)), f))
            ],
            "last_copy_count": sum(p["last_copy_count"] for _, p in previews),
            "unsynced_frames": [
                {"id": frame, "name": name or frame}
                for frame, name in names.items()
                if frame not in synced and frame != ALL
            ],
            "unknown_asset_ids": unknown + [a for _, p in previews for a in p["unknown_asset_ids"]],
            "no_copy_asset_ids": [a for _, p in previews for a in p["no_copy_asset_ids"]],
            "warnings": [],
            "message": store.CONFIRMATION_MESSAGE,
        }
        # Cautions are worked out once over every photo, so the dialog says
        # "2 photos were marked as NOT them" once, not once per frame.
        reviewed = [p for _, p in previews if p.get("reviewed_for")]
        if reviewed:
            name = reviewed[0]["reviewed_for"]
            opt_out = all(p.get("review_mode") == "opt_out" for p in reviewed)
            statuses = [entry.get("review", "not_matched") for entry in would_delete]
            merged["reviewed_for"] = name
            merged["confirmed_count"] = statuses.count("confirmed")
            if opt_out:
                merged["review_mode"] = "opt_out"
                merged["assumed_count"] = statuses.count("unanswered")
            merged["warnings"] = store.removal_warnings(name, statuses, assume_match=opt_out)
        groups = [
            (home, {entry["asset_id"]: entry["frames"] for entry in p["would_delete"]})
            for home, p in previews
            if p["would_delete"]
        ]
        return merged, groups

    async def _remove_all(
        self,
        client: Any,
        groups: list[tuple[str, dict[str, list[str]]]],
        progress: store.Progress,
    ) -> dict[str, Any]:
        """Carry out an "All frames" removal: each previewed group in turn.

        Each group is run with ``previewed`` set to what its preview listed,
        so no photo or frame beyond the preview is touched. A lost sign-in
        stops the whole run: later groups are reported, not attempted.
        """
        total = sum(len(frames) for _, frames in groups)
        merged: dict[str, Any] = {
            "status": "completed", "scope": "all_frames",
            "deleted_count": 0, "partial_count": 0, "frame_removal_count": 0,
            "deleted": [], "failed": [], "unknown_asset_ids": [], "no_copy_asset_ids": [],
        }
        offset = 0
        for position, (home, frames) in enumerate(groups):

            async def report(done: float, _total: float | None, _message: str | None,
                             offset: int = offset) -> None:
                count = offset + int(done)
                await progress(count, total, f"Removed {count}/{total} photos")

            result = await store.delete_assets_impl(
                client, self.root, home, list(frames), "all_frames", True, report, frames,
                self.removal.stop.is_set,
            )
            for key in ("deleted", "failed", "unknown_asset_ids", "no_copy_asset_ids"):
                merged[key].extend(result[key])
            for key in ("deleted_count", "partial_count", "frame_removal_count"):
                merged[key] += result[key]
            offset += len(frames)
            if result["status"] == "aborted":
                merged["status"] = "aborted"
                stopped = next(
                    (f for f in result["failed"] if f["error"].startswith("not attempted: ")),
                    {"error": "not attempted: the removal stopped early.",
                     "reason": "Not attempted: the removal stopped early."},
                )
                reason, plain = stopped["error"], stopped.get("reason") or stopped["error"]
                merged["failed"].extend(
                    {"asset_id": asset_id, "frame_id": frame, "error": reason, "reason": plain}
                    for _, later in groups[position + 1:]
                    for asset_id, targets in later.items()
                    for frame in targets
                )
                break
        merged["failed_count"] = len(merged["failed"])
        return merged

    async def api_removal(self, request: Request) -> Response:
        """Carry out the removal the user just confirmed.

        Takes only the token of the preview they were shown. The frame, the
        scope and the photos all come from that preview, never from this
        request, and a frame loaded since is left alone.
        """
        try:
            token = (await request.json()).get("token")
        except (ValueError, TypeError, AttributeError):
            token = None
        if not isinstance(token, str) or not token:
            return _fail(400, "Confirm a removal from its preview.")
        with self._start_lock:
            plan = self._plan
            if plan is None or not secrets.compare_digest(plan["token"], token):
                return _fail(409, "That preview is out of date. Open the removal again.")
            busy = self._busy()
            if busy:
                return _fail(409, f"{busy} Wait for it to finish.")
            self._plan = None  # a preview can be confirmed once

            async def work(progress: store.Progress) -> dict[str, Any]:
                client = self.client_factory()
                try:
                    if "groups" in plan:
                        return await self._remove_all(client, plan["groups"], progress)
                    return await store.delete_assets_impl(
                        client, self.root, plan["frame_id"], list(plan["frames"]),
                        plan["scope"], True, progress, plan["frames"], self.removal.stop.is_set,
                    )
                finally:
                    await client.aclose()

            self.removal.start(work)
        return JSONResponse({"state": "running"}, status_code=202)

    def api_removal_status(self, request: Request) -> Response:
        return JSONResponse(self.removal.snapshot())

    # ----------------------------------------------------------- people

    def _gallery(self) -> dict[str, Any]:
        with self._build_lock:
            status = dict(self._build)
        return {**people_index.gallery(self.root), "build": status}

    def _set_build(self, **fields: Any) -> None:
        with self._build_lock:
            self._build = {**self._build, **fields}

    def start_people_build(self) -> bool:
        """Start working out who is in the photos; False if anything is running."""
        with self._start_lock:
            if self._busy():
                return False
            with self._build_lock:
                self._build = {"state": "running", "done": 0, "total": 0, "message": "Starting"}

        async def progress(done: float, total: float | None, message: str | None) -> None:
            self._set_build(done=int(done), total=int(total or 0), message=message or "")

        def run() -> None:
            try:
                result = asyncio.run(store.find_people_impl(self.matcher, self.root, progress))
            except Exception as exc:  # noqa: BLE001 - shown to the user on the page
                plain = store.explain_error(exc)
                extra = {"detail": plain["detail"]} if plain["detail"] else {}
                self._set_build(state="error", done=0, total=0, message=plain["message"], **extra)
            else:
                self._set_build(state="done", message=f"Found {result.get('people', 0)} people")

        self._build_thread = threading.Thread(target=run, daemon=True)
        self._build_thread.start()
        return True

    def api_people(self, request: Request) -> Response:
        return JSONResponse(self._gallery())

    def api_people_build(self, request: Request) -> Response:
        if not self.matcher.model_present():
            return _fail(409, MODEL_NEEDED)
        busy = self._busy()
        if not self.start_people_build():
            return _fail(409, f"{busy or 'Something else is running.'} Wait for it to finish.")
        return JSONResponse({"state": "running"}, status_code=202)

    def api_people_status(self, request: Request) -> Response:
        with self._build_lock:
            return JSONResponse(dict(self._build))

    @staticmethod
    def _people_error(exc: ValueError) -> JSONResponse:
        return _fail(404 if str(exc).startswith("No such person") else 400, str(exc))

    async def api_edit_person(self, request: Request) -> Response:
        """Rename a person, or hide / unhide them."""
        try:
            person_id = store._check_id(request.path_params["person_id"], "person id")
            body = await request.json()
            assert isinstance(body, dict)
        except (ValueError, AssertionError, TypeError):
            return _fail(400, "Expected a name and/or hidden.")
        try:
            if "name" in body:
                people_index.rename(self.root, person_id, str(body["name"] or ""))
            if "hidden" in body:
                people_index.set_hidden(self.root, person_id, body["hidden"] is True)
            person = people_index.find(self.root, person_id)
            if person is None or person["id"] != person_id:
                raise ValueError(f"No such person: {person_id}")
        except ValueError as exc:
            return self._people_error(exc)
        return JSONResponse(
            people_index.gallery_entry(person, people_index.cached_photos(self.root))
        )

    async def api_merge_people(self, request: Request) -> Response:
        try:
            body = await request.json()
            target = store._check_id(body["target"], "person id")
            other = store._check_id(body["other"], "person id")
        except (ValueError, KeyError, TypeError):
            return _fail(400, "Expected target and other.")
        try:
            people_index.merge(self.root, target, other)
        except ValueError as exc:
            return self._people_error(exc)
        return JSONResponse(self._gallery())

    def api_suggestions(self, request: Request) -> Response:
        """Other groups that may be this person, for "more of them?"."""
        try:
            person_id = store._check_id(request.path_params["person_id"], "person id")
            return JSONResponse({"suggestions": people_index.suggestions(self.root, person_id)})
        except ValueError as exc:
            return self._people_error(exc)

    async def api_not_same(self, request: Request) -> Response:
        try:
            body = await request.json()
            first = store._check_id(body["a"], "person id")
            second = store._check_id(body["b"], "person id")
        except (ValueError, KeyError, TypeError):
            return _fail(400, "Expected a and b.")
        try:
            people_index.not_same(self.root, first, second)
        except ValueError as exc:
            return self._people_error(exc)
        return JSONResponse({"ok": True})

    def api_unmerge(self, request: Request) -> Response:
        try:
            people_index.unmerge(self.root)
        except ValueError as exc:
            return _fail(409, str(exc))
        return JSONResponse(self._gallery())

    def api_group_faces(self, request: Request) -> Response:
        """Every face in a person's group, to look through before adding it."""
        try:
            person_id = store._check_id(request.path_params["person_id"], "person id")
            return JSONResponse({"faces": people_index.group_faces(self.root, person_id)})
        except ValueError as exc:
            return self._people_error(exc)

    async def api_move_faces(self, request: Request) -> Response:
        """Some faces of one group are the person under review: take only those."""
        try:
            body = await request.json()
            target = store._check_id(body["target"], "person id")
            other = store._check_id(body["other"], "person id")
            faces = [
                [store._check_id(str(asset_id), "asset_id"), int(index)]
                for asset_id, index in body["faces"]
            ]
        except (ValueError, KeyError, TypeError):
            return _fail(400, "Expected target, other and faces.")
        try:
            people_index.move_faces(self.root, target, other, faces)
        except ValueError as exc:
            return self._people_error(exc)
        return JSONResponse(self._gallery())

    def _face_params(self, request: Request) -> tuple[str, int] | None:
        try:
            return (
                store._check_id(request.path_params["asset_id"], "asset_id"),
                int(request.path_params["index"]),
            )
        except (ValueError, KeyError):
            return None

    async def face_crop(self, request: Request) -> Response:
        face = self._face_params(request)
        path = None
        if face is not None:
            path = await run_in_threadpool(
                people_index.ensure_face_crop, self.matcher, self.root, *face
            )
        if path is None:
            return Response(status_code=404)
        return FileResponse(path, headers={"Cache-Control": "max-age=86400"})

    async def api_face_box(self, request: Request) -> Response:
        """Which frame's copy of the photo to show, and where this face is in it."""
        face = self._face_params(request)
        found = None
        if face is not None:
            found = await run_in_threadpool(people_index.face_box, self.matcher, self.root, *face)
        if found is None:
            return _fail(404, "Face not found.")
        return JSONResponse({"frame_id": found[0], "box": found[1]})

    async def person_avatar(self, request: Request) -> Response:
        try:
            person_id = store._check_id(request.path_params["person_id"], "person id")
        except ValueError:
            return Response(status_code=404)
        try:
            # Finding the face again is slow and blocking; keep it off the event loop.
            path = await run_in_threadpool(
                people_index.ensure_avatar, self.matcher, self.root, person_id
            )
        except FaceMatcherError:
            path = None
        if path is None:
            return Response(status_code=404)
        # The crop can change after a rebuild, so the browser must ask each time.
        return FileResponse(path, headers={"Cache-Control": "no-cache"})

    # ------------------------------------------------- face model consent

    def _settings(self) -> dict[str, Any]:
        data = store._read_json(self.root / "settings.json", {})
        return data if isinstance(data, dict) else {}

    def warm_model(self) -> None:
        """Load the face model in the background, if it is here. Never fetches it."""
        if self.matcher.model_present():
            threading.Thread(target=self._warm, daemon=True).start()

    def _warm(self) -> None:
        try:
            self.matcher._get_app()
        except Exception:  # noqa: BLE001 - it is tried again, and reported, on first use
            pass

    def api_setup(self, request: Request) -> Response:
        """What is needed before things work: the face model (first run) and
        the photo search model (asked for when Find is first pressed)."""
        settings = self._settings()
        return JSONResponse({
            "face_model": {
                "present": self.matcher.model_present(),
                "agreed": bool(settings.get("face_model_agreed_at")),
                "size_bytes": MODEL_DOWNLOAD_BYTES,
                "download": self.model_job.snapshot(),
            },
            "search_model": {
                "present": self.searcher.model_present(),
                "agreed": bool(settings.get("search_model_agreed_at")),
                "size_bytes": image_search.MODEL_DOWNLOAD_BYTES,
                "download": self.search_model_job.snapshot(),
            },
        })

    async def api_search_model_download(self, request: Request) -> Response:
        """The user agreed: record it and fetch the photo search model, as a job."""
        try:
            body = await request.json()
            if not isinstance(body, dict) or body.get("agree") is not True:
                raise ValueError
        except (ValueError, TypeError):
            return _fail(400, "The photo search model is only downloaded once you agree.")
        if self.searcher.model_present():
            return JSONResponse({"state": "done"})
        size = image_search.MODEL_DOWNLOAD_BYTES

        async def work(progress: store.Progress) -> dict[str, Any]:
            def arrived(got: int) -> None:
                self.search_model_job._set(
                    done=min(got, size), total=size,
                    message=f"Downloading the photo search model: {got / 1e6:,.0f} MB of about "
                            f"{size / 1e6:,.0f} MB",
                )

            await progress(0, size, "Downloading the photo search model")
            await asyncio.to_thread(self.searcher.download_model, arrived)
            return {"present": True}

        with self._start_lock:
            busy = self._busy()
            if busy:
                return _fail(409, f"{busy} Wait for it to finish.")
            settings = self._settings()
            settings["search_model_agreed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            store._write_json(self.root / "settings.json", settings)
            self.search_model_job.start(work)
        return JSONResponse({"state": "running"}, status_code=202)

    def api_search_model_status(self, request: Request) -> Response:
        return JSONResponse(self.search_model_job.snapshot())

    async def api_model_download(self, request: Request) -> Response:
        """The user agreed: record it and fetch the model, as a job."""
        try:
            body = await request.json()
            if not isinstance(body, dict) or body.get("agree") is not True:
                raise ValueError
        except (ValueError, TypeError):
            return _fail(400, "The face model is only downloaded once you agree.")
        if self.matcher.model_present():
            return JSONResponse({"state": "done"})

        async def work(progress: store.Progress) -> dict[str, Any]:
            def arrived(size: int) -> None:
                self.model_job._set(
                    done=size, total=MODEL_DOWNLOAD_BYTES,
                    message=f"Downloading the face model: {size / 1e6:,.0f} MB of about "
                            f"{MODEL_DOWNLOAD_BYTES / 1e6:,.0f} MB",
                )

            await progress(0, MODEL_DOWNLOAD_BYTES, "Downloading the face model")
            await asyncio.to_thread(self.matcher.download_model, arrived)
            await progress(MODEL_DOWNLOAD_BYTES, MODEL_DOWNLOAD_BYTES, "Getting the face model ready")
            # Load it now, so a broken download shows here and not mid-load.
            await asyncio.to_thread(self.matcher._get_app)
            return {"present": True}

        with self._start_lock:
            busy = self._busy()
            if busy:
                return _fail(409, f"{busy} Wait for it to finish.")
            settings = self._settings()
            settings["face_model_agreed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            store._write_json(self.root / "settings.json", settings)
            self.model_job.start(work)
        return JSONResponse({"state": "running"}, status_code=202)

    def api_model_status(self, request: Request) -> Response:
        return JSONResponse(self.model_job.snapshot())

    # --------------------------------------------------- putting photos back

    def _removed_frames(self) -> list[str]:
        folder = self.root / "removed"
        if not folder.is_dir():
            return []
        return sorted(
            p.name for p in folder.iterdir()
            if p.is_dir() and p.name != ALL and store._SAFE_ID.match(p.name)
        )

    def _removed_log(self, frame_id: str) -> dict[str, dict[str, Any]]:
        log = store._read_json(store.removed_dir(self.root, frame_id) / "removed.json", {})
        return log if isinstance(log, dict) else {}

    def _removed_scope(self, frame_id: str) -> list[str]:
        return self._removed_frames() if frame_id == ALL else [frame_id]

    def removed(self, frame_id: str) -> list[dict[str, Any]]:
        """Photos removed through the app, newest first; in "all", each photo once."""
        names = store.frame_names(self.root)
        found: dict[str, dict[str, Any]] = {}
        for frame in self._removed_scope(frame_id):
            for asset_id, entry in self._removed_log(frame).items():
                if not isinstance(entry, dict) or not store._SAFE_ID.match(str(asset_id)):
                    continue
                item = found.setdefault(asset_id, {
                    "id": asset_id, "frames": [], "removed_at": None, "when": None,
                    "state": None, "taken_at": entry.get("taken_at"),
                    "local_path": entry.get("local_path"),
                })
                item["frames"].append({"id": frame, "name": names.get(frame) or frame})
                # "removed" (by this app), "removing" (the app was closed part-way:
                # Aura may or may not have acted) or "missing" (found gone from
                # the frame during a load: not removed by this app).
                state = entry.get("state") or "removed"
                rank = {"removing": 0, "removed": 1, "missing": 2}
                if item["state"] is None or rank.get(state, 1) < rank.get(item["state"], 1):
                    item["state"] = state
                removed_at = entry.get("removed_at")
                if removed_at and (item["removed_at"] is None or removed_at > item["removed_at"]):
                    item["removed_at"] = removed_at
                when = removed_at or entry.get("started_at") or entry.get("found_missing_at")
                if when and (item["when"] is None or when > item["when"]):
                    item["when"] = when
                item["local_path"] = item["local_path"] or entry.get("local_path")
        return sorted(found.values(), key=lambda i: (i["when"] or "", i["id"]), reverse=True)

    def api_removed(self, request: Request) -> Response:
        try:
            frame_id = store._check_id(request.path_params["frame_id"], "frame_id")
        except ValueError as exc:
            return _fail(400, str(exc))
        return JSONResponse({"frame_id": frame_id, "removed": self.removed(frame_id)})

    def _kept_copy(self, frame_id: str, asset_id: str) -> Path | None:
        """The copy kept when a photo was removed; never anything outside removed/."""
        kept_root = (self.root / "removed").resolve()
        for frame in self._removed_scope(frame_id):
            entry = self._removed_log(frame).get(asset_id)
            if not isinstance(entry, dict):
                continue
            candidates = [store.removed_dir(self.root, frame) / f"{asset_id}.jpg"]
            if entry.get("local_path"):
                candidates.append(Path(str(entry["local_path"])))
            for path in candidates:
                resolved = path.resolve()
                if resolved.is_file() and resolved.is_relative_to(kept_root):
                    return resolved
        return None

    async def removed_thumb(self, request: Request) -> Response:
        try:
            frame_id = store._check_id(request.path_params["frame_id"], "frame_id")
            asset_id = store._check_id(request.path_params["asset_id"], "asset_id")
        except ValueError:
            return Response(status_code=404)
        source = self._kept_copy(frame_id, asset_id)
        if source is None:
            return Response(status_code=404)
        data = await run_in_threadpool(self._small_jpeg, source)
        if data is None:
            return Response(status_code=415)
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})

    @staticmethod
    def _small_jpeg(source: Path) -> bytes | None:
        import cv2

        image = cv2.imread(str(source), cv2.IMREAD_REDUCED_COLOR_4)
        if image is None:
            return None
        scale = THUMB_SHORT_SIDE / min(image.shape[:2])
        if scale < 1:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
        return encoded.tobytes() if ok else None

    async def api_restore(self, request: Request) -> Response:
        """Put removed photos back on the frames they came off, as a job."""
        try:
            body = await request.json()
            frame_id = store._check_id(body["frame_id"], "frame_id")
            asset_ids = body["asset_ids"]
            if not isinstance(asset_ids, list):
                raise TypeError
        except (ValueError, KeyError, TypeError):
            return _fail(400, "Choose photos to put back.")
        wanted = list(dict.fromkeys(str(a) for a in asset_ids))
        plan: dict[str, list[str]] = {}
        for frame in self._removed_scope(frame_id):
            log = self._removed_log(frame)
            ids = [a for a in wanted if a in log]
            if ids:
                plan[frame] = ids
        if not plan:
            return _fail(400, "None of those photos were removed with this app.")
        total = sum(len(ids) for ids in plan.values())

        async def work(progress: store.Progress) -> dict[str, Any]:
            client = self.client_factory()
            restored: list[dict[str, Any]] = []
            failed: list[dict[str, Any]] = []
            try:
                for frame, ids in plan.items():
                    if self.restore.stop.is_set():
                        break  # the app is closing; the rest stay in the Removed list
                    await progress(len(restored) + len(failed), total,
                                   f"Putting back {len(restored) + len(failed) + 1} of {total}")
                    result = await store.restore_assets_impl(
                        client, self.root, frame, ids, self.restore.stop.is_set)
                    restored.extend({"asset_id": r["asset_id"], "frame_id": frame}
                                    for r in result["restored"])
                    failed.extend(self._restore_failure(frame, f) for f in result["failed"])
                await progress(total, total, "Finished")
            finally:
                await client.aclose()
            return {"restored_count": len(restored), "failed_count": len(failed),
                    "restored": restored, "failed": failed}

        with self._start_lock:
            busy = self._busy()
            if busy or not self.restore.start(work):
                return _fail(409, f"{busy or 'Photos are being put back.'} Wait for it to finish.")
        return JSONResponse({"state": "running"}, status_code=202)

    @staticmethod
    def _restore_failure(frame: str, failure: dict[str, Any]) -> dict[str, Any]:
        copy = failure.get("local_copy")
        state = failure.get("state")
        if failure.get("kind") == "refused":
            message = {
                "removing": "This app was closed while removing this photo. ",
                "missing": "This photo was not removed by this app: it was already gone from the "
                           "frame when photos were loaded. ",
            }.get(state, "")
            message += (
                "Aura would not put this photo back, which almost always means Aura no "
                "longer has it: a photo taken off its last frame is gone from Aura."
            )
            if copy:
                message += f" The copy kept on this computer is at {copy}. You can add it again with the Aura app."
        else:
            message = failure.get("reason") or store.OTHER_TROUBLE
        return {"asset_id": failure["asset_id"], "frame_id": frame, "message": message,
                "detail": failure.get("error", ""), "local_copy": copy}

    def api_restore_status(self, request: Request) -> Response:
        return JSONResponse(self.restore.snapshot())

    def finish_work(self, timeout: float = WIND_DOWN_SECONDS) -> None:
        """Before the app exits: let a removal or put-back finish the photo it is on.

        Quitting mid-photo could leave a photo off Aura with its ledger entry
        still saying the removal had only started.
        """
        for job, doing in ((self.removal, "removing"), (self.restore, "putting back")):
            if job.thread is not None and job.thread.is_alive():
                print(f"Finishing the photo it is {doing} before closing…", flush=True)
                job.wind_down(timeout)

    # ------------------------------------------------------- update notice

    def _update_file(self) -> Path:
        return self.root / "update-check.json"

    def _refresh_latest(self) -> None:
        try:
            latest = self.update_check()
        except Exception:  # noqa: BLE001 - offline, private repository, rate limit: stay quiet
            latest = None
        try:
            store._write_json(self._update_file(), {"checked_at": time.time(), "latest": latest})
        except OSError:
            pass

    def api_version(self, request: Request) -> Response:
        """This version, and the newest release if one is known. Never waits:
        a look at GitHub runs in the background at most once a day."""
        current = app_version()
        answer: dict[str, Any] = {"version": current, "latest": None, "newer": False}
        if os.environ.get(UPDATE_CHECK_ENV):
            return JSONResponse(answer)
        cached = store._read_json(self._update_file(), {})
        cached = cached if isinstance(cached, dict) else {}
        stale = time.time() - float(cached.get("checked_at") or 0) > UPDATE_EVERY
        with self._update_lock:
            idle = self._update_thread is None or not self._update_thread.is_alive()
            if stale and idle:
                self._update_thread = threading.Thread(target=self._refresh_latest, daemon=True)
                self._update_thread.start()
        latest = cached.get("latest") if not stale else None
        if isinstance(latest, str):
            newer, have = _version_key(latest), _version_key(current)
            answer.update(latest=latest, newer=bool(newer and have and newer > have))
        return JSONResponse(answer)

    # --------------------------------------------------- phone on the wifi

    def start_lan(self) -> dict[str, Any]:
        """Open the page to the wifi and make a pairing link. Raises PhoneError."""
        with self._lan_lock:
            if self._lan is None:
                host = self.find_lan_address()
                if not host or not usable_lan_ip(host):
                    raise PhoneError(NO_NETWORK)
                try:
                    listener = self.lan_listener(self._app_for_lan(), host, 0)
                except OSError as exc:
                    raise PhoneError(
                        f"Could not open the page to your wifi on {host}. "
                        "Try again, or restart the app.", str(exc),
                    ) from exc
                self._lan = {"host": host, "port": int(listener.port), "listener": listener,
                             "sessions": [], "pair": None, "pair_until": 0.0}
                self._new_link_locked()
            return self._phone_status_locked()

    def stop_lan(self) -> None:
        """Close the wifi listener. Every link and every paired phone is dead after."""
        with self._lan_lock:
            lan, self._lan = self._lan, None  # refused from this moment on
        if lan is not None:
            try:
                lan["listener"].stop()
            except Exception:  # noqa: BLE001 - it is closed as far as requests go
                pass

    def new_phone_link(self) -> dict[str, Any] | None:
        with self._lan_lock:
            if self._lan is None:
                return None
            self._new_link_locked()
            return self._phone_status_locked()

    def phone_status(self) -> dict[str, Any]:
        with self._lan_lock:
            return self._phone_status_locked()

    def _new_link_locked(self) -> None:
        assert self._lan is not None
        self._lan["pair"] = secrets.token_urlsafe(16)
        self._lan["pair_until"] = time.time() + PAIR_TTL

    def _phone_status_locked(self) -> dict[str, Any]:
        lan = self._lan
        if lan is None:
            return {"on": False, "address": None, "link": None, "qr": None, "phones": 0,
                    "link_used": False, "link_expired": False, "expires_in": 0}
        address = f"http://{lan['host']}:{lan['port']}"
        left = lan["pair_until"] - time.time()
        live = lan["pair"] is not None and left > 0
        link = f"{address}/pair?t={lan['pair']}" if live else None
        return {
            "on": True, "address": address, "link": link, "qr": qr_svg(link) if link else None,
            "phones": len(lan["sessions"]), "link_used": lan["pair"] is None,
            "link_expired": lan["pair"] is not None and not live,
            "expires_in": max(0, int(left)) if live else 0,
        }

    def _app_for_lan(self) -> Starlette:
        return self._asgi if self._asgi is not None else self.app()

    def _pair(self, request: Request, lan: dict[str, Any]) -> Response:
        """Opened from the code on the computer: hand this phone its cookie, once."""
        offered = request.query_params.get("t", "").encode()
        session = None
        with self._lan_lock:
            if (self._lan is lan and lan["pair"] is not None and time.time() < lan["pair_until"]
                    and secrets.compare_digest(lan["pair"].encode(), offered)):
                lan["pair"] = None  # the link works once
                session = secrets.token_urlsafe(32)
                # The cookie works only from the address that paired.
                lan["sessions"].append((session, _client_ip(request)))
        headers = {"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"}
        if session is None:
            response = _fail(403, "This link was already used or is out of date. "
                                  "Show a new code on the computer and scan it again.")
            response.headers.update(headers)
            return response
        response = HTMLResponse(PAIRED_PAGE, headers=headers)
        response.set_cookie(PHONE_COOKIE, session, path="/", httponly=True, samesite="strict",
                            secure=False)
        return response

    def _wifi_refusal(self, request: Request, routes: list[Any]) -> Response | None:
        """None when a request that came in on the wifi may go on; else the refusal."""
        server = request.scope.get("server")
        with self._lan_lock:
            lan = self._lan
            sessions = list(lan["sessions"]) if lan else []
        if lan is None or not server or tuple(server) != (lan["host"], lan["port"]):
            return _fail(403, "This page only answers on this computer.")
        if request.headers.get("host") != f"{lan['host']}:{lan['port']}":
            return _fail(403, "This page only answers on this computer.")
        if request.url.path == "/pair" and request.method == "GET":
            return self._pair(request, lan)
        offered = request.cookies.get(PHONE_COOKIE, "").encode()
        here = _client_ip(request)
        paired = False
        for session, address in sessions:  # every one compared, in constant time
            paired |= secrets.compare_digest(session.encode(), offered) and address == here
        if not offered or not here or not paired:
            return _fail(403, NOT_PAIRED)
        if not _lan_allowed(routes, request.scope):
            return _fail(403, COMPUTER_ONLY)
        return None

    def api_phone(self, request: Request) -> Response:
        return JSONResponse(self.phone_status(), headers={"Cache-Control": "no-store"})

    async def api_phone_start(self, request: Request) -> Response:
        try:
            status = await run_in_threadpool(self.start_lan)
        except PhoneError as exc:
            return _fail(503, exc.message, exc.detail)
        return JSONResponse(status, headers={"Cache-Control": "no-store"})

    async def api_phone_stop(self, request: Request) -> Response:
        await run_in_threadpool(self.stop_lan)
        return JSONResponse(self.phone_status(), headers={"Cache-Control": "no-store"})

    def api_phone_new_link(self, request: Request) -> Response:
        status = self.new_phone_link()
        if status is None:
            return _fail(409, "Turn on “Use on my phone” first.")
        return JSONResponse(status, headers={"Cache-Control": "no-store"})

    def _path_roots(self) -> list[str]:
        roots = {str(self.root), str(self.root.resolve()), str(Path.home())}
        return sorted((r for r in roots if len(r) > 1), key=len, reverse=True)

    async def _for_phone(self, response: Response) -> Response:
        """Every JSON answer to a phone, with file-system paths taken out (see hide_paths)."""
        if not response.headers.get("content-type", "").startswith("application/json"):
            return response
        body = b"".join([chunk async for chunk in response.body_iterator])
        try:
            data = json.loads(body)
        except ValueError:
            data = None
        headers = {k: v for k, v in response.headers.items()
                   if k.lower() not in ("content-length", "content-type")}
        if data is None:
            return Response(body, status_code=response.status_code, headers=headers,
                            media_type=response.media_type)
        return JSONResponse(hide_paths(data, self._path_roots()),
                            status_code=response.status_code, headers=headers)

    # -------------------------------------------------------------- app

    def app(self) -> Starlette:
        async def same_origin_writes(request: Request, call_next):
            server = request.scope.get("server")
            on_wifi = not server or str(server[0]) not in LOOPBACK_SERVERS
            if on_wifi:
                # Came in on the wifi listener: see LAN_ALLOWED.
                # app.router.routes is read live: a route added later is refused too.
                refusal = self._wifi_refusal(request, app.router.routes)
                if refusal is not None:
                    return refusal
            # Only requests addressed to this computer by one of its own
            # names are answered at all (see LOCAL_HOSTS).
            elif not _is_local(request.headers.get("host")):
                return _fail(403, "This page only answers on this computer.")
            # Another website open in the browser must not be able to change
            # the selection by posting to this local address.
            if request.method not in ("GET", "HEAD"):
                origin = request.headers.get("origin")
                if origin and origin.split("://", 1)[-1] != request.headers.get("host"):
                    return _fail(403, "Cross-origin requests are not allowed.")
            response = await call_next(request)
            if on_wifi:
                response = await self._for_phone(response)
            return response

        app = Starlette(
            middleware=[Middleware(BaseHTTPMiddleware, dispatch=same_origin_writes)],
            routes=[
                Route("/", self.page),
                Route("/api/frames", self.api_frames),
                Route("/api/frames/{frame_id}/photos", self.api_photos),
                Route("/api/frames/{frame_id}/selection", self.api_set_selection, methods=["PUT"]),
                Route("/api/frames/{frame_id}/photos/{asset_id}/faces", self.api_faces),
                Route("/api/frames/{frame_id}/feedback", self.api_feedback, methods=["POST"]),
                Route("/api/frames/{frame_id}/search", self.api_search, methods=["POST"]),
                Route("/api/frames/{frame_id}/search-text", self.api_search_text, methods=["POST"]),
                Route("/api/account", self.api_account),
                Route("/api/account/sign-in", self.api_sign_in, methods=["POST"]),
                Route("/api/account/sign-out", self.api_sign_out, methods=["POST"]),
                Route("/api/account/frames", self.api_account_frames),
                Route("/api/sync", self.api_sync, methods=["POST"]),
                Route("/api/sync/status", self.api_sync_status),
                Route("/api/setup", self.api_setup),
                Route("/api/setup/face-model", self.api_model_download, methods=["POST"]),
                Route("/api/setup/face-model", self.api_model_status),
                # From this computer only (not in LAN_ALLOWED): about 610 MB.
                Route("/api/setup/search-model", self.api_search_model_download, methods=["POST"]),
                Route("/api/setup/search-model", self.api_search_model_status),
                Route("/api/frames/{frame_id}/removed", self.api_removed),
                Route("/removed-thumb/{frame_id}/{asset_id}.jpg", self.removed_thumb),
                Route("/api/restore", self.api_restore, methods=["POST"]),
                Route("/api/restore/status", self.api_restore_status),
                Route("/api/version", self.api_version),
                Route(
                    "/api/frames/{frame_id}/removal/preview",
                    self.api_removal_preview,
                    methods=["POST"],
                ),
                Route("/api/removal", self.api_removal, methods=["POST"]),
                Route("/api/removal/status", self.api_removal_status),
                Route("/api/people", self.api_people),
                Route("/api/people/build", self.api_people_build, methods=["POST"]),
                Route("/api/people/status", self.api_people_status),
                Route("/api/people/merge", self.api_merge_people, methods=["POST"]),
                Route("/api/people/unmerge", self.api_unmerge, methods=["POST"]),
                Route("/api/people/not-same", self.api_not_same, methods=["POST"]),
                Route("/api/people/{person_id}/suggestions", self.api_suggestions),
                Route("/api/people/{person_id}/faces", self.api_group_faces),
                Route("/api/people/move-faces", self.api_move_faces, methods=["POST"]),
                Route("/face/{asset_id}/{index:int}.jpg", self.face_crop),
                Route("/api/face/{asset_id}/{index:int}", self.api_face_box),
                Route("/api/people/{person_id}", self.api_edit_person, methods=["PATCH"]),
                Route("/people/{person_id}/avatar.jpg", self.person_avatar),
                Route("/api/references", self.api_add_reference, methods=["POST"]),
                Route("/api/references/{ref_id}", self.api_delete_reference, methods=["DELETE"]),
                Route("/reference/{ref_id}.jpg", self.reference_image),
                Route("/thumb/{frame_id}/{asset_id}.jpg", self.thumb),
                Route("/photo/{frame_id}/{asset_id}.jpg", self.photo),
                # Using it from a phone: from this computer only (not in LAN_ALLOWED).
                Route("/api/phone", self.api_phone),
                Route("/api/phone/start", self.api_phone_start, methods=["POST"]),
                Route("/api/phone/stop", self.api_phone_stop, methods=["POST"]),
                Route("/api/phone/new-link", self.api_phone_new_link, methods=["POST"]),
            ],
        )
        self._asgi = app
        return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Local page for reviewing cached frame photos.")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    from . import migrations

    try:
        migrations.upgrade(store.ROOT)
    except migrations.UpgradeFailed as exc:
        raise SystemExit(str(exc)) from exc
    review = Review(store.ROOT)
    frames = review.frames()
    if not frames:
        print("No photos here yet. Open the page to sign in and get them.")
    for frame in frames:
        print(f"{frame['name']}: {frame['photos']} photos")
    # Load the face model in the background so the first face lookup is quick,
    # but only if it is already here: fetching it waits for the user's yes.
    review.warm_model()
    print(f"\nOpen http://127.0.0.1:{args.port}")
    try:
        uvicorn.run(review.app(), host="127.0.0.1", port=args.port, log_level="warning")
    finally:
        review.stop_lan()  # the phone listener closes with the page
        review.finish_work()  # a removal under way ends its photo before the app does


if __name__ == "__main__":
    main()

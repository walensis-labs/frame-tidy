"""MCP server exposing Aura frame cleanup tools.

Workflow: ``list_frames`` -> ``sync_frame_photos`` -> ``find_person_in_frame``
-> (human reviews ./flagged) -> ``delete_assets``.

The tool functions are thin wrappers; the logic lives in the ``*_impl``
functions below so it can be tested without a live Aura account or the
face model.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterator, Literal

import numpy as np
from dotenv import load_dotenv
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from . import image_search, people, people_index
from .aura_client import AuraAPIError, AuraAuthError, AuraClient, AuraError, AuraRefusedError
from .face_clusters import CLUSTER_THRESHOLD
from .face_matcher import FaceMatcher, FaceMatcherError, ImageReadError
from .image_search import ImageSearcher, ImageSearchError
from .people import references_dir
from .storage import read_json as _read_json
from .storage import write_json as _write_json

Progress = Callable[[float, float | None, str | None], Awaitable[None]]

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
EMBEDDING_SAVE_EVERY = 100


def _project_root() -> Path:
    """Where ``.env``, ``cache/`` and ``flagged/`` live.

    Defaults to the repo root (this file is ``<root>/src/aura_mcp/server.py``),
    falling back to the working directory for non-editable installs.
    """
    override = os.environ.get("AURA_MCP_HOME")
    if override:
        return Path(override).expanduser().resolve()
    root = Path(__file__).resolve().parents[2]
    return root if (root / "pyproject.toml").is_file() else Path.cwd()


ROOT = _project_root()
load_dotenv(ROOT / ".env")

mcp = MCPServer(
    "aura-frames",
    instructions=(
        "Tools for removing photos of one person from Aura digital frames. "
        "Order: list_frames, sync_frame_photos, find_person_in_frame, then "
        "delete_assets. To see who is on the frames without a reference "
        "photo, run find_people and then list_people. "
        "Deletion is permanent: always show the user the "
        "flagged photos and get explicit confirmation before calling "
        "delete_assets with confirm=true."
    ),
)

_client: AuraClient | None = None
_matcher: FaceMatcher | None = None
_searcher: ImageSearcher | None = None


def session_path(root: Path) -> Path:
    return root / "session.json"


def load_session(root: Path) -> dict[str, str] | None:
    """The sign-in saved by the review page, or None."""
    data = _read_json(session_path(root), None)
    if isinstance(data, dict) and data.get("token") and data.get("user_id"):
        return {"token": str(data["token"]), "user_id": str(data["user_id"])}
    return None


def save_session(root: Path, session: dict[str, str]) -> None:
    """Keep the token Aura issued (never the password), readable only by this user."""
    path = session_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".session.{uuid.uuid4().hex[:8]}.part")
    kept = {"token": str(session["token"]), "user_id": str(session["user_id"])}
    try:
        with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as handle:
            json.dump(kept, handle)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def clear_session(root: Path) -> None:
    session_path(root).unlink(missing_ok=True)


def _env_login() -> tuple[str, str]:
    return os.environ.get("AURA_EMAIL", ""), os.environ.get("AURA_PASSWORD", "")


def signed_in(root: Path) -> bool:
    """Whether there is anything to reach Aura with: a login in .env or a saved sign-in."""
    return all(_env_login()) or load_session(root) is not None


def new_client(
    root: Path | None = None, email: str | None = None, password: str | None = None
) -> AuraClient:
    """A client for the login in ``.env``, the given login, or the saved sign-in."""
    env_email, env_password = _env_login()
    typed = email is not None
    # The base URL overrides exist to point the server at the local mock
    # (see mock_aura.py); unset, the real Aura API is used.
    return AuraClient(
        email if typed else env_email,
        (password or "") if typed else env_password,
        session=None if typed else load_session(ROOT if root is None else root),
        api_base_url=os.environ.get("AURA_API_BASE_URL"),
        image_base_url=os.environ.get("AURA_IMAGE_BASE_URL"),
    )


def get_client() -> AuraClient:
    global _client
    if _client is None:
        _client = new_client()
    return _client


def get_matcher() -> FaceMatcher:
    global _matcher
    if _matcher is None:
        _matcher = FaceMatcher()
    return _matcher


def get_searcher() -> ImageSearcher:
    global _searcher
    if _searcher is None:
        _searcher = ImageSearcher()
    return _searcher


# ------------------------------------------------------------------ helpers


def _check_id(value: str, label: str) -> str:
    """IDs become file names, so refuse anything that could escape the folder."""
    value = str(value).strip()
    if not _SAFE_ID.match(value):
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


def cache_dir(root: Path, frame_id: str) -> Path:
    return root / "cache" / frame_id


def flagged_dir(root: Path, frame_id: str) -> Path:
    return root / "flagged" / frame_id


def thumbs_dir(root: Path, frame_id: str) -> Path:
    return cache_dir(root, frame_id) / ".thumbs"


def removed_dir(root: Path, frame_id: str) -> Path:
    """Local copies of photos taken off a frame, kept so they can be put back."""
    return root / "removed" / frame_id


def selection_path(root: Path, frame_id: str) -> Path:
    return root / "selections" / f"{frame_id}.json"


def save_frame_names(root: Path, frames: list[dict[str, Any]]) -> None:
    """Remember frame names so the review page can show them, not bare IDs."""
    names = _read_json(root / "cache" / "frames.json", {})
    names.update({str(f["id"]): f.get("name") for f in frames if f.get("id")})
    _write_json(root / "cache" / "frames.json", names)


def load_index(root: Path, frame_id: str) -> dict[str, dict[str, Any]]:
    path = cache_dir(root, frame_id) / "index.json"
    if not path.is_file():
        raise ValueError(
            f"Frame {frame_id} has not been synced yet. Run sync_frame_photos first."
        )
    return json.loads(path.read_text())


def _load_embeddings(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    try:
        with np.load(path) as data:
            return {key: data[key] for key in data.files}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile):
        return {}  # corrupt cache: just recompute


def _save_embeddings(path: Path, embeddings: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Its own name (ending .npz, or numpy appends one), so a search and a
    # build saving the same frame at once cannot publish each other's half file.
    tmp = path.with_name(f"embeddings.{uuid.uuid4().hex[:8]}.part.npz")
    try:
        np.savez(tmp, **embeddings)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


async def _report(progress: Progress | None, done: int, total: int, message: str) -> None:
    if progress is None:
        return
    try:
        await progress(done, total, message)
    except Exception:
        pass  # progress is best-effort; never fail the tool over it


# ------------------------------------------------------- plain-word errors

SIGNED_OUT = "Your Aura sign-in has expired. Sign in again."
UNREACHABLE = (
    "Aura could not be reached. Check this computer's internet connection and try again."
)
AURA_TROUBLE = "Aura is having trouble right now. Try again in a few minutes."
REFUSED = "Aura refused the request."
UNEXPECTED = (
    "Aura sent back an answer this app did not expect. Aura may have changed how it "
    "works, so this app may need an update."
)
LOCAL_TROUBLE = (
    "Something went wrong saving files on this computer. Check that there is free disk space."
)
OTHER_TROUBLE = "Something unexpected went wrong. Try again; if it keeps happening, the details below may help."


def explain_error(exc: BaseException) -> dict[str, str]:
    """Turn a failure into a sentence a person can act on.

    Returns ``kind`` (signed_out, unreachable, aura_trouble, refused,
    unexpected, local, plain, other), ``message`` (the sentence) and
    ``detail`` (the technical text, kept for anyone who wants it; never the
    only thing shown). Messages this app writes itself (``ValueError``, face
    model errors) are already plain and pass through.
    """
    import httpx

    detail = str(exc) or type(exc).__name__
    if isinstance(exc, AuraAuthError):
        # Already the same sentence: nothing more to fold away.
        return {"kind": "signed_out", "message": SIGNED_OUT, "detail": "" if detail == SIGNED_OUT else detail}
    if isinstance(exc, AuraError):
        status = getattr(exc, "status_code", None)
        if isinstance(exc.__cause__, httpx.TransportError):
            kind, message = "unreachable", UNREACHABLE
        elif status is not None and status >= 500:
            kind, message = "aura_trouble", AURA_TROUBLE
        elif isinstance(exc, AuraRefusedError) or (status is not None and 400 <= status < 500):
            kind, message = "refused", REFUSED
        else:
            kind, message = "unexpected", UNEXPECTED
        return {"kind": kind, "message": message, "detail": detail}
    if isinstance(exc, (ValueError, FaceMatcherError, ImageSearchError)):
        cause = exc.__cause__
        return {"kind": "plain", "message": str(exc), "detail": str(cause) if cause else ""}
    if isinstance(exc, OSError):
        return {"kind": "local", "message": LOCAL_TROUBLE, "detail": detail}
    return {"kind": "other", "message": OTHER_TROUBLE, "detail": f"{type(exc).__name__}: {detail}"}


# ------------------------------------------------------------- tool logic


SYNC_MAX_SIDE = 1600  # long side of the copies kept for browsing and faces
SYNC_CONCURRENCY = 12  # photos downloaded at once
# What an index entry's "size" holds when the copy may be medium-size but
# nothing says so for certain (a load that stopped before writing the index).
SIZE_UNKNOWN = "unknown"

PhotoLanded = Callable[[str, Path], Awaitable[None]]


def is_original(entry: dict[str, Any]) -> bool:
    """Whether an index entry's local copy is the original file.

    A medium copy carries ``"size"``; caches made before medium copies
    existed have no ``"size"`` and hold originals.
    """
    return "size" not in entry


class _NotAnOriginal(Exception):
    """What Aura sent for the original failed check_original."""


def _whole_file(data: bytes) -> bool:
    """False for a JPEG or PNG that was cut off before its end."""
    if data[:3] == b"\xff\xd8\xff":
        # The last scan must be followed by the end-of-image marker. Inside
        # the scan data a 0xFF byte is always escaped, so it cannot fake one.
        last_scan = data.rfind(b"\xff\xda")
        return last_scan != -1 and data.find(b"\xff\xd9", last_scan) != -1
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return b"IEND" in data
    return True  # anything else: decoding it decides


def image_long_side(path: Path) -> int | None:
    """The long side of the photo in ``path``, in pixels; None if it is not a whole photo."""
    import cv2

    try:
        data = path.read_bytes()
    except OSError:
        return None
    if not data or not _whole_file(data):
        return None
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        return None
    return int(max(image.shape[:2]))


def check_original(fetched: Path, here: list[Path]) -> str | None:
    """Why ``fetched`` cannot be kept as the original, or None if it can.

    It must decode as a whole photo, and its long side must be at least that
    of every readable copy of it already on this computer: Aura's image host
    never enlarges, so a true original is never smaller than a medium copy
    made from it.
    """
    side = image_long_side(fetched)
    if side is None:
        return "what Aura sent back is not a whole photo"
    have = max((s for s in (image_long_side(p) for p in here if p.is_file()) if s), default=0)
    if side < have:
        return (
            "what Aura sent back is smaller than the copy already on this computer, "
            "so it is not the full-size photo"
        )
    return None


def _link_or_copy(source: Path, dest: Path) -> None:
    """Put ``source`` at ``dest``: a hard link where the disk allows, else a copy."""
    tmp = dest.with_name(f"{dest.name}.{uuid.uuid4().hex[:8]}.part")
    try:
        try:
            os.link(source, tmp)
        except OSError:
            shutil.copy2(source, tmp)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


async def sync_frame_photos_impl(
    client: AuraClient,
    root: Path,
    frame_id: str,
    progress: Progress | None = None,
    on_photo: PhotoLanded | None = None,
) -> dict[str, Any]:
    """Get every photo on a frame into the local cache.

    New photos are downloaded as medium-size copies (``SYNC_MAX_SIDE``),
    ``SYNC_CONCURRENCY`` at a time, newest first. ``on_photo(asset_id, path)``
    is awaited as each photo is in place, so its faces can be read while the
    rest are still arriving.
    """
    frame_id = _check_id(frame_id, "frame_id")
    assets = await client.list_assets(frame_id)
    folder = cache_dir(root, frame_id)
    folder.mkdir(parents=True, exist_ok=True)

    # What each photo already here is (original or medium), from the last sync.
    previous = _read_json(folder / "index.json", {})
    previous = previous if isinstance(previous, dict) else {}
    downloaded = already_cached = reused = 0
    failed: list[dict[str, str]] = []
    # The same photo is usually on several frames, under the same id.
    elsewhere = [p for p in folder.parent.iterdir() if p.is_dir() and p != folder]
    elsewhere_index: dict[Path, dict[str, Any]] = {}
    for other in elsewhere:
        other_index = _read_json(other / "index.json", {})
        elsewhere_index[other] = other_index if isinstance(other_index, dict) else {}

    todo: list[tuple[str, dict[str, Any]]] = []
    for item in sorted(assets, key=lambda a: str(a.get("taken_at") or ""), reverse=True):
        try:
            todo.append((_check_id(item["id"], "asset id"), item))
        except ValueError as exc:
            failed.append({"asset_id": str(item.get("id")), "error": str(exc)})
    entries: dict[str, dict[str, Any]] = {}
    finished = 0
    queue = iter(todo)

    async def fetch(asset_id: str, item: dict[str, Any]) -> None:
        nonlocal downloaded, already_cached, reused, finished
        file_name, user_id = item.get("file_name"), item.get("user_id")
        dest = folder / f"{asset_id}.jpg"
        size: Any = None
        if dest.is_file() and dest.stat().st_size > 0:
            already_cached += 1
            size = _known_size(previous.get(asset_id))
        elif (source := _reuse_photo(elsewhere, asset_id, dest)) is not None:
            reused += 1
            size = _known_size(elsewhere_index.get(source.parent, {}).get(asset_id))
        elif not file_name or not user_id:
            failed.append({"asset_id": asset_id, "error": "asset has no file_name/user_id"})
        else:
            error: AuraAPIError | None = None
            for _ in range(2):  # one retry
                try:
                    await client.download_photo(user_id, file_name, dest, max_side=SYNC_MAX_SIDE)
                except AuraAPIError as exc:
                    error = exc
                else:
                    error = None
                    break
            if error is None:
                downloaded += 1
                size = SYNC_MAX_SIDE
            else:
                failed.append({"asset_id": asset_id, "error": str(error)})

        landed = dest.is_file()
        entry = {
            "file_name": file_name,
            "user_id": user_id,
            "taken_at": item.get("taken_at"),
            "local_path": str(dest) if landed else None,
        }
        if landed and size is not None:
            entry["size"] = size
        entries[asset_id] = entry
        if landed and on_photo is not None:
            await on_photo(asset_id, dest)
        finished += 1
        await _report(progress, finished, len(assets), f"Synced {finished}/{len(assets)} photos")

    async def worker() -> None:
        for asset_id, item in queue:
            await fetch(asset_id, item)

    workers = [asyncio.create_task(worker()) for _ in range(max(1, SYNC_CONCURRENCY))]
    try:
        await asyncio.gather(*workers)
    except BaseException:
        for task in workers:
            task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        raise

    index = {asset_id: entries[asset_id] for asset_id, _ in todo if asset_id in entries}

    # Photos that have left the frame since the last sync no longer belong in
    # the cache, so they can't be flagged later. A copy the ledger does not
    # know about may be the only one left anywhere (someone took the photo
    # off with the Aura app, or a removal here was cut short), so it is moved
    # to removed/, never deleted.
    pruned = _settle_left_frame(root, frame_id, folder, index, previous)

    _write_json(folder / "index.json", index)
    return {
        "frame_id": frame_id,
        "total_synced": sum(1 for entry in index.values() if entry["local_path"]),
        "total_assets": len(assets),
        "downloaded": downloaded,
        "already_cached": already_cached,
        "reused": reused,
        "pruned_stale": pruned,
        "failed": failed,
        "cache_dir": str(folder),
    }


def _known_size(before: Any) -> Any:
    """The "size" to record for a copy already here, from its last index entry.

    The entry is trusted only if it had a file then: an entry written after
    a failed download says nothing about the file that landed since, which
    may well be a medium copy.
    """
    if isinstance(before, dict) and before.get("local_path"):
        return before.get("size")
    return SIZE_UNKNOWN


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ledger_copy(kept: Path, asset_id: str, entry: Any) -> Path | None:
    """The kept file a ledger entry points at, if it is there (never outside ``kept``)."""
    candidates = [kept / f"{asset_id}.jpg"]
    if isinstance(entry, dict) and entry.get("local_path"):
        candidates.append(Path(str(entry["local_path"])))
    for path in candidates:
        try:
            if path.is_file() and path.resolve().is_relative_to(kept.resolve()):
                return path
        except OSError:
            continue
    return None


def _settle_left_frame(
    root: Path,
    frame_id: str,
    folder: Path,
    index: dict[str, dict[str, Any]],
    previous: dict[str, Any],
) -> int:
    """Square the cache and the ledger with what is on the frame now.

    - A removal that was cut short (ledger state "removing"): if the photo is
      still on the frame, Aura never acted, so the entry and its kept copy go;
      if it is gone, the entry is marked removed.
    - A cached photo no longer on the frame: dropped from the cache if the
      ledger already keeps a copy of it, else moved to removed/ with an entry
      saying it was found missing from the frame during a load.

    Returns how many cached photos left the cache.
    """
    kept = removed_dir(root, frame_id)
    log = _read_json(kept / "removed.json", {})
    log = log if isinstance(log, dict) else {}
    changed = False
    for asset_id, entry in list(log.items()):
        if not isinstance(entry, dict) or entry.get("state") != "removing":
            continue
        if asset_id in index:
            copy = _ledger_copy(kept, asset_id, entry)
            if copy is not None:
                copy.unlink(missing_ok=True)
            del log[asset_id]
        else:
            entry["state"] = "removed"
            entry.setdefault("removed_at", entry.get("started_at"))
        changed = True

    pruned = 0
    for stale in sorted(folder.glob("*.jpg")):
        asset_id = stale.stem
        if asset_id in index:
            continue
        pruned += 1
        if asset_id in log and _ledger_copy(kept, asset_id, log[asset_id]) is not None:
            stale.unlink()
            continue
        kept.mkdir(parents=True, exist_ok=True)
        dest = kept / f"{asset_id}.jpg"
        if dest.exists():
            dest = kept / f"{asset_id}.found-{uuid.uuid4().hex[:8]}.jpg"
        os.replace(stale, dest)
        before = previous.get(asset_id)
        before = before if isinstance(before, dict) else {}
        entry = {
            "file_name": before.get("file_name"),
            "user_id": before.get("user_id"),
            "taken_at": before.get("taken_at"),
            "local_path": str(dest),
            "state": "missing",
            "found_missing_at": _now(),
        }
        size = _known_size(before) if before else SIZE_UNKNOWN
        if size is not None:
            entry["size"] = size
        log[asset_id] = entry
        changed = True
    if changed:
        _write_json(kept / "removed.json", log)
    return pruned


def _reuse_photo(elsewhere: list[Path], asset_id: str, dest: Path) -> Path | None:
    """Take a photo another frame's cache already holds instead of downloading it.

    A hard link where the disk allows (no extra space), else a copy. Each
    frame's file can be moved or deleted without touching the other's.
    Returns the file it was taken from, or None.
    """
    for other in elsewhere:
        source = other / f"{asset_id}.jpg"
        if not source.is_file() or source.stat().st_size == 0:
            continue
        try:
            _link_or_copy(source, dest)
        except OSError:
            continue
        return source
    return None


class _FaceReader:
    """Reads the faces in one frame's photos as they are handed over.

    Several photos are read at once when the matcher says that helps
    (``FaceMatcher.reads_at_once``). Faces are kept in ``embeddings.npz``,
    saved every ``EMBEDDING_SAVE_EVERY`` new photos and on ``finish``.
    ``known`` holds faces already read from the same photos on other frames,
    which are taken as they are. ``on_settled(asset_id)`` is awaited once
    per photo handed over, read or not.
    """

    def __init__(
        self,
        matcher: FaceMatcher,
        folder: Path,
        known: dict[str, np.ndarray] | None = None,
        on_settled: Callable[[str], Awaitable[None]] | None = None,
    ):
        self.matcher = matcher
        self.path = folder / "embeddings.npz"
        self.embeddings = _load_embeddings(self.path)
        self.known = known
        self.on_settled = on_settled
        self.unreadable: list[str] = []
        self._pending: dict[asyncio.Task[np.ndarray], str] = {}
        self._unsaved = 0

    async def add(self, asset_id: str, path: Path) -> None:
        if asset_id in self.embeddings:
            await self._settled(asset_id)
            return
        if self.known and asset_id in self.known:
            self.embeddings[asset_id] = self.known[asset_id]
            self._unsaved += 1
            await self._settled(asset_id)
            return
        while len(self._pending) >= max(1, int(getattr(self.matcher, "reads_at_once", 1))):
            await self._next()
        task = asyncio.create_task(asyncio.to_thread(self.matcher.face_embeddings, path))
        self._pending[task] = asset_id

    async def _next(self) -> None:
        finished, _ = await asyncio.wait(self._pending, return_when=asyncio.FIRST_COMPLETED)
        for task in finished:
            asset_id = self._pending.pop(task)
            try:
                self.embeddings[asset_id] = task.result()
                self._unsaved += 1
            except ImageReadError:
                self.unreadable.append(asset_id)
            if self._unsaved >= EMBEDDING_SAVE_EVERY:
                self.save()
            await self._settled(asset_id)

    async def _settled(self, asset_id: str) -> None:
        if self.on_settled is not None:
            await self.on_settled(asset_id)

    async def finish(self) -> None:
        while self._pending:
            await self._next()
        if self._unsaved:
            self.save()

    def save(self) -> None:
        _save_embeddings(self.path, self.embeddings)
        self._unsaved = 0

    def abandon(self) -> None:
        """Stop waiting for reads in progress, keeping what was read."""
        for task in self._pending:
            task.cancel()
        self._pending.clear()
        if self._unsaved:
            self.save()


async def scan_frame_faces(
    matcher: FaceMatcher,
    folder: Path,
    index: dict[str, dict[str, Any]],
    progress: Progress | None = None,
    known: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], list[tuple[str, Path]], list[str]]:
    """Face embeddings for every cached photo of a frame, scanning the new ones.

    Embeddings depend only on the photo, so they are kept between runs in
    ``embeddings.npz``: another search, threshold or grouping never rescans.
    ``known`` holds faces already read from the same photos on other frames,
    which are taken as they are. Returns the embeddings, the photos that exist
    on disk, and the ids of photos that could not be read.
    """
    photos = [
        (asset_id, Path(entry["local_path"]))
        for asset_id, entry in index.items()
        if entry.get("local_path") and Path(entry["local_path"]).is_file()
    ]
    settled = 0

    async def report(asset_id: str) -> None:
        nonlocal settled
        settled += 1
        if settled % 10 == 0 or settled == len(photos):
            await _report(progress, settled, len(photos), f"Scanned {settled}/{len(photos)} photos")

    reader = _FaceReader(matcher, folder, known, report)
    try:
        for asset_id, path in photos:
            await reader.add(asset_id, path)
        await reader.finish()
    except BaseException:
        reader.abandon()
        raise
    order = {asset_id: i for i, (asset_id, _) in enumerate(photos)}
    unreadable = sorted(reader.unreadable, key=order.__getitem__)
    return reader.embeddings, photos, unreadable


GROUP_FIRST_AT = 300  # photos read before people are first grouped during a load
GROUP_EVERY = 500  # and then again after about this many more


async def load_frames_impl(
    client: AuraClient,
    matcher: FaceMatcher,
    root: Path,
    frame_ids: list[str],
    progress: Progress | None = None,
    threshold: float = CLUSTER_THRESHOLD,
) -> dict[str, Any]:
    """Get the photos of several frames and work out who is in them.

    Each photo's faces are read as soon as it lands, while the rest are still
    downloading, and the people are grouped after the first
    ``GROUP_FIRST_AT`` photos and then about every ``GROUP_EVERY``, so there
    is something to show long before the load ends.
    """
    frame_ids = [_check_id(f, "frame_id") for f in frame_ids]
    names = frame_names(root)
    cache = root / "cache"
    # Faces read so far, by photo: a photo on three frames is read once.
    known: dict[str, np.ndarray] = {}
    for folder in sorted(p for p in cache.iterdir() if p.is_dir()) if cache.is_dir() else []:
        known.update(_load_embeddings(folder / "embeddings.npz"))

    counts = {f: {"got": 0, "total": 0, "read": 0} for f in frame_ids}
    readers: dict[str, _FaceReader] = {}
    current: str | None = None  # the frame whose photos are being downloaded
    read_in_load = 0
    next_grouping = GROUP_FIRST_AT

    def label(frame: str) -> str:
        return names.get(frame) or frame

    async def say(frame: str) -> None:
        c = counts[frame]
        await _report(
            progress, c["got"], c["total"],
            f"Getting photos from {label(frame)}: {c['got']:,} of {c['total']:,}"
            f" · faces read in {c['read']:,}",
        )

    def settled_in(frame: str) -> Callable[[str], Awaitable[None]]:
        async def settled(asset_id: str) -> None:
            nonlocal read_in_load, next_grouping
            counts[frame]["read"] += 1
            read_in_load += 1
            if current in (frame, None):
                await say(frame)
            if read_in_load >= next_grouping:
                next_grouping = read_in_load + GROUP_EVERY
                for reader in readers.values():
                    reader.save()
                await asyncio.to_thread(people_index.rebuild, root, threshold)

        return settled

    # One queue for every frame, in order, so reading a frame's photos can
    # run on while the next frame downloads. None ends a frame.
    queue: asyncio.Queue[tuple[str, str | None, Path | None]] = asyncio.Queue()

    async def read_faces() -> None:
        while True:
            frame, asset_id, path = await queue.get()
            if asset_id is None or path is None:
                await readers[frame].finish()
                known.update(readers[frame].embeddings)
                if frame == frame_ids[-1]:
                    return
                continue
            await readers[frame].add(asset_id, path)

    reading = asyncio.create_task(read_faces())
    loaded = []
    try:
        for frame_id in frame_ids:
            current = frame_id
            readers[frame_id] = _FaceReader(
                matcher, cache_dir(root, frame_id), known, settled_in(frame_id)
            )

            async def landed(asset_id: str, path: Path, frame: str = frame_id) -> None:
                if reading.done():
                    reading.result()  # reading faces failed: stop the load with its error
                queue.put_nowait((frame, asset_id, path))

            async def step(done: float, total: float | None, message: str | None,
                           frame: str = frame_id) -> None:
                counts[frame]["got"], counts[frame]["total"] = int(done), int(total or 0)
                await say(frame)

            await _report(progress, 0, 0, f"Getting photos from {label(frame_id)}")
            result = await sync_frame_photos_impl(client, root, frame_id, step, landed)
            queue.put_nowait((frame_id, None, None))
            loaded.append({
                "id": frame_id,
                "name": label(frame_id),
                "photos": result["total_synced"],
                "failed": len(result["failed"]),
            })
        current = None
        if frame_ids:
            await reading
        else:
            reading.cancel()
    except BaseException:
        reading.cancel()
        with contextlib.suppress(BaseException):
            await reading
        for reader in readers.values():
            reader.abandon()
        raise

    async def faces(done: float, total: float | None, message: str | None) -> None:
        await _report(progress, done, total or 0, message or "Looking at faces")

    # The final grouping, after one more try at any photo that could not be
    # read (and any frame loaded earlier whose faces were never read).
    found = await find_people_impl(matcher, root, faces, threshold)
    return {"frames": loaded, "people": found["people"], "seen_once": found["seen_once"]}


async def find_person_in_frame_impl(
    matcher: FaceMatcher,
    root: Path,
    frame_id: str,
    reference_image_path: str = "",
    similarity_threshold: float = 0.45,
    progress: Progress | None = None,
    person: str = "",
) -> dict[str, Any]:
    frame_id = _check_id(frame_id, "frame_id")
    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between -1 and 1.")
    folder = cache_dir(root, frame_id)
    sync_complete = (folder / "index.json").is_file()
    if sync_complete:
        index = load_index(root, frame_id)
    else:
        # A sync that is still running has photos on disk but no index yet.
        # Search what has arrived so far rather than making the user wait.
        index = {
            p.stem: {"file_name": None, "taken_at": None, "local_path": str(p)}
            for p in sorted(folder.glob("*.jpg"))
        }
        if not index:
            load_index(root, frame_id)  # raises the "run sync first" error

    # References plus, for a saved person, everything the user has confirmed
    # or rejected so far.
    model = await asyncio.to_thread(
        people.build_model, matcher, root, reference_image_path, person
    )

    embeddings, photos, unreadable = await scan_frame_faces(matcher, folder, index, progress)

    # Start the review folder fresh so it only ever shows this run's matches.
    review = flagged_dir(root, frame_id)
    review.mkdir(parents=True, exist_ok=True)
    for old in review.glob("*.jpg"):
        old.unlink()

    matches: list[dict[str, Any]] = []
    photos_with_faces = 0
    for asset_id, path in photos:
        faces = embeddings.get(asset_id)
        if faces is None or len(faces) == 0:
            continue
        photos_with_faces += 1
        key = people.photo_key(frame_id, asset_id)
        if key in model.rejected or asset_id in model.rejected_assets:
            continue  # the user already said this photo is not them
        confirmed = key in model.confirmed
        # A photo the user confirmed, or that the grouping of faces gave to
        # this person, stays flagged whatever it scores.
        keep = confirmed or asset_id in model.member_assets
        scored = model.best(faces)
        if keep and (scored is None or scored[0] < similarity_threshold):
            scored = model.best(faces, apply_rejections=False)
        if scored is None or not (keep or scored[0] >= similarity_threshold):
            continue
        similarity, face_index = scored
        flagged_path = review / f"{asset_id}.jpg"
        shutil.copy2(path, flagged_path)
        match = {
            "asset_id": asset_id,
            "file_name": index[asset_id].get("file_name"),
            "similarity": round(similarity, 4),
            "flagged_path": str(flagged_path),
            "face_index": face_index,
        }
        if confirmed:
            match["confirmed"] = True
        matches.append(match)
    matches.sort(key=lambda m: m["similarity"], reverse=True)
    # Read by the review page: the flagged photos, their scores, and which
    # face in each one matched (needed to record a yes/no answer about it).
    # "opt_out": the matches are taken to be the person unless the user says
    # "not them". "opt_in": each match needs a "yes" to count as confirmed.
    mode = "opt_out" if model.assume_match else "opt_in"
    _write_json(
        review / "matches.json",
        {
            "person": model.slug,
            "person_name": model.name,
            "similarity_threshold": similarity_threshold,
            "review_mode": mode,
            "matches": matches,
        },
    )

    result: dict[str, Any] = {
        "frame_id": frame_id,
        "similarity_threshold": similarity_threshold,
        "review_mode": mode,
        "photos_scanned": len(photos) - len(unreadable),
        "photos_with_faces": photos_with_faces,
        "match_count": len(matches),
        "person": model.name,
        "examples_used": {
            "is_them": len(model.positives),
            "not_them": len(model.negatives),
        },
        "flagged_dir": str(review),
        "matches": matches,
        "unreadable_asset_ids": unreadable,
    }
    if not sync_complete:
        result["note"] = (
            "The sync for this frame has not finished, so only the photos "
            "downloaded so far were searched. Run this again when it completes."
        )
    if model.reference_faces > 1:
        result["warning"] = (
            f"The reference image contains {model.reference_faces} faces; the largest "
            "one was used. Crop it to just the target person for reliable results."
        )
    return result


async def find_people_impl(
    matcher: FaceMatcher,
    root: Path,
    progress: Progress | None = None,
    threshold: float = CLUSTER_THRESHOLD,
) -> dict[str, Any]:
    """Work out who is in the cached photos of every synced frame.

    Scans any photo whose faces have not been read yet, then groups all the
    faces into people (see people_index). People already known keep their id,
    name and hidden flag.
    """
    cache = root / "cache"
    names = _read_json(cache / "frames.json", {})
    folders = sorted(
        p for p in (cache.iterdir() if cache.is_dir() else [])
        if p.is_dir() and (p / "index.json").is_file()
    )
    # Faces read so far, by photo: a photo on three frames is read once.
    known: dict[str, np.ndarray] = {}
    for folder in folders:
        known.update(_load_embeddings(folder / "embeddings.npz"))
    for folder in folders:
        label = names.get(folder.name) or folder.name

        async def report(done: float, total: float | None, message: str | None) -> None:
            await _report(progress, done, total or 0, f"{label}: {message}")

        embeddings, _, _ = await scan_frame_faces(
            matcher, folder, load_index(root, folder.name), report, known
        )
        known.update(embeddings)

    await _report(progress, 0, 0, "Grouping faces into people")
    data = await asyncio.to_thread(people_index.rebuild, root, threshold)
    shown = people_index.gallery(root)
    return {
        "frames_scanned": len(folders),
        "built_at": data["built_at"],
        "people": len(shown["people"]),
        "seen_once": len(shown["seen_once"]),
        "hidden": len(shown["hidden"]),
    }


def list_people_impl(root: Path) -> dict[str, Any]:
    """Everyone found by find_people: who has a tile, the one-offs, the hidden."""
    return people_index.gallery(root)


CLIP_DEFAULT_THRESHOLD = 0.22  # CLIP cosines sit low; this is not the 0.45 of faces
CLIP_BATCH = 64


async def find_photos_by_description_impl(
    searcher: ImageSearcher,
    root: Path,
    frame_id: str,
    query: str,
    *,
    similarity_threshold: float = CLIP_DEFAULT_THRESHOLD,
    top_k: int = 100,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Find cached photos that match a plain-language description, with CLIP.

    Open-vocabulary counterpart to :func:`find_person_in_frame_impl`: instead of
    a face it scores each photo against a phrase ("a desk", "a birthday cake").
    Image embeddings are cached in clip_embeddings.npz so re-querying the frame
    with a different description never rescans it.
    """
    frame_id = _check_id(frame_id, "frame_id")
    query = (query or "").strip()
    if not query:
        raise ValueError("Provide a description to search for.")
    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between -1 and 1.")

    folder = cache_dir(root, frame_id)
    sync_complete = (folder / "index.json").is_file()
    if sync_complete:
        index = load_index(root, frame_id)
    else:
        index = {
            p.stem: {"file_name": None, "local_path": str(p)}
            for p in sorted(folder.glob("*.jpg"))
        }
        if not index:
            load_index(root, frame_id)  # raises the "run sync first" error

    embeddings_path = folder / "clip_embeddings.npz"
    embeddings = _load_embeddings(embeddings_path)

    photos = [
        (asset_id, Path(entry["local_path"]))
        for asset_id, entry in index.items()
        if entry.get("local_path") and Path(entry["local_path"]).is_file()
    ]
    todo = [(asset_id, path) for asset_id, path in photos if asset_id not in embeddings]
    unreadable: list[str] = []
    new_since_save = 0
    for start in range(0, len(todo), CLIP_BATCH):
        batch = todo[start:start + CLIP_BATCH]
        try:
            vectors = await asyncio.to_thread(
                searcher.image_embeddings, [path for _, path in batch]
            )
            for (asset_id, _), vector in zip(batch, vectors):
                embeddings[asset_id] = vector
        except Exception:  # noqa: BLE001 - one bad file shouldn't lose the batch
            for asset_id, path in batch:
                try:
                    embeddings[asset_id] = await asyncio.to_thread(
                        searcher.image_embedding, path
                    )
                except image_search.ImageReadError:
                    unreadable.append(asset_id)
        new_since_save += len(batch)
        if new_since_save >= EMBEDDING_SAVE_EVERY:
            _save_embeddings(embeddings_path, embeddings)
            new_since_save = 0
        scanned = min(start + len(batch), len(todo))
        await _report(progress, scanned, len(todo), f"Scanned {scanned}/{len(todo)} new photos")
    if new_since_save:
        _save_embeddings(embeddings_path, embeddings)

    query_vector = await asyncio.to_thread(searcher.query_embedding, query)
    matches: list[dict[str, Any]] = []
    for asset_id, _ in photos:
        vector = embeddings.get(asset_id)
        if vector is None:
            continue
        score = float(np.asarray(vector, dtype=np.float32) @ query_vector)
        if score >= similarity_threshold:
            matches.append({
                "asset_id": asset_id,
                "file_name": index[asset_id].get("file_name"),
                "similarity": round(score, 4),
            })
    matches.sort(key=lambda m: m["similarity"], reverse=True)
    if top_k and top_k > 0:
        matches = matches[:top_k]

    review = flagged_dir(root, frame_id)
    review.mkdir(parents=True, exist_ok=True)
    # Kept separate from the person flow's matches.json so the two searches
    # never clobber each other.
    _write_json(
        review / "description.json",
        {"query": query, "similarity_threshold": similarity_threshold, "matches": matches},
    )

    result: dict[str, Any] = {
        "frame_id": frame_id,
        "query": query,
        "similarity_threshold": similarity_threshold,
        "photos_scanned": len(photos) - len(unreadable),
        "match_count": len(matches),
        "matches": matches,
        "unreadable_asset_ids": unreadable,
    }
    if not sync_complete:
        result["note"] = (
            "The sync for this frame has not finished, so only the photos "
            "downloaded so far were searched. Run this again when it completes."
        )
    return result


DeleteScope = Literal["frame", "all_frames", "account"]

SCOPE_EFFECT = {
    "frame": (
        "Removes each photo from this frame only. It stays on any other frame "
        "it was added to; a photo that is on no other frame is gone for good."
    ),
    "all_frames": (
        "Removes each photo from every frame it is on, among the frames that "
        "have been synced here. A photo left on no frame is gone from Aura "
        "for good; the copy kept on this computer is then the only one."
    ),
    "account": (
        "Permanently deletes each photo from the whole Aura account, which "
        "removes it from every frame it is on."
    ),
}

CONFIRMATION_MESSAGE = (
    "Nothing has been deleted. This cannot be undone. Show the user this list "
    "(the photos are in the flagged folder for review), tell them which scope "
    "will be used and what it does, and ask for explicit confirmation. If "
    "there are warnings, read them to the user before asking. Only call "
    "delete_assets again with confirm=true after they clearly agree."
)


def review_status(root: Path, frame_id: str) -> tuple[str | None, dict[str, str]]:
    """How each photo stands in the last search of this frame for a saved person.

    Returns the person's name and ``{asset_id: status}`` for every photo that
    search matched: "confirmed" (the user said it is them), "not_them" (the
    user said it is not) or "unanswered". Photos the search did not match are
    absent. The name is None if no saved person has been searched for.
    """
    last = _read_json(flagged_dir(root, frame_id) / "matches.json", {})
    if not isinstance(last, dict) or not last.get("person"):
        return None, {}
    answers = people.labels(root, last["person"], frame_id)
    names = {people.YES: "confirmed", people.NO: "not_them"}
    statuses = {
        m["asset_id"]: names.get(answers.get(m["asset_id"]), "unanswered")
        for m in last.get("matches", [])
    }
    if last.get("review_mode") == "opt_out":
        # "Not them" said on another frame is about the same photo here.
        for asset_id in people.rejected_assets(root, last["person"]):
            statuses[asset_id] = "not_them"
    return last.get("person_name"), statuses


def review_mode(root: Path, frame_id: str) -> str:
    """How the last search of this frame is reviewed: "opt_out" or "opt_in".

    Opt-out (a person found by grouping faces): a match counts as them unless
    the user marked it "not them". Opt-in: a match counts only once confirmed.
    """
    last = _read_json(flagged_dir(root, frame_id) / "matches.json", {})
    mode = last.get("review_mode") if isinstance(last, dict) else None
    return "opt_out" if mode == "opt_out" else "opt_in"


def removal_warnings(
    name: str | None, statuses: list[str], assume_match: bool = False
) -> list[str]:
    """Plain-language cautions about photos that were not confirmed as the person.

    With ``assume_match`` (an opt-out review) a match left unanswered is taken
    to be the person, so it draws no caution.
    """
    if not name:
        return []

    def photos(n: int) -> str:
        return "1 photo was" if n == 1 else f"{n} photos were"

    not_them, unanswered, by_hand = (
        statuses.count(s) for s in ("not_them", "unanswered", "not_matched")
    )
    warnings = []
    if not_them:
        warnings.append(f"{photos(not_them)} marked as NOT {name} in the review page.")
    if unanswered and not assume_match:
        warnings.append(
            f"{photos(unanswered)} matched as {name} but never confirmed in the review page."
        )
    if by_hand:
        warnings.append(
            f"{photos(by_hand)} not matched as {name} by the search (selected by hand)."
        )
    return warnings


def synced_frames(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """``{frame_id: index}`` for every frame whose sync has finished."""
    cache = root / "cache"
    folders = sorted(p for p in cache.iterdir() if p.is_dir()) if cache.is_dir() else []
    found = {}
    for folder in folders:
        index = _read_json(folder / "index.json", None)
        if isinstance(index, dict):
            found[folder.name] = index
    return found


def frame_names(root: Path) -> dict[str, str]:
    """Every frame Aura has listed for this account, synced here or not."""
    names = _read_json(root / "cache" / "frames.json", {})
    return {str(k): v for k, v in names.items()} if isinstance(names, dict) else {}


async def delete_assets_impl(
    client: AuraClient,
    root: Path,
    frame_id: str,
    asset_ids: list[str],
    scope: str = "frame",
    confirm: bool = False,
    progress: Progress | None = None,
    previewed: dict[str, list[str]] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Preview a removal, or with ``confirm=True`` carry it out.

    ``previewed`` is what an earlier preview listed, ``{asset_id: frames}``.
    With it, the run touches only those photos on those frames, whatever has
    been synced or selected since: the user confirmed that list and no other.
    ``should_stop`` is asked before each photo; when it says yes the rest are
    reported as not attempted (the app is closing).

    For each photo, in this order: a full-size copy is put in every target
    frame's ``removed/`` folder (fetched from Aura and checked, unless the
    copy here is already the original); its ledger entry is written with
    ``"state": "removing"``; Aura is asked; each frame it came off is marked
    ``"removed"`` and its cache copy dropped; a frame Aura said no for gets
    its entry (and a copy this run made) taken back out.
    """
    frame_id = _check_id(frame_id, "frame_id")
    if scope not in SCOPE_EFFECT:
        raise ValueError(
            f"scope must be 'frame', 'all_frames' or 'account', not {scope!r}."
        )
    index = load_index(root, frame_id)
    requested = list(dict.fromkeys(str(a).strip() for a in asset_ids))

    # Only photos that were synced from this frame are eligible. Anything else
    # (a typo, an ID from another frame) is reported and never sent to Aura.
    known = [a for a in requested if a in index]
    unknown = [a for a in requested if a not in index]
    # A photo is only removed if there is a copy of it here to keep: Aura has
    # no trash, and that copy is the one way back.
    folder = cache_dir(root, frame_id)
    no_copy = [a for a in known if not (folder / f"{a}.jpg").is_file()]
    known = [a for a in known if a not in no_copy]
    if previewed is not None:
        known = [a for a in known if a in previewed]
    review = flagged_dir(root, frame_id)
    person_name, reviewed = review_status(root, frame_id)

    # The same photo has the same id on every frame it is on.
    others = {
        other: other_index
        for other, other_index in synced_frames(root).items()
        if other != frame_id
    }
    also_on = {a: [f for f, other_index in others.items() if a in other_index] for a in known}
    everywhere = scope != "frame"
    # The frames each photo comes off, this frame first.
    targets = {a: [frame_id, *(also_on[a] if everywhere else [])] for a in known}
    if previewed is not None:
        targets = {a: [f for f in frames if f in previewed[a]] for a, frames in targets.items()}
        known = [a for a in known if targets[a]]

    def describe(asset_id: str) -> dict[str, Any]:
        flagged = review / f"{asset_id}.jpg"
        entry = {
            "asset_id": asset_id,
            "file_name": index[asset_id].get("file_name"),
            "taken_at": index[asset_id].get("taken_at"),
            "flagged_path": str(flagged) if flagged.is_file() else None,
        }
        if person_name:
            entry["review"] = reviewed.get(asset_id, "not_matched")
        return entry

    if confirm is not True:
        would_delete = [
            {
                **describe(a),
                "frames": targets[a],
                "stays_on": [] if everywhere else also_on[a],
            }
            for a in known
        ]
        names = frame_names(root)
        counts: dict[str, int] = {}
        for a in known:
            for frame in targets[a]:
                counts[frame] = counts.get(frame, 0) + 1
        preview = {
            "status": "confirmation_required",
            "scope": scope,
            "effect": SCOPE_EFFECT[scope],
            "deleted_count": 0,
            "would_delete_count": len(known),
            "would_delete": would_delete,
            "frame_removal_count": sum(counts.values()),
            "frames_affected": [
                {"id": frame, "name": names.get(frame) or frame, "count": count}
                for frame, count in counts.items()
            ],
            # Photos that will be on no frame synced here afterwards. Aura
            # keeps a photo only while it is on a frame.
            "last_copy_count": sum(1 for entry in would_delete if not entry["stays_on"]),
            "unsynced_frames": [
                {"id": frame, "name": name or frame}
                for frame, name in names.items()
                if frame != frame_id and frame not in others
            ],
            "unknown_asset_ids": unknown,
            "no_copy_asset_ids": no_copy,
            "message": CONFIRMATION_MESSAGE,
        }
        if person_name:
            statuses = [entry["review"] for entry in would_delete]
            preview["reviewed_for"] = person_name
            preview["confirmed_count"] = statuses.count("confirmed")
            opt_out = review_mode(root, frame_id) == "opt_out"
            if opt_out:
                # Matched and not marked "not them": taken to be the person.
                preview["review_mode"] = "opt_out"
                preview["assumed_count"] = statuses.count("unanswered")
            preview["warnings"] = removal_warnings(person_name, statuses, assume_match=opt_out)
        return preview

    # What is kept on this computer about each frame, loaded when first touched.
    local: dict[str, dict[str, Any]] = {}

    def frame_state(frame: str) -> dict[str, Any]:
        if frame not in local:
            folder = cache_dir(root, frame)
            kept = removed_dir(root, frame)
            log = _read_json(kept / "removed.json", {})
            local[frame] = {
                "folder": folder,
                "index": index if frame == frame_id else others[frame],
                "embeddings": _load_embeddings(folder / "embeddings.npz"),
                "kept": kept,
                "log": log if isinstance(log, dict) else {},
                "changed": False,
            }
        return local[frame]

    def save(frame: str) -> None:
        """Write one frame's ledger and index now, so a run that dies part-way
        leaves them matching the files and what Aura was asked."""
        state = local[frame]
        _write_json(state["kept"] / "removed.json", state["log"])
        if state["changed"]:
            _write_json(state["folder"] / "index.json", state["index"])

    def not_done(asset_id: str, frame: str, error: str, reason: str | None = None) -> dict[str, str]:
        # Naming the frame only matters when one photo spans several.
        entry = {"asset_id": asset_id, "frame_id": frame} if scope == "all_frames" else {"asset_id": asset_id}
        entry["error"] = error  # the raw text, for the record
        if reason:
            entry["reason"] = reason  # the same in plain words, for the page
        return entry

    def needs_original(asset_id: str) -> bool:
        """Whether this frame's copy is not known to be the original file."""
        entry = index.get(asset_id, {})
        return not (entry.get("local_path") and is_original(entry))

    async def fetch_original(asset_id: str) -> Path:
        """Download the full photo and check it, before any removal.

        Raises ``AuraError``/``OSError`` if it could not be fetched, and
        ``_NotAnOriginal`` if what came back is not a whole photo at least as
        large as the copy already here. A bad file is deleted.
        """
        entry = index[asset_id]
        file_name, user_id = entry.get("file_name"), entry.get("user_id")
        if not file_name or not user_id:
            raise AuraAPIError("Aura's listing did not say where the full-size photo is.")
        folder = removed_dir(root, targets[asset_id][0])
        dest = folder / f"{asset_id}.original-{uuid.uuid4().hex[:8]}.jpg"
        try:
            await client.download_photo(user_id, file_name, dest)
            here = [cache_dir(root, frame) / f"{asset_id}.jpg" for frame in targets[asset_id]]
            problem = await asyncio.to_thread(check_original, dest, here)
        except BaseException:
            dest.unlink(missing_ok=True)
            raise
        if problem:
            dest.unlink(missing_ok=True)
            raise _NotAnOriginal(problem)
        return dest

    async def keep_original(asset_id: str) -> tuple[dict[str, Path], set[str]]:
        """Put a full-size copy in each target frame's removed folder.

        Returns the kept file per frame, and the frames whose kept file this
        run created (there was none before). Raises as fetch_original does,
        or ``OSError`` if a copy could not be put in place; then nothing is
        left behind.
        """
        fetched = await fetch_original(asset_id) if needs_original(asset_id) else None
        source = fetched or cache_dir(root, frame_id) / f"{asset_id}.jpg"
        kept: dict[str, Path] = {}
        created: set[str] = set()
        try:
            for frame in targets[asset_id]:
                dest = frame_state(frame)["kept"] / f"{asset_id}.jpg"
                dest.parent.mkdir(parents=True, exist_ok=True)
                existed = dest.exists()
                _link_or_copy(source, dest)
                kept[frame] = dest
                if not existed:
                    created.add(frame)
        except OSError:
            for frame in created:
                kept[frame].unlink(missing_ok=True)
            raise
        finally:
            if fetched is not None:
                fetched.unlink(missing_ok=True)  # each frame has its own link or copy now
        return kept, created

    def started(asset_id: str, kept: dict[str, Path]) -> dict[str, Any]:
        """Write "removal started" into each target frame's ledger, before Aura is asked.

        Returns the entries they held before, to put back if Aura says no.
        """
        before = {}
        for frame, path in kept.items():
            state = frame_state(frame)
            before[frame] = state["log"].get(asset_id)
            entry = state["index"].get(asset_id) or index[asset_id]
            state["log"][asset_id] = {
                "file_name": entry.get("file_name"),
                "user_id": entry.get("user_id"),
                "taken_at": entry.get("taken_at"),
                "local_path": str(path),
                "scope": scope,
                "state": "removing",
                "started_at": _now(),
            }
            save(frame)
        return before

    def finished(frame: str, asset_id: str) -> None:
        """Aura took the photo off this frame: mark it removed and drop the cache copy."""
        state = frame_state(frame)
        state["log"][asset_id] = {**state["log"][asset_id], "state": "removed", "removed_at": _now()}
        for stale in (
            state["folder"] / f"{asset_id}.jpg",
            flagged_dir(root, frame) / f"{asset_id}.jpg",
            thumbs_dir(root, frame) / f"{asset_id}.jpg",
        ):
            stale.unlink(missing_ok=True)
        state["index"].pop(asset_id, None)
        state["embeddings"].pop(asset_id, None)
        state["changed"] = True
        save(frame)

    def take_back(frame: str, asset_id: str, before: Any, made_copy: bool) -> None:
        """Aura was not asked, or said no: undo the "started" entry on this frame."""
        state = frame_state(frame)
        if before is None:
            state["log"].pop(asset_id, None)
        else:
            state["log"][asset_id] = before
        if made_copy:
            (state["kept"] / f"{asset_id}.jpg").unlink(missing_ok=True)
        save(frame)

    deleted: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    aborted: str | None = None
    aborted_reason = ""
    frame_removals = 0

    def give_up_from(position: int, reason: str, plain: str) -> None:
        """Report every photo from ``position`` on as not attempted."""
        if scope == "all_frames":
            failed.extend(
                not_done(later, frame, reason, plain)
                for later in known[position:]
                for frame in targets[later]
            )
        else:
            failed.extend(not_done(later, frame_id, reason, plain) for later in known[position:])

    try:
        for position, asset_id in enumerate(known):
            if should_stop is not None and should_stop():
                aborted = "the app was closing."
                aborted_reason = "Not attempted: the app was closed before this photo."
                give_up_from(position, "not attempted: " + aborted, aborted_reason)
                break
            info = describe(asset_id)
            done: list[str] = []
            pending = list(targets[asset_id])
            # Aura has no trash: a full-size copy must be kept here, and the
            # ledger must say so, before Aura is asked to remove anything.
            try:
                kept, created = await keep_original(asset_id)
            except (AuraError, OSError, _NotAnOriginal) as exc:
                if isinstance(exc, _NotAnOriginal):
                    because = str(exc)
                    plain_cause = f"What Aura sent back {because.removeprefix('what Aura sent back ')}."
                else:
                    because = str(exc)
                    plain_cause = explain_error(exc)["message"]
                reason = (
                    "Not removed: the full-size photo could not be saved on this "
                    f"computer first, and it would be the only way back ({because})"
                )
                plain = (
                    "Not removed, because the full-size photo could not be saved on "
                    "this computer first, and it would be the only way back. " + plain_cause
                )
                if scope == "all_frames":
                    failed.extend(not_done(asset_id, frame, reason, plain) for frame in pending)
                else:
                    failed.append(not_done(asset_id, frame_id, reason, plain))
                await _report(
                    progress, position + 1, len(known),
                    f"Removed {position + 1}/{len(known)} photos",
                )
                continue
            before = started(asset_id, kept)
            try:
                if scope == "account":
                    await client.delete_asset(asset_id)
                    done, pending = pending, []
                else:
                    while pending:
                        frame = pending[0]
                        try:
                            await client.remove_asset_from_frame(frame, asset_id)
                        except AuraAuthError:
                            raise
                        except AuraAPIError as exc:
                            failed.append(
                                not_done(asset_id, frame, str(exc), explain_error(exc)["message"]))
                        else:
                            done.append(frame)
                            finished(frame, asset_id)
                        pending.pop(0)
            except AuraAuthError as exc:
                # No point hammering the API once the session is gone.
                aborted = str(exc)
                aborted_reason = "Not attempted: " + explain_error(exc)["message"]
            except AuraAPIError as exc:
                failed.append(not_done(asset_id, frame_id, str(exc), explain_error(exc)["message"]))

            if scope == "account":
                for frame in done:
                    finished(frame, asset_id)
            # Frames Aura said no for, or was never asked about: as before.
            for frame in targets[asset_id]:
                if frame not in done:
                    take_back(frame, asset_id, before[frame], frame in created)
            if done:
                info["local_copy"] = str(kept[done[0]])
                info["frames"] = done
                # Frames it was meant to come off and did not.
                info["still_on"] = [f for f in targets[asset_id] if f not in done]
                frame_removals += len(done)
                deleted.append(info)
            if aborted:
                reason = "not attempted: " + aborted
                if scope == "all_frames":
                    failed.extend(not_done(asset_id, frame, reason, aborted_reason) for frame in pending)
                    give_up_from(position + 1, reason, aborted_reason)
                else:
                    give_up_from(position, reason, aborted_reason)
                break
            await _report(
                progress, position + 1, len(known), f"Removed {position + 1}/{len(known)} photos"
            )
    finally:
        # The ledger and index are written as each photo goes (see save); the
        # face cache is large, so it is saved once here, even if the run was
        # interrupted.
        for state in local.values():
            if state["changed"] and (state["folder"] / "embeddings.npz").is_file():
                _save_embeddings(state["folder"] / "embeddings.npz", state["embeddings"])

    partial = sum(1 for entry in deleted if entry["still_on"])
    return {
        "status": "aborted" if aborted else "completed",
        "scope": scope,
        # Photos now off every frame they were meant to come off.
        "deleted_count": len(deleted) - partial,
        # Photos that came off some of those frames and are still on others.
        "partial_count": partial,
        "frame_removal_count": frame_removals,
        "failed_count": len(failed),
        "deleted": deleted,
        "failed": failed,
        "unknown_asset_ids": unknown,
        "no_copy_asset_ids": no_copy,
    }


async def restore_assets_impl(
    client: AuraClient,
    root: Path,
    frame_id: str,
    asset_ids: list[str],
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Put photos that left a frame back on it.

    Works for every ledger entry: removed through this tool, a removal that
    was cut short ("removing": Aura may or may not have acted), and a photo
    found missing from the frame during a load ("missing"). The kept copy
    takes the place of any copy still in the cache. A frame with no index
    here does not get one. ``should_stop`` is asked before each photo.
    """
    frame_id = _check_id(frame_id, "frame_id")
    kept = removed_dir(root, frame_id)
    removed_log = _read_json(kept / "removed.json", {})
    removed_log = removed_log if isinstance(removed_log, dict) else {}
    requested = list(dict.fromkeys(str(a).strip() for a in asset_ids))
    unknown = [a for a in requested if a not in removed_log]
    folder = cache_dir(root, frame_id)
    had_index = (folder / "index.json").is_file()
    index = _read_json(folder / "index.json", {}) if had_index else {}
    restored: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []

    try:
        for asset_id in requested:
            if asset_id in unknown:
                continue
            if should_stop is not None and should_stop():
                break
            entry = removed_log[asset_id]
            entry = entry if isinstance(entry, dict) else {}
            copy = _ledger_copy(kept, asset_id, entry)
            try:
                await client.add_asset_to_frame(frame_id, asset_id)
            except AuraError as exc:
                plain = explain_error(exc)
                failed.append({
                    "asset_id": asset_id,
                    "error": str(exc),
                    # "refused" almost always means Aura no longer has the photo.
                    "kind": plain["kind"],
                    "reason": plain["message"],
                    "state": entry.get("state") or "removed",
                    "local_copy": str(copy) if copy else entry.get("local_path"),
                })
                continue
            dest = folder / f"{asset_id}.jpg"
            if copy is not None:
                folder.mkdir(parents=True, exist_ok=True)
                os.replace(copy, dest)
            if had_index:
                back = {
                    "file_name": entry.get("file_name"),
                    "user_id": entry.get("user_id"),
                    "taken_at": entry.get("taken_at"),
                    "local_path": str(dest) if dest.is_file() else None,
                }
                if copy is None and dest.is_file():
                    back["size"] = SIZE_UNKNOWN  # the cache copy that was there: not known to be full-size
                elif copy is not None and "size" in entry:
                    back["size"] = entry["size"]
                index[asset_id] = back
            del removed_log[asset_id]
            restored.append({"asset_id": asset_id, "file_name": entry.get("file_name")})
    finally:
        if restored:
            _write_json(kept / "removed.json", removed_log)
            if had_index:
                _write_json(folder / "index.json", index)

    return {
        "restored_count": len(restored),
        "failed_count": len(failed),
        "restored": restored,
        "failed": failed,
        "unknown_asset_ids": unknown,
    }


def get_selection_impl(root: Path, frame_id: str) -> dict[str, Any]:
    frame_id = _check_id(frame_id, "frame_id")
    folder = cache_dir(root, frame_id)
    index = _read_json(folder / "index.json", {})
    chosen = _read_json(selection_path(root, frame_id), {}).get("asset_ids", [])
    selected = [
        {
            "asset_id": asset_id,
            "file_name": index.get(asset_id, {}).get("file_name"),
            "taken_at": index.get(asset_id, {}).get("taken_at"),
        }
        for asset_id in chosen
        if (folder / f"{asset_id}.jpg").is_file()
    ]
    references = [{"name": r["name"], "path": r["path"]} for r in people.references(root)]
    return {
        "frame_id": frame_id,
        "selected_count": len(selected),
        "selected": selected,
        "references": references,
        "people": people.summary(root),
    }


# -------------------------------------------------------------- MCP tools


@contextlib.contextmanager
def _readable_errors() -> Iterator[None]:
    """Pass expected failures to the caller with their message intact.

    The SDK hides the text of any exception that isn't a ``ToolError`` (the
    model would only see "Error executing tool ..."), which would swallow
    useful messages like a rejected login or an unsynced frame.
    """
    try:
        yield
    except (AuraError, FaceMatcherError, ImageSearchError, ValueError) as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def list_frames() -> list[dict[str, Any]]:
    """List the Aura frames on the account: id, name and asset_count."""
    with _readable_errors():
        frames = await get_client().list_frames()
        save_frame_names(ROOT, frames)
        return frames


@mcp.tool()
async def sync_frame_photos(frame_id: str, ctx: Context) -> dict[str, Any]:
    """Download every photo on a frame into the local cache.

    Photos already cached are skipped, so this is safe to re-run (and to
    resume if a large frame is interrupted). Writes cache/{frame_id}/index.json.
    The cache holds medium-size copies (1600px), several downloaded at once:
    about 0.1 s per new photo. delete_assets fetches the full original before
    removing a photo, and keeps that.
    """
    with _readable_errors():
        return await sync_frame_photos_impl(get_client(), ROOT, frame_id, ctx.report_progress)


@mcp.tool()
async def find_person_in_frame(
    frame_id: str,
    ctx: Context,
    reference_image_path: str = "",
    similarity_threshold: float = 0.45,
    person: str = "",
) -> dict[str, Any]:
    """Find cached frame photos that contain a person.

    Say who to look for with either `person` or `reference_image_path` (any
    photo of them). `person` is the name or id of someone from list_people,
    or the name of someone saved in the review page (see get_selection). For
    a known person the search also uses every match the user has confirmed
    or rejected in the review page, which makes it more accurate, especially
    at telling relatives apart.

    For someone from list_people the result has review_mode "opt_out": the
    matches are taken to be them unless the user marks one "not them" in the
    review page, so they need no one-by-one confirmation.

    Runs local face recognition (nothing is uploaded). The first face lookup
    in this server downloads InsightFace's face model (buffalo_l, about 290 MB;
    InsightFace's terms say its models are available for non-commercial
    research purposes only) into ~/.insightface, or AURA_MODEL_DIR, if it is
    not there yet. A photo matches if any
    face in it reaches similarity_threshold. Matches are copied to
    flagged/{frame_id}/ for the user to review; nothing is deleted. Requires
    sync_frame_photos to have been run for this frame. Lower the threshold to
    catch more photos, raise it to cut false positives.
    """
    with _readable_errors():
        return await find_person_in_frame_impl(
            get_matcher(), ROOT, frame_id, reference_image_path, similarity_threshold,
            ctx.report_progress, person,
        )


@mcp.tool()
async def find_people(ctx: Context) -> dict[str, Any]:
    """Work out who appears in the cached photos, across every synced frame.

    Groups the faces in all cached photos into people, using local face
    recognition (nothing is uploaded). If InsightFace's face model (about
    290 MB; available for non-commercial research purposes only, per
    InsightFace's terms) is not in ~/.insightface (or AURA_MODEL_DIR) yet, it
    is downloaded first. Photos not scanned before are scanned first, a
    fraction of a second each depending on the computer; after that it takes
    seconds. Safe to
    re-run after a sync: people already found keep their id, name and hidden
    flag. Returns how many people were found; list_people shows them.
    """
    with _readable_errors():
        return await find_people_impl(get_matcher(), ROOT, ctx.report_progress)


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def list_people() -> dict[str, Any]:
    """The people found by find_people, most photographed first.

    `people` are those seen in two or more photos (or given a name),
    `seen_once` the one-offs, `hidden` those the user chose to hide. Each has
    an id, a name if one was given in the review page, a photo_count, and
    `frames`: how many of their photos are on each frame. Pass an id or name
    as `person` to find_person_in_frame to find their photos on a frame.
    """
    with _readable_errors():
        return list_people_impl(ROOT)


@mcp.tool()
async def find_photos_in_frame(
    frame_id: str,
    query: str,
    ctx: Context,
    similarity_threshold: float = CLIP_DEFAULT_THRESHOLD,
    top_k: int = 100,
) -> dict[str, Any]:
    """Find cached frame photos matching a plain-language description.

    The first search in this server downloads the text-and-image matching
    model (CLIP ViT-B/32, about 610 MB, MIT-licensed) from Hugging Face into
    ~/.frame-tidy/search-model, or AURA_SEARCH_MODEL_DIR, if it is not there
    yet. It runs only on this computer.

    Open-vocabulary search with a local CLIP model (nothing is uploaded): give
    a phrase like "a desk", "a birthday cake", "a receipt" or "a dog" and it
    returns the photos that best fit, ranked by similarity. Use this to curate
    a frame, not just clean it of a person.

    CLIP scores sit in a narrow band (about 0.15-0.32), far below the face
    model's, so similarity_threshold defaults to 0.22; raise it for tighter
    matches, lower it for more. top_k caps how many are returned (best first).
    Requires sync_frame_photos first. Results are also written to
    flagged/{frame_id}/description.json for the review page; nothing is deleted.
    """
    with _readable_errors():
        return await find_photos_by_description_impl(
            get_searcher(), ROOT, frame_id, query,
            similarity_threshold=similarity_threshold, top_k=top_k,
            progress=ctx.report_progress,
        )


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def get_selection(frame_id: str) -> dict[str, Any]:
    """Read what the user picked in the local review page (`uv run aura-review`).

    Returns the photos they selected on this frame, and the reference faces
    they saved (name and image path). A reference path can be passed to
    find_person_in_frame; selected asset IDs can be passed to delete_assets,
    which still requires the usual preview and explicit confirmation.
    """
    with _readable_errors():
        return get_selection_impl(ROOT, frame_id)


@mcp.tool(annotations=ToolAnnotations(destructive_hint=True))
async def delete_assets(
    frame_id: str,
    asset_ids: list[str],
    scope: DeleteScope = "frame",
    confirm: bool = False,
) -> dict[str, Any]:
    """Remove photos from a frame, or delete them from the whole account.

    scope="frame" (the default) takes each photo off this frame only; it stays
    on any other frame it is on. scope="all_frames" takes each photo off every
    synced frame it is on, one frame at a time; the preview lists those frames
    and any frame that was never synced and so was not checked. scope="account"
    permanently deletes the photo from the Aura account in one call. Use
    "all_frames" or "account" only when the user asks for the photos to be
    gone everywhere. Aura keeps a photo only while it is on a frame, so a
    photo taken off its last frame is gone for good.

    With confirm=false (the default) nothing is deleted: the tool returns the
    list that would be affected. Show that list to the user, tell them the
    scope, and get their explicit confirmation before calling again with
    confirm=true. Never set confirm=true on your own initiative. No scope
    can be undone.
    """
    with _readable_errors():
        return await delete_assets_impl(
            get_client(), ROOT, frame_id, asset_ids, scope, confirm
        )


@mcp.tool()
async def restore_assets(frame_id: str, asset_ids: list[str]) -> dict[str, Any]:
    """Put photos back on a frame after delete_assets removed them.

    Works for photos removed through this server, by re-linking the existing
    Aura photo to the frame (nothing is uploaded). That only succeeds while
    Aura still has the photo, which in practice means it is on another frame;
    a photo that was on no other frame, or was deleted with scope="account",
    is gone from Aura. In that case the report gives the path of the local
    copy that was kept, which the user can add back through the Aura app.
    """
    with _readable_errors():
        return await restore_assets_impl(get_client(), ROOT, frame_id, asset_ids)


def main() -> None:
    from . import migrations

    try:
        migrations.upgrade(ROOT)
    except migrations.UpgradeFailed as exc:
        raise SystemExit(str(exc)) from exc
    mcp.run()


if __name__ == "__main__":
    main()

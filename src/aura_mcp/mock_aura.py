"""A local stand-in for the Aura API, for testing without touching a real account.

Point it at a folder where each subfolder is a frame and each image inside is
a photo on that frame::

    mock/photos/
        Living Room/   <- one frame
            a.jpg
            b.jpg
        Kitchen/       <- another frame
            a.jpg      <- same bytes as Living Room/a.jpg = one photo on two frames

Run ``uv run aura-mock`` and start the MCP server with ``AURA_API_BASE_URL``
and ``AURA_IMAGE_BASE_URL`` pointing here (see README). It mimics the request
and response shapes seen from the real API: login, frame list, cursor-paged
asset list, image download, frame removal and account deletion.

Removals only change in-memory state. Files on disk are never touched, and
restarting (or ``POST /mock/reset``) brings every photo back.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import secrets
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
# Aura's image host serves a resized copy when the file name carries these
# options (see AuraClient.image_url).
_RESIZED = re.compile(
    r"^rotate_(?P<rotate>\d+)__width_(?P<width>\d+)__height_(?P<height>\d+)"
    r"__rt_fit__quality_(?P<quality>\d+)__(?P<name>.+)$"
)
MOCK_USER_ID = "mock-user-1"
_NAMESPACE = uuid.UUID("6f1d3c52-6f0b-4b7e-9a51-3a0d1f1e2a77")


def _error(status: int, message: str, **extra: Any) -> JSONResponse:
    # Same shape as the real API: {"error": true, "message": "..."}
    return JSONResponse({"error": True, "message": message, **extra}, status_code=status)


class MockAura:
    """In-memory Aura account built from a folder of photos."""

    def __init__(self, photos_dir: Path, page_size: int = 50):
        self.photos_dir = Path(photos_dir)
        self.page_size = page_size
        self.tokens: set[str] = set()
        self.assets: dict[str, dict[str, Any]] = {}
        self.files: dict[str, Path] = {}  # served file_name -> path on disk
        self.frame_names: dict[str, str] = {}
        self.frames: dict[str, list[str]] = {}  # frame id -> asset ids
        self.load()

    def load(self) -> None:
        """(Re)build the account from disk, undoing any removals."""
        self.assets.clear()
        self.files.clear()
        self.frame_names.clear()
        self.frames.clear()
        if not self.photos_dir.is_dir():
            raise FileNotFoundError(f"Photos folder not found: {self.photos_dir}")

        for folder in sorted(p for p in self.photos_dir.iterdir() if p.is_dir()):
            frame_id = str(uuid.uuid5(_NAMESPACE, f"frame:{folder.name}"))
            self.frame_names[frame_id] = folder.name
            self.frames[frame_id] = []
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                digest = hashlib.md5(path.read_bytes()).hexdigest()
                # Identity comes from the bytes, so the same photo in two
                # frame folders is one asset that sits on both frames.
                asset_id = str(uuid.uuid5(_NAMESPACE, f"asset:{digest}"))
                if asset_id not in self.assets:
                    file_name = path.name
                    if file_name in self.files:  # same name, different photo
                        file_name = f"{digest[:12]}-{path.name}"
                    self.files[file_name] = path
                    taken = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                    self.assets[asset_id] = {
                        "id": asset_id,
                        "user_id": MOCK_USER_ID,
                        "file_name": file_name,
                        "taken_at": taken.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                        "md5_hash": digest,
                    }
                if asset_id not in self.frames[frame_id]:
                    self.frames[frame_id].append(asset_id)

    # ---------------------------------------------------------- operations

    def remove_from_frame(self, frame_id: str, asset_id: str) -> bool:
        if asset_id not in self.frames.get(frame_id, []):
            return False
        self.frames[frame_id].remove(asset_id)
        # Like the real service: a photo left on no frame is gone.
        if not any(asset_id in ids for ids in self.frames.values()):
            self.assets.pop(asset_id, None)
        return True

    def add_to_frame(self, frame_id: str, asset_id: str) -> bool:
        # Only an asset that still exists somewhere can be linked to a frame.
        if asset_id not in self.assets or asset_id in self.frames.get(frame_id, []):
            return False
        self.frames[frame_id].append(asset_id)
        return True

    def delete_from_account(self, asset_id: str) -> bool:
        if asset_id not in self.assets:
            return False
        for ids in self.frames.values():
            if asset_id in ids:
                ids.remove(asset_id)
        del self.assets[asset_id]
        return True

    def state(self) -> dict[str, Any]:
        return {
            "total_assets": len(self.assets),
            "frames": [
                {"id": fid, "name": self.frame_names[fid], "num_assets": len(ids)}
                for fid, ids in self.frames.items()
            ],
        }

    # ------------------------------------------------------------ handlers

    def _auth_failure(self, request: Request) -> JSONResponse | None:
        if request.headers.get("X-User-Id") != MOCK_USER_ID:
            return _error(400, "Missing or invalid X-User-Id")
        if request.headers.get("X-Token-Auth") not in self.tokens:
            return _error(401, "Request Unauthenticated", logout=True)
        return None

    async def login(self, request: Request) -> Response:
        try:
            user = (await request.json()).get("user") or {}
        except ValueError:
            user = {}
        # Any email/password is accepted, except the password "wrong", which
        # reproduces the real API's rejected-login response.
        if not user.get("email") or not user.get("password") or user["password"] == "wrong":
            return _error(475, "The email or password was incorrect.")
        token = secrets.token_hex(16)
        self.tokens.add(token)
        return JSONResponse(
            {
                "error": False,
                "result": {
                    "current_user": {
                        "id": MOCK_USER_ID,
                        "auth_token": token,
                        "email": user["email"],
                        "name": "Mock User",
                    }
                },
            }
        )

    async def list_frames(self, request: Request) -> Response:
        if failure := self._auth_failure(request):
            return failure
        return JSONResponse({"frames": self.state()["frames"]})

    async def list_assets(self, request: Request) -> Response:
        if failure := self._auth_failure(request):
            return failure
        frame_id = request.path_params["frame_id"]
        if frame_id not in self.frames:
            return _error(404, "Frame not found")
        try:
            limit = int(request.query_params.get("limit", self.page_size))
            offset = int(request.query_params.get("cursor") or 0)
        except ValueError:
            return _error(400, "Invalid limit or cursor")
        size = max(1, min(limit, self.page_size))
        ids = self.frames[frame_id]
        page = ids[offset : offset + size]
        more = offset + size < len(ids)
        return JSONResponse(
            {
                "assets": [self.assets[a] for a in page],
                "next_page_cursor": str(offset + size) if more else None,
            }
        )

    async def remove_asset(self, request: Request) -> Response:
        if failure := self._auth_failure(request):
            return failure
        frame_id = request.path_params["frame_id"]
        if frame_id not in self.frames:
            return _error(404, "Frame not found")
        try:
            items = (await request.json()).get("assets") or []
        except ValueError:
            return _error(400, "Invalid body")
        failed = sum(
            0 if self.remove_from_frame(frame_id, str(item.get("asset_id"))) else 1
            for item in items
        )
        return JSONResponse({"number_failed": failed})

    async def select_asset(self, request: Request) -> Response:
        if failure := self._auth_failure(request):
            return failure
        frame_id = request.path_params["frame_id"]
        if frame_id not in self.frames:
            return _error(404, "Frame not found")
        try:
            items = (await request.json()).get("assets") or []
        except ValueError:
            return _error(400, "Invalid body")
        failed = sum(
            0 if self.add_to_frame(frame_id, str(item.get("asset_id"))) else 1
            for item in items
        )
        return JSONResponse({"number_failed": failed})

    async def delete_asset(self, request: Request) -> Response:
        if failure := self._auth_failure(request):
            return failure
        if not self.delete_from_account(request.path_params["asset_id"]):
            return _error(404, "Asset not found")
        return JSONResponse({"error": False})

    async def image(self, request: Request) -> Response:
        name = request.path_params["file_name"]
        path = self.files.get(name)
        resized = None if path is not None else _RESIZED.match(name)
        if resized:
            path = self.files.get(resized["name"])
        if request.path_params["user_id"] != MOCK_USER_ID or path is None:
            return Response(status_code=404)
        if resized:
            return Response(
                _fit(path, int(resized["width"]), int(resized["height"]), int(resized["quality"])),
                media_type="image/jpeg",
            )
        return FileResponse(path)

    # Test controls. These have no counterpart in the real API.

    async def mock_state(self, request: Request) -> Response:
        return JSONResponse(self.state())

    async def mock_reset(self, request: Request) -> Response:
        self.load()
        return JSONResponse(self.state())

    async def mock_expire_tokens(self, request: Request) -> Response:
        self.tokens.clear()
        return JSONResponse({"expired": True})

    def app(self) -> Starlette:
        return Starlette(
            routes=[
                Route("/v5/login.json", self.login, methods=["POST"]),
                Route("/v5/frames.json", self.list_frames),
                Route("/v5/frames/{frame_id}/assets.json", self.list_assets),
                Route(
                    "/v5/frames/{frame_id}/remove_asset.json",
                    self.remove_asset,
                    methods=["POST"],
                ),
                Route(
                    "/v5/frames/{frame_id}/select_asset.json",
                    self.select_asset,
                    methods=["POST"],
                ),
                Route("/v5/assets/{asset_id}.json", self.delete_asset, methods=["DELETE"]),
                Route("/img/{user_id}/{file_name}", self.image),
                Route("/mock/state", self.mock_state),
                Route("/mock/reset", self.mock_reset, methods=["POST"]),
                Route("/mock/expire-tokens", self.mock_expire_tokens, methods=["POST"]),
            ]
        )


def _fit(path: Path, width: int, height: int, quality: int) -> bytes:
    """The photo shrunk to fit ``width`` x ``height`` (never enlarged), as JPEG.

    A file that is not a decodable image (the tests use stand-in bytes) is
    served as it is.
    """
    import cv2
    import numpy as np

    data = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return data
    h, w = image.shape[:2]
    scale = min(width / w, height / h, 1.0)
    if scale < 1.0:
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return encoded.tobytes() if ok else data


def main() -> None:
    import uvicorn

    default_photos = Path(__file__).resolve().parents[2] / "mock" / "photos"
    parser = argparse.ArgumentParser(description="Local mock of the Aura Frames API.")
    parser.add_argument("--photos", type=Path, default=default_photos,
                        help="Folder with one subfolder per frame (default: mock/photos)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--page-size", type=int, default=50,
                        help="Max assets per page, kept small to exercise pagination")
    args = parser.parse_args()

    mock = MockAura(args.photos, page_size=args.page_size)
    for frame in mock.state()["frames"]:
        print(f"frame {frame['id']}  {frame['name']}  ({frame['num_assets']} photos)")
    print(f"API:    http://{args.host}:{args.port}/v5")
    print(f"Images: http://{args.host}:{args.port}/img")
    uvicorn.run(mock.app(), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

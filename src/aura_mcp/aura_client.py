"""Async client for the unofficial Aura Frames API (api.pushd.com/v5).

The API is undocumented and reverse-engineered from the mobile app, so parsing
here is deliberately tolerant: unexpected shapes raise ``AuraAPIError`` with a
readable message instead of a bare ``KeyError``.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

API_BASE_URL = "https://api.pushd.com/v5"
IMAGE_BASE_URL = "https://imgproxy.pushd.com"
USER_AGENT = "Aura/4.42.0 (iPhone; iOS 17.0; Scale/3.00)"
DEVICE_ID = "00000000-0000-0000-0000-000000000000"
APP_IDENTIFIER = "com.pushd.Framelord"

# Downloads are not spaced out: callers bound how many run at once instead
# (see SYNC_CONCURRENCY in server.py).
DOWNLOAD_DELAY = 0.0
DELETE_DELAY = 0.3
ASSET_PAGE_LIMIT = 1000
MAX_ASSET_PAGES = 500

# Aura has used more than one name for the pagination cursor over time.
# `next_page_cursor` is what current responses carry; the rest are fallbacks.
CURSOR_KEYS = ("next_page_cursor", "next_page_token", "next_cursor", "cursor")


class AuraError(Exception):
    """Base class for every error raised by this client."""


class AuraAuthError(AuraError):
    """Login failed, or the session could not be re-established."""


class AuraAPIError(AuraError):
    """The API answered with an error status or an unexpected body."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class AuraRefusedError(AuraAPIError):
    """Aura answered, but reported that it did not make the change asked for
    (``number_failed`` on an otherwise successful reply)."""


def _error_detail(response: httpx.Response) -> str:
    """Pull a short human-readable reason out of an error response."""
    try:
        body = response.json()
    except ValueError:
        return response.text.strip()[:200] or response.reason_phrase
    # Aura's error bodies look like {"error": true, "message": "..."}.
    if isinstance(body, dict):
        for key in ("message", "error", "errors"):
            value = body.get(key)
            if value and not isinstance(value, bool):
                return str(value)[:200]
    return str(body)[:200]


def _describe(exc: Exception) -> str:
    """httpx timeouts stringify to "", so fall back to the exception's name."""
    return str(exc) or type(exc).__name__


class _Throttle:
    """Keeps consecutive calls at least ``delay`` seconds apart."""

    def __init__(self, delay: float):
        self.delay = delay
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            remaining = self._last + self.delay - time.monotonic()
            if remaining > 0:
                await asyncio.sleep(remaining)
            self._last = time.monotonic()


class AuraClient:
    """Authenticated session against the Aura API.

    ``transport`` exists so tests can swap in ``httpx.MockTransport``.
    """

    def __init__(
        self,
        email: str,
        password: str,
        *,
        api_base_url: str | None = None,
        image_base_url: str | None = None,
        session: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        download_delay: float = DOWNLOAD_DELAY,
        delete_delay: float = DELETE_DELAY,
        timeout: float = 30.0,
    ):
        self._email = email
        self._password = password
        self._image_base_url = (image_base_url or IMAGE_BASE_URL).rstrip("/")
        self._api = httpx.AsyncClient(
            base_url=(api_base_url or API_BASE_URL).rstrip("/"),
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=timeout,
            transport=transport,
        )
        # Separate client: the image proxy needs no credentials, so the auth
        # token is never sent to it.
        self._images = httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
        )
        # A session saved from an earlier sign-in stands in for logging in, so
        # the password itself never has to be kept anywhere.
        self._resumed = bool(session)
        self._auth_headers: dict[str, str] = (
            {"X-Token-Auth": str(session["token"]), "X-User-Id": str(session["user_id"])}
            if session
            else {}
        )
        self._login_lock = asyncio.Lock()
        self._download_throttle = _Throttle(download_delay)
        self._delete_throttle = _Throttle(delete_delay)

    async def aclose(self) -> None:
        await self._api.aclose()
        await self._images.aclose()

    async def __aenter__(self) -> "AuraClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------ auth

    @property
    def is_authenticated(self) -> bool:
        return bool(self._auth_headers)

    @property
    def session(self) -> dict[str, str] | None:
        """What to save to stay signed in: the token Aura issued, not the password."""
        if not self._auth_headers:
            return None
        return {
            "token": self._auth_headers["X-Token-Auth"],
            "user_id": self._auth_headers["X-User-Id"],
        }

    async def login(self) -> None:
        """Authenticate and store the session headers."""
        if not self._email or not self._password:
            if self._resumed:
                raise AuraAuthError("Your Aura sign-in has expired. Sign in again.")
            raise AuraAuthError(
                "Not signed in to Aura. Sign in on the review page, or set "
                "AURA_EMAIL and AURA_PASSWORD (see .env.example)."
            )
        payload = {
            "identifier_for_vendor": DEVICE_ID,
            "client_device_id": DEVICE_ID,
            "app_identifier": APP_IDENTIFIER,
            "locale": "en",
            "user": {"email": self._email, "password": self._password},
        }
        try:
            response = await self._api.post("/login.json", json=payload)
        except httpx.HTTPError as exc:
            raise AuraAPIError(f"Could not reach Aura to log in: {_describe(exc)}") from exc

        # Wrong credentials come back as a non-standard HTTP 475, so treat
        # every client error on login as a rejected login.
        if 400 <= response.status_code < 500:
            raise AuraAuthError(
                "Aura rejected the login. Check AURA_EMAIL and AURA_PASSWORD "
                f"(HTTP {response.status_code}: {_error_detail(response)})"
            )
        if response.status_code >= 500:
            raise AuraAPIError(
                f"Login failed with HTTP {response.status_code}: {_error_detail(response)}",
                response.status_code,
            )

        body: Any = None
        try:
            body = response.json()
            user = body["result"]["current_user"]
            token, user_id = user["auth_token"], user["id"]
        except (ValueError, KeyError, TypeError) as exc:
            raise AuraAuthError(
                "Login response did not contain result.current_user.auth_token/id "
                f"({_error_detail(response)})"
            ) from exc

        self._auth_headers = {"X-Token-Auth": str(token), "X-User-Id": str(user_id)}

    async def _ensure_login(self) -> None:
        if self._auth_headers:
            return
        async with self._login_lock:
            if not self._auth_headers:
                await self.login()

    async def _relogin(self, stale: dict[str, str]) -> None:
        """Re-authenticate, unless a concurrent call already did."""
        async with self._login_lock:
            if self._auth_headers == stale:
                self._auth_headers = {}
                await self.login()

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Authenticated request; logs in again once if the token has expired."""
        await self._ensure_login()
        response: httpx.Response | None = None
        for attempt in range(2):
            headers = dict(self._auth_headers)
            try:
                response = await self._api.request(method, path, headers=headers, **kwargs)
            except httpx.HTTPError as exc:
                raise AuraAPIError(f"{method} {path} failed: {_describe(exc)}") from exc
            if response.status_code != 401:
                break
            if attempt == 0:
                await self._relogin(headers)

        assert response is not None
        if response.status_code == 401:
            raise AuraAuthError(f"{method} {path} was still unauthorized after logging in again.")
        if response.status_code >= 400:
            raise AuraAPIError(
                f"{method} {path} failed with HTTP {response.status_code}: "
                f"{_error_detail(response)}",
                response.status_code,
            )
        return response

    async def _get_json(self, path: str, **kwargs: Any) -> dict[str, Any]:
        response = await self._request("GET", path, **kwargs)
        try:
            body = response.json()
        except ValueError as exc:
            raise AuraAPIError(f"GET {path} did not return JSON.") from exc
        if not isinstance(body, dict):
            raise AuraAPIError(f"GET {path} returned {type(body).__name__}, expected an object.")
        return body

    # ---------------------------------------------------------------- frames

    async def list_frames(self) -> list[dict[str, Any]]:
        body = await self._get_json("/frames.json")
        frames = body.get("frames")
        if not isinstance(frames, list):
            raise AuraAPIError(
                f"frames.json had no 'frames' list (keys: {sorted(body)})."
            )
        return [
            {
                "id": frame.get("id"),
                "name": frame.get("name"),
                "asset_count": frame.get("num_assets"),
            }
            for frame in frames
            if isinstance(frame, dict)
        ]

    async def list_assets(self, frame_id: str) -> list[dict[str, Any]]:
        """Every asset on a frame, following the cursor until it runs out."""
        path = f"/frames/{quote(frame_id, safe='')}/assets.json"
        assets: dict[str, dict[str, Any]] = {}
        seen_cursors: set[str] = set()
        cursor: str | None = None

        for _ in range(MAX_ASSET_PAGES):
            params: dict[str, Any] = {"limit": ASSET_PAGE_LIMIT, "side_load_users": "false"}
            if cursor:
                params["cursor"] = cursor
            body = await self._get_json(path, params=params)

            page = body.get("assets")
            if not isinstance(page, list):
                raise AuraAPIError(
                    f"assets.json had no 'assets' list (keys: {sorted(body)})."
                )
            before = len(assets)
            for asset in page:
                if isinstance(asset, dict) and asset.get("id"):
                    assets.setdefault(str(asset["id"]), asset)

            cursor = next((str(body[k]) for k in CURSOR_KEYS if body.get(k)), None)
            # Stop on: no cursor, a cursor we already followed, or a page that
            # added nothing new (guards against an API that ignores `cursor`).
            if not cursor or cursor in seen_cursors or len(assets) == before:
                break
            seen_cursors.add(cursor)
        else:
            raise AuraAPIError(
                f"Gave up listing frame {frame_id} after {MAX_ASSET_PAGES} pages."
            )

        return list(assets.values())

    # ---------------------------------------------------------------- photos

    def image_url(self, user_id: str, file_name: str, max_side: int | None = None) -> str:
        """Where a photo is served: the original, or with ``max_side`` a copy
        resized by Aura's image host to fit in a ``max_side`` square (aspect
        kept)."""
        name = str(file_name)
        if max_side:
            side = int(max_side)
            name = f"rotate_0__width_{side}__height_{side}__rt_fit__quality_90__{name}"
        return f"{self._image_base_url}/{quote(str(user_id), safe='')}/{quote(name, safe='')}"

    async def download_photo(
        self, user_id: str, file_name: str, dest: Path, max_side: int | None = None
    ) -> Path:
        """Download one photo to ``dest`` (written atomically).

        Without ``max_side`` this is the original file; with it, a smaller
        copy (see ``image_url``).
        """
        await self._download_throttle.wait()
        url = self.image_url(user_id, file_name, max_side)
        try:
            response = await self._images.get(url)
        except httpx.HTTPError as exc:
            raise AuraAPIError(f"Download of {file_name} failed: {_describe(exc)}") from exc
        if response.status_code >= 400:
            raise AuraAPIError(
                f"Download of {file_name} failed with HTTP {response.status_code}.",
                response.status_code,
            )
        if not response.content:
            raise AuraAPIError(f"Download of {file_name} returned an empty body.")
        # An error page sent with status 200 must never be saved as a photo.
        kind = response.headers.get("content-type", "").split(";")[0].strip().lower()
        if kind.startswith("text/") or kind in ("application/json", "application/xml"):
            raise AuraAPIError(f"Download of {file_name} was not a photo (it was {kind}).")

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        tmp.write_bytes(response.content)
        os.replace(tmp, dest)
        return dest

    async def delete_asset(self, asset_id: str) -> None:
        """Permanently delete an asset from the Aura account (every frame)."""
        await self._delete_throttle.wait()
        await self._request("DELETE", f"/assets/{quote(asset_id, safe='')}.json")

    async def remove_asset_from_frame(self, frame_id: str, asset_id: str) -> None:
        """Take an asset off one frame, leaving it on any other frame it is on."""
        await self._delete_throttle.wait()
        path = f"/frames/{quote(frame_id, safe='')}/remove_asset.json"
        response = await self._request("POST", path, json={"assets": [{"asset_id": asset_id}]})
        try:
            body = response.json()
        except ValueError:
            body = None
        # The endpoint answers 200 even when it removed nothing, and reports
        # the misses in `number_failed`.
        if isinstance(body, dict) and body.get("number_failed"):
            raise AuraRefusedError(
                f"Aura could not remove asset {asset_id} from frame {frame_id} "
                f"(number_failed={body['number_failed']})."
            )

    async def add_asset_to_frame(self, frame_id: str, asset_id: str) -> None:
        """Put a photo that already exists in Aura onto a frame.

        This links an existing asset; it uploads nothing. It only works while
        the asset still exists, which in practice means it is on another frame.
        """
        await self._delete_throttle.wait()
        path = f"/frames/{quote(frame_id, safe='')}/select_asset.json"
        response = await self._request("POST", path, json={"assets": [{"asset_id": asset_id}]})
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict) and body.get("number_failed"):
            raise AuraRefusedError(
                f"Aura could not add asset {asset_id} to frame {frame_id} "
                f"(number_failed={body['number_failed']})."
            )

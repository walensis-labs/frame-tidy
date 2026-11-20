import json

import httpx
import pytest

from aura_mcp.aura_client import (
    APP_IDENTIFIER,
    AuraAPIError,
    AuraAuthError,
    AuraClient,
)


def login_ok(token="tok-1"):
    return httpx.Response(
        200, json={"result": {"current_user": {"auth_token": token, "id": "user-1"}}}
    )


def make_client(handler, **kwargs):
    return AuraClient(
        "me@example.com",
        "pw",
        transport=httpx.MockTransport(handler),
        download_delay=0,
        delete_delay=0,
        **kwargs,
    )


async def test_missing_credentials_raise_before_any_request():
    def handler(request):
        raise AssertionError("no request should be sent without credentials")

    async with AuraClient("", "", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(AuraAuthError, match="AURA_EMAIL"):
            await client.list_frames()


async def test_login_sends_expected_payload_and_sets_auth_headers():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/v5/login.json":
            return login_ok()
        return httpx.Response(200, json={"frames": [{"id": "f1", "name": "Kitchen", "num_assets": 3}]})

    async with make_client(handler) as client:
        frames = await client.list_frames()

    login, frames_req = seen
    assert login.method == "POST"
    assert str(login.url) == "https://api.pushd.com/v5/login.json"
    body = json.loads(login.content)
    assert body["user"] == {"email": "me@example.com", "password": "pw"}
    assert body["app_identifier"] == APP_IDENTIFIER
    assert "X-Token-Auth" not in login.headers

    assert frames_req.headers["X-Token-Auth"] == "tok-1"
    assert frames_req.headers["X-User-Id"] == "user-1"
    assert frames_req.headers["User-Agent"].startswith("Aura/")
    assert frames == [{"id": "f1", "name": "Kitchen", "asset_count": 3}]


async def test_bad_password_raises_auth_error():
    # Exactly what the live API returns for wrong credentials.
    def handler(request):
        return httpx.Response(
            475, json={"error": True, "message": "The email or password was incorrect."}
        )

    async with make_client(handler) as client:
        with pytest.raises(AuraAuthError, match="The email or password was incorrect"):
            await client.list_frames()


async def test_login_without_token_raises_auth_error():
    def handler(request):
        return httpx.Response(200, json={"error": True, "message": "nope", "result": None})

    async with make_client(handler) as client:
        with pytest.raises(AuraAuthError, match="nope"):
            await client.login()


async def test_login_server_error_is_not_reported_as_bad_credentials():
    def handler(request):
        return httpx.Response(503, text="upstream down")

    async with make_client(handler) as client:
        with pytest.raises(AuraAPIError, match="HTTP 503"):
            await client.login()


async def test_reauthenticates_once_on_401():
    logins = 0
    tokens_used = []

    def handler(request):
        nonlocal logins
        if request.url.path == "/v5/login.json":
            logins += 1
            return login_ok(f"tok-{logins}")
        tokens_used.append(request.headers["X-Token-Auth"])
        if request.headers["X-Token-Auth"] == "tok-1":
            return httpx.Response(401, json={"error": True, "message": "Request Unauthenticated", "logout": True})
        return httpx.Response(200, json={"frames": []})

    async with make_client(handler) as client:
        assert await client.list_frames() == []

    assert logins == 2
    assert tokens_used == ["tok-1", "tok-2"]


async def test_persistent_401_raises_instead_of_looping():
    calls = 0

    def handler(request):
        nonlocal calls
        if request.url.path == "/v5/login.json":
            return login_ok()
        calls += 1
        return httpx.Response(401)

    async with make_client(handler) as client:
        with pytest.raises(AuraAuthError, match="still unauthorized"):
            await client.list_frames()
    assert calls == 2


async def test_list_assets_follows_cursor_until_exhausted():
    pages = {
        None: {"assets": [{"id": "a1"}, {"id": "a2"}], "next_page_cursor": "c1"},
        "c1": {"assets": [{"id": "a3"}], "next_page_cursor": "c2"},
        "c2": {"assets": [{"id": "a4"}], "next_page_cursor": None},
    }
    cursors = []

    def handler(request):
        if request.url.path == "/v5/login.json":
            return login_ok()
        assert request.url.path == "/v5/frames/frame-1/assets.json"
        cursor = request.url.params.get("cursor")
        cursors.append(cursor)
        return httpx.Response(200, json=pages[cursor])

    async with make_client(handler) as client:
        assets = await client.list_assets("frame-1")

    assert [a["id"] for a in assets] == ["a1", "a2", "a3", "a4"]
    assert cursors == [None, "c1", "c2"]


async def test_list_assets_single_page_without_cursor_key():
    def handler(request):
        if request.url.path == "/v5/login.json":
            return login_ok()
        return httpx.Response(200, json={"assets": [{"id": "a1"}]})

    async with make_client(handler) as client:
        assert [a["id"] for a in await client.list_assets("f")] == ["a1"]


async def test_list_assets_stops_if_api_ignores_cursor():
    """An API that returns the same page forever must not loop forever."""
    calls = 0

    def handler(request):
        nonlocal calls
        if request.url.path == "/v5/login.json":
            return login_ok()
        calls += 1
        return httpx.Response(200, json={"assets": [{"id": "a1"}], "next_page_cursor": f"c{calls}"})

    async with make_client(handler) as client:
        assets = await client.list_assets("f")

    assert [a["id"] for a in assets] == ["a1"]
    assert calls == 2


async def test_list_assets_unexpected_shape_is_a_clear_error():
    def handler(request):
        if request.url.path == "/v5/login.json":
            return login_ok()
        return httpx.Response(200, json={"items": []})

    async with make_client(handler) as client:
        with pytest.raises(AuraAPIError, match="no 'assets' list"):
            await client.list_assets("f")


async def test_download_uses_image_proxy_without_auth_and_writes_file(tmp_path):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, content=b"jpeg-bytes")

    dest = tmp_path / "frame" / "a1.jpg"
    async with make_client(handler) as client:
        await client.download_photo("user-9", "photo 1.jpeg", dest)

    assert dest.read_bytes() == b"jpeg-bytes"
    assert not dest.with_name("a1.jpg.part").exists()
    assert str(seen[0].url) == "https://imgproxy.pushd.com/user-9/photo%201.jpeg"
    assert "X-Token-Auth" not in seen[0].headers


def test_resized_address_asks_aura_for_a_medium_copy():
    client = AuraClient("e", "p", image_base_url="https://img.example/")
    assert client.image_url("user-9", "photo 1.jpeg") == "https://img.example/user-9/photo%201.jpeg"
    assert client.image_url("user-9", "photo 1.jpeg", max_side=1600) == (
        "https://img.example/user-9/"
        "rotate_0__width_1600__height_1600__rt_fit__quality_90__photo%201.jpeg"
    )


async def test_download_with_max_side_fetches_the_resized_address(tmp_path):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, content=b"smaller")

    dest = tmp_path / "a1.jpg"
    async with make_client(handler) as client:
        await client.download_photo("u", "f.jpg", dest, max_side=1600)
        await client.download_photo("u", "f.jpg", tmp_path / "full.jpg")

    assert seen == [
        "https://imgproxy.pushd.com/u/rotate_0__width_1600__height_1600__rt_fit__quality_90__f.jpg",
        "https://imgproxy.pushd.com/u/f.jpg",
    ]
    assert dest.read_bytes() == b"smaller"


def test_downloads_are_not_spaced_out_by_default():
    # The caller bounds how many run at once; a pause per photo made a
    # 3,000-photo frame take most of an hour.
    from aura_mcp import aura_client

    assert aura_client.DOWNLOAD_DELAY == 0


async def test_download_failure_leaves_no_file(tmp_path):
    def handler(request):
        return httpx.Response(404)

    dest = tmp_path / "a1.jpg"
    async with make_client(handler) as client:
        with pytest.raises(AuraAPIError, match="HTTP 404"):
            await client.download_photo("u", "f.jpg", dest)
    assert not dest.exists()


async def test_download_timeout_is_named_in_the_error(tmp_path):
    def handler(request):
        raise httpx.ReadTimeout("")  # what a real timeout looks like: no message

    async with make_client(handler) as client:
        with pytest.raises(AuraAPIError, match="failed: ReadTimeout"):
            await client.download_photo("u", "f.jpg", tmp_path / "a.jpg")


async def test_delete_asset_calls_delete_endpoint():
    seen = []

    def handler(request):
        if request.url.path == "/v5/login.json":
            return login_ok()
        seen.append((request.method, request.url.path, request.headers.get("X-Token-Auth")))
        return httpx.Response(200, json={})

    async with make_client(handler) as client:
        await client.delete_asset("asset-7")

    assert seen == [("DELETE", "/v5/assets/asset-7.json", "tok-1")]


async def test_delete_asset_error_surfaces_status():
    def handler(request):
        if request.url.path == "/v5/login.json":
            return login_ok()
        return httpx.Response(403, json={"error": True, "message": "not yours"})

    async with make_client(handler) as client:
        with pytest.raises(AuraAPIError, match="not yours") as excinfo:
            await client.delete_asset("asset-7")
    assert excinfo.value.status_code == 403


async def test_throttle_spaces_out_deletes():
    import time

    def handler(request):
        if request.url.path == "/v5/login.json":
            return login_ok()
        return httpx.Response(200, json={})

    client = AuraClient(
        "e", "p", transport=httpx.MockTransport(handler), delete_delay=0.05
    )
    async with client:
        await client.delete_asset("a")
        start = time.monotonic()
        await client.delete_asset("b")
        assert time.monotonic() - start >= 0.045


async def test_remove_asset_from_frame_posts_to_frame_endpoint():
    seen = []

    def handler(request):
        if request.url.path == "/v5/login.json":
            return login_ok()
        seen.append((request.method, request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"number_failed": 0})

    async with make_client(handler) as client:
        await client.remove_asset_from_frame("frame-1", "asset-7")

    assert seen == [
        ("POST", "/v5/frames/frame-1/remove_asset.json", {"assets": [{"asset_id": "asset-7"}]})
    ]


async def test_remove_asset_from_frame_reports_number_failed():
    def handler(request):
        if request.url.path == "/v5/login.json":
            return login_ok()
        return httpx.Response(200, json={"number_failed": 1})

    async with make_client(handler) as client:
        with pytest.raises(AuraAPIError, match="number_failed=1"):
            await client.remove_asset_from_frame("frame-1", "asset-7")


async def test_base_urls_can_point_at_a_mock():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        if request.url.path.endswith("/login.json"):
            return login_ok()
        if "assets" in request.url.path or "frames" in request.url.path:
            return httpx.Response(200, json={"frames": []})
        return httpx.Response(200, content=b"x")

    client = AuraClient(
        "e", "p",
        api_base_url="http://127.0.0.1:8765/v5/",
        image_base_url="http://127.0.0.1:8765/img",
        transport=httpx.MockTransport(handler),
        download_delay=0,
    )
    async with client:
        await client.list_frames()
        assert client.image_url("u1", "a.jpg") == "http://127.0.0.1:8765/img/u1/a.jpg"

    assert seen == ["http://127.0.0.1:8765/v5/login.json", "http://127.0.0.1:8765/v5/frames.json"]


async def test_add_asset_to_frame_posts_to_select_endpoint():
    seen = []

    def handler(request):
        if request.url.path == "/v5/login.json":
            return login_ok()
        seen.append((request.method, request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"number_failed": 0 if len(seen) == 1 else 1})

    async with make_client(handler) as client:
        await client.add_asset_to_frame("frame-1", "asset-7")
        with pytest.raises(AuraAPIError, match="could not add asset asset-7"):
            await client.add_asset_to_frame("frame-1", "asset-7")

    assert seen[0] == (
        "POST", "/v5/frames/frame-1/select_asset.json", {"assets": [{"asset_id": "asset-7"}]}
    )


# ---------------------------------------------------------- saved sign-in

async def test_a_saved_session_is_used_without_logging_in():
    seen = []

    def handler(request):
        seen.append(request)
        assert request.url.path != "/v5/login.json"
        return httpx.Response(200, json={"frames": []})

    session = {"token": "tok-9", "user_id": "user-9"}
    async with AuraClient("", "", session=session, transport=httpx.MockTransport(handler)) as client:
        await client.list_frames()
        assert client.session == session

    assert seen[0].headers["X-Token-Auth"] == "tok-9"
    assert seen[0].headers["X-User-Id"] == "user-9"


async def test_login_exposes_the_session_to_save():
    async with make_client(lambda request: login_ok("tok-2")) as client:
        assert client.session is None
        await client.login()
        assert client.session == {"token": "tok-2", "user_id": "user-1"}


async def test_an_expired_saved_session_asks_to_sign_in_again():
    def handler(request):
        assert request.url.path != "/v5/login.json"  # there is no password to send
        return httpx.Response(401, json={"error": True})

    session = {"token": "old", "user_id": "user-9"}
    async with AuraClient("", "", session=session, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(AuraAuthError, match="Sign in again"):
            await client.list_frames()


@pytest.mark.parametrize("content_type", ["text/html; charset=utf-8", "application/json", "text/plain"])
async def test_a_download_that_is_a_web_page_is_refused(tmp_path, content_type):
    """An error page sent with status 200 must never be saved as a photo."""

    def handler(request):
        return httpx.Response(200, content=b"<html>Service unavailable</html>",
                              headers={"content-type": content_type})

    dest = tmp_path / "a1.jpg"
    async with make_client(handler) as client:
        with pytest.raises(AuraAPIError, match="not a photo"):
            await client.download_photo("u", "f.jpg", dest)
    assert not dest.exists()

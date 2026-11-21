"""Using the page from a phone on the same wifi (WP4).

No real listener is opened here: the second server is replaced by a stand-in,
and requests "from the phone" are made with a test client whose address is
the wifi one, which is what the server sees when a phone connects to it.
"""

import json
import sys
from urllib.parse import parse_qs, urlsplit

import pytest
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from aura_mcp import review as review_mod
from aura_mcp.face_matcher import FaceMatcher
from aura_mcp.review import LAN_ALLOWED, PHONE_COOKIE, Review, usable_lan_ip

from test_review import FRAME, FakeAccount, FakeApp, FakeAura, write_photo

LAN_IP = "192.168.1.20"
LAN_PORT = 8774
LAN = f"{LAN_IP}:{LAN_PORT}"


class FakeListener:
    """Stands in for the second server: records that it was opened and closed."""

    def __init__(self, app, host, port):
        self.app, self.host, self.port, self.stopped = app, host, port or LAN_PORT, False

    def stop(self):
        self.stopped = True


@pytest.fixture
def root(tmp_path):
    folder = tmp_path / "cache" / FRAME
    for name in ("old", "new", "mid"):
        write_photo(folder / f"{name}.jpg")
    (folder / "index.json").write_text(json.dumps({
        name: {"file_name": f"{name}.heic", "taken_at": "2024-01-01T00:00:00Z",
               "local_path": str(folder / f"{name}.jpg")} for name in ("old", "new", "mid")}))
    (tmp_path / "cache" / "frames.json").write_text(json.dumps({FRAME: "Living Room"}))
    return tmp_path


def make(root, address=LAN_IP, listener=FakeListener, **kwargs):
    aura = FakeAura()
    made = []

    def factory(email=None, password=None):
        made.append(FakeAccount(email, password) if email else aura)
        return made[-1]

    review = Review(root, FaceMatcher(app=FakeApp()), client_factory=factory,
                    lan_address=lambda: address, lan_listener=listener, **kwargs)
    app = review.app()
    computer = TestClient(app, base_url="http://127.0.0.1:8766")
    return review, app, computer, aura, made


def phone(app):
    """A device on the wifi, talking to the wifi listener."""
    return TestClient(app, base_url=f"http://{LAN}")


def turn_on(computer):
    response = computer.post("/api/phone/start")
    assert response.status_code == 200, response.text
    return response.json()


def secret_of(link):
    return parse_qs(urlsplit(link).query)["t"][0]


def paired_phone(app, computer):
    status = turn_on(computer)
    device = phone(app)
    response = device.get(f"/pair?t={secret_of(status['link'])}")
    assert response.status_code == 200
    return device, status


# ------------------------------------------------------- off by default

def test_off_by_default_and_nothing_answers_on_the_wifi(root):
    review, app, computer, aura, made = make(root)
    status = computer.get("/api/phone").json()
    assert status["on"] is False and status["link"] is None
    assert phone(app).get("/api/frames").status_code == 403


# ------------------------------------------------ without the pairing cookie

def test_a_wifi_request_without_the_cookie_is_refused(root):
    review, app, computer, aura, made = make(root)
    turn_on(computer)
    device = phone(app)
    for path in ("/", "/api/frames", f"/api/frames/{FRAME}/photos", f"/photo/{FRAME}/new.jpg",
                 f"/thumb/{FRAME}/new.jpg", "/api/people", "/api/removal/status"):
        assert device.get(path).status_code == 403, path
    origin = {"Origin": f"http://{LAN}"}
    assert device.put(f"/api/frames/{FRAME}/selection", json={"asset_ids": ["new"]},
                      headers=origin).status_code == 403
    assert device.post(f"/api/frames/{FRAME}/removal/preview", headers=origin,
                       json={"scope": "frame", "asset_ids": ["new"]}).status_code == 403
    assert device.post("/api/removal", json={"token": "x"}, headers=origin).status_code == 403
    assert not (root / "selections").exists() and aura.removed == []
    # A refusal says nothing about the secret.
    assert "pair" not in device.get("/api/frames").text


def test_a_made_up_cookie_is_refused(root):
    review, app, computer, aura, made = make(root)
    paired_phone(app, computer)
    stranger = phone(app)
    stranger.cookies.set(PHONE_COOKIE, "guess")
    assert stranger.get("/api/frames").status_code == 403
    stranger.cookies.set(PHONE_COOKIE, "")
    assert stranger.get("/api/frames").status_code == 403


# --------------------------------------------------------------- pairing

def test_pairing_sets_a_strict_http_only_cookie_and_then_the_phone_may_browse(root):
    review, app, computer, aura, made = make(root)
    status = turn_on(computer)
    assert status["on"] is True and status["address"] == f"http://{LAN}"
    assert status["link"].startswith(f"http://{LAN}/pair?t=")
    assert status["qr"].lstrip().startswith("<svg") and status["link"] not in status["qr"]

    device = phone(app)
    response = device.get(f"/pair?t={secret_of(status['link'])}")
    assert response.status_code == 200
    cookie = response.headers["set-cookie"]
    assert cookie.startswith(f"{PHONE_COOKIE}=")
    parts = {p.strip().lower() for p in cookie.split(";")}
    assert "httponly" in parts and "samesite=strict" in parts and "path=/" in parts
    assert "secure" not in parts
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["referrer-policy"] == "no-referrer"

    assert device.get("/").status_code == 200
    assert device.get("/api/frames").json()["frames"][0]["id"] == FRAME
    assert device.get(f"/photo/{FRAME}/new.jpg").status_code == 200
    put = device.put(f"/api/frames/{FRAME}/selection", json={"asset_ids": ["new"]},
                     headers={"Origin": f"http://{LAN}"})
    assert put.status_code == 200
    assert computer.get("/api/phone").json()["phones"] == 1


def test_a_wrong_secret_is_refused_and_sets_nothing(root):
    review, app, computer, aura, made = make(root)
    status = turn_on(computer)
    device = phone(app)
    for wrong in ("nope", secret_of(status["link"])[:-1], ""):
        response = device.get(f"/pair?t={wrong}")
        assert response.status_code == 403 and "set-cookie" not in response.headers
    assert device.get("/pair").status_code == 403
    # Wrong guesses do not use the link up.
    assert device.get(f"/pair?t={secret_of(status['link'])}").status_code == 200


def test_the_pairing_link_works_once(root):
    review, app, computer, aura, made = make(root)
    device, status = paired_phone(app, computer)
    again = phone(app).get(f"/pair?t={secret_of(status['link'])}")
    assert again.status_code == 403 and "set-cookie" not in again.headers
    assert computer.get("/api/phone").json()["link_used"] is True


def test_another_phone_needs_a_new_link_and_the_first_stays_paired(root):
    review, app, computer, aura, made = make(root)
    first, status = paired_phone(app, computer)
    fresh = computer.post("/api/phone/new-link").json()
    assert fresh["link"] != status["link"]
    second = phone(app)
    assert second.get(f"/pair?t={secret_of(fresh['link'])}").status_code == 200
    assert first.get("/api/frames").status_code == 200
    assert second.get("/api/frames").status_code == 200


def test_an_old_link_expires(root, monkeypatch):
    review, app, computer, aura, made = make(root)
    status = turn_on(computer)
    clock = review_mod.time.time() + review_mod.PAIR_TTL + 1
    monkeypatch.setattr(review_mod.time, "time", lambda: clock)
    assert phone(app).get(f"/pair?t={secret_of(status['link'])}").status_code == 403


# ---------------------------------------------------------- the address

@pytest.mark.parametrize("host", [
    "192.168.1.21:8774", f"{LAN_IP}", f"{LAN_IP}:8775", "evil.example", f"{LAN}.evil.example",
    "127.0.0.1:8766", "localhost", f" {LAN}", f"{LAN.upper()}x",
])
def test_a_wrong_wifi_host_is_refused_even_with_the_cookie(root, host):
    review, app, computer, aura, made = make(root)
    device, _ = paired_phone(app, computer)
    assert device.get("/api/frames", headers={"host": host}).status_code == 403
    assert device.get(f"/photo/{FRAME}/new.jpg", headers={"host": host}).status_code == 403


def test_a_request_on_another_listener_is_not_taken_for_this_computer(root):
    """Saying "Host: 127.0.0.1" from the wifi does not make a request local."""
    review, app, computer, aura, made = make(root)
    device, _ = paired_phone(app, computer)
    cookie = f"{PHONE_COOKIE}={device.cookies.get(PHONE_COOKIE)}"
    assert device.get("/api/frames", headers={"cookie": cookie}).status_code == 200
    other = TestClient(app, base_url="http://192.168.1.99:8774")
    for host in ("127.0.0.1:8766", LAN):
        assert other.get("/api/frames", headers={"host": host, "cookie": cookie}).status_code == 403
    # The wifi listener itself, told it is 127.0.0.1, is still the wifi listener.
    assert phone(app).get("/api/frames", headers={"host": "127.0.0.1:8766", "cookie": cookie}).status_code == 403


def test_cross_origin_writes_from_the_wifi_are_refused(root):
    review, app, computer, aura, made = make(root)
    device, _ = paired_phone(app, computer)
    response = device.put(f"/api/frames/{FRAME}/selection", json={"asset_ids": ["new"]},
                          headers={"Origin": "http://evil.example"})
    assert response.status_code == 403


# ---------------------------------------------- what a phone may not do

DENIED_OVER_WIFI = {
    ("GET", "/api/account"),
    ("POST", "/api/account/sign-in"),
    ("POST", "/api/account/sign-out"),
    ("GET", "/api/account/frames"),
    ("POST", "/api/sync"),
    ("POST", "/api/setup/face-model"),
    ("GET", "/api/setup/face-model"),
    ("GET", "/api/phone"),
    ("POST", "/api/setup/search-model"),
    ("GET", "/api/setup/search-model"),
    ("POST", "/api/phone/start"),
    ("POST", "/api/phone/stop"),
    ("POST", "/api/phone/new-link"),
}


def test_every_route_is_allowed_or_refused_on_purpose(root):
    """A new route must be put in one list or the other by someone deciding."""
    review, app, computer, aura, made = make(root)
    routes = set()
    for route in app.router.routes:
        for method in route.methods or ():
            if method != "HEAD":
                routes.add((method, route.path))
    assert routes == LAN_ALLOWED | DENIED_OVER_WIFI
    assert not LAN_ALLOWED & DENIED_OVER_WIFI


def test_sign_in_and_the_rest_are_refused_from_a_paired_phone(root):
    review, app, computer, aura, made = make(root)
    device, _ = paired_phone(app, computer)
    origin = {"Origin": f"http://{LAN}"}
    sign_in = device.post("/api/account/sign-in", headers=origin,
                          json={"email": "a@b.c", "password": "right"})
    assert sign_in.status_code == 403
    assert "computer" in sign_in.json()["error"]
    assert [m for m in made if getattr(m, "password", None)] == []
    assert not (root / "session.json").exists()
    for method, path in sorted(DENIED_OVER_WIFI):
        response = device.request(method, path, headers=origin, json={"agree": True, "frame_ids": [FRAME]})
        assert response.status_code == 403, (method, path)
    assert computer.get("/api/phone").json()["on"] is True  # the phone could not stop it
    assert review.model_job.thread is None and review.sync.thread is None


def test_a_route_added_later_is_refused_over_the_wifi_until_allowed(root):
    review, app, computer, aura, made = make(root)
    device, _ = paired_phone(app, computer)
    app.router.routes.append(Route("/api/new-thing", lambda request: JSONResponse({"ok": True})))
    assert computer.get("/api/new-thing").status_code == 200
    assert device.get("/api/new-thing").status_code == 403


def test_a_paired_phone_can_pick_answer_and_remove_through_the_same_preview(root):
    review, app, computer, aura, made = make(root)
    device, _ = paired_phone(app, computer)
    origin = {"Origin": f"http://{LAN}"}
    device.put(f"/api/frames/{FRAME}/selection", json={"asset_ids": ["new"]}, headers=origin)
    preview = device.post(f"/api/frames/{FRAME}/removal/preview", headers=origin, json={"scope": "frame"})
    assert preview.status_code == 200
    assert device.post("/api/removal", headers=origin, json={"token": "made-up"}).status_code == 409
    go = device.post("/api/removal", headers=origin, json={"token": preview.json()["token"]})
    assert go.status_code == 202
    review.removal.thread.join(10)
    assert device.get("/api/removal/status").json()["state"] == "done"
    assert aura.removed == [(FRAME, "new")]


# ------------------------------------------------------ turning it off

def test_turning_it_off_closes_the_listener_and_kills_the_cookie_and_link(root):
    review, app, computer, aura, made = make(root)
    device, status = paired_phone(app, computer)
    listener = review._lan["listener"]
    assert computer.post("/api/phone/stop").json()["on"] is False
    assert listener.stopped
    assert device.get("/api/frames").status_code == 403

    again = turn_on(computer)
    assert again["link"] != status["link"]
    assert device.get("/api/frames").status_code == 403  # the old cookie
    assert phone(app).get(f"/pair?t={secret_of(status['link'])}").status_code == 403  # the old link


def test_starting_with_no_network_says_so_plainly(root):
    review, app, computer, aura, made = make(root, address=None)
    response = computer.post("/api/phone/start")
    assert response.status_code == 503
    assert "wifi" in response.json()["error"].lower()
    assert computer.get("/api/phone").json()["on"] is False


def test_a_listener_that_cannot_open_says_so_plainly(root):
    def broken(app, host, port):
        raise OSError(48, "Address already in use")

    review, app, computer, aura, made = make(root, listener=broken)
    response = computer.post("/api/phone/start")
    assert response.status_code == 503
    body = response.json()
    assert "could not" in body["error"].lower() and "in use" in body["detail"]
    assert computer.get("/api/phone").json()["on"] is False


def test_the_wifi_address_must_be_a_home_network_one():
    assert usable_lan_ip("192.168.1.20") and usable_lan_ip("10.0.0.5") and usable_lan_ip("172.16.4.2")
    for ip in ("127.0.0.1", "0.0.0.0", "8.8.8.8", "100.64.1.1", "::1", "fe80::1", "", "nonsense"):
        assert not usable_lan_ip(ip), ip


def test_a_public_address_is_never_opened(root):
    review, app, computer, aura, made = make(root, address="8.8.8.8")
    assert computer.post("/api/phone/start").status_code == 503


def test_the_page_says_what_a_phone_cannot_do(root):
    review, app, computer, aura, made = make(root)
    page = computer.get("/").text
    assert 'id="phone-btn"' in page
    assert "anyone on the same wifi" in page.lower()
    assert "cannot sign in" in page.lower()


# ------------------------------------------------- stops when the app quits

def test_the_desktop_app_stops_the_phone_listener_when_it_quits(tmp_path, monkeypatch):
    import uvicorn

    from aura_mcp import app as desktop

    monkeypatch.setenv("AURA_MCP_HOME", str(tmp_path))
    stopped = []

    class Server:
        def __init__(self, config):
            self.started, self.should_exit = True, False

        def run(self):
            pass

    monkeypatch.setattr(uvicorn, "Server", Server)
    monkeypatch.setattr(uvicorn, "Config", lambda *a, **k: None)
    monkeypatch.setattr(Review, "stop_lan", lambda self: stopped.append(self))
    monkeypatch.setattr(Review, "warm_model", lambda self: None)
    desktop.main(["--no-window"])
    assert len(stopped) == 1


def test_aura_review_stops_the_phone_listener_when_it_quits(tmp_path, monkeypatch):
    import uvicorn

    stopped = []
    monkeypatch.setattr(review_mod.store, "ROOT", tmp_path)
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    monkeypatch.setattr(Review, "stop_lan", lambda self: stopped.append(self))
    monkeypatch.setattr(sys, "argv", ["aura-review", "--port", "8799"])
    review_mod.main()
    assert len(stopped) == 1


def test_the_face_model_terms_are_stated_as_insightface_states_them(root):
    review, app, computer, aura, made = make(root)
    page = computer.get("/").text
    step = page[page.index('<template id="tpl-model">'):page.index("</template>")]
    assert "non-commercial research purposes only" in step
    assert "personal" not in step and "free" not in step
    assert "https://github.com/deepinsight/insightface#license" in step


# ------------------------------------------ review 2: no file paths on the phone

def _no_paths(text, root):
    assert str(root) not in text and "/Users/" not in text and "/private/" not in text


def test_reference_paths_stay_on_the_computer(root):
    review, app, computer, aura, made = make(root)
    crop = root / "references" / "me.jpg"
    from test_review import write_photo as write
    write(crop, 200, 200)
    (root / "references" / "references.json").write_text(
        json.dumps([{"id": "r1", "name": "Me", "path": str(crop)}]))
    device, _ = paired_phone(app, computer)

    assert computer.get("/api/frames").json()["references"][0]["path"] == str(crop)
    on_phone = device.get("/api/frames")
    assert on_phone.json()["references"][0]["path"] != str(crop)
    _no_paths(on_phone.text, root)


def test_removed_and_flagged_paths_stay_on_the_computer(root):
    review, app, computer, aura, made = make(root)
    device, _ = paired_phone(app, computer)
    origin = {"Origin": f"http://{LAN}"}
    flagged = root / "flagged" / FRAME / "new.jpg"
    flagged.parent.mkdir(parents=True)
    flagged.write_bytes(b"x")

    preview = device.post(f"/api/frames/{FRAME}/removal/preview", headers=origin,
                          json={"scope": "frame", "asset_ids": ["new"]})
    assert preview.json()["would_delete"][0]["flagged_path"]
    _no_paths(preview.text, root)
    device.post("/api/removal", headers=origin, json={"token": preview.json()["token"]})
    review.removal.thread.join(10)
    done = device.get("/api/removal/status")
    assert done.json()["result"]["deleted"][0]["local_copy"]
    _no_paths(done.text, root)

    removed = device.get(f"/api/frames/{FRAME}/removed")
    assert removed.json()["removed"][0]["local_path"]  # there is a copy...
    _no_paths(removed.text, root)  # ...but where it is, is shown on the computer
    assert computer.get(f"/api/frames/{FRAME}/removed").json()["removed"][0]["local_path"] == str(
        root / "removed" / FRAME / "new.jpg")


def test_a_failed_put_back_shows_no_path_on_the_phone(root):
    from aura_mcp.aura_client import AuraRefusedError

    review, app, computer, aura, made = make(root)
    device, _ = paired_phone(app, computer)
    origin = {"Origin": f"http://{LAN}"}
    preview = device.post(f"/api/frames/{FRAME}/removal/preview", headers=origin,
                          json={"scope": "frame", "asset_ids": ["new"]})
    device.post("/api/removal", headers=origin, json={"token": preview.json()["token"]})
    review.removal.thread.join(10)
    aura.restore_error = AuraRefusedError("Aura could not add asset new (number_failed=1).")

    device.post("/api/restore", headers=origin, json={"frame_id": FRAME, "asset_ids": ["new"]})
    review.restore.thread.join(10)

    on_phone = device.get("/api/restore/status")
    failure = on_phone.json()["result"]["failed"][0]
    assert "Aura app" in failure["message"] and "computer" in failure["message"]
    _no_paths(on_phone.text, root)
    kept = str(root / "removed" / FRAME / "new.jpg")
    assert kept in computer.get("/api/restore/status").json()["result"]["failed"][0]["message"]


def test_a_disk_error_shows_no_path_on_the_phone(root, monkeypatch):
    from aura_mcp import server

    review, app, computer, aura, made = make(root)
    device, _ = paired_phone(app, computer)
    origin = {"Origin": f"http://{LAN}"}
    preview = device.post(f"/api/frames/{FRAME}/removal/preview", headers=origin,
                          json={"scope": "frame", "asset_ids": ["new"]})
    where = root / "removed" / FRAME / "new.jpg"

    async def full(*args, **kwargs):
        raise OSError(28, "No space left on device", str(where))

    monkeypatch.setattr(server, "delete_assets_impl", full)
    device.post("/api/removal", headers=origin, json={"token": preview.json()["token"]})
    review.removal.thread.join(10)

    on_phone = device.get("/api/removal/status")
    assert on_phone.json()["state"] == "error" and "No space" in on_phone.json()["detail"]
    _no_paths(on_phone.text, root)
    assert str(where) in computer.get("/api/removal/status").json()["detail"]


# ------------------------------------- review 2: a phone's cookie is its own

def test_a_cookie_only_works_from_the_address_that_paired(root):
    review, app, computer, aura, made = make(root)
    status = turn_on(computer)
    mine = TestClient(app, base_url=f"http://{LAN}", client=("192.168.1.30", 50000))
    assert mine.get(f"/pair?t={secret_of(status['link'])}").status_code == 200
    assert mine.get("/api/frames").status_code == 200

    thief = TestClient(app, base_url=f"http://{LAN}", client=("192.168.1.31", 50000),
                       cookies={PHONE_COOKIE: mine.cookies[PHONE_COOKIE]})
    response = thief.get("/api/frames")
    assert response.status_code == 403 and "photos" not in response.text


@pytest.mark.parametrize("ip, ok", [
    ("10.0.0.5", True), ("172.16.4.2", True), ("172.31.255.1", True), ("192.168.1.20", True),
    ("172.32.0.1", False), ("169.254.10.10", False), ("100.64.0.1", False), ("127.0.0.1", False),
    ("0.0.0.0", False), ("8.8.8.8", False), ("192.0.2.1", False), ("198.18.0.1", False),
    ("fd00::1", False), ("fe80::1", False), ("not an address", False),
])
def test_only_home_network_addresses_are_used(ip, ok):
    assert usable_lan_ip(ip) is ok


# ------------------------------- review 2: the socket decides, not the Host header

async def _call(app, server, host, path="/api/frames"):
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET",
        "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"",
        "root_path": "", "headers": [(b"host", host.encode())],
        "client": ("192.168.1.30", 50000), "server": server,
    }
    await app(scope, receive, send)
    return next(m["status"] for m in sent if m["type"] == "http.response.start")


async def test_a_request_is_local_only_if_it_came_in_on_the_local_socket(root):
    review, app, computer, aura, made = make(root)
    turn_on(computer)

    # Came in on the wifi socket, claiming to be addressed to this computer.
    assert await _call(app, ("192.168.1.20", LAN_PORT), "127.0.0.1:8766") == 403
    assert await _call(app, ("192.168.1.20", LAN_PORT), "localhost") == 403
    # Came in on the local socket, claiming to be addressed to the wifi one.
    assert await _call(app, ("127.0.0.1", 8766), LAN) == 403
    # Each one, honestly addressed.
    assert await _call(app, ("127.0.0.1", 8766), "127.0.0.1:8766") == 200
    assert await _call(app, ("192.168.1.20", LAN_PORT), LAN) == 403  # not paired


def test_a_real_second_listener_is_not_taken_for_this_computer(root):
    """With a real socket on another loopback address (where the system has one)."""
    import http.client

    review, app, computer, aura, made = make(root)
    try:
        listener = review_mod.LanListener(app, "127.0.0.2", 0)
    except OSError:
        pytest.skip("this system has no 127.0.0.2")
    try:
        connection = http.client.HTTPConnection("127.0.0.2", listener.port, timeout=5)
        connection.request("GET", "/api/frames", headers={"Host": "127.0.0.1:8766"})
        assert connection.getresponse().status == 403
    finally:
        listener.stop()


# ------------------------------------ review 2: the window closes the listener

def test_the_desktop_window_stops_the_phone_listener_when_it_closes(tmp_path, monkeypatch):
    import types

    import uvicorn

    from aura_mcp import app as desktop

    class Server:
        def __init__(self, config):
            self.started, self.should_exit = True, False

        def run(self):
            pass

    events = []
    webview = types.ModuleType("webview")
    webview.create_window = lambda title, url, **kwargs: events.append(("window", title))
    webview.start = lambda *args, **kwargs: events.append(("closed",))
    monkeypatch.setitem(sys.modules, "webview", webview)
    monkeypatch.setenv("AURA_MCP_HOME", str(tmp_path))
    monkeypatch.setattr(uvicorn, "Server", Server)
    monkeypatch.setattr(uvicorn, "Config", lambda *a, **k: None)
    monkeypatch.setattr(Review, "stop_lan", lambda self: events.append(("stop_lan",)))

    desktop.main([])

    assert events == [("window", "Frame Tidy"), ("closed",), ("stop_lan",)]

import json
from importlib.metadata import version as _version
import threading
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from starlette.testclient import TestClient

from aura_mcp import people, server
from aura_mcp.aura_client import AuraAPIError, AuraAuthError, AuraRefusedError
from aura_mcp.face_matcher import FaceMatcher
from aura_mcp.review import Review

FRAME = "frame-1"


class FakeApp:
    """Two faces in a full 800px photo, one centred face in a crop of it,
    and nothing in an image too small to hold a face."""

    def get(self, image):
        h, w = image.shape[:2]
        emb = np.eye(512, dtype=np.float32)
        if h < 100:
            return []
        if w == 800:
            return [
                SimpleNamespace(bbox=np.array([w * 0.6, h * 0.2, w * 0.8, h * 0.5]), normed_embedding=emb[1]),
                SimpleNamespace(bbox=np.array([w * 0.1, h * 0.2, w * 0.3, h * 0.5]), normed_embedding=emb[0]),
            ]
        return [SimpleNamespace(bbox=np.array([w * 0.3, h * 0.3, w * 0.7, h * 0.7]), normed_embedding=emb[0])]


# The installed version, which /api/version reports.
APP_VERSION = _version("aura-mcp")


def write_photo(path, width=800, height=600):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((height, width, 3), 120, dtype=np.uint8))


@pytest.fixture
def root(tmp_path):
    folder = tmp_path / "cache" / FRAME
    for name in ("old", "new", "mid"):
        write_photo(folder / f"{name}.jpg")
    (folder / "index.json").write_text(json.dumps({
        "old": {"file_name": "old.heic", "taken_at": "2022-03-01T10:00:00Z", "local_path": str(folder / "old.jpg")},
        "new": {"file_name": "new.heic", "taken_at": "2025-07-04T10:00:00Z", "local_path": str(folder / "new.jpg")},
        "mid": {"file_name": "mid.heic", "taken_at": "2024-01-15T10:00:00Z", "local_path": str(folder / "mid.jpg")},
    }))
    (tmp_path / "cache" / "frames.json").write_text(json.dumps({FRAME: "Living Room"}))
    return tmp_path


@pytest.fixture
def client(root):
    return TestClient(Review(root, FaceMatcher(app=FakeApp())).app(), base_url="http://127.0.0.1")


def test_page_and_frames(client):
    page = client.get("/").text
    assert "<title>Frame Tidy</title>" in page
    assert 'data-view="people"' in page
    data = client.get("/api/frames").json()
    assert data["frames"] == [{"id": FRAME, "name": "Living Room", "photos": 3}]
    assert data["references"] == []


def test_photos_sorted_newest_first_with_scores(client, root):
    flagged = root / "flagged" / FRAME
    flagged.mkdir(parents=True)
    # the list form written by the first version is still understood
    (flagged / "matches.json").write_text(json.dumps([{"asset_id": "mid", "similarity": 0.61}]))

    data = client.get(f"/api/frames/{FRAME}/photos").json()

    assert data["flagged"]["person"] is None
    assert [p["id"] for p in data["photos"]] == ["new", "mid", "old"]
    assert [p["similarity"] for p in data["photos"]] == [None, 0.61, None]
    assert data["sync_complete"] is True and data["selection"] == []


def test_photos_listed_while_sync_is_still_running(client, root):
    (root / "cache" / FRAME / "index.json").unlink()
    data = client.get(f"/api/frames/{FRAME}/photos").json()
    assert data["sync_complete"] is False
    assert sorted(p["id"] for p in data["photos"]) == ["mid", "new", "old"]


def test_selection_round_trip_ignores_unknown_ids(client, root):
    response = client.put(
        f"/api/frames/{FRAME}/selection",
        json={"asset_ids": ["new", "not-a-photo", "../../etc/passwd", "old", "new"]},
    )
    assert response.json() == {"selected": 2}
    assert client.get(f"/api/frames/{FRAME}/photos").json()["selection"] == ["new", "old"]

    picked = server.get_selection_impl(root, FRAME)
    assert picked["selected_count"] == 2
    assert picked["selected"][0] == {
        "asset_id": "new", "file_name": "new.heic", "taken_at": "2025-07-04T10:00:00Z",
    }


def test_selection_drops_photos_that_are_gone(client, root):
    client.put(f"/api/frames/{FRAME}/selection", json={"asset_ids": ["new", "old"]})
    (root / "cache" / FRAME / "old.jpg").unlink()
    assert client.get(f"/api/frames/{FRAME}/photos").json()["selection"] == ["new"]
    assert server.get_selection_impl(root, FRAME)["selected_count"] == 1


def test_cross_origin_writes_are_refused(client, root):
    response = client.put(
        f"/api/frames/{FRAME}/selection",
        json={"asset_ids": ["new"]},
        headers={"Origin": "https://evil.example"},
    )
    assert response.status_code == 403
    assert not (root / "selections" / f"{FRAME}.json").exists()

    same = client.put(
        f"/api/frames/{FRAME}/selection",
        json={"asset_ids": ["new"]},
        headers={"Origin": "http://127.0.0.1"},
    )
    assert same.status_code == 200


def test_thumbnail_is_small_and_cached_outside_the_photo_list(client, root):
    response = client.get(f"/thumb/{FRAME}/new.jpg")
    assert response.status_code == 200
    thumb = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    assert min(thumb.shape[:2]) <= 360

    thumbs = root / "cache" / FRAME / ".thumbs"
    assert [p.name for p in thumbs.iterdir()] == ["new.jpg"]  # no temp file left
    # thumbnails must not show up as photos on the frame
    assert len(client.get(f"/api/frames/{FRAME}/photos").json()["photos"]) == 3


def test_photo_and_thumb_reject_bad_paths(client):
    assert client.get(f"/photo/{FRAME}/new.jpg").status_code == 200
    assert client.get(f"/photo/{FRAME}/missing.jpg").status_code == 404
    assert client.get(f"/thumb/{FRAME}/..%2F..%2Fsecret.jpg").status_code == 404
    assert client.get("/api/frames/..%2F..%2Fx/photos").status_code in (400, 404)


def test_faces_are_returned_left_to_right_as_fractions(client):
    faces = client.get(f"/api/frames/{FRAME}/photos/new/faces").json()["faces"]
    assert [round(f["x"], 2) for f in faces] == [0.1, 0.6]
    assert faces[0] == pytest.approx({"x": 0.1, "y": 0.2, "w": 0.2, "h": 0.3}, abs=0.01)


def test_saving_a_reference_writes_a_crop_the_tools_can_use(client, root):
    box = client.get(f"/api/frames/{FRAME}/photos/new/faces").json()["faces"][0]
    response = client.post(
        "/api/references",
        json={"frame_id": FRAME, "asset_id": "new", "box": box, "name": "Aunt Jo"},
    )
    entry = response.json()
    assert response.status_code == 200 and entry["name"] == "Aunt Jo"
    assert entry["id"].startswith("aunt-jo-")

    crop = cv2.imread(entry["path"])
    # The face is 160x180 px at x=80. A 60% margin is 96 px left and right
    # (clipped at the image edge on the left) and 108 px above and below.
    assert crop.shape[1] == pytest.approx(336, abs=3) and crop.shape[0] == pytest.approx(396, abs=3)
    assert client.get(f"/reference/{entry['id']}.jpg").status_code == 200
    assert client.get("/api/frames").json()["references"][0]["name"] == "Aunt Jo"
    assert server.get_selection_impl(root, FRAME)["references"] == [
        {"name": "Aunt Jo", "path": entry["path"]}
    ]

    removed = client.delete(f"/api/references/{entry['id']}").json()
    assert removed["references"] == []
    assert not (root / "references" / f"{entry['id']}.jpg").exists()


def test_unusable_face_is_rejected_and_leaves_nothing_behind(client, root):
    # a region so small that no face can be found in the crop
    response = client.post(
        "/api/references",
        json={"frame_id": FRAME, "asset_id": "new", "name": "Tiny",
              "box": {"x": 0.5, "y": 0.5, "w": 0.01, "h": 0.01}},
    )
    assert response.status_code == 422
    assert "too small" in response.json()["error"]
    assert not list((root / "references").glob("*.jpg"))


def test_find_writes_matches_file_and_delete_removes_thumbnail(root):
    import asyncio

    class Matcher(FaceMatcher):
        def reference_embedding(self, path):
            return np.eye(512, dtype=np.float32)[0], 1

        def face_embeddings(self, path):
            return np.eye(512, dtype=np.float32)[:1]

    class Client:
        async def remove_asset_from_frame(self, frame_id, asset_id):
            pass

    async def scenario():
        found = await server.find_person_in_frame_impl(Matcher(), root, FRAME, "ref.jpg")
        saved = json.loads((root / "flagged" / FRAME / "matches.json").read_text())
        assert saved["matches"] == found["matches"] and len(saved["matches"]) == 3
        assert saved["person"] is None  # a one-off reference photo, not a saved person

        thumbs = root / "cache" / FRAME / ".thumbs"
        thumbs.mkdir()
        (thumbs / "new.jpg").write_bytes(b"thumb")
        await server.delete_assets_impl(Client(), root, FRAME, ["new"], confirm=True)
        assert not (thumbs / "new.jpg").exists()

    asyncio.run(scenario())


def test_reference_crop_tightens_until_the_clicked_face_is_the_main_one(root):
    """A bigger neighbour in the margin must not become the reference."""

    class CrowdedApp:
        # Clicked face: x 400-500 of an 800px-wide photo. A larger neighbour
        # sits just to its left (x 250-390), inside the default margin.
        def get(self, image):
            height, width = image.shape[:2]
            emb = np.eye(512, dtype=np.float32)
            if width == 800:  # the full photo
                return [
                    SimpleNamespace(bbox=np.array([400, 200, 500, 320]), normed_embedding=emb[0]),
                    SimpleNamespace(bbox=np.array([250, 180, 390, 350]), normed_embedding=emb[1]),
                ]
            # In a crop, report whichever faces fall inside it. The crop is
            # centred on the clicked face, so its left edge tells us the margin.
            margin = (width - 100) / 2
            faces = [SimpleNamespace(bbox=np.array([margin, margin * 1.2, margin + 100, margin * 1.2 + 120]),
                                     normed_embedding=emb[0])]
            if margin >= 60:  # wide enough to include the neighbour
                faces.append(SimpleNamespace(bbox=np.array([0, 0, 140, 170]), normed_embedding=emb[1]))
            return faces

    client = TestClient(Review(root, FaceMatcher(app=CrowdedApp())).app(), base_url="http://127.0.0.1")
    response = client.post(
        "/api/references",
        json={"frame_id": FRAME, "asset_id": "new", "name": "Me",
              "box": {"x": 0.5, "y": 200 / 600, "w": 0.125, "h": 0.2}},
    )
    assert response.status_code == 200
    crop = cv2.imread(response.json()["path"])
    # 60% margin (60px) would include the neighbour; 35% (35px) does not.
    assert crop.shape[1] == pytest.approx(170, abs=3)


# ------------------------------------------------- yes / no answers on matches


class PeopleApp:
    """Full photos hold me (left) and my brother (right); crops hold one face."""

    ME = np.eye(512, dtype=np.float32)[0]
    BROTHER = (np.eye(512, dtype=np.float32)[0] * 0.8 + np.eye(512, dtype=np.float32)[1] * 0.6)

    def get(self, image):
        h, w = image.shape[:2]
        if w == 800:
            return [
                SimpleNamespace(bbox=np.array([80, 120, 240, 300]), normed_embedding=self.BROTHER),
                SimpleNamespace(bbox=np.array([480, 120, 640, 300]), normed_embedding=self.ME),
            ]
        if w == 640:  # the "brother only" photo
            return [SimpleNamespace(bbox=np.array([200, 100, 400, 350]), normed_embedding=self.BROTHER)]
        return [SimpleNamespace(bbox=np.array([w * 0.3, h * 0.3, w * 0.7, h * 0.7]), normed_embedding=self.ME)]


@pytest.fixture
def people_client(root):
    write_photo(root / "cache" / FRAME / "old.jpg", width=640, height=480)  # brother only
    client = TestClient(Review(root, FaceMatcher(app=PeopleApp())).app(), base_url="http://127.0.0.1")
    client.post("/api/references", json={
        "frame_id": FRAME, "asset_id": "new", "name": "AJ",
        "box": {"x": 0.6, "y": 0.2, "w": 0.2, "h": 0.3},
    })
    return client


def test_search_then_answers_then_search_again(people_client, root):
    client = people_client
    refused = client.post(f"/api/frames/{FRAME}/feedback", json={"asset_ids": ["new"], "label": "yes"})
    assert refused.status_code == 409  # nothing has been searched yet

    first = client.post(f"/api/frames/{FRAME}/search", json={"person": "aj"}).json()
    assert first["match_count"] == 3  # brother scores 0.8 against me: a false positive
    data = client.get(f"/api/frames/{FRAME}/photos").json()
    assert data["flagged"] == {"person": "aj", "person_name": "AJ", "similarity_threshold": 0.45,
                               "review_mode": "opt_in"}
    assert {p["id"]: p["label"] for p in data["photos"]} == {"new": None, "mid": None, "old": None}

    assert client.post(f"/api/frames/{FRAME}/feedback",
                       json={"asset_ids": ["new", "mid"], "label": "yes"}).json() == {"confirmed": 2, "rejected": 0}
    assert client.post(f"/api/frames/{FRAME}/feedback",
                       json={"asset_ids": ["old"], "label": "no"}).json() == {"confirmed": 2, "rejected": 1}
    labels = {p["id"]: p["label"] for p in client.get(f"/api/frames/{FRAME}/photos").json()["photos"]}
    assert labels == {"new": "yes", "mid": "yes", "old": "no"}

    # the stored "not them" face is the brother's, i.e. the face that matched in that photo
    stored = json.loads((root / "references" / "feedback.json").read_text())["aj"]
    assert np.allclose(stored[f"{FRAME}/old"]["embedding"], PeopleApp.BROTHER, atol=1e-4)
    assert np.allclose(stored[f"{FRAME}/new"]["embedding"], PeopleApp.ME, atol=1e-4)

    second = client.post(f"/api/frames/{FRAME}/search", json={}).json()  # same person, same cut-off
    assert second["match_count"] == 2
    assert second["examples_used"] == {"is_them": 3, "not_them": 1}
    photos = client.get(f"/api/frames/{FRAME}/photos").json()["photos"]
    assert {p["id"] for p in photos if p["similarity"] is not None} == {"new", "mid"}

    # an answer can be withdrawn, and only flagged photos can be answered about
    assert client.post(f"/api/frames/{FRAME}/feedback",
                       json={"asset_ids": ["mid"], "label": None}).json()["confirmed"] == 1
    assert client.post(f"/api/frames/{FRAME}/feedback",
                       json={"asset_ids": ["old"], "label": "yes"}).status_code == 404
    assert server.get_selection_impl(root, FRAME)["people"] == [
        {"person": "aj", "name": "AJ", "references": 1, "confirmed": 1, "rejected": 1}
    ]


def test_search_from_the_page_needs_a_known_person(people_client):
    assert people_client.post(f"/api/frames/{FRAME}/search", json={}).status_code == 409
    response = people_client.post(f"/api/frames/{FRAME}/search", json={"person": "nobody"})
    assert response.status_code == 400 and "No saved reference" in response.json()["error"]


# ------------------------------------------------- people found by grouping


class GalleryApp(PeopleApp):
    """The same photos, but the two people look nothing alike."""

    ME = np.eye(512, dtype=np.float32)[0]
    BROTHER = np.eye(512, dtype=np.float32)[1]


@pytest.fixture
def gallery_client(root):
    write_photo(root / "cache" / FRAME / "old.jpg", width=640, height=480)  # brother only
    review = Review(root, FaceMatcher(app=GalleryApp()))
    return TestClient(review.app(), base_url="http://127.0.0.1"), review


def build(client, review):
    assert client.post("/api/people/build").status_code == 202
    review._build_thread.join(10)
    assert client.get("/api/people/status").json()["state"] == "done"


def test_gallery_is_empty_before_a_build_and_survives_a_corrupt_store(gallery_client, root):
    client, _ = gallery_client
    assert client.get("/api/people").json()["people"] == []
    (root / "people").mkdir()
    (root / "people" / "people.json").write_text("{nope")
    data = client.get("/api/people").json()
    assert data["built_at"] is None and data["build"]["state"] == "idle"


def test_build_then_gallery_then_click_a_person(gallery_client, root):
    client, review = gallery_client
    build(client, review)
    people = client.get("/api/people").json()["people"]
    assert [p["photo_count"] for p in people] == [3, 2]
    me = people[1]
    assert me["frames"] == {FRAME: 2}
    assert client.get(f"/people/{me['id']}/avatar.jpg").status_code == 200
    client.put(f"/api/frames/{FRAME}/selection", json={"asset_ids": ["old"]})  # a hand pick
    found = client.post(f"/api/frames/{FRAME}/search", json={"person": me["id"], "preselect": True}).json()
    assert (found["match_count"], found["review_mode"], found["selected"]) == (2, "opt_out", 3)
    data = client.get(f"/api/frames/{FRAME}/photos").json()
    assert set(data["selection"]) == {"old", "new", "mid"}
    assert data["flagged"]["review_mode"] == "opt_out" and data["flagged"]["person"] == me["id"]


def test_rename_hide_merge_from_the_page(gallery_client):
    client, review = gallery_client
    build(client, review)
    brother, me = client.get("/api/people").json()["people"]
    assert client.patch(f"/api/people/{me['id']}", json={"name": "AJ"}).json()["name"] == "AJ"
    assert client.patch(f"/api/people/{brother['id']}", json={"hidden": True}).json()["hidden"] is True
    assert [p["id"] for p in client.get("/api/people").json()["hidden"]] == [brother["id"]]
    merged = client.post("/api/people/merge", json={"target": me["id"], "other": brother["id"]}).json()
    assert [(p["name"], p["photo_count"]) for p in merged["people"]] == [("AJ", 3)]
    assert client.patch("/api/people/p-00000000", json={"name": "x"}).status_code == 404
    assert client.post("/api/people/merge", json={"target": me["id"], "other": me["id"]}).status_code == 400
    assert client.get("/people/p-00000000/avatar.jpg").status_code == 404


def test_look_inside_a_group_and_take_only_some_of_it(gallery_client):
    client, review = gallery_client
    build(client, review)
    brother, me = client.get("/api/people").json()["people"]
    faces = client.get(f"/api/people/{brother['id']}/faces").json()["faces"]
    assert sorted(f["asset_id"] for f in faces) == ["mid", "new", "old"]
    assert {f["frame_id"] for f in faces} == {FRAME}
    old = next(f for f in faces if f["asset_id"] == "old")
    crop = client.get(f"/face/old/{old['index']}.jpg")
    assert crop.status_code == 200 and crop.headers["content-type"] == "image/jpeg"
    box = client.get(f"/api/face/old/{old['index']}").json()
    assert box["frame_id"] == FRAME and box["box"]["x"] == pytest.approx(200 / 640)
    assert client.get("/face/old/9.jpg").status_code == 404
    assert client.get("/api/face/old/9").status_code == 404
    taken = client.post("/api/people/move-faces", json={
        "target": me["id"], "other": brother["id"], "faces": [["old", old["index"]]]}).json()
    counts = {p["id"]: p["photo_count"] for p in taken["people"]}
    assert counts == {me["id"]: 3, brother["id"]: 2} and taken["can_undo"] is True
    assert client.post("/api/people/move-faces", json={
        "target": me["id"], "other": brother["id"], "faces": [["zz", 0]]}).status_code == 400
    assert client.post("/api/people/move-faces", json={"target": me["id"]}).status_code == 400
    assert client.get("/api/people/p-00000000/faces").status_code == 404


def test_a_second_build_is_refused_while_one_runs(gallery_client, monkeypatch):
    client, review = gallery_client
    gate = threading.Event()

    async def slow(*args, **kwargs):
        gate.wait(5)
        return {}

    monkeypatch.setattr(server, "find_people_impl", slow)
    assert client.post("/api/people/build").status_code == 202
    assert client.post("/api/people/build").status_code == 409
    assert client.get("/api/people/status").json()["state"] == "running"
    gate.set()
    review._build_thread.join(5)


def test_a_failed_build_reports_its_error(gallery_client, monkeypatch):
    client, review = gallery_client

    async def boom(*args, **kwargs):
        raise ValueError("too many faces")

    monkeypatch.setattr(server, "find_people_impl", boom)
    client.post("/api/people/build")
    review._build_thread.join(5)
    assert client.get("/api/people/status").json() == {
        "state": "error", "done": 0, "total": 0, "message": "too many faces"}


# ------------------------------------------------- found in the branch review


def open_person(client, person_id):
    return client.post(f"/api/frames/{FRAME}/search", json={"person": person_id, "preselect": True}).json()


def selection(client):
    return set(client.get(f"/api/frames/{FRAME}/photos").json()["selection"])


def test_opening_a_person_does_not_inherit_another_searchs_cut_off(gallery_client, root):
    client, review = gallery_client
    build(client, review)
    brother, me = client.get("/api/people").json()["people"]
    (root / "flagged" / FRAME).mkdir(parents=True)
    (root / "flagged" / FRAME / "matches.json").write_text(json.dumps(
        {"person": "someone-else", "person_name": "Someone", "similarity_threshold": 0.25, "matches": []}))
    assert open_person(client, me["id"])["similarity_threshold"] == 0.45
    lowered = client.post(f"/api/frames/{FRAME}/search", json={"person": me["id"], "similarity_threshold": 0.3})
    assert lowered.json()["similarity_threshold"] == 0.3
    again = client.post(f"/api/frames/{FRAME}/search", json={})  # "Search again": same person, same cut-off
    assert again.json()["similarity_threshold"] == 0.3


def test_opening_another_person_replaces_the_last_ones_preselection(gallery_client):
    client, review = gallery_client
    build(client, review)
    brother, me = client.get("/api/people").json()["people"]
    assert open_person(client, brother["id"])["selected"] == 3
    assert open_person(client, me["id"])["selected"] == 2
    assert selection(client) == {"new", "mid"}  # "old" was only there for the brother


def test_a_photo_unticked_by_hand_is_not_ticked_again(gallery_client):
    client, review = gallery_client
    build(client, review)
    brother, me = client.get("/api/people").json()["people"]
    open_person(client, me["id"])
    client.put(f"/api/frames/{FRAME}/selection", json={"asset_ids": ["new", "old"]})  # untick mid, hand-pick old
    assert open_person(client, me["id"])["selected"] == 2
    assert selection(client) == {"new", "old"}
    client.post(f"/api/frames/{FRAME}/search", json={})  # "Search again" follows the same rule
    assert selection(client) == {"new", "old"}


def test_a_reference_person_search_never_preselects(people_client):
    found = people_client.post(f"/api/frames/{FRAME}/search", json={"person": "aj", "preselect": True}).json()
    assert (found["review_mode"], found["selected"]) == ("opt_in", 0)


def test_merging_the_person_under_review_keeps_the_review_working(gallery_client, root):
    client, review = gallery_client
    build(client, review)
    brother, me = client.get("/api/people").json()["people"]
    open_person(client, brother["id"])
    client.post("/api/people/merge", json={"target": me["id"], "other": brother["id"]})
    assert client.get(f"/api/frames/{FRAME}/photos").json()["flagged"]["person"] == me["id"]
    said = client.post(f"/api/frames/{FRAME}/feedback", json={"asset_ids": ["old"], "label": "no"})
    assert said.status_code == 200
    assert people.labels(root, me["id"], FRAME) == {"old": "no"}
    assert client.post(f"/api/frames/{FRAME}/search", json={}).status_code == 200


def test_suggestions_no_and_undo_from_the_page(gallery_client, root):
    client, review = gallery_client
    build(client, review)
    brother, me = client.get("/api/people").json()["people"]
    assert client.get(f"/api/people/{me['id']}/suggestions").json() == {"suggestions": []}  # nothing alike
    assert client.get("/api/people/p-00000000/suggestions").status_code == 404
    assert client.post("/api/people/not-same", json={"a": me["id"], "b": brother["id"]}).status_code == 200
    assert client.post("/api/people/not-same", json={"a": me["id"], "b": "p-00000000"}).status_code == 404
    assert client.post("/api/people/unmerge").status_code == 409  # nothing merged yet
    merged = client.post("/api/people/merge", json={"target": me["id"], "other": brother["id"]}).json()
    assert merged["can_undo"] is True and len(merged["people"]) == 1
    undone = client.post("/api/people/unmerge").json()
    assert [p["id"] for p in undone["people"]] == [brother["id"], me["id"]] and undone["can_undo"] is False


# ------------------------------------------------- removing from the page

KITCHEN = "frame-2"


class FakeAura:
    """Stands in for the Aura client: records what it was asked to remove."""

    def __init__(self, error=None, gate=None, restore_error=None):
        self.removed, self.closed = [], False
        self.error, self.gate = error, gate
        self.restore_error, self.added = restore_error, []

    async def add_asset_to_frame(self, frame_id, asset_id):
        if self.gate:
            self.gate.wait(5)
        if self.restore_error:
            raise self.restore_error
        self.added.append((frame_id, asset_id))

    async def remove_asset_from_frame(self, frame_id, asset_id):
        if self.gate:
            self.gate.wait(5)
        if self.error:
            raise self.error
        self.removed.append((frame_id, asset_id))

    async def delete_asset(self, asset_id):
        raise AssertionError("the page must never delete from the account")

    async def aclose(self):
        self.closed = True


def removal_app(root, **kwargs):
    aura = FakeAura(**kwargs)
    review = Review(root, FaceMatcher(app=FakeApp()), client_factory=lambda: aura)
    return TestClient(review.app(), base_url="http://127.0.0.1"), review, aura


def preview_removal(client, scope, ids=None, frame=FRAME):
    body = {"scope": scope} if ids is None else {"scope": scope, "asset_ids": ids}
    return client.post(f"/api/frames/{frame}/removal/preview", json=body)


def confirm_removal(client, scope, ids, frame=FRAME):
    """What the page does: preview, then confirm that very preview."""
    token = preview_removal(client, scope, ids, frame).json()["token"]
    return client.post("/api/removal", json={"token": token})


def finish_removal(client, review):
    review.removal.thread.join(10)
    return client.get("/api/removal/status").json()


def add_kitchen(root, *names):
    folder = root / "cache" / KITCHEN
    for name in names:
        write_photo(folder / f"{name}.jpg")
    (folder / "index.json").write_text(json.dumps({
        name: {"file_name": f"{name}.heic", "taken_at": "2025-07-04T10:00:00Z",
               "local_path": str(folder / f"{name}.jpg")} for name in names}))
    return folder


def test_removal_preview_removes_nothing(root):
    client, review, aura = removal_app(root)
    client.put(f"/api/frames/{FRAME}/selection", json={"asset_ids": ["new", "mid"]})

    preview = preview_removal(client, "frame")

    assert preview.status_code == 200
    body = preview.json()
    assert body["status"] == "confirmation_required" and body["would_delete_count"] == 2
    assert body["frame_id"] == FRAME and body["token"]
    assert {p["asset_id"] for p in body["would_delete"]} == {"new", "mid"}
    assert aura.removed == []
    assert len(client.get(f"/api/frames/{FRAME}/photos").json()["photos"]) == 3


def test_removal_takes_exactly_the_photos_that_were_previewed(root):
    client, review, aura = removal_app(root)
    client.put(f"/api/frames/{FRAME}/selection", json={"asset_ids": ["new", "mid"]})

    assert confirm_removal(client, "frame", ["new"]).status_code == 202
    status = finish_removal(client, review)

    assert status["state"] == "done" and status["result"]["deleted_count"] == 1
    assert aura.removed == [(FRAME, "new")] and aura.closed
    listing = client.get(f"/api/frames/{FRAME}/photos").json()
    assert {p["id"] for p in listing["photos"]} == {"old", "mid"}
    assert listing["selection"] == ["mid"]
    assert (root / "removed" / FRAME / "new.jpg").is_file()  # the copy to put back


@pytest.mark.parametrize("body", [
    {}, {"token": ""}, {"token": "made-up"},
    # Naming photos, a frame or a scope is not enough: only a preview counts.
    {"scope": "frame", "asset_ids": ["new"]}, {"frame_id": FRAME, "scope": "all_frames", "asset_ids": ["new"]},
])
def test_nothing_is_removed_without_a_preview_to_confirm(root, body):
    client, review, aura = removal_app(root)
    preview_removal(client, "frame", ["new"])  # a preview exists, but is not named

    assert client.post("/api/removal", json=body).status_code in (400, 409)
    assert review.removal.thread is None and aura.removed == []


def test_a_preview_can_be_confirmed_once(root):
    client, review, aura = removal_app(root)
    token = preview_removal(client, "frame", ["new"]).json()["token"]

    assert client.post("/api/removal", json={"token": token}).status_code == 202
    finish_removal(client, review)
    assert client.post("/api/removal", json={"token": token}).status_code == 409
    assert aura.removed == [(FRAME, "new")]


def test_only_the_latest_preview_can_be_confirmed(root):
    client, review, aura = removal_app(root)
    add_kitchen(root, "new")
    first = preview_removal(client, "all_frames", ["new"]).json()["token"]
    second = preview_removal(client, "frame", ["new"]).json()["token"]

    # The dialog now shows "this frame only"; the older, wider preview is dead.
    assert client.post("/api/removal", json={"token": first}).status_code == 409
    assert client.post("/api/removal", json={"token": second}).status_code == 202
    finish_removal(client, review)
    assert aura.removed == [(FRAME, "new")]


def test_a_frame_loaded_after_the_preview_is_left_alone(root):
    client, review, aura = removal_app(root)
    token = preview_removal(client, "all_frames", ["new"]).json()["token"]
    add_kitchen(root, "new")  # finishes loading while the dialog is open

    client.post("/api/removal", json={"token": token})
    status = finish_removal(client, review)

    assert aura.removed == [(FRAME, "new")]  # what the dialog listed, no more
    assert status["result"]["frame_removal_count"] == 1
    assert (root / "cache" / KITCHEN / "new.jpg").is_file()


def test_the_page_cannot_delete_from_the_account(root):
    client, review, aura = removal_app(root)
    response = preview_removal(client, "account", ["new"])
    assert response.status_code == 400 and "token" not in response.json()
    assert review.removal.thread is None


def test_a_second_removal_is_refused_while_one_runs(root):
    gate = threading.Event()
    client, review, aura = removal_app(root, gate=gate)

    assert confirm_removal(client, "frame", ["new"]).status_code == 202
    token = preview_removal(client, "frame", ["mid"]).json()["token"]
    assert client.post("/api/removal", json={"token": token}).status_code == 409
    assert client.get("/api/removal/status").json()["state"] == "running"
    gate.set()
    assert finish_removal(client, review)["state"] == "done"
    assert aura.removed == [(FRAME, "new")]


def test_removal_from_every_frame_from_the_page(root):
    folder = add_kitchen(root, "new")
    client, review, aura = removal_app(root)

    preview = preview_removal(client, "all_frames", ["new", "mid"]).json()
    assert preview["frame_removal_count"] == 3 and preview["last_copy_count"] == 2

    client.post("/api/removal", json={"token": preview["token"]})
    status = finish_removal(client, review)

    assert status["result"]["frame_removal_count"] == 3
    assert aura.removed == [(FRAME, "new"), (KITCHEN, "new"), (FRAME, "mid")]
    assert not (folder / "new.jpg").exists()


def test_a_lost_sign_in_stops_the_removal_and_says_so(root):
    client, review, aura = removal_app(root, error=AuraAuthError("Aura rejected the login."))

    confirm_removal(client, "frame", ["new", "mid"])
    status = finish_removal(client, review)

    assert status["state"] == "done" and status["result"]["status"] == "aborted"
    assert status["result"]["deleted_count"] == 0
    assert "Aura rejected the login" in status["result"]["failed"][0]["error"]
    assert len(client.get(f"/api/frames/{FRAME}/photos").json()["photos"]) == 3


def test_a_removal_that_blows_up_reports_its_error(root, monkeypatch):
    client, review, aura = removal_app(root)
    token = preview_removal(client, "frame", ["new"]).json()["token"]

    async def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(server, "delete_assets_impl", boom)
    client.post("/api/removal", json={"token": token})
    status = finish_removal(client, review)

    # A plain sentence, with the technical text kept beside it.
    assert status["state"] == "error" and status["message"] == server.OTHER_TROUBLE
    assert status["detail"] == "RuntimeError: disk full"


def test_one_job_at_a_time_across_loading_finding_people_and_removing(root, monkeypatch):
    gate = threading.Event()
    client, review, aura = removal_app(root, gate=gate)
    assert confirm_removal(client, "frame", ["new"]).status_code == 202

    # Each would rewrite the files the removal is in the middle of changing.
    assert client.post("/api/sync", json={"frame_ids": [FRAME]}).status_code == 409
    assert client.post("/api/people/build").status_code == 409
    gate.set()
    finish_removal(client, review)
    assert client.post("/api/people/build").status_code == 202
    review._build_thread.join(10)


# --------------------------------------------- only this computer may ask

@pytest.mark.parametrize("host", ["evil.example:8766", "evil.example", "127.0.0.1.evil.example"])
def test_requests_for_another_host_name_are_refused(root, host):
    """A site whose name is re-pointed at 127.0.0.1 reaches this server with
    its own name as Host; it must not be able to read photos or remove them."""
    client, review, aura = removal_app(root)
    headers = {"host": host, "origin": f"http://{host}"}

    assert client.get(f"/photo/{FRAME}/new.jpg", headers={"host": host}).status_code == 403
    assert client.get("/api/frames", headers={"host": host}).status_code == 403
    preview = client.post(
        f"/api/frames/{FRAME}/removal/preview", headers=headers,
        json={"scope": "frame", "asset_ids": ["new"]})
    assert preview.status_code == 403
    sign_in = client.post(
        "/api/account/sign-in", headers=headers, json={"email": "a@b.c", "password": "x"})
    assert sign_in.status_code == 403
    assert aura.removed == []


@pytest.mark.parametrize("host", ["127.0.0.1:8766", "localhost:8766", "[::1]:8766", "localhost"])
def test_this_computers_own_names_are_allowed(root, host):
    client, review, aura = removal_app(root)
    assert client.get("/api/frames", headers={"host": host}).status_code == 200


# ------------------------------------------- signing in and getting the photos

class FakeAccount(FakeAura):
    """An Aura account with two frames; "new" is on both of them."""

    FRAMES = [
        {"id": FRAME, "name": "Living Room", "asset_count": 3},
        {"id": KITCHEN, "name": "Kitchen", "asset_count": 2},
    ]
    ASSETS = {FRAME: ["old", "new", "mid"], KITCHEN: ["new", "k1"]}

    def __init__(self, email=None, password=None, **kwargs):
        super().__init__(**kwargs)
        self.email, self.password = email, password
        self.session = None
        self.downloads = []

    async def login(self):
        if self.password != "right":
            raise AuraAuthError("Aura rejected the login. (HTTP 475)")
        self.session = {"token": "tok-1", "user_id": "u1"}

    async def list_frames(self):
        if self.error:
            raise self.error
        return self.FRAMES

    async def list_assets(self, frame_id):
        return [
            {"id": a, "user_id": "u1", "file_name": f"{a}.heic", "taken_at": "2024-01-01T00:00:00Z"}
            for a in self.ASSETS[frame_id]
        ]

    async def download_photo(self, user_id, file_name, dest, max_side=None):
        self.downloads.append(dest.stem)
        write_photo(dest)
        return dest


def account_app(tmp_path, monkeypatch, **kwargs):
    monkeypatch.delenv("AURA_EMAIL", raising=False)
    monkeypatch.delenv("AURA_PASSWORD", raising=False)
    made = []

    def factory(email=None, password=None):
        made.append(FakeAccount(email, password, **kwargs))
        return made[-1]

    review = Review(tmp_path, FaceMatcher(app=GalleryApp()), client_factory=factory)
    return TestClient(review.app(), base_url="http://127.0.0.1"), review, made


def test_signing_in_saves_the_token_and_never_the_password(tmp_path, monkeypatch):
    client, review, made = account_app(tmp_path, monkeypatch)
    assert client.get("/api/account").json() == {"signed_in": False}

    response = client.post(
        "/api/account/sign-in", json={"email": "me@example.com", "password": "right"})

    assert response.status_code == 200 and response.json() == {"signed_in": True}
    assert (made[0].email, made[0].password, made[0].closed) == ("me@example.com", "right", True)
    saved = (tmp_path / "session.json").read_text()
    assert json.loads(saved) == {"token": "tok-1", "user_id": "u1"}
    assert "right" not in saved and "me@example.com" not in saved
    assert client.get("/api/account").json() == {"signed_in": True}

    assert client.post("/api/account/sign-out").json() == {"signed_in": False}
    assert not (tmp_path / "session.json").exists()


def test_a_wrong_password_is_refused_plainly_and_saves_nothing(tmp_path, monkeypatch):
    client, review, made = account_app(tmp_path, monkeypatch)

    response = client.post(
        "/api/account/sign-in", json={"email": "me@example.com", "password": "wrong"})

    assert response.status_code == 401
    assert response.json()["error"] == "Aura did not accept that email and password."
    assert not (tmp_path / "session.json").exists()


@pytest.mark.parametrize("body", [{}, {"email": "me@example.com"}, {"email": "", "password": "x"}, []])
def test_sign_in_needs_an_email_and_a_password(tmp_path, monkeypatch, body):
    client, review, made = account_app(tmp_path, monkeypatch)
    assert client.post("/api/account/sign-in", json=body).status_code == 400
    assert made == []


def test_account_frames_show_what_is_on_aura_and_what_is_loaded_here(tmp_path, monkeypatch):
    client, review, made = account_app(tmp_path, monkeypatch)
    write_photo(tmp_path / "cache" / FRAME / "old.jpg")
    (tmp_path / "cache" / FRAME / "index.json").write_text("{}")

    frames = client.get("/api/account/frames").json()["frames"]

    assert frames == [
        {"id": FRAME, "name": "Living Room", "asset_count": 3, "loaded": 1, "synced": True},
        {"id": KITCHEN, "name": "Kitchen", "asset_count": 2, "loaded": 0, "synced": False},
    ]
    # Names are remembered, so the frame picker and warnings can use them.
    assert json.loads((tmp_path / "cache" / "frames.json").read_text())[KITCHEN] == "Kitchen"


def test_account_frames_say_when_the_sign_in_is_gone(tmp_path, monkeypatch):
    client, review, made = account_app(
        tmp_path, monkeypatch, error=AuraAuthError("Your Aura sign-in has expired. Sign in again."))

    response = client.get("/api/account/frames")

    assert response.status_code == 401 and "Sign in again" in response.json()["error"]


def test_getting_the_photos_loads_each_frame_and_finds_the_people(tmp_path, monkeypatch):
    client, review, made = account_app(tmp_path, monkeypatch)

    started = client.post("/api/sync", json={"frame_ids": [FRAME, KITCHEN]})
    assert started.status_code == 202
    review.sync.thread.join(20)
    status = client.get("/api/sync/status").json()

    assert status["state"] == "done", status
    assert status["result"]["frames"] == [
        {"id": FRAME, "name": "Living Room", "photos": 3, "failed": 0},
        {"id": KITCHEN, "name": "Kitchen", "photos": 2, "failed": 0},
    ]
    # "new" is on both frames but is fetched once.
    assert sorted(made[0].downloads) == ["k1", "mid", "new", "old"] and made[0].closed
    # Two frames loaded: "All frames" leads, counting "new" once.
    assert {f["name"]: f["photos"] for f in client.get("/api/frames").json()["frames"]} == {
        "All frames": 4, "Living Room": 3, "Kitchen": 2}
    assert client.get("/api/people").json()["built_at"]
    assert status["result"]["people"] == len(client.get("/api/people").json()["people"])


def test_getting_the_photos_only_loads_the_frames_asked_for(tmp_path, monkeypatch):
    client, review, made = account_app(tmp_path, monkeypatch)

    client.post("/api/sync", json={"frame_ids": [KITCHEN]})
    review.sync.thread.join(20)

    assert [f["name"] for f in client.get("/api/frames").json()["frames"]] == ["Kitchen"]
    assert client.post("/api/sync", json={"frame_ids": []}).status_code == 400
    assert client.post("/api/sync", json={"frame_ids": ["../etc"]}).status_code == 400


def test_a_failed_load_reports_why(tmp_path, monkeypatch):
    client, review, made = account_app(
        tmp_path, monkeypatch, error=AuraAuthError("Your Aura sign-in has expired. Sign in again."))

    client.post("/api/sync", json={"frame_ids": [FRAME]})
    review.sync.thread.join(20)

    status = client.get("/api/sync/status").json()
    assert status["state"] == "error" and "Sign in again" in status["message"]


# ------------------------------------------- one review across every frame

ALL = "all"
PORCH = "frame-3"


def add_frame(root, frame, *names):
    folder = root / "cache" / frame
    for name in names:
        write_photo(folder / f"{name}.jpg")
    (folder / "index.json").write_text(json.dumps({
        name: {"file_name": f"{name}.heic", "taken_at": "2025-07-04T10:00:00Z",
               "local_path": str(folder / f"{name}.jpg")} for name in names}))
    return folder


def two_frames(root):
    """Living Room holds old, new, mid; Kitchen holds new and k1."""
    add_kitchen(root, "new", "k1")
    (root / "cache" / "frames.json").write_text(json.dumps({FRAME: "Living Room", KITCHEN: "Kitchen"}))


def nothing_stored_under_all(root):
    for place in ("cache", "flagged", "removed"):
        assert not (root / place / ALL).exists(), place


def test_all_frames_is_listed_first_once_two_frames_are_loaded(root):
    client, review, aura = removal_app(root)
    assert [f["id"] for f in client.get("/api/frames").json()["frames"]] == [FRAME]  # one frame: no "all"
    two_frames(root)

    frames = client.get("/api/frames").json()["frames"]

    assert frames[0] == {"id": ALL, "name": "All frames", "photos": 4}
    assert [f["id"] for f in frames[1:]] == [FRAME, KITCHEN]


def test_the_all_listing_has_each_photo_once_with_its_frames(root):
    two_frames(root)
    client, review, aura = removal_app(root)

    data = client.get(f"/api/frames/{ALL}/photos").json()

    assert data["frame_id"] == ALL and data["sync_complete"] is True
    assert sorted(p["id"] for p in data["photos"]) == ["k1", "mid", "new", "old"]
    assert {p["id"]: p["frames"] for p in data["photos"]} == {
        "old": [FRAME], "new": [FRAME, KITCHEN], "mid": [FRAME], "k1": [KITCHEN]}
    # Per-frame listings are as they were: no "frames" on each photo.
    assert all("frames" not in p for p in client.get(f"/api/frames/{KITCHEN}/photos").json()["photos"])
    nothing_stored_under_all(root)


def test_thumbnails_and_photos_in_all_come_from_a_frame_that_holds_them(root):
    two_frames(root)
    client, review, aura = removal_app(root)

    assert client.get(f"/thumb/{ALL}/k1.jpg").status_code == 200
    assert client.get(f"/photo/{ALL}/new.jpg").status_code == 200
    assert client.get(f"/photo/{ALL}/missing.jpg").status_code == 404
    assert client.get(f"/thumb/{ALL}/..%2Fsecret.jpg").status_code == 404
    assert (root / "cache" / KITCHEN / ".thumbs" / "k1.jpg").is_file()
    assert len(client.get(f"/api/frames/{ALL}/photos/new/faces").json()["faces"]) == 2
    nothing_stored_under_all(root)


def test_the_all_selection_is_kept_on_its_own(root):
    two_frames(root)
    client, review, aura = removal_app(root)
    client.put(f"/api/frames/{FRAME}/selection", json={"asset_ids": ["old"]})

    saved = client.put(f"/api/frames/{ALL}/selection", json={"asset_ids": ["k1", "new", "nope", "../x"]})

    assert saved.json() == {"selected": 2}
    assert json.loads((root / "selections" / "all.json").read_text())["asset_ids"] == ["k1", "new"]
    assert client.get(f"/api/frames/{ALL}/photos").json()["selection"] == ["k1", "new"]
    assert client.get(f"/api/frames/{FRAME}/photos").json()["selection"] == ["old"]


def test_all_is_never_loaded_as_a_real_frame(root):
    two_frames(root)
    client, review, aura = removal_app(root)
    assert client.post("/api/sync", json={"frame_ids": [ALL]}).status_code == 400
    assert client.post("/api/sync", json={"frame_ids": [FRAME, ALL]}).status_code == 400
    # A folder of that name, however it got there, is not listed as a frame.
    write_photo(root / "cache" / ALL / "stray.jpg")
    frames = client.get("/api/frames").json()["frames"]
    assert [f["id"] for f in frames] == [ALL, FRAME, KITCHEN] and frames[0]["photos"] == 4
    assert "stray" not in {p["id"] for p in client.get(f"/api/frames/{ALL}/photos").json()["photos"]}


@pytest.fixture
def all_gallery(root):
    write_photo(root / "cache" / FRAME / "old.jpg", width=640, height=480)  # brother only
    two_frames(root)
    review = Review(root, FaceMatcher(app=GalleryApp()))
    client = TestClient(review.app(), base_url="http://127.0.0.1")
    build(client, review)
    brother, me = client.get("/api/people").json()["people"]
    return client, review, me


def open_in_all(client, person_id):
    return client.post(f"/api/frames/{ALL}/search", json={"person": person_id, "preselect": True})


def test_opening_a_person_in_all_selects_their_photos_on_every_frame(all_gallery, root):
    client, review, me = all_gallery
    assert me["frames"] == {FRAME: 2, KITCHEN: 2}
    client.put(f"/api/frames/{FRAME}/selection", json={"asset_ids": ["old"]})

    found = open_in_all(client, me["id"])

    assert found.status_code == 200
    assert (found.json()["match_count"], found.json()["review_mode"], found.json()["selected"]) == (3, "opt_out", 3)
    data = client.get(f"/api/frames/{ALL}/photos").json()
    assert set(data["selection"]) == {"new", "mid", "k1"}
    assert data["flagged"]["person"] == me["id"] and data["flagged"]["review_mode"] == "opt_out"
    assert {p["id"] for p in data["photos"] if p["similarity"] is not None} == {"new", "mid", "k1"}
    # Each frame's own selection is left as it was.
    assert client.get(f"/api/frames/{FRAME}/photos").json()["selection"] == ["old"]
    nothing_stored_under_all(root)


def test_not_them_in_all_holds_on_every_frame_the_photo_is_on(all_gallery, root):
    client, review, me = all_gallery
    open_in_all(client, me["id"])

    said = client.post(f"/api/frames/{ALL}/feedback", json={"asset_ids": ["new"], "label": "no"})

    assert said.status_code == 200 and said.json() == {"confirmed": 0, "rejected": 1}
    assert people.labels(root, me["id"], FRAME) == {"new": "no"}
    assert people.labels(root, me["id"], KITCHEN) == {"new": "no"}
    assert {p["id"]: p["label"] for p in client.get(f"/api/frames/{ALL}/photos").json()["photos"]
            if p["similarity"] is not None}["new"] == "no"
    # A search on Kitchen alone now leaves it out, as does one in "all".
    kitchen = client.post(f"/api/frames/{KITCHEN}/search", json={"person": me["id"], "preselect": True}).json()
    assert kitchen["match_count"] == 1
    assert open_in_all(client, me["id"]).json()["match_count"] == 2
    # Withdrawn on every frame too; and only flagged photos can be answered about.
    assert client.post(f"/api/frames/{ALL}/feedback", json={"asset_ids": ["new"], "label": None}).status_code == 404
    open_in_all(client, me["id"])
    assert client.post(f"/api/frames/{ALL}/feedback", json={"asset_ids": ["old"], "label": "no"}).status_code == 404


def test_withdrawing_not_them_in_all_clears_it_everywhere(all_gallery, root):
    client, review, me = all_gallery
    open_in_all(client, me["id"])
    client.post(f"/api/frames/{ALL}/feedback", json={"asset_ids": ["new"], "label": "no"})
    # Still flagged in the "all" view until searched again: withdraw it there.
    assert client.post(f"/api/frames/{ALL}/feedback", json={"asset_ids": ["new"], "label": None}).status_code == 200
    assert people.labels(root, me["id"], FRAME) == {} and people.labels(root, me["id"], KITCHEN) == {}


def test_a_reference_saved_from_all_names_the_frame_that_holds_the_photo(root):
    two_frames(root)
    client, review, aura = removal_app(root)
    box = client.get(f"/api/frames/{ALL}/photos/k1/faces").json()["faces"][0]
    entry = client.post("/api/references", json={"frame_id": ALL, "asset_id": "k1", "box": box, "name": "Jo"}).json()
    assert entry["frame_id"] == KITCHEN
    nothing_stored_under_all(root)


def test_text_search_in_all_merges_every_frame_by_best_score(root, monkeypatch):
    two_frames(root)
    client, review, aura = removal_app(root)
    scores = {FRAME: {"new": 0.30, "old": 0.25}, KITCHEN: {"new": 0.40, "k1": 0.23}}
    asked = []

    async def fake(searcher, root_, frame_id, query, **kwargs):
        asked.append(frame_id)
        matches = [{"asset_id": a, "file_name": None, "similarity": s} for a, s in scores[frame_id].items()]
        return {"query": query, "match_count": len(matches), "similarity_threshold": 0.22, "matches": matches}

    monkeypatch.setattr(server, "find_photos_by_description_impl", fake)
    monkeypatch.setattr(server, "get_searcher", lambda: None)
    review.searcher.model_present = lambda: True  # the search model is here

    found = client.post(f"/api/frames/{ALL}/search-text", json={"query": "a dog"}).json()

    assert asked == [FRAME, KITCHEN]
    assert [(m["asset_id"], m["similarity"]) for m in found["matches"]] == [
        ("new", 0.40), ("old", 0.25), ("k1", 0.23)]
    assert found["match_count"] == 3


def test_removal_from_all_takes_each_photo_off_every_frame_it_is_on(root):
    two_frames(root)
    client, review, aura = removal_app(root)
    client.put(f"/api/frames/{ALL}/selection", json={"asset_ids": ["new", "mid", "k1"]})

    preview = preview_removal(client, "all_frames", frame=ALL).json()

    assert preview["frame_id"] == ALL and preview["scope"] == "all_frames" and preview["token"]
    assert preview["would_delete_count"] == 3 and preview["frame_removal_count"] == 4
    assert preview["frames_affected"] == [
        {"id": FRAME, "name": "Living Room", "count": 2}, {"id": KITCHEN, "name": "Kitchen", "count": 2}]
    assert preview["last_copy_count"] == 3
    assert preview["unknown_asset_ids"] == [] and preview["no_copy_asset_ids"] == []
    assert preview["warnings"] == [] and preview["unsynced_frames"] == []
    assert aura.removed == []

    assert client.post("/api/removal", json={"token": preview["token"]}).status_code == 202
    status = finish_removal(client, review)

    result = status["result"]
    assert status["state"] == "done" and result["status"] == "completed"
    assert (result["deleted_count"], result["partial_count"], result["frame_removal_count"],
            result["failed_count"]) == (3, 0, 4, 0)
    assert sorted(aura.removed) == sorted([(FRAME, "new"), (KITCHEN, "new"), (FRAME, "mid"), (KITCHEN, "k1")])
    assert {p["id"] for p in client.get(f"/api/frames/{ALL}/photos").json()["photos"]} == {"old"}
    for frame, name in [(FRAME, "new"), (KITCHEN, "new"), (FRAME, "mid"), (KITCHEN, "k1")]:
        assert (root / "removed" / frame / f"{name}.jpg").is_file()
    nothing_stored_under_all(root)


def test_removal_from_all_ignores_any_scope_and_refuses_the_account(root):
    two_frames(root)
    client, review, aura = removal_app(root)
    narrow = preview_removal(client, "frame", ["new"], frame=ALL).json()
    assert narrow["scope"] == "all_frames" and narrow["frame_removal_count"] == 2
    refused = preview_removal(client, "account", ["new"], frame=ALL)
    assert refused.status_code == 400 and "token" not in refused.json()


def test_removal_from_all_never_touches_a_frame_loaded_after_the_preview(root):
    two_frames(root)
    client, review, aura = removal_app(root)
    token = preview_removal(client, "all_frames", ["new", "k1"], frame=ALL).json()["token"]
    add_frame(root, PORCH, "new", "k1")  # finishes loading while the dialog is open
    client.put(f"/api/frames/{ALL}/selection", json={"asset_ids": ["new", "k1", "old", "mid"]})

    client.post("/api/removal", json={"token": token})
    status = finish_removal(client, review)

    assert sorted(aura.removed) == sorted([(FRAME, "new"), (KITCHEN, "new"), (KITCHEN, "k1")])
    assert status["result"]["frame_removal_count"] == 3
    assert (root / "cache" / PORCH / "new.jpg").is_file() and (root / "cache" / PORCH / "k1.jpg").is_file()
    assert (root / "cache" / FRAME / "old.jpg").is_file() and (root / "cache" / FRAME / "mid.jpg").is_file()


def test_removal_from_all_only_takes_what_the_preview_listed(root):
    """Photos left out of the preview (still loading) stay, whatever is on disk now."""
    two_frames(root)
    (root / "cache" / KITCHEN / "index.json").write_text(json.dumps({}))  # Kitchen half-loaded
    client, review, aura = removal_app(root)
    preview = preview_removal(client, "all_frames", ["new", "k1"], frame=ALL).json()
    assert preview["would_delete_count"] == 1 and preview["unknown_asset_ids"] == ["k1"]
    assert preview["frames_affected"] == [{"id": FRAME, "name": "Living Room", "count": 1}]
    add_kitchen(root, "new", "k1")  # Kitchen finishes loading

    client.post("/api/removal", json={"token": preview["token"]})
    finish_removal(client, review)

    assert aura.removed == [(FRAME, "new")]
    assert (root / "cache" / KITCHEN / "k1.jpg").is_file() and (root / "cache" / KITCHEN / "new.jpg").is_file()


@pytest.mark.parametrize("body", [
    {}, {"token": ""}, {"token": "made-up"},
    {"frame_id": ALL, "scope": "all_frames", "asset_ids": ["new"]},
])
def test_nothing_is_removed_from_all_without_its_preview(root, body):
    two_frames(root)
    client, review, aura = removal_app(root)
    preview_removal(client, "all_frames", ["new"], frame=ALL)

    assert client.post("/api/removal", json=body).status_code in (400, 409)
    assert review.removal.thread is None and aura.removed == []


def test_a_stale_all_preview_is_refused(root):
    two_frames(root)
    client, review, aura = removal_app(root)
    wide = preview_removal(client, "all_frames", ["new", "k1"], frame=ALL).json()["token"]
    narrow = preview_removal(client, "frame", ["new"]).json()["token"]

    assert client.post("/api/removal", json={"token": wide}).status_code == 409
    assert client.post("/api/removal", json={"token": narrow}).status_code == 202
    finish_removal(client, review)
    assert aura.removed == [(FRAME, "new")]
    assert client.post("/api/removal", json={"token": narrow}).status_code == 409  # once only


def test_a_lost_sign_in_stops_an_all_removal_before_the_next_frame(root):
    two_frames(root)
    client, review, aura = removal_app(root, error=AuraAuthError("Aura rejected the login."))

    confirm_removal(client, "all_frames", ["new", "k1"], frame=ALL)
    status = finish_removal(client, review)

    result = status["result"]
    assert status["state"] == "done" and result["status"] == "aborted"
    assert result["deleted_count"] == 0 and aura.removed == []
    stopped = {(f["asset_id"], f.get("frame_id")) for f in result["failed"]}
    assert ("k1", KITCHEN) in stopped  # the second group was never started
    assert all("Aura rejected the login" in f["error"] for f in result["failed"])
    assert result["failed_count"] == len(result["failed"])
    assert len(client.get(f"/api/frames/{ALL}/photos").json()["photos"]) == 4


def test_all_removal_warns_about_photos_marked_not_them(all_gallery, root):
    client, review, me = all_gallery
    open_in_all(client, me["id"])
    client.post(f"/api/frames/{ALL}/feedback", json={"asset_ids": ["k1"], "label": "no"})
    aura = FakeAura()
    review.client_factory = lambda: aura

    preview = preview_removal(client, "all_frames", ["new", "k1", "old"], frame=ALL).json()

    assert preview["reviewed_for"]
    assert any("NOT" in w for w in preview["warnings"])
    assert any("selected by hand" in w for w in preview["warnings"])
    assert len(preview["warnings"]) == 2  # one line each, not one per frame



# ------------------------------------------------ WP2b: the face model, on consent

class RecordingAnalysis:
    """Stands in for insightface's FaceAnalysis once the files are there."""

    built = []

    def __init__(self, name, allowed_modules, providers, provider_options=None, root=None):
        self.root = root
        self.models = {}
        RecordingAnalysis.built.append(self)

    def prepare(self, ctx_id, det_size):
        pass

    def get(self, image):
        return []


def fake_model_files(folder):
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("det_10g.onnx", "w600k_r50.onnx"):
        (folder / name).write_bytes(b"onnx")


@pytest.fixture
def no_model(tmp_path, monkeypatch):
    """A Review whose face model folder is empty, with loading and fetching watched."""
    from aura_mcp import face_matcher

    monkeypatch.delenv("AURA_EMAIL", raising=False)
    monkeypatch.delenv("AURA_PASSWORD", raising=False)
    calls = {"load": 0, "fetch": []}

    def no_loading():
        calls["load"] += 1
        raise AssertionError("the face model was loaded before the user agreed")

    def fetch(name, root, force):
        calls["fetch"].append((name, root, force))
        fake_model_files(Path(root) / "models" / name)

    monkeypatch.setattr(face_matcher, "_face_analysis_class", no_loading)
    monkeypatch.setattr(face_matcher, "_fetch_model", fetch)
    home = tmp_path / "home"
    matcher = FaceMatcher(model_root=tmp_path / "model-root", download=False)
    made = []

    def factory(email=None, password=None):
        made.append(FakeAccount(email, password))
        return made[-1]

    review = Review(home, matcher, client_factory=factory)
    return TestClient(review.app(), base_url="http://127.0.0.1"), review, calls, home


def test_the_page_never_fetches_the_model_on_its_own():
    assert Review(Path("/nonexistent-home")).matcher.download is False


def test_nothing_loads_the_model_before_consent(no_model, tmp_path):
    client, review, calls, home = no_model

    assert client.get("/").status_code == 200
    setup = client.get("/api/setup").json()["face_model"]
    assert setup["present"] is False and setup["agreed"] is False
    assert setup["download"]["state"] == "idle"
    assert client.get("/api/frames").status_code == 200
    assert client.get("/api/people").status_code == 200
    review.warm_model()  # what start-up does: nothing, while the model is missing

    # Getting photos reads faces, so it waits for the model.
    response = client.post("/api/sync", json={"frame_ids": [FRAME]})
    assert response.status_code == 409 and "face model" in response.json()["error"]
    assert client.post("/api/people/build").status_code == 409
    assert review.sync.thread is None

    assert calls == {"load": 0, "fetch": []}
    assert not (tmp_path / "model-root").exists()


@pytest.mark.parametrize("body", [{}, {"agree": False}, {"agree": "yes"}, []])
def test_the_download_needs_an_explicit_yes(no_model, body):
    client, review, calls, home = no_model
    assert client.post("/api/setup/face-model", json=body).status_code == 400
    assert calls["fetch"] == [] and review.model_job.thread is None


def test_after_consent_the_model_is_downloaded_and_then_used(no_model, tmp_path, monkeypatch):
    from aura_mcp import face_matcher

    client, review, calls, home = no_model
    RecordingAnalysis.built = []
    monkeypatch.setattr(face_matcher, "_face_analysis_class", lambda: RecordingAnalysis)

    assert client.post("/api/setup/face-model", json={"agree": True}).status_code == 202
    review.model_job.thread.join(10)

    status = client.get("/api/setup/face-model").json()
    assert status["state"] == "done", status
    assert calls["fetch"] == [("buffalo_l", str(tmp_path / "model-root"), False)]
    setup = client.get("/api/setup").json()["face_model"]
    assert setup["present"] is True and setup["agreed"] is True
    assert json.loads((home / "settings.json").read_text())["face_model_agreed_at"]
    # Loaded from the folder it was put in, ready for the first photos.
    assert RecordingAnalysis.built and RecordingAnalysis.built[-1].root == str(tmp_path / "model-root")
    # From now on the page behaves as before.
    assert client.post("/api/sync", json={"frame_ids": [FRAME]}).status_code == 202
    review.sync.thread.join(20)


def test_a_failed_model_download_is_said_plainly(no_model, monkeypatch):
    from aura_mcp import face_matcher

    client, review, calls, home = no_model

    def offline(name, root, force):
        raise ConnectionError("Max retries exceeded with url: /deepinsight/insightface/releases")

    monkeypatch.setattr(face_matcher, "_fetch_model", offline)
    client.post("/api/setup/face-model", json={"agree": True})
    review.model_job.thread.join(10)

    status = client.get("/api/setup/face-model").json()
    assert status["state"] == "error"
    assert "could not be downloaded" in status["message"] and "Max retries" in status["detail"]
    assert client.get("/api/setup").json()["face_model"]["present"] is False


def test_the_page_has_the_consent_step(client):
    page = client.get("/").text
    step = page.split('<template id="tpl-model">', 1)[1].split("</template>", 1)[0]
    assert "InsightFace" in step and "non-commercial" in step
    assert "https://github.com/deepinsight/insightface" in step  # their terms
    assert "I agree, download it" in step


# --------------------------------------------- WP2b: a plain notice at sign-in

def test_the_sign_in_form_carries_the_notice_and_a_required_checkbox(client):
    page = client.get("/").text
    form = page.split('<template id="tpl-signin">', 1)[1].split("</template>", 1)[0]
    for words in ("independent third-party app, not affiliated with Aura", "your own Aura account", "what you approve",
                  "don't allow automation tools without its permission",
                  "password goes only to Aura", "not saved", "stay on this computer"):
        assert words in form, words
    assert '<input type="checkbox" id="signin-understand" required>' in form
    assert "I understand" in form
    # Sign in starts switched off; the page turns it on with the checkbox.
    assert '<button class="btn primary big" type="submit" id="signin-go" disabled>' in form


# ------------------------------------------------- WP2b: putting photos back

def removed_in_two_frames(root):
    """'new' removed from both frames, then 'old' from the first only."""
    two_frames(root)
    client, review, aura = removal_app(root)
    client.post(f"/api/removal", json={"token": preview_removal(
        client, "all_frames", ["new"], frame=ALL).json()["token"]})
    finish_removal(client, review)
    confirm_removal(client, "frame", ["old"])
    finish_removal(client, review)
    assert aura.removed == [(FRAME, "new"), (KITCHEN, "new"), (FRAME, "old")]
    return client, review, aura


def test_the_removed_list_covers_every_frame_newest_first(root):
    client, review, aura = removed_in_two_frames(root)
    logs = {f: json.loads((root / "removed" / f / "removed.json").read_text()) for f in (FRAME, KITCHEN)}
    logs[FRAME]["old"]["removed_at"] = "2026-11-19T12:00:00+00:00"
    logs[FRAME]["new"]["removed_at"] = logs[KITCHEN]["new"]["removed_at"] = "2026-11-18T12:00:00+00:00"
    for frame, log in logs.items():
        (root / "removed" / frame / "removed.json").write_text(json.dumps(log))

    everything = client.get(f"/api/frames/{ALL}/removed").json()["removed"]

    assert [p["id"] for p in everything] == ["old", "new"]
    new = everything[1]
    assert [f["id"] for f in new["frames"]] == [FRAME, KITCHEN]
    assert [f["name"] for f in new["frames"]] == ["Living Room", "Kitchen"]
    assert new["removed_at"] == "2026-11-18T12:00:00+00:00"
    assert new["taken_at"] == "2025-07-04T10:00:00Z"
    assert [p["id"] for p in client.get(f"/api/frames/{KITCHEN}/removed").json()["removed"]] == ["new"]
    assert [p["id"] for p in client.get(f"/api/frames/{FRAME}/removed").json()["removed"]] == ["old", "new"]
    assert client.get("/api/frames/..%2Fx/removed").status_code in (400, 404)
    nothing_stored_under_all(root)


def test_removed_thumbnails_come_only_from_the_kept_copies(root):
    client, review, aura = removed_in_two_frames(root)

    for frame in (ALL, FRAME, KITCHEN):
        response = client.get(f"/removed-thumb/{frame}/new.jpg")
        assert response.status_code == 200, frame
        thumb = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        assert min(thumb.shape[:2]) <= 360
    # A photo still on the frame is not served from here, nor anything outside.
    assert client.get(f"/removed-thumb/{FRAME}/mid.jpg").status_code == 404
    assert client.get(f"/removed-thumb/{KITCHEN}/old.jpg").status_code == 404
    assert client.get(f"/removed-thumb/{FRAME}/removed.jpg").status_code == 404
    assert client.get("/removed-thumb/..%2F..%2Fcache/mid.jpg").status_code == 404
    assert client.get(f"/removed-thumb/{FRAME}/..%2F..%2Fcache%2Fframe-1%2Fmid.jpg").status_code == 404


def finish_restore(client, review):
    review.restore.thread.join(10)
    return client.get("/api/restore/status").json()


def test_putting_a_photo_back_on_every_frame_it_came_off(root):
    client, review, aura = removed_in_two_frames(root)

    response = client.post("/api/restore", json={"frame_id": ALL, "asset_ids": ["new"]})
    assert response.status_code == 202
    status = finish_restore(client, review)

    assert status["state"] == "done", status
    result = status["result"]
    assert result["restored_count"] == 2 and result["failed_count"] == 0
    assert sorted(aura.added) == [(FRAME, "new"), (KITCHEN, "new")]
    for frame in (FRAME, KITCHEN):
        assert (root / "cache" / frame / "new.jpg").is_file()
    assert [p["id"] for p in client.get(f"/api/frames/{ALL}/removed").json()["removed"]] == ["old"]
    assert "new" in {p["id"] for p in client.get(f"/api/frames/{ALL}/photos").json()["photos"]}


def test_putting_back_says_plainly_when_aura_no_longer_has_the_photo(root):
    client, review, aura = removed_in_two_frames(root)
    aura.restore_error = AuraRefusedError(
        "Aura could not add asset old to frame frame-1 (number_failed=1).")

    client.post("/api/restore", json={"frame_id": FRAME, "asset_ids": ["old"]})
    result = finish_restore(client, review)["result"]

    kept = root / "removed" / FRAME / "old.jpg"
    assert result["restored_count"] == 0 and result["failed_count"] == 1
    failure = result["failed"][0]
    assert failure["asset_id"] == "old" and failure["frame_id"] == FRAME
    assert "no longer has" in failure["message"] and str(kept) in failure["message"]
    assert "Aura app" in failure["message"]
    assert failure["local_copy"] == str(kept) and kept.is_file()
    assert "number_failed" in failure["detail"] and "number_failed" not in failure["message"]
    # Still listed, so it can be found again.
    assert "old" in {p["id"] for p in client.get(f"/api/frames/{FRAME}/removed").json()["removed"]}


def test_putting_back_waits_for_other_work_and_checks_what_it_is_given(root):
    client, review, aura = removed_in_two_frames(root)
    gate = threading.Event()
    aura.gate = gate
    assert client.post("/api/restore", json={"frame_id": FRAME, "asset_ids": ["old"]}).status_code == 202

    # One job at a time: no removal, load, rebuild or second restore meanwhile.
    assert client.post("/api/restore", json={"frame_id": FRAME, "asset_ids": ["new"]}).status_code == 409
    assert client.post("/api/sync", json={"frame_ids": [FRAME]}).status_code == 409
    assert client.post("/api/people/build").status_code == 409
    token = preview_removal(client, "frame", ["mid"]).json()["token"]
    assert client.post("/api/removal", json={"token": token}).status_code == 409
    gate.set()
    assert finish_restore(client, review)["state"] == "done"

    for body in ({}, {"frame_id": "../x", "asset_ids": ["old"]}, {"frame_id": FRAME, "asset_ids": "old"},
                 {"frame_id": FRAME, "asset_ids": ["never-removed"]}):
        assert client.post("/api/restore", json=body).status_code == 400, body


# ---------------------------------------------- WP2b: plain words when Aura says no

def unreachable():
    import httpx

    try:
        raise httpx.ConnectTimeout("")
    except httpx.ConnectTimeout as cause:
        try:
            raise AuraAPIError("GET /frames.json failed: ConnectTimeout") from cause
        except AuraAPIError as exc:
            return exc


def test_sign_in_when_aura_cannot_be_reached(tmp_path, monkeypatch):
    client, review, made = account_app(tmp_path, monkeypatch)

    async def login(self=None):
        raise unreachable()

    monkeypatch.setattr(FakeAccount, "login", login)
    response = client.post("/api/account/sign-in", json={"email": "a@b.c", "password": "right"})

    assert response.status_code == 502
    body = response.json()
    assert body["error"] == server.UNREACHABLE and "ConnectTimeout" in body["detail"]


def test_a_wrong_password_keeps_the_detail_aside(tmp_path, monkeypatch):
    client, review, made = account_app(tmp_path, monkeypatch)
    body = client.post("/api/account/sign-in", json={"email": "a@b.c", "password": "no"}).json()
    assert body["error"] == "Aura did not accept that email and password."
    assert "475" in body["detail"]


@pytest.mark.parametrize("error, words", [
    (AuraAuthError("still unauthorized"), server.SIGNED_OUT),
    (AuraAPIError("frames.json had no 'frames' list (keys: [])."), server.UNEXPECTED),
    (AuraAPIError("GET /frames.json failed with HTTP 403: Forbidden", 403), server.REFUSED),
    (AuraAPIError("GET /frames.json failed with HTTP 502: Bad gateway", 502), server.AURA_TROUBLE),
])
def test_listing_frames_and_loading_say_what_went_wrong(tmp_path, monkeypatch, error, words):
    client, review, made = account_app(tmp_path, monkeypatch, error=error)

    listed = client.get("/api/account/frames")
    assert listed.json()["error"] == words and listed.json()["detail"] == str(error)

    client.post("/api/sync", json={"frame_ids": [FRAME]})
    review.sync.thread.join(20)
    status = client.get("/api/sync/status").json()
    assert status["state"] == "error" and status["message"] == words
    assert status["detail"] == str(error)


def test_a_failed_listing_is_unreachable_not_raw(tmp_path, monkeypatch):
    client, review, made = account_app(tmp_path, monkeypatch, error=unreachable())
    body = client.get("/api/account/frames").json()
    assert body["error"] == server.UNREACHABLE and "ConnectTimeout" in body["detail"]


def test_a_refused_removal_gives_a_plain_reason(root):
    client, review, aura = removal_app(root, error=AuraAPIError("POST /x failed with HTTP 403: no", 403))
    confirm_removal(client, "frame", ["new"])
    failure = finish_removal(client, review)["result"]["failed"][0]
    assert failure["reason"] == server.REFUSED and "HTTP 403" in failure["error"]


def test_a_lost_sign_in_during_removal_gives_a_plain_reason(root):
    client, review, aura = removal_app(root, error=AuraAuthError("still unauthorized"))
    confirm_removal(client, "frame", ["new", "mid"])
    failed = finish_removal(client, review)["result"]["failed"]
    assert all(server.SIGNED_OUT in f["reason"] for f in failed)


def test_a_lost_sign_in_in_an_all_frames_removal_carries_the_plain_reason(root):
    two_frames(root)
    add_frame(root, PORCH, "p1")
    client, review, aura = removal_app(root, error=AuraAuthError("still unauthorized"))
    token = preview_removal(client, "all_frames", ["new", "p1"], frame=ALL).json()["token"]
    client.post("/api/removal", json={"token": token})
    failed = finish_removal(client, review)["result"]["failed"]
    assert failed and all(server.SIGNED_OUT in f["reason"] for f in failed)


# ------------------------------------------- WP2b: people appear while loading

def test_the_page_says_why_removing_waits_while_loading(client):
    page = client.get("/").text
    assert 'id="loadstrip"' in page
    assert "Removing is turned on when loading has finished" in page


# --------------------------------------------------------- WP2b: update notice

def test_version_when_github_cannot_be_reached(root, monkeypatch):
    import httpx

    monkeypatch.delenv("AURA_NO_UPDATE_CHECK")
    asked = []

    def offline():
        asked.append(1)
        raise httpx.ConnectError("no route to host")

    review = Review(root, FaceMatcher(app=FakeApp()), update_check=offline)
    client = TestClient(review.app(), base_url="http://127.0.0.1")

    first = client.get("/api/version")
    assert first.status_code == 200
    assert first.json() == {"version": APP_VERSION, "latest": None, "newer": False}
    review._update_thread.join(5)
    assert client.get("/api/version").json() == {"version": APP_VERSION, "latest": None, "newer": False}
    # The failure is remembered for a day: no second try on the next page load.
    assert asked == [1]
    assert json.loads((root / "update-check.json").read_text())["latest"] is None


def test_version_reports_a_newer_release(root, monkeypatch):
    monkeypatch.delenv("AURA_NO_UPDATE_CHECK")
    review = Review(root, FaceMatcher(app=FakeApp()), update_check=lambda: "v99.0.0")
    client = TestClient(review.app(), base_url="http://127.0.0.1")
    client.get("/api/version")
    review._update_thread.join(5)
    assert client.get("/api/version").json() == {"version": APP_VERSION, "latest": "v99.0.0", "newer": True}


def test_the_update_check_can_be_turned_off(root):
    asked = []
    review = Review(root, FaceMatcher(app=FakeApp()), update_check=lambda: asked.append(1) or "v9.0.0")
    client = TestClient(review.app(), base_url="http://127.0.0.1")
    assert client.get("/api/version").json() == {"version": APP_VERSION, "latest": None, "newer": False}
    assert asked == [] and review._update_thread is None


def test_the_release_lookup_sends_nothing_personal():
    import httpx

    from aura_mcp import review as review_module

    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"tag_name": "v0.3.1", "author": {"login": "x"}})

    tag = review_module.latest_release_tag(transport=httpx.MockTransport(handler))

    assert tag == "v0.3.1"
    request = seen[0]
    assert str(request.url) == "https://api.github.com/repos/walensis-labs/frame-tidy/releases/latest"
    assert request.method == "GET" and request.content == b""
    assert not {"authorization", "cookie"} & {k.lower() for k in request.headers}
    assert request.headers["user-agent"] == "frame-tidy-update-check"


def test_a_private_repository_answers_404_and_that_is_quiet():
    import httpx

    from aura_mcp import review as review_module

    transport = httpx.MockTransport(lambda request: httpx.Response(404, json={"message": "Not Found"}))
    with pytest.raises(httpx.HTTPError):
        review_module.latest_release_tag(transport=transport)


# ---------------------------------------- review 2: the safety copy is never lost

def ledger_with(root, frame=FRAME, **entries):
    kept = root / "removed" / frame
    for name in entries:
        write_photo(kept / f"{name}.jpg")
    for name, entry in entries.items():
        entry.setdefault("local_path", str(kept / f"{name}.jpg"))
    (kept / "removed.json").write_text(json.dumps(entries))
    return kept


def test_the_removed_list_shows_a_cut_short_removal_and_a_photo_found_missing(root):
    ledger_with(root,
                cut={"state": "removing", "started_at": "2026-11-19T10:00:00+00:00", "taken_at": None},
                gone={"state": "missing", "found_missing_at": "2026-11-18T10:00:00+00:00", "taken_at": None},
                done={"state": "removed", "removed_at": "2026-11-17T10:00:00+00:00", "taken_at": None},
                older={"removed_at": "2026-11-16T10:00:00+00:00", "taken_at": None})  # before states existed
    client, review, aura = removal_app(root)

    items = client.get(f"/api/frames/{FRAME}/removed").json()["removed"]

    assert [(i["id"], i["state"]) for i in items] == [
        ("cut", "removing"), ("gone", "missing"), ("done", "removed"), ("older", "removed")]
    assert [i["when"] for i in items] == [
        "2026-11-19T10:00:00+00:00", "2026-11-18T10:00:00+00:00",
        "2026-11-17T10:00:00+00:00", "2026-11-16T10:00:00+00:00"]
    for name in ("cut", "gone"):
        assert client.get(f"/removed-thumb/{FRAME}/{name}.jpg").status_code == 200


def test_the_page_says_plainly_what_each_kind_of_kept_photo_is(client):
    page = client.get("/").text
    assert "Not removed by this app" in page
    assert "closed while removing" in page


def test_putting_back_a_cut_short_removal_from_the_page(root):
    ledger_with(root, new={"state": "removing", "started_at": "2026-11-19T10:00:00+00:00",
                           "file_name": "new.heic"})
    client, review, aura = removal_app(root)

    client.post("/api/restore", json={"frame_id": FRAME, "asset_ids": ["new"]})
    result = finish_restore(client, review)["result"]

    assert result["restored_count"] == 1 and aura.added == [(FRAME, "new")]


def test_a_refused_put_back_of_a_cut_short_removal_says_so_plainly(root):
    ledger_with(root, new={"state": "removing", "started_at": "2026-11-19T10:00:00+00:00"},
                gone={"state": "missing", "found_missing_at": "2026-11-19T10:00:00+00:00"})
    client, review, aura = removal_app(
        root, restore_error=AuraRefusedError("Aura could not add asset (number_failed=1)."))

    client.post("/api/restore", json={"frame_id": FRAME, "asset_ids": ["new", "gone"]})
    failed = {f["asset_id"]: f["message"] for f in finish_restore(client, review)["result"]["failed"]}

    assert "closed while removing" in failed["new"] and "Aura app" in failed["new"]
    assert "not removed by this app" in failed["gone"].lower()


def test_removing_and_putting_back_wait_while_photos_load(root):
    import asyncio

    client, review, aura = removal_app(root)
    ledger_with(root, old={"state": "removed", "removed_at": "2026-11-19T10:00:00+00:00"})
    gate = threading.Event()

    async def loading(progress):
        await asyncio.to_thread(gate.wait, 5)
        return {}

    assert review.sync.start(loading)
    token = preview_removal(client, "frame", ["mid"]).json()["token"]
    removal = client.post("/api/removal", json={"token": token})
    restore = client.post("/api/restore", json={"frame_id": FRAME, "asset_ids": ["old"]})
    gate.set()
    review.sync.thread.join(10)

    assert removal.status_code == 409 and "loaded" in removal.json()["error"]
    assert restore.status_code == 409 and "loaded" in restore.json()["error"]
    assert aura.removed == [] and aura.added == []
    assert review.removal.thread is None and review.restore.thread is None


def test_loading_waits_while_a_removal_runs(root):
    gate = threading.Event()
    client, review, aura = removal_app(root, gate=gate)
    review.matcher.model_present = lambda: True
    assert confirm_removal(client, "frame", ["new"]).status_code == 202

    response = client.post("/api/sync", json={"frame_ids": [FRAME]})
    gate.set()
    finish_removal(client, review)

    assert response.status_code == 409 and "removal" in response.json()["error"]
    assert review.sync.thread is None


def _busy_review(finished, job="removal"):
    """A Review whose removal (or restore) is part-way through a photo."""
    import time

    from aura_mcp import review as review_mod

    class Busy(review_mod.Review):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            running = getattr(self, job)

            def photo():
                while not running.stop.is_set():  # until asked to stop...
                    time.sleep(0.01)
                time.sleep(0.2)  # ...then the end of the photo it is on
                finished.append(job)

            running.thread = threading.Thread(target=photo, daemon=True)
            running.thread.start()

    return Busy


@pytest.mark.parametrize("job", ["removal", "restore"])
def test_aura_review_waits_for_the_photo_being_removed_before_it_quits(tmp_path, monkeypatch, job):
    import sys

    import uvicorn

    from aura_mcp import review as review_mod

    finished = []
    monkeypatch.setattr(review_mod, "Review", _busy_review(finished, job))
    monkeypatch.setattr(review_mod.store, "ROOT", tmp_path)
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["aura-review", "--port", "8799"])
    review_mod.main()
    assert finished == [job]


@pytest.mark.parametrize("job", ["removal", "restore"])
def test_the_desktop_app_waits_for_the_photo_being_removed_before_it_quits(tmp_path, monkeypatch, job):
    import uvicorn

    from aura_mcp import app as desktop
    from aura_mcp import review as review_mod

    class Server:
        def __init__(self, config):
            self.started, self.should_exit = True, False

        def run(self):
            pass

    finished = []
    monkeypatch.setenv("AURA_MCP_HOME", str(tmp_path))
    monkeypatch.setattr(review_mod, "Review", _busy_review(finished, job))
    monkeypatch.setattr(uvicorn, "Server", Server)
    monkeypatch.setattr(uvicorn, "Config", lambda *a, **k: None)
    desktop.main(["--no-window"])
    assert finished == [job]


def test_a_removal_job_stops_between_photos_when_asked(root):
    gate, inside = threading.Event(), threading.Event()
    client, review, aura = removal_app(root, gate=gate)
    real = aura.remove_asset_from_frame

    async def remove(frame_id, asset_id):
        inside.set()
        await real(frame_id, asset_id)

    aura.remove_asset_from_frame = remove
    confirm_removal(client, "frame", ["new", "mid", "old"])
    assert inside.wait(5)  # the first photo is with Aura...
    review.removal.stop.set()  # ...when the app starts closing
    gate.set()
    result = finish_removal(client, review)["result"]
    # The photo already under way is finished; the rest are not started.
    assert len(aura.removed) == 1
    assert result["failed_count"] == 2
    assert all("closed" in f["reason"] for f in result["failed"])


def test_the_app_is_called_frame_tidy_for_aura_frames(client):
    from aura_mcp.app import APP_NAME

    page = client.get("/").text
    assert APP_NAME == "Frame Tidy"
    assert f"<title>{APP_NAME}</title>" in page
    assert f"<h1>{APP_NAME}" in page and "for Aura frames" in page
    assert "Aura review" not in page


# ------------------------------- review 2: the search model waits for a yes

@pytest.fixture
def no_search_model(root, tmp_path, monkeypatch):
    from aura_mcp import image_search
    from aura_mcp.image_search import ImageSearcher

    calls = {"load": [], "fetch": []}

    def load(where, local_only):
        calls["load"].append(str(where))
        return object(), object()

    def fetch(where, arrived=None):
        calls["fetch"].append(str(where))
        if arrived:
            arrived(1234)
        for model, files in image_search.MODEL_FILES.items():
            snap = where / ("models--" + model.replace("/", "--")) / "snapshots" / "abc"
            snap.mkdir(parents=True, exist_ok=True)
            for name in files:
                (snap / name).write_bytes(b"x")

    monkeypatch.setattr(image_search, "_load_models", load)
    monkeypatch.setattr(image_search, "_fetch_models", fetch)
    searcher = ImageSearcher(model_root=tmp_path / "search-model", download=False)
    aura = FakeAura()
    review = Review(root, FaceMatcher(app=FakeApp()), client_factory=lambda: aura, searcher=searcher)
    return TestClient(review.app(), base_url="http://127.0.0.1"), review, calls


def test_the_page_never_fetches_the_search_model_on_its_own(root):
    assert Review(root).searcher.download is False


def test_text_search_waits_for_the_search_model(no_search_model, tmp_path):
    client, review, calls = no_search_model

    setup = client.get("/api/setup").json()["search_model"]
    assert setup["present"] is False and setup["agreed"] is False
    assert setup["size_bytes"] > 500_000_000 and setup["download"]["state"] == "idle"
    response = client.post(f"/api/frames/{FRAME}/search-text", json={"query": "a dog"})
    assert response.status_code == 409 and response.json()["needs"] == "search_model"
    assert calls == {"load": [], "fetch": []}
    assert not (tmp_path / "search-model").exists()


@pytest.mark.parametrize("body", [{}, {"agree": False}, {"agree": "yes"}, []])
def test_the_search_model_download_needs_an_explicit_yes(no_search_model, body):
    client, review, calls = no_search_model
    assert client.post("/api/setup/search-model", json=body).status_code == 400
    assert calls["fetch"] == [] and review.search_model_job.thread is None


def test_after_a_yes_the_search_model_is_fetched_then_used(no_search_model, root, tmp_path, monkeypatch):
    client, review, calls = no_search_model

    assert client.post("/api/setup/search-model", json={"agree": True}).status_code == 202
    review.search_model_job.thread.join(10)

    assert client.get("/api/setup/search-model").json()["state"] == "done"
    assert calls["fetch"] == [str(tmp_path / "search-model")]
    setup = client.get("/api/setup").json()["search_model"]
    assert setup["present"] is True and setup["agreed"] is True
    assert json.loads((root / "settings.json").read_text())["search_model_agreed_at"]

    async def fake(searcher, root_, frame_id, query, **kwargs):
        assert searcher is review.searcher
        return {"query": query, "match_count": 0, "similarity_threshold": 0.22, "matches": []}

    monkeypatch.setattr(server, "find_photos_by_description_impl", fake)
    assert client.post(f"/api/frames/{FRAME}/search-text", json={"query": "a dog"}).status_code == 200


def test_the_search_model_download_is_one_job_among_the_others(no_search_model):
    client, review, calls = no_search_model
    gate = threading.Event()
    import asyncio

    async def busy(progress):
        await asyncio.to_thread(gate.wait, 5)
        return {}

    review.sync.start(busy)
    refused = client.post("/api/setup/search-model", json={"agree": True})
    gate.set()
    review.sync.thread.join(10)
    assert refused.status_code == 409 and calls["fetch"] == []

    gate2 = threading.Event()
    real = review.searcher.download_model
    review.searcher.download_model = lambda arrived=None: (gate2.wait(5), real(arrived))
    assert client.post("/api/setup/search-model", json={"agree": True}).status_code == 202
    assert client.post("/api/sync", json={"frame_ids": [FRAME]}).status_code == 409
    gate2.set()
    review.search_model_job.thread.join(10)


def test_the_page_asks_before_fetching_the_search_model(client):
    page = client.get("/").text
    step = page[page.index('<template id="tpl-search-model">'):]
    step = step[:step.index("</template>")]
    assert "text-and-image" in step and "610 MB" in step and "Hugging Face" in step
    assert "MIT" in step and "only on this computer" in step
    assert "/api/setup/search-model" in page

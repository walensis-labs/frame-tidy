"""The real client and tool logic, run against the in-process mock Aura API."""

import httpx
import numpy as np
import pytest

from aura_mcp import server
from aura_mcp.aura_client import AuraAPIError, AuraAuthError, AuraClient
from aura_mcp.face_matcher import FaceMatcher
from aura_mcp.mock_aura import MockAura


def _photo(name):
    """A small real JPEG, different for each name: removal checks that the
    original it fetches is a whole photo."""
    import zlib

    import cv2

    rng = np.random.default_rng(zlib.crc32(name.encode()))
    ok, data = cv2.imencode(".jpg", rng.integers(0, 255, (24, 32, 3), dtype=np.uint8))
    assert ok
    return data.tobytes()


@pytest.fixture
def mock(tmp_path):
    photos = tmp_path / "photos"
    for frame, names in {
        "Living Room": ["p1.jpg", "p2.jpg", "p3.jpg", "p4.jpg", "shared.jpg"],
        "Kitchen": ["shared.jpg", "k1.jpg"],
    }.items():
        (photos / frame).mkdir(parents=True)
        for name in names:
            # "shared.jpg" has identical bytes in both folders
            (photos / frame / name).write_bytes(_photo(name))
    (photos / "Living Room" / "notes.txt").write_text("not a photo")
    return MockAura(photos, page_size=2)


def client_for(mock, password="pw"):
    return AuraClient(
        "me@example.com",
        password,
        api_base_url="http://mock/v5",
        image_base_url="http://mock/img",
        transport=httpx.ASGITransport(app=mock.app()),
        download_delay=0,
        delete_delay=0,
    )


def frame_id(mock, name):
    return next(f["id"] for f in mock.state()["frames"] if f["name"] == name)


async def test_frames_and_paginated_assets(mock):
    async with client_for(mock) as client:
        frames = await client.list_frames()
        assets = await client.list_assets(frame_id(mock, "Living Room"))

    assert {f["name"]: f["asset_count"] for f in frames} == {"Kitchen": 2, "Living Room": 5}
    # five photos at two per page = three pages, all collected, text file ignored
    assert sorted(a["file_name"] for a in assets) == [
        "p1.jpg", "p2.jpg", "p3.jpg", "p4.jpg", "shared.jpg",
    ]


async def test_same_photo_on_two_frames_is_one_asset(mock):
    async with client_for(mock) as client:
        living = await client.list_assets(frame_id(mock, "Living Room"))
        kitchen = await client.list_assets(frame_id(mock, "Kitchen"))
    shared = {a["id"] for a in living} & {a["id"] for a in kitchen}
    assert len(shared) == 1 and mock.state()["total_assets"] == 6


async def test_rejected_login_matches_real_api(mock):
    async with client_for(mock, password="wrong") as client:
        with pytest.raises(AuraAuthError, match="HTTP 475"):
            await client.list_frames()


async def test_client_logs_in_again_after_token_expiry(mock):
    async with client_for(mock) as client:
        await client.list_frames()
        mock.tokens.clear()
        assert len(await client.list_frames()) == 2


async def test_frame_scope_keeps_photo_on_other_frames(mock, tmp_path):
    living, kitchen = frame_id(mock, "Living Room"), frame_id(mock, "Kitchen")
    async with client_for(mock) as client:
        shared = next(a for a in await client.list_assets(living) if a["file_name"] == "shared.jpg")
        await client.remove_asset_from_frame(living, shared["id"])

        assert shared["id"] not in {a["id"] for a in await client.list_assets(living)}
        assert shared["id"] in {a["id"] for a in await client.list_assets(kitchen)}
        with pytest.raises(AuraAPIError, match="number_failed=1"):
            await client.remove_asset_from_frame(living, shared["id"])


async def test_account_scope_removes_photo_from_every_frame(mock):
    living, kitchen = frame_id(mock, "Living Room"), frame_id(mock, "Kitchen")
    async with client_for(mock) as client:
        shared = next(a for a in await client.list_assets(living) if a["file_name"] == "shared.jpg")
        await client.delete_asset(shared["id"])

        assert shared["id"] not in {a["id"] for a in await client.list_assets(living)}
        assert shared["id"] not in {a["id"] for a in await client.list_assets(kitchen)}
        with pytest.raises(AuraAPIError, match="HTTP 404"):
            await client.delete_asset(shared["id"])


async def test_reset_restores_removed_photos(mock):
    living = frame_id(mock, "Living Room")
    for asset_id in list(mock.frames[living]):
        mock.delete_from_account(asset_id)
    assert mock.state()["total_assets"] == 1
    mock.load()
    assert mock.state()["total_assets"] == 6
    # nothing on disk was touched
    assert len(list((mock.photos_dir / "Living Room").glob("*.jpg"))) == 5


@pytest.fixture
def real_photos(tmp_path):
    """Decodable photos, so the mock's resizing can be checked."""
    import cv2

    photos = tmp_path / "real"
    (photos / "Den").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for name, (h, w) in {"wide.jpg": (300, 400), "tiny.jpg": (60, 80)}.items():
        cv2.imwrite(str(photos / "Den" / name), rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
    return MockAura(photos)


async def test_mock_serves_a_really_resized_copy_at_the_resized_address(real_photos):
    import cv2

    async with client_for(real_photos) as client:
        assets = {a["file_name"]: a for a in await client.list_assets(frame_id(real_photos, "Den"))}
        response = await client._images.get(client.image_url("mock-user-1", "wide.jpg", max_side=200))
        assert response.status_code == 200
        image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        assert image.shape[:2] == (150, 200)  # fits in 200x200, aspect kept

        # Small enough already: not enlarged.
        response = await client._images.get(client.image_url("mock-user-1", "tiny.jpg", max_side=200))
        image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        assert image.shape[:2] == (60, 80)

        # The plain address is still the original file, byte for byte.
        original = await client._images.get(client.image_url("mock-user-1", "wide.jpg"))
    assert original.content == (real_photos.photos_dir / "Den" / "wide.jpg").read_bytes()
    assert set(assets) == {"wide.jpg", "tiny.jpg"}


async def test_load_keeps_medium_copies_and_a_removal_keeps_the_original(real_photos, tmp_path):
    root = tmp_path / "home"
    den = frame_id(real_photos, "Den")
    async with client_for(real_photos) as client:
        await server.sync_frame_photos_impl(client, root, den)
        index = server.load_index(root, den)
        wide = next(a for a, e in index.items() if e["file_name"] == "wide.jpg")
        assert index[wide]["size"] == server.SYNC_MAX_SIDE

        report = await server.delete_assets_impl(client, root, den, [wide], confirm=True)

    assert report["deleted_count"] == 1 and report["failed"] == []
    kept = root / "removed" / den / f"{wide}.jpg"
    assert kept.read_bytes() == (real_photos.photos_dir / "Den" / "wide.jpg").read_bytes()
    assert not (root / "cache" / den / f"{wide}.jpg").exists()
    assert wide not in real_photos.frames[den]


class StubMatcher(FaceMatcher):
    """Treats photos that look like p1.jpg or shared.jpg as the target."""

    def __init__(self):
        self.target = np.eye(512, dtype=np.float32)[0]
        self.other = np.eye(512, dtype=np.float32)[1]

    def reference_embedding(self, image_path):
        return self.target, 1

    def face_embeddings(self, image_path):
        import cv2

        seen = cv2.imread(str(image_path)).astype(np.int16)
        is_target = any(
            np.abs(seen - cv2.imdecode(np.frombuffer(_photo(name), np.uint8), cv2.IMREAD_COLOR)).mean() < 8
            for name in ("p1.jpg", "shared.jpg")
        )
        return np.stack([self.target if is_target else self.other])


async def test_full_workflow_against_mock(mock, tmp_path):
    root = tmp_path / "home"
    living, kitchen = frame_id(mock, "Living Room"), frame_id(mock, "Kitchen")

    async with client_for(mock) as client:
        synced = await server.sync_frame_photos_impl(client, root, living)
        assert synced["total_synced"] == 5 and synced["failed"] == []
        assert (root / "cache" / living / "index.json").is_file()

        found = await server.find_person_in_frame_impl(StubMatcher(), root, living, "ref.jpg")
        assert sorted(m["file_name"] for m in found["matches"]) == ["p1.jpg", "shared.jpg"]
        ids = [m["asset_id"] for m in found["matches"]]

        preview = await server.delete_assets_impl(client, root, living, ids)
        assert preview["status"] == "confirmation_required"
        assert mock.state()["total_assets"] == 6  # preview changed nothing

        report = await server.delete_assets_impl(client, root, living, ids, confirm=True)
        assert report["deleted_count"] == 2 and report["failed"] == []

    counts = {f["name"]: f["num_assets"] for f in mock.state()["frames"]}
    assert counts == {"Living Room": 3, "Kitchen": 2}  # shared photo survives in Kitchen
    assert list((root / "flagged" / living).glob("*.jpg")) == []
    assert len(list((root / "cache" / living).glob("*.jpg"))) == 3


async def test_remove_then_restore_round_trip(mock, tmp_path):
    """Frame-scope removal keeps a local copy, and restore puts the photo back."""
    root = tmp_path / "home"
    living, kitchen = frame_id(mock, "Living Room"), frame_id(mock, "Kitchen")

    async with client_for(mock) as client:
        await server.sync_frame_photos_impl(client, root, living)
        index = server.load_index(root, living)
        shared = next(a for a, e in index.items() if e["file_name"] == "shared.jpg")
        lonely = next(a for a, e in index.items() if e["file_name"] == "p1.jpg")
        original = _photo("shared.jpg")  # what Aura holds; the cache has a medium copy

        report = await server.delete_assets_impl(client, root, living, [shared, lonely], confirm=True)
        assert report["deleted_count"] == 2
        kept = root / "removed" / living
        assert (kept / f"{shared}.jpg").read_bytes() == original  # the full original is kept
        assert report["deleted"][0]["local_copy"] == str(kept / f"{shared}.jpg")
        log = server._read_json(kept / "removed.json", {})
        assert log[shared]["file_name"] == "shared.jpg" and log[shared]["scope"] == "frame"
        assert shared not in mock.frames[living] and shared in mock.frames[kitchen]

        result = await server.restore_assets_impl(client, root, living, [shared, lonely, "never-removed"])

    # the photo that was also on the Kitchen frame comes back...
    assert [r["asset_id"] for r in result["restored"]] == [shared]
    assert shared in mock.frames[living]
    assert (root / "cache" / living / f"{shared}.jpg").read_bytes() == original
    assert shared in server.load_index(root, living)
    # ...the one that was on no other frame is gone from Aura, but its copy is kept
    assert result["failed"][0]["asset_id"] == lonely
    assert "number_failed=1" in result["failed"][0]["error"]
    assert result["failed"][0]["local_copy"] == str(kept / f"{lonely}.jpg")
    assert (kept / f"{lonely}.jpg").is_file()
    assert result["unknown_asset_ids"] == ["never-removed"]
    assert set(server._read_json(kept / "removed.json", {})) == {lonely}

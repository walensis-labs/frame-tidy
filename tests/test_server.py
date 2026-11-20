import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from aura_mcp import server
from aura_mcp.aura_client import AuraAPIError, AuraAuthError, AuraRefusedError
from aura_mcp.face_matcher import FaceMatcher, ImageReadError, NoFaceFoundError

FRAME = "frame-1"


def unit(*values):
    v = np.zeros(512, dtype=np.float32)
    v[: len(values)] = values
    return v / np.linalg.norm(v)


TARGET = unit(1, 0)
OTHER = unit(0, 1)


def _jpeg(width, height):
    noise = np.random.default_rng(width * 7 + height).integers(0, 255, (height, width, 3), dtype=np.uint8)
    ok, data = cv2.imencode(".jpg", noise)
    assert ok
    return data.tobytes()


MEDIUM_SIDE, ORIGINAL_SIDE = 40, 80
_MEDIUM_JPEG, _ORIGINAL_JPEG = _jpeg(30, MEDIUM_SIDE), _jpeg(60, ORIGINAL_SIDE)


def img(name):
    """What the fake Aura sends as a medium copy: a real JPEG, named after the photo."""
    return _MEDIUM_JPEG + b"img:" + name.encode()


def original(name):
    """What the fake Aura sends as the full original: a larger real JPEG."""
    return _ORIGINAL_JPEG + b"original:" + name.encode()


class FakeClient:
    """A medium copy holds img(<name>), the original original(<name>)."""

    def __init__(self, assets=(), fail_downloads=(), delete_errors=None, fail_originals=()):
        self.assets = list(assets)
        self.fail_downloads = set(fail_downloads)
        self.fail_originals = set(fail_originals)
        self.delete_errors = delete_errors or {}
        self.downloads = []
        self.sizes = []
        self.calls = []  # downloads and removals, in the order they happened
        self.deletes = []
        self.frame_removals = []
        self.restore_errors = {}
        self.restored = []

    async def list_assets(self, frame_id):
        return self.assets

    async def download_photo(self, user_id, file_name, dest, max_side=None):
        self.downloads.append(file_name)
        self.sizes.append(max_side)
        self.calls.append(("download", file_name, max_side))
        if file_name in self.fail_downloads or (max_side is None and file_name in self.fail_originals):
            raise AuraAPIError("Download failed with HTTP 500.", 500)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(img(file_name) if max_side else original(file_name))
        return dest

    async def delete_asset(self, asset_id):
        self.deletes.append(asset_id)
        self.calls.append(("delete", asset_id))
        if asset_id in self.delete_errors:
            raise self.delete_errors[asset_id]

    async def remove_asset_from_frame(self, frame_id, asset_id):
        self.frame_removals.append((frame_id, asset_id))
        self.calls.append(("remove", frame_id, asset_id))
        for key in ((frame_id, asset_id), asset_id):
            if key in self.delete_errors:
                raise self.delete_errors[key]

    async def add_asset_to_frame(self, frame_id, asset_id):
        if asset_id in self.restore_errors:
            raise self.restore_errors[asset_id]
        self.restored.append((frame_id, asset_id))


class FakeMatcher(FaceMatcher):
    """Returns canned embeddings keyed by file stem; no model involved."""

    def __init__(self, faces, reference=TARGET, reference_faces=1):
        self.faces = faces
        self.reference = reference
        self.reference_faces = reference_faces
        self.scanned = []

    def reference_embedding(self, image_path):
        if self.reference is None:
            raise NoFaceFoundError(f"No face detected in reference image {image_path}.")
        return self.reference, self.reference_faces

    def face_embeddings(self, image_path):
        stem = Path(image_path).stem
        self.scanned.append(stem)
        if stem not in self.faces:
            raise ImageReadError(f"Could not decode image: {image_path}")
        rows = self.faces[stem]
        return np.stack(rows) if rows else np.zeros((0, 512), dtype=np.float32)


def asset(asset_id, **extra):
    return {
        "id": asset_id,
        "user_id": "u1",
        "file_name": f"{asset_id}.jpeg",
        "taken_at": "2024-01-01T00:00:00Z",
        **extra,
    }


async def synced(tmp_path, *asset_ids, frame=FRAME):
    client = FakeClient([asset(a) for a in asset_ids])
    await server.sync_frame_photos_impl(client, tmp_path, frame)
    return client


# ------------------------------------------------------------------- sync


async def test_sync_downloads_and_writes_index(tmp_path):
    client = FakeClient([asset("a1"), asset("a2")])
    result = await server.sync_frame_photos_impl(client, tmp_path, FRAME)

    folder = tmp_path / "cache" / FRAME
    assert result["total_synced"] == 2 and result["downloaded"] == 2
    assert (folder / "a1.jpg").read_bytes() == img("a1.jpeg")
    index = json.loads((folder / "index.json").read_text())
    assert index["a1"] == {
        "file_name": "a1.jpeg",
        "user_id": "u1",
        "taken_at": "2024-01-01T00:00:00Z",
        "local_path": str(folder / "a1.jpg"),
        "size": server.SYNC_MAX_SIDE,  # a medium copy, not the original
    }


async def test_sync_skips_cached_photos_on_rerun(tmp_path):
    await synced(tmp_path, "a1")
    client = FakeClient([asset("a1"), asset("a2")])
    result = await server.sync_frame_photos_impl(client, tmp_path, FRAME)

    assert client.downloads == ["a2.jpeg"]
    assert result["already_cached"] == 1 and result["downloaded"] == 1


async def test_sync_reports_failed_downloads_and_continues(tmp_path):
    client = FakeClient([asset("a1"), asset("a2")], fail_downloads={"a1.jpeg"})
    result = await server.sync_frame_photos_impl(client, tmp_path, FRAME)

    assert result["total_synced"] == 1
    assert [f["asset_id"] for f in result["failed"]] == ["a1"]
    index = server.load_index(tmp_path, FRAME)
    assert index["a1"]["local_path"] is None and index["a2"]["local_path"]


async def test_sync_prunes_photos_no_longer_on_frame(tmp_path):
    await synced(tmp_path, "a1", "gone")
    result = await server.sync_frame_photos_impl(FakeClient([asset("a1")]), tmp_path, FRAME)

    assert result["pruned_stale"] == 1
    assert not (tmp_path / "cache" / FRAME / "gone.jpg").exists()
    assert set(server.load_index(tmp_path, FRAME)) == {"a1"}


async def test_sync_rejects_path_traversal_ids(tmp_path):
    with pytest.raises(ValueError, match="Invalid frame_id"):
        await server.sync_frame_photos_impl(FakeClient(), tmp_path, "../evil")

    result = await server.sync_frame_photos_impl(
        FakeClient([asset("../../escape")]), tmp_path, FRAME
    )
    assert result["total_synced"] == 0 and len(result["failed"]) == 1
    assert not (tmp_path / "escape.jpg").exists()


# ------------------------------------------------------------------- find


async def test_find_requires_sync_first(tmp_path):
    with pytest.raises(ValueError, match="sync_frame_photos"):
        await server.find_person_in_frame_impl(FakeMatcher({}), tmp_path, FRAME, "ref.jpg")


async def test_find_searches_partial_download_while_sync_is_running(tmp_path):
    await synced(tmp_path, "a1", "a2")
    (tmp_path / "cache" / FRAME / "index.json").unlink()  # as during a running sync
    matcher = FakeMatcher({"a1": [TARGET], "a2": [OTHER]})

    result = await server.find_person_in_frame_impl(matcher, tmp_path, FRAME, "ref.jpg")

    assert [m["asset_id"] for m in result["matches"]] == ["a1"]
    assert "has not finished" in result["note"]


async def test_find_flags_photos_where_any_face_matches(tmp_path):
    await synced(tmp_path, "solo", "group", "stranger", "landscape")
    matcher = FakeMatcher(
        {
            "solo": [TARGET],
            "group": [OTHER, unit(0.9, 0.436)],  # second face is the target
            "stranger": [OTHER],
            "landscape": [],
        }
    )
    result = await server.find_person_in_frame_impl(matcher, tmp_path, FRAME, "ref.jpg", 0.45)

    assert [m["asset_id"] for m in result["matches"]] == ["solo", "group"]
    assert result["matches"][0]["similarity"] == pytest.approx(1.0)
    assert result["matches"][1]["similarity"] == pytest.approx(0.9, abs=0.01)
    assert result["matches"][0]["file_name"] == "solo.jpeg"
    assert result["photos_with_faces"] == 3

    flagged = tmp_path / "flagged" / FRAME
    assert sorted(p.name for p in flagged.glob("*.jpg")) == ["group.jpg", "solo.jpg"]
    assert result["matches"][0]["flagged_path"] == str(flagged / "solo.jpg")
    # originals stay in the cache
    assert (tmp_path / "cache" / FRAME / "solo.jpg").exists()


async def test_find_threshold_is_respected(tmp_path):
    await synced(tmp_path, "close")
    matcher = FakeMatcher({"close": [unit(0.5, 0.866)]})  # similarity 0.5

    hit = await server.find_person_in_frame_impl(matcher, tmp_path, FRAME, "r.jpg", 0.45)
    miss = await server.find_person_in_frame_impl(matcher, tmp_path, FRAME, "r.jpg", 0.6)

    assert hit["match_count"] == 1 and miss["match_count"] == 0
    # the earlier flagged copy is cleared when a later run no longer matches it
    assert list((tmp_path / "flagged" / FRAME).glob("*.jpg")) == []


async def test_find_reuses_cached_embeddings(tmp_path):
    await synced(tmp_path, "a1", "a2")
    matcher = FakeMatcher({"a1": [TARGET], "a2": []})
    await server.find_person_in_frame_impl(matcher, tmp_path, FRAME, "r.jpg")
    assert sorted(matcher.scanned) == ["a1", "a2"]

    again = FakeMatcher({})  # would fail to read anything if it rescanned
    result = await server.find_person_in_frame_impl(again, tmp_path, FRAME, "r.jpg")
    assert again.scanned == []
    assert [m["asset_id"] for m in result["matches"]] == ["a1"]


async def test_find_reports_unreadable_images(tmp_path):
    await synced(tmp_path, "ok", "broken")
    matcher = FakeMatcher({"ok": [TARGET]})
    result = await server.find_person_in_frame_impl(matcher, tmp_path, FRAME, "r.jpg")

    assert result["unreadable_asset_ids"] == ["broken"]
    assert result["photos_scanned"] == 1 and result["match_count"] == 1


async def test_find_no_face_in_reference_raises_clear_error(tmp_path):
    await synced(tmp_path, "a1")
    with pytest.raises(NoFaceFoundError, match="No face detected"):
        await server.find_person_in_frame_impl(
            FakeMatcher({}, reference=None), tmp_path, FRAME, "blank.jpg"
        )


async def test_find_warns_when_reference_has_several_faces(tmp_path):
    await synced(tmp_path, "a1")
    matcher = FakeMatcher({"a1": [TARGET]}, reference_faces=3)
    result = await server.find_person_in_frame_impl(matcher, tmp_path, FRAME, "r.jpg")
    assert "3 faces" in result["warning"]


# ----------------------------------------------------------------- delete


async def flagged_setup(tmp_path, *asset_ids):
    await synced(tmp_path, *asset_ids)
    matcher = FakeMatcher({a: [TARGET] for a in asset_ids})
    await server.find_person_in_frame_impl(matcher, tmp_path, FRAME, "r.jpg")


async def test_delete_without_confirm_touches_nothing(tmp_path):
    await flagged_setup(tmp_path, "a1", "a2")
    client = FakeClient()

    result = await server.delete_assets_impl(client, tmp_path, FRAME, ["a1", "a2"])

    assert client.deletes == []
    assert result["status"] == "confirmation_required"
    assert result["deleted_count"] == 0
    assert [d["asset_id"] for d in result["would_delete"]] == ["a1", "a2"]
    assert result["would_delete"][0]["file_name"] == "a1.jpeg"
    assert "explicit confirmation" in result["message"]
    assert (tmp_path / "cache" / FRAME / "a1.jpg").exists()
    assert (tmp_path / "flagged" / FRAME / "a1.jpg").exists()


@pytest.mark.parametrize("confirm", [False, None, 0, "yes", 1])
async def test_delete_only_runs_on_literal_true(tmp_path, confirm):
    await flagged_setup(tmp_path, "a1")
    client = FakeClient()
    result = await server.delete_assets_impl(client, tmp_path, FRAME, ["a1"], confirm=confirm)
    assert client.deletes == [] and result["status"] == "confirmation_required"


async def test_delete_confirmed_removes_remote_and_local_copies(tmp_path):
    await flagged_setup(tmp_path, "a1", "a2", "keep")
    client = FakeClient()

    result = await server.delete_assets_impl(client, tmp_path, FRAME, ["a1", "a2", "a1"], scope="account", confirm=True)

    assert client.deletes == ["a1", "a2"]  # de-duplicated
    assert result["status"] == "completed" and result["deleted_count"] == 2
    for gone in ("a1", "a2"):
        assert not (tmp_path / "cache" / FRAME / f"{gone}.jpg").exists()
        assert not (tmp_path / "flagged" / FRAME / f"{gone}.jpg").exists()
    assert (tmp_path / "cache" / FRAME / "keep.jpg").exists()
    assert set(server.load_index(tmp_path, FRAME)) == {"keep"}
    with np.load(tmp_path / "cache" / FRAME / "embeddings.npz") as data:
        assert data.files == ["keep"]


async def test_delete_never_sends_ids_that_were_not_synced(tmp_path):
    await flagged_setup(tmp_path, "a1")
    client = FakeClient()

    result = await server.delete_assets_impl(
        client, tmp_path, FRAME, ["a1", "from-another-frame"], scope="account", confirm=True)

    assert client.deletes == ["a1"]
    assert result["unknown_asset_ids"] == ["from-another-frame"]


async def test_delete_failure_keeps_local_files_and_continues(tmp_path):
    await flagged_setup(tmp_path, "a1", "a2")
    client = FakeClient(delete_errors={"a1": AuraAPIError("HTTP 403: not yours", 403)})

    result = await server.delete_assets_impl(client, tmp_path, FRAME, ["a1", "a2"], scope="account", confirm=True)

    assert result["deleted_count"] == 1 and result["failed_count"] == 1
    # The raw reply for the record, and a sentence a person can read.
    assert result["failed"][0] == {
        "asset_id": "a1", "error": "HTTP 403: not yours", "reason": "Aura refused the request."}
    assert (tmp_path / "cache" / FRAME / "a1.jpg").exists()
    assert set(server.load_index(tmp_path, FRAME)) == {"a1"}


async def test_delete_stops_when_session_is_lost(tmp_path):
    await flagged_setup(tmp_path, "a1", "a2", "a3")
    client = FakeClient(delete_errors={"a2": AuraAuthError("still unauthorized")})

    result = await server.delete_assets_impl(client, tmp_path, FRAME, ["a1", "a2", "a3"], scope="account", confirm=True)

    assert client.deletes == ["a1", "a2"]
    assert result["status"] == "aborted" and result["deleted_count"] == 1
    assert [f["asset_id"] for f in result["failed"]] == ["a2", "a3"]
    assert set(server.load_index(tmp_path, FRAME)) == {"a2", "a3"}


async def test_delete_defaults_to_frame_scope(tmp_path):
    await flagged_setup(tmp_path, "a1")
    client = FakeClient()

    preview = await server.delete_assets_impl(client, tmp_path, FRAME, ["a1"])
    assert preview["scope"] == "frame" and "this frame only" in preview["effect"]

    result = await server.delete_assets_impl(client, tmp_path, FRAME, ["a1"], confirm=True)
    assert client.frame_removals == [(FRAME, "a1")]
    assert client.deletes == []  # the account-wide endpoint is never touched
    assert result["scope"] == "frame" and result["deleted_count"] == 1
    assert not (tmp_path / "cache" / FRAME / "a1.jpg").exists()


async def test_delete_account_scope_uses_account_endpoint(tmp_path):
    await flagged_setup(tmp_path, "a1")
    client = FakeClient()

    preview = await server.delete_assets_impl(client, tmp_path, FRAME, ["a1"], scope="account")
    assert "every frame" in preview["effect"]
    await server.delete_assets_impl(client, tmp_path, FRAME, ["a1"], scope="account", confirm=True)
    assert client.deletes == ["a1"] and client.frame_removals == []


async def test_delete_rejects_unknown_scope(tmp_path):
    await flagged_setup(tmp_path, "a1")
    client = FakeClient()
    with pytest.raises(ValueError, match="scope must be"):
        await server.delete_assets_impl(client, tmp_path, FRAME, ["a1"], scope="everything", confirm=True)
    assert client.deletes == [] and client.frame_removals == []


# ------------------------------------------------- removing from every frame

KITCHEN = "frame-2"


async def two_frames(tmp_path):
    """a1 is on both frames, a2 only on the first, k1 only on the second."""
    await flagged_setup(tmp_path, "a1", "a2")
    await synced(tmp_path, "a1", "k1", frame=KITCHEN)
    matcher = FakeMatcher({"a1": [TARGET], "k1": [OTHER]})
    await server.scan_frame_faces(
        matcher, tmp_path / "cache" / KITCHEN, server.load_index(tmp_path, KITCHEN)
    )
    server.save_frame_names(
        tmp_path,
        [{"id": FRAME, "name": "Living Room"}, {"id": KITCHEN, "name": "Kitchen"}],
    )


async def test_all_frames_preview_lists_every_frame_each_photo_is_on(tmp_path):
    await two_frames(tmp_path)
    client = FakeClient()

    preview = await server.delete_assets_impl(
        client, tmp_path, FRAME, ["a1", "a2"], scope="all_frames")

    assert client.frame_removals == [] and client.deletes == []
    assert preview["status"] == "confirmation_required" and preview["scope"] == "all_frames"
    assert "every frame" in preview["effect"]
    by_id = {d["asset_id"]: d for d in preview["would_delete"]}
    assert by_id["a1"]["frames"] == [FRAME, KITCHEN] and by_id["a1"]["stays_on"] == []
    assert by_id["a2"]["frames"] == [FRAME]
    assert preview["frame_removal_count"] == 3
    assert preview["frames_affected"] == [
        {"id": FRAME, "name": "Living Room", "count": 2},
        {"id": KITCHEN, "name": "Kitchen", "count": 1},
    ]
    # After this neither photo is on a frame, so Aura no longer has them.
    assert preview["last_copy_count"] == 2


async def test_frame_preview_says_which_photos_are_on_no_other_frame(tmp_path):
    await two_frames(tmp_path)

    preview = await server.delete_assets_impl(FakeClient(), tmp_path, FRAME, ["a1", "a2"])

    by_id = {d["asset_id"]: d for d in preview["would_delete"]}
    assert by_id["a1"]["frames"] == [FRAME] and by_id["a1"]["stays_on"] == [KITCHEN]
    assert by_id["a2"]["stays_on"] == []
    assert preview["frame_removal_count"] == 2 and preview["last_copy_count"] == 1


async def test_all_frames_removes_from_each_frame_and_updates_each_cache(tmp_path):
    await two_frames(tmp_path)
    client = FakeClient()

    result = await server.delete_assets_impl(
        client, tmp_path, FRAME, ["a1", "a2"], scope="all_frames", confirm=True)

    assert client.frame_removals == [(FRAME, "a1"), (KITCHEN, "a1"), (FRAME, "a2")]
    assert client.deletes == []  # never the account-wide endpoint
    assert result["status"] == "completed" and result["deleted_count"] == 2
    assert result["frame_removal_count"] == 3 and result["failed_count"] == 0
    assert {d["asset_id"]: d["frames"] for d in result["deleted"]} == {
        "a1": [FRAME, KITCHEN], "a2": [FRAME]}
    for frame, left in ((FRAME, set()), (KITCHEN, {"k1"})):
        assert set(server.load_index(tmp_path, frame)) == left
        assert not (tmp_path / "cache" / frame / "a1.jpg").exists()
        log = json.loads((tmp_path / "removed" / frame / "removed.json").read_text())
        assert log["a1"]["scope"] == "all_frames"
        # The local copy is the only way back, so each frame keeps one.
        assert (tmp_path / "removed" / frame / "a1.jpg").is_file()
    with np.load(tmp_path / "cache" / KITCHEN / "embeddings.npz") as data:
        assert data.files == ["k1"]


async def test_all_frames_reports_the_frame_that_failed_and_leaves_it_alone(tmp_path):
    await two_frames(tmp_path)
    client = FakeClient(delete_errors={(KITCHEN, "a1"): AuraAPIError("HTTP 403: not a member", 403)})

    result = await server.delete_assets_impl(
        client, tmp_path, FRAME, ["a1", "a2"], scope="all_frames", confirm=True)

    assert result["failed"] == [
        {"asset_id": "a1", "frame_id": KITCHEN, "error": "HTTP 403: not a member",
         "reason": "Aura refused the request."}]
    by_id = {d["asset_id"]: d for d in result["deleted"]}
    assert by_id["a1"]["frames"] == [FRAME] and by_id["a1"]["still_on"] == [KITCHEN]
    assert by_id["a2"]["frames"] == [FRAME] and by_id["a2"]["still_on"] == []
    # a1 is still on a frame, so it does not count as removed.
    assert result["deleted_count"] == 1 and result["partial_count"] == 1
    assert result["frame_removal_count"] == 2
    assert (tmp_path / "cache" / KITCHEN / "a1.jpg").exists()
    assert "a1" in server.load_index(tmp_path, KITCHEN)
    assert not (tmp_path / "cache" / FRAME / "a1.jpg").exists()


async def test_a_lost_session_accounts_for_every_frame_it_did_not_reach(tmp_path):
    await two_frames(tmp_path)
    await synced(tmp_path, "a1", "a2", frame="frame-3")
    client = FakeClient(delete_errors={(KITCHEN, "a1"): AuraAuthError("still unauthorized")})

    result = await server.delete_assets_impl(
        client, tmp_path, FRAME, ["a1", "a2"], scope="all_frames", confirm=True)

    assert result["status"] == "aborted"
    assert [(f["asset_id"], f["frame_id"]) for f in result["failed"]] == [
        ("a1", KITCHEN), ("a1", "frame-3"), ("a2", FRAME), ("a2", "frame-3")]
    assert result["deleted"][0]["still_on"] == [KITCHEN, "frame-3"]
    assert result["deleted_count"] == 0 and result["partial_count"] == 1


async def test_removal_never_goes_beyond_the_frames_that_were_previewed(tmp_path):
    await two_frames(tmp_path)
    client = FakeClient()

    result = await server.delete_assets_impl(
        client, tmp_path, FRAME, ["a1", "a2"], scope="all_frames", confirm=True,
        previewed={"a1": [FRAME]})  # Kitchen was not in the preview; a2 was not either

    assert client.frame_removals == [(FRAME, "a1")]
    assert result["deleted_count"] == 1
    assert "a1" in server.load_index(tmp_path, KITCHEN)
    assert "a2" in server.load_index(tmp_path, FRAME)


async def test_each_removed_photo_is_on_record_before_the_next_is_touched(tmp_path):
    """If the app dies part-way, the photos already removed must be in the
    ledger and out of the index, or they could not be found or put back."""
    await two_frames(tmp_path)
    seen = {}

    class Watching(FakeClient):
        async def remove_asset_from_frame(self, frame_id, asset_id):
            if asset_id == "a2":
                log = tmp_path / "removed" / FRAME / "removed.json"
                seen["ledger"] = json.loads(log.read_text()) if log.is_file() else {}
                seen["index"] = set(server.load_index(tmp_path, FRAME))
            await super().remove_asset_from_frame(frame_id, asset_id)

    await server.delete_assets_impl(Watching(), tmp_path, FRAME, ["a1", "a2"], confirm=True)

    assert "a1" in seen["ledger"] and seen["index"] == {"a2"}


async def test_a_photo_with_no_copy_on_this_computer_is_not_removed(tmp_path):
    await two_frames(tmp_path)
    (tmp_path / "cache" / FRAME / "a2.jpg").unlink()  # e.g. its download failed
    client = FakeClient()

    preview = await server.delete_assets_impl(client, tmp_path, FRAME, ["a1", "a2"])
    assert preview["would_delete_count"] == 1 and preview["no_copy_asset_ids"] == ["a2"]

    result = await server.delete_assets_impl(client, tmp_path, FRAME, ["a1", "a2"], confirm=True)
    assert client.frame_removals == [(FRAME, "a1")]
    assert result["no_copy_asset_ids"] == ["a2"] and "a2" in server.load_index(tmp_path, FRAME)


async def test_all_frames_keeps_what_was_done_when_the_session_is_lost(tmp_path):
    await two_frames(tmp_path)
    client = FakeClient(delete_errors={(KITCHEN, "a1"): AuraAuthError("still unauthorized")})

    result = await server.delete_assets_impl(
        client, tmp_path, FRAME, ["a1", "a2"], scope="all_frames", confirm=True)

    assert client.frame_removals == [(FRAME, "a1"), (KITCHEN, "a1")]  # a2 never attempted
    assert result["status"] == "aborted"
    assert {d["asset_id"]: d["frames"] for d in result["deleted"]} == {"a1": [FRAME]}
    assert [(f["asset_id"], f.get("frame_id")) for f in result["failed"]] == [
        ("a1", KITCHEN), ("a2", FRAME)]
    assert "a1" not in server.load_index(tmp_path, FRAME)
    assert "a1" in server.load_index(tmp_path, KITCHEN)


async def test_account_delete_forgets_the_photo_on_every_synced_frame(tmp_path):
    await two_frames(tmp_path)
    client = FakeClient()

    result = await server.delete_assets_impl(
        client, tmp_path, FRAME, ["a1"], scope="account", confirm=True)

    assert client.deletes == ["a1"] and client.frame_removals == []
    assert result["deleted_count"] == 1
    assert "a1" not in server.load_index(tmp_path, FRAME)
    assert "a1" not in server.load_index(tmp_path, KITCHEN)
    assert (tmp_path / "removed" / KITCHEN / "a1.jpg").is_file()


async def test_all_frames_preview_names_frames_that_were_never_synced(tmp_path):
    await two_frames(tmp_path)
    server.save_frame_names(tmp_path, [{"id": "frame-3", "name": "Grandma"}])

    preview = await server.delete_assets_impl(
        FakeClient(), tmp_path, FRAME, ["a1"], scope="all_frames")

    assert preview["unsynced_frames"] == [{"id": "frame-3", "name": "Grandma"}]


async def test_delete_reports_progress(tmp_path):
    await two_frames(tmp_path)
    seen = []

    async def progress(done, total, message):
        seen.append((done, total))

    await server.delete_assets_impl(
        FakeClient(), tmp_path, FRAME, ["a1", "a2"], scope="all_frames", confirm=True,
        progress=progress)

    assert seen[-1] == (2, 2)


# ------------------------------------------- the same photo on several frames

async def test_sync_reuses_a_photo_another_frame_already_has(tmp_path):
    await synced(tmp_path, "a1", "a2")
    client = FakeClient([asset("a1"), asset("k1")])

    result = await server.sync_frame_photos_impl(client, tmp_path, KITCHEN)

    assert client.downloads == ["k1.jpeg"]  # a1 was already on this computer
    assert result["reused"] == 1 and result["downloaded"] == 1 and result["total_synced"] == 2
    copy = tmp_path / "cache" / KITCHEN / "a1.jpg"
    assert copy.read_bytes() == (tmp_path / "cache" / FRAME / "a1.jpg").read_bytes()
    assert server.load_index(tmp_path, KITCHEN)["a1"]["local_path"] == str(copy)


async def test_a_reused_photo_outlives_removal_from_the_frame_it_came_from(tmp_path):
    await synced(tmp_path, "a1")
    await server.sync_frame_photos_impl(FakeClient([asset("a1")]), tmp_path, KITCHEN)

    await server.delete_assets_impl(FakeClient(), tmp_path, FRAME, ["a1"], confirm=True)

    assert (tmp_path / "cache" / KITCHEN / "a1.jpg").read_bytes() == img("a1.jpeg")


async def test_find_people_reads_each_photo_once_however_many_frames_it_is_on(tmp_path):
    await synced(tmp_path, "a1", "a2")
    await server.sync_frame_photos_impl(FakeClient([asset("a1"), asset("k1")]), tmp_path, KITCHEN)
    matcher = FakeMatcher({"a1": [TARGET], "a2": [TARGET], "k1": [OTHER]})

    await server.find_people_impl(matcher, tmp_path)

    assert sorted(matcher.scanned) == ["a1", "a2", "k1"]
    with np.load(tmp_path / "cache" / KITCHEN / "embeddings.npz") as data:
        assert sorted(data.files) == ["a1", "k1"]


# ------------------------------------------------------------ saved sign-in

def test_session_is_saved_privately_and_never_holds_a_password(tmp_path):
    assert server.load_session(tmp_path) is None
    server.save_session(tmp_path, {"token": "tok", "user_id": "u1", "password": "nope"})

    path = tmp_path / "session.json"
    assert json.loads(path.read_text()) == {"token": "tok", "user_id": "u1"}
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    assert server.load_session(tmp_path) == {"token": "tok", "user_id": "u1"}

    server.clear_session(tmp_path)
    assert not path.exists() and server.load_session(tmp_path) is None


def test_a_damaged_session_file_counts_as_signed_out(tmp_path):
    (tmp_path / "session.json").write_text("{not json")
    assert server.load_session(tmp_path) is None
    (tmp_path / "session.json").write_text(json.dumps({"token": "tok"}))
    assert server.load_session(tmp_path) is None


def test_new_client_uses_the_saved_session_when_no_login_is_configured(tmp_path, monkeypatch):
    monkeypatch.delenv("AURA_EMAIL", raising=False)
    monkeypatch.delenv("AURA_PASSWORD", raising=False)
    assert not server.signed_in(tmp_path)
    server.save_session(tmp_path, {"token": "tok", "user_id": "u1"})

    assert server.signed_in(tmp_path)
    assert server.new_client(tmp_path).session == {"token": "tok", "user_id": "u1"}


# ------------------------------------------------------------ registration


async def test_tools_are_registered_with_expected_schema():
    tools = {tool.name: tool for tool in await server.mcp.list_tools()}
    assert set(tools) == {
        "list_frames",
        "sync_frame_photos",
        "find_person_in_frame",
        "find_photos_in_frame",
        "get_selection",
        "delete_assets",
        "restore_assets",
        "find_people",
        "list_people",
    }
    people_schema = tools["find_people"].input_schema
    assert not people_schema.get("required") and "ctx" not in people_schema.get("properties", {})
    assert not tools["list_people"].input_schema.get("required")
    delete_schema = tools["delete_assets"].input_schema
    assert delete_schema["properties"]["confirm"]["default"] is False
    assert set(delete_schema["required"]) == {"frame_id", "asset_ids"}
    assert delete_schema["properties"]["scope"]["default"] == "frame"
    assert delete_schema["properties"]["scope"]["enum"] == ["frame", "all_frames", "account"]
    find_schema = tools["find_person_in_frame"].input_schema
    assert find_schema["properties"]["similarity_threshold"]["default"] == 0.45
    assert find_schema["required"] == ["frame_id"]  # reference photo or person name
    assert {"reference_image_path", "person"} <= set(find_schema["properties"])
    assert "ctx" not in find_schema["properties"]
    desc_schema = tools["find_photos_in_frame"].input_schema
    assert set(desc_schema["required"]) == {"frame_id", "query"}
    assert desc_schema["properties"]["similarity_threshold"]["default"] == 0.22
    assert "ctx" not in desc_schema["properties"]


async def test_tool_errors_keep_their_message(monkeypatch, tmp_path):
    """Expected failures must reach the model as text, not a generic error."""
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "_client", None)
    monkeypatch.delenv("AURA_EMAIL", raising=False)
    monkeypatch.delenv("AURA_PASSWORD", raising=False)

    with pytest.raises(ToolError, match="Not signed in to Aura"):
        await server.list_frames()
    with pytest.raises(ToolError, match="has not been synced yet"):
        await server.delete_assets("never-synced", ["a1"])


async def test_delete_preview_works_without_credentials(monkeypatch, tmp_path):
    await flagged_setup(tmp_path, "a1")
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "_client", None)
    monkeypatch.delenv("AURA_EMAIL", raising=False)
    monkeypatch.delenv("AURA_PASSWORD", raising=False)

    result = await server.delete_assets(FRAME, ["a1"])
    assert result["status"] == "confirmation_required"
    assert result["would_delete_count"] == 1


# ------------------------------------------------- find photos by description

from aura_mcp import image_search  # noqa: E402


class FakeSearcher(image_search.ImageSearcher):
    """Canned CLIP vectors keyed by file stem; no model involved."""

    def __init__(self, vectors, query_vec=TARGET, bad=()):
        self.vectors = vectors
        self.query_vec = query_vec
        self.bad = set(bad)
        self.batched = []
        self.singled = []

    def image_embeddings(self, paths):
        out = []
        for p in paths:
            stem = Path(p).stem
            self.batched.append(stem)
            if stem in self.bad:
                raise RuntimeError("batch decode failed")
            out.append(self.vectors[stem])
        return out

    def image_embedding(self, path):
        stem = Path(path).stem
        self.singled.append(stem)
        if stem in self.bad:
            raise image_search.ImageReadError(f"bad image {path}")
        return self.vectors[stem]

    def query_embedding(self, query):
        if not query.strip():
            raise ValueError("Search query must not be empty.")
        return self.query_vec


async def test_find_photos_requires_sync_first(tmp_path):
    with pytest.raises(ValueError, match="sync_frame_photos"):
        await server.find_photos_by_description_impl(
            FakeSearcher({}), tmp_path, FRAME, "a desk"
        )


async def test_find_photos_rejects_empty_query(tmp_path):
    await synced(tmp_path, "a1")
    with pytest.raises(ValueError, match="description to search for"):
        await server.find_photos_by_description_impl(
            FakeSearcher({"a1": TARGET}), tmp_path, FRAME, "   "
        )


async def test_find_photos_ranks_and_thresholds(tmp_path):
    await synced(tmp_path, "hit", "mid", "miss")
    searcher = FakeSearcher(
        {"hit": unit(1, 0), "mid": unit(1, 1), "miss": unit(0, 1)},
        query_vec=unit(1, 0),
    )
    result = await server.find_photos_by_description_impl(
        searcher, tmp_path, FRAME, "a desk", similarity_threshold=0.5
    )
    ids = [m["asset_id"] for m in result["matches"]]
    assert ids == ["hit", "mid"]  # sorted by similarity, "miss" below threshold
    assert result["matches"][0]["similarity"] == pytest.approx(1.0)
    assert result["match_count"] == 2
    # Persisted for the review page, separate from the person flow's matches.json
    saved = json.loads((tmp_path / "flagged" / FRAME / "description.json").read_text())
    assert saved["query"] == "a desk" and [m["asset_id"] for m in saved["matches"]] == ids


async def test_find_photos_top_k_caps_results(tmp_path):
    await synced(tmp_path, "hit", "mid")
    searcher = FakeSearcher({"hit": unit(1, 0), "mid": unit(1, 1)}, query_vec=unit(1, 0))
    result = await server.find_photos_by_description_impl(
        searcher, tmp_path, FRAME, "a desk", similarity_threshold=-1.0, top_k=1
    )
    assert [m["asset_id"] for m in result["matches"]] == ["hit"]


async def test_find_photos_caches_embeddings_between_queries(tmp_path):
    await synced(tmp_path, "a1", "a2")
    vectors = {"a1": unit(1, 0), "a2": unit(0, 1)}
    first = FakeSearcher(vectors, query_vec=unit(1, 0))
    await server.find_photos_by_description_impl(first, tmp_path, FRAME, "a desk")
    assert set(first.batched) == {"a1", "a2"}
    assert (tmp_path / "cache" / FRAME / "clip_embeddings.npz").is_file()

    # A second query must reuse the cached embeddings, scanning nothing.
    second = FakeSearcher(vectors, query_vec=unit(0, 1))
    result = await server.find_photos_by_description_impl(second, tmp_path, FRAME, "a cat")
    assert second.batched == [] and second.singled == []
    assert result["matches"][0]["asset_id"] == "a2"  # query now points at a2


async def test_find_photos_survives_an_unreadable_photo(tmp_path):
    await synced(tmp_path, "good", "bad")
    searcher = FakeSearcher(
        {"good": unit(1, 0), "bad": unit(1, 0)}, query_vec=unit(1, 0), bad={"bad"}
    )
    result = await server.find_photos_by_description_impl(
        searcher, tmp_path, FRAME, "a desk", similarity_threshold=-1.0
    )
    assert result["unreadable_asset_ids"] == ["bad"]
    assert [m["asset_id"] for m in result["matches"]] == ["good"]


async def test_find_people_with_nothing_synced(tmp_path):
    built = await server.find_people_impl(FaceMatcher(app=object()), tmp_path)
    assert (built["frames_scanned"], built["people"], built["seen_once"]) == (0, 0, 0)
    assert server.list_people_impl(tmp_path)["people"] == []


# ------------------------------------------------- fast loading (medium copies)


async def test_sync_asks_for_the_medium_size_and_records_it(tmp_path):
    client = FakeClient([asset("a1"), asset("a2")])
    await server.sync_frame_photos_impl(client, tmp_path, FRAME)

    assert server.SYNC_MAX_SIDE == 1600
    assert client.sizes == [1600, 1600]
    index = server.load_index(tmp_path, FRAME)
    assert {a: e["size"] for a, e in index.items()} == {"a1": 1600, "a2": 1600}


async def test_sync_downloads_newest_first(tmp_path):
    assets = [
        asset("old", taken_at="2020-01-01T00:00:00Z"),
        asset("undated", taken_at=None),
        asset("new", taken_at="2025-06-01T00:00:00Z"),
        asset("mid", taken_at="2023-01-01T00:00:00Z"),
    ]
    client = FakeClient(assets)
    await server.sync_frame_photos_impl(client, tmp_path, FRAME)
    assert client.downloads == ["new.jpeg", "mid.jpeg", "old.jpeg", "undated.jpeg"]


async def test_sync_never_has_more_than_the_limit_in_flight(tmp_path, monkeypatch):
    import asyncio

    monkeypatch.setattr(server, "SYNC_CONCURRENCY", 4)
    state = {"now": 0, "most": 0}

    class Slow(FakeClient):
        async def download_photo(self, user_id, file_name, dest, max_side=None):
            state["now"] += 1
            state["most"] = max(state["most"], state["now"])
            await asyncio.sleep(0.01)
            state["now"] -= 1
            return await super().download_photo(user_id, file_name, dest, max_side)

    client = Slow([asset(f"a{i}") for i in range(20)])
    result = await server.sync_frame_photos_impl(client, tmp_path, FRAME)

    assert result["downloaded"] == 20
    assert state["most"] == 4  # several at once, never more than the limit


async def test_sync_reports_progress_as_each_photo_finishes(tmp_path):
    seen = []

    async def progress(done, total, message):
        seen.append((done, total))

    await server.sync_frame_photos_impl(
        FakeClient([asset(f"a{i}") for i in range(3)]), tmp_path, FRAME, progress)
    assert seen == [(1, 3), (2, 3), (3, 3)]


async def test_sync_retries_a_failed_download_once(tmp_path):
    class Flaky(FakeClient):
        async def download_photo(self, user_id, file_name, dest, max_side=None):
            if file_name not in self.downloads and file_name == "a1.jpeg":
                self.downloads.append(file_name)
                raise AuraAPIError("Download of a1.jpeg failed: ReadTimeout")
            return await super().download_photo(user_id, file_name, dest, max_side)

    client = Flaky([asset("a1")])
    result = await server.sync_frame_photos_impl(client, tmp_path, FRAME)
    assert result["downloaded"] == 1 and result["failed"] == []

    stubborn = FakeClient([asset("a2")], fail_downloads={"a2.jpeg"})
    result = await server.sync_frame_photos_impl(stubborn, tmp_path, FRAME)
    assert stubborn.downloads == ["a2.jpeg", "a2.jpeg"]  # tried twice, then given up
    assert [f["asset_id"] for f in result["failed"]] == ["a2"]


async def test_an_old_full_size_cache_is_left_alone(tmp_path):
    folder = tmp_path / "cache" / FRAME
    folder.mkdir(parents=True)
    (folder / "a1.jpg").write_bytes(original("a1.jpeg"))
    (folder / "index.json").write_text(json.dumps({"a1": {
        "file_name": "a1.jpeg", "user_id": "u1", "taken_at": None,
        "local_path": str(folder / "a1.jpg")}}))  # written before medium copies existed
    client = FakeClient([asset("a1"), asset("a2")])

    await server.sync_frame_photos_impl(client, tmp_path, FRAME)

    assert client.downloads == ["a2.jpeg"]
    index = server.load_index(tmp_path, FRAME)
    assert "size" not in index["a1"] and index["a2"]["size"] == 1600
    assert (folder / "a1.jpg").read_bytes() == original("a1.jpeg")


async def test_a_photo_of_unknown_size_counts_as_not_the_original(tmp_path):
    """A sync that stopped part-way leaves photos but no index: they may be
    medium copies, so they must never pass for the original."""
    folder = tmp_path / "cache" / FRAME
    folder.mkdir(parents=True)
    (folder / "a1.jpg").write_bytes(img("a1.jpeg"))

    await server.sync_frame_photos_impl(FakeClient([asset("a1")]), tmp_path, FRAME)

    assert "size" in server.load_index(tmp_path, FRAME)["a1"]


async def test_a_reused_photo_keeps_its_size(tmp_path):
    await synced(tmp_path, "a1")  # medium
    await server.sync_frame_photos_impl(FakeClient([asset("a1")]), tmp_path, KITCHEN)
    assert server.load_index(tmp_path, KITCHEN)["a1"]["size"] == 1600

    old = tmp_path / "cache" / "frame-3"
    old.mkdir(parents=True)
    (old / "o1.jpg").write_bytes(original("o1.jpeg"))
    (old / "index.json").write_text(json.dumps({"o1": {
        "file_name": "o1.jpeg", "user_id": "u1", "local_path": str(old / "o1.jpg")}}))
    client = FakeClient([asset("o1")])
    await server.sync_frame_photos_impl(client, tmp_path, KITCHEN)
    assert client.downloads == [] and "size" not in server.load_index(tmp_path, KITCHEN)["o1"]


async def test_removal_fetches_the_original_before_asking_aura(tmp_path):
    await flagged_setup(tmp_path, "a1", "a2")
    client = FakeClient()

    result = await server.delete_assets_impl(client, tmp_path, FRAME, ["a1"], confirm=True)

    assert client.calls == [("download", "a1.jpeg", None), ("remove", FRAME, "a1")]
    assert result["deleted_count"] == 1
    kept = tmp_path / "removed" / FRAME / "a1.jpg"
    assert kept.read_bytes() == original("a1.jpeg")
    assert result["deleted"][0]["local_copy"] == str(kept)
    assert not (tmp_path / "cache" / FRAME / "a1.jpg").exists()  # the medium copy is gone
    log = json.loads((tmp_path / "removed" / FRAME / "removed.json").read_text())
    assert log["a1"]["local_path"] == str(kept) and "size" not in log["a1"]
    # nothing but the kept original and the ledger is left behind
    assert sorted(p.name for p in (tmp_path / "removed" / FRAME).iterdir()) == ["a1.jpg", "removed.json"]


async def test_a_failed_original_download_blocks_that_removal(tmp_path):
    await flagged_setup(tmp_path, "a1", "a2")
    client = FakeClient(fail_originals={"a1.jpeg"})

    result = await server.delete_assets_impl(client, tmp_path, FRAME, ["a1", "a2"], confirm=True)

    assert client.frame_removals == [(FRAME, "a2")]  # nothing sent to Aura for a1
    assert result["deleted_count"] == 1 and result["failed_count"] == 1
    failure = result["failed"][0]
    assert failure["asset_id"] == "a1" and "not removed" in failure["error"].lower()
    assert "full-size" in failure["error"] and "HTTP 500" in failure["error"]
    assert (tmp_path / "cache" / FRAME / "a1.jpg").exists() and "a1" in server.load_index(tmp_path, FRAME)
    kept = tmp_path / "removed" / FRAME
    assert sorted(p.name for p in kept.iterdir()) == ["a2.jpg", "removed.json"]


async def test_a_failed_original_download_is_reported_per_frame_for_all_frames(tmp_path):
    await two_frames(tmp_path)
    client = FakeClient(fail_originals={"a1.jpeg"})

    result = await server.delete_assets_impl(
        client, tmp_path, FRAME, ["a1"], scope="all_frames", confirm=True)

    assert client.frame_removals == [] and result["deleted_count"] == 0
    assert [(f["asset_id"], f["frame_id"]) for f in result["failed"]] == [("a1", FRAME), ("a1", KITCHEN)]


async def test_all_frames_puts_one_original_in_every_frames_removed_folder(tmp_path):
    await two_frames(tmp_path)
    client = FakeClient()

    await server.delete_assets_impl(client, tmp_path, FRAME, ["a1"], scope="all_frames", confirm=True)

    assert client.downloads == ["a1.jpeg"]  # fetched once
    for frame in (FRAME, KITCHEN):
        assert (tmp_path / "removed" / frame / "a1.jpg").read_bytes() == original("a1.jpeg")
        assert not (tmp_path / "cache" / frame / "a1.jpg").exists()


async def test_a_partial_all_frames_removal_keeps_the_original_where_it_came_off(tmp_path):
    await two_frames(tmp_path)
    client = FakeClient(delete_errors={(FRAME, "a1"): AuraAPIError("HTTP 403", 403)})

    await server.delete_assets_impl(client, tmp_path, FRAME, ["a1"], scope="all_frames", confirm=True)

    assert (tmp_path / "removed" / KITCHEN / "a1.jpg").read_bytes() == original("a1.jpeg")
    assert not (tmp_path / "removed" / FRAME).exists() or not any(
        (tmp_path / "removed" / FRAME).glob("*.jpg"))
    assert (tmp_path / "cache" / FRAME / "a1.jpg").exists()


async def test_account_scope_also_keeps_the_original(tmp_path):
    await two_frames(tmp_path)
    client = FakeClient()

    await server.delete_assets_impl(client, tmp_path, FRAME, ["a1"], scope="account", confirm=True)

    assert client.calls == [("download", "a1.jpeg", None), ("delete", "a1")]
    assert (tmp_path / "removed" / KITCHEN / "a1.jpg").read_bytes() == original("a1.jpeg")


async def test_a_photo_already_kept_as_the_original_is_removed_as_before(tmp_path):
    folder = tmp_path / "cache" / FRAME
    folder.mkdir(parents=True)
    (folder / "a1.jpg").write_bytes(original("a1.jpeg"))
    (folder / "index.json").write_text(json.dumps({"a1": {
        "file_name": "a1.jpeg", "user_id": "u1", "taken_at": None,
        "local_path": str(folder / "a1.jpg")}}))
    client = FakeClient()

    await server.delete_assets_impl(client, tmp_path, FRAME, ["a1"], confirm=True)

    assert client.downloads == [] and client.frame_removals == [(FRAME, "a1")]
    assert (tmp_path / "removed" / FRAME / "a1.jpg").read_bytes() == original("a1.jpeg")


async def test_the_removal_preview_downloads_nothing(tmp_path):
    await two_frames(tmp_path)
    client = FakeClient()
    await server.delete_assets_impl(client, tmp_path, FRAME, ["a1", "a2"], scope="all_frames")
    assert client.calls == []


# ------------------------------------------- loading: faces read as photos land


async def test_load_reads_faces_while_downloading_and_groups_early(tmp_path, monkeypatch):
    import asyncio

    from aura_mcp import people_index

    monkeypatch.setattr(server, "GROUP_FIRST_AT", 3)
    monkeypatch.setattr(server, "GROUP_EVERY", 5)
    monkeypatch.setattr(server, "SYNC_CONCURRENCY", 2)
    ids = [f"a{i:02d}" for i in range(16)]

    class Slow(FakeClient):
        async def download_photo(self, user_id, file_name, dest, max_side=None):
            await asyncio.sleep(0.01)
            return await super().download_photo(user_id, file_name, dest, max_side)

    client = Slow([asset(a) for a in ids])
    matcher = FakeMatcher({a: [TARGET if i % 2 else OTHER] for i, a in enumerate(ids)})
    groupings = []
    real_rebuild = people_index.rebuild

    def watching_rebuild(root, threshold=server.CLUSTER_THRESHOLD):
        on_disk = len(list((root / "cache" / FRAME).glob("*.jpg")))
        groupings.append((len(client.downloads), on_disk, len(matcher.scanned)))
        return real_rebuild(root, threshold)

    monkeypatch.setattr(people_index, "rebuild", watching_rebuild)
    messages = []

    async def progress(done, total, message):
        messages.append(message)

    result = await server.load_frames_impl(client, matcher, tmp_path, [FRAME], progress)

    assert set(result) >= {"frames", "people", "seen_once"}
    assert result["frames"] == [{"id": FRAME, "name": FRAME, "photos": 16, "failed": 0}]
    # The first grouping came while photos were still arriving.
    first = groupings[0]
    assert first[0] < len(ids) and first[2] >= 3
    assert len(groupings) >= 3  # early, at least once more along the way, and at the end
    assert sorted(matcher.scanned) == ids  # each photo read once
    assert any("faces read in" in m for m in messages if m)
    assert result["people"] + result["seen_once"] >= 1


async def test_load_progress_reads_plainly(tmp_path, monkeypatch):
    server.save_frame_names(tmp_path, [{"id": FRAME, "name": "Den"}])
    messages = []

    async def progress(done, total, message):
        messages.append(message)

    ids = [f"a{i}" for i in range(3)]
    await server.load_frames_impl(
        FakeClient([asset(a) for a in ids]), FakeMatcher({a: [] for a in ids}),
        tmp_path, [FRAME], progress)

    assert "Getting photos from Den: 3 of 3 · faces read in 3" in messages


async def test_load_reads_a_photo_once_across_frames(tmp_path):
    client = FakeClient([asset("a1"), asset("k1")])
    matcher = FakeMatcher({"a1": [TARGET], "k1": [OTHER]})
    await server.load_frames_impl(client, matcher, tmp_path, [FRAME, KITCHEN])
    assert sorted(matcher.scanned) == ["a1", "k1"]
    with np.load(tmp_path / "cache" / KITCHEN / "embeddings.npz") as data:
        assert sorted(data.files) == ["a1", "k1"]


# ----------------------------------------------- plain words when Aura says no


def _transport_failure():
    import httpx

    try:
        raise httpx.ConnectError("[Errno 8] nodename nor servname provided")
    except httpx.ConnectError as cause:
        try:
            raise AuraAPIError("GET /frames.json failed: [Errno 8] nodename nor servname provided") from cause
        except AuraAPIError as exc:
            return exc


@pytest.mark.parametrize("exc, kind, words", [
    (AuraAuthError("POST /x was still unauthorized after logging in again."), "signed_out",
     "Your Aura sign-in has expired. Sign in again."),
    (_transport_failure(), "unreachable", "Aura could not be reached"),
    (AuraAPIError("GET /frames.json failed with HTTP 503: down", 503), "aura_trouble",
     "Aura is having trouble"),
    (AuraAPIError("POST /x failed with HTTP 422: nope", 422), "refused", "Aura refused"),
    (AuraRefusedError("Aura could not add asset a to frame f (number_failed=1)."), "refused",
     "Aura refused"),
    (AuraAPIError("frames.json had no 'frames' list (keys: ['x'])."), "unexpected",
     "may need an update"),
    (OSError(28, "No space left on device"), "local", "on this computer"),
    (RuntimeError("boom"), "other", "Something unexpected went wrong"),
])
def test_each_failure_becomes_a_plain_sentence_with_the_detail_kept(exc, kind, words):
    plain = server.explain_error(exc)
    assert plain["kind"] == kind
    assert words in plain["message"]
    # The technical text is kept, never shown alone and never as the sentence.
    assert plain["detail"] and plain["detail"] != plain["message"]
    assert "HTTP" not in plain["message"] and "Errno" not in plain["message"]


def test_our_own_plain_messages_pass_through():
    plain = server.explain_error(ValueError("Choose at least one frame."))
    assert plain == {"kind": "plain", "message": "Choose at least one frame.", "detail": ""}


async def test_a_failed_removal_says_why_in_plain_words(tmp_path):
    await flagged_setup(tmp_path, "a1", "a2")
    client = FakeClient(delete_errors={"a2": AuraAuthError("still unauthorized")})
    client.delete_errors["a1"] = _transport_failure()

    result = await server.delete_assets_impl(client, tmp_path, FRAME, ["a1", "a2"], confirm=True)

    reasons = {f["asset_id"]: f["reason"] for f in result["failed"]}
    assert "could not be reached" in reasons["a1"]
    assert "Sign in again" in reasons["a2"]


async def test_a_failed_original_download_gives_a_plain_reason(tmp_path):
    await flagged_setup(tmp_path, "a1")
    client = FakeClient(fail_originals={"a1.jpeg"})

    result = await server.delete_assets_impl(client, tmp_path, FRAME, ["a1"], confirm=True)

    reason = result["failed"][0]["reason"]
    assert reason.startswith("Not removed") and "HTTP" not in reason


# ------------------------------------------------------- putting photos back


async def test_restore_puts_a_removed_photo_back(tmp_path):
    await flagged_setup(tmp_path, "a1", "a2")
    client = FakeClient()
    await server.delete_assets_impl(client, tmp_path, FRAME, ["a1"], confirm=True)
    assert not (tmp_path / "cache" / FRAME / "a1.jpg").exists()

    result = await server.restore_assets_impl(client, tmp_path, FRAME, ["a1"])

    assert result["restored_count"] == 1 and client.restored == [(FRAME, "a1")]
    assert (tmp_path / "cache" / FRAME / "a1.jpg").read_bytes() == original("a1.jpeg")
    assert "a1" in server.load_index(tmp_path, FRAME)
    assert json.loads((tmp_path / "removed" / FRAME / "removed.json").read_text()) == {}


async def test_restore_says_when_aura_no_longer_has_the_photo(tmp_path):
    await flagged_setup(tmp_path, "a1")
    client = FakeClient()
    await server.delete_assets_impl(client, tmp_path, FRAME, ["a1"], confirm=True)
    client.restore_errors["a1"] = AuraRefusedError(
        "Aura could not add asset a1 to frame frame-1 (number_failed=1).")

    result = await server.restore_assets_impl(client, tmp_path, FRAME, ["a1"])

    failure = result["failed"][0]
    kept = tmp_path / "removed" / FRAME / "a1.jpg"
    assert failure["kind"] == "refused" and failure["local_copy"] == str(kept)
    assert kept.is_file()  # the copy stays where the user is told it is
    assert "a1" in json.loads((tmp_path / "removed" / FRAME / "removed.json").read_text())


# ------------------------------------- review 2: the safety copy is never lost


def ledger(tmp_path, frame=FRAME):
    path = tmp_path / "removed" / frame / "removed.json"
    return json.loads(path.read_text()) if path.is_file() else {}


async def _medium_landed_but_index_not_written(tmp_path, frame=FRAME):
    """Load 1 fails to get a1; load 2 gets the medium copy and quits before the index."""
    await server.sync_frame_photos_impl(
        FakeClient([asset("a1")], fail_downloads={"a1.jpeg"}), tmp_path, frame)
    assert server.load_index(tmp_path, frame)["a1"]["local_path"] is None

    class Dies(FakeClient):
        async def download_photo(self, *args, **kwargs):
            await super().download_photo(*args, **kwargs)
            raise RuntimeError("killed")

    with pytest.raises(RuntimeError):
        await server.sync_frame_photos_impl(Dies([asset("a1")]), tmp_path, frame)
    assert (tmp_path / "cache" / frame / "a1.jpg").read_bytes() == img("a1.jpeg")


async def test_a_medium_copy_after_a_failed_download_never_passes_for_the_original(tmp_path):
    await _medium_landed_but_index_not_written(tmp_path)
    await server.sync_frame_photos_impl(FakeClient([asset("a1")]), tmp_path, FRAME)

    entry = server.load_index(tmp_path, FRAME)["a1"]
    assert not server.is_original(entry)
    client = FakeClient()
    await server.delete_assets_impl(client, tmp_path, FRAME, ["a1"], confirm=True)
    assert client.calls[0] == ("download", "a1.jpeg", None)  # the original was fetched
    assert (tmp_path / "removed" / FRAME / "a1.jpg").read_bytes() == original("a1.jpeg")


async def test_a_reused_medium_copy_after_a_failed_download_never_passes_for_the_original(tmp_path):
    await _medium_landed_but_index_not_written(tmp_path, frame=KITCHEN)
    # FRAME takes KITCHEN's file instead of downloading it.
    client = FakeClient([asset("a1")])
    await server.sync_frame_photos_impl(client, tmp_path, FRAME)

    assert client.downloads == []
    assert not server.is_original(server.load_index(tmp_path, FRAME)["a1"])


class SendsBack(FakeClient):
    """Answers the original's download with whatever body it is given."""

    def __init__(self, body, **kwargs):
        super().__init__(**kwargs)
        self.body = body

    async def download_photo(self, user_id, file_name, dest, max_side=None):
        if max_side is not None:
            return await super().download_photo(user_id, file_name, dest, max_side)
        self.calls.append(("download", file_name, max_side))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.body)
        return dest


def _nothing_removed_and_nothing_lost(tmp_path, client, result):
    assert client.frame_removals == [] and client.deletes == []
    assert result["deleted_count"] == 0 and result["failed_count"] == 1
    failure = result["failed"][0]
    assert failure["asset_id"] == "a1" and failure["reason"].startswith("Not removed")
    assert "HTTP" not in failure["reason"]
    # The medium copy stays, and no bad file is left behind.
    assert (tmp_path / "cache" / FRAME / "a1.jpg").read_bytes() == img("a1.jpeg")
    assert "a1" in server.load_index(tmp_path, FRAME)
    kept = tmp_path / "removed" / FRAME
    assert not kept.exists() or not any(kept.glob("a1*"))
    assert "a1" not in ledger(tmp_path)


@pytest.mark.parametrize("body", [
    b"<html>Service temporarily unavailable</html>",  # a 200 that is not a photo
    _jpeg(20, 24) + b"original:a1.jpeg",  # a real image, smaller than the copy here
    _ORIGINAL_JPEG[: len(_ORIGINAL_JPEG) - 300],  # a truncated JPEG
], ids=["html", "smaller", "truncated"])
async def test_a_bad_original_blocks_the_removal(tmp_path, body):
    await flagged_setup(tmp_path, "a1")
    client = SendsBack(body)

    result = await server.delete_assets_impl(client, tmp_path, FRAME, ["a1"], confirm=True)

    _nothing_removed_and_nothing_lost(tmp_path, client, result)


async def test_a_bad_original_never_replaces_a_good_kept_copy(tmp_path):
    await flagged_setup(tmp_path, "a1")
    kept = tmp_path / "removed" / FRAME / "a1.jpg"
    kept.parent.mkdir(parents=True)
    kept.write_bytes(original("a1.jpeg"))  # kept by an earlier removal of the same photo
    (kept.parent / "removed.json").write_text(json.dumps({"a1": {
        "file_name": "a1.jpeg", "local_path": str(kept), "state": "removed",
        "removed_at": "2026-01-01T00:00:00+00:00"}}))
    client = SendsBack(b"<html>oops</html>")

    result = await server.delete_assets_impl(client, tmp_path, FRAME, ["a1"], confirm=True)

    assert client.frame_removals == [] and result["failed_count"] == 1
    assert kept.read_bytes() == original("a1.jpeg")
    assert ledger(tmp_path)["a1"]["state"] == "removed"
    assert sorted(p.name for p in kept.parent.iterdir()) == ["a1.jpg", "removed.json"]


def test_the_check_of_an_original(tmp_path):
    medium = tmp_path / "m.jpg"
    medium.write_bytes(img("a1.jpeg"))
    good = tmp_path / "o.jpg"
    good.write_bytes(original("a1.jpeg"))
    assert server.check_original(good, [medium]) is None
    assert server.check_original(medium, [medium]) is None  # never enlarged: same size is fine
    small = tmp_path / "s.jpg"
    small.write_bytes(_jpeg(20, 24))
    assert "smaller" in server.check_original(small, [medium])
    html = tmp_path / "h.jpg"
    html.write_bytes(b"<html></html>")
    assert "not a whole photo" in server.check_original(html, [medium])


class Interrupted(BaseException):
    """Stands in for the app being quit: not an Exception, so nothing catches it."""


async def test_quitting_right_after_aura_acted_leaves_a_ledger_entry_and_the_original(tmp_path):
    await two_frames(tmp_path)

    class QuitAfterFirstFrame(FakeClient):
        async def remove_asset_from_frame(self, frame_id, asset_id):
            await super().remove_asset_from_frame(frame_id, asset_id)
            raise Interrupted()

    client = QuitAfterFirstFrame()
    with pytest.raises(Interrupted):
        await server.delete_assets_impl(
            client, tmp_path, FRAME, ["a1"], scope="all_frames", confirm=True)

    assert client.frame_removals == [(FRAME, "a1")]
    for frame in (FRAME, KITCHEN):
        entry = ledger(tmp_path, frame)["a1"]
        assert entry["state"] == "removing" and entry["started_at"]
        kept = Path(entry["local_path"])
        assert kept == tmp_path / "removed" / frame / "a1.jpg"
        assert kept.read_bytes() == original("a1.jpeg")
    # No stray temporary file, and the medium copies are still here.
    assert sorted(p.name for p in (tmp_path / "removed" / FRAME).iterdir()) == ["a1.jpg", "removed.json"]
    assert (tmp_path / "cache" / FRAME / "a1.jpg").exists()


async def test_the_ledger_says_removing_before_aura_is_asked(tmp_path):
    await flagged_setup(tmp_path, "a1")
    seen = []

    class Watching(FakeClient):
        async def remove_asset_from_frame(self, frame_id, asset_id):
            entry = ledger(tmp_path)[asset_id]
            seen.append((entry["state"], Path(entry["local_path"]).read_bytes()))
            await super().remove_asset_from_frame(frame_id, asset_id)

    await server.delete_assets_impl(Watching(), tmp_path, FRAME, ["a1"], confirm=True)

    assert seen == [("removing", original("a1.jpeg"))]
    entry = ledger(tmp_path)["a1"]
    assert entry["state"] == "removed" and entry["removed_at"] and entry["scope"] == "frame"


async def test_an_old_full_size_copy_is_kept_before_aura_is_asked(tmp_path):
    folder = tmp_path / "cache" / FRAME
    folder.mkdir(parents=True)
    (folder / "a1.jpg").write_bytes(original("a1.jpeg"))
    (folder / "index.json").write_text(json.dumps({"a1": {
        "file_name": "a1.jpeg", "user_id": "u1", "taken_at": None,
        "local_path": str(folder / "a1.jpg")}}))
    seen = []

    class Watching(FakeClient):
        async def remove_asset_from_frame(self, frame_id, asset_id):
            entry = ledger(tmp_path)[asset_id]
            seen.append((entry["state"], Path(entry["local_path"]).read_bytes()))
            raise Interrupted()

    with pytest.raises(Interrupted):
        await server.delete_assets_impl(Watching(), tmp_path, FRAME, ["a1"], confirm=True)

    assert seen == [("removing", original("a1.jpeg"))]
    assert (tmp_path / "removed" / FRAME / "a1.jpg").read_bytes() == original("a1.jpeg")


async def test_a_refused_removal_takes_its_ledger_entry_back_out(tmp_path):
    await flagged_setup(tmp_path, "a1")
    client = FakeClient(delete_errors={"a1": AuraAPIError("HTTP 403", 403)})

    result = await server.delete_assets_impl(client, tmp_path, FRAME, ["a1"], confirm=True)

    assert result["deleted_count"] == 0
    assert ledger(tmp_path) == {}
    kept = tmp_path / "removed" / FRAME
    assert not any(kept.glob("a1*"))  # this run made the copy, so it goes again
    assert (tmp_path / "cache" / FRAME / "a1.jpg").exists()


async def test_a_refused_removal_leaves_an_earlier_entry_as_it_was(tmp_path):
    await flagged_setup(tmp_path, "a1")
    kept = tmp_path / "removed" / FRAME / "a1.jpg"
    kept.parent.mkdir(parents=True)
    kept.write_bytes(original("a1.jpeg"))
    earlier = {"a1": {"file_name": "a1.jpeg", "local_path": str(kept), "state": "removed",
                      "removed_at": "2026-01-01T00:00:00+00:00", "scope": "frame"}}
    (kept.parent / "removed.json").write_text(json.dumps(earlier))
    client = FakeClient(delete_errors={"a1": AuraAPIError("HTTP 403", 403)})

    await server.delete_assets_impl(client, tmp_path, FRAME, ["a1"], confirm=True)

    assert ledger(tmp_path) == earlier and kept.is_file()


async def test_putting_back_a_photo_whose_removal_was_cut_short(tmp_path):
    await flagged_setup(tmp_path, "a1")

    class Quit(FakeClient):
        async def remove_asset_from_frame(self, frame_id, asset_id):
            await super().remove_asset_from_frame(frame_id, asset_id)
            raise Interrupted()

    with pytest.raises(Interrupted):
        await server.delete_assets_impl(Quit(), tmp_path, FRAME, ["a1"], confirm=True)
    assert ledger(tmp_path)["a1"]["state"] == "removing"

    client = FakeClient()
    result = await server.restore_assets_impl(client, tmp_path, FRAME, ["a1"])

    assert result["restored_count"] == 1 and client.restored == [(FRAME, "a1")]
    assert ledger(tmp_path) == {}
    # The kept original took the medium copy's place; nothing is left over.
    assert (tmp_path / "cache" / FRAME / "a1.jpg").read_bytes() == original("a1.jpeg")
    assert server.is_original(server.load_index(tmp_path, FRAME)["a1"])
    assert not any((tmp_path / "removed" / FRAME).glob("*.jpg"))


async def test_putting_back_a_cut_short_removal_that_aura_refuses_says_so(tmp_path):
    await flagged_setup(tmp_path, "a1")

    class Quit(FakeClient):
        async def remove_asset_from_frame(self, frame_id, asset_id):
            raise Interrupted()

    with pytest.raises(Interrupted):
        await server.delete_assets_impl(Quit(), tmp_path, FRAME, ["a1"], confirm=True)
    client = FakeClient()
    client.restore_errors["a1"] = AuraRefusedError("number_failed=1")

    result = await server.restore_assets_impl(client, tmp_path, FRAME, ["a1"])

    failure = result["failed"][0]
    assert failure["state"] == "removing" and failure["kind"] == "refused"
    assert failure["local_copy"] == str(tmp_path / "removed" / FRAME / "a1.jpg")
    assert ledger(tmp_path)["a1"]["state"] == "removing"


async def test_a_load_moves_a_photo_gone_from_the_frame_instead_of_deleting_it(tmp_path):
    # A cache from before this branch: full-size copies, nothing fetched.
    folder = tmp_path / "cache" / FRAME
    folder.mkdir(parents=True)
    (folder / "a1.jpg").write_bytes(original("a1.jpeg"))
    (folder / "index.json").write_text(json.dumps({"a1": {
        "file_name": "a1.jpeg", "user_id": "u1", "taken_at": "2024-01-01T00:00:00Z",
        "local_path": str(folder / "a1.jpg")}}))

    # Someone took a1 off the frame with the Aura app.
    result = await server.sync_frame_photos_impl(FakeClient([asset("a2")]), tmp_path, FRAME)

    assert result["pruned_stale"] == 1
    assert not (folder / "a1.jpg").exists()
    kept = tmp_path / "removed" / FRAME / "a1.jpg"
    assert kept.read_bytes() == original("a1.jpeg")
    entry = ledger(tmp_path)["a1"]
    assert entry["state"] == "missing" and entry["found_missing_at"]
    assert entry["local_path"] == str(kept) and entry["file_name"] == "a1.jpeg"
    assert "size" not in entry  # it was the original


async def test_a_load_does_not_keep_a_second_copy_of_a_photo_the_ledger_knows(tmp_path):
    await flagged_setup(tmp_path, "a1", "a2")
    await server.delete_assets_impl(FakeClient(), tmp_path, FRAME, ["a1"], confirm=True)
    before = ledger(tmp_path)
    # A stray cache copy of the removed photo (say, a removal cut short).
    (tmp_path / "cache" / FRAME / "a1.jpg").write_bytes(img("a1.jpeg"))

    await server.sync_frame_photos_impl(FakeClient([asset("a2")]), tmp_path, FRAME)

    assert ledger(tmp_path) == before
    assert not (tmp_path / "cache" / FRAME / "a1.jpg").exists()
    assert (tmp_path / "removed" / FRAME / "a1.jpg").read_bytes() == original("a1.jpeg")


async def test_a_load_settles_a_removal_that_was_cut_short(tmp_path):
    await two_frames(tmp_path)

    class QuitAfterFirstFrame(FakeClient):
        async def remove_asset_from_frame(self, frame_id, asset_id):
            await super().remove_asset_from_frame(frame_id, asset_id)
            raise Interrupted()

    with pytest.raises(Interrupted):
        await server.delete_assets_impl(
            QuitAfterFirstFrame(), tmp_path, FRAME, ["a1"], scope="all_frames", confirm=True)

    # Aura took a1 off FRAME but never got the request for KITCHEN.
    await server.sync_frame_photos_impl(FakeClient([asset("a2")]), tmp_path, FRAME)
    await server.sync_frame_photos_impl(FakeClient([asset("a1"), asset("k1")]), tmp_path, KITCHEN)

    assert ledger(tmp_path, FRAME)["a1"]["state"] == "removed"
    assert (tmp_path / "removed" / FRAME / "a1.jpg").read_bytes() == original("a1.jpeg")
    assert not (tmp_path / "cache" / FRAME / "a1.jpg").exists()
    assert "a1" not in ledger(tmp_path, KITCHEN)
    assert not (tmp_path / "removed" / KITCHEN / "a1.jpg").exists()
    assert (tmp_path / "cache" / KITCHEN / "a1.jpg").exists()


async def test_putting_back_never_makes_an_index_for_a_frame_without_one(tmp_path):
    await flagged_setup(tmp_path, "a1")
    await server.delete_assets_impl(FakeClient(), tmp_path, FRAME, ["a1"], confirm=True)
    (tmp_path / "cache" / FRAME / "index.json").unlink()

    result = await server.restore_assets_impl(FakeClient(), tmp_path, FRAME, ["a1"])

    assert result["restored_count"] == 1
    assert not (tmp_path / "cache" / FRAME / "index.json").exists()
    assert (tmp_path / "cache" / FRAME / "a1.jpg").read_bytes() == original("a1.jpeg")


async def test_putting_back_uses_the_ledgers_copy_when_it_has_another_name(tmp_path):
    await flagged_setup(tmp_path, "a1")
    await server.delete_assets_impl(FakeClient(), tmp_path, FRAME, ["a1"], confirm=True)
    kept = tmp_path / "removed" / FRAME
    other = kept / "a1.original-1234abcd.jpg"
    (kept / "a1.jpg").rename(other)  # as an earlier build of this branch could leave it
    log = ledger(tmp_path)
    log["a1"]["local_path"] = str(other)
    (kept / "removed.json").write_text(json.dumps(log))

    await server.restore_assets_impl(FakeClient(), tmp_path, FRAME, ["a1"])

    assert (tmp_path / "cache" / FRAME / "a1.jpg").read_bytes() == original("a1.jpeg")
    assert not other.exists()


async def test_a_removal_stops_between_photos_when_asked(tmp_path):
    await flagged_setup(tmp_path, "a1", "a2")
    client = FakeClient()
    asked = []

    def stop():
        asked.append(1)
        return len(asked) > 1  # let the first photo finish, then stop

    result = await server.delete_assets_impl(
        client, tmp_path, FRAME, ["a1", "a2"], confirm=True, should_stop=stop)

    assert client.frame_removals == [(FRAME, "a1")]
    assert result["deleted_count"] == 1
    assert [f["asset_id"] for f in result["failed"]] == ["a2"]
    assert "closed" in result["failed"][0]["reason"]

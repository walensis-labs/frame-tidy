import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from aura_mcp import people, people_index, server, storage
from aura_mcp.face_matcher import FaceMatcher


def unit(*values):
    v = np.zeros(512, dtype=np.float32)
    v[: len(values)] = values
    return v / np.linalg.norm(v)


MOM, DAD, KID = unit(1, 0.1), unit(0, 0, 1, 0.1), unit(0, 0, 0, 0, 1, 0.1)


def write_frame(root, frame_id, faces):
    """A synced, scanned frame: ``faces`` maps asset id to the faces in that photo."""
    folder = root / "cache" / frame_id
    folder.mkdir(parents=True, exist_ok=True)
    index = {}
    for asset_id in faces:
        (folder / f"{asset_id}.jpg").write_bytes(b"img")
        index[asset_id] = {"file_name": f"{asset_id}.heic", "taken_at": None,
                           "local_path": str(folder / f"{asset_id}.jpg")}
    (folder / "index.json").write_text(json.dumps(index))
    server._save_embeddings(
        folder / "embeddings.npz",
        {a: np.stack(rows) if rows else np.zeros((0, 512), np.float32) for a, rows in faces.items()},
    )


# ------------------------------------------------------ pooling and rebuilds


def test_pool_counts_a_photo_on_two_frames_once(tmp_path):
    write_frame(tmp_path, "kitchen", {"a": [MOM], "b": [MOM, DAD]})
    write_frame(tmp_path, "den", {"b": [MOM, DAD], "c": [DAD]})
    rows, refs = people_index.pool_faces(tmp_path)
    assert sorted(refs) == [("a", 0), ("b", 0), ("b", 1), ("c", 0)] and rows.shape == (4, 512)


def test_pool_skips_removed_photos_and_bad_rows(tmp_path):
    write_frame(tmp_path, "kitchen", {"a": [MOM], "gone": [DAD], "z": [np.zeros(512, np.float32), KID]})
    (tmp_path / "cache" / "kitchen" / "gone.jpg").unlink()
    assert sorted(people_index.pool_faces(tmp_path)[1]) == [("a", 0), ("z", 1)]


def test_rebuild_makes_people_and_keeps_curation(tmp_path):
    write_frame(tmp_path, "kitchen", {"a": [MOM], "b": [unit(1, 0.2), DAD], "c": [DAD]})
    first = people_index.rebuild(tmp_path)
    by_size = sorted(first["people"], key=lambda p: -len(p["faces"]))
    assert [len(p["faces"]) for p in by_size] == [2, 2] and first["built_at"]
    mom = next(p for p in first["people"] if ["a", 0] in p["faces"])
    mom["name"], mom["hidden"] = "Mom", True
    people_index.save(tmp_path, first)
    write_frame(tmp_path, "den", {"d": [unit(1, 0.15)], "e": [KID]})  # more of Mom, and someone new
    second = people_index.rebuild(tmp_path)
    again = next(p for p in second["people"] if p["id"] == mom["id"])
    assert (again["name"], again["hidden"], len(again["faces"])) == ("Mom", True, 3)
    assert len(second["people"]) == 3


def test_reconcile_keeps_two_clusters_one_person_merged():
    person = {"id": "p-aaaaaaaa", "name": "Mom", "hidden": False, "faces": [], "avatar": None,
              "anchors": [{"centroid": MOM.tolist(), "size": 5}, {"centroid": KID.tolist(), "size": 1}]}
    clusters = [{"centroid": MOM, "faces": [("a", 0)], "avatar": ("a", 0)},
                {"centroid": KID, "faces": [("k", 0)], "avatar": ("k", 0)},
                {"centroid": DAD, "faces": [("d", 0)], "avatar": ("d", 0)}]
    result = people_index.reconcile([person], clusters)
    mom = next(p for p in result if p["id"] == "p-aaaaaaaa")
    assert sorted(map(tuple, mom["faces"])) == [("a", 0), ("k", 0)] and len(mom["anchors"]) == 2
    assert len(result) == 2


def test_named_person_with_no_photos_left_survives_and_unnamed_does_not():
    named = {"id": "p-aaaaaaaa", "name": "Mom", "hidden": False, "faces": [["a", 0]], "avatar": ["a", 0],
             "anchors": [{"centroid": MOM.tolist(), "size": 1}]}
    unnamed = {**named, "id": "p-bbbbbbbb", "name": None, "anchors": [{"centroid": DAD.tolist(), "size": 1}]}
    result = people_index.reconcile([named, unnamed], [])
    assert [(p["id"], p["faces"], p["avatar"]) for p in result] == [("p-aaaaaaaa", [], None)]


def test_missing_or_corrupt_store_is_just_not_built(tmp_path):
    assert people_index.load(tmp_path) == {
        "version": 1, "built_at": None, "people": [], "not_same": [], "merges": []}
    (tmp_path / "people").mkdir()
    (tmp_path / "people" / "people.json").write_text("{not json")
    assert people_index.load(tmp_path)["people"] == []


def test_rebuild_with_no_faces_at_all(tmp_path):
    assert people_index.rebuild(tmp_path)["people"] == []


# ----------------------------------------------------- curation and gallery


@pytest.fixture
def built(tmp_path):
    write_frame(tmp_path, "kitchen", {"a": [MOM], "b": [unit(1, 0.2), DAD], "c": [DAD], "s": [KID]})
    write_frame(tmp_path, "den", {"b": [unit(1, 0.2), DAD], "d": [unit(1, 0.15)]})
    people_index.rebuild(tmp_path)
    return tmp_path


def who(root, asset, face=0):
    return next(p for p in people_index.load(root)["people"] if [asset, face] in p["faces"])


def test_gallery_counts_photos_once_and_per_frame(built):
    g = people_index.gallery(built)
    mom = next(p for p in g["people"] if p["id"] == who(built, "a")["id"])
    assert (mom["photo_count"], mom["frames"]) == (3, {"kitchen": 2, "den": 2})
    assert [p["photo_count"] for p in g["people"]] == [3, 2]
    assert [p["photo_count"] for p in g["seen_once"]] == [1] and g["hidden"] == []


def test_rename_hide_and_lookup_by_name(built):
    kid = who(built, "s")["id"]
    assert people_index.rename(built, kid, "  Sam  ")["name"] == "Sam"
    assert people_index.find(built, "sam")["id"] == kid and people_index.find(built, kid)["id"] == kid
    assert kid in [p["id"] for p in people_index.gallery(built)["people"]]  # named: gets a tile
    people_index.set_hidden(built, kid, True)
    assert [p["id"] for p in people_index.gallery(built)["hidden"]] == [kid]
    assert people_index.rename(built, kid, "   ")["name"] is None
    with pytest.raises(ValueError, match="No such person"):
        people_index.rename(built, "p-00000000", "x")


def test_merge_unions_faces_and_moves_answers(built):
    mom, kid = who(built, "a")["id"], who(built, "s")["id"]
    people.set_labels(built, kid, "kitchen", {"s": KID}, "no")
    merged = people_index.merge(built, mom, kid)
    assert merged["id"] == mom and ["s", 0] in merged["faces"] and len(merged["anchors"]) == 2
    assert people_index.find(built, kid) is None
    assert people.labels(built, mom, "kitchen") == {"s": "no"}
    people_index.rebuild(built)
    assert who(built, "s")["id"] == mom  # stays merged after a rebuild
    with pytest.raises(ValueError, match="itself"):
        people_index.merge(built, mom, mom)


def test_gallery_after_photos_are_removed(built):
    mom = who(built, "a")["id"]
    for frame in ("kitchen", "den"):
        for asset in ("a", "b", "d"):
            (built / "cache" / frame / f"{asset}.jpg").unlink(missing_ok=True)
    g = people_index.gallery(built)
    assert mom not in [p["id"] for p in g["people"] + g["seen_once"]]  # unnamed, 0 photos: listed nowhere
    people_index.rename(built, mom, "Mom")
    listed = next(p for p in people_index.gallery(built)["people"] if p["id"] == mom)
    assert (listed["photo_count"], listed["frames"]) == (0, {})  # named: still there


# ------------------------------------------------------------------ avatars


class TwoFaceApp:
    """Every photo holds DAD on the left and MOM on the right."""

    def __init__(self):
        self.calls = 0

    def get(self, image):
        self.calls += 1
        return [
            SimpleNamespace(bbox=np.array([80, 120, 240, 300]), normed_embedding=DAD),
            SimpleNamespace(bbox=np.array([480, 120, 640, 300]), normed_embedding=MOM),
        ]


@pytest.fixture
def portraits(tmp_path):
    write_frame(tmp_path, "kitchen", {"a": [DAD, MOM], "b": [DAD, MOM]})
    for asset in ("a", "b"):
        cv2.imwrite(str(tmp_path / "cache" / "kitchen" / f"{asset}.jpg"), np.full((600, 800, 3), 120, np.uint8))
    people_index.rebuild(tmp_path)
    return tmp_path


def test_avatar_is_cropped_once_and_none_when_the_photo_is_gone(portraits):
    app = TwoFaceApp()
    matcher = FaceMatcher(app=app)
    mom_id = who(portraits, "a", 1)["id"]
    path = people_index.ensure_avatar(matcher, portraits, mom_id)
    assert path == people_index.avatar_path(portraits, mom_id) and path.is_file()
    h, w = cv2.imread(str(path)).shape[:2]
    assert w < 800 and h < 600  # a crop, not the photo
    assert (w, h) == (272, 306)  # around the right-hand face, MOM's
    calls = app.calls
    people_index.ensure_avatar(matcher, portraits, mom_id)
    assert app.calls == calls  # cached: no second detection
    path.unlink()
    for asset in ("a", "b"):
        (portraits / "cache" / "kitchen" / f"{asset}.jpg").unlink()
    assert people_index.ensure_avatar(matcher, portraits, mom_id) is None
    assert people_index.ensure_avatar(matcher, portraits, "p-00000000") is None


def test_a_stale_avatar_is_dropped_on_rebuild_and_merge(portraits):
    mom, dad = who(portraits, "a", 1), who(portraits, "a", 0)
    for person in (mom, dad):
        path = people_index.avatar_path(portraits, person["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"old crop")
    (portraits / "cache" / "kitchen" / f"{mom['avatar'][0]}.jpg").unlink()  # the avatar's photo leaves
    people_index.rebuild(portraits)
    assert who(portraits, "b", 1)["avatar"] != mom["avatar"]
    assert not people_index.avatar_path(portraits, mom["id"]).exists()
    people_index.avatar_path(portraits, dad["id"]).write_bytes(b"old crop")
    people_index.merge(portraits, mom["id"], dad["id"])
    assert not people_index.avatar_path(portraits, dad["id"]).exists()  # that person is gone


# ------------------------------------------------- found in the branch review


async def test_not_them_on_one_frame_holds_on_every_frame(tmp_path):
    write_frame(tmp_path, "kitchen", {"a": [MOM], "b": [unit(1, 0.2)]})
    write_frame(tmp_path, "den", {"b": [unit(1, 0.2)], "d": [unit(1, 0.15)]})
    people_index.rebuild(tmp_path)
    mom = who(tmp_path, "a")["id"]
    matcher = FaceMatcher(app=object())  # every photo is already scanned
    await server.find_person_in_frame_impl(matcher, tmp_path, "kitchen", person=mom)
    people.set_labels(tmp_path, mom, "kitchen", {"b": unit(1, 0.2)}, "no")
    result = await server.find_person_in_frame_impl(matcher, tmp_path, "den", person=mom)
    assert [m["asset_id"] for m in result["matches"]] == ["d"]
    preview = await server.delete_assets_impl(None, tmp_path, "den", ["b", "d"])
    assert preview["assumed_count"] == 1
    assert preview["warnings"] == ["1 photo was marked as NOT Unnamed person in the review page."]


def _person(person_id, name, *anchors):
    return {"id": person_id, "name": name, "hidden": False, "faces": [], "avatar": None,
            "anchors": [{"centroid": vector.tolist(), "size": size} for vector, size in anchors]}


def _group(centre, *assets):
    return {"centroid": centre, "faces": [(a, 0) for a in assets], "avatar": (assets[0], 0)}


def test_an_anchor_claims_only_its_nearest_group():
    lookalike = unit(1, 0.1, 0.9)  # cosine 0.75 to MOM: over the re-bind bar, but someone else
    result = people_index.reconcile(
        [_person("p-aaaaaaaa", "Mom", (MOM, 5))], [_group(MOM, "a"), _group(lookalike, "s")])
    mom = next(p for p in result if p["id"] == "p-aaaaaaaa")
    assert mom["faces"] == [["a", 0]] and len(mom["anchors"]) == 1
    assert sorted(len(p["faces"]) for p in result) == [1, 1]


def test_a_named_person_keeps_their_group_over_an_unnamed_one():
    grandma = _person("p-aaaaaaaa", "Grandma", (unit(1, 0.5), 3))  # cosine 0.93 to the group
    unnamed = _person("p-bbbbbbbb", None, (MOM, 40))  # cosine 1.0 to it
    result = people_index.reconcile([grandma, unnamed], [_group(MOM, "a", "b")])
    assert [(p["id"], p["name"], len(p["faces"])) for p in result] == [("p-aaaaaaaa", "Grandma", 2)]


def test_an_anchor_with_nothing_left_to_claim_is_kept():
    ex = _person("p-aaaaaaaa", "Ex", (MOM, 5), (KID, 2))  # two groups merged by hand
    after_removal = people_index.reconcile([ex], [_group(MOM, "a")])
    assert len(after_removal[0]["anchors"]) == 2
    back = people_index.reconcile(after_removal, [_group(MOM, "a"), _group(KID, "k")])
    assert [(p["id"], sorted(map(tuple, p["faces"]))) for p in back] == [("p-aaaaaaaa", [("a", 0), ("k", 0)])]


def test_a_truncated_embeddings_file_is_treated_as_missing(tmp_path):
    write_frame(tmp_path, "kitchen", {"a": [MOM]})
    path = tmp_path / "cache" / "kitchen" / "embeddings.npz"
    path.write_bytes(path.read_bytes()[:60])
    assert server._load_embeddings(path) == {}
    assert people_index.pool_faces(tmp_path)[1] == []


def test_an_unreadable_store_is_kept_aside_before_a_rebuild(tmp_path):
    write_frame(tmp_path, "kitchen", {"a": [MOM]})
    (tmp_path / "people").mkdir()
    (tmp_path / "people" / "people.json").write_text("{not json")
    people_index.rebuild(tmp_path)
    assert (tmp_path / "people" / "people.json.bak").read_text() == "{not json"


def test_each_write_uses_its_own_temp_file(tmp_path, monkeypatch):
    seen, real = [], os.replace
    monkeypatch.setattr(storage.os, "replace", lambda a, b: (seen.append(Path(a).name), real(a, b)))
    storage.write_json(tmp_path / "x.json", {})
    storage.write_json(tmp_path / "x.json", {})
    server._save_embeddings(tmp_path / "embeddings.npz", {"a": MOM[None, :]})
    server._save_embeddings(tmp_path / "embeddings.npz", {"a": MOM[None, :]})
    assert len(set(seen)) == 4 and not list(tmp_path.glob("*.part*"))


def test_an_edit_during_a_rebuild_is_not_lost(built, monkeypatch):
    kid = who(built, "s")["id"]
    entered, release, real = threading.Event(), threading.Event(), people_index.reconcile

    def slow(prior, clusters):
        entered.set()
        release.wait(5)
        return real(prior, clusters)

    monkeypatch.setattr(people_index, "reconcile", slow)
    build = threading.Thread(target=people_index.rebuild, args=(built,))
    build.start()
    assert entered.wait(5)
    edit = threading.Thread(target=people_index.rename, args=(built, kid, "Sam"))
    edit.start()
    edit.join(0.3)  # the rename must wait for the rebuild, not be overwritten by it
    release.set()
    build.join(5)
    edit.join(5)
    assert people_index.find(built, kid)["name"] == "Sam"


# --------------------------------------- tidying: suggestions and merge undo

NEAR_MOM = unit(1, 0.1, 0, 0, 0, 0, 1.9)  # cosine 0.47 to MOM: maybe her, not sure
ALSO_NEAR = unit(1, 0.1, 0, 0, 0, 0, 0, 1.6)  # cosine 0.53


def _stored(person_id, name, centre, *assets, hidden=False):
    return {"id": person_id, "name": name, "hidden": hidden, "avatar": [assets[0], 0],
            "faces": [[a, 0] for a in assets], "anchors": [{"centroid": centre.tolist(), "size": len(assets)}]}


@pytest.fixture
def untidy(tmp_path):
    """Mom, two small groups that might be her, Dad, and a hidden look-alike."""
    write_frame(tmp_path, "kitchen", {a: [MOM] for a in "abcdefgh"})
    people_index.save(tmp_path, {**people_index.load(tmp_path), "built_at": "2026-11-19T00:00:00+00:00", "people": [
        _stored("p-mom00000", "Mom", MOM, "a", "b", "c"),
        _stored("p-near0000", None, NEAR_MOM, "d"),
        _stored("p-also0000", None, ALSO_NEAR, "e", "f"),
        _stored("p-dad00000", None, DAD, "g"),
        _stored("p-hidden00", None, ALSO_NEAR, "h", hidden=True),
    ]})
    return tmp_path


def test_gallery_says_how_many_photos_make_a_main_person(untidy):
    assert people_index.gallery(untidy)["main_min_photos"] == 10


def test_suggestions_are_the_nearest_other_groups_closest_first(untidy):
    found = people_index.suggestions(untidy, "p-mom00000")
    assert [(s["id"], s["photo_count"]) for s in found] == [("p-also0000", 2), ("p-near0000", 1)]
    assert found[0]["similarity"] == pytest.approx(0.53, abs=0.01)
    assert people_index.suggestions(untidy, "p-dad00000") == []
    with pytest.raises(ValueError, match="No such person"):
        people_index.suggestions(untidy, "p-00000000")


def test_a_no_is_remembered_and_follows_a_merge(untidy):
    people_index.not_same(untidy, "p-near0000", "p-mom00000")
    assert [s["id"] for s in people_index.suggestions(untidy, "p-mom00000")] == ["p-also0000"]
    assert people_index.suggestions(untidy, "p-near0000") == [
        s for s in people_index.suggestions(untidy, "p-near0000") if s["id"] != "p-mom00000"]
    people_index.merge(untidy, "p-also0000", "p-near0000")  # the rejected group joins another
    assert people_index.suggestions(untidy, "p-mom00000") == []  # still not suggested for Mom
    with pytest.raises(ValueError, match="No such person"):
        people_index.not_same(untidy, "p-mom00000", "p-00000000")


def test_merge_can_be_undone(untidy):
    people.set_labels(untidy, "p-near0000", "kitchen", {"d": NEAR_MOM}, "no")
    before = people_index.load(untidy)["people"]
    assert people_index.gallery(untidy)["can_undo"] is False
    people_index.merge(untidy, "p-mom00000", "p-near0000")
    assert people_index.gallery(untidy)["can_undo"] is True
    assert people.labels(untidy, "p-mom00000", "kitchen") == {"d": "no"}
    restored = people_index.unmerge(untidy)
    assert restored["id"] == "p-mom00000"
    by_id = lambda people_list: {p["id"]: p for p in people_list}
    assert by_id(people_index.load(untidy)["people"]) == by_id(before)
    assert people.labels(untidy, "p-near0000", "kitchen") == {"d": "no"}
    assert people.labels(untidy, "p-mom00000", "kitchen") == {}
    assert people_index.gallery(untidy)["can_undo"] is False
    with pytest.raises(ValueError, match="Nothing to undo"):
        people_index.unmerge(untidy)


def test_a_rebuild_ends_the_chance_to_undo(built):
    mom, kid = who(built, "a")["id"], who(built, "s")["id"]
    people_index.merge(built, mom, kid)
    people_index.rebuild(built)
    assert people_index.gallery(built)["can_undo"] is False


def test_a_no_survives_a_rebuild(built):
    mom, dad = who(built, "a")["id"], who(built, "c")["id"]
    people_index.not_same(built, mom, dad)
    people_index.rebuild(built)
    assert people_index.load(built)["not_same"] == [sorted([mom, dad])]


# ------------------------------- looking inside a group, taking only some of it


@pytest.fixture
def mixed(tmp_path):
    """One group that is mostly Mom with a look-alike in it, and Kid on their own."""
    write_frame(tmp_path, "kitchen", {"a": [MOM], "b": [MOM], "c": [NEAR_MOM], "k": [KID], "j": [KID]})
    write_frame(tmp_path, "den", {"b": [MOM]})
    people_index.rebuild(tmp_path)
    group, kid = who(tmp_path, "a")["id"], who(tmp_path, "k")["id"]
    assert who(tmp_path, "c")["id"] == group  # the look-alike was grouped in
    people_index.rename(tmp_path, kid, "Kid")
    return tmp_path, group, kid


def test_group_faces_lists_each_face_with_a_frame_most_typical_first(mixed):
    root, group, _ = mixed
    faces = people_index.group_faces(root, group)
    assert [(f["asset_id"], f["index"]) for f in faces][-1] == ("c", 0)  # the odd one out last
    assert {f["asset_id"]: f["frame_id"] for f in faces} == {"a": "kitchen", "b": "den", "c": "kitchen"}
    assert faces[0]["likeness"] >= faces[-1]["likeness"]
    with pytest.raises(ValueError, match="No such person"):
        people_index.group_faces(root, "p-00000000")


def test_taking_some_of_a_group_leaves_the_rest_as_a_different_person(mixed):
    root, group, kid = mixed
    people_index.move_faces(root, kid, group, [["a", 0], ["b", 0]])
    assert sorted(map(tuple, people_index.find(root, kid)["faces"])) == [
        ("a", 0), ("b", 0), ("j", 0), ("k", 0)]
    rest = people_index.find(root, group)
    assert rest["faces"] == [["c", 0]] and rest["avatar"] == ["c", 0]
    assert sorted([kid, group]) in people_index.load(root)["not_same"]  # not offered again
    assert people_index.suggestions(root, kid) == []


def test_faces_taken_from_a_group_stay_put_after_a_rebuild(mixed):
    root, group, kid = mixed
    people_index.move_faces(root, kid, group, [["c", 0]])
    write_frame(root, "hall", {"n": [KID]})  # a sync adds a photo, and everyone is regrouped
    people_index.rebuild(root)
    assert who(root, "c")["id"] == kid
    assert who(root, "a")["id"] == group and who(root, "b")["id"] == group
    assert who(root, "n")["id"] == kid


def test_taking_some_of_a_group_can_be_undone(mixed):
    root, group, kid = mixed
    before = {p["id"]: p for p in people_index.load(root)["people"]}
    people_index.move_faces(root, kid, group, [["a", 0]])
    people_index.unmerge(root)
    assert {p["id"]: p for p in people_index.load(root)["people"]} == before
    assert people_index.load(root)["not_same"] == []


def test_taking_all_of_a_group_is_a_merge(mixed):
    root, group, kid = mixed
    people_index.move_faces(root, kid, group, [["a", 0], ["b", 0], ["c", 0]])
    assert people_index.find(root, group) is None and who(root, "c")["id"] == kid


def test_taking_faces_that_are_not_in_the_group_is_refused(mixed):
    root, group, kid = mixed
    with pytest.raises(ValueError, match="Choose at least one"):
        people_index.move_faces(root, kid, group, [])
    with pytest.raises(ValueError, match="not in that group"):
        people_index.move_faces(root, kid, group, [["k", 0]])
    with pytest.raises(ValueError, match="itself"):
        people_index.move_faces(root, kid, kid, [["k", 0]])


def test_a_face_is_found_again_in_its_photo_and_cropped_once(portraits):
    app = TwoFaceApp()
    matcher = FaceMatcher(app=app)
    frame_id, box = people_index.face_box(matcher, portraits, "a", 1)  # face 1 is MOM, on the right
    assert frame_id == "kitchen" and box["x"] == pytest.approx(0.6)
    path = people_index.ensure_face_crop(matcher, portraits, "a", 1)
    assert path.is_file() and cv2.imread(str(path)).shape[:2] == (306, 272)
    calls = app.calls
    assert people_index.ensure_face_crop(matcher, portraits, "a", 1) == path and app.calls == calls
    assert people_index.face_box(matcher, portraits, "a", 7) is None
    assert people_index.ensure_face_crop(matcher, portraits, "zz", 0) is None

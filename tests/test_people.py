import json

import numpy as np
import pytest

from aura_mcp import people, people_index, server
from aura_mcp.face_matcher import FaceMatcher, ImageReadError

FRAME = "frame-1"


def unit(*values):
    v = np.zeros(512, dtype=np.float32)
    v[: len(values)] = values
    return v / np.linalg.norm(v)


# Two brothers who look alike: their faces share most of one direction.
ME = unit(1, 0.25, 0)
BROTHER = unit(1, -0.25, 0.9)
STRANGER = unit(0, 0, 0, 1)


def me_variant(amount):
    """Me at another angle/age: drifts away from the reference photo."""
    return unit(1, 0.25, 0, 0, amount)


class Matcher(FaceMatcher):
    def __init__(self, faces, reference=ME):
        self.faces, self.reference = faces, reference

    def reference_embedding(self, image_path):
        return self.reference, 1

    def face_embeddings(self, image_path):
        from pathlib import Path

        stem = Path(image_path).stem
        if stem not in self.faces:
            raise ImageReadError(stem)
        rows = self.faces[stem]
        return np.stack(rows) if rows else np.zeros((0, 512), dtype=np.float32)


def make_frame(root, asset_ids):
    folder = root / "cache" / FRAME
    folder.mkdir(parents=True)
    index = {}
    for asset_id in asset_ids:
        (folder / f"{asset_id}.jpg").write_bytes(b"img")
        index[asset_id] = {"file_name": f"{asset_id}.heic", "taken_at": None,
                           "local_path": str(folder / f"{asset_id}.jpg")}
    (folder / "index.json").write_text(json.dumps(index))


def save_reference(root, name="Sam"):
    refs = root / "references"
    refs.mkdir(exist_ok=True)
    path = refs / f"{people.slug(name)}-1.jpg"
    path.write_bytes(b"crop")
    existing = json.loads((refs / "references.json").read_text()) if (refs / "references.json").exists() else []
    existing.append({"id": path.stem, "name": name, "path": str(path)})
    (refs / "references.json").write_text(json.dumps(existing))
    return path


async def search(root, matcher, threshold=0.45, **who):
    return await server.find_person_in_frame_impl(matcher, root, FRAME, similarity_threshold=threshold, **who)


def flagged_ids(result):
    return [m["asset_id"] for m in result["matches"]]


# ---------------------------------------------------------------- the model


def test_single_reference_scores_like_plain_cosine_similarity():
    model = people.PersonModel(positives=ME[None, :])
    score, index = model.best(np.stack([STRANGER, ME * 3]))  # unnormalised input is fine
    assert index == 1 and score == pytest.approx(1.0)
    assert model.best(np.zeros((0, 512))) is None


def test_confirmed_faces_raise_the_score_of_other_photos_of_the_person():
    hard = me_variant(1.6)
    alone = people.PersonModel(positives=ME[None, :])
    learned = people.PersonModel(positives=np.stack([ME, me_variant(1.2), me_variant(1.4)]))
    assert learned.best(hard[None, :])[0] > alone.best(hard[None, :])[0] + 0.1


def test_a_rejected_lookalike_is_ruled_out_but_the_person_is_not():
    model = people.PersonModel(positives=ME[None, :], negatives=BROTHER[None, :])
    brother_again = unit(1, -0.2, 0.85)
    assert float(model.similarities(brother_again[None, :])[0]) > 0.6  # he does look like me
    assert model.best(brother_again[None, :]) is None  # ...but more like himself
    assert model.best(ME[None, :])[0] == pytest.approx(1.0)
    # a photo of both of us still matches, on my face
    score, index = model.best(np.stack([brother_again, ME]))
    assert index == 1


# ------------------------------------------------------------- saved people


def test_labels_are_stored_per_person_and_can_be_cleared(tmp_path):
    counts = people.set_labels(tmp_path, "Sam", FRAME, {"a1": ME, "a2": BROTHER}, "yes")
    assert counts == {"confirmed": 2, "rejected": 0}
    people.set_labels(tmp_path, "Sam", FRAME, {"a2": BROTHER}, "no")
    people.set_labels(tmp_path, "Someone Else", FRAME, {"a1": STRANGER}, "yes")

    assert people.labels(tmp_path, "sam", FRAME) == {"a1": "yes", "a2": "no"}
    assert people.labels(tmp_path, "Sam", "other-frame") == {}

    assert people.set_labels(tmp_path, "Sam", FRAME, {"a1": None}, None) == {"confirmed": 0, "rejected": 1}
    with pytest.raises(ValueError, match="No face is known"):
        people.set_labels(tmp_path, "Sam", FRAME, {"a9": None}, "yes")
    with pytest.raises(ValueError, match="label must be"):
        people.set_labels(tmp_path, "Sam", FRAME, {"a1": ME}, "maybe")


def test_build_model_for_a_saved_person_uses_their_feedback(tmp_path):
    path = save_reference(tmp_path, "Sam")
    people.set_labels(tmp_path, "Sam", FRAME, {"a1": me_variant(1.0)}, "yes")
    people.set_labels(tmp_path, "Sam", FRAME, {"a2": BROTHER}, "no")
    matcher = Matcher({})

    by_name = people.build_model(matcher, tmp_path, person="sam")
    by_path = people.build_model(matcher, tmp_path, reference_image_path=str(path))

    for model in (by_name, by_path):
        assert model.name == "Sam" and model.slug == "sam"
        assert len(model.positives) == 2 and len(model.negatives) == 1
        assert model.confirmed == {f"{FRAME}/a1"} and model.rejected == {f"{FRAME}/a2"}
    assert people.summary(tmp_path) == [
        {"person": "sam", "name": "Sam", "references": 1, "confirmed": 1, "rejected": 1}
    ]


def test_build_model_one_off_reference_and_errors(tmp_path):
    model = people.build_model(Matcher({}), tmp_path, reference_image_path="/somewhere/photo.jpg")
    assert model.slug is None and len(model.positives) == 1

    with pytest.raises(ValueError, match="reference_image_path or the name"):
        people.build_model(Matcher({}), tmp_path)
    save_reference(tmp_path, "Sam")
    with pytest.raises(ValueError, match="No saved reference for 'Robin'. Saved people: Sam"):
        people.build_model(Matcher({}), tmp_path, person="Robin")


# ------------------------------------------------------- search with feedback


async def test_feedback_fixes_a_false_positive_and_a_false_negative(tmp_path):
    """The situation from the real frame: one reference can't split two brothers."""
    make_frame(tmp_path, ["me-easy", "me-hard", "brother-1", "brother-2", "stranger"])
    save_reference(tmp_path, "Sam")
    matcher = Matcher({
        "me-easy": [me_variant(0.5)],
        "me-hard": [me_variant(1.6)],          # scores lower than my brother does
        "brother-1": [BROTHER],
        "brother-2": [unit(1, -0.2, 0.85)],
        "stranger": [STRANGER],
    })

    first = await search(tmp_path, matcher, threshold=0.6, person="Sam")
    assert flagged_ids(first) == ["me-easy", "brother-2", "brother-1"]  # brother in, hard photo out

    # The user answers: easy one is me, brother-1 is not.
    by_id = {m["asset_id"]: m for m in first["matches"]}
    embeddings = server._load_embeddings(tmp_path / "cache" / FRAME / "embeddings.npz")
    face = lambda a: embeddings[a][by_id[a]["face_index"]]
    people.set_labels(tmp_path, "Sam", FRAME, {"me-easy": face("me-easy")}, "yes")
    people.set_labels(tmp_path, "Sam", FRAME, {"brother-1": face("brother-1")}, "no")

    second = await search(tmp_path, matcher, threshold=0.6, person="Sam")
    # both brother photos gone (one by name, one because it looks like the rejected face)
    assert "brother-1" not in flagged_ids(second) and "brother-2" not in flagged_ids(second)
    assert second["examples_used"] == {"is_them": 2, "not_them": 1}
    assert second["person"] == "Sam"

    lower = await search(tmp_path, matcher, threshold=0.5, person="Sam")
    assert set(flagged_ids(lower)) == {"me-easy", "me-hard"}  # recall up, brother still out


async def test_confirmed_photo_stays_flagged_and_rejected_photo_stays_out(tmp_path):
    make_frame(tmp_path, ["weak", "strong"])
    save_reference(tmp_path, "Sam")
    matcher = Matcher({"weak": [me_variant(3.0)], "strong": [ME]})
    people.set_labels(tmp_path, "Sam", FRAME, {"strong": ME}, "no")
    people.set_labels(tmp_path, "Sam", FRAME, {"weak": STRANGER}, "yes")  # even an odd face

    result = await search(tmp_path, matcher, threshold=0.9, person="Sam")

    assert flagged_ids(result) == ["weak"]
    assert result["matches"][0]["confirmed"] is True

    saved = json.loads((tmp_path / "flagged" / FRAME / "matches.json").read_text())
    assert saved["person"] == "sam" and saved["person_name"] == "Sam"
    assert saved["similarity_threshold"] == 0.9
    assert [m["asset_id"] for m in saved["matches"]] == ["weak"]


async def test_removal_preview_warns_about_anything_not_confirmed(tmp_path):
    make_frame(tmp_path, ["yes-1", "yes-2", "unanswered", "brother", "scenery"])
    save_reference(tmp_path, "Sam")
    matcher = Matcher({
        "yes-1": [ME], "yes-2": [me_variant(0.5)], "unanswered": [me_variant(0.8)],
        "brother": [BROTHER], "scenery": [],
    })
    await search(tmp_path, matcher, threshold=0.5, person="Sam")
    people.set_labels(tmp_path, "Sam", FRAME, {"yes-1": ME, "yes-2": ME}, "yes")
    people.set_labels(tmp_path, "Sam", FRAME, {"brother": BROTHER}, "no")

    class NeverCalled:
        def __getattr__(self, name):
            raise AssertionError("a preview must not contact Aura")

    clean = await server.delete_assets_impl(NeverCalled(), tmp_path, FRAME, ["yes-1", "yes-2"])
    assert clean["warnings"] == [] and clean["confirmed_count"] == 2
    assert clean["reviewed_for"] == "Sam"
    assert [d["review"] for d in clean["would_delete"]] == ["confirmed", "confirmed"]

    mixed = await server.delete_assets_impl(
        NeverCalled(), tmp_path, FRAME, ["yes-1", "unanswered", "brother", "scenery"]
    )
    assert mixed["confirmed_count"] == 1
    assert [d["review"] for d in mixed["would_delete"]] == [
        "confirmed", "unanswered", "not_them", "not_matched",
    ]
    assert mixed["warnings"] == [
        "1 photo was marked as NOT Sam in the review page.",
        "1 photo was matched as Sam but never confirmed in the review page.",
        "1 photo was not matched as Sam by the search (selected by hand).",
    ]
    assert "read them to the user" in mixed["message"]


async def test_removal_preview_has_no_review_fields_without_a_person_search(tmp_path):
    make_frame(tmp_path, ["a1"])
    preview = await server.delete_assets_impl(None, tmp_path, FRAME, ["a1"])
    assert "warnings" not in preview and "review" not in preview["would_delete"][0]


async def test_search_needs_someone_to_look_for(tmp_path):
    make_frame(tmp_path, ["a1"])
    with pytest.raises(ValueError, match="reference_image_path or the name"):
        await search(tmp_path, Matcher({"a1": [ME]}))


# ------------------------------------------- people found by grouping faces


async def test_search_for_a_clustered_person_keeps_members_and_is_opt_out(tmp_path):
    side = unit(0.3, 1, 0, 0, 0, 1)  # in profile: cosine ~0.37 to ME, its own cluster
    faces = {"a": [ME], "b": [me_variant(0.3)], "c": [side], "x": [STRANGER]}
    make_frame(tmp_path, list(faces))
    matcher = Matcher(faces)
    built = await server.find_people_impl(matcher, tmp_path)
    assert built["frames_scanned"] == 1 and (built["people"], built["seen_once"]) == (1, 2)
    store = people_index.load(tmp_path)
    mom = next(p for p in store["people"] if ["a", 0] in p["faces"])
    profile = next(p for p in store["people"] if ["c", 0] in p["faces"])
    assert profile["id"] != mom["id"]
    people_index.merge(tmp_path, mom["id"], profile["id"])
    # At a cut-off nothing reaches, the photos clustering gave this person are still flagged.
    result = await search(tmp_path, matcher, threshold=0.999, person=mom["id"])
    assert set(flagged_ids(result)) == {"a", "b", "c"}
    assert result["review_mode"] == "opt_out" and result["person"] == "Unnamed person"
    saved = json.loads((tmp_path / "flagged" / FRAME / "matches.json").read_text())
    assert (saved["person"], saved["review_mode"]) == (mom["id"], "opt_out")


async def test_not_them_drops_a_member_and_survives_rename(tmp_path):
    faces = {"a": [ME], "b": [me_variant(0.3)], "x": [STRANGER]}
    make_frame(tmp_path, list(faces))
    matcher = Matcher(faces)
    await server.find_people_impl(matcher, tmp_path)
    mom = people_index.find(tmp_path, next(p["id"] for p in people_index.gallery(tmp_path)["people"]))
    people.set_labels(tmp_path, mom["id"], FRAME, {"b": me_variant(0.3)}, "no")
    people_index.rename(tmp_path, mom["id"], "Mom")
    result = await search(tmp_path, matcher, person="mom")
    assert flagged_ids(result) == ["a"] and result["person"] == "Mom"


async def test_reference_person_search_is_still_opt_in(tmp_path):
    make_frame(tmp_path, ["a"])
    save_reference(tmp_path)
    assert (await search(tmp_path, Matcher({"a": [ME]}), person="Sam"))["review_mode"] == "opt_in"


async def test_opt_out_preview_assumes_unanswered_matches(tmp_path):
    faces = {"a": [ME], "b": [me_variant(0.3)], "x": [STRANGER]}
    make_frame(tmp_path, list(faces))
    matcher = Matcher(faces)
    await server.find_people_impl(matcher, tmp_path)
    mom = people_index.gallery(tmp_path)["people"][0]["id"]
    await search(tmp_path, matcher, person=mom)
    people.set_labels(tmp_path, mom, FRAME, {"b": me_variant(0.3)}, "no")
    preview = await server.delete_assets_impl(None, tmp_path, FRAME, ["a", "b", "x"])
    assert (preview["review_mode"], preview["assumed_count"]) == ("opt_out", 1)
    assert preview["warnings"] == [
        "1 photo was marked as NOT Unnamed person in the review page.",
        "1 photo was not matched as Unnamed person by the search (selected by hand).",
    ]

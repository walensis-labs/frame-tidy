"""Everyone found in the cached photos, across every frame on the account.

The face embeddings each frame's scan already cached are pooled and grouped
into people (see face_clusters). What is known about each person is kept in

    people/people.json     ids, names, hidden flags, which faces are theirs
    people/avatars/        one face crop per person, made on demand

A photo is identified by its Aura asset id alone: the same photo has the same
id on every frame it is on, so it is pooled and counted once.

Rebuilding never renames or re-identifies anyone. Each person keeps the
centre of every group of faces that makes them up (an "anchor"); after a fresh
grouping, a person claims the new groups nearest their anchors. That is also
what keeps two groups merged by hand together, and what lets a named person
outlive their photos being removed.
"""

from __future__ import annotations

import copy
import os
import shutil
import threading
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import people
from .face_matcher import FaceMatcher, FaceMatcherError
from .face_clusters import CLUSTER_THRESHOLD, cluster_faces
from .storage import read_json, write_json

TILE_MIN_PHOTOS = 2
# From this many photos a person is shown up front; smaller groups, which are
# mostly fragments of someone bigger, are tucked away until asked for.
MAIN_MIN_PHOTOS = 10
REBIND_MIN = 0.60
# Two people whose faces are at least this alike are offered as "the same
# person?". Not sure enough to merge unasked: close relatives land here too.
SUGGEST_MIN = 0.40
SUGGEST_LIMIT = 24
UNDO_LIMIT = 10
VERSION = 1

FaceRef = tuple[str, int]  # (asset_id, index of the face within that photo)

# Held across every read-change-write of the store, so a rename, hide or merge
# made while a rebuild is finishing is applied after it instead of being lost.
_STORE_LOCK = threading.RLock()


def people_dir(root: Path) -> Path:
    return root / "people"


def _store_path(root: Path) -> Path:
    return people_dir(root) / "people.json"


def _read_store(root: Path) -> dict[str, Any] | None:
    data = read_json(_store_path(root), None)
    if not isinstance(data, dict) or not isinstance(data.get("people"), list):
        return None
    return data


def load(root: Path) -> dict[str, Any]:
    """The saved people, or an empty not-yet-built store."""
    data = _read_store(root)
    if data is None:
        return {"version": VERSION, "built_at": None, "people": [], "not_same": [], "merges": []}
    return {
        "version": VERSION,
        "built_at": data.get("built_at"),
        "people": data["people"],
        # Pairs of ids the user said are different people: never suggested again.
        "not_same": data.get("not_same") or [],
        # What each recent merge replaced, newest last, so it can be undone.
        "merges": data.get("merges") or [],
    }


def save(root: Path, data: dict[str, Any]) -> None:
    write_json(_store_path(root), data, indent=None)


def new_id() -> str:
    return f"p-{uuid.uuid4().hex[:8]}"


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def _frame_dirs(root: Path) -> list[Path]:
    cache = root / "cache"
    return sorted(p for p in cache.iterdir() if p.is_dir()) if cache.is_dir() else []


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    try:
        with np.load(path) as data:
            return {key: data[key] for key in data.files}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile):
        return {}


def pool_faces(root: Path) -> tuple[np.ndarray, list[FaceRef]]:
    """Every usable face in the cached photos: unit rows and what each row is."""
    rows: list[np.ndarray] = []
    refs: list[FaceRef] = []
    seen: set[str] = set()
    for folder in _frame_dirs(root):
        for asset_id, faces in _load_npz(folder / "embeddings.npz").items():
            if asset_id in seen or not (folder / f"{asset_id}.jpg").is_file():
                continue
            seen.add(asset_id)
            if faces.ndim != 2:
                continue
            for index, face in enumerate(np.asarray(faces, dtype=np.float32)):
                norm = float(np.linalg.norm(face))
                if not np.isfinite(norm) or norm == 0.0:
                    continue
                rows.append(face / norm)
                refs.append((asset_id, index))
    stacked = np.stack(rows) if rows else np.zeros((0, 512), dtype=np.float32)
    return stacked, refs


def make_clusters(
    embeddings: np.ndarray, refs: list[FaceRef], labels: np.ndarray
) -> list[dict[str, Any]]:
    """Each group of faces with its centre and its most typical face."""
    members: dict[int, list[int]] = {}
    for row, label in enumerate(labels):
        members.setdefault(int(label), []).append(row)
    clusters = []
    for rows in members.values():
        centre = _unit(embeddings[rows].mean(axis=0))
        typical = rows[int(np.argmax(embeddings[rows] @ centre))]
        clusters.append(
            {"centroid": centre, "faces": [refs[r] for r in rows], "avatar": refs[typical]}
        )
    return clusters


def _anchor_vectors(person: dict[str, Any]) -> list[np.ndarray]:
    return [
        _unit(np.asarray(a["centroid"], dtype=np.float32)) for a in person.get("anchors", [])
    ]


def centroid(person: dict[str, Any]) -> np.ndarray:
    """The centre of a person's faces: their anchors, weighted by size."""
    vectors = _anchor_vectors(person)
    if not vectors:
        return np.zeros(512, dtype=np.float32)
    sizes = [max(1, int(a.get("size", 1))) for a in person["anchors"]]
    return _unit(sum(v * s for v, s in zip(vectors, sizes)))


def _with_clusters(
    person: dict[str, Any],
    claimed: list[dict[str, Any]],
    kept_anchors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    claimed = sorted(claimed, key=lambda c: len(c["faces"]), reverse=True)
    faces = list(dict.fromkeys((str(a), int(i)) for c in claimed for a, i in c["faces"]))
    return {
        "id": person["id"],
        "name": person.get("name"),
        "hidden": bool(person.get("hidden")),
        "anchors": [
            {
                "centroid": [round(float(v), 4) for v in c["centroid"]],
                "size": len(c["faces"]),
            }
            for c in claimed
        ] + list(kept_anchors or []),
        "faces": [[a, i] for a, i in faces],
        "avatar": [str(claimed[0]["avatar"][0]), int(claimed[0]["avatar"][1])],
        **({"pinned": person["pinned"]} if person.get("pinned") else {}),
    }


def reconcile(prior: list[dict[str, Any]], clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Carry the people already known over to a fresh grouping of the faces.

    Each anchor bids for the one fresh group nearest to it, and only if that
    group is close enough. It never takes a second group: a look-alike that is
    merely over the bar stays a separate, unnamed person for the user to merge
    or not. Where two people bid for the same group, one the user named or hid
    wins over one they never touched; otherwise the nearer wins.
    """
    owners = [
        (p, a, vector)
        for p, person in enumerate(prior)
        for a, vector in enumerate(_anchor_vectors(person))
    ]
    claims: dict[int, list[dict[str, Any]]] = {}
    owner_of: dict[int, int] = {}  # fresh group -> the person who has it
    settled: set[int] = set()  # anchors now represented by a claimed group

    def curated(p: int) -> bool:
        return bool(prior[p].get("name") or prior[p].get("hidden"))

    if owners and clusters:
        centres = np.stack([np.asarray(c["centroid"], dtype=np.float32) for c in clusters])
        scores = np.stack([vector for _, _, vector in owners]) @ centres.T
        nearest = scores.argmax(axis=1)
        bids = [
            (o, int(nearest[o]), float(scores[o, nearest[o]]))
            for o in range(len(owners))
            if scores[o, nearest[o]] >= REBIND_MIN
        ]
        bids.sort(key=lambda bid: (not curated(owners[bid[0]][0]), -bid[2]))
        for o, cluster, _ in bids:
            p = owners[o][0]
            if cluster not in owner_of:
                owner_of[cluster] = p
                claims.setdefault(p, []).append(clusters[cluster])
                settled.add(o)
            elif owner_of[cluster] == p:
                settled.add(o)  # two of their anchors now fall in the one group

    people = []
    for p, person in enumerate(prior):
        if p in claims:
            # An anchor with nothing to claim (its photos were removed) is
            # kept: it is how photos like those are recognised if they return.
            unclaimed = [
                person["anchors"][a]
                for o, (owner, a, _) in enumerate(owners)
                if owner == p and o not in settled
            ]
            people.append(_with_clusters(person, claims[p], unclaimed))
        elif curated(p) or person.get("pinned"):
            # Nothing of theirs is left in the cache, but who they are is kept.
            # (Someone with faces the user placed by hand gets them back below.)
            people.append({**person, "faces": [], "avatar": None})
    for c, cluster in enumerate(clusters):
        if c not in owner_of:
            people.append(_with_clusters({"id": new_id(), "name": None, "hidden": False}, [cluster]))
    return people


def rebuild(root: Path, threshold: float = CLUSTER_THRESHOLD) -> dict[str, Any]:
    """Group every cached face into people and save the result."""
    embeddings, refs = pool_faces(root)
    clusters = make_clusters(embeddings, refs, cluster_faces(embeddings, threshold))
    with _STORE_LOCK:
        path = _store_path(root)
        if path.is_file() and _read_store(root) is None:
            # Unreadable, so its names, hides and merges cannot be carried
            # over. Keep the file rather than write over the only copy.
            shutil.copy2(path, path.with_name(path.name + ".bak"))
        before = load(root)
        prior = before["people"]
        everyone = apply_pins(reconcile(prior, clusters), set(refs))
        known = {person["id"] for person in everyone}
        data = {
            "version": VERSION,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "people": everyone,
            "not_same": [pair for pair in before["not_same"] if set(pair) <= known],
            # The faces have been regrouped, so an earlier merge can no longer
            # be put back the way it was.
            "merges": [],
        }
        save(root, data)
    # A saved crop shows the face a person had before; drop it if that changed.
    now = {person["id"]: person.get("avatar") for person in data["people"]}
    for person in prior:
        if now.get(person["id"]) != person.get("avatar"):
            avatar_path(root, person["id"]).unlink(missing_ok=True)
    return data


# ------------------------------------------------------------------ curation


def find(root: Path, key: str) -> dict[str, Any] | None:
    """A person by id or, failing that, by the name they were given."""
    everyone = load(root)["people"]
    for person in everyone:
        if person["id"] == key:
            return person
    wanted = people.slug(key)
    for person in everyone:
        if person.get("name") and people.slug(person["name"]) == wanted:
            return person
    return None


def _edit(root: Path, person_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    data = load(root)
    for person in data["people"]:
        if person["id"] == person_id:
            return data, person
    raise ValueError(f"No such person: {person_id}")


def rename(root: Path, person_id: str, name: str) -> dict[str, Any]:
    """Give a person a name; an empty name makes them unnamed again."""
    with _STORE_LOCK:
        data, person = _edit(root, person_id)
        person["name"] = str(name or "").strip()[:60] or None
        save(root, data)
    return person


def set_hidden(root: Path, person_id: str, hidden: bool) -> dict[str, Any]:
    with _STORE_LOCK:
        data, person = _edit(root, person_id)
        person["hidden"] = bool(hidden)
        save(root, data)
    return person


def merge(root: Path, target_id: str, other_id: str) -> dict[str, Any]:
    """Two people are the same person: fold ``other`` into ``target``."""
    if target_id == other_id:
        raise ValueError("A person cannot be merged with itself.")
    with _STORE_LOCK:
        data, target = _edit(root, target_id)
        other = next((p for p in data["people"] if p["id"] == other_id), None)
        if other is None:
            raise ValueError(f"No such person: {other_id}")
        undo = {
            "target": copy.deepcopy(target),
            "other": copy.deepcopy(other),
            "not_same": copy.deepcopy(data["not_same"]),
        }
        target["name"] = target.get("name") or other.get("name")
        target["faces"] = [list(f) for f in dict.fromkeys(
            (a, i) for a, i in [*target.get("faces", []), *other.get("faces", [])]
        )]
        target["anchors"] = [*target.get("anchors", []), *other.get("anchors", [])]
        pins = _pins(target) + [f for f in _pins(other) if f not in _pins(target)]
        if pins:
            target["pinned"] = [list(f) for f in pins]
        if not target.get("avatar"):
            target["avatar"] = other.get("avatar")
            avatar_path(root, target_id).unlink(missing_ok=True)
        data["people"] = [p for p in data["people"] if p["id"] != other_id]
        # A "not the same as X" said about the absorbed person now holds for
        # the merged one.
        pairs = {
            tuple(sorted(target_id if pid == other_id else pid for pid in pair))
            for pair in data["not_same"]
        }
        data["not_same"] = sorted(list(pair) for pair in pairs if pair[0] != pair[1])
        # What was learned about one of them is now known about the merged person.
        undo["feedback_keys"] = people.move_feedback(root, other_id, target_id)
        data["merges"] = [*data["merges"], undo][-UNDO_LIMIT:]
        save(root, data)
    avatar_path(root, other_id).unlink(missing_ok=True)
    # A review of the absorbed person may be open on some frame. Point it at
    # the merged person, or its next answer would be filed under a dead id.
    for path in sorted((root / "flagged").glob("*/matches.json")):
        last = read_json(path, None)
        if isinstance(last, dict) and last.get("person") == other_id:
            last["person"] = target_id
            last["person_name"] = target.get("name") or people.UNNAMED
            write_json(path, last)
    return target


def unmerge(root: Path) -> dict[str, Any]:
    """Undo the most recent merge: both people come back as they were."""
    with _STORE_LOCK:
        data = load(root)
        if not data["merges"]:
            raise ValueError("Nothing to undo.")
        undo = data["merges"].pop()
        target, other = undo["target"], undo["other"]
        data["people"] = [
            target if person["id"] == target["id"] else person
            for person in data["people"]
            if person["id"] != other["id"]
        ] + [other]
        data["not_same"] = undo.get("not_same") or []
        people.move_feedback(root, target["id"], other["id"], only=undo.get("feedback_keys") or [])
        save(root, data)
    # The crops may show the merged face; they are made again when asked for.
    avatar_path(root, target["id"]).unlink(missing_ok=True)
    return target


def not_same(root: Path, person_id: str, other_id: str) -> None:
    """Remember that two people are different, so they are not suggested again."""
    with _STORE_LOCK:
        data, _ = _edit(root, person_id)
        if not any(person["id"] == other_id for person in data["people"]):
            raise ValueError(f"No such person: {other_id}")
        pair = sorted([person_id, other_id])
        if pair[0] != pair[1] and pair not in data["not_same"]:
            data["not_same"].append(pair)
            save(root, data)


def _pins(person: dict[str, Any]) -> list[FaceRef]:
    return [(str(a), int(i)) for a, i in person.get("pinned", [])]


def apply_pins(everyone: list[dict[str, Any]], present: set[FaceRef]) -> list[dict[str, Any]]:
    """Put each face the user placed by hand with the person they placed it with.

    A fresh grouping does not know that a group was split: the faces taken
    out of it would land back in it. Pins win over the grouping.
    """
    pinned_to = {face: person["id"] for person in everyone for face in _pins(person)}
    if not pinned_to:
        return everyone
    result = []
    for person in everyone:
        faces = [
            (str(a), int(i)) for a, i in person.get("faces", [])
            if pinned_to.get((str(a), int(i)), person["id"]) == person["id"]
        ]
        faces += [f for f in _pins(person) if f in present and f not in faces]
        avatar = tuple(person["avatar"]) if person.get("avatar") else None
        person = {
            **person,
            "faces": [list(f) for f in faces],
            "avatar": list(avatar) if avatar in faces else (list(faces[0]) if faces else None),
        }
        if faces or person.get("name") or person.get("hidden"):
            result.append(person)
    return result


def _face_rows(root: Path) -> tuple[np.ndarray, dict[FaceRef, int]]:
    rows, refs = pool_faces(root)
    return rows, {ref: r for r, ref in enumerate(refs)}


def group_faces(root: Path, person_id: str) -> list[dict[str, Any]]:
    """Every face in a person's group that is still in a frame, most typical first.

    The least typical faces come last: those are where a wrong face shows up.
    """
    _, person = _edit(root, person_id)
    rows, where = _face_rows(root)
    cached = cached_photos(root)
    centre = centroid(person)
    found = []
    for asset_id, index in person.get("faces", []):
        row = where.get((str(asset_id), int(index)))
        if row is None:
            continue
        frame_id = next(frame for frame, present in sorted(cached.items()) if asset_id in present)
        found.append({
            "asset_id": asset_id,
            "index": int(index),
            "frame_id": frame_id,
            "likeness": round(float(rows[row] @ centre), 4),
        })
    found.sort(key=lambda face: -face["likeness"])
    return found


def move_faces(
    root: Path, target_id: str, other_id: str, faces: list[list[Any]]
) -> dict[str, Any]:
    """Some of ``other``'s faces are ``target``: move just those.

    What is left of ``other`` stays a separate person, remembered as not
    ``target``. Every face involved is pinned, so a later regrouping keeps
    the split. Can be undone like a merge.
    """
    if target_id == other_id:
        raise ValueError("A person cannot be merged with itself.")
    chosen = list(dict.fromkeys((str(a), int(i)) for a, i in faces))
    if not chosen:
        raise ValueError("Choose at least one photo.")
    with _STORE_LOCK:
        data, target = _edit(root, target_id)
        other = next((p for p in data["people"] if p["id"] == other_id), None)
        if other is None:
            raise ValueError(f"No such person: {other_id}")
        theirs = [(str(a), int(i)) for a, i in other.get("faces", [])]
        if any(face not in theirs for face in chosen):
            raise ValueError("Those faces are not in that group.")
        remaining = [face for face in theirs if face not in chosen]
        if not remaining:
            return merge(root, target_id, other_id)
        rows, where = _face_rows(root)

        def anchor(group: list[FaceRef]) -> dict[str, Any] | None:
            vectors = [rows[where[f]] for f in group if f in where]
            if not vectors:
                return None
            centre = _unit(np.mean(vectors, axis=0))
            return {"centroid": [round(float(v), 4) for v in centre], "size": len(group)}

        undo = {
            "target": copy.deepcopy(target),
            "other": copy.deepcopy(other),
            "not_same": copy.deepcopy(data["not_same"]),
            "feedback_keys": [],
        }
        mine = [(str(a), int(i)) for a, i in target.get("faces", [])]
        target["faces"] = [list(f) for f in dict.fromkeys([*mine, *chosen])]
        if (new := anchor(chosen)) is not None:
            target["anchors"] = [*target.get("anchors", []), new]
        target["pinned"] = [list(f) for f in dict.fromkeys([*_pins(target), *chosen])]
        if not target.get("avatar"):
            target["avatar"] = list(chosen[0])
        other["faces"] = [list(f) for f in remaining]
        if (rest := anchor(remaining)) is not None:
            other["anchors"] = [rest]
        other["pinned"] = [
            list(f) for f in dict.fromkeys([*(f for f in _pins(other) if f not in chosen), *remaining])
        ]
        if not other.get("avatar") or tuple(other["avatar"]) in chosen:
            centre = centroid(other)
            other["avatar"] = list(max(
                remaining, key=lambda f: float(rows[where[f]] @ centre) if f in where else -1.0
            ))
        pair = sorted([target_id, other_id])
        if pair not in data["not_same"]:
            data["not_same"].append(pair)
        data["merges"] = [*data["merges"], undo][-UNDO_LIMIT:]
        save(root, data)
    avatar_path(root, other_id).unlink(missing_ok=True)
    return target


def suggestions(root: Path, person_id: str) -> list[dict[str, Any]]:
    """Other people who may be this person, most alike first.

    Likeness is the closest pair of anchors between the two, so a group near
    any part of someone already merged from several groups is found. Hidden
    people, people with no photos left and pairs the user has said "no" to are
    left out.
    """
    data, person = _edit(root, person_id)
    mine = _anchor_vectors(person)
    if not mine:
        return []
    refused = {tuple(pair) for pair in data["not_same"]}
    cached = cached_photos(root)
    found = []
    for other in data["people"]:
        theirs = _anchor_vectors(other)
        if other["id"] == person_id or other.get("hidden") or not theirs:
            continue
        if tuple(sorted([person_id, other["id"]])) in refused:
            continue
        likeness = float((np.stack(mine) @ np.stack(theirs).T).max())
        if likeness < SUGGEST_MIN:
            continue
        entry = gallery_entry(other, cached)
        if entry["photo_count"]:
            found.append({**entry, "similarity": round(likeness, 4)})
    found.sort(key=lambda entry: -entry["similarity"])
    return found[:SUGGEST_LIMIT]


def cached_photos(root: Path) -> dict[str, set[str]]:
    """``{frame_id: asset ids with a photo in that frame's cache}``"""
    return {folder.name: {p.stem for p in folder.glob("*.jpg")} for folder in _frame_dirs(root)}


def gallery_entry(person: dict[str, Any], cached: dict[str, set[str]]) -> dict[str, Any]:
    assets = {asset_id for asset_id, _ in person.get("faces", [])}
    frames = {frame: len(assets & present) for frame, present in cached.items()}
    present_anywhere = set().union(*cached.values()) if cached else set()
    return {
        "id": person["id"],
        "name": person.get("name"),
        "hidden": bool(person.get("hidden")),
        "photo_count": len(assets & present_anywhere),
        "frames": {frame: count for frame, count in frames.items() if count},
    }


def gallery(root: Path) -> dict[str, Any]:
    """Everyone, sorted into the people worth a tile, the one-offs and the hidden."""
    data = load(root)
    cached = cached_photos(root)
    groups: dict[str, list[dict[str, Any]]] = {"people": [], "seen_once": [], "hidden": []}
    for person in data["people"]:
        entry = gallery_entry(person, cached)
        if entry["hidden"]:
            groups["hidden"].append(entry)
        elif entry["name"] or entry["photo_count"] >= TILE_MIN_PHOTOS:
            groups["people"].append(entry)
        elif entry["photo_count"]:
            groups["seen_once"].append(entry)
    for entries in groups.values():
        entries.sort(key=lambda e: (-e["photo_count"], (e["name"] or "").lower()))
    return {
        "built_at": data["built_at"],
        "main_min_photos": MAIN_MIN_PHOTOS,
        "can_undo": bool(data["merges"]),
        **groups,
    }


# -------------------------------------------------------------------- avatars


def avatar_path(root: Path, person_id: str) -> Path:
    return people_dir(root) / "avatars" / f"{person_id}.jpg"


def ensure_avatar(matcher: FaceMatcher, root: Path, person_id: str) -> Path | None:
    """The person's face crop, made the first time it is asked for.

    None if the person is unknown or the photo their face came from is no
    longer in any frame's cache.
    """
    person = next((p for p in load(root)["people"] if p["id"] == person_id), None)
    if person is None:
        return None
    dest = avatar_path(root, person_id)
    if dest.is_file():
        return dest
    if not person.get("avatar"):
        return None
    asset_id = person["avatar"][0]
    photo = next(
        (f / f"{asset_id}.jpg" for f in _frame_dirs(root) if (f / f"{asset_id}.jpg").is_file()),
        None,
    )
    if photo is None:
        return None
    try:
        faces = matcher.detected_faces(photo)
        if not faces:
            return None
        # The detector is run again here, so pick out their face by likeness
        # rather than trusting that faces come back in the same order.
        centre = centroid(person)
        _, box = max(faces, key=lambda face: float(face[0] @ centre))
        tmp = dest.with_name(f"{dest.stem}.{uuid.uuid4().hex[:8]}.part.jpg")
        matcher.save_face_crop(photo, box, tmp, pad=0.35)
        os.replace(tmp, dest)
    except FaceMatcherError:
        return None
    return dest


def _face_vector(root: Path, asset_id: str, index: int) -> tuple[Path, np.ndarray] | None:
    """Where one stored face's photo is, and its fingerprint."""
    for folder in _frame_dirs(root):
        photo = folder / f"{asset_id}.jpg"
        if not photo.is_file():
            continue
        faces = _load_npz(folder / "embeddings.npz").get(asset_id)
        if faces is not None and faces.ndim == 2 and 0 <= index < len(faces):
            return photo, _unit(np.asarray(faces[index], dtype=np.float32))
    return None


def _find_face(
    matcher: FaceMatcher, root: Path, asset_id: str, index: int
) -> tuple[Path, dict[str, float]] | None:
    found = _face_vector(root, asset_id, index)
    if found is None:
        return None
    photo, vector = found
    try:
        faces = matcher.detected_faces(photo)
    except FaceMatcherError:
        return None
    if not faces:
        return None
    # Detection runs again here, so match the face by likeness, not by order.
    _, box = max(faces, key=lambda face: float(face[0] @ vector))
    return photo, box


def face_box(
    matcher: FaceMatcher, root: Path, asset_id: str, index: int
) -> tuple[str, dict[str, float]] | None:
    """Which frame's copy to show, and where in it this face is (fractions)."""
    found = _find_face(matcher, root, asset_id, index)
    return (found[0].parent.name, found[1]) if found else None


def face_crop_path(root: Path, asset_id: str, index: int) -> Path:
    return people_dir(root) / "faces" / f"{asset_id}_{index}.jpg"


def ensure_face_crop(matcher: FaceMatcher, root: Path, asset_id: str, index: int) -> Path | None:
    """A crop around one face, made the first time it is asked for."""
    dest = face_crop_path(root, asset_id, index)
    if dest.is_file():
        return dest
    found = _find_face(matcher, root, asset_id, index)
    if found is None:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.stem}.{uuid.uuid4().hex[:8]}.part.jpg")
    try:
        matcher.save_face_crop(found[0], found[1], tmp, pad=0.35)
        os.replace(tmp, dest)
    except FaceMatcherError:
        tmp.unlink(missing_ok=True)
        return None
    return dest

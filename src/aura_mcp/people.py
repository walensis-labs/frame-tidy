"""Who a search is looking for.

A person starts as one or more reference face crops saved under the same name.
As the user reviews matches they answer "yes, that's them" or "no", and those
answers are kept here too:

    references/references.json   the saved reference crops (name, path)
    references/feedback.json     per person: photo -> yes/no + that face's embedding

Confirmed faces are averaged with the references, which makes the person's own
photos score higher. Rejected faces are kept as "not this person" examples: a
face that looks more like one of them than like the person is not a match.
That is what separates look-alike relatives.

The face embeddings are stored with the answers, so what was learned survives
after the photos themselves have been removed from a frame.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .face_matcher import FaceMatcher
from .storage import read_json, write_json

YES, NO = "yes", "no"


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-") or "person"


def references_dir(root: Path) -> Path:
    return root / "references"


def references(root: Path) -> list[dict[str, Any]]:
    """Saved reference crops whose image still exists."""
    entries = read_json(references_dir(root) / "references.json", [])
    return [
        {**entry, "person": slug(entry.get("name", ""))}
        for entry in entries
        if entry.get("path") and Path(entry["path"]).is_file()
    ]


def _feedback_path(root: Path) -> Path:
    return references_dir(root) / "feedback.json"


def feedback(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """``{person_slug: {"frame_id/asset_id": {"label", "embedding"}}}``"""
    return read_json(_feedback_path(root), {})


def photo_key(frame_id: str, asset_id: str) -> str:
    return f"{frame_id}/{asset_id}"


def labels(root: Path, person: str, frame_id: str) -> dict[str, str]:
    """``{asset_id: "yes" | "no"}`` for one person on one frame."""
    prefix = f"{frame_id}/"
    return {
        key[len(prefix):]: entry["label"]
        for key, entry in feedback(root).get(slug(person), {}).items()
        if key.startswith(prefix)
    }


def set_labels(
    root: Path,
    person: str,
    frame_id: str,
    faces: dict[str, np.ndarray | None],
    label: str | None,
) -> dict[str, int]:
    """Record (or with ``label=None`` clear) the user's answer for some photos.

    ``faces`` maps asset id to the embedding of the face the answer is about.
    """
    if label not in (YES, NO, None):
        raise ValueError("label must be 'yes', 'no' or null.")
    data = feedback(root)
    entries = data.setdefault(slug(person), {})
    for asset_id, embedding in faces.items():
        key = photo_key(frame_id, asset_id)
        if label is None:
            entries.pop(key, None)
        elif embedding is None:
            raise ValueError(f"No face is known for photo {asset_id}.")
        else:
            entries[key] = {
                "label": label,
                "embedding": [round(float(v), 5) for v in embedding],
            }
    write_json(_feedback_path(root), data, indent=None)
    return {
        "confirmed": sum(1 for e in entries.values() if e["label"] == YES),
        "rejected": sum(1 for e in entries.values() if e["label"] == NO),
    }


def move_feedback(
    root: Path, from_key: str, to_key: str, only: list[str] | None = None
) -> list[str]:
    """Hand one person's answers to another; the receiver's own answers win.

    With ``only``, just the answers about those photos are moved. Returns the
    photo keys that changed hands, which is what undoing the move needs.
    """
    data = feedback(root)
    source = data.get(slug(from_key), {})
    wanted = [key for key in source if only is None or key in only]
    if not wanted:
        return []
    kept = data.setdefault(slug(to_key), {})
    moved = []
    for key in wanted:
        entry = source.pop(key)
        if key not in kept:
            kept[key] = entry
            moved.append(key)
    if not source:
        data.pop(slug(from_key), None)
    write_json(_feedback_path(root), data, indent=None)
    return moved


def summary(root: Path) -> list[dict[str, Any]]:
    """Each known person with how much is known about them."""
    data = feedback(root)
    people: dict[str, dict[str, Any]] = {}
    for ref in references(root):
        person = people.setdefault(
            ref["person"], {"person": ref["person"], "name": ref["name"], "references": 0}
        )
        person["references"] += 1
    for person in people.values():
        entries = data.get(person["person"], {}).values()
        person["confirmed"] = sum(1 for e in entries if e["label"] == YES)
        person["rejected"] = sum(1 for e in entries if e["label"] == NO)
    return list(people.values())


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


@dataclass
class PersonModel:
    """What a search compares faces against."""

    positives: np.ndarray  # (n, 512): reference faces + confirmed faces
    negatives: np.ndarray = field(default_factory=lambda: np.zeros((0, 512), np.float32))
    slug: str | None = None
    name: str | None = None
    confirmed: frozenset[str] = frozenset()  # photo keys answered "yes"
    rejected: frozenset[str] = frozenset()  # photo keys answered "no"
    reference_faces: int = 1  # most faces seen in any one reference image
    # For a person found by grouping faces (see people_index): the photos the
    # grouping gave them, which a search flags whatever they score.
    member_assets: frozenset[str] = frozenset()
    # True for such a person: their matches are taken to be them unless the
    # user says otherwise, instead of each one needing a "yes".
    assume_match: bool = False
    # For such a person, photos answered "no" on any frame. The same photo is
    # usually on several frames, and "not them" is about the photo.
    rejected_assets: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        self.positives = np.atleast_2d(np.asarray(self.positives, dtype=np.float32))
        self.negatives = np.asarray(self.negatives, dtype=np.float32).reshape(-1, self.positives.shape[1])
        self._centre = _unit(self.positives.mean(axis=0))

    def similarities(self, faces: np.ndarray) -> np.ndarray:
        """How much each face looks like the person (cosine similarity)."""
        faces = np.atleast_2d(np.asarray(faces, dtype=np.float32))
        norms = np.linalg.norm(faces, axis=1)
        return (faces @ self._centre) / np.where(norms == 0, 1.0, norms)

    def best(self, faces: np.ndarray, *, apply_rejections: bool = True) -> tuple[float, int] | None:
        """Score and index of the face most like the person, or None.

        With ``apply_rejections``, a face that resembles a "not this person"
        example at least as much as it resembles the person is ruled out.
        """
        if len(faces) == 0:
            return None
        scores = self.similarities(faces)
        if apply_rejections and len(self.negatives):
            faces = np.atleast_2d(np.asarray(faces, dtype=np.float32))
            unlike = (faces @ self.negatives.T).max(axis=1)
            scores = np.where(scores > unlike, scores, -np.inf)
        index = int(np.argmax(scores))
        return (float(scores[index]), index) if np.isfinite(scores[index]) else None


UNNAMED = "Unnamed person"


def rejected_assets(root: Path, person: str) -> frozenset[str]:
    """Asset ids answered "not them" for this person, on whichever frame."""
    return frozenset(
        key.split("/", 1)[1]
        for key, entry in feedback(root).get(slug(person), {}).items()
        if entry["label"] == NO and "/" in key
    )


def _grouped_person_model(root: Path, person: dict[str, Any], centre: np.ndarray) -> PersonModel:
    """The model for a person found by grouping faces.

    Their answers are filed under their id, not their name, so renaming or
    merging them never loses what was learned.
    """
    entries = feedback(root).get(slug(person["id"]), {})
    positives, negatives = [centre], []
    for entry in entries.values():
        vector = _unit(np.asarray(entry["embedding"], dtype=np.float32))
        (positives if entry["label"] == YES else negatives).append(vector)
    return PersonModel(
        positives=np.stack(positives),
        negatives=np.stack(negatives) if negatives else np.zeros((0, 512), np.float32),
        slug=person["id"],
        name=person.get("name") or UNNAMED,
        confirmed=frozenset(k for k, e in entries.items() if e["label"] == YES),
        rejected=frozenset(k for k, e in entries.items() if e["label"] == NO),
        member_assets=frozenset(asset_id for asset_id, _ in person.get("faces", [])),
        assume_match=True,
        rejected_assets=rejected_assets(root, person["id"]),
    )


def build_model(
    matcher: FaceMatcher, root: Path, reference_image_path: str = "", person: str = ""
) -> PersonModel:
    """The model for a named person, or for a one-off reference image.

    A ``reference_image_path`` that points at a saved reference crop is
    treated as that person, so their feedback is used too.
    """
    saved = references(root)
    if not person and reference_image_path:
        target = Path(reference_image_path).expanduser().resolve()
        person = next(
            (r["name"] for r in saved if Path(r["path"]).resolve() == target), ""
        )

    if not person:
        if not reference_image_path:
            raise ValueError("Give a reference_image_path or the name of a saved person.")
        embedding, count = matcher.reference_embedding(reference_image_path)
        return PersonModel(positives=embedding[None, :], reference_faces=count)

    key = slug(person)
    mine = [r for r in saved if r["person"] == key]
    if not mine:
        from . import people_index  # imported here: people_index imports this module

        found = people_index.find(root, person)
        if found is not None:
            return _grouped_person_model(root, found, people_index.centroid(found))
        known = sorted({r["name"] for r in saved})
        raise ValueError(
            f"No saved reference for {person!r}. "
            + (f"Saved people: {', '.join(known)}." if known else "Save one in the review page first.")
        )
    positives, most_faces = [], 1
    for ref in mine:
        embedding, count = matcher.reference_embedding(ref["path"])
        positives.append(embedding)
        most_faces = max(most_faces, count)

    entries = feedback(root).get(key, {})
    negatives = []
    for entry in entries.values():
        vector = _unit(np.asarray(entry["embedding"], dtype=np.float32))
        (positives if entry["label"] == YES else negatives).append(vector)
    return PersonModel(
        positives=np.stack(positives),
        negatives=np.stack(negatives) if negatives else np.zeros((0, 512), np.float32),
        slug=key,
        name=mine[0]["name"],
        confirmed=frozenset(k for k, e in entries.items() if e["label"] == YES),
        rejected=frozenset(k for k, e in entries.items() if e["label"] == NO),
        reference_faces=most_faces,
    )

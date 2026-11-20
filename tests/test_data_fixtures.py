"""Saved data from earlier versions must keep opening.

``tests/fixtures/data-v1`` is a frozen, synthetic data folder in format 1 (no
real photos or people). Never regenerate it with newer code: it stands for
what users of that version have on disk. Paths in it are written ``{ROOT}``
and the photo copies are made here, so the fixture holds no images.

When a release adds an upgrade step (see aura_mcp.migrations), add a frozen
folder for the format it replaces next to this one, and a test like these.
"""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from starlette.testclient import TestClient

from aura_mcp import migrations, people, people_index
from aura_mcp.face_matcher import FaceMatcher
from aura_mcp.review import Review

FIXTURES = Path(__file__).parent / "fixtures"
KITCHEN, DEN = "11111111-1111-4111-8111-111111111111", "22222222-2222-4222-8222-222222222222"
FIRST, SHARED, REMOVED = (
    "aaaaaaa1-0000-4000-8000-000000000001",
    "aaaaaaa2-0000-4000-8000-000000000002",
    "aaaaaaa5-0000-4000-8000-000000000005",
)


def open_fixture(name: str, tmp_path: Path) -> Path:
    root = tmp_path / "data"
    shutil.copytree(FIXTURES / name, root)
    for path in root.rglob("*.json"):
        path.write_text(path.read_text().replace("{ROOT}", str(root)))
    for path in root.rglob("*.json"):
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            continue
        for entry in data.values():
            if isinstance(entry, dict) and str(entry.get("local_path", "")).endswith(".jpg"):
                cv2.imwrite(entry["local_path"], np.full((60, 80, 3), 128, np.uint8))
    return root


@pytest.fixture
def v1(tmp_path):
    root = open_fixture("data-v1", tmp_path)
    result = migrations.upgrade(root)
    assert result["from"] == 1 and result["to"] == migrations.CURRENT
    return root


def test_format_1_is_recognised_and_brought_up_to_date(v1):
    assert migrations.read_version(v1) == migrations.CURRENT


def test_format_1_opens_in_the_page(v1):
    client = TestClient(Review(v1, FaceMatcher(app=object())).app(), base_url="http://127.0.0.1")
    frames = {f["id"]: (f["name"], f["photos"]) for f in client.get("/api/frames").json()["frames"]}
    assert frames[KITCHEN] == ("Kitchen", 3) and frames[DEN] == ("Den", 2)

    kitchen = client.get(f"/api/frames/{KITCHEN}/photos").json()
    assert kitchen["selection"] == [FIRST]
    assert kitchen["flagged"]["person_name"] == "Sam"
    labels = {p["id"]: p["label"] for p in kitchen["photos"]}
    assert labels[SHARED] == "no"  # the "not them" answer survived

    gallery = client.get("/api/people").json()
    sam = next(p for p in gallery["people"] if p["name"] == "Sam")
    assert sam["photo_count"] == 3 and sam["frames"] == {KITCHEN: 2, DEN: 2}

    removed = client.get(f"/api/frames/{KITCHEN}/removed").json()
    assert [item["id"] for item in removed["removed"]] == [REMOVED]


def test_format_1_people_and_answers_read_directly(v1):
    sam = people_index.find(v1, "Sam")
    assert sam is not None and len(sam["anchors"]) >= 1
    assert people.labels(v1, sam["id"], KITCHEN) == {SHARED: "no"}

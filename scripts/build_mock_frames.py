"""Build mock frames from the LFW face dataset, with known right answers.

LFW (Labeled Faces in the Wild) is a public research dataset of news photos,
one folder per person. This picks one person as the "target", holds one of
their photos out as the reference image, and spreads the rest across two mock
frames among photos of other people:

    mock/photos/Living Room/   most target photos + many other people + no-face images
    mock/photos/Kitchen/       a few of the SAME target photos + different other people
    mock/reference/<person>.jpg
    mock/ground_truth.json     which file names really show the target

Get the dataset first (about 180 MB):

    mkdir -p mock/downloads/lfw
    curl -L -o mock/downloads/lfw.tgz https://ndownloader.figshare.com/files/5976018
    tar -xzf mock/downloads/lfw.tgz -C mock/downloads/lfw

Then:  uv run python scripts/build_mock_frames.py
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lfw", type=Path, default=ROOT / "mock/downloads/lfw/lfw")
    parser.add_argument("--out", type=Path, default=ROOT / "mock")
    parser.add_argument("--person", default="Serena_Williams",
                        help="LFW folder name of the target person")
    parser.add_argument("--others", type=int, default=150,
                        help="Photos of other people on the main frame")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    target_dir = args.lfw / args.person
    if not target_dir.is_dir():
        raise SystemExit(f"No such person in LFW: {target_dir}")
    rng = random.Random(args.seed)

    target = sorted(target_dir.glob("*.jpg"))
    reference, target = target[0], target[1:]
    rng.shuffle(target)
    shared = target[:10]  # on both frames

    # One photo each from a random set of other people.
    people = sorted(p for p in args.lfw.iterdir() if p.is_dir() and p.name != args.person)
    rng.shuffle(people)
    others = [sorted(p.glob("*.jpg"))[0] for p in people[: args.others + 40]]
    living_others, kitchen_others = others[: args.others], others[args.others :]

    photos = args.out / "photos"
    shutil.rmtree(photos, ignore_errors=True)
    living, kitchen = photos / "Living Room", photos / "Kitchen"
    living.mkdir(parents=True)
    kitchen.mkdir(parents=True)

    for src in target + living_others:
        shutil.copy2(src, living / src.name)
    for src in shared + kitchen_others:
        shutil.copy2(src, kitchen / src.name)

    # A few photos with nobody in them, like the scenery shots on a real frame.
    noise = np.random.default_rng(args.seed)
    for i in range(3):
        gradient = np.linspace(0, 255, 400, dtype=np.float32)[None, :, None]
        image = gradient * noise.uniform(0.3, 1.0, size=(1, 1, 3)) + noise.normal(0, 6, (300, 400, 3))
        cv2.imwrite(str(living / f"scenery_{i + 1}.jpg"), np.clip(image, 0, 255).astype(np.uint8))

    ref_dir = args.out / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref_path = ref_dir / f"{args.person}.jpg"
    shutil.copy2(reference, ref_path)

    truth = {
        "person": args.person,
        "reference": str(ref_path),
        "target_file_names": sorted(p.name for p in target),
        "frames": {
            "Living Room": {"target": len(target), "total": len(list(living.iterdir()))},
            "Kitchen": {"target": len(shared), "total": len(list(kitchen.iterdir()))},
        },
    }
    (args.out / "ground_truth.json").write_text(json.dumps(truth, indent=2))
    print(json.dumps({k: truth[k] for k in ("person", "reference", "frames")}, indent=2))


if __name__ == "__main__":
    main()

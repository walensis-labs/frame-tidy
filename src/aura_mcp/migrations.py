"""Bring a data folder up to the format this version of the app uses.

Everything the app keeps (photo copies, the people store, the removal log,
answers, selections, the sign-in token) lives in one data folder that
outlives the app: a new version is installed over the old one and opens the
same folder. When a release changes how something is saved, it adds a step
here instead of teaching every reader about every old format.

    data-version.json   {"version": N}: the format the folder is in

How a release adds a step:

1. Raise ``CURRENT`` by one and add ``STEPS[CURRENT] = fn``. ``fn(root)``
   turns a folder in format ``CURRENT - 1`` into format ``CURRENT``. It
   should be safe to run again on a folder it already changed (a step cut
   short is run again from the start).
2. Freeze a copy of a folder in the old format under ``tests/fixtures/`` and
   test that the step and the current readers handle it (see
   ``tests/test_data_fixtures.py``).

Before any step runs, the small files (``.json`` and ``.npz``, not the photo
copies) are copied to ``backups/before-v{N}-{time}/``. The version on disk
is written after each step, so a failure leaves the folder at the last
format that was finished. A folder written by a newer app is never touched.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

APP_NAME = "Frame Tidy"
VERSION_FILE = "data-version.json"

# The format this version of the app reads and writes.
CURRENT = 1

# STEPS[n] turns format n - 1 into format n.
STEPS: dict[int, Callable[[Path], None]] = {}

# What a data folder holds. A folder with none of these has no data in it
# (when run from the source tree, the data folder is the project folder).
DATA_ENTRIES = (
    "cache", "people", "references", "removed", "selections", "flagged",
    "session.json", "settings.json",
)
# Copied before an upgrade. Photo copies are left out: they are large, and no
# step should need to change them.
BACKUP_SUFFIXES = (".json", ".npz")


class UpgradeFailed(Exception):
    """The folder could not be brought up to date. ``backup`` is where the copies are."""

    def __init__(self, message: str, backup: Path | None = None) -> None:
        super().__init__(message)
        self.backup = backup


class DataTooNew(UpgradeFailed):
    """The folder was written by a newer version of the app."""


def has_data(root: Path) -> bool:
    return any((root / name).exists() for name in DATA_ENTRIES)


def read_version(root: Path) -> int | None:
    """The format on disk; None when the file is missing. Raises UpgradeFailed if unreadable."""
    path = root / VERSION_FILE
    if not path.is_file():
        return None
    try:
        version = json.loads(path.read_text())["version"]
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise ValueError(version)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise UpgradeFailed(
            f"{APP_NAME} could not read {path}. Nothing was changed. "
            "If you edited or replaced that file, put the original back."
        ) from exc
    return version


def write_version(root: Path, version: int) -> None:
    path = root / VERSION_FILE
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.part")
    tmp.write_text(json.dumps({"version": version}) + "\n")
    os.replace(tmp, path)


def backup(root: Path, target: int) -> Path:
    """Copy the small data files, keeping their places, before an upgrade to ``target``."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    dest = root / "backups" / f"before-v{target}-{stamp}-{uuid.uuid4().hex[:4]}"
    for name in DATA_ENTRIES:
        source = root / name
        if source.is_file():
            files = [source]
        elif source.is_dir():
            files = [p for p in source.rglob("*") if p.is_file() and p.suffix in BACKUP_SUFFIXES]
        else:
            continue
        for path in files:
            copy = dest / path.relative_to(root)
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, copy)
    if (root / VERSION_FILE).is_file():
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / VERSION_FILE, dest / VERSION_FILE)
    return dest


def upgrade(
    root: Path,
    *,
    current: int = CURRENT,
    steps: dict[int, Callable[[Path], None]] | None = None,
) -> dict[str, Any]:
    """Bring ``root`` to format ``current``. Returns ``{"from", "to", "backup"}``.

    ``from`` is None for a folder with no data yet. Raises DataTooNew for a
    folder from a newer app and UpgradeFailed when a step fails.
    """
    steps = STEPS if steps is None else steps
    root.mkdir(parents=True, exist_ok=True)
    found = read_version(root)
    if found is None:
        if not has_data(root):
            write_version(root, current)
            return {"from": None, "to": current, "backup": None}
        found = 1  # saved before versions were recorded
    if found > current:
        raise DataTooNew(
            f"This data was saved by a newer version of {APP_NAME} "
            f"(format {found}; this version reads up to {current}). "
            f"Install the newest {APP_NAME} to open it. Nothing was changed."
        )
    if found == current:
        if not (root / VERSION_FILE).is_file():
            write_version(root, current)
        return {"from": found, "to": current, "backup": None}

    for n in range(found + 1, current + 1):
        assert n in steps, f"no upgrade step to format {n}"
    saved = backup(root, current)
    for n in range(found + 1, current + 1):
        try:
            steps[n](root)
        except Exception as exc:
            raise UpgradeFailed(
                f"{APP_NAME} could not update your saved data to format {n}: {exc}. "
                f"It is still in format {n - 1}, and a copy from before the update "
                f"is in {saved}.",
                saved,
            ) from exc
        write_version(root, n)
    return {"from": found, "to": current, "backup": saved}

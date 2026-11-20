"""Small JSON file helpers shared by the server, the review page and people."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any


def read_json(path: Path, default: Any) -> Any:
    """The parsed file, or ``default`` if it is missing or unreadable."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def write_json(path: Path, data: Any, *, indent: int | None = 2) -> None:
    """Write atomically, so a crash never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # A name of its own, so two writers (the review page, a build running in
    # the background, the MCP server) never share a half-written file.
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.part")
    try:
        tmp.write_text(json.dumps(data, indent=indent, sort_keys=True))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)

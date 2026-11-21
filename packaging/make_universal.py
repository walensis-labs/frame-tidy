"""Join an Apple-silicon app and an Intel app into one Universal app.

    python3 packaging/make_universal.py <arm64.app> <x86_64.app> <out.app>

The two apps are built from the same code and locked libraries, so they hold
the same files. Each binary present in both is merged with ``lipo`` into one
file holding both versions; macOS runs the one that matches the Mac. Files
present in only one app (OpenCV's differently named helper libraries) are
copied as they are. Any other file must be identical in both apps, apart from
the few listed in SAME_IN_EFFECT, or the join stops.

The result is unsigned (merging breaks the signatures PyInstaller made).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from check_app import is_macho

# Files that differ between the two builds without differing in effect: the
# per-file hash lists and wheel tags of installed packages, the order of
# imports in a generated OpenCV typing file, and the zip of the standard
# library (same files, different zip timestamps; compared by content below).
SAME_IN_EFFECT = ("RECORD", "WHEEL")
SAME_IN_EFFECT_PATHS = ("Contents/Resources/cv2/typing/__init__.py",)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _zip_contents(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as z:
        return {n: hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()}


def _tree(root: Path) -> tuple[dict[Path, Path], dict[Path, str]]:
    files, links = {}, {}
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if rel.parts[:2] == ("Contents", "_CodeSignature"):
            continue
        if path.is_symlink():
            links[rel] = os.readlink(path)
        elif path.is_file():
            files[rel] = path
    return files, links


def join(arm: Path, intel: Path, out: Path) -> None:
    arm_files, arm_links = _tree(arm)
    intel_files, intel_links = _tree(intel)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(arm, out, symlinks=True)
    shutil.rmtree(out / "Contents/_CodeSignature", ignore_errors=True)

    problems, merged, added = [], 0, 0
    for rel, intel_path in sorted(intel_files.items()):
        target = out / rel
        arm_path = arm_files.get(rel)
        if arm_path is None:
            if rel in arm_links:
                problems.append(f"{rel}: a file in one app, a link in the other")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(intel_path, target)
            added += 1
        elif is_macho(arm_path):
            if not is_macho(intel_path):
                problems.append(f"{rel}: a binary in one app only")
                continue
            target.unlink()
            subprocess.run(["lipo", "-create", "-output", str(target), str(arm_path), str(intel_path)],
                           check=True)
            merged += 1
        elif _sha(arm_path) != _sha(intel_path):
            same = (
                rel.name in SAME_IN_EFFECT
                or str(rel) in SAME_IN_EFFECT_PATHS
                or (rel.suffix == ".zip" and _zip_contents(arm_path) == _zip_contents(intel_path))
            )
            if not same:
                problems.append(f"{rel}: differs between the two apps")
    for rel, link in sorted(intel_links.items()):
        if rel in arm_links:
            if arm_links[rel] != link:
                problems.append(f"{rel}: links to different places")
            continue
        if rel in arm_files:
            problems.append(f"{rel}: a link in one app, a file in the other")
            continue
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        os.symlink(link, out / rel)
        added += 1
    if problems:
        print("ERROR: the two apps do not join cleanly:", *problems[:30], sep="\n  ", file=sys.stderr)
        sys.exit(1)
    print(f"Joined: {merged} binaries merged, {added} files and links only in the Intel app added.")


if __name__ == "__main__":
    join(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))

"""Check every binary in a built app before it is signed.

    python3 packaging/check_app.py "dist/Frame Tidy.app" arm64 [x86_64]

Fails unless:
- every binary is built only for the listed kinds of Mac (arm64, x86_64);
- the main program holds all of them (a Universal app when both are listed);
- no binary needs a newer macOS than LSMinimumSystemVersion in Info.plist,
  so an older Mac gets macOS's "needs a newer version" message instead of a
  crash.

Libraries that exist for only one kind of Mac are allowed: the OpenCV builds
for Apple silicon and Intel bundle differently named helper libraries, and
each kind of Mac loads only its own.
"""

from __future__ import annotations

import plistlib
import subprocess
import sys
from pathlib import Path


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split("."))


def is_macho(path: Path) -> bool:
    out = subprocess.run(["file", "-b", str(path)], capture_output=True, text=True).stdout
    return out.startswith("Mach-O")


def archs(path: Path) -> list[str]:
    return subprocess.run(["lipo", "-archs", str(path)], capture_output=True, text=True).stdout.split()


def min_macos(path: Path, arch: str) -> str | None:
    lines = [line.split() for line in subprocess.run(
        ["otool", "-arch", arch, "-l", str(path)], capture_output=True, text=True).stdout.splitlines()]
    for i, line in enumerate(lines):
        if line[:1] == ["minos"]:  # LC_BUILD_VERSION
            return line[1]
        if line == ["cmd", "LC_VERSION_MIN_MACOSX"]:  # older binaries
            for later in lines[i + 1:i + 4]:
                if later[:1] == ["version"]:
                    return later[1]
    return None


def check(app: Path, wanted: list[str]) -> list[str]:
    info = plistlib.loads((app / "Contents/Info.plist").read_bytes())
    declared = info["LSMinimumSystemVersion"]
    main = app / "Contents/MacOS" / info["CFBundleExecutable"]
    problems = []
    if sorted(archs(main)) != sorted(wanted):
        problems.append(f"main program is {archs(main)}, expected {wanted}")
    count = 0
    for path in sorted(app.rglob("*")):
        if not path.is_file() or path.is_symlink() or not is_macho(path):
            continue
        count += 1
        rel = path.relative_to(app)
        have = archs(path)
        extra = [a for a in have if a not in wanted]
        if extra or not have:
            problems.append(f"{rel}: built for {have}")
            continue
        for arch in have:
            needs = min_macos(path, arch)
            if needs and _version(needs) > _version(declared):
                problems.append(f"{rel} ({arch}): needs macOS {needs}")
    if not problems:
        print(f"{count} binaries checked: {' + '.join(wanted)}, macOS {declared} or later.")
    return problems


def main() -> None:
    app, wanted = Path(sys.argv[1]), sys.argv[2:]
    problems = check(app, wanted)
    if problems:
        print("ERROR: the app is not ready to sign:", *problems[:30], sep="\n  ", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

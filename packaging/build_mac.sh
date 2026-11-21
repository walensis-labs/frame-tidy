#!/usr/bin/env bash
# Build "Frame Tidy.app" and "Frame Tidy.dmg" into dist/ for one kind of Mac.
#
#   packaging/build_mac.sh            this Mac's kind
#   ARCH=arm64  packaging/build_mac.sh   Apple silicon
#   ARCH=x86_64 packaging/build_mac.sh   Intel (works on Apple silicon through Rosetta)
#
# The app is built for macOS 12 and later: every library is installed in the
# version made for macOS 12, in a separate environment per kind of Mac, and the
# build fails if anything inside the app needs a newer macOS than the app says.
# No face-model files are included: InsightFace downloads them on first use.
set -euo pipefail
cd "$(dirname "$0")/.."

ARCH="${ARCH:-$(uname -m)}"
case "$ARCH" in
  arm64)  PY_ARCH=aarch64 ;;
  x86_64) PY_ARCH=x86_64 ;;
  *) echo "ARCH must be arm64 or x86_64" >&2; exit 2 ;;
esac
export MACOSX_DEPLOYMENT_TARGET=12.0
PLATFORM="$PY_ARCH-apple-darwin"
VENV="build/venv-$ARCH"

# The same locked versions as development, installed for the chosen kind of Mac.
uv venv --allow-existing -q --python "cpython-3.12-macos-$PY_ARCH-none" "$VENV"
mkdir -p build
uv export --frozen --no-dev --group build --no-emit-project --no-hashes -q > "build/requirements-$ARCH.txt"
uv pip install -q -p "$VENV" --python-platform "$PLATFORM" -r "build/requirements-$ARCH.txt"
uv pip install -q -p "$VENV" --python-platform "$PLATFORM" --no-deps --reinstall-package aura-mcp .

rm -rf dist "build/pyinstaller-$ARCH"
arch "-$ARCH" "$VENV/bin/python" -m PyInstaller --noconfirm --clean --distpath dist \
  --workpath "build/pyinstaller-$ARCH" packaging/frame_tidy.spec

APP="dist/Frame Tidy.app"
test -d "$APP"
# Never ship model files by accident.
if find "$APP" -name '*.onnx' | grep -q .; then
  echo "ERROR: .onnx model files found inside the app bundle" >&2
  exit 1
fi

# Every binary must be for this kind of Mac and must not need a newer macOS
# than the app declares; otherwise it would crash on an older Mac instead of
# macOS saying the app needs a newer system.
MIN="$(/usr/libexec/PlistBuddy -c 'Print :LSMinimumSystemVersion' "$APP/Contents/Info.plist")"
python3 - "$APP" "$ARCH" "$MIN" <<'EOF'
import subprocess, sys
from pathlib import Path
app, arch, declared = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
key = lambda v: tuple(int(x) for x in v.split("."))
bad = []
for f in app.rglob("*"):
    if not f.is_file() or f.is_symlink():
        continue
    if not subprocess.run(["file", "-b", str(f)], capture_output=True, text=True).stdout.startswith("Mach-O"):
        continue
    archs = subprocess.run(["lipo", "-archs", str(f)], capture_output=True, text=True).stdout.split()
    if arch not in archs:
        bad.append(f"{f.relative_to(app)}: built for {archs}, not {arch}")
        continue
    lines = [l.split() for l in subprocess.run(
        ["otool", "-arch", arch, "-l", str(f)], capture_output=True, text=True).stdout.splitlines()]
    mins = []
    for i, l in enumerate(lines):
        if l[:1] == ["minos"]:  # LC_BUILD_VERSION
            mins.append(l[1])
        elif l == ["cmd", "LC_VERSION_MIN_MACOSX"]:  # older binaries
            mins += [m[1] for m in lines[i + 1:i + 4] if m[:1] == ["version"]]
    mins = mins[:1]
    if mins and key(mins[0]) > key(declared):
        bad.append(f"{f.relative_to(app)}: needs macOS {mins[0]}")
if bad:
    print(f"ERROR: the app says macOS {declared}, but:", *bad[:20], sep="\n  ", file=sys.stderr)
    sys.exit(1)
print(f"All binaries are {arch} and run on macOS {declared}.")
EOF

STAGE="$(mktemp -d)"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
rm -f "dist/Frame Tidy.dmg"
hdiutil create -volname "Frame Tidy" -srcfolder "$STAGE" -ov -format UDZO "dist/Frame Tidy.dmg"
rm -rf "$STAGE"

du -sh "$APP" "dist/Frame Tidy.dmg"

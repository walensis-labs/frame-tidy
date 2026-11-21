#!/usr/bin/env bash
# Build "Frame Tidy.app" and "Frame Tidy.dmg" into dist/.
#
#   packaging/build_mac.sh                  Universal: Apple silicon and Intel in one app
#   ARCH=arm64  packaging/build_mac.sh      Apple silicon only (quicker, for testing; no updater)
#   ARCH=x86_64 packaging/build_mac.sh      Intel only (works on Apple silicon through Rosetta)
#
# The app is built for macOS 12.3 and later: every library is installed in the
# version made for macOS 12, in a separate environment per kind of Mac, and
# packaging/check_app.py fails the build if any binary is for the wrong kind of
# Mac or needs a newer macOS than the app says. A Universal app is the two
# single builds joined by packaging/make_universal.py.
# No face-model files are included: InsightFace downloads them on first use.
set -euo pipefail
cd "$(dirname "$0")/.."

ARCH="${ARCH:-universal}"
export MACOSX_DEPLOYMENT_TARGET=12.0

# Build the app for one kind of Mac into build/app-<arch>/Frame Tidy.app.
build_one() {
  local arch="$1" py_arch
  case "$arch" in
    arm64)  py_arch=aarch64 ;;
    x86_64) py_arch=x86_64 ;;
    *) echo "ARCH must be universal, arm64 or x86_64" >&2; exit 2 ;;
  esac
  local platform="$py_arch-apple-darwin" venv="build/venv-$arch"
  # The same locked versions as development, installed for that kind of Mac.
  uv venv --allow-existing -q --python "cpython-3.12-macos-$py_arch-none" "$venv"
  mkdir -p build
  uv export --frozen --no-dev --group build --no-emit-project --no-hashes -q > "build/requirements-$arch.txt"
  uv pip install -q -p "$venv" --python-platform "$platform" -r "build/requirements-$arch.txt"
  uv pip install -q -p "$venv" --python-platform "$platform" --no-deps --reinstall-package aura-mcp .
  rm -rf "build/app-$arch" "build/pyinstaller-$arch"
  arch "-$arch" "$venv/bin/python" -m PyInstaller --noconfirm --clean --distpath "build/app-$arch" \
    --workpath "build/pyinstaller-$arch" packaging/frame_tidy.spec
  python3 packaging/check_app.py "build/app-$arch/Frame Tidy.app" "$arch"
}

APP="dist/Frame Tidy.app"
rm -rf dist
mkdir -p dist
if [ "$ARCH" = universal ]; then
  build_one arm64
  build_one x86_64
  python3 packaging/make_universal.py "build/app-arm64/Frame Tidy.app" "build/app-x86_64/Frame Tidy.app" "$APP"
  # Sparkle, the updater (Universal already). Its XPC services are only for
  # sandboxed apps, which this is not.
  packaging/fetch_sparkle.sh
  ditto build/sparkle/Sparkle.framework "$APP/Contents/Frameworks/Sparkle.framework"
  rm -rf "$APP/Contents/Frameworks/Sparkle.framework/Versions/B/XPCServices" \
         "$APP/Contents/Frameworks/Sparkle.framework/XPCServices"
  python3 packaging/check_app.py "$APP" arm64 x86_64
  # Joining breaks PyInstaller's local signatures, and Apple silicon will not run
  # unsigned code. Sign locally (ad hoc) so the app runs on this Mac;
  # sign_and_notarize.sh replaces these with the real signature.
  packaging/codesign_app.sh "$APP" -
else
  build_one "$ARCH"
  cp -R "build/app-$ARCH/Frame Tidy.app" "$APP"
fi

# Never ship model files by accident.
if find "$APP" -name '*.onnx' | grep -q .; then
  echo "ERROR: .onnx model files found inside the app bundle" >&2
  exit 1
fi

STAGE="$(mktemp -d)"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
rm -f "dist/Frame Tidy.dmg"
hdiutil create -volname "Frame Tidy" -srcfolder "$STAGE" -ov -format UDZO "dist/Frame Tidy.dmg"
test -s "dist/Frame Tidy.dmg" || {
  echo "ERROR: the dmg was not made. If hdiutil said 'Operation not permitted', macOS's" >&2
  echo "App Management protection is blocking this terminal from copying a Frame Tidy app:" >&2
  echo "allow it in System Settings > Privacy & Security > App Management." >&2
  exit 1
}
rm -rf "$STAGE"

du -sh "$APP" "dist/Frame Tidy.dmg"

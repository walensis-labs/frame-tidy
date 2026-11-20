#!/usr/bin/env bash
# Build "Frame Tidy.app" and "Frame Tidy.dmg" into dist/. Run from anywhere.
# No face-model files are included: InsightFace downloads them on first use.
set -euo pipefail
cd "$(dirname "$0")/.."

uv sync --group build
rm -rf build dist
uv run --group build pyinstaller --noconfirm --clean --distpath dist --workpath build \
  packaging/frame_tidy.spec

APP="dist/Frame Tidy.app"
test -d "$APP"
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
rm -rf "$STAGE"

du -sh "$APP" "dist/Frame Tidy.dmg"

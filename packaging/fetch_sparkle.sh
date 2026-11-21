#!/usr/bin/env bash
# Fetch Sparkle (the macOS updater the app uses) into build/sparkle, once.
# The version and checksum are pinned: a different file stops the build.
#
#   build/sparkle/Sparkle.framework   copied into the app by build_mac.sh
#   build/sparkle/bin/sign_update     signs a release for the appcast
#   build/sparkle/bin/generate_keys   makes the update signing key (once, by the owner)
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION=2.10.1
SHA256=1a856d2a9edb8e628de88121d7be04cf0e2167f8641650b78fca00b14244cf27
DEST=build/sparkle

if [ -f "$DEST/.version" ] && [ "$(cat "$DEST/.version")" = "$VERSION" ]; then
  exit 0
fi
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
curl -fsSLo "$TMP/sparkle.tar.xz" \
  "https://github.com/sparkle-project/Sparkle/releases/download/$VERSION/Sparkle-$VERSION.tar.xz"
echo "$SHA256  $TMP/sparkle.tar.xz" | shasum -a 256 -c - >/dev/null || {
  echo "ERROR: Sparkle $VERSION download does not match the pinned checksum" >&2
  exit 1
}
rm -rf "$DEST"
mkdir -p "$DEST"
tar -xJf "$TMP/sparkle.tar.xz" -C "$DEST" Sparkle.framework bin LICENSE
echo "$VERSION" > "$DEST/.version"
echo "Sparkle $VERSION ready in $DEST"

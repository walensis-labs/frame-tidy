#!/usr/bin/env bash
# Sign every piece of code in an app, inside out, then the app itself.
#
#   packaging/codesign_app.sh "dist/Frame Tidy.app" -                 ad hoc (runs on this Mac)
#   packaging/codesign_app.sh "dist/Frame Tidy.app" "Developer ID…"   for release (timestamped)
#
# Order matters: every binary first, then nested bundles (Sparkle's
# Updater.app and Sparkle.framework) deepest first, then the main program,
# then the app. All with the hardened runtime, which notarization requires.
# Any signing error stops the script.
set -euo pipefail
APP="$1"
IDENTITY="$2"
ENT="$(cd "$(dirname "$0")" && pwd)/entitlements.plist"
MAIN="$APP/Contents/MacOS/$(/usr/libexec/PlistBuddy -c 'Print :CFBundleExecutable' "$APP/Contents/Info.plist")"

SIGN=(codesign --force --options runtime --entitlements "$ENT" --sign "$IDENTITY")
[ "$IDENTITY" = "-" ] || SIGN+=(--timestamp)

sign() {
  local out
  if ! out="$("${SIGN[@]}" "$@" 2>&1)"; then
    printf '%s\n' "$out" >&2
    echo "ERROR: signing failed" >&2
    exit 1
  fi
}

# 1. Every Mach-O file, with or without an extension, except the main program.
binaries=()
while IFS= read -r -d '' f; do
  [ "$f" = "$MAIN" ] && continue
  if file -b "$f" | grep -q '^Mach-O'; then binaries+=("$f"); fi
done < <(find "$APP" -type f -print0)
for ((i = 0; i < ${#binaries[@]}; i += 20)); do
  sign "${binaries[@]:i:20}"
done

# 2. Nested bundles, deepest first, so each seals the already-signed code inside it.
while IFS= read -r -d '' bundle; do
  sign "$bundle"
done < <(find "$APP/Contents" -depth -type d \( -name '*.app' -o -name '*.framework' -o -name '*.xpc' \) -print0)

# 3. The main program, then the app.
sign "$MAIN"
sign "$APP"
codesign --verify --deep --strict "$APP"
echo "Signed ${#binaries[@]} binaries and the app ($IDENTITY)."

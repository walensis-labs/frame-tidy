#!/usr/bin/env bash
# Sign "Frame Tidy.app" with a Developer ID, notarize it with Apple, staple the
# ticket and rebuild the .dmg. Run by the GitHub build (see .github/workflows/build.yml)
# or by hand. First run by hand 2026-11-20: app and dmg notarized and stapled.
# NOTARIZE=0 signs only (for testing updates locally; such a build must not be shared).
#
# Environment:
#   APPLE_TEAM_ID            10-character team id (required)
#   APPLE_SIGNING_IDENTITY   optional; found in the keychain from the team id
#                            when unset ("Developer ID Application: ... (TEAMID)")
# Notarization credentials, one of:
#   APPLE_NOTARY_PROFILE     keychain profile made once with
#                            `xcrun notarytool store-credentials <name> --apple-id <you> --team-id <TEAMID>`
#                            (default "frame-tidy"; keeps the password out of the shell)
#   APPLE_ID + APPLE_APP_PASSWORD   for CI, where there is no keychain profile
# Run packaging/build_mac.sh first.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${APPLE_TEAM_ID:?set APPLE_TEAM_ID (see the comments at the top of this script)}"
if [ -z "${APPLE_SIGNING_IDENTITY:-}" ]; then
  APPLE_SIGNING_IDENTITY="$(security find-identity -v -p codesigning |
    sed -n "s/.*\"\(Developer ID Application: .*($APPLE_TEAM_ID)\)\"/\1/p" | head -1)"
  : "${APPLE_SIGNING_IDENTITY:?no Developer ID Application certificate for team $APPLE_TEAM_ID in the keychain}"
fi
if [ -n "${APPLE_ID:-}" ] && [ -n "${APPLE_APP_PASSWORD:-}" ]; then
  NOTARY=(--apple-id "$APPLE_ID" --team-id "$APPLE_TEAM_ID" --password "$APPLE_APP_PASSWORD")
else
  NOTARY=(--keychain-profile "${APPLE_NOTARY_PROFILE:-frame-tidy}")
fi
echo "Signing as: $APPLE_SIGNING_IDENTITY"

APP="dist/Frame Tidy.app"
DMG="dist/Frame Tidy.dmg"
test -d "$APP"

# Sign every binary, nested bundle (Sparkle) and the app, inside out, with the
# hardened runtime that notarization requires.
packaging/codesign_app.sh "$APP" "$APPLE_SIGNING_IDENTITY"
codesign --verify --deep --strict --verbose=2 "$APP"

if [ "${NOTARIZE:-1}" != 0 ]; then
  # Notarize the app (zip it for upload), then staple the ticket to it.
  ZIP="$(mktemp -d)/FrameTidy.zip"
  ditto -c -k --keepParent "$APP" "$ZIP"
  xcrun notarytool submit "$ZIP" "${NOTARY[@]}" --wait
  # Stapling fails unless Apple accepted the upload, so a rejection stops here.
  xcrun stapler staple "$APP"
  spctl --assess --type execute -v "$APP"
fi

# Rebuild the dmg around the stapled app, then sign, notarize and staple it.
STAGE="$(mktemp -d)"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
rm -f "$DMG"
hdiutil create -volname "Frame Tidy" -srcfolder "$STAGE" -ov -format UDZO "$DMG"
test -s "$DMG" || {
  echo "ERROR: the dmg was not made. If hdiutil said 'Operation not permitted', macOS's" >&2
  echo "App Management protection is blocking this terminal from copying a Frame Tidy app:" >&2
  echo "allow it in System Settings > Privacy & Security > App Management." >&2
  exit 1
}
rm -rf "$STAGE"
codesign --force --timestamp --sign "$APPLE_SIGNING_IDENTITY" "$DMG"
if [ "${NOTARIZE:-1}" = 0 ]; then
  echo "Signed, not notarized (NOTARIZE=0): $DMG is for local testing only."
  exit 0
fi
xcrun notarytool submit "$DMG" "${NOTARY[@]}" --wait
xcrun stapler staple "$DMG"
spctl --assess --type open --context context:primary-signature -v "$DMG"
echo "Done: $DMG"

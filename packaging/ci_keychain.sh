#!/usr/bin/env bash
# For GitHub Actions only: put the Developer ID certificate into a throwaway
# keychain so codesign can use it, or delete that keychain again.
#
#   packaging/ci_keychain.sh import   needs APPLE_CERTIFICATE_P12 (base64 of the
#                                     exported .p12) and APPLE_CERTIFICATE_PASSWORD
#   packaging/ci_keychain.sh delete
#
# The keychain lives in $RUNNER_TEMP with a random password, is added to the
# search list (so `security find-identity` and codesign see it) and is deleted
# when the job ends. Nothing is printed about the certificate's contents.
set -euo pipefail

KEYCHAIN="${RUNNER_TEMP:?run this on GitHub Actions}/frame-tidy-signing.keychain-db"

case "${1:-}" in
  import)
    : "${APPLE_CERTIFICATE_P12:?set the APPLE_CERTIFICATE_P12 secret}"
    : "${APPLE_CERTIFICATE_PASSWORD:?set the APPLE_CERTIFICATE_PASSWORD secret}"
    KEYCHAIN_PASSWORD="$(openssl rand -hex 24)"
    P12="$RUNNER_TEMP/frame-tidy-signing.p12"
    trap 'rm -f "$P12"' EXIT
    printf '%s' "$APPLE_CERTIFICATE_P12" | base64 --decode > "$P12"

    security create-keychain -p "$KEYCHAIN_PASSWORD" "$KEYCHAIN"
    security set-keychain-settings -lut 21600 "$KEYCHAIN"
    security unlock-keychain -p "$KEYCHAIN_PASSWORD" "$KEYCHAIN"
    security import "$P12" -k "$KEYCHAIN" -P "$APPLE_CERTIFICATE_PASSWORD" \
      -f pkcs12 -T /usr/bin/codesign -T /usr/bin/security
    # Let codesign use the key without a prompt.
    security set-key-partition-list -S apple-tool:,apple:,codesign: -s \
      -k "$KEYCHAIN_PASSWORD" "$KEYCHAIN" > /dev/null
    # Search this keychain as well as the usual ones.
    existing=()
    while IFS= read -r line; do
      line="${line#"${line%%[![:space:]]*}"}"
      line="${line%\"}"; line="${line#\"}"
      [ -n "$line" ] && existing+=("$line")
    done < <(security list-keychains -d user)
    security list-keychains -d user -s "$KEYCHAIN" "${existing[@]}"
    # Fail here, not halfway through signing, if the certificate is not usable.
    security find-identity -v -p codesigning "$KEYCHAIN" | grep -q "Developer ID Application" || {
      echo "The certificate imported, but no 'Developer ID Application' identity is usable." >&2
      exit 1
    }
    echo "Signing certificate ready."
    ;;
  delete)
    security delete-keychain "$KEYCHAIN" 2>/dev/null || true
    ;;
  *)
    echo "usage: $0 import|delete" >&2
    exit 2
    ;;
esac

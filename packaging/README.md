# Packaging Frame Tidy

Frame Tidy is a working name. The app is the review page in `src/aura_mcp/`
wrapped by `src/aura_mcp/app.py` (chooses a free port, keeps data in the
per-user application folder, opens its own window).

## Run without building

    uv run aura-app                 # own window
    uv run aura-app --browser       # default browser instead
    uv run aura-app --no-window     # serve only (Ctrl-C to stop), for testing

Data lives in `~/Library/Application Support/Frame Tidy` (macOS), `%APPDATA%\Frame Tidy`
(Windows), `~/.local/share/Frame Tidy` (Linux), unless `AURA_MCP_HOME` is set.
To test safely against the mock Aura service, set `AURA_MCP_HOME` to a scratch
folder, `AURA_API_BASE_URL` / `AURA_IMAGE_BASE_URL` to the mock, and unset
`AURA_EMAIL` / `AURA_PASSWORD`.

## Build (macOS)

    packaging/build_mac.sh               # Universal: Apple silicon and Intel in one app
    ARCH=arm64  packaging/build_mac.sh   # Apple silicon only (quicker, for testing)
    ARCH=x86_64 packaging/build_mac.sh   # Intel only

Produces `dist/Frame Tidy.app` and `dist/Frame Tidy.dmg`. The Universal app
(about 700 MB installed) is the Apple-silicon and Intel builds joined by
`packaging/make_universal.py`: each binary in both is merged with `lipo`, and
OpenCV's helper libraries, which have different names in its two builds, are
both kept. On an Apple-silicon Mac the Intel build runs through Rosetta
(`softwareupdate --install-rosetta` once). The joined app is signed ad hoc so it
runs locally; `sign_and_notarize.sh` replaces that with the real signature.

### Older Macs

The app runs on macOS 12.3 and later, on Apple silicon and Intel. The build
installs the locked libraries into a separate environment per kind of Mac
(`build/venv-<arch>`) with `MACOSX_DEPLOYMENT_TARGET=12.0`, so each library's
macOS 12 build is used. `packaging/check_app.py` then fails the build if any
binary is for the wrong kind of Mac, if the main program does not hold every
kind, or if anything needs a newer macOS than `LSMinimumSystemVersion` in
`packaging/frame_tidy.spec` (12.3, set by SciPy).

Some libraries dropped macOS 12 or Intel in later releases, so `pyproject.toml`
holds them back on macOS (Python 3.12): onnxruntime 1.19, opencv 4.10, onnx 1.22
and cryptography 48. Development and tests use the same versions. Before
raising any of them, check that the new release still has macOS 12 builds for
both `arm64` and `x86_64`, or raise the minimum macOS for everyone.
The face model is not inside: InsightFace downloads it into `~/.insightface`
on first use. The script fails if any `.onnx` file ends up in the bundle.
PyInstaller ad-hoc signs the app, which is enough to run on the machine that built it;
other Macs will show an "unidentified developer" warning until it is notarized.

## Sign and notarize (needs a paid Apple Developer ID)

### On your Mac

Needs the "Developer ID Application" certificate in the login keychain (with
Apple's "Developer ID Certification Authority (G2)" certificate beside it if
Xcode is not installed) and a notarytool keychain profile named `frame-tidy`,
made once with
`xcrun notarytool store-credentials frame-tidy --apple-id <you> --team-id <TEAMID>`.

    packaging/build_mac.sh
    APPLE_TEAM_ID=<TEAMID> packaging/sign_and_notarize.sh

It signs every binary with the hardened runtime, notarizes the app, staples
it, checks it with Gatekeeper, rebuilds the dmg, then signs, notarizes and
staples that. First run 2026-11-20: Apple accepted the app and the dmg. The
bundle id is `com.walensis.frametidy`; keep it unchanged between releases.

### In GitHub Actions

`.github/workflows/build.yml` runs the same script when these are set under
Settings > Secrets and variables > Actions. A release tag fails rather than
publish an unsigned dmg. Set up once:

1. In Keychain Access, find "Developer ID Application: … (TEAMID)" under
   My Certificates, right-click > Export, and save it as a `.p12` with a strong
   password.
2. Add the secret `APPLE_CERTIFICATE_P12`:
   `base64 -i DeveloperID.p12 | gh secret set APPLE_CERTIFICATE_P12`.
3. Add the secrets `APPLE_CERTIFICATE_PASSWORD` (the export password),
   `APPLE_ID` (the Apple ID email) and `APPLE_APP_PASSWORD` (an app-specific
   password from account.apple.com), and the *variable* `APPLE_TEAM_ID`.
4. Keep the `.p12` and its password in a password manager, and delete the file.

Then run the workflow by hand (Actions > build > Run workflow) and check the
artifact opens on another Mac before tagging a release.

## Automatic updates (Sparkle)

The Universal app bundles [Sparkle](https://sparkle-project.org) 2.10.1
(`packaging/fetch_sparkle.sh`, pinned by checksum). Once a day it reads
`https://github.com/walensis-labs/frame-tidy/releases/latest/download/appcast.xml`,
the appcast attached to the newest release, and offers the update with its
release notes; Help > Check for Updates… looks at once. It installs an update
only if the dmg is signed with the update key (`SUPublicEDKey`, from
`packaging/sparkle_public_key.txt`) and the app inside is signed by the same
Apple developer. Without the public key file the app is built without updates.
Code: `src/aura_mcp/updater.py`; settings: `packaging/frame_tidy.spec`.

**The update key** (once, by the owner): `build/sparkle/bin/generate_keys`
creates it in your login keychain and prints the public key; save that line in
`packaging/sparkle_public_key.txt` and commit it. For GitHub Actions, export the
private key with `build/sparkle/bin/generate_keys -x update.key`, add the file's
contents as the secret `SPARKLE_PRIVATE_KEY`, keep a copy in a password manager
and delete the file. Losing the private key means installed copies can no
longer update themselves (they would need one manual install of a build with a
new key), so back it up.

**Testing an update locally:** build two versions with a throwaway key
(`SPARKLE_PUBLIC_KEY=<test public key>`), sign them with `NOTARIZE=0`, make an
appcast for the newer one with `make_appcast.py --key-file <test key> --url
http://localhost:<port>/...`, serve both files, and start the older app with
`AURA_UPDATE_FEED=http://localhost:<port>/appcast.xml`.

## Release

1. Set the version in `pyproject.toml` (the app reads it from there) and write
   `docs/release-notes/<version>.md`: it is the release text on GitHub and what
   Sparkle shows in its update window (paragraphs, `##` headings, `-` lists,
   **bold**, `code` and links).
2. Commit, then tag and push: `git tag v0.1.4 && git push origin v0.1.4`.
3. `.github/workflows/build.yml` builds the Universal app, signs and notarizes
   it, signs the update and writes `appcast.xml`, and publishes the release
   with both. It refuses a tag when a signing secret, the update key, the
   public key file or the notes are missing, or the tag does not match the
   version. The landing page and installed copies pick up the new release on
   their own.

By hand instead (on your Mac, with the keychain profile and update key):
`packaging/build_mac.sh`, `APPLE_TEAM_ID=… packaging/sign_and_notarize.sh`,
rename the dmg to `Frame-Tidy-<version>.dmg`, then
`python3 packaging/make_appcast.py dist/Frame-Tidy-<version>.dmg <version>
docs/release-notes/<version>.md dist/appcast.xml`, and attach both files to a
release named `v<version>`.

## Windows

Built by CI only, never run by us. pywebview uses Edge WebView2 there; if it is
missing the app falls back to the default browser.

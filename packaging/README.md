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

    packaging/build_mac.sh               # this Mac's kind
    ARCH=arm64  packaging/build_mac.sh   # Apple silicon
    ARCH=x86_64 packaging/build_mac.sh   # Intel; works on Apple silicon through Rosetta

Produces `dist/Frame Tidy.app` and `dist/Frame Tidy.dmg` for one kind of Mac
(Apple silicon about 315 MB / 135 MB, Intel about 390 MB / 165 MB). Build one
kind, sign it, and move the dmg away before building the other: both use `dist/`.

### Older Macs

The app runs on macOS 12.3 and later, on Apple silicon and Intel. The build
installs the locked libraries into a separate environment per kind of Mac
(`build/venv-<arch>`) with `MACOSX_DEPLOYMENT_TARGET=12.0`, so each library's
macOS 12 build is used, and then fails if any binary in the app is for the
wrong kind of Mac or needs a newer macOS than `LSMinimumSystemVersion` in
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

## Release

1. Bump the version in `pyproject.toml` (the app bundle reads it from there).
2. Tag and push: `git tag v0.2.0 && git push origin v0.2.0`.
3. `.github/workflows/build.yml` builds, signs and notarizes the macOS dmg and
   builds a Windows zip, and attaches them to a release. A tag fails if the
   signing secrets are missing. The Windows job is untested.

## Windows

Built by CI only, never run by us. pywebview uses Edge WebView2 there; if it is
missing the app falls back to the default browser.

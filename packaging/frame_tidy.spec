# PyInstaller spec shared by the macOS and Windows builds.
# Run from the repository root:  uv run --group build pyinstaller packaging/frame_tidy.spec
import os
import sys
import tomllib
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

ROOT = Path(SPECPATH).parent  # noqa: F821  (SPECPATH is provided by PyInstaller)
NAME = "Frame Tidy"
# One version number, from pyproject.toml, for the app bundle and the update check.
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]

# Modules loaded by name at run time, or imported lazily, which PyInstaller
# may not see:
#   - uvicorn picks its loop, protocol and lifespan modules by name; the phone
#     listener uses lifespan="off", the page server "on" (so: all of uvicorn);
#   - segno draws the pairing QR code, imported when "Use on my phone" is pressed;
#   - fastembed (photo search), huggingface_hub (its download), insightface
#     (faces) and cv2 are imported inside functions.
hidden = (
    collect_submodules("insightface.app")
    + collect_submodules("insightface.model_zoo")
    + collect_submodules("insightface.utils")
    + collect_submodules("uvicorn")
    + collect_submodules("segno")
    + ["fastembed", "huggingface_hub", "cv2", "segno"]
)
if sys.platform == "win32":
    hidden += ["webview.platforms.edgechromium"]

datas = [(str(ROOT / "src/aura_mcp/review.html"), "aura_mcp")]
datas += collect_data_files("fastembed")
datas += collect_data_files("segno")
# Frozen, insightface looks for its data files in "objects" at the top of the
# bundle; without them it prints "[Error] File not found: ...meanshape_68.pkl".
datas += [(src, "objects") for src, dest in collect_data_files("insightface") if dest.endswith("objects")]
# importlib.metadata.version("aura-mcp") is how the app knows its version
# (GET /api/version and the update notice); without this it says 0.0.0.
datas += copy_metadata("aura-mcp")
for name in ("fastembed", "huggingface_hub", "onnxruntime"):
    datas += copy_metadata(name)

# The app never imports these; they only get pulled in by optional hooks.
excludes = [
    "tkinter", "matplotlib", "IPython", "jupyter", "notebook", "pytest",
    "torch", "tensorflow", "pandas", "sphinx", "PyQt5", "PyQt6", "PySide2", "PySide6",
    "albumentations", "tensorboard",
]

a = Analysis(  # noqa: F821
    [str(ROOT / "packaging/entry.py")],
    pathex=[str(ROOT / "src")],
    datas=datas,
    hiddenimports=hidden,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821
exe = EXE(  # noqa: F821
    pyz, a.scripts, [],
    exclude_binaries=True,
    name=NAME,
    console=False,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name=NAME, upx=False)  # noqa: F821

# Sparkle, the macOS updater (see src/aura_mcp/updater.py). The public half of
# the update signing key is committed in packaging/sparkle_public_key.txt;
# SPARKLE_PUBLIC_KEY overrides it for a local test with a throwaway key.
# Without a key the app is built without automatic updates.
FEED_URL = "https://github.com/walensis-labs/frame-tidy/releases/latest/download/appcast.xml"
_key_file = ROOT / "packaging/sparkle_public_key.txt"
SPARKLE_KEY = os.environ.get("SPARKLE_PUBLIC_KEY") or (
    _key_file.read_text().strip() if _key_file.is_file() else "")
updates = {
    "SUFeedURL": FEED_URL,
    "SUEnableAutomaticChecks": True,
    "SUScheduledCheckInterval": 86400,
    "SUAllowsAutomaticUpdates": True,
} if SPARKLE_KEY else {}
if SPARKLE_KEY:
    updates["SUPublicEDKey"] = SPARKLE_KEY

if sys.platform == "darwin":
    app = BUNDLE(  # noqa: F821
        coll,
        name=f"{NAME}.app",
        icon=None,
        bundle_identifier="com.walensis.frametidy",  # never change after release: macOS treats a new id as a new app
        info_plist={
            "CFBundleDisplayName": NAME,
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "12.3",  # SciPy needs 12.3; build_mac.sh checks every binary against this
            **updates,
        },
    )

"""Automatic updates for the packaged macOS app, through Sparkle.

Sparkle (https://sparkle-project.org) is bundled in
``Frame Tidy.app/Contents/Frameworks/Sparkle.framework`` by
``packaging/build_mac.sh``. Its settings are in the app's Info.plist (set in
``packaging/frame_tidy.spec``):

    SUFeedURL         the appcast: the newest GitHub release's appcast.xml
    SUPublicEDKey     the public half of the update signing key; Sparkle
                      refuses any update not signed with the private half
    SUEnableAutomaticChecks, SUScheduledCheckInterval: once a day

Sparkle shows its own windows: "A new version is available", with the
release notes and an Install button, then downloads, checks and installs the
update and reopens the app. Users can also let it install updates by itself.

Nothing here runs from source or on Windows; there the page's update note
(``/api/version``) is the way people hear about new versions.

For testing, ``AURA_UPDATE_FEED`` points Sparkle at another appcast.
"""

from __future__ import annotations

import os
import plistlib
import sys
from pathlib import Path
from typing import Any

_controller: Any = None
_delegate: Any = None


def _contents() -> Path | None:
    """The app bundle's Contents folder, when running as the packaged macOS app."""
    if sys.platform != "darwin" or not getattr(sys, "frozen", False):
        return None
    contents = Path(sys.executable).resolve().parent.parent
    return contents if contents.name == "Contents" else None


def framework_path() -> Path | None:
    contents = _contents()
    if contents is None:
        return None
    path = contents / "Frameworks" / "Sparkle.framework"
    return path if path.is_dir() else None


def available() -> bool:
    """True when this copy can update itself: packaged, with Sparkle and a signing key."""
    contents = _contents()
    if contents is None or framework_path() is None:
        return False
    try:
        info = plistlib.loads((contents / "Info.plist").read_bytes())
    except (OSError, plistlib.InvalidFileException):
        return False
    return bool(info.get("SUFeedURL") and info.get("SUPublicEDKey"))


def start() -> bool:
    """Start Sparkle's scheduled checks. Call on the main thread, once the app is running."""
    global _controller, _delegate
    if _controller is not None:
        return True
    if not available():
        return False
    import objc
    from Foundation import NSBundle, NSObject

    bundle = NSBundle.bundleWithPath_(str(framework_path()))
    if bundle is None or not bundle.load():
        return False
    controller_class = objc.lookUpClass("SPUStandardUpdaterController")

    feed = os.environ.get("AURA_UPDATE_FEED")
    if feed:
        class FeedOverride(NSObject):  # SPUUpdaterDelegate, testing only
            def feedURLStringForUpdater_(self, updater: Any) -> str:
                return feed

        _delegate = FeedOverride.alloc().init()
    _controller = controller_class.alloc().initWithStartingUpdater_updaterDelegate_userDriverDelegate_(
        True, _delegate, None
    )
    return True


def check_now() -> None:
    """Look for an update now and show the result (for a "Check for Updates…" menu item)."""
    if _controller is not None:
        _controller.checkForUpdates_(None)


def on_main_thread(fn: Any) -> None:
    """Run ``fn`` on the Cocoa main thread (Sparkle's windows must be made there)."""
    from PyObjCTools import AppHelper

    AppHelper.callAfter(fn)

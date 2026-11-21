from aura_mcp import updater


def test_running_from_source_never_updates_itself():
    assert updater.framework_path() is None
    assert updater.available() is False
    assert updater.start() is False
    updater.check_now()  # harmless without Sparkle

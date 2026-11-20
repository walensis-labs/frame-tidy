import json

import pytest

from aura_mcp import migrations


def make_data(root):
    """A data folder from before versions were recorded."""
    (root / "cache" / "f1").mkdir(parents=True)
    (root / "cache" / "f1" / "index.json").write_text('{"a": {"file_name": "a.heic"}}')
    (root / "cache" / "f1" / "a.jpg").write_bytes(b"photo")
    (root / "people").mkdir()
    (root / "people" / "people.json").write_text('{"version": 1, "people": []}')
    (root / "session.json").write_text('{"token": "t"}')


def version_on_disk(root):
    return json.loads((root / migrations.VERSION_FILE).read_text())["version"]


def test_a_new_empty_folder_starts_at_the_current_version(tmp_path):
    result = migrations.upgrade(tmp_path, current=3, steps={})
    assert result == {"from": None, "to": 3, "backup": None}
    assert version_on_disk(tmp_path) == 3


def test_an_unrelated_folder_is_not_mistaken_for_data(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "README.md").write_text("hi")
    assert migrations.upgrade(tmp_path, current=2, steps={})["from"] is None


def test_data_from_before_versions_is_version_one(tmp_path):
    make_data(tmp_path)
    result = migrations.upgrade(tmp_path, current=1, steps={})
    assert result == {"from": 1, "to": 1, "backup": None}
    assert version_on_disk(tmp_path) == 1


def test_steps_run_in_order_after_a_backup(tmp_path):
    make_data(tmp_path)
    ran = []

    def to_2(root):
        ran.append(2)
        (root / "people" / "people.json").write_text('{"version": 2, "people": []}')

    def to_3(root):
        ran.append(3)
        assert version_on_disk(root) == 2  # each finished step is recorded at once

    result = migrations.upgrade(tmp_path, current=3, steps={2: to_2, 3: to_3})
    assert ran == [2, 3] and result["from"] == 1 and result["to"] == 3
    assert version_on_disk(tmp_path) == 3
    backup = result["backup"]
    assert backup.parent == tmp_path / "backups" and backup.name.startswith("before-v3-")
    assert json.loads((backup / "people" / "people.json").read_text())["version"] == 1
    assert (backup / "cache" / "f1" / "index.json").is_file() and (backup / "session.json").is_file()
    assert not (backup / "cache" / "f1" / "a.jpg").exists()  # photos are not copied


def test_only_the_missing_steps_run(tmp_path):
    make_data(tmp_path)
    (tmp_path / migrations.VERSION_FILE).write_text('{"version": 2}')
    ran = []
    migrations.upgrade(tmp_path, current=3, steps={2: lambda r: ran.append(2), 3: lambda r: ran.append(3)})
    assert ran == [3]


def test_a_failed_step_stops_and_leaves_the_last_good_version(tmp_path):
    make_data(tmp_path)

    def broken(root):
        raise RuntimeError("disk full")

    with pytest.raises(migrations.UpgradeFailed, match="backups") as caught:
        migrations.upgrade(tmp_path, current=3, steps={2: lambda r: None, 3: broken})
    assert version_on_disk(tmp_path) == 2
    assert caught.value.backup.is_dir()


def test_data_from_a_newer_app_is_left_alone(tmp_path):
    make_data(tmp_path)
    (tmp_path / migrations.VERSION_FILE).write_text('{"version": 5}')
    with pytest.raises(migrations.DataTooNew, match="newer version of Frame Tidy"):
        migrations.upgrade(tmp_path, current=3, steps={})
    assert version_on_disk(tmp_path) == 5 and not (tmp_path / "backups").exists()


def test_an_unreadable_version_file_is_not_guessed_at(tmp_path):
    make_data(tmp_path)
    (tmp_path / migrations.VERSION_FILE).write_text("{oops")
    with pytest.raises(migrations.UpgradeFailed, match="data-version.json"):
        migrations.upgrade(tmp_path, current=3, steps={})


def test_a_missing_step_is_a_programming_error(tmp_path):
    make_data(tmp_path)
    with pytest.raises(AssertionError):
        migrations.upgrade(tmp_path, current=3, steps={3: lambda r: None})


def test_the_shipped_steps_lead_to_the_current_version():
    assert sorted(migrations.STEPS) == list(range(2, migrations.CURRENT + 1))

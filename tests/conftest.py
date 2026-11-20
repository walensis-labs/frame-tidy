import pytest


@pytest.fixture(autouse=True)
def _no_update_check(monkeypatch):
    """The page's look-up of newer releases goes to GitHub: never in tests."""
    monkeypatch.setenv("AURA_NO_UPDATE_CHECK", "1")


@pytest.fixture(autouse=True)
def _search_model_elsewhere(monkeypatch, tmp_path_factory):
    """No test looks in, or downloads to, the real photo search model folder."""
    monkeypatch.setenv("AURA_SEARCH_MODEL_DIR", str(tmp_path_factory.mktemp("search-model")))

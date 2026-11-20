import numpy as np
import pytest

from aura_mcp.image_search import ImageSearcher, _unit


def vec(*values):
    v = np.zeros(512, dtype=np.float32)
    v[: len(values)] = values
    return v


def test_similarity_is_cosine_and_ignores_magnitude():
    assert ImageSearcher.similarity(vec(2, 0), vec(5, 0)) == pytest.approx(1.0)
    assert ImageSearcher.similarity(vec(2, 0), vec(0, 3)) == pytest.approx(0.0, abs=1e-6)
    assert ImageSearcher.similarity(vec(1, 1), vec(1, 0)) == pytest.approx(
        1 / np.sqrt(2), abs=1e-5
    )


def test_unit_leaves_a_zero_vector_alone():
    z = np.zeros(4, dtype=np.float32)
    assert np.array_equal(_unit(z), z)
    assert _unit(np.array([3.0, 4.0])) == pytest.approx([0.6, 0.8])


def test_query_embedding_rejects_empty_without_loading_the_model():
    # Must raise on the empty string before any (slow, 0.6 GB) model load.
    with pytest.raises(ValueError, match="must not be empty"):
        ImageSearcher().query_embedding("   ")


# ---------------------------------- the search model is fetched only after a yes

from aura_mcp import image_search  # noqa: E402
from aura_mcp.image_search import ImageSearchError  # noqa: E402


def fake_search_model(root):
    """What fastembed leaves in its folder once both halves are downloaded."""
    for model, files in image_search.MODEL_FILES.items():
        snapshot = root / ("models--" + model.replace("/", "--")) / "snapshots" / "abc123"
        snapshot.mkdir(parents=True, exist_ok=True)
        for name in files:
            (snapshot / name).write_bytes(b"x")


@pytest.fixture
def watched(monkeypatch):
    calls = {"load": [], "fetch": []}

    def load(root, local_only):
        calls["load"].append((str(root), local_only))
        return object(), object()

    def fetch(root, arrived=None):
        calls["fetch"].append(str(root))
        fake_search_model(root)

    monkeypatch.setattr(image_search, "_load_models", load)
    monkeypatch.setattr(image_search, "_fetch_models", fetch)
    return calls


def test_the_model_folder_can_be_set_for_tests(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_SEARCH_MODEL_DIR", str(tmp_path / "m"))
    assert image_search.default_model_root() == tmp_path / "m"
    assert ImageSearcher().model_root == tmp_path / "m"


def test_present_only_when_both_halves_are_there(tmp_path):
    searcher = ImageSearcher(model_root=tmp_path, download=False)
    assert searcher.model_present() is False
    fake_search_model(tmp_path)
    assert searcher.model_present() is True


def test_without_consent_nothing_is_fetched_or_loaded(tmp_path, watched):
    searcher = ImageSearcher(model_root=tmp_path / "m", download=False)
    with pytest.raises(ImageSearchError, match="needs"):
        searcher.query_embedding("a dog")
    assert watched == {"load": [], "fetch": []}
    assert not (tmp_path / "m").exists()


def test_the_mcp_tool_still_fetches_on_first_use(tmp_path, watched):
    searcher = ImageSearcher(model_root=tmp_path / "m")  # download=True, as the MCP tool has it
    searcher._ensure_loaded()
    assert watched["fetch"] == [str(tmp_path / "m")]
    assert watched["load"] == [(str(tmp_path / "m"), True)]


def test_download_model_reports_what_has_arrived(tmp_path, watched):
    searcher = ImageSearcher(model_root=tmp_path / "m", download=False)
    seen = []
    searcher.download_model(seen.append)
    assert searcher.model_present() and seen and seen[-1] > 0


def test_the_size_told_to_the_user_is_the_measured_one():
    assert 590_000_000 < image_search.MODEL_DOWNLOAD_BYTES < 630_000_000

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from aura_mcp.face_matcher import (
    FaceMatcher,
    ImageReadError,
    NoFaceFoundError,
    cosine_similarity,
)


def face(embedding, bbox=(0, 0, 10, 10)):
    emb = np.asarray(embedding, dtype=np.float32)
    return SimpleNamespace(normed_embedding=emb / np.linalg.norm(emb), bbox=np.array(bbox))


class FakeApp:
    def __init__(self, faces):
        self.faces = faces

    def get(self, image):
        return self.faces


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "photo.jpg"
    cv2.imwrite(str(path), np.zeros((32, 32, 3), dtype=np.uint8))
    return path


def test_cosine_similarity_handles_unnormalised_and_zero_vectors():
    sims = cosine_similarity(np.array([[2.0, 0.0], [0.0, 3.0], [0.0, 0.0]]), np.array([5.0, 0.0]))
    assert sims == pytest.approx([1.0, 0.0, 0.0])


def test_reference_embedding_raises_when_no_face(image):
    with pytest.raises(NoFaceFoundError, match="No face detected"):
        FaceMatcher(app=FakeApp([])).reference_embedding(image)


def test_reference_embedding_uses_largest_face(image):
    small = face([1, 0], bbox=(0, 0, 5, 5))
    large = face([0, 1], bbox=(0, 0, 50, 50))
    embedding, count = FaceMatcher(app=FakeApp([small, large])).reference_embedding(image)
    assert count == 2
    assert embedding == pytest.approx([0.0, 1.0])


def test_face_embeddings_returns_one_row_per_face(image):
    matcher = FaceMatcher(app=FakeApp([face([1, 0]), face([0, 1])]))
    assert matcher.face_embeddings(image).shape == (2, 2)
    assert FaceMatcher(app=FakeApp([])).face_embeddings(image).shape == (0, 512)


def test_best_similarity_takes_the_closest_face():
    faces = np.array([[0.0, 1.0], [0.8, 0.6]], dtype=np.float32)
    assert FaceMatcher.best_similarity(faces, np.array([1.0, 0.0])) == pytest.approx(0.8)
    assert FaceMatcher.best_similarity(np.zeros((0, 512)), np.array([1.0, 0.0])) is None


def test_missing_and_corrupt_images_raise_image_read_error(tmp_path):
    matcher = FaceMatcher(app=FakeApp([]))
    with pytest.raises(ImageReadError, match="not found"):
        matcher.face_embeddings(tmp_path / "nope.jpg")
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(ImageReadError, match="Could not decode"):
        matcher.face_embeddings(bad)


def test_detected_faces_pairs_each_embedding_with_its_box(tmp_path):
    photo = tmp_path / "wide.jpg"
    cv2.imwrite(str(photo), np.zeros((600, 800, 3), dtype=np.uint8))
    e = np.eye(512, dtype=np.float32)
    matcher = FaceMatcher(app=FakeApp([face(e[0], (80, 120, 240, 300)), face(e[1], (480, 120, 640, 300))]))
    faces = matcher.detected_faces(photo)
    assert [int(np.argmax(emb)) for emb, _ in faces] == [0, 1]
    assert faces[1][1] == pytest.approx({"x": 0.6, "y": 0.2, "w": 0.2, "h": 0.3})
    assert FaceMatcher(app=FakeApp([])).detected_faces(photo) == []


class FakeAnalysis:
    """Stands in for insightface's FaceAnalysis: records how it was built."""

    built = []
    fail_coreml = False

    def __init__(self, name, allowed_modules, providers, provider_options=None, root=None):
        self.root = root
        if providers[0] == "CoreMLExecutionProvider" and self.fail_coreml:
            raise RuntimeError("failed to create a CoreML session")
        self.providers = providers
        self.provider_options = provider_options
        self.models = {"detection": SimpleNamespace(session=SimpleNamespace(get_providers=lambda: providers))}
        FakeAnalysis.built.append(self)

    def prepare(self, ctx_id, det_size):
        self.ctx_id = ctx_id
        if ctx_id < 0:  # what insightface does: a negative ctx_id forces the CPU
            self.models["detection"].session.get_providers = lambda: ["CPUExecutionProvider"]

    def get(self, image):
        return []


@pytest.fixture
def analysis(monkeypatch):
    from aura_mcp import face_matcher

    FakeAnalysis.built, FakeAnalysis.fail_coreml = [], False
    monkeypatch.setattr(face_matcher, "_face_analysis_class", lambda: FakeAnalysis)
    return face_matcher


def test_mac_uses_coreml_on_cpu_and_gpu_and_reads_two_at_once(analysis, monkeypatch):
    monkeypatch.setattr(analysis.sys, "platform", "darwin")
    monkeypatch.setattr(analysis, "_available_providers",
                        lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"])
    matcher = FaceMatcher()
    assert matcher.reads_at_once == 1  # nothing loaded yet

    matcher._get_app()

    app = FakeAnalysis.built[-1]
    assert app.providers == ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    # The Neural Engine gives slightly different fingerprints; CPU+GPU does not.
    assert app.provider_options[0] == {"MLComputeUnits": "CPUAndGPU"}
    assert app.ctx_id == 0  # -1 would silently put it back on the CPU
    assert matcher.reads_at_once == 2


def test_falls_back_to_the_cpu_when_coreml_cannot_start(analysis, monkeypatch):
    monkeypatch.setattr(analysis.sys, "platform", "darwin")
    monkeypatch.setattr(analysis, "_available_providers",
                        lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"])
    FakeAnalysis.fail_coreml = True
    matcher = FaceMatcher()

    matcher._get_app()

    app = FakeAnalysis.built[-1]
    assert app.providers == ["CPUExecutionProvider"] and app.ctx_id == -1
    assert matcher.reads_at_once == 1


def test_elsewhere_the_cpu_is_used(analysis, monkeypatch):
    monkeypatch.setattr(analysis.sys, "platform", "win32")
    monkeypatch.setattr(analysis, "_available_providers",
                        lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"])
    matcher = FaceMatcher()
    matcher._get_app()
    assert FakeAnalysis.built[-1].providers == ["CPUExecutionProvider"]
    assert matcher.reads_at_once == 1


def test_reading_on_several_threads_leaves_stdout_alone(image):
    import sys
    from concurrent.futures import ThreadPoolExecutor

    before = sys.stdout
    matcher = FaceMatcher(app=FakeApp([face([1, 0])]))
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda _: matcher.face_embeddings(image), range(200)))
    assert sys.stdout is before


# ------------------------------------------------ the model is fetched on consent


def fake_model(folder, files=("det_10g.onnx", "w600k_r50.onnx")):
    folder.mkdir(parents=True, exist_ok=True)
    for name in files:
        (folder / name).write_bytes(b"onnx")


def test_the_model_folder_can_be_pointed_elsewhere(tmp_path, monkeypatch):
    from aura_mcp import face_matcher

    monkeypatch.setenv("AURA_MODEL_DIR", str(tmp_path / "models-here"))
    assert face_matcher.default_model_root() == tmp_path / "models-here"
    matcher = FaceMatcher()
    assert matcher.model_dir == tmp_path / "models-here" / "models" / "buffalo_l"
    assert FaceMatcher(model_root=tmp_path / "x").model_dir == tmp_path / "x" / "models" / "buffalo_l"
    monkeypatch.delenv("AURA_MODEL_DIR")
    assert face_matcher.default_model_root() == Path.home() / ".insightface"


def test_model_present_needs_the_detector_and_the_recogniser(tmp_path):
    matcher = FaceMatcher(model_root=tmp_path)
    assert not matcher.model_present()
    fake_model(matcher.model_dir, files=("det_10g.onnx",))  # half an unzip
    assert not matcher.model_present()
    fake_model(matcher.model_dir)
    assert matcher.model_present()
    # An injected app (tests) counts as present.
    assert FaceMatcher(app=FakeApp([]), model_root=tmp_path / "empty").model_present()


def test_without_permission_to_download_nothing_is_loaded(tmp_path, monkeypatch):
    from aura_mcp import face_matcher

    def never(*args, **kwargs):
        raise AssertionError("the model must not be loaded or fetched")

    monkeypatch.setattr(face_matcher, "_face_analysis_class", never)
    monkeypatch.setattr(face_matcher, "_fetch_model", never)
    matcher = FaceMatcher(model_root=tmp_path, download=False)

    with pytest.raises(face_matcher.ModelMissingError):
        matcher._get_app()
    assert not (tmp_path / "models").exists()


def test_the_model_is_loaded_from_the_chosen_folder(analysis, monkeypatch, tmp_path):
    monkeypatch.setattr(analysis.sys, "platform", "linux")
    matcher = FaceMatcher(model_root=tmp_path, download=False)
    fake_model(matcher.model_dir)
    matcher._get_app()
    assert FakeAnalysis.built[-1].root == str(tmp_path)


def test_download_model_reports_bytes_and_leaves_the_files(tmp_path, monkeypatch):
    from aura_mcp import face_matcher

    seen = []

    def fake_fetch(name, root, force):
        # What insightface does: write the zip a chunk at a time, then unpack.
        zip_path = Path(root) / "models" / f"{name}.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with open(zip_path, "wb") as handle:
            for _ in range(3):
                handle.write(b"x" * 1000)
                handle.flush()
                import time
                time.sleep(0.15)
        fake_model(Path(root) / "models" / name)
        seen.append(force)

    monkeypatch.setattr(face_matcher, "_fetch_model", fake_fetch)
    matcher = FaceMatcher(model_root=tmp_path, download=False)
    sizes = []
    matcher.download_model(sizes.append, interval=0.05)

    assert matcher.model_present()
    assert seen == [False]
    assert sizes and max(sizes) == 3000 and sizes == sorted(sizes)


def test_a_half_unpacked_model_is_fetched_again(tmp_path, monkeypatch):
    from aura_mcp import face_matcher

    seen = []
    monkeypatch.setattr(
        face_matcher, "_fetch_model",
        lambda name, root, force: (seen.append(force), fake_model(Path(root) / "models" / name)))
    matcher = FaceMatcher(model_root=tmp_path, download=False)
    fake_model(matcher.model_dir, files=("det_10g.onnx",))
    matcher.download_model()
    assert seen == [True] and matcher.model_present()


def test_a_failed_download_says_so_plainly(tmp_path, monkeypatch):
    from aura_mcp import face_matcher

    def broken(name, root, force):
        raise RuntimeError("Failed downloading url https://example/buffalo_l.zip")

    monkeypatch.setattr(face_matcher, "_fetch_model", broken)
    matcher = FaceMatcher(model_root=tmp_path, download=False)
    with pytest.raises(face_matcher.FaceMatcherError) as caught:
        matcher.download_model()
    assert "could not be downloaded" in str(caught.value)
    assert "Failed downloading" in str(caught.value.__cause__)

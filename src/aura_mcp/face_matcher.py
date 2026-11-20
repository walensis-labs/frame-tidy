"""Face embedding extraction and comparison using InsightFace (buffalo_l)."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np

MODEL_NAME = "buffalo_l"
DET_SIZE = (640, 640)

# Where InsightFace keeps its models (it adds models/<name>/). AURA_MODEL_DIR
# points it elsewhere, which tests and trial runs use to start from nothing.
MODEL_DIR_ENV = "AURA_MODEL_DIR"
# The two files face finding needs: the detector and the recogniser. The
# other three in the pack (landmarks, age/gender) are never loaded.
MODEL_FILES = ("det_10g.onnx", "w600k_r50.onnx")
# Size of InsightFace's buffalo_l.zip, for the download progress.
MODEL_DOWNLOAD_BYTES = 288_621_354

COREML = "CoreMLExecutionProvider"
CPU = "CPUExecutionProvider"
# CoreML on the CPU and GPU (not the Neural Engine) gives the same face
# fingerprints as the CPU (cosine >= 0.9999 on LFW test photos) about eight
# times faster on an M1. The Neural Engine's lower precision moved some
# fingerprints to 0.997, so it is left out.
COREML_OPTIONS = {"MLComputeUnits": "CPUAndGPU"}
# With CoreML two photos read at once finish about 1.7 times as fast as one
# at a time; on the CPU alone reading in parallel did not help.
ACCELERATED_READS = 2


def _face_analysis_class() -> Any:
    from insightface.app import FaceAnalysis

    return FaceAnalysis


def _fetch_model(name: str, root: str, force: bool) -> None:
    """InsightFace's own download: its release zip, unpacked into root/models/name."""
    from insightface.utils.storage import download

    download("models", name, force=force, root=root)


def default_model_root() -> Path:
    override = os.environ.get(MODEL_DIR_ENV)
    return Path(override).expanduser() if override else Path.home() / ".insightface"


def _available_providers() -> list[str]:
    import onnxruntime

    return list(onnxruntime.get_available_providers())


class _Quiet:
    """Send stdout to stderr while any thread is inside, thread-safely.

    ``contextlib.redirect_stdout`` swaps a global; two threads using it at
    once can leave stdout pointing at stderr for good, which under the stdio
    MCP transport would cut the protocol off.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._depth = 0
        self._saved: Any = None

    def __enter__(self) -> None:
        with self._lock:
            if self._depth == 0:
                self._saved, sys.stdout = sys.stdout, sys.stderr
            self._depth += 1

    def __exit__(self, *exc: object) -> None:
        with self._lock:
            self._depth -= 1
            if self._depth == 0:
                sys.stdout, self._saved = self._saved, None


_quiet = _Quiet()


class FaceMatcherError(Exception):
    """Base class for face matching errors."""


class ImageReadError(FaceMatcherError):
    """The file is missing or is not a decodable image."""


class NoFaceFoundError(FaceMatcherError):
    """No face could be detected in the reference image."""


class ModelMissingError(FaceMatcherError):
    """The face model is not on this computer, and fetching it was not agreed to."""


def cosine_similarity(embeddings: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Cosine similarity of each row of ``embeddings`` against ``reference``."""
    embeddings = np.atleast_2d(np.asarray(embeddings, dtype=np.float32))
    reference = np.asarray(reference, dtype=np.float32)
    denom = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(reference)
    return (embeddings @ reference) / np.where(denom == 0, 1.0, denom)


class FaceMatcher:
    """Lazy wrapper around ``insightface.app.FaceAnalysis``.

    The model (~280 MB) lives in ``model_root`` (``~/.insightface`` unless
    ``AURA_MODEL_DIR`` says otherwise) and is loaded once per process. With
    ``download=True`` (the MCP tools) InsightFace fetches it on first use if
    it is missing. With ``download=False`` (the review page) a missing model
    raises ``ModelMissingError`` instead, and only ``download_model``, called
    once the user has agreed, fetches it. ``app`` can be injected for tests.

    On a Mac it runs through CoreML (see ``COREML_OPTIONS``), falling back to
    the CPU if CoreML cannot start. ``reads_at_once`` says how many photos
    are worth reading at the same time once the model is loaded.
    """

    reads_at_once = 1

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        app: Any | None = None,
        model_root: str | Path | None = None,
        download: bool = True,
    ):
        self._model_name = model_name
        self._app = app
        self._lock = threading.Lock()
        self.model_root = Path(model_root).expanduser() if model_root else default_model_root()
        self.download = download

    @property
    def model_dir(self) -> Path:
        return self.model_root / "models" / self._model_name

    def model_present(self) -> bool:
        """Whether the model can be loaded without fetching anything."""
        if self._app is not None:
            return True
        return all((self.model_dir / name).is_file() for name in MODEL_FILES)

    def download_model(
        self, on_bytes: Callable[[int], None] | None = None, interval: float = 0.5
    ) -> None:
        """Fetch the model with InsightFace's own downloader (blocking).

        ``on_bytes`` is told how much of the zip has arrived so far, read
        from the size of the file InsightFace is writing.
        """
        zip_path = self.model_root / "models" / f"{self._model_name}.zip"
        # A folder left half unpacked would be taken as complete; fetch again.
        force = self.model_dir.exists() and not self.model_present()
        done = threading.Event()

        def watch() -> None:
            last = -1
            while not done.wait(interval):
                try:
                    size = zip_path.stat().st_size
                except OSError:
                    continue
                if on_bytes and size > last:
                    last = size
                    on_bytes(size)

        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        try:
            with _quiet:
                _fetch_model(self._model_name, str(self.model_root), force)
        except Exception as exc:  # noqa: BLE001 - network, disk, a bad zip...
            raise FaceMatcherError(
                "The face model could not be downloaded. Check this computer's "
                "internet connection and free disk space, then try again."
            ) from exc
        finally:
            done.set()
            watcher.join(timeout=2)
        if on_bytes and zip_path.is_file():
            on_bytes(zip_path.stat().st_size)
        if not self.model_present():
            raise FaceMatcherError(
                "The face model was downloaded but some of its files are missing. Try again."
            )

    def _get_app(self) -> Any:
        with self._lock:
            if self._app is None:
                if not self.download and not self.model_present():
                    raise ModelMissingError(
                        "The face model is not on this computer yet. Open Frames to download it."
                    )
                # InsightFace prints to stdout while loading. Under the stdio
                # MCP transport stdout carries protocol messages, so anything
                # it prints is sent to stderr instead.
                with _quiet:
                    app = None
                    if sys.platform == "darwin" and COREML in _available_providers():
                        try:
                            app = self._load([COREML, CPU], [COREML_OPTIONS, {}], ctx_id=0)
                        except Exception:  # noqa: BLE001 - any CoreML failure: use the CPU
                            app = None
                        if app is not None and not self._runs_on(app, COREML):
                            app = None
                    if app is None:
                        app = self._load([CPU], None, ctx_id=-1)
                    else:
                        self.reads_at_once = ACCELERATED_READS
                self._app = app
            return self._app

    def _load(self, providers: list[str], options: list[dict[str, str]] | None, ctx_id: int) -> Any:
        extra = {"provider_options": options} if options else {}
        app = _face_analysis_class()(
            name=self._model_name,
            root=str(self.model_root),
            allowed_modules=["detection", "recognition"],
            providers=providers,
            **extra,
        )
        # A negative ctx_id makes InsightFace move every model to the CPU.
        app.prepare(ctx_id=ctx_id, det_size=DET_SIZE)
        return app

    @staticmethod
    def _runs_on(app: Any, provider: str) -> bool:
        """Whether every loaded model really runs on ``provider``."""
        try:
            return all(
                model.session.get_providers()[0] == provider for model in app.models.values()
            )
        except (AttributeError, IndexError, TypeError):
            return False

    @staticmethod
    def _read(image_path: str | Path) -> np.ndarray:
        import cv2

        path = Path(image_path).expanduser()
        if not path.is_file():
            raise ImageReadError(f"Image not found: {path}")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ImageReadError(f"Could not decode image: {path}")
        return image

    def _detect(self, image_path: str | Path) -> list[Any]:
        image = self._read(image_path)
        app = self._get_app()
        with _quiet:
            return list(app.get(image))

    def face_boxes(self, image_path: str | Path) -> list[dict[str, float]]:
        """Where the faces are, as fractions of the image width and height."""
        image = self._read(image_path)
        height, width = image.shape[:2]
        app = self._get_app()
        with _quiet:
            faces = list(app.get(image))
        boxes = [self._fraction_box(face.bbox, width, height) for face in faces]
        return sorted(boxes, key=lambda b: b["x"])

    @staticmethod
    def _fraction_box(bbox: Any, width: int, height: int) -> dict[str, float]:
        x1, y1, x2, y2 = (float(v) for v in bbox)
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(width), x2), min(float(height), y2)
        return {
            "x": x1 / width,
            "y": y1 / height,
            "w": (x2 - x1) / width,
            "h": (y2 - y1) / height,
        }

    def detected_faces(self, image_path: str | Path) -> list[tuple[np.ndarray, dict[str, float]]]:
        """Each face found: its embedding and where it is (as face_boxes gives it).

        In the order the detector reports them, which is the order of the rows
        that face_embeddings returns.
        """
        image = self._read(image_path)
        height, width = image.shape[:2]
        app = self._get_app()
        with _quiet:
            faces = list(app.get(image))
        return [
            (
                np.asarray(face.normed_embedding, dtype=np.float32),
                self._fraction_box(face.bbox, width, height),
            )
            for face in faces
        ]

    @classmethod
    def save_face_crop(
        cls, image_path: str | Path, box: dict[str, float], dest: Path, pad: float = 0.6
    ) -> tuple[float, float]:
        """Save the region around one face (``box`` as returned by face_boxes).

        ``pad`` is extra margin on every side, as a fraction of the face size,
        so the detector can find the face again in the crop. Returns where the
        centre of that face lies in the saved crop, in pixels.
        """
        import cv2

        image = cls._read(image_path)
        height, width = image.shape[:2]
        x, y = box["x"] * width, box["y"] * height
        w, h = box["w"] * width, box["h"] * height
        if w <= 1 or h <= 1:
            raise FaceMatcherError("That face region is empty.")
        x1, y1 = int(max(0, x - w * pad)), int(max(0, y - h * pad))
        x2, y2 = int(min(width, x + w * (1 + pad))), int(min(height, y + h * (1 + pad)))
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(dest), image[y1:y2, x1:x2]):
            raise FaceMatcherError(f"Could not write {dest}")
        return x + w / 2 - x1, y + h / 2 - y1

    def faces_in(self, image_path: str | Path) -> list[tuple[float, float, float, float]]:
        """Pixel boxes ``(x1, y1, x2, y2)`` of the faces found, largest first."""
        boxes = [tuple(float(v) for v in face.bbox) for face in self._detect(image_path)]
        return sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)

    def face_embeddings(self, image_path: str | Path) -> np.ndarray:
        """L2-normalised embeddings for every face found, shape ``(n, 512)``."""
        faces = self._detect(image_path)
        if not faces:
            return np.zeros((0, 512), dtype=np.float32)
        return np.stack([np.asarray(f.normed_embedding, dtype=np.float32) for f in faces])

    def reference_embedding(self, image_path: str | Path) -> tuple[np.ndarray, int]:
        """Embedding of the target person, plus how many faces the image held.

        If the reference shows several people, the largest face is used.
        """
        faces = self._detect(image_path)
        if not faces:
            raise NoFaceFoundError(
                f"No face detected in reference image {image_path}. "
                "Use a clear, front-facing photo of the person."
            )

        def area(face: Any) -> float:
            x1, y1, x2, y2 = face.bbox
            return float((x2 - x1) * (y2 - y1))

        largest = max(faces, key=area)
        return np.asarray(largest.normed_embedding, dtype=np.float32), len(faces)

    @staticmethod
    def best_similarity(embeddings: np.ndarray, reference: np.ndarray) -> float | None:
        """Highest similarity of any face to the reference; None if no faces."""
        if len(embeddings) == 0:
            return None
        return float(cosine_similarity(embeddings, reference).max())

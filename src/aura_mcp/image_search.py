"""Open-vocabulary photo search with CLIP (ViT-B/32, ONNX via fastembed).

Finds photos that match a plain-language description -- "a desk", "a birthday
cake", "a receipt" -- so a frame can be curated the same way it is cleaned of a
person. CLIP's image and text encoders share one 512-d space, so a photo's fit
to a phrase is just the cosine between the photo's image embedding and the
phrase's text embedding.

Two deliberate choices:

* ONNX runtime via ``fastembed``, no torch -- same stack as the face model.
* CLIP is MIT-licensed (unlike the non-commercial face model), so this search
  is safe to ship in a paid tier.

CLIP similarities sit in a narrow positive band (roughly 0.15-0.32), much lower
than the face model's, so callers should think in terms of ranking and a low
threshold, not the 0.45 used for faces.
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np

VISION_MODEL = "Qdrant/clip-ViT-B-32-vision"
TEXT_MODEL = "Qdrant/clip-ViT-B-32-text"

# Where the two models are kept (fastembed's Hugging Face cache layout).
# AURA_SEARCH_MODEL_DIR overrides it (tests point it at a temporary folder).
MODEL_DIR_ENV = "AURA_SEARCH_MODEL_DIR"
# The files each half needs, inside models--<org>--<name>/snapshots/<rev>/.
MODEL_FILES = {
    VISION_MODEL: ("model.onnx", "preprocessor_config.json"),
    TEXT_MODEL: ("model.onnx", "tokenizer.json"),
}
# Measured: both halves as fastembed 0.9 keeps them, 608,019,462 bytes
# (the listing says 0.34 + 0.25 GB).
MODEL_DOWNLOAD_BYTES = 608_019_462
MODEL_NEEDED = (
    "Finding photos by description needs the photo search model first. "
    "Press Find on the computer to download it."
)


def default_model_root() -> Path:
    override = os.environ.get(MODEL_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".frame-tidy" / "search-model"


def _load_models(root: Path, local_only: bool) -> tuple[Any, Any]:
    """Build both models from ``root``; with ``local_only``, never reach the network."""
    from fastembed import ImageEmbedding, TextEmbedding

    kwargs = {"cache_dir": str(root), "local_files_only": local_only}
    return ImageEmbedding(VISION_MODEL, **kwargs), TextEmbedding(TEXT_MODEL, **kwargs)


def _fetch_models(root: Path, arrived: Callable[[int], None] | None = None) -> None:
    """Download both models from Hugging Face into ``root`` (fastembed's own download)."""
    _load_models(root, local_only=False)


def _folder_bytes(root: Path) -> int:
    total = 0
    for folder, _, files in os.walk(root):
        for name in files:
            path = Path(folder) / name
            if not path.is_symlink():
                try:
                    total += path.stat().st_size
                except OSError:
                    pass
    return total

# Prompt ensembling: averaging a few phrasings of the same query is a standard
# CLIP trick that steadies scores against wording.
QUERY_TEMPLATES = ("{q}", "a photo of {q}", "a photo containing {q}")


class ImageSearchError(Exception):
    """Base class for errors from the open-vocabulary searcher."""


class ImageReadError(ImageSearchError):
    """A photo could not be read or embedded."""


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


class ImageSearcher:
    """CLIP image/text embeddings for open-vocabulary photo search.

    The ONNX models (about 610 MB) load lazily on first use and print download and
    progress noise to stdout, which would corrupt the MCP stdio channel, so
    every call that touches them redirects stdout to stderr.
    """

    def __init__(self, model_root: str | Path | None = None, download: bool = True) -> None:
        self._img = None
        self._txt = None
        self.model_root = Path(model_root).expanduser() if model_root else default_model_root()
        # The review page passes download=False: the models are fetched only
        # after the user agrees (see download_model). The MCP tool fetches
        # them on first use, as its description says.
        self.download = download
        self._lock = threading.Lock()

    def model_present(self) -> bool:
        """Whether both halves of the model are in ``model_root``."""
        for model, files in MODEL_FILES.items():
            snapshots = self.model_root / ("models--" + model.replace("/", "--")) / "snapshots"
            if not snapshots.is_dir() or not any(
                all((snap / name).is_file() for name in files) for snap in snapshots.iterdir()
            ):
                return False
        return True

    def download_model(self, arrived: Callable[[int], None] | None = None) -> None:
        """Fetch the model into ``model_root``. ``arrived(bytes)`` is called as it lands."""
        self.model_root.mkdir(parents=True, exist_ok=True)
        done = threading.Event()

        def watch() -> None:
            while not done.wait(0.5):
                if arrived is not None:
                    arrived(_folder_bytes(self.model_root))

        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        try:
            with contextlib.redirect_stdout(sys.stderr):
                _fetch_models(self.model_root, arrived)
        except Exception as exc:  # noqa: BLE001 - surface a readable message
            raise ImageSearchError(
                "The photo search model could not be downloaded. Check this computer's "
                "internet connection and try again."
            ) from exc
        finally:
            done.set()
            watcher.join(2)
        if arrived is not None:
            arrived(_folder_bytes(self.model_root))
        if not self.model_present():
            raise ImageSearchError("The photo search model did not download completely. Try again.")

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._img is not None:
                return
            present = self.model_present()
            if not present and not self.download:
                raise ImageSearchError(MODEL_NEEDED)
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    if not present:
                        _fetch_models(self.model_root)
                    self._img, self._txt = _load_models(self.model_root, local_only=True)
            except Exception as exc:  # noqa: BLE001 - surface a readable message
                raise ImageSearchError(f"Could not load the CLIP model: {exc}") from exc

    # ---------------------------------------------------------------- images

    def image_embedding(self, path: str | Path) -> np.ndarray:
        """A single photo's unit image embedding (512-d)."""
        self._ensure_loaded()
        try:
            with contextlib.redirect_stdout(sys.stderr):
                vector = next(iter(self._img.embed([str(path)])))
        except Exception as exc:  # noqa: BLE001
            raise ImageReadError(f"Could not read image {path}: {exc}") from exc
        return _unit(np.asarray(vector, dtype=np.float32))

    def image_embeddings(self, paths: list[str | Path]) -> list[np.ndarray]:
        """Unit image embeddings for several photos, batched for speed.

        fastembed batches internally; one unreadable file would abort the whole
        batch, so callers that need per-photo resilience should fall back to
        :meth:`image_embedding` on failure.
        """
        self._ensure_loaded()
        with contextlib.redirect_stdout(sys.stderr):
            vectors = list(self._img.embed([str(p) for p in paths]))
        return [_unit(np.asarray(v, dtype=np.float32)) for v in vectors]

    # ----------------------------------------------------------------- text

    def text_embedding(self, text: str) -> np.ndarray:
        """A single phrase's unit text embedding (512-d)."""
        self._ensure_loaded()
        with contextlib.redirect_stdout(sys.stderr):
            vector = next(iter(self._txt.embed([text])))
        return _unit(np.asarray(vector, dtype=np.float32))

    def query_embedding(self, query: str) -> np.ndarray:
        """A description's unit embedding, averaged over a few phrasings."""
        query = query.strip()
        if not query:
            raise ValueError("Search query must not be empty.")
        phrasings = [tpl.format(q=query) for tpl in QUERY_TEMPLATES]
        self._ensure_loaded()
        with contextlib.redirect_stdout(sys.stderr):
            vectors = list(self._txt.embed(phrasings))
        mean = np.mean([_unit(np.asarray(v, dtype=np.float32)) for v in vectors], axis=0)
        return _unit(mean)

    # ----------------------------------------------------------------- score

    @staticmethod
    def similarity(image_embedding: np.ndarray, text_embedding: np.ndarray) -> float:
        """Cosine similarity between a photo and a phrase."""
        return float(_unit(np.asarray(image_embedding, dtype=np.float32))
                     @ _unit(np.asarray(text_embedding, dtype=np.float32)))

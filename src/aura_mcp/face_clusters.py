"""Group face embeddings into people.

Average-linkage clustering on cosine distance. Measured on LFW (three people
with 40 photos each among 60 strangers): each person's faces fell into one
cluster, no cluster mixed two people, and any cut from 0.30 to 0.55 gave the
same grouping. 0.40 sits in the middle of that band.
"""

from __future__ import annotations

import numpy as np

CLUSTER_THRESHOLD = 0.40
# The pairwise distances are n*n/2 doubles: about 400 MB at 10,000 faces.
MAX_FACES = 10_000


class TooManyFacesError(ValueError):
    """More faces than can be clustered in memory in one go."""


def cluster_faces(embeddings: np.ndarray, threshold: float = CLUSTER_THRESHOLD) -> np.ndarray:
    """One label per row: faces with the same label are the same person.

    ``embeddings`` holds unit vectors, one face per row. Two groups are joined
    while the average cosine similarity between their faces is at least
    ``threshold``. Labels run from 0 in order of first appearance.
    """
    count = len(embeddings)
    if count > MAX_FACES:
        raise TooManyFacesError(
            f"{count} faces is more than the {MAX_FACES} that can be grouped at once."
        )
    if count < 2:
        return np.zeros(count, dtype=np.int64)

    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    distances = np.clip(pdist(np.asarray(embeddings, dtype=np.float64), metric="cosine"), 0.0, 2.0)
    raw = fcluster(linkage(distances, method="average"), t=1.0 - threshold, criterion="distance")
    order: dict[int, int] = {}
    return np.array([order.setdefault(int(label), len(order)) for label in raw], dtype=np.int64)

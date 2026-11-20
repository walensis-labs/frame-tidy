import numpy as np
import pytest

from aura_mcp.face_clusters import MAX_FACES, TooManyFacesError, cluster_faces


def unit(*values):
    v = np.zeros(512, dtype=np.float32)
    v[: len(values)] = values
    return v / np.linalg.norm(v)


A, B = unit(1, 0.1), unit(0, 0, 1, 0.1)


def test_same_person_groups_and_strangers_stay_apart():
    faces = np.stack([A, unit(1, 0.3), B, unit(1, -0.2), unit(0, 0, 1, -0.2), unit(0, 0, 0, 0, 1)])
    labels = cluster_faces(faces)
    assert labels.tolist() == [0, 0, 1, 0, 1, 2]


def test_nothing_and_one_face():
    assert cluster_faces(np.zeros((0, 512), np.float32)).tolist() == []
    assert cluster_faces(A[None, :]).tolist() == [0]


def test_threshold_is_the_cut():
    near = np.stack([unit(1, 0), unit(1, 1)])  # cosine 0.707
    assert cluster_faces(near, 0.70).tolist() == [0, 0]
    assert cluster_faces(near, 0.75).tolist() == [0, 1]


def test_refuses_more_faces_than_fit_in_memory():
    with pytest.raises(TooManyFacesError, match=str(MAX_FACES)):
        cluster_faces(np.zeros((MAX_FACES + 1, 2), np.float32))

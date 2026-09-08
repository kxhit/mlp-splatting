from __future__ import annotations

import numpy as np
import pytest

from mip360.utils.sampling import _collect_leaves, build_octree


def test_octree_preserves_points_on_the_root_maximum_faces():
    points = np.asarray(
        [
            [x, y, z]
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for z in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )
    root = build_octree(points, min_points=1)
    recovered = np.concatenate([leaf.points for leaf in _collect_leaves(root)], axis=0)

    assert len(recovered) == len(points)
    assert {tuple(row) for row in recovered} == {tuple(row) for row in points}


def test_octree_rejects_empty_or_nonfinite_input():
    with pytest.raises(ValueError, match="min_points"):
        build_octree(np.zeros((1, 3)), min_points=0)
    with pytest.raises(ValueError, match="N > 0"):
        build_octree(np.empty((0, 3)))
    with pytest.raises(ValueError, match="finite"):
        build_octree(np.asarray([[0.0, np.nan, 1.0]]))

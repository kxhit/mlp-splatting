from __future__ import annotations

import pytest
import torch

from mip360.train import _assemble_feature_mosaic


def test_assemble_feature_mosaic_matches_dense_stack_selection():
    features = [
        torch.arange(4 * 5 * 3, dtype=torch.float32).reshape(4, 5, 3) + 1000 * index
        for index in range(4)
    ]
    view_map = torch.tensor(
        [
            [0, 0, 1, 1, 2],
            [0, 3, 3, 1, 2],
            [2, 2, 3, 0, 0],
            [1, 3, 2, 1, 0],
        ],
        dtype=torch.long,
    )
    stacked = torch.stack(features, dim=0)
    yy = torch.arange(4).unsqueeze(1).expand(4, 5)
    xx = torch.arange(5).unsqueeze(0).expand(4, 5)
    reference = stacked[view_map, yy, xx]

    candidate = _assemble_feature_mosaic(features, view_map)

    torch.testing.assert_close(candidate, reference, rtol=0.0, atol=0.0)
    assert candidate.is_contiguous()


def test_assemble_feature_mosaic_validates_view_indices():
    with pytest.raises(ValueError, match="out-of-range"):
        _assemble_feature_mosaic([torch.zeros((2, 2, 3))], torch.ones((2, 2), dtype=torch.long))

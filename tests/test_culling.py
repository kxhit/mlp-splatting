from __future__ import annotations

import pytest
import torch

from mip360.cumodel import _overlap_matrix_to_sparse


def test_overlap_matrix_to_sparse_preserves_row_major_candidate_order():
    overlap = torch.tensor(
        [
            [False, True, False, True, True],
            [False, False, False, False, False],
            [True, False, True, False, False],
        ]
    )

    candidates, offsets = _overlap_matrix_to_sparse(overlap)

    assert candidates.dtype == torch.int32
    assert offsets.dtype == torch.int32
    assert candidates.tolist() == [1, 3, 4, 0, 2]
    assert offsets.tolist() == [0, 3, 3, 5]


def test_overlap_matrix_to_sparse_handles_empty_masks():
    candidates, offsets = _overlap_matrix_to_sparse(torch.zeros((4, 7), dtype=torch.bool))

    assert candidates.shape == (0,)
    assert offsets.tolist() == [0, 0, 0, 0, 0]


def test_overlap_matrix_to_sparse_rejects_non_matrix_input():
    with pytest.raises(ValueError, match="num_tiles, num_primitives"):
        _overlap_matrix_to_sparse(torch.zeros(3, dtype=torch.bool))

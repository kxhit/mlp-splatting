from __future__ import annotations

import torch

from mip360.eval_lseg_table2 import _load_decoder_if_needed
from mip360.feature_decoder import FeatureUpsampleDecoder


def test_lseg_eval_loads_decoder_embedded_in_scene_checkpoint(tmp_path):
    expected = FeatureUpsampleDecoder(in_dim=3, out_dim=5, hidden_dim=4)
    scene_path = tmp_path / "scene_epoch_00012.pt"
    torch.save(
        {
            "format_version": 2,
            "epoch": 12,
            "feature_decoder_state_dict": expected.state_dict(),
        },
        scene_path,
    )

    actual, source = _load_decoder_if_needed(
        scene_feature_dim=3,
        target_feature_dim=5,
        decoder_ckpt="",
        scene_ckpt_path=str(scene_path),
        device=torch.device("cpu"),
    )

    assert actual is not None
    assert source.endswith(":feature_decoder_state_dict")
    for expected_value, actual_value in zip(
        expected.state_dict().values(), actual.state_dict().values(), strict=True
    ):
        torch.testing.assert_close(actual_value, expected_value)

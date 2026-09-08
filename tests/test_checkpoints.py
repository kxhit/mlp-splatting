from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from mip360.checkpoints import (
    convert_checkpoint,
    convert_feature_slots,
    infer_scene_spec,
    normalize_scene_state_dict,
    split_checkpoint,
)


def make_state(*, n: int = 3, hidden: int = 8, pe: int = 6, slots: int = 5, dim: int = 7):
    encoded = 12 * pe
    return {
        "positions": torch.randn(n, 3),
        "log_scales": torch.randn(n, 3),
        "quats": torch.randn(n, 4),
        "W1": torch.randn(n, hidden, encoded),
        "b1": torch.randn(n, hidden),
        "W2": torch.randn(n, hidden, hidden),
        "b2": torch.randn(n, hidden),
        "W3": torch.randn(n, 4, hidden),
        "b3": torch.randn(n, 4),
        "feat_attr": torch.randn(n, slots, dim),
    }


def directional_value(feature: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    if feature.shape[1] == 5:
        return feature[:, 0] + (gates[..., None] * feature[:, 1:]).sum(dim=1)
    return (gates[..., None] * feature).sum(dim=1)


def test_infer_dimensions_and_validate_shapes():
    state = make_state(n=11, hidden=32, pe=10, slots=5, dim=512)
    spec = infer_scene_spec(state)
    assert spec.num_primitives == 11
    assert spec.hidden_dim == 32
    assert spec.encoded_dim == 120
    assert spec.pe_frequencies == 10
    assert spec.feature_slots == 5
    assert spec.feature_dim == 512

    state["W3"] = torch.empty(11, 32, 4)
    with pytest.raises(ValueError, match="W3 has shape"):
        infer_scene_spec(state)

    state = make_state()
    del state["positions"]
    with pytest.raises(KeyError, match="positions"):
        infer_scene_spec(state)


def test_module_prefix_and_legacy_mlp_remap():
    current = make_state()
    reverse = {"W1": "mlps.lin1.weight", "b1": "mlps.lin1.bias", "W2": "mlps.lin2.weight", "b2": "mlps.lin2.bias", "W3": "mlps.lin3.weight", "b3": "mlps.lin3.bias"}
    legacy = {f"module._orig_mod.{reverse.get(k, k)}": v for k, v in current.items()}
    normalized = normalize_scene_state_dict(legacy)
    assert all(key in normalized for key in ("W1", "b1", "W2", "b2", "W3", "b3"))
    assert not any(key.startswith(("module.", "_orig_mod.", "mlps.")) for key in normalized)
    assert infer_scene_spec(normalized) == infer_scene_spec(current)


def test_five_to_four_is_algebraically_equivalent():
    generator = torch.Generator().manual_seed(17)
    feature5 = torch.randn(9, 5, 13, generator=generator, dtype=torch.float64)
    gates = torch.softmax(torch.randn(9, 4, generator=generator, dtype=torch.float64), dim=1)
    feature4 = convert_feature_slots(feature5, 4)
    torch.testing.assert_close(
        directional_value(feature5, gates), directional_value(feature4, gates), rtol=1e-13, atol=1e-13
    )


def test_four_to_five_round_trip_is_exact():
    feature4 = torch.randn(5, 4, 19, dtype=torch.float64)
    feature5 = convert_feature_slots(feature4, 5)
    torch.testing.assert_close(convert_feature_slots(feature5, 4), feature4, rtol=0, atol=0)


def test_wrapper_metadata_is_preserved_and_shape_metadata_updated():
    raw = {
        "format_version": 1,
        "epoch": 60_000,
        "engine": "tile_cuda",
        "feature_slots": 5,
        "feature_dim_low": 7,
        "optimizer_state_dict": {"state": {123: {"step": torch.tensor(8)}}},
        "metadata": {
            "feature_slots": 5,
            "feature_dim": 7,
            "feature_slot_layout": "legacy_base_plus_four_deltas",
            "keep": "yes",
        },
        "custom_metadata": {"keep": [1, 2, 3]},
        "scene_state_dict": make_state(),
    }
    converted, before, after = convert_checkpoint(raw)
    assert split_checkpoint(converted).wrapper_key == "scene_state_dict"
    assert converted["epoch"] == 60_000
    assert "optimizer_state_dict" not in converted
    assert converted["custom_metadata"] is raw["custom_metadata"]
    assert converted["feature_slots"] == 4
    assert converted["feature_dim_low"] == 7
    assert converted["metadata"] == {
        "feature_slots": 4,
        "feature_dim": 7,
        "feature_slot_layout": "direct_tetrahedral",
        "keep": "yes",
    }
    assert before.feature_slots == 5 and after.feature_slots == 4


def test_bare_state_dict_stays_bare():
    converted, _, after = convert_checkpoint(make_state())
    assert "scene_state_dict" not in converted
    assert converted["feat_attr"].shape == (3, 4, 7)
    assert after.feature_slots == 4


def test_common_state_dict_wrapper_is_supported():
    raw = {"state_dict": make_state(), "epoch": 12, "keep": "metadata"}
    view = split_checkpoint(raw)
    assert view.wrapper_key == "state_dict"

    converted, _, after = convert_checkpoint(raw)
    assert converted["epoch"] == 12
    assert converted["keep"] == "metadata"
    assert converted["state_dict"]["feat_attr"].shape == (3, 4, 7)
    assert after.feature_slots == 4


def test_cli_writes_new_file_and_refuses_overwrite(tmp_path: Path):
    source = tmp_path / "scene.pt"
    torch.save({"scene_state_dict": make_state(), "epoch": 4}, source)
    script = Path(__file__).parents[1] / "tools" / "convert_legacy_checkpoint.py"
    command = [sys.executable, str(script), str(source)]
    first = subprocess.run(command, check=False, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    output = tmp_path / "scene.slots4.pt"
    assert output.is_file()

    saved = torch.load(output, map_location="cpu", weights_only=False)
    assert saved["epoch"] == 4
    assert saved["scene_state_dict"]["feat_attr"].shape[1] == 4

    second = subprocess.run(command, check=False, capture_output=True, text=True)
    assert second.returncode != 0
    assert "refusing to overwrite" in second.stderr

"""Checkpoint compatibility helpers with no renderer or CUDA dependency.

The project has historically saved both bare scene ``state_dict`` objects and
larger training-state wrappers.  This module normalizes either representation,
maps the old ``mlps.lin*`` parameter names to the tile renderer names, and can
losslessly translate between the two directional-feature layouts used by the
project.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch


_PREFIXES = ("module.", "_orig_mod.")
_LEGACY_MLP_KEYS = {
    "mlps.lin1.weight": "W1",
    "mlps.lin1.bias": "b1",
    "mlps.lin2.weight": "W2",
    "mlps.lin2.bias": "b2",
    "mlps.lin3.weight": "W3",
    "mlps.lin3.bias": "b3",
}
_REQUIRED_MLP_KEYS = ("W1", "b1", "W2", "b2", "W3", "b3")


@dataclass(frozen=True)
class CheckpointView:
    """A checkpoint split into its scene state and outer representation."""

    scene_state_dict: dict[str, Any]
    wrapper_key: Literal["scene_state_dict", "state_dict"] | None


@dataclass(frozen=True)
class SceneSpec:
    """Dimensions inferred solely from checkpoint tensors."""

    num_primitives: int
    hidden_dim: int
    encoded_dim: int
    pe_frequencies: int
    feature_dim: int
    feature_slots: int


def _is_state_dict(value: Any) -> bool:
    """Return whether *value* resembles a PyTorch state dictionary."""

    if not isinstance(value, Mapping) or not value:
        return False
    keys = value.keys()
    return any(
        isinstance(key, str)
        and (
            key.endswith("positions")
            or key.endswith("W1")
            or key.endswith("mlps.lin1.weight")
        )
        for key in keys
    )


def split_checkpoint(checkpoint: Mapping[str, Any]) -> CheckpointView:
    """Locate the scene state dict in a bare or wrapped checkpoint.

    ``scene_state_dict`` is the native training-state wrapper.  ``state_dict``
    is also accepted for compatibility with common trainer frameworks.  A bare
    mapping containing scene parameters is returned as-is.
    """

    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"checkpoint must be a mapping, got {type(checkpoint).__name__}")

    for key in ("scene_state_dict", "state_dict"):
        candidate = checkpoint.get(key)
        if _is_state_dict(candidate):
            return CheckpointView(dict(candidate), key)

    if _is_state_dict(checkpoint):
        return CheckpointView(dict(checkpoint), None)

    raise ValueError(
        "could not find a scene state_dict; expected a bare scene state_dict "
        "or a 'scene_state_dict'/'state_dict' wrapper"
    )


def _strip_known_prefixes(key: str) -> str:
    previous = None
    while key != previous:
        previous = key
        for prefix in _PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
    return key


def normalize_scene_state_dict(
    state_dict: Mapping[str, Any], *, remap_legacy_mlp: bool = True
) -> dict[str, Any]:
    """Strip compiler/DDP prefixes and optionally canonicalize legacy MLP keys.

    Unknown entries (including runtime statistics) are retained.  A collision
    is accepted only when both names point to the same object; otherwise it is
    rejected so conversion cannot silently select the wrong weights.
    """

    if not isinstance(state_dict, Mapping):
        raise TypeError("state_dict must be a mapping")

    normalized: dict[str, Any] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise TypeError(f"state_dict key must be str, got {type(key).__name__}")
        canonical = _strip_known_prefixes(key)
        if canonical in normalized and normalized[canonical] is not value:
            raise ValueError(f"key collision after prefix removal: {key!r} -> {canonical!r}")
        normalized[canonical] = value

    if not remap_legacy_mlp:
        return normalized

    present_legacy = {old for old in _LEGACY_MLP_KEYS if old in normalized}
    if present_legacy and present_legacy != set(_LEGACY_MLP_KEYS):
        missing = sorted(set(_LEGACY_MLP_KEYS) - present_legacy)
        raise KeyError(f"incomplete legacy MLP state; missing {missing}")

    for old, new in _LEGACY_MLP_KEYS.items():
        if old not in normalized:
            continue
        legacy_value = normalized.pop(old)
        if new in normalized and normalized[new] is not legacy_value:
            current = normalized[new]
            if not (
                torch.is_tensor(current)
                and torch.is_tensor(legacy_value)
                and current.shape == legacy_value.shape
                and torch.equal(current, legacy_value)
            ):
                raise ValueError(f"both legacy key {old!r} and canonical key {new!r} differ")
        else:
            normalized[new] = legacy_value
    return normalized


def _tensor(state_dict: Mapping[str, Any], key: str) -> torch.Tensor:
    value = state_dict.get(key)
    if not torch.is_tensor(value):
        raise KeyError(f"required tensor {key!r} is missing")
    return value


def _expect_shape(name: str, tensor: torch.Tensor, expected: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != expected:
        raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {expected}")


def infer_scene_spec(state_dict: Mapping[str, Any]) -> SceneSpec:
    """Infer and validate N, hidden width, PE count, and feature dimensions."""

    state = normalize_scene_state_dict(state_dict)
    w1 = _tensor(state, "W1")
    if w1.ndim != 3:
        raise ValueError(f"W1 must have shape [N,H,encoded_dim], got {tuple(w1.shape)}")
    n, hidden_dim, encoded_dim = map(int, w1.shape)
    if n <= 0 or hidden_dim <= 0 or encoded_dim <= 0:
        raise ValueError("scene dimensions must be positive")
    if encoded_dim % 12:
        raise ValueError(
            f"encoded_dim={encoded_dim} is not divisible by 12 "
            "(6D input x sin/cos positional encoding)"
        )

    _expect_shape("b1", _tensor(state, "b1"), (n, hidden_dim))
    _expect_shape("W2", _tensor(state, "W2"), (n, hidden_dim, hidden_dim))
    _expect_shape("b2", _tensor(state, "b2"), (n, hidden_dim))
    _expect_shape("W3", _tensor(state, "W3"), (n, 4, hidden_dim))
    _expect_shape("b3", _tensor(state, "b3"), (n, 4))

    geometry_shapes = {"positions": (n, 3), "log_scales": (n, 3), "quats": (n, 4)}
    for name, shape in geometry_shapes.items():
        _expect_shape(name, _tensor(state, name), shape)

    feature_dim = 0
    feature_slots = 0
    feature = state.get("feat_attr")
    if feature is not None:
        if not torch.is_tensor(feature):
            raise TypeError("feat_attr must be a tensor")
        if feature.ndim == 2:
            _expect_shape("feat_attr", feature, (n, int(feature.shape[1])))
            feature_dim, feature_slots = int(feature.shape[1]), 1
        elif feature.ndim == 3:
            if int(feature.shape[0]) != n:
                raise ValueError(f"feat_attr has N={feature.shape[0]}, expected {n}")
            feature_slots, feature_dim = int(feature.shape[1]), int(feature.shape[2])
        else:
            raise ValueError(f"feat_attr must have shape [N,D] or [N,S,D], got {tuple(feature.shape)}")
        if feature_dim <= 0 or feature_slots <= 0:
            raise ValueError("feature dimensions must be positive")

    return SceneSpec(
        num_primitives=n,
        hidden_dim=hidden_dim,
        encoded_dim=encoded_dim,
        pe_frequencies=encoded_dim // 12,
        feature_dim=feature_dim,
        feature_slots=feature_slots,
    )


def convert_feature_slots(feature: torch.Tensor, target_slots: Literal[4, 5]) -> torch.Tensor:
    """Translate directional feature parameterizations without changing output.

    Slots 1..4 of the five-slot layout are deltas from slot 0.  Four-slot
    tensors contain the corresponding directional embeddings directly.  The
    conversion is exact for any gating weights whose four components sum to 1.
    """

    if not torch.is_tensor(feature) or feature.ndim != 3:
        raise ValueError("feature must be a tensor with shape [N,S,D]")
    source_slots = int(feature.shape[1])
    if target_slots not in (4, 5):
        raise ValueError("target_slots must be 4 or 5")
    if source_slots == target_slots:
        return feature.clone()
    if source_slots == 5 and target_slots == 4:
        return feature[:, :1, :] + feature[:, 1:, :]
    if source_slots == 4 and target_slots == 5:
        # A zero base is the unique practical choice that also gives a bitwise
        # exact 4 -> 5 -> 4 round trip in floating point.
        base = torch.zeros_like(feature[:, :1, :])
        return torch.cat((base, feature), dim=1)
    raise ValueError(f"cannot convert {source_slots}-slot features to {target_slots} slots")


def convert_checkpoint(
    checkpoint: Mapping[str, Any], *, target_feature_slots: Literal[4, 5] = 4
) -> tuple[dict[str, Any], SceneSpec, SceneSpec]:
    """Normalize a checkpoint and convert its directional feature layout.

    The returned checkpoint has the same bare/wrapped representation as the
    input.  Wrapper metadata is retained and synchronized with the converted
    tensor shape.  Optimizer state is dropped when the slot layout changes:
    Adam's per-parameter moment tensors keep the old ``feat_attr`` shape and
    cannot be resumed safely after conversion.
    """

    view = split_checkpoint(checkpoint)
    state = normalize_scene_state_dict(view.scene_state_dict)
    before = infer_scene_spec(state)
    if before.feature_slots not in (4, 5):
        raise ValueError(
            f"checkpoint has {before.feature_slots} feature slots; only 4 and 5 are convertible"
        )
    converted_state = dict(state)
    converted_state["feat_attr"] = convert_feature_slots(
        _tensor(state, "feat_attr"), target_feature_slots
    )
    after = infer_scene_spec(converted_state)

    if view.wrapper_key is None:
        output: dict[str, Any] = converted_state
    else:
        output = dict(checkpoint)
        output[view.wrapper_key] = converted_state
        if "feature_slots" in output:
            output["feature_slots"] = after.feature_slots
        if "feature_dim_low" in output:
            output["feature_dim_low"] = after.feature_dim
        metadata = output.get("metadata")
        if isinstance(metadata, Mapping):
            metadata = dict(metadata)
            metadata["feature_slots"] = after.feature_slots
            metadata["feature_dim"] = after.feature_dim
            if "feature_dim_low" in metadata:
                metadata["feature_dim_low"] = after.feature_dim
            metadata["feature_slot_layout"] = (
                "direct_tetrahedral"
                if after.feature_slots == 4
                else "legacy_base_plus_four_deltas"
            )
            output["metadata"] = metadata
        if before.feature_slots != after.feature_slots:
            # Optimizer state is indexed by parameter position and contains
            # shape-specific moments.  It cannot be mapped robustly without
            # the original optimizer/parameter-group objects.
            output.pop("optimizer_state_dict", None)
    return output, before, after

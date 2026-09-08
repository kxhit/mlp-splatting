"""Shared fail-fast validation for COLMAP cameras and feature payloads."""

from __future__ import annotations

import os
import re

import numpy as np
import torch

FEATURE_CHANNELS = {"lseg": 512, "sam": 256}
MAX_FOCAL_ASYMMETRY = 0.01
MAX_PRINCIPAL_POINT_OFFSET_PX = 0.5 + 1e-6


def parse_supported_camera(camera, *, image_name: str) -> dict[str, float | int | str]:
    """Return intrinsics supported by the current centered-pinhole ray model."""
    model = str(camera.model)
    params = np.asarray(camera.params, dtype=np.float64).reshape(-1)
    width = int(camera.width)
    height = int(camera.height)
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid COLMAP camera dimensions for image '{image_name}': {width}x{height}."
        )

    if model == "SIMPLE_PINHOLE":
        if params.size != 3:
            raise ValueError(
                f"COLMAP SIMPLE_PINHOLE camera for image '{image_name}' must have "
                f"3 parameters, got {params.size}."
            )
        fx, cx, cy = map(float, params)
        fy = fx
    elif model == "PINHOLE":
        if params.size != 4:
            raise ValueError(
                f"COLMAP PINHOLE camera for image '{image_name}' must have "
                f"4 parameters, got {params.size}."
            )
        fx, fy, cx, cy = map(float, params)
    else:
        raise ValueError(
            f"Unsupported COLMAP camera model '{model}' for image '{image_name}'. "
            "The current renderer accepts only undistorted SIMPLE_PINHOLE or PINHOLE "
            "cameras with centered principal points and approximately square pixels."
        )

    values = np.asarray((fx, fy, cx, cy), dtype=np.float64)
    if not np.isfinite(values).all() or fx <= 0.0 or fy <= 0.0:
        raise ValueError(
            f"Invalid COLMAP intrinsics for image '{image_name}': "
            f"fx={fx}, fy={fy}, cx={cx}, cy={cy}."
        )
    focal_mean = 0.5 * (fx + fy)
    focal_asymmetry = abs(fx - fy) / focal_mean
    if focal_asymmetry >= MAX_FOCAL_ASYMMETRY:
        raise ValueError(
            f"Unsupported non-square COLMAP pixels for image '{image_name}': "
            f"fx={fx}, fy={fy}, relative difference={focal_asymmetry:.3%}; "
            "the current renderer requires a difference below 1%."
        )

    expected_cx = width * 0.5
    expected_cy = height * 0.5
    if (
        abs(cx - expected_cx) > MAX_PRINCIPAL_POINT_OFFSET_PX
        or abs(cy - expected_cy) > MAX_PRINCIPAL_POINT_OFFSET_PX
    ):
        raise ValueError(
            f"Unsupported off-center COLMAP principal point for image '{image_name}': "
            f"(cx, cy)=({cx}, {cy}), expected approximately "
            f"({expected_cx}, {expected_cy}) for {width}x{height}."
        )
    return {
        "model": model,
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
    }


def validated_camera_intrinsics_by_image(images, cameras) -> dict[str, dict]:
    """Validate that every COLMAP image uses one shared supported calibration."""
    by_base: dict[str, dict] = {}
    reference = None
    reference_name = None
    for image_id in sorted(images):
        image = images[image_id]
        base = os.path.basename(image.name)
        if base in by_base:
            raise ValueError(f"Duplicated COLMAP image basename detected: '{base}'")
        if image.camera_id not in cameras:
            raise KeyError(
                f"COLMAP image '{base}' references missing camera id {image.camera_id}."
            )
        intrinsics = parse_supported_camera(cameras[image.camera_id], image_name=base)
        if reference is None:
            reference = intrinsics
            reference_name = base
        else:
            same_shape_and_model = (
                intrinsics["model"] == reference["model"]
                and intrinsics["width"] == reference["width"]
                and intrinsics["height"] == reference["height"]
            )
            numeric_keys = ("fx", "fy", "cx", "cy")
            same_values = np.allclose(
                [intrinsics[key] for key in numeric_keys],
                [reference[key] for key in numeric_keys],
                rtol=1e-6,
                atol=1e-6,
            )
            if not (same_shape_and_model and same_values):
                raise ValueError(
                    "Per-image COLMAP intrinsics are unsupported: "
                    f"image '{base}' (camera {image.camera_id}) differs from "
                    f"image '{reference_name}'. The current renderer requires one shared "
                    "camera model, resolution, focal length, and principal point."
                )
        by_base[base] = intrinsics
    return by_base


def normalize_feature_stem(stem: str) -> str:
    lower_stem = stem.lower()
    # Feature extractors commonly append both a payload tag and an explicit
    # layout hint.  Strip the longest forms first so the hint can disambiguate
    # CHW/HWC tensors without breaking filename-stem alignment with the RGB.
    for suffix in (
        "_fmap_cxhxw",
        "_fmap_hxwxc",
        "_feature_cxhxw",
        "_feature_hxwxc",
        "_fmap_chw",
        "_fmap_hwc",
        "_feature_chw",
        "_feature_hwc",
        "_fmap",
        "_feature",
    ):
        if lower_stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _feature_layout_hint(feat_path: str) -> str | None:
    name = os.path.basename(feat_path).lower()
    chw = "cxhxw" in name or re.search(r"(?:^|[_-])chw(?:[_.-]|$)", name) is not None
    hwc = "hxwxc" in name or re.search(r"(?:^|[_-])hwc(?:[_.-]|$)", name) is not None
    if chw == hwc:
        return None
    return "chw" if chw else "hwc"


def infer_feature_layout(
    shape, image_hw: tuple[int, int], feature_type: str, feat_path: str
) -> str:
    """Infer CHW/HWC from declared feature channels, spatial shape, and hints."""
    shape = tuple(int(dim) for dim in shape)
    if len(shape) != 3:
        raise ValueError(f"Expected a 3D feature tensor, got shape {shape} from {feat_path}")
    kind = str(feature_type).lower()
    expected_channels = FEATURE_CHANNELS.get(kind)
    if expected_channels is None:
        raise ValueError(
            f"Cannot infer feature layout for feature_type={feature_type!r}; "
            f"expected one of {sorted(FEATURE_CHANNELS)}."
        )

    candidates = []
    if shape[0] == expected_channels:
        candidates.append("chw")
    if shape[2] == expected_channels:
        candidates.append("hwc")
    if not candidates:
        raise ValueError(
            f"Feature tensor {feat_path} has shape {shape}; {kind} features require "
            f"{expected_channels} channels on either the first (CHW) or last (HWC) axis."
        )
    if len(candidates) == 1:
        return candidates[0]

    target_h, target_w = map(int, image_hw)

    def _spatial_score(layout: str) -> int:
        spatial = shape[1:] if layout == "chw" else shape[:2]
        if spatial == (target_h, target_w):
            return 2
        if min(*spatial, target_h, target_w) <= 0:
            return 0
        feature_aspect = spatial[1] / spatial[0]
        target_aspect = target_w / target_h
        relative_error = abs(feature_aspect - target_aspect) / target_aspect
        return 1 if relative_error < 0.01 else 0

    scores = {layout: _spatial_score(layout) for layout in candidates}
    if scores["chw"] != scores["hwc"]:
        return max(scores, key=scores.get)
    hint = _feature_layout_hint(feat_path)
    if hint in candidates:
        return hint
    raise ValueError(
        f"Ambiguous feature layout for {feat_path}: shape={shape}, "
        f"target image={target_h}x{target_w}, expected channels={expected_channels}. "
        "Use an unambiguous channel axis or an explicit CxHxW/CHW or HxWxC/HWC filename hint."
    )


def select_feature_payload(payload, *, source: str):
    """Select an explicit or unambiguous tensor/array from a feature payload."""
    if torch.is_tensor(payload) or isinstance(payload, np.ndarray):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(
            f"Feature payload in {source} must be a tensor, array, or dictionary, "
            f"got {type(payload).__name__}."
        )
    if "feat" in payload:
        selected = payload["feat"]
        if not (torch.is_tensor(selected) or isinstance(selected, np.ndarray)):
            raise TypeError(f"Feature payload 'feat' in {source} is not a tensor or array.")
        return selected
    tensor_items = [
        (str(key), value)
        for key, value in payload.items()
        if torch.is_tensor(value) or isinstance(value, np.ndarray)
    ]
    if len(tensor_items) != 1:
        keys = sorted(key for key, _ in tensor_items)
        raise ValueError(
            f"Ambiguous feature payload in {source}: expected key 'feat' or exactly "
            f"one tensor/array value, found {len(tensor_items)} at keys {keys}."
        )
    return tensor_items[0][1]

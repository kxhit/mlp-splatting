from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mip360 import data_validation, mip360_loader
from replica import replica_loader

LOADER_MODULES = (mip360_loader, replica_loader)


def _camera(model: str, params, *, camera_id: int = 1, width: int = 638, height: int = 478):
    return SimpleNamespace(
        id=camera_id,
        model=model,
        width=width,
        height=height,
        params=np.asarray(params, dtype=np.float64),
    )


def _image(name: str, camera_id: int = 1):
    return SimpleNamespace(name=name, camera_id=camera_id)


def test_room0_pinhole_intrinsics_are_supported_and_keep_fx():
    camera = _camera("PINHOLE", [319.8482, 320.2420, 319.0, 239.0])
    result = data_validation.validated_camera_intrinsics_by_image(
        {1: _image("frame000001.jpg")}, {1: camera}
    )

    intrinsics = result["frame000001.jpg"]
    assert intrinsics["fx"] == pytest.approx(319.8482)
    assert intrinsics["fy"] == pytest.approx(320.2420)
    assert intrinsics["width"] == 638
    assert intrinsics["height"] == 478


def test_simple_pinhole_intrinsics_are_supported():
    camera = _camera("SIMPLE_PINHOLE", [320.0, 319.0, 239.0])
    result = data_validation.validated_camera_intrinsics_by_image(
        {1: _image("frame.png")}, {1: camera}
    )
    assert result["frame.png"]["fx"] == result["frame.png"]["fy"] == 320.0


@pytest.mark.parametrize(
    ("camera", "error"),
    [
        (
            _camera("OPENCV", [320.0, 320.0, 319.0, 239.0, 0.0, 0.0, 0.0, 0.0]),
            "Unsupported COLMAP camera model",
        ),
        (
            _camera("PINHOLE", [300.0, 330.0, 319.0, 239.0]),
            "non-square COLMAP pixels",
        ),
        (
            _camera("PINHOLE", [320.0, 320.0, 321.0, 239.0]),
            "off-center COLMAP principal point",
        ),
    ],
)
def test_unsupported_intrinsics_fail_fast(camera, error):
    with pytest.raises(ValueError, match=error):
        data_validation.validated_camera_intrinsics_by_image(
            {1: _image("frame.png")}, {1: camera}
        )


def test_per_image_intrinsics_are_rejected():
    images = {1: _image("a.png", 1), 2: _image("b.png", 2)}
    cameras = {
        1: _camera("SIMPLE_PINHOLE", [320.0, 319.0, 239.0], camera_id=1),
        2: _camera("SIMPLE_PINHOLE", [321.0, 319.0, 239.0], camera_id=2),
    }
    with pytest.raises(ValueError, match="Per-image COLMAP intrinsics are unsupported"):
        data_validation.validated_camera_intrinsics_by_image(images, cameras)


def test_feature_layout_uses_expected_channel_axis():
    assert data_validation.infer_feature_layout(
        (512, 12, 16), (48, 64), "lseg", "frame.pt"
    ) == "chw"
    assert data_validation.infer_feature_layout(
        (12, 16, 512), (48, 64), "lseg", "frame.pt"
    ) == "hwc"
    assert data_validation.infer_feature_layout(
        (256, 12, 16), (48, 64), "sam", "frame.pt"
    ) == "chw"


def test_feature_layout_ambiguity_requires_a_hint():
    shape = (512, 32, 512)
    with pytest.raises(ValueError, match="Ambiguous feature layout"):
        data_validation.infer_feature_layout(shape, (64, 64), "lseg", "frame.pt")
    assert data_validation.infer_feature_layout(
        shape, (64, 64), "lseg", "frame_fmap_CxHxW.pt"
    ) == "chw"


def test_feature_layout_rejects_wrong_channel_count():
    with pytest.raises(ValueError, match="512 channels"):
        data_validation.infer_feature_layout(
            (128, 12, 16), (48, 64), "lseg", "frame.pt"
        )


@pytest.mark.parametrize("loader_module", LOADER_MODULES)
def test_duplicate_normalized_feature_stems_are_rejected(loader_module, tmp_path):
    (tmp_path / "frame_feature.pt").touch()
    (tmp_path / "frame_fmap.npy").touch()
    dataset = loader_module.mip360Dataset.__new__(loader_module.mip360Dataset)
    dataset.image_paths = [str(tmp_path / "frame.png")]

    with pytest.raises(ValueError, match="Duplicate feature stem 'frame'"):
        dataset._build_feature_path_list(str(tmp_path))


@pytest.mark.parametrize("loader_module", LOADER_MODULES)
@pytest.mark.parametrize(
    "layout_suffix",
    ("fmap_CxHxW", "fmap_CHW", "feature_HxWxC", "feature_HWC"),
)
def test_layout_hint_suffix_preserves_feature_path_alignment(
    loader_module, tmp_path, layout_suffix
):
    feature_path = tmp_path / f"frame_{layout_suffix}.pt"
    feature_path.touch()
    dataset = loader_module.mip360Dataset.__new__(loader_module.mip360Dataset)
    dataset.image_paths = [str(tmp_path / "frame.png")]

    assert dataset._build_feature_path_list(str(tmp_path)) == [str(feature_path)]


@pytest.mark.parametrize("loader_module", LOADER_MODULES)
@pytest.mark.parametrize("extension", ("pt", "npz"))
def test_ambiguous_feature_dictionary_payload_is_rejected(
    loader_module, tmp_path, extension
):
    path = tmp_path / f"frame.{extension}"
    payload = {
        "first": torch.zeros((512, 2, 3)),
        "second": torch.zeros((512, 2, 3)),
    }
    if extension == "pt":
        torch.save(payload, path)
    else:
        np.savez(path, **{key: value.numpy() for key, value in payload.items()})

    dataset = loader_module.mip360Dataset.__new__(loader_module.mip360Dataset)
    dataset.images = torch.zeros((1, 3, 8, 12))
    dataset.feature_type = "lseg"
    dataset.feature_resize_to_image = False
    with pytest.raises(ValueError, match="Ambiguous feature payload"):
        dataset._load_feature_tensor(str(path))


@pytest.mark.parametrize("loader_module", LOADER_MODULES)
@pytest.mark.parametrize(("layout", "shape"), (("chw", (512, 2, 3)), ("hwc", (2, 3, 512))))
def test_feature_tensor_is_normalized_to_hwc(loader_module, tmp_path, layout, shape):
    path = tmp_path / f"frame_{layout}.pt"
    torch.save({"feat": torch.arange(np.prod(shape)).reshape(shape)}, path)

    dataset = loader_module.mip360Dataset.__new__(loader_module.mip360Dataset)
    dataset.images = torch.zeros((1, 3, 8, 12))
    dataset.feature_type = "lseg"
    dataset.feature_resize_to_image = False
    feature = dataset._load_feature_tensor(str(path))

    assert feature.shape == (2, 3, 512)
    assert feature.dtype == torch.float32

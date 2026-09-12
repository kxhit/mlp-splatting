from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def tile_ext():
    if not torch.cuda.is_available():
        pytest.skip("CUDA renderer tests require a CUDA device")
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        pytest.skip("CUDA renderer tests require an nvcc toolchain")

    default_source = Path(__file__).parents[1] / "mip360" / "cuda" / "np_tile_kernels.cu"
    source = Path(os.environ.get("MLPSP_CUDA_TEST_SOURCE", default_source))
    extension_name = os.environ.get("MLPSP_CUDA_TEST_EXT_NAME", "np_tile_kernels_cuda_tests")
    extra_cuda_cflags = []
    max_registers = os.environ.get("MLPSP_NVCC_MAXRREGCOUNT", "").strip()
    if max_registers:
        extra_cuda_cflags.append(f"-maxrregcount={max_registers}")
    tilecenter_threads = os.environ.get("MLPSP_TILECENTER_BLOCK_THREADS", "").strip()
    if tilecenter_threads:
        extra_cuda_cflags.append(f"-DMLPSP_TILECENTER_BLOCK_THREADS={tilecenter_threads}")
    recompute_geometry = os.environ.get(
        "MLPSP_BACKWARD_RECOMPUTE_GEOMETRY", ""
    ).strip()
    if recompute_geometry:
        extra_cuda_cflags.append(
            f"-DMLPSP_BACKWARD_RECOMPUTE_GEOMETRY={recompute_geometry}"
        )
    max_hits = os.environ.get("MLPSP_TILECENTER_MAX_HITS", "").strip()
    if max_hits:
        extra_cuda_cflags.append(f"-DMLPSP_TILECENTER_MAX_HITS={max_hits}")
    min_blocks = os.environ.get("MLPSP_BACKWARD_MIN_BLOCKS_PER_SM", "").strip()
    if min_blocks:
        extra_cuda_cflags.append(f"-DMLPSP_BACKWARD_MIN_BLOCKS_PER_SM={min_blocks}")
    return load(
        name=extension_name,
        sources=[str(source)],
        verbose=False,
        extra_cuda_cflags=extra_cuda_cflags,
    )


def test_public_scene_defaults_match_release_architecture():
    from mip360.cumodel import NeuralPrimitiveSceneTileCUDA

    scene = NeuralPrimitiveSceneTileCUDA(num_primitives=2)
    assert scene.hidden_dim == 32
    assert scene.pe_frequencies == 6
    assert scene.encoder.out_dim == 72
    assert scene.W1.shape == (2, 32, 72)


def _public_scene(device: torch.device):
    from mip360.cumodel import NeuralPrimitiveSceneTileCUDA

    scene = NeuralPrimitiveSceneTileCUDA(num_primitives=1).to(device)
    scene.set_image_size(1, 1)
    scene.tile_order_mode_override = "tilecenter"
    with torch.no_grad():
        scene.positions.zero_()
        scene.positions[:, 2] = 1.0
        scene.log_scales.fill_(-2.0)
        scene.quats.zero_()
        scene.quats[:, 0] = 1.0
    rays_o = torch.zeros((1, 3), device=device)
    rays_d = torch.tensor([[0.0, 0.0, 1.0]], device=device)
    view = torch.eye(4, device=device)
    projection = torch.eye(4, device=device)
    return scene, rays_o, rays_d, view, projection


def _install_test_extension(monkeypatch, tile_ext):
    import mip360.cumodel as cumodel

    monkeypatch.setattr(cumodel, "_tile_ext", tile_ext)
    monkeypatch.setattr(cumodel, "_tile_ext_key", cumodel._tile_ext_build_config()[0])


def test_public_no_grad_helpers_dispatch_and_return_detached_outputs(tile_ext, monkeypatch):
    _install_test_extension(monkeypatch, tile_ext)
    device = torch.device("cuda", torch.cuda.current_device())
    scene, rays_o, rays_d, view, projection = _public_scene(device)

    rgb, alpha = scene.render_rgb_alpha_no_grad(
        rays_o, rays_d, view_matrix=view, proj_matrix=projection
    )
    rgb_depth, depth, alpha_depth = scene.render_rgb_t_no_grad(
        rays_o, rays_d, view_matrix=view, proj_matrix=projection
    )

    for output in (rgb, alpha, rgb_depth, depth, alpha_depth):
        assert output.device == device
        assert torch.isfinite(output).all()
        assert not output.requires_grad


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two visible CUDA devices")
def test_public_renderer_selects_the_models_nondefault_device(tile_ext, monkeypatch):
    _install_test_extension(monkeypatch, tile_ext)
    target = torch.device("cuda", 1)
    scene, rays_o, rays_d, view, projection = _public_scene(target)
    torch.cuda.set_device(0)

    rgb = scene(
        rays_o,
        rays_d,
        view_matrix=view,
        proj_matrix=projection,
    )

    assert torch.cuda.current_device() == target.index
    assert rgb.device == target
    assert torch.isfinite(rgb).all()


def _renderer_inputs(n: int) -> dict[str, torch.Tensor | float | int]:
    device = torch.device("cuda")
    hidden = 1
    frequencies = 1
    encoded = 6 * frequencies * 2

    positions = torch.zeros((n, 3), device=device)
    positions[:, 2] = 1.0
    quats = torch.zeros((n, 4), device=device)
    quats[:, 0] = 1.0
    return {
        "rays_o": torch.zeros((1, 3), device=device),
        "rays_d": torch.tensor([[0.0, 0.0, 1.0]], device=device),
        "positions": positions,
        "log_scales": torch.zeros((n, 3), device=device),
        "quats": quats,
        "candidates": torch.arange(n, device=device, dtype=torch.int32),
        "offsets": torch.tensor([0, n], device=device, dtype=torch.int32),
        "freq": torch.ones(frequencies, device=device),
        "attenuation": torch.ones(frequencies, device=device),
        "W1": torch.zeros((n, hidden, encoded), device=device),
        "b1": torch.zeros((n, hidden), device=device),
        "W2": torch.zeros((n, hidden, hidden), device=device),
        "b2": torch.zeros((n, hidden), device=device),
        "W3": torch.zeros((n, 4, hidden), device=device),
        "b3": torch.zeros((n, 4), device=device),
        "feat_attr": torch.zeros((n, 4, 2), device=device),
        "grad_rgb": torch.ones((1, 3), device=device),
        "grad_t": torch.zeros(1, device=device),
        "grad_feat": torch.ones((1, 2), device=device),
    }


def _base_args(data):
    return (
        data["rays_o"],
        data["rays_d"],
        data["positions"],
        data["log_scales"],
        data["quats"],
        data["candidates"],
        data["offsets"],
        data["freq"],
        data["attenuation"],
        data["W1"],
        data["b1"],
        data["W2"],
        data["b2"],
        data["W3"],
        data["b3"],
    )


def _invoke(tile_ext, name: str, data):
    base = _base_args(data)
    dims = (1, 1, 1, 1, 1.0e-3)
    if name == "render_tiles_forward":
        return tile_ext.render_tiles_forward(*base, *dims)
    if name == "render_tiles_autograd":
        return tile_ext.render_tiles_autograd(*base, *dims)
    if name == "render_tiles_forward_per_pixel_with_feat":
        return tile_ext.render_tiles_forward_per_pixel_with_feat(
            *base, data["feat_attr"], 1.0, 1.0, *dims
        )
    if name == "render_tiles_backward_count_per_pixel":
        return tile_ext.render_tiles_backward_count_per_pixel(*base, *dims)
    if name == "render_tiles_backward_featattr_per_pixel":
        return tile_ext.render_tiles_backward_featattr_per_pixel(
            *base, data["feat_attr"], data["grad_feat"], 1.0, 1.0, *dims
        )
    if name == "render_tiles_backward_emit_per_pixel":
        counts = torch.full((1,), data["positions"].shape[0], device="cuda", dtype=torch.int32)
        record_offsets = torch.zeros(1, device="cuda", dtype=torch.int32)
        return tile_ext.render_tiles_backward_emit_per_pixel(
            *base,
            data["grad_rgb"],
            data["grad_t"],
            counts,
            record_offsets,
            int(counts[0]),
            *dims,
        )
    if name == "render_tiles_forward_tilecenter":
        return tile_ext.render_tiles_forward_tilecenter(*base, *dims)
    if name == "render_tiles_forward_tilecenter_with_feat":
        return tile_ext.render_tiles_forward_tilecenter_with_feat(
            *base, data["feat_attr"], 1.0, 1.0, *dims
        )
    if name == "render_tiles_backward_tilecenter_full":
        return tile_ext.render_tiles_backward_tilecenter_full(
            *base, data["grad_rgb"], data["grad_t"], *dims
        )
    if name == "render_tiles_backward_featattr_tilecenter":
        return tile_ext.render_tiles_backward_featattr_tilecenter(
            *base, data["feat_attr"], data["grad_feat"], 1.0, 1.0, *dims
        )
    if name == "render_tiles_backward_tilecenter_full_with_feat":
        return tile_ext.render_tiles_backward_tilecenter_full_with_feat(
            *base,
            data["feat_attr"],
            data["grad_rgb"],
            data["grad_t"],
            data["grad_feat"],
            1.0,
            1.0,
            *dims,
        )
    raise AssertionError(f"unknown test entry point: {name}")


def test_reduced_thread_tilecenter_build_rejects_tile16(tile_ext):
    configured_threads = os.environ.get("MLPSP_TILECENTER_BLOCK_THREADS", "").strip()
    if configured_threads not in {"64", "128"}:
        pytest.skip("only applies to reduced-thread builds")

    data = _renderer_inputs(1)
    with pytest.raises(RuntimeError, match="MLPSP_TILECENTER_BLOCK_THREADS"):
        tile_ext.render_tiles_forward_tilecenter(
            *_base_args(data), 1, 1, 16, 16, 1.0e-3
        )


def test_tilecenter_backward_matches_reference(tile_ext):
    reference_path = os.environ.get("MLPSP_CUDA_REFERENCE_SO", "").strip()
    if not reference_path:
        pytest.skip("set MLPSP_CUDA_REFERENCE_SO for cross-build numerical comparison")
    reference_name = os.environ.get(
        "MLPSP_CUDA_REFERENCE_NAME", "np_tile_kernels_cuda_tests"
    )
    spec = importlib.util.spec_from_file_location(reference_name, reference_path)
    assert spec is not None and spec.loader is not None
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)

    torch.manual_seed(20260912)
    data = _renderer_inputs(12)
    data["positions"][:, 0] = torch.linspace(-0.08, 0.08, 12, device="cuda")
    data["positions"][:, 1] = torch.linspace(0.05, -0.05, 12, device="cuda")
    data["log_scales"].fill_(-0.5)
    for name in ("W1", "b1", "W2", "b2", "W3", "b3"):
        data[name].normal_(mean=0.0, std=0.1)

    for entry_point in (
        "render_tiles_forward_tilecenter",
        "render_tiles_forward_tilecenter_with_feat",
        "render_tiles_backward_tilecenter_full",
        "render_tiles_backward_featattr_tilecenter",
    ):
        expected = _invoke(reference, entry_point, data)
        actual = _invoke(tile_ext, entry_point, data)
        if isinstance(actual, torch.Tensor):
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
            continue
        for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
            torch.testing.assert_close(actual_tensor, expected_tensor, atol=2e-5, rtol=2e-5)

    # Exercise a full tile whose pixels hit different primitive subsequences.
    # This catches reduced-thread block specializations on divergent hit lists.
    data = _renderer_inputs(24)
    axis = torch.linspace(-0.45, 0.45, 8, device="cuda")
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    data["rays_o"] = torch.stack(
        (grid_x.reshape(-1), grid_y.reshape(-1), torch.zeros(64, device="cuda")), dim=-1
    )
    data["rays_d"] = torch.tensor([0.0, 0.0, 1.0], device="cuda").repeat(64, 1)
    data["positions"][:, 0] = torch.linspace(-0.35, 0.35, 24, device="cuda")
    data["positions"][:, 1] = torch.linspace(0.3, -0.3, 24, device="cuda")
    data["log_scales"].fill_(-1.8)
    data["grad_rgb"] = torch.randn((64, 3), device="cuda")
    data["grad_t"] = torch.zeros(64, device="cuda")
    data["grad_feat"] = torch.randn((64, 2), device="cuda")
    data["feat_attr"].normal_(mean=0.0, std=0.1)
    for name in ("W1", "b1", "W2", "b2", "W3", "b3"):
        data[name].normal_(mean=0.0, std=0.1)
    dimensions = (8, 8, 8, 8, 1.0e-3)
    expected = reference.render_tiles_backward_tilecenter_full(
        *_base_args(data), data["grad_rgb"], data["grad_t"], *dimensions
    )
    actual = tile_ext.render_tiles_backward_tilecenter_full(
        *_base_args(data), data["grad_rgb"], data["grad_t"], *dimensions
    )
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=5e-4, rtol=5e-4)

    expected = reference.render_tiles_forward_tilecenter_with_feat(
        *_base_args(data), data["feat_attr"], 1.0, 1.0, *dimensions
    )
    actual = tile_ext.render_tiles_forward_tilecenter_with_feat(
        *_base_args(data), data["feat_attr"], 1.0, 1.0, *dimensions
    )
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=5e-4, rtol=5e-4)

    expected = reference.render_tiles_backward_featattr_tilecenter(
        *_base_args(data), data["feat_attr"], data["grad_feat"], 1.0, 1.0, *dimensions
    )
    actual = tile_ext.render_tiles_backward_featattr_tilecenter(
        *_base_args(data), data["feat_attr"], data["grad_feat"], 1.0, 1.0, *dimensions
    )
    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=5e-4)


@pytest.mark.parametrize("feature_slots", [4, 5])
def test_fused_rgb_feature_backward_matches_separate_kernels(tile_ext, feature_slots):
    torch.manual_seed(20260912 + feature_slots)
    data = _renderer_inputs(16)
    data["positions"][:, 0] = torch.linspace(-0.12, 0.12, 16, device="cuda")
    data["positions"][:, 1] = torch.linspace(0.08, -0.08, 16, device="cuda")
    data["log_scales"].fill_(-0.7)
    data["feat_attr"] = torch.randn((16, feature_slots, 7), device="cuda")
    data["grad_feat"] = torch.randn((1, 7), device="cuda")
    data["grad_rgb"] = torch.randn((1, 3), device="cuda")
    for name in ("W1", "b1", "W2", "b2", "W3", "b3"):
        data[name].normal_(mean=0.0, std=0.1)

    rgb_grads = _invoke(tile_ext, "render_tiles_backward_tilecenter_full", data)
    feature_grad = _invoke(tile_ext, "render_tiles_backward_featattr_tilecenter", data)
    fused_grads = _invoke(
        tile_ext, "render_tiles_backward_tilecenter_full_with_feat", data
    )

    assert len(fused_grads) == len(rgb_grads) + 1
    for fused, separate in zip(fused_grads[:-1], rgb_grads, strict=True):
        torch.testing.assert_close(fused, separate, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(fused_grads[-1], feature_grad, atol=2e-5, rtol=2e-5)


def test_tilecenter_reduced_hit_buffer_reports_overflow(tile_ext):
    if os.environ.get("MLPSP_TILECENTER_MAX_HITS", "").strip() != "128":
        pytest.skip("only applies to the optimized 128-hit build")

    data = _renderer_inputs(129)
    data["b3"][:, 3] = -10.0
    with pytest.raises(RuntimeError, match="backward hit capacity 128"):
        _invoke(tile_ext, "render_tiles_backward_tilecenter_full", data)


def test_warp_aggregated_atomic_handles_sparse_peer_masks(tile_ext):
    # Each warp has several non-contiguous equal-address groups.  In particular,
    # lanes {0, 3, 17, 31} catches the old fixed shfl-down tree, which lost the
    # values from lanes not connected to the leader by power-of-two offsets.
    lanes = torch.arange(32, dtype=torch.int32)
    lane_dest = 2 + lanes.remainder(5)
    lane_dest[[0, 3, 17, 31]] = 0
    lane_dest[[1, 2, 16, 30]] = 1
    indices = torch.cat([lane_dest + warp * 8 for warp in range(8)]).cuda()
    values = (torch.arange(indices.numel(), dtype=torch.float32).remainder(7) + 1).cuda()

    actual = tile_ext._test_atomic_add_warp_agg(indices.contiguous(), values.contiguous(), 64)
    expected = torch.zeros(64, device="cuda").scatter_add_(0, indices.long(), values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "name,capacity,error",
    [
        ("render_tiles_autograd", 256, "per-pixel intersection capacity 256 exceeded"),
        ("render_tiles_forward", 256, "per-pixel intersection capacity 256 exceeded"),
        ("render_tiles_forward_per_pixel_with_feat", 256, "per-pixel intersection capacity 256 exceeded"),
        ("render_tiles_backward_count_per_pixel", 256, "per-pixel intersection capacity 256 exceeded"),
        ("render_tiles_backward_emit_per_pixel", 256, "per-pixel intersection capacity 256 exceeded"),
        ("render_tiles_backward_featattr_per_pixel", 256, "per-pixel intersection capacity 256 exceeded"),
        ("render_tiles_forward_tilecenter", 768, "tile candidate capacity 768 exceeded"),
        ("render_tiles_forward_tilecenter_with_feat", 768, "tile candidate capacity 768 exceeded"),
        ("render_tiles_backward_tilecenter_full", 768, "tile candidate capacity 768 exceeded"),
        ("render_tiles_backward_featattr_tilecenter", 768, "tile candidate capacity 768 exceeded"),
        ("render_tiles_backward_tilecenter_full_with_feat", 768, "tile candidate capacity 768 exceeded"),
    ],
)
def test_rgb_and_feature_entry_points_fail_explicitly_on_capacity_overflow(
    tile_ext, name: str, capacity: int, error: str
):
    data = _renderer_inputs(capacity + 1)
    with pytest.raises(RuntimeError, match=error):
        _invoke(tile_ext, name, data)


@pytest.mark.parametrize(
    "name,capacity",
    [
        ("render_tiles_forward", 256),
        ("render_tiles_forward_tilecenter", 512),
    ],
)
def test_capacity_boundary_is_accepted(tile_ext, name: str, capacity: int):
    outputs = _invoke(tile_ext, name, _renderer_inputs(capacity))
    assert all(torch.isfinite(output).all() for output in outputs)


@pytest.mark.parametrize(
    "name",
    [
        "render_tiles_autograd",
        "render_tiles_forward",
        "render_tiles_forward_per_pixel_with_feat",
        "render_tiles_backward_count_per_pixel",
        "render_tiles_backward_emit_per_pixel",
        "render_tiles_backward_featattr_per_pixel",
    ],
)
def test_per_pixel_allows_large_tile_candidate_lists_when_hits_fit(tile_ext, name: str):
    data = _renderer_inputs(386)
    # Preserve 256 true ray/ellipsoid intersections, while keeping a candidate
    # span representative of the published room0 benchmark.  The remaining
    # candidates are valid but far enough off-axis not to hit the test ray.
    data["positions"][256:, 0] = 100.0
    outputs = _invoke(tile_ext, name, data)
    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
    assert all(torch.isfinite(output).all() for output in outputs)


@pytest.mark.parametrize(
    "name",
    [
        "render_tiles_forward_per_pixel_with_feat",
        "render_tiles_forward_tilecenter_with_feat",
        "render_tiles_backward_featattr_per_pixel",
        "render_tiles_backward_featattr_tilecenter",
    ],
)
def test_removed_eight_slot_cube_gating_is_rejected(tile_ext, name: str):
    data = _renderer_inputs(1)
    data["feat_attr"] = torch.zeros((1, 8, 2), device="cuda")
    with pytest.raises(RuntimeError, match=r"slots in \{4,5\}"):
        _invoke(tile_ext, name, data)


def test_tile_offsets_must_cover_the_candidate_list(tile_ext):
    data = _renderer_inputs(2)
    data["offsets"] = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="tile_offsets must start at zero"):
        _invoke(tile_ext, "render_tiles_forward", data)

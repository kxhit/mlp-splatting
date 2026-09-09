#!/usr/bin/env python3
"""Evaluate RGB image quality metrics on train/test split for tile-cuda checkpoints.

Supports:
- scene checkpoints: scene_epoch_xxx.pt (model-only) or train_state_xxx.pt
- dataset loaders: replica / mip360
- split controls compatible with existing training/eval scripts
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
try:
    from .utils.loss_utils import ssim
    from .checkpoints import infer_scene_spec, normalize_scene_state_dict, split_checkpoint
    from .config import parse_args_with_config
except ImportError:  # Support direct script execution.
    from utils.loss_utils import ssim
    from checkpoints import infer_scene_spec, normalize_scene_state_dict, split_checkpoint
    from config import parse_args_with_config


def _strip_state_dict_prefixes(state_dict: dict) -> dict:
    return normalize_scene_state_dict(state_dict, remap_legacy_mlp=False)


def _extract_scene_state_dict(raw_ckpt: dict) -> dict:
    return split_checkpoint(raw_ckpt).scene_state_dict


def _infer_scene_dims(scene_sd: dict) -> tuple[int, int, int, int, int]:
    """Infer (num_primitives, hidden_dim, pe_frequencies, feature_dim_low, feature_slots)."""
    spec = infer_scene_spec(scene_sd)
    return (
        spec.num_primitives,
        spec.hidden_dim,
        spec.pe_frequencies,
        spec.feature_dim,
        spec.feature_slots,
    )


def _remap_legacy_mlp_to_tile_keys(scene_sd: dict) -> dict:
    """Map legacy vmodel MLP keys to tile_cuda keys if needed."""
    return normalize_scene_state_dict(scene_sd)


def _filter_kwargs_for_callable(func, kwargs: dict) -> dict:
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _build_projection_from_focal(
    focal: float,
    width: int,
    height: int,
    near: float = 0.001,
    far: float = 100.0,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    focal = float(focal)
    fov_y = 2.0 * float(torch.atan(torch.tensor((0.5 * height) / (focal + 1e-8))))
    aspect = float(width) / float(height)
    t = float(torch.tan(torch.tensor(fov_y / 2.0)))
    r = t * aspect
    p = torch.zeros(4, 4, dtype=torch.float32, device=device)
    p[0, 0] = 1.0 / r
    p[1, 1] = 1.0 / t
    p[2, 2] = -(far + near) / (far - near)
    p[3, 2] = 1.0
    p[2, 3] = -(2 * far * near) / (far - near)
    return p


def _c2w_to_view_matrix(c2w_3x4: torch.Tensor, device: str | torch.device) -> torch.Tensor:
    c2w_h = torch.eye(4, dtype=torch.float32, device=device)
    c2w_h[:3, :4] = c2w_3x4.to(device=device, dtype=torch.float32)
    return torch.inverse(c2w_h)


def _get_frame_identity(dataset, split_idx: int) -> dict:
    image_path = ""
    image_stem = f"{int(split_idx):04d}"
    if hasattr(dataset, "image_paths_selected") and len(getattr(dataset, "image_paths_selected", [])) > split_idx:
        image_path = str(dataset.image_paths_selected[split_idx])
        image_stem = Path(image_path).stem

    global_idx = int(split_idx)
    if hasattr(dataset, "select_ids") and len(getattr(dataset, "select_ids", [])) > split_idx:
        global_idx = int(dataset.select_ids[split_idx])

    return {
        "split_idx": int(split_idx),
        "global_idx": int(global_idx),
        "image_stem": str(image_stem),
        "image_path": str(image_path),
    }


def _load_dataset_class(loader_name: str):
    loader_name = str(loader_name).lower()
    if loader_name == "mip360":
        try:
            from .mip360_loader import mip360Dataset
        except ImportError:
            from mip360_loader import mip360Dataset

        return mip360Dataset
    if loader_name == "replica":
        this_dir = Path(__file__).resolve().parent
        repo_root = this_dir.parent
        replica_dir = repo_root / "replica"
        if not replica_dir.is_dir():
            raise FileNotFoundError(f"Replica loader directory not found: {replica_dir}")
        sys.path.insert(0, str(replica_dir))
        from replica_loader import mip360Dataset  # pylint: disable=import-error

        return mip360Dataset
    raise ValueError(f"Unknown --dataset_loader={loader_name}")


def _build_dataset(dataset_cls, args):
    kwargs = dict(
        basedir=args.data_dir,
        factor=int(args.factor),
        llffhold=int(args.llffhold),
        split=args.split,
        split_mode=args.split_mode,
        split_test_offset=args.split_test_offset,
        feature_type="none",
        feature_dir=None,
        preload_features=False,
        feature_cache_size=int(args.feature_cache_size),
        feature_resize_to_image=False,
        ray_cache_size=int(args.ray_cache_size),
    )
    return dataset_cls(**_filter_kwargs_for_callable(dataset_cls.__init__, kwargs))


def _load_scene_tile_cuda(
    scene_ckpt_path: str,
    device: torch.device,
    min_points: int,
    tile_size: int = 8,
    feature_gating_beta: float | None = None,
    feature_gating_temp: float | None = None,
):
    try:
        from .cumodel import NeuralPrimitiveSceneTileCUDA
    except ImportError:
        from cumodel import NeuralPrimitiveSceneTileCUDA

    raw_ckpt = torch.load(scene_ckpt_path, map_location="cpu")
    if not isinstance(raw_ckpt, dict):
        raise TypeError(f"Unsupported checkpoint format: {scene_ckpt_path}")
    metadata = raw_ckpt.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    gating_beta = float(
        metadata.get("feature_gating_beta", 1.5)
        if feature_gating_beta is None
        else feature_gating_beta
    )
    gating_temp = float(
        metadata.get("feature_gating_temperature", 2.0)
        if feature_gating_temp is None
        else feature_gating_temp
    )
    encoding_kind = str(metadata.get("encoding_kind", "attenuated_phase_v1"))
    if encoding_kind not in {"attenuated_phase_v1", "fourier_features_v1"}:
        raise ValueError(f"Unsupported checkpoint encoding_kind={encoding_kind!r}")
    use_integrated_pe = encoding_kind == "attenuated_phase_v1"
    pe_sigma = float(metadata.get("encoding_sigma", 0.01))

    scene_sd = _extract_scene_state_dict(raw_ckpt)
    if not isinstance(scene_sd, dict):
        raise TypeError(f"Scene checkpoint does not contain a state_dict dictionary: {scene_ckpt_path}")
    scene_sd = _strip_state_dict_prefixes(scene_sd)
    scene_sd = _remap_legacy_mlp_to_tile_keys(scene_sd)

    n, hidden_dim, pe_frequencies, feature_dim_low, feature_slots = _infer_scene_dims(scene_sd)
    scene = NeuralPrimitiveSceneTileCUDA(
        num_primitives=n,
        hidden_dim=hidden_dim,
        pe_frequencies=pe_frequencies,
        use_integrated_pe=use_integrated_pe,
        pe_sigma=pe_sigma,
        initial_positions=None,
        min_points=int(min_points),
        tile_size=int(tile_size),
        feature_dim_low=int(feature_dim_low),
        feature_slots=int(feature_slots),
        feature_gating_beta=gating_beta,
        feature_gating_temp=gating_temp,
    ).to(device)

    missing, unexpected = scene.load_state_dict(scene_sd, strict=False)
    benign_missing = {
        "encoder.freq_bands",
        "stat_sum_w",
        "stat_max_w",
        "stat_hits",
        "stat_contrib_count",
        "stat_sum_werr",
        "alpha_accum",
    }
    missing = [k for k in missing if k not in benign_missing]
    if missing:
        raise RuntimeError(
            f"Checkpoint is missing required scene tensors: {missing[:10]} "
            f"(total={len(missing)})"
        )
    if unexpected:
        print(f"[Warn] Unexpected keys while loading scene: {unexpected[:10]} (total={len(unexpected)})")
    scene.eval()
    return scene


def _call_scene_rgb(
    scene,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    score_threshold: float,
    view_matrix: torch.Tensor,
    proj_matrix: torch.Tensor,
):
    """Call scene forward in a signature-compatible way across variants."""
    call_target = scene.forward if hasattr(scene, "forward") else scene
    base_kwargs = {
        "score_threshold": float(score_threshold),
        "bg_color": None,
        "view_matrix": view_matrix,
        "proj_matrix": proj_matrix,
        "return_feature": False,
    }

    sig = inspect.signature(call_target)
    params = sig.parameters
    if "ray_o" in params:
        base_kwargs["ray_o"] = rays_o
    if "ray_d" in params:
        base_kwargs["ray_d"] = rays_d
    if "rays_o" in params:
        base_kwargs["rays_o"] = rays_o
    if "rays_d" in params:
        base_kwargs["rays_d"] = rays_d

    filtered = _filter_kwargs_for_callable(call_target, base_kwargs)
    has_named_rays = (("ray_o" in filtered) or ("rays_o" in filtered)) and (("ray_d" in filtered) or ("rays_d" in filtered))
    if has_named_rays:
        out = scene(**filtered)
    else:
        for k in ("ray_o", "ray_d", "rays_o", "rays_d"):
            filtered.pop(k, None)
        out = scene(rays_o, rays_d, **filtered)

    if isinstance(out, (tuple, list)):
        if len(out) == 0:
            raise RuntimeError("Scene forward returned empty tuple/list.")
        out = out[0]
    return out


def _ensure_hwc_rgb_tensor(img_like: torch.Tensor | np.ndarray) -> torch.Tensor:
    if torch.is_tensor(img_like):
        t = img_like.detach()
    else:
        t = torch.from_numpy(np.asarray(img_like))

    if t.ndim != 3:
        raise ValueError(f"RGB image must be 3D, got shape={tuple(t.shape)}")

    if t.shape[-1] == 3:
        out = t
    elif t.shape[0] == 3:
        out = t.permute(1, 2, 0).contiguous()
    else:
        raise ValueError(f"Cannot interpret image layout as HWC/CHW RGB: shape={tuple(t.shape)}")

    out = out.float()
    if out.numel() > 0 and float(out.max()) > 1.5:
        out = out / 255.0
    return out.clamp(0.0, 1.0)


def _resize_hwc_bilinear(img_hwc: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if int(img_hwc.shape[0]) == int(height) and int(img_hwc.shape[1]) == int(width):
        return img_hwc
    t = img_hwc.permute(2, 0, 1).unsqueeze(0)
    t = F.interpolate(t, size=(int(height), int(width)), mode="bilinear", align_corners=False)
    return t.squeeze(0).permute(1, 2, 0).contiguous()


def _build_lpips_model(net: str, device: torch.device):
    try:
        import lpips  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Failed to import lpips. Install dependency (e.g. pip install lpips), "
            "or run with --disable_lpips."
        ) from exc

    model = lpips.LPIPS(net=str(net)).to(device)
    model.eval()
    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate RGB metrics (PSNR/SSIM/LPIPS) for tile-cuda checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        help="Versioned JSON recipe. Command-line values override its defaults.",
    )
    parser.add_argument("--data", "--data-dir", "--data_dir", dest="data_dir", type=str, required=True, help="Scene directory.")
    parser.add_argument(
        "--checkpoint", "--scene-ckpt", "--scene_ckpt", dest="scene_ckpt",
        type=str, required=True,
        help="Path to scene_epoch_xxx.pt or train_state_xxx.pt.",
    )
    parser.add_argument("--dataset", "--dataset-loader", "--dataset_loader", dest="dataset_loader", type=str, default="replica", choices=["replica", "mip360"], help="Dataset loader implementation.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--factor", type=int, default=1, help="Image downsample factor used by dataset loader.")
    parser.add_argument("--llffhold", type=int, default=8, help="Train/test split period.")
    parser.add_argument("--split-mode", "--split_mode", dest="split_mode", type=str, default="replica_feature3dgs", help="Split mode (e.g. llff/replica_feature3dgs).")
    parser.add_argument("--split-test-offset", "--split_test_offset", dest="split_test_offset", type=int, default=None, help="Override split test offset if loader supports it.")
    parser.add_argument("--expected-views", dest="expected_views", type=int, default=0, help="Require exactly this many split views (0 disables the check).")
    parser.add_argument("--ray-cache-size", "--ray_cache_size", dest="ray_cache_size", type=int, default=8, help="Dataset ray cache size.")
    parser.add_argument("--feature-cache-size", "--feature_cache_size", dest="feature_cache_size", type=int, default=16, help="Dataset feature cache size (kept for loader compatibility).")
    parser.add_argument(
        "--tile-order", "--tile_order", dest="tile_order",
        type=str,
        default="tilecenter",
        choices=["tilecenter", "per_pixel"],
        help="Tile traversal mode for renderer.",
    )
    parser.add_argument("--score-threshold", "--score_threshold", dest="score_threshold", type=float, default=3e-3, help="Tile renderer score threshold.")
    parser.add_argument(
        "--max-points-per-leaf", "--min-points", "--min_points",
        dest="max_points_per_leaf", type=int, default=10,
        help="Training octree threshold m, recorded here for provenance.",
    )
    parser.add_argument("--tile-size", "--tile_size", dest="tile_size", type=int, default=8, choices=[8, 16], help="Tile pixel size used by scene constructor (must match training).")
    parser.add_argument("--limit-frames", "--limit_frames", dest="limit_frames", type=int, default=0, help="Evaluate first N frames only (0 means all).")
    parser.add_argument("--warmup-frames", "--warmup_frames", dest="warmup_frames", type=int, default=1, help="Initial frames excluded from FPS timing.")
    parser.add_argument("--disable-lpips", "--disable_lpips", dest="disable_lpips", action="store_true", help="Disable LPIPS computation.")
    parser.add_argument(
        "--lpips-net", "--lpips_net", dest="lpips_net",
        type=str,
        default="vgg",
        choices=["alex", "vgg", "squeeze"],
        help="LPIPS backbone network.",
    )
    parser.add_argument("--output", "--save-json", "--save_json", dest="save_json", type=str, default="", help="Optional output JSON path.")
    parser.add_argument("--device", type=str, default="cuda", help="Device, e.g. cuda or cuda:0.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parse_args_with_config(parser, argv, section="evaluation")
    if args.expected_views < 0:
        parser.error("--expected-views cannot be negative")
    if args.max_points_per_leaf <= 0:
        parser.error("--max-points-per-leaf must be positive")

    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")
    device = torch.device(args.device if (not str(args.device).startswith("cuda") or torch.cuda.is_available()) else "cpu")
    if device.type == "cuda":
        device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        torch.cuda.set_device(device)

    os.environ["MLPSP_TILE_ORDER"] = str(args.tile_order)

    dataset_cls = _load_dataset_class(args.dataset_loader)
    dataset = _build_dataset(dataset_cls, args)
    if len(dataset) == 0:
        raise RuntimeError("Dataset split is empty.")
    if int(args.expected_views) > 0 and len(dataset) != int(args.expected_views):
        raise RuntimeError(
            f"{args.split} split has {len(dataset)} views; expected {int(args.expected_views)}. "
            "Check the dataset archive and split settings."
        )

    scene_ckpt = os.path.abspath(args.scene_ckpt)
    scene = _load_scene_tile_cuda(
        scene_ckpt_path=scene_ckpt,
        device=device,
        min_points=int(args.max_points_per_leaf),
        tile_size=int(args.tile_size),
    )
    lpips_model = None if bool(args.disable_lpips) else _build_lpips_model(net=str(args.lpips_net), device=device)

    max_frames = len(dataset) if int(args.limit_frames) <= 0 else min(len(dataset), int(args.limit_frames))
    if max_frames <= 0:
        raise RuntimeError("No frames selected for evaluation.")

    print(
        f"[EvalRGB] split={args.split} frames={max_frames} loader={args.dataset_loader} "
        f"tile_order={args.tile_order} tile_size={args.tile_size} device={device}"
    )

    per_image: list[dict] = []
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    lpips_values: list[float] | None = ([] if lpips_model is not None else None)
    sum_sse = 0.0
    sum_count = 0
    render_time_s = 0.0
    timed_frames = 0

    for idx in range(max_frames):
        frame_meta = _get_frame_identity(dataset, idx)
        sample = dataset[idx]
        if not isinstance(sample, (tuple, list)) or len(sample) < 4:
            raise RuntimeError(f"Unexpected dataset sample format at idx={idx}")

        rays_o_hw = sample[0]
        rays_d_hw = sample[1]
        gt_img_hwc = _ensure_hwc_rgb_tensor(sample[2])
        pose = sample[3] if len(sample) == 4 else sample[-1]

        if rays_o_hw.ndim < 2 or rays_o_hw.shape[-1] != 3:
            raise ValueError(f"Unexpected rays_o shape at idx={idx}: {tuple(rays_o_hw.shape)}")
        h = int(gt_img_hwc.shape[0])
        w = int(gt_img_hwc.shape[1])

        rays_o = rays_o_hw.reshape(-1, 3).to(device, non_blocking=True)
        rays_d = rays_d_hw.reshape(-1, 3).to(device, non_blocking=True)
        gt_img = gt_img_hwc.to(device, non_blocking=True)

        if hasattr(scene, "set_image_size"):
            scene.set_image_size(w, h)
        view_matrix = _c2w_to_view_matrix(pose, device=device)
        proj_matrix = _build_projection_from_focal(dataset.focal, w, h, device=device)

        use_timing = idx >= int(max(0, args.warmup_frames))
        if use_timing and device.type == "cuda":
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_end = torch.cuda.Event(enable_timing=True)
            ev_start.record()
        elif use_timing:
            t0 = time.perf_counter()

        with torch.no_grad():
            pred_flat = _call_scene_rgb(
                scene=scene,
                rays_o=rays_o,
                rays_d=rays_d,
                score_threshold=float(args.score_threshold),
                view_matrix=view_matrix,
                proj_matrix=proj_matrix,
            )

        if use_timing and device.type == "cuda":
            ev_end.record()
            torch.cuda.synchronize(device)
            render_time_s += float(ev_start.elapsed_time(ev_end) / 1000.0)
            timed_frames += 1
        elif use_timing:
            render_time_s += float(time.perf_counter() - t0)
            timed_frames += 1

        if pred_flat.ndim != 2 or pred_flat.shape[1] != 3:
            raise RuntimeError(f"Unexpected renderer output shape at idx={idx}: {tuple(pred_flat.shape)}")
        if pred_flat.shape[0] != h * w:
            raise RuntimeError(
                f"Renderer output ray count mismatch at idx={idx}: got {pred_flat.shape[0]}, expected {h*w}."
            )
        pred_img = pred_flat.view(h, w, 3).clamp(0.0, 1.0)

        if int(gt_img.shape[0]) != h or int(gt_img.shape[1]) != w:
            gt_img = _resize_hwc_bilinear(gt_img, height=h, width=w)

        mse = torch.mean((pred_img - gt_img) ** 2)
        mse_scalar = float(mse.item())
        if mse_scalar <= 0.0:
            psnr = float("inf")
        else:
            psnr = float(-10.0 * torch.log10(mse).item())
        pred_nchw = pred_img.permute(2, 0, 1).unsqueeze(0)
        gt_nchw = gt_img.permute(2, 0, 1).unsqueeze(0)
        ssim_val = float(ssim(pred_nchw, gt_nchw).item())
        lpips_val = None
        if lpips_model is not None:
            pred_for_lpips = pred_nchw * 2.0 - 1.0
            gt_for_lpips = gt_nchw * 2.0 - 1.0
            lpips_val = float(lpips_model(pred_for_lpips, gt_for_lpips).mean().item())

        numel = int(pred_img.numel())
        sum_sse += float(mse_scalar * numel)
        sum_count += numel
        psnr_values.append(psnr)
        ssim_values.append(ssim_val)
        if lpips_values is not None and lpips_val is not None:
            lpips_values.append(lpips_val)

        per_image.append(
            {
                **frame_meta,
                "height": int(h),
                "width": int(w),
                "mse": float(mse_scalar),
                "psnr": (None if not math.isfinite(psnr) else float(psnr)),
                "ssim": (None if not math.isfinite(ssim_val) else float(ssim_val)),
                "lpips": (None if (lpips_val is None or not math.isfinite(lpips_val)) else float(lpips_val)),
            }
        )
        psnr_show = "inf" if not math.isfinite(psnr) else f"{psnr:.4f}"
        ssim_show = "nan" if not math.isfinite(ssim_val) else f"{ssim_val:.4f}"
        if lpips_val is None:
            lpips_show = "N/A"
        else:
            lpips_show = "nan" if not math.isfinite(lpips_val) else f"{lpips_val:.4f}"
        print(
            f"[{idx+1:04d}/{max_frames:04d}] split_idx={frame_meta['split_idx']} "
            f"psnr={psnr_show} dB ssim={ssim_show} lpips={lpips_show}"
        )

    finite_psnr = np.asarray([v for v in psnr_values if np.isfinite(v)], dtype=np.float64)
    avg_psnr = float(np.mean(finite_psnr)) if finite_psnr.size > 0 else float("inf")
    median_psnr = float(np.median(finite_psnr)) if finite_psnr.size > 0 else float("inf")
    finite_ssim = np.asarray([v for v in ssim_values if np.isfinite(v)], dtype=np.float64)
    avg_ssim = float(np.mean(finite_ssim)) if finite_ssim.size > 0 else float("nan")
    median_ssim = float(np.median(finite_ssim)) if finite_ssim.size > 0 else float("nan")
    if lpips_values is None:
        avg_lpips = None
        median_lpips = None
    else:
        finite_lpips = np.asarray([v for v in lpips_values if np.isfinite(v)], dtype=np.float64)
        avg_lpips = float(np.mean(finite_lpips)) if finite_lpips.size > 0 else float("nan")
        median_lpips = float(np.median(finite_lpips)) if finite_lpips.size > 0 else float("nan")

    if sum_count <= 0:
        global_psnr = float("nan")
    else:
        global_mse = float(sum_sse / float(sum_count))
        global_psnr = float("inf") if global_mse <= 0.0 else float(-10.0 * math.log10(global_mse))

    fps_render = float(timed_frames / render_time_s) if (timed_frames > 0 and render_time_s > 0.0) else float("nan")

    summary = {
        "data_dir": os.path.abspath(args.data_dir),
        "scene_ckpt": scene_ckpt,
        "dataset_loader": str(args.dataset_loader),
        "split": str(args.split),
        "split_mode": str(args.split_mode),
        "factor": int(args.factor),
        "tile_order": str(args.tile_order),
        "tile_size": int(args.tile_size),
        "max_points_per_leaf": int(args.max_points_per_leaf),
        "config": args.config,
        "score_threshold": float(args.score_threshold),
        "lpips_enabled": bool(lpips_model is not None),
        "lpips_net": (None if lpips_model is None else str(args.lpips_net)),
        "num_frames_evaluated": int(max_frames),
        "num_frames_timed": int(timed_frames),
        "warmup_frames": int(max(0, args.warmup_frames)),
        "mean_psnr_db": (None if not math.isfinite(avg_psnr) else float(avg_psnr)),
        "median_psnr_db": (None if not math.isfinite(median_psnr) else float(median_psnr)),
        "global_psnr_db": (None if not math.isfinite(global_psnr) else float(global_psnr)),
        "mean_ssim": (None if not math.isfinite(avg_ssim) else float(avg_ssim)),
        "median_ssim": (None if not math.isfinite(median_ssim) else float(median_ssim)),
        "mean_lpips": (
            None
            if (avg_lpips is None or not math.isfinite(float(avg_lpips)))
            else float(avg_lpips)
        ),
        "median_lpips": (
            None
            if (median_lpips is None or not math.isfinite(float(median_lpips)))
            else float(median_lpips)
        ),
        "render_fps": (None if not math.isfinite(fps_render) else float(fps_render)),
        "per_image": per_image,
    }

    print("")
    print("=== RGB PSNR Summary ===")
    print(f"Frames: {max_frames}")
    print(f"Mean PSNR (image-wise): {'inf' if not math.isfinite(avg_psnr) else f'{avg_psnr:.6f}'} dB")
    print(f"Median PSNR: {'inf' if not math.isfinite(median_psnr) else f'{median_psnr:.6f}'} dB")
    print(f"Global PSNR (all pixels): {'inf' if not math.isfinite(global_psnr) else f'{global_psnr:.6f}'} dB")
    print(f"Mean SSIM: {'nan' if not math.isfinite(avg_ssim) else f'{avg_ssim:.6f}'}")
    print(f"Median SSIM: {'nan' if not math.isfinite(median_ssim) else f'{median_ssim:.6f}'}")
    if avg_lpips is None or median_lpips is None:
        print("Mean LPIPS: N/A (disabled)")
        print("Median LPIPS: N/A (disabled)")
    else:
        print(f"Mean LPIPS: {'nan' if not math.isfinite(avg_lpips) else f'{avg_lpips:.6f}'}")
        print(f"Median LPIPS: {'nan' if not math.isfinite(median_lpips) else f'{median_lpips:.6f}'}")
    if math.isfinite(fps_render):
        print(f"Render FPS (after warmup): {fps_render:.3f}")

    if args.save_json.strip():
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[EvalRGB] Saved JSON: {out_path}")


if __name__ == "__main__":
    main()

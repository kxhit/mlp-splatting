#!/usr/bin/env python3
"""Reproducible RGB and fused RGB+feature inference benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mip360 import eval_rgb_psnr as evaluation  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _portable_command(argv: list[str]) -> str:
    """Keep benchmark arguments without leaking machine-specific paths."""

    replacements = {
        "--data": "<DATA_DIR>",
        "--data-dir": "<DATA_DIR>",
        "--data_dir": "<DATA_DIR>",
        "--checkpoint": "<CHECKPOINT>",
        "--output": "<OUTPUT_JSON>",
    }
    result: list[str] = [Path(argv[0]).name] if argv else []
    index = 1
    while index < len(argv):
        token = argv[index]
        matched = next((option for option in replacements if token == option), None)
        if matched is not None and index + 1 < len(argv):
            result.extend((token, replacements[matched]))
            index += 2
            continue
        matched = next((option for option in replacements if token.startswith(f"{option}=")), None)
        if matched is not None:
            result.append(f"{matched}={replacements[matched]}")
        else:
            result.append(token)
        index += 1
    return " ".join(result)


def _portable_view_identity(identity: dict) -> dict:
    """Retain split identity while reducing an image path to its filename."""

    result = dict(identity)
    image_path = result.pop("image_path", None)
    if image_path:
        result["image_file"] = Path(str(image_path)).name
    return result


def _timing_stats(values: list[float]) -> dict[str, float | int]:
    samples = np.asarray(values, dtype=np.float64)
    mean = float(samples.mean())
    median = float(np.median(samples))
    std = float(samples.std(ddof=1)) if samples.size > 1 else 0.0
    return {
        "samples": int(samples.size),
        "mean_ms": mean,
        "median_ms": median,
        "p95_ms": float(np.percentile(samples, 95)),
        "std_ms": std,
        "coefficient_of_variation": std / mean if mean else 0.0,
        "fps_from_mean": 1000.0 / mean,
        "fps_from_median": 1000.0 / median,
    }


def _quality(predictions: list[torch.Tensor], targets: list[torch.Tensor]) -> dict:
    per_view = []
    total_squared_error = 0.0
    total_values = 0
    for prediction, target in zip(predictions, targets, strict=True):
        squared_error = (prediction - target).square()
        mse = float(squared_error.mean().item())
        per_view.append(float(-10.0 * math.log10(mse)) if mse else float("inf"))
        total_squared_error += float(squared_error.sum().item())
        total_values += squared_error.numel()
    global_mse = total_squared_error / total_values
    return {
        "mean_psnr_db": float(np.mean(per_view)),
        "median_psnr_db": float(np.median(per_view)),
        "global_psnr_db": float(-10.0 * math.log10(global_mse)),
        "per_view_psnr_db": per_view,
    }


def _render(scene, frame, threshold: float, return_feature: bool):
    rays_o, rays_d, view, projection, width, height = frame
    scene.set_image_size(width, height)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    start.record()
    with torch.inference_mode():
        output = scene(
            rays_o,
            rays_d,
            score_threshold=threshold,
            view_matrix=view,
            proj_matrix=projection,
            return_feature=return_feature,
        )
    end.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    cuda_ms = float(start.elapsed_time(end))
    if return_feature:
        rgb, feature = output
        feature_shape = list(feature.shape)
        del feature
    else:
        rgb = output
        feature_shape = None
    return rgb.view(height, width, 3).clamp(0.0, 1.0), wall_ms, cuda_ms, feature_shape


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the canonical tile CUDA renderer on a fixed view split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", "--data-dir", "--data_dir", dest="data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="Destination JSON file.")
    parser.add_argument("--dataset-loader", choices=("replica", "mip360"), default="replica")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--split-mode", default="replica_feature3dgs")
    parser.add_argument("--split-test-offset", type=int, default=2)
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--tile-order", "--order", choices=("tilecenter", "per_pixel"), default="tilecenter")
    parser.add_argument("--tile-size", type=int, choices=(8, 16), default=8)
    parser.add_argument("--score-threshold", "--threshold", type=float, default=5e-3)
    parser.add_argument("--warmup-cycles", type=int, default=3)
    parser.add_argument("--timed-cycles", type=int, default=3)
    parser.add_argument("--expected-views", type=int, default=0, help="Fail unless the split has this size; 0 disables the check.")
    parser.add_argument(
        "--max-points-per-leaf", "--min-points", "--min_points",
        dest="max_points_per_leaf",
        type=int,
        default=10,
        help="Octree leaf capacity; legacy --min-points spelling is retained for compatibility.",
    )
    parser.add_argument("--feature-gating-beta", type=float, default=None)
    parser.add_argument("--feature-gating-temp", type=float, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ray-cache-size", type=int, default=16)
    parser.add_argument("--feature-cache-size", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.warmup_cycles < 0 or args.timed_cycles <= 0:
        raise ValueError("warmup cycles must be non-negative and timed cycles must be positive")
    if not 0.0 < args.score_threshold < 1.0:
        raise ValueError("score threshold must lie strictly between zero and one")
    if not torch.cuda.is_available():
        raise RuntimeError("the tile renderer requires CUDA")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must select a CUDA device")
    device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    os.environ["MLPSP_TILE_ORDER"] = args.tile_order

    dataset_args = argparse.Namespace(
        data_dir=args.data_dir,
        factor=args.factor,
        llffhold=args.llffhold,
        split=args.split,
        split_mode=args.split_mode,
        split_test_offset=args.split_test_offset,
        feature_cache_size=args.feature_cache_size,
        ray_cache_size=args.ray_cache_size,
    )
    dataset_cls = evaluation._load_dataset_class(args.dataset_loader)
    dataset = evaluation._build_dataset(dataset_cls, dataset_args)
    if not len(dataset):
        raise RuntimeError("the selected split is empty")
    if args.expected_views and len(dataset) != args.expected_views:
        raise RuntimeError(f"expected {args.expected_views} views, found {len(dataset)}")

    scene = evaluation._load_scene_tile_cuda(
        args.checkpoint,
        device,
        args.max_points_per_leaf,
        args.tile_size,
        args.feature_gating_beta,
        args.feature_gating_temp,
    )
    scene.tile_order_mode_override = args.tile_order
    if scene.feat_attr is None:
        raise RuntimeError("checkpoint has no feature attributes; fused feature timing is unavailable")

    frames = []
    targets = []
    identities = []
    for index in range(len(dataset)):
        sample = dataset[index]
        rays_o_hw, rays_d_hw, target_like = sample[:3]
        pose = sample[3] if len(sample) == 4 else sample[-1]
        target = evaluation._ensure_hwc_rgb_tensor(target_like).to(device)
        height, width = map(int, target.shape[:2])
        frames.append(
            (
                rays_o_hw.reshape(-1, 3).to(device),
                rays_d_hw.reshape(-1, 3).to(device),
                evaluation._c2w_to_view_matrix(pose, device),
                evaluation._build_projection_from_focal(dataset.focal, width, height, device=device),
                width,
                height,
            )
        )
        targets.append(target)
        identities.append(evaluation._get_frame_identity(dataset, index))

    for return_feature in (False, True):
        for _ in range(args.warmup_cycles):
            for frame in frames:
                prediction, _, _, _ = _render(scene, frame, args.score_threshold, return_feature)
                del prediction

    results = {}
    rgb_reference: list[torch.Tensor] = []
    for return_feature, label in ((False, "rgb_only"), (True, "rgb_feature")):
        wall_times: list[float] = []
        cuda_times: list[float] = []
        first_cycle_predictions = []
        feature_shapes = []
        parity_max_abs = 0.0
        parity_squared_error = 0.0
        parity_values = 0
        for cycle in range(args.timed_cycles):
            for index, frame in enumerate(frames):
                prediction, wall_ms, cuda_ms, feature_shape = _render(
                    scene, frame, args.score_threshold, return_feature
                )
                wall_times.append(wall_ms)
                cuda_times.append(cuda_ms)
                if cycle == 0:
                    first_cycle_predictions.append(prediction)
                    if return_feature:
                        delta = prediction - rgb_reference[index]
                        parity_max_abs = max(parity_max_abs, float(delta.abs().max().item()))
                        parity_squared_error += float(delta.square().sum().item())
                        parity_values += delta.numel()
                        feature_shapes.append(feature_shape)
                    else:
                        rgb_reference.append(prediction.detach().clone())
                del prediction
        result = {
            "wall": _timing_stats(wall_times),
            "cuda_event": _timing_stats(cuda_times),
            "quality": _quality(first_cycle_predictions, targets),
            "per_frame_wall_ms": wall_times,
            "per_frame_cuda_ms": cuda_times,
        }
        if return_feature:
            rmse = math.sqrt(parity_squared_error / parity_values)
            result["rgb_parity_vs_rgb_only"] = {
                "max_abs": parity_max_abs,
                "rmse": rmse,
                "psnr_db": float(-20.0 * math.log10(rmse)) if rmse else None,
            }
            result["feature_shapes"] = feature_shapes
        results[label] = result

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    device_properties = torch.cuda.get_device_properties(device)
    source_files = [
        REPO_ROOT / "mip360/checkpoints.py",
        REPO_ROOT / "mip360/cumodel.py",
        REPO_ROOT / "mip360/cuda/np_tile_kernels.cu",
        REPO_ROOT / "mip360/data_validation.py",
        REPO_ROOT / "mip360/eval_rgb_psnr.py",
        REPO_ROOT / "mip360/mip360_loader.py",
        REPO_ROOT / "replica/replica_loader.py",
        REPO_ROOT / "tools/benchmark_inference.py",
    ]
    status = _git("status", "--short")
    report = {
        "schema_version": 1,
        "created_unix": time.time(),
        "configuration": {
            "tile_order": args.tile_order,
            "tile_size": args.tile_size,
            "score_threshold": args.score_threshold,
            "warmup_cycles": args.warmup_cycles,
            "timed_cycles": args.timed_cycles,
            "views": len(frames),
            "dataset_loader": args.dataset_loader,
            "split": args.split,
            "split_mode": args.split_mode,
            "split_test_offset": args.split_test_offset,
            "factor": args.factor,
            "max_points_per_leaf": args.max_points_per_leaf,
            "feature_gating_beta": scene.feature_gating_beta,
            "feature_gating_temperature": scene.feature_gating_temp,
            "feature_slots": scene.feature_slots,
            "feature_dimension": scene.feature_dim_low,
        },
        "hardware": {
            "device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": torch.cuda.get_device_name(device),
            "compute_capability": (
                f"{device_properties.major}.{device_properties.minor}"
            ),
            "total_memory_gib": device_properties.total_memory / (1024 ** 3),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "provenance": {
            "git_head": _git("rev-parse", "HEAD"),
            "git_status_porcelain": status.splitlines() if status else [],
            "checkpoint_file": checkpoint.name,
            "checkpoint_sha256": _sha256(checkpoint),
            "source_sha256": {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in source_files},
            "views": [_portable_view_identity(identity) for identity in identities],
            "command": _portable_command(sys.argv),
        },
        "results": results,
    }
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(destination),
                "rgb_fps": results["rgb_only"]["wall"]["fps_from_mean"],
                "feature_fps": results["rgb_feature"]["wall"]["fps_from_mean"],
                "psnr_db": results["rgb_only"]["quality"]["mean_psnr_db"],
                "rgb_parity": results["rgb_feature"]["rgb_parity_vs_rgb_only"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

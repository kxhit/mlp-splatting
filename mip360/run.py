#!/usr/bin/env python3
"""Train the canonical tile-based MLP Splatting model."""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

try:
    from .train import train_scene_llff
except ImportError:  # Support ``python mip360/run.py``.
    from train import train_scene_llff


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train MLP Splatting with the feature-capable tile CUDA renderer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", "--data_dir", dest="data_dir", required=True, help="Scene directory.")
    parser.add_argument(
        "--output", "--output_dir", dest="output_dir", default="outputs",
        help="Root directory for checkpoints and logs.",
    )
    parser.add_argument("--factor", type=int, default=1, choices=(1, 2, 4, 8), help="Image downsample factor.")
    parser.add_argument(
        "--iterations", "--num_epochs", dest="num_epochs", type=int, default=60_000,
        help="Number of optimization iterations.",
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--min-points", "--min_points", dest="min_points", type=int, default=10,
        help="Minimum COLMAP points per octree leaf.",
    )

    renderer = parser.add_argument_group("tile renderer")
    renderer.add_argument(
        "--tile-order", "--tile_order", dest="tile_order",
        choices=("tilecenter", "per_pixel"), default="tilecenter",
        help="Tile-center ordering is fast; per-pixel ordering is the quality reference.",
    )
    renderer.add_argument(
        "--tile-size", "--tile_size", dest="tile_size", type=int,
        choices=(8, 16), default=8,
    )
    renderer.add_argument(
        "--tile-mosaic", "--tile_mosaic", dest="tile_mosaic",
        action=argparse.BooleanOptionalAction, default=True,
        help="Compose each training frame from tiles sampled across views.",
    )
    renderer.add_argument(
        "--mosaic-views", "--tile_mosaic_k_views", dest="tile_mosaic_k_views",
        type=int, default=4,
    )

    feature = parser.add_argument_group("feature distillation")
    feature.add_argument(
        "--feature-type", "--feature_type", dest="feature_type",
        choices=("none", "lseg", "sam"), default="none",
    )
    feature.add_argument("--feature-dir", "--feature_dir", dest="feature_dir", default=None)
    feature.add_argument(
        "--feature-loss-weight", "--feature_loss_weight", dest="feature_loss_weight",
        type=float, default=1.0,
    )
    feature.add_argument(
        "--feature-lowdim-divisor", "--feature_lowdim_divisor",
        dest="feature_lowdim_divisor", type=int, default=1,
        help="Use 1 for the paper's direct 512D LSeg features.",
    )
    feature.add_argument(
        "--feature-slots", "--feature_slots", dest="feature_slots", type=int,
        choices=(4, 5), default=4,
        help="Four direct tetrahedral slots are canonical; five is accepted only for legacy resumes.",
    )
    feature.add_argument(
        "--gating-beta", "--feature_gating_beta", dest="feature_gating_beta",
        type=float, default=1.5,
    )
    feature.add_argument(
        "--gating-temperature", "--feature_gating_temp", dest="feature_gating_temp",
        type=float, default=2.0,
    )
    feature.add_argument(
        "--decoder-hidden-dim", "--feature_decoder_hidden_dim",
        dest="feature_decoder_hidden_dim", type=int, default=128,
    )

    optimization = parser.add_argument_group("optimization")
    optimization.add_argument("--learning-rate", "--lr", dest="lr", type=float, default=1e-3)
    optimization.add_argument(
        "--log-scale-lr", "--log_scale_lr", dest="log_scale_lr", type=float, default=1e-2,
    )
    optimization.add_argument(
        "--lambda-dssim", "--lambda_dssim", dest="lambda_dssim", type=float, default=0.1,
    )
    optimization.add_argument(
        "--log-scale-min", "--log_scale_min", dest="log_scale_min", type=float, default=-12.0,
    )
    optimization.add_argument(
        "--log-scale-max", "--log_scale_max", dest="log_scale_max", type=float, default=4.0,
    )
    optimization.add_argument(
        "--log-scale-grad-clip", "--log_scale_grad_clip",
        dest="log_scale_grad_clip", type=float, default=0.0,
    )
    optimization.add_argument(
        "--black-border-mask", "--black_border_mask", dest="black_border_mask",
        choices=("auto", "on", "off"), default="auto",
    )
    optimization.add_argument(
        "--black-border-threshold", "--black_border_threshold",
        dest="black_border_threshold", type=float, default=1e-3,
    )

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--resume", "--resume_ckpt", dest="resume_ckpt", default=None)
    runtime.add_argument("--tag", default=None)
    runtime.add_argument(
        "--preload-features", "--preload_features", dest="preload_features", action="store_true",
    )
    runtime.add_argument(
        "--feature-cache-size", "--feature_cache_size",
        dest="feature_cache_size", type=int, default=64,
    )
    runtime.add_argument(
        "--ray-cache-size", "--ray_cache_size", dest="ray_cache_size", type=int, default=64,
    )
    runtime.add_argument(
        "--workers", "--dataloader_num_workers", dest="dataloader_num_workers", type=int, default=2,
    )
    runtime.add_argument(
        "--pin-memory", dest="dataloader_pin_memory",
        action=argparse.BooleanOptionalAction, default=True,
    )
    runtime.add_argument(
        "--persistent-workers", dest="dataloader_persistent_workers",
        action=argparse.BooleanOptionalAction, default=True,
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.num_epochs <= 0:
        parser.error("--iterations must be positive")
    if args.feature_lowdim_divisor <= 0:
        parser.error("--feature-lowdim-divisor must be positive")
    if args.feature_slots == 5 and args.resume_ckpt is None:
        parser.error("--feature-slots 5 is reserved for resuming legacy checkpoints")
    set_seed(args.seed)

    # The public entrypoint freezes the experimental switches that accumulated
    # during development: no PE ramp, density control, center warm-up, or late
    # learning-rate decay.
    train_scene_llff(
        args.data_dir,
        args.output_dir,
        factor=args.factor,
        num_epochs=args.num_epochs,
        lr=args.lr,
        # Used only if a loader has no point-cloud initialization.  Standard
        # COLMAP/Replica scenes derive the count from --min-points.
        initial_num_primitives=1024,
        min_points=args.min_points,
        engine="tile_cuda",
        tile_order=args.tile_order,
        tile_size=args.tile_size,
        resume_ckpt=args.resume_ckpt,
        run_tag=args.tag,
        enable_pe_ramp=False,
        enable_center_lr_warmup=False,
        tile_mosaic=args.tile_mosaic,
        tile_mosaic_k_views=args.tile_mosaic_k_views,
        tile_mosaic_warmup_steps=0,
        feature_type=args.feature_type,
        feature_dir=args.feature_dir,
        feature_loss_weight=args.feature_loss_weight,
        feature_decoder_hidden_dim=args.feature_decoder_hidden_dim,
        feature_lowdim_divisor=args.feature_lowdim_divisor,
        feature_slots=args.feature_slots,
        feature_gating_beta=args.feature_gating_beta,
        feature_gating_temp=args.feature_gating_temp,
        feature_mock_from_rgb=False,
        lambda_dssim=args.lambda_dssim,
        black_border_mask=args.black_border_mask,
        black_border_threshold=args.black_border_threshold,
        late_cosine_start=args.num_epochs + 1,
        late_cosine_total_steps=args.num_epochs + 2,
        late_cosine_min_mlp=1.0,
        late_cosine_min_scale=1.0,
        late_cosine_min_position=1.0,
        late_cosine_min_extra=1.0,
        log_scale_lr=args.log_scale_lr,
        log_scale_min=args.log_scale_min,
        log_scale_max=args.log_scale_max,
        log_scale_grad_clip=args.log_scale_grad_clip,
        renorm_quats_after_step=True,
        sanitize_nonfinite_params=True,
        reset_nonfinite_optimizer_state=True,
        preload_features=args.preload_features,
        feature_cache_size=args.feature_cache_size,
        ray_cache_size=args.ray_cache_size,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=args.dataloader_pin_memory,
        dataloader_persistent_workers=args.dataloader_persistent_workers,
        dataloader_prefetch_factor=2,
        tile_mosaic_feature_prefetch=True,
        tile_mosaic_prefetch_workers=max(1, args.dataloader_num_workers),
        tile_mosaic_prefetch_depth=2,
        tile_mosaic_prefetch_cache_size=0,
    )


if __name__ == "__main__":
    main()

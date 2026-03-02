import argparse
import random

import numpy as np
import torch

from train import train_scene_llff

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/media/sjkim/HDD_3TB/dataset/360_v2/bonsai")
    parser.add_argument("--output_dir", type=str, default="/home/sjkim/Documents/mlpsp/output/mip360")
    parser.add_argument("--num_epochs", type=int, default=800000)
    parser.add_argument("--lr", type=float, default=1e-3, help="Base learning rate for trainable parameter groups.")
    parser.add_argument("--batch_size", type=int, default=12000, help="Used by vmodel engine (random rays).")
    parser.add_argument("--initial_num_primitives", type=int, default=2183)
    parser.add_argument("--min_points", type=int, default=35, help="Minimum points per octree leaf when tile_cuda constructor supports it.")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--engine",
        type=str,
        default="vmodel",
        choices=["vmodel", "tile_cuda"],
        help="Training backend. 'tile_cuda' requires full-frame training and a CUDA backward implementation.",
    )
    parser.add_argument(
        "--tile_order",
        type=str,
        default="per_pixel",
        choices=["per_pixel", "tilecenter", "tilecenter_split", "globaldepth"],
        help="Only used when --engine tile_cuda; maps to MLPSP_TILE_ORDER.",
    )
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default="",
        help="Path to checkpoint for continuing training. Supports train-state or model-only checkpoints.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional run tag. If provided, used as random_tag; otherwise keep current default behavior.",
    )
    parser.add_argument(
        "--enable_pe_ramp",
        action="store_true",
        help="Enable BARF-style coarse-to-fine positional encoding frequency ramp.",
    )
    parser.add_argument(
        "--pe_ramp_start",
        type=int,
        default=0,
        help="Training step to start PE frequency ramp.",
    )
    parser.add_argument(
        "--pe_ramp_steps",
        type=int,
        default=50000,
        help="Number of steps to reach full PE frequencies after pe_ramp_start.",
    )
    parser.add_argument(
        "--enable_center_lr_warmup",
        action="store_true",
        help="Enable center(position) learning-rate warmup.",
    )
    parser.add_argument(
        "--center_lr_warmup_steps",
        type=int,
        default=5000,
        help="Number of warmup steps for center(position) learning rate.",
    )
    parser.add_argument(
        "--center_lr_start_factor",
        type=float,
        default=0.1,
        help="Start factor for center(position) LR warmup (0~1].",
    )
    parser.add_argument(
        "--tile_mosaic",
        action="store_true",
        help="Enable tile-wise multi-view mosaic training (tile_cuda + tilecenter only).",
    )
    parser.add_argument(
        "--tile_mosaic_k_views",
        type=int,
        default=8,
        help="Max number of candidate views sampled per mosaic step.",
    )
    parser.add_argument(
        "--tile_mosaic_warmup_steps",
        type=int,
        default=0,
        help="Warmup steps before enabling tile_mosaic (0 means immediate).",
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        default="none",
        choices=["none", "lseg", "sam"],
        help="Enable featuremetric branch with pre-extracted foundational features.",
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        default="",
        help="Optional explicit feature directory path. If omitted, dataset defaults are used.",
    )
    parser.add_argument(
        "--feature_loss_weight",
        type=float,
        default=1.0,
        help="Weight for L1 featuremetric loss term.",
    )
    parser.add_argument(
        "--feature_decoder_hidden_dim",
        type=int,
        default=128,
        help="Hidden channel width for 1x1-3x3-1x1 feature decoder.",
    )
    parser.add_argument(
        "--feature_lowdim_divisor",
        type=int,
        default=4,
        help="Low-dim rendered feature is GT dim divided by this factor (Feature-3DGS-style speedup).",
    )
    parser.add_argument(
        "--feature_mock_from_rgb",
        action="store_true",
        help="If GT feature files are missing, use deterministic RGB-projected mock features (for smoke tests).",
    )
    parser.add_argument(
        "--lambda_dssim",
        type=float,
        default=0.1,
        help="RGB loss blend coefficient: (1-lambda)*L1 + lambda*(1-SSIM).",
    )
    parser.add_argument(
        "--late_cosine_start",
        type=int,
        default=80000,
        help="Training step to start late-cosine LR decay.",
    )
    parser.add_argument(
        "--late_cosine_total_steps",
        type=int,
        default=110000,
        help="Training step where late-cosine reaches final min multiplier.",
    )
    parser.add_argument(
        "--late_cosine_min_mlp",
        type=float,
        default=1e-2,
        help="Minimum LR multiplier for MLP parameter group at late-cosine end.",
    )
    parser.add_argument(
        "--late_cosine_min_scale",
        type=float,
        default=0.2,
        help="Minimum LR multiplier for log-scale parameter group at late-cosine end.",
    )
    parser.add_argument(
        "--late_cosine_min_position",
        type=float,
        default=0.3,
        help="Minimum LR multiplier for center(position) parameter group at late-cosine end.",
    )
    parser.add_argument(
        "--late_cosine_min_extra",
        type=float,
        default=0.2,
        help="Minimum LR multiplier for optional extra groups (quats/features/decoder).",
    )
    parser.add_argument(
        "--enable_densify_prune",
        action="store_true",
        help="Re-enable densification/pruning (disabled by default for controlled experiments).",
    )
    parser.add_argument(
        "--densify_prune_interval",
        type=int,
        default=5000,
        help="Interval (steps) for densify/prune stats loop when enabled.",
    )
    parser.add_argument(
        "--densify_prune_warmup_steps",
        type=int,
        default=2000,
        help="Warmup steps before first densify/prune stats loop.",
    )
    parser.add_argument(
        "--densify_prune_num_views",
        type=int,
        default=20,
        help="Number of training views sampled to accumulate stats before densify/prune.",
    )
    parser.add_argument(
        "--preload_features",
        action="store_true",
        help="Preload all feature tensors into RAM at startup.",
    )
    parser.add_argument(
        "--feature_cache_size",
        type=int,
        default=64,
        help="LRU cache size for lazy feature loading (used when --preload_features is off).",
    )
    parser.add_argument(
        "--ray_cache_size",
        type=int,
        default=64,
        help="LRU cache size for per-view ray tensors in dataset.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=2,
        help="Number of DataLoader workers for tile_cuda per-image sampling.",
    )
    parser.add_argument(
        "--dataloader_pin_memory",
        action="store_true",
        help="Enable pinned-memory DataLoader batches for faster H2D transfers.",
    )
    parser.add_argument(
        "--dataloader_persistent_workers",
        action="store_true",
        help="Keep DataLoader workers alive across epochs (requires num_workers > 0).",
    )
    parser.add_argument(
        "--dataloader_prefetch_factor",
        type=int,
        default=2,
        help="Number of prefetched batches per worker (requires num_workers > 0).",
    )
    parser.set_defaults(
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
    )
    args = parser.parse_args()

    set_seed(args.seed)
    train_scene_llff(
        args.data_dir,
        args.output_dir,
        num_epochs=args.num_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        initial_num_primitives=args.initial_num_primitives,
        min_points=args.min_points,
        engine=args.engine,
        tile_order=args.tile_order,
        resume_ckpt=args.resume_ckpt if args.resume_ckpt else None,
        run_tag=args.tag if args.tag else None,
        enable_pe_ramp=args.enable_pe_ramp,
        pe_ramp_start=args.pe_ramp_start,
        pe_ramp_steps=args.pe_ramp_steps,
        enable_center_lr_warmup=args.enable_center_lr_warmup,
        center_lr_warmup_steps=args.center_lr_warmup_steps,
        center_lr_start_factor=args.center_lr_start_factor,
        tile_mosaic=args.tile_mosaic,
        tile_mosaic_k_views=args.tile_mosaic_k_views,
        tile_mosaic_warmup_steps=args.tile_mosaic_warmup_steps,
        feature_type=args.feature_type,
        feature_dir=(args.feature_dir if args.feature_dir else None),
        feature_loss_weight=args.feature_loss_weight,
        feature_decoder_hidden_dim=args.feature_decoder_hidden_dim,
        feature_lowdim_divisor=args.feature_lowdim_divisor,
        feature_mock_from_rgb=args.feature_mock_from_rgb,
        lambda_dssim=args.lambda_dssim,
        late_cosine_start=args.late_cosine_start,
        late_cosine_total_steps=args.late_cosine_total_steps,
        late_cosine_min_mlp=args.late_cosine_min_mlp,
        late_cosine_min_scale=args.late_cosine_min_scale,
        late_cosine_min_position=args.late_cosine_min_position,
        late_cosine_min_extra=args.late_cosine_min_extra,
        disable_densify_prune=(not args.enable_densify_prune),
        densify_prune_interval=args.densify_prune_interval,
        densify_prune_warmup_steps=args.densify_prune_warmup_steps,
        densify_prune_num_views=args.densify_prune_num_views,
        preload_features=args.preload_features,
        feature_cache_size=args.feature_cache_size,
        ray_cache_size=args.ray_cache_size,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=args.dataloader_pin_memory,
        dataloader_persistent_workers=args.dataloader_persistent_workers,
        dataloader_prefetch_factor=args.dataloader_prefetch_factor,
    )
    

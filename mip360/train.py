from __future__ import annotations

from multiprocessing.util import info
import os
import re
import uuid
from tqdm import tqdm
from datetime import datetime
import copy
import time
from collections import OrderedDict
import wandb
USE_WANDB = True

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from vmodel import NeuralPrimitiveScene
# from vmodel import NeuralPrimitiveScene
from mip360_loader import mip360Dataset
from feature_decoder import FeatureUpsampleDecoder


def _barf_frequency_weights(alpha: float, num_frequencies: int, device: torch.device | str) -> torch.Tensor:
    """BARF-style smooth frequency ramp weights."""
    if num_frequencies <= 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    k = torch.arange(num_frequencies, dtype=torch.float32, device=device)
    x = torch.clamp(torch.tensor(float(alpha), dtype=torch.float32, device=device) - k, 0.0, 1.0)
    return 0.5 * (1.0 - torch.cos(torch.pi * x))


def _set_scene_frequency_weights(scene: nn.Module, weights: torch.Tensor):
    encoder = getattr(scene, "encoder", None)
    if encoder is None:
        return
    if hasattr(encoder, "set_runtime_frequency_weights"):
        encoder.set_runtime_frequency_weights(weights)


def _build_mock_feature_projection(out_dim: int, device: torch.device | str) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(12345)
    proj = torch.randn(3, out_dim, generator=g, dtype=torch.float32)
    proj = proj / (proj.norm(dim=0, keepdim=True) + 1e-12)
    return proj.to(device=device)


def zero_grads_for_ids(scene: nn.Module, fixed_ids: torch.Tensor):
    # Densify/prune is disabled in this branch; keep compatibility with legacy call sites.
    return


def _extract_epoch_from_path(path: str) -> int:
    match = re.search(r"scene_epoch_(\d+)\.pt$", path)
    if match is None:
        return 0
    return int(match.group(1))


def _infer_num_primitives(state_dict: dict) -> int | None:
    pos = state_dict.get("positions")
    if not isinstance(pos, torch.Tensor):
        return None
    if pos.ndim != 2 or pos.shape[0] <= 0:
        return None
    return int(pos.shape[0])


def _infer_feature_dim_low(state_dict: dict) -> int | None:
    feat_attr = state_dict.get("feat_attr")
    if not isinstance(feat_attr, torch.Tensor):
        return None
    if feat_attr.ndim != 2 or feat_attr.shape[1] <= 0:
        return None
    return int(feat_attr.shape[1])


def _strip_state_dict_prefixes(state_dict: dict) -> dict:
    out = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("_orig_mod."):
            nk = nk[len("_orig_mod."):]
        if nk.startswith("module."):
            nk = nk[len("module."):]
        out[nk] = v
    return out


def _load_decoder_state(feature_decoder: nn.Module, decoder_state_dict: dict, source: str) -> None:
    sd = _strip_state_dict_prefixes(decoder_state_dict)
    model_sd = feature_decoder.state_dict()
    missing = [k for k in model_sd.keys() if k not in sd]
    if missing:
        raise KeyError(f"[Resume] Decoder checkpoint at {source} is missing keys: {missing}")
    shape_mismatch = []
    for k, v in model_sd.items():
        ckpt_v = sd.get(k)
        if isinstance(ckpt_v, torch.Tensor) and ckpt_v.shape != v.shape:
            shape_mismatch.append((k, tuple(ckpt_v.shape), tuple(v.shape)))
    if shape_mismatch:
        raise ValueError(f"[Resume] Decoder checkpoint at {source} has shape mismatch: {shape_mismatch[:3]}")
    filtered_sd = {k: sd[k] for k in model_sd.keys()}
    feature_decoder.load_state_dict(filtered_sd, strict=True)


def _iter_resume_decoder_candidates(resume_ckpt: str, epoch: int | None, output_dir: str | None):
    candidates = []
    resume_ckpt = os.path.abspath(resume_ckpt)
    base = os.path.basename(resume_ckpt)
    ckpt_dir = os.path.dirname(resume_ckpt)

    if base.startswith("scene_"):
        candidates.append(_paired_decoder_path(resume_ckpt))
    if base.startswith("train_state_"):
        scene_base = "scene_" + base[len("train_state_"):]
        candidates.append(_paired_decoder_path(os.path.join(ckpt_dir, scene_base)))

    if isinstance(output_dir, str) and output_dir:
        if epoch is not None and int(epoch) >= 0:
            ep = int(epoch)
            candidates.append(_paired_decoder_path(os.path.join(output_dir, f"scene_epoch_{ep}.pt")))
            candidates.append(_paired_decoder_path(os.path.join(output_dir, f"scene_epoch_{ep:05d}.pt")))
        candidates.append(_paired_decoder_path(os.path.join(output_dir, "scene_ckpt_tmp.pt")))
        candidates.append(_paired_decoder_path(os.path.join(output_dir, "scene_ckpt.pt")))

    seen = set()
    for p in candidates:
        norm = os.path.abspath(p)
        if norm in seen:
            continue
        seen.add(norm)
        yield norm


def _save_train_state(
    path: str,
    scene: nn.Module,
    optimizer: torch.optim.Optimizer,
    feature_decoder: nn.Module | None,
    feature_decoder_enabled: bool,
    feature_dim_low: int,
    feature_dim_gt: int,
    epoch: int,
    engine: str,
    tile_order: str,
    output_dir: str,
):
    payload = {
        "format_version": 1,
        "epoch": int(epoch),
        "engine": str(engine),
        "tile_order": str(tile_order),
        "output_dir": str(output_dir),
        "feature_decoder_enabled": bool(feature_decoder_enabled),
        "feature_dim_low": int(feature_dim_low),
        "feature_dim_gt": int(feature_dim_gt),
        "scene_state_dict": scene.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if feature_decoder is not None:
        payload["feature_decoder_state_dict"] = feature_decoder.state_dict()
    torch.save(payload, path)


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: str | torch.device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _load_resume_state(
    resume_ckpt: str,
    scene: nn.Module,
    optimizer: torch.optim.Optimizer,
    feature_decoder: nn.Module | None,
    expect_feature_decoder: bool,
    device: str | torch.device,
) -> tuple[int, bool]:
    raw = torch.load(resume_ckpt, map_location="cpu")
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported checkpoint type at {resume_ckpt}")

    # New-format training-state checkpoint.
    if "scene_state_dict" in raw:
        ckpt_sd = raw["scene_state_dict"]
        model_sd = scene.state_dict()
        filtered_sd = {k: v for k, v in ckpt_sd.items() if k in model_sd and model_sd[k].shape == v.shape}
        model_sd.update(filtered_sd)
        scene.load_state_dict(model_sd, strict=False)
        print(f"[Resume] Loaded scene_state_dict (matched {len(filtered_sd)} tensors)")
        resumed_epoch = int(raw.get("epoch", 0))
        optimizer_loaded = False
        opt_sd = raw.get("optimizer_state_dict")
        if isinstance(opt_sd, dict):
            try:
                optimizer.load_state_dict(opt_sd)
                _move_optimizer_state_to_device(optimizer, device=device)
                optimizer_loaded = True
            except Exception as exc:
                print(f"[Resume] Failed to restore optimizer state: {exc}")
        ckpt_decoder_enabled = raw.get("feature_decoder_enabled")
        if isinstance(ckpt_decoder_enabled, bool) and ckpt_decoder_enabled != bool(expect_feature_decoder):
            raise ValueError(
                "[Resume] feature decoder mode mismatch: "
                f"checkpoint={ckpt_decoder_enabled}, current_run={bool(expect_feature_decoder)}. "
                "Use matching feature_lowdim settings when resuming."
            )
        decoder_loaded = False
        if feature_decoder is not None:
            dec_sd = raw.get("feature_decoder_state_dict")
            if isinstance(dec_sd, dict):
                _load_decoder_state(feature_decoder, dec_sd, source=f"{resume_ckpt}:feature_decoder_state_dict")
                decoder_loaded = True
            else:
                for cand in _iter_resume_decoder_candidates(
                    resume_ckpt=resume_ckpt,
                    epoch=resumed_epoch,
                    output_dir=raw.get("output_dir"),
                ):
                    if not os.path.isfile(cand):
                        continue
                    cand_sd = torch.load(cand, map_location="cpu")
                    if not isinstance(cand_sd, dict):
                        continue
                    try:
                        _load_decoder_state(feature_decoder, cand_sd, source=cand)
                        decoder_loaded = True
                        break
                    except Exception:
                        continue
            if not decoder_loaded and bool(expect_feature_decoder):
                raise FileNotFoundError(
                    "[Resume] Feature decoder is required but could not be restored from "
                    "train_state or paired decoder checkpoint."
                )
            if decoder_loaded:
                print("[Resume] Restored feature decoder state")
        return resumed_epoch, optimizer_loaded

    # Legacy model-only checkpoint (state_dict only).
    model_dict = scene.state_dict()
    ckpt_dict = {k: v for k, v in raw.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(ckpt_dict)
    scene.load_state_dict(model_dict, strict=False)
    print(f"[Resume] Loaded model weights from legacy checkpoint (matched {len(ckpt_dict)} tensors)")
    if feature_decoder is not None:
        decoder_path = _paired_decoder_path(resume_ckpt)
        if not os.path.isfile(decoder_path):
            if bool(expect_feature_decoder):
                raise FileNotFoundError(f"[Resume] Missing paired decoder checkpoint: {decoder_path}")
        else:
            dec_raw = torch.load(decoder_path, map_location="cpu")
            if not isinstance(dec_raw, dict):
                raise TypeError(f"[Resume] Decoder checkpoint must be a state_dict dictionary: {decoder_path}")
            _load_decoder_state(feature_decoder, dec_raw, source=decoder_path)
            print(f"[Resume] Restored feature decoder state from {decoder_path}")
    return _extract_epoch_from_path(resume_ckpt), False


def _paired_decoder_path(scene_ckpt_path: str) -> str:
    base = os.path.basename(scene_ckpt_path)
    if base.startswith("scene_"):
        dec_base = "decoder_" + base[len("scene_"):]
    else:
        dec_base = "decoder_" + base
    return os.path.join(os.path.dirname(scene_ckpt_path), dec_base)


def _save_scene_with_optional_decoder(scene_ckpt_path: str, scene: nn.Module, feature_decoder: nn.Module | None):
    torch.save(scene.state_dict(), scene_ckpt_path)
    if feature_decoder is not None:
        torch.save(feature_decoder.state_dict(), _paired_decoder_path(scene_ckpt_path))


def _build_projection_from_focal(focal: float, W: int, H: int, near: float = 0.001, far: float = 100.0, device: str | torch.device = "cpu"):
    """Match the viewer's perspective projection convention used by tile_cuda culling."""
    focal = float(focal)
    # vertical fov from focal length (pixels)
    fov_y = 2.0 * float(torch.atan(torch.tensor((0.5 * H) / (focal + 1e-8))))
    aspect = float(W) / float(H)
    t = float(torch.tan(torch.tensor(fov_y / 2.0)))
    r = t * aspect
    P = torch.zeros(4, 4, dtype=torch.float32, device=device)
    P[0, 0] = 1.0 / r
    P[1, 1] = 1.0 / t
    P[2, 2] = -(far + near) / (far - near)
    P[3, 2] = 1.0
    P[2, 3] = -(2 * far * near) / (far - near)
    return P


def _c2w_to_view_matrix(c2w_3x4: torch.Tensor, device: str | torch.device):
    """Convert (3,4) camera-to-world to (4,4) world-to-camera (view) matrix."""
    c2w_h = torch.eye(4, dtype=torch.float32, device=device)
    c2w_h[:3, :4] = c2w_3x4.to(device=device, dtype=torch.float32)
    view = torch.inverse(c2w_h)
    return view


def _sample_tile_mosaic_batch(
    dataset,
    view_ids: torch.Tensor,
    tile_size: int,
    view_cache: OrderedDict,
    include_feature: bool = False,
):
    """Compose one full-frame mosaic from random tile views.

    When include_feature=True, returns an additional HWC feature GT mosaic that
    reuses the same tile->view assignment as RGB/rays.
    """
    view_ids = view_ids.to(dtype=torch.long, device="cpu")
    if view_ids.numel() == 0:
        raise ValueError("view_ids must not be empty")

    first_img = dataset.images[0]
    Himg, Wimg = int(first_img.shape[1]), int(first_img.shape[2])
    Ty = (Himg + tile_size - 1) // tile_size
    Tx = (Wimg + tile_size - 1) // tile_size
    Hpad = Ty * tile_size
    Wpad = Tx * tile_size
    num_tiles = Ty * Tx
    K = int(view_ids.numel())

    # Local tile -> local view assignment in [0, K).
    tile_view_local = torch.randint(0, K, (num_tiles,), dtype=torch.long)
    tile_img_ids = view_ids.index_select(0, tile_view_local)

    def _parse_view_sample(sample):
        if not isinstance(sample, (tuple, list)):
            raise TypeError(f"Unexpected dataset sample type: {type(sample)}")
        if len(sample) == 4:
            rays_o_hw, rays_d_hw, img_hw3, _pose = sample
            feat_hwc = None
        elif len(sample) == 5:
            rays_o_hw, rays_d_hw, img_hw3, feat_hwc, _pose = sample
        else:
            raise ValueError(f"Unexpected dataset sample tuple length: {len(sample)}")
        return rays_o_hw, rays_d_hw, img_hw3, feat_hwc

    def _load_feature_for_view(idx: int):
        if getattr(dataset, "features", None) is not None:
            feat = dataset.features[int(idx)]
            return feat.contiguous().cpu()
        if getattr(dataset, "_feature_paths", None) is not None:
            feat = dataset._get_feature_lazy(int(idx))
            return feat.contiguous().cpu()
        return None

    def _get_view(idx: int):
        key = int(idx)
        cached = view_cache.get(key)
        feat_hwc = None
        if cached is not None:
            view_cache.move_to_end(key)
            rays_o_hw, rays_d_hw, img_hw3 = cached
            if include_feature:
                feat_hwc = _load_feature_for_view(key)
            return rays_o_hw, rays_d_hw, img_hw3, feat_hwc
        rays_o_hw, rays_d_hw, img_hw3, feat_hwc = _parse_view_sample(dataset[key])
        cached = (
            rays_o_hw.contiguous().cpu(),
            rays_d_hw.contiguous().cpu(),
            img_hw3.contiguous().cpu(),
        )
        view_cache[key] = cached
        # Keep cache bounded; avoid unbounded CPU memory growth.
        if len(view_cache) > 32:
            view_cache.popitem(last=False)
        if include_feature:
            if feat_hwc is None:
                feat_hwc = _load_feature_for_view(key)
            elif torch.is_tensor(feat_hwc):
                feat_hwc = feat_hwc.contiguous().cpu()
        return cached[0], cached[1], cached[2], feat_hwc

    rays_o_views = []
    rays_d_views = []
    rgb_views = []
    feat_views = [] if include_feature else None
    feat_shape = None
    for i in range(K):
        rid = int(view_ids[i].item())
        ro, rd, rgb, feat = _get_view(rid)
        rays_o_views.append(ro)
        rays_d_views.append(rd)
        rgb_views.append(rgb)
        if include_feature:
            if feat is None:
                raise RuntimeError(
                    "include_feature=True but feature tensor is unavailable for "
                    f"view index {rid}."
                )
            if feat_shape is None:
                feat_shape = tuple(feat.shape)
            elif tuple(feat.shape) != feat_shape:
                raise ValueError(
                    "All mosaic candidate views must share the same feature shape; "
                    f"got {tuple(feat.shape)} vs {feat_shape}."
                )
            feat_views.append(feat)

    def _to_tiles(x_hw3: torch.Tensor):
        if x_hw3.shape[0] != Hpad or x_hw3.shape[1] != Wpad:
            pad_h = Hpad - x_hw3.shape[0]
            pad_w = Wpad - x_hw3.shape[1]
            x_hw3 = F.pad(x_hw3.permute(2, 0, 1), (0, pad_w, 0, pad_h)).permute(1, 2, 0).contiguous()
        return x_hw3.view(Ty, tile_size, Tx, tile_size, 3).permute(0, 2, 1, 3, 4).contiguous()

    rays_o_tiles = torch.stack([_to_tiles(x) for x in rays_o_views], dim=0)  # (K,Ty,Tx,ts,ts,3)
    rays_d_tiles = torch.stack([_to_tiles(x) for x in rays_d_views], dim=0)
    rgb_tiles = torch.stack([_to_tiles(x) for x in rgb_views], dim=0)

    tile_y = torch.arange(Ty, dtype=torch.long).unsqueeze(1).expand(Ty, Tx).reshape(-1)
    tile_x = torch.arange(Tx, dtype=torch.long).unsqueeze(0).expand(Ty, Tx).reshape(-1)
    k_sel = tile_view_local

    sel_ro = rays_o_tiles[k_sel, tile_y, tile_x]
    sel_rd = rays_d_tiles[k_sel, tile_y, tile_x]
    sel_rgb = rgb_tiles[k_sel, tile_y, tile_x]

    def _from_tiles(x_tiles: torch.Tensor):
        x = x_tiles.view(Ty, Tx, tile_size, tile_size, 3).permute(0, 2, 1, 3, 4).reshape(Hpad, Wpad, 3)
        return x[:Himg, :Wimg, :].contiguous()

    rays_o_hw = _from_tiles(sel_ro)
    rays_d_hw = _from_tiles(sel_rd)
    rgbs_hw = _from_tiles(sel_rgb)

    feat_hwc = None
    if include_feature:
        tile_view_grid = tile_view_local.view(Ty, Tx)
        tile_view_hw = tile_view_grid.repeat_interleave(tile_size, dim=0).repeat_interleave(tile_size, dim=1)
        tile_view_hw = tile_view_hw[:Himg, :Wimg].contiguous()

        feat_stack = torch.stack(feat_views, dim=0)  # (K, Hf, Wf, Cf)
        Hf, Wf = int(feat_stack.shape[1]), int(feat_stack.shape[2])
        if Hf == Himg and Wf == Wimg:
            tile_view_feat = tile_view_hw
        else:
            tile_view_feat = F.interpolate(
                tile_view_hw.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                size=(Hf, Wf),
                mode="nearest",
            ).squeeze(0).squeeze(0).to(dtype=torch.long)

        yy = torch.arange(Hf, dtype=torch.long).unsqueeze(1).expand(Hf, Wf)
        xx = torch.arange(Wf, dtype=torch.long).unsqueeze(0).expand(Hf, Wf)
        feat_hwc = feat_stack[tile_view_feat, yy, xx].contiguous()

    return rays_o_hw, rays_d_hw, rgbs_hw, feat_hwc, tile_view_local, tile_img_ids



def train_scene_llff(
    data_dir,
    output_root,
    num_epochs=10,
    batch_size=4096,
    lr=1e-3,
    device='cuda',
    initial_num_primitives=1024,
    engine: str = "vmodel",
    tile_order: str = "per_pixel",
    resume_ckpt: str | None = None,
    run_tag: str | None = None,
    enable_pe_ramp: bool = False,
    pe_ramp_start: int = 0,
    pe_ramp_steps: int = 50000,
    enable_center_lr_warmup: bool = False,
    center_lr_warmup_steps: int = 5000,
    center_lr_start_factor: float = 0.1,
    tile_mosaic: bool = False,
    tile_mosaic_k_views: int = 8,
    tile_mosaic_warmup_steps: int = 0,
    feature_type: str = "none",
    feature_dir: str | None = None,
    feature_loss_weight: float = 1.0,
    feature_decoder_hidden_dim: int = 128,
    feature_lowdim_divisor: int = 4,
    feature_mock_from_rgb: bool = False,
    disable_densify_prune: bool = True,
    preload_features: bool = False,
    feature_cache_size: int = 16,
    ray_cache_size: int = 8,
    dataloader_num_workers: int = 0,
    dataloader_pin_memory: bool = False,
    dataloader_persistent_workers: bool = False,
    dataloader_prefetch_factor: int = 2,
):
    def _nvtx_range(msg: str):
        class _NVTX:
            def __enter__(self_inner):
                if torch.cuda.is_available():
                    torch.cuda.nvtx.range_push(msg)
            def __exit__(self_inner, exc_type, exc, tb):
                if torch.cuda.is_available():
                    torch.cuda.nvtx.range_pop()
        return _NVTX()
    # Resolve resume metadata early so model shape can match checkpoint tensors.
    resume_probe = None
    resume_state_dict = None
    if resume_ckpt:
        resume_ckpt = os.path.abspath(resume_ckpt)
        if not os.path.exists(resume_ckpt):
            raise FileNotFoundError(f"--resume_ckpt not found: {resume_ckpt}")
        resume_probe = torch.load(resume_ckpt, map_location="cpu")
        if not isinstance(resume_probe, dict):
            raise TypeError(f"Unsupported checkpoint type at {resume_ckpt}")
        if "scene_state_dict" in resume_probe:
            resume_state_dict = resume_probe["scene_state_dict"]
        else:
            resume_state_dict = resume_probe
        inferred_n = _infer_num_primitives(resume_state_dict)
        if inferred_n is not None and inferred_n != int(initial_num_primitives):
            print(f"[Resume] Override num_primitives from {initial_num_primitives} -> {inferred_n}")
            initial_num_primitives = int(inferred_n)

    # Set saving directory
    os.makedirs(output_root, exist_ok=True)
    if resume_ckpt:
        resume_output_dir = None
        if isinstance(resume_probe, dict):
            probe_output_dir = resume_probe.get("output_dir")
            if isinstance(probe_output_dir, str) and probe_output_dir:
                resume_output_dir = probe_output_dir
        output_dir = resume_output_dir or os.path.dirname(resume_ckpt)
        random_tag = os.path.basename(os.path.normpath(output_dir)) or "resumed"
    else:
        random_tag = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        # random_tag = "delta_b"  # for debugging
        output_dir = os.path.join(output_root, random_tag)
    if run_tag is not None and str(run_tag).strip():
        random_tag = str(run_tag).strip()
        if not resume_ckpt:
            output_dir = os.path.join(output_root, random_tag)
    os.makedirs(output_dir, exist_ok=True)

    feature_type = str(feature_type).lower()
    use_feature_metric = feature_type in {"lseg", "sam"}
    feature_dim_gt = 0
    if feature_type == "lseg":
        feature_dim_gt = 512
    elif feature_type == "sam":
        feature_dim_gt = 256
    elif feature_type != "none":
        raise ValueError(f"Unsupported feature_type: {feature_type}")
    feature_dim_low = feature_dim_gt // int(max(1, feature_lowdim_divisor)) if use_feature_metric else 0
    resume_feature_dim_low = _infer_feature_dim_low(resume_state_dict) if isinstance(resume_state_dict, dict) else None
    if use_feature_metric and (resume_feature_dim_low is not None) and (int(resume_feature_dim_low) != int(feature_dim_low)):
        print(f"[Resume] Override feature_dim_low from {feature_dim_low} -> {int(resume_feature_dim_low)}")
        feature_dim_low = int(resume_feature_dim_low)
    if use_feature_metric and int(feature_dim_low) <= 0:
        raise ValueError("featuremetric training requires feature_dim_low > 0.")
    use_feature_decoder = bool(use_feature_metric and int(feature_dim_low) != int(feature_dim_gt))
    if use_feature_metric:
        if use_feature_decoder:
            print(f"[Feature] Decoder enabled (low_dim={feature_dim_low}, gt_dim={feature_dim_gt})")
        else:
            print(f"[Feature] Decoder disabled (low_dim == gt_dim == {feature_dim_gt})")

    use_mock_feature = False
    try:
        dataset = mip360Dataset(
            data_dir,
            feature_type=feature_type,
            feature_dir=feature_dir,
            preload_features=bool(preload_features),
            feature_cache_size=int(feature_cache_size),
            feature_resize_to_image=False,
            ray_cache_size=int(ray_cache_size),
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        if use_feature_metric and feature_mock_from_rgb:
            print(f"[Feature] {exc}")
            print("[Feature] Falling back to RGB-projected mock features for pipeline verification.")
            dataset = mip360Dataset(
                data_dir,
                feature_type="none",
                feature_dir=None,
                preload_features=False,
                feature_cache_size=0,
                feature_resize_to_image=False,
                ray_cache_size=int(ray_cache_size),
            )
            use_mock_feature = True
        else:
            raise
    tile_size = 16
    tile_mosaic_k_views = int(max(1, tile_mosaic_k_views))
    tile_mosaic_warmup_steps = int(max(0, tile_mosaic_warmup_steps))
    use_tile_mosaic = bool(engine == "tile_cuda" and tile_mosaic)
    if use_tile_mosaic and tile_order != "tilecenter":
        raise ValueError("tile_mosaic currently supports tile_order='tilecenter' only.")
    if use_tile_mosaic and use_feature_metric:
        print("[Feature] tile_mosaic is enabled; semantic GT will be mosaicked per tile-view assignment.")

    """
    test on how to oganize parameters is suitable
    """
    if engine == "tile_cuda":
        # NOTE: tile_cuda training requires a CUDA backward path; we only wire the option here.
        # This renderer consumes full-frame rays (B == W*H) and view/proj matrices.
        from cumodel import NeuralPrimitiveSceneTileCUDA
        init_positions = None if resume_ckpt else dataset.positions
        scene = NeuralPrimitiveSceneTileCUDA(
            num_primitives=initial_num_primitives,
            hidden_dim=32,
            pe_frequencies=6,
            initial_positions=init_positions,
            feature_dim_low=feature_dim_low,
        ).to(device)
        # ensure tile order mode for this run
        os.environ["MLPSP_TILE_ORDER"] = str(tile_order)
        if use_tile_mosaic:
            # Mosaic training only supports tilecenter ordering for now.
            os.environ["MLPSP_TILE_ORDER"] = "tilecenter"
    else:
        init_positions = None if resume_ckpt else dataset.positions
        scene = NeuralPrimitiveScene(
            num_primitives=initial_num_primitives,
            hidden_dim=32,
            pe_frequencies=6,
            initial_positions=init_positions,
        ).to(device)
        if use_feature_metric:
            raise ValueError("featuremetric training is currently supported with --engine tile_cuda only.")
    #optimizer = torch.optim.Adam(scene.parameters(), lr=lr)


    """
    # 加载已有 checkpoint 权重
    model_ckpt_path = "/home/yuzhou/Desktop/output/mip360/2183/scene_epoch_176000.pt"
    if os.path.exists(model_ckpt_path):
        ckpt = torch.load(model_ckpt_path, map_location=device)
        scene.load_state_dict(ckpt)
        print(f"[Checkpoint] Loaded model from {model_ckpt_path}")
    """

    feature_decoder = None
    if use_feature_decoder:
        feature_decoder = FeatureUpsampleDecoder(
            in_dim=feature_dim_low,
            out_dim=feature_dim_gt,
            hidden_dim=int(feature_decoder_hidden_dim),
        ).to(device)

    if engine == "tile_cuda":
        mlp_params = [scene.W1, scene.b1, scene.W2, scene.b2, scene.W3, scene.b3]
    else:
        mlp_params = list(scene.mlps.parameters())

    param_groups = [
        {"params": mlp_params},  # 默认 1e-3
        {"params": scene.log_scales, "lr": 1e-2},
        {"params": scene.positions, "lr": 1e-4},
        #{"params": scene.rotations, "lr": 1e-2},
    ]
    if use_feature_metric:
        if getattr(scene, "feat_attr", None) is None:
            raise RuntimeError("Feature training requested but scene has no feat_attr.")
        param_groups.append({"params": [scene.feat_attr], "lr": float(lr)})
        if feature_decoder is not None:
            param_groups.append({"params": feature_decoder.parameters(), "lr": float(lr)})
    optimizer = torch.optim.Adam(param_groups, lr=float(lr))
    base_center_lr = float(optimizer.param_groups[2]["lr"])
    center_lr_start_factor = float(max(1e-6, min(1.0, center_lr_start_factor)))
    pe_ramp_start = int(max(0, pe_ramp_start))
    pe_ramp_steps = int(max(1, pe_ramp_steps))
    center_lr_warmup_steps = int(max(1, center_lr_warmup_steps))

    if enable_pe_ramp:
        print(
            f"[PE Ramp] Enabled: start={pe_ramp_start}, steps={pe_ramp_steps}, "
            f"num_freq={getattr(scene.encoder, 'num_frequencies', 'unknown')}"
        )
    if enable_center_lr_warmup:
        print(
            f"[Center LR Warmup] Enabled: base_lr={base_center_lr:.6g}, "
            f"start_factor={center_lr_start_factor:.4f}, steps={center_lr_warmup_steps}"
        )


    #scheduler = StepLR(optimizer, step_size=1000, gamma=0.8)

    start_epoch = 0
    optimizer_resumed = False
    if resume_ckpt:
        start_epoch, optimizer_resumed = _load_resume_state(
            resume_ckpt=resume_ckpt,
            scene=scene,
            optimizer=optimizer,
            feature_decoder=feature_decoder,
            expect_feature_decoder=use_feature_decoder,
            device=device,
        )
        print(
            f"[Resume] Loaded from {resume_ckpt} | start_epoch={start_epoch} "
            f"| optimizer_restored={optimizer_resumed}"
        )
    else:
        ckpt_path = os.path.join(output_dir, "scene_epoch_0.pt")
        _save_scene_with_optional_decoder(ckpt_path, scene, feature_decoder)
        print(f"[Checkpoint] Saved model to {ckpt_path}")
        _save_train_state(
            path=os.path.join(output_dir, "train_state_latest.pt"),
            scene=scene,
            optimizer=optimizer,
            feature_decoder=feature_decoder,
            feature_decoder_enabled=use_feature_decoder,
            feature_dim_low=feature_dim_low,
            feature_dim_gt=feature_dim_gt,
            epoch=0,
            engine=engine,
            tile_order=tile_order,
            output_dir=output_dir,
        )


    if USE_WANDB:
        wandb.init(
            project="mlp-splatting",
            name=f"mip360_run_{random_tag}",
            config={
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "lr": lr,
                "num_primitives": initial_num_primitives,
                "hidden_dim": 32,
                "pe_frequencies": 6,
                "num_layers": 2,   # 当前是单隐藏层，如果以后改成两层可以记录成2
                "enable_pe_ramp": bool(enable_pe_ramp),
                "pe_ramp_start": int(pe_ramp_start),
                "pe_ramp_steps": int(pe_ramp_steps),
                "enable_center_lr_warmup": bool(enable_center_lr_warmup),
                "center_lr_warmup_steps": int(center_lr_warmup_steps),
                "center_lr_start_factor": float(center_lr_start_factor),
                "tile_mosaic": bool(use_tile_mosaic),
                "tile_mosaic_k_views": int(tile_mosaic_k_views),
                "tile_mosaic_warmup_steps": int(tile_mosaic_warmup_steps),
                "tile_mosaic_tile_size": int(tile_size),
                "feature_type": feature_type,
                "feature_metric_enabled": bool(use_feature_metric),
                "feature_dim_gt": int(feature_dim_gt),
                "feature_dim_low": int(feature_dim_low),
                "feature_decoder_enabled": bool(use_feature_decoder),
                "feature_decoder_hidden_dim": int(feature_decoder_hidden_dim),
                "feature_loss_weight": float(feature_loss_weight),
                "disable_densify_prune": bool(disable_densify_prune),
                "preload_features": bool(preload_features),
                "feature_cache_size": int(feature_cache_size),
                "ray_cache_size": int(ray_cache_size),
                "dataloader_num_workers": int(dataloader_num_workers),
                "dataloader_pin_memory": bool(dataloader_pin_memory),
                "dataloader_persistent_workers": bool(dataloader_persistent_workers),
                "dataloader_prefetch_factor": int(dataloader_prefetch_factor),
            }
        )

    loss_accum = torch.zeros((), dtype=torch.float32, device=device)
    loss_count = 0

    """
    # Load checkpoint for continued training
    ckpt = torch.load("scene_ckpt_tmp.pt")
    model_dict = scene.state_dict()
    ckpt_dict = {k: v for k, v in ckpt.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(ckpt_dict)
    scene.load_state_dict(model_dict)
    print(f"[Checkpoint] Partially loaded model from scene_ckpt_tmp.pt (matched {len(ckpt_dict)} tensors)")
    """

    rays_o_all = rays_d_all = rgbs_all = None
    N = 0
    if engine != "tile_cuda":
        rays_o_all, rays_d_all, rgbs_all = dataset.get_all_rays(device=None)
        N = rays_o_all.shape[0]
        assert batch_size <= N, "Batch size must be less than or equal to the number of rays."
        perm = torch.randperm(N)
        ptr = 0
    else:
        tile_loader_kwargs = {
            "batch_size": 1,
            "shuffle": True,
            "num_workers": int(max(0, dataloader_num_workers)),
            "pin_memory": bool(dataloader_pin_memory),
            "drop_last": False,
        }
        if tile_loader_kwargs["num_workers"] > 0:
            tile_loader_kwargs["persistent_workers"] = bool(dataloader_persistent_workers)
            tile_loader_kwargs["prefetch_factor"] = int(max(1, dataloader_prefetch_factor))
        tile_dataloader = DataLoader(dataset, **tile_loader_kwargs)
        tile_data_iter = iter(tile_dataloader)
        print(
            "[DataLoader] tile_cuda enabled "
            f"(workers={tile_loader_kwargs['num_workers']}, "
            f"pin_memory={tile_loader_kwargs['pin_memory']}, "
            f"persistent_workers={tile_loader_kwargs.get('persistent_workers', False)}, "
            f"prefetch_factor={tile_loader_kwargs.get('prefetch_factor', 0)})"
        )


    # train loop
    printed_batch = False
    profile_enabled = os.environ.get("MLPSP_PROFILE", "0") == "1"
    profile_start = int(os.environ.get("MLPSP_PROFILE_START", "10"))
    profile_steps = int(os.environ.get("MLPSP_PROFILE_STEPS", "1"))
    profiler_active = False
    timing_enabled = os.environ.get("MLPSP_TIMING", "0") == "1"
    timing_every = int(os.environ.get("MLPSP_TIMING_EVERY", "50"))
    did_densify = False
    if not bool(disable_densify_prune):
        raise ValueError("This training path enforces densify/prune disabled for controlled experiments.")
    mock_feature_proj = None
    if use_feature_metric and use_mock_feature:
        mock_feature_proj = _build_mock_feature_projection(feature_dim_gt, device=device)
    fixed_ids = torch.empty(0, dtype=torch.long, device=scene.positions.device)
    tile_view_cache = OrderedDict()


    for epoch in tqdm(range(start_epoch, num_epochs), initial=start_epoch, total=num_epochs):
        wall_start = time.perf_counter() if timing_enabled and (epoch % timing_every == 0) else None
        data_start = wall_start

        bg_color = torch.rand(3, device=device)
        feat_gt_hwc = None

        # (1) BARF-style PE frequency ramp.
        if enable_pe_ramp and hasattr(scene, "encoder") and hasattr(scene.encoder, "num_frequencies"):
            if epoch < pe_ramp_start:
                alpha = 0.0
            else:
                progress = min(1.0, float(epoch - pe_ramp_start) / float(pe_ramp_steps))
                alpha = progress * float(scene.encoder.num_frequencies)
            freq_weights = _barf_frequency_weights(alpha, int(scene.encoder.num_frequencies), device=scene.positions.device)
            _set_scene_frequency_weights(scene, freq_weights)

        # (2) Center(position) LR warmup.
        if enable_center_lr_warmup:
            warmup_progress = min(1.0, float(epoch) / float(center_lr_warmup_steps))
            lr_scale = center_lr_start_factor + (1.0 - center_lr_start_factor) * warmup_progress
            optimizer.param_groups[2]["lr"] = base_center_lr * lr_scale

        if engine == "tile_cuda":
            use_mosaic_step = use_tile_mosaic and (epoch >= tile_mosaic_warmup_steps)
            effective_k = min(tile_mosaic_k_views, len(dataset))
            if use_mosaic_step and effective_k > 1:
                # Build full-frame rays/GT by stitching tiles from up to k random views.
                perm = torch.randperm(len(dataset))
                view_ids = perm[:effective_k].to(torch.long)
                rays_o_hw, rays_d_hw, img_hw3, feat_hwc_mosaic, tile_view_local, _tile_img_ids = _sample_tile_mosaic_batch(
                    dataset=dataset,
                    view_ids=view_ids,
                    tile_size=tile_size,
                    view_cache=tile_view_cache,
                    include_feature=bool(use_feature_metric and (not use_mock_feature)),
                )
                Himg, Wimg = img_hw3.shape[0], img_hw3.shape[1]
                scene.set_image_size(Wimg, Himg)

                rays_o_batch = rays_o_hw.reshape(-1, 3).to(device, non_blocking=True)
                rays_d_batch = rays_d_hw.reshape(-1, 3).to(device, non_blocking=True)
                rgbs_batch = img_hw3.reshape(-1, 3).to(device, non_blocking=True)
                if feat_hwc_mosaic is not None:
                    feat_gt_hwc = feat_hwc_mosaic.to(device, non_blocking=True)
                elif use_feature_metric and use_mock_feature:
                    feat_gt_hwc = torch.matmul(rgbs_batch, mock_feature_proj).view(Himg, Wimg, feature_dim_gt)
                else:
                    feat_gt_hwc = None

                poses_sel = dataset.poses.index_select(0, view_ids)
                view_mats = torch.stack(
                    [_c2w_to_view_matrix(poses_sel[i], device=device) for i in range(poses_sel.shape[0])],
                    dim=0,
                )
                proj_matrix = _build_projection_from_focal(dataset.focal, Wimg, Himg, device=device)
                tile_view_local_dev = tile_view_local.to(device=device, non_blocking=True)
                culled_prim_indices, tile_offsets = scene.build_tilecenter_culling_from_tile_views(
                    tile_view_ids=tile_view_local_dev,
                    view_matrices=view_mats,
                    proj_matrix=proj_matrix,
                    score_threshold=1e-3,
                )
                view_matrix = None
            else:
                # Fast path: one training image per step from DataLoader.
                try:
                    sample = next(tile_data_iter)
                except StopIteration:
                    tile_data_iter = iter(tile_dataloader)
                    sample = next(tile_data_iter)
                if use_feature_metric and (not use_mock_feature):
                    rays_o_hw, rays_d_hw, img_hw3, feat_hwc, pose = sample
                else:
                    rays_o_hw, rays_d_hw, img_hw3, pose = sample
                    feat_hwc = None
                rays_o_hw = rays_o_hw.squeeze(0)
                rays_d_hw = rays_d_hw.squeeze(0)
                img_hw3 = img_hw3.squeeze(0)
                pose = pose.squeeze(0)
                if feat_hwc is not None:
                    feat_hwc = feat_hwc.squeeze(0)
                Himg, Wimg = img_hw3.shape[0], img_hw3.shape[1]
                scene.set_image_size(Wimg, Himg)

                rays_o_batch = rays_o_hw.reshape(-1, 3).to(device, non_blocking=True)
                rays_d_batch = rays_d_hw.reshape(-1, 3).to(device, non_blocking=True)
                rgbs_batch = img_hw3.reshape(-1, 3).to(device, non_blocking=True)
                if feat_hwc is not None:
                    feat_gt_hwc = feat_hwc.to(device, non_blocking=True)
                elif use_feature_metric and use_mock_feature:
                    flat_rgb = img_hw3.reshape(-1, 3).to(device, non_blocking=True)
                    feat_gt_hwc = torch.matmul(flat_rgb, mock_feature_proj).view(Himg, Wimg, feature_dim_gt)
                else:
                    feat_gt_hwc = None

                view_matrix = _c2w_to_view_matrix(pose, device=device)
                proj_matrix = _build_projection_from_focal(dataset.focal, Wimg, Himg, device=device)
                culled_prim_indices = None
                tile_offsets = None
            if not printed_batch:
                if use_mosaic_step and effective_k > 1:
                    print(
                        f"[Batch] engine=tile_cuda batch_size={Wimg * Himg} "
                        f"(tile_mosaic, full frame {Wimg}x{Himg}, k_views={effective_k}, tile={tile_size})"
                    )
                else:
                    print(f"[Batch] engine=tile_cuda batch_size={Wimg * Himg} (full frame {Wimg}x{Himg})")
                printed_batch = True
        else:
            if ptr + batch_size > N:
                perm = torch.randperm(N)
                ptr = 0

            idx = perm[ptr:ptr + batch_size]
            ptr += batch_size

            rays_o_batch = rays_o_all[idx].to(device, non_blocking=True)
            rays_d_batch = rays_d_all[idx].to(device, non_blocking=True)
            rgbs_batch   = rgbs_all[idx].to(device, non_blocking=True)
            if not printed_batch:
                print(f"[Batch] engine={engine} batch_size={int(batch_size)} (random rays)")
                printed_batch = True

        data_end = time.perf_counter() if wall_start is not None else None
        if wall_start is not None and torch.cuda.is_available():
            fwd_start = torch.cuda.Event(enable_timing=True)
            fwd_end = torch.cuda.Event(enable_timing=True)
            bwd_start = torch.cuda.Event(enable_timing=True)
            bwd_end = torch.cuda.Event(enable_timing=True)
            opt_start = torch.cuda.Event(enable_timing=True)
            opt_end = torch.cuda.Event(enable_timing=True)
            fwd_start.record()


        """
        if epoch % 500 < 500:
            # use random sampled rays
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
            rgbs = rgbs.reshape(-1, 3)
            rand_idx = torch.randperm(rays_o.size(0))[:batch_size]
            rays_o_batch = rays_o[rand_idx]
            rays_d_batch = rays_d[rand_idx]
            rgbs_batch = rgbs[rand_idx]
        else:
            rays_o_batch, rays_d_batch, rgbs_batch = sample_patch_rays(
                        rays_o, rays_d, rgbs,
                        num_patches_h=4, num_patches_w=4  # horizontal 4, vertical 4 patches
                    )  
        """

        if profile_enabled and (epoch == profile_start):
            torch.cuda.profiler.start()
            profiler_active = True

        with _nvtx_range(f"train/step_{epoch}"):
            optimizer.zero_grad()                       # 1. Clear gradients
            if engine == "tile_cuda":
                with _nvtx_range("train/forward_tile_cuda"):
                    scene_out = scene(
                        rays_o_batch, rays_d_batch,
                        score_threshold=1e-3,
                        bg_color=bg_color,
                        view_matrix=view_matrix,
                        proj_matrix=proj_matrix,
                        culled_prim_indices=culled_prim_indices,
                        tile_offsets=tile_offsets,
                        return_feature=bool(use_feature_metric),
                    )
                    if use_feature_metric:
                        pred, feat_low = scene_out
                    else:
                        pred = scene_out
                        feat_low = None
            else:
                with _nvtx_range("train/forward_vmodel"):
                    pred = scene(rays_o_batch, rays_d_batch, score_threshold=1e-3, bg_color=bg_color)    # 2. Forward pass
                    feat_low = None
            loss_rgb = F.mse_loss(pred, rgbs_batch)         # 3. Compute loss (MSE)
            loss = loss_rgb
            loss_feat = None
            if use_feature_metric:
                if feat_gt_hwc is None:
                    raise RuntimeError("Feature GT is missing for featuremetric training.")
                if feat_low is None:
                    raise RuntimeError("scene(..., return_feature=True) did not provide feat_low.")
                feat_h, feat_w = int(feat_gt_hwc.shape[0]), int(feat_gt_hwc.shape[1])
                feat_low_img = feat_low.view(Himg, Wimg, feature_dim_low).permute(2, 0, 1).unsqueeze(0)
                if feat_h != Himg or feat_w != Wimg:
                    feat_low_img = F.interpolate(
                        feat_low_img, size=(feat_h, feat_w), mode="bilinear", align_corners=False
                    )
                if feature_decoder is not None:
                    feat_pred_h = feature_decoder(feat_low_img)
                else:
                    if int(feat_low_img.shape[1]) != int(feature_dim_gt):
                        raise RuntimeError(
                            f"Decoder is disabled but feature channels mismatch: "
                            f"pred={int(feat_low_img.shape[1])}, gt={int(feature_dim_gt)}"
                        )
                    feat_pred_h = feat_low_img
                feat_pred_hwc = feat_pred_h.squeeze(0).permute(1, 2, 0).contiguous()
                loss_feat = F.l1_loss(feat_pred_hwc, feat_gt_hwc)
                loss = loss_rgb + float(feature_loss_weight) * loss_feat
            #loss = F.l1_loss(pred, rgbs_batch)

            if wall_start is not None and torch.cuda.is_available():
                fwd_end.record()
            with _nvtx_range("train/backward"):
                if wall_start is not None and torch.cuda.is_available():
                    bwd_start.record()
                loss.backward()                             # 4. Backpropagation
                if wall_start is not None and torch.cuda.is_available():
                    bwd_end.record()

        # import pdb; pdb.set_trace()

        #print("Position grad norm:", scene.positions.grad.norm().item())
        #with torch.no_grad():
            #print(f"[Epoch {epoch+1}] Position mean: {scene.positions.mean().item():.6f}, std: {scene.positions.std().item():.6f}")
        if wall_start is not None and torch.cuda.is_available():
            opt_start.record()


        zero_grads_for_ids(scene, fixed_ids)
        optimizer.step()

        if wall_start is not None and torch.cuda.is_available():
            opt_end.record()

        if profile_enabled and profiler_active and (epoch == profile_start + profile_steps):
            torch.cuda.profiler.stop()
            profiler_active = False

        if wall_start is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                fwd_ms = fwd_start.elapsed_time(fwd_end)
                bwd_ms = bwd_start.elapsed_time(bwd_end)
                opt_ms = opt_start.elapsed_time(opt_end)
            else:
                fwd_ms = bwd_ms = opt_ms = 0.0
            data_ms = (data_end - data_start) * 1000.0 if data_end is not None else 0.0
            wall_ms = (time.perf_counter() - wall_start) * 1000.0
            print(
                f"[Timing] step={epoch} data_ms={data_ms:.2f} fwd_ms={fwd_ms:.2f} "
                f"bwd_ms={bwd_ms:.2f} opt_ms={opt_ms:.2f} wall_ms={wall_ms:.2f}",
                flush=True,
            )

        # Accumulate loss
        loss_accum += loss.detach()
        loss_count += 1

        # Print average loss every 100 epochs
        if (epoch + 1) % 100 == 0:
            avg_loss = loss_accum / loss_count
            avg_loss_scalar = float(avg_loss.item())
            print(f"[Epoch {epoch+1}] avg {loss_count} loss : {avg_loss_scalar:.6f}")
            # Reset accumulators
            loss_accum.zero_()
            loss_count = 0
            if USE_WANDB:   # log to Weights & Biases
                metrics = {"avg_loss": avg_loss_scalar, "epoch": epoch + 1}
                if use_feature_metric:
                    metrics["loss_rgb"] = float(loss_rgb.item())
                    if loss_feat is not None:
                        metrics["loss_feat"] = float(loss_feat.item())
                wandb.log(metrics)
                # Reset again for safety (redundant but harmless)
                loss_accum.zero_()
                loss_count = 0

        if False and (engine == "tile_cuda") and (epoch % 5000 == 0) and (epoch > 0):
            """
            tile_results = sample_worst_tiles_l1(
                scene, dataset,
                k_images=100,
                m_tiles=20,
                tile_size=32,
                score_threshold=1e-3,
                bg_color=torch.zeros(3, device="cuda"),
                seed=0,
            )

            sampled_ids = [r["img_idx"] for r in tile_results]
            print(sampled_ids)

            torch.save(tile_results, os.path.join(output_dir, "tile_results.pt"))

            #print(f"[Tile Results] {tile_results}")
            pointcloud_xyz = tiles_to_points_from_tmap(
                scene, dataset,
                tile_results,
                tile_size=32,
                score_threshold=1e-3,
                bg_color=torch.zeros(3, device="cuda"),
                alpha_min=1e-10,
            )

            # 你已有的 densify 接口直接吃 (M,3)
            stats = densify_tilecuda_from_pointcloud_random(scene, pointcloud_xyz, DENSIFY_K)

            
            
            stats = densify_tilecuda_stage1_from_worst_tiles(
            scene, dataset,
            k_images=100, m_tiles=20, tile_size=32,
        )

            _save_scene_with_optional_decoder(
                os.path.join(output_dir, "after_densification.pt"), scene, feature_decoder
            )
            did_densify = True
            print(f"[densify] {stats}")

            # 跑通优先：重建 optimizer，让新增 Parameter 被优化
            optimizer = torch.optim.Adam([
                {'params': [scene.W1, scene.b1, scene.W2, scene.b2, scene.W3, scene.b3], 'lr': 1e-3},
                {'params': [scene.log_scales], 'lr': 1e-2},
                {'params': [scene.positions], 'lr': 1e-4},
            ])
            print("[densify] optimizer rebuilt.")
            """

            densify_scores = compute_multiview_densify_scores(
                scene, dataset,
                k_images=20,
                r_threshold=0.10,
                cover_mult=3.716,
                chunk_prims=128,
                seed=0,
            )

            # 例如：选 score > densify_thr 的 primitives 作为 densify target（覆盖多视角高误差区域）
            densify_thr = 500.0
            densify_ids = torch.nonzero(densify_scores > densify_thr, as_tuple=False).squeeze(-1)

            # prune_scores = compute_multiview_prune_scores(
            #     scene, dataset,
            #     k_images=20,
            #     r_threshold=0.10,
            #     cover_mult=3.716,
            #     chunk_prims=128,
            #     seed=0,
            # )
            prune_ids, prune_info = compute_prune_ids_from_scale_sigma(
                scene,
                scale_rel_min=0.03,
                sigma_max=0.15,
                return_details=True,
            )
            # prune_thr = 0.9
            # prune_ids = torch.nonzero(prune_scores > float(prune_thr), as_tuple=False).squeeze(-1)
            print("[prune] scale/sigma criteria:", prune_info)
            densify_prune_tilecuda_and_rebuild_optimizeroptimizer, info, fixed_ids = densify_prune_tilecuda_and_rebuild_optimizer(
                scene, optimizer,
                densify_ids=densify_ids,   # 旧空间 ids
                prune_ids=prune_ids,       # 旧空间 ids（高分删）
                init_std=0.01,
                jitter_frac=0.10,
                scale_div=2.0,
            )
            print(info)

            optimizer = torch.optim.Adam([
                {'params': [scene.W1, scene.b1, scene.W2, scene.b2, scene.W3, scene.b3], 'lr': 1e-3},
                {'params': [scene.log_scales], 'lr': 1e-2},
                {'params': [scene.positions], 'lr': 1e-4},
            ])
            _save_scene_with_optional_decoder(
                os.path.join(output_dir, "after_pruning.pt"), scene, feature_decoder
            )

            did_densify = True
        
        if epoch % 5000 == 500 and epoch > 500:
            fixed_ids = torch.empty(0, dtype=torch.long, device=scene.positions.device)


        # Eval after 500 epochs
        if (epoch + 1) % 500 == 0:
            _save_scene_with_optional_decoder(
                os.path.join(output_dir, "scene_ckpt_tmp.pt"), scene, feature_decoder
            )
            _save_train_state(
                path=os.path.join(output_dir, "train_state_latest.pt"),
                scene=scene,
                optimizer=optimizer,
                feature_decoder=feature_decoder,
                feature_decoder_enabled=use_feature_decoder,
                feature_dim_low=feature_dim_low,
                feature_dim_gt=feature_dim_gt,
                epoch=epoch + 1,
                engine=engine,
                tile_order=tile_order,
                output_dir=output_dir,
            )
            #evaluate_and_visualize(data_dir, model_ckpt=scene.state_dict(), N=scene.num_primitives, device=device)
        
        # Save at every 2000 epochs
        if (epoch + 1) % 2000 == 0:
            ckpt_path = os.path.join(output_dir, f"scene_epoch_{epoch+1:05d}.pt")
            _save_scene_with_optional_decoder(ckpt_path, scene, feature_decoder)
            _save_train_state(
                path=os.path.join(output_dir, f"train_state_epoch_{epoch+1:05d}.pt"),
                scene=scene,
                optimizer=optimizer,
                feature_decoder=feature_decoder,
                feature_decoder_enabled=use_feature_decoder,
                feature_dim_low=feature_dim_low,
                feature_dim_gt=feature_dim_gt,
                epoch=epoch + 1,
                engine=engine,
                tile_order=tile_order,
                output_dir=output_dir,
            )
            print(f"[Checkpoint] Saved model to {ckpt_path}")


    _save_scene_with_optional_decoder(os.path.join(output_dir, "scene_ckpt.pt"), scene, feature_decoder)
    _save_train_state(
        path=os.path.join(output_dir, "train_state_latest.pt"),
        scene=scene,
        optimizer=optimizer,
        feature_decoder=feature_decoder,
        feature_decoder_enabled=use_feature_decoder,
        feature_dim_low=feature_dim_low,
        feature_dim_gt=feature_dim_gt,
        epoch=num_epochs,
        engine=engine,
        tile_order=tile_order,
        output_dir=output_dir,
    )


def prune_primitives(scene, alpha_thresh=0.01, occupancy_thresh=0.01, scale_min=0.01, scale_max=10.0):
    alpha = scene.alpha_accum
    occ = scene.occupancy_accum

    # Convert log_scale to scale (input shape: (N, 1))
    scales = torch.exp(scene.log_scales).squeeze(-1)  # shape: (N,)

    # alpha + occupancy + scale restrictions
    keep_mask = (
        (alpha > alpha_thresh) &
        (occ > occupancy_thresh) &
        (scales > scale_min) &
        (scales < scale_max)
    )
    keep_indices = torch.nonzero(keep_mask).squeeze(-1)

    print(f"Pruned {scene.num_primitives - keep_indices.numel()} primitives.")

    scene.positions = nn.Parameter(scene.positions[keep_indices])
    scene.log_scales = nn.Parameter(scene.log_scales[keep_indices])
    scene.mlps = nn.ModuleList([scene.mlps[i] for i in keep_indices])
    scene.num_primitives = keep_indices.numel()
    scene.alpha_accum = scene.alpha_accum[keep_indices]
    scene.occupancy_accum = scene.occupancy_accum[keep_indices]


# TODO: improve densification strategy (current version is a placeholder)
def densify_primitives(scene, grad_thresh=0.002, scale_thresh=0.1, shrink_factor=1.6, num_new=32):
    grad = scene.positions.grad          # (N, 3)
    grad_norm = torch.norm(grad, dim=1)  # (N,)
    log_scales = scene.log_scales.squeeze(-1)  # (N,)
    scales = torch.exp(log_scales)

    # 找出 grad_norm 大于阈值的 primitive
    mask = grad_norm > grad_thresh
    if not mask.any():
        print("No primitives met gradient threshold for densification.")
        return

    indices = torch.nonzero(mask).squeeze(-1)
    grad_norm_valid = grad_norm[indices]

    # 只保留前 num_new 个（按 grad_norm 降序）
    topk = min(num_new, indices.numel())
    _, topk_order = torch.topk(grad_norm_valid, k=topk)
    selected_indices = indices[topk_order]

    # 分类：大 scale / 小 scale
    selected_scales = scales[selected_indices]
    selected_log_scales = log_scales[selected_indices]
    selected_grad = grad[selected_indices]
    selected_pos = scene.positions[selected_indices]

    mask_large = selected_scales > scale_thresh
    mask_small = ~mask_large

    # --- Case 1: scale > threshold ---
    pos_large = selected_pos[mask_large]
    scale_large = selected_scales[mask_large] / shrink_factor
    log_scale_large = torch.log(scale_large)

    # 更新原始 primitive（scale /= shrink_factor）
    scene.positions.data[selected_indices[mask_large]] = pos_large
    scene.log_scales.data[selected_indices[mask_large]] = log_scale_large.unsqueeze(-1)

    # 复制：加高斯噪声
    noise = torch.randn_like(pos_large)
    new_pos_large = pos_large + scale_large.unsqueeze(1) * noise
    new_log_scale_large = log_scale_large
    new_mlps_large = [copy.deepcopy(scene.mlps[i]) for i in selected_indices[mask_large].tolist()]

    # --- Case 2: scale <= threshold ---
    pos_small = selected_pos[mask_small]
    scale_small = selected_scales[mask_small]
    log_scale_small = selected_log_scales[mask_small]
    grad_small = selected_grad[mask_small]

    direction = grad_small / (grad_small.norm(dim=1, keepdim=True) + 1e-8)
    offset = direction * scale_small.unsqueeze(-1)
    new_pos_small = pos_small + offset
    new_log_scale_small = log_scale_small
    new_mlps_small = [copy.deepcopy(scene.mlps[i]) for i in selected_indices[mask_small].tolist()]

    # 合并新增 primitive
    new_positions = torch.cat([new_pos_large, new_pos_small], dim=0)
    new_log_scales = torch.cat([new_log_scale_large, new_log_scale_small], dim=0)
    new_mlps = new_mlps_large + new_mlps_small

    scene.positions = nn.Parameter(torch.cat([scene.positions, new_positions], dim=0))
    scene.log_scales = nn.Parameter(torch.cat([scene.log_scales, new_log_scales.unsqueeze(-1)], dim=0))
    scene.mlps.extend(new_mlps)
    scene.num_primitives += len(new_mlps)

    # 扩展统计 buffer
    device = scene.alpha_accum.device
    zeros = torch.zeros(len(new_mlps), device=device)
    scene.alpha_accum = torch.cat([scene.alpha_accum, zeros], dim=0)
    scene.occupancy_accum = torch.cat([scene.occupancy_accum, zeros], dim=0)

    print(f"Densified {len(new_mlps)} new primitives (max {num_new}, grad_thresh={grad_thresh}).")



"""
def primitive_consistency_loss(model, ray_o, ray_d, eps=1e-3):
    B = ray_o.shape[0]
    N = model.num_primitives

    ray_d = F.normalize(ray_d, dim=-1)
    noise = torch.randn_like(ray_o) * eps
    ray_o_perturbed = ray_o + noise

    ray_o_expand = ray_o.unsqueeze(0).expand(N, B, 3)
    ray_o_perturbed_expand = ray_o_perturbed.unsqueeze(0).expand(N, B, 3)
    ray_d_expand = ray_d.unsqueeze(0).expand(N, B, 3)
    center_expand = model.positions[:, None, :].expand(N, B, 3)

    # 原始投影距离
    vec = center_expand - ray_o_expand
    t = torch.sum(vec * ray_d_expand, dim=-1, keepdim=True)
    proj = ray_o_expand + t * ray_d_expand
    dist = center_expand - proj  # (N, B, 3)

    # 偏移后的投影距离
    vec2 = center_expand - ray_o_perturbed_expand
    t2 = torch.sum(vec2 * ray_d_expand, dim=-1, keepdim=True)
    proj2 = ray_o_perturbed_expand + t2 * ray_d_expand
    dist2 = center_expand - proj2  # (N, B, 3)

    loss_total = 0.0
    count = 0

    for i in range(N):
        norm = torch.norm(dist[i], dim=-1, keepdim=True)
        mask = norm <= 5 * model.scales[i]  # (B, 1)

        if not mask.any():
            continue

        out1 = model.mlps[i](ray_d, dist[i])  # (B, 4)
        out2 = model.mlps[i](ray_d, dist2[i])  # (B, 4)

        rgb1 = out1[..., :3]
        rgb2 = out2[..., :3]

        diff = F.mse_loss(rgb1[mask.squeeze(-1)], rgb2[mask.squeeze(-1)])
        loss_total += diff
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=ray_o.device)
    else:
        return loss_total / count
"""


def sample_patch_rays(rays_o, rays_d, rgbs, num_patches_h=4, num_patches_w=4):
    """
    从 (N, H, W, 3) 的 rays 中采样一个随机 patch。

    参数:
        rays_o, rays_d, rgbs: shape (N, H, W, 3)
        num_patches_h: 高度方向的划分数
        num_patches_w: 宽度方向的划分数

    返回:
        rays_o_batch, rays_d_batch, rgbs_batch: shape (patch_H * patch_W, 3)
    """
    N, H, W, _ = rays_o.shape
    patch_H = H // num_patches_h
    patch_W = W // num_patches_w

    # 随机选图像
    img_idx = torch.randint(0, N, ()).item()

    # 随机选 patch 行列索引
    row = torch.randint(0, num_patches_h, ()).item()
    col = torch.randint(0, num_patches_w, ()).item()

    h0, h1 = row * patch_H, (row + 1) * patch_H
    w0, w1 = col * patch_W, (col + 1) * patch_W

    # 提取 patch 并 reshape 成 (patch_H * patch_W, 3)
    rays_o_patch = rays_o[img_idx, h0:h1, w0:w1].reshape(-1, 3)
    rays_d_patch = rays_d[img_idx, h0:h1, w0:w1].reshape(-1, 3)
    rgbs_patch   = rgbs[img_idx, h0:h1, w0:w1].reshape(-1, 3)

    return rays_o_patch, rays_d_patch, rgbs_patch



# 设置用于 position 的 warm-up 和 linear decay 学习率函数
def get_position_lr(iteration, warmup_steps=500, max_steps=100000, init_lr=5e-3, final_lr=5e-4):
    if iteration < warmup_steps:
        return 5e-4
    else:
        return 5e-3
    # 线性衰减
    #else:
    #    alpha = (iteration - warmup_steps) / (max_steps - warmup_steps)
    #    return init_lr * (1 - alpha) + final_lr * alpha


def get_mlp_lr(iteration, max_steps=100000, init_lr=1e-3, final_lr=1e-4):
    halfway = max_steps // 2
    if iteration < halfway:
        return init_lr
    else:
        alpha = (iteration - halfway) / (max_steps - halfway)
        return init_lr * (1 - alpha) + final_lr * alpha
    

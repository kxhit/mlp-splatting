from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import inspect
import math
import os
import re
import threading
from tqdm import tqdm
import time
from collections import OrderedDict, deque
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    import wandb
except ImportError:  # Experiment tracking is optional for a clean local run.
    wandb = None

USE_WANDB = wandb is not None and os.environ.get("MLPSP_WANDB", "0") == "1"

try:
    from .mip360_loader import mip360Dataset
    from .feature_decoder import FeatureUpsampleDecoder
    from .checkpoints import normalize_scene_state_dict, split_checkpoint
    from .utils.loss_utils import ssim
except ImportError:  # Support `python mip360/run.py`.
    from mip360_loader import mip360Dataset
    from feature_decoder import FeatureUpsampleDecoder
    from checkpoints import normalize_scene_state_dict, split_checkpoint
    from utils.loss_utils import ssim


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


def _is_scannet_scene_path(scene_dir: str) -> bool:
    abs_path = os.path.abspath(str(scene_dir))
    abs_path_lower = abs_path.lower()
    scene_name = os.path.basename(os.path.normpath(abs_path)).lower()
    return ("scannet" in abs_path_lower) or (re.fullmatch(r"scene\d{4}_\d{2}", scene_name) is not None)


def _compute_valid_rgb_mask(gt_rgb: torch.Tensor, threshold: float) -> torch.Tensor:
    # gt_rgb: [B, 3] in [0, 1]
    valid = (torch.amax(gt_rgb, dim=-1, keepdim=True) > float(threshold)).to(dtype=gt_rgb.dtype)
    if float(valid.sum().item()) <= 0.0:
        valid = torch.ones_like(valid)
    return valid


def _masked_l1_loss(pred: torch.Tensor, gt: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    weight = valid_mask.expand_as(pred)
    denom = torch.clamp(weight.sum(), min=1.0)
    return (torch.abs(pred - gt) * weight).sum() / denom


def _masked_mse_loss(pred: torch.Tensor, gt: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    weight = valid_mask.expand_as(pred)
    denom = torch.clamp(weight.sum(), min=1.0)
    return (((pred - gt) ** 2) * weight).sum() / denom


def _filter_kwargs_for_callable(func, kwargs: dict) -> dict:
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _zero_optimizer_state_rows(optimizer: torch.optim.Optimizer, param: torch.nn.Parameter, row_ids: torch.Tensor) -> None:
    if row_ids.numel() == 0:
        return
    state = optimizer.state.get(param, None)
    if not isinstance(state, dict):
        return
    for v in state.values():
        if not torch.is_tensor(v):
            continue
        if v.ndim == 0:
            continue
        if v.shape[0] != param.shape[0]:
            continue
        v[row_ids.to(device=v.device)] = 0


@torch.no_grad()
def _stabilize_scene_after_step(
    scene: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    log_scale_min: float,
    log_scale_max: float,
    renorm_quats_after_step: bool,
    sanitize_nonfinite_params: bool,
    reset_nonfinite_optimizer_state: bool,
) -> dict[str, int]:
    stats = {"sanitized_primitives": 0}

    if not hasattr(scene, "positions") or not isinstance(getattr(scene, "positions"), torch.nn.Parameter):
        return stats

    num_prims = int(scene.positions.shape[0])
    if num_prims <= 0:
        return stats

    lo = float(min(log_scale_min, log_scale_max))
    hi = float(max(log_scale_min, log_scale_max))

    if hasattr(scene, "log_scales") and isinstance(getattr(scene, "log_scales"), torch.nn.Parameter):
        scene.log_scales.data.clamp_(min=lo, max=hi)

    if renorm_quats_after_step and hasattr(scene, "quats") and isinstance(getattr(scene, "quats"), torch.nn.Parameter):
        q = scene.quats.data
        q_norm = torch.linalg.norm(q, dim=-1, keepdim=True)
        valid = torch.isfinite(q).all(dim=-1, keepdim=True) & (q_norm > 1e-12)
        q_unit = q / torch.clamp(q_norm, min=1e-12)
        q.copy_(torch.where(valid, q_unit, torch.zeros_like(q)))
        invalid_rows = (~valid.squeeze(-1)).nonzero(as_tuple=False).squeeze(-1)
        if invalid_rows.numel() > 0:
            q[invalid_rows] = 0
            q[invalid_rows, 0] = 1

    if not sanitize_nonfinite_params:
        return stats

    per_prim_param_names = ["positions", "log_scales", "quats", "W1", "b1", "W2", "b2", "W3", "b3", "feat_attr"]
    bad_rows = torch.zeros(num_prims, dtype=torch.bool, device=scene.positions.device)
    tracked_params: dict[str, torch.nn.Parameter] = {}
    for name in per_prim_param_names:
        p = getattr(scene, name, None)
        if not isinstance(p, torch.nn.Parameter):
            continue
        if p.ndim == 0 or p.shape[0] != num_prims:
            continue
        tracked_params[name] = p
        row_finite = torch.isfinite(p.data).reshape(num_prims, -1).all(dim=1)
        bad_rows |= ~row_finite

    bad_ids = bad_rows.nonzero(as_tuple=False).squeeze(-1)
    if bad_ids.numel() == 0:
        return stats

    stats["sanitized_primitives"] = int(bad_ids.numel())

    for name, p in tracked_params.items():
        if name == "log_scales":
            p.data[bad_ids] = lo
            p.data[bad_ids].clamp_(min=lo, max=hi)
        elif name == "quats":
            p.data[bad_ids] = 0
            p.data[bad_ids, 0] = 1
        else:
            p.data[bad_ids] = 0

    if reset_nonfinite_optimizer_state:
        for p in tracked_params.values():
            _zero_optimizer_state_rows(optimizer, p, bad_ids)

    return stats


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
    if feat_attr.ndim == 2 and feat_attr.shape[1] > 0:
        return int(feat_attr.shape[1])
    if feat_attr.ndim == 3 and feat_attr.shape[2] > 0:
        return int(feat_attr.shape[2])
    return None


def _infer_feature_slots(state_dict: dict) -> int | None:
    feat_attr = state_dict.get("feat_attr")
    if not isinstance(feat_attr, torch.Tensor):
        return None
    if feat_attr.ndim == 2:
        return 1
    if feat_attr.ndim == 3 and feat_attr.shape[1] > 0:
        return int(feat_attr.shape[1])
    return None


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
    feature_slots: int,
    feature_dim_gt: int,
    epoch: int,
    engine: str,
    tile_order: str,
    output_dir: str,
):
    metadata = dict(getattr(scene, "checkpoint_metadata", {}))
    payload = {
        "format_version": 2,
        "metadata": metadata,
        "epoch": int(epoch),
        "engine": str(engine),
        "tile_order": str(tile_order),
        "output_dir": str(output_dir),
        "feature_decoder_enabled": bool(feature_decoder_enabled),
        "feature_dim_low": int(feature_dim_low),
        "feature_slots": int(feature_slots),
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
    checkpoint_view = split_checkpoint(raw)
    normalized_scene_state = normalize_scene_state_dict(
        checkpoint_view.scene_state_dict
    )

    # New-format training-state checkpoint.
    if checkpoint_view.wrapper_key is not None:
        ckpt_sd = normalized_scene_state
        model_sd = scene.state_dict()
        filtered_sd = {k: v for k, v in ckpt_sd.items() if k in model_sd and model_sd[k].shape == v.shape}
        required = {
            "positions", "log_scales", "quats",
            "W1", "b1", "W2", "b2", "W3", "b3",
        }
        if getattr(scene, "feat_attr", None) is not None:
            required.add("feat_attr")
        missing_required = sorted(required - filtered_sd.keys())
        if missing_required:
            raise ValueError(
                f"Resume checkpoint is missing shape-compatible scene tensors: {missing_required}"
            )
        model_sd.update(filtered_sd)
        scene.load_state_dict(model_sd, strict=False)
        print(
            f"[Resume] Loaded {checkpoint_view.wrapper_key} "
            f"(matched {len(filtered_sd)} tensors)"
        )
        resumed_epoch = int(raw.get("epoch", _extract_epoch_from_path(resume_ckpt)))
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
    ckpt_dict = {
        k: v
        for k, v in normalized_scene_state.items()
        if k in model_dict and model_dict[k].shape == v.shape
    }
    required = {
        "positions", "log_scales", "quats",
        "W1", "b1", "W2", "b2", "W3", "b3",
    }
    if getattr(scene, "feat_attr", None) is not None:
        required.add("feat_attr")
    missing_required = sorted(required - ckpt_dict.keys())
    if missing_required:
        raise ValueError(
            f"Legacy resume checkpoint is missing shape-compatible scene tensors: {missing_required}"
        )
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


def _save_scene_with_optional_decoder(
    scene_ckpt_path: str,
    scene: nn.Module,
    feature_decoder: nn.Module | None,
    *,
    epoch: int,
):
    payload = {
        "format_version": 2,
        "metadata": dict(getattr(scene, "checkpoint_metadata", {})),
        "epoch": int(epoch),
        "scene_state_dict": scene.state_dict(),
    }
    if feature_decoder is not None:
        payload["feature_decoder_state_dict"] = feature_decoder.state_dict()
    torch.save(payload, scene_ckpt_path)
    if feature_decoder is not None:
        # Retain the paired file for compatibility with older evaluation tools.
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
    c2w = c2w_3x4.to(device=device, dtype=torch.float32, non_blocking=True)
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    Rt = R.transpose(0, 1)
    view = torch.zeros((4, 4), dtype=torch.float32, device=device)
    view[:3, :3] = Rt
    view[:3, 3] = -(Rt @ t)
    view[3, 3] = 1.0
    return view


def _c2w_to_view_matrices(c2w_nx3x4: torch.Tensor, device: str | torch.device):
    """Vectorized (N,3,4) camera-to-world -> (N,4,4) world-to-camera."""
    c2w = c2w_nx3x4.to(device=device, dtype=torch.float32, non_blocking=True)
    R = c2w[..., :3]
    t = c2w[..., 3]
    Rt = R.transpose(-1, -2)
    view = torch.zeros((c2w.shape[0], 4, 4), dtype=torch.float32, device=device)
    view[:, :3, :3] = Rt
    view[:, :3, 3] = -(Rt @ t.unsqueeze(-1)).squeeze(-1)
    view[:, 3, 3] = 1.0
    return view


class _TileMosaicFeaturePrefetcher:
    """Best-effort async feature prefetcher for tile_mosaic steps."""

    def __init__(self, dataset, *, max_workers: int, cache_size: int):
        self.dataset = dataset
        self.max_workers = int(max(1, max_workers))
        self.cache_size = int(max(1, cache_size))
        self._ready = OrderedDict()
        self._inflight: dict[int, Future] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="tilefeat",
        )
        self.prefetch_submitted = 0
        self.cache_hits = 0
        self.waited_inflight = 0
        self.sync_loads = 0
        self.load_failures = 0

    def close(self):
        with self._lock:
            inflight = list(self._inflight.values())
            self._inflight.clear()
        for fut in inflight:
            fut.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def summary(self) -> str:
        return (
            f"submitted={self.prefetch_submitted}, "
            f"cache_hits={self.cache_hits}, "
            f"waited_inflight={self.waited_inflight}, "
            f"sync_loads={self.sync_loads}, "
            f"failures={self.load_failures}"
        )

    def _load_one(self, idx: int):
        feat_path = self.dataset._feature_paths[int(idx)]
        feat = self.dataset._load_feature_tensor(feat_path)
        return feat.contiguous().cpu()

    def _store_ready_locked(self, idx: int, feat: torch.Tensor):
        key = int(idx)
        self._ready[key] = feat
        self._ready.move_to_end(key)
        while len(self._ready) > self.cache_size:
            self._ready.popitem(last=False)

    def _collect_done_locked(self):
        done_items = []
        for idx, fut in list(self._inflight.items()):
            if fut.done():
                done_items.append((idx, fut))
                del self._inflight[idx]
        for idx, fut in done_items:
            try:
                feat = fut.result()
            except Exception:
                self.load_failures += 1
                continue
            self._store_ready_locked(idx, feat)

    def prefetch(self, indices):
        if indices is None:
            return
        uniq_indices = []
        seen = set()
        for i in indices:
            key = int(i)
            if key in seen:
                continue
            seen.add(key)
            uniq_indices.append(key)
        if len(uniq_indices) == 0:
            return

        with self._lock:
            self._collect_done_locked()
            for idx in uniq_indices:
                if idx in self._ready or idx in self._inflight:
                    continue
                self._inflight[idx] = self._executor.submit(self._load_one, idx)
                self.prefetch_submitted += 1

    def get(self, idx: int) -> torch.Tensor:
        key = int(idx)
        fut = None
        with self._lock:
            self._collect_done_locked()
            cached = self._ready.get(key)
            if cached is not None:
                self._ready.move_to_end(key)
                self.cache_hits += 1
                return cached
            fut = self._inflight.pop(key, None)

        if fut is not None:
            self.waited_inflight += 1
            try:
                feat = fut.result()
            except Exception:
                self.load_failures += 1
                feat = self._load_one(key)
                self.sync_loads += 1
            with self._lock:
                self._store_ready_locked(key, feat)
            return feat

        feat = self._load_one(key)
        self.sync_loads += 1
        with self._lock:
            self._store_ready_locked(key, feat)
        return feat


def _sample_tile_mosaic_batch(
    dataset,
    view_ids: torch.Tensor,
    tile_size: int,
    view_cache: OrderedDict,
    include_feature: bool = False,
    feature_getter=None,
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
    use_manual_view_fetch = bool(
        include_feature
        and (feature_getter is not None)
        and hasattr(dataset, "_get_rays_cached")
        and hasattr(dataset, "images")
        and hasattr(dataset, "poses")
    )

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
        if feature_getter is not None:
            feat = feature_getter(int(idx))
            if feat is not None:
                return feat.contiguous().cpu()
        if getattr(dataset, "features", None) is not None:
            feat = dataset.features[int(idx)]
            return feat.contiguous().cpu()
        if getattr(dataset, "_feature_paths", None) is not None:
            feat = dataset._get_feature_lazy(int(idx))
            return feat.contiguous().cpu()
        return None

    def _to_tiles(x_hw3: torch.Tensor):
        if x_hw3.shape[0] != Hpad or x_hw3.shape[1] != Wpad:
            pad_h = Hpad - x_hw3.shape[0]
            pad_w = Wpad - x_hw3.shape[1]
            x_hw3 = F.pad(x_hw3.permute(2, 0, 1), (0, pad_w, 0, pad_h)).permute(1, 2, 0).contiguous()
        return x_hw3.view(Ty, tile_size, Tx, tile_size, 3).permute(0, 2, 1, 3, 4).contiguous()

    def _load_view_without_feature(idx: int):
        key = int(idx)
        img = dataset.images[key]
        pose = dataset.poses[key]
        rays_o_hw, rays_d_hw = dataset._get_rays_cached(key, img.shape[1:], pose)
        img_hw3 = img.permute(1, 2, 0)
        return rays_o_hw, rays_d_hw, img_hw3

    def _get_view(idx: int):
        key = int(idx)
        cached = view_cache.get(key)
        feat_hwc = None
        if cached is not None:
            view_cache.move_to_end(key)
            rays_o_tiles, rays_d_tiles, img_tiles = cached
            if include_feature:
                feat_hwc = _load_feature_for_view(key)
            return rays_o_tiles, rays_d_tiles, img_tiles, feat_hwc
        if use_manual_view_fetch:
            rays_o_hw, rays_d_hw, img_hw3 = _load_view_without_feature(key)
        else:
            rays_o_hw, rays_d_hw, img_hw3, feat_hwc = _parse_view_sample(dataset[key])
        cached = (
            _to_tiles(rays_o_hw.contiguous().cpu()),
            _to_tiles(rays_d_hw.contiguous().cpu()),
            _to_tiles(img_hw3.contiguous().cpu()),
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

    rays_o_tiles_views = []
    rays_d_tiles_views = []
    rgb_tiles_views = []
    feat_views = [] if include_feature else None
    feat_shape = None
    for i in range(K):
        rid = int(view_ids[i].item())
        ro_tiles, rd_tiles, rgb_tiles, feat = _get_view(rid)
        rays_o_tiles_views.append(ro_tiles)
        rays_d_tiles_views.append(rd_tiles)
        rgb_tiles_views.append(rgb_tiles)
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

    rays_o_tiles = torch.stack(rays_o_tiles_views, dim=0)  # (K,Ty,Tx,ts,ts,3)
    rays_d_tiles = torch.stack(rays_d_tiles_views, dim=0)
    rgb_tiles = torch.stack(rgb_tiles_views, dim=0)

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
    dataset_class=None,
    dataset_name: str = "mip360",
    factor: int = 1,
    llffhold: int = 8,
    split_mode: str = "llff",
    split_test_offset: int | None = None,
    expected_views: int = 0,
    expected_primitives: int = 0,
    num_epochs=60000,
    batch_size=4096,
    lr=1e-3,
    device='cuda',
    initial_num_primitives=1024,
    min_points: int = 10,
    engine: str = "tile_cuda",
    tile_order: str = "tilecenter",
    tile_size: int = 8,
    resume_ckpt: str | None = None,
    run_tag: str | None = None,
    enable_pe_ramp: bool = False,
    pe_ramp_start: int = 0,
    pe_ramp_steps: int = 50000,
    enable_center_lr_warmup: bool = False,
    center_lr_warmup_steps: int = 5000,
    center_lr_start_factor: float = 0.1,
    tile_mosaic: bool = True,
    tile_mosaic_k_views: int = 4,
    tile_mosaic_warmup_steps: int = 0,
    feature_type: str = "none",
    feature_dir: str | None = None,
    feature_loss_weight: float = 1.0,
    feature_decoder_hidden_dim: int = 128,
    feature_lowdim_divisor: int = 1,
    feature_slots: int = 4,
    feature_gating_beta: float = 1.5,
    feature_gating_temp: float = 2.0,
    feature_mock_from_rgb: bool = False,
    lambda_dssim: float = 0.1,
    black_border_mask: str = "auto",
    black_border_threshold: float = 1e-3,
    late_cosine_start: int = 80000,
    late_cosine_total_steps: int = 110000,
    late_cosine_min_mlp: float = 1e-2,
    late_cosine_min_scale: float = 0.2,
    late_cosine_min_position: float = 0.3,
    late_cosine_min_extra: float = 0.2,
    log_scale_lr: float = 1e-2,
    log_scale_min: float = -12.0,
    log_scale_max: float = 4.0,
    log_scale_grad_clip: float = 0.0,
    renorm_quats_after_step: bool = True,
    sanitize_nonfinite_params: bool = True,
    reset_nonfinite_optimizer_state: bool = True,
    preload_features: bool = False,
    feature_cache_size: int = 16,
    ray_cache_size: int = 8,
    dataloader_num_workers: int = 0,
    dataloader_pin_memory: bool = False,
    dataloader_persistent_workers: bool = False,
    dataloader_prefetch_factor: int = 2,
    tile_mosaic_feature_prefetch: bool = True,
    tile_mosaic_prefetch_workers: int = 2,
    tile_mosaic_prefetch_depth: int = 2,
    tile_mosaic_prefetch_cache_size: int = 0,
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
    engine = str(engine).strip().lower()
    if engine != "tile_cuda":
        raise ValueError(
            "The clean release supports the feature-capable tile_cuda training path only. "
            "Use tile_order='per_pixel' for the reference ordering."
        )
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("tile_cuda training requires an available CUDA device")
    device = torch.device(
        "cuda", torch.cuda.current_device() if device.index is None else device.index
    )
    torch.cuda.set_device(device)

    # Resolve resume metadata early so model shape can match checkpoint tensors.
    resume_probe = None
    resume_state_dict = None
    resume_metadata = {}
    if resume_ckpt:
        resume_ckpt = os.path.abspath(resume_ckpt)
        if not os.path.exists(resume_ckpt):
            raise FileNotFoundError(f"--resume_ckpt not found: {resume_ckpt}")
        resume_probe = torch.load(resume_ckpt, map_location="cpu")
        if not isinstance(resume_probe, dict):
            raise TypeError(f"Unsupported checkpoint type at {resume_ckpt}")
        resume_state_dict = normalize_scene_state_dict(
            split_checkpoint(resume_probe).scene_state_dict
        )
        if isinstance(resume_probe.get("metadata"), dict):
            resume_metadata = dict(resume_probe["metadata"])
        inferred_n = _infer_num_primitives(resume_state_dict)
        if inferred_n is not None and inferred_n != int(initial_num_primitives):
            print(f"[Resume] Override num_primitives from {initial_num_primitives} -> {inferred_n}")
            initial_num_primitives = int(inferred_n)

    # ``output_root`` is the exact run directory unless an explicit tag asks
    # for one named child.  This mirrors common 3DGS/NeRF command-line usage.
    os.makedirs(output_root, exist_ok=True)
    random_tag = str(run_tag).strip() if run_tag and str(run_tag).strip() else os.path.basename(os.path.abspath(output_root))
    output_dir = os.path.join(output_root, random_tag) if run_tag and str(run_tag).strip() else output_root
    os.makedirs(output_dir, exist_ok=True)

    feature_type = str(feature_type).lower()
    if resume_metadata and "feature_type" in resume_metadata:
        checkpoint_feature_type = str(resume_metadata["feature_type"]).lower()
        if checkpoint_feature_type != feature_type:
            print(
                f"[Resume] Override feature_type from {feature_type!r} "
                f"-> {checkpoint_feature_type!r}"
            )
        feature_type = checkpoint_feature_type
    use_feature_metric = feature_type in {"lseg", "sam"}
    feature_dim_gt = 0
    if feature_type == "lseg":
        feature_dim_gt = 512
    elif feature_type == "sam":
        feature_dim_gt = 256
    elif feature_type != "none":
        raise ValueError(f"Unsupported feature_type: {feature_type}")
    feature_dim_low = feature_dim_gt // int(max(1, feature_lowdim_divisor)) if use_feature_metric else 0
    feature_slots = int(max(1, feature_slots))
    feature_gating_beta = float(feature_gating_beta)
    feature_gating_temp = float(feature_gating_temp)
    if resume_metadata:
        tile_order = str(resume_metadata.get("tile_order", tile_order))
        tile_size = int(resume_metadata.get("tile_size", tile_size))
        feature_gating_beta = float(
            resume_metadata.get("feature_gating_beta", feature_gating_beta)
        )
        feature_gating_temp = float(
            resume_metadata.get("feature_gating_temperature", feature_gating_temp)
        )
        encoding_kind = str(resume_metadata.get("encoding_kind", "attenuated_phase_v1"))
        if encoding_kind != "attenuated_phase_v1":
            raise ValueError(
                f"Unsupported checkpoint encoding_kind={encoding_kind!r}; "
                "the clean training path uses 'attenuated_phase_v1'."
            )
        pe_sigma = float(resume_metadata.get("encoding_sigma", 0.01))
    else:
        pe_sigma = 0.01
    resume_feature_dim_low = _infer_feature_dim_low(resume_state_dict) if isinstance(resume_state_dict, dict) else None
    resume_feature_slots = _infer_feature_slots(resume_state_dict) if isinstance(resume_state_dict, dict) else None
    if feature_type == "none" and resume_feature_dim_low not in (None, 0):
        raise ValueError(
            "The resume checkpoint contains feature attributes but has no feature_type "
            "metadata. Pass --feature-type lseg or --feature-type sam explicitly."
        )
    if use_feature_metric and (resume_feature_dim_low is not None) and (int(resume_feature_dim_low) != int(feature_dim_low)):
        print(f"[Resume] Override feature_dim_low from {feature_dim_low} -> {int(resume_feature_dim_low)}")
        feature_dim_low = int(resume_feature_dim_low)
    if use_feature_metric and (resume_feature_slots is not None) and (int(resume_feature_slots) != int(feature_slots)):
        print(f"[Resume] Override feature_slots from {feature_slots} -> {int(resume_feature_slots)}")
        feature_slots = int(resume_feature_slots)
    if use_feature_metric and int(feature_dim_low) <= 0:
        raise ValueError("featuremetric training requires feature_dim_low > 0.")
    if use_feature_metric and feature_gating_beta <= 0.0:
        raise ValueError("feature_gating_beta must be > 0.")
    if use_feature_metric and feature_gating_temp <= 0.0:
        raise ValueError("feature_gating_temp must be > 0.")
    use_feature_decoder = bool(use_feature_metric and int(feature_dim_low) != int(feature_dim_gt))
    if use_feature_metric:
        if use_feature_decoder:
            print(f"[Feature] Decoder enabled (low_dim={feature_dim_low}, gt_dim={feature_dim_gt})")
        else:
            print(f"[Feature] Decoder disabled (low_dim == gt_dim == {feature_dim_gt})")

    use_mock_feature = False
    resolved_dataset_class = mip360Dataset if dataset_class is None else dataset_class
    dataset_kwargs = {
        "basedir": data_dir,
        "factor": factor,
        "llffhold": int(llffhold),
        "split": "train",
        "split_mode": split_mode,
        "split_test_offset": split_test_offset,
        "feature_type": feature_type,
        "feature_dir": feature_dir,
        "preload_features": bool(preload_features),
        "feature_cache_size": int(feature_cache_size),
        "feature_resize_to_image": False,
        "ray_cache_size": int(ray_cache_size),
    }
    try:
        dataset = resolved_dataset_class(
            **_filter_kwargs_for_callable(resolved_dataset_class.__init__, dataset_kwargs)
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        if use_feature_metric and feature_mock_from_rgb:
            print(f"[Feature] {exc}")
            print("[Feature] Falling back to RGB-projected mock features for pipeline verification.")
            fallback_kwargs = dict(dataset_kwargs)
            fallback_kwargs.update(
                feature_type="none",
                feature_dir=None,
                preload_features=False,
                feature_cache_size=0,
            )
            dataset = resolved_dataset_class(
                **_filter_kwargs_for_callable(resolved_dataset_class.__init__, fallback_kwargs)
            )
            use_mock_feature = True
        else:
            raise
    if int(expected_views) > 0 and len(dataset) != int(expected_views):
        raise RuntimeError(
            f"Training split has {len(dataset)} views; expected {int(expected_views)}. "
            "Check the dataset archive and split settings."
        )
    tile_size = int(tile_size)
    if tile_size <= 0:
        raise ValueError("tile_size must be a positive integer.")
    if tile_size not in (8, 16):
        raise ValueError("tile_cuda currently supports tile_size in {8, 16}.")
    tile_mosaic_k_views = int(max(1, tile_mosaic_k_views))
    tile_mosaic_warmup_steps = int(max(0, tile_mosaic_warmup_steps))
    use_tile_mosaic = bool(tile_mosaic)
    if use_tile_mosaic and tile_order not in {"tilecenter", "per_pixel"}:
        raise ValueError("tile_mosaic currently supports tile_order in {'tilecenter', 'per_pixel'} only.")
    if use_tile_mosaic and use_feature_metric:
        print("[Feature] tile_mosaic is enabled; semantic GT will be mosaicked per tile-view assignment.")
    log_scale_lr = float(max(0.0, log_scale_lr))
    log_scale_grad_clip = float(max(0.0, log_scale_grad_clip))
    log_scale_min = float(log_scale_min)
    log_scale_max = float(log_scale_max)
    if log_scale_max < log_scale_min:
        log_scale_min, log_scale_max = log_scale_max, log_scale_min
    renorm_quats_after_step = bool(renorm_quats_after_step)
    sanitize_nonfinite_params = bool(sanitize_nonfinite_params)
    reset_nonfinite_optimizer_state = bool(reset_nonfinite_optimizer_state)

    # The public path uses the full-frame CUDA renderer for both RGB and feature
    # supervision.  ``per_pixel`` remains available through tile_order as the
    # accuracy reference without introducing a second model implementation.
    try:
        from .cumodel import NeuralPrimitiveSceneTileCUDA
    except ImportError:
        from cumodel import NeuralPrimitiveSceneTileCUDA
    init_positions = None if resume_ckpt else dataset.positions
    scene_kwargs = {
        "num_primitives": initial_num_primitives,
        "hidden_dim": 32,
        "pe_frequencies": 6,
        "use_integrated_pe": True,
        "pe_sigma": pe_sigma,
        "initial_positions": init_positions,
        "feature_dim_low": feature_dim_low,
        "feature_slots": feature_slots,
        "feature_gating_beta": feature_gating_beta,
        "feature_gating_temp": feature_gating_temp,
        "min_points": int(min_points),
        "tile_size": int(tile_size),
    }
    scene = NeuralPrimitiveSceneTileCUDA(
        **_filter_kwargs_for_callable(NeuralPrimitiveSceneTileCUDA.__init__, scene_kwargs)
    ).to(device)
    if (
        not resume_ckpt
        and int(expected_primitives) > 0
        and int(scene.num_primitives) != int(expected_primitives)
    ):
        raise RuntimeError(
            f"Octree initialized {int(scene.num_primitives)} primitives; expected "
            f"{int(expected_primitives)}. Check the point cloud and octree threshold."
        )
    os.environ["MLPSP_TILE_ORDER"] = str(tile_order)
    slot_layout = (
        "direct_tetrahedral"
        if int(feature_slots) == 4
        else "legacy_base_plus_four_deltas"
    ) if use_feature_metric else "none"
    scene.checkpoint_metadata = {
        "dataset": str(dataset_name),
        "dataset_split": "train",
        "factor": int(factor),
        "llffhold": int(llffhold),
        "split_mode": str(split_mode),
        "split_test_offset": (
            None if split_test_offset is None else int(split_test_offset)
        ),
        "training_views": int(len(dataset)),
        "iterations": int(num_epochs),
        "seed": int(torch.initial_seed()),
        "renderer": "tile_cuda",
        "tile_order": str(tile_order),
        "tile_size": int(tile_size),
        "score_threshold": 1e-3,
        "num_primitives": int(scene.num_primitives),
        "hidden_dim": 32,
        "pe_frequencies": 6,
        "encoding_kind": "attenuated_phase_v1",
        "encoding_sigma": float(pe_sigma),
        "feature_type": str(feature_type),
        "feature_dim": int(feature_dim_low),
        "feature_slots": int(feature_slots),
        "feature_slot_layout": slot_layout,
        "feature_gating_beta": float(feature_gating_beta),
        "feature_gating_temperature": float(feature_gating_temp),
        "feature_decoder_enabled": bool(use_feature_decoder),
        "tile_mosaic": bool(use_tile_mosaic),
        "tile_mosaic_views": int(tile_mosaic_k_views),
        "max_points_per_leaf": int(min_points),
        "learning_rate": float(lr),
        "position_learning_rate": 1e-4,
        "log_scale_learning_rate": float(log_scale_lr),
        "lambda_dssim": float(lambda_dssim),
        "feature_loss_weight": float(feature_loss_weight),
    }

    feature_decoder = None
    if use_feature_decoder:
        feature_decoder = FeatureUpsampleDecoder(
            in_dim=feature_dim_low,
            out_dim=feature_dim_gt,
            hidden_dim=int(feature_decoder_hidden_dim),
        ).to(device)

    mlp_params = [scene.W1, scene.b1, scene.W2, scene.b2, scene.W3, scene.b3]

    param_groups = [
        {"params": mlp_params, "lr": float(lr)},
        {"params": [scene.log_scales], "lr": float(log_scale_lr)},
        {"params": [scene.positions], "lr": 1e-4},
        #{"params": scene.rotations, "lr": 1e-2},
    ]
    if hasattr(scene, "quats") and isinstance(getattr(scene, "quats"), torch.nn.Parameter):
        param_groups.append({"params": [scene.quats], "lr": float(lr)})
    if use_feature_metric:
        if getattr(scene, "feat_attr", None) is None:
            raise RuntimeError("Feature training requested but scene has no feat_attr.")
        param_groups.append({"params": [scene.feat_attr], "lr": float(lr)})
        if feature_decoder is not None:
            param_groups.append({"params": feature_decoder.parameters(), "lr": float(lr)})
    optimizer = torch.optim.Adam(param_groups, lr=float(lr))
    base_group_lrs = [float(g["lr"]) for g in optimizer.param_groups]
    base_center_lr = float(base_group_lrs[2])
    lambda_dssim = float(max(0.0, min(1.0, lambda_dssim)))
    black_border_threshold = float(max(0.0, black_border_threshold))
    black_border_mask_mode = str(black_border_mask).strip().lower()
    if black_border_mask_mode not in {"auto", "on", "off"}:
        raise ValueError("black_border_mask must be one of {'auto', 'on', 'off'}.")
    if black_border_mask_mode == "on":
        use_black_border_mask = True
    elif black_border_mask_mode == "off":
        use_black_border_mask = False
    else:
        use_black_border_mask = _is_scannet_scene_path(data_dir)
    if use_black_border_mask:
        print(
            f"[Mask] Black-border loss mask enabled (mode={black_border_mask_mode}, "
            f"threshold={black_border_threshold:.6g})"
        )
    else:
        print(f"[Mask] Black-border loss mask disabled (mode={black_border_mask_mode})")

    late_cosine_start = int(max(0, late_cosine_start))
    late_cosine_total_steps = int(max(late_cosine_start + 1, late_cosine_total_steps))
    late_cosine_min_mlp = float(max(0.0, min(1.0, late_cosine_min_mlp)))
    late_cosine_min_scale = float(max(0.0, min(1.0, late_cosine_min_scale)))
    late_cosine_min_position = float(max(0.0, min(1.0, late_cosine_min_position)))
    late_cosine_min_extra = float(max(0.0, min(1.0, late_cosine_min_extra)))
    group_min_mults = []
    for gi in range(len(base_group_lrs)):
        if gi == 0:
            group_min_mults.append(late_cosine_min_mlp)
        elif gi == 1:
            group_min_mults.append(late_cosine_min_scale)
        elif gi == 2:
            group_min_mults.append(late_cosine_min_position)
        else:
            group_min_mults.append(late_cosine_min_extra)

    def _late_cosine_multiplier(step: int, min_mult: float) -> float:
        if step < late_cosine_start:
            return 1.0
        t = float(step - late_cosine_start) / float(max(1, late_cosine_total_steps - late_cosine_start))
        t = max(0.0, min(1.0, t))
        return float(min_mult + 0.5 * (1.0 - min_mult) * (1.0 + math.cos(math.pi * t)))

    def _apply_late_cosine_lrs(step: int):
        for gi, group in enumerate(optimizer.param_groups):
            group["lr"] = float(base_group_lrs[gi]) * _late_cosine_multiplier(step, group_min_mults[gi])

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
    print(
        f"[Stability] log_scale_lr={log_scale_lr:.6g}, "
        f"log_scale_clamp=[{log_scale_min:.6g}, {log_scale_max:.6g}], "
        f"log_scale_grad_clip={log_scale_grad_clip:.6g}, "
        f"renorm_quats_after_step={renorm_quats_after_step}, "
        f"sanitize_nonfinite_params={sanitize_nonfinite_params}"
    )
    print(
        f"[LR] Late-cosine enabled: start={late_cosine_start}, total_steps={late_cosine_total_steps}, "
        f"group_min_mults={group_min_mults}"
    )

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
        _save_scene_with_optional_decoder(ckpt_path, scene, feature_decoder, epoch=0)
        print(f"[Checkpoint] Saved model to {ckpt_path}")
        _save_train_state(
            path=os.path.join(output_dir, "train_state_latest.pt"),
            scene=scene,
            optimizer=optimizer,
            feature_decoder=feature_decoder,
            feature_decoder_enabled=use_feature_decoder,
            feature_dim_low=feature_dim_low,
            feature_slots=feature_slots,
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
                "num_primitives": int(scene.num_primitives),
                "min_points": int(min_points),
                "hidden_dim": 32,
                "pe_frequencies": 6,
                "num_layers": 2,
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
                "feature_slots": int(feature_slots),
                "feature_gating_beta": float(feature_gating_beta),
                "feature_gating_temp": float(feature_gating_temp),
                "feature_decoder_enabled": bool(use_feature_decoder),
                "feature_decoder_hidden_dim": int(feature_decoder_hidden_dim),
                "feature_loss_weight": float(feature_loss_weight),
                "lambda_dssim": float(lambda_dssim),
                "late_cosine_start": int(late_cosine_start),
                "late_cosine_total_steps": int(late_cosine_total_steps),
                "late_cosine_min_mlp": float(late_cosine_min_mlp),
                "late_cosine_min_scale": float(late_cosine_min_scale),
                "late_cosine_min_position": float(late_cosine_min_position),
                "late_cosine_min_extra": float(late_cosine_min_extra),
                "preload_features": bool(preload_features),
                "feature_cache_size": int(feature_cache_size),
                "ray_cache_size": int(ray_cache_size),
                "dataloader_num_workers": int(dataloader_num_workers),
                "dataloader_pin_memory": bool(dataloader_pin_memory),
                "dataloader_persistent_workers": bool(dataloader_persistent_workers),
                "dataloader_prefetch_factor": int(dataloader_prefetch_factor),
                "tile_mosaic_feature_prefetch": bool(tile_mosaic_feature_prefetch),
                "tile_mosaic_prefetch_workers": int(tile_mosaic_prefetch_workers),
                "tile_mosaic_prefetch_depth": int(tile_mosaic_prefetch_depth),
                "tile_mosaic_prefetch_cache_size": int(tile_mosaic_prefetch_cache_size),
            }
        )

    loss_accum = torch.zeros((), dtype=torch.float32, device=device)
    loss_accum2 = torch.zeros((), dtype=torch.float32, device=device)
    wandb_accum = torch.zeros((), dtype=torch.float32, device=device)
    loss_count = 0

    rays_o_all = rays_d_all = rgbs_all = None
    N = 0
    if (
        engine == "tile_cuda"
        and use_tile_mosaic
        and int(tile_mosaic_k_views) > 1
        and int(tile_mosaic_warmup_steps) == 0
        and int(dataloader_num_workers) > 0
    ):
        print(
            "[DataLoader] tile_mosaic(k>1, warmup=0): override num_workers "
            f"{int(dataloader_num_workers)} -> 0 (fast path is not consumed)."
        )
        dataloader_num_workers = 0
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
    timing_enabled = os.environ.get("MLPSP_TIMING", "0") == "1"
    timing_every = int(os.environ.get("MLPSP_TIMING_EVERY", "50"))
    mock_feature_proj = None
    if use_feature_metric and use_mock_feature:
        mock_feature_proj = _build_mock_feature_projection(feature_dim_gt, device=device)
    tile_view_cache = OrderedDict()
    tile_feature_prefetcher = None
    tile_mosaic_view_plan = deque()
    tile_mosaic_prefetch_depth = int(max(1, tile_mosaic_prefetch_depth))
    if (
        use_tile_mosaic
        and use_feature_metric
        and (not use_mock_feature)
        and bool(tile_mosaic_feature_prefetch)
        and getattr(dataset, "features", None) is None
        and getattr(dataset, "_feature_paths", None) is not None
    ):
        prefetch_cache_size = int(tile_mosaic_prefetch_cache_size)
        if prefetch_cache_size <= 0:
            prefetch_cache_size = int(max(1, feature_cache_size))
        tile_feature_prefetcher = _TileMosaicFeaturePrefetcher(
            dataset=dataset,
            max_workers=int(max(1, tile_mosaic_prefetch_workers)),
            cache_size=int(max(1, prefetch_cache_size)),
        )
        print(
            "[FeaturePrefetch] tile_mosaic async prefetch enabled "
            f"(workers={int(max(1, tile_mosaic_prefetch_workers))}, "
            f"depth={tile_mosaic_prefetch_depth}, cache={int(max(1, prefetch_cache_size))})"
        )
        while len(tile_mosaic_view_plan) < (tile_mosaic_prefetch_depth + 1):
            tile_mosaic_view_plan.append(torch.randperm(len(dataset))[:tile_mosaic_k_views].to(torch.long))
        warmup_ids = []
        for planned_ids in tile_mosaic_view_plan:
            warmup_ids.extend(int(i) for i in planned_ids.tolist())
        tile_feature_prefetcher.prefetch(warmup_ids)

    mosaic_include_feature = bool(use_feature_metric and (not use_mock_feature))
    tile_mosaic_cpu_overlap = bool(
        engine == "tile_cuda"
        and use_tile_mosaic
        and (os.environ.get("MLPSP_TILE_MOSAIC_CPU_OVERLAP", "1") == "1")
    )
    tile_mosaic_cpu_executor = None
    tile_mosaic_cpu_future: Future | None = None

    def _next_mosaic_view_ids(k_views: int) -> torch.Tensor:
        k = int(max(1, min(int(k_views), len(dataset))))
        if tile_feature_prefetcher is not None:
            while len(tile_mosaic_view_plan) < (tile_mosaic_prefetch_depth + 1):
                tile_mosaic_view_plan.append(torch.randperm(len(dataset))[:k].to(torch.long))
            view_ids_local = tile_mosaic_view_plan.popleft()
            while len(tile_mosaic_view_plan) < tile_mosaic_prefetch_depth:
                tile_mosaic_view_plan.append(torch.randperm(len(dataset))[:k].to(torch.long))
            prefetch_ids = []
            for planned_ids in tile_mosaic_view_plan:
                prefetch_ids.extend(int(i) for i in planned_ids.tolist())
            tile_feature_prefetcher.prefetch(prefetch_ids)
            return view_ids_local
        perm = torch.randperm(len(dataset))
        return perm[:k].to(torch.long)

    def _build_mosaic_cpu_batch(view_ids_cpu: torch.Tensor) -> dict[str, torch.Tensor | None]:
        feature_getter = tile_feature_prefetcher.get if tile_feature_prefetcher is not None else None
        rays_o_hw, rays_d_hw, img_hw3, feat_hwc_mosaic, tile_view_local, tile_img_ids = _sample_tile_mosaic_batch(
            dataset=dataset,
            view_ids=view_ids_cpu,
            tile_size=tile_size,
            view_cache=tile_view_cache,
            include_feature=mosaic_include_feature,
            feature_getter=feature_getter,
        )
        return {
            "view_ids": view_ids_cpu,
            "rays_o_hw": rays_o_hw,
            "rays_d_hw": rays_d_hw,
            "img_hw3": img_hw3,
            "feat_hwc_mosaic": feat_hwc_mosaic,
            "tile_view_local": tile_view_local,
            "tile_img_ids": tile_img_ids,
        }

    if tile_mosaic_cpu_overlap and int(tile_mosaic_k_views) > 1:
        tile_mosaic_cpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tilemosaiccpu")
        print(
            "[Pipeline] tile_mosaic CPU lookahead enabled "
            "(set MLPSP_TILE_MOSAIC_CPU_OVERLAP=0 to disable)."
        )
        initial_use_mosaic = bool(start_epoch >= tile_mosaic_warmup_steps)
        initial_k = min(tile_mosaic_k_views, len(dataset))
        if initial_use_mosaic and initial_k > 1:
            init_view_ids = _next_mosaic_view_ids(initial_k)
            tile_mosaic_cpu_future = tile_mosaic_cpu_executor.submit(_build_mosaic_cpu_batch, init_view_ids)


    for epoch in tqdm(range(start_epoch, num_epochs), initial=start_epoch, total=num_epochs):
        wall_start = time.perf_counter() if timing_enabled and (epoch % timing_every == 0) else None
        data_start = wall_start

        bg_color = torch.rand(3, device=device)
        feat_gt_hwc = None

        # (0) Late-cosine LR schedule.
        _apply_late_cosine_lrs(epoch)

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
            optimizer.param_groups[2]["lr"] = float(optimizer.param_groups[2]["lr"]) * lr_scale

        if engine == "tile_cuda":
            use_mosaic_step = use_tile_mosaic and (epoch >= tile_mosaic_warmup_steps)
            effective_k = min(tile_mosaic_k_views, len(dataset))
            if use_mosaic_step and effective_k > 1:
                # Build full-frame rays/GT by stitching tiles from up to k random views.
                if tile_mosaic_cpu_executor is not None:
                    if tile_mosaic_cpu_future is None:
                        view_ids = _next_mosaic_view_ids(effective_k)
                        tile_mosaic_cpu_future = tile_mosaic_cpu_executor.submit(_build_mosaic_cpu_batch, view_ids)
                    try:
                        mosaic_batch = tile_mosaic_cpu_future.result()
                    except Exception as exc:
                        raise RuntimeError("tile_mosaic CPU lookahead batch generation failed.") from exc

                    next_use_mosaic = bool(
                        (epoch + 1) < num_epochs
                        and use_tile_mosaic
                        and ((epoch + 1) >= tile_mosaic_warmup_steps)
                    )
                    next_k = min(tile_mosaic_k_views, len(dataset))
                    if next_use_mosaic and next_k > 1:
                        next_view_ids = _next_mosaic_view_ids(next_k)
                        tile_mosaic_cpu_future = tile_mosaic_cpu_executor.submit(
                            _build_mosaic_cpu_batch,
                            next_view_ids,
                        )
                    else:
                        tile_mosaic_cpu_future = None

                    view_ids = mosaic_batch["view_ids"]
                    rays_o_hw = mosaic_batch["rays_o_hw"]
                    rays_d_hw = mosaic_batch["rays_d_hw"]
                    img_hw3 = mosaic_batch["img_hw3"]
                    feat_hwc_mosaic = mosaic_batch["feat_hwc_mosaic"]
                    tile_view_local = mosaic_batch["tile_view_local"]
                    _tile_img_ids = mosaic_batch["tile_img_ids"]
                else:
                    view_ids = _next_mosaic_view_ids(effective_k)
                    feature_getter = tile_feature_prefetcher.get if tile_feature_prefetcher is not None else None
                    rays_o_hw, rays_d_hw, img_hw3, feat_hwc_mosaic, tile_view_local, _tile_img_ids = _sample_tile_mosaic_batch(
                        dataset=dataset,
                        view_ids=view_ids,
                        tile_size=tile_size,
                        view_cache=tile_view_cache,
                        include_feature=mosaic_include_feature,
                        feature_getter=feature_getter,
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
                view_mats = _c2w_to_view_matrices(poses_sel, device=device)
                proj_matrix = _build_projection_from_focal(dataset.focal, Wimg, Himg, device=device)
                tile_view_local_dev = tile_view_local.to(device=device, non_blocking=True)
                culled_prim_indices, tile_offsets = scene.build_tilecenter_culling_from_tile_views(
                    tile_view_ids=tile_view_local_dev,
                    view_matrices=view_mats,
                    proj_matrix=proj_matrix,
                    score_threshold=1e-3,
                    tile_size=tile_size,
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
            valid_rgb_mask = None
            if use_black_border_mask:
                valid_rgb_mask = _compute_valid_rgb_mask(rgbs_batch, black_border_threshold)
            # 3DGS-style RGB reconstruction: (1-lambda)*L1 + lambda*(1-SSIM).
            if valid_rgb_mask is not None:
                loss_l1 = _masked_l1_loss(pred, rgbs_batch, valid_rgb_mask)
            else:
                loss_l1 = F.l1_loss(pred, rgbs_batch)
            if engine == "tile_cuda":
                pred_img = pred.view(Himg, Wimg, 3).permute(2, 0, 1).unsqueeze(0)
                gt_img = rgbs_batch.view(Himg, Wimg, 3).permute(2, 0, 1).unsqueeze(0)
                if valid_rgb_mask is not None:
                    mask_img = valid_rgb_mask.view(Himg, Wimg, 1).permute(2, 0, 1).unsqueeze(0)
                    pred_img_for_ssim = pred_img * mask_img + gt_img * (1.0 - mask_img)
                else:
                    pred_img_for_ssim = pred_img
                loss_ssim = 1.0 - ssim(pred_img_for_ssim, gt_img)
                loss_rgb = (1.0 - lambda_dssim) * loss_l1 + lambda_dssim * loss_ssim
            else:
                loss_ssim = None
                loss_rgb = loss_l1
            loss = loss_rgb

            if valid_rgb_mask is not None:
                loss_mse = _masked_mse_loss(pred, rgbs_batch, valid_rgb_mask)
            else:
                loss_mse = F.mse_loss(pred, rgbs_batch)
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
                if valid_rgb_mask is not None:
                    mask_hw = valid_rgb_mask.view(Himg, Wimg)
                    if feat_h != Himg or feat_w != Wimg:
                        mask_hw = F.interpolate(
                            mask_hw.unsqueeze(0).unsqueeze(0),
                            size=(feat_h, feat_w),
                            mode="nearest",
                        ).squeeze(0).squeeze(0)
                    feat_weight = mask_hw.unsqueeze(-1)
                    feat_denom = torch.clamp(feat_weight.sum() * feat_pred_hwc.shape[-1], min=1.0)
                    loss_feat = (torch.abs(feat_pred_hwc - feat_gt_hwc) * feat_weight).sum() / feat_denom
                else:
                    loss_feat = F.l1_loss(feat_pred_hwc, feat_gt_hwc)
                loss = loss_rgb + float(feature_loss_weight) * loss_feat
            #loss = F.l1_loss(pred, rgbs_batch)

            if wall_start is not None and torch.cuda.is_available():
                fwd_end.record()
            with _nvtx_range("train/backward"):
                if wall_start is not None and torch.cuda.is_available():
                    bwd_start.record()
                loss.backward()                             # 4. Backpropagation
                if (
                    log_scale_grad_clip > 0.0
                    and hasattr(scene, "log_scales")
                    and isinstance(getattr(scene, "log_scales"), torch.nn.Parameter)
                    and scene.log_scales.grad is not None
                ):
                    scene.log_scales.grad.data.clamp_(-log_scale_grad_clip, log_scale_grad_clip)
                if wall_start is not None and torch.cuda.is_available():
                    bwd_end.record()

        if wall_start is not None and torch.cuda.is_available():
            opt_start.record()


        optimizer.step()
        stab_stats = _stabilize_scene_after_step(
            scene,
            optimizer,
            log_scale_min=log_scale_min,
            log_scale_max=log_scale_max,
            renorm_quats_after_step=renorm_quats_after_step,
            sanitize_nonfinite_params=sanitize_nonfinite_params,
            reset_nonfinite_optimizer_state=reset_nonfinite_optimizer_state,
        )
        if stab_stats.get("sanitized_primitives", 0) > 0:
            print(
                f"[Stability] epoch={epoch + 1} sanitized_primitives={stab_stats['sanitized_primitives']}",
                flush=True,
            )

        if wall_start is not None and torch.cuda.is_available():
            opt_end.record()

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
        loss_accum += loss_mse.detach()
        if loss_feat is not None:
            loss_accum2 += loss_feat.detach()
        wandb_accum += loss.detach()
        loss_count += 1

        # Print average loss every 100 epochs
        if (epoch + 1) % 100 == 0:
            avg_loss = loss_accum / loss_count
            avg_loss2 = loss_accum2 / loss_count
            wandb_loss = wandb_accum / loss_count
            avg_loss_scalar = float(avg_loss.item())
            avg_loss_scalar2 = float(avg_loss2.item())
            wandb_loss_scalar = float(wandb_loss.item())
            print(f"[Epoch {epoch+1}] avg {loss_count} SME loss : {avg_loss_scalar:.6f} Semantic loss : {avg_loss_scalar2:.6f}")
            # Reset accumulators
            loss_accum.zero_()
            loss_accum2.zero_()
            wandb_accum.zero_()
            loss_count = 0
            if USE_WANDB:   # log to Weights & Biases
                metrics = {
                    "avg_loss": wandb_loss_scalar,
                    "epoch": epoch + 1,
                    "loss_rgb": float(loss_rgb.item()),
                    "loss_l1": float(loss_l1.item()),
                }
                if loss_ssim is not None:
                    metrics["loss_ssim"] = float(loss_ssim.item())
                if use_feature_metric:
                    if loss_feat is not None:
                        metrics["loss_feat"] = float(loss_feat.item())
                wandb.log(metrics)
                # Reset again for safety (redundant but harmless)
                loss_accum.zero_()
                loss_accum2.zero_()
                wandb_accum.zero_()
                loss_count = 0

        # Eval after 500 epochs
        if (epoch + 1) % 500 == 0:
            _save_scene_with_optional_decoder(
                os.path.join(output_dir, "scene_ckpt_tmp.pt"),
                scene,
                feature_decoder,
                epoch=epoch + 1,
            )
            _save_train_state(
                path=os.path.join(output_dir, "train_state_latest.pt"),
                scene=scene,
                optimizer=optimizer,
                feature_decoder=feature_decoder,
                feature_decoder_enabled=use_feature_decoder,
                feature_dim_low=feature_dim_low,
                feature_slots=feature_slots,
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
            _save_scene_with_optional_decoder(
                ckpt_path, scene, feature_decoder, epoch=epoch + 1
            )
            _save_train_state(
                path=os.path.join(output_dir, f"train_state_epoch_{epoch+1:05d}.pt"),
                scene=scene,
                optimizer=optimizer,
                feature_decoder=feature_decoder,
                feature_decoder_enabled=use_feature_decoder,
                feature_dim_low=feature_dim_low,
                feature_slots=feature_slots,
                feature_dim_gt=feature_dim_gt,
                epoch=epoch + 1,
                engine=engine,
                tile_order=tile_order,
                output_dir=output_dir,
            )
            print(f"[Checkpoint] Saved model to {ckpt_path}")

    if tile_feature_prefetcher is not None:
        print(f"[FeaturePrefetch] summary: {tile_feature_prefetcher.summary()}")
        tile_feature_prefetcher.close()
    if tile_mosaic_cpu_future is not None:
        tile_mosaic_cpu_future.cancel()
        tile_mosaic_cpu_future = None
    if tile_mosaic_cpu_executor is not None:
        tile_mosaic_cpu_executor.shutdown(wait=False, cancel_futures=True)

    _save_scene_with_optional_decoder(
        os.path.join(output_dir, "scene_ckpt.pt"),
        scene,
        feature_decoder,
        epoch=num_epochs,
    )
    _save_train_state(
        path=os.path.join(output_dir, "train_state_latest.pt"),
        scene=scene,
        optimizer=optimizer,
        feature_decoder=feature_decoder,
        feature_decoder_enabled=use_feature_decoder,
        feature_dim_low=feature_dim_low,
        feature_slots=feature_slots,
        feature_dim_gt=feature_dim_gt,
        epoch=num_epochs,
        engine=engine,
        tile_order=tile_order,
        output_dir=output_dir,
    )

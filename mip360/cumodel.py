from __future__ import annotations

import math
import os
import subprocess
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .utils.sampling import octree_sample
except ImportError:  # Support direct script execution.
    from utils.sampling import octree_sample


def _is_extension_build_active(ext_name: str, sources: list[str]) -> bool:
    """Best-effort check for active nvcc/cicc/ninja related to this extension."""
    try:
        out = subprocess.check_output(["ps", "-eo", "cmd"], text=True)
    except Exception:
        return False
    ext_name_l = ext_name.lower()
    source_tokens = [os.path.basename(str(s)).lower() for s in sources]
    for line in out.splitlines():
        l = line.lower()
        if ("nvcc" not in l) and ("cicc" not in l) and ("ninja" not in l):
            continue
        if ext_name_l in l:
            return True
        for token in source_tokens:
            if token and token in l:
                return True
    return False


def _maybe_cleanup_stale_ext_lock(ext_name: str, sources: list[str], verbose: bool = False) -> None:
    """Remove stale torch extension lock files to avoid infinite wait."""
    stale_sec_raw = os.environ.get("MLPSP_EXT_LOCK_STALE_SEC", "900").strip()
    try:
        stale_sec = int(stale_sec_raw)
    except ValueError:
        stale_sec = 900
    if stale_sec <= 0:
        return
    try:
        from torch.utils.cpp_extension import _get_build_directory  # type: ignore
        build_dir = _get_build_directory(ext_name, verbose)
    except Exception:
        return
    lock_path = os.path.join(build_dir, "lock")
    if not os.path.exists(lock_path):
        return
    try:
        age_sec = max(0.0, time.time() - os.path.getmtime(lock_path))
    except OSError:
        return
    if age_sec < float(stale_sec):
        return
    if _is_extension_build_active(ext_name, sources):
        return
    try:
        os.remove(lock_path)
        print(f"[ExtLock] Removed stale lock: {lock_path} (age={age_sec:.0f}s)")
    except FileNotFoundError:
        return
    except OSError as exc:
        print(f"[ExtLock] Failed to remove lock {lock_path}: {exc}")


def _safe_load_extension(load_fn, *, name: str, sources: list[str], verbose: bool = False, **kwargs):
    _maybe_cleanup_stale_ext_lock(name, sources, verbose=verbose)
    return load_fn(name=name, sources=sources, verbose=verbose, **kwargs)

# Optional: tile-based renderer extension (loaded on demand)
_tile_ext = None
_tile_error = None
_tile_ext_key = None
try:
    from torch.utils.cpp_extension import load as load_ext
    this_dir = os.path.dirname(os.path.abspath(__file__))
    tile_src = os.path.join(this_dir, 'cuda', 'np_tile_kernels.cu')
    if os.path.exists(tile_src):
        # Delay actual compilation until first use to keep import fast
        _tile_src_path = tile_src
    else:
        _tile_error = f'Tile CUDA source missing at {tile_src}'
        _tile_src_path = None
except Exception as e:  # noqa: BLE001
    _tile_ext = None
    _tile_error = str(e)


def _tile_ext_build_config():
    """Build key/name/flags for tile CUDA extension, with optional maxrregcount tuning."""
    maxrreg_raw = os.environ.get("MLPSP_NVCC_MAXRREGCOUNT", "").strip()
    extra_cuda_cflags = []
    suffix = "default"
    if maxrreg_raw:
        try:
            maxrreg = int(maxrreg_raw)
            if maxrreg > 0:
                extra_cuda_cflags.append(f"-maxrregcount={maxrreg}")
                suffix = f"rreg{maxrreg}"
        except ValueError:
            # Ignore invalid values and fall back to default compile options.
            pass
    name = f"np_tile_kernels_{suffix}"
    key = (name, tuple(extra_cuda_cflags))
    return key, name, extra_cuda_cflags

class IntegratedPositionalEncoding(nn.Module):
    def __init__(self, in_dim=6, num_frequencies=10, sigma=0.01):
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        self.sigma2 = sigma ** 2
        freq_bands = 2. ** torch.linspace(0., num_frequencies - 1, num_frequencies) * torch.pi
        self.register_buffer("freq_bands", freq_bands)
        self.register_buffer("runtime_frequency_weights", torch.ones(num_frequencies), persistent=False)

    def set_runtime_frequency_weights(self, weights: torch.Tensor):
        with torch.no_grad():
            self.runtime_frequency_weights.copy_(weights.to(device=self.runtime_frequency_weights.device, dtype=self.runtime_frequency_weights.dtype))

    def forward(self, x):
        x_proj = x[..., None] * self.freq_bands
        attenuation = torch.exp(-0.5 * (self.freq_bands ** 2) * self.sigma2) * self.runtime_frequency_weights
        x_proj = x_proj * attenuation
        x_proj = x_proj.view(*x.shape[:-1], -1)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    @property
    def out_dim(self):
        return self.in_dim * self.num_frequencies * 2


class FourierFeatureEncoding(nn.Module):
    def __init__(self, in_dim=6, num_frequencies=10):
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        freq_bands = 2. ** torch.linspace(0., num_frequencies - 1, num_frequencies)
        self.register_buffer("freq_bands", freq_bands)
        self.register_buffer("runtime_frequency_weights", torch.ones(num_frequencies), persistent=False)

    def set_runtime_frequency_weights(self, weights: torch.Tensor):
        with torch.no_grad():
            self.runtime_frequency_weights.copy_(weights.to(device=self.runtime_frequency_weights.device, dtype=self.runtime_frequency_weights.dtype))

    def forward(self, x):
        x_proj = x[..., None] * self.freq_bands
        w = self.runtime_frequency_weights.view(*([1] * (x_proj.ndim - 1)), self.num_frequencies)
        sin_feat = (torch.sin(x_proj) * w).view(*x.shape[:-1], -1)
        cos_feat = (torch.cos(x_proj) * w).view(*x.shape[:-1], -1)
        return torch.cat([sin_feat, cos_feat], dim=-1)

    @property
    def out_dim(self):
        return self.in_dim * self.num_frequencies * 2


def _encode_x6(x6: torch.Tensor, freq: torch.Tensor, att: torch.Tensor) -> torch.Tensor:
    # x6: [M, 6], freq/att: [F]
    xproj = x6[:, :, None] * freq[None, None, :] * att[None, None, :]
    # Match CUDA layout used by mlp_eval_rgb_sigma:
    # [all sin (6*F), then all cos (6*F)].
    sin_blk = torch.sin(xproj).reshape(x6.shape[0], -1)
    cos_blk = torch.cos(xproj).reshape(x6.shape[0], -1)
    return torch.cat([sin_blk, cos_blk], dim=-1)


def _mlp_forward_indexed(
    x_enc: torch.Tensor,
    prim_ids: torch.Tensor,
    W1: torch.Tensor, b1: torch.Tensor,
    W2: torch.Tensor, b2: torch.Tensor,
    W3: torch.Tensor, b3: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Match CUDA Hcap behavior
    Hcap = min(int(W1.shape[1]), 64)
    W1s = W1.index_select(0, prim_ids)[:, :Hcap, :]
    b1s = b1.index_select(0, prim_ids)[:, :Hcap]
    W2s = W2.index_select(0, prim_ids)[:, :Hcap, :Hcap]
    b2s = b2.index_select(0, prim_ids)[:, :Hcap]
    W3s = W3.index_select(0, prim_ids)[:, :, :Hcap]
    b3s = b3.index_select(0, prim_ids)

    h1 = torch.einsum("mhi,mi->mh", W1s, x_enc) + b1s
    h1 = F.relu(h1)
    h2 = torch.einsum("moh,mh->mo", W2s, h1) + b2s
    h2 = F.relu(h2)
    o = torch.einsum("moh,mh->mo", W3s, h2) + b3s
    rgb = torch.sigmoid(o[:, :3])
    sigma_raw = F.softplus(o[:, 3:4])
    return rgb, sigma_raw


class _TileRenderAutogradFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        rays_o, rays_d,
        positions, log_scales, quats,
        culled_prim_indices, tile_offsets,
        freq, att,
        W1, b1, W2, b2, W3, b3,
        Wimg: int, Himg: int, TW: int, TH: int,
        threshold: float,
        mode: int,
        bg_color: torch.Tensor | None,
    ):
        ext = NeuralPrimitiveSceneTileCUDA._ensure_tile_ext_static()

        if mode == 0:
            rgb, alpha = ext.render_tiles_forward(
                rays_o, rays_d, positions, log_scales, quats,
                culled_prim_indices, tile_offsets,
                freq, att, W1, b1, W2, b2, W3, b3,
                int(Wimg), int(Himg), int(TW), int(TH), float(threshold)
            )
        elif mode == 1:
            rgb, alpha = ext.render_tiles_forward_tilecenter(
                rays_o, rays_d, positions, log_scales, quats,
                culled_prim_indices, tile_offsets,
                freq, att, W1, b1, W2, b2, W3, b3,
                int(Wimg), int(Himg), int(TW), int(TH), float(threshold)
            )
        else:
            raise ValueError("Unsupported tile mode for backward")

        if bg_color is not None:
            bg = bg_color
            if bg.numel() == 1:
                bg = bg.repeat(3)
            rgb = rgb + (1.0 - alpha).clamp(0.0, 1.0) * bg.view(1, 3)

        ctx.save_for_backward(
            rays_o, rays_d, positions, log_scales, quats,
            culled_prim_indices, tile_offsets,
            freq, att, W1, b1, W2, b2, W3, b3
        )
        ctx.Wimg = int(Wimg)
        ctx.Himg = int(Himg)
        ctx.TW = int(TW)
        ctx.TH = int(TH)
        ctx.threshold = float(threshold)
        ctx.mode = int(mode)
        ctx.bg_color = bg_color
        return rgb

    @staticmethod
    def backward(ctx, grad_out_rgb):
        (rays_o, rays_d, positions, log_scales, quats,
         culled_prim_indices, tile_offsets,
         freq, att, W1, b1, W2, b2, W3, b3) = ctx.saved_tensors


        grad_out_rgb = grad_out_rgb.contiguous()
        B = rays_o.shape[0]
        if ctx.bg_color is None:
            grad_T_end = torch.zeros(B, device=grad_out_rgb.device, dtype=grad_out_rgb.dtype)
        else:
            bg = ctx.bg_color
            if not torch.is_tensor(bg):
                bg = torch.tensor(bg, dtype=grad_out_rgb.dtype, device=grad_out_rgb.device)
            else:
                bg = bg.to(device=grad_out_rgb.device, dtype=grad_out_rgb.dtype)
            if bg.numel() == 1:
                bg = bg.repeat(3)
            grad_T_end = (grad_out_rgb * bg.view(1, 3)).sum(dim=-1)

        ext = NeuralPrimitiveSceneTileCUDA._ensure_tile_ext_static()
        chunk_size = int(os.environ.get("MLPSP_BWD_CHUNK", "65536"))

        if ctx.mode == 0:
            grad_quats = torch.zeros_like(quats)
            counts = ext.render_tiles_backward_count_per_pixel(
                rays_o, rays_d, positions, log_scales, quats,
                culled_prim_indices, tile_offsets,
                freq, att, W1, b1, W2, b2, W3, b3,
                ctx.Wimg, ctx.Himg, ctx.TW, ctx.TH, ctx.threshold
            )
            offsets = torch.zeros_like(counts)
            if counts.numel() > 1:
                offsets[1:] = torch.cumsum(counts[:-1], dim=0)
            total = int(offsets[-1].item() + counts[-1].item()) if counts.numel() > 0 else 0
            if total == 0:
                grad_positions = torch.zeros_like(positions)
                grad_log_scales = torch.zeros_like(log_scales)
                grad_quats = torch.zeros_like(quats)
                grad_W1 = torch.zeros_like(W1)
                grad_b1 = torch.zeros_like(b1)
                grad_W2 = torch.zeros_like(W2)
                grad_b2 = torch.zeros_like(b2)
                grad_W3 = torch.zeros_like(W3)
                grad_b3 = torch.zeros_like(b3)
            else:
                (prim_id, x6, grad_rgb, grad_sigma_raw,
                 grad_positions, grad_log_scales, grad_quats) = ext.render_tiles_backward_emit_per_pixel(
                    rays_o, rays_d, positions, log_scales, quats,
                    culled_prim_indices, tile_offsets,
                    freq, att, W1, b1, W2, b2, W3, b3,
                    grad_out_rgb, grad_T_end,
                    counts,
                    offsets, total,
                    ctx.Wimg, ctx.Himg, ctx.TW, ctx.TH, ctx.threshold
                )
                grad_W1 = torch.zeros_like(W1)
                grad_b1 = torch.zeros_like(b1)
                grad_W2 = torch.zeros_like(W2)
                grad_b2 = torch.zeros_like(b2)
                grad_W3 = torch.zeros_like(W3)
                grad_b3 = torch.zeros_like(b3)
                with torch.enable_grad():
                    for start in range(0, total, chunk_size):
                        end = min(total, start + chunk_size)
                        prim_id_chunk = prim_id[start:end].to(torch.long)
                        x6_chunk = x6[start:end]
                        grad_rgb_chunk = grad_rgb[start:end]
                        grad_sigma_chunk = grad_sigma_raw[start:end]
                        x_enc = _encode_x6(x6_chunk, freq, att).detach()
                        rgb, sigma_raw = _mlp_forward_indexed(x_enc, prim_id_chunk, W1, b1, W2, b2, W3, b3)
                        gW1, gb1, gW2, gb2, gW3, gb3 = torch.autograd.grad(
                            outputs=[rgb, sigma_raw],
                            inputs=[W1, b1, W2, b2, W3, b3],
                            grad_outputs=[grad_rgb_chunk, grad_sigma_chunk],
                            retain_graph=False,
                            allow_unused=False,
                        )
                        grad_W1.add_(gW1)
                        grad_b1.add_(gb1)
                        grad_W2.add_(gW2)
                        grad_b2.add_(gb2)
                        grad_W3.add_(gW3)
                        grad_b3.add_(gb3)
        elif ctx.mode == 1:
            (grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3,
             grad_positions, grad_log_scales, grad_quats) = ext.render_tiles_backward_tilecenter_full(
                rays_o, rays_d, positions, log_scales, quats,
                culled_prim_indices, tile_offsets,
                freq, att, W1, b1, W2, b2, W3, b3,
                grad_out_rgb, grad_T_end,
                ctx.Wimg, ctx.Himg, ctx.TW, ctx.TH, ctx.threshold
            )
        else:
            raise ValueError("Unsupported tile mode for backward")

        return (
            None, None,  # rays_o, rays_d
            grad_positions, grad_log_scales, grad_quats,
            None, None,  # culled_prim_indices, tile_offsets
            None, None,  # freq, att
            grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3,
            None, None, None, None, None,  # Wimg, Himg, TW, TH, threshold
            None, None,  # mode, bg_color

        )


class _TileRenderWithFeatAutogradFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        rays_o, rays_d,
        positions, log_scales, quats,
        culled_prim_indices, tile_offsets,
        freq, att,
        W1, b1, W2, b2, W3, b3,
        feat_attr,
        gating_beta: float,
        gating_temp: float,
        Wimg: int, Himg: int, TW: int, TH: int,
        threshold: float,
        bg_color: torch.Tensor | None,
    ):
        ext = NeuralPrimitiveSceneTileCUDA._ensure_tile_ext_static()
        rgb, alpha, feat = ext.render_tiles_forward_tilecenter_with_feat(
            rays_o, rays_d, positions, log_scales, quats,
            culled_prim_indices, tile_offsets,
            freq, att,
            W1, b1, W2, b2, W3, b3,
            feat_attr,
            float(gating_beta), float(gating_temp),
            int(Wimg), int(Himg), int(TW), int(TH), float(threshold)
        )

        if bg_color is not None:
            bg = bg_color
            if bg.numel() == 1:
                bg = bg.repeat(3)
            rgb = rgb + (1.0 - alpha).clamp(0.0, 1.0) * bg.view(1, 3)

        ctx.save_for_backward(
            rays_o, rays_d, positions, log_scales, quats,
            culled_prim_indices, tile_offsets,
            freq, att, W1, b1, W2, b2, W3, b3, feat_attr
        )
        ctx.Wimg = int(Wimg)
        ctx.Himg = int(Himg)
        ctx.TW = int(TW)
        ctx.TH = int(TH)
        ctx.threshold = float(threshold)
        ctx.bg_color = bg_color
        ctx.gating_beta = float(gating_beta)
        ctx.gating_temp = float(gating_temp)
        return rgb, feat

    @staticmethod
    def backward(ctx, grad_out_rgb, grad_out_feat):
        (rays_o, rays_d, positions, log_scales, quats,
         culled_prim_indices, tile_offsets,
         freq, att, W1, b1, W2, b2, W3, b3, feat_attr) = ctx.saved_tensors

        ext = NeuralPrimitiveSceneTileCUDA._ensure_tile_ext_static()
        B = rays_o.shape[0]

        grad_positions = torch.zeros_like(positions)
        grad_log_scales = torch.zeros_like(log_scales)
        grad_quats = torch.zeros_like(quats)
        grad_W1 = torch.zeros_like(W1)
        grad_b1 = torch.zeros_like(b1)
        grad_W2 = torch.zeros_like(W2)
        grad_b2 = torch.zeros_like(b2)
        grad_W3 = torch.zeros_like(W3)
        grad_b3 = torch.zeros_like(b3)
        grad_feat_attr = None

        if grad_out_rgb is not None:
            grad_out_rgb = grad_out_rgb.contiguous()
            if ctx.bg_color is None:
                grad_T_end = torch.zeros(B, device=grad_out_rgb.device, dtype=grad_out_rgb.dtype)
            else:
                bg = ctx.bg_color
                if not torch.is_tensor(bg):
                    bg = torch.tensor(bg, dtype=grad_out_rgb.dtype, device=grad_out_rgb.device)
                else:
                    bg = bg.to(device=grad_out_rgb.device, dtype=grad_out_rgb.dtype)
                if bg.numel() == 1:
                    bg = bg.repeat(3)
                grad_T_end = (grad_out_rgb * bg.view(1, 3)).sum(dim=-1)
            (grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3,
             grad_positions, grad_log_scales, grad_quats) = ext.render_tiles_backward_tilecenter_full(
                rays_o, rays_d, positions, log_scales, quats,
                culled_prim_indices, tile_offsets,
                freq, att, W1, b1, W2, b2, W3, b3,
                grad_out_rgb, grad_T_end,
                ctx.Wimg, ctx.Himg, ctx.TW, ctx.TH, ctx.threshold
            )

        if grad_out_feat is not None:
            grad_out_feat = grad_out_feat.contiguous()
            grad_feat_attr = ext.render_tiles_backward_featattr_tilecenter(
                rays_o, rays_d, positions, log_scales, quats,
                culled_prim_indices, tile_offsets,
                freq, att, W1, b1, W2, b2, W3, b3,
                feat_attr,
                grad_out_feat,
                float(ctx.gating_beta), float(ctx.gating_temp),
                ctx.Wimg, ctx.Himg, ctx.TW, ctx.TH, ctx.threshold
            )

        return (
            None, None,  # rays_o, rays_d
            grad_positions, grad_log_scales, grad_quats,
            None, None,  # culled_prim_indices, tile_offsets
            None, None,  # freq, att
            grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3,
            grad_feat_attr,
            None, None,  # gating_beta, gating_temp
            None, None, None, None, None,  # Wimg, Himg, TW, TH, threshold
            None,  # bg_color
        )


class _TileRenderWithFeatPerPixelAutogradFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        rays_o, rays_d,
        positions, log_scales, quats,
        culled_prim_indices, tile_offsets,
        freq, att,
        W1, b1, W2, b2, W3, b3,
        feat_attr,
        gating_beta: float,
        gating_temp: float,
        Wimg: int, Himg: int, TW: int, TH: int,
        threshold: float,
        bg_color: torch.Tensor | None,
    ):
        ext = NeuralPrimitiveSceneTileCUDA._ensure_tile_ext_static()
        rgb, alpha, feat = ext.render_tiles_forward_per_pixel_with_feat(
            rays_o, rays_d, positions, log_scales, quats,
            culled_prim_indices, tile_offsets,
            freq, att,
            W1, b1, W2, b2, W3, b3,
            feat_attr,
            float(gating_beta), float(gating_temp),
            int(Wimg), int(Himg), int(TW), int(TH), float(threshold)
        )

        if bg_color is not None:
            bg = bg_color
            if bg.numel() == 1:
                bg = bg.repeat(3)
            rgb = rgb + (1.0 - alpha).clamp(0.0, 1.0) * bg.view(1, 3)

        ctx.save_for_backward(
            rays_o, rays_d, positions, log_scales, quats,
            culled_prim_indices, tile_offsets,
            freq, att, W1, b1, W2, b2, W3, b3, feat_attr
        )
        ctx.Wimg = int(Wimg)
        ctx.Himg = int(Himg)
        ctx.TW = int(TW)
        ctx.TH = int(TH)
        ctx.threshold = float(threshold)
        ctx.bg_color = bg_color
        ctx.gating_beta = float(gating_beta)
        ctx.gating_temp = float(gating_temp)
        return rgb, feat

    @staticmethod
    def backward(ctx, grad_out_rgb, grad_out_feat):
        (rays_o, rays_d, positions, log_scales, quats,
         culled_prim_indices, tile_offsets,
         freq, att, W1, b1, W2, b2, W3, b3, feat_attr) = ctx.saved_tensors

        ext = NeuralPrimitiveSceneTileCUDA._ensure_tile_ext_static()
        B = rays_o.shape[0]

        grad_positions = torch.zeros_like(positions)
        grad_log_scales = torch.zeros_like(log_scales)
        grad_quats = torch.zeros_like(quats)
        grad_W1 = torch.zeros_like(W1)
        grad_b1 = torch.zeros_like(b1)
        grad_W2 = torch.zeros_like(W2)
        grad_b2 = torch.zeros_like(b2)
        grad_W3 = torch.zeros_like(W3)
        grad_b3 = torch.zeros_like(b3)
        grad_feat_attr = None

        if grad_out_rgb is not None:
            grad_out_rgb = grad_out_rgb.contiguous()
            if ctx.bg_color is None:
                grad_T_end = torch.zeros(B, device=grad_out_rgb.device, dtype=grad_out_rgb.dtype)
            else:
                bg = ctx.bg_color
                if not torch.is_tensor(bg):
                    bg = torch.tensor(bg, dtype=grad_out_rgb.dtype, device=grad_out_rgb.device)
                else:
                    bg = bg.to(device=grad_out_rgb.device, dtype=grad_out_rgb.dtype)
                if bg.numel() == 1:
                    bg = bg.repeat(3)
                grad_T_end = (grad_out_rgb * bg.view(1, 3)).sum(dim=-1)

            counts = ext.render_tiles_backward_count_per_pixel(
                rays_o, rays_d, positions, log_scales, quats,
                culled_prim_indices, tile_offsets,
                freq, att, W1, b1, W2, b2, W3, b3,
                ctx.Wimg, ctx.Himg, ctx.TW, ctx.TH, ctx.threshold
            )
            offsets = torch.zeros_like(counts)
            if counts.numel() > 1:
                offsets[1:] = torch.cumsum(counts[:-1], dim=0)
            total = int(offsets[-1].item() + counts[-1].item()) if counts.numel() > 0 else 0
            if total == 0:
                grad_positions = torch.zeros_like(positions)
                grad_log_scales = torch.zeros_like(log_scales)
                grad_quats = torch.zeros_like(quats)
                grad_W1 = torch.zeros_like(W1)
                grad_b1 = torch.zeros_like(b1)
                grad_W2 = torch.zeros_like(W2)
                grad_b2 = torch.zeros_like(b2)
                grad_W3 = torch.zeros_like(W3)
                grad_b3 = torch.zeros_like(b3)
            else:
                (prim_id, x6, grad_rgb, grad_sigma_raw,
                 grad_positions, grad_log_scales, grad_quats) = ext.render_tiles_backward_emit_per_pixel(
                    rays_o, rays_d, positions, log_scales, quats,
                    culled_prim_indices, tile_offsets,
                    freq, att, W1, b1, W2, b2, W3, b3,
                    grad_out_rgb, grad_T_end,
                    counts,
                    offsets, total,
                    ctx.Wimg, ctx.Himg, ctx.TW, ctx.TH, ctx.threshold
                )
                grad_W1 = torch.zeros_like(W1)
                grad_b1 = torch.zeros_like(b1)
                grad_W2 = torch.zeros_like(W2)
                grad_b2 = torch.zeros_like(b2)
                grad_W3 = torch.zeros_like(W3)
                grad_b3 = torch.zeros_like(b3)
                chunk_size = int(os.environ.get("MLPSP_BWD_CHUNK", "65536"))
                with torch.enable_grad():
                    for start in range(0, total, chunk_size):
                        end = min(total, start + chunk_size)
                        prim_id_chunk = prim_id[start:end].to(torch.long)
                        x6_chunk = x6[start:end]
                        grad_rgb_chunk = grad_rgb[start:end]
                        grad_sigma_chunk = grad_sigma_raw[start:end]
                        x_enc = _encode_x6(x6_chunk, freq, att).detach()
                        rgb, sigma_raw = _mlp_forward_indexed(x_enc, prim_id_chunk, W1, b1, W2, b2, W3, b3)
                        gW1, gb1, gW2, gb2, gW3, gb3 = torch.autograd.grad(
                            outputs=[rgb, sigma_raw],
                            inputs=[W1, b1, W2, b2, W3, b3],
                            grad_outputs=[grad_rgb_chunk, grad_sigma_chunk],
                            retain_graph=False,
                            allow_unused=False,
                        )
                        grad_W1.add_(gW1)
                        grad_b1.add_(gb1)
                        grad_W2.add_(gW2)
                        grad_b2.add_(gb2)
                        grad_W3.add_(gW3)
                        grad_b3.add_(gb3)

        if grad_out_feat is not None:
            grad_out_feat = grad_out_feat.contiguous()
            grad_feat_attr = ext.render_tiles_backward_featattr_per_pixel(
                rays_o, rays_d, positions, log_scales, quats,
                culled_prim_indices, tile_offsets,
                freq, att, W1, b1, W2, b2, W3, b3,
                feat_attr,
                grad_out_feat,
                float(ctx.gating_beta), float(ctx.gating_temp),
                ctx.Wimg, ctx.Himg, ctx.TW, ctx.TH, ctx.threshold
            )

        return (
            None, None,  # rays_o, rays_d
            grad_positions, grad_log_scales, grad_quats,
            None, None,  # culled_prim_indices, tile_offsets
            None, None,  # freq, att
            grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3,
            grad_feat_attr,
            None, None,  # gating_beta, gating_temp
            None, None, None, None, None,  # Wimg, Himg, TW, TH, threshold
            None,  # bg_color
        )

class NeuralPrimitiveSceneTileCUDA(nn.Module):
    """Tile-based single-pass MLP splatting renderer.

    - Requires full-frame rays (B == W*H); call set_image_size first.
    - One CUDA kernel pass per frame: iterate depth-ordered primitives per tile,
      evaluate its pixels in parallel, and alpha blend with early termination.
    """

    requires_full_image = True

    def __init__(
        self,
        num_primitives=1024,
        hidden_dim=32,
        pe_frequencies=6,
        initial_positions=None,
        use_integrated_pe=True,
        pe_sigma: float = 0.01,
        min_points: int = 10,
        tile_size: int = 8,
        feature_dim_low: int = 0,
        feature_slots: int = 4,
        feature_gating_beta: float = 1.5,
        feature_gating_temp: float = 2.0,
    ):
        super().__init__()
        if int(hidden_dim) > 32:
            raise ValueError(
                "NeuralPrimitiveSceneTileCUDA currently supports hidden_dim <= 32 "
                "(kernel scratch-cap tuning in backward/full path)."
            )
        self.num_primitives = num_primitives
        self.hidden_dim = hidden_dim
        self.pe_frequencies = pe_frequencies
        self.use_integrated_pe = bool(use_integrated_pe)
        self.pe_sigma = float(pe_sigma)
        if self.pe_sigma <= 0.0:
            raise ValueError("pe_sigma must be > 0.")
        self.image_size = None  # set by renderer
        self.tile_size = int(tile_size)
        if self.tile_size <= 0:
            raise ValueError("tile_size must be a positive integer.")
        if self.tile_size not in (8, 16):
            raise ValueError("NeuralPrimitiveSceneTileCUDA supports tile_size in {8, 16}.")
        # Runtime override for tile ordering mode (falls back to MLPSP_TILE_ORDER when None).
        self.tile_order_mode_override = None
        self.feature_dim_low = int(max(0, feature_dim_low))
        self.feature_slots = int(max(1, feature_slots))
        self.feature_gating_beta = float(feature_gating_beta)
        self.feature_gating_temp = float(feature_gating_temp)
        if (self.feature_dim_low > 0) and (self.feature_slots not in (4, 5)):
            raise ValueError("feature_slots must be 4 (canonical) or 5 (legacy) when features are enabled.")
        if self.feature_gating_beta <= 0.0:
            raise ValueError("feature_gating_beta must be > 0.")
        if self.feature_gating_temp <= 0.0:
            raise ValueError("feature_gating_temp must be > 0.")

        if initial_positions is None:
            bound = 1.0
            self.positions = nn.Parameter(torch.rand(num_primitives, 3) * 2 * bound - bound)
            sampled_scales = torch.ones(num_primitives) * 0.1
        else:
            sampled_positions, sampled_scales = octree_sample(initial_positions, min_points)
            print("Sampled", len(sampled_positions), "initial positions from point cloud's", len(initial_positions), "points")
            sampled_scales /= 10
            self.positions = nn.Parameter(sampled_positions.clone().detach())
            self.num_primitives = int(sampled_positions.shape[0])
        self.log_scales = nn.Parameter(torch.log(sampled_scales).unsqueeze(-1).repeat(1, 3))
        self.quats = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.positions.device, dtype=self.positions.dtype).view(1, 4).repeat(self.num_primitives, 1))
        # Keep encoder object to export freq bands
        if self.use_integrated_pe:
            self.encoder = IntegratedPositionalEncoding(
                in_dim=6,
                num_frequencies=pe_frequencies,
                sigma=self.pe_sigma,
            )
        else:
            self.encoder = FourierFeatureEncoding(in_dim=6, num_frequencies=pe_frequencies)

        # Per-primitive MLP weights (float32 for simplicity)
        IN = self.encoder.out_dim
        H = hidden_dim
        OUT = 4
        self.W1 = nn.Parameter(torch.randn(self.num_primitives, H, IN) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(self.num_primitives, H))
        self.W2 = nn.Parameter(torch.randn(self.num_primitives, H, H) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(self.num_primitives, H))
        self.W3 = nn.Parameter(torch.randn(self.num_primitives, OUT, H) * 0.01)
        self.b3 = nn.Parameter(torch.zeros(self.num_primitives, OUT))
        if self.feature_dim_low > 0:
            if self.feature_slots == 5:
                feat_init = torch.zeros(self.num_primitives, self.feature_slots, self.feature_dim_low)
                feat_init[:, 0, :].normal_(mean=0.0, std=0.01)  # base slot; deltas start from 0
            else:
                feat_init = torch.randn(self.num_primitives, self.feature_slots, self.feature_dim_low) * 0.01
            self.feat_attr = nn.Parameter(feat_init)
        else:
            self.feat_attr = None

        self.register_buffer("stat_sum_w", torch.zeros(self.num_primitives, dtype=torch.float32))
        self.register_buffer("stat_max_w", torch.zeros(self.num_primitives, dtype=torch.float32))
        self.register_buffer("stat_hits",  torch.zeros(self.num_primitives, dtype=torch.int32))
        self.register_buffer("stat_contrib_count",  torch.zeros(self.num_primitives, dtype=torch.int32))
        self.register_buffer("stat_sum_werr",  torch.zeros(self.num_primitives, dtype=torch.float32))


        # Reserved for pruning/statistics workflows; keep accumulation in float32.
        self.register_buffer("alpha_accum", torch.zeros(self.num_primitives, dtype=torch.float32))

    @torch.no_grad()
    def reset_w_stats(self):
        """Zero the per-primitive w-stat buffers."""
        N = self.positions.shape[0]
        dev = self.positions.device

        self.stat_sum_w = torch.zeros(N, device=dev, dtype=torch.float32)
        self.stat_hits = torch.zeros(N, device=dev, dtype=torch.int32)
        self.stat_max_w = torch.zeros(N, device=dev, dtype=torch.float32)
        self.stat_sum_werr = torch.zeros(N, device=dev, dtype=torch.float32)
        self.stat_contrib_count = torch.zeros(N, device=dev, dtype=torch.int32)


    def set_image_size(self, W: int, H: int):
        self.image_size = (int(W), int(H))

    def _activate_cuda_device(self) -> torch.device:
        """Select the model's device before dispatching a custom CUDA kernel."""
        device = self.positions.device
        if device.type != "cuda":
            raise RuntimeError(
                "NeuralPrimitiveSceneTileCUDA must be moved to a CUDA device before rendering."
            )
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device.index)
        return device

    def _resolve_tile_order_mode(self) -> str:
        mode = self.tile_order_mode_override
        if mode is None or str(mode).strip() == "":
            mode = os.environ.get("MLPSP_TILE_ORDER", "tilecenter")
        mode = str(mode).strip().lower()
        aliases = {"tile_center": "tilecenter", "tile-center": "tilecenter"}
        mode = aliases.get(mode, mode)
        if mode not in {"tilecenter", "per_pixel"}:
            raise ValueError(
                f"Unsupported tile order {mode!r}; use 'tilecenter' or 'per_pixel'."
            )
        return mode

    def _ensure_tile_ext(self):
        return self._ensure_tile_ext_static()

    @staticmethod
    def _ensure_tile_ext_static():
        global _tile_ext, _tile_ext_key
        if _tile_ext is None:
            if _tile_error is not None and _tile_error.startswith('Tile CUDA source missing'):
                raise RuntimeError(_tile_error)
            if '_tile_src_path' not in globals() or _tile_src_path is None:
                raise RuntimeError(f"Tile CUDA not available: {_tile_error}")
            key, name, extra_cuda_cflags = _tile_ext_build_config()
            _tile_ext = _safe_load_extension(
                load_ext,
                name=name,
                sources=[_tile_src_path],
                verbose=False,
                extra_cuda_cflags=extra_cuda_cflags,
            )
            _tile_ext_key = key
        else:
            key, name, extra_cuda_cflags = _tile_ext_build_config()
            if _tile_ext_key != key:
                _tile_ext = _safe_load_extension(
                    load_ext,
                    name=name,
                    sources=[_tile_src_path],
                    verbose=False,
                    extra_cuda_cflags=extra_cuda_cflags,
                )
                _tile_ext_key = key
        return _tile_ext

    @torch.no_grad()
    def build_tilecenter_culling_from_tile_views(
        self,
        tile_view_ids: torch.Tensor,
        view_matrices: torch.Tensor,
        proj_matrix: torch.Tensor,
        score_threshold: float = 1e-3,
        tile_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build tile->primitive sparse lists for tilecenter mode using per-tile view assignment."""
        if self.image_size is None:
            raise ValueError("image_size must be set before building tile culling.")
        W, H = self.image_size
        device = self._activate_cuda_device()
        tile_size = int(self.tile_size if tile_size is None else tile_size)
        if tile_size <= 0:
            raise ValueError("tile_size must be a positive integer.")
        num_tiles_x = (W + tile_size - 1) // tile_size
        num_tiles_y = (H + tile_size - 1) // tile_size
        num_tiles = num_tiles_x * num_tiles_y

        tile_view_ids = tile_view_ids.to(device=device, dtype=torch.long).reshape(-1)
        if tile_view_ids.numel() != num_tiles:
            raise ValueError(f"tile_view_ids length mismatch: got {tile_view_ids.numel()}, expected {num_tiles}")
        if view_matrices.ndim != 3 or view_matrices.shape[1:] != (4, 4):
            raise ValueError("view_matrices must have shape (K,4,4)")
        if int(view_matrices.shape[0]) <= 0:
            raise ValueError("view_matrices must contain at least one view")
        if int(tile_view_ids.min().item()) < 0 or int(tile_view_ids.max().item()) >= int(view_matrices.shape[0]):
            raise ValueError("tile_view_ids contains out-of-range view indices")

        tile_indices = torch.arange(num_tiles, device=device)
        tile_x_coords = tile_indices % num_tiles_x
        tile_y_coords = tile_indices // num_tiles_x
        tile_min_px_x_all = tile_x_coords * tile_size
        tile_max_px_x_all = (tile_x_coords + 1) * tile_size
        tile_min_px_y_all = tile_y_coords * tile_size
        tile_max_px_y_all = (tile_y_coords + 1) * tile_size

        N = self.num_primitives
        pos_world_h = torch.cat([self.positions, torch.ones(N, 1, device=device)], dim=-1)
        scales = torch.exp(self.log_scales).amax(dim=-1)
        focal_x = proj_matrix[0, 0] * W / 2.0
        focal_y = proj_matrix[1, 1] * H / 2.0
        focal = (focal_x + focal_y) * 0.5
        radius_multiplier = torch.sqrt(-2.0 * torch.log(torch.tensor(score_threshold, device=device)))
        world_radius = scales * radius_multiplier

        pair_tile_ids = []
        pair_prim_ids = []
        unique_views = torch.unique(tile_view_ids, sorted=False)

        for uv in unique_views.tolist():
            view_matrix = view_matrices[int(uv)].to(device=device, dtype=torch.float32)
            tile_ids = torch.nonzero(tile_view_ids == int(uv), as_tuple=False).squeeze(-1)
            if tile_ids.numel() == 0:
                continue

            pos_view_h = pos_world_h @ view_matrix.T
            depth = pos_view_h[:, 2]
            near_plane = 0.001
            in_front = depth > near_plane

            pixel_radius = world_radius / (depth + 1e-8) * focal
            pos_clip = pos_view_h @ proj_matrix.T
            w = pos_clip[..., 3:4]
            w = w + 1e-6 * (w == 0)
            pos_ndc = pos_clip[..., :3] / w
            pos_screen_x = (pos_ndc[:, 0] + 1) * 0.5 * W
            pos_screen_y = (pos_ndc[:, 1] + 1) * 0.5 * H

            prim_min_px_x = (pos_screen_x - pixel_radius).unsqueeze(0)
            prim_max_px_x = (pos_screen_x + pixel_radius).unsqueeze(0)
            prim_min_px_y = (pos_screen_y - pixel_radius).unsqueeze(0)
            prim_max_px_y = (pos_screen_y + pixel_radius).unsqueeze(0)

            tile_min_px_x = tile_min_px_x_all.index_select(0, tile_ids).unsqueeze(1)
            tile_max_px_x = tile_max_px_x_all.index_select(0, tile_ids).unsqueeze(1)
            tile_min_px_y = tile_min_px_y_all.index_select(0, tile_ids).unsqueeze(1)
            tile_max_px_y = tile_max_px_y_all.index_select(0, tile_ids).unsqueeze(1)

            overlap_x = (prim_min_px_x < tile_max_px_x) & (prim_max_px_x > tile_min_px_x)
            overlap_y = (prim_min_px_y < tile_max_px_y) & (prim_max_px_y > tile_min_px_y)
            overlap = overlap_x & overlap_y & in_front.unsqueeze(0)

            local_pairs = overlap.nonzero(as_tuple=False)
            if local_pairs.numel() == 0:
                continue
            pair_tile_ids.append(tile_ids[local_pairs[:, 0]])
            pair_prim_ids.append(local_pairs[:, 1].to(torch.int32))

        if len(pair_tile_ids) == 0:
            return (
                torch.empty(0, dtype=torch.int32, device=device),
                torch.zeros(num_tiles + 1, dtype=torch.int32, device=device),
            )

        tile_ids_all = torch.cat(pair_tile_ids, dim=0)
        prim_ids_all = torch.cat(pair_prim_ids, dim=0)
        sorted_indices = torch.argsort(tile_ids_all)
        sorted_tile_ids = tile_ids_all[sorted_indices]
        culled_prim_indices = prim_ids_all[sorted_indices]

        tile_offsets = torch.zeros(num_tiles + 1, dtype=torch.long, device=device)
        unique_tile_ids, counts = torch.unique_consecutive(sorted_tile_ids, return_counts=True)
        tile_offsets.scatter_add_(0, unique_tile_ids + 1, counts)
        tile_offsets = torch.cumsum(tile_offsets, dim=0).to(torch.int32)
        return culled_prim_indices, tile_offsets

    def forward(
        self,
        ray_o,
        ray_d,
        score_threshold=1e-3,
        bg_color=None,
        view_matrix=None,
        proj_matrix=None,
        culled_prim_indices=None,
        tile_offsets=None,
        return_feature: bool = False,
    ):
        if self.image_size is None:
            raise ValueError("image_size must be set on NeuralPrimitiveSceneTileCUDA before calling forward.")

        W, H = self.image_size
        B = ray_o.shape[0]
        assert B == W * H, "Tile renderer requires full-frame rays (B == W*H)"
        device = self._activate_cuda_device()
        tile_size = int(self.tile_size)
        num_tiles_x = (W + tile_size - 1) // tile_size
        num_tiles_y = (H + tile_size - 1) // tile_size
        num_tiles = num_tiles_x * num_tiles_y

        use_external_culling = (culled_prim_indices is not None) or (tile_offsets is not None)
        if use_external_culling:
            if culled_prim_indices is None or tile_offsets is None:
                raise ValueError("Both culled_prim_indices and tile_offsets must be provided together.")
            culled_prim_indices = culled_prim_indices.to(device=device, dtype=torch.int32)
            tile_offsets = tile_offsets.to(device=device, dtype=torch.int32)
            if tile_offsets.numel() != num_tiles + 1:
                raise ValueError(f"tile_offsets length mismatch: got {tile_offsets.numel()}, expected {num_tiles + 1}")
            depth = None
        else:
            if view_matrix is None or proj_matrix is None:
                raise ValueError("Tile renderer requires view_matrix/proj_matrix unless external culling is provided.")
            # --- 1. Tile Culling (3DGS-style) ---
            with torch.no_grad():
                view_proj_matrix = proj_matrix @ view_matrix
                N = self.num_primitives
                pos_world_h = torch.cat([self.positions, torch.ones(N, 1, device=device)], dim=-1)

                # Get positions in view space
                pos_view_h = pos_world_h @ view_matrix.T
                depth = pos_view_h[:, 2]
                near_plane = 0.001
                in_front = depth > near_plane

                # Get projected radius in pixels
                scales = torch.exp(self.log_scales).amax(dim=-1)
                focal_x = proj_matrix[0, 0] * W / 2.0
                focal_y = proj_matrix[1, 1] * H / 2.0
                focal = (focal_x + focal_y) * 0.5

                # Derive radius from score threshold, similar to the Triton kernel logic
                # sc = exp(-0.5 * perp2 / s^2) > score_threshold
                # -> perp2 < -2 * s^2 * log(score_threshold)
                # -> perp < s * sqrt(-2 * log(score_threshold))
                radius_multiplier = torch.sqrt(-2.0 * torch.log(torch.tensor(score_threshold, device=device)))
                world_radius = scales * radius_multiplier
                pixel_radius = world_radius / (depth + 1e-8) * focal

                # Project centers to screen
                pos_clip = pos_view_h @ proj_matrix.T
                w = pos_clip[..., 3:4]
                w = w + 1e-6 * (w == 0)
                pos_ndc = pos_clip[..., :3] / w
                pos_screen_x = (pos_ndc[:, 0] + 1) * 0.5 * W
                pos_screen_y = (pos_ndc[:, 1] + 1) * 0.5 * H

                # Vectorized tile-primitive overlap check
                tile_indices = torch.arange(num_tiles, device=device)
                tile_x_coords = tile_indices % num_tiles_x
                tile_y_coords = tile_indices // num_tiles_x

                tile_min_px_x = (tile_x_coords * tile_size).unsqueeze(1)
                tile_max_px_x = ((tile_x_coords + 1) * tile_size).unsqueeze(1)
                tile_min_px_y = (tile_y_coords * tile_size).unsqueeze(1)
                tile_max_px_y = ((tile_y_coords + 1) * tile_size).unsqueeze(1)

                prim_min_px_x = (pos_screen_x - pixel_radius).unsqueeze(0)
                prim_max_px_x = (pos_screen_x + pixel_radius).unsqueeze(0)
                prim_min_px_y = (pos_screen_y - pixel_radius).unsqueeze(0)
                prim_max_px_y = (pos_screen_y + pixel_radius).unsqueeze(0)

                overlap_x = (prim_min_px_x < tile_max_px_x) & (prim_max_px_x > tile_min_px_x)
                overlap_y = (prim_min_px_y < tile_max_px_y) & (prim_max_px_y > tile_min_px_y)

                overlap_matrix = overlap_x & overlap_y & in_front.unsqueeze(0)

            # Convert overlap matrix to sorted sparse format for kernel
            # Per-pixel ordering is the quality reference; tile-center ordering
            # shares one approximate front-to-back order across each tile.
            tile_order_mode = self._resolve_tile_order_mode()
            use_tilecenter = tile_order_mode == "tilecenter"

            tile_prim_pairs = overlap_matrix.nonzero(as_tuple=False)
            if tile_prim_pairs.numel() == 0:
                # Handle case with no intersections
                culled_prim_indices = torch.empty(0, dtype=torch.int32, device=device)
                tile_offsets = torch.zeros(num_tiles + 1, dtype=torch.int32, device=device)
            else:
                sorted_indices = torch.argsort(tile_prim_pairs[:, 0])
                sorted_pairs = tile_prim_pairs[sorted_indices]

                tile_ids = sorted_pairs[:, 0]
                culled_prim_indices = sorted_pairs[:, 1].to(torch.int32)

                # Build offsets array
                tile_offsets = torch.zeros(num_tiles + 1, dtype=torch.long, device=device)
                unique_tile_ids, counts = torch.unique_consecutive(tile_ids, return_counts=True)
                tile_offsets.scatter_add_(0, unique_tile_ids + 1, counts)
                tile_offsets = torch.cumsum(tile_offsets, dim=0).to(torch.int32)

        tile_order_mode = self._resolve_tile_order_mode()
        use_tilecenter = tile_order_mode == "tilecenter"

        # --- 2. Kernel Call ---
        ext = self._ensure_tile_ext()

        if isinstance(self.encoder, IntegratedPositionalEncoding):
            freq = self.encoder.freq_bands.to(device=device, dtype=torch.float32)
            sigma2 = float(self.encoder.sigma2)
            att = torch.exp(-0.5 * (freq * freq) * sigma2)
        else:
            freq = self.encoder.freq_bands.to(device=device, dtype=torch.float32)
            att = torch.ones_like(freq)
        if hasattr(self.encoder, "runtime_frequency_weights"):
            att = att * self.encoder.runtime_frequency_weights.to(device=device, dtype=torch.float32)

        ray_d_n = F.normalize(ray_d, dim=-1)

        if bg_color is not None:
            if not torch.is_tensor(bg_color):
                bg = torch.tensor(bg_color, dtype=ray_o.dtype, device=ray_o.device)
            else:
                bg = bg_color.to(ray_o.device, dtype=ray_o.dtype)
        else:
            bg = None

        if (not return_feature) or (self.feat_attr is None):
            mode = 1 if use_tilecenter else 0
            rgb = _TileRenderAutogradFn.apply(
                ray_o.contiguous(), ray_d_n.contiguous(),
                self.positions.contiguous(), self.log_scales.contiguous(), F.normalize(self.quats, dim=-1).contiguous(),
                culled_prim_indices, tile_offsets.to(torch.int32),
                freq.contiguous(), att.contiguous(),
                self.W1.contiguous(), self.b1.contiguous(),
                self.W2.contiguous(), self.b2.contiguous(),
                self.W3.contiguous(), self.b3.contiguous(),
                int(W), int(H), tile_size, tile_size,
                float(score_threshold),
                int(mode),
                bg
            )
            if return_feature and self.feat_attr is None:
                raise RuntimeError("return_feature=True requested but feat_attr is not initialized.")
            return rgb
        if use_tilecenter:
            rgb, feat_low = _TileRenderWithFeatAutogradFn.apply(
                ray_o.contiguous(), ray_d_n.contiguous(),
                self.positions.contiguous(), self.log_scales.contiguous(), F.normalize(self.quats, dim=-1).contiguous(),
                culled_prim_indices, tile_offsets.to(torch.int32),
                freq.contiguous(), att.contiguous(),
                self.W1.contiguous(), self.b1.contiguous(),
                self.W2.contiguous(), self.b2.contiguous(),
                self.W3.contiguous(), self.b3.contiguous(),
                self.feat_attr.contiguous(),
                float(self.feature_gating_beta),
                float(self.feature_gating_temp),
                int(W), int(H), tile_size, tile_size,
                float(score_threshold),
                bg,
            )
        else:
            rgb, feat_low = _TileRenderWithFeatPerPixelAutogradFn.apply(
                ray_o.contiguous(), ray_d_n.contiguous(),
                self.positions.contiguous(), self.log_scales.contiguous(), F.normalize(self.quats, dim=-1).contiguous(),
                culled_prim_indices, tile_offsets.to(torch.int32),
                freq.contiguous(), att.contiguous(),
                self.W1.contiguous(), self.b1.contiguous(),
                self.W2.contiguous(), self.b2.contiguous(),
                self.W3.contiguous(), self.b3.contiguous(),
                self.feat_attr.contiguous(),
                float(self.feature_gating_beta),
                float(self.feature_gating_temp),
                int(W), int(H), tile_size, tile_size,
                float(score_threshold),
                bg,
            )
        return rgb, feat_low

    def render_feature_map(
        self,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
        score_threshold: float = 1e-3,
        view_matrix: torch.Tensor | None = None,
        proj_matrix: torch.Tensor | None = None,
        culled_prim_indices: torch.Tensor | None = None,
        tile_offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Render low-dim feature map via the active tile ordering mode."""
        if self.feat_attr is None:
            raise RuntimeError("Feature rendering requested but feature_dim_low == 0.")
        if self.image_size is None:
            raise ValueError("image_size must be set before calling render_feature_map.")
        W, H = self.image_size
        B = ray_o.shape[0]
        if B != W * H:
            raise ValueError("render_feature_map requires full-frame rays (B == W*H).")
        # Reuse forward path to build/validate culling and execute fused feature kernel.
        _rgb, feat = self.forward(
            ray_o=ray_o,
            ray_d=ray_d,
            score_threshold=score_threshold,
            bg_color=None,
            view_matrix=view_matrix,
            proj_matrix=proj_matrix,
            culled_prim_indices=culled_prim_indices,
            tile_offsets=tile_offsets,
            return_feature=True,
        )
        return feat


    @torch.no_grad()
    def render_rgb_alpha_no_grad(
        self,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
        score_threshold: float = 1e-3,
        bg_color=None,
        view_matrix: torch.Tensor | None = None,
        proj_matrix: torch.Tensor | None = None,
        collect_w_stats: bool = False,
        gt_rgb: torch.Tensor | None = None,
    ):
        """Render RGB/alpha without autograd.

        Intended for sweeping the dataset to collect per-primitive contribution statistics
        (w = T * alpha) under the *tilecenter* pipeline.

        Args:
            collect_w_stats: if True, accumulates into `stat_sum_w/stat_max_w/stat_hits`.

        Returns:
            rgb:   (B, 3)
            alpha: (B, 1)
        """
        if self.image_size is None:
            raise ValueError("image_size must be set on NeuralPrimitiveSceneTileCUDA before calling render_rgb_alpha_no_grad.")
        if view_matrix is None or proj_matrix is None:
            raise ValueError("Tile renderer requires view_matrix and proj_matrix for culling.")

        W, H = self.image_size
        B = ray_o.shape[0]
        assert B == W * H, "Tile renderer requires full-frame rays (B == W*H)"
        device = self._activate_cuda_device()
        tile_size = int(self.tile_size)

        # --- 1. Tile Culling (3DGS-style) ---
        with torch.no_grad():
            view_proj_matrix = proj_matrix @ view_matrix
            N = self.num_primitives
            pos_world_h = torch.cat([self.positions, torch.ones(N, 1, device=device)], dim=-1)

            # Get positions in view space
            pos_view_h = pos_world_h @ view_matrix.T
            depth = pos_view_h[:, 2]
            near_plane = 0.001
            in_front = depth > near_plane

            # Get projected radius in pixels
            scales = torch.exp(self.log_scales).amax(dim=-1)
            focal_x = proj_matrix[0, 0] * W / 2.0
            focal_y = proj_matrix[1, 1] * H / 2.0
            focal = (focal_x + focal_y) * 0.5

            # Derive radius from score threshold, similar to the Triton kernel logic
            # sc = exp(-0.5 * perp2 / s^2) > score_threshold
            # -> perp2 < -2 * s^2 * log(score_threshold)
            # -> perp < s * sqrt(-2 * log(score_threshold))
            radius_multiplier = torch.sqrt(-2.0 * torch.log(torch.tensor(score_threshold, device=device)))
            world_radius = scales * radius_multiplier
            pixel_radius = world_radius / (depth + 1e-8) * focal

            # Project centers to screen
            pos_clip = pos_view_h @ proj_matrix.T
            w = pos_clip[..., 3:4]
            w = w + 1e-6 * (w == 0)
            pos_ndc = pos_clip[..., :3] / w
            pos_screen_x = (pos_ndc[:, 0] + 1) * 0.5 * W
            pos_screen_y = (pos_ndc[:, 1] + 1) * 0.5 * H

            # Vectorized tile-primitive overlap check
            num_tiles_x = (W + tile_size - 1) // tile_size
            num_tiles_y = (H + tile_size - 1) // tile_size
            num_tiles = num_tiles_x * num_tiles_y

            tile_indices = torch.arange(num_tiles, device=device)
            tile_x_coords = tile_indices % num_tiles_x
            tile_y_coords = tile_indices // num_tiles_x

            tile_min_px_x = (tile_x_coords * tile_size).unsqueeze(1)
            tile_max_px_x = ((tile_x_coords + 1) * tile_size).unsqueeze(1)
            tile_min_px_y = (tile_y_coords * tile_size).unsqueeze(1)
            tile_max_px_y = ((tile_y_coords + 1) * tile_size).unsqueeze(1)

            prim_min_px_x = (pos_screen_x - pixel_radius).unsqueeze(0)
            prim_max_px_x = (pos_screen_x + pixel_radius).unsqueeze(0)
            prim_min_px_y = (pos_screen_y - pixel_radius).unsqueeze(0)
            prim_max_px_y = (pos_screen_y + pixel_radius).unsqueeze(0)

            overlap_x = (prim_min_px_x < tile_max_px_x) & (prim_max_px_x > tile_min_px_x)
            overlap_y = (prim_min_px_y < tile_max_px_y) & (prim_max_px_y > tile_min_px_y)

            overlap_matrix = overlap_x & overlap_y & in_front.unsqueeze(0)

        tile_order_mode = self._resolve_tile_order_mode()
        use_tilecenter = tile_order_mode == "tilecenter"
        if not use_tilecenter:
            raise RuntimeError("render_rgb_alpha_no_grad currently supports MLPSP_TILE_ORDER=tilecenter only.")

        # Convert overlap matrix to sorted sparse format for kernel
        tile_prim_pairs = overlap_matrix.nonzero(as_tuple=False)
        if tile_prim_pairs.numel() == 0:
            culled_prim_indices = torch.empty(0, dtype=torch.int32, device=device)
            tile_offsets = torch.zeros(num_tiles + 1, dtype=torch.int32, device=device)
        else:
            sorted_indices = torch.argsort(tile_prim_pairs[:, 0])
            sorted_pairs = tile_prim_pairs[sorted_indices]
            tile_ids = sorted_pairs[:, 0]
            culled_prim_indices = sorted_pairs[:, 1].to(torch.int32)

            tile_offsets = torch.zeros(num_tiles + 1, dtype=torch.long, device=device)
            unique_tile_ids, counts = torch.unique_consecutive(tile_ids, return_counts=True)
            tile_offsets.scatter_add_(0, unique_tile_ids + 1, counts)
            tile_offsets = torch.cumsum(tile_offsets, dim=0).to(torch.int32)

        # --- 2. Kernel call (no autograd) ---
        ext = self._ensure_tile_ext()

        if isinstance(self.encoder, IntegratedPositionalEncoding):
            freq = self.encoder.freq_bands.to(device=device, dtype=torch.float32)
            sigma2 = float(self.encoder.sigma2)
            att = torch.exp(-0.5 * (freq * freq) * sigma2)
        else:
            freq = self.encoder.freq_bands.to(device=device, dtype=torch.float32)
            att = torch.ones_like(freq)
        if hasattr(self.encoder, "runtime_frequency_weights"):
            att = att * self.encoder.runtime_frequency_weights.to(device=device, dtype=torch.float32)

        ray_d_n = F.normalize(ray_d, dim=-1)

        if collect_w_stats:
            if gt_rgb is None:
                raise ValueError("collect_w_stats=True requires gt_rgb (shape=(B,3), B=W*H) to compute pixel_error inside the CUDA kernel.")
            if gt_rgb.dim() != 2 or gt_rgb.shape[0] != B or gt_rgb.shape[1] != 3:
                raise ValueError(f"gt_rgb must have shape (B,3) with B=W*H={B}, got {tuple(gt_rgb.shape)}")
            gt_rgb = gt_rgb.to(device=device, dtype=torch.float32).contiguous()

            if not hasattr(self, "stat_sum_werr") or self.stat_sum_werr is None or self.stat_sum_werr.numel() != self.num_primitives:
                self.register_buffer("stat_sum_werr", torch.zeros(self.num_primitives, dtype=torch.float32, device=device))
            if not hasattr(self, "stat_contrib_count") or self.stat_contrib_count is None or self.stat_contrib_count.numel() != self.num_primitives:
                self.register_buffer("stat_contrib_count", torch.zeros(self.num_primitives, dtype=torch.int32, device=device))

            rgb, alpha = ext.render_tiles_forward_tilecenter_stats(
                ray_o.contiguous(), ray_d_n.contiguous(),
                self.positions.contiguous(), self.log_scales.contiguous(), F.normalize(self.quats, dim=-1).contiguous(),
                culled_prim_indices, tile_offsets.to(torch.int32),
                freq.contiguous(), att.contiguous(),
                int(freq.numel()),
                self.W1.contiguous(), self.b1.contiguous(),
                self.W2.contiguous(), self.b2.contiguous(),
                self.W3.contiguous(), self.b3.contiguous(),
                int(self.W1.shape[0]), int(self.W1.shape[1]), int(self.W1.shape[2]),
                int(W), int(H), tile_size, tile_size,
                float(score_threshold),
                gt_rgb,
                self.stat_sum_w.contiguous(),
                self.stat_max_w.contiguous(),
                self.stat_hits.contiguous(),
                self.stat_sum_werr.contiguous(),
                self.stat_contrib_count.contiguous(),
            )

        else:
            rgb, alpha = ext.render_tiles_forward_tilecenter(
                ray_o.contiguous(), ray_d_n.contiguous(),
                self.positions.contiguous(), self.log_scales.contiguous(),
                F.normalize(self.quats, dim=-1).contiguous(),
                culled_prim_indices, tile_offsets.to(torch.int32),
                freq.contiguous(), att.contiguous(),
                self.W1.contiguous(), self.b1.contiguous(),
                self.W2.contiguous(), self.b2.contiguous(),
                self.W3.contiguous(), self.b3.contiguous(),
                int(W), int(H), tile_size, tile_size,
                float(score_threshold),
            )

        # Background compositing (matches forward)
        if bg_color is not None:
            if not torch.is_tensor(bg_color):
                bg = torch.tensor(bg_color, dtype=ray_o.dtype, device=ray_o.device)
            else:
                bg = bg_color.to(ray_o.device, dtype=ray_o.dtype)
            if bg.numel() == 1:
                bg = bg.repeat(3)
            rgb = rgb + (1.0 - alpha).clamp(0.0, 1.0) * bg.view(1, 3)

        return rgb, alpha

    @torch.no_grad()
    def render_rgb_t_no_grad(
            self,
            ray_o: torch.Tensor,
            ray_d: torch.Tensor,
            score_threshold: float = 1e-3,
            bg_color=None,
            view_matrix: torch.Tensor | None = None,
            proj_matrix: torch.Tensor | None = None,
        ):
            """Render RGB and expected depth (t) without autograd.

            This is intended for geometry analysis. The computation path (culling, ordering,
            ray normalization, kernel selection) matches `forward()` as closely as possible, except
            that it calls the no-grad CUDA entrypoint that additionally returns per-pixel expected depth `t`.

            Returns:
                rgb: (B, 3) float tensor
                t:   (B, 1) float tensor (expected depth along the *normalized* ray)
                alpha: (B, 1) float tensor (accumulated opacity), returned for convenience
            """
            if self.image_size is None:
                raise ValueError("image_size must be set on NeuralPrimitiveSceneTileCUDA before calling render_rgb_t_no_grad.")
            if view_matrix is None or proj_matrix is None:
                raise ValueError("Tile renderer requires view_matrix and proj_matrix for culling.")

            W, H = self.image_size
            B = ray_o.shape[0]
            assert B == W * H, "Tile renderer requires full-frame rays (B == W*H)"
            device = self._activate_cuda_device()
            tile_size = int(self.tile_size)

            # --- 1. Tile Culling (identical to forward) ---
            N = self.num_primitives
            pos_world_h = torch.cat([self.positions, torch.ones(N, 1, device=device)], dim=-1)

            pos_view_h = pos_world_h @ view_matrix.T
            depth_center = pos_view_h[:, 2]
            near_plane = 0.001
            in_front = depth_center > near_plane

            scales = torch.exp(self.log_scales).amax(dim=-1)
            focal_x = proj_matrix[0, 0] * W / 2.0
            focal_y = proj_matrix[1, 1] * H / 2.0
            focal = (focal_x + focal_y) * 0.5

            radius_multiplier = torch.sqrt(-2.0 * torch.log(torch.tensor(score_threshold, device=device)))
            world_radius = scales * radius_multiplier
            pixel_radius = world_radius / (depth_center + 1e-8) * focal

            pos_clip = pos_view_h @ proj_matrix.T
            w = pos_clip[..., 3:4]
            w = w + 1e-6 * (w == 0)
            pos_ndc = pos_clip[..., :3] / w
            pos_screen_x = (pos_ndc[:, 0] + 1) * 0.5 * W
            pos_screen_y = (pos_ndc[:, 1] + 1) * 0.5 * H

            num_tiles_x = (W + tile_size - 1) // tile_size
            num_tiles_y = (H + tile_size - 1) // tile_size
            num_tiles = num_tiles_x * num_tiles_y

            tile_indices = torch.arange(num_tiles, device=device)
            tile_x_coords = tile_indices % num_tiles_x
            tile_y_coords = tile_indices // num_tiles_x

            tile_min_px_x = (tile_x_coords * tile_size).unsqueeze(1)
            tile_max_px_x = ((tile_x_coords + 1) * tile_size).unsqueeze(1)
            tile_min_px_y = (tile_y_coords * tile_size).unsqueeze(1)
            tile_max_px_y = ((tile_y_coords + 1) * tile_size).unsqueeze(1)

            prim_min_px_x = (pos_screen_x - pixel_radius).unsqueeze(0)
            prim_max_px_x = (pos_screen_x + pixel_radius).unsqueeze(0)
            prim_min_px_y = (pos_screen_y - pixel_radius).unsqueeze(0)
            prim_max_px_y = (pos_screen_y + pixel_radius).unsqueeze(0)

            overlap_x = (prim_min_px_x < tile_max_px_x) & (prim_max_px_x > tile_min_px_x)
            overlap_y = (prim_min_px_y < tile_max_px_y) & (prim_max_px_y > tile_min_px_y)
            overlap_matrix = overlap_x & overlap_y & in_front.unsqueeze(0)

            # --- 1.5. Convert overlap matrix to sorted sparse format for kernel ---
            tile_order_mode = self._resolve_tile_order_mode()
            use_tilecenter = tile_order_mode == "tilecenter"
            if not use_tilecenter:
                raise RuntimeError("render_rgb_t_no_grad requires MLPSP_TILE_ORDER=tilecenter (depth output is implemented for tilecenter path).")

            tile_prim_pairs = overlap_matrix.nonzero(as_tuple=False)
            if tile_prim_pairs.numel() == 0:
                culled_prim_indices = torch.empty(0, dtype=torch.int32, device=device)
                tile_offsets = torch.zeros(num_tiles + 1, dtype=torch.int32, device=device)
            else:
                sorted_indices = torch.argsort(tile_prim_pairs[:, 0])
                sorted_pairs = tile_prim_pairs[sorted_indices]
                tile_ids = sorted_pairs[:, 0]
                culled_prim_indices = sorted_pairs[:, 1].to(torch.int32)

                tile_offsets = torch.zeros(num_tiles + 1, dtype=torch.long, device=device)
                unique_tile_ids, counts = torch.unique_consecutive(tile_ids, return_counts=True)
                tile_offsets.scatter_add_(0, unique_tile_ids + 1, counts)
                tile_offsets = torch.cumsum(tile_offsets, dim=0).to(torch.int32)

            # --- 2. Kernel Call (tilecenter, no autograd, with t) ---
            ext = self._ensure_tile_ext()

            if isinstance(self.encoder, IntegratedPositionalEncoding):
                freq = self.encoder.freq_bands.to(device=device, dtype=torch.float32)
                sigma2 = float(self.encoder.sigma2)
                att = torch.exp(-0.5 * (freq * freq) * sigma2)
            else:
                freq = self.encoder.freq_bands.to(device=device, dtype=torch.float32)
                att = torch.ones_like(freq)

            ray_d_n = F.normalize(ray_d, dim=-1)

            if bg_color is not None:
                if not torch.is_tensor(bg_color):
                    bg = torch.tensor(bg_color, dtype=ray_o.dtype, device=ray_o.device)
                else:
                    bg = bg_color.to(ray_o.device, dtype=ray_o.dtype)
            else:
                bg = None
            rgb, alpha, t = ext.render_tiles_forward_tilecenter_with_t(
                ray_o.contiguous(), ray_d_n.contiguous(),
                self.positions.contiguous(), self.log_scales.contiguous(),
                F.normalize(self.quats, dim=-1).contiguous(),
                culled_prim_indices, tile_offsets.to(torch.int32),
                freq.contiguous(), att.contiguous(),
                self.W1.contiguous(), self.b1.contiguous(),
                self.W2.contiguous(), self.b2.contiguous(),
                self.W3.contiguous(), self.b3.contiguous(),
                int(W), int(H), tile_size, tile_size,
                float(score_threshold)
            )

            if bg is not None:
                if bg.numel() == 1:
                    bg = bg.repeat(3)
                rgb = rgb + (1.0 - alpha).clamp(0.0, 1.0) * bg.view(1, 3)

            return rgb, t, alpha

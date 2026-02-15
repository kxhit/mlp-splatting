import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # package-relative import when using mip360 as a package
    from .utils.sampling import octree_sample  # type: ignore
except Exception:
    # fallback for script-style execution
    from utils.sampling import octree_sample  # type: ignore

# Optional: try to load custom CUDA extension for pass-1/2 selection
_ext = None
_ext_error = None
try:
    from torch.utils.cpp_extension import load
    this_dir = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(this_dir, 'cuda', 'np_cuda_kernels.cu')
    if os.path.exists(src_path):
        _ext = load(name='np_cuda_kernels', sources=[src_path], verbose=False)
    else:
        _ext_error = f'CUDA source missing at {src_path}'
except Exception as e:  # noqa: BLE001
    _ext = None
    _ext_error = str(e)

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

# Optional: cooperative splat tile-based renderer extension (loaded on demand)
_splattile_ext = None
_splattile_error = None
try:
    from torch.utils.cpp_extension import load as load_ext
    this_dir = os.path.dirname(os.path.abspath(__file__))
    splattile_src = os.path.join(this_dir, 'cuda', 'np_splattile_kernels.cu')
    if os.path.exists(splattile_src):
        # Delay actual compilation until first use to keep import fast
        _splattile_src_path = splattile_src
    else:
        _splattile_error = f'Splat Tile CUDA source missing at {splattile_src}'
        _splattile_src_path = None
except Exception as e:  # noqa: BLE001
    _splattile_ext = None
    _splattile_error = str(e)


class IntegratedPositionalEncoding(nn.Module):
    def __init__(self, in_dim=6, num_frequencies=10, sigma=0.01):
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        self.sigma2 = sigma ** 2
        freq_bands = 2. ** torch.linspace(0., num_frequencies - 1, num_frequencies) * torch.pi
        self.register_buffer("freq_bands", freq_bands)

    def forward(self, x):
        x_proj = x[..., None] * self.freq_bands
        attenuation = torch.exp(-0.5 * (self.freq_bands ** 2) * self.sigma2)
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

    def forward(self, x):
        x_proj = x[..., None] * self.freq_bands
        x_proj = x_proj.view(*x.shape[:-1], -1)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    @property
    def out_dim(self):
        return self.in_dim * self.num_frequencies * 2


class IndexedLinearFlat(nn.Module):
    def __init__(self, num_primitives, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_primitives, out_features, in_features))
        self.bias = nn.Parameter(torch.empty(num_primitives, out_features))
        self.in_features = in_features
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, idx):
        w = self.weight.index_select(0, idx)
        b = self.bias.index_select(0, idx)
        y = torch.einsum("md,mhd->mh", x, w) + b
        return y


class IndexedMLP(nn.Module):
    def __init__(self, num_primitives, encoded_dim, hidden_dim=16):
        super().__init__()
        self.lin1 = IndexedLinearFlat(num_primitives, encoded_dim, hidden_dim)
        self.lin2 = IndexedLinearFlat(num_primitives, hidden_dim, hidden_dim)
        self.lin3 = IndexedLinearFlat(num_primitives, hidden_dim, 4)

    def forward(self, x_enc, idx):
        h1 = F.relu(self.lin1(x_enc, idx))
        h2 = F.relu(self.lin2(h1, idx))
        o = self.lin3(h2, idx)
        rgb = torch.sigmoid(o[:, :3])
        sigma = F.softplus(o[:, 3:4])
        return rgb, sigma


@torch.no_grad()
def _select_variable_k_cuda(ray_o, ray_d, positions, log_scales, threshold: float):
    if _ext is None:
        raise RuntimeError(f"CUDA extension not available: {_ext_error}")
    device = ray_o.device
    B = ray_o.shape[0]
    N = positions.shape[0]

    ray_d_n = F.normalize(ray_d, dim=-1)

    # Pass-1
    counts = _ext.count_above_threshold(ray_o, ray_d_n, positions, log_scales.squeeze(-1), float(threshold)).to(torch.int64)

    if B == 1:
        offsets = torch.zeros_like(counts)
    else:
        offsets = torch.empty_like(counts)
        offsets[0] = 0
        offsets[1:] = torch.cumsum(counts[:-1], dim=0)

    M = int(counts.sum().item())
    if M == 0:
        empty_l = torch.empty(0, dtype=torch.long, device=device)
        empty_f = torch.empty(0, dtype=torch.float32, device=device)
        empty_f3 = torch.empty(0, 3, dtype=torch.float32, device=device)
        return empty_l, empty_l, empty_f, empty_f3, counts, offsets, ray_d_n

    # Pass-2
    out_ray, out_prim, out_t, out_dx, out_dy, out_dz = _ext.write_above_threshold(
        ray_o, ray_d_n, positions, log_scales.squeeze(-1), offsets, float(threshold), M)

    dist_vec = torch.stack([out_dx, out_dy, out_dz], dim=-1)
    return out_ray.to(torch.int64), out_prim.to(torch.int64), out_t, dist_vec, counts, offsets, ray_d_n


class NeuralPrimitiveScene(nn.Module):
    """API-compatible scene with CUDA pass-1/2 selection.

    This class mirrors mip360.vmodel.NeuralPrimitiveScene but replaces the Triton
    selection with CUDA kernels when available. The rest of the pipeline (sorting,
    encoding, MLP, compositing) stays in PyTorch to preserve behavior.
    """

    def __init__(self, num_primitives=1024, hidden_dim=16, pe_frequencies=10,
                 initial_positions=None, use_integrated_pe=True):
        super().__init__()
        self.num_primitives = num_primitives

        if initial_positions is None:
            bound = 1.0
            self.positions = nn.Parameter(torch.rand(num_primitives, 3) * 2 * bound - bound)
            sampled_scales = torch.ones(num_primitives) * 0.1
        else:
            sampled_positions, sampled_scales = octree_sample(initial_positions)
            print("Sampled", len(sampled_positions), "initial positions from point cloud")
            assert len(sampled_positions) == num_primitives
            sampled_scales /= 10
            self.positions = nn.Parameter(sampled_positions.clone().detach())

        self.log_scales = nn.Parameter(torch.log(sampled_scales).unsqueeze(-1))

        if use_integrated_pe:
            self.encoder = IntegratedPositionalEncoding(in_dim=6, num_frequencies=pe_frequencies, sigma=0.01)
        else:
            self.encoder = FourierFeatureEncoding(in_dim=6, num_frequencies=pe_frequencies)

        self.mlps = IndexedMLP(num_primitives, self.encoder.out_dim, hidden_dim)

        self.register_buffer("alpha_accum", torch.zeros(num_primitives))

    @torch.no_grad()
    def _select_variable_k(self, ray_o, ray_d, threshold: float):
        # Prefer CUDA kernels; if unavailable, fall back to Triton in vmodel
        try:
            return _select_variable_k_cuda(ray_o, ray_d, self.positions, self.log_scales, threshold)
        except Exception:
            # Fallback path using the reference Triton-based implementation
            from .vmodel import NeuralPrimitiveScene as RefScene  # type: ignore
            ref = RefScene(self.num_primitives)
            # share parameters to avoid allocations
            ref.positions = self.positions
            ref.log_scales = self.log_scales
            return ref._select_variable_k(ray_o, ray_d, threshold)

    def forward(self, ray_o, ray_d, score_threshold=1e-4, bg_color=None):
        B = ray_o.shape[0]
        device = ray_o.device

        ray_ids, prim_ids, t_m, dist_m, seg_len, offsets, ray_d_n = \
            self._select_variable_k(ray_o, ray_d, float(score_threshold))

        if prim_ids.numel() == 0:
            out = torch.zeros(B, 3, device=device)
            if bg_color is not None:
                if not torch.is_tensor(bg_color):
                    bg = torch.tensor(bg_color, dtype=out.dtype, device=device)
                else:
                    bg = bg_color.to(device=device, dtype=out.dtype)
                if bg.numel() == 1:
                    bg = bg.repeat(3)
                out = out + bg.view(1, 3)
            return out

        # sort by (ray_id, t)
        t_shift = t_m - t_m.min()
        t_norm = (t_shift / (t_shift.max() + 1e-9) * (2**32 - 1)).to(torch.int64)
        key = (ray_ids.to(torch.int64) << 32) | t_norm
        order = torch.argsort(key)

        ray_ids = ray_ids[order]
        prim_ids = prim_ids[order]
        dist_m = dist_m[order]

        rd_m = ray_d_n[ray_ids]
        x_enc = self.encoder(torch.cat([rd_m, dist_m], dim=-1))
        rgb_m, sigma_raw_m = self.mlps(x_enc, prim_ids)

        scales = torch.exp(self.log_scales).squeeze(-1)
        s_m = scales.index_select(0, prim_ids).unsqueeze(-1)
        decay = torch.exp(- (torch.norm(dist_m, dim=-1, keepdim=True) ** 2) / (2.0 * (s_m ** 2) + 1e-12))
        sigma = sigma_raw_m * decay

        alpha = 1.0 - torch.exp(-sigma)
        s = torch.log(1.0 - alpha + 1e-12).squeeze(-1)
        cs = torch.cumsum(s, dim=0)
        cs_before = torch.zeros_like(cs)
        cs_before[1:] = cs[:-1]
        start_for_pos = offsets[ray_ids]
        baseline = cs_before.gather(0, start_for_pos)
        trans = torch.exp(cs_before - baseline).unsqueeze(-1)
        weights = alpha * trans

        out = torch.zeros(B, 3, device=device)
        out.index_add_(0, ray_ids, weights * rgb_m)

        if bg_color is not None:
            bg = bg_color
            if not torch.is_tensor(bg):
                bg = torch.tensor(bg, dtype=out.dtype, device=device)
            else:
                bg = bg.to(device=device, dtype=out.dtype)
            if bg.numel() == 1:
                bg = bg.repeat(3)
            sum_w = torch.zeros(B, 1, device=device)
            sum_w.index_add_(0, ray_ids, weights)
            out = out + (1.0 - sum_w).clamp(0.0, 1.0) * bg.view(1, 3)

        with torch.no_grad():
            a_sum = alpha.squeeze(-1)
            add = torch.zeros_like(self.alpha_accum)
            add.index_add_(0, prim_ids, a_sum)
            sel = torch.unique(prim_ids)
            old = self.alpha_accum.index_select(0, sel)
            new = 0.2 * old + 0.8 * add.index_select(0, sel)
            self.alpha_accum.scatter_(0, sel, new)

        return out

    def forward_depth(self, ray_o, ray_d, T_wc, score_threshold=1e-4):
        B = ray_o.shape[0]
        device = ray_o.device

        ray_ids, prim_ids, t_m, dist_m, seg_len, offsets, ray_d_n = \
            self._select_variable_k(ray_o, ray_d, float(score_threshold))

        if prim_ids.numel() == 0:
            return torch.zeros(B, 3, device=device), torch.zeros(B, 1, device=device)

        t_shift = t_m - t_m.min()
        t_norm = (t_shift / (t_shift.max() + 1e-9) * (2**32 - 1)).to(torch.int64)
        key = (ray_ids.to(torch.int64) << 32) | t_norm
        order = torch.argsort(key)

        ray_ids = ray_ids[order]
        prim_ids = prim_ids[order]
        dist_m = dist_m[order]
        t_m = t_m[order]

        rd_m = ray_d_n[ray_ids]
        x_enc = self.encoder(torch.cat([rd_m, dist_m], dim=-1))
        rgb_m, sigma_raw_m = self.mlps(x_enc, prim_ids)

        scales = torch.exp(self.log_scales).squeeze(-1)
        s_m = scales.index_select(0, prim_ids).unsqueeze(-1)
        decay = torch.exp(- (torch.norm(dist_m, dim=-1, keepdim=True) ** 2) / (2.0 * (s_m ** 2) + 1e-12))
        sigma = sigma_raw_m * decay

        alpha = 1.0 - torch.exp(-sigma)
        s = torch.log(1.0 - alpha + 1e-12).squeeze(-1)
        cs = torch.cumsum(s, dim=0)
        cs_before = torch.zeros_like(cs)
        cs_before[1:] = cs[:-1]
        start_for_pos = offsets[ray_ids]
        baseline = cs_before.gather(0, start_for_pos)
        trans = torch.exp(cs_before - baseline).unsqueeze(-1)
        weights = alpha * trans

        p_m = ray_o[ray_ids] + t_m.unsqueeze(-1) * ray_d_n[ray_ids]
        p_m_h = torch.cat([p_m, torch.ones_like(p_m[:, :1])], dim=-1)
        pose = torch.eye(4, device=device)
        pose[:3, ] = T_wc
        T_cw = torch.inverse(pose)
        p_c_h = (T_cw @ p_m_h.T).T
        depth_m = p_c_h[:, 2:3]

        depth = torch.zeros(B, 1, device=device)
        depth.index_add_(0, ray_ids, weights * depth_m)
        return depth


def _encode_x6(x6: torch.Tensor, freq: torch.Tensor, att: torch.Tensor) -> torch.Tensor:
    # x6: [M, 6], freq/att: [F]
    xproj = x6[:, :, None] * freq[None, None, :] * att[None, None, :]
    sincos = torch.cat([torch.sin(xproj), torch.cos(xproj)], dim=-1)
    return sincos.reshape(x6.shape[0], -1)


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
    h2 = torch.einsum("mho,mh->mo", W2s, h1) + b2s
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
        positions, log_scales,
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
                rays_o, rays_d, positions, log_scales,
                culled_prim_indices, tile_offsets,
                freq, att, W1, b1, W2, b2, W3, b3,
                int(Wimg), int(Himg), int(TW), int(TH), float(threshold)
            )
        elif mode == 1:
            rgb, alpha = ext.render_tiles_forward_tilecenter(
                rays_o, rays_d, positions, log_scales,
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
            rays_o, rays_d, positions, log_scales,
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
        (rays_o, rays_d, positions, log_scales,
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
        traversal_only = os.environ.get("MLPSP_BWD_TRAVERSAL_ONLY", "0").strip() in {"1", "true", "yes", "on"}
        stage = os.environ.get("MLPSP_BWD_STAGE", "full").strip().lower()
        if traversal_only:
            stage = "traversal"

        if ctx.mode == 0:
            counts = ext.render_tiles_backward_count_per_pixel(
                rays_o, rays_d, positions, log_scales,
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
                grad_W1 = torch.zeros_like(W1)
                grad_b1 = torch.zeros_like(b1)
                grad_W2 = torch.zeros_like(W2)
                grad_b2 = torch.zeros_like(b2)
                grad_W3 = torch.zeros_like(W3)
                grad_b3 = torch.zeros_like(b3)
            else:
                (prim_id, x6, grad_rgb, grad_sigma_raw,
                 grad_positions, grad_log_scales) = ext.render_tiles_backward_emit_per_pixel(
                    rays_o, rays_d, positions, log_scales,
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
            if stage in {"traversal", "forward"}:
                # Profiling-only path: run traversal + MLP forward, skip backprop.
                ext.render_tiles_backward_tilecenter_traversal(
                    rays_o, rays_d, positions, log_scales,
                    culled_prim_indices, tile_offsets,
                    freq, att, W1, b1, W2, b2, W3, b3,
                    ctx.Wimg, ctx.Himg, ctx.TW, ctx.TH, ctx.threshold
                )
                grad_positions = torch.zeros_like(positions)
                grad_log_scales = torch.zeros_like(log_scales)
                grad_W1 = torch.zeros_like(W1)
                grad_b1 = torch.zeros_like(b1)
                grad_W2 = torch.zeros_like(W2)
                grad_b2 = torch.zeros_like(b2)
                grad_W3 = torch.zeros_like(W3)
                grad_b3 = torch.zeros_like(b3)
            elif stage in {"backprop_noatomic", "backprop-noatomic", "noatomic"}:
                # Profiling-only path: run backprop math without atomic accumulation.
                ext.render_tiles_backward_tilecenter_backprop_noatomic(
                    rays_o, rays_d, positions, log_scales,
                    culled_prim_indices, tile_offsets,
                    freq, att, W1, b1, W2, b2, W3, b3,
                    grad_out_rgb, grad_T_end,
                    ctx.Wimg, ctx.Himg, ctx.TW, ctx.TH, ctx.threshold
                )
                grad_positions = torch.zeros_like(positions)
                grad_log_scales = torch.zeros_like(log_scales)
                grad_W1 = torch.zeros_like(W1)
                grad_b1 = torch.zeros_like(b1)
                grad_W2 = torch.zeros_like(W2)
                grad_b2 = torch.zeros_like(b2)
                grad_W3 = torch.zeros_like(W3)
                grad_b3 = torch.zeros_like(b3)
            else:
                (grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3,
                 grad_positions, grad_log_scales) = ext.render_tiles_backward_tilecenter_full(
                    rays_o, rays_d, positions, log_scales,
                    culled_prim_indices, tile_offsets,
                    freq, att, W1, b1, W2, b2, W3, b3,
                    grad_out_rgb, grad_T_end,
                    ctx.Wimg, ctx.Himg, ctx.TW, ctx.TH, ctx.threshold
                )
        else:
            raise ValueError("Unsupported tile mode for backward")

        return (
            None, None,  # rays_o, rays_d
            grad_positions, grad_log_scales,
            None, None,  # culled_prim_indices, tile_offsets
            None, None,  # freq, att
            grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3,
            None, None, None, None, None,  # Wimg, Himg, TW, TH, threshold
            None, None,  # mode, bg_color
        )

class NeuralPrimitiveSceneTileCUDA(nn.Module):
    """Tile-based single-pass MLP splatting renderer.

    - Same constructor signature and forward API as vmodel.NeuralPrimitiveScene.
    - Requires full-frame rays (B == W*H). renderer.py sets image_size before calling.
    - One CUDA kernel pass per frame: per-tile iterate primitives (depth-ordered),
      batch-forward 16x16 pixels, and immediate alpha blending with early-stop.
    """

    requires_full_image = True

    def __init__(self, num_primitives=1024, hidden_dim=16, pe_frequencies=10,
                 initial_positions=None, use_integrated_pe=True):
        super().__init__()
        if int(hidden_dim) > 32:
            raise ValueError(
                "NeuralPrimitiveSceneTileCUDA currently supports hidden_dim <= 32 "
                "(kernel scratch-cap tuning in backward/full path)."
            )
        self.num_primitives = num_primitives
        self.hidden_dim = hidden_dim
        self.pe_frequencies = pe_frequencies
        self.use_integrated_pe = use_integrated_pe
        self.image_size = None  # set by renderer

        if initial_positions is None:
            bound = 1.0
            self.positions = nn.Parameter(torch.rand(num_primitives, 3) * 2 * bound - bound)
            sampled_scales = torch.ones(num_primitives) * 0.1
        else:
            sampled_positions, sampled_scales = octree_sample(initial_positions)
            print("Sampled", len(sampled_positions), "initial positions from point cloud's", len(initial_positions), "points")

            #assert len(sampled_positions) == num_primitives
            sampled_scales /= 10
            self.positions = nn.Parameter(sampled_positions.clone().detach())

        self.log_scales = nn.Parameter(torch.log(sampled_scales).unsqueeze(-1))

        # Keep encoder object to export freq bands
        if use_integrated_pe:
            self.encoder = IntegratedPositionalEncoding(in_dim=6, num_frequencies=pe_frequencies, sigma=0.01)
        else:
            self.encoder = FourierFeatureEncoding(in_dim=6, num_frequencies=pe_frequencies)

        # Per-primitive MLP weights (float32 for simplicity)
        IN = self.encoder.out_dim
        H = hidden_dim
        OUT = 4
        self.W1 = nn.Parameter(torch.randn(num_primitives, H, IN) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(num_primitives, H))
        self.W2 = nn.Parameter(torch.randn(num_primitives, H, H) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(num_primitives, H))
        self.W3 = nn.Parameter(torch.randn(num_primitives, OUT, H) * 0.01)
        self.b3 = nn.Parameter(torch.zeros(num_primitives, OUT))

        self.register_buffer("alpha_accum", torch.zeros(num_primitives))

    def set_image_size(self, W: int, H: int):
        self.image_size = (int(W), int(H))

    def load_legacy_state_dict(self, legacy_state):
        """Loads weights from a legacy model's state_dict."""
        with torch.no_grad():
            self.W1.copy_(legacy_state['mlps.lin1.weight'])
            self.b1.copy_(legacy_state['mlps.lin1.bias'])
            self.W2.copy_(legacy_state['mlps.lin2.weight'])
            self.b2.copy_(legacy_state['mlps.lin2.bias'])
            self.W3.copy_(legacy_state['mlps.lin3.weight'])
            self.b3.copy_(legacy_state['mlps.lin3.bias'])

            if 'positions' in legacy_state:
                self.positions.copy_(legacy_state['positions'])
            if 'log_scales' in legacy_state:
                self.log_scales.copy_(legacy_state['log_scales'])

    def _ensure_tile_ext(self):
        global _tile_ext, _tile_ext_key
        if _tile_ext is None:
            if _tile_error is not None and _tile_error.startswith('Tile CUDA source missing'):
                raise RuntimeError(_tile_error)
            if '_tile_src_path' not in globals() or _tile_src_path is None:
                raise RuntimeError(f"Tile CUDA not available: {_tile_error}")
            # JIT compile on first use; allow env-controlled maxrregcount tuning.
            key, name, extra_cuda_cflags = _tile_ext_build_config()
            _tile_ext = load_ext(
                name=name,
                sources=[_tile_src_path],
                verbose=False,
                extra_cuda_cflags=extra_cuda_cflags,
            )
            _tile_ext_key = key
        else:
            key, name, extra_cuda_cflags = _tile_ext_build_config()
            if _tile_ext_key != key:
                _tile_ext = load_ext(
                    name=name,
                    sources=[_tile_src_path],
                    verbose=False,
                    extra_cuda_cflags=extra_cuda_cflags,
                )
                _tile_ext_key = key
        return _tile_ext

    @staticmethod
    def _ensure_tile_ext_static():
        global _tile_ext, _tile_ext_key
        if _tile_ext is None:
            if _tile_error is not None and _tile_error.startswith('Tile CUDA source missing'):
                raise RuntimeError(_tile_error)
            if '_tile_src_path' not in globals() or _tile_src_path is None:
                raise RuntimeError(f"Tile CUDA not available: {_tile_error}")
            key, name, extra_cuda_cflags = _tile_ext_build_config()
            _tile_ext = load_ext(
                name=name,
                sources=[_tile_src_path],
                verbose=False,
                extra_cuda_cflags=extra_cuda_cflags,
            )
            _tile_ext_key = key
        else:
            key, name, extra_cuda_cflags = _tile_ext_build_config()
            if _tile_ext_key != key:
                _tile_ext = load_ext(
                    name=name,
                    sources=[_tile_src_path],
                    verbose=False,
                    extra_cuda_cflags=extra_cuda_cflags,
                )
                _tile_ext_key = key
        return _tile_ext

    def forward(self, ray_o, ray_d, score_threshold=1e-4, bg_color=None, view_matrix=None, proj_matrix=None):
        if self.image_size is None:
            raise ValueError("image_size must be set on NeuralPrimitiveSceneTileCUDA before calling forward.")
        if view_matrix is None or proj_matrix is None:
            raise ValueError("Tile renderer now requires view_matrix and proj_matrix for culling.")

        W, H = self.image_size
        B = ray_o.shape[0]
        assert B == W * H, "Tile renderer requires full-frame rays (B == W*H)"
        device = self.positions.device
        tile_size = 16

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
            scales = torch.exp(self.log_scales.squeeze(-1))
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

        # Convert overlap matrix to sorted sparse format for kernel
        # Tile traversal mode:
        # - per_pixel (default): per-pixel hit-list + per-pixel depth sort (exact within MAX_INTERSECTIONS)
        # - tilecenter: tile-level approximate ordering (tile-center ray) + prim-major traversal (in-kernel per-tile sort)
        # - globaldepth: tile-level approximate ordering using each primitive center's camera-space forward (view-space z),
        #                but sorting is performed OUTSIDE the kernel. Kernel assumes per-tile lists are already ordered.
        tile_order_mode = os.environ.get("MLPSP_TILE_ORDER", "per_pixel").strip().lower()
        use_tilecenter = tile_order_mode in {"tilecenter", "tile_center", "tile-center"}
        use_globaldepth = tile_order_mode in {"globaldepth", "global_depth", "global-depth", "globalz", "global_z", "global-z"}

        tile_prim_pairs = overlap_matrix.nonzero(as_tuple=False)
        if tile_prim_pairs.numel() == 0:
            # Handle case with no intersections
            culled_prim_indices = torch.empty(0, dtype=torch.int32, device=device)
            tile_offsets = torch.zeros(num_tiles + 1, dtype=torch.int32, device=device)
        else:
            # Sort pairs:
            # - default: by tile_id only (tile lists are NOT ordered internally)
            # - globaldepth: by (tile_id, global depth rank) so each tile's list is already ordered
            if use_globaldepth:
                # Build a global rank based on primitive center camera-space forward (view-space z).
                # Smaller depth = closer => front-to-back traversal (matches transmittance accumulation).
                prim_order = torch.argsort(depth)  # ascending
                prim_rank = torch.empty(N, dtype=torch.int32, device=device)
                prim_rank[prim_order] = torch.arange(N, device=device, dtype=torch.int32)

                pair_ranks = prim_rank[tile_prim_pairs[:, 1]]
                key = (tile_prim_pairs[:, 0].to(torch.int64) << 32) | pair_ranks.to(torch.int64)
                sorted_indices = torch.argsort(key)
            else:
                # Sort pairs by tile_id
                sorted_indices = torch.argsort(tile_prim_pairs[:, 0])
            sorted_pairs = tile_prim_pairs[sorted_indices]
            
            tile_ids = sorted_pairs[:, 0]
            culled_prim_indices = sorted_pairs[:, 1].to(torch.int32)

            # Build offsets array
            tile_offsets = torch.zeros(num_tiles + 1, dtype=torch.long, device=device)
            unique_tile_ids, counts = torch.unique_consecutive(tile_ids, return_counts=True)
            tile_offsets.scatter_add_(0, unique_tile_ids + 1, counts)
            tile_offsets = torch.cumsum(tile_offsets, dim=0).to(torch.int32)

        # --- 2. Kernel Call ---
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

        if use_globaldepth:
            if torch.is_grad_enabled():
                raise RuntimeError("globaldepth backward is not implemented; use per_pixel or tilecenter.")
            rgb, alpha = ext.render_tiles_autograd_tilecenter_presorted(
                ray_o.contiguous(), ray_d_n.contiguous(),
                self.positions.contiguous(), self.log_scales.squeeze(-1).contiguous(),
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
            return rgb

        mode = 1 if use_tilecenter else 0
        rgb = _TileRenderAutogradFn.apply(
            ray_o.contiguous(), ray_d_n.contiguous(),
            self.positions.contiguous(), self.log_scales.squeeze(-1).contiguous(),
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
        return rgb

    @torch.no_grad()
    def render_rgb_t_no_grad(
        self,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
        score_threshold: float = 1e-4,
        bg_color=None,
        view_matrix: torch.Tensor | None = None,
        proj_matrix: torch.Tensor | None = None,
    ):
        """Render RGB and expected depth (t) without autograd.

        This is intended for densification / analysis. The computation path (culling, ordering,
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
        device = self.positions.device
        tile_size = 16

        # --- 1. Tile Culling (identical to forward) ---
        view_proj_matrix = proj_matrix @ view_matrix  # noqa: F841 (kept for parity / debugging)
        N = self.num_primitives
        pos_world_h = torch.cat([self.positions, torch.ones(N, 1, device=device)], dim=-1)

        pos_view_h = pos_world_h @ view_matrix.T
        depth_center = pos_view_h[:, 2]
        near_plane = 0.001
        in_front = depth_center > near_plane

        scales = torch.exp(self.log_scales.squeeze(-1))
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
        tile_order_mode = os.environ.get("MLPSP_TILE_ORDER", "per_pixel").strip().lower()
        use_tilecenter = tile_order_mode in {"tilecenter", "tile_center", "tile-center"}
        use_globaldepth = tile_order_mode in {"globaldepth", "global_depth", "global-depth", "globalz", "global_z", "global-z"}

        if use_globaldepth:
            raise RuntimeError("render_rgb_t_no_grad currently supports tilecenter/per_pixel ordering only (globaldepth is unsupported here).")
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
            self.positions.contiguous(), self.log_scales.squeeze(-1).contiguous(),
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

class NeuralPrimitiveSceneSplatTileCUDA(nn.Module):
    """
    Cooperative tile-based MLP splatting renderer.
    - Culling, sorting, and rendering are all performed inside a single cooperative kernel.
    - Takes camera parameters and resolution, generates rays internally.
    """
    requires_full_image = True

    def __init__(self, num_primitives=1024, hidden_dim=16, pe_frequencies=10,
                 initial_positions=None, use_integrated_pe=True):
        super().__init__()
        self.num_primitives = num_primitives
        self.hidden_dim = hidden_dim
        self.pe_frequencies = pe_frequencies
        self.use_integrated_pe = use_integrated_pe

        if initial_positions is None:
            bound = 1.0
            self.positions = nn.Parameter(torch.rand(num_primitives, 3) * 2 * bound - bound)
            sampled_scales = torch.ones(num_primitives) * 0.1
        else:
            sampled_positions, sampled_scales = octree_sample(initial_positions)
            print("Sampled", len(sampled_positions), "initial positions from point cloud")
            assert len(sampled_positions) == num_primitives
            sampled_scales /= 10
            self.positions = nn.Parameter(sampled_positions.clone().detach())

        self.log_scales = nn.Parameter(torch.log(sampled_scales).unsqueeze(-1))

        if use_integrated_pe:
            self.encoder = IntegratedPositionalEncoding(in_dim=6, num_frequencies=pe_frequencies, sigma=0.01)
        else:
            self.encoder = FourierFeatureEncoding(in_dim=6, num_frequencies=pe_frequencies)

        IN = self.encoder.out_dim
        H = hidden_dim
        OUT = 4
        self.W1 = nn.Parameter(torch.randn(num_primitives, H, IN) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(num_primitives, H))
        self.W2 = nn.Parameter(torch.randn(num_primitives, H, H) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(num_primitives, H))
        self.W3 = nn.Parameter(torch.randn(num_primitives, OUT, H) * 0.01)
        self.b3 = nn.Parameter(torch.zeros(num_primitives, OUT))

        self.register_buffer("alpha_accum", torch.zeros(num_primitives))

    def _ensure_splattile_ext(self):
        global _splattile_ext
        if _splattile_ext is None:
            if _splattile_error is not None and _splattile_error.startswith('Splat Tile CUDA source missing'):
                raise RuntimeError(_splattile_error)
            if '_splattile_src_path' not in globals() or _splattile_src_path is None:
                raise RuntimeError(f"Splat Tile CUDA not available: {_splattile_error}")
            # JIT compile on first use
            _splattile_ext = load_ext(name='np_splattile_kernels', sources=[_splattile_src_path], verbose=False)
        return _splattile_ext

    def forward(self, resolution, view_matrix, proj_matrix, camera_pos, score_threshold=1e-4, bg_color=None):
        W, H = resolution
        device = self.positions.device
        
        ext = self._ensure_splattile_ext()

        if isinstance(self.encoder, IntegratedPositionalEncoding):
            freq = self.encoder.freq_bands.to(device=device, dtype=torch.float32)
            sigma2 = float(self.encoder.sigma2)
            att = torch.exp(-0.5 * (freq * freq) * sigma2)
        else:
            freq = self.encoder.freq_bands.to(device=device, dtype=torch.float32)
            att = torch.ones_like(freq)

        rgb, alpha = ext.render_splattile_autograd(
            view_matrix.contiguous(), proj_matrix.contiguous(), camera_pos.contiguous(),
            int(W), int(H),
            self.positions.contiguous(), self.log_scales.squeeze(-1).contiguous(),
            self.W1.contiguous(), self.b1.contiguous(),
            self.W2.contiguous(), self.b2.contiguous(),
            self.W3.contiguous(), self.b3.contiguous(),
            freq.contiguous(), att.contiguous(),
            float(score_threshold)
        )

        if bg_color is not None:
            if not torch.is_tensor(bg_color):
                bg = torch.tensor(bg_color, dtype=rgb.dtype, device=rgb.device)
            else:
                bg = bg_color.to(rgb.device, dtype=rgb.dtype)
            if bg.numel() == 1:
                bg = bg.repeat(3)
            # Reshape alpha to (H, W, 1) for broadcasting
            alpha_reshaped = alpha.unsqueeze(-1)
            rgb = rgb + (1.0 - alpha_reshaped).clamp(0.0, 1.0) * bg.view(1, 1, 3)

        return rgb
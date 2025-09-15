import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.sampling import octree_sample

# =========================
# Positional Encodings
# =========================


# 放在 vmodel.py 顶部合适位置
import os, time

class CUDATimer:
    def __init__(self, enabled=True):
        self.enabled = enabled and torch.cuda.is_available()
        if self.enabled:
            self._events = {}
            self._starts = {}
            self.records = {}
    def start(self, name):
        if not self.enabled: return
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self._starts[name] = ev
    def stop(self, name):
        if not self.enabled: return
        ev_end = torch.cuda.Event(enable_timing=True)
        ev_end.record()
        torch.cuda.synchronize()
        ms = self._starts[name].elapsed_time(ev_end)
        self.records[name] = self.records.get(name, 0.0) + ms
    def dump(self, header="PROFILE (per-iter ms)"):
        if not self.enabled: return
        total = sum(self.records.values())
        parts = " | ".join(f"{k}:{v:.2f}" for k,v in self.records.items())
        print(f"{header}: {parts} | total:{total:.2f} ms")
        self.records.clear()



class IntegratedPositionalEncoding(nn.Module):
    def __init__(self, in_dim=6, num_frequencies=10, sigma=0.01):
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        self.sigma2 = sigma ** 2
        freq_bands = 2. ** torch.linspace(0., num_frequencies - 1, num_frequencies) * torch.pi
        self.register_buffer("freq_bands", freq_bands)

    def forward(self, x):
        # x: (..., in_dim)
        x_proj = x[..., None] * self.freq_bands  # (..., in_dim, num_freq)
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
        x_proj = x[..., None] * self.freq_bands  # (..., in_dim, num_freq)
        x_proj = x_proj.view(*x.shape[:-1], -1)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    @property
    def out_dim(self):
        return self.in_dim * self.num_frequencies * 2


# =========================
# Triton Indexed Linear
# =========================

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _indexed_linear_fwd(
        X_ptr,          # (B,K,D)
        W_ptr,          # (N,H,D)
        IDX_ptr,        # (B,K)  int32
        Y_ptr,          # (B,K,H)
        B: tl.constexpr,
        K: tl.constexpr,
        N: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        SXB, SXK, SXD,
        SWN, SWH, SWD,
        SYB, SYK, SYH,
        SIB, SIK,
        BLOCK_D: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        pid0 = tl.program_id(0)  # over B*K
        pid1 = tl.program_id(1)  # over H tiles

        bk = pid0
        b = bk // K
        k = bk % K

        h_off = pid1 * BLOCK_H + tl.arange(0, BLOCK_H)
        d_off = tl.arange(0, BLOCK_D)
        mask_h = h_off < H

        # load idx[b,k]
        idx_val = tl.load(IDX_ptr + b * SIB + k * SIK, mask=True).to(tl.int32)

        # base pointers
        x_ptr = X_ptr + b * SXB + k * SXK

        acc = tl.zeros([BLOCK_H], dtype=tl.float32)

        for d0 in range(0, D, BLOCK_D):
            d_idx = d_off + d0
            mask_d = d_idx < D

            x_val = tl.load(x_ptr + d_idx * SXD, mask=mask_d, other=0.0)  # (BLOCK_D,)

            w_base = W_ptr + idx_val * SWN
            w_ptr = w_base + h_off[:, None] * SWH + d_idx[None, :] * SWD
            w_val = tl.load(w_ptr, mask=mask_h[:, None] & mask_d[None, :], other=0.0)  # (BLOCK_H,BLOCK_D)

            acc += tl.sum(w_val * x_val[None, :], axis=1)

        y_ptr = Y_ptr + b * SYB + k * SYK + h_off * SYH
        tl.store(y_ptr, acc, mask=mask_h)

    # dX kernel: dx[b,k,d] = sum_h dy[b,k,h] * W[idx[b,k],h,d]
    @triton.jit
    def _indexed_linear_dx(
        DY_ptr,        # (B,K,H)
        W_ptr,         # (N,H,D)
        IDX_ptr,       # (B,K) int32
        DX_ptr,        # (B,K,D)
        B: tl.constexpr,
        K: tl.constexpr,
        N: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        S_dy_B, S_dy_K, S_dy_H,
        S_W_N, S_W_H, S_W_D,
        S_idx_B, S_idx_K,
        S_dx_B, S_dx_K, S_dx_D,
        BLOCK_H: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid0 = tl.program_id(0)   # over B*K
        pid1 = tl.program_id(1)   # over D tiles

        bk = pid0
        b = bk // K
        k = bk % K

        d_off = pid1 * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = d_off < D

        # load idx[b,k]
        idx_val = tl.load(IDX_ptr + b * S_idx_B + k * S_idx_K).to(tl.int32)

        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        for h0 in range(0, H, BLOCK_H):
            h_off = h0 + tl.arange(0, BLOCK_H)
            mask_h = h_off < H

            # dy[b,k,h]
            dy_ptr = DY_ptr + b * S_dy_B + k * S_dy_K + h_off * S_dy_H
            dy_val = tl.load(dy_ptr, mask=mask_h, other=0.0)  # (BLOCK_H,)

            # W[idx,h,d]
            w_base = W_ptr + idx_val * S_W_N
            w_ptr  = w_base + h_off[:, None] * S_W_H + d_off[None, :] * S_W_D
            w_val  = tl.load(w_ptr, mask=mask_h[:, None] & mask_d[None, :], other=0.0)  # (BLOCK_H,BLOCK_D)

            # acc_d += sum_h dy * W
            acc += tl.sum(w_val * dy_val[:, None], axis=0)

        # store dx
        dx_ptr = DX_ptr + b * S_dx_B + k * S_dx_K + d_off * S_dx_D
        tl.store(dx_ptr, acc, mask=mask_d)

    # dW kernel: for each (b,k,h,d) tile, atomic add to dW[idx[b,k],h,d] += dy[b,k,h] * x[b,k,d]
    @triton.jit
    def _indexed_linear_dW(
        DY_ptr,        # (B,K,H)
        X_ptr,         # (B,K,D)
        IDX_ptr,       # (B,K) int32
        dW_ptr,        # (N,H,D)
        B: tl.constexpr,
        K: tl.constexpr,
        N: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        S_dy_B, S_dy_K, S_dy_H,
        S_x_B, S_x_K, S_x_D,
        S_idx_B, S_idx_K,
        S_dW_N, S_dW_H, S_dW_D,
        BLOCK_H: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        # grid: (over B*K, over H tiles, over D tiles)
        pid0 = tl.program_id(0)  # bk
        pid1 = tl.program_id(1)  # h tile
        pid2 = tl.program_id(2)  # d tile

        bk = pid0
        b = bk // K
        k = bk % K

        h_off = pid1 * BLOCK_H + tl.arange(0, BLOCK_H)
        d_off = pid2 * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_h = h_off < H
        mask_d = d_off < D

        # load idx[b,k]
        idx_val = tl.load(IDX_ptr + b * S_idx_B + k * S_idx_K, mask=True).to(tl.int32)

        # load slices
        dy_ptr = DY_ptr + b * S_dy_B + k * S_dy_K + h_off * S_dy_H
        dy_val = tl.load(dy_ptr, mask=mask_h, other=0.0)                      # (BLOCK_H,)

        x_ptr = X_ptr + b * S_x_B + k * S_x_K + d_off * S_x_D
        x_val = tl.load(x_ptr, mask=mask_d, other=0.0)                         # (BLOCK_D,)

        # outer = dy[:,None] * x[None,:]  -> (BLOCK_H, BLOCK_D)
        outer = dy_val[:, None] * x_val[None, :]

        # atomic add to dW[idx_val, h_off, d_off]
        base = dW_ptr + idx_val * S_dW_N
        w_ptr = base + h_off[:, None] * S_dW_H + d_off[None, :] * S_dW_D
        tl.atomic_add(w_ptr, outer, mask=mask_h[:, None] & mask_d[None, :])


class _IndexedLinearFunction(torch.autograd.Function):
    """
    y[b,k,h] = < W[idx[b,k],h,:], x[b,k,:] >
    W: (N,H,D), x: (B,K,D), idx: (B,K) long/int
    """
    @staticmethod
    def forward(ctx, x, W, idx, block_d=128, block_h=64):
        assert x.is_cuda and W.is_cuda and idx.is_cuda, "Use CUDA tensors."
        B, K, D = x.shape
        N, H, Dw = W.shape
        assert D == Dw, "D mismatch"

        y = x.new_empty((B, K, H))

        if TRITON_AVAILABLE:
            grid = (B * K, (H + block_h - 1) // block_h)
            _indexed_linear_fwd[grid](
                x, W, idx.to(torch.int32), y,
                B, K, N, H, D,
                x.stride(0), x.stride(1), x.stride(2),
                W.stride(0), W.stride(1), W.stride(2),
                y.stride(0), y.stride(1), y.stride(2),
                idx.stride(0), idx.stride(1),
                BLOCK_D=block_d, BLOCK_H=block_h,
            )
        else:
            # PyTorch fallback（较慢）：按 H 分块避免峰值
            H_CHUNK = min(128, H)
            for hs in range(0, H, H_CHUNK):
                he = min(hs + H_CHUNK, H)
                W_h = W[:, hs:he, :]          # (N,hC,D)
                y[..., hs:he] = torch.einsum('bkd,bkhd->bkh', x, W_h[idx])

        ctx.save_for_backward(x, W, idx)
        ctx.block_d = block_d
        ctx.block_h = block_h
        return y

    @staticmethod
    def backward(ctx, dy):
        x, W, idx = ctx.saved_tensors
        block_d = ctx.block_d
        block_h = ctx.block_h
        B, K, D = x.shape
        N, H, D = W.shape

        if TRITON_AVAILABLE:
            # dX kernel
            dx = torch.empty_like(x)
            grid_dx = (B * K, (D + block_d - 1) // block_d)
            _indexed_linear_dx[grid_dx](
                dy, W, idx.to(torch.int32), dx,
                B, K, N, H, D,
                dy.stride(0), dy.stride(1), dy.stride(2),
                W.stride(0), W.stride(1), W.stride(2),
                idx.stride(0), idx.stride(1),
                dx.stride(0), dx.stride(1), dx.stride(2),
                BLOCK_H=block_h, BLOCK_D=block_d,
            )

            # dW kernel with atomics
            dW = torch.zeros_like(W)
            grid_dW = (B * K,
                       (H + block_h - 1) // block_h,
                       (D + block_d - 1) // block_d)
            _indexed_linear_dW[grid_dW](
                dy, x, idx.to(torch.int32), dW,
                B, K, N, H, D,
                dy.stride(0), dy.stride(1), dy.stride(2),
                x.stride(0),  x.stride(1),  x.stride(2),
                idx.stride(0), idx.stride(1),
                dW.stride(0), dW.stride(1), dW.stride(2),
                BLOCK_H=block_h, BLOCK_D=block_d,
            )
        else:
            # Fallback（分块，避免 OOM）
            dx = torch.zeros_like(x)
            H_CHUNK = min(128, H)
            for hs in range(0, H, H_CHUNK):
                he = min(hs + H_CHUNK, H)
                dY_h = dy[..., hs:he]    # (B,K,hC)
                W_h = W[:, hs:he, :]     # (N,hC,D)
                dx += torch.einsum('bkh,bkhd->bkd', dY_h, W_h[idx])

            dW = torch.zeros_like(W)
            dW_flat = dW.view(N, H * D)
            idx_flat = idx.reshape(-1)
            dY_flat = dy.reshape(B * K, H)
            X_flat  = x.reshape(B * K, D)
            CH = 65536
            for s in range(0, B * K, CH):
                e = min(s + CH, B * K)
                outer = torch.einsum('mh,md->mhd', dY_flat[s:e, :], X_flat[s:e, :]).reshape(e - s, H * D)
                dW_flat.index_add_(0, idx_flat[s:e], outer)

        return dx, dW, None, None, None


class IndexedLinear(nn.Module):
    """ nn.Linear 的索引化版本：权重库 (N,H,D)，按 idx[b,k] 选择一条权重。 """
    def __init__(self, num_primitives, in_features, out_features):
        super().__init__()
        self.N = num_primitives
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(self.N, self.out_features, self.in_features))
        self.bias   = nn.Parameter(torch.empty(self.N, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, idx):
        # x: (B,K,D), idx: (B,K)
        y = _IndexedLinearFunction.apply(x, self.weight, idx)  # (B,K,H)
        b = self.bias[idx]                                     # (B,K,H)
        return y + b


class IndexedMLP(nn.Module):
    """ 两层： encoded_dim -> H -> 4。输入 x_enc (B,K,Denc)，输出 (B,K,3/1) """
    def __init__(self, num_primitives, encoded_dim, hidden_dim=16):
        super().__init__()
        self.lin1 = IndexedLinear(num_primitives, encoded_dim, hidden_dim)
        self.lin2 = IndexedLinear(num_primitives, hidden_dim, 4)

    def forward(self, x_enc, idx):
        h = F.relu(self.lin1(x_enc, idx))
        out = self.lin2(h, idx)
        rgb = torch.sigmoid(out[..., :3])
        sigma = F.softplus(out[..., 3:4])
        return rgb, sigma


# =========================
# Scene with per-ray Top-K
# =========================

class NeuralPrimitiveScene(nn.Module):
    def __init__(self, num_primitives=1024, hidden_dim=16, pe_frequencies=10, topk=32, initial_positions=None, use_integrated_pe=True):
        super().__init__()
        self.num_primitives = num_primitives
        self.topk = topk

        # positions & scales
        if initial_positions is None:
            bound = 1.0
            self.positions = nn.Parameter(torch.rand(num_primitives, 3) * 2 * bound - bound)
            sampled_scales = torch.ones(num_primitives) * 0.1
        else:
            sampled_positions, sampled_scales = octree_sample(initial_positions)
            sampled_scales /= 10
            self.positions = nn.Parameter(sampled_positions.clone().detach())

        self.log_scales = nn.Parameter(torch.log(sampled_scales).unsqueeze(-1))

        # encoder (6 -> encoded_dim)
        in_dim = 6
        if use_integrated_pe:
            self.encoder = IntegratedPositionalEncoding(in_dim=in_dim, num_frequencies=pe_frequencies, sigma=0.01)
        else:
            self.encoder = FourierFeatureEncoding(in_dim=in_dim, num_frequencies=pe_frequencies)
        encoded_dim = self.encoder.out_dim

        # indexed MLP (no weight replication)
        self.mlps = IndexedMLP(num_primitives=num_primitives, encoded_dim=encoded_dim, hidden_dim=hidden_dim)

        # stats
        self.register_buffer("alpha_accum", torch.zeros(num_primitives))
        self.register_buffer("occupancy_accum", torch.zeros(num_primitives))
        self.invalid_nums = 0

    @torch.no_grad()
    def _topk_indices_by_decay(self, ray_o, ray_d, positions, radii, k):
        """
        使用 (B,N) 级矩阵计算每条 ray 到每个 primitive 的垂直距离平方：
          perp2 = ||c - o||^2 - ((c - o)·d)^2
        评分 score = exp( - perp2 / (2 r^2) )
        返回 per-ray Top-K 的索引 (B,K)。
        """
        B = ray_o.shape[0]
        # ||o||^2 and ||c||^2
        o2 = (ray_o * ray_o).sum(dim=1, keepdim=True)           # (B,1)
        c2 = (positions * positions).sum(dim=1, keepdim=True).T # (1,N)
        oc = ray_o @ positions.T                                # (B,N)
        delta2 = c2 + o2 - 2.0 * oc                             # (B,N)

        od = (ray_o * ray_d).sum(dim=1, keepdim=True)           # (B,1)
        cd = ray_d @ positions.T                                # (B,N)
        proj = cd - od                                          # (B,N)

        perp2 = torch.clamp(delta2 - proj * proj, min=0.0)      # (B,N)
        r2 = (radii.squeeze(-1) ** 2).unsqueeze(0)              # (1,N)
        score = torch.exp(- perp2 / (2.0 * (r2 + 1e-12)))       # (B,N)
        K = min(k, positions.shape[0])
        topk_inds = torch.topk(score, K, dim=1, largest=True)[1]  # (B,K)
        return topk_inds

    def forward(self, ray_o, ray_d, mlp_id=None):
        """
        - mlp_id is None：自动 per-ray TopK（B,K）
        - mlp_id shape == (B,K)：使用外部提供的 per-ray TopK
        - mlp_id shape == (M,) ：兼容旧式 (M,B)（内部转成 per-ray）
        """
        device = ray_o.device
        ray_d = F.normalize(ray_d, dim=-1)
        B = ray_o.shape[0]
        N = self.num_primitives
        scales = torch.exp(self.log_scales)  # (N,1)


        PROFILE = os.environ.get("MLP_PROFILE", "0") == "1"
        timer = CUDATimer(enabled=PROFILE)

        # ===== per-ray TopK 路径 =====
        # 1) Top-K 选择（基于衰减的打分）
        timer.start("topk_score+select")
        if (mlp_id is None) or (isinstance(mlp_id, torch.Tensor) and mlp_id.dim() == 2):
            with torch.no_grad():
                if mlp_id is None:
                    topk_inds = self._topk_indices_by_decay(ray_o, ray_d, self.positions, scales, self.topk)
                else:
                    topk_inds = mlp_id
            timer.stop("topk_score+select")

            # 2) 几何准备（t, dist, decay）
            timer.start("geometry")
            B, K = topk_inds.shape
            centers_bk = self.positions[topk_inds]                     # (B,K,3)
            vec = centers_bk - ray_o.unsqueeze(1)                      # (B,K,3)
            t = (vec * ray_d.unsqueeze(1)).sum(dim=-1, keepdim=True)   # (B,K,1)
            proj = ray_o.unsqueeze(1) + t * ray_d.unsqueeze(1)         # (B,K,3)
            dist_bk = centers_bk - proj                                # (B,K,3)
            scales_bk = scales.squeeze(-1)[topk_inds].unsqueeze(-1)    # (B,K,1)
            norm_bk = torch.norm(dist_bk, dim=-1, keepdim=True)        # (B,K,1)
            decay_bk = torch.exp(- (norm_bk ** 2) / (2.0 * (scales_bk ** 2) + 1e-12))
            timer.stop("geometry")

            # 3) 编码（PE）
            timer.start("encode_PE")
            ray_d_bk = ray_d.unsqueeze(1).expand(B, K, 3)
            x_bk = torch.cat([ray_d_bk, dist_bk], dim=-1)              # (B,K,6)
            x_enc = self.encoder(x_bk)                                  # (B,K,Denc)
            timer.stop("encode_PE")

            # 4) 索引化 MLP 前向（Triton indexed linear）
            timer.start("mlp_fwd")
            rgb_bk, sigma_raw_bk = self.mlps(x_enc, topk_inds)          # (B,K,3),(B,K,1)
            sigma_bk = sigma_raw_bk * decay_bk
            timer.stop("mlp_fwd")

            # 5) 深度排序 + gather
            timer.start("sort+gather")
            _, sort_idx = torch.topk(t.squeeze(-1), k=K, dim=1, largest=False, sorted=True)
            gather_rgb = torch.gather(rgb_bk, 1, sort_idx.unsqueeze(-1).expand(-1, -1, 3))
            gather_sigma = torch.gather(sigma_bk, 1, sort_idx.unsqueeze(-1))
            timer.stop("sort+gather")

            # 6) 体渲染合成
            timer.start("composite")
            alpha = 1.0 - torch.exp(-gather_sigma)
            trans = torch.cumprod(
                torch.cat([torch.ones(B,1,1,device=device), 1.0 - alpha + 1e-12], dim=1)[:, :-1],
                dim=1
            )
            blended = torch.sum(alpha * trans * gather_rgb, dim=1)
            timer.stop("composite")

            # 7) 统计更新（no grad）
            timer.start("stats")
            with torch.no_grad():
                alpha_sum_flat = alpha.squeeze(-1).reshape(-1)
                idx_flat = topk_inds.reshape(-1)
                added = torch.zeros_like(self.alpha_accum)
                added.index_add_(0, idx_flat, alpha_sum_flat)
                sel = torch.unique(idx_flat)
                old_sel = self.alpha_accum.index_select(0, sel)
                new_sel = 0.2 * old_sel + 0.8 * added.index_select(0, sel)
                self.alpha_accum.scatter_(0, sel, new_sel)
            timer.stop("stats")

            if PROFILE:  # 每次 forward 打一行
                timer.dump("FORWARD PROFILE")

            return blended


        # ===== 兼容旧式 (M,B)：把 M 复制到每条 ray 的 K=M，仍用 indexed 计算 =====
        else:
            mlp_id_1d = mlp_id            # (M,)
            M = mlp_id_1d.shape[0]
            # 统一转成 per-ray：(B,M)
            topk_inds = mlp_id_1d.unsqueeze(0).expand(B, M)   # (B,M)
            centers_bm = self.positions[mlp_id_1d]            # (M,3)
            vec = centers_bm.unsqueeze(0) - ray_o.unsqueeze(1)                # (B,M,3)
            t = (vec * ray_d.unsqueeze(1)).sum(dim=-1, keepdim=True)          # (B,M,1)
            proj = ray_o.unsqueeze(1) + t * ray_d.unsqueeze(1)                # (B,M,3)
            dist_bm = centers_bm.unsqueeze(0) - proj                           # (B,M,3)

            scales_bm = scales.squeeze(-1)[mlp_id_1d].unsqueeze(0).unsqueeze(-1)  # (1,M,1)
            norm_bm = torch.norm(dist_bm, dim=-1, keepdim=True)                    # (B,M,1)
            decay_bm = torch.exp(- (norm_bm ** 2) / (2.0 * (scales_bm ** 2) + 1e-12))

            ray_d_bm = ray_d.unsqueeze(1).expand(B, M, 3)
            x_bm = torch.cat([ray_d_bm, dist_bm], dim=-1)
            x_enc = self.encoder(x_bm)                                           # (B,M,Denc)

            rgb_bm, sigma_raw_bm = self.mlps(x_enc, topk_inds)                   # (B,M,3),(B,M,1)
            sigma_bm = sigma_raw_bm * decay_bm

            # 统计（不进图）
            with torch.no_grad():
                alpha_bm = 1.0 - torch.exp(-sigma_bm)                            # (B,M,1)
                alpha_sum = alpha_bm.squeeze(-1).sum(dim=0)                      # (M,)
                old = self.alpha_accum.index_select(0, mlp_id_1d)
                new = 0.2 * old + 0.8 * alpha_sum
                self.alpha_accum.scatter_(0, mlp_id_1d, new)

            # 深度排序 & 体渲染
            sort_idx = torch.argsort(t.squeeze(-1), dim=1)                       # (B,M)
            gather_rgb = torch.gather(rgb_bm, 1, sort_idx.unsqueeze(-1).expand(-1, -1, 3))
            gather_sigma = torch.gather(sigma_bm, 1, sort_idx.unsqueeze(-1))

            alpha = 1.0 - torch.exp(-gather_sigma)
            trans = torch.cumprod(
                torch.cat([torch.ones(B, 1, 1, device=device), 1.0 - alpha + 1e-12], dim=1)[:, :-1],
                dim=1
            )
            weights = alpha * trans
            blended = torch.sum(weights * gather_rgb, dim=1)                      # (B,3)
            return blended

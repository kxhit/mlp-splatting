import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.sampling import octree_sample

import triton
import triton.language as tl


# =========================
# Positional Encodings
# =========================
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


# ============================================================
# Triton: Pass-1 计数（按阈值统计每条 ray 的命中个数）
# ============================================================
@triton.jit
def _count_above_threshold(
    RAYO, RAYD, POS, LOGS, THRESH,    # (B,3), (B,3), (N,3), (N,1), scalar
    B: tl.constexpr, N: tl.constexpr,
    OUT_CNT,                          # (B,)
    S_ro_B, S_rd_B, S_pos_N, S_logs_N,
):
    pid = tl.program_id(0)
    if pid >= B:
        return

    # 读取该 ray
    rox = tl.load(RAYO + pid * S_ro_B + 0)
    roy = tl.load(RAYO + pid * S_ro_B + 1)
    roz = tl.load(RAYO + pid * S_ro_B + 2)
    rdx = tl.load(RAYD + pid * S_rd_B + 0)
    rdy = tl.load(RAYD + pid * S_rd_B + 1)
    rdz = tl.load(RAYD + pid * S_rd_B + 2)

    thr = tl.load(THRESH)
    cnt = 0

    for n in range(0, N):
        cx = tl.load(POS + n * S_pos_N + 0)
        cy = tl.load(POS + n * S_pos_N + 1)
        cz = tl.load(POS + n * S_pos_N + 2)
        logs = tl.load(LOGS + n * S_logs_N + 0)
        r = tl.exp(logs)
        r2 = r * r + 1e-12

        vx = cx - rox; vy = cy - roy; vz = cz - roz
        t = vx * rdx + vy * rdy + vz * rdz
        px = rox + t * rdx; py = roy + t * rdy; pz = roz + t * rdz
        dx = cx - px; dy = cy - py; dz = cz - pz
        perp2 = tl.maximum(dx*dx + dy*dy + dz*dz, 0.0)
        sc = tl.exp(-0.5 * perp2 / r2)

        if (sc > thr) and (t > 0):
            cnt += 1

    tl.store(OUT_CNT + pid, cnt)


# ============================================================
# Triton: Pass-2 写出（把满足阈值的记录写入平铺数组）
# ============================================================
@triton.jit
def _write_above_threshold(
    RAYO, RAYD, POS, LOGS, THRESH,
    B: tl.constexpr, N: tl.constexpr,
    OFFSETS,                           # (B,) 每条 ray 的起始偏移（独占前缀和）
    OUT_RAY, OUT_PRIM, OUT_T, OUT_DX, OUT_DY, OUT_DZ,
    S_ro_B, S_rd_B, S_pos_N, S_logs_N,
):
    pid = tl.program_id(0)
    if pid >= B:
        return

    # 读取该 ray
    rox = tl.load(RAYO + pid * S_ro_B + 0)
    roy = tl.load(RAYO + pid * S_ro_B + 1)
    roz = tl.load(RAYO + pid * S_ro_B + 2)
    rdx = tl.load(RAYD + pid * S_rd_B + 0)
    rdy = tl.load(RAYD + pid * S_rd_B + 1)
    rdz = tl.load(RAYD + pid * S_rd_B + 2)

    thr = tl.load(THRESH)
    base = tl.load(OFFSETS + pid)   # 该 ray 的输出起点
    c = 0                           # 局部计数器

    for n in range(0, N):
        cx = tl.load(POS + n * S_pos_N + 0)
        cy = tl.load(POS + n * S_pos_N + 1)
        cz = tl.load(POS + n * S_pos_N + 2)
        logs = tl.load(LOGS + n * S_logs_N + 0)
        r = tl.exp(logs)
        r2 = r * r + 1e-12

        vx = cx - rox; vy = cy - roy; vz = cz - roz
        t = vx * rdx + vy * rdy + vz * rdz
        px = rox + t * rdx; py = roy + t * rdy; pz = roz + t * rdz
        dx = cx - px; dy = cy - py; dz = cz - pz
        perp2 = tl.maximum(dx*dx + dy*dy + dz*dz, 0.0)
        sc = tl.exp(-0.5 * perp2 / r2)

        if (sc > thr) and (t > 0):
            out_i = base + c
            tl.store(OUT_RAY  + out_i, pid)
            tl.store(OUT_PRIM + out_i, n)
            tl.store(OUT_T    + out_i, t)
            tl.store(OUT_DX   + out_i, dx)
            tl.store(OUT_DY   + out_i, dy)
            tl.store(OUT_DZ   + out_i, dz)
            c += 1


# ============================================================
# Flat Indexed Layers / MLP
# ============================================================
class IndexedLinearFlat(nn.Module):
    """x (M,D), idx (M,) -> y (M,H) （纯 PyTorch 实现，简单稳妥）"""
    def __init__(self, num_primitives, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_primitives, out_features, in_features))
        self.bias   = nn.Parameter(torch.empty(num_primitives, out_features))
        self.in_features = in_features
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, idx):
        w = self.weight.index_select(0, idx)  # (M,H,D)
        b = self.bias.index_select(0, idx)    # (M,H)
        y = torch.einsum("md,mhd->mh", x, w) + b
        return y


class IndexedMLP(nn.Module):
    """Flat MLP: (M, Denc) + idx(M,) -> (M,3), (M,1)
       现在是两层隐藏层：enc -> H -> H -> 4
    """
    def __init__(self, num_primitives, encoded_dim, hidden_dim=16):
        super().__init__()
        self.lin1 = IndexedLinearFlat(num_primitives, encoded_dim, hidden_dim)   # enc -> H
        self.lin2 = IndexedLinearFlat(num_primitives, hidden_dim,  hidden_dim)   # H   -> H
        self.lin3 = IndexedLinearFlat(num_primitives, hidden_dim,  4)            # H   -> 4

    def forward(self, x_enc, idx):
        h1 = F.relu(self.lin1(x_enc, idx))
        h2 = F.relu(self.lin2(h1, idx))
        o  = self.lin3(h2, idx)
        rgb   = torch.sigmoid(o[:, :3])
        sigma = F.softplus(o[:, 3:4])
        return rgb, sigma



# ============================================================
# Scene
# ============================================================
class NeuralPrimitiveScene(nn.Module):
    def __init__(self, num_primitives=1024, hidden_dim=16, pe_frequencies=10,
                 initial_positions=None, use_integrated_pe=True):
        super().__init__()
        self.num_primitives = num_primitives

        # positions & scales
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

        # encoder
        if use_integrated_pe:
            self.encoder = IntegratedPositionalEncoding(in_dim=6, num_frequencies=pe_frequencies, sigma=0.01)
        else:
            self.encoder = FourierFeatureEncoding(in_dim=6, num_frequencies=pe_frequencies)

        # flat MLP
        self.mlps = IndexedMLP(num_primitives, self.encoder.out_dim, hidden_dim)

        # stats
        self.register_buffer("alpha_accum", torch.zeros(num_primitives))

    # ---------- variable-K 主流程 ----------
    @torch.no_grad()
    def _select_variable_k(self, ray_o, ray_d, threshold: float):
        device = ray_o.device
        B = ray_o.shape[0]
        N = self.num_primitives

        ray_d = F.normalize(ray_d, dim=-1)

        # Pass-1
        counts = torch.empty(B, dtype=torch.int32, device=device)
        _count_above_threshold[(B,)](
            ray_o, ray_d, self.positions, self.log_scales,
            torch.tensor(threshold, device=device, dtype=torch.float32),
            B, N,
            counts,
            ray_o.stride(0), ray_d.stride(0),
            self.positions.stride(0), self.log_scales.stride(0),
        )

        seg_len = counts.to(torch.int64)
        if B == 1:
            offsets = torch.zeros_like(seg_len)
        else:
            offsets = torch.empty_like(seg_len)
            offsets[0] = 0
            offsets[1:] = torch.cumsum(seg_len[:-1], dim=0)

        M = int(seg_len.sum().item())
        if M == 0:
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.float32, device=device),
                torch.empty(0, 3, dtype=torch.float32, device=device),
                seg_len, offsets, ray_d,
            )

        # Pass-2
        ray_ids  = torch.empty(M, dtype=torch.int64, device=device)
        prim_ids = torch.empty(M, dtype=torch.int64, device=device)
        t_m      = torch.empty(M, dtype=torch.float32, device=device)
        dx_m     = torch.empty(M, dtype=torch.float32, device=device)
        dy_m     = torch.empty(M, dtype=torch.float32, device=device)
        dz_m     = torch.empty(M, dtype=torch.float32, device=device)

        _write_above_threshold[(B,)](
            ray_o, ray_d, self.positions, self.log_scales,
            torch.tensor(threshold, device=device, dtype=torch.float32),
            B, N,
            offsets,
            ray_ids, prim_ids, t_m, dx_m, dy_m, dz_m,
            ray_o.stride(0), ray_d.stride(0),
            self.positions.stride(0), self.log_scales.stride(0),
        )

        dist_vec = torch.stack([dx_m, dy_m, dz_m], dim=-1)
        return ray_ids, prim_ids, t_m, dist_vec, seg_len, offsets, ray_d

    def forward(self, ray_o, ray_d, score_threshold=1e-4, bg_color=None):
        B = ray_o.shape[0]
        device = ray_o.device

        ray_ids, prim_ids, t_m, dist_m, seg_len, offsets, ray_d_n = \
            self._select_variable_k(ray_o, ray_d, float(score_threshold))

        if prim_ids.numel() == 0:
            return torch.zeros(B, 3, device=device)

        # ---- sort ----
        t_shift = t_m - t_m.min()
        t_norm = (t_shift / (t_shift.max() + 1e-9) * (2**32 - 1)).to(torch.int64)
        key = (ray_ids.to(torch.int64) << 32) | t_norm
        order = torch.argsort(key)

        ray_ids = ray_ids[order]
        prim_ids = prim_ids[order]
        dist_m = dist_m[order]

        # ---- MLP ----
        rd_m = ray_d_n[ray_ids]
        x_enc = self.encoder(torch.cat([rd_m, dist_m], dim=-1))
        rgb_m, sigma_raw_m = self.mlps(x_enc, prim_ids)

        scales = torch.exp(self.log_scales).squeeze(-1)
        s_m = scales.index_select(0, prim_ids).unsqueeze(-1)
        decay = torch.exp(- (torch.norm(dist_m, dim=-1, keepdim=True) ** 2) /
                          (2.0 * (s_m ** 2) + 1e-12))
        sigma = sigma_raw_m * decay

        alpha = 1.0 - torch.exp(-sigma)

        s = torch.log(1.0 - alpha + 1e-12).squeeze(-1)
        cs = torch.cumsum(s, dim=0)
        cs_before = torch.zeros_like(cs); cs_before[1:] = cs[:-1]
        start_for_pos = offsets[ray_ids]
        baseline = cs_before.gather(0, start_for_pos)
        trans = torch.exp(cs_before - baseline).unsqueeze(-1)
        weights = alpha * trans

        out = torch.zeros(B, 3, device=device)
        out.index_add_(0, ray_ids, weights * rgb_m)

        # Optional background compositing: add remaining transmittance onto bg_color
        if bg_color is not None:
            if not torch.is_tensor(bg_color):
                bg_color = torch.tensor(bg_color, dtype=out.dtype, device=device)
            if bg_color.numel() == 1:
                bg_color = bg_color.repeat(3)
            # sum of weights per ray gives accumulated opacity
            sum_w = torch.zeros(B, 1, device=device)
            sum_w.index_add_(0, ray_ids, weights)
            out = out + (1.0 - sum_w).clamp(0.0, 1.0) * bg_color.view(1, 3)

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
        """
        渲染给定相机位姿下的颜色图和深度图
        ray_o, ray_d: 世界坐标系下的射线起点和方向 (B,3)
        T_wc: (3,4) 相机 -> 世界 的齐次变换矩阵
        返回:
            out:   (B,3) 颜色
            depth: (B,1) 相机坐标系下的深度 (z)
        """
        B = ray_o.shape[0]
        device = ray_o.device

        ray_ids, prim_ids, t_m, dist_m, seg_len, offsets, ray_d_n = \
            self._select_variable_k(ray_o, ray_d, float(score_threshold))

        if prim_ids.numel() == 0:
            return torch.zeros(B, 3, device=device), torch.zeros(B, 1, device=device)

        # ---- sort ----
        t_shift = t_m - t_m.min()
        t_norm = (t_shift / (t_shift.max() + 1e-9) * (2**32 - 1)).to(torch.int64)
        key = (ray_ids.to(torch.int64) << 32) | t_norm
        order = torch.argsort(key)

        ray_ids = ray_ids[order]
        prim_ids = prim_ids[order]
        dist_m = dist_m[order]
        t_m = t_m[order]
        

        # ---- MLP ----
        rd_m = ray_d_n[ray_ids]
        x_enc = self.encoder(torch.cat([rd_m, dist_m], dim=-1))
        rgb_m, sigma_raw_m = self.mlps(x_enc, prim_ids)

        scales = torch.exp(self.log_scales).squeeze(-1)
        s_m = scales.index_select(0, prim_ids).unsqueeze(-1)
        decay = torch.exp(- (torch.norm(dist_m, dim=-1, keepdim=True) ** 2) /
                        (2.0 * (s_m ** 2) + 1e-12))
        sigma = sigma_raw_m * decay

        alpha = 1.0 - torch.exp(-sigma)

        s = torch.log(1.0 - alpha + 1e-12).squeeze(-1)
        cs = torch.cumsum(s, dim=0)
        cs_before = torch.zeros_like(cs); cs_before[1:] = cs[:-1]
        start_for_pos = offsets[ray_ids]
        baseline = cs_before.gather(0, start_for_pos)
        trans = torch.exp(cs_before - baseline).unsqueeze(-1)
        weights = alpha * trans


        # ---- Depth compositing (相机坐标系 z) ----
        # 交点的世界坐标
        p_m = ray_o[ray_ids] + t_m.unsqueeze(-1) * ray_d_n[ray_ids]  # (M,3)
        # 齐次坐标
        p_m_h = torch.cat([p_m, torch.ones_like(p_m[:, :1])], dim=-1)  # (M,4)
        # 世界 -> 相机
        pose = torch.eye(4, device=device)
        pose[:3, ] = T_wc
        T_cw = torch.inverse(pose)
        p_c_h = (T_cw @ p_m_h.T).T  # (M,4)
        depth_m = p_c_h[:, 2:3]     # (M,1) 相机系 z 分量

        depth = torch.zeros(B, 1, device=device)
        depth.index_add_(0, ray_ids, weights * depth_m)

        return depth


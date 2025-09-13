import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.sampling import octree_sample


class IntegratedPositionalEncoding(nn.Module):
    def __init__(self, in_dim=4, num_frequencies=10, sigma=0.01):
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
    def __init__(self, in_dim=4, num_frequencies=10):
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


class BatchedPrimitiveMLP(nn.Module):
    def __init__(self, num_primitives, input_dim=6, hidden_dim=16, num_frequencies=10, use_integrated_pe=True):
        super().__init__()
        self.num_primitives = num_primitives
        if use_integrated_pe:
            self.encoder = IntegratedPositionalEncoding(input_dim, num_frequencies)
        else:
            self.encoder = FourierFeatureEncoding(input_dim, num_frequencies)

        # use encoder.out_dim directly
        self.encoded_dim = self.encoder.out_dim
        self.hidden_dim = hidden_dim

        self.W1 = nn.Parameter(torch.empty(num_primitives, hidden_dim, self.encoded_dim))
        self.b1 = nn.Parameter(torch.empty(num_primitives, hidden_dim))
        self.W2 = nn.Parameter(torch.empty(num_primitives, 4, hidden_dim))
        self.b2 = nn.Parameter(torch.empty(num_primitives, 4))
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        fan_in = self.W1.size(2)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.b1, -bound, bound)

        nn.init.xavier_uniform_(self.W2)
        fan_in2 = self.W2.size(2)
        bound2 = 1 / math.sqrt(fan_in2)
        nn.init.uniform_(self.b2, -bound2, bound2)


    def forward(self, ray_d, dist, idx=None):
        B = ray_d.shape[0]
        if idx is None:
            W1, b1, W2, b2 = self.W1, self.b1, self.W2, self.b2
            M = self.num_primitives
        else:
            W1 = self.W1.index_select(0, idx)
            b1 = self.b1.index_select(0, idx)
            W2 = self.W2.index_select(0, idx)
            b2 = self.b2.index_select(0, idx)
            M = idx.shape[0]

        ray_d_mb = ray_d.unsqueeze(0).expand(M, B, 3)
        x = torch.cat([ray_d_mb, dist], dim=-1)           # (M,B,6)
        x_enc = self.encoder(x)                           # (M,B,encoded_dim)

        h = torch.einsum('mbd,mhd->mbh', x_enc, W1) + b1.unsqueeze(1)   # (M,B,H)
        h = F.relu(h)
        out = torch.einsum('mbh,moh->mbo', h, W2) + b2.unsqueeze(1)     # (M,B,4)

        rgb = torch.sigmoid(out[..., :3])   # (M,B,3)
        sigma = F.softplus(out[..., 3:4])   # (M,B,1)
        return rgb, sigma


class NeuralPrimitiveScene(nn.Module):
    def __init__(self, num_primitives=1024, hidden_dim=16, pe_frequencies=10, topk=128, initial_positions=None):
        super().__init__()
        self.num_primitives = num_primitives
        if initial_positions is None:
            bound = 1.0
            self.positions = nn.Parameter(torch.rand(num_primitives, 3) * 2 * bound - bound)
            sampled_scales = torch.ones(num_primitives) * 0.1
        else:
            sampled_positions, sampled_scales = octree_sample(initial_positions)
            sampled_scales /= 10
            self.positions = nn.Parameter(sampled_positions.clone().detach())

        self.log_scales = nn.Parameter(torch.log(sampled_scales).unsqueeze(-1))
        self.mlps = BatchedPrimitiveMLP(
            num_primitives=num_primitives,
            input_dim=6,
            hidden_dim=hidden_dim,
            num_frequencies=pe_frequencies,
            use_integrated_pe=True,
        )

        self.register_buffer("alpha_accum", torch.zeros(num_primitives))
        self.register_buffer("occupancy_accum", torch.zeros(num_primitives))
        self.invalid_nums = 0
        self.topk = topk

    def forward(self, ray_o, ray_d, mlp_id=None):
        self.invalid_nums = 0
        B = ray_o.shape[0]
        N = self.num_primitives
        device = ray_o.device
        ray_d = F.normalize(ray_d, dim=-1)

        ray_o_nb = ray_o.unsqueeze(0).expand(N, B, 3)
        ray_d_nb = ray_d.unsqueeze(0).expand(N, B, 3)
        centers = self.positions[:, None, :].expand(N, B, 3)

        vec = centers - ray_o_nb
        t = torch.sum(vec * ray_d_nb, dim=-1, keepdim=True)
        proj = ray_o_nb + t * ray_d_nb
        vec_perp = centers - proj

        scales = torch.exp(self.log_scales)

        if mlp_id is None:
            mlp_id = torch.arange(N, device=device)
        M = mlp_id.shape[0]

        dist = vec_perp.index_select(0, mlp_id)       # (M,B,3)
        t_sel = t.index_select(0, mlp_id).squeeze(-1) # (M,B)
        scales_sel = scales.index_select(0, mlp_id)   # (M,1)

        norm = torch.norm(dist, dim=-1, keepdim=True) # (M,B,1)
        # mask = norm <= scales_sel.unsqueeze(1)      # (M,B,1)  # 暂时不用

        decay = torch.exp(- (norm ** 2) / (2.0 * (scales_sel.unsqueeze(1) ** 2 + 1e-12)))

        rgb, sigma_raw = self.mlps(ray_d, dist, idx=mlp_id)
        sigma = sigma_raw * decay

        with torch.no_grad():
            alpha = 1.0 - torch.exp(-sigma)
            # alpha_sum = (alpha.squeeze(-1) * mask.squeeze(-1).float()).sum(dim=1)  # (M,)  # 注释掉 mask
            alpha_sum = alpha.squeeze(-1).sum(dim=1)  # 简化版
            old = self.alpha_accum.index_select(0, mlp_id)
            new = 0.2 * old + 0.8 * alpha_sum
            self.alpha_accum.scatter_(0, mlp_id, new)

        sorted_inds = torch.argsort(t_sel, dim=0)  # (M,B)
        gather_rgb = torch.gather(rgb, 0, sorted_inds.unsqueeze(-1).expand(-1, -1, 3))
        gather_sigma = torch.gather(sigma, 0, sorted_inds.unsqueeze(-1))

        alpha_sorted = 1.0 - torch.exp(-gather_sigma)
        trans = torch.cumprod(
            torch.cat([torch.ones(1, B, 1, device=device), 1.0 - alpha_sorted + 1e-12], dim=0)[:-1],
            dim=0
        )
        weights = alpha_sorted * trans
        blended = torch.sum(weights * gather_rgb, dim=0)  # (B,3)
        return blended

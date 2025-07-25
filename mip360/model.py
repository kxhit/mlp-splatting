import torch
import torch.nn as nn
import torch.nn.functional as F


class IntegratedPositionalEncoding(nn.Module):
    def __init__(self, in_dim=4, num_frequencies=10, sigma=0.01):
        super().__init__()
        self.sigma2 = sigma ** 2
        freq_bands = 2. ** torch.linspace(0., num_frequencies - 1, num_frequencies) * torch.pi
        self.register_buffer("freq_bands", freq_bands)

    def forward(self, x):
        # x: (..., in_dim)
        x_proj = x[..., None] * self.freq_bands  # (..., in_dim, num_freq)
        attenuation = torch.exp(-0.5 * (self.freq_bands ** 2) * self.sigma2)  # (num_freq,)
        x_proj = x_proj * attenuation  # broadcast to (..., in_dim, num_freq)
        x_proj = x_proj.view(*x.shape[:-1], -1)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    

class FourierFeatureEncoding(nn.Module):
    def __init__(self, in_dim=4, num_frequencies=10):
        super().__init__()
        freq_bands = 2. ** torch.linspace(0., num_frequencies - 1, num_frequencies)
        self.register_buffer("freq_bands", freq_bands)

    def forward(self, x):
        x_proj = x[..., None] * self.freq_bands  # uses already-on-device buffer
        x_proj = x_proj.view(*x.shape[:-1], -1)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class PrimitiveMLP(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, num_frequencies=10):
        super().__init__()
        #self.encoder = FourierFeatureEncoding(input_dim, num_frequencies)
        self.encoder = IntegratedPositionalEncoding(input_dim, num_frequencies)

        self.net = nn.Sequential(
            nn.Linear(input_dim * num_frequencies * 2, hidden_dim),
            nn.ReLU(),
            #nn.Linear(hidden_dim, hidden_dim),
            #nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # RGB + density
        )

    def forward(self, ray_d, dist):
        x = torch.cat([ray_d, dist], dim=-1)
        x_encoded = self.encoder(x)
        raw_out = self.net(x_encoded)
        rgb = torch.sigmoid(raw_out[..., :3])  # [0, 1]
        sigma = F.softplus(raw_out[..., 3:])  # >= 0
        # # make sigma binary, 0 or 1
        # sigma = F.sigmoid(raw_out[..., 3:])
        # sigma = (sigma > 0.5).float()  # Binary sigma
        return torch.cat([rgb, sigma], dim=-1)


class NeuralPrimitiveScene(nn.Module):
    def __init__(self, num_primitives=1024, hidden_dim=16, pe_frequencies=10, initial_positions=None):
        super().__init__()
        self.num_primitives = num_primitives


        if initial_positions == None:
            bound = 1.0
            # self.positions = nn.Parameter(torch.randn(num_primitives, 3))
            self.positions = nn.Parameter(
                torch.rand(num_primitives, 3) * 2 * bound - bound
            )
        else:
            # 从 initial_positions 中随机采样 num_primitives 个点
            assert initial_positions.shape[1] == 3, "initial_positions should be of shape (N, 3)"
            N = initial_positions.shape[0]
            assert N >= self.num_primitives, f"initial_positions must have at least {self.num_primitives} points"

            indices = torch.randperm(N)[:self.num_primitives]
            sampled_positions = initial_positions[indices]
            self.positions = nn.Parameter(sampled_positions.clone().detach())  # make it a learnable parameter

        #self.scales = nn.Parameter(torch.ones(num_primitives, 1) * 0.1)  # Learnable per-primitive scale
        self.log_scales = nn.Parameter(torch.log(torch.ones(num_primitives, 1) * 0.1))


        self.mlps = nn.ModuleList([
            PrimitiveMLP(input_dim=6, hidden_dim=hidden_dim, num_frequencies=pe_frequencies)
            for _ in range(num_primitives)
        ])

        self.register_buffer("alpha_accum", torch.zeros(num_primitives))
        self.register_buffer("occupancy_accum", torch.zeros(num_primitives))

        self.invalid_nums = 0


    
    def forward(self, ray_o, ray_d):
        self.invalid_nums = 0
        B = ray_o.shape[0]
        N = self.num_primitives

        ray_d = F.normalize(ray_d, dim=-1)
        ray_o_expand = ray_o.unsqueeze(0).expand(N, B, 3)
        ray_d_expand = ray_d.unsqueeze(0).expand(N, B, 3)
        center_expand = self.positions[:, None, :].expand(N, B, 3)

        vec = center_expand - ray_o_expand
        t = torch.sum(vec * ray_d_expand, dim=-1, keepdim=True)
        proj = ray_o_expand + t * ray_d_expand
        vec_perp = center_expand - proj

        z_vals = t.squeeze(-1)
        sorted_inds = torch.argsort(z_vals, dim=0)

        rgb_list = []
        sigma_list = []

        scales = torch.exp(self.log_scales)

        for i in range(N):
            dist = vec_perp[i]  # (B, 3)
            norm = torch.norm(dist, dim=-1, keepdim=True)  # (B, 1)

            mask = norm <= scales[i]  # (B, 1)

            if not mask.any():
                # If no points are within the scale, return zero RGB and sigma
                #print(f"Primitive {i} has no points within scale, returning zeros.")
                self.invalid_nums += 1
                #rgb_list.append(torch.zeros(B, 3, device=ray_o.device))
                #sigma_list.append(torch.zeros(B, 1, device=ray_o.device))
                # TODO log the alpha for pruning

                #continue


            out = self.mlps[i](ray_d, dist)

            # TODO 
            decay = torch.exp(-norm ** 2 / (2 * scales[i] ** 2))  # (B, 1)

            sigma = out[:, 3:] * decay  # soft bound
            rgb = out[:, :3]

            #valid_mask = mask.float()
            # 用 mask 选择有效的输出，同时保留梯度
            #sigma = sigma * valid_mask #+ (1 - valid_mask) * sigma.detach() 
            #rgb = rgb * valid_mask #+ (1 - valid_mask) * rgb.detach()


            rgb_list.append(rgb)
            sigma_list.append(sigma)

            with torch.no_grad():
                alpha = 1.0 - torch.exp(-sigma)
                alpha_sum = (alpha.squeeze(-1) * mask.float().squeeze(-1)).sum()
                occ = mask.float().sum()
                decay = 0.9
                self.alpha_accum[i] = (1 - decay) * self.alpha_accum[i] + decay * alpha_sum.detach()
                self.occupancy_accum[i] = (1 - decay) * self.occupancy_accum[i] + decay * occ.detach()

        rgb = torch.stack(rgb_list, dim=0)
        sigma = torch.stack(sigma_list, dim=0)

        rgb_sorted = torch.gather(rgb, 0, sorted_inds.unsqueeze(-1).expand(-1, -1, 3))
        sigma_sorted = torch.gather(sigma, 0, sorted_inds.unsqueeze(-1))

        alpha = 1.0 - torch.exp(-sigma_sorted)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones(1, B, 1, device=alpha.device), 1.0 - alpha + 1e-10], dim=0)[:-1],
            dim=0
        )
        blended = torch.sum(weights * rgb_sorted, dim=0)
        return blended
    

    """
    def forward(self, ray_o, ray_d):
        self.invalid_nums = 0
        B = ray_o.shape[0]
        N = self.num_primitives

        ray_d = F.normalize(ray_d, dim=-1)
        scales = torch.exp(self.log_scales)  # [N, 1]

        rgb_list = []
        sigma_list = []

        # Expand to [N, B, 3] for vectorized geometric computation
        ray_o_expand = ray_o.unsqueeze(0).expand(N, B, 3)
        ray_d_expand = ray_d.unsqueeze(0).expand(N, B, 3)
        center_expand = self.positions[:, None, :].expand(N, B, 3)

        vec = center_expand - ray_o_expand  # [N, B, 3]
        t = torch.sum(vec * ray_d_expand, dim=-1, keepdim=True)  # [N, B, 1]
        proj = ray_o_expand + t * ray_d_expand
        vec_perp = center_expand - proj  # [N, B, 3]
        norm = torch.norm(vec_perp, dim=-1, keepdim=True)  # [N, B, 1]
        z_vals = t.squeeze(-1)  # [N, B]

        sorted_inds = torch.argsort(z_vals, dim=0)  # sort along N

        for i in range(N):
            dist_i = vec_perp[i]           # [B, 3]
            norm_i = norm[i]               # [B, 1]
            scale_i = scales[i]            # scalar
            mask = (norm_i <= scale_i)     # [B, 1]
            valid_idx = mask.squeeze(-1).nonzero(as_tuple=False).squeeze(-1)  # 1D idx list

            if valid_idx.numel() == 0:
                self.invalid_nums += 1
                rgb_list.append(torch.zeros(B, 3, device=ray_o.device))
                sigma_list.append(torch.zeros(B, 1, device=ray_o.device))
                continue

            # Select valid rays only
            ray_d_sub = ray_d[valid_idx]       # [Bᵢ, 3]
            dist_sub = dist_i[valid_idx]       # [Bᵢ, 3]
            norm_sub = norm_i[valid_idx]       # [Bᵢ, 1]

            out = self.mlps[i](ray_d_sub, dist_sub)  # [Bᵢ, 4]
            rgb_sub = out[:, :3]
            sigma_sub = out[:, 3:] * torch.exp(-norm_sub ** 2 / (2 * scale_i ** 2))  # [Bᵢ, 1]

            # Scatter back to full B-length tensors
            rgb_full = torch.zeros(B, 3, device=ray_o.device)
            sigma_full = torch.zeros(B, 1, device=ray_o.device)
            rgb_full[valid_idx] = rgb_sub
            sigma_full[valid_idx] = sigma_sub

            rgb_list.append(rgb_full)
            sigma_list.append(sigma_full)

            with torch.no_grad():
                alpha = 1.0 - torch.exp(-sigma_sub)  # [Bᵢ, 1]
                alpha_sum = alpha.sum()
                occ = valid_idx.numel()
                decay = 0.9
                self.alpha_accum[i] = (1 - decay) * self.alpha_accum[i] + decay * alpha_sum.detach()
                self.occupancy_accum[i] = (1 - decay) * self.occupancy_accum[i] + decay * occ

        rgb = torch.stack(rgb_list, dim=0)       # [N, B, 3]
        sigma = torch.stack(sigma_list, dim=0)   # [N, B, 1]

        # Sort along z for each ray
        rgb_sorted = torch.gather(rgb, 0, sorted_inds.unsqueeze(-1).expand(-1, -1, 3))       # [N, B, 3]
        sigma_sorted = torch.gather(sigma, 0, sorted_inds.unsqueeze(-1))                    # [N, B, 1]

        alpha = 1.0 - torch.exp(-sigma_sorted)   # [N, B, 1]
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones(1, B, 1, device=alpha.device), 1.0 - alpha + 1e-10], dim=0)[:-1],
            dim=0
        )
        blended = torch.sum(weights * rgb_sorted, dim=0)  # [B, 3]

        return blended
    """



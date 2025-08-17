import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree


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

        # ======= Inside your class =======
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

    def farthest_point_sampling(self, points, num_samples):
        """
        Farthest Point Sampling (FPS)
        Args:
            points: (N, 3) input point cloud
            num_samples: number of samples to select
        Returns:
            indices: (num_samples,) indices of selected points
        """
        N = points.shape[0]
        selected_inds = torch.zeros(num_samples, dtype=torch.long, device=points.device)
        distances = torch.full((N,), float('inf'), device=points.device)

        # randomly pick first point
        farthest = torch.randint(0, N, (1,), device=points.device)
        selected_inds[0] = farthest

        for i in range(1, num_samples):
            dist = torch.cdist(points[farthest].unsqueeze(0), points, p=2).squeeze(0)
            distances = torch.minimum(distances, dist)
            farthest = torch.argmax(distances)
            selected_inds[i] = farthest

        return selected_inds


    
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


            """
            if scales[i] > 0.005: #or scales[i] > 200:
                #print(f"Primitive {i} has scale < 0.001, skipping.")
                rgb_list.append(torch.zeros(B, 3, device=ray_o.device))
                sigma_list.append(torch.zeros(B, 1, device=ray_o.device))
                continue
            """

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

            # TODO 
            with torch.no_grad():
                alpha = 1.0 - torch.exp(-sigma)
                alpha_mean = (alpha.squeeze(-1) * mask.float().squeeze(-1)).sum()
                self.alpha_accum[i] = 0.2 * self.alpha_accum[i] + 0.8 * alpha_mean.detach()

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
    

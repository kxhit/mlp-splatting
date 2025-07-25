import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.func import vmap, functional_call

class FourierFeatureEncoding(nn.Module):
    def __init__(self, in_dim=4, num_frequencies=10):
        super().__init__()
        self.freq_bands = 2. ** torch.linspace(0., num_frequencies - 1, num_frequencies)

    def forward(self, x):
        x_proj = x[..., None] * self.freq_bands.to(x.device)
        x_proj = x_proj.view(*x.shape[:-1], -1)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class PrimitiveMLP(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, num_frequencies=10):
        super().__init__()
        self.encoder = FourierFeatureEncoding(input_dim, num_frequencies)
        self.net = nn.Sequential(
            nn.Linear(input_dim * num_frequencies * 2, hidden_dim),
            nn.ReLU(),
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
    def __init__(self, num_primitives=2, hidden_dim=64, pe_frequencies=6):
        super().__init__()
        self.num_primitives = num_primitives
        bound = 1.0
        self.positions = nn.Parameter(torch.rand(num_primitives, 3) * 2 * bound - bound)
        self.scales = nn.Parameter(torch.ones(num_primitives, 1) * 0.5)

        template = PrimitiveMLP(input_dim=6, hidden_dim=hidden_dim, num_frequencies=pe_frequencies)
        self.shared_model = template
        self.buffer_dict = dict(template.named_buffers())

        # Per-primitive parameters
        self.param_keys = list(dict(template.named_parameters()).keys())
        self.primitive_params = nn.ParameterDict()
        for key in self.param_keys:
            safe_key = key.replace('.', '_')
            param_tensor = torch.stack([dict(template.named_parameters())[key].clone()
                                        for _ in range(num_primitives)])
            self.primitive_params[safe_key] = nn.Parameter(param_tensor)
        self.reverse_map = {k.replace('.', '_'): k for k in self.param_keys}

    def forward(self, ray_o, ray_d):
        B = ray_o.shape[0]
        N = self.num_primitives

        # Expand rays and primitives
        ray_o_exp = ray_o.unsqueeze(0).expand(N, B, 3)
        ray_d_exp = F.normalize(ray_d.unsqueeze(0).expand(N, B, 3), dim=-1)
        pos_exp = self.positions[:, None, :].expand(N, B, 3)

        vec = pos_exp - ray_o_exp
        t = torch.sum(vec * ray_d_exp, dim=-1, keepdim=True)
        proj = ray_o_exp + t * ray_d_exp
        dist_vec = pos_exp - proj
        dist_norm = torch.norm(dist_vec, dim=-1)  # (N, B)

        mask = dist_norm <= self.scales.view(-1, 1)  # (N, B)
        ray_idx, prim_idx = torch.nonzero(mask, as_tuple=True)

        # Select sparse inputs
        ray_d_sparse = ray_d[ray_idx]
        dist_sparse = dist_vec[prim_idx, ray_idx]

        # Batched per-primitive parameters
        param_map = prim_idx
        param_dict = {
            self.reverse_map[k]: self.primitive_params[k][param_map]
            for k in self.reverse_map
        }

        def fwd(params, ray_d_i, dist_i):
            return functional_call(self.shared_model, (params, self.buffer_dict), (ray_d_i, dist_i))

        out = vmap(fwd)(param_dict, ray_d_sparse, dist_sparse)
        rgb, sigma = out[:, :3], out[:, 3:]
        alpha = 1.0 - torch.exp(-sigma)

        # Basic alpha compositing
        blended = torch.zeros(B, 3, device=ray_o.device)
        accum = torch.zeros(B, 1, device=ray_o.device)

        for i in range(ray_idx.shape[0]):
            w = alpha[i] * (1.0 - accum[ray_idx[i]])
            blended[ray_idx[i]] += w * rgb[i]
            accum[ray_idx[i]] += w

        return blended




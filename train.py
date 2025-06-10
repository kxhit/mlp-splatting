import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from model import NeuralPrimitiveScene
# from vmodel import NeuralPrimitiveScene
from llff_loader import LLFFDataset
from tqdm import tqdm
from save_model import save_model
from visualize import evaluate_and_visualize

import copy

def train_scene_llff(data_dir, num_epochs=10, batch_size=4096, lr=1e-3, device='cuda'):
    dataset = LLFFDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    scene = NeuralPrimitiveScene().to(device)
    optimizer = torch.optim.Adam(scene.parameters(), lr=lr)

    for epoch in tqdm(range(num_epochs)):
        for rays_o, rays_d, rgbs in dataloader:
            # rays_o, rays_d, rgbs: [B, H*W, 3]
            rays_o = rays_o.view(-1, 3).to(device)  # [B*H*W, 3]
            rays_d = rays_d.view(-1, 3).to(device)
            rgbs = rgbs.view(-1, 3).to(device)

            assert batch_size <= rays_o.size(0), "Batch size must be less than or equal to the number of rays."
            rand_idx = torch.randperm(rays_o.size(0))[:batch_size]
            rays_o_batch = rays_o[rand_idx]
            rays_d_batch = rays_d[rand_idx]
            rgbs_batch = rgbs[rand_idx]

            pred = scene(rays_o_batch, rays_d_batch)
            loss = F.mse_loss(pred, rgbs_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[Epoch {epoch+1}] Loss: {loss.item():.6f}")

        # Prune & densify primitives
        if epoch % 25 == 0 and epoch > 50:
            prune_primitives(scene)
            densify_primitives(scene, num_new=16)
            print(f"After pruning and densifying, scene has {scene.num_primitives} primitives.")

        # Eval after each epoch (first image only)
        if (epoch + 1) % 50 == 0 or epoch == num_epochs - 1:
            torch.save(scene.state_dict(), "scene_ckpt_tmp.pt")
            evaluate_and_visualize(data_dir, model_ckpt=scene.state_dict(), N=scene.num_primitives, device=device)

    save_model(scene, "scene_ckpt.pt")


def prune_primitives(scene, alpha_thresh=0.01, occupancy_thresh=0.01):
    alpha = scene.alpha_accum
    occ = scene.occupancy_accum

    keep_mask = (alpha > alpha_thresh) & (occ > occupancy_thresh)
    keep_indices = torch.nonzero(keep_mask).squeeze(-1)

    print(f"Pruned {scene.num_primitives - keep_indices.numel()} primitives.")

    scene.positions = nn.Parameter(scene.positions[keep_indices])
    scene.scales = nn.Parameter(scene.scales[keep_indices])
    scene.mlps = nn.ModuleList([scene.mlps[i] for i in keep_indices])
    scene.num_primitives = keep_indices.numel()
    scene.alpha_accum = scene.alpha_accum[keep_indices]
    scene.occupancy_accum = scene.occupancy_accum[keep_indices]

def densify_primitives(scene, num_new=32, noise_scale=0.01):
    topk = min(num_new, scene.num_primitives)
    top_indices = torch.randperm(scene.num_primitives)[:topk]

    new_positions = scene.positions[top_indices] + noise_scale * torch.randn_like(scene.positions[top_indices])
    new_scales = scene.scales[top_indices].clone()
    new_mlps = [copy.deepcopy(scene.mlps[i]) for i in top_indices]

    scene.positions = nn.Parameter(torch.cat([scene.positions, new_positions], dim=0))
    scene.scales = nn.Parameter(torch.cat([scene.scales, new_scales], dim=0))
    scene.mlps.extend(new_mlps)
    scene.num_primitives += len(new_mlps)

    # ✅ 扩展统计 buffer
    device = scene.alpha_accum.device
    zeros = torch.zeros(len(new_mlps), device=device)
    scene.alpha_accum = torch.cat([scene.alpha_accum, zeros], dim=0)
    scene.occupancy_accum = torch.cat([scene.occupancy_accum, zeros], dim=0)

    print(f"Densify: {len(new_mlps)} new primitives added.")

import os
import uuid
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from model import NeuralPrimitiveScene
# from vmodel import NeuralPrimitiveScene
from mip360_loader import mip360Dataset
from tqdm import tqdm
from save_model import save_model
from visualize import evaluate_and_visualize


import copy
from utils.loss_utils import ssim



"""
def primitive_consistency_loss(model, ray_o, ray_d, eps=1e-3):
    B = ray_o.shape[0]
    N = model.num_primitives

    ray_d = F.normalize(ray_d, dim=-1)
    noise = torch.randn_like(ray_o) * eps
    ray_o_perturbed = ray_o + noise

    ray_o_expand = ray_o.unsqueeze(0).expand(N, B, 3)
    ray_o_perturbed_expand = ray_o_perturbed.unsqueeze(0).expand(N, B, 3)
    ray_d_expand = ray_d.unsqueeze(0).expand(N, B, 3)
    center_expand = model.positions[:, None, :].expand(N, B, 3)

    # 原始投影距离
    vec = center_expand - ray_o_expand
    t = torch.sum(vec * ray_d_expand, dim=-1, keepdim=True)
    proj = ray_o_expand + t * ray_d_expand
    dist = center_expand - proj  # (N, B, 3)

    # 偏移后的投影距离
    vec2 = center_expand - ray_o_perturbed_expand
    t2 = torch.sum(vec2 * ray_d_expand, dim=-1, keepdim=True)
    proj2 = ray_o_perturbed_expand + t2 * ray_d_expand
    dist2 = center_expand - proj2  # (N, B, 3)

    loss_total = 0.0
    count = 0

    for i in range(N):
        norm = torch.norm(dist[i], dim=-1, keepdim=True)
        mask = norm <= 5 * model.scales[i]  # (B, 1)

        if not mask.any():
            continue

        out1 = model.mlps[i](ray_d, dist[i])  # (B, 4)
        out2 = model.mlps[i](ray_d, dist2[i])  # (B, 4)

        rgb1 = out1[..., :3]
        rgb2 = out2[..., :3]

        diff = F.mse_loss(rgb1[mask.squeeze(-1)], rgb2[mask.squeeze(-1)])
        loss_total += diff
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=ray_o.device)
    else:
        return loss_total / count
"""


def sample_patch_rays(rays_o, rays_d, rgbs, num_patches_h=4, num_patches_w=4):
    """
    从 (N, H, W, 3) 的 rays 中采样一个随机 patch。

    参数:
        rays_o, rays_d, rgbs: shape (N, H, W, 3)
        num_patches_h: 高度方向的划分数
        num_patches_w: 宽度方向的划分数

    返回:
        rays_o_batch, rays_d_batch, rgbs_batch: shape (patch_H * patch_W, 3)
    """
    N, H, W, _ = rays_o.shape
    patch_H = H // num_patches_h
    patch_W = W // num_patches_w

    # 随机选图像
    img_idx = torch.randint(0, N, ()).item()

    # 随机选 patch 行列索引
    row = torch.randint(0, num_patches_h, ()).item()
    col = torch.randint(0, num_patches_w, ()).item()

    h0, h1 = row * patch_H, (row + 1) * patch_H
    w0, w1 = col * patch_W, (col + 1) * patch_W

    # 提取 patch 并 reshape 成 (patch_H * patch_W, 3)
    rays_o_patch = rays_o[img_idx, h0:h1, w0:w1].reshape(-1, 3)
    rays_d_patch = rays_d[img_idx, h0:h1, w0:w1].reshape(-1, 3)
    rgbs_patch   = rgbs[img_idx, h0:h1, w0:w1].reshape(-1, 3)

    return rays_o_patch, rays_d_patch, rgbs_patch



# 设置用于 position 的 warm-up 和 linear decay 学习率函数
def get_position_lr(iteration, warmup_steps=500, max_steps=100000, init_lr=5e-3, final_lr=5e-4):
    if iteration < warmup_steps:
        return 5e-4
    else:
        return 5e-3
    # 线性衰减
    #else:
    #    alpha = (iteration - warmup_steps) / (max_steps - warmup_steps)
    #    return init_lr * (1 - alpha) + final_lr * alpha


def get_mlp_lr(iteration, max_steps=100000, init_lr=1e-3, final_lr=1e-4):
    halfway = max_steps // 2
    if iteration < halfway:
        return init_lr
    else:
        alpha = (iteration - halfway) / (max_steps - halfway)
        return init_lr * (1 - alpha) + final_lr * alpha


def train_scene_llff(data_dir, num_epochs=10, batch_size=4096, lr=1e-3, device='cuda', initial_num_primitives=1024):
    # Set saving directory
    output_root = "/home/yuzhou/Desktop/output/mip360"
    os.makedirs(output_root, exist_ok=True)
    random_tag = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    output_dir = os.path.join(output_root, random_tag)
    os.makedirs(output_dir, exist_ok=True)

    dataset = mip360Dataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=200, shuffle=True)

    scene = NeuralPrimitiveScene(num_primitives=initial_num_primitives, initial_positions=dataset.positions).to(device)
    #optimizer = torch.optim.Adam(scene.parameters(), lr=lr)

    optimizer = torch.optim.Adam([
        {"params": scene.mlps.parameters()},  # 默认 1e-3
        {"params": scene.log_scales, "lr": 1e-1},
        {"params": scene.positions, "lr": 5e-3},  # 初始设为较小值，保守一些
    ], lr=1e-3)


    #scheduler = StepLR(optimizer, step_size=1000, gamma=0.8)


    ckpt_path = os.path.join(output_dir, f"scene_epoch_0.pt")
    torch.save(scene.state_dict(), ckpt_path)
    print(f"[Checkpoint] Saved model to {ckpt_path}")


    loss_accum = 0.0
    loss_count = 0


    """
    #加载check point继续训练
    ckpt = torch.load("scene_ckpt_tmp.pt")
    model_dict = scene.state_dict()
    ckpt_dict = {k: v for k, v in ckpt.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(ckpt_dict)
    scene.load_state_dict(model_dict)
    print(f"[Checkpoint] Partially loaded model from scene_ckpt_tmp.pt (matched {len(ckpt_dict)} tensors)")
    """

        
    for epoch in tqdm(range(num_epochs)):

        # Update learning rates
        #mlp_lr = get_mlp_lr(epoch, max_steps=num_epochs)
        #position_lr = get_position_lr(epoch, warmup_steps=500, max_steps=num_epochs)
        #optimizer.param_groups[0]['lr'] = mlp_lr  # param_groups[0] 是 scene.mlps
        #optimizer.param_groups[2]['lr'] = position_lr  # param_groups[2] 是 scene.positions


        for rays_o, rays_d, rgbs in dataloader:
            rays_o = rays_o.to(device)  
            rays_d = rays_d.to(device)
            rgbs = rgbs.to(device)

            assert batch_size <= rays_o.reshape(-1,3).size(0), "Batch size must be less than or equal to the number of rays."

            # 使用随机采样 rays
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
            rgbs = rgbs.reshape(-1, 3)
            rand_idx = torch.randperm(rays_o.size(0))[:batch_size]
            rays_o_batch = rays_o[rand_idx]
            rays_d_batch = rays_d[rand_idx]
            rgbs_batch = rgbs[rand_idx]

            """
            if epoch % 500 < 500:
                # 使用随机采样 rays
                rays_o = rays_o.reshape(-1, 3)
                rays_d = rays_d.reshape(-1, 3)
                rgbs = rgbs.reshape(-1, 3)
                rand_idx = torch.randperm(rays_o.size(0))[:batch_size]
                rays_o_batch = rays_o[rand_idx]
                rays_d_batch = rays_d[rand_idx]
                rgbs_batch = rgbs[rand_idx]
            else:
                rays_o_batch, rays_d_batch, rgbs_batch = sample_patch_rays(
                            rays_o, rays_d, rgbs,
                            num_patches_h=4, num_patches_w=4  # 纵向4块，横向4块
                        )  
            """

            pred = scene(rays_o_batch, rays_d_batch)
            loss = F.mse_loss(pred, rgbs_batch) 
            optimizer.zero_grad()
            loss.backward()
            # import pdb; pdb.set_trace()

            #print("Position grad norm:", scene.positions.grad.norm().item())
            #with torch.no_grad():
                #print(f"[Epoch {epoch+1}] Position mean: {scene.positions.mean().item():.6f}, std: {scene.positions.std().item():.6f}")

            optimizer.step()

        # 累加 loss
        loss_accum += loss.item()
        loss_count += 1
        # 每 100 epoch 打印平均 loss
        if (epoch + 1) % 100 == 0:
            avg_loss = loss_accum / loss_count
            print(f"[Epoch {epoch+1}] avg {loss_count} loss : {avg_loss:.6f}")
            loss_accum = 0.0
            loss_count = 0

        # TODO wandb


        # Prune & densify primitives
        if epoch % 500 == 0 and epoch > 500:
            prune_primitives(scene)
            MAX_NUM_PRIMITIVES = initial_num_primitives
            if scene.num_primitives < MAX_NUM_PRIMITIVES:
                densify_primitives(scene, num_new=min(16, MAX_NUM_PRIMITIVES - scene.num_primitives))
            print(f"After pruning and densifying, scene has {scene.num_primitives} primitives.")

        # Eval after 500 epoch 
        if (epoch + 1) % 500 == 0:
            torch.save(scene.state_dict(), os.path.join(output_dir, "scene_ckpt_tmp.pt"))
            #evaluate_and_visualize(data_dir, model_ckpt=scene.state_dict(), N=scene.num_primitives, device=device)
     
        # Save at every 1000 epochs
        if (epoch + 1) % 1000 == 0:
            ckpt_path = os.path.join(output_dir, f"scene_epoch_{epoch+1:05d}.pt")
            torch.save(scene.state_dict(), ckpt_path)
            print(f"[Checkpoint] Saved model to {ckpt_path}")


    save_model(scene, os.path.join(output_dir, "scene_ckpt.pt"))


def prune_primitives(scene, alpha_thresh=0.01, occupancy_thresh=0.01):
    alpha = scene.alpha_accum
    occ = scene.occupancy_accum

    keep_mask = (alpha > alpha_thresh) & (occ > occupancy_thresh)
    keep_indices = torch.nonzero(keep_mask).squeeze(-1)

    print(f"Pruned {scene.num_primitives - keep_indices.numel()} primitives.")

    scene.positions = nn.Parameter(scene.positions[keep_indices])
    scene.log_scales = nn.Parameter(scene.log_scales[keep_indices])
    scene.mlps = nn.ModuleList([scene.mlps[i] for i in keep_indices])
    scene.num_primitives = keep_indices.numel()
    scene.alpha_accum = scene.alpha_accum[keep_indices]
    scene.occupancy_accum = scene.occupancy_accum[keep_indices]

def densify_primitives(scene, num_new=32, noise_scale=0.01, alpha_thresh=0.05):
    # 筛选出 alpha 大于阈值的 primitive
    candidate_mask = scene.alpha_accum > alpha_thresh
    candidate_indices = torch.nonzero(candidate_mask).squeeze(-1)

    if candidate_indices.numel() == 0:
        print("No suitable primitives found for densification.")
        return

    # 从中选择最多 num_new 个进行 densify（可随机，也可 topk）
    if candidate_indices.numel() > num_new:
        selected = candidate_indices[torch.randperm(candidate_indices.numel())[:num_new]]
    else:
        selected = candidate_indices

    new_positions = scene.positions[selected] + noise_scale * torch.randn_like(scene.positions[selected])
    new_scales = scene.log_scales[selected].clone()
    new_mlps = [copy.deepcopy(scene.mlps[i]) for i in selected]

    scene.positions = nn.Parameter(torch.cat([scene.positions, new_positions], dim=0))
    scene.log_scales = nn.Parameter(torch.cat([scene.log_scales, new_scales], dim=0))
    scene.mlps.extend(new_mlps)
    scene.num_primitives += len(new_mlps)

    # 扩展统计 buffer
    device = scene.alpha_accum.device
    zeros = torch.zeros(len(new_mlps), device=device)
    scene.alpha_accum = torch.cat([scene.alpha_accum, zeros], dim=0)
    scene.occupancy_accum = torch.cat([scene.occupancy_accum, zeros], dim=0)

    print(f"Densify: {len(new_mlps)} new primitives added (alpha_thresh={alpha_thresh}).")

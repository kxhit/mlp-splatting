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
from synthetic_loader import SyntheticDataset
from tqdm import tqdm
from save_model import save_model
from visualize import evaluate_and_visualize


import copy
from utils.loss_utils import ssim



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


def train_scene_nerf_synthetic(data_dir, scene_name, num_epochs=10, batch_size=4096, lr=1e-3, device='cuda', initial_num_primitives=1024):
    # Set saving directory
    output_root = "/home/yuzhou/Desktop/output/nerf_synthetic"
    os.makedirs(output_root, exist_ok=True)
    random_tag = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    output_dir = os.path.join(output_root, random_tag)
    os.makedirs(output_dir, exist_ok=True)

    dataset = SyntheticDataset(data_dir, scene_name)
    scene = NeuralPrimitiveScene(num_primitives=initial_num_primitives, initial_positions=dataset.positions).to(device)
    #optimizer = torch.optim.Adam(scene.parameters(), lr=lr)


    optimizer = torch.optim.Adam([
        {"params": scene.mlps.parameters()},  # 默认 1e-3
        {"params": scene.log_scales, "lr": 1e-1},
        {"params": scene.positions, "lr": 0},  # 初始设为较小值，保守一些
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

    rays_o_all, rays_d_all, rgbs_all = dataset.get_all_rays(device=None)

    N = rays_o_all.shape[0]
    assert batch_size <= N, "Batch size must be less than or equal to the number of rays."
    perm = torch.randperm(N)
    ptr = 0


    # 训练循环
    for epoch in tqdm(range(num_epochs)):

        # Update learning rates
        #mlp_lr = get_mlp_lr(epoch, max_steps=num_epochs)
        #position_lr = get_position_lr(epoch, warmup_steps=500, max_steps=num_epochs)
        #optimizer.param_groups[0]['lr'] = mlp_lr  # param_groups[0] 是 scene.mlps
        #optimizer.param_groups[2]['lr'] = position_lr  # param_groups[2] 是 scene.positions

        if ptr + batch_size > N:
            perm = torch.randperm(N)
            ptr = 0

        idx = perm[ptr:ptr + batch_size]
        ptr += batch_size

        # 采样 patch rays
        rays_o_batch = rays_o_all[idx].to(device, non_blocking=True)
        rays_d_batch = rays_d_all[idx].to(device, non_blocking=True)
        rgbs_batch   = rgbs_all[idx].to(device, non_blocking=True)

        # 训练一步
        optimizer.zero_grad()                       # 1. 清梯度
        pred = scene(rays_o_batch, rays_d_batch)    # 2. 前向
        loss = F.mse_loss(pred, rgbs_batch)         # 3. 计算 loss
        loss.backward()                             # 4. 反向传播

        # import pdb; pdb.set_trace()

        #print("Position grad norm:", scene.positions.grad.norm().item())
        #with torch.no_grad():
            #print(f"[Epoch {epoch+1}] Position mean: {scene.positions.mean().item():.6f}, std: {scene.positions.std().item():.6f}")

        optimizer.step()                            # 5. 更新参数


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
            #prune_primitives(scene)
            #MAX_NUM_PRIMITIVES = initial_num_primitives
            #if scene.num_primitives < MAX_NUM_PRIMITIVES:
            #    densify_primitives(scene, num_new=min(16, MAX_NUM_PRIMITIVES - scene.num_primitives))
            print(f"After pruning and densifying, scene has {scene.num_primitives} primitives.")

        # Eval after 500 epoch 
        if (epoch + 1) % 500 == 0:
            torch.save(scene.state_dict(), os.path.join(output_dir, "scene_ckpt_tmp.pt"))
            #evaluate_and_visualize(data_dir, model_ckpt=scene.state_dict(), N=scene.num_primitives, device=device)
        
        # Save at every 2000 epochs
        if (epoch + 1) % 2000 == 0:
            ckpt_path = os.path.join(output_dir, f"scene_epoch_{epoch+1:05d}.pt")
            torch.save(scene.state_dict(), ckpt_path)
            print(f"[Checkpoint] Saved model to {ckpt_path}")


    save_model(scene, os.path.join(output_dir, "scene_ckpt.pt"))


def prune_primitives(scene, alpha_thresh=0.01, occupancy_thresh=0.01, scale_min=0.01, scale_max=10.0):
    alpha = scene.alpha_accum
    occ = scene.occupancy_accum

    # 将 log_scale 转换为 scale（假设 shape 为 (N, 1)）
    scales = torch.exp(scene.log_scales).squeeze(-1)  # shape: (N,)

    # alpha + occupancy + scale 限制
    keep_mask = (
        (alpha > alpha_thresh) &
        (occ > occupancy_thresh) &
        (scales > scale_min) &
        (scales < scale_max)
    )
    keep_indices = torch.nonzero(keep_mask).squeeze(-1)

    print(f"Pruned {scene.num_primitives - keep_indices.numel()} primitives.")

    scene.positions = nn.Parameter(scene.positions[keep_indices])
    scene.log_scales = nn.Parameter(scene.log_scales[keep_indices])
    scene.mlps = nn.ModuleList([scene.mlps[i] for i in keep_indices])
    scene.num_primitives = keep_indices.numel()
    scene.alpha_accum = scene.alpha_accum[keep_indices]
    scene.occupancy_accum = scene.occupancy_accum[keep_indices]



def densify_primitives(scene, grad_thresh=0.002, scale_thresh=0.1, shrink_factor=1.6, num_new=32):
    grad = scene.positions.grad          # (N, 3)
    grad_norm = torch.norm(grad, dim=1)  # (N,)
    log_scales = scene.log_scales.squeeze(-1)  # (N,)
    scales = torch.exp(log_scales)

    # 找出 grad_norm 大于阈值的 primitive
    mask = grad_norm > grad_thresh
    if not mask.any():
        print("No primitives met gradient threshold for densification.")
        return

    indices = torch.nonzero(mask).squeeze(-1)
    grad_norm_valid = grad_norm[indices]

    # 只保留前 num_new 个（按 grad_norm 降序）
    topk = min(num_new, indices.numel())
    _, topk_order = torch.topk(grad_norm_valid, k=topk)
    selected_indices = indices[topk_order]

    # 分类：大 scale / 小 scale
    selected_scales = scales[selected_indices]
    selected_log_scales = log_scales[selected_indices]
    selected_grad = grad[selected_indices]
    selected_pos = scene.positions[selected_indices]

    mask_large = selected_scales > scale_thresh
    mask_small = ~mask_large

    # --- Case 1: scale > threshold ---
    pos_large = selected_pos[mask_large]
    scale_large = selected_scales[mask_large] / shrink_factor
    log_scale_large = torch.log(scale_large)

    # 更新原始 primitive（scale /= shrink_factor）
    scene.positions.data[selected_indices[mask_large]] = pos_large
    scene.log_scales.data[selected_indices[mask_large]] = log_scale_large.unsqueeze(-1)

    # 复制：加高斯噪声
    noise = torch.randn_like(pos_large)
    new_pos_large = pos_large + scale_large.unsqueeze(1) * noise
    new_log_scale_large = log_scale_large
    new_mlps_large = [copy.deepcopy(scene.mlps[i]) for i in selected_indices[mask_large].tolist()]

    # --- Case 2: scale <= threshold ---
    pos_small = selected_pos[mask_small]
    scale_small = selected_scales[mask_small]
    log_scale_small = selected_log_scales[mask_small]
    grad_small = selected_grad[mask_small]

    direction = grad_small / (grad_small.norm(dim=1, keepdim=True) + 1e-8)
    offset = direction * scale_small.unsqueeze(-1)
    new_pos_small = pos_small + offset
    new_log_scale_small = log_scale_small
    new_mlps_small = [copy.deepcopy(scene.mlps[i]) for i in selected_indices[mask_small].tolist()]

    # 合并新增 primitive
    new_positions = torch.cat([new_pos_large, new_pos_small], dim=0)
    new_log_scales = torch.cat([new_log_scale_large, new_log_scale_small], dim=0)
    new_mlps = new_mlps_large + new_mlps_small

    scene.positions = nn.Parameter(torch.cat([scene.positions, new_positions], dim=0))
    scene.log_scales = nn.Parameter(torch.cat([scene.log_scales, new_log_scales.unsqueeze(-1)], dim=0))
    scene.mlps.extend(new_mlps)
    scene.num_primitives += len(new_mlps)

    # 扩展统计 buffer
    device = scene.alpha_accum.device
    zeros = torch.zeros(len(new_mlps), device=device)
    scene.alpha_accum = torch.cat([scene.alpha_accum, zeros], dim=0)
    scene.occupancy_accum = torch.cat([scene.occupancy_accum, zeros], dim=0)

    print(f"Densified {len(new_mlps)} new primitives (max {num_new}, grad_thresh={grad_thresh}).")


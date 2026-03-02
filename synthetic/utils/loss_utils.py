#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    



"""
import torch

@torch.no_grad()
def update_knn_cache(scene, k=8, chunk=2048, device=None):
    pos = scene.positions.detach()
    if device is not None:
        pos = pos.to(device)
    N = pos.shape[0]

    knn_idx = torch.empty((N, k), device=pos.device, dtype=torch.long)
    knn_dist2 = torch.empty((N, k), device=pos.device, dtype=torch.float32)

    # 逐块计算与所有点的距离，取 topk (最小)
    for i0 in range(0, N, chunk):
        i1 = min(N, i0 + chunk)
        q = pos[i0:i1]  # (C,3)

        # (C,N) squared distances
        # torch.cdist 默认返回 L2；这里用 cdist^2 或手写更省
        d = torch.cdist(q, pos, p=2)  # (C,N)
        d2 = d * d

        # 排除自身（对角线）：对 query block 内对应的全局 index 置 inf
        rows = torch.arange(i1 - i0, device=pos.device)
        cols = torch.arange(i0, i1, device=pos.device)
        d2[rows, cols] = float("inf")

        # 取最近 k 个
        vals, inds = torch.topk(d2, k=k, dim=1, largest=False, sorted=False)
        knn_idx[i0:i1] = inds
        knn_dist2[i0:i1] = vals

    return knn_idx, knn_dist2



def scale_knn_loss_weighted(scene, knn_idx, knn_dist2, sigma=0.05, eps=1e-4):
    ls = scene.log_scales.squeeze(-1)          # (N,)
    nbr = ls[knn_idx]                          # (N,k)
    diff = ls[:, None] - nbr                   # (N,k)

    sigma2 = sigma * sigma
    w = torch.exp(-knn_dist2 / (2.0 * sigma2)) # (N,k)

    num = (w * torch.sqrt(diff * diff + eps * eps)).sum()
    den = w.sum() + 1e-12
    return num / den
"""
# densify_prune.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm


def compute_multiview_densify_scores(
    scene,
    dataset,
    k_images: int = 8,
    r_threshold: float = 0.1,       # 新增：Min-Max 后的归一化阈值
    score_threshold: float = 1e-3,  # 渲染器内部的 threshold
    bg_color: torch.Tensor | None = None,
    cover_mult: float = 3.716,      
    chunk_prims: int = 128,         # 增加 chunk 大小以提高效率
    device: str | torch.device = "cuda",
    seed: int | None = None,
):
    scene_device = scene.positions.device
    N = int(scene.positions.shape[0])
    importance = torch.zeros(N, device=scene_device, dtype=torch.float32)

    # --- 随机挑选 k 个视图 ---
    n_imgs = len(dataset)
    k = min(k_images, n_imgs)
    indices = torch.randperm(n_imgs, generator=torch.Generator().manual_seed(seed) if seed else None)[:k].tolist()

    focal = float(dataset.focal)
    Xw = scene.positions.data
    s  = torch.exp(scene.log_scales.data.squeeze(-1))

    for img_idx in indices:
        rays_o_hw, rays_d_hw, img_hw3, pose = dataset[img_idx]
        H, W = img_hw3.shape[0], img_hw3.shape[1]
        
        # 1. 渲染当前视角
        view = _c2w_to_view_matrix(pose, device=scene_device)
        proj = _build_projection_from_focal(focal, W, H, device=scene_device)
        
        with torch.no_grad():
            pred_rgb, _, _ = scene.render_rgb_t_no_grad(
                rays_o_hw.reshape(-1,3).to(scene_device), 
                rays_d_hw.reshape(-1,3).to(scene_device),
                score_threshold=score_threshold,
                bg_color=bg_color,
                view_matrix=view,
                proj_matrix=proj
            )
        
        # 2. 计算归一化 Mask
        gt = img_hw3.reshape(-1, 3).to(scene_device)
        l1 = (pred_rgb - gt).abs().sum(dim=-1).view(H, W)
        
        l1_min, l1_max = l1.min(), l1.max()
        diff = l1_max - l1_min
        l1_norm = (l1 - l1_min) / diff if diff > 1e-7 else torch.zeros_like(l1)
        
        # 大于阈值 r 的像素标记为 1
        mask = (l1_norm >= r_threshold).float() 

        # 3. 投影所有 Primitive 到像素坐标
        # [投影逻辑同你之前的代码]
        ones = torch.ones((N, 1), device=scene_device)
        Xc_h = torch.cat([Xw, ones], dim=-1) @ view.T
        x, y, z = Xc_h[:, 0], Xc_h[:, 1], Xc_h[:, 2]
        in_front = z > 1e-2
        
        u = focal * (x / z) + (0.5 * W)
        v = focal * (y / z) + (0.5 * H)
        r_pix = (float(cover_mult) * s * focal / z).abs()
        r_pix = torch.clamp(r_pix, 0.0, float(max(H, W)))

        # 4. 分块计算覆盖 Mask 的像素数量
        ys, xs = torch.arange(H, device=scene_device), torch.arange(W, device=scene_device)
        Y, X = torch.meshgrid(ys, xs, indexing="ij")

        for a in range(0, N, chunk_prims):
            b = min(N, a + chunk_prims)
            valid = in_front[a:b] & (r_pix[a:b] > 0.1)
            if not valid.any(): continue
            
            # 只有有效且在视野内的才计算
            idx_local = torch.nonzero(valid).squeeze(-1)
            # 这里的计算量较大，对于每个 Primitive 检查圆内像素并求和 Mask 值
            for i in idx_local:
                curr_idx = a + i
                dist_sq = (X - u[curr_idx])**2 + (Y - v[curr_idx])**2
                inside = (dist_sq <= r_pix[curr_idx]**2)
                importance[curr_idx] += (inside.float() * mask).sum()

    return importance / float(k)




@torch.no_grad()
def compute_multiview_prune_scores(
    scene,
    dataset,
    k_images: int = 10,
    r_threshold: float = 0.10,     
    score_threshold: float = 1e-3,
    mask_min_pixels: int = 1024,     # 防止 mask 过小
    cover_mult: float = 2.5,         # footprint 半径 = cover_mult * scale（world）
    chunk_prims: int = 64,
    device: str | torch.device = "cuda",
    seed: int | None = None,
    eps: float = 1e-8,
):
    """
    返回：
      scores_raw: (N,) 未归一化 s_i
      scores_01 : (N,) min-max 到 [0,1] 的 s_i
    """
    dev = scene.positions.device
    N = int(scene.positions.shape[0])
    scores = torch.zeros(N, device=dev, dtype=torch.float32)

    # pick k views
    n_imgs = len(dataset)
    K = int(min(k_images, n_imgs))
    if seed is not None:
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        perm = torch.randperm(n_imgs, generator=g)
    else:
        perm = torch.randperm(n_imgs)
    pick = perm[:K].tolist()

    focal = float(getattr(dataset, "focal", None) or getattr(dataset, "focal_length", None))
    if focal is None:
        raise AttributeError("dataset.focal (or focal_length) is required.")

    # primitive geometry
    Xw = scene.positions.data.to(dev)  # (N,3)
    s  = torch.exp(scene.log_scales.data.squeeze(-1)).to(dev)  # (N,)

    for img_idx in pick:
        rays_o_hw, rays_d_hw, img_hw3, pose = dataset[int(img_idx)]
        H, W = int(img_hw3.shape[0]), int(img_hw3.shape[1])

        if hasattr(scene, "set_image_size"):
            scene.set_image_size(W, H)

        rays_o = rays_o_hw.reshape(-1, 3).to(dev, non_blocking=True)
        rays_d = rays_d_hw.reshape(-1, 3).to(dev, non_blocking=True)
        gt = img_hw3.reshape(-1, 3).to(dev, non_blocking=True)

        view = _c2w_to_view_matrix(pose, device=dev)
        proj = _build_projection_from_focal(focal, W=W, H=H, device=dev)

        # forward rgb
        pred_rgb = scene(
            rays_o, rays_d,
            score_threshold=float(score_threshold),
            view_matrix=view,
            proj_matrix=proj,
        )

        # 计算归一化 Mask
        l1 = (pred_rgb - gt).abs().sum(dim=-1).view(H, W)
        E_photo = l1.mean() 
        l1_min, l1_max = l1.min(), l1.max()
        diff = l1_max - l1_min
        l1_norm = (l1 - l1_min) / diff if diff > 1e-7 else torch.zeros_like(l1)
        # 大于阈值 r 的像素标记为 1
        mask = (l1_norm >= r_threshold).float()

        # ensure mask not too tiny
        if int(mask.sum().item()) < int(mask_min_pixels):
            flat = l1.flatten()
            topk = min(int(mask_min_pixels), flat.numel())
            _, idxs = torch.topk(flat, k=topk, largest=True, sorted=False)
            mask = torch.zeros_like(flat, dtype=torch.bool)
            mask[idxs] = True
            mask = mask.view(H, W)

        mask_f = mask.to(torch.float32)  # (H,W)

        # pixel grids
        xs = torch.arange(W, device=dev, dtype=torch.float32)
        ys = torch.arange(H, device=dev, dtype=torch.float32)
        Y, X = torch.meshgrid(ys, xs, indexing="ij")  # (H,W)

        # project all primitives to this view
        ones = torch.ones((N, 1), device=dev, dtype=torch.float32)
        Xw_h = torch.cat([Xw.to(torch.float32), ones], dim=-1)   # (N,4)
        Xc_h = (Xw_h @ view.T)                                    # (N,4), row-vector
        x = Xc_h[:, 0]
        y = Xc_h[:, 1]
        z = Xc_h[:, 2]

        in_front = z > 1e-6
        if not in_front.any():
            continue

        cx = 0.5 * float(W)
        cy = 0.5 * float(H)
        fx = focal
        fy = focal

        u = fx * (x / z) + cx
        v = fy * (y / z) + cy

        r_world = float(cover_mult) * s.to(torch.float32)       # (N,)
        r_pix = (r_world * fx / z).abs()                        # (N,)
        max_radius = float(max(H, W))
        r_pix = torch.clamp(r_pix, 0.0, max_radius)

        # per-primitive exact mask count, chunked
        for a in range(0, N, int(chunk_prims)):
            b = min(N, a + int(chunk_prims))
            uu = u[a:b]
            vv = v[a:b]
            rr = r_pix[a:b]
            valid = in_front[a:b] & torch.isfinite(uu) & torch.isfinite(vv) & torch.isfinite(rr) & (rr > 0.0)
            if not valid.any():
                continue

            uu = uu[valid]
            vv = vv[valid]
            rr = rr[valid]

            du2 = (X[None, :, :] - uu[:, None, None]) ** 2
            dv2 = (Y[None, :, :] - vv[:, None, None]) ** 2
            inside = (du2 + dv2) <= (rr[:, None, None] ** 2)     # (C,H,W) bool

            cnt = (inside.to(torch.float32) * mask_f[None, :, :]).sum(dim=(1, 2))  # (C,)

            idx_local = torch.nonzero(valid, as_tuple=False).squeeze(-1) + a
            scores[idx_local] += cnt * E_photo

    # min-max normalize to [0,1]
    smin = scores.min()
    smax = scores.max()
    scores_01 = (scores - smin) / (smax - smin + eps)

    return scores_01



@torch.no_grad()
def get_prune_ids_by_opacity_bias(scene, threshold: float = 0.005):
    """
    根據 MLP 最后一層的 Bias (b3[3]) 來判斷哪些 Primitive 太透明，需要被剪枝。
    
    參數:
      scene: 包含 b3 參數的模型
      threshold: 透明度閾值，預設 0.005
    返回:
      prune_ids: torch.Tensor (long), 需要被刪除的索引
    """
    # b3 的形狀通常是 (N, 4)，最後一維是 [R, G, B, Opacity_logit]
    # 我們只取最後一維
    opacity_logits = scene.b3.data[:, 3]
    
    # 計算基礎透明度 (Base Opacity)
    base_opacity = torch.sigmoid(opacity_logits)
    
    # 找出小於閾值的索引
    prune_mask = base_opacity < threshold
    prune_ids = torch.nonzero(prune_mask).squeeze(-1)
    
    return prune_ids


@torch.no_grad()
def get_prune_ids_by_scale(scene, threshold: float = 1.5):
    scale_scales = torch.exp(scene.log_scales.squeeze(-1))

    # 找出大於閾值的索引
    prune_mask = scale_scales > threshold
    prune_ids = torch.nonzero(prune_mask).squeeze(-1)
    
    return prune_ids



@torch.no_grad()
def densify_prune_tilecuda_and_rebuild_optimizer(
    scene,
    optimizer,
    densify_ids,
    prune_ids,
    *,
    init_std: float = 0.01,
    jitter_frac: float = 0.1,   # δ = jitter_frac * (scale_old) * rand_unit
    scale_div: float = 2.0,      # scale /= 2
    scale_threshold: float = 0.05, # scale > threshold => split; else clone
    reset_alpha_for_touched: bool = True,
    keep_new_from_prune: bool = True,  # prune_ids 默认是旧 index 空间；不去 prune 新的
):
    """
    Densify (split) + Prune + Rebuild optimizer.

    densify:
      - for each old id i in densify_ids:
          * create one new primitive appended at end
          * old & new scales /= scale_div
          * old pos += δ, new pos -= δ (paired jitter)
          * old & new MLP params random init
    prune:
      - remove prune_ids (by default interpreted in old index space)
    returns:
      new_optimizer, info, unaffected_ids_final
    """

    device = scene.positions.device
    N0 = int(scene.num_primitives)

    densify_ids = torch.as_tensor(densify_ids, device=device, dtype=torch.long)
    prune_ids   = torch.as_tensor(prune_ids,   device=device, dtype=torch.long)

    """
    all_grads = scene.positions.grad[densify_ids] # (K, 3)
    grad_norms = torch.norm(all_grads, dim=-1)    # (K,)
    eps = 1e-9
    mask = grad_norms > eps
    densify_ids = densify_ids[mask]
    """

    densify_ids = torch.unique(densify_ids)
    prune_ids   = torch.unique(prune_ids)

    # sanitize ranges (old index space)
    if densify_ids.numel() > 0:
        if int(densify_ids.min()) < 0 or int(densify_ids.max()) >= N0:
            raise ValueError(f"densify_ids out of range: [0,{N0-1}]")
    if prune_ids.numel() > 0:
        if int(prune_ids.min()) < 0 or int(prune_ids.max()) >= N0:
            raise ValueError(f"prune_ids out of range (assumed old space): [0,{N0-1}]")


    """
    # If an id is both densify and prune: pruning wins, so skip densify for it.
    if densify_ids.numel() > 0 and prune_ids.numel() > 0:
        densify_ids = densify_ids[~torch.isin(densify_ids, prune_ids)]
    """

    K = int(densify_ids.numel())




    # ==========
    # 1) DENSIFY (Split vs Clone based on Scale)
    # ==========
    if K > 0:
        pos = scene.positions.data
        lsc = scene.log_scales.data
        W1, b1, W2, b2, W3, b3 = scene.W1.data, scene.b1.data, scene.W2.data, scene.b2.data, scene.W3.data, scene.b3.data

        # --- A. 获取梯度方向 ---
        state = optimizer.state.get(scene.positions, None)
        if state is not None and "exp_avg" in state:
            m = state["exp_avg"]  # (N,3)
            dirs = m[densify_ids]
            dir_unit = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-12)

        # --- B. 判断分类：Split (大) vs Clone (小) ---
        scale_old = torch.exp(lsc[densify_ids].squeeze(-1))
        is_large = scale_old > scale_threshold  # (K,) bool mask

        # --- C. 计算位移与 Scale 变化 ---
        # 大 scale: 移动 0.2, 且 scale 缩小 (scale_div=1.5)
        # 小 scale: 移动 0.5, 且 scale 保持
        jitter = torch.where(is_large, 1, 1)
        # 旧的 B Loss 下降
        delta = (jitter * scale_old)[:, None] * (-dir_unit)

        old_pos_new = pos.clone()
        old_pos_new[densify_ids] += delta  # B
        new_pos = pos[densify_ids] -+ delta  # C

        old_lsc_new = lsc.clone()
        # 只有大的需要缩小体积 (scale /= 1.5 => log_scale -= log(1.5))
        lsc_shrink = torch.where(is_large, math.log(5), 0.0)
        lsc_shrink = lsc_shrink.unsqueeze(-1)
        old_lsc_new[densify_ids] -= lsc_shrink
        new_lsc = lsc[densify_ids] - lsc_shrink

        # --- D. 透明度补偿 (保持透過率一致) ---
        old_bias_w = b3[densify_ids, 3]

        """
        # --- D. 综合诊断打印 (前 50 个) ---
        grad_norms = torch.norm(dirs, dim=-1)
        
        print(f"\n[Densify Diagnosis] Total IDs: {K}")
        # 这里的标题改为 Opacity(Phys)，代表物理透明度
        print(f"{'ID_idx':>8} | {'Opacity(Phys)':>15} | {'Grad_Norm':>12} | {'Scale':>10}")
        print("-" * 60)
        
        # 修改打印逻辑：打印前 50 个，如果不足 50 则打印全部
        num_to_print = min(K, 50)
        display_ids = list(range(num_to_print))
        
        # 如果总数超过 50, 最后再补 5 个结尾的，方便看整体跨度
        if K > 55:
            display_ids += [-1] # 占位符
            display_ids += list(range(K-5, K))

        for i in display_ids:
            if i == -1:
                print("   ...      |       ...       |     ...      |    ...")
                continue
            
            # 1. 物理透明度 (Sigmoid 之后)
            # 如果 old_bias_w[i] 是 0，这里会显示 0.500000
            b_phys = torch.sigmoid(old_bias_w[i]).item()
            
            # 2. 梯度模长
            g_val = grad_norms[i].item()
            
            # 3. 尺寸
            s_val = scale_old[i].item()
            
            print(f"{i:8d} | {b_phys:15.6f} | {g_val:12.6e} | {s_val:10.5f}")
            
        print("-" * 60)
        
        # 统计小信息：有多少是“死点”
        num_zeros = (grad_norms < 1e-12).sum().item()
        print(f"Summary: {num_zeros}/{K} points have ZERO gradient (Norm < 1e-12)")
        print("-" * 60 + "\n")
        """
        
        """
        alpha = torch.sigmoid(old_bias_w)
        alpha = torch.clamp(alpha, 1e-6, 0.99) 
        new_alpha = 1.0 - torch.sqrt(1.0 - alpha)
        new_bias_w = torch.log(new_alpha / (1.0 - new_alpha + 1e-8))
        """

        #y = F.softplus(old_bias_w)               # sigma_raw_old (bias-only approx)
        #new_bias_w = torch.log(torch.expm1(0.5 * y))
        new_bias_w = old_bias_w 


        # --- E. MLP 权重继承与扰动 ---
        old_W1_new, old_b1_new = W1.clone(), b1.clone()
        old_W2_new, old_b2_new = W2.clone(), b2.clone()
        old_W3_new, old_b3_new = W3.clone(), b3.clone()
        old_b3_new[densify_ids, 3] = new_bias_w 

        new_W1, new_b1 = W1[densify_ids].clone(), b1[densify_ids].clone()
        new_W2, new_b2 = W2[densify_ids].clone(), b2[densify_ids].clone()
        new_W3, new_b3 = W3[densify_ids].clone(), b3[densify_ids].clone()
        new_b3[:, 3] = new_bias_w  

        # 打破权重对称性
        noise_std = 1e-2
        new_W3 += torch.randn_like(new_W3) * noise_std
        new_W2 += torch.randn_like(new_W2) * (noise_std * 0.1)

        # --- F. 参数写回 ---
        scene.positions  = nn.Parameter(torch.cat([old_pos_new, new_pos], dim=0))
        scene.log_scales = nn.Parameter(torch.cat([old_lsc_new, new_lsc], dim=0))
        scene.W1 = nn.Parameter(torch.cat([old_W1_new, new_W1], dim=0))
        scene.b1 = nn.Parameter(torch.cat([old_b1_new, new_b1], dim=0))
        scene.W2 = nn.Parameter(torch.cat([old_W2_new, new_W2], dim=0))
        scene.b2 = nn.Parameter(torch.cat([old_b2_new, new_b2], dim=0))
        scene.W3 = nn.Parameter(torch.cat([old_W3_new, new_W3], dim=0))
        scene.b3 = nn.Parameter(torch.cat([old_b3_new, new_b3], dim=0))

        # 处理 alpha_accum 等辅助变量
        if hasattr(scene, "alpha_accum") and scene.alpha_accum is not None:
            aa = scene.alpha_accum
            # 新加入的元素累积透明度设为 0
            new_aa = torch.zeros(K, device=device, dtype=aa.dtype)
            scene.alpha_accum = torch.cat([aa, new_aa], dim=0)

        scene.num_primitives = int(scene.positions.shape[0])

    N1 = int(scene.num_primitives)

    # ==========
    # 2) PRUNE
    # ==========
    if prune_ids.numel() > 0:
        # prune_ids assumed in OLD index space [0, N0)
        # after densify, old indices are still [0, N0), new are [N0, N0+K)
        # optionally keep new always
        if keep_new_from_prune:
            prune_ids_eff = prune_ids
        else:
            prune_ids_eff = prune_ids  # (or allow passing new ids separately)

        keep = torch.ones(N1, device=device, dtype=torch.bool)
        keep[prune_ids_eff] = False

        scene.positions  = nn.Parameter(scene.positions.data[keep])
        scene.log_scales = nn.Parameter(scene.log_scales.data[keep])

        scene.W1 = nn.Parameter(scene.W1.data[keep])
        scene.b1 = nn.Parameter(scene.b1.data[keep])
        scene.W2 = nn.Parameter(scene.W2.data[keep])
        scene.b2 = nn.Parameter(scene.b2.data[keep])
        scene.W3 = nn.Parameter(scene.W3.data[keep])
        scene.b3 = nn.Parameter(scene.b3.data[keep])

        if hasattr(scene, "alpha_accum") and scene.alpha_accum is not None:
            scene.alpha_accum = scene.alpha_accum[keep]

        scene.num_primitives = int(scene.positions.shape[0])

    N2 = int(scene.num_primitives)  # final

    # ==========
    # 3) REBUILD OPTIMIZER (with Stage 1 Strategy)
    # ==========
    defaults = optimizer.defaults
    
    # 基础学习率获取 (保持原有的配置)
    def group_lr(g_idx):
        if len(optimizer.param_groups) > g_idx:
            return optimizer.param_groups[g_idx].get("lr", defaults.get("lr", 1e-3))
        return defaults.get("lr", 1e-3)

    base_lr_mlp = group_lr(0)
    base_lr_lsc = group_lr(1)
    base_lr_pos = group_lr(2)

    # --- 阶段 1 压制逻辑 ---
    # 我们调低 Scale 和 Position 的学习率，迫使 Opacity (MLP Bias) 成为主要优化手段
    stage1_scale_mult = 1  # 压制 Scale 增长，防止其通过变大来遮掩空隙
    stage1_pos_mult   = 1  # 稳定位置，防止剧烈跳动破坏补偿逻辑

    mlp_params = [scene.W1, scene.b1, scene.W2, scene.b2, scene.W3, scene.b3]

    new_optimizer = torch.optim.Adam(
        [
            {
                "params": mlp_params, 
                "lr": base_lr_mlp, 
                "name": "mlp"
            },
            {
                "params": [scene.log_scales], 
                "lr": base_lr_lsc * stage1_scale_mult, 
                "name": "log_scales"
            },
            {
                "params": [scene.positions], 
                "lr": base_lr_pos * stage1_pos_mult, 
                "name": "positions"
            },
        ],
        lr=base_lr_mlp,
        betas=(0.7, 0.999), # 调低 beta1 (0.9 -> 0.7) 让优化器对分裂后的新梯度更“敏感”
        eps=defaults.get("eps", 1e-3)
    )

    # ==========
    # 4) RETURN UNAFFECTED IDS (final index space)
    # ==========
    # “unaffected” means: old primitives not densified and not pruned.
    # Compute in old space first, then map through final keep-mask.
    unaffected_old = torch.ones(N0, device=device, dtype=torch.bool)
    if densify_ids.numel() > 0:
        unaffected_old[densify_ids] = False
    if prune_ids.numel() > 0:
        unaffected_old[prune_ids] = False
    unaffected_old_ids = torch.nonzero(unaffected_old, as_tuple=False).squeeze(-1)  # in [0,N0)

    # Map old ids -> final ids: after pruning, indices compress.
    # We can rebuild a mapping by simulating final keep mask on the *post-densify* index space.
    # Construct keep mask on N1, then compute new indices via cumsum.
    # If no pruning happened, mapping is identity for old ids.
    if prune_ids.numel() == 0:
        unaffected_final_ids = unaffected_old_ids
    else:
        # keep mask used during prune: keep[prune_ids]=False, keep[new]=True
        keepN1 = torch.ones(N1, device=device, dtype=torch.bool)
        keepN1[prune_ids] = False
        # new index = cumsum(keep)-1 for kept entries
        new_index = torch.cumsum(keepN1.to(torch.int64), dim=0) - 1
        unaffected_final_ids = new_index[unaffected_old_ids]

    info = {
        "N0": N0,
        "densified": int(K),
        "pruned": int(prune_ids.numel()),
        "N_final": N2,
    }

    # ==========
    # 5) COMPUTE AFFECTED IDS (final index space)
    # ==========
    # 在最终的 N2 长度空间中，创建一个全为 True 的 mask
    is_affected = torch.ones(N2, device=device, dtype=torch.bool)
    
    # 将不受影响的索引位置设为 False
    if unaffected_final_ids.numel() > 0:
        is_affected[unaffected_final_ids] = False
    
    # 提取剩下的索引，即为受影响的（包括分裂产生的、克隆产生的、以及位置/尺度被修改过的）
    affected_final_ids = torch.nonzero(is_affected, as_tuple=False).squeeze(-1)

    return new_optimizer, info, unaffected_final_ids, affected_final_ids


@torch.no_grad()
def densify_and_prune(
    scene,
    optimizer,
    *,
    keep_new_from_prune: bool = True,  # prune_ids 默认是旧 index 空间；不去 prune 新的
    elong_thr: float = 5.0,
    eps: float = 1e-8,
):
    """
    Densify (split) + Prune + Rebuild optimizer.

    densify:
      - for each old id i in densify_ids:
          * create one new primitive appended at end
          * old & new scales /= scale_div
          * old pos += δ, new pos -= δ (paired jitter)
          * old & new MLP params random init
    prune:
      - remove prune_ids (by default interpreted in old index space)
    returns:
      new_optimizer, info, unaffected_ids_final
    """

    Ssum = scene.stat_sum_w.float()
    Hhit = scene.stat_hits.float()
    ratio = Hhit / (Ssum + 1e-12)
    q99 = torch.quantile(Hhit, 0.99)

    w1 = (Hhit < q99) & (ratio > 500)
    w2 = (Hhit >= q99) & (ratio > 200)
    w_transparent = w1 | w2
    w_small = (Ssum <= 20)
    w_prune = w_small | w_transparent


    device = scene.positions.device
    N0 = int(scene.num_primitives)
    S = scene.stat_sum_w.float()
    E = scene.stat_contrib_count.float()
    Err = scene.stat_sum_werr.float() / (E + 1e-6)
    score = S * Err   # ⭐ 核心
    K = int(0.1 * S.numel())  # top 10%
    cand_ids = torch.topk(score, k=K).indices


    # ---- densify = cand_ids excluding prune
    keep = ~w_prune[cand_ids]          # (K,) 只在 cand_ids 上索引
    densify_ids = cand_ids[keep]
    N = scene.stat_sum_w.numel()
    w_densify = torch.zeros(N, dtype=torch.bool, device=device)
    w_densify[densify_ids] = True
    w_densify.sum()


    prune_ids = torch.nonzero(w_prune, as_tuple=False).squeeze(1)
    densify_ids = torch.nonzero(w_densify, as_tuple=False).squeeze(1)
    densify_ids = torch.unique(densify_ids)
    prune_ids   = torch.unique(prune_ids)


    # sanitize ranges (old index space)
    if densify_ids.numel() > 0:
        if int(densify_ids.min()) < 0 or int(densify_ids.max()) >= N0:
            raise ValueError(f"densify_ids out of range: [0,{N0-1}]")
    if prune_ids.numel() > 0:
        if int(prune_ids.min()) < 0 or int(prune_ids.max()) >= N0:
            raise ValueError(f"prune_ids out of range (assumed old space): [0,{N0-1}]")


    
    # ==========
    # 1) DENSIFY (Split vs Clone based on Scale)
    # ==========
    K = int(densify_ids.numel())
    if K > 0:
        # --- scales statistics ---
        sc = scene.log_scales.detach().exp()                 # (N,3)
        smax = sc.max(dim=-1).values                         # (N,)
        smed = sc.median(dim=-1).values                      # (N,)
        elong = smax / (smed + eps)                          # (N,)
        is_elong = elong > elong_thr                         # (N,)
        typ = torch.quantile(smax, 0.90)                     # scalar

        # 只取 densify 的那部分
        sc_d   = sc[densify_ids]                             # (K,3)
        smed_d = smed[densify_ids]                           # (K,)
        is_elong_d = is_elong[densify_ids]                   # (K,)

        # --- longest axis direction in world ---
        # 取最长轴是哪一维：0/1/2
        axis_idx = sc_d.argmax(dim=-1)                       # (K,)
        # 旋转矩阵
        R = quat_to_rotmat_wxyz(scene.quats.detach()[densify_ids])  # (K,3,3)

        # 局部基向量 e0/e1/e2 -> world：取 R 的对应列
        # R[..., :, j] 是第 j 个局部轴在 world 的方向
        axis_world = R.gather(
            dim=2,
            index=axis_idx.view(-1, 1, 1).expand(-1, 3, 1)
        ).squeeze(-1)                                        # (K,3)
        axis_world = axis_world / (axis_world.norm(dim=-1, keepdim=True) + 1e-12)

        # --- decide sign using positional Adam exp_avg ---
        state = optimizer.state.get(scene.positions, None)
        if state is not None and "exp_avg" in state:
            m = state["exp_avg"].detach()                    # (N,3)
            gdir = m[densify_ids]                            # (K,3)
            gdir = gdir / (gdir.norm(dim=-1, keepdim=True) + 1e-12)
            # dot > 0 => axis_world 与 gdir 同向；否则取反
            sign = torch.where((axis_world * gdir).sum(dim=-1, keepdim=True) >= 0,
                            torch.ones((densify_ids.numel(),1), device=device),
                            -torch.ones((densify_ids.numel(),1), device=device))
        else:
            # 没有 exp_avg 就默认 + 方向（或你也可以随机）
            sign = torch.ones((densify_ids.numel(),1), device=device)

        # --- offset magnitude d ---
        # 经验上：细长的用 smed 驱动（更稳），并 clamp 到 0.3*typ
        max_offset = 0.3 * typ
        # normal 的也可以 clamp 到同一个 max_offset，避免飞
        # 你也可以加一个 min_offset 防止几乎不动：
        min_offset = 1e-4 * typ
        d_elong  = torch.clamp(0.5 * smed_d, min=min_offset, max=max_offset)  # (K,)
        r_normal = sc_d.max(dim=-1).values                                  # (K,)
        d_normal = torch.clamp(0.5 * r_normal, min=min_offset, max=max_offset) # (K,)
        d = torch.where(is_elong_d, d_elong, d_normal).view(-1, 1)             # (K,1)
        pos_old = scene.positions.detach()[densify_ids]        # (K,3)
        pos_new = pos_old + sign * axis_world * d              # (K,3)

        # --- new properties ---
        positions_new = torch.cat([scene.positions.data, pos_new], dim=0)
        
        log_scales_old = scene.log_scales.data
        log_scales_app = log_scales_old[densify_ids].clone()
        log_scales_new = torch.cat([log_scales_old, log_scales_app], dim=0)
        shrink = math.log(5.0)
        m_old = log_scales_new[densify_ids].median(dim=-1).values   # (K,)
        log_scales_new[densify_ids] = m_old[:, None].expand(-1, 3)  # (K,3)
        log_scales_new[densify_ids] -= shrink                       # /5
        m_new = log_scales_new[N0:N0+K].median(dim=-1).values       # (K,)
        log_scales_new[N0:N0+K] = m_new[:, None].expand(-1, 3)      # (K,3)
        log_scales_new[N0:N0+K] -= shrink                           # /5

        quats_old      = scene.quats.data
        quats_app      = quats_old[densify_ids].clone()
        quats_new      = torch.cat([quats_old, quats_app],      dim=0)

        # ---- 3) append MLP params (NO helper functions) ----
        # 假设 scene.W1..b3 的 shape 已经是正确的：
        #   W1: (N0, H, IN)
        #   b1: (N0, H)
        #   W2: (N0, H, H)
        #   b2: (N0, H)
        #   W3: (N0, OUT, H)   # OUT 通常是 4 (rgb+sigma) 或 3/4，按你项目实际
        #   b3: (N0, OUT)
        std = 0.01
        dtype = scene.W1.dtype
        device = scene.W1.device
        new_ids = torch.arange(N0, N0 + K, device=device, dtype=torch.long)
        touched = torch.cat([densify_ids, new_ids], dim=0)
        W1_app = torch.randn((K,) + scene.W1.shape[1:], device=device, dtype=dtype) * std
        b1_app = torch.zeros((K,) + scene.b1.shape[1:], device=device, dtype=dtype)
        W2_app = torch.randn((K,) + scene.W2.shape[1:], device=device, dtype=dtype) * std
        b2_app = torch.zeros((K,) + scene.b2.shape[1:], device=device, dtype=dtype)
        W3_app = torch.randn((K,) + scene.W3.shape[1:], device=device, dtype=dtype) * std
        b3_app = torch.zeros((K,) + scene.b3.shape[1:], device=device, dtype=dtype)


        W1_new = torch.cat([scene.W1.data, W1_app], dim=0)
        b1_new = torch.cat([scene.b1.data, b1_app], dim=0)
        W2_new = torch.cat([scene.W2.data, W2_app], dim=0)
        b2_new = torch.cat([scene.b2.data, b2_app], dim=0)
        W3_new = torch.cat([scene.W3.data, W3_app], dim=0)
        b3_new = torch.cat([scene.b3.data, b3_app], dim=0)


        W1_new[touched] = torch.randn_like(W1_new[touched]) * std
        b1_new[touched] = 0
        W2_new[touched] = torch.randn_like(W2_new[touched]) * std
        b2_new[touched] = 0
        W3_new[touched] = torch.randn_like(W3_new[touched]) * std
        b3_new[touched] = 0

        # ---- 4) 写回 scene（用新 Parameter 替换）----
        scene.positions  = nn.Parameter(positions_new, requires_grad=True)
        scene.log_scales = nn.Parameter(log_scales_new, requires_grad=True)
        scene.quats      = nn.Parameter(quats_new, requires_grad=True)

        scene.W1 = nn.Parameter(W1_new, requires_grad=True)
        scene.b1 = nn.Parameter(b1_new, requires_grad=True)
        scene.W2 = nn.Parameter(W2_new, requires_grad=True)
        scene.b2 = nn.Parameter(b2_new, requires_grad=True)
        scene.W3 = nn.Parameter(W3_new, requires_grad=True)
        scene.b3 = nn.Parameter(b3_new, requires_grad=True)

        scene.num_primitives = int(scene.positions.shape[0])

    N1 = int(scene.num_primitives)

    # ==========
    # 2) PRUNE
    # ==========
    if prune_ids.numel() > 0:
        # prune_ids assumed in OLD index space [0, N0)
        # after densify, old indices are still [0, N0), new are [N0, N0+K)
        # optionally keep new always
        if keep_new_from_prune:
            prune_ids_eff = prune_ids
        else:
            prune_ids_eff = prune_ids  # (or allow passing new ids separately)

        keep = torch.ones(N1, device=device, dtype=torch.bool)
        keep[prune_ids_eff] = False

        scene.positions  = nn.Parameter(scene.positions.data[keep])
        scene.log_scales = nn.Parameter(scene.log_scales.data[keep])
        scene.quats = nn.Parameter(scene.quats.data[keep])

        scene.W1 = nn.Parameter(scene.W1.data[keep])
        scene.b1 = nn.Parameter(scene.b1.data[keep])
        scene.W2 = nn.Parameter(scene.W2.data[keep])
        scene.b2 = nn.Parameter(scene.b2.data[keep])
        scene.W3 = nn.Parameter(scene.W3.data[keep])
        scene.b3 = nn.Parameter(scene.b3.data[keep])

        scene.num_primitives = int(scene.positions.shape[0])

    N2 = int(scene.num_primitives)  # final

    # ==========
    # 3) REBUILD OPTIMIZER (with Stage 1 Strategy)
    # ==========
    optimizer = torch.optim.Adam([
        {"params": [scene.W1, scene.b1, scene.W2, scene.b2, scene.W3, scene.b3]},  # 默认 1e-3
        {"params": scene.log_scales, "lr": 1e-2},
        {"params": scene.positions, "lr": 1e-4},
        {"params": scene.quats, "lr": 1e-3},
        #{"params": scene.rotations, "lr": 1e-2},
    ], lr=1e-3)

    # ==========
    # 4) RETURN UNAFFECTED IDS (final index space)
    # ==========
    # “unaffected” means: old primitives not densified and not pruned.
    # Compute in old space first, then map through final keep-mask.
    unaffected_old = torch.ones(N0, device=device, dtype=torch.bool)
    if densify_ids.numel() > 0:
        unaffected_old[densify_ids] = False
    if prune_ids.numel() > 0:
        unaffected_old[prune_ids] = False
    unaffected_old_ids = torch.nonzero(unaffected_old, as_tuple=False).squeeze(-1)  # in [0,N0)

    # Map old ids -> final ids: after pruning, indices compress.
    # We can rebuild a mapping by simulating final keep mask on the *post-densify* index space.
    # Construct keep mask on N1, then compute new indices via cumsum.
    # If no pruning happened, mapping is identity for old ids.
    if prune_ids.numel() == 0:
        unaffected_final_ids = unaffected_old_ids
    else:
        # keep mask used during prune: keep[prune_ids]=False, keep[new]=True
        keepN1 = torch.ones(N1, device=device, dtype=torch.bool)
        keepN1[prune_ids] = False
        # new index = cumsum(keep)-1 for kept entries
        new_index = torch.cumsum(keepN1.to(torch.int64), dim=0) - 1
        unaffected_final_ids = new_index[unaffected_old_ids]

    info = {
        "N0": N0,
        "densified": int(K),
        "pruned": int(prune_ids.numel()),
        "N_final": N2,
    }

    # ==========
    # 5) COMPUTE AFFECTED IDS (final index space)
    # ==========
    # 在最终的 N2 长度空间中，创建一个全为 True 的 mask
    is_affected = torch.ones(N2, device=device, dtype=torch.bool)
    
    # 将不受影响的索引位置设为 False
    if unaffected_final_ids.numel() > 0:
        is_affected[unaffected_final_ids] = False
    
    # 提取剩下的索引，即为受影响的（包括分裂产生的、克隆产生的、以及位置/尺度被修改过的）
    affected_final_ids = torch.nonzero(is_affected, as_tuple=False).squeeze(-1)

    return optimizer, info, unaffected_final_ids, affected_final_ids





@torch.no_grad()
def zero_grads_for_ids(scene, fixed_ids):
    if fixed_ids.numel() == 0:
        return
    fixed_ids = fixed_ids.to(scene.positions.device)

    for p in [scene.positions, scene.log_scales, scene.quats, scene.W1, scene.b1, scene.W2, scene.b2, scene.W3, scene.b3]:
        if p.grad is None:
            continue
        p.grad[fixed_ids] = 0




@torch.no_grad()
def compute_multi_view_score(scene, dataset, idxs, threshold=0.05):
    scene.reset_w_stats()
    scene.eval()

    device = "cuda"
    for img_idx in tqdm(idxs):
        # ---------------- Load data ----------------
        rays_o_hw, rays_d_hw, img_hw3, pose = dataset[img_idx]
        Himg, Wimg = img_hw3.shape[0], img_hw3.shape[1]
        scene.set_image_size(Wimg, Himg)

        rays_o_batch = rays_o_hw.reshape(-1, 3).to(device, non_blocking=True)
        rays_d_batch = rays_d_hw.reshape(-1, 3).to(device, non_blocking=True)

        view_matrix = _c2w_to_view_matrix(pose, device=device)
        proj_matrix = _build_projection_from_focal(dataset.focal, Wimg, Himg, device=device)


        _, _ = scene.render_rgb_alpha_no_grad(
            rays_o_batch, rays_d_batch,
            score_threshold=1e-3,
            bg_color=torch.zeros(3, device=device),
            view_matrix=view_matrix,
            proj_matrix=proj_matrix,
            collect_w_stats=True,
        )



    sum_w = scene.stat_sum_w.detach().float().reshape(-1) 
    hits  = scene.stat_hits.detach().float().reshape(-1) 
    hits = torch.clamp(hits, min=1.0)  # 防止除零


    scales = torch.exp(scene.log_scales).reshape(-1)  # 只保留有效项的scales
    w = (sum_w / hits) * (scales) * (scales)# 只保留有效项
    N = w.numel()
    k = int(threshold * N)   # 前10%

    w_sorted, sorted_idx = torch.sort(w)   # 升序
    smallest_idx = sorted_idx[:k]                # 前10%最小

    return smallest_idx.cpu()  # 返回 CPU 上的索引






























@torch.no_grad()
def densify_tilecuda_from_pointcloud_random(scene, pointcloud_xyz: torch.Tensor, k: int = 2000):
    """
    One-shot densification for cumodel.NeuralPrimitiveSceneTileCUDA.

    scene has Parameters:
      positions:  (N,3)
      log_scales: (N,1)
      W1,b1,W2,b2,W3,b3 with leading dim N
      alpha_accum buffer: (N,)
      num_primitives int

    pointcloud_xyz: (M,3) tensor (cpu or cuda)
    """

    k = torch.min(torch.tensor(k), torch.tensor(int(pointcloud_xyz.shape[0])))

    device = scene.positions.device
    dtype_pos = scene.positions.dtype

    assert pointcloud_xyz.dim() == 2 and pointcloud_xyz.shape[1] == 3, \
        f"pointcloud_xyz must be (M,3), got {tuple(pointcloud_xyz.shape)}"
    M = int(pointcloud_xyz.shape[0])
    if M <= 0:
        raise ValueError("Point cloud is empty.")

    # --- sample new positions ---
    idx = torch.randint(0, M, (k,), device=pointcloud_xyz.device)
    new_pos = pointcloud_xyz[idx].to(device=device, dtype=dtype_pos).contiguous()  # (k,3)

    # --- median scale from current ---
    cur_scales = torch.exp(scene.log_scales.squeeze(-1))  # (N,)
    med_scale = torch.median(cur_scales).clamp_min(1e-6)
    new_log_scales = torch.log(med_scale).expand(k, 1).to(device=device, dtype=dtype_pos).contiguous()

    # --- init new per-primitive MLP params ---
    IN = int(scene.encoder.out_dim)
    H = int(scene.hidden_dim)
    OUT = 4

    # keep dtype consistent with existing weights (might be fp16/bf16 later)
    dtype_w = scene.W1.dtype

    new_W1 = (torch.randn(k, H, IN, device=device, dtype=dtype_w) * 0.01).contiguous()
    new_b1 = torch.zeros(k, H, device=device, dtype=dtype_w).contiguous()
    new_W2 = (torch.randn(k, H, H, device=device, dtype=dtype_w) * 0.01).contiguous()
    new_b2 = torch.zeros(k, H, device=device, dtype=dtype_w).contiguous()
    new_W3 = (torch.randn(k, OUT, H, device=device, dtype=dtype_w) * 0.01).contiguous()
    new_b3 = torch.zeros(k, OUT, device=device, dtype=dtype_w).contiguous()

    # --- concat + rewrap as nn.Parameter ---
    scene.positions = nn.Parameter(torch.cat([scene.positions.data, new_pos], dim=0))
    scene.log_scales = nn.Parameter(torch.cat([scene.log_scales.data, new_log_scales], dim=0))

    scene.W1 = nn.Parameter(torch.cat([scene.W1.data, new_W1], dim=0))
    scene.b1 = nn.Parameter(torch.cat([scene.b1.data, new_b1], dim=0))
    scene.W2 = nn.Parameter(torch.cat([scene.W2.data, new_W2], dim=0))
    scene.b2 = nn.Parameter(torch.cat([scene.b2.data, new_b2], dim=0))
    scene.W3 = nn.Parameter(torch.cat([scene.W3.data, new_W3], dim=0))
    scene.b3 = nn.Parameter(torch.cat([scene.b3.data, new_b3], dim=0))

    # --- buffers / metadata ---
    scene.alpha_accum = torch.cat(
        [scene.alpha_accum, torch.zeros(k, device=device, dtype=scene.alpha_accum.dtype)],
        dim=0
    )
    scene.num_primitives = int(scene.positions.shape[0])

    return {
        "added": k,
        "new_N": scene.num_primitives,
        "median_scale": float(med_scale.detach().cpu()),
    }



# ==================== Tile-L1 based densification utilities ====================

def _build_projection_from_focal(focal: float, W: int, H: int, near: float = 0.001, far: float = 100.0,
                                 device: str | torch.device = "cpu") -> torch.Tensor:
    """Match train.py / viewer perspective projection convention used by tile_cuda culling."""
    focal = float(focal)
    fov_y = 2.0 * float(torch.atan(torch.tensor((0.5 * H) / (focal + 1e-8), device=device)))
    aspect = float(W) / float(H)
    t = float(torch.tan(torch.tensor(fov_y / 2.0, device=device)))
    r = t * aspect
    P = torch.zeros(4, 4, dtype=torch.float32, device=device)
    P[0, 0] = 1.0 / r
    P[1, 1] = 1.0 / t
    P[2, 2] = -(far + near) / (far - near)
    P[3, 2] = 1.0
    P[2, 3] = -(2 * far * near) / (far - near)
    return P


def _c2w_to_view_matrix(c2w_3x4: torch.Tensor, device: str | torch.device) -> torch.Tensor:
    """Convert (3,4) camera-to-world to (4,4) world-to-camera (view) matrix."""
    c2w_h = torch.eye(4, dtype=torch.float32, device=device)
    c2w_h[:3, :4] = c2w_3x4.to(device=device, dtype=torch.float32)
    return torch.inverse(c2w_h)


@torch.no_grad()
def sample_worst_tiles_l1(
    scene,
    dataset,
    k_images: int = 8,
    m_tiles: int = 10,
    tile_size: int = 32,
    score_threshold: float = 1e-3,
    bg_color: torch.Tensor | None = None,
    device: str | torch.device = "cuda",
    seed: int | None = None,
):
    """Randomly sample k images, compute per-tile L1(rgb) error, and return top-m worst tiles per image.

    Requirements:
      - dataset[idx] returns (rays_o_hw, rays_d_hw, img_hw3, pose_3x4) like mip360Dataset
      - scene is a NeuralPrimitiveSceneTileCUDA-like model and provides either:
          * scene.render_rgb_t_no_grad(...) -> (rgb, t, alpha)  [preferred], or
          * scene(...) -> rgb (B,3) for tile_cuda engine.
      - scene must be configured for full-frame rays (B == W*H).
    Returns:
      results: list of dict per sampled image:
        {
          'img_idx': int,
          'H': int, 'W': int,
          'tile_scores': (Ty, Tx) float tensor on CPU,
          'top_tiles_yx': (m,2) long tensor on CPU,  # (ty, tx)
          'top_scores': (m,) float tensor on CPU,
        }
    """
    if seed is not None:
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        perm = torch.randperm(len(dataset), generator=g)
    else:
        perm = torch.randperm(len(dataset))

    k_images = int(min(k_images, len(dataset)))
    pick = perm[:k_images].tolist()

    results = []
    scene_device = scene.positions.device if hasattr(scene, "positions") else torch.device(device)

    for img_idx in pick:
        rays_o_hw, rays_d_hw, img_hw3, pose = dataset[int(img_idx)]
        H, W = int(img_hw3.shape[0]), int(img_hw3.shape[1])

        # ensure renderer knows image size (tile_cuda path uses this for tile grid)
        if hasattr(scene, "set_image_size"):
            scene.set_image_size(W, H)

        rays_o = rays_o_hw.reshape(-1, 3).to(scene_device, non_blocking=True)
        rays_d = rays_d_hw.reshape(-1, 3).to(scene_device, non_blocking=True)
        gt = img_hw3.reshape(-1, 3).to(scene_device, non_blocking=True)

        view = _c2w_to_view_matrix(pose, device=scene_device)
        focal = float(getattr(dataset, "focal", None) or getattr(dataset, "focal_length", None))
        proj = _build_projection_from_focal(focal, W=W, H=H, device=scene_device)

        if bg_color is None:
            bg = torch.zeros(3, device=scene_device, dtype=torch.float32)
        else:
            bg = bg_color.to(device=scene_device, dtype=torch.float32)

        # forward (RGB only is fine for tile scoring)
        if hasattr(scene, "render_rgb_t_no_grad"):
            pred_rgb, _tmap, _alpha = scene.render_rgb_t_no_grad(
                rays_o, rays_d,
                score_threshold=float(score_threshold),
                bg_color=bg,
                view_matrix=view,
                proj_matrix=proj,
            )
        else:
            pred_rgb = scene(
                rays_o, rays_d,
                score_threshold=float(score_threshold),
                bg_color=bg,
                view_matrix=view,
                proj_matrix=proj,
            )

        # per-pixel L1 (sum over channels): (B,)
        l1 = (pred_rgb - gt).abs().sum(dim=-1)  # (B,)
        l1_hw = l1.view(H, W)  # (H,W)

        # pad to tile multiple (match tile grid logic)
        Ty = (H + tile_size - 1) // tile_size
        Tx = (W + tile_size - 1) // tile_size
        Hpad, Wpad = Ty * tile_size, Tx * tile_size
        pad_bottom = Hpad - H
        pad_right = Wpad - W
        if pad_bottom or pad_right:
            l1_hw = F.pad(l1_hw, (0, pad_right, 0, pad_bottom), mode="constant", value=0.0)

        # reshape and sum per tile -> (Ty, Tx)
        tile_scores = l1_hw.view(Ty, tile_size, Tx, tile_size).sum(dim=(1, 3))

        # top-m worst tiles
        flat = tile_scores.flatten()
        m_eff = int(min(m_tiles, flat.numel()))
        top_scores, top_idx = torch.topk(flat, k=m_eff, largest=True, sorted=True)
        top_ty = (top_idx // Tx).to(torch.long)
        top_tx = (top_idx % Tx).to(torch.long)
        top_yx = torch.stack([top_ty, top_tx], dim=-1)

        results.append({
            "img_idx": int(img_idx),
            "H": H, "W": W,
            "tile_scores": tile_scores.detach().cpu(),
            "top_tiles_yx": top_yx.detach().cpu(),
            "top_scores": top_scores.detach().cpu(),
        })

    return results



@torch.no_grad()
def tiles_to_points_from_tmap(
    scene,
    dataset,
    tile_results: list,
    tile_size: int = 32,
    score_threshold: float = 1e-3,
    bg_color: torch.Tensor | None = None,
    alpha_min: float = 1e-3,
    device: str | torch.device = "cuda",
):
    """Convert worst-tile selections into a point cloud using rendered per-pixel depth t.

    For each (image idx, top_tiles_yx) entry in tile_results:
      1) Re-render that image with scene.render_rgb_t_no_grad(...) to obtain t_map and alpha.
      2) For each selected tile, pick the center pixel and back-project:
            X = ray_o + t * normalize(ray_d)
      3) Collect all such X into a (M,3) tensor suitable for densify_tilecuda_from_pointcloud_*.

    Notes:
      - Requires scene.render_rgb_t_no_grad to exist and to return per-pixel t_map (B,1).
      - Uses per-pixel rays from dataset[idx] (rays_o_hw, rays_d_hw).
      - Filters invalid points with alpha < alpha_min, non-finite t, or t <= 0.
    """
    if not hasattr(scene, "render_rgb_t_no_grad"):
        raise AttributeError("scene.render_rgb_t_no_grad(...) is required to obtain t_map for back-projection.")

    scene_device = scene.positions.device if hasattr(scene, "positions") else torch.device(device)

    all_pts = []
    for r in tile_results:
        img_idx = int(r["img_idx"])
        top_yx_cpu = r["top_tiles_yx"]  # (m,2) on CPU
        if top_yx_cpu.numel() == 0:
            continue

        rays_o_hw, rays_d_hw, img_hw3, pose = dataset[img_idx]
        H, W = int(img_hw3.shape[0]), int(img_hw3.shape[1])

        if hasattr(scene, "set_image_size"):
            scene.set_image_size(W, H)

        rays_o = rays_o_hw.reshape(-1, 3).to(scene_device, non_blocking=True)
        rays_d = rays_d_hw.reshape(-1, 3).to(scene_device, non_blocking=True)

        view = _c2w_to_view_matrix(pose, device=scene_device)
        focal = float(getattr(dataset, "focal", None) or getattr(dataset, "focal_length", None))
        proj = _build_projection_from_focal(focal, W=W, H=H, device=scene_device)

        if bg_color is None:
            bg = torch.zeros(3, device=scene_device, dtype=torch.float32)
        else:
            bg = bg_color.to(device=scene_device, dtype=torch.float32)

        # We only need t_map and alpha; rgb is ignored here.
        _rgb, t_map, alpha = scene.render_rgb_t_no_grad(
            rays_o, rays_d,
            score_threshold=float(score_threshold),
            bg_color=bg,
            view_matrix=view,
            proj_matrix=proj,
        )
        # t_map, alpha: (B,1)
        t_map = t_map.view(-1)          # (B,)
        alpha = alpha.view(-1)          # (B,)

        # normalize ray_d to match renderer's definition of t
        rays_d_n = F.normalize(rays_d, dim=-1)

        # tile center pixels (clamped inside image)
        top_yx = top_yx_cpu.to(device=scene_device)
        ty = top_yx[:, 0]
        tx = top_yx[:, 1]
        cy = ty * tile_size + (tile_size // 2)
        cx = tx * tile_size + (tile_size // 2)
        cy = torch.clamp(cy, 0, H - 1)
        cx = torch.clamp(cx, 0, W - 1)

        pix = cy * W + cx  # (m,)
        pix = pix.to(torch.long)

        t_sel = t_map[pix]
        a_sel = alpha[pix]
        ro_sel = rays_o[pix]
        rd_sel = rays_d_n[pix]

        valid = torch.isfinite(t_sel) & (t_sel > 0.0) & torch.isfinite(a_sel) & (a_sel >= float(alpha_min))
        if valid.any():
            t_sel = t_sel[valid].unsqueeze(-1)  # (m',1)
            pts = ro_sel[valid] + t_sel * rd_sel[valid]  # (m',3)
            all_pts.append(pts)

    if not all_pts:
        # return empty tensor on scene_device
        return torch.empty((0, 3), device=scene_device, dtype=torch.float32)

    pointcloud_xyz = torch.cat(all_pts, dim=0).contiguous()  # (M,3)
    return pointcloud_xyz



import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

@torch.no_grad()
def densify_tilecuda_from_points_reset_pair_nearest(
    scene,
    new_pos_xyz: torch.Tensor,
    k: int = 2000,
    pos_jitter: float = 0.0,
    init_std: float = 0.01,
    chunk_n_old: int = 32768,
    reset_alpha_accum_for_old: bool = True,
):
    """
    Pair-reset densification (your new plan):
      - For each new position, find nearest existing primitive (nn).
      - Initialize BOTH:
          (a) the new primitive MLP params with random init
          (b) the nearest old primitive MLP params with random init
      - New primitive's log_scale is copied from nearest old primitive.
      - Old primitive's position/log_scale are unchanged.
      - (Optional) reset alpha_accum for touched old primitives to 0.

    This is intentionally "violent" reset to force competition by removing old local dominance.
    """

    device = scene.positions.device
    dtype_pos = scene.positions.dtype
    dtype_w = scene.W1.dtype

    assert new_pos_xyz.dim() == 2 and new_pos_xyz.size(1) == 3, \
        f"new_pos_xyz must be (M,3), got {tuple(new_pos_xyz.shape)}"
    M = int(new_pos_xyz.size(0))
    if M <= 0:
        return {"added": 0, "touched_old": 0, "new_N": int(scene.positions.size(0))}

    k_eff = int(min(int(k), M))
    pick = torch.randint(0, M, (k_eff,), device=new_pos_xyz.device)
    new_pos = new_pos_xyz[pick].to(device=device, dtype=dtype_pos).contiguous()
    if pos_jitter > 0:
        new_pos = new_pos + (float(pos_jitter) * torch.randn_like(new_pos))

    # --- nearest old indices ---
    old_pos = scene.positions.data
    N_old = int(old_pos.size(0))

    new_pos_f = new_pos.to(torch.float32)
    old_pos_f = old_pos.to(torch.float32)

    nn_idx = torch.empty((k_eff,), device=device, dtype=torch.long)
    nn_d2  = torch.full((k_eff,), float("inf"), device=device, dtype=torch.float32)

    for s in range(0, N_old, int(chunk_n_old)):
        e = min(N_old, s + int(chunk_n_old))
        chunk = old_pos_f[s:e]  # (C,3)
        d2 = torch.cdist(new_pos_f, chunk).square()  # (k,C)
        d2_min, d2_arg = d2.min(dim=1)               # (k,)
        better = d2_min < nn_d2
        nn_d2[better] = d2_min[better]
        nn_idx[better] = d2_arg[better] + s

    touched_old = torch.unique(nn_idx)  # (U,)

    # --- new log_scales = nearest old log_scales ---
    new_log_sc = scene.log_scales.data[nn_idx].to(dtype_pos).contiguous()  # (k,1)

    # --- random init for NEW MLP params ---
    IN = int(scene.encoder.out_dim)
    H = int(scene.hidden_dim)
    OUT = 4
    std = float(init_std)

    new_W1 = (torch.randn(k_eff, H, IN, device=device, dtype=dtype_w) * std).contiguous()
    new_b1 = torch.zeros(k_eff, H, device=device, dtype=dtype_w).contiguous()
    new_W2 = (torch.randn(k_eff, H, H, device=device, dtype=dtype_w) * std).contiguous()
    new_b2 = torch.zeros(k_eff, H, device=device, dtype=dtype_w).contiguous()
    new_W3 = (torch.randn(k_eff, OUT, H, device=device, dtype=dtype_w) * std).contiguous()
    new_b3 = torch.zeros(k_eff, OUT, device=device, dtype=dtype_w).contiguous()

    # --- random init for TOUCHED OLD MLP params (in-place reset) ---
    # keep positions/log_scales unchanged for old
    U = int(touched_old.numel())
    scene.W1.data[touched_old].copy_(torch.randn(U, H, IN, device=device, dtype=dtype_w) * std)
    scene.b1.data[touched_old].zero_()
    scene.W2.data[touched_old].copy_(torch.randn(U, H, H, device=device, dtype=dtype_w) * std)
    scene.b2.data[touched_old].zero_()
    scene.W3.data[touched_old].copy_(torch.randn(U, OUT, H, device=device, dtype=dtype_w) * std)
    scene.b3.data[touched_old].zero_()

    if reset_alpha_accum_for_old and hasattr(scene, "alpha_accum") and scene.alpha_accum is not None:
        scene.alpha_accum[touched_old] = 0

    # --- append NEW primitives ---
    scene.positions  = nn.Parameter(torch.cat([scene.positions.data,  new_pos],   dim=0))
    scene.log_scales = nn.Parameter(torch.cat([scene.log_scales.data, new_log_sc], dim=0))

    scene.W1 = nn.Parameter(torch.cat([scene.W1.data, new_W1], dim=0))
    scene.b1 = nn.Parameter(torch.cat([scene.b1.data, new_b1], dim=0))
    scene.W2 = nn.Parameter(torch.cat([scene.W2.data, new_W2], dim=0))
    scene.b2 = nn.Parameter(torch.cat([scene.b2.data, new_b2], dim=0))
    scene.W3 = nn.Parameter(torch.cat([scene.W3.data, new_W3], dim=0))
    scene.b3 = nn.Parameter(torch.cat([scene.b3.data, new_b3], dim=0))

    if hasattr(scene, "alpha_accum") and scene.alpha_accum is not None:
        scene.alpha_accum = torch.cat(
            [scene.alpha_accum, torch.zeros(k_eff, device=device, dtype=scene.alpha_accum.dtype)],
            dim=0
        )

    scene.num_primitives = int(scene.positions.size(0))

    return {
        "added": int(k_eff),
        "touched_old": int(U),
        "new_N": int(scene.num_primitives),
        "mean_nn_dist": float(torch.sqrt(nn_d2.mean()).detach().cpu()),
        "init_std": std,
    }




@torch.no_grad()
def densify_tilecuda_stage1_from_worst_tiles(
    scene, dataset,
    k_images=8, m_tiles=10, tile_size=32,
    score_threshold=1e-3, alpha_min=1e-3,
    add_k=2000,
    scale_mult=0.8, weight_noise=1e-3, pos_jitter=0.0,
    device="cuda", seed=None,
):
    tile_res = sample_worst_tiles_l1(
        scene, dataset, k_images=k_images, m_tiles=m_tiles,
        tile_size=tile_size, score_threshold=score_threshold,
        device=device, seed=seed
    )
    pts = tiles_to_points_from_tmap(
        scene, dataset, tile_res, tile_size=tile_size,
        score_threshold=score_threshold, alpha_min=alpha_min,
        device=device
    )
    return densify_tilecuda_from_points_reset_pair_nearest(
        scene, pts, k=add_k,
        pos_jitter=pos_jitter,
    )


@torch.no_grad()
def quat_to_rotmat_wxyz(q: torch.Tensor) -> torch.Tensor:
    """
    q: (...,4) in (w,x,y,z)
    return: (...,3,3)
    """
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(dim=-1)

    ww = w*w; xx = x*x; yy = y*y; zz = z*z
    wx = w*x; wy = w*y; wz = w*z
    xy = x*y; xz = x*z; yz = y*z

    r00 = 1 - 2*(yy + zz)
    r01 = 2*(xy - wz)
    r02 = 2*(xz + wy)

    r10 = 2*(xy + wz)
    r11 = 1 - 2*(xx + zz)
    r12 = 2*(yz - wx)

    r20 = 2*(xz - wy)
    r21 = 2*(yz + wx)
    r22 = 1 - 2*(xx + yy)

    R = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=-2)
    return R

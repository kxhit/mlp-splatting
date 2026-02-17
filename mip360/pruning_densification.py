# densify_prune.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math



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
def densify_prune_tilecuda_and_rebuild_optimizer(
    scene,
    optimizer,
    densify_ids,
    prune_ids,
    *,
    init_std: float = 0.01,
    jitter_frac: float = 0.1,   # δ = jitter_frac * (scale_old) * rand_unit
    scale_div: float = 2.0,      # scale /= 2
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
    # 1) DENSIFY (MLP Splitting with Opacity Compensation)
    # ==========
    if K > 0:
        pos = scene.positions.data
        lsc = scene.log_scales.data
        W1, b1, W2, b2, W3, b3 = scene.W1.data, scene.b1.data, scene.W2.data, scene.b2.data, scene.W3.data, scene.b3.data

        # --- A. 获取梯度方向 (打破对称性的关键) ---
        # 我们利用 positions 的梯度来决定新 MLP 的位移方向
        if scene.positions.grad is not None:
            # 提取要分裂点的梯度
            grads = scene.positions.grad[densify_ids] 
            # 这里的梯度方向代表 Loss 上升最快方向，我们向负梯度方向（Loss 下降方向）移动
            # 只取方向，归一化处理
            grad_dir = -grads / (torch.norm(grads, dim=-1, keepdim=True) + 1e-12)
        else:
            # 如果万一没有梯度信息，回退到随机方向
            grad_dir = torch.randn((K, 3), device=device, dtype=pos.dtype)
            grad_dir = grad_dir / (grad_dir.norm(dim=-1, keepdim=True) + 1e-12)

        # --- B. 透明度补偿 ---
        with torch.no_grad():
            old_bias_w = b3[densify_ids, 3]
            alpha = torch.sigmoid(old_bias_w)
            alpha = torch.clamp(alpha, 1e-6, 0.99) 
            new_alpha = 1.0 - torch.sqrt(1.0 - alpha)
            new_bias_w = torch.log(new_alpha / (1.0 - new_alpha + 1e-8))

        # --- C. 空间属性：旧的不动，新的沿梯度移动 ---
        scale_old = torch.exp(lsc[densify_ids].squeeze(-1))
        # 偏移量 δ = jitter_frac * scale
        delta = (float(jitter_frac) * scale_old)[:, None] * grad_dir 

        old_pos_new = pos.clone()  # 旧的 B 保持在 pos[densify_ids]
        new_pos = pos[densify_ids] + delta  # 新的 C 朝梯度优化方向跳一步

        old_lsc_new = lsc.clone()
        new_lsc = lsc[densify_ids].clone()

        # --- D. MLP 权重继承与扰动 ---
        old_W1_new, old_b1_new = W1.clone(), b1.clone()
        old_W2_new, old_b2_new = W2.clone(), b2.clone()
        old_W3_new, old_b3_new = W3.clone(), b3.clone()
        old_b3_new[densify_ids, 3] = new_bias_w

        new_W1, new_b1 = W1[densify_ids].clone(), b1[densify_ids].clone()
        new_W2, new_b2 = W2[densify_ids].clone(), b2[densify_ids].clone()
        new_W3, new_b3 = W3[densify_ids].clone(), b3[densify_ids].clone()
        new_b3[:, 3] = new_bias_w

        # 加入极小权重噪声进一步打破 MLP 内部的对称性
        noise_std = 1e-3
        old_W3_new[densify_ids] += torch.randn_like(old_W3_new[densify_ids]) * noise_std
        new_W3 += torch.randn_like(new_W3) * noise_std
        new_W2 += torch.randn_like(new_W2) * (noise_std * 0.1)

        
        # --- E. 参数写回 ---
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
    # 3) REBUILD OPTIMIZER (preserve group LRs/hparams)
    # ==========
    defaults = optimizer.defaults

    def group_lr(g):
        return g.get("lr", defaults.get("lr", 1e-3))

    lr0 = group_lr(optimizer.param_groups[0]) if len(optimizer.param_groups) > 0 else defaults.get("lr", 1e-3)
    lr1 = group_lr(optimizer.param_groups[1]) if len(optimizer.param_groups) > 1 else lr0
    lr2 = group_lr(optimizer.param_groups[2]) if len(optimizer.param_groups) > 2 else lr0

    mlp_params = [scene.W1, scene.b1, scene.W2, scene.b2, scene.W3, scene.b3]

    new_optimizer = torch.optim.Adam(
        [
            {"params": mlp_params,         "lr": lr0},
            {"params": [scene.log_scales], "lr": lr1},
            {"params": [scene.positions],  "lr": lr2},
        ],
        lr=lr0,
        betas=defaults.get("betas", (0.9, 0.999)),
        eps=defaults.get("eps", 1e-8),
        weight_decay=defaults.get("weight_decay", 0.0),
        amsgrad=defaults.get("amsgrad", False),
    )

    torch.cuda.empty_cache()

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
        "lrs": (float(lr0), float(lr1), float(lr2)),
        #"new_ids_appended_preprune": (list(range(N0, N0 + K)) if K > 0 else []),
    }

    return new_optimizer, info, unaffected_final_ids




@torch.no_grad()
def zero_grads_for_ids(scene, fixed_ids):
    if fixed_ids.numel() == 0:
        return
    fixed_ids = fixed_ids.to(scene.positions.device)

    for p in [scene.positions, scene.log_scales, scene.W1, scene.b1, scene.W2, scene.b2, scene.W3, scene.b3]:
        if p.grad is None:
            continue
        #p.grad[fixed_ids] = 0
























































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

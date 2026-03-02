from multiprocessing.util import info
import os
import uuid
from tqdm import tqdm
from datetime import datetime
import copy
import time
import wandb
USE_WANDB = True

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from vmodel import NeuralPrimitiveScene
# from vmodel import NeuralPrimitiveScene
from mip360_loader import mip360Dataset
from save_model import save_model
from pruning_densification import *
from utils.loss_utils import *



def _build_projection_from_focal(focal: float, W: int, H: int, near: float = 0.001, far: float = 100.0, device: str | torch.device = "cpu"):
    """Match the viewer's perspective projection convention used by tile_cuda culling."""
    focal = float(focal)
    # vertical fov from focal length (pixels)
    fov_y = 2.0 * float(torch.atan(torch.tensor((0.5 * H) / (focal + 1e-8))))
    aspect = float(W) / float(H)
    t = float(torch.tan(torch.tensor(fov_y / 2.0)))
    r = t * aspect
    P = torch.zeros(4, 4, dtype=torch.float32, device=device)
    P[0, 0] = 1.0 / r
    P[1, 1] = 1.0 / t
    P[2, 2] = -(far + near) / (far - near)
    P[3, 2] = 1.0
    P[2, 3] = -(2 * far * near) / (far - near)
    return P


def _c2w_to_view_matrix(c2w_3x4: torch.Tensor, device: str | torch.device):
    """Convert (3,4) camera-to-world to (4,4) world-to-camera (view) matrix."""
    c2w_h = torch.eye(4, dtype=torch.float32, device=device)
    c2w_h[:3, :4] = c2w_3x4.to(device=device, dtype=torch.float32)
    view = torch.inverse(c2w_h)
    return view



def train_scene_llff(
    data_dir,
    output_root,
    num_epochs=10,
    batch_size=4096,
    lr=1e-3,
    device='cuda',
    initial_num_primitives=1024,
    engine: str = "vmodel",
    tile_order: str = "per_pixel",
):
    def _nvtx_range(msg: str):
        class _NVTX:
            def __enter__(self_inner):
                if torch.cuda.is_available():
                    torch.cuda.nvtx.range_push(msg)
            def __exit__(self_inner, exc_type, exc, tb):
                if torch.cuda.is_available():
                    torch.cuda.nvtx.range_pop()
        return _NVTX()
    # Set saving directory
    os.makedirs(output_root, exist_ok=True)
    random_tag = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    random_tag = "16698"  # for debugging
    output_dir = os.path.join(output_root, random_tag)
    os.makedirs(output_dir, exist_ok=True)

    dataset = mip360Dataset(data_dir)

    """
    test on how to oganize parameters is suitable
    """
    if engine == "tile_cuda":
        # NOTE: tile_cuda training requires a CUDA backward path; we only wire the option here.
        # This renderer consumes full-frame rays (B == W*H) and view/proj matrices.
        from cumodel import NeuralPrimitiveSceneTileCUDA
        scene = NeuralPrimitiveSceneTileCUDA(
            num_primitives=initial_num_primitives,
            hidden_dim=32,
            pe_frequencies=6,
            initial_positions=dataset.positions,
        ).to(device)
        # ensure tile order mode for this run
        os.environ["MLPSP_TILE_ORDER"] = str(tile_order)
    else:
        scene = NeuralPrimitiveScene(
            num_primitives=initial_num_primitives,
            hidden_dim=32,
            pe_frequencies=6,
            initial_positions=dataset.positions,
        ).to(device)
    #optimizer = torch.optim.Adam(scene.parameters(), lr=lr)


    """
    # 加载已有 checkpoint 权重
    model_ckpt_path = "/home/yuzhou/Desktop/output/mip360/15581/scene_epoch_142000.pt"
    if os.path.exists(model_ckpt_path):
        ckpt = torch.load(model_ckpt_path, map_location=device)
        scene.load_state_dict(ckpt, strict=False)
        print(f"[Checkpoint] Loaded model from {model_ckpt_path}")
    """

    if engine == "tile_cuda":
        mlp_params = [scene.W1, scene.b1, scene.W2, scene.b2, scene.W3, scene.b3]
    else:
        mlp_params = list(scene.mlps.parameters())

    """
    optimizer = torch.optim.Adam([
        {"params": mlp_params},  # 默认 1e-3
        {"params": scene.log_scales, "lr": 5e-3},
        {"params": scene.positions, "lr": 2e-4},
        #{"params": scene.rotations, "lr": 1e-2},
    ], lr=1e-3)
    """



    start_decay = 80000
    total_iters = 110000

    base_lrs  = [1e-3, 1e-2, 1e-4]
    min_mults = [1e-2, 0.2, 0.3]   # 终点倍率（相对各自 base_lr）

    def late_cosine_mult(step, min_mult):
        # 0 ~ start_decay: 完全不变
        if step < start_decay:
            return 1.0

        # start_decay ~ total_iters: cosine 到 min_mult
        t = (step - start_decay) / max(1, (total_iters - start_decay))
        t = min(max(t, 0.0), 1.0)
        return min_mult + 0.5 * (1.0 - min_mult) * (1.0 + math.cos(math.pi * t))

    optimizer = torch.optim.Adam([
        {"params": mlp_params,        "lr": base_lrs[0]},
        {"params": scene.log_scales,  "lr": base_lrs[1]},
        {"params": scene.positions,   "lr": base_lrs[2]},
    ])

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[
            lambda s: late_cosine_mult(s, min_mults[0]),
            lambda s: late_cosine_mult(s, min_mults[1]),
            lambda s: late_cosine_mult(s, min_mults[2]),
        ]
    )

    #scheduler = StepLR(optimizer, step_size=1000, gamma=0.8)

    ckpt_path = os.path.join(output_dir, f"scene_epoch_0.pt")
    torch.save(scene.state_dict(), ckpt_path)
    print(f"[Checkpoint] Saved model to {ckpt_path}")


    if USE_WANDB:
        wandb.init(
            project="mlp-splatting",
            name=f"mip360_run_{random_tag}",
            config={
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "lr": lr,
                "num_primitives": initial_num_primitives,
                "hidden_dim": 32,
                "pe_frequencies": 6,
                "num_layers": 2,   # 当前是单隐藏层，如果以后改成两层可以记录成2
            }
        )

    loss_accum = 0.0
    loss_count = 0

    """
    # Load checkpoint for continued training
    ckpt = torch.load("scene_ckpt_tmp.pt")
    model_dict = scene.state_dict()
    ckpt_dict = {k: v for k, v in ckpt.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(ckpt_dict)
    scene.load_state_dict(model_dict)
    print(f"[Checkpoint] Partially loaded model from scene_ckpt_tmp.pt (matched {len(ckpt_dict)} tensors)")
    """

    rays_o_all, rays_d_all, rgbs_all = dataset.get_all_rays(device=None)

    N = rays_o_all.shape[0]
    if engine != "tile_cuda":
        assert batch_size <= N, "Batch size must be less than or equal to the number of rays."
        perm = torch.randperm(N)
        ptr = 0


    # train loop
    printed_batch = False
    profile_enabled = os.environ.get("MLPSP_PROFILE", "0") == "1"
    profile_start = int(os.environ.get("MLPSP_PROFILE_START", "10"))
    profile_steps = int(os.environ.get("MLPSP_PROFILE_STEPS", "1"))
    profiler_active = False
    timing_enabled = os.environ.get("MLPSP_TIMING", "0") == "1"
    timing_every = int(os.environ.get("MLPSP_TIMING_EVERY", "50"))
    did_densify = False
    WARMUP_ITERS = 2000
    fixed_ids = torch.empty(0, dtype=torch.long, device=scene.positions.device)



    """
    with torch.no_grad():
        for i in range(scene.num_primitives):
            scene.b3[i, 3] = 2.1972
    """
    for epoch in tqdm(range(num_epochs)):
        """
        if epoch % 500 == 0:
            print(torch.min(scene.b3[:, 3]), torch.max(scene.b3[:, 3]), torch.mean(scene.b3[:, 3]))
            print(torch.min(scene.b3[:, :3]), torch.max(scene.b3[:, :3]), torch.mean(scene.b3[:, :3]))
        """
        wall_start = time.perf_counter() if timing_enabled and (epoch % timing_every == 0) else None
        data_start = wall_start

        bg_color = torch.rand(3, device=device)

        # Update learning rates
        #mlp_lr = get_mlp_lr(epoch, max_steps=num_epochs)
        #position_lr = get_position_lr(epoch, warmup_steps=500, max_steps=num_epochs)
        #optimizer.param_groups[0]['lr'] = mlp_lr  # param_groups[0] 是 scene.mlps
        #optimizer.param_groups[2]['lr'] = position_lr  # param_groups[2] 是 scene.positions

        if engine == "tile_cuda":
            # Full-frame training path (one randomly selected training image per step).
            # dataset[idx] returns rays_o(H,W,3), rays_d(H,W,3), img(H,W,3), pose(3,4)
            img_idx = int(torch.randint(0, len(dataset), ()).item())
            rays_o_hw, rays_d_hw, img_hw3, pose = dataset[img_idx]
            Himg, Wimg = img_hw3.shape[0], img_hw3.shape[1]
            scene.set_image_size(Wimg, Himg)

            rays_o_batch = rays_o_hw.reshape(-1, 3).to(device, non_blocking=True)
            rays_d_batch = rays_d_hw.reshape(-1, 3).to(device, non_blocking=True)
            rgbs_batch = img_hw3.reshape(-1, 3).to(device, non_blocking=True)

            view_matrix = _c2w_to_view_matrix(pose, device=device)
            proj_matrix = _build_projection_from_focal(dataset.focal, Wimg, Himg, device=device)
            if not printed_batch:
                print(f"[Batch] engine=tile_cuda batch_size={Wimg * Himg} (full frame {Wimg}x{Himg})")
                printed_batch = True
        else:
            if ptr + batch_size > N:
                perm = torch.randperm(N)
                ptr = 0

            idx = perm[ptr:ptr + batch_size]
            ptr += batch_size

            rays_o_batch = rays_o_all[idx].to(device, non_blocking=True)
            rays_d_batch = rays_d_all[idx].to(device, non_blocking=True)
            rgbs_batch   = rgbs_all[idx].to(device, non_blocking=True)
            if not printed_batch:
                print(f"[Batch] engine={engine} batch_size={int(batch_size)} (random rays)")
                printed_batch = True

        data_end = time.perf_counter() if wall_start is not None else None
        if wall_start is not None and torch.cuda.is_available():
            fwd_start = torch.cuda.Event(enable_timing=True)
            fwd_end = torch.cuda.Event(enable_timing=True)
            bwd_start = torch.cuda.Event(enable_timing=True)
            bwd_end = torch.cuda.Event(enable_timing=True)
            opt_start = torch.cuda.Event(enable_timing=True)
            opt_end = torch.cuda.Event(enable_timing=True)
            fwd_start.record()


        """
        if epoch % 500 < 500:
            # use random sampled rays
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
                        num_patches_h=4, num_patches_w=4  # horizontal 4, vertical 4 patches
                    )  
        """

        if profile_enabled and (epoch == profile_start):
            torch.cuda.profiler.start()
            profiler_active = True


        with _nvtx_range(f"train/step_{epoch}"):
            optimizer.zero_grad()                       # 1. Clear gradients
            if engine == "tile_cuda":
                with _nvtx_range("train/forward_tile_cuda"):
                    pred = scene(
                        rays_o_batch, rays_d_batch,
                        score_threshold=1e-3,
                        bg_color=bg_color,
                        view_matrix=view_matrix,
                        proj_matrix=proj_matrix,
                    )
            else:
                with _nvtx_range("train/forward_vmodel"):
                    pred = scene(rays_o_batch, rays_d_batch, score_threshold=1e-3, bg_color=bg_color)    # 2. Forward pass
            loss_mse = F.mse_loss(pred, rgbs_batch)         # 3. Compute loss (MSE)

            H, W = Himg, Wimg
            # reshape to BCHW for SSIM
            pred_img = pred.view(H, W, 3).permute(2, 0, 1).unsqueeze(0)      # (1,3,H,W)
            gt_img   = rgbs_batch.view(H, W, 3).permute(2, 0, 1).unsqueeze(0)# (1,3,H,W)
            # L1 can stay flattened OR use images; either is fine
            l1 = F.l1_loss(pred, rgbs_batch)  # scalar
            # SSIM similarity in [0,1]
            ssim_val = ssim(pred_img, gt_img)
            # 3DGS-style combined loss: (1-λ)*L1 + λ*(1-SSIM)
            lambda_dssim = 0.1
            loss = (1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - ssim_val) 

            def compute_repulsive_loss(scene, affected_ids, multiplier=0.1):
                if affected_ids.numel() < 2:
                    return torch.tensor(0.0, device=affected_ids.device)

                # 提取受影响点的坐标和尺度
                pos = scene.positions[affected_ids]  # (K, 3)
                # 假设 log_scales 是 (N, 1) 或 (N, 3)，取平均半径或最大半径
                scales = torch.exp(scene.log_scales[affected_ids]).mean(dim=-1) # (K,)

                # 计算两两之间的距离平方 (K, K)
                # dist_sq = |p_i|^2 + |p_j|^2 - 2 p_i·p_j
                dist_sq = torch.cdist(pos, pos, p=2)**2

                # 计算临界距离平方 (s_i + s_j)^2
                # 使用 broadcasting 得到 (K, K) 的半径和矩阵
                s_sum = scales[:, None] + scales[None, :]
                s_sum_sq = s_sum**2

                # 计算高斯排斥核
                # 当距离越小，loss 越大；当距离大于半径之和时，迅速衰减
                repulsion = torch.exp(-dist_sq / (2 * s_sum_sq + 1e-8))

                # 去掉对角线（自己对自己的排斥）
                mask = torch.eye(len(affected_ids), device=affected_ids.device).bool()
                repulsion = repulsion.masked_fill(mask, 0)

                return repulsion.mean() * multiplier
            
            if (epoch % 5000 > 10000000000000) and (epoch % 5000 < 1000) and (epoch < 20000):
                repulsive_loss = compute_repulsive_loss(scene, affected_ids, multiplier=0.01)
                loss += repulsive_loss

            if wall_start is not None and torch.cuda.is_available():
                fwd_end.record()
            with _nvtx_range("train/backward"):
                if wall_start is not None and torch.cuda.is_available():
                    bwd_start.record()
                loss.backward()                             # 4. Backpropagation
                if wall_start is not None and torch.cuda.is_available():
                    bwd_end.record()

        # import pdb; pdb.set_trace()

        #print("Position grad norm:", scene.positions.grad.norm().item())
        #with torch.no_grad():
            #print(f"[Epoch {epoch+1}] Position mean: {scene.positions.mean().item():.6f}, std: {scene.positions.std().item():.6f}")
        if wall_start is not None and torch.cuda.is_available():
            opt_start.record()


        zero_grads_for_ids(scene, fixed_ids)
        optimizer.step()
        scheduler.step()

        if wall_start is not None and torch.cuda.is_available():
            opt_end.record()

        if profile_enabled and profiler_active and (epoch == profile_start + profile_steps):
            torch.cuda.profiler.stop()
            profiler_active = False

        if wall_start is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                fwd_ms = fwd_start.elapsed_time(fwd_end)
                bwd_ms = bwd_start.elapsed_time(bwd_end)
                opt_ms = opt_start.elapsed_time(opt_end)
            else:
                fwd_ms = bwd_ms = opt_ms = 0.0
            data_ms = (data_end - data_start) * 1000.0 if data_end is not None else 0.0
            wall_ms = (time.perf_counter() - wall_start) * 1000.0
            print(
                f"[Timing] step={epoch} data_ms={data_ms:.2f} fwd_ms={fwd_ms:.2f} "
                f"bwd_ms={bwd_ms:.2f} opt_ms={opt_ms:.2f} wall_ms={wall_ms:.2f}",
                flush=True,
            )

        # Accumulate loss
        loss_accum += loss_mse.item()
        loss_count += 1

        # Print average loss every 100 epochs
        if (epoch + 1) % 100 == 0:
            avg_loss = loss_accum / loss_count
            print(f"[Epoch {epoch+1}] avg {loss_count} loss : {avg_loss:.6f}")
            # Reset accumulators
            loss_accum = 0.0
            loss_count = 0
            if USE_WANDB:   # log to Weights & Biases
                wandb.log({"avg_loss": avg_loss, "epoch": epoch + 1})
                # Reset again for safety (redundant but harmless)
                loss_accum = 0.0
                loss_count = 0

        """
        if epoch == 99:
            torch.save(scene.state_dict(), os.path.join(output_dir, "before_pruning.pt"))

        if (epoch == 1):
            scene.reset_w_stats()
            for img_idx in tqdm(range(len(dataset))):
                # ---------------- Load data ----------------
                rays_o_hw, rays_d_hw, img_hw3, pose = dataset[img_idx]
                Himg, Wimg = img_hw3.shape[0], img_hw3.shape[1]
                scene.set_image_size(Wimg, Himg)
                rays_o_batch = rays_o_hw.reshape(-1, 3).to(device, non_blocking=True)
                rays_d_batch = rays_d_hw.reshape(-1, 3).to(device, non_blocking=True)
                rgbs_batch = img_hw3.reshape(-1, 3).to(device, non_blocking=True)
                view_matrix = _c2w_to_view_matrix(pose, device=device)
                proj_matrix = _build_projection_from_focal(dataset.focal, Wimg, Himg, device=device)

                pred, _ = scene.render_rgb_alpha_no_grad(
                    rays_o_batch, rays_d_batch,
                    score_threshold=1e-3,
                    bg_color=torch.zeros(3, device=device),
                    view_matrix=view_matrix,
                    proj_matrix=proj_matrix,
                    collect_w_stats=True,
                    gt_rgb=rgbs_batch,
                )
        """
        if (engine == "tile_cuda") and (epoch == -200):
            """
            densify_scores = compute_multiview_densify_scores(
                scene, dataset,
                k_images=20,
                r_threshold=0.10,
                cover_mult=3.716,
                chunk_prims=128,
                seed=0,
            )

            # 例如：选 score > densify_thr 的 primitives 作为 densify target（覆盖多视角高误差区域）
            densify_thr = 2000.0
            densify_ids = torch.nonzero(densify_scores > densify_thr, as_tuple=False).squeeze(-1)
            """
            #densify_ids = torch.empty(0, dtype=torch.long, device=scene.positions.device) 

            """
            prune_scores = compute_multiview_prune_scores(
                scene, dataset,
                k_images=20,
                r_threshold=0.10,
                cover_mult=3.716,
                chunk_prims=128,
                seed=0,
            )
            prune_thr = 0.8
            prune_ids = torch.nonzero(prune_scores > float(prune_thr), as_tuple=False).squeeze(-1)
            prune_ids = torch.cat([prune_ids, get_prune_ids_by_scale(scene, threshold=1)])
            prune_ids = torch.unique(prune_ids)
            """

            """
            idxs = torch.randperm(len(dataset))[:] 
            prune_ids = compute_multi_view_score(scene, dataset, idxs, threshold=0.025)


            optimizer_, info, fixed_ids, affected_ids = densify_prune_tilecuda_and_rebuild_optimizer(
                scene, optimizer,
                densify_ids=densify_ids,   # 旧空间 ids
                prune_ids=prune_ids,       # 旧空间 ids（高分删）
                init_std=0.01,
                jitter_frac=0.10,
                scale_threshold=0.02, # scale > threshold => split; else clone  
                scale_div=1.5,
            )
            """
            optimizer, info, fixed_ids, affected_ids = densify_and_prune(scene, optimizer)
            optimizer.zero_grad(set_to_none=True)
            scene.reset_w_stats()
            print(info, affected_ids)
            torch.save(scene.state_dict(), os.path.join(output_dir, "after_pruning.pt"))
        

        """
        if epoch % 5000 == 1000 and epoch < 20000:
            # 在 train.py 的循环中
            print("--- Stage 2: Restoring learning rates for full optimization ---")
            for g in optimizer.param_groups:
                if g['name'] == "log_scales":
                    g['lr'] /= 1  # 恢复原始 Scale LR
                if g['name'] == "positions":
                    g['lr'] /= 1  # 恢复原始 Position LR
            fixed_ids = torch.empty(0, dtype=torch.long, device=scene.positions.device)
        """

        # Eval after 500 epochs
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

    # Convert log_scale to scale (input shape: (N, 1))
    scales = torch.exp(scene.log_scales).squeeze(-1)  # shape: (N,)

    # alpha + occupancy + scale restrictions
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


# TODO: improve densification strategy (current version is a placeholder)
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
    

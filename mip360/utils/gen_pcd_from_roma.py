# gen_pcd_from_roma.py
import sys, os, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../submodules"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../submodules/RoMa"))

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import open3d as o3d
from PIL import Image
from utils import read_write_model
from utils.read_write_model import qvec2rotmat

from romatch import roma_outdoor, roma_indoor
from romatch.utils import get_tuple_transform_ops



# ========== 简单相机类 ==========
class SimpleCamera:
    def __init__(self, image_id, qvec, tvec, K, image_path):
        self.image_id = image_id
        self.full_proj_transform = torch.from_numpy(
            self._build_proj_matrix(qvec, tvec, K)
        ).float()                    # (4,4) 放 CPU 即可，计算时搬到 GPU
        self.world_view_transform = self.full_proj_transform.flatten()
        self.image_path = image_path
        self.camera_center = torch.from_numpy(-qvec2rotmat(qvec).T @ tvec).float()  # CPU

    def _build_proj_matrix(self, qvec, tvec, K):
        R = qvec2rotmat(qvec)
        t = tvec.reshape(3, 1)
        Rt = np.hstack([R, t])       # [R|t]
        P = K @ Rt                   # 3x4
        P4 = np.eye(4)
        P4[:3, :4] = P
        return P4

    def load_image(self):
        im = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) / 255.0
        return torch.tensor(im.transpose(2, 0, 1), dtype=torch.float32)  # (C,H,W)


# ========== 简单场景类 ==========
class SimpleScene:
    def __init__(self, colmap_sparse, image_root):
        if os.path.exists(os.path.join(colmap_sparse, "cameras.bin")):
            cameras = read_write_model.read_cameras_binary(os.path.join(colmap_sparse, "cameras.bin"))
            images = read_write_model.read_images_binary(os.path.join(colmap_sparse, "images.bin"))
        else:
            cameras = read_write_model.read_cameras_text(os.path.join(colmap_sparse, "cameras.txt"))
            images = read_write_model.read_images_text(os.path.join(colmap_sparse, "images.txt"))

        self.cameras = []
        for img_id, img in images.items():
            cam = cameras[img.camera_id]
            # 这里假定 COLMAP PINHOLE：params=[fx, fy, cx, cy]
            K = np.array([[cam.params[0], 0,             cam.params[2]],
                          [0,             cam.params[1], cam.params[3]],
                          [0,             0,             1]])
            image_path = os.path.join(image_root, img.name)
            self.cameras.append(SimpleCamera(img_id, img.qvec, img.tvec, K, image_path))

    def getTrainCameras(self):
        return self.cameras


# ========== 工具函数 ==========
def pairwise_distances(matrix):
    return torch.cdist(matrix, matrix, p=2)

def k_closest_vectors(matrix, k):
    # 返回每个向量的 k 个最近邻索引（排除自身）
    distances = pairwise_distances(matrix)
    distances.fill_diagonal_(float('inf'))
    _, indices = torch.topk(distances, k, largest=False, dim=1)
    return indices

def select_cameras_kmeans(cameras, K):
    # 从相机姿态向量中用 K-means 选 K 个代表性相机（中心最近）
    from scipy.cluster.vq import kmeans, vq
    from scipy.spatial.distance import cdist
    if not isinstance(cameras, np.ndarray):
        cameras = np.asarray(cameras)
    cluster_centers, _ = kmeans(cameras, K)
    cluster_assignments, _ = vq(cameras, cluster_centers)
    selected_indices = []
    for k in range(K):
        cluster_members = cameras[cluster_assignments == k]
        distances = cdist([cluster_centers[k]], cluster_members)[0]
        nearest_camera_idx = np.where(cluster_assignments == k)[0][np.argmin(distances)]
        selected_indices.append(nearest_camera_idx)
    return selected_indices


def compute_warp_and_confidence(viewpoint_cam1, viewpoint_cam2, roma_model, device="cuda"):
    # 懒加载图像
    imA = viewpoint_cam1.load_image()
    imB = viewpoint_cam2.load_image()
    imA_np = imA.numpy().transpose(1, 2, 0)
    imB_np = imB.numpy().transpose(1, 2, 0)
    imA_pil = Image.fromarray((imA_np * 255).astype(np.uint8))
    imB_pil = Image.fromarray((imB_np * 255).astype(np.uint8))

    ws, hs = roma_model.w_resized, roma_model.h_resized
    test_transform = get_tuple_transform_ops(resize=(hs, ws), normalize=True)
    im_A, im_B = test_transform((imA_pil, imB_pil))
    batch = {"im_A": im_A[None].to(device), "im_B": im_B[None].to(device)}

    corresps = roma_model.forward(batch) if not roma_model.symmetric else roma_model.forward_symmetric(batch)
    finest_scale = 1
    hs, ws = roma_model.upsample_res if roma_model.upsample_preds else (hs, ws)

    certainty = corresps[finest_scale]["certainty"]
    im_A_to_im_B = corresps[finest_scale]["flow"]

    if roma_model.upsample_preds:
        im_A_to_im_B = F.interpolate(im_A_to_im_B, size=(hs, ws), mode="bilinear", align_corners=False)
        certainty = F.interpolate(certainty, size=(hs, ws), mode="bilinear", align_corners=False)

    im_A_to_im_B = im_A_to_im_B.permute(0, 2, 3, 1)
    im_A_coords = torch.stack(torch.meshgrid(
        torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=device),
        torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=device),
        indexing='ij'
    ), dim=0).permute(1, 2, 0).unsqueeze(0).expand(im_A_to_im_B.size(0), -1, -1, -1)

    warp = torch.cat((im_A_coords, im_A_to_im_B), dim=-1)  # (1,H,W,4) 每像素(xA_norm,yA_norm,dx,dy)
    certainty = certainty.sigmoid()                         # (1,1,H,W)

    return certainty[0, 0], warp[0], imB_np


def resize_batch(tensors_3d, tensors_4d, target_shape):
    # tensors_3d: (B,H,W)，tensors_4d: (B,H,W,4)
    target_H, target_W = target_shape
    resized_tensors_3d = F.interpolate(
        tensors_3d.unsqueeze(1), size=(target_H, target_W), mode="bilinear", align_corners=False
    ).squeeze(1)
    resized_tensors_4d = F.interpolate(
        tensors_4d.permute(0, 3, 1, 2), size=(target_H, target_W), mode="bilinear", align_corners=False
    ).permute(0, 2, 3, 1)
    return resized_tensors_3d, resized_tensors_4d


def aggregate_confidences_and_warps(viewpoint_stack, closest_indices, roma_model, source_idx):
    certainties_all, warps_all, imB_compound = [], [], []
    for nn in closest_indices[source_idx]:
        c, w, imB = compute_warp_and_confidence(viewpoint_stack[source_idx], viewpoint_stack[nn], roma_model)
        certainties_all.append(c); warps_all.append(w); imB_compound.append(imB)

    certainties_all = torch.stack(certainties_all, dim=0)  # (Nn,Hs,Ws)
    target_shape = imB_compound[0].shape[:2]               # (H_img, W_img)

    certainties_all_resized, warps_all_resized = resize_batch(
        certainties_all, torch.stack(warps_all, dim=0), target_shape
    )
    certainties_max, certainties_max_idcs = torch.max(certainties_all_resized, dim=0)   # (H,W), (H,W)

    H, W = certainties_max.shape
    warps_max = warps_all_resized[certainties_max_idcs, torch.arange(H).unsqueeze(1), torch.arange(W)]
    imA = viewpoint_stack[source_idx].load_image().numpy().transpose(1, 2, 0)
    imA = (imA * 255).astype(np.uint8)

    return certainties_max, warps_max, certainties_max_idcs, imA, imB_compound, certainties_all_resized, warps_all_resized


def extract_keypoints_and_colors(imA, imB_compound, certainties_max, certainties_max_idcs, matches, roma_model):
    # matches: (N,4) = (x1_norm, y1_norm, x2_norm, y2_norm)（来自 warp）
    H_A, W_A, _ = imA.shape
    H, W = certainties_max.shape

    # 将归一化坐标转回像素
    kptsA, kptsB = roma_model.to_pixel_coordinates(matches, W_A, H_A, H, W)
    kptsA_np = kptsA.detach().cpu().numpy()[:, [1, 0]]   # (N,2) = (row=yA, col=xA) -> [y, x]
    kptsB_np = kptsB.detach().cpu().numpy()

    # 颜色采样（A）
    kA_y = np.clip(kptsA_np[:, 0].astype(int), 0, H_A - 1)
    kA_x = np.clip(kptsA_np[:, 1].astype(int), 0, W_A - 1)
    kptsA_color = imA[kA_y, kA_x]

    # 确定 B 的来源（每像素来自哪个 NN）
    imB_compound_np = np.stack(imB_compound, axis=0)  # (Nn, H, W, 3)
    certainties_max_idcs_np = certainties_max_idcs.detach().cpu().numpy()

    # 颜色采样（B）
    kB_y = np.clip(kptsB_np[:, 1].astype(int), 0, H - 1)
    kB_x = np.clip(kptsB_np[:, 0].astype(int), 0, W - 1)
    kptsB_proj_matrices_idx = certainties_max_idcs_np[kB_y, kB_x]
    kptsB_color = imB_compound_np[kptsB_proj_matrices_idx, kB_y, kB_x]

    return kptsA_np, kptsB_np, kptsB_proj_matrices_idx, kptsA_color, kptsB_color


def triangulate_points(P1, P2, k1_x, k1_y, k2_x, k2_y, device="cuda"):
    """
    批量线性三角化 + 同时计算两个视角的重投影误差(EDGS 风格需要）
    返回:
      X_h: (N,4) 齐次坐标
      err1, err2: numpy (N,) 重投影误差
    """
    if not isinstance(P1, torch.Tensor): P1 = torch.tensor(P1, dtype=torch.float32)
    if not isinstance(P2, torch.Tensor): P2 = torch.tensor(P2, dtype=torch.float32)
    P1, P2 = P1.to(device).float(), P2.to(device).float()

    k1_x = torch.as_tensor(k1_x, dtype=torch.float32, device=device).view(-1, 1)
    k1_y = torch.as_tensor(k1_y, dtype=torch.float32, device=device).view(-1, 1)
    k2_x = torch.as_tensor(k2_x, dtype=torch.float32, device=device).view(-1, 1)
    k2_y = torch.as_tensor(k2_y, dtype=torch.float32, device=device).view(-1, 1)

    if P1.shape == (4, 4): P1 = P1[:3, :]
    if P2.shape == (4, 4): P2 = P2[:3, :]

    A = torch.stack([
        k1_x * P1[2, :] - P1[0, :],
        k1_y * P1[2, :] - P1[1, :],
        k2_x * P2[2, :] - P2[0, :],
        k2_y * P2[2, :] - P2[1, :],
    ], dim=1)   # (N,4,4)

    _, _, Vh = torch.linalg.svd(A)
    X_h = Vh[:, -1, :]
    X_h = X_h / (X_h[:, -1:] + 1e-8)  # 归一化

    # 重投影误差
    ones = torch.ones((X_h.shape[0], 1), device=device)
    X = X_h
    x1 = (P1 @ X.T).T; x1 = x1[:, :2] / x1[:, 2:3]
    x2 = (P2 @ X.T).T; x2 = x2[:, :2] / x2[:, 2:3]
    proj1_target = torch.cat([k1_x, k1_y], dim=1)
    proj2_target = torch.cat([k2_x, k2_y], dim=1)
    err1 = (x1 - proj1_target).norm(dim=1).detach().cpu().numpy()
    err2 = (x2 - proj2_target).norm(dim=1).detach().cpu().numpy()
    return X_h, err1, err2


def select_best_keypoints(NNs_triangulated_points, NNs_errors_proj1, NNs_errors_proj2, device="cuda"):
    """
    从多个邻居的三角化结果里, 选择投影误差(max(err1,err2)）最小的那个。
    Inputs:
      NNs_triangulated_points: torch (Nn, M, 4)
      NNs_errors_proj1: numpy (Nn, M)
      NNs_errors_proj2: numpy (Nn, M)
    Returns:
      best_points: torch (M, 4)
      best_errors: numpy (M,)
    """
    NNs_errors_proj = np.maximum(NNs_errors_proj1, NNs_errors_proj2)
    nn_best = np.argmin(NNs_errors_proj, axis=0)  # (M,)
    indices = torch.from_numpy(nn_best).long().to(device)
    n_indices = torch.arange(NNs_triangulated_points.shape[1]).long().to(device)
    best_points = NNs_triangulated_points[indices, n_indices, :]
    best_errors = np.min(NNs_errors_proj, axis=0)
    return best_points, best_errors


# ========== 主流程 ==========
def extract_pointcloud_with_corr(scene, cfg, device="cuda", roma_model=None, output_ply="output.ply"):
    if roma_model is None:
        roma_model = roma_indoor(device=device) if cfg.roma_model == "indoors" else roma_outdoor(device=device)
        roma_model.upsample_preds = False
        roma_model.symmetric = False

    M = cfg.matches_per_ref
    upper_thresh = roma_model.sample_thresh
    expansion_factor = 1  # 和 EDGS 一致，通常取 1

    viewpoint_stack = scene.getTrainCameras().copy()
    NUM_REFERENCE_FRAMES = min(cfg.num_refs, len(viewpoint_stack))
    NUM_NNS_PER_REFERENCE = min(cfg.nns_per_ref, len(viewpoint_stack))

    viewpoint_cam_all = torch.stack([x.world_view_transform.flatten() for x in viewpoint_stack], axis=0)
    selected_indices = sorted(select_cameras_kmeans(viewpoint_cam_all.cpu().numpy(), K=NUM_REFERENCE_FRAMES))
    closest_indices = k_closest_vectors(viewpoint_cam_all, NUM_NNS_PER_REFERENCE)
    closest_indices_selected = closest_indices.cpu().numpy()  # (Ncam, K)

    all_xyz, all_rgb = [], []

    # 1) warm-up（避免第一次前向巨慢/显存 spike）
    with torch.no_grad():
        if len(viewpoint_stack) >= 2:
            imA = viewpoint_stack[0].load_image().numpy().transpose(1, 2, 0)
            imB = viewpoint_stack[1].load_image().numpy().transpose(1, 2, 0)
            roma_model.match(Image.fromarray((imA * 255).astype(np.uint8)),
                             Image.fromarray((imB * 255).astype(np.uint8)),
                             device=device)
            torch.cuda.empty_cache()

    # 2) 遍历参考帧
    for source_idx in tqdm(selected_indices, desc="Extracting point cloud"):
        certainties_max, warps_max, certainties_max_idcs, imA, imB_compound, certainties_all, warps_all = \
            aggregate_confidences_and_warps(viewpoint_stack, closest_indices_selected, roma_model, source_idx)

        # 按 RoMa 的 certainty 抽样像素（matches）
        matches = warps_max.reshape(-1, 4)
        certainty = certainties_max.reshape(-1).clone()
        certainty[certainty > upper_thresh] = 1
        good_samples = torch.multinomial(
            certainty,
            num_samples=min(expansion_factor * M, len(certainty)),
            replacement=False
        )

        # 3) 对每个邻居 NN 都进行三角化，保存误差
        triangulated_all = []
        errors1_all, errors2_all = [], []
        kptsA_color_cache = None  # 用于最后颜色取值（A）

        for NN_idx in range(len(warps_all)):
            matches_NN = warps_all[NN_idx].reshape(-1, 4)[good_samples]
            kptsA_np, kptsB_np, _, kptsA_color, kptsB_color = extract_keypoints_and_colors(
                imA, imB_compound, certainties_max, certainties_max_idcs, matches_NN, roma_model
            )
            if kptsA_color_cache is None:
                kptsA_color_cache = kptsA_color  # 记录同一批像素的 A 端颜色

            P1 = viewpoint_stack[source_idx].full_proj_transform
            P2 = viewpoint_stack[closest_indices_selected[source_idx, NN_idx]].full_proj_transform

            triangulated_points, err1, err2 = triangulate_points(
                P1=P1, P2=P2,
                k1_x=kptsA_np[:M, 0], k1_y=kptsA_np[:M, 1],
                k2_x=kptsB_np[:M, 0], k2_y=kptsB_np[:M, 1],
                device=device
            )
            triangulated_all.append(triangulated_points)  # (M,4)
            errors1_all.append(err1)                      # (M,)
            errors2_all.append(err2)                      # (M,)

        # 4) 跨 NN 选最佳（三角化 + 误差）
        triangulated_all = torch.stack(triangulated_all, dim=0)  # (Nn, M, 4)
        best_points, best_errors = select_best_keypoints(
            NNs_triangulated_points=triangulated_all,
            NNs_errors_proj1=np.stack(errors1_all, axis=0),
            NNs_errors_proj2=np.stack(errors2_all, axis=0),
            device=device
        )

        # 5) 误差过滤 + 颜色
        xyz = best_points[:, :3]  # (M,3)
        mask = best_errors < cfg.proj_err_tolerance
        xyz = xyz[mask].detach().cpu().numpy()
        rgb = kptsA_color_cache[:M][mask]  # 用 A 侧颜色（也可做 A/B 融合）

        all_xyz.append(xyz)
        all_rgb.append(rgb)

    # 6) 写出 PLY
    if len(all_xyz) == 0:
        print("[WARN] 没有采到有效点，检查采样阈值/参数设置。")
        return

    all_xyz = np.concatenate(all_xyz, axis=0)
    all_rgb = np.concatenate(all_rgb, axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_xyz)
    pcd.colors = o3d.utility.Vector3dVector(all_rgb.astype(np.float32) / 255.0)
    o3d.io.write_point_cloud(output_ply, pcd)
    print(f"[INFO] 保存点云到 {output_ply}, 共 {all_xyz.shape[0]} 个点.")


def main():
    ap = argparse.ArgumentParser(description="Extract colored point cloud with RoMa correspondences (EDGS-style minimal)")
    ap.add_argument("--colmap_sparse", type=str, required=True, help="COLMAP sparse/0 目录")
    ap.add_argument("--image_root", type=str, required=True, help="图像目录（与 images.txt 的相对路径一致）")
    ap.add_argument("--out_ply", type=str, default="roma_points.ply", help="输出点云路径")
    ap.add_argument("--roma_model", type=str, default="outdoors", choices=["indoors", "outdoors"])
    ap.add_argument("--num_refs", type=int, default=180, help="参考帧数量(K-means 选）")
    ap.add_argument("--nns_per_ref", type=int, default=3, help="每个参考帧使用的最近邻数量")
    ap.add_argument("--matches_per_ref", type=int, default=15000, help="每个参考帧抽样的像素/匹配数量")
    ap.add_argument("--proj_err_tolerance", type=float, default=0.01, help="重投影误差阈值（像素）")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    class Cfg: pass
    cfg = Cfg()
    cfg.roma_model = args.roma_model
    cfg.num_refs = args.num_refs
    cfg.nns_per_ref = args.nns_per_ref
    cfg.matches_per_ref = args.matches_per_ref
    cfg.proj_err_tolerance = args.proj_err_tolerance

    scene = SimpleScene(args.colmap_sparse, args.image_root)
    extract_pointcloud_with_corr(scene, cfg, device=device, output_ply=args.out_ply)


if __name__ == "__main__":
    main()

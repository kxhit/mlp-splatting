from visualize import evaluate_and_visualize
import torch

 
"""
if __name__ == "__main__":
    #data_dir = "/home/xin/data/nerf/nerf_example_data/nerf_llff_data/fern" # change to your scene path

    data_dir = "/home/yuzhou/Desktop/datasets/nerf_llff_data/fern"  # change to your scene path


    model_ckpt_path = "/home/yuzhou/Desktop/output/20250716_174737_ba2d8537/scene_ckpt_tmp.pt"           # saved model path

    
    model_ckpt = torch.load(model_ckpt_path, map_location='cpu') if model_ckpt_path else None
    mlp_idx = list(range(0, 1))
    mlp_idx = None
    evaluate_and_visualize(data_dir, model_ckpt=model_ckpt, block=True, mlp_idx=mlp_idx, N=1024)
"""


def strip_orig_mod_prefix(state_dict):
    """Remove '_orig_mod.' prefix in keys, if exists."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_k = k[len("_orig_mod."):]
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict

if __name__ == "__main__":
    # Change to your scene path
    data_dir = "/home/yuzhou/Desktop/datasets/360_v2/bonsai"
    model_ckpt_path = "/home/yuzhou/Desktop/output/mip360/20250731_215818_67f2c929/scene_epoch_80000.pt"

    model_ckpt = torch.load(model_ckpt_path, map_location='cpu') if model_ckpt_path else None
    
    # ✅ 处理 key 前缀
    if model_ckpt is not None:
        model_ckpt = strip_orig_mod_prefix(model_ckpt)

    # 可修改为 list(range(N)) 以渲染所有 MLPs
    mlp_idx = list(range(894, 895))  
    #mlp_idx = None

    # 如果训练时是 N=1024 个 primitive，那这里也得一致
    evaluate_and_visualize(data_dir, model_ckpt=model_ckpt, block=True, mlp_idx=mlp_idx, N=1024)



"""
# ---------------- 下面画图，是为了看视锥还有mlp的位置，mlp的位置应该在视锥向前延伸范围内 ----------------------

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

from model import NeuralPrimitiveScene
from llff_loader import LLFFDataset
from torch.utils.data import DataLoader

def load_positions(ckpt_path, num_primitives):
    scene = NeuralPrimitiveScene(num_primitives=num_primitives).cpu()
    scene.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    return scene.positions.detach().cpu().numpy()

def get_all_rays(data_dir):
    dataset = LLFFDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    rays_o_list = []
    rays_d_list = []

    for rays_o, rays_d, _ in dataloader:
        rays_o_list.append(rays_o[0].numpy())
        rays_d_list.append(rays_d[0].numpy())
        break  # 只取第一张图像的 rays

    rays_o = np.concatenate(rays_o_list, axis=0)
    rays_d = np.concatenate(rays_d_list, axis=0)
    return rays_o, rays_d

if __name__ == "__main__":
    data_dir = "/home/yuzhou/Desktop/datasets/nerf_llff_data/fern"  # change to your scene path

    ckpt = "/home/yuzhou/Desktop/mlp-splatting/scene_ckpt_tmp.pt"           # saved model path
    #ckpt = "/home/yuzhou/Desktop/output/scene_epoch_0.pt"

    N = 1024

    positions = load_positions(ckpt, N)
    rays_o, rays_d = get_all_rays(data_dir)

    # 归一化方向向量
    rays_d = rays_d / np.linalg.norm(rays_d, axis=1, keepdims=True)

    # 可选：下采样以避免图太密
    step = 200
    rays_o = rays_o[::step]
    rays_d = rays_d[::step]
    rays_end = rays_o + rays_d

    # === 可视化 ===
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 蓝色：primitive 起点
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               s=5, c='blue', label='Primitives')

    # 红色：射线箭头
    ax.quiver(rays_o[:, 0], rays_o[:, 1], rays_o[:, 2],
              rays_d[:, 0], rays_d[:, 1], rays_d[:, 2],
              length=1.0, normalize=False, color='red', linewidth=0.5, label='Rays (length=1)')

    # 绿色：射线终点
    ax.scatter(rays_end[:, 0], rays_end[:, 1], rays_end[:, 2],
               s=5, c='green', label='Ray Endpoints')

    ax.set_title("Primitive Centers and Rays from First Image")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.show()
"""

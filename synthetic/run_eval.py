from visualize import evaluate_and_visualize
import torch



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
    # Path to dataset (change to your own scene path)
    #data_dir = "/home/yuzhou/Desktop/datasets/360_v2/bicycle"
    data_dir = "/home/yuzhou/Desktop/datasets/360_v2/room"
    # Path to trained model checkpoint
    #model_ckpt_path = "/home/yuzhou/Desktop/output/mip360/20250918_025504_67a196c4/scene_epoch_100000.pt"
    
    model_ckpt_path = "/home/yuzhou/Desktop/output/mip360/20250918_004540_b6d99f62/scene_epoch_100000.pt"
    model_ckpt_path = "/home/yuzhou/Desktop/output/mip360/20250926_203723_6024d5dd//scene_epoch_500000.pt"
    #model_ckpt_path = "/home/yuzhou/Desktop/output/mip360/20250922_233544_71123b7b/scene_epoch_500000.pt"   #this is bicycle
    model_ckpt_path = "/home/yuzhou/Desktop/output/mip360/20250928_162800_331d34d5/scene_epoch_500000.pt"   #this is room
    
    # Load model checkpoint (CPU map by default)
    model_ckpt = torch.load(model_ckpt_path, map_location='cpu') if model_ckpt_path else None

    # ✅ Clean up checkpoint keys (remove possible "module." prefix from DDP)
    if model_ckpt is not None:
        model_ckpt = strip_orig_mod_prefix(model_ckpt)

    # Select subset of MLP primitives to render
    # Can set to list(range(N)) to render all MLPs
    #mlp_idx = list(range(312, 325))  
    mlp_idx = None  # Set to None to use all primitives

    """
    # Debug: inspect checkpoint contents
    print(model_ckpt.keys())  

    # Check shapes of some parameters
    print("freq_bands shape:", model_ckpt["mlps.0.encoder.freq_bands"].shape)
    print("net.0.weight shape:", model_ckpt["mlps.0.net.0.weight"].shape)
    print("net.0.bias shape:", model_ckpt["mlps.0.net.0.bias"].shape)
    """

    # Number of primitives must match training configuration
    evaluate_and_visualize(
        data_dir, 
        model_ckpt=model_ckpt, 
        block=True, 
        mlp_idx=mlp_idx, 
        N=9407
    )



 
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


"""
# ---------------- Visualization script: check primitive positions relative to camera frustum ----------------

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

from model import NeuralPrimitiveScene
from llff_loader import LLFFDataset
from torch.utils.data import DataLoader


def load_positions(ckpt_path, num_primitives):
    #Load primitive positions from checkpoint.

    scene = NeuralPrimitiveScene(num_primitives=num_primitives).cpu()
    scene.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    return scene.positions.detach().cpu().numpy()


def get_all_rays(data_dir):
    #Extract rays (origins and directions) from the first image of the dataset.

    dataset = LLFFDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    rays_o_list, rays_d_list = [], []

    for rays_o, rays_d, _ in dataloader:
        rays_o_list.append(rays_o[0].numpy())
        rays_d_list.append(rays_d[0].numpy())
        break  # only use the first image

    rays_o = np.concatenate(rays_o_list, axis=0)
    rays_d = np.concatenate(rays_d_list, axis=0)
    return rays_o, rays_d


if __name__ == "__main__":
    # Path to dataset and checkpoint
    data_dir = "/home/yuzhou/Desktop/datasets/nerf_llff_data/fern"
    ckpt = "/home/yuzhou/Desktop/mlp-splatting/scene_ckpt_tmp.pt"  # saved model path
    # ckpt = "/home/yuzhou/Desktop/output/scene_epoch_0.pt"

    N = 1024  # number of primitives (must match training)

    positions = load_positions(ckpt, N)
    rays_o, rays_d = get_all_rays(data_dir)

    # Normalize ray directions
    rays_d = rays_d / np.linalg.norm(rays_d, axis=1, keepdims=True)

    # Optional: subsample rays for clearer visualization
    step = 200
    rays_o = rays_o[::step]
    rays_d = rays_d[::step]
    rays_end = rays_o + rays_d

    # === Visualization ===
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Blue: primitive centers
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               s=5, c='blue', label='Primitives')

    # Red: ray directions (arrows)
    ax.quiver(rays_o[:, 0], rays_o[:, 1], rays_o[:, 2],
              rays_d[:, 0], rays_d[:, 1], rays_d[:, 2],
              length=1.0, normalize=False, color='red', linewidth=0.5, label='Rays (length=1)')

    # Green: ray endpoints
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

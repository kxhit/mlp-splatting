from train import train_scene_llff
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed(2025)
    #torch.set_float32_matmul_precision('high')
    #data_dir = "/home/xin/data/nerf/nerf_example_data/nerf_llff_data/fern"
    data_dir = "/home/yuzhou/Desktop/datasets/360_v2/bicycle"
    train_scene_llff(data_dir, num_epochs=500000, batch_size=3000, initial_num_primitives=10350)
    


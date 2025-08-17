from train import train_scene_llff
import torch
import numpy as np
import random
import os
import uuid
from datetime import datetime

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多卡
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed(2025)
    #torch.set_float32_matmul_precision('high')
<<<<<<< HEAD
    #data_dir = "/home/xin/data/nerf/nerf_example_data/nerf_llff_data/fern"
    data_dir = "/home/yzcheng/Desktop/datasets/360_v2/bonsai"
    train_scene_llff(data_dir, num_epochs=100000, batch_size=9000, initial_num_primitives=1024)
=======
    # TODO Change to your scene path
    # data_dir = "/home/yuzhou/Desktop/datasets/360_v2/bonsai"
    # output_root = "/home/yuzhou/Desktop/output/mip360"
    data_dir = "/home/xin/data/mipnerf360/bonsai"
    output_root = "./output/mip360"
    # Set saving directory
    random_tag = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    output_dir = os.path.join(output_root, random_tag)
    os.makedirs(output_dir, exist_ok=True)
    train_scene_llff(data_dir, output_dir, num_epochs=100000, batch_size=4096*3, initial_num_primitives=1024)
>>>>>>> 8cc65eef06208a75988afbfce503d9d1b23a9363

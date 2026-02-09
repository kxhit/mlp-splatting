import argparse
import random

import numpy as np
import torch

from train import train_scene_llff

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/media/sjkim/HDD_3TB/dataset/360_v2/bonsai")
    parser.add_argument("--num_epochs", type=int, default=800000)
    parser.add_argument("--batch_size", type=int, default=12000, help="Used by vmodel engine (random rays).")
    parser.add_argument("--initial_num_primitives", type=int, default=2183)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--engine",
        type=str,
        default="vmodel",
        choices=["vmodel", "tile_cuda"],
        help="Training backend. 'tile_cuda' requires full-frame training and a CUDA backward implementation.",
    )
    parser.add_argument(
        "--tile_order",
        type=str,
        default="per_pixel",
        choices=["per_pixel", "tilecenter", "globaldepth"],
        help="Only used when --engine tile_cuda; maps to MLPSP_TILE_ORDER.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    train_scene_llff(
        args.data_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        initial_num_primitives=args.initial_num_primitives,
        engine=args.engine,
        tile_order=args.tile_order,
    )
    


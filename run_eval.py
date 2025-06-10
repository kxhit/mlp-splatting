from visualize import evaluate_and_visualize
import torch

if __name__ == "__main__":
    data_dir = "/home/xin/data/nerf/nerf_example_data/nerf_llff_data/fern" # change to your scene path
    model_ckpt_path = "scene_ckpt_tmp.pt"           # saved model path
    model_ckpt = torch.load(model_ckpt_path, map_location='cpu') if model_ckpt_path else None
    mlp_idx = list(range(0, 128))
    evaluate_and_visualize(data_dir, model_ckpt=model_ckpt, block=True, mlp_idx=mlp_idx)
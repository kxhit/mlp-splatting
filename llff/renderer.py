
import torch
import numpy as np
import dearpygui.dearpygui as dpg
import matplotlib.pyplot as plt
from model import NeuralPrimitiveScene
from llff_loader import LLFFDataset

device = 'cuda'
scene = None
dataset = None
current_pose_idx = 0

@torch.no_grad()
def render_scene():
    global scene, dataset

    rays_o, rays_d, rgbs = dataset[current_pose_idx]
    rays_o = rays_o.to(device)
    rays_d = rays_d.to(device)
    gt_rgb = rgbs.to(device)

    batch_size = 4096
    pred_rgb = []
    for i in range(0, rays_o.shape[0], batch_size):
        pred_rgb.append(scene(rays_o[i:i+batch_size], rays_d[i:i+batch_size]))
    pred_rgb = torch.cat(pred_rgb, dim=0)

    pred_img = pred_rgb.view(1, 378, 504, 3).clamp(0, 1).cpu()
    gt_img = gt_rgb.view(1, 378, 504, 3).cpu()
    psnr = -10 * torch.log10(torch.mean((pred_img - gt_img) ** 2)).item()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(gt_img[0].numpy())
    axes[0].set_title("Ground Truth")
    axes[1].imshow(pred_img[0].numpy())
    axes[1].set_title(f"Prediction (PSNR: {psnr:.2f} dB)")
    plt.tight_layout()
    plt.show()

def next_pose_callback():
    global current_pose_idx
    current_pose_idx = (current_pose_idx + 1) % len(dataset)
    render_scene()

def prev_pose_callback():
    global current_pose_idx
    current_pose_idx = (current_pose_idx - 1) % len(dataset)
    render_scene()

def setup(data_dir, model_ckpt=None, mlp_idx=None):
    global scene, dataset
    dataset = LLFFDataset(data_dir)
    scene = NeuralPrimitiveScene().to(device)
    if model_ckpt:
        scene.load_state_dict(model_ckpt)
    if mlp_idx is not None:
        scene.num_primitives = len(mlp_idx)
        scene.mlps = torch.nn.ModuleList([scene.mlps[i] for i in mlp_idx])
        scene.positions = torch.nn.Parameter(scene.positions[mlp_idx])

def run_gui(data_dir, model_ckpt=None, mlp_idx=None):
    setup(data_dir, model_ckpt, mlp_idx)

    dpg.create_context()
    dpg.create_viewport(title='MLP Splat Controller', width=400, height=200)

    with dpg.window(label="Controller", width=400, height=200):
        dpg.add_button(label="Prev Pose", callback=prev_pose_callback)
        dpg.add_button(label="Next Pose", callback=next_pose_callback)
        dpg.add_button(label="Render", callback=render_scene)

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


# Example usage
if __name__ == "__main__":
    data_dir = "/home/xin/data/nerf/nerf_example_data/nerf_llff_data/fern"
    model_ckpt = torch.load("scene_ckpt_tmp.pt", map_location='cpu')
    run_gui(data_dir, model_ckpt=model_ckpt, mlp_idx=None)

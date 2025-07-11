import torch
import matplotlib.pyplot as plt
from model import NeuralPrimitiveScene
from llff_loader import LLFFDataset

fig = None
ax0_img = None
ax1_img = None

@torch.no_grad()
def evaluate_and_visualize(data_dir, model_ckpt=None, N=128, device='cuda', block=False, mlp_idx=None):
    global fig, ax0_img, ax1_img

    dataset = LLFFDataset(data_dir)
    rays_o, rays_d, rgbs = dataset[0]  # pick first image for evaluation

    rays_o = rays_o.to(device)
    rays_d = rays_d.to(device)
    gt_rgb = rgbs.to(device)

    scene = NeuralPrimitiveScene(N).to(device)
    if model_ckpt:
        scene.load_state_dict(model_ckpt)
    # TODO debug, only viz subset mlps
    if mlp_idx is not None:
        scene.num_primitives = len(mlp_idx)
        scene.mlps = torch.nn.ModuleList([scene.mlps[i] for i in mlp_idx])
        scene.positions = torch.nn.Parameter(scene.positions[mlp_idx])
        scene.log_scales = torch.nn.Parameter(scene.log_scales[mlp_idx])
    # TODO debug, only viz subset mlps
    # import pdb; pdb.set_trace()
    print('scene positions ', scene.positions)
    print('scene scales ', torch.exp(scene.log_scales))
    pred_rgb = []
    # batch_size = 4096
    batch_size = rays_o.shape[0]
    for i in range(0, rays_o.shape[0], batch_size):
        pred_rgb.append(scene(rays_o[i:i+batch_size], rays_d[i:i+batch_size]))
    print('invalid num ', scene.invalid_nums)
    pred_rgb = torch.cat(pred_rgb, dim=0)

    pred_img = pred_rgb.view(1, 378, 504, 3).clamp(0, 1).cpu()
    gt_img = gt_rgb.view(1, 378, 504, 3).cpu()

    psnr = -10 * torch.log10(torch.mean((pred_img - gt_img) ** 2)).item()

    if fig is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax0_img = axes[0].imshow(gt_img[0].numpy())
        axes[0].set_title("Ground Truth")
        ax1_img = axes[1].imshow(pred_img[0].numpy())
        axes[1].set_title(f"Prediction (PSNR: {psnr:.2f} dB)")
        plt.tight_layout()
        plt.ion()
        plt.show()
    else:
        ax0_img.set_data(gt_img[0].numpy())
        ax1_img.set_data(pred_img[0].numpy())
        ax1_img.axes.set_title(f"Prediction (PSNR: {psnr:.2f} dB)")
        fig.canvas.draw()
        fig.canvas.flush_events()

    plt.pause(0.001)
    plt.show(block=block)

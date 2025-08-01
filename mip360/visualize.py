import torch
import matplotlib.pyplot as plt
from model import NeuralPrimitiveScene
from mip360_loader import mip360Dataset

fig = None
ax0_img = None
ax1_img = None

@torch.no_grad()
def evaluate_and_visualize(data_dir, model_ckpt=None, N=1024, device='cuda', block=False, mlp_idx=None):
    global fig, ax0_img, ax1_img
    dataset = mip360Dataset(data_dir, split='train')
    img_idx = 200
    rays_o, rays_d, rgbs = dataset[img_idx]  # pick first image for evaluation

    rays_o = rays_o.to(device).view(-1, 3)  # [H*W, 3]
    rays_d = rays_d.to(device).view(-1, 3)
    gt_rgb = rgbs.to(device).view(-1, 3)

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
    #print('scene positions ', scene.positions)
    print('scene scales ', torch.exp(scene.log_scales))

    scales = torch.exp(scene.log_scales)
    num_small_scales = (scales > 10).sum().item()
    print('Number of scales > 10:', num_small_scales)

    #top10_values, top10_indices = torch.topk(scales.view(-1), k=10, largest=True)
    #print("Top 10 largest scales:")
    #print(top10_values)

    pred_rgb = []
    batch_size = 100000
    #batch_size = rays_o.shape[0]
    for i in range(0, rays_o.shape[0], batch_size):
        pred_rgb.append(scene(rays_o[i:i+batch_size], rays_d[i:i+batch_size]))
    print('invalid num ', scene.invalid_nums)
    pred_rgb = torch.cat(pred_rgb, dim=0)

    pred_img = pred_rgb.view(1, 260, 390, 3).clamp(0, 1).cpu()
    gt_img = gt_rgb.view(1, 260, 390, 3).cpu()

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
    #plt.savefig("img_{img_idx}_mlp_{mlp_idx}.png".format(img_idx=img_idx, mlp_idx=mlp_idx if mlp_idx is not None else "all"))
    plt.show(block=block)

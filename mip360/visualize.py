import torch
import matplotlib.pyplot as plt
from model import NeuralPrimitiveScene
from mip360_loader import mip360Dataset
import os
import imageio
import numpy as np
from tqdm import tqdm


@torch.no_grad()
def evaluate_and_visualize(data_dir, output_dir, model_ckpt=None, N=1024, device='cuda', block=False, mlp_idx=None, loop_all=False):
    dataset = mip360Dataset(data_dir, split='train')
    img_idx = 200
    rays_o, rays_d, rgbs = dataset[img_idx]  # pick first image for evaluation
    height, width = rgbs.shape[:2]
    rays_o = rays_o.to(device).view(-1, 3)  # [H*W, 3]
    rays_d = rays_d.to(device).view(-1, 3)
    gt_rgb = rgbs.to(device).view(-1, 3)

    scene = NeuralPrimitiveScene(N).to(device)
    if model_ckpt:
        scene.load_state_dict(model_ckpt)
    # # TODO debug, only viz subset mlps
    # if mlp_idx is not None:
    #     scene.num_primitives = len(mlp_idx)
    #     scene.mlps = torch.nn.ModuleList([scene.mlps[i] for i in mlp_idx])
    #     scene.positions = torch.nn.Parameter(scene.positions[mlp_idx])
    #     scene.log_scales = torch.nn.Parameter(scene.log_scales[mlp_idx])
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

    # get the index of scales between 0.1 - 10
    scale_idx = torch.where((scales > 0.1) & (scales < 10))[0]
    mlp_idx_t = torch.as_tensor(mlp_idx, device=scale_idx.device)
    mlp_idx = mlp_idx_t[scale_idx.long()]
    mlp_idx = mlp_idx.tolist()

    pred_rgb = []
    batch_size = 100000
    if loop_all:    # loop over each mlp id individually
        for mlp_id in tqdm(mlp_idx):
            pred_rgb = []
            for i in range(0, rays_o.shape[0], batch_size):
                pred_rgb.append(scene(rays_o[i:i + batch_size], rays_d[i:i + batch_size], [mlp_id]))    # model forward can take chosen mlp_idx
            pred_rgb = torch.cat(pred_rgb, dim=0)
            pred_img = pred_rgb.view(1, height, width, 3).clamp(0, 1).cpu()
            gt_img = gt_rgb.view(1, height, width, 3).cpu()
            viz(pred_img, gt_img, [mlp_id], img_idx, scales[mlp_id], output_dir, block=block)
    else:   # infer the list of mlp_idx together
        for i in range(0, rays_o.shape[0], batch_size):
            pred_rgb.append(scene(rays_o[i:i+batch_size], rays_d[i:i+batch_size], mlp_idx))
        print('invalid num ', scene.invalid_nums)
        pred_rgb = torch.cat(pred_rgb, dim=0)
        pred_img = pred_rgb.view(1, height, width, 3).clamp(0, 1).cpu()
        gt_img = gt_rgb.view(1, height, width, 3).cpu()
        viz(pred_img, gt_img, mlp_idx, img_idx, scales, output_dir, block=block)


def viz(pred_img, gt_img, mlp_idx, img_idx, scale, output_dir, block=False):
    psnr = -10 * torch.log10(torch.mean((pred_img - gt_img) ** 2)).item()

    # show gt and pred side by side
    viz_img = torch.cat([gt_img[0], pred_img[0]], dim=1)
    viz_img = viz_img.cpu().numpy()
    viz_img = (viz_img * 255).astype('uint8')  # convert to uint8 for visualization
    if block:
        # show img with psnr in title
        plt.imshow(viz_img)
        plt.title(f"PSNR: {psnr:.2f} dB, MLPs: {mlp_idx[0]}-{mlp_idx[-1]}")
        plt.axis('off')
        plt.show()
    imageio.imwrite(os.path.join(output_dir, f"img_{img_idx}_mlp_{mlp_idx[0]}_{mlp_idx[-1]}_scale_{scale[0].item():.3f}.png"), viz_img)
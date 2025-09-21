import torch
import torch.nn.functional as F


@torch.inference_mode()
def render_scene(scene, rays_o: torch.Tensor, rays_d: torch.Tensor, H: int, W: int, batch_size: int = 50000) -> torch.Tensor:
    device = next(scene.parameters()).device
    rays_o = rays_o.to(device, non_blocking=True)
    rays_d = rays_d.to(device, non_blocking=True)

    preds = []
    for i in range(0, rays_o.shape[0], batch_size):
        preds.append(scene(rays_o[i:i+batch_size], rays_d[i:i+batch_size]))
    rgb = torch.cat(preds, dim=0)
    rgb = rgb.clamp(0, 1).reshape(H, W, 3).contiguous()
    return rgb



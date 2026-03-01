import torch
import torch.nn.functional as F
import numpy as np

def qvec2rotmat(q):
    # q: (...,4), (qx,qy,qz,qw)
    q = F.normalize(q, dim=-1)
    qw, qx, qy, qz = q[..., 3], q[..., 0], q[..., 1], q[..., 2]

    R = torch.stack([
        1 - 2*qy**2 - 2*qz**2,   2*qx*qy - 2*qz*qw,       2*qx*qz + 2*qy*qw,
        2*qx*qy + 2*qz*qw,       1 - 2*qx**2 - 2*qz**2,   2*qy*qz - 2*qx*qw,
        2*qx*qz - 2*qy*qw,       2*qy*qz + 2*qx*qw,       1 - 2*qx**2 - 2*qy**2
    ], dim=-1).view(*q.shape[:-1], 3, 3)
    return R

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
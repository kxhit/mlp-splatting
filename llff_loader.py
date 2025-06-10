import numpy as np
import imageio
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# class LLFFDataset(Dataset):
#     def __init__(self, basedir, factor=8):
#         self.basedir = basedir
#         self.transform = transforms.ToTensor()
#         self._load_meta(factor)
#
#     def _load_meta(self, factor):
#         poses_bounds = np.load(os.path.join(self.basedir, "poses_bounds.npy"))
#         poses = poses_bounds[:, :-2].reshape([-1, 3, 5])
#         self.poses = poses[:, :, :4]
#         self.poses = torch.from_numpy(self.poses).float()
#         import pdb; pdb.set_trace()
#
#         image_dir = os.path.join(self.basedir, f"images_{factor}")
#         self.image_paths = sorted([
#             os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('JPG', 'jpg', 'png'))
#         ])
#         self.images = [self.transform(imageio.imread(p)) for p in self.image_paths]
#         self.images = torch.stack(self.images)
#
#         self.poses = self.poses[:1]
#         self.images = self.images[:1]
#         print(f"Loaded {len(self.images)} images from {image_dir}")
#
#     def __len__(self):
#         return len(self.images)
#
#     def __getitem__(self, idx):
#         img = self.images[idx]
#         pose = self.poses[idx]
#         rays_o, rays_d = self._get_rays(img.shape[1:], pose)
#         return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3), img.permute(1, 2, 0).reshape(-1, 3)
#
#     def _get_rays(self, img_hw, c2w):
#         H, W = img_hw
#         i, j = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
#         dirs = torch.stack([(i - W * 0.5) / W, -(j - H * 0.5) / H, -torch.ones_like(i)], -1)
#         rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
#         rays_o = c2w[:3, 3].expand(rays_d.shape)
#         return rays_o, rays_d


class LLFFDataset(Dataset):
    def __init__(self, basedir, factor=8):
        self.basedir = basedir
        self.transform = transforms.ToTensor()
        self._load_meta(factor)

    def _load_meta(self, factor):
        poses_bounds = np.load(os.path.join(self.basedir, "poses_bounds.npy"))
        poses = poses_bounds[:, :-2].reshape([-1, 3, 5])
        self.poses = poses[:, :, :4]

        # 添加位姿归一化
        positions = self.poses[:, :, 3]
        avg_position = np.mean(positions, axis=0)
        dists = np.linalg.norm(positions - avg_position, axis=1)
        scale = 1.0 / (np.max(dists) * 0.75)  # 0.75是典型缩放因子

        # 应用归一化
        self.poses[:, :, 3] -= avg_position
        self.poses[:, :, 3] *= scale

        self.poses = torch.from_numpy(self.poses).float()

        image_dir = os.path.join(self.basedir, f"images_{factor}")
        self.image_paths = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('JPG', 'jpg', 'png'))
        ])
        self.images = [self.transform(imageio.imread(p)) for p in self.image_paths]
        self.images = torch.stack(self.images)

        # 保存归一化参数用于后续处理
        self.scale = scale
        self.avg_position = avg_position

        # 计算焦距 (从poses_bounds获取)
        hwf = poses[0, :, 4]
        self.focal = hwf[2] / factor  # 根据下采样因子调整焦距

        # self.poses = self.poses[:1]
        # self.images = self.images[:1]
        print(f"Loaded {len(self.images)} images from {image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        pose = self.poses[idx]
        rays_o, rays_d = self._get_rays(img.shape[1:], pose)
        return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3), img.permute(1, 2, 0).reshape(-1, 3)

    def _get_rays(self, img_hw, c2w):
        H, W = img_hw
        i, j = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")

        # 使用存储的焦距而不是硬编码值
        dirs = torch.stack([
            (i - W * 0.5) / self.focal,  # 使用实际焦距
            -(j - H * 0.5) / self.focal,  # 使用实际焦距
            -torch.ones_like(i)
        ], -1)

        rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
        rays_o = c2w[:3, 3].expand(rays_d.shape)
        return rays_o, rays_d
import numpy as np
import imageio
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils.read_write_model import *


class mip360Dataset(Dataset):
    def __init__(self, basedir, factor=4, llffhold=8, split='train'):
        self.basedir = basedir
        self.transform = transforms.ToTensor()
        self.llffhold = llffhold
        self.split = split
        self._load_meta(factor)

    def _load_meta(self, factor):
        # === Load poses from COLMAP ===
        colmap_dir = os.path.join(self.basedir, "colmap_results/triangulated")
        images = read_images_binary(os.path.join(colmap_dir, "images.bin"))
        cameras = read_cameras_binary(os.path.join(colmap_dir, "cameras.bin"))

        # --- Load image paths first (defines the canonical ordering) ---
        if self.split == "train":
            image_dir = os.path.join(self.basedir, "train")

            def img_num_from_path(p: str) -> int:
                # expects r_123.png / r_0001.png
                base = os.path.basename(p)
                return int(base.split("_")[1].split(".")[0])

            self.image_paths = sorted(
                [
                    os.path.join(image_dir, f)
                    for f in os.listdir(image_dir)
                    if f.endswith(("JPG", "jpg", "png"))
                ],
                key=img_num_from_path,
            )
        elif self.split == "test":
            return
        else:
            raise ValueError(f"Unknown split type: {self.split}")

        # --- Build name -> (pose, cam_id) mapping from COLMAP ---
        name2pose_cam = {}
        for _, im in images.items():
            R = qvec2rotmat(im.qvec)
            t = im.tvec.reshape(3,)

            c2w = np.eye(4)
            c2w[:3, :3] = R.T
            c2w[:3, 3] = -R.T @ t

            key = os.path.basename(im.name)  # robust to COLMAP storing subdirs
            name2pose_cam[key] = (c2w[:3, :4], im.camera_id)

        # --- Reorder poses to match self.image_paths exactly ---
        poses = []
        hwf_list = []
        for p in self.image_paths:
            key = os.path.basename(p)
            if key not in name2pose_cam:
                raise KeyError(
                    f"Image {key} not found in COLMAP images.bin. "
                    f"Check that COLMAP was run on the same image set/paths as {image_dir}."
                )

            pose, cam_id = name2pose_cam[key]
            cam = cameras[cam_id]

            poses.append(pose)
            hwf_list.append([cam.height, cam.width, cam.params[0]])  # assume fx=fy

        poses = np.stack(poses, axis=0)  # (N,3,4)
        hwf = np.array(hwf_list)         # (N,3)

        # === Normalize poses (keep your original logic) ===
        positions = poses[:, :, 3]  # (N,3)
        avg_position = np.mean(positions, axis=0)
        dists = np.linalg.norm(positions - avg_position, axis=1)
        scale = 1.0 / (np.max(dists) * 0.75)  # 0.75 is a typical scaling factor

        poses[:, :, 3] -= avg_position
        poses[:, :, 3] *= scale

        self.poses = torch.from_numpy(poses).float()

        # === Load and normalize COLMAP point cloud ===
        points3D = read_points3D_binary(os.path.join(colmap_dir, "points3D.bin"))
        all_xyz = []
        for pid in sorted(points3D.keys()):
            all_xyz.append(points3D[pid].xyz)
        all_xyz = np.stack(all_xyz, axis=0)  # (M,3)

        # Apply same normalization to points
        all_xyz -= avg_position
        all_xyz *= scale
        self.positions = torch.from_numpy(all_xyz).float()  # (M,3)

        # === Load RGBA images once; split into RGB and opacity ===
        imgs = [self.transform(imageio.imread(p)) for p in self.image_paths]  # each: [4,H,W]
        self.rgb = torch.stack([im[:3] for im in imgs])      # (N,3,H,W)
        self.opacity = torch.stack([im[3:4] for im in imgs]) # (N,1,H,W)

        # Save normalization parameters for later use
        self.scale = scale
        self.avg_position = avg_position

        # Focal length from COLMAP intrinsics (per-image, but usually constant)
        self.focal = float(hwf[0, 2])

        # Keep COLMAP dicts if you need them later (avoid name collision with self.rgb)
        self.colmap_images = images
        self.colmap_cameras = cameras

        print(f"Loaded {len(self.image_paths)} images from {image_dir}")


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        rgb = self.rgb[idx]
        opacity = self.opacity[idx]
        pose = self.poses[idx]
        rays_o, rays_d = self._get_rays(rgb.shape[1:], pose)
        #return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3), img.permute(1, 2, 0).reshape(-1, 3)
        return rays_o, rays_d, rgb.permute(1, 2, 0), pose, opacity

    def _get_rays(self, img_hw, c2w):
        H, W = img_hw
        i, j = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")

        dirs = torch.stack([
            (i - W * 0.5) / self.focal,
            (j - H * 0.5) / self.focal,
            torch.ones_like(i)  # z-axis is forward
        ], -1)  # shape: (H, W, 3)

        # Rotate ray directions from camera coordinates to world coordinates
        rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
        # Ray origins (camera center), broadcast to match ray directions
        rays_o = c2w[:3, 3].expand(rays_d.shape)

        return rays_o, rays_d

    
    def get_all_rays(self, device=None, return_img_ids=False):
        """
        Stays on CPU unless an explicit device (e.g., 'cuda') is given
        """
        rays_o_list, rays_d_list, rgbs_list, opacity_list = [], [], [], []
        img_id_list = [] if return_img_ids else None

        for i in range(len(self.image_paths)):
            rgb  = self.rgb[i]                 # (3, H, W)
            opacity = self.opacity[i]
            pose = self.poses[i]                  # (3, 4)
            H, W = rgb.shape[1:]

            rays_o, rays_d = self._get_rays((H, W), pose)  # (H, W, 3)
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
            rgbs   = rgb.permute(1, 2, 0).reshape(-1, 3)
            opacities = opacity.permute(1,2,0).reshape(-1, 1)

            rays_o_list.append(rays_o)
            rays_d_list.append(rays_d)
            rgbs_list.append(rgbs)
            opacity_list.append(opacities)

            if return_img_ids:
                img_id_list.append(torch.full((H * W,), i, dtype=torch.long))

        rays_o = torch.cat(rays_o_list,     dim=0)
        rays_d = torch.cat(rays_d_list,     dim=0)
        rgbs   = torch.cat(rgbs_list,       dim=0)
        opacities = torch.cat(opacity_list, dim=0)

        if return_img_ids:
            img_ids = torch.cat(img_id_list, dim=0)

        # Pass 'device' only if moving tensors to GPU
        if device is not None:
            rays_o = rays_o.to(device, non_blocking=True)
            rays_d = rays_d.to(device, non_blocking=True)
            rgbs   = rgbs.to(device, non_blocking=True)
            opacities = opacities.to(device, non_blocking=True)
            if return_img_ids:
                img_ids = img_ids.to(device, non_blocking=True)

        if return_img_ids:
            return rays_o, rays_d, rgbs, opacities, img_ids
        return rays_o, rays_d, rgbs, opacities



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


import numpy as np
import imageio
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils.read_write_model import read_images_binary, read_cameras_binary, read_points3D_binary, qvec2rotmat


class mip360Dataset(Dataset):
    def __init__(self, basedir, factor=8, llffhold=8, split='train'):
        self.basedir = basedir
        self.transform = transforms.ToTensor()
        self.llffhold = llffhold
        self.split = split
        self._load_meta(factor)

    def _load_meta(self, factor):
        # === Modified: load poses from COLMAP ===
        colmap_dir = os.path.join(self.basedir, "sparse/0")
        images = read_images_binary(os.path.join(colmap_dir, "images.bin"))
        cameras = read_cameras_binary(os.path.join(colmap_dir, "cameras.bin"))

        image_ids_sorted = sorted(images.keys())

        poses = []
        hwf_list = []
        for image_id in image_ids_sorted:
            image = images[image_id]
            cam = cameras[image.camera_id]

            # pose: w2c → c2w
            R = qvec2rotmat(image.qvec)
            t = image.tvec.reshape(3,)
            c2w = np.eye(4)
            c2w[:3, :3] = R.T
            c2w[:3, 3] = -R.T @ t
            poses.append(c2w[:3, :4])

            # intrinsic: assume fx = fy
            H, W = cam.height, cam.width
            fx = cam.params[0]
            hwf_list.append([H, W, fx])

        poses = np.stack(poses, axis=0)         # (N, 3, 4)
        hwf = np.array(hwf_list)                # (N, 3)

        # === Normalize poses (keep original logic) ===
        positions = poses[:, :, 3]
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
            xyz = points3D[pid].xyz
            all_xyz.append(xyz)

        all_xyz = np.stack(all_xyz, axis=0)  # shape: (M, 3)

        # === Apply same normalization: translation + scaling ===
        all_xyz -= avg_position
        all_xyz *= scale


        self.positions = torch.from_numpy(all_xyz).float()  # shape: (M, 3)


        # === Keep the remaining part unchanged ===
        # Load images
        if factor == 1:
            image_dir = os.path.join(self.basedir, f"images")
        else:
            image_dir = os.path.join(self.basedir, f"images_{factor}")
        self.image_paths = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('JPG', 'jpg', 'png'))
        ])
        self.images = [self.transform(imageio.imread(p)) for p in self.image_paths]
        self.images = torch.stack(self.images)

        # Save normalization parameters for later use
        self.scale = scale
        self.avg_position = avg_position

        # Compute focal length (derived from hwf)
        self.focal = hwf[0, 2] / factor  # Adjusted by downsampling factor


        # === Train/test split following llffhold convention ===
        num_images = self.images.shape[0]
        i_test = np.arange(num_images)[::self.llffhold]
        i_train = np.array([i for i in range(num_images) if i not in i_test])


        if self.split == 'train':
            select_ids = i_train
        elif self.split == 'test':
            select_ids = i_test
        else:
            raise ValueError(f"Unknown split type: {self.split}")

        # Apply split
        self.images = self.images[select_ids]
        self.poses = self.poses[select_ids]

        # Expose selected indices and their corresponding image paths for UI/inspection
        try:
            self.select_ids = select_ids.tolist() if hasattr(select_ids, "tolist") else list(select_ids)
        except Exception:
            # fallback best-effort
            self.select_ids = [int(i) for i in select_ids]
        try:
            self.image_paths_selected = [self.image_paths[i] for i in self.select_ids]
        except Exception:
            # keep parity with dataset length if something goes wrong
            self.image_paths_selected = []

        print(f"Loaded {len(self.images)} images from {image_dir}")


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        pose = self.poses[idx]
        rays_o, rays_d = self._get_rays(img.shape[1:], pose)
        #return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3), img.permute(1, 2, 0).reshape(-1, 3)
        return rays_o, rays_d, img.permute(1, 2, 0)

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
        rays_o_list, rays_d_list, rgbs_list = [], [], []
        img_id_list = [] if return_img_ids else None

        for i in range(len(self.images)):
            img  = self.images[i]                 # (3, H, W)
            pose = self.poses[i]                  # (3, 4)
            H, W = img.shape[1:]

            rays_o, rays_d = self._get_rays((H, W), pose)  # (H, W, 3)
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
            rgbs   = img.permute(1, 2, 0).reshape(-1, 3)

            rays_o_list.append(rays_o)
            rays_d_list.append(rays_d)
            rgbs_list.append(rgbs)

            if return_img_ids:
                img_id_list.append(torch.full((H * W,), i, dtype=torch.long))

        rays_o = torch.cat(rays_o_list, dim=0)
        rays_d = torch.cat(rays_d_list, dim=0)
        rgbs   = torch.cat(rgbs_list,   dim=0)

        if return_img_ids:
            img_ids = torch.cat(img_id_list, dim=0)

        # Pass 'device' only if moving tensors to GPU
        if device is not None:
            rays_o = rays_o.to(device, non_blocking=True)
            rays_d = rays_d.to(device, non_blocking=True)
            rgbs   = rgbs.to(device, non_blocking=True)
            if return_img_ids:
                img_ids = img_ids.to(device, non_blocking=True)

        if return_img_ids:
            return rays_o, rays_d, rgbs, img_ids
        return rays_o, rays_d, rgbs



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


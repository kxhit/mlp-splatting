from __future__ import annotations

import os
import re
from collections import OrderedDict

import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

try:
    from .utils.read_write_model import (
        qvec2rotmat,
        read_cameras_binary,
        read_images_binary,
        read_points3D_binary,
    )
except ImportError:  # Support direct script execution.
    from utils.read_write_model import (
        qvec2rotmat,
        read_cameras_binary,
        read_images_binary,
        read_points3D_binary,
    )

try:
    from .data_validation import (
        infer_feature_layout,
        normalize_feature_stem,
        select_feature_payload,
        validated_camera_intrinsics_by_image,
    )
except ImportError:  # Support direct script execution.
    from data_validation import (
        infer_feature_layout,
        normalize_feature_stem,
        select_feature_payload,
        validated_camera_intrinsics_by_image,
    )


def _basename_no_ext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _last_int_suffix(name_no_ext: str) -> tuple[int, int]:
    match = re.search(r"(\d+)$", name_no_ext)
    if match:
        return (1, int(match.group(1)))
    return (0, -1)


def _smart_image_sort_key(path: str):
    base = _basename_no_ext(path)
    has_num, num = _last_int_suffix(base)
    return (-has_num, num, base)


class mip360Dataset(Dataset):
    def __init__(
        self,
        basedir,
        factor=4,
        llffhold=8,
        split='train',
        feature_type: str = "none",
        feature_dir: str | None = None,
        preload_features: bool = False,
        feature_cache_size: int = 16,
        feature_resize_to_image: bool = False,
        ray_cache_size: int = 8,
    ):
        self.basedir = basedir
        self.transform = transforms.ToTensor()
        self.llffhold = llffhold
        self.split = split
        self.feature_type = str(feature_type).lower()
        self.feature_dir = feature_dir
        self.preload_features = bool(preload_features)
        self.feature_cache_size = int(max(0, feature_cache_size))
        self.feature_resize_to_image = bool(feature_resize_to_image)
        self._feature_cache = OrderedDict()
        self._feature_paths = None
        self.ray_cache_size = int(max(0, ray_cache_size))
        self._ray_cache = OrderedDict()
        self.features = None
        self._load_meta(factor)

    def _load_meta(self, factor):
        # === Modified: load poses from COLMAP ===
        colmap_dir = os.path.join(self.basedir, "sparse/0")
        images = read_images_binary(os.path.join(colmap_dir, "images.bin"))
        cameras = read_cameras_binary(os.path.join(colmap_dir, "cameras.bin"))
        intrinsics_by_base = validated_camera_intrinsics_by_image(images, cameras)

        # Load images from disk with a numeric-suffix-aware order (e.g., image_2 before image_10).
        if factor == 1:
            image_dir = os.path.join(self.basedir, "images")
        else:
            image_dir = os.path.join(self.basedir, f"images_{factor}")
        exts = (".JPG", ".jpg", ".png", ".PNG", ".jpeg", ".JPEG")
        image_paths_all = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.endswith(exts)
        ]
        image_paths_all.sort(key=_smart_image_sort_key)

        # Build COLMAP pose/intrinsics lookup by basename (with extension).
        pose_by_base = {}
        hwf_by_base = {}
        for image_id in sorted(images.keys()):
            image = images[image_id]
            base = os.path.basename(image.name)

            # pose: w2c → c2w
            R = qvec2rotmat(image.qvec)
            t = image.tvec.reshape(3,)
            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, :3] = R.T
            c2w[:3, 3] = -R.T @ t
            pose_by_base[base] = c2w[:3, :4]

            # Validation above guarantees one centered, undistorted calibration.
            intrinsics = intrinsics_by_base[base]
            H = int(intrinsics["height"])
            W = int(intrinsics["width"])
            fx = float(intrinsics["fx"])
            hwf_by_base[base] = [H, W, fx]

        # Keep only images that are present in both disk directory and COLMAP.
        matched_paths = []
        matched_poses = []
        matched_hwf = []
        missing_in_colmap = []
        for img_path in image_paths_all:
            base = os.path.basename(img_path)
            if base not in pose_by_base:
                missing_in_colmap.append(base)
                continue
            matched_paths.append(img_path)
            matched_poses.append(pose_by_base[base])
            matched_hwf.append(hwf_by_base[base])

        if len(matched_paths) == 0:
            raise RuntimeError(
                f"No images matched between {image_dir} and COLMAP images.bin. "
                f"Example disk file: {os.path.basename(image_paths_all[0]) if image_paths_all else 'N/A'}; "
                f"Example colmap name: {next(iter(pose_by_base.keys())) if pose_by_base else 'N/A'}"
            )
        if len(missing_in_colmap) > 0:
            print(
                f"[Warn] {len(missing_in_colmap)} images exist on disk but not in COLMAP. "
                f"First few: {missing_in_colmap[:10]}"
            )

        self.image_paths = matched_paths

        poses = np.stack(matched_poses, axis=0)  # (N, 3, 4), aligned to self.image_paths
        hwf = np.array(matched_hwf, dtype=np.float32)  # (N, 3), aligned to self.image_paths

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
        self.images = [self.transform(imageio.imread(p)) for p in self.image_paths]
        self.images = torch.stack(self.images)

        # Optional: load pre-extracted foundation features (Feature-3DGS-compatible assumption).
        if self.feature_type != "none":
            if self.feature_dir is None:
                if self.feature_type == "lseg":
                    feature_dir = os.path.join(self.basedir, "rgb_feature_langseg")
                elif self.feature_type == "sam":
                    feature_dir = os.path.join(self.basedir, "sam_embeddings")
                else:
                    raise ValueError(f"Unknown feature_type: {self.feature_type}")
            else:
                feature_dir = self.feature_dir
            self._feature_paths = self._build_feature_path_list(feature_dir)

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
        if self._feature_paths is not None:
            self._feature_paths = [self._feature_paths[int(i)] for i in select_ids]
            if self.preload_features:
                self.features = self._preload_features()

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

        print(f"Loaded {len(self.images)} images from {image_dir} (matched from {len(self.image_paths)} total).")


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        pose = self.poses[idx]
        rays_o, rays_d = self._get_rays_cached(idx, img.shape[1:], pose)
        if self.features is None:
            if self._feature_paths is None:
                return rays_o, rays_d, img.permute(1, 2, 0), pose
            feat = self._get_feature_lazy(idx)
            return rays_o, rays_d, img.permute(1, 2, 0), feat, pose
        return rays_o, rays_d, img.permute(1, 2, 0), self.features[idx], pose

    def _build_feature_path_list(self, feature_dir: str):
        if not os.path.isdir(feature_dir):
            raise FileNotFoundError(f"Feature directory not found: {feature_dir}")

        feature_paths = sorted(
            (
                os.path.join(feature_dir, f)
                for f in os.listdir(feature_dir)
                if f.lower().endswith((".npy", ".npz", ".pt"))
            ),
            key=lambda path: os.path.normcase(os.path.normpath(path)),
        )
        if len(feature_paths) == 0:
            raise FileNotFoundError(f"No feature files found in {feature_dir}")

        feature_by_stem = {}
        for p in feature_paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            norm_stem = normalize_feature_stem(stem)
            if norm_stem in feature_by_stem:
                raise ValueError(
                    f"Duplicate feature stem '{norm_stem}' after normalization: "
                    f"{feature_by_stem[norm_stem]} and {p}"
                )
            feature_by_stem[norm_stem] = p

        for img_path in self.image_paths:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            if stem not in feature_by_stem:
                raise KeyError(f"Feature missing for image stem '{stem}' in {feature_dir}")
        feature_paths_aligned = [feature_by_stem[os.path.splitext(os.path.basename(p))[0]] for p in self.image_paths]
        print(f"Indexed {len(feature_paths_aligned)} feature files from {feature_dir}")
        return feature_paths_aligned

    def _load_feature_tensor(self, feat_path: str):
        H_img, W_img = int(self.images.shape[2]), int(self.images.shape[3])
        extension = os.path.splitext(feat_path)[1].lower()
        if extension == ".npy":
            payload = np.load(feat_path)
        elif extension == ".npz":
            with np.load(feat_path) as npz:
                payload = {key: npz[key] for key in npz.files}
        else:
            payload = torch.load(feat_path, map_location="cpu")
        selected = select_feature_payload(payload, source=feat_path)
        feat = torch.as_tensor(selected).float()

        layout = infer_feature_layout(
            feat.shape, (H_img, W_img), self.feature_type, feat_path
        )
        if layout == "chw":
            feat = feat.permute(1, 2, 0).contiguous()
        else:
            feat = feat.contiguous()

        if self.feature_resize_to_image and (feat.shape[0] != H_img or feat.shape[1] != W_img):
            feat_chw = feat.permute(2, 0, 1).unsqueeze(0)
            feat_chw = torch.nn.functional.interpolate(
                feat_chw, size=(H_img, W_img), mode="bilinear", align_corners=False
            )
            feat = feat_chw.squeeze(0).permute(1, 2, 0).contiguous()
        return feat

    def _preload_features(self):
        # Keep lazy-loading path unchanged (float32), but store preloaded features as float16
        # to reduce host-memory footprint when --preload_features is enabled.
        feats = [self._load_feature_tensor(p).to(dtype=torch.float16) for p in self._feature_paths]
        feats = torch.stack(feats, dim=0)
        print(f"Preloaded {len(feats)} feature maps into memory (dtype={feats.dtype})")
        return feats

    def _get_feature_lazy(self, idx: int):
        key = int(idx)
        cached = self._feature_cache.get(key)
        if cached is not None:
            self._feature_cache.move_to_end(key)
            return cached
        feat_path = self._feature_paths[key]
        feat = self._load_feature_tensor(feat_path)
        self._feature_cache[key] = feat
        if self.feature_cache_size > 0:
            while len(self._feature_cache) > self.feature_cache_size:
                self._feature_cache.popitem(last=False)
        else:
            self._feature_cache.clear()
        return feat

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

    def _get_rays_cached(self, idx: int, img_hw, c2w):
        key = int(idx)
        cached = self._ray_cache.get(key)
        if cached is not None:
            self._ray_cache.move_to_end(key)
            return cached
        rays = self._get_rays(img_hw, c2w)
        self._ray_cache[key] = rays
        if self.ray_cache_size > 0:
            while len(self._ray_cache) > self.ray_cache_size:
                self._ray_cache.popitem(last=False)
        else:
            self._ray_cache.clear()
        return rays

    
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

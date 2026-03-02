# mip360/utils/sampling.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import torch

__all__ = ["octree_sample", "obb_tree_sample"]


# ---------------------------
# Octree (your existing)
# ---------------------------

class OctreeNode:
    def __init__(self, center, size, points, depth=0):
        self.center = center
        self.size = size
        self.points = points
        self.children = []
        self.depth = depth


def build_octree(points, min_points=200, depth=0, center=None, root_size=None):
    points = np.asarray(points)

    if depth == 0:
        mins, maxs = points.min(axis=0), points.max(axis=0)
        center = (mins + maxs) / 2
        root_size = np.max(maxs - mins)

    size = root_size / (2 ** depth)
    node = OctreeNode(center, size, points, depth)

    if len(points) <= min_points:
        return node

    child_size = size / 2
    quarter = size / 4
    offsets = np.array([[dx, dy, dz] for dx in [-1, 1]
                                     for dy in [-1, 1]
                                     for dz in [-1, 1]]) * quarter
    for off in offsets:
        sub_center = center + off
        mask = np.all((points >= sub_center - child_size/2) & (points < sub_center + child_size/2), axis=1)
        if np.any(mask):
            child = build_octree(points[mask], min_points, depth+1, sub_center, root_size)
            node.children.append(child)

    return node


def collect_leaf_info(node, use_centroid=True):
    if not node.children:
        pos = node.points.mean(axis=0) if use_centroid else node.center
        radius = node.size / 2 * np.sqrt(3)
        return [(pos, radius)]
    results = []
    for c in node.children:
        results.extend(collect_leaf_info(c, use_centroid))
    return results


def octree_sample(points, min_points=10, device="cpu"):
    root = build_octree(points, min_points=min_points)
    leaf_info = collect_leaf_info(root, use_centroid=True)

    positions = np.array([p for p, _ in leaf_info])
    radii = np.array([r for _, r in leaf_info])

    positions = torch.from_numpy(positions).float().to(device)
    radii = torch.from_numpy(radii).float().to(device)
    return positions, radii


"""

# ---------------------------
# OBB Tree
# ---------------------------

@dataclass
class OBBNode:
    c: np.ndarray
    R: np.ndarray
    h: np.ndarray
    idx: np.ndarray
    children: List["OBBNode"] = field(default_factory=list)
    split_dim: Optional[int] = None  # 2 or 3 or None


def _pca_axes(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = points.mean(axis=0)
    X = points - mu
    C = (X.T @ X) / max(len(points), 1)
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    if np.linalg.det(evecs) < 0:
        evecs[:, 2] *= -1.0
    return mu, evals, evecs


def _fit_obb(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu, _, R = _pca_axes(points)
    Y = (points - mu) @ R
    ymin = Y.min(axis=0)
    ymax = Y.max(axis=0)
    h = 0.5 * (ymax - ymin)
    y_center = 0.5 * (ymax + ymin)
    c = mu + R @ y_center
    return c, R, h


def build_anisotropic_obb_tree(points: np.ndarray, min_points_leaf: int, tau_planar: float) -> OBBNode:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N,3).")
    N = points.shape[0]
    if N == 0:
        raise ValueError("points is empty.")
    all_idx = np.arange(N, dtype=np.int32)

    def _recurse_fixed(idx: np.ndarray, c: np.ndarray, R: np.ndarray, h: np.ndarray) -> OBBNode:
        node = OBBNode(c=c, R=R, h=h, idx=idx)
        if idx.size < min_points_leaf:
            return node

        P = points[idx]
        _, evals, _ = _pca_axes(P)
        lam2, lam3 = float(evals[1]), float(evals[2])
        planar_ratio = lam2 / max(lam3, 1e-12)
        is_2d = planar_ratio >= tau_planar
        node.split_dim = 2 if is_2d else 3

        Y = (P - c) @ R

        if is_2d:
            b0 = (Y[:, 0] >= 0.0).astype(np.int32)
            b1 = (Y[:, 1] >= 0.0).astype(np.int32)
            child_id = b0 + 2 * b1
            ch = np.array([h[0] * 0.5, h[1] * 0.5, h[2]], dtype=np.float64)
            for cid in range(4):
                mask = (child_id == cid)
                if not np.any(mask):
                    continue
                sx = -1.0 if (cid & 1) == 0 else +1.0
                sy = -1.0 if (cid & 2) == 0 else +1.0
                cc = c + R @ np.array([sx * ch[0], sy * ch[1], 0.0], dtype=np.float64)
                node.children.append(_recurse_fixed(idx[mask], cc, R, ch))
        else:
            b0 = (Y[:, 0] >= 0.0).astype(np.int32)
            b1 = (Y[:, 1] >= 0.0).astype(np.int32)
            b2 = (Y[:, 2] >= 0.0).astype(np.int32)
            child_id = b0 + 2 * b1 + 4 * b2
            ch = h * 0.5
            for cid in range(8):
                mask = (child_id == cid)
                if not np.any(mask):
                    continue
                sx = -1.0 if (cid & 1) == 0 else +1.0
                sy = -1.0 if (cid & 2) == 0 else +1.0
                sz = -1.0 if (cid & 4) == 0 else +1.0
                cc = c + R @ np.array([sx * ch[0], sy * ch[1], sz * ch[2]], dtype=np.float64)
                node.children.append(_recurse_fixed(idx[mask], cc, R, ch))

        return node

    root_c, root_R, root_h = _fit_obb(points)
    return _recurse_fixed(all_idx, root_c, root_R, root_h)


def attach_points_ref(node: OBBNode, points: np.ndarray):
    node._points_ref = points  # dynamic attribute
    for c in node.children:
        attach_points_ref(c, points)


def collect_leaf_info_obb(node: OBBNode, use_centroid=True):
    if not node.children:
        if use_centroid:
            pos = node._points_ref[node.idx].mean(axis=0)
        else:
            pos = node.c
        radius = float(np.linalg.norm(node.h))  # half diagonal length
        return [(pos, radius)]
    out = []
    for c in node.children:
        out.extend(collect_leaf_info_obb(c, use_centroid))
    return out


def obb_tree_sample(points, min_points=320, tau_planar=2000000000.0, device="cpu", use_centroid=True):
    points = np.asarray(points, dtype=np.float64)
    root = build_anisotropic_obb_tree(points, min_points_leaf=min_points, tau_planar=tau_planar)
    attach_points_ref(root, points)
    leaf_info = collect_leaf_info_obb(root, use_centroid=use_centroid)

    positions = np.array([p for p, _ in leaf_info], dtype=np.float32)
    radii = np.array([r for _, r in leaf_info], dtype=np.float32)

    positions = torch.from_numpy(positions).to(device)
    radii = torch.from_numpy(radii).to(device)
    return positions, radii
"""
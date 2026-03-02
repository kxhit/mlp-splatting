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
        self.center = center   # cube 中心
        self.size = size       # cube 边长 (严格 1/2^depth)
        self.points = points
        self.children = []
        self.depth = depth

def build_octree(points, min_points=200, depth=0, center=None, root_size=None):
    points = np.asarray(points)

    # 初始化 root
    if depth == 0:
        mins, maxs = points.min(axis=0), points.max(axis=0)
        center = (mins + maxs) / 2
        root_size = np.max(maxs - mins)

    size = root_size / (2 ** depth)
    node = OctreeNode(center, size, points, depth)

    if len(points) <= min_points:
        return node

    # 子 cube 划分
    child_size = size / 2
    quarter = size / 4
    offsets = np.array([[dx, dy, dz] for dx in [-1, 1]
                                     for dy in [-1, 1]
                                     for dz in [-1, 1]]) * quarter
    for off in offsets:
        sub_center = center + off
        # 用严格区间避免边界点重复归属
        mask = np.all((points >= sub_center - child_size/2) & (points < sub_center + child_size/2), axis=1)
        if np.any(mask):
            child = build_octree(points[mask], min_points, depth+1, sub_center, root_size)
            node.children.append(child)

    return node



"后处理，让大空间比如说墙体尽可能获得更多的点"

def _max_depth(root):
    md = 0
    stack = [root]
    while stack:
        n = stack.pop()
        md = max(md, n.depth)
        stack.extend(n.children)
    return md

def _collect_leaves(root):
    leaves = []
    stack = [root]
    while stack:
        n = stack.pop()
        if not n.children:
            leaves.append(n)
        else:
            stack.extend(n.children)
    return leaves

def _collect_leaves_by_depth(root):
    d2 = {}
    stack = [root]
    while stack:
        n = stack.pop()
        if not n.children:
            d2.setdefault(n.depth, []).append(n)
        else:
            stack.extend(n.children)
    return d2

def _split_leaf_inplace(node, root_size):
    """
    将一个 leaf node 原地 8 分（只创建有点的 child，空的 child 不创建）。
    返回新增的 leaf 数量（new_leaves - old_leaves），以及新建 child 数。
    """
    assert not node.children, "Only split leaf nodes"
    pts = np.asarray(node.points)
    if pts.shape[0] == 0:
        return 0, 0

    # node.size 在你的实现中应该满足 node.size = root_size / (2**node.depth)
    # 这里用 node.size 直接算子块尺寸，保证一致
    size = float(node.size)
    child_size = size / 2.0
    quarter = size / 4.0

    center = np.asarray(node.center, dtype=np.float64)
    offsets = np.array([[dx, dy, dz] for dx in [-1, 1]
                                 for dy in [-1, 1]
                                 for dz in [-1, 1]], dtype=np.float64) * quarter

    children = []
    for off in offsets:
        sub_center = center + off
        lo = sub_center - child_size / 2.0
        hi = sub_center + child_size / 2.0
        mask = np.all((pts >= lo) & (pts < hi), axis=1)
        if np.any(mask):
            child = OctreeNode(
                center=sub_center,
                size=child_size,              # = root_size / (2**(depth+1))
                points=pts[mask],
                depth=node.depth + 1
            )
            children.append(child)

    if len(children) == 0:
        return 0, 0

    # 原 leaf 1 个 -> 变成 len(children) 个 leaf
    node.children = children
    added = len(children) - 1
    return added, len(children)


def refine_octree_postprocess(root, add_frac=0.10, stop_half_depth=True, verbose=True):
    """
    Postprocessing: 从浅层开始，把“当前深度的 leaf”逐层 8 分，
    直到：
      (1) leaf 总数增加 >= add_frac * 原 leaf 数
      或
      (2) refine 深度达到 half_depth = (max_depth+1)//2 （如果 stop_half_depth=True）

    注意：只 refine “当时还是 leaf 的节点”；一旦被 split，它就不再是 leaf。
    """
    # 估 root_size：由 root.size 与 root.depth 反推（root.depth通常为0）
    # root.size = root_size / (2**root.depth) => root_size = root.size * 2**root.depth
    root_size = float(root.size) * (2 ** int(root.depth))

    orig_leaves = _collect_leaves(root)
    orig_leaf_count = len(orig_leaves)
    target_add = int(np.ceil(orig_leaf_count * add_frac))

    init_max_d = _max_depth(root)
    half_depth = (init_max_d + 1) // 2  # 你说的 (深度+1)/2 的整数版本

    added_total = 0
    refined_nodes = 0

    if verbose:
        print(f"[refine_post] orig_leaf_count={orig_leaf_count}, target_add={target_add}")
        print(f"[refine_post] init_max_depth={init_max_d}, half_depth_stop={half_depth} (enabled={stop_half_depth})")

    # 按 depth 从小到大逐层处理
    d = 0
    while True:
        if added_total >= target_add:
            if verbose:
                print(f"[refine_post] stop: reached target_add ({added_total} >= {target_add})")
            break
        if stop_half_depth and d >= half_depth:
            if verbose:
                print(f"[refine_post] stop: reached half_depth (d={d} >= {half_depth})")
            break

        # 重新收集当前 leaf（因为上一轮 split 会改变树）
        leaf_by_depth = _collect_leaves_by_depth(root)
        leaves_d = leaf_by_depth.get(d, [])
        if verbose:
            print(f"[refine_post] depth={d}: leaf_count={len(leaves_d)}")

        if len(leaves_d) == 0:
            d += 1
            # 若已经超过当前最大 depth，也可以提前停（可选）
            if d > _max_depth(root) + 1:
                break
            continue

        # 逐个 split（按你的策略：这一层所有 leaf 都 split）
        for leaf in leaves_d:
            if added_total >= target_add:
                break
            add, nchild = _split_leaf_inplace(leaf, root_size=root_size)
            if nchild > 0:
                refined_nodes += 1
                added_total += add

        d += 1

    new_leaf_count = len(_collect_leaves(root))
    if verbose:
        print(f"[refine_post] refined_nodes={refined_nodes}, added_leaves={added_total}")
        print(f"[refine_post] new_leaf_count={new_leaf_count} (delta={new_leaf_count - orig_leaf_count})")

    return root


def collect_leaf_info(node, use_centroid=True):
    """收集所有 leaf 节点的 (位置, 半径)"""
    if not node.children:  # leaf
        pos = node.points.mean(axis=0) if use_centroid else node.center
        radius = node.size / 2 * np.sqrt(3)  # 半径为半对角线长度
        return [(pos, radius)]
    results = []
    for c in node.children:
        results.extend(collect_leaf_info(c, use_centroid))
    return results

def octree_sample(points, min_points=35, device="cpu"):
    """从点云中采样，返回 (positions, radii)"""
    root = build_octree(points, min_points=min_points)
    # postprocess：只增加 15% leaf，且最多 refine 到一半深度
    root = refine_octree_postprocess(root, add_frac=0.1, stop_half_depth=True, verbose=True)

    leaf_info = collect_leaf_info(root, use_centroid=True)

    positions = np.array([p for p, r in leaf_info])
    radii = np.array([r for p, r in leaf_info])

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
"""Octree initialization for MLP Splatting primitives."""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["octree_sample"]


class OctreeNode:
    """A populated axis-aligned octree node."""

    def __init__(self, center, size, points, depth=0):
        self.center = center
        self.size = size
        self.points = points
        self.children = []
        self.depth = depth


def build_octree(points, min_points=200, depth=0, center=None, root_size=None):
    """Split until each leaf contains at most ``min_points`` points.

    ``min_points`` is the historical internal name for the paper's octree
    stopping threshold ``m``.  The public CLI calls it max-points-per-leaf.
    """

    points = np.asarray(points)
    if depth == 0:
        if int(min_points) <= 0:
            raise ValueError("min_points must be positive")
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError("points must have shape [N, 3] with N > 0")
        if not np.isfinite(points).all():
            raise ValueError("points must contain only finite values")
        mins, maxs = points.min(axis=0), points.max(axis=0)
        center = (mins + maxs) / 2
        root_size = np.max(maxs - mins)

    size = root_size / (2**depth)
    node = OctreeNode(center, size, points, depth)
    if len(points) <= min_points or size <= np.finfo(np.float64).eps:
        return node

    child_size = size / 2
    quarter = size / 4
    offsets = (
        np.array(
            [
                [dx, dy, dz]
                for dx in (-1, 1)
                for dy in (-1, 1)
                for dz in (-1, 1)
            ]
        )
        * quarter
    )
    positive_side = points >= np.asarray(center)
    octant_ids = (
        positive_side[:, 0].astype(np.uint8) * 4
        + positive_side[:, 1].astype(np.uint8) * 2
        + positive_side[:, 2].astype(np.uint8)
    )
    for octant_id, offset in enumerate(offsets):
        child_center = center + offset
        # Center comparisons assign every point, including the root's maximum
        # faces, to exactly one child.
        mask = octant_ids == octant_id
        if np.any(mask):
            node.children.append(
                build_octree(
                    points[mask],
                    min_points,
                    depth + 1,
                    child_center,
                    root_size,
                )
            )
    return node


def _max_depth(root):
    maximum = 0
    stack = [root]
    while stack:
        node = stack.pop()
        maximum = max(maximum, node.depth)
        stack.extend(node.children)
    return maximum


def _collect_leaves(root):
    leaves = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.children:
            stack.extend(node.children)
        else:
            leaves.append(node)
    return leaves


def _collect_leaves_by_depth(root):
    leaves_by_depth = {}
    stack = [root]
    while stack:
        node = stack.pop()
        if node.children:
            stack.extend(node.children)
        else:
            leaves_by_depth.setdefault(node.depth, []).append(node)
    return leaves_by_depth


def _split_leaf_inplace(node):
    """Split one leaf into its non-empty octants and return leaf growth."""

    if node.children:
        raise ValueError("only leaf nodes can be split")
    points = np.asarray(node.points)
    if points.shape[0] == 0:
        return 0, 0

    child_size = float(node.size) / 2
    if child_size <= np.finfo(np.float64).eps:
        return 0, 0
    quarter = float(node.size) / 4
    center = np.asarray(node.center, dtype=np.float64)
    offsets = (
        np.array(
            [
                [dx, dy, dz]
                for dx in (-1, 1)
                for dy in (-1, 1)
                for dz in (-1, 1)
            ],
            dtype=np.float64,
        )
        * quarter
    )

    children = []
    positive_side = points >= center
    octant_ids = (
        positive_side[:, 0].astype(np.uint8) * 4
        + positive_side[:, 1].astype(np.uint8) * 2
        + positive_side[:, 2].astype(np.uint8)
    )
    for octant_id, offset in enumerate(offsets):
        child_center = center + offset
        mask = octant_ids == octant_id
        if np.any(mask):
            children.append(
                OctreeNode(
                    center=child_center,
                    size=child_size,
                    points=points[mask],
                    depth=node.depth + 1,
                )
            )

    if not children:
        return 0, 0
    node.children = children
    return len(children) - 1, len(children)


def refine_octree_postprocess(root, add_frac=0.10, stop_half_depth=True, verbose=True):
    """Refine shallow leaves until the requested leaf-count growth is reached."""

    original_leaf_count = len(_collect_leaves(root))
    target_growth = int(np.ceil(original_leaf_count * add_frac))
    initial_max_depth = _max_depth(root)
    half_depth = (initial_max_depth + 1) // 2
    added_total = 0
    refined_nodes = 0

    if verbose:
        print(
            f"[refine_post] original_leaves={original_leaf_count}, "
            f"target_growth={target_growth}, max_depth={initial_max_depth}, "
            f"half_depth_stop={half_depth} (enabled={stop_half_depth})"
        )

    depth = 0
    while added_total < target_growth:
        if stop_half_depth and depth >= half_depth:
            break
        leaves = _collect_leaves_by_depth(root).get(depth, [])
        for leaf in leaves:
            if added_total >= target_growth:
                break
            added, child_count = _split_leaf_inplace(leaf)
            if child_count:
                refined_nodes += 1
                added_total += added
        depth += 1
        if not leaves and depth > _max_depth(root) + 1:
            break

    if verbose:
        new_leaf_count = len(_collect_leaves(root))
        print(
            f"[refine_post] refined_nodes={refined_nodes}, "
            f"added_leaves={added_total}, new_leaves={new_leaf_count}"
        )
    return root


def collect_leaf_info(node, use_centroid=True):
    """Collect one position and circumsphere radius per leaf."""

    if not node.children:
        position = node.points.mean(axis=0) if use_centroid else node.center
        radius = node.size * np.sqrt(3) / 2
        return [(position, radius)]
    result = []
    for child in node.children:
        result.extend(collect_leaf_info(child, use_centroid))
    return result


def octree_sample(points, min_points=35, device="cpu"):
    """Return primitive positions and initial radii sampled from a point cloud."""

    root = build_octree(points, min_points=min_points)
    root = refine_octree_postprocess(
        root,
        add_frac=0.1,
        stop_half_depth=True,
        verbose=True,
    )
    leaf_info = collect_leaf_info(root, use_centroid=True)
    positions = torch.from_numpy(np.array([p for p, _ in leaf_info])).float().to(device)
    radii = torch.from_numpy(np.array([r for _, r in leaf_info])).float().to(device)
    return positions, radii

import numpy as np
import torch

class OctreeNode:
    def __init__(self, center, size, points, depth=0):
        self.center = center   # cube center
        self.size = size       # cube side length (exactly 1/2^depth)
        self.points = points
        self.children = []
        self.depth = depth

def build_octree(points, min_points=200, depth=0, center=None, root_size=None):
    points = np.asarray(points)

    # Initialize root
    if depth == 0:
        mins, maxs = points.min(axis=0), points.max(axis=0)
        center = (mins + maxs) / 2
        root_size = np.max(maxs - mins)

    size = root_size / (2 ** depth)
    node = OctreeNode(center, size, points, depth)

    if len(points) <= min_points:
        return node

    # Sub-cube partition
    child_size = size / 2
    quarter = size / 4
    offsets = np.array([[dx, dy, dz] for dx in [-1, 1]
                                     for dy in [-1, 1]
                                     for dz in [-1, 1]]) * quarter
    for off in offsets:
        sub_center = center + off
        # Use strict interval to avoid duplicate assignment of boundary points
        mask = np.all((points >= sub_center - child_size/2) & (points < sub_center + child_size/2), axis=1)
        if np.any(mask):
            child = build_octree(points[mask], min_points, depth+1, sub_center, root_size)
            node.children.append(child)

    return node


def collect_leaf_info(node, use_centroid=True):
    """Collect all leaf nodes as (position, radius)"""
    if not node.children:  # leaf
        pos = node.points.mean(axis=0) if use_centroid else node.center
        radius = node.size / 2 * np.sqrt(3)  # radius = half diagonal length
        return [(pos, radius)]
    results = []
    for c in node.children:
        results.extend(collect_leaf_info(c, use_centroid))
    return results

def octree_sample(points, min_points=69, device="cpu"):
    """Sample from point cloud and return (positions, radii)"""
    root = build_octree(points, min_points=min_points)
    leaf_info = collect_leaf_info(root, use_centroid=True)

    positions = np.array([p for p, r in leaf_info])
    radii = np.array([r for p, r in leaf_info])

    positions = torch.from_numpy(positions).float().to(device)
    radii = torch.from_numpy(radii).float().to(device)

    return positions, radii


# Motivation
3D representation is the key in 3D computer vision / reconstruction / novel view sysnthesis. 
Gaussian Splatting is a handcraft primitive but mainly benefits from:
1. the ultra fast parallel "splatting" to convert 3D primitive property (e.g., density, color, radius) to 2D.
2. the 3D gaussian is very lightweight, with only a few float numbers

However, a 3D scene needs over 10K 3dgs to represent / reconstruct.

Can we find a much more flexible and less handcraft 3D representation that can be optimized efficiently?

1. MLP is the most basic and flexible 3D representation, it is light weight and can be parallelized.
2. The NeRF way of interacting with MLP is very compicated due to the dense sampling along the ray. In contrast, 3dgs is a simple "splatting" and light field is per ray computation (no sampling along the ray).
Therefore, we treat each MLP as a 3D primitive, and use the "splatting" way, each ray (ray_d, a normalised 3d vector, the ray direction), interacts with MLP (weights, scale, center), will have a distance (dist, a scalar), then the output (density, rgb) of this ray from this MLP is identical.
Mathematically:
```math
density, rgb = MLP(ray_d, dist)
```


# TODO
[ ] Visualize all MLPs for one image, check every MLP
[ ] Visualize a single MLP on different images / angles, to check its 3D shape

# Replica recipes

`room0_lseg.json` is the primary release recipe. It uses the paper's `m=10`
octree setting, initializes exactly 2,398 primitives, and evaluates the
feature-aware tile-center renderer at an 8×8 tile size and a `5e-3` score
threshold. The released checkpoint reaches 36.2243 dB on the ten-view
Feature-3DGS split, within 0.006 dB of the paper's 36.23 dB room0 result.

`room0_lseg_fast.json` is retained only as the compact `m=20` ablation:
1,344 primitives and a 35.88 dB paper target. It is not the main-table
reproduction.

The paper specifies 60k iterations, 8×8 tiles, four-view mosaics, four
directional 512D feature slots, and learning rates of `1e-2` for log-scale,
`1e-4` for position, and `1e-3` otherwise. Historical training checkpoints
parameterize the four effective slots as a shared base plus four residual
tensors (`feature_slots=5`). Release checkpoints fold these losslessly into
four direct tensors for inference.

The paper does not specify every implementation value. These recipes therefore
freeze the recovered implementation defaults: seed 2025, DSSIM weight 0.1,
attenuated-phase encoding with `sigma=0.01`, and feature-gate constants
`beta=1.5` and temperature `2.0`.

Feature supervision has a deliberately narrow gradient path in this release:
the feature loss updates feature attributes and the decoder, while primitive
geometry, opacity, and appearance remain RGB-supervised. See the root README
for the benchmark protocol and exact reproduction scope.

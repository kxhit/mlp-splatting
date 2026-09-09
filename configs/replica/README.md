# Replica recipes

`room0_lseg.json` freezes the paper-quality `m=10` setup. The paper reports
2,398 primitives and 36.23 dB for this room0 setting.

`room0_lseg_fast.json` matches the released 1,344-primitive checkpoint and the
paper's `m=20` ablation (35.88 dB). The clean release reproduces 35.849 dB on
the ten-view Feature-3DGS split.

The paper specifies 60k iterations, 8x8 tiles, four-view mosaics, four
directional 512D feature slots, and learning rates of `1e-2` for log-scale,
`1e-4` for position, and `1e-3` otherwise. The original runs parameterize those
four effective slots as one shared base plus four residual tensors; release
checkpoints may be converted algebraically to four direct tensors for inference.
The paper does not specify every implementation value.
The checked-in recipes therefore distinguish the paper values from frozen
release defaults such as seed 2025, DSSIM weight 0.1, score threshold `3e-3`,
and the recovered gate/encoding parameters. See the root README for current
reproduction scope and known differences.

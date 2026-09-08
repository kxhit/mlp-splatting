# Bigboy inference Pareto benchmark

Source selection started from the `mlpsp-perpx` candidate at base commit
`6ea35a2dcad3f45deffe8b37b1494c81229d38bd`.

The benchmark uses only physical GPU 0 and 1, both NVIDIA RTX 6000 Ada
Generation cards. Each process has a dedicated `TORCH_EXTENSIONS_DIR`. The
dataset is Replica `room0`, factor 1 (638×478), using the fixed Feature-3DGS
test split (`test_offset=2`, 10 views). The checkpoint contains 512-dimensional
features and 1,344 primitives.

Each screening result runs three complete 10-view warmup cycles followed by one
timed 10-view cycle for both RGB-only and fused RGB+feature inference. Finalist
results repeat the timed cycle three times on each physical GPU (30 samples per
GPU and mode). FPS is the reciprocal of mean end-to-end wall latency; the JSON
also contains CUDA-event latency, median/p95/CV, per-frame latency, per-image
PSNR, global PSNR, GPU/software metadata, exact command, split manifest, dirty
status, and hashes.

## Cross-GPU finalists

| traversal | tile | threshold | mean per-image PSNR | RGB FPS | fused RGB+feature FPS | feature speed vs per-pixel |
|---|---:|---:|---:|---:|---:|---:|
| per-pixel | 16 | 3e-3 | 35.90228 | 61.30 | 11.36 | baseline |
| tile-center | 8 | 3e-3 | 35.84900 | 28.38 | **12.45** | **+9.6%** |
| tile-center | 8 | 1e-2 | 35.69236 | 32.43 | **14.42** | **+26.9%** |

The highest-PSNR screened configuration was per-pixel/tile-8/1e-3 at
35.91970 dB. The recommended quality-gated fused configuration is therefore
tile-center/tile-8/3e-3: it loses 0.071 dB versus that reference while gaining
9.6% fused throughput versus the cross-GPU per-pixel finalist. Threshold 1e-2
is the more aggressive throughput preset (0.227 dB below the best screened
quality). A development-only tile size 4 was substantially slower and is not
exposed by the clean release. Tile size 16
with tile-center reached up to 17.69 fused FPS in screening, but its PSNR loss
was 0.743 dB, so it was rejected by the 0.10 dB quality gate.

RGB emitted by the fused feature kernels was bit-exact with the RGB-only path
for every screened and finalist configuration (`max_abs=0`).

## Five-slot to four-slot conversion

For legacy checkpoints, the algebraic conversion is:

```text
direct_slot[k] = base_slot + delta_slot[k], k = 0..3
```

Across both GPUs at the recommended tile-center/tile-8/3e-3 preset, the
five-slot and converted four-slot fused rates were 12.4657 and 12.4768 FPS
respectively (a statistically negligible +0.09%). RGB remained bit-exact;
feature parity was `max_abs=1.907e-6`, `RMSE=1.382e-8`. The conversion is useful
for the cleaner paper-aligned representation and reduces the feature-attribute
tensor element count by 20%, but should not be presented as a renderer speedup.

## Artifacts

- `pareto_all.csv`: flattened screening and cross-GPU measurements.
- `pareto_summary.json`: aggregate finalist and slot-conversion results.
- `release_gpu{0,1}_tilecenter_t8_s3e-3.json`: final release reruns with
  per-frame timings, view manifests, source/checkpoint hashes, and sanitized
  commands.
- `reference_gpu{0,1}_perpixel_t16_s3e-3.json`: clean-release quality-reference reruns.
- `aggressive_gpu{0,1}_tilecenter_t8_s1e-2.json`: clean-release throughput-preset reruns.
- `commands.txt`: portable command templates.

Screening-source hashes (before public-tree curation):

```text
516267b668d9daadca7b4bf94e7c66843192de6279dc1084c436c64ad7159ccd  mip360/cumodel.py
b5388710b53fc0dee9d5cb5e28d6f35902e1fde8ae2c64a2c6114b42aba5cedf  mip360/cuda/np_tile_kernels.cu
f017c083ca52582bad9c55e84aee4e1cf1d1bede91a1070cfbb1008f89eba3d1  mip360/eval_rgb_psnr.py
d52da43a6ceee9912817359108985ac4f208d6aeaaf9277c8a7a4c5fd2eea748  replica/replica_loader.py
7abd5a50a1499f78969fddaf7d83e55ec3ce194ebd4c38ea4e7b5872fbefbb4c  scene_epoch_60000.pt
```

## Clean release verification

After curating the source tree, commit `30cbdc5b39861138ee2d39f0ade6771740bfdf10`
was cloned cleanly and its CUDA extension compiled independently for each GPU.
All three finalists were rerun with the same three-cycle warm-up and timing
protocol. Each cell below is mean FPS over 30 frames; PSNR is identical on both
cards.

| configuration | PSNR | GPU 0 RGB | GPU 1 RGB | GPU 0 fused | GPU 1 fused | cross-GPU fused mean |
|---|---:|---:|---:|---:|---:|---:|
| per-pixel / 16 / `3e-3` | 35.90228 | 61.7820 | 60.7310 | 11.9669 | 11.7030 | 11.8349 |
| **tile-center / 8 / `3e-3`** | **35.84900** | 28.5105 | 28.0571 | **12.6309** | **12.4295** | **12.5302** |
| tile-center / 8 / `1e-2` | 35.69236 | 32.5897 | 32.2469 | 14.5264 | 14.4145 | 14.4704 |

Every report records a clean Git status, 10 fixed views, 30 timed samples per
output mode, and `max_abs=0` RGB parity. The converted four-slot checkpoint is
30,046,256 bytes with SHA-256
`9fc118ba64494351084b5f2779873fbccec9053bb438c07654f20c970ab12e29`.
The source hashes recorded by all six JSON files include
`b8a9661a…` for `mip360/cumodel.py`, `907629b4…` for the CUDA kernel, and
the checkpoint, loader, validation, evaluation, and benchmark modules. See the
JSON files for full hashes, exact frames, timings, software stack, and commands.

<div align="center">

# MLP Splatting

### Object-Centric Neural Fields

Compact neural primitives for photorealistic rendering, scene decomposition,
and feature-aware interaction.

[**Paper**](https://arxiv.org/abs/2606.03877) ·
[**Project page**](https://www.shinjeongkim.com/mlp-splatting/) ·
[**Interactive demo**](https://www.shinjeongkim.com/mlp-splatting/#interactive-demo) ·
[**Checkpoint**](https://github.com/kxhit/mlp-splatting/releases/tag/v0.1.0)

[![arXiv](https://img.shields.io/badge/arXiv-2606.03877-b31b1b.svg)](https://arxiv.org/abs/2606.03877)
[![ECCV](https://img.shields.io/badge/ECCV-2026-6f42c1.svg)](https://www.shinjeongkim.com/mlp-splatting/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](#installation)
[![CUDA](https://img.shields.io/badge/renderer-CUDA-76B900.svg?logo=nvidia&logoColor=white)](#installation)

<img src="assets/teaser.svg" alt="MLP Splatting overview" width="100%">

</div>

MLP Splatting represents a scene with a few expressive, spatially localized
MLPs. Each primitive predicts radiance and opacity inside an anisotropic
Gaussian support; optional feature distillation turns the same compact scene
representation into an open-vocabulary interface for selection and editing.

This release contains one canonical implementation: the feature-aware,
tile-based CUDA renderer selected from the development branches and verified
on both GPUs of our RTX 6000 Ada machine.

## Reproduce room0

The fastest path verifies the released 1,344-primitive LSeg checkpoint on ten
held-out Replica `room0` views. It reproduces **35.849 dB**, within 0.031 dB of
the paper's **35.88 dB** result for the `m=20` ablation.

### Installation

You need Linux, Python 3.10+, an NVIDIA GPU, and a CUDA toolkit containing
`nvcc`. The frozen benchmark environment used Python 3.11.14, PyTorch
2.9.0+cu128, torchvision 0.24.0+cu128, and CUDA toolkit 12.1.

```bash
git clone https://github.com/kxhit/mlp-splatting.git
cd mlp-splatting

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install a PyTorch wheel compatible with your driver first.
# See https://pytorch.org/get-started/locally/
python -m pip install torch torchvision
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps

python scripts/check_install.py
```

The extension in `mip360/cuda/np_tile_kernels.cu` compiles lazily on the first
render. Set `TORCH_EXTENSIONS_DIR` if you want to control its build cache.

Install perceptual and semantic evaluation dependencies only when needed:

```bash
python -m pip install -r requirements-optional.txt
```

### Data

Download the
[Feature-3DGS processed Replica archive](https://drive.google.com/file/d/1sC2ZJUBRHKeWXXVUj7rIBEM-xaibvGw7/view?usp=sharing).
Its scene folders use the expected COLMAP layout:

```text
<room0>/
├── images/
├── sparse/0/
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
└── rgb_feature_langseg/         # 512D LSeg features
```

The checked-in recipe requires 80 ordered images and applies the Feature-3DGS
split (`llffhold=8`, offset 2): 70 training views and 10 test views. A mismatch
fails before training or evaluation rather than silently changing the protocol.
Please retain the terms and citation of the
[Replica dataset](https://github.com/facebookresearch/Replica-Dataset).

### Checkpoint evaluation

```bash
mkdir -p checkpoints
curl -L \
  https://github.com/kxhit/mlp-splatting/releases/download/v0.1.0/room0_lseg512_tile8_60000_slots4.pt \
  -o checkpoints/room0_lseg512_tile8_60000_slots4.pt

echo "9fc118ba64494351084b5f2779873fbccec9053bb438c07654f20c970ab12e29  checkpoints/room0_lseg512_tile8_60000_slots4.pt" \
  | sha256sum --check

bash scripts/reproduce_room0.sh \
  /path/to/room0 \
  checkpoints/room0_lseg512_tile8_60000_slots4.pt
```

The script runs the same portable command available directly at the repository
root and checks both the ten-view manifest and a conservative PSNR interval:

```bash
python evaluate.py \
  --config configs/replica/room0_lseg_fast.json \
  --data /path/to/room0 \
  --checkpoint checkpoints/room0_lseg512_tile8_60000_slots4.pt \
  --disable-lpips \
  --output results/room0_lseg_fast.json
```

## Training

Important entrypoints stay at the repository root; implementation details stay
inside the package. `--output` is the exact run directory and receives a
`resolved_config.json` plus all checkpoints.

Paper-quality room0 recipe (`m=10`, paper target: 2,398 primitives / 36.23 dB):

```bash
python train.py \
  --config configs/replica/room0_lseg.json \
  --data /path/to/room0 \
  --output outputs/room0_lseg_paper
```

Faster recipe matching the released checkpoint (`m=20`, 1,344 primitives):

```bash
python train.py \
  --config configs/replica/room0_lseg_fast.json \
  --data /path/to/room0 \
  --output outputs/room0_lseg_fast
```

Both configs freeze 60,000 iterations, 8×8 tile-center traversal, four-view
tile mosaics, four effective 512D feature slots (trained as a shared base plus
four residual tensors), the data split, optimizer values, and evaluation
settings. Command-line options override config values. Run
`python train.py --help` for the full interface. Use `--feature-type none` for
an RGB-only experiment.

Evaluate a newly trained checkpoint with the matching recipe:

```bash
python evaluate.py \
  --config configs/replica/room0_lseg.json \
  --data /path/to/room0 \
  --checkpoint outputs/room0_lseg_paper/scene_epoch_60000.pt \
  --output results/room0_lseg_paper.json
```

This computes image-wise PSNR, SSIM, LPIPS, and synchronized render FPS. Add
`--disable-lpips` when only PSNR/SSIM and renderer timing are required.

## Semantic evaluation

`mip360/eval_lseg_table2.py` evaluates rendered LSeg features against semantic
labels. Labels and matching LSeg text embeddings are dataset-specific and are
not redistributed here.

```bash
python mip360/eval_lseg_table2.py \
  --data_dir /path/to/room0 \
  --scene_ckpt outputs/room0_lseg_paper/scene_epoch_60000.pt \
  --factor 1 \
  --gt_mode semantic_png \
  --gt_semantic_dir /path/to/labels \
  --labels_file /path/to/labels.txt \
  --text_feature_pt /path/to/lseg_text_features.pt \
  --tile_order tilecenter \
  --tile_size 8 \
  --score_threshold 3e-3 \
  --save_json results/room0_semantics.json
```

Omit `--text_feature_pt` to encode the label list with the pinned official
OpenAI CLIP dependency.

## Renderer selection

We compared the clean feature renderer on physical GPU 0 and 1, both NVIDIA
RTX 6000 Ada cards. Each result uses the same 638×478 `room0` checkpoint and ten
fixed held-out views, with three complete warm-up cycles followed by three
timed cycles per GPU and output mode.

| Traversal | Tile | Threshold | PSNR ↑ | RGB FPS ↑ | RGB + feature FPS ↑ |
|---|---:|---:|---:|---:|---:|
| per-pixel | 16×16 | `3e-3` | **35.9023** | **61.26** | 11.83 |
| **tile-center** | **8×8** | **`3e-3`** | **35.8490** | 28.28 | **12.53** |
| tile-center, aggressive | 8×8 | `1e-2` | 35.6924 | 32.42 | 14.47 |

The released default is tile-center / 8×8 / `3e-3`: fused feature inference is
5.9% faster than the per-pixel quality reference for a 0.053 dB difference.
RGB from the fused and RGB-only kernels was bit-exact in every run. These are
engineering measurements on RTX 6000 Ada, not replacements for the paper's
RTX 4090 tables.

Raw per-frame timings, commands, source hashes, checkpoint hash, and hardware
metadata are archived as a
[release asset](https://github.com/kxhit/mlp-splatting/releases/download/v0.1.0/room0_rtx6000ada_benchmarks.zip)
(SHA-256 `c10d070e765e7e3592c1581cd682c641927f1b34b209feb3bdf3f2e641788faa`)
instead of living in the source tree.

## Repository layout

```text
train.py                         public training entrypoint
evaluate.py                      public RGB evaluation entrypoint
configs/replica/                 paper-quality and fast room0 recipes
mip360/                          model, trainer, evaluation, COLMAP loader
└── cuda/np_tile_kernels.cu      feature-aware tile renderer
replica/                         Replica-specific dataset loader
scripts/                         install check and one-command reproduction
tools/                           benchmark and checkpoint migration utilities
tests/                           CPU tests plus optional CUDA regressions
```

## Reproduction scope

The released checkpoint path has been reproduced from a clean checkout on both
GPUs: 10/10 expected views, 35.849 dB tile-center PSNR, bit-exact fused/RGB
color, and successful CUDA regression tests. The `m=10` config records the
paper-quality recipe, but a fresh 60k run has not yet been independently
certified in this curated repository; do not cite it as a completed Table 1
reproduction without running it.

Some values needed by historical checkpoints were not stated in the paper and
are therefore labeled as frozen implementation defaults in
[`configs/replica/README.md`](configs/replica/README.md). In particular, the
released checkpoint uses attenuated-phase encoding (`sigma=0.01`) and recovered
feature-gate constants (`beta=1.5`, temperature `2.0`). The current feature
backward updates feature attributes; primitive geometry and appearance remain
RGB-supervised.

Rendering is CUDA-only. Tile candidate capacities are checked explicitly and
raise on overflow rather than truncating silently. Tile-center ordering is an
approximation, so validate against `per_pixel` when moving to a new dataset or
resolution.

No open-source license has been declared yet. Until the authors add one, this
repository is source-available without an implied reuse license. Third-party
attribution is recorded in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Citation

```bibtex
@inproceedings{kim2026mlpsplatting,
  title     = {MLP Splatting: Object-Centric Neural Fields},
  author    = {Kim, Shinjeong and Cheng, Yuzhou and Kong, Xin and Kelly, Paul H. J. and Davison, Andrew J.},
  booktitle = {The European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

See [`CITATION.cff`](CITATION.cff) for citation-manager metadata.

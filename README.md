<div align="center">

# MLP Splatting

### Object-Centric Neural Fields

Compact neural primitives for photorealistic rendering, scene decomposition,
and feature-aware interaction.

[**Paper**](https://arxiv.org/abs/2606.03877) ·
[**Project page**](https://www.shinjeongkim.com/mlp-splatting/) ·
[**Teaser video**](https://www.shinjeongkim.com/mlp-splatting/static/videos/teaser.mp4) ·
[**Interactive demo**](https://www.shinjeongkim.com/mlp-splatting/#visualization) ·
[**Models & logs**](https://github.com/kxhit/mlp-splatting/releases/tag/v0.2.0)

[![arXiv](https://img.shields.io/badge/arXiv-2606.03877-b31b1b.svg)](https://arxiv.org/abs/2606.03877)
[![ECCV](https://img.shields.io/badge/ECCV-2026-6f42c1.svg)](https://www.shinjeongkim.com/mlp-splatting/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](#installation)
[![CUDA](https://img.shields.io/badge/renderer-CUDA-76B900.svg?logo=nvidia&logoColor=white)](#installation)

<img src="assets/teaser.svg" alt="MLP Splatting overview" width="100%">

</div>

MLP Splatting represents a scene with a small set of spatially localized
light-field MLPs. Each primitive predicts radiance and opacity within an
anisotropic Gaussian support, combining the compactness of a neural field with
independent, object-level control. Optional semantic feature distillation adds
open-vocabulary selection, segmentation, and editing to the same representation.

This repository provides the canonical feature-aware, tile-based CUDA
implementation, pretrained models, and configurations used in the paper.

## Installation

You need Linux, Python 3.10+, an NVIDIA GPU, and a CUDA toolkit containing
`nvcc`. The verified benchmark environment used Python 3.11.14, PyTorch
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
render. Set `TORCH_EXTENSIONS_DIR` to choose its build cache. Perceptual and
semantic evaluation have a separate dependency set:

```bash
python -m pip install -r requirements-optional.txt
```

## Data

Download the
[Feature-3DGS processed Replica archive](https://drive.google.com/file/d/1sC2ZJUBRHKeWXXVUj7rIBEM-xaibvGw7/view?usp=sharing).
The `room0` directory should have this layout:

```text
room0/
├── images/
├── sparse/0/
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
└── rgb_feature_langseg/         # 512D LSeg features
```

The recipe requires 80 ordered images and applies the Feature-3DGS split
(`llffhold=8`, offset 2): 70 training views and 10 test views. A mismatch fails
before training or evaluation instead of silently changing the protocol.
Please retain the terms and citation of the
[Replica dataset](https://github.com/facebookresearch/Replica-Dataset).

## Evaluate a pretrained model

```bash
mkdir -p checkpoints
curl -L \
  https://github.com/kxhit/mlp-splatting/releases/download/v0.2.0/room0_lseg512_tile8_60000_m10_slots4.pt \
  -o checkpoints/room0_lseg512_tile8_60000_m10_slots4.pt

echo "dc49ba8de3eec72e6f8b5205501817b2a654c20fc5fd3dc8860167eb0aeb9f89  checkpoints/room0_lseg512_tile8_60000_m10_slots4.pt" \
  | sha256sum --check

bash scripts/reproduce_room0.sh \
  /path/to/room0 \
  checkpoints/room0_lseg512_tile8_60000_m10_slots4.pt
```

The script verifies the ten-view split and should report approximately
**36.22 dB**. The equivalent direct command is:

```bash
python evaluate.py \
  --config configs/replica/room0_lseg.json \
  --data /path/to/room0 \
  --checkpoint checkpoints/room0_lseg512_tile8_60000_m10_slots4.pt \
  --disable-lpips \
  --output results/room0_lseg.json
```

## Training

The two public experiment entrypoints live at the repository root. Dataset
recipes live under `configs/`; models, loaders, and CUDA implementation details
remain inside their packages. `--output` is the exact run directory and stores
the resolved configuration alongside every checkpoint.

```bash
python train.py \
  --config configs/replica/room0_lseg.json \
  --data /path/to/room0 \
  --output outputs/room0_lseg
```

The configuration records the complete `m=10` setup: 60,000 iterations, the
Feature-3DGS split, optimizer values, tile traversal, view mosaics, and feature
settings. Each run writes its resolved configuration beside the checkpoints.

Evaluate a newly trained checkpoint with the same recipe:

```bash
python evaluate.py \
  --config configs/replica/room0_lseg.json \
  --data /path/to/room0 \
  --checkpoint outputs/room0_lseg/scene_epoch_60000.pt \
  --output results/room0_lseg_trained.json
```

Evaluation reports image-wise PSNR, SSIM, LPIPS, and synchronized render FPS.
Use `--disable-lpips` when only PSNR/SSIM and renderer timing are required.
Command-line arguments override config values; run `python train.py --help` or
`python evaluate.py --help` for the complete interfaces.

Additional recipes and their provenance are documented in
[`configs/replica/README.md`](configs/replica/README.md).

## Semantic evaluation

`mip360/eval_lseg_table2.py` evaluates rendered LSeg features against semantic
labels. Labels and matching LSeg text embeddings are dataset-specific and are
not redistributed here.

```bash
python mip360/eval_lseg_table2.py \
  --data_dir /path/to/room0 \
  --scene_ckpt outputs/room0_lseg/scene_epoch_60000.pt \
  --factor 1 \
  --gt_mode semantic_png \
  --gt_semantic_dir /path/to/labels \
  --labels_file /path/to/labels.txt \
  --text_feature_pt /path/to/lseg_text_features.pt \
  --tile_order tilecenter \
  --tile_size 8 \
  --score_threshold 5e-3 \
  --save_json results/room0_semantics.json
```

Omit `--text_feature_pt` to encode the label list with the pinned official
OpenAI CLIP dependency.

## Repository layout

```text
train.py                         public training entrypoint
evaluate.py                      public RGB evaluation entrypoint
configs/replica/                 main and ablation room0 recipes
mip360/                          model, trainer, evaluation, COLMAP loader
└── cuda/np_tile_kernels.cu      feature-aware tile renderer
replica/                         Replica-specific dataset loader
scripts/                         install check and one-command reproduction
tools/                           benchmark and checkpoint migration utilities
tests/                           CPU tests plus optional CUDA regressions
```

## Reproducibility notes

Configuration files are versioned and validated strictly; dataset, split, and
primitive-count mismatches fail early. Exact benchmark commands, environment
metadata, and per-frame logs are available with the
[v0.2.0 release](https://github.com/kxhit/mlp-splatting/releases/tag/v0.2.0).

Rendering requires CUDA. Feature supervision optimizes the distilled feature
attributes and decoder, while primitive geometry and appearance remain
RGB-supervised. Further implementation details are recorded with each config.

Third-party attribution is recorded in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). A project license will be
added before the repository is made public.

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

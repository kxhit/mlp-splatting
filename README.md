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

This release distills the development code into one canonical feature-aware,
tile-based CUDA path. Its primary Replica `room0` recipe uses the paper's
`m=10` setting: **2,398 primitives and 36.2243 dB**, within **0.006 dB** of the
paper's 36.23 dB result.

## Reproduce room0

The released checkpoint and recipe evaluate ten fixed held-out views from the
Feature-3DGS Replica split. The default is the fastest tested fused RGB+feature
configuration inside a 0.10 dB quality budget.

| | Primitives | PSNR ↑ | Δ vs. paper |
|---|---:|---:|---:|
| Paper, `m=10` | 2,398 | 36.23 dB | — |
| **This release** | **2,398** | **36.2243 dB** | **−0.0057 dB** |

### Installation

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

### Data

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

### Checkpoint evaluation

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

The checkpoint is 53,605,040 bytes. The script uses the public root entrypoint,
checks the exact ten-view manifest, and requires PSNR in `[36.18, 36.27]` dB.
The equivalent direct command is:

```bash
python evaluate.py \
  --config configs/replica/room0_lseg.json \
  --data /path/to/room0 \
  --checkpoint checkpoints/room0_lseg512_tile8_60000_m10_slots4.pt \
  --disable-lpips \
  --output results/room0_lseg.json
```

## Train

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

This recipe freezes the `m=10` / 2,398-primitive setup, 60,000 iterations, 8×8
tile-center traversal, four-view tile mosaics, the Feature-3DGS split, optimizer
values, and feature settings. Historical training runs store five feature
tensors per primitive: one shared base plus four residuals, yielding four
effective directional 512D slots. The deployment checkpoint above folds these
losslessly into four direct slots.

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

The smaller `configs/replica/room0_lseg_fast.json` recipe is retained only to
reproduce the paper's `m=20` compact ablation (1,344 primitives / 35.88 dB). It
is not the main `m=10` reproduction.

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

## Renderer selection

We benchmarked the 2,398-primitive checkpoint independently on physical GPU 0
and GPU 1 of `bigboy`, both NVIDIA RTX 6000 Ada cards. Each result covers the
same ten 638×478 held-out views, with three complete warm-up cycles followed by
three timed cycles per GPU and output mode. The table reports the two-GPU mean.

| Traversal | Tile | Threshold | PSNR ↑ | RGB FPS ↑ | RGB + feature FPS ↑ |
|---|---:|---:|---:|---:|---:|
| per-pixel reference | 16×16 | `1e-3` | **36.3095** | **32.0040** | 7.6644 |
| **tile-center release** | **8×8** | **`5e-3`** | **36.2243** | 22.4804 | **10.1972** |

Tile-center / 8×8 / `5e-3` is the released feature-aware default: it improves
fused RGB+feature throughput by **33.1% on average per GPU** while changing
PSNR by **−0.0852 dB** relative to the per-pixel quality reference. Fused and
RGB-only paths produced bit-exact RGB in every timed run. For RGB-only
throughput, the per-pixel path is faster and remains available through the CLI.

These measurements characterize RTX 6000 Ada execution and are not a direct
replacement for the paper's RTX 4090 timing table. Raw per-frame timings,
commands, source and checkpoint hashes, and hardware metadata are preserved in
the
[v0.2.0 benchmark archive](https://github.com/kxhit/mlp-splatting/releases/download/v0.2.0/room0_m10_rtx6000ada_benchmarks.zip),
not in the source tree.

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

## Reproduction scope

The numbers above are checkpoint-reproduction results from a clean checkout on
both `bigboy` GPUs: 10/10 expected views, 36.2243 dB tile-center PSNR, and
bit-exact fused/RGB color. A full quality pass measured 0.9606 SSIM and 0.09334
LPIPS-VGG. A real one-iteration `m=10` training smoke test also
loaded all 80 images and features, constructed the expected 70/10 split,
initialized exactly 2,398 primitives, wrote a reloadable checkpoint, and
evaluated a held-out view. A fresh 60k training run has not yet been
independently repeated in this curated repository; do not present the released
checkpoint evaluation as a newly trained result.

Some historical implementation values are absent from the paper and are
therefore frozen and labeled in
[`configs/replica/README.md`](configs/replica/README.md). In particular, the
released checkpoint uses attenuated-phase encoding (`sigma=0.01`) and recovered
feature-gate constants (`beta=1.5`, temperature `2.0`). Feature loss updates
feature attributes and the decoder; primitive geometry, opacity, and appearance
remain RGB-supervised.

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

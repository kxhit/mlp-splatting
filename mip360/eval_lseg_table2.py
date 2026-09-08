#!/usr/bin/env python3
"""Evaluate MLP Splatting with LSeg-style metrics (mIoU / accuracy / FPS).

This script is designed for Feature-3DGS Table-2-style evaluation on test views.
It renders per-view semantic features from a trained mlpsp scene checkpoint,
converts features to semantic labels via CLIP text similarity, and reports:
  - mIoU (Feature-3DGS style: per-image top-k classes by teacher+student frequency)
  - pixel accuracy
  - FPS (render-only and render+semantic-query pipeline)

Typical usage (Replica split + manual semantic PNG GT):
  python mip360/eval_lseg_table2.py \
    --data_dir /path/to/scene \
    --scene_ckpt /path/to/scene_epoch_05000.pt \
    --dataset_loader replica \
    --gt_mode semantic_png \
    --gt_semantic_dir /path/to/manual_relabel_pngs \
    --miou_topk 7 \
    --feature_dim_target 512

Typical usage (teacher-feature proxy GT):
  python mip360/eval_lseg_table2.py \
    --data_dir /path/to/scene \
    --scene_ckpt /path/to/scene_epoch_05000.pt \
    --dataset_loader replica \
    --gt_mode teacher_feature \
    --teacher_feature_dir /path/to/rgb_feature_langseg \
    --miou_topk 7 \
    --feature_dim_target 512
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

try:
    from .checkpoints import infer_scene_spec, normalize_scene_state_dict, split_checkpoint
except ImportError:  # Support direct script execution.
    from checkpoints import infer_scene_spec, normalize_scene_state_dict, split_checkpoint

FEATURE3DGS_MANUAL_RELABEL_1BASED = {
    90: 15,  # TV -> door
    29: 4,   # rug -> floor
    58: 40,  # pillow -> cushion
}


def _strip_state_dict_prefixes(state_dict: dict) -> dict:
    return normalize_scene_state_dict(state_dict, remap_legacy_mlp=False)


def _extract_scene_state_dict(raw_ckpt: dict) -> dict:
    return split_checkpoint(raw_ckpt).scene_state_dict


def _infer_scene_dims(scene_sd: dict) -> tuple[int, int, int, int, int]:
    """Infer (num_primitives, hidden_dim, pe_frequencies, feature_dim_low, feature_slots)."""
    spec = infer_scene_spec(scene_sd)
    return (
        spec.num_primitives,
        spec.hidden_dim,
        spec.pe_frequencies,
        spec.feature_dim,
        spec.feature_slots,
    )


def _remap_legacy_mlp_to_tile_keys(scene_sd: dict) -> dict:
    """Map legacy vmodel MLP keys to tile_cuda keys if needed."""
    return normalize_scene_state_dict(scene_sd)


def _paired_decoder_path(scene_ckpt_path: str) -> str:
    base = os.path.basename(scene_ckpt_path)
    if base.startswith("scene_"):
        dec_base = "decoder_" + base[len("scene_") :]
    else:
        dec_base = "decoder_" + base
    return os.path.join(os.path.dirname(scene_ckpt_path), dec_base)


def _filter_kwargs_for_callable(func, kwargs: dict) -> dict:
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _call_scene_with_feature(
    scene,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    score_threshold: float,
    view_matrix: torch.Tensor,
    proj_matrix: torch.Tensor,
):
    """Call scene forward in a signature-compatible way across cumodel variants.

    Some variants use (ray_o, ray_d, ...), others use (rays_o, rays_d, ...).
    """
    call_target = scene.forward if hasattr(scene, "forward") else scene
    base_kwargs = {
        "score_threshold": float(score_threshold),
        "bg_color": None,
        "view_matrix": view_matrix,
        "proj_matrix": proj_matrix,
        "return_feature": True,
    }

    sig = inspect.signature(call_target)
    params = sig.parameters
    if "ray_o" in params:
        base_kwargs["ray_o"] = rays_o
    if "ray_d" in params:
        base_kwargs["ray_d"] = rays_d
    if "rays_o" in params:
        base_kwargs["rays_o"] = rays_o
    if "rays_d" in params:
        base_kwargs["rays_d"] = rays_d

    filtered = _filter_kwargs_for_callable(call_target, base_kwargs)
    has_named_rays = (("ray_o" in filtered) or ("rays_o" in filtered)) and (("ray_d" in filtered) or ("rays_d" in filtered))
    if has_named_rays:
        return scene(**filtered)

    # Fallback for unexpected signatures: pass rays positionally.
    for k in ("ray_o", "ray_d", "rays_o", "rays_d"):
        filtered.pop(k, None)
    return scene(rays_o, rays_d, **filtered)


def _build_projection_from_focal(
    focal: float,
    width: int,
    height: int,
    near: float = 0.001,
    far: float = 100.0,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    focal = float(focal)
    fov_y = 2.0 * float(torch.atan(torch.tensor((0.5 * height) / (focal + 1e-8))))
    aspect = float(width) / float(height)
    t = float(torch.tan(torch.tensor(fov_y / 2.0)))
    r = t * aspect
    p = torch.zeros(4, 4, dtype=torch.float32, device=device)
    p[0, 0] = 1.0 / r
    p[1, 1] = 1.0 / t
    p[2, 2] = -(far + near) / (far - near)
    p[3, 2] = 1.0
    p[2, 3] = -(2 * far * near) / (far - near)
    return p


def _c2w_to_view_matrix(c2w_3x4: torch.Tensor, device: str | torch.device) -> torch.Tensor:
    c2w_h = torch.eye(4, dtype=torch.float32, device=device)
    c2w_h[:3, :4] = c2w_3x4.to(device=device, dtype=torch.float32)
    return torch.inverse(c2w_h)


def _smart_sort_key(path_obj: Path) -> tuple[int, int, str]:
    stem = path_obj.stem
    match = re.search(r"(\d+)$", stem)
    if match:
        return (-1, int(match.group(1)), stem)
    return (0, -1, stem)


def _get_frame_identity(dataset, split_idx: int) -> dict:
    image_path = ""
    image_stem = f"{int(split_idx):04d}"
    if hasattr(dataset, "image_paths_selected") and len(getattr(dataset, "image_paths_selected", [])) > split_idx:
        image_path = str(dataset.image_paths_selected[split_idx])
        image_stem = Path(image_path).stem

    global_idx = int(split_idx)
    if hasattr(dataset, "select_ids") and len(getattr(dataset, "select_ids", [])) > split_idx:
        global_idx = int(dataset.select_ids[split_idx])

    return {
        "split_idx": int(split_idx),
        "global_idx": int(global_idx),
        "image_stem": str(image_stem),
        "image_path": str(image_path),
    }


def _load_ade20k_labels(ade20k_label_file: str | None) -> list[str]:
    candidate_paths = []
    if ade20k_label_file:
        candidate_paths.append(Path(ade20k_label_file))
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    candidate_paths.extend(
        [
            repo_root / "label_files" / "ade20k_objectInfo150.txt",
            repo_root.parent / "feature-3dgs" / "encoders" / "lseg_encoder" / "label_files" / "ade20k_objectInfo150.txt",
            Path.cwd() / "feature-3dgs" / "encoders" / "lseg_encoder" / "label_files" / "ade20k_objectInfo150.txt",
        ]
    )

    label_file = None
    for c in candidate_paths:
        if c.is_file():
            label_file = c
            break
    if label_file is None:
        raise FileNotFoundError(
            "ADE20K label file not found. Pass --ade20k_label_file explicitly, or provide --labels_csv / --labels_file."
        )

    labels = []
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            label = line.strip().split(",")[-1].split(";")[0]
            labels.append(label)
    # Match LSeg behavior (drop header line for ADE20K).
    labels = labels[1:]
    return labels


def _parse_labels(args) -> list[str]:
    if args.labels_csv.strip():
        labels = [x.strip() for x in args.labels_csv.split(",") if x.strip()]
        if not labels:
            raise ValueError("--labels_csv was provided but no valid labels were parsed.")
        return labels

    if args.labels_file.strip():
        p = Path(args.labels_file)
        if not p.is_file():
            raise FileNotFoundError(f"--labels_file not found: {p}")
        labels = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not labels:
            raise ValueError(f"No labels found in {p}")
        return labels

    return _load_ade20k_labels(args.ade20k_label_file)


def _load_text_features(args, labels: list[str], device: torch.device) -> torch.Tensor:
    if args.text_feature_pt.strip():
        text_raw = torch.load(args.text_feature_pt, map_location="cpu")
        if isinstance(text_raw, dict):
            if "text_feature" in text_raw and torch.is_tensor(text_raw["text_feature"]):
                text_feat = text_raw["text_feature"]
            elif "feat" in text_raw and torch.is_tensor(text_raw["feat"]):
                text_feat = text_raw["feat"]
            else:
                tensor_vals = [v for v in text_raw.values() if torch.is_tensor(v)]
                if not tensor_vals:
                    raise KeyError(f"No tensor found in --text_feature_pt dictionary: {args.text_feature_pt}")
                text_feat = tensor_vals[0]
        elif torch.is_tensor(text_raw):
            text_feat = text_raw
        else:
            raise TypeError("--text_feature_pt must be a tensor or dict containing a tensor.")
        if text_feat.ndim != 2:
            raise ValueError(f"Text feature must be 2D [num_classes, dim], got {tuple(text_feat.shape)}")
        if int(text_feat.shape[0]) != len(labels):
            raise ValueError(
                f"Text feature class count mismatch: tensor={int(text_feat.shape[0])}, labels={len(labels)}. "
                "Adjust labels or provide aligned text features."
            )
        return text_feat.float().to(device)

    try:
        from .text_encoder import CLIPTextEncoder
    except ImportError:
        from text_encoder import CLIPTextEncoder

    encoder = CLIPTextEncoder(
        device=device,
        model_name=str(args.clip_model),
        download_root=str(args.clip_download_root),
    )
    texts = [x.replace("_", " ") for x in labels]
    return encoder.encode_texts(texts).float().to(device)


def _parse_label_remap(remap_text: str) -> dict[int, int]:
    remap = {}
    txt = remap_text.strip()
    if txt == "":
        return remap
    for token in txt.split(","):
        token = token.strip()
        if token == "":
            continue
        if ":" not in token:
            raise ValueError(f"Invalid remap token '{token}'. Expected format 'src:dst'.")
        src_s, dst_s = token.split(":", 1)
        src = int(src_s.strip())
        dst = int(dst_s.strip())
        remap[src] = dst
    return remap


def _shift_label_remap(remap: dict[int, int], delta: int) -> dict[int, int]:
    if not remap:
        return {}
    shifted = {}
    for src, dst in remap.items():
        shifted[int(src) + int(delta)] = int(dst) + int(delta)
    return shifted


def _validate_label_remap(remap: dict[int, int], num_classes: int):
    if not remap:
        return
    invalid = []
    for src, dst in remap.items():
        if src < 0 or src >= num_classes or dst < 0 or dst >= num_classes:
            invalid.append((int(src), int(dst)))
    if invalid:
        raise ValueError(
            "Label remap contains out-of-range indices for num_classes="
            f"{num_classes}: {invalid}. "
            "If you provided ADE20K IDs (1..150), use --label_remap_one_based."
        )


def _validate_label_remap_one_based(remap: dict[int, int], num_classes: int):
    if not remap:
        return
    invalid = []
    for src, dst in remap.items():
        if src < 1 or src > num_classes or dst < 1 or dst > num_classes:
            invalid.append((int(src), int(dst)))
    if invalid:
        raise ValueError(
            "Label remap contains out-of-range 1-based indices for num_classes="
            f"{num_classes}: {invalid}. Expected IDs in [1, {num_classes}]."
        )


def _apply_label_remap(arr: np.ndarray, remap: dict[int, int]) -> np.ndarray:
    if not remap:
        return arr
    out = arr.copy()
    for src, dst in remap.items():
        out[out == int(src)] = int(dst)
    return out


def _ensure_hwc_feature(feat: torch.Tensor) -> torch.Tensor:
    if feat.ndim != 3:
        raise ValueError(f"Feature map must be 3D, got {tuple(feat.shape)}")
    # HWC vs CHW heuristic.
    if feat.shape[0] <= 8 and feat.shape[-1] > 8:
        return feat.permute(1, 2, 0).contiguous()
    if feat.shape[-1] <= 8 and feat.shape[0] > 8:
        return feat.contiguous()
    if feat.shape[0] > 8 and feat.shape[-1] > 8:
        # Ambiguous; prefer HWC (same default as dataset loader).
        return feat.contiguous()
    return feat.permute(1, 2, 0).contiguous()


def _resize_label_nearest(label_hw: np.ndarray, height: int, width: int) -> np.ndarray:
    if label_hw.shape[0] == height and label_hw.shape[1] == width:
        return label_hw
    t = torch.from_numpy(label_hw.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(height, width), mode="nearest")
    return t.squeeze(0).squeeze(0).to(torch.int64).cpu().numpy()


def _looks_zero_based_label_map(arr: np.ndarray, num_classes: int) -> bool:
    if arr.size == 0:
        return False
    mn = int(np.min(arr))
    mx = int(np.max(arr))
    return mn >= 0 and mx < int(num_classes)


def _to_feature3dgs_label_domain(arr: np.ndarray, num_classes: int) -> np.ndarray:
    # Feature-3DGS uses testset.make_pred(...) where ADE20K pred_offset=1.
    if _looks_zero_based_label_map(arr, num_classes=num_classes):
        return arr + 1
    return arr


def _feature3dgs_accuracy(teacher: np.ndarray, student: np.ndarray) -> float:
    correct_predictions = np.sum(teacher == student)
    total_pixels = int(np.prod(teacher.shape))
    return float(correct_predictions / total_pixels) if total_pixels > 0 else float("nan")


def _feature3dgs_iou(teacher: np.ndarray, student: np.ndarray, num_classes: int) -> float:
    miou, _selected, _iou_by_label = _feature3dgs_iou_with_top_labels(
        teacher=teacher,
        student=student,
        num_classes=num_classes,
    )
    return miou


def _feature3dgs_iou_with_top_labels(
    teacher: np.ndarray, student: np.ndarray, num_classes: int
) -> tuple[float, np.ndarray, dict[int, float]]:
    flat_teacher = teacher.reshape(-1)
    flat_student = student.reshape(-1)
    unique_labels, counts = np.unique(np.concatenate((flat_teacher, flat_student)), return_counts=True)
    sorted_labels = unique_labels[np.argsort(-counts)]

    if int(num_classes) > 0:
        labels = sorted_labels[: int(num_classes)]
    else:
        labels = sorted_labels

    iou_scores = []
    iou_by_label: dict[int, float] = {}
    for cls_idx in labels:
        true_labels = teacher == cls_idx
        predicted_labels = student == cls_idx
        intersection = np.logical_and(true_labels, predicted_labels)
        union = np.logical_or(true_labels, predicted_labels)
        union_sum = int(np.sum(union))
        if union_sum == 0:
            score = np.nan
        else:
            score = float(np.sum(intersection) / union_sum)
        iou_scores.append(score)
        iou_by_label[int(cls_idx)] = float(score) if not np.isnan(score) else float("nan")

    if len(iou_scores) == 0:
        return float("nan"), np.asarray([], dtype=np.int64), iou_by_label
    return (
        float(np.nanmean(np.asarray(iou_scores, dtype=np.float64))),
        np.asarray(labels, dtype=np.int64),
        iou_by_label,
    )


def _label_name_from_eval_index(labels: list[str], eval_class_idx: int, one_based: bool) -> str:
    idx = int(eval_class_idx) - 1 if bool(one_based) else int(eval_class_idx)
    if 0 <= idx < len(labels):
        return str(labels[idx])
    return f"class_{int(eval_class_idx)}"


def _sanitize_token(text: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", str(text)).strip("_")
    return safe if safe else "unknown"


def _deterministic_color(eval_class_idx: int) -> np.ndarray:
    key = (int(eval_class_idx) * 2654435761) & 0xFFFFFFFF
    return np.array(
        [
            64 + (key & 0x7F),
            64 + ((key >> 8) & 0x7F),
            64 + ((key >> 16) & 0x7F),
        ],
        dtype=np.uint8,
    )


def _to_uint8_rgb(
    img_like: torch.Tensor | np.ndarray | None,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    if img_like is None:
        return np.zeros((int(target_h), int(target_w), 3), dtype=np.uint8)
    if torch.is_tensor(img_like):
        arr = img_like.detach().cpu().numpy()
    else:
        arr = np.asarray(img_like)

    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.ndim != 3:
        return np.zeros((int(target_h), int(target_w), 3), dtype=np.uint8)

    if arr.shape[-1] > 3:
        arr = arr[..., :3]

    arr = arr.astype(np.float32)
    if float(np.max(arr)) <= 1.5:
        arr = arr * 255.0
    arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)

    if int(arr.shape[0]) != int(target_h) or int(arr.shape[1]) != int(target_w):
        arr = np.array(
            Image.fromarray(arr).resize((int(target_w), int(target_h)), Image.BILINEAR),
            dtype=np.uint8,
        )
    return arr


def _load_rgb_for_overlay(
    frame_meta: dict,
    sample_img_hw3,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    image_path = str(frame_meta.get("image_path", "")).strip()
    if image_path and Path(image_path).is_file():
        with Image.open(image_path) as img:
            arr = np.array(img.convert("RGB"), dtype=np.uint8)
        if int(arr.shape[0]) != int(target_h) or int(arr.shape[1]) != int(target_w):
            arr = np.array(
                Image.fromarray(arr).resize((int(target_w), int(target_h)), Image.BILINEAR),
                dtype=np.uint8,
            )
        return arr
    return _to_uint8_rgb(sample_img_hw3, target_h=target_h, target_w=target_w)


def _make_label_map(mask: np.ndarray, color: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[mask] = color
    return out


def _make_overlay(rgb_uint8: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    out = rgb_uint8.copy()
    if np.any(mask):
        out[mask] = (
            (1.0 - float(alpha)) * out[mask].astype(np.float32) + float(alpha) * color.astype(np.float32)
        ).astype(np.uint8)
    return out


def _add_panel_title(npimg: np.ndarray, title: str) -> np.ndarray:
    pil = Image.fromarray(npimg)
    draw = ImageDraw.Draw(pil)
    draw.rectangle((0, 0, pil.width, 18), fill=(0, 0, 0))
    draw.text((4, 3), str(title), fill=(255, 255, 255))
    return np.array(pil, dtype=np.uint8)


def _compose_2x2(
    teacher_map: np.ndarray,
    pred_map: np.ndarray,
    teacher_overlay: np.ndarray,
    pred_overlay: np.ndarray,
) -> np.ndarray:
    top = np.concatenate((teacher_map, pred_map), axis=1)
    bottom = np.concatenate((teacher_overlay, pred_overlay), axis=1)
    return np.concatenate((top, bottom), axis=0)


def _save_topk_visualizations(
    vis_root: Path,
    frame_meta: dict,
    rgb_uint8: np.ndarray,
    teacher_label_hw: np.ndarray,
    pred_label_hw: np.ndarray,
    selected_labels: np.ndarray,
    labels: list[str],
    one_based_label_domain: bool,
):
    frame_name = str(frame_meta.get("image_stem", f"{int(frame_meta.get('split_idx', 0)):04d}"))
    frame_dir = vis_root / frame_name
    frame_dir.mkdir(parents=True, exist_ok=True)

    for rank, cls_idx in enumerate(selected_labels.tolist(), start=1):
        color = _deterministic_color(int(cls_idx))
        class_name = _label_name_from_eval_index(
            labels=labels,
            eval_class_idx=int(cls_idx),
            one_based=bool(one_based_label_domain),
        )
        safe_class_name = _sanitize_token(class_name)

        teacher_mask = teacher_label_hw == int(cls_idx)
        pred_mask = pred_label_hw == int(cls_idx)

        teacher_map = _make_label_map(teacher_mask, color, teacher_label_hw.shape)
        pred_map = _make_label_map(pred_mask, color, pred_label_hw.shape)
        teacher_overlay = _make_overlay(rgb_uint8, teacher_mask, color)
        pred_overlay = _make_overlay(rgb_uint8, pred_mask, color)

        teacher_map = _add_panel_title(teacher_map, "teacher pseudo GT label map")
        pred_map = _add_panel_title(pred_map, "estimated label map")
        teacher_overlay = _add_panel_title(teacher_overlay, "teacher pseudo GT overlay")
        pred_overlay = _add_panel_title(pred_overlay, "estimated overlay")

        panel = _compose_2x2(teacher_map, pred_map, teacher_overlay, pred_overlay)
        out_name = f"top{rank:02d}_label{int(cls_idx):03d}_{safe_class_name}.png"
        Image.fromarray(panel).save(frame_dir / out_name)


def _resize_feature_hwc_bilinear(
    feature_hwc: torch.Tensor,
    height: int,
    width: int,
    align_corners: bool,
) -> torch.Tensor:
    if int(feature_hwc.shape[0]) == int(height) and int(feature_hwc.shape[1]) == int(width):
        return feature_hwc
    feat_chw = feature_hwc.permute(2, 0, 1).unsqueeze(0).float()
    feat_chw = F.interpolate(
        feat_chw,
        size=(int(height), int(width)),
        mode="bilinear",
        align_corners=bool(align_corners),
    )
    return feat_chw.squeeze(0).permute(1, 2, 0).contiguous()


def _classify_feature_map(feature_hwc: torch.Tensor, text_feature: torch.Tensor) -> torch.Tensor:
    """Return label map with shape [H, W], 0-based class indices."""
    if feature_hwc.ndim != 3:
        raise ValueError(f"feature_hwc must be 3D, got {tuple(feature_hwc.shape)}")
    h, w, d = int(feature_hwc.shape[0]), int(feature_hwc.shape[1]), int(feature_hwc.shape[2])
    if text_feature.ndim != 2:
        raise ValueError(f"text_feature must be 2D, got {tuple(text_feature.shape)}")
    if int(text_feature.shape[1]) != d:
        raise ValueError(
            f"Feature dim mismatch between rendered feature ({d}) and text embedding ({int(text_feature.shape[1])}). "
            "Use matching text encoder or pass --text_feature_pt."
        )

    image_feature = feature_hwc.view(-1, d).float()
    text_feature = text_feature.float()
    image_feature = F.normalize(image_feature, dim=1)
    text_feature = F.normalize(text_feature, dim=1)
    logits = torch.matmul(image_feature, text_feature.t())
    pred = torch.argmax(logits, dim=1)
    return pred.view(h, w)


def _update_confusion(
    confusion: np.ndarray,
    gt_hw: np.ndarray,
    pred_hw: np.ndarray,
    num_classes: int,
    ignore_index: int,
) -> tuple[int, int]:
    gt = gt_hw.reshape(-1).astype(np.int64)
    pr = pred_hw.reshape(-1).astype(np.int64)
    mask = (gt != int(ignore_index)) & (gt >= 0) & (gt < num_classes) & (pr >= 0) & (pr < num_classes)
    if not np.any(mask):
        return 0, 0
    gt = gt[mask]
    pr = pr[mask]
    idx = num_classes * gt + pr
    hist = np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    confusion += hist
    correct = int((gt == pr).sum())
    valid = int(gt.shape[0])
    return correct, valid


def _compute_metrics(confusion: np.ndarray, miou_topk: int) -> tuple[float, float, np.ndarray, np.ndarray]:
    eps = 1e-12
    gt_count = confusion.sum(axis=1).astype(np.float64)
    pred_count = confusion.sum(axis=0).astype(np.float64)
    diag = np.diag(confusion).astype(np.float64)
    union = gt_count + pred_count - diag
    iou = np.full_like(diag, np.nan, dtype=np.float64)
    valid_cls = union > 0
    iou[valid_cls] = diag[valid_cls] / (union[valid_cls] + eps)

    if miou_topk > 0:
        order = np.argsort(-gt_count)
        selected = [int(i) for i in order if gt_count[int(i)] > 0][: int(miou_topk)]
    else:
        selected = [int(i) for i in np.where(gt_count > 0)[0].tolist()]

    if len(selected) == 0:
        miou = float("nan")
    else:
        miou = float(np.nanmean(iou[selected]))

    total = float(confusion.sum())
    acc = float(diag.sum() / total) if total > 0 else float("nan")
    return miou, acc, iou, np.array(selected, dtype=np.int64)


class SemanticPNGProvider:
    def __init__(self, mask_dir: str, dataset, num_total_images: int | None = None):
        self.mask_dir = Path(mask_dir)
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"GT semantic directory not found: {self.mask_dir}")
        self.mask_paths = sorted(self.mask_dir.glob("*.png"), key=_smart_sort_key)
        if len(self.mask_paths) == 0:
            raise FileNotFoundError(f"No PNG files found in GT semantic directory: {self.mask_dir}")
        self.mask_by_stem = {p.stem: p for p in self.mask_paths}
        self.mask_by_suffix_int = {}
        for p in self.mask_paths:
            m = re.search(r"(\d+)$", p.stem)
            if m:
                self.mask_by_suffix_int[int(m.group(1))] = p
        self.dataset = dataset
        self.num_total_images = int(num_total_images) if num_total_images is not None else None

    def _resolve_path(self, split_idx: int) -> Path:
        image_stem = None
        if hasattr(self.dataset, "image_paths_selected") and len(getattr(self.dataset, "image_paths_selected", [])) > split_idx:
            image_stem = Path(self.dataset.image_paths_selected[split_idx]).stem
            if image_stem in self.mask_by_stem:
                return self.mask_by_stem[image_stem]

        global_idx = split_idx
        if hasattr(self.dataset, "select_ids") and len(getattr(self.dataset, "select_ids", [])) > split_idx:
            global_idx = int(self.dataset.select_ids[split_idx])

        canonical_name = f"semantic_class_{global_idx}"
        if canonical_name in self.mask_by_stem:
            return self.mask_by_stem[canonical_name]

        if global_idx in self.mask_by_suffix_int:
            return self.mask_by_suffix_int[global_idx]

        if len(self.mask_paths) == len(self.dataset):
            return self.mask_paths[split_idx]

        if self.num_total_images is not None and len(self.mask_paths) == self.num_total_images:
            if 0 <= global_idx < len(self.mask_paths):
                return self.mask_paths[global_idx]

        raise FileNotFoundError(
            f"Failed to map GT semantic PNG for split_idx={split_idx}, global_idx={global_idx}, image_stem={image_stem}."
        )

    def get_label(self, split_idx: int, target_h: int, target_w: int) -> np.ndarray:
        p = self._resolve_path(split_idx)
        arr = np.array(Image.open(p))
        if arr.ndim == 3:
            arr = arr[..., 0]
        arr = arr.astype(np.int64)
        return _resize_label_nearest(arr, target_h, target_w)


def _load_decoder_if_needed(
    scene_feature_dim: int,
    target_feature_dim: int,
    decoder_ckpt: str,
    scene_ckpt_path: str,
    device: torch.device,
):
    if scene_feature_dim == target_feature_dim:
        return None, ""
    if scene_feature_dim <= 0:
        raise RuntimeError("Scene has no semantic feature branch (feat_attr).")

    ckpt_path = decoder_ckpt.strip()
    raw = None
    decoder_source = ""
    if ckpt_path:
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Feature decoder checkpoint not found: {ckpt_path}")
        raw = torch.load(ckpt_path, map_location="cpu")
        decoder_source = ckpt_path
    else:
        scene_raw = torch.load(scene_ckpt_path, map_location="cpu")
        if (
            isinstance(scene_raw, dict)
            and isinstance(scene_raw.get("feature_decoder_state_dict"), dict)
        ):
            raw = scene_raw["feature_decoder_state_dict"]
            decoder_source = f"{scene_ckpt_path}:feature_decoder_state_dict"
        else:
            ckpt_path = _paired_decoder_path(scene_ckpt_path)
            if os.path.isfile(ckpt_path):
                raw = torch.load(ckpt_path, map_location="cpu")
                decoder_source = ckpt_path
            else:
                raise FileNotFoundError(
                    "Feature decoder checkpoint is required "
                    f"(scene feature dim={scene_feature_dim}, target={target_feature_dim}); "
                    "none was embedded in the scene checkpoint and the paired file "
                    f"was not found: {ckpt_path}"
                )

    if isinstance(raw, dict) and "feature_decoder_state_dict" in raw and isinstance(raw["feature_decoder_state_dict"], dict):
        raw = raw["feature_decoder_state_dict"]
    if not isinstance(raw, dict):
        raise TypeError(f"Decoder checkpoint must be a state_dict dictionary: {decoder_source}")
    sd = _strip_state_dict_prefixes(raw)

    w0 = sd.get("net.0.weight", None)
    w4 = sd.get("net.4.weight", None)
    if (not torch.is_tensor(w0)) or (not torch.is_tensor(w4)):
        raise KeyError("Decoder checkpoint is missing 'net.0.weight'/'net.4.weight'.")
    if w0.ndim != 4 or w4.ndim != 4:
        raise ValueError("Decoder checkpoint has unexpected conv weight shapes.")

    in_dim = int(w0.shape[1])
    out_dim = int(w4.shape[0])
    hidden_dim = int(w0.shape[0])
    if in_dim != scene_feature_dim or out_dim != target_feature_dim:
        raise ValueError(
            f"Decoder dim mismatch: checkpoint in/out=({in_dim},{out_dim}), expected ({scene_feature_dim},{target_feature_dim})."
        )

    try:
        from .feature_decoder import FeatureUpsampleDecoder
    except ImportError:
        from feature_decoder import FeatureUpsampleDecoder

    decoder = FeatureUpsampleDecoder(in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim).to(device)
    decoder.load_state_dict(sd, strict=True)
    decoder.eval()
    return decoder, decoder_source


def _build_dataset(dataset_cls, args, split: str, feature_type: str, feature_dir: str | None):
    kwargs = dict(
        basedir=args.data_dir,
        factor=int(args.factor),
        llffhold=int(args.llffhold),
        split=split,
        split_mode=args.split_mode,
        split_test_offset=args.split_test_offset,
        feature_type=feature_type,
        feature_dir=feature_dir,
        preload_features=bool(args.preload_features),
        feature_cache_size=int(args.feature_cache_size),
        feature_resize_to_image=bool(args.feature_resize_to_image),
        ray_cache_size=int(args.ray_cache_size),
    )
    return dataset_cls(**_filter_kwargs_for_callable(dataset_cls.__init__, kwargs))


def _load_dataset_class(loader_name: str):
    loader_name = str(loader_name).lower()
    if loader_name == "mip360":
        try:
            from .mip360_loader import mip360Dataset
        except ImportError:
            from mip360_loader import mip360Dataset

        return mip360Dataset
    if loader_name == "replica":
        this_dir = Path(__file__).resolve().parent
        repo_root = this_dir.parent
        replica_dir = repo_root / "replica"
        if not replica_dir.is_dir():
            raise FileNotFoundError(f"Replica loader directory not found: {replica_dir}")
        sys.path.insert(0, str(replica_dir))
        from replica_loader import mip360Dataset  # pylint: disable=import-error

        return mip360Dataset
    raise ValueError(f"Unknown --dataset_loader={loader_name}")


def _load_scene_tile_cuda(
    scene_ckpt_path: str,
    device: torch.device,
    min_points: int,
    tile_size: int = 8,
    feature_gating_beta: float | None = None,
    feature_gating_temp: float | None = None,
):
    try:
        from .cumodel import NeuralPrimitiveSceneTileCUDA
    except ImportError:
        from cumodel import NeuralPrimitiveSceneTileCUDA

    raw_ckpt = torch.load(scene_ckpt_path, map_location="cpu")
    if not isinstance(raw_ckpt, dict):
        raise TypeError(f"Unsupported checkpoint format: {scene_ckpt_path}")
    metadata = raw_ckpt.get("metadata", {}) if isinstance(raw_ckpt.get("metadata", {}), dict) else {}
    gating_beta = float(
        metadata.get("feature_gating_beta", 1.5)
        if feature_gating_beta is None
        else feature_gating_beta
    )
    gating_temp = float(
        metadata.get("feature_gating_temperature", 2.0)
        if feature_gating_temp is None
        else feature_gating_temp
    )
    encoding_kind = str(metadata.get("encoding_kind", "attenuated_phase_v1"))
    if encoding_kind not in {"attenuated_phase_v1", "fourier_features_v1"}:
        raise ValueError(f"Unsupported checkpoint encoding_kind={encoding_kind!r}")
    use_integrated_pe = encoding_kind == "attenuated_phase_v1"
    pe_sigma = float(metadata.get("encoding_sigma", 0.01))

    scene_sd = _extract_scene_state_dict(raw_ckpt)
    if not isinstance(scene_sd, dict):
        raise TypeError(f"Scene checkpoint does not contain a state_dict dictionary: {scene_ckpt_path}")
    scene_sd = _strip_state_dict_prefixes(scene_sd)
    scene_sd = _remap_legacy_mlp_to_tile_keys(scene_sd)

    n, hidden_dim, pe_frequencies, feature_dim_low, feature_slots = _infer_scene_dims(scene_sd)
    scene = NeuralPrimitiveSceneTileCUDA(
        num_primitives=n,
        hidden_dim=hidden_dim,
        pe_frequencies=pe_frequencies,
        use_integrated_pe=use_integrated_pe,
        pe_sigma=pe_sigma,
        initial_positions=None,
        min_points=int(min_points),
        tile_size=int(tile_size),
        feature_dim_low=int(feature_dim_low),
        feature_slots=int(feature_slots),
        feature_gating_beta=gating_beta,
        feature_gating_temp=gating_temp,
    ).to(device)
    missing, unexpected = scene.load_state_dict(scene_sd, strict=False)
    benign_missing = {
        "encoder.freq_bands",
        "stat_sum_w",
        "stat_max_w",
        "stat_hits",
        "stat_contrib_count",
        "stat_sum_werr",
        "alpha_accum",
    }
    missing = [k for k in missing if k not in benign_missing]
    if missing:
        raise RuntimeError(
            f"Checkpoint is missing required scene tensors: {missing[:10]} "
            f"(total={len(missing)})"
        )
    if unexpected:
        print(f"[Warn] Unexpected keys while loading scene: {unexpected[:10]} (total={len(unexpected)})")
    scene.eval()
    return scene, int(feature_dim_low)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MLP Splatting semantic metrics with LSeg features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Scene directory.")
    parser.add_argument("--scene_ckpt", type=str, required=True, help="Scene checkpoint (.pt).")
    parser.add_argument("--decoder_ckpt", type=str, default="", help="Optional decoder checkpoint if scene feature dim != target dim.")
    parser.add_argument("--dataset_loader", type=str, default="replica", choices=["replica", "mip360"], help="Dataset loader implementation.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--factor", type=int, default=4, help="Image downsample factor.")
    parser.add_argument("--llffhold", type=int, default=8, help="Train/test split period.")
    parser.add_argument("--split_mode", type=str, default="replica_feature3dgs", help="Replica loader split mode (llff/replica_feature3dgs).")
    parser.add_argument("--split_test_offset", type=int, default=None, help="Override split test offset if loader supports it.")
    parser.add_argument("--ray_cache_size", type=int, default=8, help="Dataset ray cache size.")
    parser.add_argument("--feature_cache_size", type=int, default=16, help="Dataset feature cache size.")
    parser.add_argument("--preload_features", action="store_true", help="Preload GT features in dataset loader.")
    parser.add_argument("--feature_resize_to_image", action="store_true", help="Resize GT feature maps to image size in dataset loader.")

    parser.add_argument("--gt_mode", type=str, default="semantic_png", choices=["semantic_png", "teacher_feature"], help="Ground-truth source.")
    parser.add_argument("--gt_semantic_dir", type=str, default="", help="Directory containing GT semantic PNG files (for gt_mode=semantic_png).")
    parser.add_argument("--gt_num_total_images", type=int, default=None, help="Total number of full-scene images for semantic PNG index mapping.")
    parser.add_argument("--teacher_feature_dir", type=str, default="", help="Teacher LSeg feature directory (for gt_mode=teacher_feature).")

    parser.add_argument("--labels_csv", type=str, default="", help="Comma-separated label names.")
    parser.add_argument("--labels_file", type=str, default="", help="Label file (one label per line).")
    parser.add_argument("--ade20k_label_file", type=str, default="", help="ADE20K label file path (optional).")
    parser.add_argument("--text_feature_pt", type=str, default="", help="Optional precomputed text feature tensor [C,D].")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32", help="CLIP model name for text encoding.")
    parser.add_argument("--clip_download_root", type=str, default="/tmp/tmp_clip", help="CLIP cache directory.")

    parser.add_argument("--feature_dim_target", type=int, default=512, help="Target rendered semantic feature dim (LSeg full=512).")
    parser.add_argument(
        "--miou_topk",
        type=int,
        default=7,
        help="Feature-3DGS mIoU uses top-k classes by combined (teacher+student) pixel frequency per image.",
    )
    parser.add_argument(
        "--metric_style",
        type=str,
        default="feature3dgs",
        choices=["feature3dgs", "legacy_confusion"],
        help="feature3dgs: match Feature-3DGS segmentation_metric.py exactly; legacy_confusion: previous confusion-matrix metric.",
    )
    parser.add_argument(
        "--metric_resize_hw",
        type=str,
        default="119,159",
        help="Resize labels to H,W before metric (Feature-3DGS default: 119,159).",
    )
    parser.add_argument(
        "--feature3dgs_preclass_resize_hw",
        type=str,
        default="360,480",
        help="Feature-map resize H,W before text-similarity classification in feature3dgs metric mode.",
    )
    parser.add_argument("--ignore_index", type=int, default=-1, help="Ignore label index in GT.")
    parser.add_argument("--gt_label_offset", type=int, default=0, help="Add offset to GT labels before metric (e.g., -1 for 1-based PNG labels).")
    parser.add_argument("--label_remap", type=str, default="", help="Optional label remap pairs. By default interpreted as 0-based eval indices, e.g. '89:14,28:3,57:39'.")
    parser.add_argument("--label_remap_one_based", action="store_true", help="Interpret --label_remap as 1-based ADE20K IDs (e.g. '90:15,29:4,58:40').")
    parser.add_argument("--disable_default_teacher_relabel", action="store_true", help="Disable default Feature-3DGS manual relabel in --gt_mode=teacher_feature.")

    parser.add_argument("--limit_frames", type=int, default=0, help="Evaluate only first N frames (0 means all).")
    parser.add_argument("--warmup_frames", type=int, default=1, help="Number of initial frames excluded from FPS timing.")
    parser.add_argument("--score_threshold", type=float, default=3e-3, help="Tile renderer score threshold.")
    parser.add_argument("--min_points", type=int, default=10, help="min_points used by scene constructor.")
    parser.add_argument("--tile_size", type=int, default=8, choices=[8, 16], help="Tile pixel size used by scene constructor (must match training).")
    parser.add_argument(
        "--tile_order",
        type=str,
        default="tilecenter",
        choices=["tilecenter", "per_pixel"],
        help="Tile traversal mode; tilecenter is the release default.",
    )
    parser.add_argument("--feature_gating_beta", type=float, default=None, help="Override checkpoint gating beta.")
    parser.add_argument("--feature_gating_temp", type=float, default=None, help="Override checkpoint gating temperature.")
    parser.add_argument("--save_pred_dir", type=str, default="", help="Optional directory to save predicted label PNGs (uint16).")
    parser.add_argument(
        "--save_topk_vis_dir",
        type=str,
        default="",
        help="Optional directory to save per-frame top-k label 2x2 visualization panels.",
    )
    parser.add_argument("--save_json", type=str, default="", help="Optional path to save evaluation summary JSON.")
    parser.add_argument("--device", type=str, default="cuda", help="Device, e.g. cuda or cuda:0.")
    args = parser.parse_args()

    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")
    device = torch.device(args.device if (not str(args.device).startswith("cuda") or torch.cuda.is_available()) else "cpu")
    if device.type == "cuda":
        device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        torch.cuda.set_device(device)

    os.environ["MLPSP_TILE_ORDER"] = str(args.tile_order)

    labels = _parse_labels(args)
    text_feature = _load_text_features(args, labels, device=device)
    num_classes = int(text_feature.shape[0])
    if num_classes != len(labels):
        raise RuntimeError("Label count and text feature count mismatch after loading.")

    dataset_cls = _load_dataset_class(args.dataset_loader)
    dataset = _build_dataset(dataset_cls, args, split=args.split, feature_type="none", feature_dir=None)
    if len(dataset) == 0:
        raise RuntimeError("Dataset split is empty.")

    scene_ckpt = os.path.abspath(args.scene_ckpt)
    scene, scene_feature_dim = _load_scene_tile_cuda(
        scene_ckpt,
        device=device,
        min_points=args.min_points,
        tile_size=args.tile_size,
        feature_gating_beta=args.feature_gating_beta,
        feature_gating_temp=args.feature_gating_temp,
    )
    if scene_feature_dim <= 0:
        raise RuntimeError("Loaded scene has no semantic feature branch (feat_attr).")

    decoder, decoder_path = _load_decoder_if_needed(
        scene_feature_dim=scene_feature_dim,
        target_feature_dim=int(args.feature_dim_target),
        decoder_ckpt=str(args.decoder_ckpt),
        scene_ckpt_path=scene_ckpt,
        device=device,
    )

    gt_provider = None
    teacher_dataset = None
    if args.gt_mode == "semantic_png":
        if args.gt_semantic_dir.strip() == "":
            raise ValueError("--gt_semantic_dir is required when --gt_mode=semantic_png.")
        gt_provider = SemanticPNGProvider(
            mask_dir=args.gt_semantic_dir,
            dataset=dataset,
            num_total_images=args.gt_num_total_images,
        )
    elif args.gt_mode == "teacher_feature":
        teacher_feature_dir = args.teacher_feature_dir.strip() or None
        teacher_dataset = _build_dataset(
            dataset_cls,
            args,
            split=args.split,
            feature_type="lseg",
            feature_dir=teacher_feature_dir,
        )
        if len(teacher_dataset) != len(dataset):
            raise RuntimeError(
                f"Teacher feature dataset length mismatch: {len(teacher_dataset)} vs render dataset {len(dataset)}."
            )

    remap_user = _parse_label_remap(args.label_remap)
    if str(args.metric_style) == "feature3dgs":
        if remap_user:
            remap = remap_user if args.label_remap_one_based else _shift_label_remap(remap_user, +1)
        elif args.gt_mode == "teacher_feature" and (not args.disable_default_teacher_relabel):
            remap = dict(FEATURE3DGS_MANUAL_RELABEL_1BASED)
            print("[Eval] Applying default Feature-3DGS manual relabel for teacher_feature: 90->15, 29->4, 58->40 (1-based ADE20K IDs).")
        else:
            remap = {}
        _validate_label_remap_one_based(remap, num_classes=num_classes)
        if remap_user:
            if args.label_remap_one_based:
                print(f"[Eval] Applying user label remap (1-based IDs): {remap}")
            else:
                print(f"[Eval] Applying user label remap converted 0->1 based: {remap}")
    else:
        if remap_user:
            remap = _shift_label_remap(remap_user, -1) if args.label_remap_one_based else remap_user
        elif args.gt_mode == "teacher_feature" and (not args.disable_default_teacher_relabel):
            remap = _shift_label_remap(FEATURE3DGS_MANUAL_RELABEL_1BASED, -1)
            print(
                "[Eval] Applying default Feature-3DGS manual relabel for teacher_feature: "
                "90->15, 29->4, 58->40 (1-based ADE20K), converted to 0-based indices."
            )
        else:
            remap = {}
        _validate_label_remap(remap, num_classes=num_classes)
        if remap_user:
            print(f"[Eval] Applying user label remap: {remap}")

    resize_tokens = [x.strip() for x in str(args.metric_resize_hw).split(",") if x.strip()]
    if len(resize_tokens) != 2:
        raise ValueError(f"--metric_resize_hw must be 'H,W', got: {args.metric_resize_hw}")
    metric_h = int(resize_tokens[0])
    metric_w = int(resize_tokens[1])
    if metric_h <= 0 or metric_w <= 0:
        raise ValueError(f"--metric_resize_hw must be positive, got: {args.metric_resize_hw}")

    preclass_tokens = [x.strip() for x in str(args.feature3dgs_preclass_resize_hw).split(",") if x.strip()]
    if len(preclass_tokens) != 2:
        raise ValueError(f"--feature3dgs_preclass_resize_hw must be 'H,W', got: {args.feature3dgs_preclass_resize_hw}")
    preclass_h = int(preclass_tokens[0])
    preclass_w = int(preclass_tokens[1])
    if preclass_h <= 0 or preclass_w <= 0:
        raise ValueError(f"--feature3dgs_preclass_resize_hw must be positive, got: {args.feature3dgs_preclass_resize_hw}")

    if args.save_pred_dir.strip():
        pred_dir = Path(args.save_pred_dir)
        pred_dir.mkdir(parents=True, exist_ok=True)
    else:
        pred_dir = None

    if args.save_topk_vis_dir.strip():
        topk_vis_dir = Path(args.save_topk_vis_dir)
    elif args.save_json.strip():
        save_json_path = Path(args.save_json)
        topk_vis_dir = save_json_path.parent / f"{save_json_path.stem}_miou_top{int(args.miou_topk)}_vis"
    else:
        topk_vis_dir = Path("results") / f"miou_top{int(args.miou_topk)}_vis"
    topk_vis_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Eval] Saving top-k visualization panels to: {topk_vis_dir}")

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    correct_total = 0
    valid_total = 0
    feature3dgs_acc_sum = 0.0
    feature3dgs_iou_sum = 0.0
    feature3dgs_count = 0
    per_image_metrics: list[dict] = []

    render_time_s = 0.0
    pipeline_time_s = 0.0
    timed_frames = 0

    max_frames = len(dataset) if int(args.limit_frames) <= 0 else min(len(dataset), int(args.limit_frames))
    print(
        f"[Eval] split={args.split} frames={max_frames} num_classes={num_classes} "
        f"scene_feature_dim={scene_feature_dim} target_dim={args.feature_dim_target} "
        f"decoder={'none' if decoder is None else decoder_path}"
    )

    for idx in range(max_frames):
        frame_meta = _get_frame_identity(dataset, idx)
        sample = dataset[idx]
        if not isinstance(sample, (tuple, list)) or len(sample) < 4:
            raise RuntimeError(f"Unexpected dataset sample format at idx={idx}")
        rays_o_hw, rays_d_hw, img_hw3, pose = sample[0], sample[1], sample[2], sample[3]
        h, w = int(rays_o_hw.shape[0]), int(rays_o_hw.shape[1])
        scene.set_image_size(w, h)

        rays_o = rays_o_hw.reshape(-1, 3).to(device, non_blocking=True)
        rays_d = rays_d_hw.reshape(-1, 3).to(device, non_blocking=True)
        view_matrix = _c2w_to_view_matrix(pose, device=device)
        proj_matrix = _build_projection_from_focal(dataset.focal, w, h, device=device)

        # FPS timing (render-only and render+query).
        use_timing = idx >= int(max(0, args.warmup_frames))
        if use_timing and device.type == "cuda":
            ev_pipeline_start = torch.cuda.Event(enable_timing=True)
            ev_render_start = torch.cuda.Event(enable_timing=True)
            ev_render_end = torch.cuda.Event(enable_timing=True)
            ev_pipeline_end = torch.cuda.Event(enable_timing=True)
            ev_pipeline_start.record()
            ev_render_start.record()
        elif use_timing:
            tp0 = time.perf_counter()
            tr0 = tp0

        with torch.no_grad():
            scene_out = _call_scene_with_feature(
                scene=scene,
                rays_o=rays_o,
                rays_d=rays_d,
                score_threshold=float(args.score_threshold),
                view_matrix=view_matrix,
                proj_matrix=proj_matrix,
            )
            if not isinstance(scene_out, (tuple, list)) or len(scene_out) != 2:
                raise RuntimeError("scene(..., return_feature=True) did not return (rgb, feat_low).")
            _rgb, feat_low = scene_out
            feat_low_img = feat_low.view(h, w, scene_feature_dim).permute(2, 0, 1).unsqueeze(0).float()
            if decoder is not None:
                feat_pred = decoder(feat_low_img).squeeze(0).permute(1, 2, 0).contiguous()
            else:
                feat_pred = feat_low_img.squeeze(0).permute(1, 2, 0).contiguous()

            if use_timing and device.type == "cuda":
                ev_render_end.record()
            elif use_timing:
                tr1 = time.perf_counter()

            feat_pred_for_cls = feat_pred
            if str(args.metric_style) == "feature3dgs" and int(w) > int(preclass_w):
                feat_pred_for_cls = _resize_feature_hwc_bilinear(
                    feat_pred_for_cls,
                    height=preclass_h,
                    width=preclass_w,
                    align_corners=True,
                )
            pred_label = _classify_feature_map(feat_pred_for_cls, text_feature).detach()
            pred_np = pred_label.cpu().numpy().astype(np.int64)

            if args.gt_mode == "semantic_png":
                gt_np = gt_provider.get_label(idx, target_h=h, target_w=w)
            else:
                teacher_sample = teacher_dataset[idx]
                if not isinstance(teacher_sample, (tuple, list)) or len(teacher_sample) < 5:
                    raise RuntimeError(f"Unexpected teacher dataset sample format at idx={idx}")
                gt_feat = _ensure_hwc_feature(teacher_sample[3].float())
                if int(gt_feat.shape[0]) != h or int(gt_feat.shape[1]) != w:
                    gt_feat = (
                        F.interpolate(
                            gt_feat.permute(2, 0, 1).unsqueeze(0),
                            size=(h, w),
                            mode="bilinear",
                            align_corners=True,
                        )
                        .squeeze(0)
                        .permute(1, 2, 0)
                        .contiguous()
                    )
                gt_feat_for_cls = gt_feat
                if str(args.metric_style) == "feature3dgs" and int(w) > int(preclass_w):
                    gt_feat_for_cls = _resize_feature_hwc_bilinear(
                        gt_feat_for_cls,
                        height=preclass_h,
                        width=preclass_w,
                        align_corners=True,
                    )
                gt_label = _classify_feature_map(gt_feat_for_cls.to(device), text_feature).detach()
                gt_np = gt_label.cpu().numpy().astype(np.int64)

        if use_timing and device.type == "cuda":
            ev_pipeline_end.record()
            torch.cuda.synchronize(device)
            render_time_s += float(ev_render_start.elapsed_time(ev_render_end) / 1000.0)
            pipeline_time_s += float(ev_pipeline_start.elapsed_time(ev_pipeline_end) / 1000.0)
            timed_frames += 1
        elif use_timing:
            tp1 = time.perf_counter()
            render_time_s += float(tr1 - tr0)
            pipeline_time_s += float(tp1 - tp0)
            timed_frames += 1

        metric_pred_np = pred_np
        metric_gt_np = gt_np
        if int(args.gt_label_offset) != 0:
            metric_gt_np = metric_gt_np + int(args.gt_label_offset)

        if str(args.metric_style) == "feature3dgs":
            metric_pred_np = _to_feature3dgs_label_domain(metric_pred_np, num_classes=num_classes)
            metric_gt_np = _to_feature3dgs_label_domain(metric_gt_np, num_classes=num_classes)
            if remap:
                metric_gt_np = _apply_label_remap(metric_gt_np, remap)
                metric_pred_np = _apply_label_remap(metric_pred_np, remap)

            metric_gt_np = _resize_label_nearest(metric_gt_np, metric_h, metric_w)
            metric_pred_np = _resize_label_nearest(metric_pred_np, metric_h, metric_w)

            acc_i = _feature3dgs_accuracy(metric_gt_np, metric_pred_np)
            iou_i, frame_top_labels, _frame_iou_by_label = _feature3dgs_iou_with_top_labels(
                teacher=metric_gt_np,
                student=metric_pred_np,
                num_classes=int(args.miou_topk),
            )
            feature3dgs_acc_sum += float(acc_i)
            feature3dgs_iou_sum += float(iou_i)
            feature3dgs_count += 1
            vis_label_domain_one_based = True

            valid_pixels_i = int(np.prod(metric_gt_np.shape))
            per_image_metrics.append(
                {
                    **frame_meta,
                    "mIoU": (None if math.isnan(float(iou_i)) else float(iou_i)),
                    "accuracy": (None if math.isnan(float(acc_i)) else float(acc_i)),
                    "valid_pixels": int(valid_pixels_i),
                }
            )

            if not math.isnan(float(acc_i)):
                correct_total += int(np.sum(metric_gt_np == metric_pred_np))
                valid_total += int(np.prod(metric_gt_np.shape))
        else:
            if remap:
                metric_gt_np = _apply_label_remap(metric_gt_np, remap)
                metric_pred_np = _apply_label_remap(metric_pred_np, remap)

            c, v = _update_confusion(
                confusion=confusion,
                gt_hw=metric_gt_np,
                pred_hw=metric_pred_np,
                num_classes=num_classes,
                ignore_index=int(args.ignore_index),
            )
            correct_total += c
            valid_total += v

            frame_conf = np.zeros_like(confusion)
            _update_confusion(
                confusion=frame_conf,
                gt_hw=metric_gt_np,
                pred_hw=metric_pred_np,
                num_classes=num_classes,
                ignore_index=int(args.ignore_index),
            )
            frame_miou, frame_acc, _frame_iou_per_class, _frame_selected_cls = _compute_metrics(
                confusion=frame_conf,
                miou_topk=int(args.miou_topk),
            )
            frame_top_labels = _frame_selected_cls
            vis_label_domain_one_based = False
            per_image_metrics.append(
                {
                    **frame_meta,
                    "mIoU": (None if math.isnan(float(frame_miou)) else float(frame_miou)),
                    "accuracy": (None if math.isnan(float(frame_acc)) else float(frame_acc)),
                    "valid_pixels": int(v),
                }
            )

        if pred_dir is not None:
            out_name = f"{frame_meta['image_stem']}.png"
            pred_img = Image.fromarray(metric_pred_np.astype(np.uint16), mode="I;16")
            pred_img.save(pred_dir / out_name)

        rgb_for_overlay = _load_rgb_for_overlay(
            frame_meta=frame_meta,
            sample_img_hw3=img_hw3,
            target_h=int(metric_gt_np.shape[0]),
            target_w=int(metric_gt_np.shape[1]),
        )
        _save_topk_visualizations(
            vis_root=topk_vis_dir,
            frame_meta=frame_meta,
            rgb_uint8=rgb_for_overlay,
            teacher_label_hw=metric_gt_np,
            pred_label_hw=metric_pred_np,
            selected_labels=np.asarray(frame_top_labels, dtype=np.int64),
            labels=labels,
            one_based_label_domain=vis_label_domain_one_based,
        )

        if (idx + 1) % 1 == 0:
            frame_result = per_image_metrics[-1]
            print(
                f"[Eval] {idx + 1}/{max_frames} "
                f"{frame_result['image_stem']} "
                f"mIoU={frame_result['mIoU']} "
                f"acc={frame_result['accuracy']}",
                flush=True,
            )

    if str(args.metric_style) == "feature3dgs":
        miou = (feature3dgs_iou_sum / feature3dgs_count) if feature3dgs_count > 0 else float("nan")
        pixel_acc = (feature3dgs_acc_sum / feature3dgs_count) if feature3dgs_count > 0 else float("nan")
        iou_per_class = np.array([], dtype=np.float64)
        selected_cls = np.array([], dtype=np.int64)
    else:
        miou, pixel_acc, iou_per_class, selected_cls = _compute_metrics(confusion=confusion, miou_topk=int(args.miou_topk))

    if valid_total > 0:
        pixel_acc_direct = float(correct_total / valid_total)
    else:
        pixel_acc_direct = float("nan")

    fps_render = float(timed_frames / render_time_s) if (timed_frames > 0 and render_time_s > 0) else float("nan")
    fps_pipeline = float(timed_frames / pipeline_time_s) if (timed_frames > 0 and pipeline_time_s > 0) else float("nan")
    avg_render_ms = float((render_time_s / timed_frames) * 1000.0) if timed_frames > 0 else float("nan")
    avg_pipeline_ms = float((pipeline_time_s / timed_frames) * 1000.0) if timed_frames > 0 else float("nan")

    selected_cls_info = []
    if str(args.metric_style) != "feature3dgs":
        selected_cls_info = [
            {
                "index": int(ci),
                "label": labels[int(ci)] if int(ci) < len(labels) else f"class_{int(ci)}",
                "iou": (None if math.isnan(float(iou_per_class[int(ci)])) else float(iou_per_class[int(ci)])),
                "gt_pixels": int(confusion[int(ci), :].sum()),
            }
            for ci in selected_cls.tolist()
        ]

    summary = {
        "data_dir": os.path.abspath(args.data_dir),
        "scene_ckpt": os.path.abspath(args.scene_ckpt),
        "decoder_ckpt": (decoder_path if decoder is not None else ""),
        "dataset_loader": args.dataset_loader,
        "split": args.split,
        "tile_order": str(args.tile_order),
        "tile_size": int(args.tile_size),
        "min_points": int(args.min_points),
        "score_threshold": float(args.score_threshold),
        "feature_gating_beta": float(scene.feature_gating_beta),
        "feature_gating_temperature": float(scene.feature_gating_temp),
        "num_frames_total_split": int(len(dataset)),
        "num_frames_evaluated": int(max_frames),
        "num_frames_timed": int(timed_frames),
        "warmup_frames": int(args.warmup_frames),
        "gt_mode": args.gt_mode,
        "metric_style": str(args.metric_style),
        "metric_resize_hw": [int(metric_h), int(metric_w)],
        "feature3dgs_preclass_resize_hw": [int(preclass_h), int(preclass_w)],
        "num_classes": int(num_classes),
        "feature_dim_scene": int(scene_feature_dim),
        "feature_dim_target": int(args.feature_dim_target),
        "miou_topk": int(args.miou_topk),
        "topk_vis_dir": str(topk_vis_dir.resolve()),
        "mIoU": (None if math.isnan(miou) else float(miou)),
        "accuracy": (None if math.isnan(pixel_acc) else float(pixel_acc)),
        "accuracy_direct_valid_pixels": (None if math.isnan(pixel_acc_direct) else float(pixel_acc_direct)),
        "fps_render_only": (None if math.isnan(fps_render) else float(fps_render)),
        "fps_render_plus_semantic": (None if math.isnan(fps_pipeline) else float(fps_pipeline)),
        "avg_render_ms": (None if math.isnan(avg_render_ms) else float(avg_render_ms)),
        "avg_render_plus_semantic_ms": (None if math.isnan(avg_pipeline_ms) else float(avg_pipeline_ms)),
        "valid_pixels": int(valid_total),
        "num_frames_metric_averaged": int(feature3dgs_count) if str(args.metric_style) == "feature3dgs" else int(max_frames),
        "selected_classes": selected_cls_info,
        "per_image_metrics": per_image_metrics,
    }

    print("\n=== Evaluation Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.save_json.strip():
        out_json = Path(args.save_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[Eval] Saved summary JSON to: {out_json}")


if __name__ == "__main__":
    main()

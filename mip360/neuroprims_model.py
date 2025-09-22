
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import torch
from torch import nn, Tensor

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.data.scene_box import SceneBox

# IMPORTANT: your project root must be on PYTHONPATH so this import works.
# e.g. export PYTHONPATH=/path/to/your/project:$PYTHONPATH
from model import NeuralPrimitiveScene  # type: ignore


def _strip_prefix(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    out = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            out[k[10:]] = v
        else:
            out[k] = v
    return out


@dataclass
class NeuroPrimsModelConfig(ModelConfig):
    """Config for wrapping your NeuralPrimitiveScene into a Nerfstudio Model."""
    _target: Type = field(default_factory=lambda: NeuroPrimsModel)

    num_primitives: int = 1024
    ckpt_path: Optional[Path] = None
    batch_size: int = 131072


class NeuroPrimsModel(Model):
    """Forward RayBundle -> NeuralPrimitiveScene, return rgb for the viewer."""
    config: NeuroPrimsModelConfig

    def __init__(self, config: NeuroPrimsModelConfig, scene_box: SceneBox, num_train_data: int, **kwargs) -> None:
        super().__init__(config=config, scene_box=scene_box, num_train_data=num_train_data, **kwargs)

        self.scene = NeuralPrimitiveScene(num_primitives=self.config.num_primitives, initial_positions=None)    # TODO

        # Optional checkpoint loading (safe / partial)
        if self.config.ckpt_path is not None and Path(self.config.ckpt_path).exists():
            ckpt = torch.load(self.config.ckpt_path, map_location="cpu")
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            model_sd = self.scene.state_dict()
            for k, v in _strip_prefix(ckpt).items():
                if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape):
                    model_sd[k] = v
            self.scene.load_state_dict(model_sd, strict=False)

        self.scene.to(self.device)
        self.scene.eval()

    def get_param_groups(self) -> Dict[str, List[nn.Parameter]]:
        return {
            "mlps": list(self.scene.mlps.parameters()),  # 默认 1e-3
            "log_scales": [self.scene.log_scales],  # 1e-1
            "positions": [self.scene.positions],  # 5e-3
        }

    # @torch.no_grad()
    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, Tensor]:  # type: ignore
        rays_o = ray_bundle.origins
        rays_d = ray_bundle.directions

        H = W = None
        if rays_o.ndim == 3:  # [H, W, 3] during eval
            H, W = rays_o.shape[0], rays_o.shape[1]
        rays_o = rays_o.reshape(-1, 3).to(self.device)
        rays_d = rays_d.reshape(-1, 3).to(self.device)

        preds = []
        bs = int(self.config.batch_size)
        for i in range(0, rays_o.shape[0], bs):
            preds.append(self.scene(rays_o[i:i+bs], rays_d[i:i+bs], mlp_id=None))
        rgb = torch.cat(preds, dim=0).clamp(0, 1)

        if H is not None and W is not None:
            rgb = rgb.view(H, W, 3)

        return {"rgb": rgb}

    @torch.no_grad()
    def get_metrics_dict(self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        metrics = {}
        if "image" in batch and "rgb" in outputs:
            gt = batch["image"].clamp(0, 1).to(self.device)
            if len(gt.shape) == 2:  # RGBA
                gt = gt[:, :3]  # Ignore alpha channel
            elif len(gt.shape) == 3:  # [H, W, C]
                gt = gt[:, :, :3]
            else:
                raise ValueError(f"Unexpected shape for gt image: {gt.shape}")
            pred = outputs["rgb"].clamp(0, 1)
            if pred.shape != gt.shape:
                try:
                    pred = pred.view_as(gt)
                except Exception:
                    pass
            mse = torch.mean((pred - gt) ** 2)
            psnr = -10.0 * torch.log10(mse + 1e-12)
            metrics["mse"] = mse.detach()
            metrics["psnr"] = psnr.detach()
        return metrics

    def get_loss_dict(self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor],
                      metrics_dict: Dict[str, Tensor] | None = None) -> Dict[str, Tensor]:
        loss_dict = {}
        if "image" in batch and "rgb" in outputs:
            gt = batch["image"].to(self.device)[:,:3]
            pred = outputs["rgb"]
            if pred.shape != gt.shape:
                try:
                    pred = pred.view_as(gt)
                except Exception:
                    pass
            loss_dict["rgb_mse"] = torch.mean((pred - gt) ** 2)
        return loss_dict

    @torch.no_grad()
    def get_image_metrics_and_images(
            self,
            outputs: Dict[str, Tensor],
            batch: Dict[str, Tensor],
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """Return (metrics_dict, images_dict) for eval/visualization."""
        metrics: Dict[str, Tensor] = {}

        pred = outputs.get("rgb", None)
        images: Dict[str, Tensor] = {}
        if pred is not None:
            images["pred_rgb"] = pred.clamp(0, 1).detach().cpu()

        gt = batch.get("image", None)[:,:,:3]
        if gt is not None:
            metrics_dict = self.get_metrics_dict(outputs, batch)
            metrics.update(metrics_dict)
            images["gt_rgb"] = gt.clamp(0, 1).detach().cpu()

        return metrics, images
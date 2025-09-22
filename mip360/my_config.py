
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Type

from nerfstudio.configs.base_config import ViewerConfig, MachineConfig, LoggingConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.instant_ngp_dataparser import InstantNGPDataParserConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig


from .neuroprims_model import NeuroPrimsModelConfig


NeuroPrimsMethod = MethodSpecification(
    config=TrainerConfig(
        method_name="neuroprims",
        pipeline=VanillaPipelineConfig(
            model=NeuroPrimsModelConfig(),
            datamanager=VanillaDataManagerConfig(
                # dataparser=InstantNGPDataParserConfig(),
                dataparser=BlenderDataParserConfig(),
                train_num_rays_per_batch=4096*3,
                eval_num_rays_per_batch=4096*3,
                # camera_res_scale_factor=0.25,   # 800*0.25=200
            ),
        ),
        optimizers={
            "mlps": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-8, weight_decay=0.0),
                "scheduler": ExponentialDecaySchedulerConfig(
                    warmup_steps=1000, max_steps=20000,
                    lr_final=1e-5, lr_pre_warmup=1e-8, ramp="cosine",
                ),
            },
            "log_scales": {
                "optimizer": AdamOptimizerConfig(lr=1e-1, eps=1e-8, weight_decay=0.0),
                "scheduler": ExponentialDecaySchedulerConfig(
                    warmup_steps=1000, max_steps=20000,
                    lr_final=1e-3, lr_pre_warmup=1e-8, ramp="cosine",
                ),
            },
            "positions": {
                "optimizer": AdamOptimizerConfig(lr=0, eps=1e-8, weight_decay=0.0), # TODO dont move positions
                "scheduler": ExponentialDecaySchedulerConfig(
                    warmup_steps=1000, max_steps=20000,
                    lr_final=5e-5, lr_pre_warmup=1e-8, ramp="cosine",
                ),
            },
        },
        viewer=ViewerConfig(
            quit_on_train_completion=False,
            default_composite_depth=False,
        ),
        machine=MachineConfig(),
        logging=LoggingConfig(),
        max_num_iterations=20000,
        # steps_per_eval_batch=5,
        # steps_per_eval_image=5,
    ),
    description="NeuralPrimitiveScene + InstantNGP (transforms.json) dataparser for full Nerfstudio viewer.",
)

#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig
from lerobot.utils.constants import ACTION


@PreTrainedConfig.register_subclass("diffusion_policy")
@dataclass
class DiffusionPolicyConfig(PreTrainedConfig):
    """LeRobot-compatible config skeleton for the from-scratch Diffusion Policy.

    This config intentionally inherits from `PreTrainedConfig` so it participates in the same LeRobot
    contracts as registered policies: feature schemas, normalization metadata, device/AMP defaults,
    optimizer/scheduler presets, temporal dataset sampling, and Hub config save/load behavior.

    `n_obs_steps`, `horizon`, and `n_action_steps` describe different time windows and should not be
    treated as aliases. `n_obs_steps` is the amount of observation history used for conditioning,
    `horizon` is the full future action trajectory denoised by the diffusion model, and
    `n_action_steps` is the receding-horizon prefix executed before the policy is queried again.

    `drop_n_last_frames` exists because LeRobotDataset samples future action windows from episode data.
    Near episode ends, those windows require copy-padding. Diffusion Policy traditionally drops the final
    `horizon - n_action_steps - n_obs_steps + 1` frames to reduce padded training examples.
    """

    # LeRobot compatibility fields. These drive dataset temporal sampling and processor normalization.
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # With the default DP timing, this is 16 - 8 - 2 + 1. Keeping it explicit mirrors the dataset API.
    drop_n_last_frames: int = 7

    # DP image encoder fields. The actual processor/model implementation is deferred to later phases.
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False

    # DP temporal U-Net architecture fields. These are validated now but not instantiated in Phase 1.
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True

    # Diffusion scheduler and inference fields.
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0
    num_inference_steps: int | None = None

    # Training loss field. Advanced keyframe/gripper weighting is deliberately deferred and disabled.
    do_mask_loss_for_padding: bool = False

    # Optimizer/scheduler preset fields consumed by LeRobot training utilities.
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.n_obs_steps < 1:
            raise ValueError(f"`n_obs_steps` must be >= 1. Got {self.n_obs_steps}.")
        if self.horizon < 1:
            raise ValueError(f"`horizon` must be >= 1. Got {self.horizon}.")
        if self.n_action_steps < 1:
            raise ValueError(f"`n_action_steps` must be >= 1. Got {self.n_action_steps}.")

        max_action_steps = self.horizon - self.n_obs_steps + 1
        if self.n_action_steps > max_action_steps:
            raise ValueError(
                "`n_action_steps` must be <= `horizon - n_obs_steps + 1`. "
                f"Got {self.n_action_steps=} with maximum {max_action_steps}."
            )

        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

        supported_prediction_types = {"epsilon", "sample"}
        if self.prediction_type not in supported_prediction_types:
            raise ValueError(
                f"`prediction_type` must be one of {sorted(supported_prediction_types)}. "
                f"Got {self.prediction_type}."
            )

        supported_noise_schedulers = {"DDPM", "DDIM"}
        if self.noise_scheduler_type not in supported_noise_schedulers:
            raise ValueError(
                f"`noise_scheduler_type` must be one of {sorted(supported_noise_schedulers)}. "
                f"Got {self.noise_scheduler_type}."
            )

        # The temporal U-Net downsamples by 2 at each stage, so its input horizon must divide cleanly.
        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                "`horizon` must be divisible by the temporal U-Net downsampling factor "
                f"`2 ** len(down_dims)`. Got {self.horizon=} and {self.down_dims=}."
            )

        if self.crop_shape is not None:
            if len(self.crop_shape) != 2:
                raise ValueError(f"`crop_shape` must be a tuple of (height, width). Got {self.crop_shape}.")
            if self.crop_shape[0] < 1 or self.crop_shape[1] < 1:
                raise ValueError(f"`crop_shape` dimensions must be >= 1. Got {self.crop_shape}.")

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("You must provide at least one image or environment-state input.")

        if self.action_feature is None:
            raise ValueError(f"You must provide `{ACTION}` in the output features.")

        if len(self.image_features) > 0:
            first_image_key, first_image_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if len(image_ft.shape) != 3:
                    raise ValueError(
                        f"`{key}` must have image shape (channels, height, width). Got {image_ft.shape}."
                    )
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_image_key}`; all image shapes must match."
                    )

        if self.crop_shape is not None:
            crop_height, crop_width = self.crop_shape
            for key, image_ft in self.image_features.items():
                image_height, image_width = image_ft.shape[1], image_ft.shape[2]
                if crop_height > image_height or crop_width > image_width:
                    raise ValueError(
                        f"`crop_shape` should fit within image shapes. Got {self.crop_shape} for "
                        f"`crop_shape` and {image_ft.shape} for `{key}`."
                    )

    @property
    def observation_delta_indices(self) -> list[int]:
        # LeRobotDataset uses these offsets to collect observation history ending at the current frame.
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list[int]:
        # Actions start at the oldest observation index and span the full denoising horizon.
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        # Diffusion Policy does not require reward targets for supervised behavior cloning.
        return None

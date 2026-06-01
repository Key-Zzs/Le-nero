#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
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
"""Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"

TODO(alexander-soare):
  - Remove reliance on diffusers for DDPMScheduler and LR scheduler.
"""

import logging
import math
from collections import deque
from collections.abc import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor, nn

from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

GRIPPER_ACTION_DIM_NAMES = {
    "left_gripper_cmd",
    "right_gripper_cmd",
    "left_gripper_cmd_bin",
    "right_gripper_cmd_bin",
    "left_gripper",
    "right_gripper",
    "gripper",
}

_LOSS_WEIGHTING_WARNINGS: set[str] = set()

logger = logging.getLogger(__name__)


def _warn_loss_weighting_once(message: str) -> None:
    if message in _LOSS_WEIGHTING_WARNINGS:
        return
    _LOSS_WEIGHTING_WARNINGS.add(message)
    logger.warning("Diffusion loss_weighting: %s", message)


def infer_gripper_dim_indices(
    action_dim: int,
    *,
    explicit_indices: list[int] | None = None,
    action_dim_names: list[str] | tuple[str, ...] | None = None,
) -> list[int]:
    """Resolve gripper action dimensions from explicit config or action feature names."""
    if explicit_indices is not None:
        resolved = []
        for index in explicit_indices:
            index = int(index)
            if 0 <= index < action_dim and index not in resolved:
                resolved.append(index)
            else:
                _warn_loss_weighting_once(
                    f"ignoring invalid gripper_dim_indices entry {index} for action_dim={action_dim}"
                )
        return resolved

    if action_dim_names is None:
        return []
    if len(action_dim_names) != action_dim:
        _warn_loss_weighting_once(
            "cannot infer gripper dims because action feature names length "
            f"({len(action_dim_names)}) does not match action_dim ({action_dim})"
        )
        return []

    resolved = []
    for index, name in enumerate(action_dim_names):
        normalized_name = str(name).strip().lower()
        leaf_name = normalized_name.rsplit(".", maxsplit=1)[-1].rsplit("/", maxsplit=1)[-1]
        if normalized_name in GRIPPER_ACTION_DIM_NAMES or leaf_name in GRIPPER_ACTION_DIM_NAMES:
            resolved.append(index)
    return resolved


def compute_unweighted_denoising_mse_loss(
    pred: Tensor,
    target: Tensor,
    action_is_pad: Tensor | None,
    config: DiffusionConfig,
) -> Tensor:
    """Original Diffusion denoising MSE, preserving the final mean denominator exactly."""
    mse_per_dim = F.mse_loss(pred, target, reduction="none")
    if config.do_mask_loss_for_padding:
        if action_is_pad is None:
            raise ValueError(
                "You need to provide 'action_is_pad' in the batch when "
                f"{config.do_mask_loss_for_padding=}."
            )
        if tuple(action_is_pad.shape) != tuple(pred.shape[:2]):
            raise ValueError(
                "action_is_pad must have shape [B, H]. "
                f"Got {tuple(action_is_pad.shape)}, expected {tuple(pred.shape[:2])}."
            )
        mse_per_dim = mse_per_dim * (~action_is_pad.to(device=pred.device, dtype=torch.bool)).unsqueeze(-1)
    return mse_per_dim.mean()


def _get_horizon_weight(
    batch: dict[str, Tensor],
    key: str,
    *,
    shape: tuple[int, int],
    like: Tensor,
    max_weight: float,
) -> Tensor:
    raw_weight = batch.get(key)
    if raw_weight is None:
        _warn_loss_weighting_once(f"missing '{key}', falling back to horizon weight 1")
        return torch.ones((*shape, 1), dtype=like.dtype, device=like.device)
    if not isinstance(raw_weight, Tensor):
        _warn_loss_weighting_once(f"'{key}' is not a tensor, falling back to horizon weight 1")
        return torch.ones((*shape, 1), dtype=like.dtype, device=like.device)

    weight = raw_weight.to(device=like.device, dtype=like.dtype)
    if weight.ndim == 3 and weight.shape[-1] == 1:
        weight = weight.squeeze(-1)
    if tuple(weight.shape) != shape:
        _warn_loss_weighting_once(
            f"'{key}' has shape {tuple(weight.shape)}, expected {shape}; falling back to horizon weight 1"
        )
        return torch.ones((*shape, 1), dtype=like.dtype, device=like.device)

    weight = torch.nan_to_num(weight, nan=1.0, posinf=max_weight, neginf=1.0)
    return weight.clamp(min=0.0, max=max_weight).unsqueeze(-1)


def _scale_horizon_weight(horizon_weight: Tensor, scale: float, max_weight: float) -> Tensor:
    scaled = 1.0 + (horizon_weight - 1.0) * float(scale)
    return scaled.clamp(min=0.0, max=max_weight)


def _build_horizon_dim_weight(
    horizon_weight: Tensor,
    *,
    action_dim: int,
    gripper_dim_indices: list[int],
    config: DiffusionConfig,
) -> Tensor:
    weighting_cfg = config.loss_weighting
    if not weighting_cfg.use_horizon_weight:
        return torch.ones(
            (*horizon_weight.shape[:2], action_dim),
            dtype=horizon_weight.dtype,
            device=horizon_weight.device,
        )

    weight = torch.ones(
        (*horizon_weight.shape[:2], action_dim),
        dtype=horizon_weight.dtype,
        device=horizon_weight.device,
    )
    gripper_mask = torch.zeros(action_dim, dtype=torch.bool, device=horizon_weight.device)
    if gripper_dim_indices:
        gripper_mask[gripper_dim_indices] = True
    pose_mask = ~gripper_mask

    if weighting_cfg.apply_to_pose_dims and pose_mask.any():
        weight[..., pose_mask] = _scale_horizon_weight(
            horizon_weight,
            weighting_cfg.pose_keyframe_weight_scale,
            weighting_cfg.max_weight,
        )
    if weighting_cfg.apply_to_gripper_dims and gripper_mask.any():
        weight[..., gripper_mask] = _scale_horizon_weight(
            horizon_weight,
            weighting_cfg.gripper_keyframe_weight_scale,
            weighting_cfg.max_weight,
        )
    return weight


def _build_action_dim_weight(
    action_dim: int,
    *,
    gripper_dim_indices: list[int],
    like: Tensor,
    config: DiffusionConfig,
) -> Tensor:
    weighting_cfg = config.loss_weighting
    action_dim_weight = torch.ones((1, 1, action_dim), dtype=like.dtype, device=like.device)
    if weighting_cfg.use_action_dim_weight and gripper_dim_indices:
        action_dim_weight[..., gripper_dim_indices] = float(weighting_cfg.gripper_dim_weight)
    return torch.nan_to_num(
        action_dim_weight,
        nan=1.0,
        posinf=weighting_cfg.max_weight,
        neginf=1.0,
    ).clamp(min=0.0, max=weighting_cfg.max_weight)


def _build_valid_mask(
    action_is_pad: Tensor | None,
    *,
    shape: tuple[int, int],
    like: Tensor,
    config: DiffusionConfig,
) -> Tensor:
    if not config.do_mask_loss_for_padding:
        return torch.ones((*shape, 1), dtype=like.dtype, device=like.device)
    if action_is_pad is None:
        raise ValueError(
            "You need to provide 'action_is_pad' in the batch when "
            f"{config.do_mask_loss_for_padding=}."
        )
    if tuple(action_is_pad.shape) != shape:
        raise ValueError(
            "action_is_pad must have shape [B, H]. "
            f"Got {tuple(action_is_pad.shape)}, expected {shape}."
        )
    valid_mask = (~action_is_pad.to(device=like.device, dtype=torch.bool)).unsqueeze(-1)
    return valid_mask.to(dtype=like.dtype)


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor | None:
    mask = mask.to(dtype=value.dtype, device=value.device)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    if mask.shape != value.shape:
        mask = mask.expand_as(value)
    denominator = mask.sum()
    if denominator <= 0:
        return None
    return (value * mask).sum() / denominator.clamp_min(1e-6)


def _add_metric(metrics: dict[str, Tensor], key: str, value: Tensor | None) -> None:
    if value is not None:
        metrics[key] = value.detach()


def _get_event_tensor(
    batch: dict[str, Tensor],
    key: str,
    *,
    shape: tuple[int, int],
    device: torch.device,
) -> Tensor | None:
    event = batch.get(key)
    if event is None or not isinstance(event, Tensor):
        return None
    event = event.to(device=device)
    if event.ndim == 3 and event.shape[-1] == 1:
        event = event.squeeze(-1)
    if tuple(event.shape) != shape:
        _warn_loss_weighting_once(
            f"'{key}' has shape {tuple(event.shape)}, expected {shape}; skipping event breakdown"
        )
        return None
    return event.to(dtype=torch.long)


def compute_weighted_denoising_mse_loss(
    pred: Tensor,
    target: Tensor,
    action_is_pad: Tensor | None,
    batch: dict[str, Tensor],
    config: DiffusionConfig,
    action_dim_names: list[str] | tuple[str, ...] | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute enabled Diffusion weighted denoising MSE and detached logging metrics."""
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError(
            "pred and target must have matching [B, H, D] shapes. "
            f"Got {tuple(pred.shape)} and {tuple(target.shape)}."
        )

    batch_size, horizon, action_dim = pred.shape
    weighting_cfg = config.loss_weighting
    mse_per_dim = F.mse_loss(pred, target, reduction="none")

    valid_mask = _build_valid_mask(
        action_is_pad,
        shape=(batch_size, horizon),
        like=mse_per_dim,
        config=config,
    )

    # This is an action-horizon step weight [B, H, 1], not a diffusion scheduler timestep weight [B].
    if weighting_cfg.use_horizon_weight:
        horizon_weight = _get_horizon_weight(
            batch,
            weighting_cfg.keyframe_weight_column,
            shape=(batch_size, horizon),
            like=mse_per_dim,
            max_weight=weighting_cfg.max_weight,
        )
    else:
        horizon_weight = torch.ones((batch_size, horizon, 1), dtype=mse_per_dim.dtype, device=mse_per_dim.device)

    gripper_dim_indices = []
    needs_gripper_dims = (
        weighting_cfg.use_action_dim_weight
        or weighting_cfg.log_weighted_loss_breakdown
        or (
            weighting_cfg.use_horizon_weight
            and (
                weighting_cfg.apply_to_pose_dims != weighting_cfg.apply_to_gripper_dims
                or weighting_cfg.pose_keyframe_weight_scale != weighting_cfg.gripper_keyframe_weight_scale
            )
        )
    )
    if needs_gripper_dims:
        names = action_dim_names if weighting_cfg.infer_gripper_dim_from_feature_names else None
        gripper_dim_indices = infer_gripper_dim_indices(
            action_dim,
            explicit_indices=weighting_cfg.gripper_dim_indices,
            action_dim_names=names,
        )
        if weighting_cfg.use_action_dim_weight and not gripper_dim_indices:
            _warn_loss_weighting_once(
                "could not resolve gripper action dims; using horizon weights without gripper dim weight"
            )

    horizon_dim_weight = _build_horizon_dim_weight(
        horizon_weight,
        action_dim=action_dim,
        gripper_dim_indices=gripper_dim_indices,
        config=config,
    )
    action_dim_weight = _build_action_dim_weight(
        action_dim,
        gripper_dim_indices=gripper_dim_indices,
        like=mse_per_dim,
        config=config,
    )

    effective_weight = valid_mask * horizon_dim_weight * action_dim_weight
    weighted_mse = mse_per_dim * effective_weight
    if weighting_cfg.normalize_weighted_loss:
        denominator = effective_weight.sum().clamp_min(1e-6)
    else:
        denominator = torch.tensor(mse_per_dim.numel(), dtype=mse_per_dim.dtype, device=mse_per_dim.device)
    loss = weighted_mse.sum() / denominator

    metrics: dict[str, Tensor] = {}
    if not weighting_cfg.log_weighted_loss_breakdown:
        return loss, metrics

    with torch.no_grad():
        valid_horizon_mask = valid_mask.squeeze(-1) > 0
        valid_dim_mask = valid_mask.expand_as(mse_per_dim) > 0
        keyframe_mask = (horizon_weight.squeeze(-1) > 1.0) & valid_horizon_mask
        normal_mask = (horizon_weight.squeeze(-1) <= 1.0) & valid_horizon_mask

        denoising_mse_unweighted = _masked_mean(mse_per_dim, valid_dim_mask)
        _add_metric(metrics, "loss/denoising_mse_unweighted", denoising_mse_unweighted)
        _add_metric(metrics, "loss/dp_denoising_mse_unweighted", denoising_mse_unweighted)
        metrics["loss/denoising_mse_weighted"] = loss.detach()
        metrics["loss/dp_denoising_mse_weighted"] = loss.detach()
        keyframe_mse = _masked_mean(mse_per_dim, keyframe_mask)
        normal_mse = _masked_mean(mse_per_dim, normal_mask)
        _add_metric(metrics, "loss/keyframe_mse", keyframe_mse)
        _add_metric(metrics, "loss/dp_keyframe_mse", keyframe_mse)
        _add_metric(metrics, "loss/normal_mse", normal_mse)
        _add_metric(metrics, "loss/dp_normal_mse", normal_mse)
        valid_horizon_count = valid_horizon_mask.to(dtype=mse_per_dim.dtype).sum().clamp_min(1e-6)
        metrics["loss/keyframe_ratio"] = (
            keyframe_mask.to(dtype=mse_per_dim.dtype).sum() / valid_horizon_count
        ).detach()
        metrics["loss/normal_ratio"] = (
            normal_mask.to(dtype=mse_per_dim.dtype).sum() / valid_horizon_count
        ).detach()
        mean_annotation_weight = _masked_mean(horizon_weight, valid_horizon_mask)
        max_annotation_weight = horizon_weight[valid_horizon_mask].max() if valid_horizon_mask.any() else None
        _add_metric(metrics, "loss/mean_horizon_weight", mean_annotation_weight)
        _add_metric(
            metrics,
            "loss/max_horizon_weight",
            max_annotation_weight,
        )
        _add_metric(metrics, "loss/mean_annotation_weight", mean_annotation_weight)
        _add_metric(metrics, "loss/max_annotation_weight", max_annotation_weight)
        _add_metric(metrics, "loss/mean_effective_weight", _masked_mean(effective_weight, valid_dim_mask))
        _add_metric(
            metrics,
            "loss/max_effective_weight",
            effective_weight[valid_dim_mask].max() if valid_dim_mask.any() else None,
        )

        gripper_dim_mask = torch.zeros(action_dim, dtype=torch.bool, device=mse_per_dim.device)
        if gripper_dim_indices:
            gripper_dim_mask[gripper_dim_indices] = True
            gripper_mse = _masked_mean(mse_per_dim, valid_dim_mask & gripper_dim_mask)
            _add_metric(metrics, "loss/gripper_mse", gripper_mse)
            _add_metric(metrics, "loss/dp_gripper_mse", gripper_mse)
        pose_dim_mask = ~gripper_dim_mask
        if pose_dim_mask.any():
            pose_mse = _masked_mean(mse_per_dim, valid_dim_mask & pose_dim_mask)
            _add_metric(metrics, "loss/pose_mse", pose_mse)
            _add_metric(metrics, "loss/dp_pose_mse", pose_mse)

        event = _get_event_tensor(
            batch,
            weighting_cfg.gripper_event_column,
            shape=(batch_size, horizon),
            device=mse_per_dim.device,
        )
        if event is not None:
            closing_mse = _masked_mean(mse_per_dim, (event == 2) & valid_horizon_mask)
            opening_mse = _masked_mean(mse_per_dim, (event == 5) & valid_horizon_mask)
            _add_metric(metrics, "loss/closing_mse", closing_mse)
            _add_metric(metrics, "loss/dp_closing_mse", closing_mse)
            _add_metric(metrics, "loss/opening_mse", opening_mse)
            _add_metric(metrics, "loss/dp_opening_mse", opening_mse)

    return loss, metrics


class DiffusionPolicy(PreTrainedPolicy):
    """
    Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
    (paper: https://huggingface.co/papers/2303.04137, code: https://github.com/real-stanford/diffusion_policy).
    """

    config_class = DiffusionConfig
    name = "diffusion"

    def __init__(
        self,
        config: DiffusionConfig,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None

        self.diffusion = DiffusionModel(config)

        self.reset()

    def get_optim_params(self) -> dict:
        return self.diffusion.parameters()

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        # stack n latest observations from the queue
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        actions = self.diffusion.generate_actions(batch, noise=noise)

        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations.

        This method handles caching a history of observations and an action trajectory generated by the
        underlying diffusion model. Here's how it works:
          - `n_obs_steps` steps worth of observations are cached (for the first steps, the observation is
            copied `n_obs_steps` times to fill the cache).
          - The diffusion model generates `horizon` steps worth of actions.
          - `n_action_steps` worth of actions are actually kept for execution, starting from the current step.
        Schematically this looks like:
            ----------------------------------------------------------------------------------------------
            (legend: o = n_obs_steps, h = horizon, a = n_action_steps)
            |timestep            | n-o+1 | n-o+2 | ..... | n     | ..... | n+a-1 | n+a   | ..... | n-o+h |
            |observation is used | YES   | YES   | YES   | YES   | NO    | NO    | NO    | NO    | NO    |
            |action is generated | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   |
            |action is used      | NO    | NO    | NO    | YES   | YES   | YES   | NO    | NO    | NO    |
            ----------------------------------------------------------------------------------------------
        Note that this means we require: `n_action_steps <= horizon - n_obs_steps + 1`. Also, note that
        "horizon" may not the best name to describe what the variable actually means, because this period is
        actually measured from the first observation which (if `n_obs_steps` > 1) happened in the past.
        """
        # NOTE: for offline evaluation, we have action in the batch, so we need to pop it out
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        # NOTE: It's important that this happens after stacking the images into a single key.
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        loss, loss_breakdown = self.diffusion.compute_loss_and_metrics(batch)
        loss_dict = {key: value.item() for key, value in loss_breakdown.items()} if loss_breakdown else None
        return loss, loss_dict


def _make_noise_scheduler(name: str, **kwargs: dict) -> DDPMScheduler | DDIMScheduler:
    """
    Factory for noise scheduler instances of the requested type. All kwargs are passed
    to the scheduler.
    """
    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    elif name == "DDIM":
        return DDIMScheduler(**kwargs)
    else:
        raise ValueError(f"Unsupported noise scheduler type {name}")


class DiffusionModel(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config

        # Build observation encoders (depending on which observations are provided).
        global_cond_dim = self.config.robot_state_feature.shape[0]
        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]

        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim * config.n_obs_steps)

        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )

        if config.num_inference_steps is None:
            self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        else:
            self.num_inference_steps = config.num_inference_steps

    # ========= inference  ============
    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        # Sample prior.
        sample = (
            noise
            if noise is not None
            else torch.randn(
                size=(batch_size, self.config.horizon, self.config.action_feature.shape[0]),
                dtype=dtype,
                device=device,
                generator=generator,
            )
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            # Predict model output.
            model_output = self.unet(
                sample,
                torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
                global_cond=global_cond,
            )
            # Compute previous image: x_t -> x_t-1
            sample = self.noise_scheduler.step(model_output, t, sample, generator=generator).prev_sample

        return sample

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode image features and concatenate them all together along with the state vector."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        global_cond_feats = [batch[OBS_STATE]]
        # Extract image features.
        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                # Combine batch and sequence dims while rearranging to make the camera index dimension first.
                images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
                img_features_list = torch.cat(
                    [
                        encoder(images)
                        for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
                    ]
                )
                # Separate batch and sequence dims back out. The camera index dim gets absorbed into the
                # feature dim (effectively concatenating the camera features).
                img_features = einops.rearrange(
                    img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            else:
                # Combine batch, sequence, and "which camera" dims before passing to shared encoder.
                img_features = self.rgb_encoder(
                    einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
                )
                # Separate batch dim and sequence dim back out. The camera index dim gets absorbed into the
                # feature dim (effectively concatenating the camera features).
                img_features = einops.rearrange(
                    img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        # Concatenate features then flatten to (B, global_cond_dim).
        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

    def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """
        This function expects `batch` to have:
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, n_obs_steps, environment_dim)
        }
        """
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # run sampling
        actions = self.conditional_sample(batch_size, global_cond=global_cond, noise=noise)

        # Extract `n_action_steps` steps worth of actions (from the current observation).
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]

        return actions

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        loss, _ = self.compute_loss_and_metrics(batch)
        return loss

    def compute_loss_and_metrics(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        """
        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, n_obs_steps, environment_dim)

            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon) when do_mask_loss_for_padding is true
        }
        """
        # Input validation.
        assert set(batch).issuperset({OBS_STATE, ACTION})
        assert OBS_IMAGES in batch or OBS_ENV_STATE in batch
        n_obs_steps = batch[OBS_STATE].shape[1]
        horizon = batch[ACTION].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # Forward diffusion.
        trajectory = batch[ACTION]
        # Sample noise to add to the trajectory.
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        # Sample a random noising timestep for each item in the batch.
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        # Add noise to the clean trajectories according to the noise magnitude at each timestep.
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)

        # Run the denoising network (that might denoise the trajectory, or attempt to predict the noise).
        pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

        # Compute the loss.
        # The target is either the original trajectory, or the noise.
        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = batch[ACTION]
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")

        action_is_pad = batch.get("action_is_pad")
        if self.config.loss_weighting.enabled:
            loss, loss_breakdown = compute_weighted_denoising_mse_loss(
                pred,
                target,
                action_is_pad,
                batch,
                self.config,
                action_dim_names=getattr(self.config, "_action_feature_names", None),
            )
            loss_dict = {key: value.detach().item() for key, value in loss_breakdown.items()}
            weighted_mse = loss.detach().item()
            loss_dict.setdefault("loss/denoising_mse_weighted", weighted_mse)
            loss_dict.setdefault("loss/dp_denoising_mse_weighted", weighted_mse)
            loss_dict["loss/total"] = weighted_mse
            return loss, loss_dict

        loss = compute_unweighted_denoising_mse_loss(pred, target, action_is_pad, self.config)
        return loss, {"loss/total": loss.detach().item()}


class SpatialSoftmax(nn.Module):
    """
    Spatial Soft Argmax operation described in "Deep Spatial Autoencoders for Visuomotor Learning" by Finn et al.
    (https://huggingface.co/papers/1509.06113). A minimal port of the robomimic implementation.

    At a high level, this takes 2D feature maps (from a convnet/ViT) and returns the "center of mass"
    of activations of each channel, i.e., keypoints in the image space for the policy to focus on.

    Example: take feature maps of size (512x10x12). We generate a grid of normalized coordinates (10x12x2):
    -----------------------------------------------------
    | (-1., -1.)   | (-0.82, -1.)   | ... | (1., -1.)   |
    | (-1., -0.78) | (-0.82, -0.78) | ... | (1., -0.78) |
    | ...          | ...            | ... | ...         |
    | (-1., 1.)    | (-0.82, 1.)    | ... | (1., 1.)    |
    -----------------------------------------------------
    This is achieved by applying channel-wise softmax over the activations (512x120) and computing the dot
    product with the coordinates (120x2) to get expected points of maximal activation (512x2).

    The example above results in 512 keypoints (corresponding to the 512 input channels). We can optionally
    provide num_kp != None to control the number of keypoints. This is achieved by a first applying a learnable
    linear mapping (in_channels, H, W) -> (num_kp, H, W).
    """

    def __init__(self, input_shape, num_kp=None):
        """
        Args:
            input_shape (list): (C, H, W) input feature map shape.
            num_kp (int): number of keypoints in output. If None, output will have the same number of channels as input.
        """
        super().__init__()

        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        # we could use torch.linspace directly but that seems to behave slightly differently than numpy
        # and causes a small degradation in pc_success of pre-trained models.
        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        # register as buffer so it's moved to the correct device.
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: (B, C, H, W) input feature maps.
        Returns:
            (B, K, 2) image-space coordinates of keypoints.
        """
        if self.nets is not None:
            features = self.nets(features)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        features = features.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(features, dim=-1)
        # [B * K, H * W] x [H * W, 2] -> [B * K, 2] for spatial coordinate mean in x and y dimensions
        expected_xy = attention @ self.pos_grid
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)

        return feature_keypoints


class DiffusionRgbEncoder(nn.Module):
    """Encodes an RGB image into a 1D feature vector.

    Includes the ability to normalize and crop the image first.
    """

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        # Set up optional preprocessing.
        if config.crop_shape is not None:
            self.do_crop = True
            # Always use center crop for eval
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        # Set up backbone.
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        # Note: This assumes that the layer4 feature map is children()[-3]
        # TODO(alexander-soare): Use a safer alternative.
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError(
                    "You can't replace BatchNorm in a pretrained model without ruining the weights!"
                )
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        # Set up pooling and final layers.
        # Use a dry run to get the feature map shape.
        # The dummy input should take the number of image channels from `config.image_features` and it should
        # use the height and width from `config.crop_shape` if it is provided, otherwise it should use the
        # height and width from `config.image_features`.

        # Note: we have a check in the config class to make sure all images have the same shape.
        images_shape = next(iter(config.image_features.values())).shape
        dummy_shape_h_w = config.crop_shape if config.crop_shape is not None else images_shape[1:]
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].
        Returns:
            (B, D) image feature.
        """
        # Preprocess: maybe crop (if it was set up in the __init__).
        if self.do_crop:
            if self.training:  # noqa: SIM108
                x = self.maybe_random_crop(x)
            else:
                # Always use center crop for eval.
                x = self.center_crop(x)
        # Extract backbone feature.
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        # Final linear layer with non-linearity.
        x = self.relu(self.out(x))
        return x


def _replace_submodules(
    root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]
) -> nn.Module:
    """
    Args:
        root_module: The module for which the submodules need to be replaced
        predicate: Takes a module as an argument and must return True if the that module is to be replaced.
        func: Takes a module as an argument and returns a new module to replace it with.
    Returns:
        The root module with its submodules replaced.
    """
    if predicate(root_module):
        return func(root_module)

    replace_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    assert not any(predicate(m) for _, m in root_module.named_modules(remove_duplicate=True))
    return root_module


class DiffusionSinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embeddings as in Attention is All You Need."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiffusionConv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish"""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class DiffusionConditionalUnet1d(nn.Module):
    """A 1D convolutional UNet with FiLM modulation for conditioning.

    Note: this removes local conditioning as compared to the original diffusion policy code.
    """

    def __init__(self, config: DiffusionConfig, global_cond_dim: int):
        super().__init__()

        self.config = config

        # Encoder for the diffusion timestep.
        self.diffusion_step_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        )

        # The FiLM conditioning dimension.
        cond_dim = config.diffusion_step_embed_dim + global_cond_dim

        # In channels / out channels for each downsampling block in the Unet's encoder. For the decoder, we
        # just reverse these.
        in_out = [(config.action_feature.shape[0], config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:], strict=True)
        )

        # Unet encoder.
        common_res_block_kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
            "use_film_scale_modulation": config.use_film_scale_modulation,
        }
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        DiffusionConditionalResidualBlock1d(dim_in, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # Downsample as long as it is not the last block.
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Processing in the middle of the auto-encoder.
        self.mid_modules = nn.ModuleList(
            [
                DiffusionConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
                DiffusionConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
            ]
        )

        # Unet decoder.
        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        # dim_in * 2, because it takes the encoder's skip connection as well
                        DiffusionConditionalResidualBlock1d(dim_in * 2, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # Upsample as long as it is not the last block.
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            DiffusionConv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size),
            nn.Conv1d(config.down_dims[0], config.action_feature.shape[0], 1),
        )

    def forward(self, x: Tensor, timestep: Tensor | int, global_cond=None) -> Tensor:
        """
        Args:
            x: (B, T, input_dim) tensor for input to the Unet.
            timestep: (B,) tensor of (timestep_we_are_denoising_from - 1).
            global_cond: (B, global_cond_dim)
            output: (B, T, input_dim)
        Returns:
            (B, T, input_dim) diffusion model prediction.
        """
        # For 1D convolutions we'll need feature dimension first.
        x = einops.rearrange(x, "b t d -> b d t")

        timesteps_embed = self.diffusion_step_encoder(timestep)

        # If there is a global conditioning feature, concatenate it to the timestep embedding.
        if global_cond is not None:
            global_feature = torch.cat([timesteps_embed, global_cond], axis=-1)
        else:
            global_feature = timesteps_embed

        # Run encoder, keeping track of skip features to pass to the decoder.
        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # Run decoder, using the skip features from the encoder.
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, "b d t -> b t d")
        return x


class DiffusionConditionalResidualBlock1d(nn.Module):
    """ResNet style 1D convolutional block with FiLM modulation for conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        # Set to True to do scale modulation with FiLM as well as bias modulation (defaults to False meaning
        # FiLM just modulates bias).
        use_film_scale_modulation: bool = False,
    ):
        super().__init__()

        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_channels = out_channels

        self.conv1 = DiffusionConv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)

        # FiLM modulation (https://huggingface.co/papers/1709.07871) outputs per-channel bias and (maybe) scale.
        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))

        self.conv2 = DiffusionConv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)

        # A final convolution for dimension matching the residual (if needed).
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x: (B, in_channels, T)
            cond: (B, cond_dim)
        Returns:
            (B, out_channels, T)
        """
        out = self.conv1(x)

        # Get condition embedding. Unsqueeze for broadcasting to `out`, resulting in (B, out_channels, 1).
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            # Treat the embedding as a list of scales and biases.
            scale = cond_embed[:, : self.out_channels]
            bias = cond_embed[:, self.out_channels :]
            out = scale * out + bias
        else:
            # Treat the embedding as biases.
            out = out + cond_embed

        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return out

#!/usr/bin/env python

"""Tensor shape utilities for the from-scratch Diffusion Policy.

This module deliberately stays independent from model code. Future phases can call these helpers before
SpatialSoftmax, RGB encoders, U-Nets, losses, or action queues so tensor contract failures are reported at
the boundary instead of being hidden inside model internals.
"""

from collections.abc import Iterable

import torch

from lerobot.policies.diffusion_policy.configuration_diffusion_policy import DiffusionPolicyConfig
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

ACTION_IS_PAD = "action_is_pad"


def get_image_feature_keys(config: DiffusionPolicyConfig) -> list[str]:
    """Return configured visual input keys in the same order as LeRobot's feature schema."""

    return list(config.image_features)


def stack_image_features(
    batch: dict[str, torch.Tensor],
    image_feature_keys: Iterable[str],
    *,
    output_key: str = OBS_IMAGES,
) -> dict[str, torch.Tensor]:
    """Stack separate camera image tensors into the shared `observation.images` key.

    This mirrors the legacy Diffusion Policy convention:
    `torch.stack([batch[key] for key in config.image_features], dim=-4)`.
    For training tensors, `[B, T, C, H, W]` becomes `[B, T, N, C, H, W]`.
    For single-step inference tensors, `[B, C, H, W]` becomes `[B, N, C, H, W]`.
    """

    image_feature_keys = list(image_feature_keys)
    if len(image_feature_keys) == 0:
        raise ValueError("`image_feature_keys` must contain at least one image feature key.")

    tensors: list[torch.Tensor] = []
    reference_key = image_feature_keys[0]
    reference_shape: tuple[int, ...] | None = None

    for key in image_feature_keys:
        tensor = _require_tensor(batch, key)

        if tensor.ndim not in {4, 5}:
            raise ValueError(
                f"Image feature `{key}` must be a single-step tensor [B, C, H, W] or a training "
                f"tensor [B, T, C, H, W]. Got shape {tuple(tensor.shape)}."
            )

        if reference_shape is None:
            reference_shape = tuple(tensor.shape)
        elif tuple(tensor.shape) != reference_shape:
            raise ValueError(
                f"Image feature `{key}` shape {tuple(tensor.shape)} does not match "
                f"`{reference_key}` shape {reference_shape}."
            )

        tensors.append(tensor)

    stacked_batch = dict(batch)
    stacked_batch[output_key] = torch.stack(tensors, dim=-4)
    return stacked_batch


def validate_training_batch_contract(
    batch: dict[str, torch.Tensor],
    config: DiffusionPolicyConfig,
) -> None:
    """Validate the post-stacking training batch contract expected by future model phases."""

    state = _require_tensor(batch, OBS_STATE)
    action = _require_tensor(batch, ACTION)

    if OBS_IMAGES not in batch and OBS_ENV_STATE not in batch:
        raise ValueError(
            f"Training batch must contain at least one conditioning input: `{OBS_IMAGES}` or "
            f"`{OBS_ENV_STATE}`."
        )

    _expect_ndim(OBS_STATE, state, 3, "[B, n_obs_steps, state_dim]")
    batch_size, n_obs_steps, state_dim = state.shape
    _expect_size(OBS_STATE, "n_obs_steps", n_obs_steps, config.n_obs_steps)

    # Some focused tests may construct config objects without full feature metadata. In that case, keep the
    # contract useful by validating rank, batch size, and temporal dimensions, and only validate feature dims
    # when the config tells us what those dims should be.
    expected_state_dim = _feature_dim(config.robot_state_feature)
    if expected_state_dim is not None:
        _expect_size(OBS_STATE, "state_dim", state_dim, expected_state_dim)

    _expect_ndim(ACTION, action, 3, "[B, horizon, action_dim]")
    _expect_size(ACTION, "batch size", action.shape[0], batch_size)
    _expect_size(ACTION, "horizon", action.shape[1], config.horizon)

    expected_action_dim = _feature_dim(config.action_feature)
    if expected_action_dim is not None:
        _expect_size(ACTION, "action_dim", action.shape[2], expected_action_dim)

    if OBS_IMAGES in batch:
        images = _require_tensor(batch, OBS_IMAGES)
        _expect_ndim(OBS_IMAGES, images, 6, "[B, n_obs_steps, num_cameras, C, H, W]")
        _expect_size(OBS_IMAGES, "batch size", images.shape[0], batch_size)
        _expect_size(OBS_IMAGES, "n_obs_steps", images.shape[1], config.n_obs_steps)

        image_feature_keys = get_image_feature_keys(config)
        if len(image_feature_keys) > 0:
            _expect_size(OBS_IMAGES, "num_cameras", images.shape[2], len(image_feature_keys))

        expected_image_shape = _first_image_shape(config)
        if expected_image_shape is not None:
            observed_image_shape = tuple(images.shape[-3:])
            if observed_image_shape != expected_image_shape:
                raise ValueError(
                    f"`{OBS_IMAGES}` image shape must be {expected_image_shape}. "
                    f"Got {observed_image_shape} from full shape {tuple(images.shape)}."
                )

    if OBS_ENV_STATE in batch:
        env_state = _require_tensor(batch, OBS_ENV_STATE)
        _expect_ndim(OBS_ENV_STATE, env_state, 3, "[B, n_obs_steps, env_dim]")
        _expect_size(OBS_ENV_STATE, "batch size", env_state.shape[0], batch_size)
        _expect_size(OBS_ENV_STATE, "n_obs_steps", env_state.shape[1], config.n_obs_steps)

        expected_env_dim = _feature_dim(config.env_state_feature)
        if expected_env_dim is not None:
            _expect_size(OBS_ENV_STATE, "env_dim", env_state.shape[2], expected_env_dim)

    if ACTION_IS_PAD in batch:
        action_is_pad = _require_tensor(batch, ACTION_IS_PAD)
        _expect_ndim(ACTION_IS_PAD, action_is_pad, 2, "[B, horizon]")
        _expect_size(ACTION_IS_PAD, "batch size", action_is_pad.shape[0], batch_size)
        _expect_size(ACTION_IS_PAD, "horizon", action_is_pad.shape[1], config.horizon)


def validate_single_step_observation_contract(
    batch: dict[str, torch.Tensor],
    config: DiffusionPolicyConfig,
) -> None:
    """Validate a LeRobot-style single-step inference observation before queue stacking."""

    state = _require_tensor(batch, OBS_STATE)
    _expect_ndim(OBS_STATE, state, 2, "[B, state_dim]")
    batch_size, state_dim = state.shape

    expected_state_dim = _feature_dim(config.robot_state_feature)
    if expected_state_dim is not None:
        _expect_size(OBS_STATE, "state_dim", state_dim, expected_state_dim)

    image_feature_keys = get_image_feature_keys(config)
    provided_image_keys = [key for key in image_feature_keys if key in batch]
    has_env_state = OBS_ENV_STATE in batch

    if len(provided_image_keys) == 0 and not has_env_state:
        raise ValueError(
            "Single-step observation must contain at least one configured image feature key or "
            f"`{OBS_ENV_STATE}`."
        )

    if provided_image_keys:
        missing_image_keys = [key for key in image_feature_keys if key not in batch]
        if missing_image_keys:
            raise KeyError(
                "Missing image feature key(s) in single-step observation: "
                f"{', '.join(f'`{key}`' for key in missing_image_keys)}."
            )

        reference_key = image_feature_keys[0]
        reference_shape: tuple[int, ...] | None = None
        for key in image_feature_keys:
            image = _require_tensor(batch, key)
            _expect_ndim(key, image, 4, "[B, C, H, W]")
            _expect_size(key, "batch size", image.shape[0], batch_size)

            if reference_shape is None:
                reference_shape = tuple(image.shape)
            elif tuple(image.shape) != reference_shape:
                raise ValueError(
                    f"Image feature `{key}` shape {tuple(image.shape)} does not match "
                    f"`{reference_key}` shape {reference_shape}."
                )

            expected_image_shape = _image_shape_for_key(config, key)
            if expected_image_shape is not None:
                observed_image_shape = tuple(image.shape[-3:])
                if observed_image_shape != expected_image_shape:
                    raise ValueError(
                        f"Image feature `{key}` shape must end with {expected_image_shape}. "
                        f"Got {observed_image_shape} from full shape {tuple(image.shape)}."
                    )

    if has_env_state:
        env_state = _require_tensor(batch, OBS_ENV_STATE)
        _expect_ndim(OBS_ENV_STATE, env_state, 2, "[B, env_dim]")
        _expect_size(OBS_ENV_STATE, "batch size", env_state.shape[0], batch_size)

        expected_env_dim = _feature_dim(config.env_state_feature)
        if expected_env_dim is not None:
            _expect_size(OBS_ENV_STATE, "env_dim", env_state.shape[1], expected_env_dim)


def _require_tensor(batch: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    if key not in batch:
        raise KeyError(f"Missing required tensor key `{key}`.")

    tensor = batch[key]
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"`{key}` must be a torch.Tensor. Got {type(tensor).__name__}.")
    return tensor


def _expect_ndim(name: str, tensor: torch.Tensor, expected_ndim: int, expected_shape: str) -> None:
    if tensor.ndim != expected_ndim:
        raise ValueError(
            f"`{name}` must have shape {expected_shape} with ndim={expected_ndim}. "
            f"Got shape {tuple(tensor.shape)}."
        )


def _expect_size(name: str, dimension_name: str, observed: int, expected: int) -> None:
    if observed != expected:
        raise ValueError(
            f"`{name}` has wrong {dimension_name}: got {observed}, expected {expected}."
        )


def _feature_dim(feature: object | None) -> int | None:
    shape = getattr(feature, "shape", None)
    if shape is None or len(shape) == 0:
        return None
    return int(shape[0])


def _first_image_shape(config: DiffusionPolicyConfig) -> tuple[int, int, int] | None:
    for feature in config.image_features.values():
        shape = getattr(feature, "shape", None)
        if shape is not None and len(shape) == 3:
            return tuple(int(dim) for dim in shape)
    return None


def _image_shape_for_key(config: DiffusionPolicyConfig, key: str) -> tuple[int, int, int] | None:
    feature = config.image_features.get(key)
    shape = getattr(feature, "shape", None)
    if shape is None or len(shape) != 3:
        return None
    return tuple(int(dim) for dim in shape)

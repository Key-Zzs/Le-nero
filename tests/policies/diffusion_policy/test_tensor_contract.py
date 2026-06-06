#!/usr/bin/env python

"""Scoped tests for the Diffusion Policy tensor contract utilities."""

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion_policy.configuration_diffusion_policy import DiffusionPolicyConfig
from lerobot.policies.diffusion_policy.tensor_contract import (
    stack_image_features,
    validate_single_step_observation_contract,
    validate_training_batch_contract,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

ACTION_IS_PAD = "action_is_pad"
CAM0 = f"{OBS_IMAGES}.cam0"
CAM1 = f"{OBS_IMAGES}.cam1"


def state_feature(shape: tuple[int, ...] = (5,)) -> PolicyFeature:
    return PolicyFeature(type=FeatureType.STATE, shape=shape)


def visual_feature(shape: tuple[int, ...] = (3, 8, 8)) -> PolicyFeature:
    return PolicyFeature(type=FeatureType.VISUAL, shape=shape)


def env_feature(shape: tuple[int, ...] = (7,)) -> PolicyFeature:
    return PolicyFeature(type=FeatureType.ENV, shape=shape)


def action_feature(shape: tuple[int, ...] = (4,)) -> PolicyFeature:
    return PolicyFeature(type=FeatureType.ACTION, shape=shape)


def make_config(
    *,
    image_keys: tuple[str, ...] = (CAM0, CAM1),
    include_env: bool = False,
    state_dim: int = 5,
    env_dim: int = 7,
    action_dim: int = 4,
) -> DiffusionPolicyConfig:
    input_features = {OBS_STATE: state_feature((state_dim,))}
    input_features.update({key: visual_feature() for key in image_keys})

    if include_env:
        input_features[OBS_ENV_STATE] = env_feature((env_dim,))

    return DiffusionPolicyConfig(
        n_obs_steps=2,
        horizon=16,
        n_action_steps=8,
        input_features=input_features,
        output_features={ACTION: action_feature((action_dim,))},
        device="cpu",
    )


def make_valid_training_batch(
    config: DiffusionPolicyConfig,
    *,
    batch_size: int = 3,
    include_images: bool = True,
    include_env: bool = False,
    include_action_is_pad: bool = False,
) -> dict[str, torch.Tensor]:
    state_dim = config.robot_state_feature.shape[0]
    action_dim = config.action_feature.shape[0]
    batch = {
        OBS_STATE: torch.randn(batch_size, config.n_obs_steps, state_dim),
        ACTION: torch.randn(batch_size, config.horizon, action_dim),
    }

    if include_images:
        num_cameras = len(config.image_features)
        channels, height, width = next(iter(config.image_features.values())).shape
        batch[OBS_IMAGES] = torch.randn(
            batch_size,
            config.n_obs_steps,
            num_cameras,
            channels,
            height,
            width,
        )

    if include_env:
        env_dim = config.env_state_feature.shape[0]
        batch[OBS_ENV_STATE] = torch.randn(batch_size, config.n_obs_steps, env_dim)

    if include_action_is_pad:
        batch[ACTION_IS_PAD] = torch.zeros(batch_size, config.horizon, dtype=torch.bool)

    return batch


def test_stack_image_features_for_training_batch_uses_legacy_camera_dimension():
    batch_size, n_obs_steps, channels, height, width = 2, 3, 3, 8, 8
    batch = {
        CAM0: torch.zeros(batch_size, n_obs_steps, channels, height, width),
        CAM1: torch.ones(batch_size, n_obs_steps, channels, height, width),
    }

    stacked = stack_image_features(batch, [CAM0, CAM1])

    assert stacked is not batch
    assert OBS_IMAGES not in batch
    assert stacked[CAM0] is batch[CAM0]
    assert stacked[OBS_IMAGES].shape == (batch_size, n_obs_steps, 2, channels, height, width)
    torch.testing.assert_close(stacked[OBS_IMAGES][:, :, 0], batch[CAM0])
    torch.testing.assert_close(stacked[OBS_IMAGES][:, :, 1], batch[CAM1])


def test_stack_image_features_for_single_step_inference_uses_legacy_camera_dimension():
    batch_size, channels, height, width = 2, 3, 8, 8
    batch = {
        CAM0: torch.zeros(batch_size, channels, height, width),
        CAM1: torch.ones(batch_size, channels, height, width),
    }

    stacked = stack_image_features(batch, [CAM0, CAM1])

    assert stacked[OBS_IMAGES].shape == (batch_size, 2, channels, height, width)
    torch.testing.assert_close(stacked[OBS_IMAGES][:, 0], batch[CAM0])
    torch.testing.assert_close(stacked[OBS_IMAGES][:, 1], batch[CAM1])


def test_stack_image_features_raises_for_missing_image_key():
    batch = {CAM0: torch.zeros(2, 3, 8, 8)}

    with pytest.raises(KeyError, match="cam1"):
        stack_image_features(batch, [CAM0, CAM1])


def test_stack_image_features_raises_for_mismatched_image_shapes():
    batch = {
        CAM0: torch.zeros(2, 3, 8, 8),
        CAM1: torch.zeros(2, 3, 10, 8),
    }

    with pytest.raises(ValueError, match="does not match"):
        stack_image_features(batch, [CAM0, CAM1])


def test_validate_training_batch_contract_accepts_valid_image_only_batch():
    config = make_config()
    batch = make_valid_training_batch(config, include_images=True, include_action_is_pad=True)

    validate_training_batch_contract(batch, config)


def test_validate_training_batch_contract_accepts_valid_env_state_only_batch():
    config = make_config(image_keys=(), include_env=True)
    batch = make_valid_training_batch(config, include_images=False, include_env=True)

    validate_training_batch_contract(batch, config)


def test_validate_training_batch_contract_rejects_missing_state():
    config = make_config()
    batch = make_valid_training_batch(config)
    batch.pop(OBS_STATE)

    with pytest.raises(KeyError, match="observation.state"):
        validate_training_batch_contract(batch, config)


def test_validate_training_batch_contract_rejects_missing_action():
    config = make_config()
    batch = make_valid_training_batch(config)
    batch.pop(ACTION)

    with pytest.raises(KeyError, match=ACTION):
        validate_training_batch_contract(batch, config)


def test_validate_training_batch_contract_rejects_missing_images_and_env_state():
    config = make_config()
    batch = make_valid_training_batch(config, include_images=False)

    with pytest.raises(ValueError, match="at least one conditioning input"):
        validate_training_batch_contract(batch, config)


def test_validate_training_batch_contract_rejects_wrong_n_obs_steps():
    config = make_config()
    batch = make_valid_training_batch(config)
    batch[OBS_STATE] = torch.randn(3, config.n_obs_steps + 1, config.robot_state_feature.shape[0])

    with pytest.raises(ValueError, match="n_obs_steps"):
        validate_training_batch_contract(batch, config)


def test_validate_training_batch_contract_rejects_wrong_horizon():
    config = make_config()
    batch = make_valid_training_batch(config)
    batch[ACTION] = torch.randn(3, config.horizon - 1, config.action_feature.shape[0])

    with pytest.raises(ValueError, match="horizon"):
        validate_training_batch_contract(batch, config)


def test_validate_training_batch_contract_rejects_wrong_action_is_pad_shape():
    config = make_config()
    batch = make_valid_training_batch(config, include_action_is_pad=True)
    batch[ACTION_IS_PAD] = torch.zeros(3, config.horizon, 1, dtype=torch.bool)

    with pytest.raises(ValueError, match="action_is_pad"):
        validate_training_batch_contract(batch, config)


def test_validate_training_batch_contract_rejects_inconsistent_batch_size():
    config = make_config()
    batch = make_valid_training_batch(config)
    batch[OBS_IMAGES] = torch.randn(
        4,
        config.n_obs_steps,
        len(config.image_features),
        3,
        8,
        8,
    )

    with pytest.raises(ValueError, match="batch size"):
        validate_training_batch_contract(batch, config)


def test_validate_single_step_observation_contract_accepts_valid_image_observation():
    config = make_config()
    batch = {
        OBS_STATE: torch.randn(3, config.robot_state_feature.shape[0]),
        CAM0: torch.randn(3, 3, 8, 8),
        CAM1: torch.randn(3, 3, 8, 8),
    }

    validate_single_step_observation_contract(batch, config)


def test_validate_single_step_observation_contract_rejects_missing_state():
    config = make_config()
    batch = {
        CAM0: torch.randn(3, 3, 8, 8),
        CAM1: torch.randn(3, 3, 8, 8),
    }

    with pytest.raises(KeyError, match="observation.state"):
        validate_single_step_observation_contract(batch, config)


def test_validate_single_step_observation_contract_rejects_temporal_state():
    config = make_config()
    batch = {
        OBS_STATE: torch.randn(3, config.n_obs_steps, config.robot_state_feature.shape[0]),
        CAM0: torch.randn(3, 3, 8, 8),
        CAM1: torch.randn(3, 3, 8, 8),
    }

    with pytest.raises(ValueError, match="state_dim"):
        validate_single_step_observation_contract(batch, config)


def test_validate_single_step_observation_contract_rejects_temporal_image():
    config = make_config()
    batch = {
        OBS_STATE: torch.randn(3, config.robot_state_feature.shape[0]),
        CAM0: torch.randn(3, config.n_obs_steps, 3, 8, 8),
        CAM1: torch.randn(3, config.n_obs_steps, 3, 8, 8),
    }

    with pytest.raises(ValueError, match="\\[B, C, H, W\\]"):
        validate_single_step_observation_contract(batch, config)


def test_validate_single_step_observation_contract_rejects_missing_images_and_env_state():
    config = make_config()
    batch = {OBS_STATE: torch.randn(3, config.robot_state_feature.shape[0])}

    with pytest.raises(ValueError, match="at least one configured image feature key"):
        validate_single_step_observation_contract(batch, config)

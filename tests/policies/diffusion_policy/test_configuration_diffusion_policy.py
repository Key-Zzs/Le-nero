#!/usr/bin/env python

import pytest

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig
from lerobot.policies.diffusion_policy.configuration_diffusion_policy import DiffusionPolicyConfig
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGE, OBS_STATE


def action_features(shape: tuple[int, ...] = (6,)) -> dict[str, PolicyFeature]:
    return {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=shape)}


def visual_feature(shape: tuple[int, ...] = (3, 96, 96)) -> PolicyFeature:
    return PolicyFeature(type=FeatureType.VISUAL, shape=shape)


def env_feature(shape: tuple[int, ...] = (10,)) -> PolicyFeature:
    return PolicyFeature(type=FeatureType.ENV, shape=shape)


def test_default_initialization_matches_minimal_baseline():
    config = DiffusionPolicyConfig()

    assert config.type == "diffusion_policy"
    assert config.n_obs_steps == 2
    assert config.horizon == 16
    assert config.n_action_steps == 8
    assert config.noise_scheduler_type == "DDPM"
    assert config.prediction_type == "epsilon"
    assert config.normalization_mapping == {
        "VISUAL": NormalizationMode.MEAN_STD,
        "STATE": NormalizationMode.MIN_MAX,
        "ACTION": NormalizationMode.MIN_MAX,
    }


def test_temporal_delta_indices_match_diffusion_policy_contract():
    config = DiffusionPolicyConfig()

    assert config.observation_delta_indices == [-1, 0]
    assert config.action_delta_indices == list(range(-1, 15))
    assert config.reward_delta_indices is None


def test_horizon_must_match_unet_downsampling_factor():
    with pytest.raises(ValueError, match="downsampling factor"):
        DiffusionPolicyConfig(horizon=15, down_dims=(512, 1024, 2048))


def test_n_action_steps_must_fit_receding_horizon_window():
    with pytest.raises(ValueError, match="horizon - n_obs_steps"):
        DiffusionPolicyConfig(n_obs_steps=2, horizon=16, n_action_steps=16)


@pytest.mark.parametrize(
    "kwargs, error_match",
    [
        ({"n_obs_steps": 0}, "n_obs_steps"),
        ({"horizon": 0}, "horizon"),
        ({"n_action_steps": 0}, "n_action_steps"),
    ],
)
def test_temporal_lengths_must_be_positive(kwargs, error_match):
    with pytest.raises(ValueError, match=error_match):
        DiffusionPolicyConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs, error_match",
    [
        ({"noise_scheduler_type": "BAD"}, "noise_scheduler_type"),
        ({"prediction_type": "BAD"}, "prediction_type"),
        ({"vision_backbone": "vit_base_patch16_224"}, "vision_backbone"),
    ],
)
def test_scheduler_prediction_and_backbone_validation(kwargs, error_match):
    with pytest.raises(ValueError, match=error_match):
        DiffusionPolicyConfig(**kwargs)


def test_image_only_features_are_valid():
    config = DiffusionPolicyConfig(
        input_features={f"{OBS_IMAGE}.front": visual_feature()},
        output_features=action_features(),
    )

    config.validate_features()


def test_environment_state_only_features_are_valid():
    config = DiffusionPolicyConfig(
        input_features={OBS_ENV_STATE: env_feature()},
        output_features=action_features(),
    )

    config.validate_features()


def test_requires_visual_or_environment_state_input():
    config = DiffusionPolicyConfig(
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,))},
        output_features=action_features(),
    )

    with pytest.raises(ValueError, match="at least one image or environment-state"):
        config.validate_features()


def test_requires_action_output_feature():
    config = DiffusionPolicyConfig(
        input_features={f"{OBS_IMAGE}.front": visual_feature()},
        output_features={},
    )

    with pytest.raises(ValueError, match=ACTION):
        config.validate_features()


def test_multi_camera_image_shapes_must_match():
    config = DiffusionPolicyConfig(
        input_features={
            f"{OBS_IMAGE}.front": visual_feature((3, 96, 96)),
            f"{OBS_IMAGE}.wrist": visual_feature((3, 128, 96)),
        },
        output_features=action_features(),
    )

    with pytest.raises(ValueError, match="all image shapes must match"):
        config.validate_features()


def test_crop_shape_must_fit_all_image_features():
    config = DiffusionPolicyConfig(
        crop_shape=(128, 84),
        input_features={f"{OBS_IMAGE}.front": visual_feature((3, 96, 96))},
        output_features=action_features(),
    )

    with pytest.raises(ValueError, match="crop_shape"):
        config.validate_features()


def test_optimizer_and_scheduler_presets_preserve_config_values():
    config = DiffusionPolicyConfig(optimizer_lr=3e-4, scheduler_warmup_steps=123)

    optimizer = config.get_optimizer_preset()
    scheduler = config.get_scheduler_preset()

    assert isinstance(optimizer, AdamConfig)
    assert optimizer.lr == 3e-4
    assert optimizer.betas == config.optimizer_betas
    assert optimizer.eps == config.optimizer_eps
    assert optimizer.weight_decay == config.optimizer_weight_decay

    assert isinstance(scheduler, DiffuserSchedulerConfig)
    assert scheduler.name == config.scheduler_name
    assert scheduler.num_warmup_steps == 123

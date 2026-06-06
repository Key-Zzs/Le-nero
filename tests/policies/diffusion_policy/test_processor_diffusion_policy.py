#!/usr/bin/env python

"""Scoped tests for the from-scratch Diffusion Policy processor factory."""

import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.diffusion_policy.configuration_diffusion_policy import DiffusionPolicyConfig
from lerobot.policies.diffusion_policy.processor_diffusion_policy import (
    make_diffusion_policy_pre_post_processors,
)
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGE,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


def create_default_config() -> DiffusionPolicyConfig:
    config = DiffusionPolicyConfig()
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 2, 2)),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,)),
    }
    config.normalization_mapping = {
        FeatureType.STATE: NormalizationMode.MIN_MAX,
        FeatureType.VISUAL: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.MIN_MAX,
    }
    config.device = "cpu"
    return config


def create_default_stats() -> dict[str, dict[str, torch.Tensor]]:
    return {
        OBS_STATE: {
            "min": torch.tensor([0.0, -2.0]),
            "max": torch.tensor([10.0, 2.0]),
        },
        OBS_IMAGE: {
            "mean": torch.full((3, 2, 2), 10.0),
            "std": torch.full((3, 2, 2), 2.0),
        },
        ACTION: {
            "min": torch.tensor([-1.0, -2.0, 0.0]),
            "max": torch.tensor([1.0, 2.0, 10.0]),
        },
    }


def create_single_step_batch() -> dict[str, torch.Tensor]:
    return {
        OBS_STATE: torch.tensor([5.0, 0.0]),
        OBS_IMAGE: torch.full((3, 2, 2), 12.0),
        ACTION: torch.tensor([0.0, 0.0, 5.0]),
    }


def test_processor_factory_imports():
    from lerobot.policies.diffusion_policy.processor_diffusion_policy import (
        make_diffusion_policy_pre_post_processors as imported_factory,
    )

    assert imported_factory is make_diffusion_policy_pre_post_processors


def test_pipeline_construction_uses_default_names():
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_diffusion_policy_pre_post_processors(config, stats)

    assert preprocessor is not None
    assert postprocessor is not None
    assert preprocessor.name == POLICY_PREPROCESSOR_DEFAULT_NAME
    assert postprocessor.name == POLICY_POSTPROCESSOR_DEFAULT_NAME


def test_processor_step_composition_matches_legacy_order():
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_diffusion_policy_pre_post_processors(config, stats)

    assert [type(step) for step in preprocessor.steps] == [
        RenameObservationsProcessorStep,
        AddBatchDimensionProcessorStep,
        DeviceProcessorStep,
        NormalizerProcessorStep,
    ]
    assert [type(step) for step in postprocessor.steps] == [
        UnnormalizerProcessorStep,
        DeviceProcessorStep,
    ]


def test_preprocessor_normalizes_observations_visuals_and_training_action_targets():
    config = create_default_config()
    stats = create_default_stats()
    preprocessor, _ = make_diffusion_policy_pre_post_processors(config, stats)

    processed = preprocessor(create_single_step_batch())

    torch.testing.assert_close(processed[OBS_STATE], torch.tensor([[0.0, 0.0]]))
    torch.testing.assert_close(processed[OBS_IMAGE], torch.ones(1, 3, 2, 2))
    torch.testing.assert_close(processed[ACTION], torch.tensor([[0.0, 0.0, 0.0]]))


def test_preprocessor_adds_batch_dimension_and_moves_tensors_to_config_device():
    config = create_default_config()
    stats = create_default_stats()
    preprocessor, _ = make_diffusion_policy_pre_post_processors(config, stats)

    processed = preprocessor(create_single_step_batch())

    assert processed[OBS_STATE].shape == (1, 2)
    assert processed[OBS_IMAGE].shape == (1, 3, 2, 2)
    assert processed[ACTION].shape == (1, 3)
    assert processed[OBS_STATE].device.type == config.device
    assert processed[OBS_IMAGE].device.type == config.device
    assert processed[ACTION].device.type == config.device


def test_postprocessor_unnormalizes_policy_action_and_returns_cpu_tensor():
    config = create_default_config()
    stats = create_default_stats()
    _, postprocessor = make_diffusion_policy_pre_post_processors(config, stats)

    postprocessed = postprocessor(torch.tensor([[0.0, 0.0, 0.0]]))

    torch.testing.assert_close(postprocessed, torch.tensor([[0.0, 0.0, 5.0]]))
    assert postprocessed.device.type == "cpu"


def test_postprocessor_converter_functions_match_legacy_diffusion_behavior():
    config = create_default_config()
    stats = create_default_stats()
    _, postprocessor = make_diffusion_policy_pre_post_processors(config, stats)

    assert postprocessor.to_transition is policy_action_to_transition
    assert postprocessor.to_output is transition_to_policy_action

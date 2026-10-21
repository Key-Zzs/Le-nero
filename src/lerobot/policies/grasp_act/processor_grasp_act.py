#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.grasp_act.configuration_grasp_act import GraspACTConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    EnvTransition,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TransitionKey,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import OBS_STATE, POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


@dataclass
@ProcessorStepRegistry.register(name="grasp_act_state_slice_processor")
class GraspACTStateSliceProcessorStep(ProcessorStep):
    """Selects the Grasp-ACT proprioceptive state view from raw observation.state."""

    indices: tuple[int, ...] = field(default_factory=lambda: tuple(range(14, 28)))
    feature_key: str = OBS_STATE

    def __post_init__(self) -> None:
        self.indices = tuple(int(index) for index in self.indices)
        if not self.indices:
            raise ValueError("GraspACTStateSliceProcessorStep requires at least one state index.")

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is None or self.feature_key not in observation:
            return new_transition

        new_observation = dict(observation)
        new_observation[self.feature_key] = _slice_last_dim(new_observation[self.feature_key], self.indices)
        new_transition[TransitionKey.OBSERVATION] = new_observation
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"indices": list(self.indices), "feature_key": self.feature_key}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        new_features = features.copy()
        observation_features = dict(new_features.get(PipelineFeatureType.OBSERVATION, {}))
        if self.feature_key in observation_features:
            feature = observation_features[self.feature_key]
            observation_features[self.feature_key] = PolicyFeature(
                type=feature.type,
                shape=(len(self.indices),),
            )
            new_features[PipelineFeatureType.OBSERVATION] = observation_features
        return new_features


def _slice_last_dim(value: Any, indices: tuple[int, ...]) -> Any:
    max_index = max(indices)
    target_dim = len(indices)

    if torch.is_tensor(value):
        if value.shape[-1] == target_dim:
            return value
        if value.shape[-1] <= max_index:
            raise ValueError(
                f"Cannot slice state dimension {value.shape[-1]} with indices {indices}."
            )
        index_tensor = torch.tensor(indices, dtype=torch.long, device=value.device)
        return value.index_select(dim=-1, index=index_tensor)

    index_list = list(indices)
    if isinstance(value, np.ndarray):
        if value.shape[-1] == target_dim:
            return value
        if value.shape[-1] <= max_index:
            raise ValueError(
                f"Cannot slice state dimension {value.shape[-1]} with indices {indices}."
            )
        return value[..., index_list]

    array = np.asarray(value)
    if array.shape[-1] == target_dim:
        return value
    if array.shape[-1] <= max_index:
        raise ValueError(f"Cannot slice state dimension {array.shape[-1]} with indices {indices}.")
    return array[..., index_list]


def _slice_stat_value(value: Any, indices: tuple[int, ...]) -> Any:
    max_index = max(indices)
    target_dim = len(indices)

    if torch.is_tensor(value):
        if value.ndim == 0 or value.shape[-1] == target_dim:
            return value
        if value.shape[-1] <= max_index:
            return value
        index_tensor = torch.tensor(indices, dtype=torch.long, device=value.device)
        return value.index_select(dim=-1, index=index_tensor)

    array = np.asarray(value)
    if array.ndim == 0 or array.shape[-1] == target_dim:
        return value
    if array.shape[-1] <= max_index:
        return value

    sliced = array[..., list(indices)]
    if isinstance(value, (list, tuple)):
        return sliced.tolist()
    return sliced


def _slice_state_stats(
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None,
    indices: tuple[int, ...] | None,
) -> dict[str, dict[str, torch.Tensor]] | None:
    if not dataset_stats or not indices or OBS_STATE not in dataset_stats:
        return dataset_stats

    sliced_stats = deepcopy(dataset_stats)
    sliced_stats[OBS_STATE] = {
        stat_name: _slice_stat_value(stat_value, indices)
        for stat_name, stat_value in sliced_stats[OBS_STATE].items()
    }
    return sliced_stats


def make_grasp_act_pre_post_processors(
    config: GraspACTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Creates the pre- and post-processing pipelines for the ACT policy.

    The pre-processing pipeline handles normalization, batching, and device placement for the model inputs.
    The post-processing pipeline handles unnormalization and moves the model outputs back to the CPU.

    Args:
        config (ACTConfig): The ACT policy configuration object.
        dataset_stats (dict[str, dict[str, torch.Tensor]] | None): A dictionary containing dataset
            statistics (e.g., mean and std) used for normalization. Defaults to None.

    Returns:
        tuple[PolicyProcessorPipeline[dict[str, Any], dict[str, Any]], PolicyProcessorPipeline[PolicyAction, PolicyAction]]: A tuple containing the
        pre-processor pipeline and the post-processor pipeline.
    """

    state_indices = tuple(config.low_dim_state_indices) if config.low_dim_state_indices else None
    stats = _slice_state_stats(dataset_stats, state_indices)

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
    ]
    if state_indices is not None:
        input_steps.append(GraspACTStateSliceProcessorStep(indices=state_indices))
    input_steps.extend(
        [
            DeviceProcessorStep(device=config.device),
            NormalizerProcessorStep(
                features={**config.input_features, **config.output_features},
                norm_map=config.normalization_mapping,
                stats=stats,
                device=config.device,
            ),
        ]
    )
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )

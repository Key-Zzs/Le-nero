#!/usr/bin/env python

from __future__ import annotations

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
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import WeightedRandomSampler


class EpisodeAwareSampler:
    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
    ):
        """Sampler that optionally incorporates episode boundary information.

        Args:
            dataset_from_indices: List of indices containing the start of each episode in the dataset.
            dataset_to_indices: List of indices containing the end of each episode in the dataset.
            episode_indices_to_use: List of episode indices to use. If None, all episodes are used.
                                    Assumes that episodes are indexed from 0 to N-1.
            drop_n_first_frames: Number of frames to drop from the start of each episode.
            drop_n_last_frames: Number of frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
        """
        indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(dataset_from_indices, dataset_to_indices, strict=True)
        ):
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                indices.extend(range(start_index + drop_n_first_frames, end_index - drop_n_last_frames))

        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            for i in torch.randperm(len(self.indices)):
                yield self.indices[i]
        else:
            for i in self.indices:
                yield i

    def __len__(self) -> int:
        return len(self.indices)


DEFAULT_KEYFRAME_SAMPLER_CONFIG = {
    "enabled": False,
    "annotation_weight_column": "annotation.keyframe_weight",
    "annotation_event_column": "annotation.gripper_event",
    "positive_if_weight_gt": 1.0,
    "positive_event_ids": [2, 5],
    "include_pre_post_events": True,
    "positive_sample_weight": 3.0,
    "normal_sample_weight": 1.0,
    "max_sample_weight": 4.0,
    "require_annotation": False,
    "seed": 0,
    "log_sampler_stats": True,
}

PRE_POST_GRIPPER_EVENT_IDS = (1, 3, 4, 6)


@dataclass
class KeyframeSamplerResult:
    sampler: WeightedRandomSampler | None
    weights: torch.DoubleTensor | None
    positive_mask: torch.BoolTensor | None
    stats: dict[str, Any]


@dataclass
class KeyframeSampleWeightsResult:
    weights: torch.DoubleTensor | None
    positive_mask: torch.BoolTensor | None
    stats: dict[str, Any]


def _cfg_get(config: Mapping[str, Any] | Any | None, key: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def normalize_keyframe_sampler_config(config: Mapping[str, Any] | Any | None) -> dict[str, Any]:
    cfg = dict(DEFAULT_KEYFRAME_SAMPLER_CONFIG)
    if config is not None:
        for key in cfg:
            cfg[key] = _cfg_get(config, key, cfg[key])

    cfg["enabled"] = bool(cfg["enabled"])
    cfg["annotation_weight_column"] = str(cfg["annotation_weight_column"])
    cfg["annotation_event_column"] = str(cfg["annotation_event_column"])
    cfg["positive_if_weight_gt"] = float(cfg["positive_if_weight_gt"])
    cfg["positive_event_ids"] = [int(event_id) for event_id in cfg["positive_event_ids"]]
    cfg["include_pre_post_events"] = bool(cfg["include_pre_post_events"])
    cfg["positive_sample_weight"] = float(cfg["positive_sample_weight"])
    cfg["normal_sample_weight"] = float(cfg["normal_sample_weight"])
    cfg["max_sample_weight"] = float(cfg["max_sample_weight"])
    cfg["require_annotation"] = bool(cfg["require_annotation"])
    cfg["seed"] = int(cfg["seed"])
    cfg["log_sampler_stats"] = bool(cfg["log_sampler_stats"])

    if cfg["max_sample_weight"] <= 0:
        raise ValueError("`keyframe_sampler.max_sample_weight` must be > 0.")
    if cfg["positive_sample_weight"] < 0:
        raise ValueError("`keyframe_sampler.positive_sample_weight` must be >= 0.")
    if cfg["normal_sample_weight"] < 0:
        raise ValueError("`keyframe_sampler.normal_sample_weight` must be >= 0.")
    if cfg["positive_sample_weight"] == 0 and cfg["normal_sample_weight"] == 0:
        raise ValueError(
            "`keyframe_sampler.positive_sample_weight` and "
            "`keyframe_sampler.normal_sample_weight` cannot both be 0."
        )
    return cfg


def keyframe_sampler_enabled(config: Mapping[str, Any] | Any | None) -> bool:
    return bool(_cfg_get(config, "enabled", False))


def _base_stats(config: Mapping[str, Any] | Any | None, *, enabled: bool | None = None) -> dict[str, Any]:
    cfg = normalize_keyframe_sampler_config(config)
    return {
        "keyframe_sampler/enabled": cfg["enabled"] if enabled is None else bool(enabled),
        "keyframe_sampler/positive_sample_count": 0,
        "keyframe_sampler/positive_sample_ratio": 0.0,
        "keyframe_sampler/mean_sample_weight": None,
        "keyframe_sampler/max_sample_weight": None,
        "keyframe_sampler/normal_sample_weight": cfg["normal_sample_weight"],
        "keyframe_sampler/positive_sample_weight": min(
            cfg["positive_sample_weight"], cfg["max_sample_weight"]
        ),
        "keyframe_sampler/annotation_missing": False,
        "keyframe_sampler/fallback_to_default": True,
        "keyframe_sampler/eligible_sample_count": 0,
        "keyframe_sampler/num_samples": 0,
        "keyframe_sampler/disabled_reason": None,
    }


def _feature_names(dataset: Any) -> set[str]:
    names: set[str] = set()
    meta = getattr(dataset, "meta", None)
    meta_features = getattr(meta, "features", None)
    if isinstance(meta_features, Mapping):
        names.update(str(key) for key in meta_features)

    hf_dataset = getattr(dataset, "hf_dataset", None)
    hf_features = getattr(hf_dataset, "features", None)
    if isinstance(hf_features, Mapping):
        names.update(str(key) for key in hf_features)
    column_names = getattr(hf_dataset, "column_names", None)
    if column_names is not None:
        names.update(str(key) for key in column_names)
    if isinstance(hf_dataset, Mapping):
        names.update(str(key) for key in hf_dataset)
    return names


def _has_column(dataset: Any, column: str) -> bool:
    return column in _feature_names(dataset)


def _to_1d_tensor(values: Any, *, dtype: torch.dtype, column: str) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        tensor = values.detach().cpu()
    else:
        try:
            tensor = torch.as_tensor(values)
        except (TypeError, ValueError):
            tensor = torch.stack([torch.as_tensor(value).squeeze() for value in values])
    tensor = tensor.to(dtype=dtype)
    while tensor.ndim > 1 and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    if tensor.ndim != 1:
        raise ValueError(f"`{column}` must be a scalar per frame. Got shape {tuple(tensor.shape)}.")
    return tensor


def _get_column_tensor(dataset: Any, column: str, *, dtype: torch.dtype) -> torch.Tensor:
    source = getattr(dataset, "hf_dataset", dataset)
    if isinstance(source, Mapping):
        values = source[column]
    else:
        try:
            values = source[column]
        except (KeyError, TypeError, ValueError):
            values = [source[index][column] for index in range(len(dataset))]
    return _to_1d_tensor(values, dtype=dtype, column=column)


def _episode_ranges(dataset: Any) -> list[tuple[int, int]]:
    episodes = getattr(getattr(dataset, "meta", None), "episodes", None)
    if episodes is None:
        raise ValueError("Dataset metadata is missing episode ranges.")

    try:
        starts = episodes["dataset_from_index"]
        ends = episodes["dataset_to_index"]
        return [(int(start), int(end)) for start, end in zip(starts, ends)]
    except (KeyError, TypeError, ValueError):
        ranges = []
        for episode in episodes:
            ranges.append((int(episode["dataset_from_index"]), int(episode["dataset_to_index"])))
        return ranges


def _episode_indices_for_dataset(dataset: Any, *, length: int) -> torch.LongTensor:
    if _has_column(dataset, "episode_index"):
        episode_indices = _get_column_tensor(dataset, "episode_index", dtype=torch.long)
        if len(episode_indices) != length:
            raise ValueError(
                f"`episode_index` length ({len(episode_indices)}) does not match dataset length ({length})."
            )
        return episode_indices

    episode_indices = torch.empty(length, dtype=torch.long)
    for episode_idx, (start, end) in enumerate(_episode_ranges(dataset)):
        episode_indices[start:end] = int(episode_idx)
    return episode_indices


def _positive_event_ids(positive_event_ids: list[int], include_pre_post_events: bool) -> set[int]:
    event_ids = {int(event_id) for event_id in positive_event_ids}
    if include_pre_post_events:
        event_ids.update(PRE_POST_GRIPPER_EVENT_IDS)
    return event_ids


def _compute_keyframe_sample_weights_with_stats(
    dataset: Any,
    action_delta_indices: list[int] | tuple[int, ...],
    *,
    annotation_weight_column: str = "annotation.keyframe_weight",
    annotation_event_column: str = "annotation.gripper_event",
    positive_if_weight_gt: float = 1.0,
    positive_event_ids: list[int] | tuple[int, ...] = (2, 5),
    include_pre_post_events: bool = True,
    positive_sample_weight: float = 3.0,
    normal_sample_weight: float = 1.0,
    max_sample_weight: float = 4.0,
    require_annotation: bool = False,
) -> KeyframeSampleWeightsResult:
    cfg = {
        "enabled": True,
        "annotation_weight_column": annotation_weight_column,
        "annotation_event_column": annotation_event_column,
        "positive_if_weight_gt": positive_if_weight_gt,
        "positive_event_ids": list(positive_event_ids),
        "include_pre_post_events": include_pre_post_events,
        "positive_sample_weight": positive_sample_weight,
        "normal_sample_weight": normal_sample_weight,
        "max_sample_weight": max_sample_weight,
        "require_annotation": require_annotation,
        "seed": 0,
        "log_sampler_stats": True,
    }
    cfg = normalize_keyframe_sampler_config(cfg)
    stats = _base_stats(cfg, enabled=True)

    length = len(dataset)
    stats["keyframe_sampler/eligible_sample_count"] = int(length)
    stats["keyframe_sampler/num_samples"] = int(length)
    if length == 0:
        stats["keyframe_sampler/mean_sample_weight"] = 0.0
        stats["keyframe_sampler/max_sample_weight"] = 0.0
        stats["keyframe_sampler/disabled_reason"] = "empty_dataset"
        return KeyframeSampleWeightsResult(
            torch.empty(0, dtype=torch.double),
            torch.empty(0, dtype=torch.bool),
            stats,
        )

    has_weight = _has_column(dataset, cfg["annotation_weight_column"])
    has_event = _has_column(dataset, cfg["annotation_event_column"])
    if not has_weight and not has_event:
        message = (
            "Keyframe sampler annotation columns are missing: "
            f"{cfg['annotation_weight_column']!r}, {cfg['annotation_event_column']!r}"
        )
        if cfg["require_annotation"]:
            raise ValueError(message)
        stats["keyframe_sampler/annotation_missing"] = True
        stats["keyframe_sampler/disabled_reason"] = "annotation_missing"
        return KeyframeSampleWeightsResult(None, None, stats)

    weight_tensor = (
        _get_column_tensor(dataset, cfg["annotation_weight_column"], dtype=torch.float32)
        if has_weight
        else None
    )
    event_tensor = (
        _get_column_tensor(dataset, cfg["annotation_event_column"], dtype=torch.long)
        if has_event
        else None
    )
    if weight_tensor is not None and len(weight_tensor) != length:
        raise ValueError(
            f"`{cfg['annotation_weight_column']}` length ({len(weight_tensor)}) "
            f"does not match dataset length ({length})."
        )
    if event_tensor is not None and len(event_tensor) != length:
        raise ValueError(
            f"`{cfg['annotation_event_column']}` length ({len(event_tensor)}) "
            f"does not match dataset length ({length})."
        )

    deltas = [int(delta) for delta in action_delta_indices]
    if not deltas:
        raise ValueError("`action_delta_indices` must not be empty when keyframe sampler is enabled.")

    episode_indices = _episode_indices_for_dataset(dataset, length=length)
    ranges = _episode_ranges(dataset)
    positive_ids = _positive_event_ids(cfg["positive_event_ids"], cfg["include_pre_post_events"])
    positive_mask = torch.zeros(length, dtype=torch.bool)

    for sample_idx in range(length):
        episode_idx = int(episode_indices[sample_idx].item())
        if episode_idx < 0 or episode_idx >= len(ranges):
            raise ValueError(f"Invalid episode_index={episode_idx} at dataset index {sample_idx}.")
        episode_start, episode_end = ranges[episode_idx]
        if episode_end <= episode_start:
            continue

        query_indices = [
            max(episode_start, min(episode_end - 1, sample_idx + delta)) for delta in deltas
        ]

        is_positive = False
        if weight_tensor is not None:
            query_weights = torch.nan_to_num(
                weight_tensor[query_indices],
                nan=1.0,
                posinf=cfg["max_sample_weight"],
                neginf=1.0,
            )
            is_positive = bool((query_weights > cfg["positive_if_weight_gt"]).any().item())
        if not is_positive and event_tensor is not None:
            is_positive = any(int(event_tensor[index].item()) in positive_ids for index in query_indices)
        positive_mask[sample_idx] = is_positive

    normal_weight = min(cfg["normal_sample_weight"], cfg["max_sample_weight"])
    positive_weight = min(cfg["positive_sample_weight"], cfg["max_sample_weight"])
    weights = torch.full((length,), normal_weight, dtype=torch.double)
    weights[positive_mask] = positive_weight
    weights = torch.nan_to_num(weights, nan=normal_weight, posinf=cfg["max_sample_weight"], neginf=0.0)
    weights = weights.clamp(min=0.0, max=cfg["max_sample_weight"])

    positive_count = int(positive_mask.sum().item())
    stats["keyframe_sampler/positive_sample_count"] = positive_count
    stats["keyframe_sampler/positive_sample_ratio"] = positive_count / length
    stats["keyframe_sampler/mean_sample_weight"] = float(weights.mean().item())
    stats["keyframe_sampler/max_sample_weight"] = float(weights.max().item())
    stats["keyframe_sampler/fallback_to_default"] = False
    return KeyframeSampleWeightsResult(weights, positive_mask, stats)


def compute_keyframe_sample_weights(
    dataset: Any,
    action_delta_indices: list[int] | tuple[int, ...],
    annotation_weight_column: str = "annotation.keyframe_weight",
    annotation_event_column: str = "annotation.gripper_event",
    positive_if_weight_gt: float = 1.0,
    positive_event_ids: list[int] | tuple[int, ...] = (2, 5),
    include_pre_post_events: bool = True,
    positive_sample_weight: float = 3.0,
    normal_sample_weight: float = 1.0,
    max_sample_weight: float = 4.0,
    require_annotation: bool = False,
) -> torch.DoubleTensor | None:
    """Return per-dataset-index weights for action-window keyframe oversampling.

    Each sample is positive when any timestep in its action_delta_indices-aligned
    window has keyframe_weight > positive_if_weight_gt or a configured gripper
    event id. Window indices are clamped to the current episode, matching
    LeRobotDataset's temporal query behavior.
    """
    return _compute_keyframe_sample_weights_with_stats(
        dataset,
        action_delta_indices,
        annotation_weight_column=annotation_weight_column,
        annotation_event_column=annotation_event_column,
        positive_if_weight_gt=positive_if_weight_gt,
        positive_event_ids=positive_event_ids,
        include_pre_post_events=include_pre_post_events,
        positive_sample_weight=positive_sample_weight,
        normal_sample_weight=normal_sample_weight,
        max_sample_weight=max_sample_weight,
        require_annotation=require_annotation,
    ).weights


def build_keyframe_weighted_sampler(
    dataset: Any,
    action_delta_indices: list[int] | tuple[int, ...] | None,
    keyframe_sampler_config: Mapping[str, Any] | Any | None,
    *,
    eligible_indices: list[int] | torch.Tensor | None = None,
) -> KeyframeSamplerResult:
    cfg = normalize_keyframe_sampler_config(keyframe_sampler_config)
    stats = _base_stats(cfg)
    if not cfg["enabled"]:
        stats["keyframe_sampler/disabled_reason"] = "disabled_by_config"
        return KeyframeSamplerResult(None, None, None, stats)
    if action_delta_indices is None:
        raise ValueError("keyframe_sampler.enabled=true requires policy.action_delta_indices.")

    sample_result = _compute_keyframe_sample_weights_with_stats(
        dataset,
        action_delta_indices,
        annotation_weight_column=cfg["annotation_weight_column"],
        annotation_event_column=cfg["annotation_event_column"],
        positive_if_weight_gt=cfg["positive_if_weight_gt"],
        positive_event_ids=cfg["positive_event_ids"],
        include_pre_post_events=cfg["include_pre_post_events"],
        positive_sample_weight=cfg["positive_sample_weight"],
        normal_sample_weight=cfg["normal_sample_weight"],
        max_sample_weight=cfg["max_sample_weight"],
        require_annotation=cfg["require_annotation"],
    )
    stats.update(sample_result.stats)
    stats["keyframe_sampler/enabled"] = True

    if sample_result.weights is None or sample_result.positive_mask is None:
        stats["keyframe_sampler/fallback_to_default"] = True
        return KeyframeSamplerResult(None, None, None, stats)

    weights = sample_result.weights.clone()
    positive_mask = sample_result.positive_mask.clone()
    eligible_mask = torch.ones(len(dataset), dtype=torch.bool)
    if eligible_indices is not None:
        eligible_mask = torch.zeros(len(dataset), dtype=torch.bool)
        eligible_tensor = torch.as_tensor(eligible_indices, dtype=torch.long)
        if eligible_tensor.numel() > 0:
            eligible_mask[eligible_tensor] = True
        weights[~eligible_mask] = 0.0

    eligible_count = int(eligible_mask.sum().item())
    eligible_positive_count = int((positive_mask & eligible_mask).sum().item())
    stats["keyframe_sampler/eligible_sample_count"] = eligible_count
    stats["keyframe_sampler/num_samples"] = eligible_count
    stats["keyframe_sampler/positive_sample_count"] = eligible_positive_count
    stats["keyframe_sampler/positive_sample_ratio"] = (
        eligible_positive_count / eligible_count if eligible_count else 0.0
    )
    if eligible_count:
        eligible_weights = weights[eligible_mask]
        stats["keyframe_sampler/mean_sample_weight"] = float(eligible_weights.mean().item())
        stats["keyframe_sampler/max_sample_weight"] = float(eligible_weights.max().item())
    else:
        stats["keyframe_sampler/mean_sample_weight"] = 0.0
        stats["keyframe_sampler/max_sample_weight"] = 0.0

    normal_weight = min(cfg["normal_sample_weight"], cfg["max_sample_weight"])
    positive_weight = min(cfg["positive_sample_weight"], cfg["max_sample_weight"])
    if eligible_count == 0:
        stats["keyframe_sampler/fallback_to_default"] = True
        stats["keyframe_sampler/disabled_reason"] = "no_eligible_samples"
        return KeyframeSamplerResult(None, weights, positive_mask, stats)
    if eligible_positive_count == 0:
        stats["keyframe_sampler/fallback_to_default"] = True
        stats["keyframe_sampler/disabled_reason"] = "no_positive_samples"
        return KeyframeSamplerResult(None, weights, positive_mask, stats)
    if eligible_positive_count == eligible_count or positive_weight == normal_weight:
        stats["keyframe_sampler/fallback_to_default"] = True
        stats["keyframe_sampler/disabled_reason"] = "no_weight_contrast"
        return KeyframeSamplerResult(None, weights, positive_mask, stats)
    if float(weights.sum().item()) <= 0.0:
        stats["keyframe_sampler/fallback_to_default"] = True
        stats["keyframe_sampler/disabled_reason"] = "zero_total_weight"
        return KeyframeSamplerResult(None, weights, positive_mask, stats)

    generator = torch.Generator()
    generator.manual_seed(cfg["seed"])
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=eligible_count,
        replacement=True,
        generator=generator,
    )
    stats["keyframe_sampler/fallback_to_default"] = False
    stats["keyframe_sampler/disabled_reason"] = None
    return KeyframeSamplerResult(sampler, weights, positive_mask, stats)


def format_keyframe_sampler_stats(stats: dict[str, Any]) -> str:
    return (
        "keyframe_sampler: "
        f"enabled={stats.get('keyframe_sampler/enabled')} "
        f"positive_sample_count={stats.get('keyframe_sampler/positive_sample_count')} "
        f"positive_sample_ratio={stats.get('keyframe_sampler/positive_sample_ratio')} "
        f"mean_sample_weight={stats.get('keyframe_sampler/mean_sample_weight')} "
        f"max_sample_weight={stats.get('keyframe_sampler/max_sample_weight')} "
        f"normal_sample_weight={stats.get('keyframe_sampler/normal_sample_weight')} "
        f"positive_sample_weight={stats.get('keyframe_sampler/positive_sample_weight')} "
        f"annotation_missing={stats.get('keyframe_sampler/annotation_missing')} "
        f"fallback_to_default={stats.get('keyframe_sampler/fallback_to_default')} "
        f"disabled_reason={stats.get('keyframe_sampler/disabled_reason')}"
    )

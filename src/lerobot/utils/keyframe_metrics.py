#!/usr/bin/env python

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

DEFAULT_ANNOTATION_WEIGHT_COLUMN = "annotation.keyframe_weight"
DEFAULT_ANNOTATION_EVENT_COLUMN = "annotation.gripper_event"

DEFAULT_DEBUG_METRICS_CONFIG = {
    "enabled": True,
    "write_annotation_summary_json": True,
    "write_sampler_summary_json": True,
    "write_batch_metrics_preview": False,
    "max_preview_batches": 5,
    "plot_annotation_distribution": False,
}

EVENT_NORMAL = 0
EVENT_PRE_CLOSING = 1
EVENT_CLOSING = 2
EVENT_POST_CLOSING = 3
EVENT_PRE_OPENING = 4
EVENT_OPENING = 5
EVENT_POST_OPENING = 6
EVENT_TRANSITION_UNKNOWN = 7

EVENT_NAMES = {
    EVENT_NORMAL: "normal",
    EVENT_PRE_CLOSING: "pre_closing",
    EVENT_CLOSING: "closing",
    EVENT_POST_CLOSING: "post_closing",
    EVENT_PRE_OPENING: "pre_opening",
    EVENT_OPENING: "opening",
    EVENT_POST_OPENING: "post_opening",
    EVENT_TRANSITION_UNKNOWN: "transition_unknown",
}

EVENT_COUNT_KEYS = {
    EVENT_NORMAL: "normal_count",
    EVENT_PRE_CLOSING: "pre_closing_count",
    EVENT_CLOSING: "closing_count",
    EVENT_POST_CLOSING: "post_closing_count",
    EVENT_PRE_OPENING: "pre_opening_count",
    EVENT_OPENING: "opening_count",
    EVENT_POST_OPENING: "post_opening_count",
    EVENT_TRANSITION_UNKNOWN: "transition_unknown_count",
}


def cfg_get(config: Mapping[str, Any] | Any | None, key: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def normalize_debug_metrics_config(config: Mapping[str, Any] | Any | None) -> dict[str, Any]:
    cfg = dict(DEFAULT_DEBUG_METRICS_CONFIG)
    if config is not None:
        for key in cfg:
            cfg[key] = cfg_get(config, key, cfg[key])

    cfg["enabled"] = bool(cfg["enabled"])
    cfg["write_annotation_summary_json"] = bool(cfg["write_annotation_summary_json"])
    cfg["write_sampler_summary_json"] = bool(cfg["write_sampler_summary_json"])
    cfg["write_batch_metrics_preview"] = bool(cfg["write_batch_metrics_preview"])
    cfg["max_preview_batches"] = max(0, int(cfg["max_preview_batches"]))
    cfg["plot_annotation_distribution"] = bool(cfg["plot_annotation_distribution"])
    return cfg


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


def _dataset_length(dataset: Any) -> int:
    num_frames = getattr(dataset, "num_frames", None)
    if num_frames is not None:
        return int(num_frames)
    return int(len(dataset))


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


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    denominator = float(denominator)
    if denominator <= 0:
        return 0.0
    return float(numerator) / denominator


def _json_safe_scalar(value: Any) -> bool | int | float | str | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.numel() != 1:
            return None
        value = tensor.item()
    elif hasattr(value, "item") and callable(value.item):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass

    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, str):
        return value
    return None


def to_jsonable(value: Any) -> Any:
    scalar = _json_safe_scalar(value)
    if scalar is not None or value is None:
        return scalar
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.numel() > 1000:
            return f"<tensor shape={tuple(tensor.shape)}>"
        return to_jsonable(tensor.tolist())
    return str(value)


def scalarize_log_dict(metrics: Mapping[str, Any], *, include_strings: bool = False) -> dict[str, int | float | bool | str]:
    scalar_metrics: dict[str, int | float | bool | str] = {}
    for key, value in metrics.items():
        scalar = _json_safe_scalar(value)
        if scalar is None:
            continue
        if isinstance(scalar, str) and not include_strings:
            continue
        scalar_metrics[str(key)] = scalar
    return scalar_metrics


def format_scalar_metrics(metrics: Mapping[str, Any], *, max_items: int = 24) -> str:
    scalar_metrics = scalarize_log_dict(metrics, include_strings=False)
    items = sorted(scalar_metrics.items())[:max_items]
    parts = []
    for key, value in items:
        if isinstance(value, bool):
            display = str(value).lower()
        elif isinstance(value, float):
            display = f"{value:.6g}"
        else:
            display = str(value)
        parts.append(f"{key}={display}")
    remaining = len(scalar_metrics) - len(items)
    if remaining > 0:
        parts.append(f"...(+{remaining} more)")
    return " ".join(parts)


def write_debug_json(output_dir: str | Path, relative_path: str, payload: Any) -> Path:
    path = Path(output_dir) / "debug" / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(to_jsonable(payload), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def summarize_annotation_distribution(
    dataset: Any,
    annotation_weight_column: str = DEFAULT_ANNOTATION_WEIGHT_COLUMN,
    annotation_event_column: str = DEFAULT_ANNOTATION_EVENT_COLUMN,
) -> dict[str, Any]:
    num_frames = _dataset_length(dataset)
    has_weight = _has_column(dataset, annotation_weight_column)
    has_event = _has_column(dataset, annotation_event_column)
    has_annotation = bool(has_weight or has_event)

    summary: dict[str, Any] = {
        "annotation/has_annotation": has_annotation,
        "annotation/num_frames": num_frames,
        "annotation/weight_column": annotation_weight_column,
        "annotation/event_column": annotation_event_column,
        "annotation/keyframe_frame_count": 0,
        "annotation/keyframe_frame_ratio": 0.0,
        "annotation/mean_keyframe_weight": None,
        "annotation/max_keyframe_weight": None,
        "annotation/mean_weight": None,
        "annotation/max_weight": None,
    }
    for suffix in EVENT_COUNT_KEYS.values():
        summary[f"annotation/{suffix}"] = 0

    if not has_annotation or num_frames == 0:
        return summary

    weight_tensor = (
        _get_column_tensor(dataset, annotation_weight_column, dtype=torch.float32) if has_weight else None
    )
    event_tensor = (
        _get_column_tensor(dataset, annotation_event_column, dtype=torch.long) if has_event else None
    )
    if weight_tensor is not None and len(weight_tensor) != num_frames:
        raise ValueError(
            f"`{annotation_weight_column}` length ({len(weight_tensor)}) does not match dataset length ({num_frames})."
        )
    if event_tensor is not None and len(event_tensor) != num_frames:
        raise ValueError(
            f"`{annotation_event_column}` length ({len(event_tensor)}) does not match dataset length ({num_frames})."
        )

    keyframe_mask = torch.zeros(num_frames, dtype=torch.bool)
    if weight_tensor is not None:
        weight_tensor = torch.nan_to_num(weight_tensor, nan=1.0, posinf=1_000_000.0, neginf=1.0)
        keyframe_mask |= weight_tensor > 1.0
        summary["annotation/mean_weight"] = float(weight_tensor.mean().item()) if num_frames else 0.0
        summary["annotation/max_weight"] = float(weight_tensor.max().item()) if num_frames else 0.0
    if event_tensor is not None:
        keyframe_mask |= event_tensor != EVENT_NORMAL
        for event_id, suffix in EVENT_COUNT_KEYS.items():
            summary[f"annotation/{suffix}"] = int((event_tensor == event_id).sum().item())
        summary["annotation/event_distribution"] = {
            EVENT_NAMES[event_id]: int((event_tensor == event_id).sum().item()) for event_id in EVENT_NAMES
        }

    keyframe_count = int(keyframe_mask.sum().item())
    summary["annotation/keyframe_frame_count"] = keyframe_count
    summary["annotation/keyframe_frame_ratio"] = _safe_ratio(keyframe_count, num_frames)
    if weight_tensor is not None and keyframe_count > 0:
        keyframe_weights = weight_tensor[keyframe_mask]
        summary["annotation/mean_keyframe_weight"] = float(keyframe_weights.mean().item())
        summary["annotation/max_keyframe_weight"] = float(keyframe_weights.max().item())
    elif weight_tensor is not None:
        summary["annotation/mean_keyframe_weight"] = 0.0
        summary["annotation/max_keyframe_weight"] = 0.0
    return summary


def _squeeze_last_singleton(tensor: torch.Tensor) -> torch.Tensor:
    while tensor.ndim > 0 and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    return tensor


def _batch_temporal_shape(batch: Mapping[str, Any], *candidate_keys: str) -> tuple[int, int] | None:
    for key in candidate_keys:
        value = batch.get(key)
        if isinstance(value, torch.Tensor) and value.ndim >= 2:
            return int(value.shape[0]), int(value.shape[1])
    return None


def _batch_field_2d(batch: Mapping[str, Any], key: str, *, dtype: torch.dtype) -> torch.Tensor | None:
    value = batch.get(key)
    if value is None or not isinstance(value, torch.Tensor):
        return None
    tensor = _squeeze_last_singleton(value.detach()).to(dtype=dtype)
    if tensor.ndim != 2:
        return None
    return tensor


def compute_batch_annotation_metrics(
    batch: Mapping[str, Any],
    annotation_weight_column: str = DEFAULT_ANNOTATION_WEIGHT_COLUMN,
    annotation_event_column: str = DEFAULT_ANNOTATION_EVENT_COLUMN,
) -> dict[str, int | float | bool]:
    weight_tensor = _batch_field_2d(batch, annotation_weight_column, dtype=torch.float32)
    event_tensor = _batch_field_2d(batch, annotation_event_column, dtype=torch.long)
    if weight_tensor is None and event_tensor is None:
        return {}

    temporal_shape = _batch_temporal_shape(
        batch,
        annotation_weight_column,
        annotation_event_column,
        "action",
    )
    if temporal_shape is None:
        return {}

    if weight_tensor is not None and tuple(weight_tensor.shape) != temporal_shape:
        return {}
    if event_tensor is not None and tuple(event_tensor.shape) != temporal_shape:
        return {}

    pad_tensor = _batch_field_2d(batch, "action_is_pad", dtype=torch.bool)
    if pad_tensor is not None and tuple(pad_tensor.shape) == temporal_shape:
        valid_mask = ~pad_tensor
    else:
        device = weight_tensor.device if weight_tensor is not None else event_tensor.device
        valid_mask = torch.ones(temporal_shape, dtype=torch.bool, device=device)

    valid_count = int(valid_mask.sum().item())
    total_count = int(valid_mask.numel())
    padded_count = total_count - valid_count
    if valid_count <= 0:
        return {
            "batch/has_annotation": True,
            "batch/keyframe_ratio": 0.0,
            "batch/mean_keyframe_weight": 0.0,
            "batch/max_keyframe_weight": 0.0,
            "batch/opening_count": 0,
            "batch/closing_count": 0,
            "batch/normal_count": 0,
            "batch/valid_action_count": 0,
            "batch/padded_action_count": padded_count,
        }

    keyframe_mask = torch.zeros(temporal_shape, dtype=torch.bool, device=valid_mask.device)
    metrics: dict[str, int | float | bool] = {"batch/has_annotation": True}
    if weight_tensor is not None:
        weight_tensor = torch.nan_to_num(
            weight_tensor.to(device=valid_mask.device), nan=1.0, posinf=1_000_000.0, neginf=1.0
        )
        keyframe_mask |= weight_tensor > 1.0
        valid_weights = weight_tensor[valid_mask]
        metrics["batch/mean_keyframe_weight"] = float(valid_weights.mean().item())
        metrics["batch/max_keyframe_weight"] = float(valid_weights.max().item())
    else:
        metrics["batch/mean_keyframe_weight"] = 1.0
        metrics["batch/max_keyframe_weight"] = 1.0

    if event_tensor is not None:
        event_tensor = event_tensor.to(device=valid_mask.device)
        keyframe_mask |= event_tensor != EVENT_NORMAL
        metrics["batch/opening_count"] = int(((event_tensor == EVENT_OPENING) & valid_mask).sum().item())
        metrics["batch/closing_count"] = int(((event_tensor == EVENT_CLOSING) & valid_mask).sum().item())
    else:
        metrics["batch/opening_count"] = 0
        metrics["batch/closing_count"] = 0

    keyframe_count = int((keyframe_mask & valid_mask).sum().item())
    metrics["batch/keyframe_ratio"] = _safe_ratio(keyframe_count, valid_count)
    metrics["batch/normal_count"] = int(valid_count - keyframe_count)
    metrics["batch/valid_action_count"] = valid_count
    metrics["batch/padded_action_count"] = padded_count
    return metrics


def write_annotation_plots(
    dataset: Any,
    output_dir: str | Path,
    annotation_weight_column: str = DEFAULT_ANNOTATION_WEIGHT_COLUMN,
    annotation_event_column: str = DEFAULT_ANNOTATION_EVENT_COLUMN,
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional matplotlib install
        logging.warning("Skipping annotation debug plots because matplotlib is unavailable: %s", exc)
        return []

    written: list[Path] = []
    debug_dir = Path(output_dir) / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    if _has_column(dataset, annotation_event_column):
        event_tensor = _get_column_tensor(dataset, annotation_event_column, dtype=torch.long)
        labels = [EVENT_NAMES[event_id] for event_id in EVENT_NAMES]
        counts = [int((event_tensor == event_id).sum().item()) for event_id in EVENT_NAMES]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(labels, counts)
        ax.set_ylabel("frames")
        ax.set_title("Annotation event distribution")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        path = debug_dir / "annotation_event_distribution.png"
        fig.savefig(path)
        plt.close(fig)
        written.append(path)

    if _has_column(dataset, annotation_weight_column):
        weight_tensor = _get_column_tensor(dataset, annotation_weight_column, dtype=torch.float32)
        weight_tensor = torch.nan_to_num(weight_tensor, nan=1.0, posinf=1_000_000.0, neginf=1.0)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(weight_tensor.tolist(), bins=50)
        ax.set_xlabel("keyframe weight")
        ax.set_ylabel("frames")
        ax.set_title("Keyframe weight histogram")
        fig.tight_layout()
        path = debug_dir / "keyframe_weight_histogram.png"
        fig.savefig(path)
        plt.close(fig)
        written.append(path)

    return written


def log_training_debug_startup(
    *,
    dataset: Any,
    output_dir: str | Path,
    debug_metrics_config: Mapping[str, Any] | Any | None,
    annotation_weight_column: str = DEFAULT_ANNOTATION_WEIGHT_COLUMN,
    annotation_event_column: str = DEFAULT_ANNOTATION_EVENT_COLUMN,
    sampler_stats: Mapping[str, Any] | None = None,
    wandb_logger: Any | None = None,
    step: int = 0,
) -> dict[str, Any]:
    debug_cfg = normalize_debug_metrics_config(debug_metrics_config)
    if not debug_cfg["enabled"]:
        return {"annotation_summary": {}, "written_files": [], "wandb_metrics": {}}

    annotation_summary = summarize_annotation_distribution(
        dataset,
        annotation_weight_column=annotation_weight_column,
        annotation_event_column=annotation_event_column,
    )
    logging.info("annotation summary: %s", format_scalar_metrics(annotation_summary))

    written_files: list[Path] = []
    if debug_cfg["write_annotation_summary_json"]:
        written_files.append(write_debug_json(output_dir, "annotation_summary.json", annotation_summary))
    if sampler_stats is not None and debug_cfg["write_sampler_summary_json"]:
        written_files.append(write_debug_json(output_dir, "keyframe_sampler_summary.json", dict(sampler_stats)))
    if debug_cfg["plot_annotation_distribution"]:
        written_files.extend(
            write_annotation_plots(
                dataset,
                output_dir,
                annotation_weight_column=annotation_weight_column,
                annotation_event_column=annotation_event_column,
            )
        )

    wandb_metrics = scalarize_log_dict(annotation_summary)
    if sampler_stats is not None:
        wandb_metrics.update(scalarize_log_dict(sampler_stats))
    if wandb_logger is not None and wandb_metrics:
        wandb_logger.log_dict(wandb_metrics, step=step)

    if written_files:
        logging.info("debug metrics written: %s", ", ".join(str(path) for path in written_files))
    return {
        "annotation_summary": annotation_summary,
        "written_files": written_files,
        "wandb_metrics": wandb_metrics,
    }

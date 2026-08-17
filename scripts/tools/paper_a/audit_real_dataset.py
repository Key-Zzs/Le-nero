#!/usr/bin/env python3
"""Read-only Stage 1 audit for real LeRobot RGB-D datasets.

The auditor is deliberately independent of ``LeRobotDataset`` loading: it
reads metadata, Parquet in batches, and Zarr in bounded frame chunks.  It never
opens a raw dataset in write mode and never changes source files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq


CAMERAS = ("head", "left_wrist", "right_wrist")
MODALITIES = ("depth", "left_ir", "right_ir")
JOIN_FIELDS = ("index", "episode_index", "frame_index", "global_frame_index", "robot_timestamp")
DEFAULT_CHUNK_FRAMES = 16
CONTROL_SUFFIXES = {".json", ".parquet", ".txt", ".yaml", ".yml"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def canonical_hash(value: Any) -> str:
    payload = json.dumps(jsonable(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_datasets(root: Path) -> list[Path]:
    """Detect only directories with LeRobot metadata and a data directory."""

    root = root.expanduser().resolve()
    found: set[Path] = set()
    for info_path in root.rglob("meta/info.json"):
        dataset_root = info_path.parent.parent
        if dataset_root.is_dir() and (dataset_root / "data").is_dir():
            found.add(dataset_root)
    return sorted(found)


def _payload_class(relative: Path) -> str:
    parts = relative.parts
    if parts and parts[0] == "sidecars":
        return "raw_sidecar_chunk" if relative.suffix not in {".json", ".zarray", ".zattrs", ".zgroup"} else "sidecar_schema"
    if parts and parts[0] == "videos":
        return "video_payload"
    if parts and parts[0] == "meta":
        return "metadata"
    return "parquet_payload" if relative.suffix == ".parquet" else "other"


def build_source_manifest(dataset_root: Path, *, hash_video_payloads: bool = True) -> dict[str, Any]:
    """Return a reproducible file inventory without hashing Zarr frame chunks."""

    dataset_root = dataset_root.expanduser().resolve()
    entries: list[dict[str, Any]] = []
    raw_chunk_count = 0
    raw_chunk_bytes = 0
    raw_chunk_dirs: dict[str, dict[str, int]] = {}
    for path in sorted(p for p in dataset_root.rglob("*") if p.is_file()):
        relative = path.relative_to(dataset_root)
        kind = _payload_class(relative)
        if kind == "raw_sidecar_chunk":
            raw_chunk_count += 1
            raw_chunk_bytes += path.stat().st_size
            directory = relative.parent.as_posix()
            summary = raw_chunk_dirs.setdefault(directory, {"files": 0, "bytes": 0})
            summary["files"] += 1
            summary["bytes"] += path.stat().st_size
            continue
        entry: dict[str, Any] = {
            "path": relative.as_posix(),
            "kind": kind,
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        if kind != "raw_sidecar_chunk" and (kind != "video_payload" or hash_video_payloads):
            entry["sha256"] = sha256_file(path)
        else:
            entry["sha256"] = None
        entries.append(entry)
    return {
        "dataset_root": str(dataset_root),
        "hash_policy": "SHA-256 metadata, Parquet, Zarr schema, and video files; Zarr frame chunks are inventory-only",
        "files": entries,
        "total_files": len(entries),
        "raw_sidecar_chunk_summary": {"files": raw_chunk_count, "bytes": raw_chunk_bytes, "directories": raw_chunk_dirs},
        "total_files_including_raw_chunks": len(entries) + raw_chunk_count,
        "total_bytes": sum(int(item["bytes"]) for item in entries) + raw_chunk_bytes,
        "manifest_hash": canonical_hash({"files": entries, "raw_sidecar_chunk_summary": {"files": raw_chunk_count, "bytes": raw_chunk_bytes, "directories": raw_chunk_dirs}}),
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _info(dataset_root: Path) -> dict[str, Any]:
    value = _load_json(dataset_root / "meta/info.json")
    return value if value is not None else {}


def _data_files(dataset_root: Path) -> list[Path]:
    return sorted((dataset_root / "data").rglob("*.parquet"))


def _episode_files(dataset_root: Path) -> list[Path]:
    return sorted((dataset_root / "meta/episodes").rglob("*.parquet"))


def _flatten_arrow(column: Any, dtype: np.dtype) -> np.ndarray:
    values = column.to_pylist()
    array = np.asarray(values, dtype=dtype)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return array


def _scalar_arrow(column: Any, dtype: np.dtype) -> np.ndarray:
    return np.asarray(column.to_numpy(zero_copy_only=False), dtype=dtype).reshape(-1)


def _stats(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(array)
    valid = array[finite]
    if valid.size == 0:
        return {"count": int(array.size), "finite_count": 0, "nan_inf_count": int(array.size)}
    return {
        "count": int(array.size),
        "finite_count": int(valid.size),
        "nan_inf_count": int(array.size - valid.size),
        "min": float(np.min(valid)),
        "median": float(np.median(valid)),
        "p95": float(np.percentile(valid, 95)),
        "max": float(np.max(valid)),
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
    }


def _timestamp_unit(field: str) -> dict[str, Any]:
    if field == "robot_timestamp":
        return {"source": "time.time() in FlexivDualArm", "unit": "seconds", "seconds_scale": 1.0}
    if field.endswith("_rgbd_timestamp"):
        return {
            "source": "RealSense color_frame.get_timestamp(), persisted without conversion",
            "unit": "milliseconds",
            "seconds_scale": 1e-3,
        }
    if field == "timestamp":
        return {"source": "LeRobot episode clock", "unit": "seconds", "seconds_scale": 1.0}
    return {"source": "not a timestamp", "unit": None, "seconds_scale": None}


def _state_schema(info: dict[str, Any]) -> dict[str, Any]:
    feature = info.get("features", {}).get("observation.state", {})
    names = list(feature.get("names") or [])
    dimension = int((feature.get("shape") or [0])[0]) if feature.get("shape") else 0
    declared = info.get("robot_state_schema")
    if declared:
        name = declared.get("state_schema", declared.get("name", "declared"))
    elif dimension == 28 and any("ee_pose.r" in name for name in names):
        name = "legacy_state_28d_absolute_rotvec"
    elif dimension == 34 and any("rotation_6d" in name for name in names):
        name = "flexiv_abs_rot6d_v2"
    elif dimension == 48:
        name = "raw_force_48d_or_other_48d"
    else:
        name = f"unannotated_state_{dimension}d"
    return {"name": name, "dimension": dimension, "names": names, "declared_metadata": declared}


def _action_schema(info: dict[str, Any]) -> dict[str, Any]:
    feature = info.get("features", {}).get("action", {})
    dimension = int((feature.get("shape") or [0])[0]) if feature.get("shape") else 0
    return {"dimension": dimension, "names": list(feature.get("names") or [])}


def _episode_metadata(dataset_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    episodes: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in _episode_files(dataset_root):
        try:
            table = pq.read_table(path)
        except Exception as exc:  # pragma: no cover - corrupt parquet depends on engine
            errors.append(f"cannot read {path.relative_to(dataset_root)}: {exc}")
            continue
        rows = table.to_pylist()
        for row in rows:
            clean = jsonable(row)
            clean["metadata_source"] = str(path.relative_to(dataset_root))
            clean["episode_provenance_hash"] = canonical_hash(clean)
            episodes.append(clean)
    episodes.sort(key=lambda row: int(row.get("episode_index", -1)))
    return episodes, errors


def _video_paths(dataset_root: Path) -> dict[str, list[Path]]:
    output: dict[str, list[Path]] = {}
    videos = dataset_root / "videos"
    if not videos.is_dir():
        return output
    for path in sorted(videos.glob("**/*.mp4")):
        key = path.relative_to(videos).parts[0]
        output.setdefault(key, []).append(path)
    return output


def _probe_video(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "bytes": path.stat().st_size, "frame_count": None}
    try:
        import av  # noqa: PLC0415

        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            count = sum(1 for _ in container.decode(stream))
            result.update({"frame_count": count, "width": stream.codec_context.width, "height": stream.codec_context.height, "average_rate": str(stream.average_rate)})
    except Exception as exc:  # pragma: no cover - codec availability is environment-specific
        result["probe_error"] = str(exc)
    return result


def _audit_calibration(dataset_root: Path, manifest: dict[str, Any] | None) -> dict[str, Any]:
    calibration_path = dataset_root / "meta/realsense_calibration.json"
    result: dict[str, Any] = {"status": "FAIL", "path": str(calibration_path), "cameras": {}, "errors": []}
    if not calibration_path.is_file():
        result["errors"].append("missing calibration manifest")
        return result
    calibration = _load_json(calibration_path)
    if calibration is None:
        result["errors"].append("calibration manifest is not valid JSON")
        return result
    result["sha256"] = sha256_file(calibration_path)
    expected = (manifest or {}).get("calibration", {}).get("sha256")
    result["manifest_sha256"] = expected
    if expected and expected != result["sha256"]:
        result["errors"].append("calibration manifest SHA-256 mismatch")
    camera_payloads = calibration.get("cameras", {})
    for camera in CAMERAS:
        logical = f"{camera}_rgb"
        payload = camera_payloads.get(logical)
        if not isinstance(payload, dict):
            result["errors"].append(f"missing calibration for {logical}")
            continue
        streams = payload.get("streams", {})
        missing_streams = [name for name in ("color", "depth", "infrared1", "infrared2") if name not in streams]
        serial = payload.get("device", {}).get("serial")
        scale = payload.get("depth_scale_m_per_unit")
        if missing_streams or serial is None or scale is None:
            result["errors"].append(f"incomplete calibration for {logical}: missing={missing_streams}")
        refs = {name: stream.get("intrinsics", {}) for name, stream in streams.items()}
        result["cameras"][camera] = {
            "logical_camera": logical,
            "serial": serial,
            "depth_scale_m_per_unit": scale,
            "streams": {name: {key: stream.get(key) for key in ("width", "height", "fps", "format")} for name, stream in streams.items()},
            "intrinsics_present": {name: bool(value.get("K") or value.get("fx")) for name, value in refs.items()},
            "extrinsics_present": sorted(payload.get("extrinsics", {}).keys()),
        }
        calibration_file = payload.get("calibration_file")
        if calibration_file:
            referenced = dataset_root / "meta/calibration" / Path(calibration_file).name
            if not referenced.is_file():
                result["errors"].append(f"missing per-camera calibration file for {logical}: {referenced.name}")
            else:
                result["cameras"][camera]["calibration_file"] = str(referenced)
                result["cameras"][camera]["calibration_file_sha256"] = sha256_file(referenced)
    result["status"] = "PASS" if not result["errors"] else "FAIL"
    return result


def _stream_array_stats(array: Any, *, chunk_frames: int, positive_only: bool = False) -> dict[str, Any]:
    total = int(array.shape[0]) if len(array.shape) else 0
    finite = positive = 0
    sample_parts: list[np.ndarray] = []
    minimum = math.inf
    maximum = -math.inf
    for start in range(0, total, chunk_frames):
        block = np.asarray(array[start : min(total, start + chunk_frames)])
        flat = block.reshape(-1)
        finite_mask = np.isfinite(flat)
        finite += int(finite_mask.sum())
        positive += int((finite_mask & (flat > 0)).sum())
        valid = flat[finite_mask]
        if valid.size:
            minimum = min(minimum, float(np.min(valid)))
            maximum = max(maximum, float(np.max(valid)))
            stride = max(1, valid.size // 2048)
            sample_parts.append(valid[::stride][:2048].astype(np.float64, copy=False))
    sample = np.concatenate(sample_parts) if sample_parts else np.empty(0, dtype=np.float64)
    if sample.size:
        values = sample[sample > 0] if positive_only else sample
        quantiles = {"min": float(np.min(values)), "median": float(np.median(values)), "p95": float(np.percentile(values, 95)), "max": float(np.max(values))} if values.size else {}
    else:
        quantiles = {}
    return {
        "frames": total,
        "dtype": str(array.dtype),
        "shape": [int(v) for v in array.shape],
        "finite_count": finite,
        "finite_ratio": finite / max(1, int(np.prod(array.shape))),
        "positive_count": positive,
        "positive_ratio": positive / max(1, int(np.prod(array.shape))),
        "sample_quantiles": quantiles,
        "global_min": None if minimum is math.inf else minimum,
        "global_max": None if maximum == -math.inf else maximum,
        "quantiles_method": "deterministic stratified sample per chunk",
    }


def _audit_sensor(dataset_root: Path, info: dict[str, Any], manifest: dict[str, Any] | None, total_frames: int, chunk_frames: int) -> tuple[dict[str, Any], Any | None, list[str]]:
    errors: list[str] = []
    features = info.get("features", {})
    videos = _video_paths(dataset_root)
    cameras: dict[str, Any] = {}
    try:
        import zarr  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        zarr = None
        errors.append(f"zarr unavailable: {exc}")
    reader = None
    sidecar_path = dataset_root / "sidecars/realsense.zarr"
    if manifest is None or not (dataset_root / "meta/rgbd_sidecar.json").is_file():
        errors.append("raw-sidecar manifest missing")
    elif not sidecar_path.is_dir():
        errors.append("raw-sidecar Zarr store missing")
    elif zarr is not None:
        try:
            reader = zarr.open_group(str(sidecar_path), mode="r")
        except Exception as exc:
            errors.append(f"cannot open raw-sidecar Zarr read-only: {exc}")
    for camera in CAMERAS:
        rgb_key = f"observation.images.{camera}_rgb"
        legacy_key = f"observation.images.{camera}_image"
        declared = rgb_key if rgb_key in features else legacy_key if legacy_key in features else None
        video_key = rgb_key if rgb_key in videos else legacy_key if legacy_key in videos else rgb_key
        video_reports = [_probe_video(path) for path in videos.get(video_key, [])]
        camera_report: dict[str, Any] = {
            "rgb": {"declared_feature": declared, "video_files": video_reports, "frame_count": sum(int(item["frame_count"]) for item in video_reports if item.get("frame_count") is not None)},
            "depth": None,
            "left_ir": None,
            "right_ir": None,
        }
        if declared is None or not video_reports:
            errors.append(f"{camera}: RGB video missing")
        for modality in MODALITIES:
            if reader is None:
                continue
            try:
                array = reader[f"data/{camera}/{modality}"]
                camera_report[modality] = _stream_array_stats(array, chunk_frames=chunk_frames, positive_only=modality == "depth")
                camera_report[modality]["missing_frames"] = max(0, total_frames - int(array.shape[0]))
                camera_report[modality]["extra_frames"] = max(0, int(array.shape[0]) - total_frames)
                if int(array.shape[0]) != total_frames:
                    errors.append(f"{camera}/{modality}: frame count {array.shape[0]} != Parquet {total_frames}")
            except Exception as exc:
                errors.append(f"{camera}/{modality}: missing or unreadable Zarr array: {exc}")
        if reader is not None:
            try:
                reused = np.asarray(reader[f"data/{camera}/rgbd_reused"][:], dtype=bool)
                camera_report["reused_frames"] = {
                    "count": int(reused.sum()),
                    "ratio": float(reused.mean()) if reused.size else 0.0,
                    "frames": int(reused.size),
                }
            except Exception as exc:
                errors.append(f"{camera}: missing or unreadable rgbd_reused array: {exc}")
        if camera_report["depth"] is None:
            errors.append(f"{camera}: depth missing")
        cameras[camera] = camera_report
    if manifest is not None and manifest.get("status") != "complete":
        errors.append(f"raw-sidecar status is {manifest.get('status')!r}, not complete")
    sidecar_summary = {
        "path": str(dataset_root / "meta/rgbd_sidecar.json"),
        "present": manifest is not None,
        "status": (manifest or {}).get("status"),
        "relative_path": (manifest or {}).get("relative_path"),
        "committed_frames": (manifest or {}).get("committed_frames"),
        "committed_episodes": (manifest or {}).get("committed_episodes"),
        "calibration": (manifest or {}).get("calibration"),
    }
    for sensor_report in cameras.values():
        rgb_count = sensor_report.get("rgb", {}).get("frame_count")
        depth_count = (sensor_report.get("depth") or {}).get("frames")
        sensor_report["rgb_depth_count_match"] = bool(rgb_count is not None and depth_count is not None and rgb_count == depth_count)
    return {"storage": "zarr_v2" if reader is not None else None, "raw_sidecar_manifest": sidecar_summary, "cameras": cameras, "errors": errors}, reader, errors


def _read_main_rows(dataset_root: Path, columns: Iterable[str], batch_size: int) -> tuple[dict[str, np.ndarray], list[str], int]:
    files = _data_files(dataset_root)
    errors: list[str] = []
    values: dict[str, list[np.ndarray]] = defaultdict(list)
    row_count = 0
    for path in files:
        try:
            parquet = pq.ParquetFile(path)
            available = [column for column in columns if column in parquet.schema_arrow.names]
            missing = [column for column in columns if column not in available]
            if missing:
                errors.append(f"{path.relative_to(dataset_root)} missing columns: {missing}")
            for batch in parquet.iter_batches(batch_size=batch_size, columns=available):
                row_count += batch.num_rows
                for name in available:
                    column = batch.column(batch.schema.get_field_index(name))
                    if name in {"action", "observation.state"}:
                        values[name].append(_flatten_arrow(column, np.dtype(np.float64)))
                    else:
                        values[name].append(_scalar_arrow(column, np.dtype(bool if name.endswith("_reused") else np.float64 if name == "robot_timestamp" or name.endswith("_timestamp") or name == "timestamp" else np.int64)))
        except Exception as exc:
            errors.append(f"cannot read {path.relative_to(dataset_root)}: {exc}")
    joined = {name: np.concatenate(parts) if parts else np.empty((0, 1) if name in {"action", "observation.state"} else 0) for name, parts in values.items()}
    return joined, errors, row_count


def _summary_per_dimension(values: np.ndarray, names: list[str]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    finite = np.isfinite(array).all(axis=1) if array.size else np.ones(0, dtype=bool)
    dimensions: list[dict[str, Any]] = []
    for index in range(array.shape[1] if array.ndim == 2 else 0):
        dimensions.append({"index": index, "name": names[index] if index < len(names) else f"dim_{index}", **_stats(array[:, index])})
    return {"dimension": int(array.shape[1]) if array.ndim == 2 else 0, "row_count": int(array.shape[0]) if array.ndim == 2 else 0, "all_rows_finite": bool(finite.all()), "dimensions": dimensions}


def _action_coverage(actions: np.ndarray, episodes: np.ndarray, action_names: list[str]) -> dict[str, Any]:
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2:
        values = values.reshape(-1, 1)
    finite_rows = np.isfinite(values).all(axis=1) if values.size else np.empty(0, dtype=bool)
    finite_values = values[finite_rows]
    finite_episodes = np.asarray(episodes)[finite_rows] if len(episodes) == len(values) else np.empty(0, dtype=np.int64)
    result: dict[str, Any] = {"proxy_name": "action_coverage_proxy", "nonfinite_rows": int(len(values) - len(finite_values)), "action_covariance_eigenvalues": [], "near_zero_dimensions": [], "per_arm": {}, "gripper_transitions": {}, "horizon_chunk_variance": {}}
    if finite_values.shape[0] > 1:
        covariance = np.cov(finite_values, rowvar=False)
        covariance = np.atleast_2d(covariance)
        result["action_covariance_eigenvalues"] = np.maximum(np.linalg.eigvalsh(covariance), 0.0).tolist()
    result["near_zero_dimensions"] = [int(i) for i, std in enumerate(np.std(finite_values, axis=0)) if float(std) <= 1e-8] if finite_values.size else []
    for label, start in (("left", 0), ("right", 6)):
        translation = np.linalg.norm(finite_values[:, start : start + 3], axis=1) if finite_values.shape[1] >= start + 3 else np.zeros(finite_values.shape[0])
        rotation = np.linalg.norm(finite_values[:, start + 3 : start + 6], axis=1) if finite_values.shape[1] >= start + 6 else np.zeros(finite_values.shape[0])
        result["per_arm"][label] = {
            "translation_norm": _stats(translation),
            "rotation_norm": _stats(rotation),
            "active_rate_threshold_1e-6": float(np.mean((translation + rotation) > 1e-6)) if finite_values.shape[0] else 0.0,
            "static_proportion_threshold_1e-6": float(np.mean((translation + rotation) <= 1e-6)) if finite_values.shape[0] else 1.0,
        }
    for index, label in ((12, "left"), (13, "right")):
        if finite_values.shape[1] > index:
            series = finite_values[:, index]
            result["gripper_transitions"][label] = {"count": int(np.count_nonzero(np.diff(series) != 0)), "unique_values": np.unique(series).tolist()}
    for horizon in (2, 4, 8):
        chunks: list[np.ndarray] = []
        for episode in sorted(set(finite_episodes.tolist())) if finite_episodes.size else []:
            episode_values = finite_values[finite_episodes == episode]
            for start in range(0, max(0, len(episode_values) - horizon + 1)):
                chunks.append(np.mean(episode_values[start : start + horizon], axis=0))
        chunk_array = np.asarray(chunks)
        result["horizon_chunk_variance"][str(horizon)] = {"windows": int(len(chunk_array)), "mean_variance": float(np.mean(np.var(chunk_array, axis=0))) if len(chunk_array) else 0.0, "per_dimension": np.var(chunk_array, axis=0).tolist() if len(chunk_array) else []}
    result["limited_coverage"] = any(item["static_proportion_threshold_1e-6"] > 0.95 for item in result["per_arm"].values()) or bool(result["near_zero_dimensions"])
    return result


def _timestamp_audit(main: dict[str, np.ndarray], reader: Any | None, total_frames: int, *, chunk_frames: int) -> dict[str, Any]:
    errors: list[str] = []
    fields: dict[str, Any] = {}
    for field in JOIN_FIELDS:
        if field not in main:
            errors.append(f"missing timestamp/join field {field}")
    for field in ("robot_timestamp", "timestamp", *[f"{camera}_rgbd_timestamp" for camera in CAMERAS]):
        if field in main:
            values = np.asarray(main[field], dtype=np.float64)
            if field == "timestamp" and "episode_index" in main:
                episodes = np.asarray(main["episode_index"], dtype=np.int64)
                non_monotonic = sum(int(np.count_nonzero(np.diff(values[episodes == episode]) < 0)) for episode in sorted(set(episodes.tolist())))
            else:
                non_monotonic = int(np.count_nonzero(np.diff(values) < 0))
            fields[field] = {"units": _timestamp_unit(field), "count": int(len(values)), "expected_count": int(total_frames), "count_mismatch": len(values) != total_frames, "stats": _stats(values), "non_monotonic_count": non_monotonic, "missing_count": int(np.count_nonzero(~np.isfinite(values)))}
            if fields[field]["count_mismatch"]:
                errors.append(f"{field} count mismatch: {len(values)} != {total_frames}")
            if fields[field]["non_monotonic_count"]:
                errors.append(f"{field} non-monotonic timestamps")
            if fields[field]["missing_count"]:
                errors.append(f"{field} contains missing/non-finite timestamps")
    indices = {field: np.asarray(main[field]) for field in JOIN_FIELDS if field in main}
    index_audit: dict[str, Any] = {}
    for field, values in indices.items():
        duplicate_count = int(len(values) - len(np.unique(values))) if field in {"index", "global_frame_index"} else 0
        non_monotonic_count = int(np.count_nonzero(np.diff(values) < 0)) if field in {"index", "global_frame_index", "episode_index"} else 0
        index_audit[field] = {"count": int(len(values)), "duplicate_count": duplicate_count, "non_monotonic_count": non_monotonic_count}
        if duplicate_count or non_monotonic_count:
            errors.append(f"{field} duplicate or non-monotonic")
    if reader is not None:
        for field in ("index", "episode_index", "frame_index", "global_frame_index", "robot_timestamp"):
            path = f"meta/{field}"
            try:
                zarr_values = np.asarray(reader[path][:])
                if field in main and not np.array_equal(zarr_values, main[field]):
                    errors.append(f"Parquet/Zarr join mismatch for {field}")
            except Exception as exc:
                errors.append(f"missing Zarr join array {path}: {exc}")
        for camera in CAMERAS:
            for suffix in ("rgbd_timestamp", "rgbd_reused"):
                path = f"data/{camera}/{suffix}"
                field = f"{camera}_{suffix}"
                try:
                    zarr_values = np.asarray(reader[path][:])
                    if field in main and not np.array_equal(zarr_values, main[field]):
                        errors.append(f"Parquet/Zarr join mismatch for {field}")
                except Exception as exc:
                    errors.append(f"missing Zarr join array {path}: {exc}")
    normalized: dict[str, np.ndarray] = {}
    for field in ("robot_timestamp", *[f"{camera}_rgbd_timestamp" for camera in CAMERAS]):
        if field in main:
            normalized[field] = np.asarray(main[field], dtype=np.float64) * float(_timestamp_unit(field)["seconds_scale"])
    pairwise: dict[str, Any] = {}
    if "robot_timestamp" in normalized:
        for camera in CAMERAS:
            field = f"{camera}_rgbd_timestamp"
            if field in normalized:
                pairwise[camera] = _stats(normalized[field] - normalized["robot_timestamp"])
    if all(f"{camera}_rgbd_timestamp" in normalized for camera in CAMERAS):
        stacked = np.stack([normalized[f"{camera}_rgbd_timestamp"] for camera in CAMERAS], axis=1)
        pairwise["camera_internal_skew"] = _stats(np.max(stacked, axis=1) - np.min(stacked, axis=1))
    episode_values = np.asarray(main.get("episode_index", np.empty(0)), dtype=np.int64)
    frame_values = np.asarray(main.get("frame_index", np.empty(0)), dtype=np.int64)
    if len(episode_values) == len(frame_values):
        expected = np.concatenate([np.arange(int(np.count_nonzero(episode_values == episode)), dtype=np.int64) for episode in sorted(set(episode_values.tolist()))]) if len(episode_values) else np.empty(0, dtype=np.int64)
        if len(expected) == len(frame_values) and not np.array_equal(expected, frame_values):
            errors.append("frame_index is not a per-episode 0..N-1 sequence")
    return {"status": "PASS" if not errors else "FAIL", "fields": fields, "indices": index_audit, "robot_camera_difference_seconds": pairwise, "errors": errors, "timestamp_unit_source": "derived from the actual recorder code paths; values were not unit-converted in source data"}


def audit_dataset(dataset_root: Path, *, chunk_frames: int = DEFAULT_CHUNK_FRAMES) -> dict[str, Any]:
    dataset_root = dataset_root.expanduser().resolve()
    info_path = dataset_root / "meta/info.json"
    report: dict[str, Any] = {"dataset_root": str(dataset_root), "repo_id": dataset_root.name, "audit_started_at": utc_now(), "read_only": True, "errors": [], "warnings": [], "reason_codes": []}
    if not info_path.is_file():
        report.update({"stage2_status": "RECOLLECT", "reason_codes": ["dataset_metadata_missing"], "errors": [f"missing {info_path}"]})
        return report
    info = _info(dataset_root)
    report["identity"] = {"absolute_path": str(dataset_root), "repo_id": dataset_root.name, "robot_type": info.get("robot_type"), "total_frames_metadata": info.get("total_frames"), "total_episodes_metadata": info.get("total_episodes"), "fps": info.get("fps"), "dataset_version": info.get("codebase_version"), "metadata_mtime": datetime.fromtimestamp(info_path.stat().st_mtime, timezone.utc).isoformat(), "source_state_schema": _state_schema(info), "action_schema": _action_schema(info)}
    source_manifest = build_source_manifest(dataset_root)
    report["identity"].update({"manifest_hash": source_manifest["manifest_hash"], "source_tree_bytes": source_manifest["total_bytes"], "source_file_count": source_manifest["total_files"]})
    episodes, episode_errors = _episode_metadata(dataset_root)
    report["episodes"] = {"count": len(episodes), "items": [{key: value for key, value in episode.items() if key not in {"stats/action/min", "stats/action/max", "stats/action/mean", "stats/action/std", "stats/observation.state/min", "stats/observation.state/max", "stats/observation.state/mean", "stats/observation.state/std"}} for episode in episodes], "episode_metadata_hash": canonical_hash(episodes)}
    report["errors"].extend(episode_errors)
    data_files = _data_files(dataset_root)
    parquet_rows = sum(int(pq.ParquetFile(path).metadata.num_rows) for path in data_files) if data_files else 0
    report["identity"]["total_frames_parquet"] = parquet_rows
    report["identity"]["data_file_count"] = len(data_files)
    if not episodes and parquet_rows == 0 and int(info.get("total_frames", 0) or 0) == 0:
        report["reason_codes"].append("empty_dataset")
    if int(info.get("total_frames", parquet_rows) or 0) != parquet_rows:
        report["errors"].append(f"meta total_frames={info.get('total_frames')} != Parquet rows={parquet_rows}")
        report["reason_codes"].append("corrupted_episode_boundaries")
    expected_episode_frames = sum(int(item.get("length", 0) or 0) for item in episodes)
    if expected_episode_frames != parquet_rows or (episodes and [int(item.get("episode_index", -1)) for item in episodes] != list(range(len(episodes)))):
        report["errors"].append(f"episode metadata lengths/indexes do not match Parquet rows: metadata={expected_episode_frames}, parquet={parquet_rows}")
        report["reason_codes"].append("corrupted_episode_boundaries")
    manifest = _load_json(dataset_root / "meta/rgbd_sidecar.json")
    report["calibration"] = _audit_calibration(dataset_root, manifest)
    if report["calibration"]["status"] != "PASS":
        report["reason_codes"].append("missing_calibration" if not (dataset_root / "meta/realsense_calibration.json").is_file() else "calibration_mismatch")
    sensor, reader, sensor_errors = _audit_sensor(dataset_root, info, manifest, parquet_rows, chunk_frames)
    report["sensors"] = sensor
    if sensor_errors:
        report["errors"].extend(sensor_errors)
        if any("depth" in error for error in sensor_errors):
            report["reason_codes"].append("no_depth")
        if any("RGB" in error for error in sensor_errors):
            report["reason_codes"].append("no_rgb")
        if any("frame count" in error for error in sensor_errors):
            report["reason_codes"].append("rgbd_frame_count_mismatch")
    columns = [*JOIN_FIELDS, "timestamp", "action", "observation.state", *[f"{camera}_rgbd_timestamp" for camera in CAMERAS], *[f"{camera}_rgbd_reused" for camera in CAMERAS]]
    main, main_errors, _ = _read_main_rows(dataset_root, columns, batch_size=max(64, chunk_frames))
    report["errors"].extend(main_errors)
    state_schema = _state_schema(info)
    action_schema = _action_schema(info)
    state = main.get("observation.state", np.empty((0, state_schema["dimension"])))
    action = main.get("action", np.empty((0, action_schema["dimension"])))
    report["state_action"] = {"state": _summary_per_dimension(state, state_schema["names"]), "action": _summary_per_dimension(action, action_schema["names"]), "rotation_representation": "legacy absolute rotvec" if "legacy" in state_schema["name"] else "declared/unknown", "clipped_saturated_proportion": "NOT_ASSESSED: no reliable actuator bounds are persisted"}
    if report["state_action"]["state"]["all_rows_finite"] is False or report["state_action"]["action"]["all_rows_finite"] is False:
        report["errors"].append("state/action contains NaN or Inf")
        report["reason_codes"].append("state_action_invalid")
    if "observation.state" not in main or "action" not in main:
        report["reason_codes"].append("state_action_invalid")
    episodes_array = np.asarray(main.get("episode_index", np.empty(0)), dtype=np.int64)
    report["timestamps"] = _timestamp_audit(main, reader, parquet_rows, chunk_frames=chunk_frames)
    report["errors"].extend(report["timestamps"]["errors"])
    if report["timestamps"]["errors"]:
        report["reason_codes"].append("irrecoverable_timestamp_alignment")
    report["action_coverage"] = _action_coverage(action, episodes_array, action_schema["names"])
    for camera, sensor_report in report["sensors"].get("cameras", {}).items():
        depth_report = sensor_report.get("depth") or {}
        if depth_report.get("positive_ratio", 1.0) < 0.99:
            report["warnings"].append(f"{camera} depth contains zero/non-positive pixels")
            report["reason_codes"].append("partial_depth_missing")
    if report["action_coverage"]["limited_coverage"]:
        report["warnings"].append("action coverage proxy indicates a static arm or near-zero dimensions")
        report["reason_codes"].append("limited_action_coverage")
    if info.get("robot_type") == "flexiv_dual_arm" and state_schema["name"] != "flexiv_abs_rot6d_raw_force_v3":
        report["warnings"].append("legacy state requires an explicit offline conversion before Stage 2 policy use")
        report["reason_codes"].append("legacy_state_requires_explicit_conversion")
    report["tracker_feasibility_status"] = "HUMAN_REVIEW_REQUIRED"
    report["warnings"].append("object visibility/tracking/pose quality were not evaluated by a tracker in Stage 1")
    report["reason_codes"].append("object_visibility_human_review_required")
    report["stage2_status"] = "RECOLLECT" if report["errors"] or "empty_dataset" in report["reason_codes"] else "CONDITIONAL_PASS" if report["warnings"] or report["reason_codes"] else "PASS"
    if report["stage2_status"] == "RECOLLECT" and not report["errors"]:
        report["reason_codes"].append("unusable_rgbd")
    report["reason_codes"] = sorted(set(report["reason_codes"]))
    report["errors"] = sorted(set(report["errors"]))
    report["warnings"] = sorted(set(report["warnings"]))
    report["audit_completed_at"] = utc_now()
    return jsonable(report)


def audit_root(root: Path, *, chunk_frames: int = DEFAULT_CHUNK_FRAMES) -> dict[str, Any]:
    datasets = discover_datasets(root)
    return {"schema_version": 1, "audit_type": "paper_a_stage1_real_dataset_audit", "raw_root": str(root.expanduser().resolve()), "read_only": True, "generated_at": utc_now(), "datasets": [audit_dataset(dataset, chunk_frames=chunk_frames) for dataset in datasets], "dataset_count": len(datasets)}


def markdown_report(payload: dict[str, Any]) -> str:
    lines = ["# Paper A Stage 1 Dataset Audit", "", f"- Raw root: `{payload['raw_root']}`", f"- Generated: `{payload['generated_at']}`", f"- Dataset count: `{payload['dataset_count']}`", "- Raw source mutation: `not performed`", "", "## Stage 0 baseline", "", "The repository/branch/submodule/acquisition contract is frozen in [BASELINE_SNAPSHOT.md](../BASELINE_SNAPSHOT.md) and [ACQUISITION_DATA_CONTRACT.md](../ACQUISITION_DATA_CONTRACT.md). The Stage 0 machine-readable snapshot is [stage0_baseline_snapshot.json](stage0_baseline_snapshot.json).", "", "## Summary", "", "| Dataset | Frames | Episodes | State | Sensor | Timestamp | Calibration | Stage 2 | Reasons |", "|---|---:|---:|---|---|---|---|---|---|"]
    for item in payload["datasets"]:
        identity = item.get("identity", {})
        lines.append(f"| `{item.get('repo_id')}` | {identity.get('total_frames_parquet', '?')} | {item.get('episodes', {}).get('count', '?')} | {identity.get('source_state_schema', {}).get('name', '?')} | {'FAIL' if item.get('sensors', {}).get('errors') else 'PASS'} | {item.get('timestamps', {}).get('status', '?')} | {item.get('calibration', {}).get('status', '?')} | **{item.get('stage2_status', '?')}** | {', '.join(item.get('reason_codes', [])) or '-'} |")
    for item in payload["datasets"]:
        lines.extend(["", f"## `{item.get('repo_id')}`", "", f"- Source: `{item.get('dataset_root')}`", f"- Manifest hash: `{item.get('identity', {}).get('manifest_hash', '?')}`", f"- Tracker feasibility: `{item.get('tracker_feasibility_status', '?')}`", f"- State/action: `{item.get('state_action', {}).get('state', {}).get('dimension', '?')}D / {item.get('state_action', {}).get('action', {}).get('dimension', '?')}D`", f"- Reason codes: {', '.join(f'`{x}`' for x in item.get('reason_codes', [])) or 'none'}"])
        lines.append("\n### Sensor completeness\n")
        lines.append("| Camera | RGB frames | Depth positive ratio | Reused frames | Left IR | Right IR |\n|---|---:|---:|---:|---:|---:|")
        for camera, sensor in item.get("sensors", {}).get("cameras", {}).items():
            depth = sensor.get("depth") or {}
            lines.append(f"| `{camera}` | {sensor.get('rgb', {}).get('frame_count', '?')} | {depth.get('positive_ratio', '?')} | {sensor.get('reused_frames', {}).get('count', '?')} | {sensor.get('left_ir', {}).get('frames', '?')} | {sensor.get('right_ir', {}).get('frames', '?')} |")
        lines.extend(["", "### Timestamp / calibration / action coverage", "", f"- Timestamp status: `{item.get('timestamps', {}).get('status', '?')}`; robot-camera differences are reported in seconds after applying the source-code units.", f"- Calibration status: `{item.get('calibration', {}).get('status', '?')}`; manifest/file hashes are checked.", f"- Action coverage proxy: `{item.get('action_coverage', {}).get('proxy_name', '?')}`; no effect rank is claimed.", "", "### Human review boundary", "", "Stage 1 does not run a tracker or synthesize phase labels. Object visibility, occlusion, table geometry, and suitability for geometric segmentation remain human review items."])
    candidates = [item for item in payload["datasets"] if item.get("stage2_status") in {"PASS", "CONDITIONAL_PASS"}]
    candidates.sort(key=lambda item: (-int(item.get("identity", {}).get("source_state_schema", {}).get("dimension", 0)), item.get("repo_id", "")))
    lines.extend(["", "## Recommended Stage 2 priority", "", "The first human-review candidate is **`pick_place_20260717_v01`** because its audited 34D rotation-6D state is closer to the current 48D Flexiv acquisition contract than the 28D absolute-rotvec `pick_place_20260713_v05`. This is a workflow priority, not a scientific success claim; both still require explicit offline schema conversion, action-coverage review, depth review, and human object-visibility review.", "", "| Priority | Dataset | Gate state | Entry condition |", "|---:|---|---|---|"])
    for index, item in enumerate(candidates, start=1):
        lines.append(f"| {index} | `{item.get('repo_id')}` | `{item.get('stage2_status')}` | resolve `{', '.join(item.get('reason_codes', []))}` |")
    lines.extend(["", "## Suggested supplemental collection", "", "No dataset meets the hard `RECOLLECT` gate: RGB/video counts, Zarr depth/IR counts, calibration hashes, and timestamp joins passed. Targeted supplemental collection is nevertheless recommended for both datasets if human review confirms unusable views or if the Paper A protocol requires non-zero depth coverage and balanced bimanual action coverage. The audit evidence does not authorize relabeling or repairing the existing raw source.", "", "## Unresolved issues", "", "- Convert legacy 28D/34D data explicitly to the current Flexiv 48D contract, or record a new native-48D set; do not pad or rewrite raw data.", "- Human-review the 50-frame-per-dataset preview sets for object visibility, occlusion, table geometry, and geometric-segmentation suitability.", "- Investigate partial/non-positive depth pixels and decide whether they meet the Stage 2 depth policy.", "- Decide whether the static-arm/action-coverage proxy is acceptable for Paper A or requires targeted supplemental demonstrations.", "- Stage 2 tracker, pose quality, object effect, and phase semantics remain unevaluated."])
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Raw root or one dataset root.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--chunk-frames", type=int, default=DEFAULT_CHUNK_FRAMES)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = audit_root(args.root, chunk_frames=args.chunk_frames)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown_report(payload), encoding="utf-8")
    print(json.dumps({"datasets": payload["dataset_count"], "output_json": str(args.output_json), "output_md": str(args.output_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

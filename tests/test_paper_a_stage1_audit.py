"""Synthetic, read-only coverage for Paper A Stage 1 audit contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.tools.paper_a.audit_real_dataset import audit_dataset, discover_datasets


def _write_dataset(
    root: Path,
    *,
    state_dim: int = 2,
    state_names: list[str] | None = None,
    action_dim: int = 2,
    rows: int = 3,
    robot_timestamps: list[float] | None = None,
    state_values: list[list[float]] | None = None,
    action_values: list[list[float]] | None = None,
    total_frames: int | None = None,
    episode_lengths: list[int] | None = None,
    calibration_hash: str | None = None,
    with_calibration: bool = False,
) -> Path:
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    if state_names is None:
        state_names = [f"state_{i}" for i in range(state_dim)]
    info = {
        "codebase_version": "v3.0",
        "robot_type": "flexiv_dual_arm",
        "total_episodes": 1,
        "total_frames": rows if total_frames is None else total_frames,
        "fps": 30,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [state_dim], "names": state_names},
            "action": {"dtype": "float32", "shape": [action_dim], "names": [f"action_{i}" for i in range(action_dim)]},
            "global_frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "robot_timestamp": {"dtype": "float64", "shape": [1], "names": None},
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    if episode_lengths is not None:
        length = episode_lengths[0]
    else:
        length = rows
    pq.write_table(
        pa.table(
            {
                "index": pa.array(range(rows), type=pa.int64()),
                "episode_index": pa.array([0] * rows, type=pa.int64()),
                "frame_index": pa.array(range(rows), type=pa.int64()),
                "global_frame_index": pa.array(range(rows), type=pa.int64()),
                "robot_timestamp": pa.array(robot_timestamps or [1000.0 + i for i in range(rows)], type=pa.float64()),
                "observation.state": pa.array(state_values or [[float(i)] * state_dim for i in range(rows)], type=pa.list_(pa.float32())),
                "action": pa.array(action_values or [[float(i)] * action_dim for i in range(rows)], type=pa.list_(pa.float32())),
            }
        ),
        root / "data/chunk-000/file-000.parquet",
    )
    pq.write_table(
        pa.table({"episode_index": [0], "length": [length], "dataset_from_index": [0], "dataset_to_index": [length]}),
        root / "meta/episodes/chunk-000/file-000.parquet",
    )
    if with_calibration:
        calibration = {"cameras": {camera + "_rgb": {"device": {"serial": str(i)}, "depth_scale_m_per_unit": 0.001, "streams": {name: {"width": 2, "height": 2, "fps": 30, "intrinsics": {"K": [[1]]}} for name in ("color", "depth", "infrared1", "infrared2")}} for i, camera in enumerate(("head", "left_wrist", "right_wrist"))}}
        calibration_path = root / "meta/realsense_calibration.json"
        calibration_path.write_text(json.dumps(calibration))
        expected = calibration_hash or hashlib.sha256(calibration_path.read_bytes()).hexdigest()
        (root / "meta/rgbd_sidecar.json").write_text(json.dumps({"calibration": {"relative_path": "meta/realsense_calibration.json", "sha256": expected}}))
    return root


def test_dataset_detection_ignores_ordinary_directories(tmp_path):
    valid = _write_dataset(tmp_path / "nested" / "valid")
    (tmp_path / "ordinary").mkdir()
    assert discover_datasets(tmp_path) == [valid.resolve()]


def test_missing_metadata_is_recollect(tmp_path):
    report = audit_dataset(tmp_path / "missing")
    assert report["stage2_status"] == "RECOLLECT"
    assert "dataset_metadata_missing" in report["reason_codes"]


def test_corrupted_frame_count_and_partial_episode_are_recollect(tmp_path):
    report = audit_dataset(_write_dataset(tmp_path / "partial", total_frames=4, episode_lengths=[2]))
    assert report["stage2_status"] == "RECOLLECT"
    assert "corrupted_episode_boundaries" in report["reason_codes"]


def test_non_monotonic_timestamp_is_reported(tmp_path):
    report = audit_dataset(_write_dataset(tmp_path / "timestamps", robot_timestamps=[1000.0, 1002.0, 1001.0]))
    assert report["timestamps"]["status"] == "FAIL"
    assert any("non-monotonic" in error for error in report["timestamps"]["errors"])


def test_calibration_hash_mismatch_is_reported(tmp_path):
    report = audit_dataset(_write_dataset(tmp_path / "calibration", with_calibration=True, calibration_hash="0" * 64))
    assert report["calibration"]["status"] == "FAIL"
    assert "calibration_mismatch" in report["reason_codes"]


def test_state_action_nan_is_reported(tmp_path):
    report = audit_dataset(_write_dataset(tmp_path / "nan", state_values=[[0.0, np.nan], [1.0, 2.0], [2.0, 3.0]], action_values=[[0.0, 0.0], [1.0, np.inf], [2.0, 2.0]]))
    assert "state_action_invalid" in report["reason_codes"]
    assert report["state_action"]["state"]["all_rows_finite"] is False
    assert report["state_action"]["action"]["all_rows_finite"] is False


def test_legacy_schema_is_explicitly_conditional(tmp_path):
    names = ["left_ee_pose.rx", *[f"legacy_{i}" for i in range(27)]]
    report = audit_dataset(_write_dataset(tmp_path / "legacy", state_dim=28, state_names=names))
    assert "legacy_state_requires_explicit_conversion" in report["reason_codes"]


def test_empty_dataset_is_recollect(tmp_path):
    root = tmp_path / "empty"
    root.mkdir(parents=True)
    (root / "meta").mkdir()
    (root / "data").mkdir()
    (root / "meta/info.json").write_text(json.dumps({"total_frames": 0, "total_episodes": 0, "robot_type": "flexiv_dual_arm", "features": {}}))
    report = audit_dataset(root)
    assert report["stage2_status"] == "RECOLLECT"
    assert "empty_dataset" in report["reason_codes"]


def test_audit_is_read_only(tmp_path):
    root = _write_dataset(tmp_path / "readonly")
    before = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("*") if path.is_file()}
    report = audit_dataset(root)
    after = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("*") if path.is_file()}
    assert report["read_only"] is True
    assert before == after

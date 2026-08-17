#!/usr/bin/env python3
"""Generate traceable RGB-D contact sheets without running object tracking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

from audit_real_dataset import CAMERAS, _data_files, _episode_metadata, _load_json, discover_datasets


def _rows(dataset_root: Path, indices: set[int]) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    offset = 0
    columns = ["index", "episode_index", "frame_index", "global_frame_index", "robot_timestamp", *[f"{camera}_rgbd_timestamp" for camera in CAMERAS]]
    for path in _data_files(dataset_root):
        parquet = pq.ParquetFile(path)
        count = parquet.metadata.num_rows
        wanted = sorted(index - offset for index in indices if offset <= index < offset + count)
        if wanted:
            table = parquet.read(columns=[column for column in columns if column in parquet.schema_arrow.names])
            for local in wanted:
                row = table.slice(local, 1).to_pylist()[0]
                row["_source_file"] = str(path)
                row["_source_file_row"] = local
                output[offset + local] = row
        offset += count
    return output


def _video_frames(dataset_root: Path, wanted: set[int]) -> dict[str, dict[int, np.ndarray]]:
    result: dict[str, dict[int, np.ndarray]] = {camera: {} for camera in CAMERAS}
    try:
        import av  # noqa: PLC0415
    except ImportError:
        return result
    for camera in CAMERAS:
        key = f"observation.images.{camera}_rgb"
        paths = sorted((dataset_root / "videos" / key).glob("**/*.mp4"))
        if not paths:
            continue
        target = set(wanted)
        ordinal = 0
        try:
            with av.open(str(paths[0])) as container:
                stream = container.streams.video[0]
                for frame in container.decode(stream):
                    if ordinal in target:
                        result[camera][ordinal] = frame.to_ndarray(format="rgb24")
                    ordinal += 1
                    if ordinal > max(target, default=-1):
                        break
        except Exception:
            continue
    return result


def _depth_frame(dataset_root: Path, camera: str, index: int) -> np.ndarray | None:
    manifest = _load_json(dataset_root / "meta/rgbd_sidecar.json")
    if not manifest:
        return None
    try:
        import zarr  # noqa: PLC0415

        store = dataset_root / manifest.get("relative_path", "sidecars/realsense.zarr")
        group = zarr.open_group(str(store), mode="r")
        return np.asarray(group[f"data/{camera}/depth"][index])
    except Exception:
        return None


def _depth_image(depth: np.ndarray | None) -> Image.Image:
    if depth is None:
        return Image.new("RGB", (320, 240), (80, 80, 80))
    values = depth.astype(np.float32)
    valid = np.isfinite(values) & (values > 0)
    if not np.any(valid):
        return Image.new("RGB", (values.shape[1], values.shape[0]), (80, 80, 80))
    low, high = np.percentile(values[valid], [1, 99])
    high = max(high, low + 1)
    normalized = np.clip((values - low) / (high - low), 0, 1)
    gray = (normalized * 255).astype(np.uint8)
    try:
        import cv2  # noqa: PLC0415

        return Image.fromarray(cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB))
    except Exception:
        return Image.fromarray(np.repeat(gray[..., None], 3, axis=-1))


def _fit(image: Image.Image, size: tuple[int, int] = (320, 240)) -> Image.Image:
    copy = image.convert("RGB").copy()
    copy.thumbnail(size)
    canvas = Image.new("RGB", size, "black")
    canvas.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return canvas


def generate_preview(dataset_root: Path, output_dir: Path, samples_per_episode: int = 5) -> dict[str, Any]:
    dataset_root = dataset_root.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes, errors = _episode_metadata(dataset_root)
    samples: list[dict[str, Any]] = []
    for item in episodes:
        length = int(item.get("length", 0) or 0)
        start = int(item.get("dataset_from_index", 0) or 0)
        if length <= 0:
            continue
        count = max(3, min(samples_per_episode, length))
        positions = sorted(set(np.linspace(0, length - 1, count, dtype=int).tolist()))
        for frame_index in positions:
            samples.append({"episode_index": int(item.get("episode_index", 0)), "episode_frame_index": frame_index, "dataset_index": start + frame_index})
    row_map = _rows(dataset_root, {item["dataset_index"] for item in samples})
    frames = _video_frames(dataset_root, {item["dataset_index"] for item in samples})
    manifest: dict[str, Any] = {"dataset_root": str(dataset_root), "tracker_feasibility_status": "HUMAN_REVIEW_REQUIRED", "object_tracking": "NOT_RUN", "phase_labels": "NOT_GENERATED", "samples_per_episode_requested": samples_per_episode, "samples": [], "contact_sheets": [], "errors": errors}
    for episode in sorted({item["episode_index"] for item in samples}):
        episode_samples = [item for item in samples if item["episode_index"] == episode]
        sheet = Image.new("RGB", (4 * 320, len(episode_samples) * 270), "black")
        draw = ImageDraw.Draw(sheet)
        labels = ["head RGB", "head depth", "left wrist RGB", "right wrist RGB"]
        for column, label in enumerate(labels):
            draw.text((column * 320 + 4, 2), label, fill="white")
        for row_index, item in enumerate(episode_samples):
            dataset_index = item["dataset_index"]
            row = row_map.get(dataset_index, {})
            rgb_head = frames["head"].get(dataset_index)
            rgb_left = frames["left_wrist"].get(dataset_index)
            rgb_right = frames["right_wrist"].get(dataset_index)
            images = [_fit(Image.fromarray(rgb_head) if rgb_head is not None else Image.new("RGB", (320, 240), "gray")), _fit(_depth_image(_depth_frame(dataset_root, "head", dataset_index))), _fit(Image.fromarray(rgb_left) if rgb_left is not None else Image.new("RGB", (320, 240), "gray")), _fit(Image.fromarray(rgb_right) if rgb_right is not None else Image.new("RGB", (320, 240), "gray"))]
            for column, image in enumerate(images):
                sheet.paste(image, (column * 320, row_index * 270 + 25))
            draw.text((4, row_index * 270 + 250), f"ep={episode} frame={item['episode_frame_index']} index={dataset_index}", fill="white")
            record = {**item, "robot_timestamp": row.get("robot_timestamp"), "camera_timestamps": {camera: row.get(f"{camera}_rgbd_timestamp") for camera in CAMERAS}, "source_file": row.get("_source_file"), "source_file_row": row.get("_source_file_row"), "rgb_frames_available": {camera: dataset_index in frames[camera] for camera in CAMERAS}, "depth_source": str(dataset_root / "sidecars/realsense.zarr/data/head/depth")}
            manifest["samples"].append(record)
        path = output_dir / f"episode_{episode:06d}_contact_sheet.png"
        sheet.save(path)
        manifest["contact_sheets"].append(str(path))
    manifest["point_cloud_snapshot"] = {"status": "NOT_GENERATED", "reason": "Stage 1 does not add a point-cloud algorithm; depth snapshots are retained for human review."}
    manifest["manifest_hash"] = __import__("hashlib").sha256(json.dumps(manifest["samples"], sort_keys=True, default=str).encode()).hexdigest()
    (output_dir / "preview_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="One dataset root or parent raw root.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--samples-per-episode", type=int, default=5)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    datasets = [root] if (root / "meta/info.json").is_file() else discover_datasets(root)
    for dataset in datasets:
        output = args.output_root / dataset.name
        manifest = generate_preview(dataset, output, args.samples_per_episode)
        print(f"{dataset.name}: {len(manifest['samples'])} samples, {len(manifest['contact_sheets'])} sheets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

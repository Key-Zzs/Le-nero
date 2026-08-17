# Paper A acquisition and raw-data contract

This contract is frozen from the current Le-nero code and the two existing
datasets under `/home/deepcybo/.cache/huggingface/lerobot/flexiv_dual_arm_3d`.
The source data remain read-only.

## Sensors and storage

Each valid dataset has three RealSense streams named `head`, `left_wrist`, and
`right_wrist`:

- RGB: `observation.images.{camera}_rgb`, 640x480 RGB, 30 FPS, normal LeRobot
  video files under `videos/`.
- Depth: native RealSense `uint16` under
  `/data/{camera}/depth`; values use the recorded RealSense depth units.
- IR: `uint8` under `/data/{camera}/left_ir` and
  `/data/{camera}/right_ir`.
- Per-camera scalar sidecar fields are
  `{camera}_rgbd_timestamp` and `{camera}_rgbd_reused` in the main Parquet
  timeline and mirrored in the Zarr sidecar.
- The authoritative raw sidecar is `meta/rgbd_sidecar.json` plus
  `sidecars/realsense.zarr`; it is Zarr v2, committed-prefix based, and joins
  row-for-row to the main Parquet data.

## Time and join fields

The Flexiv source writes `robot_timestamp = time.time()` as float64 seconds.
The RealSense camera wrapper writes `float(color_frame.get_timestamp())` with
no conversion; the RealSense SDK timestamp is milliseconds. The audit records
these source units explicitly and normalizes only derived difference statistics
to seconds. It does not rewrite source values.

The join contract is:

- `index`: main Parquet row ordinal and Zarr `/meta/index`;
- `episode_index`: episode identity;
- `frame_index`: zero-based frame position within an episode;
- `global_frame_index`: monotonic acquisition identity;
- `robot_timestamp`: robot-side clock;
- per-camera RGB-D timestamps and reused flags as listed above.

Episode boundaries come from `meta/episodes/**/*.parquet` (`length`,
`dataset_from_index`, `dataset_to_index`) and Zarr `/meta/episode_ends`.

## State, action, and calibration

The current Flexiv source contract is `flexiv_abs_rot6d_raw_force_v3`:

- `observation.state`: float32, 48D; 34D kinematic prefix with absolute
  rotation-6D (`matrix_columns_0_1`) plus 14 raw wrench/gripper-force fields;
- `action`: float32, 14D; left/right xyz + rotvec deltas followed by two
  gripper commands;
- state/action names and field order are persisted in `meta/info.json` for new
  recordings.

The audited existing datasets declare 28D legacy absolute-rotvec state and 14D
action. They are kept distinct and are not converted in place.

Calibration is stored in `meta/realsense_calibration.json`, with per-camera
files in `meta/calibration/`. The manifest stores the calibration SHA-256;
Stage 1 checks that hash, camera serial, depth scale, color/depth/IR stream
dimensions, intrinsics, and color/depth plus IR extrinsics where present.

## Provenance and immutability

`stage1_dataset_audit.json` includes absolute paths, metadata versions, source
schema, data/episode counts, file-inventory hash, episode metadata hashes,
sensor statistics, timestamp joins, calibration hashes, and state/action
statistics. `stage1_source_manifest.json` provides the corresponding source
file inventory. Zarr frame chunks are inventoried by path/size/mtime rather than
loaded into RAM or rewritten.


# Paper A Stage 0 baseline snapshot

Snapshot date: 2026-08-17 (Asia/Shanghai)

## Repository provenance

- Repository root: `/home/deepcybo/flexiv_ws/Le-nero`
- Base branch: `develop/3d_policy/dual_arm`
- Base commit SHA: `fc1ae884c6d5a2f610f40a5f61e71917ae02b0d0`
- Research branch: `research/paper-a-teb-data`
- Remote: `origin https://github.com/Key-Zzs/Le-nero.git`
- Submodule path: `dual_arm_data_collection/lerobot_dual_arm_teleop`
- Submodule URL: `https://github.com/Shenzhaolong1330/dual_arm_teleop.git`
- Submodule configured branch: `develop/3d_policy/dual_flexiv_GN01`
- Submodule pinned commit: `d363eb6b9de2c1a20bc9947330265fb6b9eb03c8`
- Submodule working tree: clean; current branch tracks its configured origin branch.
- Python: `/home/deepcybo/miniconda3/envs/dual_arm_teleop/bin/python` 3.10.20
- OS: Linux 6.8.0-124-generic-x86_64, glibc 2.35
- Relevant packages: LeRobot 0.3.4, NumPy 2.2.6, PyArrow 24.0.0, Zarr 2.18.3, PyAV 15.1.0, OpenCV 4.12.0, PyTorch 2.7.1+cu128.

The baseline was clean before branch creation. `git fetch origin` and
`git pull --ff-only` made no content change. No push and no commit were done.

## Acquisition-side baseline

The collection implementation lives in the pinned `dual_arm_teleop` submodule;
the LeRobot dataset/core handling and configs are in this repository. The
current Flexiv acquisition source declares `flexiv_abs_rot6d_raw_force_v3`, a
48D `observation.state`, and a 14D action. The already-recorded datasets below
are older, unannotated 28D absolute-rotvec state data and must not be silently
treated as current 48D recordings.

The current acquisition source also records RGB in normal LeRobot MP4 fields,
native depth/IR in a separate Zarr v2 sidecar, scalar join fields in Parquet,
and calibration in `meta/realsense_calibration.json`. The submodule is only
verified here; it is not redesigned or replaced by Stage 0/1.

## Information boundary

- Raw LeRobot and raw RGB-D/IR sidecar files are immutable source evidence.
- Stage 2 hand/object pose and task-effect annotations are derived artifacts;
  they are not policy observations.
- Phase semantic annotations are not Paper A core inputs and Stage 1 creates no
  phase labels.
- Task-effect annotations may be generated offline later, but cannot be used
  to claim Stage 1 object tracking or effect identification.
- The Stage 1 action analysis is named `action coverage proxy`; it is not an
  effect rank or an object-effect label.


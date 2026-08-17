# Paper A Stage 0 + Stage 1 plan and acceptance boundary

## Stage 0

1. Freeze `develop/3d_policy/dual_arm` at the recorded SHA and work only on
   `research/paper-a-teb-data`.
2. Verify the pinned `dual_arm_teleop` submodule without changing it.
3. Record the repository, Python environment, acquisition fields, timestamp
   units, calibration, state/action, episode, and provenance contracts.

## Stage 1

1. Discover only directories containing LeRobot `meta/info.json` and `data/`;
   ordinary directories are not datasets.
2. Audit metadata, episode boundaries, RGB/video, chunked depth/IR content,
   join indices, timestamp order/skew, calibration hashes, state/action
   finite values and dimensions, and the action coverage proxy.
3. Export five uniformly spaced source frames per episode when possible. Each
   sample retains episode index, episode frame index, dataset index, source
   row/timestamps, and the raw sidecar path.
4. Produce contact sheets for human review. No tracker, foundation model,
   artificial phase label, or effect label is run.

## Status semantics

- `PASS`: raw contract and state/action are valid; no conditional review gate.
- `CONDITIONAL_PASS`: raw contract is usable but explicit conversion, limited
  coverage, or human review remains.
- `RECOLLECT`: the source has an unrecoverable raw/metadata/timestamp/calibration
  failure or is empty.

`tracker_feasibility_status` is always `HUMAN_REVIEW_REQUIRED` in Stage 1. A
contact sheet is evidence for later human review, not a visibility score.


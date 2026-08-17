# Paper A Stage 1 Dataset Audit

- Raw root: `/home/deepcybo/.cache/huggingface/lerobot/flexiv_dual_arm_3d`
- Generated: `2026-08-17T09:09:03.108312+00:00`
- Dataset count: `2`
- Raw source mutation: `not performed`

## Stage 0 baseline

The repository/branch/submodule/acquisition contract is frozen in [BASELINE_SNAPSHOT.md](../BASELINE_SNAPSHOT.md) and [ACQUISITION_DATA_CONTRACT.md](../ACQUISITION_DATA_CONTRACT.md). The Stage 0 machine-readable snapshot is [stage0_baseline_snapshot.json](stage0_baseline_snapshot.json).

## Summary

| Dataset | Frames | Episodes | State | Sensor | Timestamp | Calibration | Stage 2 | Reasons |
|---|---:|---:|---|---|---|---|---|---|
| `pick_place_20260713_v05` | 4209 | 10 | legacy_state_28d_absolute_rotvec | PASS | PASS | PASS | **CONDITIONAL_PASS** | legacy_state_requires_explicit_conversion, limited_action_coverage, object_visibility_human_review_required, partial_depth_missing |
| `pick_place_20260717_v01` | 4983 | 10 | flexiv_abs_rot6d_v2 | PASS | PASS | PASS | **CONDITIONAL_PASS** | legacy_state_requires_explicit_conversion, limited_action_coverage, object_visibility_human_review_required, partial_depth_missing |

## `pick_place_20260713_v05`

- Source: `/home/deepcybo/.cache/huggingface/lerobot/flexiv_dual_arm_3d/pick_place_20260713_v05`
- Manifest hash: `fe5298c450b6150c5298f65b1a3ca65af8c4c27b48a39f278dca6dcdb05902be`
- Tracker feasibility: `HUMAN_REVIEW_REQUIRED`
- State/action: `28D / 14D`
- Reason codes: `legacy_state_requires_explicit_conversion`, `limited_action_coverage`, `object_visibility_human_review_required`, `partial_depth_missing`

### Sensor completeness

| Camera | RGB frames | Depth positive ratio | Reused frames | Left IR | Right IR |
|---|---:|---:|---:|---:|---:|
| `head` | 4209 | 0.9029581491112794 | 2 | 4209 | 4209 |
| `left_wrist` | 4209 | 0.7686966637710858 | 3 | 4209 | 4209 |
| `right_wrist` | 4209 | 0.7649677727414469 | 3 | 4209 | 4209 |

### Timestamp / calibration / action coverage

- Timestamp status: `PASS`; robot-camera differences are reported in seconds after applying the source-code units.
- Calibration status: `PASS`; manifest/file hashes are checked.
- Action coverage proxy: `action_coverage_proxy`; no effect rank is claimed.

### Human review boundary

Stage 1 does not run a tracker or synthesize phase labels. Object visibility, occlusion, table geometry, and suitability for geometric segmentation remain human review items.

## `pick_place_20260717_v01`

- Source: `/home/deepcybo/.cache/huggingface/lerobot/flexiv_dual_arm_3d/pick_place_20260717_v01`
- Manifest hash: `3d24abfece4008cc706bdd80bba0be8140d39cb3d5662fda8402534e320adbed`
- Tracker feasibility: `HUMAN_REVIEW_REQUIRED`
- State/action: `34D / 14D`
- Reason codes: `legacy_state_requires_explicit_conversion`, `limited_action_coverage`, `object_visibility_human_review_required`, `partial_depth_missing`

### Sensor completeness

| Camera | RGB frames | Depth positive ratio | Reused frames | Left IR | Right IR |
|---|---:|---:|---:|---:|---:|
| `head` | 4983 | 0.8939740985235216 | 5 | 4983 | 4983 |
| `left_wrist` | 4983 | 0.7735025029109388 | 4 | 4983 | 4983 |
| `right_wrist` | 4983 | 0.7648102728965984 | 9 | 4983 | 4983 |

### Timestamp / calibration / action coverage

- Timestamp status: `PASS`; robot-camera differences are reported in seconds after applying the source-code units.
- Calibration status: `PASS`; manifest/file hashes are checked.
- Action coverage proxy: `action_coverage_proxy`; no effect rank is claimed.

### Human review boundary

Stage 1 does not run a tracker or synthesize phase labels. Object visibility, occlusion, table geometry, and suitability for geometric segmentation remain human review items.

## Recommended Stage 2 priority

The first human-review candidate is **`pick_place_20260717_v01`** because its audited 34D rotation-6D state is closer to the current 48D Flexiv acquisition contract than the 28D absolute-rotvec `pick_place_20260713_v05`. This is a workflow priority, not a scientific success claim; both still require explicit offline schema conversion, action-coverage review, depth review, and human object-visibility review.

| Priority | Dataset | Gate state | Entry condition |
|---:|---|---|---|
| 1 | `pick_place_20260717_v01` | `CONDITIONAL_PASS` | resolve `legacy_state_requires_explicit_conversion, limited_action_coverage, object_visibility_human_review_required, partial_depth_missing` |
| 2 | `pick_place_20260713_v05` | `CONDITIONAL_PASS` | resolve `legacy_state_requires_explicit_conversion, limited_action_coverage, object_visibility_human_review_required, partial_depth_missing` |

## Suggested supplemental collection

No dataset meets the hard `RECOLLECT` gate: RGB/video counts, Zarr depth/IR counts, calibration hashes, and timestamp joins passed. Targeted supplemental collection is nevertheless recommended for both datasets if human review confirms unusable views or if the Paper A protocol requires non-zero depth coverage and balanced bimanual action coverage. The audit evidence does not authorize relabeling or repairing the existing raw source.

## Unresolved issues

- Convert legacy 28D/34D data explicitly to the current Flexiv 48D contract, or record a new native-48D set; do not pad or rewrite raw data.
- Human-review the 50-frame-per-dataset preview sets for object visibility, occlusion, table geometry, and geometric-segmentation suitability.
- Investigate partial/non-positive depth pixels and decide whether they meet the Stage 2 depth policy.
- Decide whether the static-arm/action-coverage proxy is acceptable for Paper A or requires targeted supplemental demonstrations.
- Stage 2 tracker, pose quality, object effect, and phase semantics remain unevaluated.

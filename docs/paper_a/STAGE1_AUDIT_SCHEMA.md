# Stage 1 audit schema

The machine-readable report is `docs/paper_a/reports/stage1_dataset_audit.json`.
Each dataset record contains:

| Field | Meaning |
|---|---|
| `identity` | absolute path, repo/dataset identity, counts, FPS/version, schemas, provenance hash |
| `episodes` | episode lengths, boundaries, metadata hash, per-episode provenance hash |
| `sensors` | RGB video files/counts and chunked depth/IR dtype, shape, finite/positive ratios and sampled quantiles |
| `timestamps` | source units, missing/duplicate/non-monotonic counts, Parquet/Zarr joins, robot-camera differences, camera skew |
| `calibration` | manifest/file SHA-256, camera serials, depth scale, stream dimensions, intrinsics/extrinsics presence |
| `state_action` | dimensions/names, NaN/Inf, per-dimension statistics, rotation representation |
| `action_coverage` | covariance spectrum, near-zero dimensions, per-arm activity, gripper transitions, h=2/4/8 proxy variance |
| `tracker_feasibility_status` | `HUMAN_REVIEW_REQUIRED`; Stage 1 has no tracker |
| `stage2_status` | `PASS`, `CONDITIONAL_PASS`, or `RECOLLECT` |
| `reason_codes` | evidence-backed classification causes |

Depth quantiles are computed from a deterministic stratified sample per bounded
Zarr chunk; full depth/IR arrays are never loaded into RAM. The report labels
this method explicitly. Action coverage is a proxy only and never an effect
rank.


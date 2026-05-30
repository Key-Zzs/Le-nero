# Le-nero

This repository is based on LeRobot and adds dual-arm robot teleoperation, data collection, policy training, and a round-based DAgger loop. This README only covers the repository structure, configuration entry points, and runtime call flow needed for daily use.

## Repository Setup and Environment

For a first-time clone, fetch submodules together with the main repository:

```bash
git clone --recurse-submodules <Le-nero repository URL> Le-nero
cd Le-nero
```

If the repository has already been cloned but submodule directories are empty or incomplete, run this from the repository root:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

To update the main repository and submodules during daily development:

```bash
cd Le-nero
git pull --ff-only
git submodule sync --recursive
git submodule update --init --recursive
git submodule update --remote --merge --recursive
```

To switch the main repository branch:

```bash
git fetch origin
git switch <branch_name>
git submodule update --init --recursive
```

To switch or update the dual-arm teleoperation submodule:

```bash
cd Le-nero/dual_arm_data_collection/lerobot_dual_arm_teleop
git fetch origin
git switch main
git pull --ff-only
```

Create the Python environment and install both the root repository and the dual-arm teleoperation package:

```bash
conda create -n dual_arm_teleop python=3.10 -y
conda activate dual_arm_teleop
python -m pip install --upgrade pip

cd Le-nero
pip install -e .

cd dual_arm_data_collection/lerobot_dual_arm_teleop
pip install -e .
```

Oculus Reader is not managed by the current `.gitmodules`, so it must be cloned separately into the required location:

```bash
cd Le-nero/dual_arm_data_collection/lerobot_dual_arm_teleop/teleoperators/oculus_teleoperator/oculus
git clone https://github.com/rail-berkeley/oculus_reader.git
cd oculus_reader
pip install -e .
```

If the directory already exists, update it and reinstall:

```bash
cd Le-nero/dual_arm_data_collection/lerobot_dual_arm_teleop/teleoperators/oculus_teleoperator/oculus/oculus_reader
git pull --ff-only
pip install -e .
```

Oculus connectivity also requires ADB:

```bash
sudo apt install android-tools-adb
adb devices
```

On the first USB connection, allow USB debugging in the headset. For wireless connection, first use `adb shell ip route` to find the headset IP, then run `adb connect <Oculus_IP>:5555`.

## Core Modules and Runtime Flow

The runtime flow can be understood as:

```text
scripts/config/*.yaml
        |
        v
scripts/core/*.py command entry points
        |
        +--> robots create the real robot interface
        +--> teleoperators create Oculus teleoperation input
        +--> src/lerobot/policies create policy models
        |
        v
LeRobot dataset / train / replay / visualize
```

### Policy Layer

Policy code is located in:

```text
src/lerobot/policies
```

This directory keeps LeRobot policy abstractions and implementations such as `act`, `diffusion`, `smolvla`, and `pi0`. The dual-arm teleoperation scripts mainly use `lerobot.policies.factory.make_policy` and `make_pre_post_processors` to create policy objects and their pre/post-processors.

The most commonly used policy config files in the dual-arm workflow are:

- `scripts/config/policy_config/act_train_config.yaml`: ACT training config.
- `scripts/config/policy_config/act_reason_config.yaml`: ACT inference/deployment config.
- `scripts/config/policy_config/diffusion_train_config.yaml`: Diffusion Policy training config.
- `scripts/config/policy_config/diffusion_reason_config.yaml`: Diffusion Policy inference/deployment config.

`scripts/core/policy_config_utils.py` resolves policy config paths from `record_cfg.yaml`, `train_cfg.yaml`, or `dagger_rounds_cfg.yaml`. Relative paths are resolved first against the `lerobot_dual_arm_teleop` project root, and absolute paths are also supported.

### Robot Communication Interface Layer

Robot interfaces are located in:

```text
dual_arm_data_collection/lerobot_dual_arm_teleop/robots
```

`robots/__init__.py` is the robot registry. The currently registered robot types include:

- `franka`
- `dobot_dual_arm`
- `nero_dual_arm`
- `franka_dual_arm`

Scripts do not instantiate a concrete robot class directly. Instead, they use the configured `robot_type` to call:

```python
create_robot_config(robot_type, **robot_cfg)
create_robot(robot_type, robot_config)
```

Each concrete robot class implements the robot interface expected by LeRobot, such as `connect()`, `reset()`, `send_action()`, camera initialization, observation fields, and action fields. For example, `dual_agilex_nero/nero_dual_arm.py` connects to the dual-arm zerorpc service through `NeroDualArmClient`, then organizes dual-arm end-effector poses, joint states, gripper commands, and RealSense cameras into a LeRobot-compatible data structure.

Hardware-specific parameters should usually live in config files instead of runtime scripts:

```text
dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/config/DAS_config
```

For example, `nero_cofig.yaml` defines the Nero robot IP, port, gripper parameters, Oculus mapping, and camera serial numbers. `run_record.py`, `run_replay.py`, and `reset_robot.py` automatically load the corresponding DAS config based on `record.robot_type`. You can also explicitly set `das_config_path` in `record_cfg.yaml`.

### Tool Scripts, Data Collection, and Policy Configs

Main scripts are located in:

```text
dual_arm_data_collection/lerobot_dual_arm_teleop/scripts
```

Common directories:

- `scripts/core`: command entry implementations for record, replay, visualize, reset, train, and DAgger.
- `scripts/config`: main workflow configs, including `record_cfg.yaml`, `train_cfg.yaml`, and `dagger_rounds_cfg.yaml`.
- `scripts/config/policy_config`: policy hyperparameter configs, split into train and reason configs.
- `scripts/config/DAS_config`: hardware and teleoperation detail configs.
- `scripts/tools`: dataset checks, RealSense device checks, dataset patching, renaming, and related utilities.

Core config files:

- `record_cfg.yaml`: main config shared by data collection, policy inference, mixed control, replay, and visualization.
- `train_cfg.yaml`: policy training config, including dataset paths, output directory, GPU settings, batch size, training steps, and wandb.
- `dagger_rounds_cfg.yaml`: round-based DAgger controller config that connects collection, export, and next-round training.
- `*_train_config.yaml`: model structure and training-related hyperparameters used during policy training.
- `*_reason_config.yaml`: model structure, device, and checkpoint parameters used during inference or deployment.

The three `robot-record` modes are controlled by `record.run_mode`:

- `run_record`: pure teleoperation data collection.
- `run_policy`: load a policy checkpoint and let the policy control the robot.
- `run_mix`: policy execution with operator takeover, used for DAgger data collection.

## Core Module Usage

After installing `dual_arm_data_collection/lerobot_dual_arm_teleop/setup.py`, the following console commands are registered. Install with:

```bash
cd Le-nero/dual_arm_data_collection/lerobot_dual_arm_teleop
pip install -e .
```

Command entry points:

| Command | Purpose | Default config |
| --- | --- | --- |
| `robot-record` | Teleoperation collection, policy execution, or run_mix mixed collection | `scripts/config/record_cfg.yaml` |
| `robot-replay` | Replay a collected episode | `scripts/config/record_cfg.yaml` `replay` section |
| `robot-visualize` | Visualize a dataset episode with Rerun | `scripts/config/record_cfg.yaml` `visualize` section |
| `robot-reset` | Connect to the configured robot and return it home | `scripts/config/record_cfg.yaml` |
| `robot-train` | Train ACT or Diffusion Policy | `scripts/config/train_cfg.yaml` |
| `robot-dagger` | Run round-based DAgger: collect, export, then train the next-round policy | `scripts/config/dagger_rounds_cfg.yaml` |
| `robot-dagger-export` | Export DAgger training data from raw run_mix logs | `scripts/config/dagger_rounds_cfg.yaml` `dagger_export` section |
| `tools-check-dataset` | Inspect local LeRobot dataset information | Command arguments |
| `tools-check-dagger-dataset` | Inspect an exported DAgger dataset | Command arguments |
| `tools-check-rs` | Show RealSense device serial numbers | None |
| `robot-help` | Print the command summary | None |

All core commands support an explicit config file. It is recommended to pass the path during debugging:

```bash
robot-record --config scripts/config/record_cfg.yaml
robot-replay --config scripts/config/record_cfg.yaml
robot-visualize --config scripts/config/record_cfg.yaml
robot-reset --config scripts/config/record_cfg.yaml
robot-train --config scripts/config/train_cfg.yaml
robot-dagger --config scripts/config/dagger_rounds_cfg.yaml
robot-dagger-export --config scripts/config/dagger_rounds_cfg.yaml
```

Before data collection, usually edit `scripts/config/record_cfg.yaml`:

- `record.repo_id`: dataset name. Recommended format: `<robot_task<num>_step<num>/<description>`, for example `nero_task3_step1/2mL_empty_right`.
- `record.robot_type`: choose a robot type such as `nero_dual_arm` or `franka_dual_arm`.
- `record.run_mode`: choose `run_record`, `run_policy`, or `run_mix`.
- `record.policy.type`, `config_path`, `pretrained_path`: required only for `run_policy` or `run_mix`.
- `record.task`: task description, number of episodes, resume behavior, and whether to record success labels.
- `record.time`: max episode duration, reset duration, and metadata save period.
- `replay`, `visualize`: default dataset and episode used by replay and visualization.

Hardware parameters are usually edited in `scripts/config/DAS_config/*.yaml`:

- `teleop.oculus_config.ip`: Oculus Quest IP.
- `teleop.oculus_config.*_pose_scaler` and `*_channel_signs`: mapping from left/right controllers to robot actions.
- `robot.robot_ip`, `robot.robot_port`: robot service address.
- `robot.use_gripper` and gripper parameters: enable grippers, close/open thresholds, max opening width, and force.
- `cameras.*_serial`, `width`, `height`: RealSense serial numbers and resolution.

Before training, usually edit `scripts/config/train_cfg.yaml`:

- `train.dataset.repo_id` and `train.dataset.root`: training dataset.
- `train.policy.type` and `train.policy.config_path`: policy type and training config.
- `train.output_dir`, `job_name`: model and log output location.
- `train.training`: visible GPUs, memory cap, TF32, and other training device settings.
- `train.steps`, `batch_size`, `num_workers`, `save_freq`: training scale.
- `train.wandb`: wandb project and mode.

Before DAgger, usually edit `scripts/config/dagger_rounds_cfg.yaml`:

- `dagger_rounds.seed_repo_id` or `seed_dataset_path`: seed dataset used by round 0.
- `dagger_rounds.initial_pretrained_path`: optional initial checkpoint, if one already exists.
- `dagger_rounds.policy`: policy type used in rounds, plus train/reason config paths.
- `dagger_rounds.episodes_per_round`, `num_rounds`, `round_schedule`: collection count per round, number of rounds, and training step schedule.
- `dagger_rounds.output_root`: DAgger round output directory.
- `dagger_rounds.record_cfg_path`, `train_cfg_path`: base configs dynamically modified and called by the controller.
- `dagger_rounds.policy_backend.export`: rules for exporting run_mix logs into training data.

## Quest Controller Buttons

| Control | Function |
| --- | --- |
| Left grip `LG` | Hold to enable left-arm end-effector motion. In `run_mix`, this starts or continues expert override for the left arm. |
| Right grip `RG` | Hold to enable right-arm end-effector motion. In `run_mix`, this starts or continues expert override for the right arm. |
| Left trigger `LTr` | Control the left gripper. Pressing closes the gripper; releasing opens it. |
| Right trigger `RTr` | Control the right gripper. Pressing closes the gripper; releasing opens it. |
| `Y` button | In `run_mix`, release the left gripper channel back to policy control. |
| `B` button | In `run_mix`, release the right gripper channel back to policy control. |
| `A` button | Request robot reset, if supported by the active teleoperator/robot implementation. |
| Controller pose | Controls the corresponding end-effector delta pose while the corresponding grip is held. |

If `mirror_teleop` is enabled, the left/right controller assignment is swapped and pose deltas are mirrored before being sent to the robot.

## DAgger/run_mix Controls

- Policy is the default controller. Human input overrides only the channels being actively controlled.
- Holding `LG` or `RG` makes the corresponding arm an expert override. The first override frame is marked as `takeover_start`; continued override frames are marked as `recovery`.
- `LTr` and `RTr` control grippers independently from arm motion. Gripper takeover uses soft takeover: the trigger command must match the current held gripper value before manual gripper control becomes active, which avoids sudden jumps.
- Press `Y` for the left gripper or `B` for the right gripper to hand that gripper back to the policy. The trigger must be released before that gripper can be manually reacquired.
- Use the left arrow to discard failed, incomplete, low-quality, or not-trainable episodes before saving. This is required when `full_episode.success_policy` is `recorded_is_success`.

Common workflow example:

```bash
cd Le-nero/dual_arm_data_collection/lerobot_dual_arm_teleop

# 1. Show camera serial numbers and fill them into scripts/config/DAS_config/*.yaml
tools-check-rs

# 2. Check that policy configs resolve correctly. Recommended before run_policy/run_mix
robot-record --config scripts/config/record_cfg.yaml --dry-run-policy-config

# 3. Connect to the robot and return it home
robot-reset --config scripts/config/record_cfg.yaml

# 4. Collect teleoperation data
robot-record --config scripts/config/record_cfg.yaml

# 5. Visualize or replay data
robot-visualize --config scripts/config/record_cfg.yaml
robot-replay --config scripts/config/record_cfg.yaml

# 6. Train a policy
robot-train --config scripts/config/train_cfg.yaml

# 7. Run the round-based DAgger loop
robot-dagger --config scripts/config/dagger_rounds_cfg.yaml
```

Common key controls during collection:

- Right arrow: stop the current episode and save it.
- Left arrow: discard the current episode.
- Esc: stop the recording session.
- Enter: continue to the next teleoperation segment or next episode.
- Ctrl+C: interrupt and clean up the incomplete dataset.

## TODO

### Gripper Transition Keyframe Weighting TODO

Scope for this TODO: design and implementation planning only. The current training default behavior for ACT and Diffusion Policy must stay unchanged until an explicit disabled-by-default implementation phase lands.

Codebase inspection summary:

- Dataset entry and chunk construction: `src/lerobot/datasets/factory.py`, `src/lerobot/datasets/lerobot_dataset.py`, `src/lerobot/datasets/utils.py`.
- Training dataloaders and logging: `src/lerobot/scripts/lerobot_train.py`, `dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/core/run_train.py`, `src/lerobot/utils/logging_utils.py`, `src/lerobot/rl/wandb_utils.py`.
- ACT loss and config: `src/lerobot/policies/act/modeling_act.py`, `src/lerobot/policies/act/configuration_act.py`, `dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/config/policy_config/act_train_config.yaml`.
- Diffusion Policy loss and config: `src/lerobot/policies/diffusion/modeling_diffusion.py`, `src/lerobot/policies/diffusion/configuration_diffusion.py`, `dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/config/policy_config/diffusion_train_config.yaml`.
- Policy feature inference and preprocessing: `src/lerobot/policies/factory.py`, `src/lerobot/processor/converters.py`, `src/lerobot/policies/act/processor_act.py`, `src/lerobot/policies/diffusion/processor_diffusion.py`.
- Existing annotation/edit/sampling references: `dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/debug/annotate_dataset_phase.py`, `dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/tools/preprocess_dataset.py`, `dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/tools/patch_lerobot_dataset_metadata.py`, `dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/tools/merge_lerobot_tasks.py`, `src/lerobot/scripts/lerobot_edit_dataset.py`, `dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/core/dagger_sampling.py`.
- Tests to extend later: `tests/datasets/test_datasets.py`, `tests/datasets/test_sampler.py`, `tests/processor/test_act_processor.py`, `tests/processor/test_diffusion_processor.py`, `tests/policies/test_policies.py`, `dual_arm_data_collection/lerobot_dual_arm_teleop/tests/test_dagger_sampling.py`.

Current behavior to preserve:

- ACT builds action chunks from `ACTConfig.action_delta_indices = range(chunk_size)`. `LeRobotDataset._get_query_indices()` clamps indices to episode boundaries and creates `action_is_pad` for padded chunk positions. `ACTPolicy.forward()` currently computes L1 action loss with `F.l1_loss(..., reduction="none")`, multiplies by `~batch["action_is_pad"].unsqueeze(-1)`, then calls `.mean()`. If VAE is enabled, KLD is added as `l1_loss + kl_weight * kld_loss`.
- Diffusion Policy builds the action horizon from `DiffusionConfig.action_delta_indices = range(1 - n_obs_steps, 1 - n_obs_steps + horizon)`. `DiffusionModel.compute_loss()` uses `F.mse_loss(pred, target, reduction="none")` over `[B, horizon, action_dim]`, optionally masks with `action_is_pad` only when `do_mask_loss_for_padding` is true, then returns `loss.mean()`. The sampled diffusion noise timestep variable is `timesteps` with shape `[B]`; this is separate from action horizon step `H`.
- New parquet columns can be loaded only if they are registered in `meta/info.json` features, because `LeRobotDataset.load_hf_dataset()` builds Hugging Face features from `self.features`. However, policy preprocessing currently drops arbitrary non-observation annotation keys: `src/lerobot/processor/converters.py` preserves observation keys, action, padding keys containing `_is_pad`, task/index metadata, reward/done/truncated, but not generic keys such as `annotation.keyframe_weight`.
- Action dimension names live in `dataset.meta.features["action"]["names"]`; `PolicyFeature` only carries shape/type. Gripper dim inference should use dataset feature names when available, with explicit config indices as fallback. Current dual-arm action names include patterns such as `left_gripper_cmd`, `right_gripper_cmd`, `left_gripper_cmd_bin`, and `right_gripper_cmd_bin`.

Future config sketch, README-only for now:

```yaml
loss_weighting:
  enabled: false
  keyframe_weight_column: "annotation.keyframe_weight"
  gripper_event_column: "annotation.gripper_event"
  use_timestep_weight: true
  use_action_dim_weight: true
  gripper_dim_indices: null
  infer_gripper_dim_from_feature_names: true
  gripper_dim_weight: 2.0
  max_weight: 10.0
  normalize_weighted_loss: true
  apply_to_pose_dims: true
  pose_keyframe_weight_scale: 1.0
  apply_to_gripper_dims: true
  gripper_keyframe_weight_scale: 1.0
```

Phase 0: Codebase inspection and design confirmation

- Goal: Freeze the implementation surface before code changes: dataset propagation, ACT loss, DP loss, config shape, logging, sampler, and tests.
- Files involved: `src/lerobot/datasets/factory.py`, `src/lerobot/datasets/lerobot_dataset.py`, `src/lerobot/datasets/utils.py`, `src/lerobot/processor/converters.py`, `src/lerobot/policies/factory.py`, `src/lerobot/scripts/lerobot_train.py`, `dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/core/run_train.py`, ACT/DP files listed above.
- TODO items: document whether annotation fields should be treated as queryable temporal features; decide whether `resolve_delta_timestamps()` should add the same action delta indices for `annotation.keyframe_weight` and `annotation.gripper_event`; decide whether annotation fields are complementary data or first-class batch keys after preprocessing; confirm gripper dim inference from `meta.info["features"]["action"]["names"]`; define exact disabled behavior tests.
- Acceptance criteria: a design note or PR description can state exact tensor shapes for ACT `[B, chunk_size]` weights and DP `[B, horizon]` weights; old datasets without annotation columns fall back to all-ones weights; disabled config is numerically equivalent to current code.
- Risks: annotation keys may be silently dropped by preprocessing; changing mean denominator can change loss scale; action horizon step weights can be confused with diffusion noise timesteps.
- Do not change in this phase: training loss code, dataset schema, policy configs, dataloader sampling, scripts, or tests.

Phase 1: Offline gripper transition annotation

- Goal: Add a later offline annotation tool that detects gripper opening/closing transition keyframes without mutating the original dataset.
- Files involved: future `dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/debug/annotate_gripper_transition.py`; use patterns from `dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/debug/annotate_dataset_phase.py`, `dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/tools/preprocess_dataset.py`, `dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/tools/patch_lerobot_dataset_metadata.py`, `dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/tools/merge_lerobot_tasks.py`.
- TODO items: detect opening and closing from action or gripper state; support continuous and binary gripper values; support left and right arms independently; write per-frame `annotation.gripper_event` and `annotation.keyframe_weight`; optionally write `annotation.left_gripper_event` and `annotation.right_gripper_event`; support `pre_window` and `post_window`; implement dry-run, statistics, plots, and CSV export.
- Weight defaults to evaluate: `normal = 1.0`, `pre_closing = 2.0`, `closing = 4.0-8.0`, `post_closing = 2.0-3.0`, `pre_opening = 2.0`, `opening = 4.0-8.0`, `post_opening = 2.0-3.0`.
- Acceptance criteria: dry-run reports transition counts and event ratios per episode; export creates a distinct dataset root with updated `meta/info.json` schema and parquet columns; videos and original data are preserved unless an explicit output copy is requested.
- Risks: gripper conventions differ (`open=1/close=0`, `_cmd` vs `_cmd_bin`, reversed gripper config); noisy commands can create false transitions; too-wide windows can label most of an episode as keyframe.
- Do not change in this phase: ACT/DP training, dataloaders, policy configs, deployment, robot control, or the source dataset in place.

Phase 2: Dataset feature propagation

- Goal: Ensure annotation columns become correctly aligned batch tensors for action chunks/horizons.
- Files involved: `src/lerobot/datasets/factory.py`, `src/lerobot/datasets/lerobot_dataset.py`, `src/lerobot/datasets/utils.py`, `src/lerobot/processor/converters.py`, `src/lerobot/policies/act/processor_act.py`, `src/lerobot/policies/diffusion/processor_diffusion.py`.
- TODO items: register `annotation.keyframe_weight` and `annotation.gripper_event` in dataset `meta/info.json` features and Hugging Face schema; update temporal query resolution so annotation columns use the same delta indices as `action`; verify `LeRobotDataset._get_query_indices()` returns annotation tensors and padding aligned to `action_is_pad`; preserve annotation tensors through preprocessing without normalization; add all-ones fallback weights when columns are absent; document old dataset compatibility.
- Acceptance criteria: ACT batches expose `annotation.keyframe_weight` as `[B, chunk_size]`; DP batches expose it as `[B, horizon]`; padding is aligned with `action_is_pad`; old datasets and disabled config produce the same batch fields used by current loss code.
- Risks: current `resolve_delta_timestamps()` only handles reward, action, and observation keys; current `batch_to_transition()` drops generic annotation keys; adding annotation keys to policy features would incorrectly normalize or classify them.
- Do not change in this phase: actual loss weighting math, sampler behavior, policy architecture, robot data collection, or deployment.

Phase 3: ACT weighted loss

- Goal: Replace ACT action reconstruction loss with disabled-by-default per-timestep and per-action-dim weighting.
- Files involved: `src/lerobot/policies/act/modeling_act.py`, `src/lerobot/policies/act/configuration_act.py`, `dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/config/policy_config/act_train_config.yaml` for future config documentation only.
- TODO items: keep current L1 loss as the exact disabled path; when enabled, compute `loss_per_dim = abs(pred_action - target_action)` with shape `[B, chunk_size, D]`; apply `timestep_weight` from `annotation.keyframe_weight`; apply `action_dim_weight` with gripper dims inferred from action feature names or explicit config; apply `action_is_pad` before reduction; clamp by `max_weight`; keep KLD weighting unchanged.
- Target formula, not implemented yet:

```python
loss = mean(abs(pred_action - target_action))
loss = masked_weighted_mean(loss_per_dim * timestep_weight * action_dim_weight)
```

- Acceptance criteria: disabled config returns bitwise or tight numeric equality with current `l1_loss`; enabled config broadcasts `[B, S]`, `[D]`, and `[B, S, D]` correctly; padded timesteps do not contribute; `loss_dict` can report `normal_frame_loss`, `keyframe_loss`, `gripper_loss`, `pose_loss`, `opening_loss`, and `closing_loss`.
- Risks: normalizing by weight sum vs element count changes gradient scale; weighting pose dims during gripper events may overfit transition context; gripper-only weighting can ignore approach/release pose corrections.
- Do not change in this phase: ACT model architecture, inference queueing, temporal ensembling, VAE KLD semantics, dataset writer, or DP loss.

Phase 4: DP weighted denoising loss

- Goal: Add disabled-by-default weighted denoising loss over action horizon steps, without confusing horizon step weights with diffusion noise timesteps.
- Files involved: `src/lerobot/policies/diffusion/modeling_diffusion.py`, `src/lerobot/policies/diffusion/configuration_diffusion.py`, `dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/config/policy_config/diffusion_train_config.yaml` for future config documentation only.
- TODO items: keep current `F.mse_loss(pred, target, reduction="none").mean()` as the exact disabled path; when enabled, compute `mse_per_dim` over `[B, H, D]`; apply `horizon_weight` from `annotation.keyframe_weight`; apply `action_dim_weight`; keep `timesteps` reserved for diffusion scheduler noise timestep `[B]`; respect `action_is_pad` according to existing `do_mask_loss_for_padding` semantics unless the new config explicitly opts into masking.
- Target formula, not implemented yet:

```python
loss = mean(mse(pred_noise, target_noise))
loss = weighted_mean(mse_per_dim * horizon_weight * action_dim_weight)
```

- Acceptance criteria: disabled config is numerically equivalent to current DP loss; enabled config broadcasts weights over `[B, H, D]`; horizon weighting never indexes the diffusion scheduler timestep; padding behavior is documented and tested.
- Risks: DP default currently does not mask padding; changing that by accident changes behavior; sparse transition weights can bias denoising toward gripper events and harm smooth approach trajectories.
- Do not change in this phase: scheduler behavior, `num_train_timesteps`, inference sampling, U-Net architecture, ACT loss, or sampler.

Phase 5: Optional keyframe-aware sampling

- Goal: Evaluate sampling as a second-priority tool after loss weighting, only to increase batches containing transition chunks.
- Files involved: `src/lerobot/datasets/sampler.py`, `src/lerobot/scripts/lerobot_train.py`, `dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/core/run_train.py`, `dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/core/dagger_sampling.py`.
- TODO items: design a sampler that marks an index/chunk positive if its action chunk or horizon contains a transition; cap oversampling to roughly 2-4x; avoid repeating entire episodes; define interaction with existing DP `EpisodeAwareSampler` and DAgger source-aware `WeightedRandomSampler`; log transition chunk sample ratio.
- Acceptance criteria: sampler is optional and disabled by default; loss weighting works without it; sampler never bypasses episode boundary and padding rules; batches show higher transition chunk ratio without dominating the epoch.
- Risks: over-sampling can cause premature closing/opening or gripper jitter; PyTorch DataLoader supports only one sampler, so combining DP episode-aware dropping, DAgger source weighting, and keyframe weighting needs one deliberate sampler path.
- Do not change in this phase: loss weighting, annotation schema, ACT/DP model code, or dataset contents.

Phase 6: Metrics, debugging, and visualization

- Goal: Make weighting auditable during training and annotation review.
- Files involved: `src/lerobot/scripts/lerobot_train.py`, `dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/core/run_train.py`, `src/lerobot/utils/logging_utils.py`, `src/lerobot/rl/wandb_utils.py`, ACT/DP `loss_dict` outputs, future annotation script.
- TODO items: log `total_loss`, `normal_frame_loss`, `keyframe_loss`, `gripper_loss`, `pose_loss`, `opening_loss`, `closing_loss`, `keyframe_ratio_per_batch`, `weighted_loss_mean_weight`, `max_weight`, and `transition_chunk_sample_ratio`; add annotation distribution summaries and CSV/plot outputs; ensure DDP/Accelerate logging uses scalar values only.
- Acceptance criteria: wandb receives train-prefixed scalar metrics through existing `WandBLogger.log_dict()`; local logs remain readable; annotation reports make opening/closing imbalance visible before training.
- Risks: per-batch metrics can be noisy; logging tensors or non-scalars is ignored by current wandb wrapper; per-process metrics may need aggregation before logging.
- Do not change in this phase: training math, model architecture, dataset export format, or deployment behavior.

Phase 7: Tests and regression safety

- Goal: Add focused tests before enabling the feature in real training.
- Files involved: `tests/datasets/test_datasets.py`, `tests/datasets/test_sampler.py`, `tests/processor/test_act_processor.py`, `tests/processor/test_diffusion_processor.py`, `tests/policies/test_policies.py`, possible new focused tests under `tests/policies/` and `dual_arm_data_collection/lerobot_dual_arm_teleop/tests/`.
- TODO items: annotation script unit tests; dataset new-column read and temporal alignment tests; ACT weighted loss shape/mask tests; DP weighted loss shape/horizon weight tests; disabled config numeric equivalence tests; padding mask exclusion tests; gripper dim weight broadcast tests; old dataset missing annotation fallback-to-ones tests; sampler cap tests if Phase 5 is implemented.
- Acceptance criteria: disabled ACT and DP losses match old implementation values; old datasets load and train without annotation fields; padding never receives positive weighted contribution; gripper dims are inferred correctly from left/right gripper feature names.
- Risks: end-to-end policy artifact tests are expensive and may be too broad for early loss math; random DP noise requires fixed seeds or isolated deterministic loss helpers.
- Do not change in this phase: production configs, default training behavior, dataset files, or test artifacts unless an implementation PR explicitly requires it.

Phase 8: Training and rollout validation

- Goal: Validate weights gradually before real robot rollout.
- Files involved: `dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/config/train_cfg.yaml`, `dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/config/policy_config/act_train_config.yaml`, `dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/config/policy_config/diffusion_train_config.yaml`, deployment configs only after offline validation.
- TODO items: run a small annotated-dataset smoke test; enable ACT first with conservative weights; then test DP; sweep `gripper_dim_weight`, event weights, and pre/post windows; compare normal vs keyframe loss curves; inspect rollout for premature close, premature open, and gripper jitter; compare before/after success metrics before real robot use.
- Acceptance criteria: training runs with disabled and enabled configs; enabled runs show measurable keyframe loss signal without exploding total loss; rollout videos show transitions at intended task phases; real robot validation has an explicit rollback checkpoint.
- Risks: over-weighting rare events can harm non-transition behavior; annotation mistakes can be amplified; simulation/offline metrics may not predict real gripper timing.
- Do not change in this phase: annotation definitions mid-run, robot safety limits, default production checkpoints, or deployment policy without a validated rollback path.

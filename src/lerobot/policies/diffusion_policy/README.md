# Diffusion Policy From Scratch for LeRobot Replacement

## 1. Goal

Build a clean, from-scratch Diffusion Policy implementation for learning, debugging, and testing, while keeping a path to full drop-in replacement of the current LeRobot policy under `src/lerobot/policies/diffusion/`.

This directory is currently a planning area only. Do not add the new model implementation here until the compatibility contract and skeleton phases are started.

## 2. Scope

In scope for the replacement:

- A LeRobot-compatible config class.
- A LeRobot-compatible policy class.
- A processor factory for pre/post processing.
- The same normalization behavior expected by LeRobot train/eval/deploy tools.
- `forward(batch)` training behavior returning a loss and logging dictionary.
- `select_action(batch)` inference behavior with receding-horizon action chunking.
- Dataset temporal indices for observation/action sampling.
- Action padding mask support.
- Checkpoint save/load compatibility through LeRobot policy APIs.

Out of scope for the first implementation pass:

- Editing policy registries or import paths before the new code is ready.
- Changing training scripts, dataset code, robot communication code, or the legacy implementation.
- Adding paper-faithful extensions before the minimal replacement passes integration tests.

Matching only an `obs` input shape and an `action` output shape is not enough. A replacement must also match LeRobot's config semantics, processor pipeline, normalization mapping, temporal sampling, loss return format, action queue behavior, device placement, and checkpoint semantics.

## 3. Relationship to Existing LeRobot Diffusion Policy

The existing implementation is in `src/lerobot/policies/diffusion/` and should remain untouched during this planning stage. It contains:

- `configuration_diffusion.py`: config, optimizer/scheduler presets, feature validation, temporal indices, and optional loss-weighting config.
- `processor_diffusion.py`: LeRobot pre/post processor factory for normalization, batch dimension handling, device transfer, and action conversion.
- `modeling_diffusion.py`: current policy, model, RGB encoder, SpatialSoftmax, DDPM/DDIM sampling, denoising loss, optional weighted loss, and receding-horizon action queue.

A legacy copy is stored under `src/lerobot/policies/diffusion_legacy/` as a reference snapshot. The new implementation should use the legacy copy for behavioral comparison, not as source to paste from.

Current LeRobot registration points that reference the built-in policy:

- `src/lerobot/policies/__init__.py` exports `DiffusionConfig`.
- `src/lerobot/policies/factory.py` imports `DiffusionConfig`, maps `policy_type == "diffusion"` to config creation, dynamically imports `DiffusionPolicy`, creates diffusion processors, and records action feature names from dataset metadata.
- `src/lerobot/async_inference/helpers.py` imports `DiffusionConfig` through `lerobot.policies`.
- `src/lerobot/__init__.py` mentions `"diffusion"` in policy documentation/examples.

Do not change those registration points until the new implementation reaches the drop-in replacement phase.

## 4. Current Dependency Map

### Configuration Dependencies

| Imported module path | Imported symbols | Imported by | Function served | Replacement decision |
| --- | --- | --- | --- | --- |
| `dataclasses` | `dataclass`, `field` | `configuration_diffusion.py` | Dataclass config definitions and default factories. | Reuse directly. |
| `lerobot.configs.policies` | `PreTrainedConfig` | `configuration_diffusion.py` | Base config, `type` registry, feature schema fields, save/load behavior, device/AMP defaults. | Must reuse for LeRobot compatibility. |
| `lerobot.configs.types` | `NormalizationMode` | `configuration_diffusion.py` | Normalization mapping keys for visual/state/action features. | Must reuse. |
| `lerobot.optim.optimizers` | `AdamConfig` | `configuration_diffusion.py` | Optimizer preset returned by `get_optimizer_preset`. | Should reuse initially. |
| `lerobot.optim.schedulers` | `DiffuserSchedulerConfig` | `configuration_diffusion.py` | LR scheduler preset, currently backed by diffusers optimization helpers. | Should reuse initially; can wrap later if removing diffusers optimizer dependency. |
| `DiffusionLossWeightingConfig` | Local dataclass | `configuration_diffusion.py`, `modeling_diffusion.py` | Optional keyframe/gripper weighted denoising loss settings. | Avoid in minimal phase; defer compatibility implementation after baseline loss works. |

Feature schema properties come from `PreTrainedConfig`, not from the diffusion config itself:

- `input_features` and `output_features` are the source of policy I/O schema.
- `image_features` selects `FeatureType.VISUAL` inputs.
- `robot_state_feature` selects `observation.state`.
- `env_state_feature` selects environment state inputs.
- `action_feature` selects the `action` output.

These must remain aligned with LeRobot's feature types and constants.

### Processor Dependencies

| Imported module path | Imported symbols | Imported by | Function served | Replacement decision |
| --- | --- | --- | --- | --- |
| `typing` | `Any` | `processor_diffusion.py` | Processor type annotations. | Reuse directly. |
| `torch` | `Tensor` via `torch.Tensor` annotation | `processor_diffusion.py` | Dataset statistics and tensor typing. | Reuse directly. |
| `lerobot.policies.diffusion.configuration_diffusion` | `DiffusionConfig` | `processor_diffusion.py` | Current processor factory config type. | Replace with the new config only when registry migration begins. |
| `lerobot.processor` | `AddBatchDimensionProcessorStep`, `DeviceProcessorStep`, `NormalizerProcessorStep`, `PolicyAction`, `PolicyProcessorPipeline`, `RenameObservationsProcessorStep`, `UnnormalizerProcessorStep` | `processor_diffusion.py` | Pre/post processor pipeline, normalization, device transfer, batch dimension, action type. | Must reuse for LeRobot compatibility. |
| `lerobot.processor.converters` | `policy_action_to_transition`, `transition_to_policy_action` | `processor_diffusion.py` | Converts policy action tensors to/from transition representation for postprocessing. | Must reuse. |
| `lerobot.utils.constants` | `POLICY_PREPROCESSOR_DEFAULT_NAME`, `POLICY_POSTPROCESSOR_DEFAULT_NAME` | `processor_diffusion.py` | Stable serialized processor names. | Must reuse. |

Current preprocessor order:

1. `RenameObservationsProcessorStep(rename_map={})`
2. `AddBatchDimensionProcessorStep()`
3. `DeviceProcessorStep(device=config.device)`
4. `NormalizerProcessorStep(features={**input_features, **output_features}, norm_map=normalization_mapping, stats=dataset_stats)`

Current postprocessor order:

1. `UnnormalizerProcessorStep(features=output_features, norm_map=normalization_mapping, stats=dataset_stats)`
2. `DeviceProcessorStep(device="cpu")`

### Model and Runtime Dependencies

| Imported module path | Imported symbols | Imported by | Function served | Replacement decision |
| --- | --- | --- | --- | --- |
| `logging` | `logging`, module logger | `modeling_diffusion.py` | Weighted-loss warnings. | Reuse directly. |
| `math` | `math.log` | `modeling_diffusion.py` | Sinusoidal timestep embedding. | Reuse directly. |
| `collections` | `deque` | `modeling_diffusion.py` | Observation/action queues for receding-horizon inference. | Should reuse initially or match behavior exactly. |
| `collections.abc` | `Callable` | `modeling_diffusion.py` | Helper typing for module replacement. | Reuse directly if needed. |
| `einops` | `rearrange` | `modeling_diffusion.py` | Image/camera/time/action tensor rearrangement. | Should reuse initially for clarity; can replace with torch reshapes later. |
| `numpy` | `np.meshgrid`, `np.linspace` | `modeling_diffusion.py` | SpatialSoftmax position grid. | Reimplement with torch in the new SpatialSoftmax. |
| `torch` | Core tensor ops, random sampling, no-grad, cat/stack/full/randn/randint | `modeling_diffusion.py` | Runtime, training, noising, sampling, queue tensors. | Must reuse. |
| `torch.nn.functional` | `F.mse_loss`, `F.softmax` | `modeling_diffusion.py` | Denoising loss and SpatialSoftmax attention. | Reuse torch ops, but reimplement DP modules from scratch. |
| `torchvision` | `models`, `transforms.CenterCrop`, `transforms.RandomCrop` | `modeling_diffusion.py` | ResNet backbone and RGB crop preprocessing. | Wrap or reimplement in the new image encoder; avoid direct leakage into core architecture. |
| `diffusers.schedulers.scheduling_ddim` | `DDIMScheduler` | `modeling_diffusion.py` | Optional inference scheduler. | Defer to paper-faithful or advanced phase. |
| `diffusers.schedulers.scheduling_ddpm` | `DDPMScheduler` | `modeling_diffusion.py` | Training noising and default inference scheduler. | Wrap behind a scheduler adapter initially or implement minimal DDPM. |
| `torch` | `Tensor`, `nn` | `modeling_diffusion.py` | Module definitions and annotations. | Must reuse. |
| `lerobot.policies.diffusion.configuration_diffusion` | `DiffusionConfig` | `modeling_diffusion.py` | Current model config type. | Replace with new config when implementation starts. |
| `lerobot.policies.pretrained` | `PreTrainedPolicy` | `modeling_diffusion.py` | LeRobot policy base class, save/load interface, abstract methods. | Must reuse. |
| `lerobot.policies.utils` | `get_device_from_parameters`, `get_dtype_from_parameters`, `get_output_shape`, `populate_queues` | `modeling_diffusion.py` | Sampling device/dtype, RGB encoder dry-run shape, receding-horizon queue filling. | Should reuse initially; queue behavior must be preserved. |
| `lerobot.utils.constants` | `ACTION`, `OBS_ENV_STATE`, `OBS_IMAGES`, `OBS_STATE` | `modeling_diffusion.py` | Stable batch keys for train/inference. | Must reuse. |

### Internal Functional Dependencies

| Functional dependency | Current location | Function served | Replacement decision |
| --- | --- | --- | --- |
| Normalization / unnormalization | Processor pipeline | Normalizes observations/actions and unnormalizes selected actions using dataset stats. | Must reuse. |
| Device transfer | Processor pipeline and model utilities | Moves policy input to config device and selected action back to CPU. | Must reuse via processors; model utilities can be reused initially. |
| Feature renaming | Processor pipeline | Placeholder currently uses empty rename map. | Must keep hook for compatibility; reuse current processor step. |
| Temporal indexing | `DiffusionConfig.observation_delta_indices`, `action_delta_indices`, `drop_n_last_frames` | Drives LeRobotDataset temporal sampling for observation history and future action chunks. | Must reuse semantics exactly. |
| Action padding mask | `action_is_pad`, `do_mask_loss_for_padding` | Masks copy-padded action timesteps in loss. | Must support before dataset smoke tests. |
| Queue-based receding-horizon inference | `DiffusionPolicy.reset`, `select_action`, `predict_action_chunk`, `populate_queues` | Caches observations and action chunks across environment steps. | Must match before replacement. |
| Checkpoint compatibility | `PreTrainedPolicy`, `PreTrainedConfig`, safetensors save/load | Loads config and weights through LeRobot/Hub APIs. | Must reuse base APIs; old weight key compatibility can be phased later. |
| Optimizer/scheduler presets | `get_optimizer_preset`, `get_scheduler_preset` | Provides training defaults. | Should reuse initially. |
| LeRobot policy registration | `policies/__init__.py`, `factory.py` | Maps `"diffusion"` to config, policy, and processors. | Do not modify in this stage; plan migration only after replacement tests. |

## 5. Reuse / Reimplement / Defer Decisions

Must reuse:

- `PreTrainedConfig` and `PreTrainedPolicy`.
- `PolicyFeature`, `FeatureType`, `NormalizationMode`, and LeRobot constants such as `ACTION`, `OBS_STATE`, `OBS_IMAGES`, `OBS_ENV_STATE`.
- `input_features`, `output_features`, `image_features`, `robot_state_feature`, `env_state_feature`, and `action_feature` semantics.
- Processor pipeline classes, normalizer/unnormalizer steps, processor converters, and serialized processor names.
- LeRobot dataset temporal index contract: observation deltas, action deltas, and `drop_n_last_frames`.
- LeRobot save/load behavior through config JSON and safetensors policy weights.

Should reuse initially:

- `populate_queues` for action chunking and observation history behavior.
- `get_device_from_parameters`, `get_dtype_from_parameters`, and `get_output_shape`.
- `AdamConfig` and `DiffuserSchedulerConfig` training presets.
- `einops.rearrange` for explicit tensor shape transformations.
- A scheduler adapter around diffusers DDPM if a minimal in-house DDPM scheduler is not ready in Phase 9.

Should reimplement from scratch:

- RGB/image encoder wrapper and crop handling.
- SpatialSoftmax.
- Sinusoidal timestep embedding.
- Conv1d block and FiLM residual block.
- Conditional 1D U-Net.
- DDPM noising/denoising training loss.
- Conditional action sampling logic.
- Minimal loss dictionary and shape validation.

Can defer:

- DDIM sampling.
- EMA.
- Advanced keyframe/gripper loss weighting.
- Advanced loss weighting metrics.
- Multi-camera separate encoders beyond the minimum shared-encoder contract.
- Paper-faithful local conditioning variants.
- Robot-specific action adapters.
- Loading old diffusion checkpoints with identical parameter key names.

## 6. Target Code Architecture

The future implementation should stay small and explicit at first:

- `configuration_diffusion_policy.py`: LeRobot-compatible config skeleton with the same key public fields needed for training, inference, temporal sampling, normalization, optimizer preset, and scheduler preset.
- `processor_diffusion_policy.py`: processor factory matching the current pre/post pipeline behavior.
- `modeling_diffusion_policy.py`: `DiffusionPolicy` replacement class using `PreTrainedPolicy`, a small `DiffusionModel`, and from-scratch DP modules.
- Internal modules may be split later only if tests show the single model file is too large to work with.

Keep public LeRobot-facing names and registry changes out of the new package until the replacement is ready. During development, use explicit imports from `lerobot.policies.diffusion_policy` in local tests.

## 7. LeRobot Compatibility Contract

The replacement must eventually expose the same LeRobot-facing capabilities as the original implementation:

- Config class registered through LeRobot's policy config mechanism.
- Policy class subclassing `PreTrainedPolicy`.
- Processor factory accepted by `make_pre_post_processors`.
- Normalization behavior using LeRobot dataset stats and `NormalizationMode`.
- `forward(batch)` for training/validation, returning `(loss, loss_dict_or_none)`.
- `select_action(batch)` for one-step inference.
- `predict_action_chunk(batch)` for chunked inference.
- `reset()` for environment resets and queue clearing.
- Temporal indices for LeRobotDataset sampling.
- Action chunking with `horizon`, `n_obs_steps`, and `n_action_steps`.
- Action padding mask handling through `action_is_pad` when configured.
- Checkpoint loading and saving with `PreTrainedPolicy.from_pretrained` / `save_pretrained`.
- Compatibility with LeRobot train/eval/deploy tools.

Minimum compatibility requirements:

- `n_action_steps <= horizon - n_obs_steps + 1`.
- Training batches must include `observation.state` and `action`.
- Training batches must include either stacked images or `observation.environment_state`.
- Image features must support multiple camera keys through `config.image_features`.
- Output loss dictionary must include at least `loss/total`.
- Inference must remove any `action` key from offline-eval batches before action selection.

## 8. Tensor Shape Contract

Training `forward(batch)` input after processor and policy-local stacking:

- `observation.state`: `(B, n_obs_steps, state_dim)`.
- `observation.images`: `(B, n_obs_steps, num_cameras, C, H, W)` when visual inputs exist.
- `observation.environment_state`: `(B, n_obs_steps, env_dim)` when environment state exists.
- `action`: `(B, horizon, action_dim)`.
- `action_is_pad`: `(B, horizon)` when `do_mask_loss_for_padding` is true.

Training output:

- `loss`: scalar tensor.
- `loss_dict`: logging-friendly dict, at minimum `{"loss/total": float}`.

Single-step inference input to `select_action(batch)`:

- `observation.state`: `(B, state_dim)`.
- Each image feature key in `config.image_features`: `(B, C, H, W)`.
- `observation.environment_state`: `(B, env_dim)` when configured.

Policy-local queue stacking converts single-step observations to:

- `observation.state`: `(B, n_obs_steps, state_dim)`.
- `observation.images`: `(B, n_obs_steps, num_cameras, C, H, W)`.
- `observation.environment_state`: `(B, n_obs_steps, env_dim)`.

Sampling and action output:

- Full diffusion sample: `(B, horizon, action_dim)`.
- Action chunk kept for execution: `(B, n_action_steps, action_dim)`.
- `select_action` output: `(B, action_dim)`.

## 9. Implementation Phases

- [x] Phase 0 - Dependency inspection and legacy backup
- [x] Phase 1 - Config skeleton compatible with LeRobot
  - Status: config skeleton and scoped tests added.
- [x] Phase 2 - Processor compatibility
  - Status: processor factory and tests added, full pytest pending because of local environment issue: `pytest` and `torch` are missing in available interpreters; `uv run` fails while building `egl-probe`.
- [x] Phase 3 - Minimal image/state/action tensor contract
  - Status: tensor contract utilities and scoped tests added; full pytest pending because of local environment issue: `pytest`/`torch`/`draccus` are missing in available interpreters, and `uv run` still fails while building `egl-probe`.
- [ ] Phase 4 - SpatialSoftmax from scratch
- [ ] Phase 5 - RGB encoder from scratch
- [ ] Phase 6 - Sinusoidal timestep embedding
- [ ] Phase 7 - FiLM residual block
- [ ] Phase 8 - Conditional 1D U-Net
- [ ] Phase 9 - DDPM scheduler wrapper or minimal scheduler
- [ ] Phase 10 - Denoising loss and training forward
- [ ] Phase 11 - Conditional sampling
- [ ] Phase 12 - Receding-horizon `select_action`
- [ ] Phase 13 - Fake batch shape tests
- [ ] Phase 14 - LeRobotDataset batch smoke test
- [ ] Phase 15 - Training-step overfit test
- [ ] Phase 16 - Checkpoint save/load compatibility
- [ ] Phase 17 - Drop-in replacement test against existing LeRobot DP interface
- [ ] Phase 18 - Optional paper-faithful extensions

## 10. Test Plan

- Static import tests for the new config, policy, and processor factory once they exist.
- Config tests for default values, feature validation, temporal indices, optimizer preset, and scheduler preset.
- Processor tests with fake dataset stats to verify normalization, unnormalization, device transfer, batch dimension, and CPU action postprocessing.
- Fake batch shape tests for image-only, environment-state-only, and image-plus-environment-state inputs.
- `forward(batch)` tests verifying scalar loss, `loss/total`, `action_is_pad` masking, and no mutation of unrelated batch keys.
- `select_action(batch)` tests verifying queue warmup, action chunk refill, `reset()`, and output shape.
- Deterministic sampling tests using injected `noise`.
- LeRobotDataset smoke test verifying temporal indices and real batch keys.
- Small overfit test on a fake or tiny dataset to confirm gradients and optimizer/scheduler integration.
- Save/load smoke test through `save_pretrained` and `from_pretrained`.
- Final drop-in interface test against the existing policy's public behavior before registry migration.

## 11. Open Questions

- Should Phase 9 prefer a minimal in-house DDPM scheduler first, or a thin adapter over diffusers for faster integration?
- Should old diffusion checkpoints load into the new implementation exactly, or is config-level compatibility enough until a later migration phase?
- Should the first RGB encoder use a torchvision ResNet backbone behind a clean wrapper, or start with a smaller educational CNN for shape tests?
- Which datasets should define the first LeRobotDataset smoke test and overfit test?
- When the replacement is ready, should the new package take over the existing `"diffusion"` policy type directly, or should it first register a temporary experimental type?

# 面向 LeRobot 替换的从零实现 Diffusion Policy

## 1. 目标

构建一个干净、便于学习、调试和测试的 Diffusion Policy 从零实现，同时保留将来完整替换当前 LeRobot 内置实现的路径。当前内置实现位于 `src/lerobot/policies/diffusion/`。

本目录目前只是规划区。在兼容性契约和代码骨架阶段开始之前，不要在这里加入新的模型实现。

## 2. 范围

替换目标包含：

- 与 LeRobot 兼容的 config 类。
- 与 LeRobot 兼容的 policy 类。
- 用于前处理和后处理的 processor factory。
- 与 LeRobot train/eval/deploy 工具预期一致的归一化行为。
- `forward(batch)` 训练行为，返回 loss 和日志字典。
- `select_action(batch)` 推理行为，支持 receding-horizon action chunking。
- 用于 observation/action 采样的数据集 temporal indices。
- action padding mask 支持。
- 通过 LeRobot policy API 进行 checkpoint 保存和加载的兼容性。

第一轮实现不包含：

- 在新代码准备好之前修改 policy registry 或 import path。
- 修改训练脚本、数据集代码、机器人通信代码或 legacy 实现。
- 在最小替换版本通过集成测试之前加入 paper-faithful 扩展。

只匹配 `obs` 输入形状和 `action` 输出形状是不够的。真正的替换实现还必须匹配 LeRobot 的 config 语义、processor pipeline、normalization mapping、temporal sampling、loss 返回格式、action queue 行为、device placement 和 checkpoint 语义。

## 3. 与现有 LeRobot Diffusion Policy 的关系

现有实现位于 `src/lerobot/policies/diffusion/`，在当前规划阶段应保持不变。它包含：

- `configuration_diffusion.py`：config、optimizer/scheduler preset、feature validation、temporal indices，以及可选的 loss-weighting config。
- `processor_diffusion.py`：LeRobot 前处理/后处理 processor factory，负责归一化、batch 维度处理、设备转移和 action 转换。
- `modeling_diffusion.py`：当前 policy、model、RGB encoder、SpatialSoftmax、DDPM/DDIM sampling、denoising loss、可选 weighted loss，以及 receding-horizon action queue。

legacy 参考快照保存在 `src/lerobot/policies/diffusion_legacy/`。新实现可以用 legacy 版本做行为对比，但不应该把它当作直接复制粘贴的源码来源。

当前引用内置 policy 的 LeRobot 注册点：

- `src/lerobot/policies/__init__.py` 导出 `DiffusionConfig`。
- `src/lerobot/policies/factory.py` 导入 `DiffusionConfig`，将 `policy_type == "diffusion"` 映射到 config 创建，动态导入 `DiffusionPolicy`，创建 diffusion processors，并从 dataset metadata 记录 action feature names。
- `src/lerobot/async_inference/helpers.py` 通过 `lerobot.policies` 导入 `DiffusionConfig`。
- `src/lerobot/__init__.py` 在 policy 文档/示例中提到 `"diffusion"`。

在新实现进入 drop-in replacement 阶段之前，不要修改这些注册点。

## 4. 当前依赖图

### Configuration 依赖

| 导入模块路径 | 导入符号 | 导入文件 | 作用 | 替换决策 |
| --- | --- | --- | --- | --- |
| `dataclasses` | `dataclass`, `field` | `configuration_diffusion.py` | 定义 dataclass config 和 default factory。 | 直接复用。 |
| `lerobot.configs.policies` | `PreTrainedConfig` | `configuration_diffusion.py` | config 基类、`type` registry、feature schema 字段、保存/加载行为、device/AMP 默认逻辑。 | 为了 LeRobot 兼容，必须复用。 |
| `lerobot.configs.types` | `NormalizationMode` | `configuration_diffusion.py` | visual/state/action feature 的 normalization mapping key。 | 必须复用。 |
| `lerobot.optim.optimizers` | `AdamConfig` | `configuration_diffusion.py` | `get_optimizer_preset` 返回的 optimizer preset。 | 初期建议复用。 |
| `lerobot.optim.schedulers` | `DiffuserSchedulerConfig` | `configuration_diffusion.py` | LR scheduler preset，当前由 diffusers optimization helpers 支撑。 | 初期建议复用；如果后续移除 diffusers optimizer 依赖，可以再包一层或替换。 |
| `DiffusionLossWeightingConfig` | 本地 dataclass | `configuration_diffusion.py`, `modeling_diffusion.py` | 可选 keyframe/gripper weighted denoising loss 设置。 | 最小阶段先避免；baseline loss 可用后再补兼容实现。 |

Feature schema 属性来自 `PreTrainedConfig`，不是 diffusion config 自己定义的：

- `input_features` 和 `output_features` 是 policy I/O schema 的来源。
- `image_features` 选择 `FeatureType.VISUAL` 输入。
- `robot_state_feature` 选择 `observation.state`。
- `env_state_feature` 选择 environment state 输入。
- `action_feature` 选择 `action` 输出。

这些语义必须和 LeRobot 的 feature types 及 constants 保持一致。

### Processor 依赖

| 导入模块路径 | 导入符号 | 导入文件 | 作用 | 替换决策 |
| --- | --- | --- | --- | --- |
| `typing` | `Any` | `processor_diffusion.py` | processor 类型注解。 | 直接复用。 |
| `torch` | 通过 `torch.Tensor` 注解使用 `Tensor` | `processor_diffusion.py` | dataset statistics 和 tensor 类型。 | 直接复用。 |
| `lerobot.policies.diffusion.configuration_diffusion` | `DiffusionConfig` | `processor_diffusion.py` | 当前 processor factory 的 config 类型。 | 仅在 registry migration 开始时替换为新 config。 |
| `lerobot.processor` | `AddBatchDimensionProcessorStep`, `DeviceProcessorStep`, `NormalizerProcessorStep`, `PolicyAction`, `PolicyProcessorPipeline`, `RenameObservationsProcessorStep`, `UnnormalizerProcessorStep` | `processor_diffusion.py` | 前/后处理 pipeline、归一化、设备转移、batch 维度、action 类型。 | 为了 LeRobot 兼容，必须复用。 |
| `lerobot.processor.converters` | `policy_action_to_transition`, `transition_to_policy_action` | `processor_diffusion.py` | 在后处理阶段将 policy action tensor 和 transition 表示相互转换。 | 必须复用。 |
| `lerobot.utils.constants` | `POLICY_PREPROCESSOR_DEFAULT_NAME`, `POLICY_POSTPROCESSOR_DEFAULT_NAME` | `processor_diffusion.py` | 稳定的 serialized processor 名称。 | 必须复用。 |

当前 preprocessor 顺序：

1. `RenameObservationsProcessorStep(rename_map={})`
2. `AddBatchDimensionProcessorStep()`
3. `DeviceProcessorStep(device=config.device)`
4. `NormalizerProcessorStep(features={**input_features, **output_features}, norm_map=normalization_mapping, stats=dataset_stats)`

当前 postprocessor 顺序：

1. `UnnormalizerProcessorStep(features=output_features, norm_map=normalization_mapping, stats=dataset_stats)`
2. `DeviceProcessorStep(device="cpu")`

### Model 和 Runtime 依赖

| 导入模块路径 | 导入符号 | 导入文件 | 作用 | 替换决策 |
| --- | --- | --- | --- | --- |
| `logging` | `logging`, module logger | `modeling_diffusion.py` | weighted-loss warning。 | 直接复用。 |
| `math` | `math.log` | `modeling_diffusion.py` | sinusoidal timestep embedding。 | 直接复用。 |
| `collections` | `deque` | `modeling_diffusion.py` | receding-horizon inference 的 observation/action queues。 | 初期建议复用，或精确匹配其行为。 |
| `collections.abc` | `Callable` | `modeling_diffusion.py` | module replacement helper 的类型注解。 | 如有需要直接复用。 |
| `einops` | `rearrange` | `modeling_diffusion.py` | image/camera/time/action tensor 重排。 | 初期建议复用以保持形状变换清晰；以后可用 torch reshape 替换。 |
| `numpy` | `np.meshgrid`, `np.linspace` | `modeling_diffusion.py` | SpatialSoftmax position grid。 | 在新的 SpatialSoftmax 中用 torch 重新实现。 |
| `torch` | core tensor ops、random sampling、no-grad、cat/stack/full/randn/randint | `modeling_diffusion.py` | runtime、training、noising、sampling、queue tensor。 | 必须复用。 |
| `torch.nn.functional` | `F.mse_loss`, `F.softmax` | `modeling_diffusion.py` | denoising loss 和 SpatialSoftmax attention。 | 复用 torch ops，但 DP 模块本身从零实现。 |
| `torchvision` | `models`, `transforms.CenterCrop`, `transforms.RandomCrop` | `modeling_diffusion.py` | ResNet backbone 和 RGB crop preprocessing。 | 在新 image encoder 中包一层或重新实现；避免让它直接泄漏到核心架构。 |
| `diffusers.schedulers.scheduling_ddim` | `DDIMScheduler` | `modeling_diffusion.py` | 可选 inference scheduler。 | 推迟到 paper-faithful 或高级阶段。 |
| `diffusers.schedulers.scheduling_ddpm` | `DDPMScheduler` | `modeling_diffusion.py` | training noising 和默认 inference scheduler。 | 初期可放在 scheduler adapter 后面，或实现最小 DDPM。 |
| `torch` | `Tensor`, `nn` | `modeling_diffusion.py` | module 定义和类型注解。 | 必须复用。 |
| `lerobot.policies.diffusion.configuration_diffusion` | `DiffusionConfig` | `modeling_diffusion.py` | 当前 model config 类型。 | 开始新实现时替换为新 config。 |
| `lerobot.policies.pretrained` | `PreTrainedPolicy` | `modeling_diffusion.py` | LeRobot policy 基类、保存/加载接口、抽象方法。 | 必须复用。 |
| `lerobot.policies.utils` | `get_device_from_parameters`, `get_dtype_from_parameters`, `get_output_shape`, `populate_queues` | `modeling_diffusion.py` | sampling device/dtype、RGB encoder dry-run shape、receding-horizon queue 填充。 | 初期建议复用；queue 行为必须保留。 |
| `lerobot.utils.constants` | `ACTION`, `OBS_ENV_STATE`, `OBS_IMAGES`, `OBS_STATE` | `modeling_diffusion.py` | train/inference 的稳定 batch key。 | 必须复用。 |

### 内部功能依赖

| 功能依赖 | 当前位置 | 作用 | 替换决策 |
| --- | --- | --- | --- |
| Normalization / unnormalization | Processor pipeline | 使用 dataset stats 归一化 observations/actions，并反归一化被选中的 actions。 | 必须复用。 |
| Device transfer | Processor pipeline 和 model utilities | 将 policy input 移到 config device，并将 selected action 移回 CPU。 | 通过 processors 必须复用；model utilities 初期可复用。 |
| Feature renaming | Processor pipeline | 当前使用空 rename map 的占位 hook。 | 为兼容性保留 hook；复用当前 processor step。 |
| Temporal indexing | `DiffusionConfig.observation_delta_indices`, `action_delta_indices`, `drop_n_last_frames` | 驱动 LeRobotDataset 对 observation history 和 future action chunks 的 temporal sampling。 | 必须精确复用语义。 |
| Action padding mask | `action_is_pad`, `do_mask_loss_for_padding` | 在 loss 中 mask copy-padded action timesteps。 | dataset smoke test 前必须支持。 |
| Queue-based receding-horizon inference | `DiffusionPolicy.reset`, `select_action`, `predict_action_chunk`, `populate_queues` | 在环境步之间缓存 observations 和 action chunks。 | 替换前必须匹配。 |
| Checkpoint compatibility | `PreTrainedPolicy`, `PreTrainedConfig`, safetensors save/load | 通过 LeRobot/Hub API 加载 config 和 weights。 | 必须复用基础 API；旧 weight key 兼容可以分阶段实现。 |
| Optimizer/scheduler presets | `get_optimizer_preset`, `get_scheduler_preset` | 提供训练默认值。 | 初期建议复用。 |
| LeRobot policy registration | `policies/__init__.py`, `factory.py` | 将 `"diffusion"` 映射到 config、policy 和 processors。 | 当前阶段不要修改；只在 replacement tests 通过后规划迁移。 |

## 5. 复用 / 重写 / 推迟决策

必须复用：

- `PreTrainedConfig` 和 `PreTrainedPolicy`。
- `PolicyFeature`、`FeatureType`、`NormalizationMode`，以及 LeRobot constants，例如 `ACTION`、`OBS_STATE`、`OBS_IMAGES`、`OBS_ENV_STATE`。
- `input_features`、`output_features`、`image_features`、`robot_state_feature`、`env_state_feature` 和 `action_feature` 的语义。
- Processor pipeline classes、normalizer/unnormalizer steps、processor converters 和 serialized processor names。
- LeRobot dataset temporal index contract：observation deltas、action deltas 和 `drop_n_last_frames`。
- 通过 config JSON 和 safetensors policy weights 实现的 LeRobot save/load 行为。

初期建议复用：

- 用于 action chunking 和 observation history 行为的 `populate_queues`。
- `get_device_from_parameters`、`get_dtype_from_parameters` 和 `get_output_shape`。
- `AdamConfig` 和 `DiffuserSchedulerConfig` 训练 preset。
- 用于显式 tensor shape transformation 的 `einops.rearrange`。
- 如果 Phase 9 还没有准备好最小自研 DDPM scheduler，则先用围绕 diffusers DDPM 的 scheduler adapter。

应该从零重写：

- RGB/image encoder wrapper 和 crop handling。
- SpatialSoftmax。
- Sinusoidal timestep embedding。
- Conv1d block 和 FiLM residual block。
- Conditional 1D U-Net。
- DDPM noising/denoising training loss。
- Conditional action sampling logic。
- 最小 loss dictionary 和 shape validation。

可以推迟：

- DDIM sampling。
- EMA。
- 高级 keyframe/gripper loss weighting。
- 高级 loss weighting metrics。
- 超出最小 shared-encoder contract 的 multi-camera separate encoders。
- Paper-faithful local conditioning variants。
- Robot-specific action adapters。
- 用完全相同 parameter key names 加载旧 diffusion checkpoints。

## 6. 目标代码架构

未来实现初期应该保持小而明确：

- `configuration_diffusion_policy.py`：与 LeRobot 兼容的 config skeleton，包含训练、推理、temporal sampling、normalization、optimizer preset 和 scheduler preset 所需的关键 public fields。
- `processor_diffusion_policy.py`：匹配当前前/后处理 pipeline 行为的 processor factory。
- `modeling_diffusion_policy.py`：使用 `PreTrainedPolicy` 的 `DiffusionPolicy` replacement class、小型 `DiffusionModel`，以及从零实现的 DP modules。
- 只有当测试表明单个 model 文件过大、难以维护时，再拆分内部模块。

在 replacement 准备好之前，不要把 public LeRobot-facing names 和 registry changes 放进新 package。开发期间，在本地测试中显式从 `lerobot.policies.diffusion_policy` 导入。

## 7. LeRobot 兼容性契约

替换实现最终必须暴露与原始实现相同的 LeRobot-facing capabilities：

- 通过 LeRobot policy config 机制注册的 config class。
- 继承 `PreTrainedPolicy` 的 policy class。
- 能被 `make_pre_post_processors` 接受的 processor factory。
- 使用 LeRobot dataset stats 和 `NormalizationMode` 的 normalization behavior。
- 用于训练/验证的 `forward(batch)`，返回 `(loss, loss_dict_or_none)`。
- 用于单步推理的 `select_action(batch)`。
- 用于 chunked inference 的 `predict_action_chunk(batch)`。
- 用于环境 reset 和 queue 清空的 `reset()`。
- 用于 LeRobotDataset sampling 的 temporal indices。
- 使用 `horizon`、`n_obs_steps` 和 `n_action_steps` 的 action chunking。
- 配置开启时，通过 `action_is_pad` 处理 action padding mask。
- 通过 `PreTrainedPolicy.from_pretrained` / `save_pretrained` 加载和保存 checkpoint。
- 与 LeRobot train/eval/deploy tools 兼容。

最低兼容要求：

- `n_action_steps <= horizon - n_obs_steps + 1`。
- 训练 batch 必须包含 `observation.state` 和 `action`。
- 训练 batch 必须包含 stacked images 或 `observation.environment_state` 至少一种。
- Image features 必须通过 `config.image_features` 支持多个 camera keys。
- 输出 loss dictionary 必须至少包含 `loss/total`。
- 推理时必须先从 offline-eval batches 中移除任何 `action` key，再进行 action selection。

## 8. Tensor Shape 契约

经过 processor 和 policy-local stacking 后，训练 `forward(batch)` 输入：

- `observation.state`：`(B, n_obs_steps, state_dim)`。
- `observation.images`：当存在 visual inputs 时为 `(B, n_obs_steps, num_cameras, C, H, W)`。
- `observation.environment_state`：当存在 environment state 时为 `(B, n_obs_steps, env_dim)`。
- `action`：`(B, horizon, action_dim)`。
- `action_is_pad`：当 `do_mask_loss_for_padding` 为 true 时为 `(B, horizon)`。

训练输出：

- `loss`：标量 tensor。
- `loss_dict`：日志友好的 dict，至少包含 `{"loss/total": float}`。

`select_action(batch)` 的单步推理输入：

- `observation.state`：`(B, state_dim)`。
- `config.image_features` 中每个 image feature key：`(B, C, H, W)`。
- `observation.environment_state`：配置存在时为 `(B, env_dim)`。

Policy-local queue stacking 会将单步 observations 转换为：

- `observation.state`：`(B, n_obs_steps, state_dim)`。
- `observation.images`：`(B, n_obs_steps, num_cameras, C, H, W)`。
- `observation.environment_state`：`(B, n_obs_steps, env_dim)`。

Sampling 和 action 输出：

- 完整 diffusion sample：`(B, horizon, action_dim)`。
- 保留用于执行的 action chunk：`(B, n_action_steps, action_dim)`。
- `select_action` 输出：`(B, action_dim)`。

## 9. 实现阶段

- [x] Phase 0 - 依赖检查和 legacy 备份
- [x] Phase 1 - 与 LeRobot 兼容的 config skeleton
  - 状态：config skeleton 和 scoped tests 已添加。
- [x] Phase 2 - Processor 兼容性
  - 状态：processor factory 和测试已添加，完整 pytest 因本地环境问题暂未完成：可用解释器缺少 `pytest` 和 `torch`，`uv run` 在构建 `egl-probe` 时失败。
- [ ] Phase 3 - 最小 image/state/action tensor contract
- [ ] Phase 4 - 从零实现 SpatialSoftmax
- [ ] Phase 5 - 从零实现 RGB encoder
- [ ] Phase 6 - Sinusoidal timestep embedding
- [ ] Phase 7 - FiLM residual block
- [ ] Phase 8 - Conditional 1D U-Net
- [ ] Phase 9 - DDPM scheduler wrapper 或最小 scheduler
- [ ] Phase 10 - Denoising loss 和 training forward
- [ ] Phase 11 - Conditional sampling
- [ ] Phase 12 - Receding-horizon `select_action`
- [ ] Phase 13 - Fake batch shape tests
- [ ] Phase 14 - LeRobotDataset batch smoke test
- [ ] Phase 15 - Training-step overfit test
- [ ] Phase 16 - Checkpoint save/load compatibility
- [ ] Phase 17 - 针对现有 LeRobot DP interface 的 drop-in replacement test
- [ ] Phase 18 - 可选 paper-faithful extensions

## 10. 测试计划

- 在新 config、policy 和 processor factory 存在后，添加 static import tests。
- Config tests：覆盖默认值、feature validation、temporal indices、optimizer preset 和 scheduler preset。
- Processor tests：使用 fake dataset stats 验证 normalization、unnormalization、device transfer、batch dimension 和 CPU action postprocessing。
- Fake batch shape tests：覆盖 image-only、environment-state-only、image-plus-environment-state 输入。
- `forward(batch)` tests：验证 scalar loss、`loss/total`、`action_is_pad` masking，以及不会修改无关 batch keys。
- `select_action(batch)` tests：验证 queue warmup、action chunk refill、`reset()` 和输出形状。
- 使用注入 `noise` 的 deterministic sampling tests。
- LeRobotDataset smoke test：验证 temporal indices 和真实 batch keys。
- 在 fake 或 tiny dataset 上做 small overfit test，确认 gradients 和 optimizer/scheduler integration。
- 通过 `save_pretrained` 和 `from_pretrained` 做 save/load smoke test。
- 在 registry migration 前，针对现有 policy public behavior 做最终 drop-in interface test。

## 11. 开放问题

- Phase 9 应该优先实现最小自研 DDPM scheduler，还是先用 diffusers 的 thin adapter 以便更快集成？
- 旧 diffusion checkpoints 是否需要精确加载到新实现中，还是在后续 migration 阶段之前只保证 config-level compatibility 即可？
- 第一版 RGB encoder 应该在 clean wrapper 后面使用 torchvision ResNet backbone，还是先用更小的教学型 CNN 做 shape tests？
- 第一个 LeRobotDataset smoke test 和 overfit test 应该使用哪些 datasets？
- 当 replacement 准备好后，新 package 应该直接接管现有 `"diffusion"` policy type，还是先注册一个临时 experimental type？

# Diffusion Policy Implementation Log

## Phase 1 Config skeleton compatible with LeRobot

### 1. Phase 1 的目标

Phase 1 只实现新的 Diffusion Policy 配置骨架：

- 新增 `DiffusionPolicyConfig`。
- 继承 LeRobot 的 `PreTrainedConfig`。
- 注册为实验 policy type：`"diffusion_policy"`。
- 保留 LeRobot 数据集采样、特征描述、normalization、optimizer/scheduler preset 所需的公共语义。
- 添加配置级 validation 和单元测试。

这个阶段刻意不实现以下内容：

- processor
- model
- RGB encoder
- SpatialSoftmax
- U-Net
- diffusion scheduler 运行逻辑
- loss
- inference / action queue
- factory 或 `__init__.py` 注册迁移

这样做的思路是先把“配置契约”稳定下来，再进入 processor 和模型实现。Diffusion Policy 的模型代码后续会依赖这些字段，如果一开始配置语义不稳定，后面的 shape、dataset temporal sampling、normalization、训练默认值都会跟着摇。

### 2. 文件位置

Phase 1 的主要实现文件：

```text
src/lerobot/policies/diffusion_policy/configuration_diffusion_policy.py
```

对应测试文件：

```text
tests/policies/diffusion_policy/test_configuration_diffusion_policy.py
```

### 3. 为什么继承 `PreTrainedConfig`

`DiffusionPolicyConfig` 继承：

```python
class DiffusionPolicyConfig(PreTrainedConfig):
```

原因是 LeRobot 的 policy config 并不只是普通参数容器。`PreTrainedConfig` 提供了很多 LeRobot 生态需要的基础契约：

- `input_features`
- `output_features`
- `image_features`
- `robot_state_feature`
- `env_state_feature`
- `action_feature`
- policy config registry
- `type` 属性
- device 自动选择
- AMP 可用性处理
- config save/load 基础行为

也就是说，继承 `PreTrainedConfig` 是让新配置进入 LeRobot policy 体系的关键步骤。否则后续 processor、dataset、trainer、checkpoint API 都很难自然接上。

### 4. 为什么注册为 `"diffusion_policy"`

当前实现使用：

```python
@PreTrainedConfig.register_subclass("diffusion_policy")
```

没有使用原版的 `"diffusion"`。

原因：

- 现有 LeRobot 内置实现仍在 `src/lerobot/policies/diffusion/`。
- 当前 Phase 1 只是新实现的配置骨架，还不能替代原版。
- 如果现在抢占 `"diffusion"`，会影响现有训练、测试、加载逻辑。
- 使用 `"diffusion_policy"` 可以安全地做实验，不破坏旧路径。

等后续 processor、model、loss、inference、save/load 测试稳定后，再考虑是否迁移 factory 和正式替代 `"diffusion"`。

### 5. 字段设计思路

配置字段分成几组。

#### 5.1 LeRobot 兼容字段

```python
n_obs_steps: int = 2
horizon: int = 16
n_action_steps: int = 8
```

这三个字段都和时间有关，但语义不同。

`n_obs_steps` 表示模型条件输入里包含多少帧 observation。默认值 `2` 表示当前帧和上一帧。

`horizon` 表示 diffusion model 一次去噪预测的完整 action trajectory 长度。默认 `16` 表示模型训练目标是 16 步 action 序列。

`n_action_steps` 表示 inference 时真正执行多少步动作，然后再次调用 policy。默认 `8` 表示每次生成 16 步，但只执行前 8 步，这是 receding-horizon 控制思路。

它们不能混用。简单说：

```text
n_obs_steps   = 看多少历史 observation
horizon       = 预测多少未来 action
n_action_steps = 实际执行多少 action
```

#### 5.2 Normalization mapping

```python
normalization_mapping = {
    "VISUAL": NormalizationMode.MEAN_STD,
    "STATE": NormalizationMode.MIN_MAX,
    "ACTION": NormalizationMode.MIN_MAX,
}
```

这和原版 Diffusion Policy 保持一致：

- 图像用 mean/std。
- 状态用 min/max。
- 动作用 min/max。

这个字段后续会被 processor 使用，当前 Phase 1 只是先稳定配置语义。

#### 5.3 `drop_n_last_frames`

```python
drop_n_last_frames: int = 7
```

默认值来自：

```text
horizon - n_action_steps - n_obs_steps + 1
= 16 - 8 - 2 + 1
= 7
```

这个字段和 `LeRobotDataset` 的 temporal sampling 有关。Diffusion Policy 训练时需要从当前帧附近采样一段未来 action。如果 episode 快结束，未来 action 不够，就会产生 padding。

原版实现会丢掉 episode 最后 7 帧，减少过多 padding 样本。这既保留原版行为，也让后续 dataset smoke test 更容易对齐。

#### 5.4 图像 encoder 配置

```python
vision_backbone = "resnet18"
crop_shape = (84, 84)
crop_is_random = True
pretrained_backbone_weights = None
use_group_norm = True
spatial_softmax_num_keypoints = 32
use_separate_rgb_encoder_per_camera = False
```

这些字段为后续 RGB encoder 和 SpatialSoftmax 做准备。

Phase 1 不创建任何图像模型，只保留配置入口和 validation。这样后续实现 encoder 时，不需要再改配置接口。

#### 5.5 U-Net 配置

```python
down_dims = (512, 1024, 2048)
kernel_size = 5
n_groups = 8
diffusion_step_embed_dim = 128
use_film_scale_modulation = True
```

这些字段描述后续 Conditional 1D U-Net 的结构。

当前阶段唯一使用它们的是 validation：`horizon` 必须能被 U-Net temporal downsampling factor 整除。

#### 5.6 Diffusion scheduler / inference 配置

```python
noise_scheduler_type = "DDPM"
num_train_timesteps = 100
beta_schedule = "squaredcos_cap_v2"
beta_start = 0.0001
beta_end = 0.02
prediction_type = "epsilon"
clip_sample = True
clip_sample_range = 1.0
num_inference_steps = None
```

这些字段描述 diffusion forward/noise schedule 和 inference sampling 设置。

Phase 1 只校验：

- `noise_scheduler_type` 必须是 `"DDPM"` 或 `"DDIM"`。
- `prediction_type` 必须是 `"epsilon"` 或 `"sample"`。

真正的 diffusion scheduler 运行逻辑留到后续阶段。

#### 5.7 Loss / optimizer / scheduler preset

```python
do_mask_loss_for_padding = False
optimizer_lr = 1e-4
optimizer_betas = (0.95, 0.999)
optimizer_eps = 1e-8
optimizer_weight_decay = 1e-6
scheduler_name = "cosine"
scheduler_warmup_steps = 500
```

`do_mask_loss_for_padding` 是给后续 loss 用的：如果 action 序列里有 padding，可以选择在 loss 里 mask 掉 padding 部分。

optimizer/scheduler 字段用于 `get_optimizer_preset()` 和 `get_scheduler_preset()`，让 LeRobot 训练工具可以读取默认训练配置。

### 6. `__post_init__` 的用途、流程和思路

`__post_init__` 在 dataclass 初始化后自动执行。

实现流程：

1. 调用父类初始化：

   ```python
   super().__post_init__()
   ```

   这一步交给 `PreTrainedConfig` 处理 device 自动选择和 AMP 检查。

2. 检查 temporal 参数必须为正：

   ```python
   n_obs_steps >= 1
   horizon >= 1
   n_action_steps >= 1
   ```

   这些字段会直接进入 dataset temporal sampling，如果是 0 或负数，会产生不合理的 range。

3. 检查 `n_action_steps` 是否超过可执行窗口：

   ```python
   max_action_steps = horizon - n_obs_steps + 1
   n_action_steps <= max_action_steps
   ```

   这是 Diffusion Policy receding-horizon 行为的关键约束。默认：

   ```text
   horizon = 16
   n_obs_steps = 2
   max_action_steps = 15
   ```

   所以默认 `n_action_steps = 8` 合法，但 `16` 不合法。

4. 检查视觉 backbone：

   ```python
   vision_backbone.startswith("resnet")
   ```

   当前只支持 ResNet 系列，和原版 DP 的图像 encoder 假设对齐。

5. 检查 `prediction_type`：

   允许：

   ```text
   epsilon
   sample
   ```

6. 检查 `noise_scheduler_type`：

   允许：

   ```text
   DDPM
   DDIM
   ```

7. 检查 `horizon` 和 U-Net 下采样倍率兼容：

   ```python
   downsampling_factor = 2 ** len(down_dims)
   horizon % downsampling_factor == 0
   ```

   默认 `down_dims` 长度为 3，所以 downsampling factor 是 8。默认 `horizon=16` 可以整除。

8. 检查 `crop_shape` 自身合法：

   - 如果不是 `None`，必须长度为 2。
   - 高和宽都必须大于等于 1。

设计思路：尽量把明显错误提前暴露在 config 初始化阶段。这样后续还没进入 model 或 dataset，就能知道配置不合法。

### 7. `get_optimizer_preset` 的用途、流程和思路

实现：

```python
def get_optimizer_preset(self) -> AdamConfig:
    return AdamConfig(
        lr=self.optimizer_lr,
        betas=self.optimizer_betas,
        eps=self.optimizer_eps,
        weight_decay=self.optimizer_weight_decay,
    )
```

用途：

- 返回 LeRobot optimizer config object。
- 不是直接创建 `torch.optim.Adam`。
- 训练 factory 后续会用这个 config 去 build 真正 optimizer。

思路：

- 保留原版 DP 的 Adam 默认值。
- 让配置字段和 optimizer preset 一一对应。
- 方便测试确认 lr、betas、eps、weight_decay 没有在传递过程中丢失。

### 8. `get_scheduler_preset` 的用途、流程和思路

实现：

```python
def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
    return DiffuserSchedulerConfig(
        name=self.scheduler_name,
        num_warmup_steps=self.scheduler_warmup_steps,
    )
```

用途：

- 返回 LeRobot 学习率 scheduler config object。
- 注意这里是 learning-rate scheduler，不是 diffusion denoising scheduler。

默认：

```text
name = cosine
num_warmup_steps = 500
```

思路：

- 跟原版训练 preset 对齐。
- Phase 1 不引入新的 scheduler 抽象，先复用 LeRobot 现有 `DiffuserSchedulerConfig`。

### 9. `validate_features` 的用途、流程和思路

`validate_features()` 检查 dataset/policy feature schema。

它依赖 `PreTrainedConfig` 提供的属性：

- `image_features`
- `env_state_feature`
- `action_feature`

实现流程：

1. 至少有 image 或 environment state：

   ```python
   if len(self.image_features) == 0 and self.env_state_feature is None:
       raise ValueError(...)
   ```

   合法组合包括：

   - image-only
   - env-state-only
   - image + env-state

   不合法：

   - 只有 `observation.state`
   - 完全没有 observation 输入

2. 必须有 action output：

   ```python
   if self.action_feature is None:
       raise ValueError(...)
   ```

   这是为了确保后续 policy 一定能预测 LeRobot 标准 action key。

3. 如果有图像，图像 shape 必须是 3 维：

   ```text
   (channels, height, width)
   ```

   这一步能提前捕获错误 shape，比如二维图像或者错误排列。

4. 多相机图像 shape 必须完全一致：

   原版 Diffusion Policy 也有这个限制。多个 camera key 可以存在，但它们的 shape 必须一样。

5. `crop_shape` 必须能放进所有图像：

   如果图像是 `(3, 96, 96)`，那么默认 crop `(84, 84)` 合法；如果 crop 是 `(128, 84)` 就不合法。

设计思路：

- 保留原版 DP 的 feature contract。
- 额外补充 action 和 image rank 检查，让错误更早、更清楚。
- 不在 Phase 1 做模型 shape 推导，因为模型还没有实现。

### 10. Temporal index properties

#### 10.1 `observation_delta_indices`

实现：

```python
return list(range(1 - self.n_obs_steps, 1))
```

默认：

```python
n_obs_steps = 2
range(-1, 1) -> [-1, 0]
```

用途：

- 告诉 `LeRobotDataset` 训练样本应该取哪些 observation 帧。
- 默认取上一帧和当前帧。

#### 10.2 `action_delta_indices`

实现：

```python
return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))
```

默认：

```python
n_obs_steps = 2
horizon = 16
range(-1, 15)
```

也就是：

```text
[-1, 0, 1, ..., 14]
```

用途：

- 告诉 `LeRobotDataset` 训练时要取多长的 action target。
- action window 从最早 observation 对齐点开始，跨度为完整 `horizon`。

#### 10.3 `reward_delta_indices`

实现：

```python
return None
```

用途：

- Diffusion Policy 这里是 supervised behavior cloning，不需要 reward target。

### 11. 测试覆盖内容

新增测试文件：

```text
tests/policies/diffusion_policy/test_configuration_diffusion_policy.py
```

主要覆盖：

- 默认初始化。
- 默认字段值：
  - `n_obs_steps == 2`
  - `horizon == 16`
  - `n_action_steps == 8`
  - `noise_scheduler_type == "DDPM"`
  - `prediction_type == "epsilon"`
- temporal indices：
  - observation 是 `[-1, 0]`
  - action 是 `range(-1, 15)`
  - reward 是 `None`
- `horizon` 必须能被 U-Net 下采样倍率整除。
- `n_action_steps` 不能超过 `horizon - n_obs_steps + 1`。
- scheduler/prediction/backbone 非法值会报错。
- feature validation：
  - image-only 合法。
  - env-state-only 合法。
  - 没有 image/env-state 不合法。
  - 没有 action output 不合法。
  - 多相机图像 shape 不一致不合法。
  - crop 大于图像不合法。
- optimizer/scheduler preset 类型和值。

### 12. 和原版配置的区别

原版参考：

```text
src/lerobot/policies/diffusion_legacy/configuration_diffusion.py
src/lerobot/policies/diffusion/configuration_diffusion.py
```

#### 12.1 注册名不同

原版：

```python
@PreTrainedConfig.register_subclass("diffusion")
```

新版本：

```python
@PreTrainedConfig.register_subclass("diffusion_policy")
```

原因是当前新实现还不能替代原版，需要避免影响现有 `"diffusion"`。

#### 12.2 类名不同

原版类名：

```python
DiffusionConfig
```

新版本类名：

```python
DiffusionPolicyConfig
```

这样能明确区分新目录下的 from-scratch replacement config。

#### 12.3 没有实现高级 loss weighting

原版包含：

```python
DiffusionLossWeightingConfig
loss_weighting
```

并且会校验：

- `loss_weighting.max_weight > 0`
- `loss_weighting.gripper_dim_weight >= 0`

新版本没有这些字段。原因是 Phase 1 不做 keyframe/gripper loss weighting，避免把高级训练逻辑提前固定进配置。

#### 12.4 新版本增加 temporal 参数合法性检查

新版本额外检查：

- `n_obs_steps >= 1`
- `horizon >= 1`
- `n_action_steps >= 1`
- `n_action_steps <= horizon - n_obs_steps + 1`

原版没有这些显式检查。

#### 12.5 新版本显式检查 action output

原版 `validate_features()` 没有显式检查 `"action"` output。

新版本会检查：

```python
self.action_feature is not None
```

这是为了后续 policy/model 能可靠拿到标准 action 输出定义。

#### 12.6 新版本检查图像 shape rank

新版本要求图像 shape 是 3 维：

```text
(channels, height, width)
```

原版主要检查 crop 和多相机 shape 是否一致，没有单独检查 rank。

#### 12.7 新版本检查 `crop_shape` 自身

新版本在 `__post_init__()` 中检查：

- `crop_shape` 长度必须是 2。
- crop 高宽都必须 >= 1。

原版只在有 image features 时检查 crop 是否超过图像大小。

#### 12.8 temporal indices 保持一致

这部分新旧一致：

```python
observation_delta_indices = list(range(1 - n_obs_steps, 1))
action_delta_indices = list(range(1 - n_obs_steps, 1 - n_obs_steps + horizon))
reward_delta_indices = None
```

这是必须保持的 LeRobotDataset 采样兼容点。

#### 12.9 optimizer/scheduler preset 保持一致

新版本继续返回：

- `AdamConfig`
- `DiffuserSchedulerConfig`

默认 lr、betas、eps、weight decay、scheduler name、warmup steps 都与原版保持一致。

#### 12.10 没有修改旧实现和 factory

新版本没有改动：

- `src/lerobot/policies/diffusion/`
- `src/lerobot/policies/diffusion_legacy/`
- `src/lerobot/policies/factory.py`
- `src/lerobot/policies/__init__.py`
- training scripts
- dataset code
- robot communication code

这保证 Phase 1 是低风险增量。

### 13. 后续复盘重点

进入 Phase 2 之前，需要重点记住：

1. `DiffusionPolicyConfig` 现在已经定义了 processor 后续要读取的字段。
2. `normalization_mapping` 会被 processor 使用，不是在 model 里手动 normalize。
3. temporal indices 是 dataset sampling 的核心契约，后续不要随意改。
4. `"diffusion_policy"` 只是实验注册名，不等于正式替代 `"diffusion"`。
5. 当前 `validate_features()` 只验证配置和 schema，不验证真实 tensor batch。
6. 真正的 tensor shape contract 要等 processor 和 model fake batch 测试阶段继续补。

推荐下一阶段：

```text
Phase 2 - Processor compatibility
```

## Phase 2 - Processor 兼容性

### 1. 本阶段目标

Phase 2 只实现新的 from-scratch Diffusion Policy 的 processor 兼容层：

- 新增 `processor_diffusion_policy.py`。
- 新增 `make_diffusion_policy_pre_post_processors(...)`。
- 复用 LeRobot 现有 processor primitives。
- 让新目录下的 processor factory 在行为上对齐原版 Diffusion Policy。
- 添加 scoped processor tests，验证 pipeline 构建、step 顺序、normalization、device、batch dimension、postprocessor unnormalization 和 converter wiring。

本阶段仍然不实现：

- model
- RGB encoder
- SpatialSoftmax
- U-Net
- scheduler
- loss
- sampling
- action queue
- inference / `select_action`
- factory 注册迁移

原因是 processor 是 LeRobot policy 的数据边界。先稳定 processor contract，可以保证后续模型实现拿到的 batch、action target、device placement 和 normalization 行为都与原版保持一致。

### 2. 修改文件

本阶段修改/新增文件：

```text
src/lerobot/policies/diffusion_policy/processor_diffusion_policy.py
tests/policies/diffusion_policy/test_processor_diffusion_policy.py
src/lerobot/policies/diffusion_policy/configuration_diffusion_policy.py
src/lerobot/policies/diffusion_policy/README.md
src/lerobot/policies/diffusion_policy/README_zh-CN.md
src/lerobot/policies/diffusion_policy/implementation_log.md
```

其中 `configuration_diffusion_policy.py` 只做了一个 Phase 1 小修正：

```text
horizon: 8 -> 16
n_action_steps: 4 -> 8
drop_n_last_frames: 3 -> 7
```

修正原因：Phase 1 的 README、implementation log 和已有配置测试都记录默认值应为 `horizon=16`、`n_action_steps=8`、`drop_n_last_frames=7`，而当前源文件实际写成了 `8/4/3`。这个不一致一旦 pytest 环境可用，会导致 Phase 1 config tests 失败。该修正只恢复 Phase 1 已记录的配置契约，没有加入新的模型行为。

没有修改：

```text
src/lerobot/policies/factory.py
src/lerobot/policies/__init__.py
src/lerobot/policies/diffusion/
src/lerobot/policies/diffusion_legacy/
training scripts
dataset code
robot communication code
```

仓库中已有中文 README 文件名是：

```text
README_zh-CN.md
```

本阶段使用了现有文件，没有创建重复的 `README_zh-CH.md`。

### 3. 新增函数逐个解释

#### `make_diffusion_policy_pre_post_processors(...)`

- 函数用途

  构造一组 LeRobot-compatible policy preprocessor 和 postprocessor，用于新的 `DiffusionPolicyConfig`。

  函数位置：

  ```text
  src/lerobot/policies/diffusion_policy/processor_diffusion_policy.py
  ```

  函数名：

  ```python
  make_diffusion_policy_pre_post_processors
  ```

- 输入参数

  ```python
  config: DiffusionPolicyConfig
  ```

  新的 from-scratch Diffusion Policy 配置。processor 会读取：

  - `config.input_features`
  - `config.output_features`
  - `config.normalization_mapping`
  - `config.device`

  ```python
  dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None
  ```

  dataset normalization stats。和原版一样，stats 可以是 `None`。如果 stats 缺失，LeRobot 的 `NormalizerProcessorStep` / `UnnormalizerProcessorStep` 会保留无 stats 的行为，而不是在本函数里手动处理。

- 返回值

  返回二元组：

  ```python
  (
      PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
      PolicyProcessorPipeline[PolicyAction, PolicyAction],
  )
  ```

  第一个是 preprocessor，用于 dataset/offline batch 输入。

  第二个是 postprocessor，用于 policy 输出 action。

- 处理流程

  preprocessor steps：

  ```text
  RenameObservationsProcessorStep(rename_map={})
  AddBatchDimensionProcessorStep()
  DeviceProcessorStep(device=config.device)
  NormalizerProcessorStep(
      features={**config.input_features, **config.output_features},
      norm_map=config.normalization_mapping,
      stats=dataset_stats,
  )
  ```

  postprocessor steps：

  ```text
  UnnormalizerProcessorStep(
      features=config.output_features,
      norm_map=config.normalization_mapping,
      stats=dataset_stats,
  )
  DeviceProcessorStep(device="cpu")
  ```

  pipeline 名称使用 LeRobot 默认常量：

  ```python
  POLICY_PREPROCESSOR_DEFAULT_NAME
  POLICY_POSTPROCESSOR_DEFAULT_NAME
  ```

  postprocessor converter 使用原版 Diffusion Policy 相同函数：

  ```python
  policy_action_to_transition
  transition_to_policy_action
  ```

- 每一个 processor step 的作用

  `RenameObservationsProcessorStep(rename_map={})`

  当前 rename map 为空，行为上不改 key。保留这个 step 是为了对齐 LeRobot 现有 policy processor 结构，并保留未来需要 observation key rename 时的兼容 hook。

  `AddBatchDimensionProcessorStep()`

  将单步输入补成 batch size 1。例如：

  ```text
  observation.state: (state_dim,) -> (1, state_dim)
  observation.image: (C, H, W) -> (1, C, H, W)
  action: (action_dim,) -> (1, action_dim)
  ```

  这让单步 inference-like 输入和 batched training 输入都能通过同一条 processor pipeline。

  `DeviceProcessorStep(device=config.device)`

  将 batch 中的 tensor 移到配置指定设备。后续模型还没实现，但 processor contract 必须先保证模型收到的 tensor 已经位于 policy device。

  `NormalizerProcessorStep(...)`

  使用 LeRobot 内置 normalization 逻辑，根据 feature type 和 `config.normalization_mapping` 对 observation 和 action target 归一化。

  `UnnormalizerProcessorStep(...)`

  将模型输出的 normalized policy action 反归一化回 dataset/robot action scale。

  `DeviceProcessorStep(device="cpu")`

  将 postprocessed action 移回 CPU，保持原版 LeRobot Diffusion Policy 的输出边界。

- 为什么要把 `input_features` 和 `output_features` 合并后送入 `NormalizerProcessorStep`

  Diffusion Policy 的训练 batch 不只包含 observation，也包含 action target。

  observation 属于 input features：

  ```text
  observation.state
  observation.image
  observation.environment_state
  ```

  action 属于 output features：

  ```text
  action
  ```

  训练阶段的 denoising loss 需要在 normalized action space 中计算。如果 preprocessor 只传入 `input_features`，那么 observation 会归一化，但 `action` target 不会归一化，后续训练目标会和 diffusion sampling/denoising 的动作尺度不一致。

  因此 preprocessor 必须和原版一致：

  ```python
  features={**config.input_features, **config.output_features}
  ```

- 为什么 action target 在训练阶段需要归一化

  Diffusion Policy 通常在规范化后的 action space 中学习噪声预测或样本预测。这样不同关节/动作维度的数值范围会被拉到更稳定的尺度，默认 `ACTION` 使用 `MIN_MAX` 映射到 `[-1, 1]`。

  如果 action target 不归一化，模型后续会同时面对 normalized observation 和原始尺度 action target，训练 loss 的数值尺度会不稳定，也会和 postprocessor 的 unnormalization 方向不匹配。

- 为什么 postprocessor 只处理 `output_features`

  postprocessor 的输入是模型输出的 `PolicyAction`，不是完整训练 batch。它只需要知道 action 的 feature schema 和 action 的 normalization stats。

  所以 postprocessor 使用：

  ```python
  features=config.output_features
  ```

  这也和原版 `make_diffusion_pre_post_processors(...)` 保持一致。

- 为什么 action 输出要反归一化

  模型内部输出的是 normalized action。真实环境、机器人控制、离线评估日志通常需要原始 action scale。

  postprocessor 负责把：

  ```text
  normalized action -> dataset/robot action scale
  ```

  这样模型实现不需要手写反归一化逻辑，也不会把 normalization 细节散落在 inference 代码里。

- 为什么最终要移动到 CPU

  原版 Diffusion Policy postprocessor 在反归一化后调用：

  ```python
  DeviceProcessorStep(device="cpu")
  ```

  这会让 policy 输出边界保持 CPU tensor。这样上层 eval/deploy/robot 代码不需要假设 action 仍在 CUDA/MPS 上，也避免 GPU tensor 泄漏到机器人通信层或日志层。

- 与原版 `make_diffusion_pre_post_processors(...)` 的相同点

  行为保持一致：

  - preprocessor step 顺序一致；
  - postprocessor step 顺序一致；
  - 使用相同 processor primitives；
  - 使用相同 default processor names；
  - preprocessor 同时 normalizes input features 和 output/action features；
  - postprocessor 只 unnormalizes output/action features；
  - postprocessed action 移回 CPU；
  - postprocessor 使用相同 converter：
    - `policy_action_to_transition`
    - `transition_to_policy_action`

- 与原版 `make_diffusion_pre_post_processors(...)` 的区别

  区别只在命名和 config 类型：

  ```text
  原版 config: DiffusionConfig
  新版 config: DiffusionPolicyConfig
  ```

  ```text
  原版函数: make_diffusion_pre_post_processors
  新版函数: make_diffusion_policy_pre_post_processors
  ```

  新函数导入：

  ```python
  from lerobot.policies.diffusion_policy.configuration_diffusion_policy import DiffusionPolicyConfig
  ```

  本阶段没有把新 processor 注册进全局 factory，也没有替换旧的 `"diffusion"` policy。

### 4. 测试文件逐项解释

测试文件：

```text
tests/policies/diffusion_policy/test_processor_diffusion_policy.py
```

本测试文件使用局部 helper 构造最小 fake config 和 fake stats，不依赖真实模型或真实 dataset。

#### `test_processor_factory_imports`

目的：

- 验证新函数可以从新模块正常 import。

验证逻辑：

- 从 `lerobot.policies.diffusion_policy.processor_diffusion_policy` import `make_diffusion_policy_pre_post_processors`。
- 断言导入到的对象就是测试文件顶部使用的 factory。

对应风险：

- 新文件路径、函数名、模块 import 失败会阻断后续所有使用者。

#### `test_pipeline_construction_uses_default_names`

目的：

- 验证 preprocessor/postprocessor 可以成功构造。
- 验证 pipeline name 使用 LeRobot 默认名称。

验证逻辑：

- 构造 fake `DiffusionPolicyConfig`。
- 构造 fake dataset stats。
- 调用 factory。
- 断言：

  ```text
  preprocessor.name == POLICY_PREPROCESSOR_DEFAULT_NAME
  postprocessor.name == POLICY_POSTPROCESSOR_DEFAULT_NAME
  ```

对应风险：

- processor name 错误会影响保存、加载和 LeRobot 默认 processor 文件命名约定。

#### `test_processor_step_composition_matches_legacy_order`

目的：

- 验证 step 类型和顺序与 legacy Diffusion Policy 一致。

验证逻辑：

- preprocessor 必须是：

  ```text
  RenameObservationsProcessorStep
  AddBatchDimensionProcessorStep
  DeviceProcessorStep
  NormalizerProcessorStep
  ```

- postprocessor 必须是：

  ```text
  UnnormalizerProcessorStep
  DeviceProcessorStep
  ```

对应风险：

- step 顺序改变会改变可观察行为。例如如果先 normalize 再加 batch，broadcast 行为和 device placement 都可能不同；如果先 CPU 再 unnormalize，stats/device 适配也会不同。

#### `test_preprocessor_normalizes_observations_visuals_and_training_action_targets`

目的：

- 验证 observation 和 action target 都走 normalization。
- 验证 visual feature 也可以走配置的 mean/std normalization 路径。

验证逻辑：

- fake state：

  ```text
  min = [0, -2]
  max = [10, 2]
  input = [5, 0]
  expected normalized = [0, 0]
  ```

- fake action：

  ```text
  min = [-1, -2, 0]
  max = [1, 2, 10]
  input = [0, 0, 5]
  expected normalized = [0, 0, 0]
  ```

- fake image：

  ```text
  mean = 10
  std = 2
  input = 12
  expected normalized = 1
  ```

对应风险：

- 如果 output/action features 没有被传入 preprocessor normalizer，训练 action target 会保持原始尺度。
- 如果 visual normalization 路径断掉，后续 image encoder 前的数据契约会不稳定。

#### `test_preprocessor_adds_batch_dimension_and_moves_tensors_to_config_device`

目的：

- 验证单步输入会被补 batch dimension。
- 验证 tensor 会移动到 `config.device`。

验证逻辑：

- 输入：

  ```text
  observation.state: (2,)
  observation.image: (3, 2, 2)
  action: (3,)
  ```

- 输出：

  ```text
  observation.state: (1, 2)
  observation.image: (1, 3, 2, 2)
  action: (1, 3)
  ```

- 当前测试使用 CPU-only 配置，断言所有 tensor 的 `device.type == "cpu"`。

对应风险：

- 没有 batch dimension 时，后续模型 forward 很容易收到 rank 不一致的 tensor。
- device 不一致会导致模型和输入 tensor 不在同一设备。

#### `test_postprocessor_unnormalizes_policy_action_and_returns_cpu_tensor`

目的：

- 验证 postprocessor 可以把 normalized action 反归一化回原始 action scale。
- 验证输出 action 位于 CPU。

验证逻辑：

- 输入 normalized action：

  ```text
  [[0, 0, 0]]
  ```

- fake action stats：

  ```text
  min = [-1, -2, 0]
  max = [1, 2, 10]
  ```

- expected unnormalized action：

  ```text
  [[0, 0, 5]]
  ```

对应风险：

- 如果 postprocessor 不反归一化，上层环境会收到 normalized action，而不是 robot/dataset scale action。
- 如果输出不回 CPU，部署层可能收到 GPU tensor。

#### `test_postprocessor_converter_functions_match_legacy_diffusion_behavior`

目的：

- 验证 postprocessor converter wiring 与 legacy Diffusion Policy 一致。

验证逻辑：

- 断言：

  ```python
  postprocessor.to_transition is policy_action_to_transition
  postprocessor.to_output is transition_to_policy_action
  ```

对应风险：

- 如果 converter 错误，postprocessor 可能无法把裸 `PolicyAction` tensor 放进 `EnvTransition`，或者无法从最终 transition 取回 policy action。

### 5. 与原版实现的区别

保持一致的行为：

- preprocessor step 顺序一致；
- postprocessor step 顺序一致；
- processor pipeline names 一致；
- normalization/unnormalization 复用同一套 LeRobot primitives；
- action target 在训练 preprocessor 中归一化；
- policy output action 在 postprocessor 中反归一化；
- postprocessed action 移回 CPU；
- postprocessor converter 函数一致。

命名变化：

- 原版模块：

  ```text
  lerobot.policies.diffusion.processor_diffusion
  ```

- 新模块：

  ```text
  lerobot.policies.diffusion_policy.processor_diffusion_policy
  ```

- 原版函数：

  ```text
  make_diffusion_pre_post_processors
  ```

- 新函数：

  ```text
  make_diffusion_policy_pre_post_processors
  ```

- 原版 config：

  ```text
  DiffusionConfig
  ```

- 新 config：

  ```text
  DiffusionPolicyConfig
  ```

仍未实现的高级功能：

- model；
- RGB encoder；
- SpatialSoftmax；
- Conditional 1D U-Net；
- DDPM/DDIM scheduler；
- denoising loss；
- conditional sampling；
- receding-horizon action queue；
- `forward(batch)`；
- `select_action(batch)`；
- checkpoint 权重兼容；
- global factory 注册。

为什么本阶段没有实现 model/U-Net/scheduler/loss/inference：

- processor contract 是模型之前的数据边界；
- 先确保 normalization、device、batch 和 converter 与原版一致，后续模型实现才能用稳定输入；
- 如果现在同时加入 model 和 scheduler，问题会混在一起，难以判断失败来自数据处理还是模型逻辑；
- 当前阶段目标是 scoped compatibility，不是可训练 policy。

### 6. 当前限制

当前 processor 只建立了数据预处理/后处理契约：

- 可以构造 preprocessor/postprocessor；
- 可以对 fake batch 做 batch dimension、device transfer、normalization；
- 可以对 fake normalized action 做 unnormalization 和 CPU transfer；
- 可以保持 legacy converter wiring。

当前还没有真实模型：

- 还不能进行训练 `forward(batch)`；
- 还不能计算 denoising loss；
- 还不能进行 `select_action` 推理；
- 还没有 action queue；
- 还不能替换 LeRobot 内置 `"diffusion"` policy。

本阶段 pytest 受本地环境阻塞，具体结果：

```text
python -m pytest tests/policies/diffusion_policy/test_configuration_diffusion_policy.py -q
```

失败原因：

```text
/usr/bin/python: No module named pytest
```

```text
.venv/bin/python -m pytest tests/policies/diffusion_policy/test_configuration_diffusion_policy.py tests/policies/diffusion_policy/test_processor_diffusion_policy.py -q
```

失败原因：

```text
.venv/bin/python: No module named pytest
```

```text
uv run python -m pytest tests/policies/diffusion_policy/test_configuration_diffusion_policy.py tests/policies/diffusion_policy/test_processor_diffusion_policy.py -q
```

第一次在 sandbox 内失败：

```text
snap-confine is packaged without necessary permissions and cannot continue
required permitted capability cap_dac_override not found in current capabilities
```

按 sandbox 规则在外部重试后，`uv run` 仍未进入 pytest，依赖构建失败：

```text
Failed to download and build `egl-probe @ git+https://github.com/huggingface/egl_probe.git#egg=egl_probe`
Package metadata name `hf-egl-probe` does not match given name `egl-probe`
```

compile 检查通过：

```text
.venv/bin/python -m compileall src/lerobot/policies/diffusion_policy tests/policies/diffusion_policy
```

结果：

```text
configuration_diffusion_policy.py compiled
processor_diffusion_policy.py compiled
test_configuration_diffusion_policy.py compiled
test_processor_diffusion_policy.py compiled
```

import smoke test 暂未完成，原因是可用解释器缺少 `torch`：

```text
.venv/bin/python -c "import torch; print(torch.__version__)"
ModuleNotFoundError: No module named 'torch'
```

系统 Python 也是同样缺少 `torch`，且版本为 Python 3.8.10，不满足项目 `pyproject.toml` 中的 `requires-python >=3.10`。

### 7. 下一阶段建议

推荐下一阶段：

```text
Phase 3 - 最小 image/state/action tensor contract
```

原因：

- Phase 2 只验证 processor 边界，还没有定义模型内部如何接收经过 processor 的 batch。
- 进入 SpatialSoftmax、RGB encoder、U-Net 之前，应该先用 fake tensors 固定最小 shape contract。
- 下一阶段应明确：
  - `observation.state` 的 batch/time/action 维度如何组织；
  - 单相机和多相机 image key 如何汇总；
  - `action` target 的 `(B, horizon, action_dim)` 合同；
  - 缺失 image/state/env-state 时应如何报错；
  - processor 输出和未来 model 输入之间的 key/shape 对齐。
- 如果跳过这个阶段直接写 SpatialSoftmax、RGB encoder 或 U-Net，很容易把 shape 错误藏进模型内部，后续排查成本会高很多。

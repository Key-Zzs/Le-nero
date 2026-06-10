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

## Phase 3 - 最小 image/state/action tensor contract

### 1. 本阶段目标

Phase 3 的目标是先固定 Diffusion Policy 进入模型前的最小 tensor contract，而不是开始写模型。

本阶段明确并测试以下几类 shape：

- LeRobot-style training batch。
- LeRobot-style single-step inference observation。
- multi-camera image stacking。
- action target shape。
- optional `action_is_pad` shape。

具体来说，本阶段把 legacy diffusion policy 中散落在 `forward(batch)`、`select_action(batch)`、`generate_actions(...)`、`compute_loss_and_metrics(...)`、`_prepare_global_conditioning(...)` 里的 shape 假设，抽成独立、显式、可测试的小工具函数。

### 2. 修改文件

本阶段新增：

```text
src/lerobot/policies/diffusion_policy/tensor_contract.py
tests/policies/diffusion_policy/test_tensor_contract.py
```

本阶段更新：

```text
src/lerobot/policies/diffusion_policy/README.md
src/lerobot/policies/diffusion_policy/README_zh-CN.md
src/lerobot/policies/diffusion_policy/implementation_log.md
```

本阶段没有修改：

```text
src/lerobot/policies/factory.py
src/lerobot/policies/__init__.py
src/lerobot/policies/diffusion/
src/lerobot/policies/diffusion_legacy/
training scripts
dataset code
robot communication code
```

### 3. 为什么本阶段仍不实现模型

在实现 SpatialSoftmax、RGB encoder、U-Net、scheduler、loss 之前，必须先固定 batch tensor contract。原因是这些模块都依赖相同的前置 shape 假设：

- state 必须知道是 `[B, n_obs_steps, state_dim]` 还是 `[B, state_dim]`。
- image 必须知道多相机维度 `N` 放在哪里。
- action 必须知道训练 target 是 `[B, horizon, action_dim]`。
- `action_is_pad` 必须知道 mask 的形状是否能和 action 的 batch/time 维度对齐。

如果这些 contract 没有先独立固定，后续模型报错时很难判断问题来自 processor、camera stacking、queue stacking、RGB encoder，还是 U-Net 本身。

所以本阶段只做 tensor contract，不做以下内容：

- `modeling_diffusion_policy.py`
- SpatialSoftmax
- RGB encoder
- U-Net
- scheduler
- diffusion loss
- sampling
- action queue
- `select_action`
- training `forward`

### 4. 新增函数逐个解释

#### `stack_image_features(...)`

函数用途：

- 把多个 camera image feature key 合并到统一的 `observation.images` key。
- 保持 legacy Diffusion Policy 的 camera stacking 语义。
- 返回一个新的浅拷贝 batch，避免修改调用者传进来的原始 dict。

输入参数：

- `batch: dict[str, torch.Tensor]`：包含独立 image feature key 的 batch。
- `image_feature_keys: Iterable[str]`：需要堆叠的 camera key 列表，通常来自 `config.image_features`。
- `output_key: str = OBS_IMAGES`：输出 key，默认是 `observation.images`。

返回值：

- 返回新的 dict。
- 原 batch 中已有 tensor object 会被复用。
- 新 dict 会额外包含 `output_key`。

处理流程：

1. 将 `image_feature_keys` 转成 list，确保至少有一个 key。
2. 逐个检查 key 是否存在。
3. 检查每个 image value 都是 `torch.Tensor`。
4. 检查 image tensor rank 只能是 4 或 5：
   - rank 5 用于训练：`[B, T, C, H, W]`。
   - rank 4 用于单步推理：`[B, C, H, W]`。
5. 检查所有 camera tensor shape 完全一致。
6. 用 `torch.stack(tensors, dim=-4)` 进行堆叠。
7. 返回 shallow copy。

training image shape 的变化：

```text
cam0: [B, T, C, H, W]
cam1: [B, T, C, H, W]

torch.stack(..., dim=-4)

observation.images: [B, T, N, C, H, W]
```

其中 `N` 是 camera 数量。

inference image shape 的变化：

```text
cam0: [B, C, H, W]
cam1: [B, C, H, W]

torch.stack(..., dim=-4)

observation.images: [B, N, C, H, W]
```

为什么使用浅拷贝避免修改原 batch：

- legacy `forward` 和 `select_action` 里也会先 `batch = dict(batch)`。
- 这样只是在新的 dict 上加入 `observation.images`。
- 调用者原始 batch 不会被追加 key。
- 这对调试、测试和后续 processor/model 边界都更安全。

为什么要检查 missing key 和 shape mismatch：

- legacy 里直接 list comprehension 取 key，缺 key 时会自然抛异常，但错误上下文不够明确。
- 多相机 shape 不一致时，`torch.stack` 会报错，但错误信息离 DP contract 太远。
- 本阶段把这两个错误提前变成明确的 `KeyError` 或 `ValueError`。

与 legacy `torch.stack([batch[key] ...], dim=-4)` 的关系：

- stacking 的 `dim=-4` 完全保持一致。
- training 和 inference 两种 rank 下产生的 camera 维度位置与 legacy 一致。

与 legacy 的区别：

- legacy 是在 `forward` 和 `select_action` 里临时做隐式 stacking。
- 本阶段把这段逻辑抽成独立函数，后续 model、policy 和 tests 都可以复用。

#### `validate_training_batch_contract(...)`

函数用途：

- 验证 policy-local image stacking 之后，训练 batch 是否满足未来模型和 loss 的最小输入契约。
- 这个函数只验证 shape，不计算 loss，不调用模型。

输入参数：

- `batch: dict[str, torch.Tensor]`：post-stacking training batch。
- `config: DiffusionPolicyConfig`：提供 `n_obs_steps`、`horizon`、feature metadata。

返回值：

- 返回 `None`。
- 如果 contract 不满足，抛出清晰的 `KeyError` 或 `ValueError`。

检查哪些 key：

- 必须有 `observation.state`。
- 必须有 `action`。
- 必须至少有 `observation.images` 或 `observation.environment_state` 一种 conditioning input。
- 如果存在 `action_is_pad`，则验证它的 shape。

检查哪些 shape：

```text
observation.state: [B, n_obs_steps, state_dim]
observation.images: [B, n_obs_steps, num_cameras, C, H, W]
observation.environment_state: [B, n_obs_steps, env_dim]
action: [B, horizon, action_dim]
action_is_pad: [B, horizon]
```

如何使用 `config.n_obs_steps`：

- 检查 `observation.state.shape[1]`。
- 如果有 `observation.images`，检查 `observation.images.shape[1]`。
- 如果有 `observation.environment_state`，检查 `observation.environment_state.shape[1]`。

如何使用 `config.horizon`：

- 检查 `action.shape[1]`。
- 如果存在 `action_is_pad`，检查 `action_is_pad.shape[1]`。

如何处理 `action_is_pad`：

- 本阶段把 `action_is_pad` 视为 optional key。
- 如果 key 不存在，不报错。
- 如果 key 存在，必须是 `[B, horizon]`。
- 它的 batch size 必须和 state/action 一致。

为什么 training batch 必须有 `ACTION`：

- Diffusion Policy 训练是行为克隆式监督训练。
- loss 需要真实 action trajectory 作为 denoising target 或 sample target。
- 没有 `action`，后续 `compute_loss_and_metrics(...)` 无法定义训练目标。

为什么 training batch 必须有 image 或 env-state 至少一种：

- 模型需要 conditioning input。
- state 是必需的，但 legacy contract 也要求 image 或 env-state 至少一种出现在 batch 中。
- 这保证未来 `_prepare_global_conditioning(...)` 至少能拼接出有效上下文。

与 legacy `compute_loss_and_metrics(...)` 里的 assert 逻辑关系：

legacy 中的关键检查是：

```python
assert set(batch).issuperset({OBS_STATE, ACTION})
assert OBS_IMAGES in batch or OBS_ENV_STATE in batch
assert horizon == self.config.horizon
assert n_obs_steps == self.config.n_obs_steps
```

本阶段保持这些语义，但把 `assert` 改成更明确的异常信息，并补充：

- tensor type 检查。
- ndim 检查。
- batch size 一致性检查。
- action dim 检查。
- camera 数量检查。
- image `(C, H, W)` 检查。
- env-state dim 检查。
- `action_is_pad` shape 检查。

如果某些测试或临时 config 没有完整 feature metadata，函数会跳过无法安全确定的 feature dim，只保留 rank、batch size、time/horizon 等确定性检查。这一点在代码注释中写明，避免测试为了构造最小 config 而被迫补齐无关 metadata。

#### `validate_single_step_observation_contract(...)`

函数用途：

- 验证单步推理 observation。
- 这是进入 action queue stacking 之前的形状检查。
- 不验证 action target。

输入参数：

- `batch: dict[str, torch.Tensor]`：单步 observation batch。
- `config: DiffusionPolicyConfig`：提供 image feature keys 和可选 feature dims。

返回值：

- 返回 `None`。
- 如果 contract 不满足，抛出 `KeyError` 或 `ValueError`。

为什么推理单步 observation 不应该要求 `ACTION`：

- 在线推理时只有 observation，没有 ground-truth action。
- offline eval batch 里可能带着 action，但 legacy `select_action` 会先移除 action。
- 所以这个函数不要求 action，也不把 action 当作推理输入 contract 的一部分。

为什么 state 是 `[B, state_dim]` 而不是 `[B, T, state_dim]`：

- 这个函数验证的是单个环境 step 的 observation。
- 历史 `T = n_obs_steps` 维度由后续 queue stacking 产生。
- 如果这里已经传入 `[B, T, state_dim]`，说明调用边界混淆了单步 observation 和历史 observation。

为什么 image 是 `[B, C, H, W]` 而不是 `[B, T, C, H, W]`：

- 单步推理每个 camera key 对应当前 step 的一张 batch image。
- 时间维度同样由 queue stacking 后续生成。
- 本阶段不实现 queue，因此只验证 queue 之前的单步 image。

该函数和后续 `select_action` queue stacking 的关系：

- 后续 `select_action` 会先接收单步 observation。
- image keys 会先被 stack 成 `[B, N, C, H, W]`。
- queue 再把连续 step stack 成 `[B, n_obs_steps, N, C, H, W]`。
- 本函数只负责第一步边界，不负责 action queue。

与 legacy `select_action(...)` 输入逻辑的关系：

- legacy `select_action` 接收单步 observation。
- 如果 batch 中有 `action`，legacy 会先 pop 掉。
- 如果有 image features，legacy 会用 `torch.stack(..., dim=-4)` 得到 `[B, N, C, H, W]`。
- 然后 `populate_queues` 才产生 observation history。
- 本阶段只把前半段的 shape contract 单独抽出来。

#### `get_image_feature_keys(...)`

函数用途：

- 返回 `config.image_features` 中的 visual input keys。

为什么从 `config.image_features` 取 key：

- `PreTrainedConfig.image_features` 已经是 LeRobot 对 visual feature schema 的标准入口。
- 不需要在测试或未来 policy 中重复筛选 `FeatureType.VISUAL`。

它如何避免重复访问 config internals：

- 测试可以直接调用 helper 或通过 contract 函数间接使用。
- 未来 `modeling_diffusion_policy.py` 可以从同一个 helper 取 camera key 顺序。
- camera stacking 顺序就能和 config feature schema 保持一致。

### 5. 测试文件逐项解释

测试文件：

```text
tests/policies/diffusion_policy/test_tensor_contract.py
```

`test_stack_image_features_for_training_batch_uses_legacy_camera_dimension`

- 构造两个 camera key，shape 都是 `[B, T, C, H, W]`。
- 调用 `stack_image_features(...)`。
- 验证输出 `observation.images` 是 `[B, T, 2, C, H, W]`。
- 验证原始 batch 没有被加入 `observation.images`。
- 防止 training image camera 维度放错位置。

`test_stack_image_features_for_single_step_inference_uses_legacy_camera_dimension`

- 构造两个 single-step camera key，shape 都是 `[B, C, H, W]`。
- 验证输出是 `[B, 2, C, H, W]`。
- 防止 inference image 被误当成带时间维度的 training image。

`test_stack_image_features_raises_for_missing_image_key`

- 只提供 `cam0`，但要求 stack `cam0` 和 `cam1`。
- 验证抛出清晰 `KeyError`。
- 防止多相机 batch 缺 camera 时错误延迟到模型内部。

`test_stack_image_features_raises_for_mismatched_image_shapes`

- 构造两个 camera tensor，但 `(C, H, W)` 不一致。
- 验证抛出 `ValueError`。
- 防止 `torch.stack` 的底层错误成为唯一提示。

`test_validate_training_batch_contract_accepts_valid_image_only_batch`

- 构造包含 state、stacked images、action、`action_is_pad` 的训练 batch。
- 验证 training contract 通过。
- 这个测试代表最常见的视觉 DP training batch。

`test_validate_training_batch_contract_accepts_valid_env_state_only_batch`

- 构造没有 image、但有 `observation.environment_state` 的训练 batch。
- 验证 contract 通过。
- 防止实现把 DP 强行写成只支持视觉输入。

`test_validate_training_batch_contract_rejects_missing_state`

- 删除 `observation.state`。
- 验证抛出 `KeyError`。
- 防止训练 batch 少了必需 robot state。

`test_validate_training_batch_contract_rejects_missing_action`

- 删除 `action`。
- 验证抛出 `KeyError`。
- 防止 loss 目标缺失。

`test_validate_training_batch_contract_rejects_missing_images_and_env_state`

- 只保留 state 和 action，不提供 image/env-state。
- 验证抛出 `ValueError`。
- 防止 conditioning input 缺失。

`test_validate_training_batch_contract_rejects_wrong_n_obs_steps`

- 把 state 的 time 维改成 `config.n_obs_steps + 1`。
- 验证错误包含 `n_obs_steps`。
- 防止 dataset temporal sampling 和 config 不一致。

`test_validate_training_batch_contract_rejects_wrong_horizon`

- 把 action horizon 改成 `config.horizon - 1`。
- 验证错误包含 `horizon`。
- 防止 denoising target 长度和模型 horizon 不一致。

`test_validate_training_batch_contract_rejects_wrong_action_is_pad_shape`

- 把 `action_is_pad` 变成 `[B, horizon, 1]`。
- 验证错误包含 `action_is_pad`。
- 防止 padding mask 无法和 `[B, horizon, action_dim]` 对齐。

`test_validate_training_batch_contract_rejects_inconsistent_batch_size`

- 让 `observation.images` 的 batch size 不同于 state/action。
- 验证错误包含 `batch size`。
- 防止不同 key 来自不同 batch slice 的集成错误。

`test_validate_single_step_observation_contract_accepts_valid_image_observation`

- 构造单步 state `[B, state_dim]` 和两个 camera `[B, C, H, W]`。
- 验证 single-step inference contract 通过。
- 这是后续 `select_action` 的输入边界。

`test_validate_single_step_observation_contract_rejects_missing_state`

- 删除 state。
- 验证抛出 `KeyError`。
- 防止推理 observation 没有 robot state。

`test_validate_single_step_observation_contract_rejects_temporal_state`

- 把 state 写成 `[B, T, state_dim]`。
- 验证抛出 `ValueError`。
- 防止调用者把 queue-stacked observation 传到单步 contract。

`test_validate_single_step_observation_contract_rejects_temporal_image`

- 把 image 写成 `[B, T, C, H, W]`。
- 验证抛出 `ValueError`。
- 防止单步推理和 training batch image shape 混用。

`test_validate_single_step_observation_contract_rejects_missing_images_and_env_state`

- 只提供 state，不提供 image feature key，也不提供 env-state。
- 验证抛出 `ValueError`。
- 防止推理 observation 没有任何 conditioning input。

### 6. 与原版实现的区别

保持一致的 shape 语义：

- training state 仍是 `[B, n_obs_steps, state_dim]`。
- training images 仍是 `[B, n_obs_steps, num_cameras, C, H, W]`。
- training env-state 仍是 `[B, n_obs_steps, env_dim]`。
- training action 仍是 `[B, horizon, action_dim]`。
- optional `action_is_pad` 仍是 `[B, horizon]`。
- single-step inference state 仍是 `[B, state_dim]`。
- single-step inference image feature 仍是 `[B, C, H, W]`。
- image stacking 仍使用 legacy 的 `dim=-4`。

被抽成显式 contract 的 legacy 隐式逻辑：

- `forward(batch)` 中的 camera stacking。
- `select_action(batch)` 中的 camera stacking。
- `compute_loss_and_metrics(...)` 中的 state/action/image-env presence check。
- `compute_loss_and_metrics(...)` 中的 `n_obs_steps` 和 `horizon` check。
- `_prepare_global_conditioning(...)` 对 `[B, T, N, C, H, W]` 的隐式假设。
- `generate_actions(...)` 对 `[B, n_obs_steps, state_dim]` 的隐式假设。

仍没有实现的行为：

- 不创建 `modeling_diffusion_policy.py`。
- 不实现 SpatialSoftmax。
- 不实现 RGB encoder。
- 不实现 U-Net。
- 不实现 scheduler。
- 不实现 denoising loss。
- 不实现 sampling。
- 不实现 action queue。
- 不实现 `select_action`。
- 不实现 training `forward`。

为什么本阶段不涉及 processor normalization：

- Phase 2 已经覆盖 processor factory、normalizer、unnormalizer、device transfer、batch dimension 和 action converter wiring。
- Phase 3 关注的是 processor 之后、模型之前的 tensor shape。
- normalization 数值正确性和 shape contract 是两个边界，分开测试更容易定位问题。

为什么本阶段不涉及 model forward：

- model forward 会引入 encoder、conditioning、denoising network、scheduler 和 loss。
- 这些都依赖当前 contract。
- 在 contract 稳定前加入 model forward，会让 shape bug 和模型 bug 混在一起。

### 7. 当前限制

当前只有 tensor contract：

- 可以 stack 多相机 image feature。
- 可以验证 post-stacking training batch。
- 可以验证 single-step inference observation。
- 可以验证 optional `action_is_pad` 的 shape。

当前还没有：

- 模型；
- SpatialSoftmax；
- RGB encoder；
- U-Net；
- scheduler；
- diffusion loss；
- sampling；
- action queue；
- training `forward`；
- `select_action` inference。

因此当前还不能训练，也还不能推理。

本阶段 pytest 受本地环境阻塞，具体结果：

```text
python -m pytest tests/policies/diffusion_policy/test_configuration_diffusion_policy.py tests/policies/diffusion_policy/test_processor_diffusion_policy.py tests/policies/diffusion_policy/test_tensor_contract.py -q
```

失败原因：

```text
/usr/bin/python: No module named pytest
```

可用解释器的依赖 spot check 也显示系统 Python 和 `.venv` 都缺少测试所需的 `pytest` 和 `torch`：

```text
python -c "import pytest; print(pytest.__version__)"
ModuleNotFoundError: No module named 'pytest'

python -c "import torch; print(torch.__version__)"
ModuleNotFoundError: No module named 'torch'

.venv/bin/python -c "import pytest; print(pytest.__version__)"
ModuleNotFoundError: No module named 'pytest'

.venv/bin/python -c "import torch; print(torch.__version__)"
ModuleNotFoundError: No module named 'torch'
```

```text
uv run python -m pytest tests/policies/diffusion_policy/test_configuration_diffusion_policy.py tests/policies/diffusion_policy/test_processor_diffusion_policy.py tests/policies/diffusion_policy/test_tensor_contract.py -q
```

sandbox 内失败原因：

```text
snap-confine is packaged without necessary permissions and cannot continue
required permitted capability cap_dac_override not found in current capabilities
```

按权限规则在 sandbox 外重试后，`uv run` 仍未进入 pytest，依赖构建失败：

```text
Failed to download and build `egl-probe @ git+https://github.com/huggingface/egl_probe.git#egg=egl_probe`
Package metadata name `hf-egl-probe` does not match given name `egl-probe`
```

compile 检查通过：

```text
python -m compileall src/lerobot/policies/diffusion_policy tests/policies/diffusion_policy
```

结果包含：

```text
Compiling 'src/lerobot/policies/diffusion_policy/tensor_contract.py'...
Compiling 'tests/policies/diffusion_policy/test_tensor_contract.py'...
```

`.venv` 的 compile 检查也通过：

```text
.venv/bin/python -m compileall src/lerobot/policies/diffusion_policy tests/policies/diffusion_policy
```

import smoke test 使用 `PYTHONPATH=src` 仍被本地依赖阻塞：

```text
ModuleNotFoundError: No module named 'draccus'
```

直接不设置 `PYTHONPATH` 时，解释器还会先报：

```text
ModuleNotFoundError: No module named 'lerobot'
```

这些失败都发生在测试环境和项目依赖加载阶段，不是 tensor contract 断言失败。

### 8. 下一阶段建议

推荐下一阶段：

```text
Phase 4 - 从零实现 SpatialSoftmax
```

原因：

- Phase 3 已经把 image/state/action 的最小 shape 边界固定下来。
- SpatialSoftmax 是第一个真正进入视觉模型路径的 DP 模块。
- 它只需要处理 feature map 到 keypoint-style feature 的局部逻辑，适合作为模型实现的第一步。
- 有了 Phase 3 的 contract，后续 RGB encoder 和 U-Net 可以依赖稳定的 image/state/action shape，不必在模型内部反复猜测输入维度。

## Phase 4 - 从零实现 SpatialSoftmax

### 1. 本阶段目标

本阶段只实现新的从零版 `SpatialSoftmax`：

- 新增 `src/lerobot/policies/diffusion_policy/spatial_softmax.py`。
- 新增 `tests/policies/diffusion_policy/test_spatial_softmax.py`。
- 保持 legacy Diffusion Policy 中 SpatialSoftmax 的数学行为。
- 使用 PyTorch 构造坐标网格，不再依赖 numpy。
- 支持 `num_keypoints=None` 和正整数 `num_keypoints`。
- 将固定坐标网格注册为 buffer。
- 添加局部 scoped tests，先把 feature map 到 keypoint coordinate 的行为固定下来。

本阶段刻意不实现：

- RGB encoder
- ResNet wrapper
- image crop
- U-Net
- timestep embedding
- FiLM block
- scheduler
- diffusion loss
- sampling
- action queue
- `select_action`
- training `forward`
- `modeling_diffusion_policy.py`

这样做是为了让视觉路径的第一个真实模块足够小、可测试，也方便后续 RGB encoder 接入时直接复用已经验证过的 keypoint pooling 行为。

### 2. 修改文件

本阶段新增：

```text
src/lerobot/policies/diffusion_policy/spatial_softmax.py
tests/policies/diffusion_policy/test_spatial_softmax.py
```

本阶段更新：

```text
src/lerobot/policies/diffusion_policy/README.md
src/lerobot/policies/diffusion_policy/README_zh-CN.md
src/lerobot/policies/diffusion_policy/implementation_log.md
```

Phase 3 在 README 中已经是 `[x]`，所以本阶段没有额外修正 Phase 3 checkbox。Phase 4 已在英文和中文 README 中从 `[ ]` 更新为 `[x]`，并记录了本地 pytest 环境阻塞原因。

### 3. SpatialSoftmax 的数学作用

`SpatialSoftmax` 的输入是 CNN 或未来 RGB encoder 产生的 feature map：

```text
[B, C, H, W]
```

含义是：

- `B`：batch size。
- `C`：feature channel 数。
- `H, W`：feature map 的空间尺寸。

普通 max pooling 会在局部窗口里取最大值，输出更强的局部响应，但会丢掉精确的连续位置信息。hard argmax 虽然能找出最大激活的位置，但位置选择是离散操作，通常不可导，不适合直接作为可反向传播的神经网络模块。

`SpatialSoftmax` 可以看作 differentiable soft-argmax：

1. 对每个 channel 的 `H * W` 个空间位置做 softmax。
2. 得到一个空间概率分布。
3. 用这个概率分布对归一化坐标网格求期望。
4. 输出每个 channel 或 keypoint map 的连续二维坐标。

坐标网格使用：

```text
x in [-1, 1]
y in [-1, 1]
```

其中 `x` 沿 width 方向变化，`y` 沿 height 方向变化。归一化到 `[-1, 1]` 的好处是输出和真实图像分辨率解耦：无论 feature map 是 `10x12` 还是 `7x7`，输出都在同一个坐标尺度里。后续 policy 可以把它当作紧凑、稳定的视觉特征使用。

输出 shape 是：

```text
[B, K, 2]
```

其中：

- `K = C`，当 `num_keypoints is None`。
- `K = num_keypoints`，当启用 learnable `1x1 Conv2d` 投影。
- 最后一维的 `2` 表示 `(x, y)`。

这个模块适合机器人视觉策略，因为机器人控制通常关心“重要物体或可操作部位在哪里”。`SpatialSoftmax` 不直接输出整张 feature map，而是把每个响应图压缩成类似关键点的坐标，使后续低维控制网络更容易利用空间信息。

### 4. 新增类和函数逐个解释

#### `SpatialSoftmax.__init__(...)`

函数用途：

- 初始化一个从 `[B, C, H, W]` feature map 到 `[B, K, 2]` 坐标的 differentiable soft-argmax 模块。
- 保存输入 feature map 的静态 `(C, H, W)` contract。
- 可选创建 learnable keypoint projection。
- 构造并注册固定归一化坐标网格。

输入参数：

```python
def __init__(
    self,
    input_shape: tuple[int, int, int],
    num_keypoints: int | None = None,
) -> None:
```

`input_shape` 必须是 `(C, H, W)`，并且三个值都必须是正整数。本阶段显式校验这些条件，如果传入长度不是 3，或者 `C/H/W` 中有 0、负数、非整数，会抛出清晰的 `ValueError`。

`num_keypoints` 可以是：

- `None`：不做 channel 投影，输出 keypoint 数 `K` 等于输入 channel 数 `C`。
- 正整数：先用 `nn.Conv2d(C, num_keypoints, kernel_size=1)` 把 channel 数从 `C` 投影到 `K`。

为什么使用 `1x1 Conv2d`：

- 它只在 channel 维做 learnable linear mixing。
- 它不改变空间尺寸 `H, W`。
- 它可以让模型从许多 CNN channels 中学习出固定数量的 keypoint heatmaps。
- 这和 legacy SpatialSoftmax 的 `num_kp` 行为一致。

`pos_grid` 的构造流程：

1. 用 `torch.linspace(-1.0, 1.0, steps=width)` 构造 x 坐标。
2. 用 `torch.linspace(-1.0, 1.0, steps=height)` 构造 y 坐标。
3. 用 `torch.meshgrid(..., indexing="ij")` 得到 `[H, W]` 网格。
4. 展平成 `[H * W, 2]`。
5. 每一行是一个空间位置的 `(x, y)`。

`pos_grid` 是 buffer 而不是 parameter，原因是：

- 它是固定几何坐标，不应该被 optimizer 更新。
- 它需要跟随 module 一起移动到 CPU/GPU。
- 它需要出现在 `state_dict` 中，方便保存和加载模块状态。
- `register_buffer("pos_grid", ...)` 正好满足这些需求。

与 legacy 实现的相同点：

- 输入 contract 仍是 `[B, C, H, W]`。
- 可选 `1x1 Conv2d` 仍用于把 `C` 映射到 `num_keypoints`。
- spatial softmax 仍作用在 `H * W` 空间维。
- 坐标仍归一化到 `[-1, 1]`。
- 输出仍是 `[B, K, 2]`。

与 legacy 实现的区别：

- legacy 用 `numpy.meshgrid` 和 `np.linspace` 构造 grid。
- 新实现用 PyTorch 的 `torch.linspace` 和 `torch.meshgrid` 构造 grid。
- legacy 参数名是 `num_kp`，新实现参数名是 `num_keypoints`，语义更明确。
- legacy 内部 projection 名为 `nets`，新实现命名为 `keypoint_projection`。
- 新实现增加了 constructor 参数校验，避免非法 shape 隐式进入 forward。

#### `SpatialSoftmax.forward(...)`

函数用途：

- 接收 `[B, C, H, W]` feature map。
- 返回 `[B, K, 2]` normalized expected coordinate。

输入 shape：

```text
[B, C, H, W]
```

输出 shape：

```text
[B, K, 2]
```

forward 流程：

1. validate shape

   检查输入是否是 4D tensor，并检查输入的 `C, H, W` 是否和 constructor 的 `input_shape` 完全一致。shape 不匹配时抛出 `ValueError`，错误信息包含期望 shape 和实际 shape。

2. optional 1x1 projection

   如果 `num_keypoints` 不为 `None`，先通过 `keypoint_projection` 把 `[B, C, H, W]` 变成 `[B, K, H, W]`。如果 `num_keypoints is None`，这一步跳过，`K = C`。

3. flatten `H * W`

   将 `[B, K, H, W]` reshape 成：

   ```text
   [B * K, H * W]
   ```

   这样每一行都是一个 channel/keypoint map 在所有空间位置上的激活。

4. softmax over spatial dimension

   对最后一维 `H * W` 做 softmax：

   ```python
   attention = torch.softmax(features, dim=-1)
   ```

   softmax 维度是空间维，而不是 channel 维。原因是 SpatialSoftmax 的目标是“每个 channel/keypoint map 在图像空间中关注哪里”，所以每个 channel 都应该独立形成自己的空间概率分布。如果对 channel 维做 softmax，就会变成不同 channels 互相竞争，丢掉每个 channel 自己的空间位置解释。

5. expectation over coordinate grid

   用 attention 乘以固定坐标网格：

   ```text
   [B * K, H * W] @ [H * W, 2] -> [B * K, 2]
   ```

   这就是对 `(x, y)` 坐标求期望。激活越强的位置，概率越大，对最终坐标的贡献也越大。

6. reshape to `[B, K, 2]`

   最后 reshape 回 batch 结构：

   ```text
   [B, K, 2]
   ```

为什么该操作可反向传播：

- `1x1 Conv2d` 可导。
- reshape 可导。
- softmax 可导。
- matrix multiplication 可导。
- 输出坐标对输入 feature map 的梯度可以一路回传。

shape mismatch 时如何报错：

- 输入不是 4D：报错说明必须是 `[B, C, H, W]`。
- channel 不匹配：报错说明 constructor 中的 `input_shape` 和实际 `C, H, W` 不一致。
- height/width 不匹配：同样报错，并带出期望和实际 shape。

与 legacy implementation 的关系：

- 本实现保留 legacy 的数学流程。
- 本实现不复制 legacy 的源码结构，而是用更小、更显式的 PyTorch 模块重写。
- 本实现先独立于 RGB encoder，是因为 RGB encoder 的输出将直接喂给 SpatialSoftmax。先固定这个模块，可以让下一阶段只关注 backbone/crop/multi-camera 逻辑。

### 5. 测试文件逐项解释

测试文件：

```text
tests/policies/diffusion_policy/test_spatial_softmax.py
```

`test_output_shape_without_learnable_keypoint_projection`

- 验证 `num_keypoints=None` 时输出 shape 是 `[B, C, 2]`。
- 防止后续 RGB encoder 接入时误以为 SpatialSoftmax 一定输出固定 keypoint 数。

`test_output_shape_with_learnable_keypoint_projection`

- 验证 `num_keypoints=K` 时输出 shape 是 `[B, K, 2]`。
- 防止 `1x1 Conv2d` projection 后 reshape 仍错误使用原始 channel 数。

`test_uniform_feature_map_returns_grid_center_without_projection`

- 构造全零 feature map。
- 全零经过 spatial softmax 后是 uniform distribution。
- 对称网格的期望应接近 `(0, 0)`。
- 防止坐标网格顺序、flatten 顺序或 softmax 维度写错。

`test_uniform_feature_map_returns_grid_center_with_projection`

- 启用 `num_keypoints=K`。
- 将 `1x1 Conv2d` weight 和 bias 都置零。
- 无论输入 features 是什么，projection 输出都是全零 map。
- 期望输出仍接近 `(0, 0)`。
- 防止 projection 分支和非 projection 分支数学行为不一致。

`test_dominant_activation_returns_expected_normalized_coordinate`

- 在小网格 `H=3, W=5` 中，把某一个空间位置设为强激活，其余位置设为很小值。
- 输出坐标应接近该位置在 `[-1, 1]` 网格里的 `(x, y)`。
- 防止 `x/y` 维度反了、height/width 网格构造反了、flatten 顺序和 grid 顺序不匹配。

`test_position_grid_is_registered_buffer_not_parameter_and_moves_with_module`

- 检查 `pos_grid` 出现在 `named_buffers()`。
- 检查 `pos_grid` 不出现在 `named_parameters()`。
- 将 module 移到当前可用 device，确认 buffer 跟随移动。
- 防止未来误把固定几何 grid 当成 learnable parameter。

`test_gradient_flow_to_features_and_optional_projection`

- 分别覆盖 `num_keypoints=None` 和 `num_keypoints=K`。
- 对 `out.sum()` 调用 `backward()`。
- 检查输入 features 有有限梯度。
- 当存在 `1x1 Conv2d` 时，也检查 projection 参数有有限梯度。
- 防止后续把 soft-argmax 改成不可导的 hard argmax 或 detach 操作。

`test_invalid_constructor_input_shape_raises_clear_error`

- 覆盖 `input_shape` 长度不是 3。
- 覆盖 `C/H/W` 为 0 或负数。
- 防止非法 feature map contract 在初始化时沉默通过。

`test_invalid_constructor_num_keypoints_raises_clear_error`

- 覆盖 `num_keypoints=0` 和负数。
- 防止创建无意义的 keypoint projection。

`test_forward_rejects_non_4d_input`

- 输入不是 `[B, C, H, W]` 时应报错。
- 防止训练或 encoder 接入时把 `[C, H, W]` 或其他 rank 的 tensor 静默 reshape。

`test_forward_rejects_channel_mismatch`

- 输入 channel 与 constructor 的 `C` 不一致时应报错。
- 防止 RGB encoder 输出 channel 变化后未同步更新 SpatialSoftmax input shape。

`test_forward_rejects_height_or_width_mismatch`

- 输入 `H/W` 与 constructor 不一致时应报错。
- 防止 crop/backbone 输出尺寸变化后 grid 仍使用旧尺寸。

这些测试共同保护后续视觉 encoder 集成：RGB encoder 只要输出 `[B, C, H, W]`，SpatialSoftmax 就会明确验证 shape，并稳定返回 `[B, K, 2]`。

### 6. 与原版实现的区别

数学行为保持一致：

- 都对每个 channel/keypoint map 的 `H * W` 做 softmax。
- 都用 `[-1, 1]` 坐标网格求 expected xy。
- 都输出 `[B, K, 2]`。
- 都支持不投影时 `K=C`，投影时 `K=num_keypoints`。

grid 构造方式变化：

- 原版使用 numpy：

  ```text
  np.meshgrid(np.linspace(...), np.linspace(...))
  ```

- 新版使用 PyTorch：

  ```text
  torch.linspace(...)
  torch.meshgrid(..., indexing="ij")
  ```

参数命名变化：

- 原版：`num_kp`
- 新版：`num_keypoints`

内部模块命名变化：

- 原版 projection 名为 `nets`。
- 新版 projection 名为 `keypoint_projection`。

是否兼容后续 RGB encoder：

- 兼容。后续 RGB encoder 只需要输出静态 shape `[B, C, H, W]`，并把对应 `(C, H, W)` 传给 `SpatialSoftmax`。
- 如果 RGB encoder 输出尺寸变化，forward validation 会立刻报错，而不是静默产生错误坐标。

为什么本阶段没有实现 ResNet/RGB encoder：

- SpatialSoftmax 是 RGB encoder 的下游模块。
- 先验证 SpatialSoftmax，可以把后续 RGB encoder 的测试拆成两个问题：backbone/crop 是否产生正确 feature map，以及 SpatialSoftmax 是否正确压缩 feature map。
- 如果这两个问题混在一个阶段实现，shape 错误和数学错误会更难定位。

### 7. 当前限制

当前只有 `SpatialSoftmax`。

仍未实现：

- RGB encoder
- image crop
- ResNet/backbone wrapper
- multi-camera encoder 组合逻辑
- U-Net
- timestep embedding
- FiLM residual block
- scheduler
- diffusion loss
- training `forward`
- sampling
- action queue
- `select_action`
- inference
- factory/global registry 注册

测试运行状态：

尝试运行：

```text
python -m pytest tests/policies/diffusion_policy/test_configuration_diffusion_policy.py tests/policies/diffusion_policy/test_processor_diffusion_policy.py tests/policies/diffusion_policy/test_tensor_contract.py tests/policies/diffusion_policy/test_spatial_softmax.py -q
```

结果：

```text
/usr/bin/python: No module named pytest
```

尝试运行：

```text
uv run python -m pytest tests/policies/diffusion_policy/test_configuration_diffusion_policy.py tests/policies/diffusion_policy/test_processor_diffusion_policy.py tests/policies/diffusion_policy/test_tensor_contract.py tests/policies/diffusion_policy/test_spatial_softmax.py -q
```

结果：

```text
snap-confine is packaged without necessary permissions and cannot continue
required permitted capability cap_dac_override not found in current capabilities:
  =
```

compile 检查通过：

```text
python -m compileall src/lerobot/policies/diffusion_policy tests/policies/diffusion_policy
```

其中包含：

```text
Compiling 'src/lerobot/policies/diffusion_policy/spatial_softmax.py'...
Compiling 'tests/policies/diffusion_policy/test_spatial_softmax.py'...
```

import smoke test 使用系统 Python 被本地依赖阻塞：

```text
ModuleNotFoundError: No module named 'torch'
```

`.venv/bin/python` 的 import smoke test 也被本地依赖阻塞：

```text
ModuleNotFoundError: No module named 'torch'
```

因此本阶段 pytest 未完整运行是本地环境问题，不是 SpatialSoftmax 代码逻辑或测试断言失败。

### 8. 下一阶段建议

推荐下一阶段：

```text
Phase 5 - 从零实现 RGB encoder
```

原因：

- SpatialSoftmax 已经提供稳定的 feature map 到 keypoint 坐标转换。
- RGB encoder 的自然输出就是 `[B, C, H, W]` feature map。
- 下一阶段可以专注实现 image crop、backbone feature extraction 和输出 shape 推断。
- RGB encoder 接上 SpatialSoftmax 后，可以把视觉输入压缩成 `[B, K * 2]` 或等价的低维视觉特征，再进入后续 state/action 条件模型。

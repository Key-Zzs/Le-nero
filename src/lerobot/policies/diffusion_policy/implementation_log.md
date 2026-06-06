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

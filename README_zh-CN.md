# Le-nero

本仓库基于 LeRobot，增加了双臂机器人遥操作、数据采集、策略训练和 DAgger 轮次闭环流程。本文只介绍日常使用需要理解的仓库结构、配置入口和运行调用关系。

下文假设仓库目录为：

实际使用时可以替换为自己的本地路径。

## 仓库获取与环境配置

首次拉取仓库时建议直接带上子模块：

```bash
git clone --recurse-submodules <Le-nero 仓库地址> Le-nero
cd Le-nero
```

如果已经 clone 过仓库，但子模块目录为空或缺文件，在仓库根目录执行：

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

日常更新主仓库和子模块：

```bash
cd Le-nero
git pull --ff-only
git submodule sync --recursive
git submodule update --init --recursive
git submodule update --remote --merge --recursive
```

切换主仓库分支：

```bash
git fetch origin
git switch <branch_name>
git submodule update --init --recursive
```

切换或更新双臂遥操作子模块：

```bash
cd Le-nero/dual_arm_data_collection/lerobot_dual_arm_teleop
git fetch origin
git switch main
git pull --ff-only
```

创建 Python 环境并安装根仓库与双臂遥操作包：

```bash
conda create -n dual_arm_teleop python=3.10 -y
conda activate dual_arm_teleop
python -m pip install --upgrade pip

cd Le-nero
pip install -e .

cd dual_arm_data_collection/lerobot_dual_arm_teleop
pip install -e .
```

Oculus Reader 不是通过当前 `.gitmodules` 管理的子模块，需要单独 clone 到指定目录：

```bash
cd Le-nero/dual_arm_data_collection/lerobot_dual_arm_teleop/teleoperators/oculus_teleoperator/oculus
git clone https://github.com/rail-berkeley/oculus_reader.git
cd oculus_reader
pip install -e .
```

如果该目录已经存在，只需要更新并重新安装：

```bash
cd Le-nero/dual_arm_data_collection/lerobot_dual_arm_teleop/teleoperators/oculus_teleoperator/oculus/oculus_reader
git pull --ff-only
pip install -e .
```

Oculus 连接还需要 ADB：

```bash
sudo apt install android-tools-adb
adb devices
```

首次 USB 连接时需要在头显中允许 USB 调试；无线连接时可以先通过 `adb shell ip route` 查看头显 IP，再执行 `adb connect <Oculus_IP>:5555`。

## 核心模块与调用机理

仓库的运行链路可以简化理解为：

```text
scripts/config/*.yaml
        |
        v
scripts/core/*.py 命令入口
        |
        +--> robots 创建真实机器人接口
        +--> teleoperators 创建 Oculus 遥操作输入
        +--> src/lerobot/policies 创建策略模型
        |
        v
LeRobot dataset / train / replay / visualize
```

### 策略层

策略代码位于：

```text
src/lerobot/policies
```

这里保留 LeRobot 的策略抽象和具体实现，例如 `act`、`diffusion`、`smolvla`、`pi0` 等。双臂遥操作脚本主要通过 `lerobot.policies.factory.make_policy` 和 `make_pre_post_processors` 创建策略对象及前后处理器。

当前双臂流程中最常用的是：

- `scripts/config/policy_config/act_train_config.yaml`：ACT 训练配置。
- `scripts/config/policy_config/act_reason_config.yaml`：ACT 推理/部署配置。
- `scripts/config/policy_config/diffusion_train_config.yaml`：Diffusion Policy 训练配置。
- `scripts/config/policy_config/diffusion_reason_config.yaml`：Diffusion Policy 推理/部署配置。

`scripts/core/policy_config_utils.py` 负责解析 `record_cfg.yaml`、`train_cfg.yaml` 或 `dagger_rounds_cfg.yaml` 中的策略配置路径。相对路径会优先按 `lerobot_dual_arm_teleop` 项目根目录解析，也支持直接写绝对路径。

### 机器人通讯接口定义层

机器人接口位于：

```text
dual_arm_data_collection/lerobot_dual_arm_teleop/robots
```

`robots/__init__.py` 是机器人注册表，当前注册的类型包括：

- `franka`
- `dobot_dual_arm`
- `nero_dual_arm`
- `franka_dual_arm`

脚本不会直接实例化某个具体机器人类，而是根据配置中的 `robot_type` 调用：

```python
create_robot_config(robot_type, **robot_cfg)
create_robot(robot_type, robot_config)
```

具体机器人类负责实现 LeRobot 期望的机器人接口，例如 `connect()`、`reset()`、`send_action()`、相机初始化、观测字段和动作字段定义。以 `nero_dual_arm` 为例，`dual_agilex_nero/nero_dual_arm.py` 通过 `NeroDualArmClient` 连接双臂 zerorpc 服务，并把双臂末端位姿、关节状态、夹爪命令和 RealSense 相机组织成 LeRobot 可记录的数据结构。

硬件相关参数不建议直接写在运行脚本里，而是放在：

```text
dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/config/DAS_config
```

例如 `nero_cofig.yaml` 定义 Nero 的机器人 IP、端口、夹爪参数、Oculus 映射和相机序列号。`run_record.py`、`run_replay.py`、`reset_robot.py` 会根据 `record.robot_type` 自动加载对应 DAS 配置；也可以在 `record_cfg.yaml` 中通过 `das_config_path` 显式指定。

### 工具脚本、数采系统与策略配置

主要脚本位于：

```text
dual_arm_data_collection/lerobot_dual_arm_teleop/scripts
```

常用目录含义：

- `scripts/core`：命令入口实现，包括采集、回放、可视化、重置、训练、DAgger。
- `scripts/config`：主流程配置，包含 `record_cfg.yaml`、`train_cfg.yaml`、`dagger_rounds_cfg.yaml`。
- `scripts/config/policy_config`：策略超参数配置，区分 train 和 reason 两类。
- `scripts/config/DAS_config`：硬件和遥操作细节配置。
- `scripts/tools`：数据集检查、RealSense 设备检查、数据集修补和重命名等工具。

核心配置文件：

- `record_cfg.yaml`：数据采集、策略推理、混合控制、回放和可视化共用的主配置。
- `train_cfg.yaml`：策略训练配置，包括数据集路径、输出目录、GPU、batch size、训练步数和 wandb。
- `dagger_rounds_cfg.yaml`：轮次式 DAgger 控制器配置，负责把采集、导出、训练串成闭环。
- `*_train_config.yaml`：策略训练时使用的模型结构和训练相关超参数。
- `*_reason_config.yaml`：策略推理或部署时使用的模型结构、设备和 checkpoint 参数。

`robot-record` 的三种运行模式由 `record.run_mode` 控制：

- `run_record`：纯遥操作采集。
- `run_policy`：加载策略 checkpoint，由策略控制机器人。
- `run_mix`：策略执行为主，操作者可接管，用于 DAgger 数据采集。

## 核心模块使用

`dual_arm_data_collection/lerobot_dual_arm_teleop/setup.py` 安装后会注册以下命令。安装命令为：

```bash
cd Le-nero/dual_arm_data_collection/lerobot_dual_arm_teleop
pip install -e .
```

命令入口：

| 命令 | 作用 | 默认配置 |
| --- | --- | --- |
| `robot-record` | 遥操作采集、策略执行或 run_mix 混合采集 | `scripts/config/record_cfg.yaml` |
| `robot-replay` | 回放已采集 episode | `scripts/config/record_cfg.yaml` 的 `replay` 段 |
| `robot-visualize` | 用 Rerun 可视化数据集 episode | `scripts/config/record_cfg.yaml` 的 `visualize` 段 |
| `robot-reset` | 根据配置连接机器人并回 home | `scripts/config/record_cfg.yaml` |
| `robot-train` | 训练 ACT 或 Diffusion Policy | `scripts/config/train_cfg.yaml` |
| `robot-dagger` | 运行轮次式 DAgger：采集、导出、训练下一轮策略 | `scripts/config/dagger_rounds_cfg.yaml` |
| `robot-dagger-export` | 从 raw run_mix 日志单独导出 DAgger 训练数据 | `scripts/config/dagger_rounds_cfg.yaml` 的 `dagger_export` 段 |
| `tools-check-dataset` | 检查本地 LeRobot 数据集信息 | 命令参数 |
| `tools-check-dagger-dataset` | 检查导出的 DAgger 数据集 | 命令参数 |
| `tools-check-rs` | 查看 RealSense 设备序列号 | 无 |
| `robot-help` | 打印命令摘要 | 无 |

所有核心命令都支持显式传入配置文件，推荐调试时总是写明路径：

```bash
robot-record --config scripts/config/record_cfg.yaml
robot-replay --config scripts/config/record_cfg.yaml
robot-visualize --config scripts/config/record_cfg.yaml
robot-reset --config scripts/config/record_cfg.yaml
robot-train --config scripts/config/train_cfg.yaml
robot-dagger --config scripts/config/dagger_rounds_cfg.yaml
robot-dagger-export --config scripts/config/dagger_rounds_cfg.yaml
```

采集前通常需要修改 `scripts/config/record_cfg.yaml`：

- `record.repo_id`：数据集名称，建议使用 `<robot_task<num>_step<num>/<description>`，例如 `nero_task3_step1/2mL_empty_right`。
- `record.robot_type`：选择 `nero_dual_arm`、`franka_dual_arm` 等机器人类型。
- `record.run_mode`：选择 `run_record`、`run_policy` 或 `run_mix`。
- `record.policy.type`、`config_path`、`pretrained_path`：仅在 `run_policy` 或 `run_mix` 时需要确认。
- `record.task`：任务描述、episode 数量、是否 resume、是否记录 success。
- `record.time`：episode 最大时长、reset 时长和 metadata 保存周期。
- `replay`、`visualize`：回放和可视化默认使用的数据集和 episode。

硬件参数通常在 `scripts/config/DAS_config/*.yaml` 中修改：

- `teleop.oculus_config.ip`：Oculus Quest IP。
- `teleop.oculus_config.*_pose_scaler` 和 `*_channel_signs`：左右手柄到机器人动作的映射。
- `robot.robot_ip`、`robot.robot_port`：机器人服务地址。
- `robot.use_gripper` 和夹爪参数：夹爪启用、开合阈值、最大开口和力。
- `cameras.*_serial`、`width`、`height`：RealSense 序列号和分辨率。

训练前通常需要修改 `scripts/config/train_cfg.yaml`：

- `train.dataset.repo_id` 和 `train.dataset.root`：训练数据集。
- `train.policy.type` 和 `train.policy.config_path`：策略类型和训练配置。
- `train.output_dir`、`job_name`：模型和日志输出位置。
- `train.training`：GPU 可见卡、显存限制、TF32 等训练设备设置。
- `train.steps`、`batch_size`、`num_workers`、`save_freq`：训练规模。
- `train.wandb`：wandb 项目和模式。

DAgger 前通常需要修改 `scripts/config/dagger_rounds_cfg.yaml`：

- `dagger_rounds.seed_repo_id` 或 `seed_dataset_path`：round 0 使用的种子数据集。
- `dagger_rounds.initial_pretrained_path`：可选，已有初始 checkpoint 时填写。
- `dagger_rounds.policy`：轮次中使用的策略类型，以及 train/reason 配置路径。
- `dagger_rounds.episodes_per_round`、`num_rounds`、`round_schedule`：每轮采集数量、轮数和训练步数策略。
- `dagger_rounds.output_root`：DAgger 轮次输出目录。
- `dagger_rounds.record_cfg_path`、`train_cfg_path`：被控制器动态改写并调用的基础配置。
- `dagger_rounds.policy_backend.export`：run_mix 日志导出为训练数据的规则。

## Quest 控制器按键

| 控制键 | 功能 |
| --- | --- |
| 左握持键 `LG` | 按住以启动左臂末端运动。在 `run_mix` 中会开始或持续左臂专家接管。 |
| 右握持键 `RG` | 按住以启动右臂末端运动。在 `run_mix` 中会开始或持续右臂专家接管。 |
| 左扳机 `LTr` | 控制左夹爪；按下关闭，松开打开。 |
| 右扳机 `RTr` | 控制右夹爪；按下关闭，松开打开。 |
| `Y` 按钮 | 在 `run_mix` 中将左夹爪通道交还给策略控制。 |
| `B` 按钮 | 在 `run_mix` 中将右夹爪通道交还给策略控制。 |
| `A` 按钮 | 在当前 teleoperator/robot 实现支持时请求机器人复位。 |
| 控制器位姿 | 在对应握持键按住时，控制对应机械臂的末端增量位姿。 |

如果启用了 `mirror_teleop`，左右控制器的对应关系会交换，并在发送给机器人前对位姿增量做镜像。

## DAgger/run_mix 控制定义

- 默认由策略控制机器人；人工输入只覆盖正在主动控制的通道。
- 按住 `LG` 或 `RG` 会让对应手臂进入专家接管。接管的第一帧标记为 `takeover_start`，持续接管帧标记为 `recovery`。
- `LTr` 和 `RTr` 独立控制夹爪，不要求同时接管手臂。夹爪接管使用 soft takeover：扳机命令需要先接近当前保持的夹爪值，手动夹爪控制才会生效，避免夹爪突然跳变。
- 按 `Y` 可将左夹爪交还给策略，按 `B` 可将右夹爪交还给策略；交还后需要先松开对应扳机，才能再次手动接管该夹爪。
- 使用左箭头丢弃失败、不完整、质量差或不适合作为训练示范的 episode。`full_episode.success_policy` 为 `recorded_is_success` 时，这一点尤其重要。

常用流程示例：

```bash
cd Le-nero/dual_arm_data_collection/lerobot_dual_arm_teleop

# 1. 查看相机序列号，填入 scripts/config/DAS_config/*.yaml
tools-check-rs

# 2. 检查策略配置能否正常解析，run_policy/run_mix 前推荐执行
robot-record --config scripts/config/record_cfg.yaml --dry-run-policy-config

# 3. 连接机器人并回 home
robot-reset --config scripts/config/record_cfg.yaml

# 4. 遥操作采集数据
robot-record --config scripts/config/record_cfg.yaml

# 5. 可视化或回放数据
robot-visualize --config scripts/config/record_cfg.yaml
robot-replay --config scripts/config/record_cfg.yaml

# 6. 训练策略
robot-train --config scripts/config/train_cfg.yaml

# 7. 运行 DAgger 轮次闭环
robot-dagger --config scripts/config/dagger_rounds_cfg.yaml
```

采集时常用按键约定：

- 右箭头：停止当前 episode 并保存。
- 左箭头：丢弃当前 episode。
- Esc：停止整个录制任务。
- Enter：继续下一段遥操作或下一条 episode。
- Ctrl+C：中断并清理未完成数据集。

## TODO

### 夹爪开合关键帧加权训练 TODO

本小节只记录代码检查结果和后续实施规划。真正实现前，ACT 和 Diffusion Policy 的默认训练行为必须保持不变；所有新能力都应通过默认关闭的配置显式启用。

代码路径检查摘要：

- 数据集入口与 action chunk / horizon 构造：`src/lerobot/datasets/factory.py`、`src/lerobot/datasets/lerobot_dataset.py`、`src/lerobot/datasets/utils.py`。
- 训练 dataloader 与日志：`src/lerobot/scripts/lerobot_train.py`、`dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/core/run_train.py`、`src/lerobot/utils/logging_utils.py`、`src/lerobot/rl/wandb_utils.py`。
- ACT loss 与配置：`src/lerobot/policies/act/modeling_act.py`、`src/lerobot/policies/act/configuration_act.py`、`dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/config/policy_config/act_train_config.yaml`。
- Diffusion Policy loss 与配置：`src/lerobot/policies/diffusion/modeling_diffusion.py`、`src/lerobot/policies/diffusion/configuration_diffusion.py`、`dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/config/policy_config/diffusion_train_config.yaml`。
- policy feature 推断与 batch 预处理：`src/lerobot/policies/factory.py`、`src/lerobot/processor/converters.py`、`src/lerobot/policies/act/processor_act.py`、`src/lerobot/policies/diffusion/processor_diffusion.py`。
- 可参考的已有标注、编辑、采样实现：`dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/debug/annotate_dataset_phase.py`、`dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/tools/preprocess_dataset.py`、`dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/tools/patch_lerobot_dataset_metadata.py`、`dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/tools/merge_lerobot_tasks.py`、`src/lerobot/scripts/lerobot_edit_dataset.py`、`dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/core/dagger_sampling.py`。
- 后续测试入口：`tests/datasets/test_datasets.py`、`tests/datasets/test_sampler.py`、`tests/processor/test_act_processor.py`、`tests/processor/test_diffusion_processor.py`、`tests/policies/test_policies.py`、`dual_arm_data_collection/lerobot_dual_arm_teleop/tests/test_dagger_sampling.py`。

必须保持的当前行为：

- ACT 通过 `ACTConfig.action_delta_indices = range(chunk_size)` 构造 action chunk。`LeRobotDataset._get_query_indices()` 会把跨越 episode 边界的索引 clamp 到 episode 内，并生成 `action_is_pad`。`ACTPolicy.forward()` 当前使用 `F.l1_loss(..., reduction="none")`，乘以 `~batch["action_is_pad"].unsqueeze(-1)` 后直接 `.mean()`；启用 VAE 时再加 `kl_weight * kld_loss`。
- Diffusion Policy 通过 `DiffusionConfig.action_delta_indices = range(1 - n_obs_steps, 1 - n_obs_steps + horizon)` 构造 action horizon。`DiffusionModel.compute_loss()` 当前在 `[B, horizon, action_dim]` 上计算 `F.mse_loss(pred, target, reduction="none")`，仅当 `do_mask_loss_for_padding` 为 true 时使用 `action_is_pad` mask，最后 `loss.mean()`。变量 `timesteps` 是 diffusion scheduler 的噪声 timestep，shape 为 `[B]`，不能和 action horizon step 混淆。
- parquet 新列只有写入 `meta/info.json` 的 features 后，才能通过 `LeRobotDataset.load_hf_dataset()` 使用 Hugging Face schema 读出。但是当前 `src/lerobot/processor/converters.py` 只保留 observation key、action、包含 `_is_pad` 的 padding key、task/index 元信息和 reward/done/truncated；普通 `annotation.*` 字段会在 policy preprocessing 中被丢弃。
- action 维度名称在 `dataset.meta.features["action"]["names"]` 中，`PolicyFeature` 只有 shape/type。夹爪维度应优先从 action feature names 推断，显式配置 indices 作为 fallback。当前双臂动作名包括 `left_gripper_cmd`、`right_gripper_cmd`、`left_gripper_cmd_bin`、`right_gripper_cmd_bin` 等模式。

后续配置草案，仅记录在 README 中，当前不要修改配置文件：

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

Phase 0: 代码检查与设计确认

- 目标：在实现前固定改动边界，包括 dataset 传播、ACT loss、DP loss、配置结构、日志、sampler 和测试。
- 涉及文件：`src/lerobot/datasets/factory.py`、`src/lerobot/datasets/lerobot_dataset.py`、`src/lerobot/datasets/utils.py`、`src/lerobot/processor/converters.py`、`src/lerobot/policies/factory.py`、`src/lerobot/scripts/lerobot_train.py`、`dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/core/run_train.py` 以及上文列出的 ACT/DP 文件。
- TODO 子项：确认 annotation 字段是否作为可时间查询的 feature；确认 `resolve_delta_timestamps()` 是否要给 `annotation.keyframe_weight` 和 `annotation.gripper_event` 加上与 `action` 相同的 delta indices；确认 annotation 在 preprocessing 后作为 complementary data 还是普通 batch key 保留；确认夹爪维度从 `meta.info["features"]["action"]["names"]` 推断；定义 disabled 配置下的数值等价测试。
- 验收标准：能明确写出 ACT 权重 shape 为 `[B, chunk_size]`，DP 权重 shape 为 `[B, horizon]`；旧数据集没有 annotation 字段时 fallback 到全 1 权重；disabled 配置与当前训练数值等价。
- 风险点：annotation key 可能被 preprocessing 静默丢弃；mean 分母变化会改变 loss scale；action horizon step 容易和 diffusion timestep 混淆。
- 本阶段不应改动的内容：训练 loss、dataset schema、policy config、dataloader sampler、脚本和测试。

Phase 1: 离线夹爪 transition 标注

- 目标：后续新增离线标注工具，识别夹爪 opening/closing 关键帧，并且不破坏原始 dataset。
- 涉及文件：未来新增 `dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/debug/annotate_gripper_transition.py`；参考 `annotate_dataset_phase.py`、`preprocess_dataset.py`、`patch_lerobot_dataset_metadata.py`、`merge_lerobot_tasks.py`。
- TODO 子项：从 action 或 gripper state 检测 opening/closing；支持连续夹爪值和二值夹爪值；支持左右臂分别标注；生成 `annotation.gripper_event` 和 `annotation.keyframe_weight`；可选生成 `annotation.left_gripper_event` 和 `annotation.right_gripper_event`；支持 `pre_window`、`post_window`；支持 dry-run、统计、可视化和 CSV 导出。
- 权重初始建议：`normal = 1.0`、`pre_closing = 2.0`、`closing = 4.0-8.0`、`post_closing = 2.0-3.0`、`pre_opening = 2.0`、`opening = 4.0-8.0`、`post_opening = 2.0-3.0`。
- 验收标准：dry-run 能输出每个 episode 的 transition 数量和比例；导出模式写入独立 dataset root，并更新 `meta/info.json` schema 和 parquet 列；视频和原始数据不被原地破坏。
- 风险点：不同机器人夹爪约定可能不同，如 `open=1/close=0`、`_cmd` 与 `_cmd_bin`、反向夹爪配置；噪声命令可能造成误标；window 过宽会把大段 episode 标成 keyframe。
- 本阶段不应改动的内容：ACT/DP 训练、dataloader、policy config、部署、机器人控制和原始 dataset。

Phase 2: Dataset feature 传播

- 目标：让 annotation 列在 action chunk / horizon 上对齐，并成为 loss 可见的 batch tensor。
- 涉及文件：`src/lerobot/datasets/factory.py`、`src/lerobot/datasets/lerobot_dataset.py`、`src/lerobot/datasets/utils.py`、`src/lerobot/processor/converters.py`、`src/lerobot/policies/act/processor_act.py`、`src/lerobot/policies/diffusion/processor_diffusion.py`。
- TODO 子项：在 dataset `meta/info.json` features 和 Hugging Face schema 中注册 `annotation.keyframe_weight`、`annotation.gripper_event`；让 annotation 字段使用与 `action` 相同的 temporal delta indices；验证 `LeRobotDataset._get_query_indices()` 返回与 `action_is_pad` 对齐的 annotation tensor；让 preprocessing 保留 annotation tensor 但不归一化；旧数据缺列时 fallback 到全 1 权重。
- 验收标准：ACT batch 中 `annotation.keyframe_weight` 为 `[B, chunk_size]`；DP batch 中为 `[B, horizon]`；padding 与 `action_is_pad` 对齐；旧数据集和 disabled config 保持当前 loss 使用的字段不变。
- 风险点：当前 `resolve_delta_timestamps()` 只处理 reward、action 和 observation；当前 `batch_to_transition()` 会丢弃普通 annotation key；如果把 annotation 加进 policy features，可能被错误归一化或错误分类。
- 本阶段不应改动的内容：loss weighting 数学、sampler 行为、policy 架构、机器人采集和部署。

Phase 3: ACT weighted loss

- 目标：将 ACT action reconstruction loss 扩展为默认关闭的 per-timestep / per-action-dim weighted loss。
- 涉及文件：`src/lerobot/policies/act/modeling_act.py`、`src/lerobot/policies/act/configuration_act.py`、`dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/config/policy_config/act_train_config.yaml`，其中配置文件只在后续实现阶段记录新增配置，不应在当前阶段改默认行为。
- TODO 子项：disabled 路径保持当前 L1 loss 完全等价；enabled 时计算 `[B, chunk_size, D]` 的 `loss_per_dim = abs(pred_action - target_action)`；从 `annotation.keyframe_weight` 读取 timestep weight；根据夹爪维度生成 `action_dim_weight`；先应用 `action_is_pad` 再 reduce；按 `max_weight` clamp；VAE KLD 部分保持现有语义。
- 目标公式，当前不要实现：

```python
loss = mean(abs(pred_action - target_action))
loss = masked_weighted_mean(loss_per_dim * timestep_weight * action_dim_weight)
```

- 验收标准：disabled config 与当前 `l1_loss` 数值一致；enabled config 正确广播 `[B, S]`、`[D]`、`[B, S, D]`；padding timestep 不参与 loss；`loss_dict` 可上报 `normal_frame_loss`、`keyframe_loss`、`gripper_loss`、`pose_loss`、`opening_loss`、`closing_loss`。
- 风险点：按权重和归一化或按元素数归一化会改变梯度尺度；transition 事件中过度加权 pose dim 可能过拟合上下文；只加权 gripper dim 可能忽略靠近和释放阶段的位姿修正。
- 本阶段不应改动的内容：ACT 模型结构、推理队列、temporal ensembling、VAE KLD 语义、dataset 写入和 DP loss。

Phase 4: DP weighted denoising loss

- 目标：为 Diffusion Policy 增加默认关闭的 action horizon step 加权 denoising loss，并避免与 diffusion noise timestep 混淆。
- 涉及文件：`src/lerobot/policies/diffusion/modeling_diffusion.py`、`src/lerobot/policies/diffusion/configuration_diffusion.py`、`dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/config/policy_config/diffusion_train_config.yaml`，其中配置文件只在后续实现阶段记录新增配置，不应在当前阶段改默认行为。
- TODO 子项：disabled 路径保持当前 `F.mse_loss(pred, target, reduction="none").mean()` 完全等价；enabled 时计算 `[B, H, D]` 的 `mse_per_dim`；从 `annotation.keyframe_weight` 读取 horizon weight；应用 `action_dim_weight`；保留 `timesteps` 专指 diffusion scheduler 噪声 timestep `[B]`；padding 行为遵循当前 `do_mask_loss_for_padding` 语义，除非新配置明确要求改变。
- 目标公式，当前不要实现：

```python
loss = mean(mse(pred_noise, target_noise))
loss = weighted_mean(mse_per_dim * horizon_weight * action_dim_weight)
```

- 验收标准：disabled config 与当前 DP loss 数值一致；enabled config 正确广播 `[B, H, D]` 权重；horizon weight 不参与 diffusion scheduler timestep 索引；padding 行为被文档和测试覆盖。
- 风险点：DP 默认当前不 mask padding，误改会改变行为；稀疏 transition 加权可能让 denoising 过度偏向夹爪事件，损伤平滑靠近轨迹。
- 本阶段不应改动的内容：scheduler 行为、`num_train_timesteps`、推理采样、U-Net 架构、ACT loss 和 sampler。

Phase 5: 可选 keyframe-aware sampling

- 目标：将采样加权作为第二优先级能力，只用于提高包含 transition chunk 的 batch 出现概率；主方案仍是训练侧 loss weighting。
- 涉及文件：`src/lerobot/datasets/sampler.py`、`src/lerobot/scripts/lerobot_train.py`、`dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/core/run_train.py`、`dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/core/dagger_sampling.py`。
- TODO 子项：如果一个 index 对应的 action chunk / horizon 含 transition，则标为正样本；采样倍率限制在约 2-4 倍；不要过度重复整个 episode；定义与 DP `EpisodeAwareSampler` 和 DAgger source-aware `WeightedRandomSampler` 的组合策略；记录 transition chunk sample ratio。
- 验收标准：sampler 默认关闭且可选；不开 sampler 时 loss weighting 仍可工作；sampler 不绕过 episode boundary 和 padding 规则；batch 中 transition chunk 比例提高但不过度主导 epoch。
- 风险点：过采样可能导致提前闭合、提前张开或夹爪抖动；PyTorch DataLoader 只能直接使用一个 sampler，需要为 DP episode-aware dropping、DAgger source weighting 和 keyframe weighting 设计统一路径。
- 本阶段不应改动的内容：loss weighting、annotation schema、ACT/DP 模型代码和 dataset 内容。

Phase 6: 指标、调试与可视化

- 目标：让加权训练过程可审计，便于发现标注和权重问题。
- 涉及文件：`src/lerobot/scripts/lerobot_train.py`、`dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/core/run_train.py`、`src/lerobot/utils/logging_utils.py`、`src/lerobot/rl/wandb_utils.py`、ACT/DP 的 `loss_dict` 输出、未来 annotation 脚本。
- TODO 子项：记录 `total_loss`、`normal_frame_loss`、`keyframe_loss`、`gripper_loss`、`pose_loss`、`opening_loss`、`closing_loss`、`keyframe_ratio_per_batch`、`weighted_loss_mean_weight`、`max_weight`、`transition_chunk_sample_ratio`；增加 annotation 分布统计和 CSV/plot 输出；确保 DDP/Accelerate 下只记录 scalar。
- 验收标准：wandb 通过现有 `WandBLogger.log_dict()` 收到 train 前缀 scalar 指标；本地日志可读；annotation report 能在训练前暴露 opening/closing 分布不均。
- 风险点：per-batch 指标可能噪声大；当前 wandb wrapper 会忽略 tensor 或非 scalar；多进程训练可能需要先聚合再记录。
- 本阶段不应改动的内容：训练数学、模型结构、dataset 导出格式和部署行为。

Phase 7: 测试与回归安全

- 目标：在真实训练启用前补齐针对性测试。
- 涉及文件：`tests/datasets/test_datasets.py`、`tests/datasets/test_sampler.py`、`tests/processor/test_act_processor.py`、`tests/processor/test_diffusion_processor.py`、`tests/policies/test_policies.py`，以及可能新增到 `tests/policies/` 和 `dual_arm_data_collection/lerobot_dual_arm_teleop/tests/` 的专门测试。
- TODO 子项：annotation 脚本单元测试；dataset 新增列读取和 temporal alignment 测试；ACT weighted loss shape/mask 测试；DP weighted loss shape/horizon weight 测试；disabled config 数值等价测试；padding mask 不参与加权测试；gripper dim weight 广播测试；旧 dataset 缺 annotation 字段时 fallback 到全 1 权重测试；若实现 Phase 5，补 sampler cap 测试。
- 验收标准：disabled ACT 和 DP loss 与旧实现数值一致；旧 dataset 可以无 annotation 字段正常加载和训练；padding 不产生正向加权贡献；左右夹爪 feature name 能正确推断 gripper dim。
- 风险点：端到端 policy artifact 测试成本高，早期 loss 数学更适合小型确定性测试；DP 随机噪声需要固定 seed 或拆出 deterministic loss helper。
- 本阶段不应改动的内容：生产配置、默认训练行为、dataset 文件和 test artifacts，除非实现 PR 明确需要。

Phase 8: 训练与 rollout 验证

- 目标：在真机前逐步验证权重效果和副作用。
- 涉及文件：`dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/config/train_cfg.yaml`、`dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/config/policy_config/act_train_config.yaml`、`dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/config/policy_config/diffusion_train_config.yaml`，部署配置只应在离线验证通过后修改。
- TODO 子项：小数据 annotated smoke test；先只开 ACT conservative weights；再开 DP；逐步 sweep `gripper_dim_weight`、event weight 和 pre/post window；比较 normal/keyframe loss 曲线；检查 rollout 是否出现提前闭合、提前张开、夹爪抖动；真机验证前后对比成功率和失败类型。
- 验收标准：disabled 和 enabled 配置都能完成训练；enabled 训练有可见 keyframe loss 信号但 total loss 不爆炸；rollout 视频中的开合发生在预期任务阶段；真机验证前有明确 rollback checkpoint。
- 风险点：稀有事件过度加权可能损伤非 transition 行为；标注错误会被放大；离线或仿真指标不一定预测真实夹爪时序。
- 本阶段不应改动的内容：训练过程中临时更改 annotation 定义、机器人安全限制、默认生产 checkpoint，以及没有 rollback 的部署策略。

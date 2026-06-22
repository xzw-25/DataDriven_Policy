# Data-Driven Proxy Model

本工程实现一个面向车辆轨迹与速度联合控制的神经网络控制器。整体目标是用专家控制器生成监督数据，先进行模仿学习，再基于车辆动力学闭环 rollout 做可微分微调，最后导出 TorchScript / ONNX 模型并在部署运行时中验证。

控制器固定使用 22 维输入特征，神经网络输出归一化的方向盘转角和有符号纵向加速度需求。部署管线会将神经网络输出反归一化，经过限幅、安全检查、纵向驱动/制动分配后生成最终执行指令。

## 文档索引

- [神经网络车辆控制器整体设计方案](neural_network_vehicle_controller_design.md)
- [工程目录结构与模块说明](docs/project_structure.md)
- [训练 loss 计算方法](docs/training_losses.md)
- [工程总体架构图](docs/diagrams/vehicle_controller_project_architecture.drawio)
- [神经网络结构图](docs/diagrams/vehicle_controller_neural_network_architecture.drawio)
- [架构图说明](docs/diagrams/README.md)

## 整体框架

```text
典型场景仿真数据                 原始 PKL 数据
    |                               |
    v                               v
专家控制器 + 车辆模型闭环仿真      task_manifest + record_pkl 切片
    |                               |
    v                               v
NPZ 训练数据 <------------------ raw_data -> features / targets
    |  features / raw_features / targets / physical_targets
    v
模仿学习训练
    |
    v
baseline checkpoint
    |--------------------------|
    v                          v
离线验证/训练展示              可微分闭环 rollout 微调
                               |
                               v
                         closed-loop checkpoint
                               |
                               v
                      TorchScript / ONNX 导出
                               |
                               v
                    部署包验证 + 实时控制管线
```

核心运行链路由以下部分组成：

- `data/`：专家控制器、典型场景、数据生成、NPZ 数据集读取。
- `features/`：坐标系转换后的 22 维特征构建、误差计算、归一化、输入校验。
- `models/`：MLP、Direct MLP、GRU 控制器模型。
- `training/`：监督训练、loss、指标、检查点、离线图表、可微分闭环训练。
- `vehicle/`：运动学自行车模型、执行器参数和车辆参数。
- `control/`：神经网络策略、命令限幅、纵向分配、安全监督和控制 pipeline。
- `simulation/`：闭环仿真、典型场景 rollout、训练展示图生成。
- `deployment/`：TorchScript / ONNX 运行时、实时控制器、健康监控和部署包验证。
- `adapters/`：回放适配器、ROS2 适配器占位与接口封装。

## 环境安装

推荐 Python 3.10+。

```bash
python3 -m pip install -e .
```

开发与测试依赖：

```bash
python3 -m pip install -e ".[dev]"
```

数据处理依赖：

```bash
python3 -m pip install -e ".[data]"
```

ONNX 导出和部署验证依赖：

```bash
python3 -m pip install -e ".[deploy]"
```

仓库中的 `scripts/` 入口会自动加载 `src/`，通常不需要手动设置 `PYTHONPATH`。在其他工程中调用本包时，建议按上面的方式安装为 editable package。

## 快速开始

从生成训练数据到导出部署模型的推荐流程：

```bash
# 1. 生成专家控制器闭环训练数据
python3 scripts/generate_training_data.py --config configs/default.yaml

# 2. 模仿学习训练 baseline
python3 scripts/train_imitation.py \
  --config configs/default.yaml \
  --dataset data/processed/simulated_controller_dataset.npz

# 3. 离线验证网络输出与生成数据 expert 控制输出
python3 scripts/evaluate_offline.py \
  artifacts/checkpoints/baseline.pt \
  data/processed/simulated_controller_dataset.npz \
  --model-config configs/model/direct_mlp_controller.yaml

# 4. 闭环验证
python3 scripts/evaluate_closed_loop.py \
  --config configs/default.yaml \
  --checkpoint artifacts/checkpoints/baseline.pt

# 5. 可微分闭环 rollout 微调
python3 scripts/train_closed_loop.py \
  --config configs/default.yaml \
  --initial-checkpoint artifacts/checkpoints/baseline.pt

# 6. 导出部署模型
python3 scripts/export_model.py \
  artifacts/checkpoints/closed_loop.pt \
  --format both
```

基于原始 PKL 数据的监督训练与验证流程：

```bash
# 1. 从 task manifest 和 record_pkl 提取每个 entry/part 对应的原始信号片段
python3 scripts/extract_task_raw_data.py \
  --task-manifest data/raw/ai_control_dataset/task_manifest/clean_ad_policy_sim_v1_aba9e399.pkl \
  --record-root data/raw/ai_control_dataset/record_pkl \
  --output data/interim/clean_ad_policy_sim_v1_aba9e399_raw_data.pkl

# 2. 将 reference_traj + pose 构造 22 维 feature，并提取 control_signal 作为 2 维监督 target
python3 scripts/build_features_from_raw_data.py \
  --raw-data data/interim/clean_ad_policy_sim_v1_aba9e399_raw_data.pkl \
  --output data/processed/clean_ad_policy_sim_v1_aba9e399_imitation_dataset.npz

# 3. 一键构建 NPZ 并运行模仿学习训练
python3 scripts/train_imitation_from_raw_data.py \
  --raw-data data/interim/clean_ad_policy_sim_v1_aba9e399_raw_data.pkl \
  --dataset-output data/processed/clean_ad_policy_sim_v1_aba9e399_imitation_dataset.npz \
  --checkpoint-output artifacts/checkpoints/raw_data_imitation.pt \
  --epochs 50

# 4. 对比网络控制器输出和原始监督 target
python3 scripts/validate_supervised_controller.py \
  --checkpoint artifacts/checkpoints/raw_data_imitation.pt \
  --dataset data/processed/clean_ad_policy_sim_v1_aba9e399_imitation_dataset.npz \
  --output-dir artifacts/reports/supervised_validation \
  --dataset-label raw_data_imitation

# 5. 使用训练得到的网络控制器做闭环评估
python3 scripts/evaluate_closed_loop.py \
  --config configs/default.yaml \
  --checkpoint artifacts/checkpoints/raw_data_imitation.pt
```

运行测试：

```bash
python3 -m pytest
```

## 输入输出契约

神经网络输入顺序固定为：

```text
x1, y1, x2, y2, x3, y3, x4, y4, x5, y5,
kappa, e_lat, e_v, e_s, a_ref, v_ref, s_ref,
vx, vy, ax, ay, r
```
构造feature和target使用到的pkl中的信号：
```text
reference_traj/
  status/
    is_replan: int16 (5993,)  (values=[-1, 0, 1])
  time/
    timestamp: float64 (5993,)  (min=1.779e+09 max=1.779e+09 mean=1.779e+09)
  trajectory/
    points/
      a: float32 (5993, 140)  (min=-6.05 max=4.768)
      da: float32 (5993, 140)  (min=-13.74 max=15.92)
      kappa: float32 (5993, 140)  (min=-0.04497 max=0.1075)
      relative_time: float32 (5993, 140)  (min=-0.9075 max=5.116)
      s: float32 (5993, 140)  (min=-9.358 max=84.29)
      theta: float32 (5993, 140)  (min=0.6309 max=1.452)
      v: float32 (5993, 140)  (min=1.587 max=17.34)
      x: float32 (5993, 140)  (min=1262 max=1827)
      y: float32 (5993, 140)  (min=3534 max=4185)
    valid_length: int16 (5993,)  (min=0 max=140)

pose/
  motion/
    angular_velocity_flu: float64 (5993, 3)  (min=-0.08516 max=0.08335)
    linear_acceleration_flu: float64 (5993, 3)  (min=-1.942 max=2.623)
    linear_velocity_enu: float64 (5993, 3)  (min=-0.3993 max=12.42)
  orientation/
    heading: float64 (5993,)  (min=0.75 max=0.9347 mean=0.8596)
    quaternion_xyzw: float64 (5993, 4)  (min=0 max=0.9305)
  position/
    position_enu: float64 (5993, 3)  (min=0 max=4130)
  time/
    timestamp: float64 (5993,)  (min=1.779e+09 max=1.779e+09 mean=1.779e+09)

control_signal/
  command/
    target_longitudinal_acceleration: float32 (5993,)  (min=-0.4251 max=0.9454 mean=0.07632)
    target_longitudinal_torque: float32 (5993,)  (min=0 max=821 mean=101)
    target_steering_wheel_angle: float32 (5993,)  (min=-16.16 max=33.28 mean=1.973)
  longitudinal_status/
    acc_brakepreferred: int16 (5993,)  (values=[-1, 0, 1])
    acc_torque_request_status: int16 (5993,)  (values=[-1, 0, 1])
    driveoff_request: int16 (5993,)  (values=[-1, 0])
    standstill_request: int16 (5993,)  (values=[-1, 0])
  mode/
    control_mode: int16 (5993,)  (values=[-1, 0, 6])
    control_noa_enable: int16 (5993,)  (values=[-1, 0, 1])
    lat_override: int16 (5993,)  (values=[-1, 0])
    lon_override: int16 (5993,)  (values=[-1, 0, 1])
  time/
    timestamp: float64 (5993,)  (min=1.779e+09 max=1.779e+09 mean=1.779e+09)

```



其中 5 个参考轨迹点不再使用固定米制前视距离，而是由预瞄时间决定。默认配置为：

```yaml
preview_times_s: [0.1, 0.2, 0.3, 0.4, 0.5]
```

每个控制周期按参考速度规划换算为弧长偏移：

```text
preview_distance = max(v_ref * t + 0.5 * a_ref * t^2, 0)
```

含义：

- `x1..x5, y1..y5`：车辆局部坐标系下的 5 个预瞄参考轨迹点，默认对应 `0.1s, 0.2s, 0.3s, 0.4s, 0.5s`。
- `kappa`：按配置权重计算的参考曲率。
- `e_lat`：横向误差，使用当前 pose 与参考轨迹的空间最近点计算。
- `e_v`：速度误差，使用按当前 `pose.timestamp` 插值得到的纵向参考速度计算，`e_v = v_ref - vx`。
- `e_s`：纵向进度误差，使用按当前 `pose.timestamp` 插值得到的纵向参考 `s_ref` 计算，`e_s = s_ref - s_vehicle`。
- `a_ref`：从自车在参考线上的投影位置向前预瞄 `0.5s` 时，对应参考轨迹点的加速度。
- `v_ref, s_ref`：当前 pose 绝对时间对应的参考速度、参考纵向位置。纵向时间轴直接使用 `reference_traj.time.timestamp + relative_time`，再根据 `pose.timestamp` 在线性插值得到。
- `vx, vy, ax, ay, r`：车辆当前状态量。

神经网络直接输出：

```text
steering_normalized, signed_accel_normalized
```

部署侧按模型配置中的 `steering_limit_deg` 和 `accel_limit_mps2` 反归一化得到：

```text
steering_wheel_angle_cmd_deg, signed_acceleration_cmd
```

之后由纵向分配模块生成互斥的：

```text
drive_torque_cmd, brake_decel_cmd
```

车辆执行器和动力学模型内部仍使用 rad，控制 pipeline 会在写入 `VehicleCommand.steering_wheel_angle_rad` 前完成 `deg -> rad` 转换。

绘图中的控制量命名约定：

- `raw neural command`：神经网络输出反归一化后的原始控制需求。
- `limited neural command`：经过控制限幅后的网络控制需求。
- `executed command`：经过限幅、纵向分配、安全监督和执行器模型后的最终执行控制量。
- `generated-data expert`：生成训练数据时专家控制器在对应状态下输出的控制量。

## 配置文件

主配置入口为 [configs/default.yaml](configs/default.yaml)。

常用配置：

- [configs/data/dataset.yaml](configs/data/dataset.yaml)：数据目录、采样周期、预瞄时间、曲率权重、数据划分比例。
- [configs/data/normalization.yaml](configs/data/normalization.yaml)：22 维特征归一化参数。
- [configs/data/simulation_generation.yaml](configs/data/simulation_generation.yaml)：仿真数据生成、随机扰动、专家控制器参数。
- [configs/model/direct_mlp_controller.yaml](configs/model/direct_mlp_controller.yaml)：Direct MLP 结构和控制输出尺度。
- [configs/model/mlp_controller.yaml](configs/model/mlp_controller.yaml)：三分支 MLP 结构。
- [configs/model/gru_controller.yaml](configs/model/gru_controller.yaml)：GRU 时序控制器结构。
- [configs/training/imitation.yaml](configs/training/imitation.yaml)：模仿学习 batch、epoch、学习率和 loss 权重。
- [configs/training/closed_loop.yaml](configs/training/closed_loop.yaml)：闭环 rollout horizon、学习率和闭环 loss 权重。
- [configs/vehicle/vehicle_params.yaml](configs/vehicle/vehicle_params.yaml)：车辆动力学参数。
- [configs/vehicle/actuator_limits.yaml](configs/vehicle/actuator_limits.yaml)：执行器物理限制。
- [configs/deployment/runtime.yaml](configs/deployment/runtime.yaml)：部署运行时后端、模型路径、超时等。
- [configs/deployment/safety_limits.yaml](configs/deployment/safety_limits.yaml)：部署侧输入和车辆状态安全边界。

## Pipeline 详解

### 0. 原始 PKL 数据到监督训练和验证

原始数据链路面向已构建好的 `task_manifest` 与 `record_pkl`。它不再用仿真 expert 生成标签，而是直接使用记录中的控制信号作为网络输出监督目标。

输入文件示例：

```text
data/raw/ai_control_dataset/task_manifest/clean_ad_policy_sim_v1_aba9e399.pkl
data/raw/ai_control_dataset/record_pkl/PP381/20260521/*_RPKL.pkl
```

#### 0.1 提取 raw_data

```bash
python3 scripts/extract_task_raw_data.py \
  --task-manifest data/raw/ai_control_dataset/task_manifest/clean_ad_policy_sim_v1_aba9e399.pkl \
  --record-root data/raw/ai_control_dataset/record_pkl \
  --output data/interim/clean_ad_policy_sim_v1_aba9e399_raw_data.pkl
```

处理逻辑：

- 遍历 task manifest 中的 `entries`。
- 对每个 `entry.parts[]`，根据 `record_pkl_id` 在 `record_pkl` 根目录查找对应 pkl。
- 使用闭区间 `[start_index_in_bag, end_index_in_bag]` 切片。
- 提取并按 part 顺序拼接：
  - `reference_traj`
  - `pose`
  - `control_signal`

输出字段位于每个 entry 的 `raw_data` 下：

```text
raw_data/
  reference_traj/
  pose/
  control_signal/
  parts/
  frame_count
```

#### 0.2 构造 22 维输入和 2 维监督 target

```bash
python3 scripts/build_features_from_raw_data.py \
  --raw-data data/interim/clean_ad_policy_sim_v1_aba9e399_raw_data.pkl \
  --output data/processed/clean_ad_policy_sim_v1_aba9e399_imitation_dataset.npz
```

逐帧处理逻辑：

- `reference_traj.trajectory.points.x/y` 与 `valid_length` 组成有效轨迹点。
- 同帧 `s/kappa/v/a` 写入 `TrajectoryPoint`，用于曲率、预瞄距离和误差计算。
- `pose.position.position_enu` 提供自车全局位置。
- `pose.orientation.heading` 提供自车航向。
- `pose.motion.linear_velocity_enu` 按 heading 转成车体系 `vx/vy`。
- `pose.motion.linear_acceleration_flu` 提供 `ax/ay`。
- `pose.motion.angular_velocity_flu[:, 2]` 提供横摆角速度 `r`。
- 横向误差使用当前 pose 与有效参考轨迹点的空间最近点计算。
- 5 个预瞄 sample point 不从 reference 历史拼接点开始采样，而是先将自车位置投影到有效参考轨迹线段上，再从该投影点沿参考轨迹向前按预瞄距离采样。
- 纵向参考点独立按时间插值：使用当前帧有效 `relative_time` 构造点级绝对时间
  `reference_point_time = reference_traj.time.timestamp + relative_time`。
- 使用 `pose.time.timestamp` 在 `reference_point_time` 上对 `x/y/s/kappa/v/a` 线性插值，得到当前纵向参考点；`e_v/e_s/v_ref/s_ref` 和按预瞄时间换算的预瞄距离使用该插值参考点。
- `a_ref` 独立从自车投影点向前预瞄 `0.5s` 对应的轨迹点取得，预瞄距离同样由当前纵向参考点的 `v_ref/a_ref` 换算得到。
- 调用 `vehicle_controller.data.feature_builder` 和 `vehicle_controller.features.FeatureBuilder` 生成固定顺序 22 维输入。
- 从 `control_signal.command` 中提取：
  - `target_steering_wheel_angle`
  - `target_longitudinal_acceleration`

NPZ 字段：

```text
features              (N, 22)  # 归一化后的网络输入
raw_features          (N, 22)  # 原始物理量输入
targets               (N, 2)   # 归一化监督标签，训练使用
physical_targets      (N, 2)   # 物理单位监督标签，分析和画图使用
target_valid_mask     (N,)     # 标签有效性，训练加载时自动过滤无效帧
target_names          (2,)     # steering_target_deg, signed_accel_target_mps2
clip_ids              (N,)
entry_indices         (N,)
frame_indices         (N,)
timestamps_s          (N,)
metadata_json
```

`targets` 与 `physical_targets` 顺序均为：

```text
steering_target_deg, signed_accel_target_mps2
```

其中 `targets` 会按模型配置归一化：

```text
targets[:, 0] = physical_targets[:, 0] / steering_limit_deg
targets[:, 1] = physical_targets[:, 1] / accel_limit_mps2
```

`physical_targets` 保留真实物理单位，便于输出对比验证和部署侧反归一化检查。

#### 0.3 可视化 reference_traj 轨迹

```bash
python3 scripts/plot_reference_trajectories.py \
  --raw-data data/interim/clean_ad_policy_sim_v1_aba9e399_raw_data.pkl \
  --output-dir artifacts/reports/reference_trajectories
```

该脚本将每帧的 `x/y` 组成轨迹点，输出总览图和每个 clip 的 reference trajectory 曲线。

#### 0.4 模仿学习训练

可分两步运行：

```bash
python3 scripts/build_features_from_raw_data.py \
  --raw-data data/interim/clean_ad_policy_sim_v1_aba9e399_raw_data.pkl \
  --output data/processed/clean_ad_policy_sim_v1_aba9e399_imitation_dataset.npz

python3 scripts/train_imitation.py \
  --config configs/default.yaml \
  --dataset data/processed/clean_ad_policy_sim_v1_aba9e399_imitation_dataset.npz \
  --output artifacts/checkpoints/raw_data_imitation.pt
```

也可以一键构建数据并训练：

```bash
python3 scripts/train_imitation_from_raw_data.py \
  --raw-data data/interim/clean_ad_policy_sim_v1_aba9e399_raw_data.pkl \
  --dataset-output data/processed/clean_ad_policy_sim_v1_aba9e399_imitation_dataset.npz \
  --checkpoint-output artifacts/checkpoints/raw_data_imitation.pt \
  --epochs 50
```

训练时实际使用：

```text
model_input  = features[:, 0:22]
supervision  = targets[:, 0:2]
```

`loss_curve.png` 的纵轴使用对数坐标，便于观察训练后期小 loss 的变化。

#### 0.5 输出对比验证

```bash
python3 scripts/validate_supervised_controller.py \
  --checkpoint artifacts/checkpoints/raw_data_imitation.pt \
  --dataset data/processed/clean_ad_policy_sim_v1_aba9e399_imitation_dataset.npz \
  --output-dir artifacts/reports/supervised_validation \
  --dataset-label raw_data_imitation
```

验证内容：

- 将 NPZ 中的 22 维 `features` 输入 checkpoint 对应的神经网络控制器。
- 得到归一化网络输出，并按模型配置反归一化为物理控制量。
- 与 `physical_targets` 中保存的原始监督标签对比。
- 输出归一化误差和物理单位误差。
- 绘制每个 clip 的方向盘转角、纵向加速度对比图。

输出示例：

```text
artifacts/reports/supervised_validation/
  raw_data_imitation_metrics.json
  raw_data_imitation_predictions.npz
  raw_data_imitation_<clip_id>_control_comparison.png
```

#### 0.6 闭环评估测试

```bash
python3 scripts/evaluate_closed_loop.py \
  --config configs/default.yaml \
  --checkpoint artifacts/checkpoints/raw_data_imitation.pt
```

闭环评估会把网络输出重新接入控制 pipeline 和车辆模型，检查离线拟合后的网络在闭环 rollout 中的跟踪误差、速度响应、控制输出和稳定性。

### 1. 生成仿真训练数据

```bash
python3 scripts/generate_training_data.py --config configs/default.yaml
```

默认使用 `configs/data/simulation_generation.yaml` 中的设置，基于左转、右转、启停、左变道、右变道、双移线等典型场景生成数据。

专家控制器：

- 横向：曲率预瞄前馈 + 航向误差反馈 + 横向误差反馈。
- 纵向：位置外环 PID + 速度内环 PID。
- 被控对象：运动学自行车模型。

默认输出：

```text
data/processed/simulated_controller_dataset.npz
```

NPZ 字段：

- `features`：归一化后的 22 维输入。
- `raw_features`：物理单位输入特征。
- `targets`：归一化控制标签。
- `physical_targets`：物理单位 expert 控制标签。
- `scenario_ids`：场景或 rollout 标识。
- `timestamps_s`：样本时间戳。
- `metadata`：生成配置和数据说明。

常用参数：

```bash
python3 scripts/generate_training_data.py \
  --config configs/default.yaml \
  --output data/processed/custom_dataset.npz \
  --repetitions 8 \
  --seed 7
```

### 2. 外部 CSV 数据转换

如果已有外部表格数据，可转换为训练用 NPZ：

```bash
python3 scripts/prepare_dataset.py input.csv data/processed/controller_dataset.npz
```

CSV 必须包含固定 22 维输入列和目标控制列。列名定义在 `vehicle_controller.constants.FEATURE_NAMES` 和 `vehicle_controller.data.schema.TARGET_NAMES` 中。

### 3. 计算归一化参数

```bash
python3 scripts/compute_normalization.py \
  data/processed/simulated_controller_dataset.npz \
  artifacts/normalization/normalization.json
```

如果 NPZ 中存在 `raw_features`，脚本会优先使用原始物理量计算均值和标准差。

### 4. 模仿学习训练

```bash
python3 scripts/train_imitation.py \
  --config configs/default.yaml \
  --dataset data/processed/simulated_controller_dataset.npz \
  --output artifacts/checkpoints/baseline.pt
```

训练输出：

- `artifacts/checkpoints/baseline.pt`：模型检查点，包含模型参数、优化器状态和模型配置。
- `artifacts/reports/imitation_training/loss_curve.png`：loss 随迭代下降曲线。
- `artifacts/reports/imitation_training/loss_history.csv`：batch 和 epoch loss 历史。
- `artifacts/reports/training_showcase/`：训练典型场景闭环展示图。

使用其他模型配置：

```bash
python3 scripts/train_imitation.py \
  --model-config configs/model/mlp_controller.yaml \
  --output artifacts/checkpoints/mlp.pt
```

只训练不生成展示图：

```bash
python3 scripts/train_imitation.py --no-showcase
```

loss 计算方法见 [docs/training_losses.md](docs/training_losses.md)。

### 5. 离线验证

离线验证用于比较数据集中 expert 控制输出和当前神经网络输出：

```bash
python3 scripts/evaluate_offline.py \
  artifacts/checkpoints/baseline.pt \
  data/processed/simulated_controller_dataset.npz \
  --model-config configs/model/direct_mlp_controller.yaml
```

输出包括归一化控制误差指标，以及按场景绘制的：

- expert steering vs neural steering；
- expert signed acceleration vs neural signed acceleration。

默认图像目录：

```text
artifacts/reports/offline_validation/
```

验证集场景图会优先使用生成数据时保存的 `physical_targets`，保证 expert 曲线和网络输出都在物理单位下对齐。

### 6. 闭环验证

```bash
python3 scripts/evaluate_closed_loop.py \
  --config configs/default.yaml \
  --checkpoint artifacts/checkpoints/baseline.pt
```

默认执行两类验证：

- `straight_smoke`：直线跟踪 smoke 场景。
- `training_scenarios`：生成训练数据时同源的典型闭环场景。

默认输出目录：

```text
artifacts/reports/closed_loop/
```

常用选项：

```bash
# 只跑直线 smoke 场景
python3 scripts/evaluate_closed_loop.py \
  --checkpoint artifacts/checkpoints/baseline.pt \
  --smoke-only

# 只打印指标，不生成图
python3 scripts/evaluate_closed_loop.py \
  --checkpoint artifacts/checkpoints/baseline.pt \
  --no-plots
```

闭环图中会对比参考轨迹、实际轨迹、跟踪误差、速度、方向盘转角、有符号加速度，以及生成数据 expert 控制和神经网络控制。

### 7. 可微分闭环微调

闭环微调使用模仿学习得到的 checkpoint 作为初值，在典型参考场景上展开可微分车辆模型 rollout，通过跟踪误差、稳定性和舒适性 loss 更新神经网络参数。

```bash
python3 scripts/train_closed_loop.py \
  --config configs/default.yaml \
  --closed-loop-config configs/training/closed_loop.yaml \
  --initial-checkpoint artifacts/checkpoints/baseline.pt \
  --output artifacts/checkpoints/closed_loop.pt
```

输出：

- `artifacts/checkpoints/closed_loop.pt`
- `artifacts/reports/closed_loop_training/loss_curve.png`
- `artifacts/reports/closed_loop_training/loss_history.csv`

常用调试参数：

```bash
python3 scripts/train_closed_loop.py \
  --initial-checkpoint artifacts/checkpoints/baseline.pt \
  --epochs 3 \
  --horizon-steps 200 \
  --batch-size 2
```

当前闭环微调主要面向 MLP / Direct MLP 这类前馈控制器。GRU 导出和推理支持已存在，但闭环可微分训练暂不作为默认路径。

### 8. 模型导出

```bash
python3 scripts/export_model.py \
  artifacts/checkpoints/closed_loop.pt \
  --format both \
  --output-dir artifacts/exported_models/controller
```

导出目录包含：

- `model.pt`：TorchScript 模型。
- `model.onnx`：ONNX 模型，使用 `--format onnx` 或 `--format both` 时生成。
- `metadata.json`：输入输出契约、模型类型、尺度、导出格式。
- `normalization.json`：部署侧归一化参数。

默认 ONNX opset 为 18，可通过 `--onnx-opset` 修改。

仅导出 TorchScript：

```bash
python3 scripts/export_model.py artifacts/checkpoints/baseline.pt
```

### 9. 部署包验证

导出后可调用部署验证 API 检查模型包：

```bash
PYTHONPATH=src python3 - <<'PY'
from vehicle_controller.deployment import validate_deployment_package

result = validate_deployment_package("artifacts/exported_models/controller")
print(result)
PY
```

验证内容：

- `metadata.json` 与 `normalization.json` 是否存在且字段完整。
- feature 名称、数量、输入 shape、输出名称是否与工程契约一致。
- TorchScript 模型能否加载和前向推理。
- ONNX 模型存在时，能否加载并与 TorchScript 输出对齐。
- 输出是否为 `(batch, 2)`，数值是否有限，归一化控制量是否在合理范围内。

### 10. 运行时回放与性能测试

简单控制 pipeline 回放：

```bash
python3 scripts/replay_controller.py
```

神经网络原始推理耗时测试：

```bash
python3 scripts/benchmark_runtime.py
```

## 模块说明

### `src/vehicle_controller/constants.py`

定义固定输入维度、特征名称和常量。任何数据、模型、部署代码都应以这里的 `FEATURE_NAMES` / `FEATURE_COUNT` 为准。

### `src/vehicle_controller/types.py`

定义车辆状态、位姿、轨迹点、参考轨迹和控制命令等核心 dataclass。

### `src/vehicle_controller/geometry/`

提供坐标转换、参考轨迹采样、曲率计算。它负责把全局参考路径转换为车辆坐标系下的预瞄点和曲率特征。

### `src/vehicle_controller/features/`

负责构造神经网络输入：

- `error_calculator.py`：横向、航向、纵向、速度误差。
- `feature_builder.py`：组装固定顺序的 22 维输入。
- `normalizer.py`：均值方差归一化和裁剪。
- `validator.py`：部署前输入检查。

### `src/vehicle_controller/data/`

负责数据来源和数据集对象：

- `simulation_generator.py`：基于专家控制器和车辆模型生成训练样本。
- `feature_builder.py`：从原始 `raw_data` 的 `reference_traj`、`pose`、`control_signal` 构造监督训练 NPZ 所需的 22 维输入和 2 维 target。
- `expert_controller.py`：生成数据时使用的专家控制器。
- `synthetic_scenarios.py`：左转、右转、启停、变道、双移线等典型参考场景。
- `dataset.py`：单步监督训练数据集。
- `sequence_dataset.py`：时序模型数据集。
- `augmentation.py`、`sampler.py`、`split.py`：数据增强、采样和划分。

### `src/vehicle_controller/models/`

提供可训练控制器：

- `mlp_controller.py`：多分支 MLP，分别编码轨迹、误差和状态。
- `direct_mlp_controller.py`：无显式输入分支的 Direct MLP，当前默认基线。
- `gru_controller.py`：序列输入控制器。
- `model_factory.py`：根据 YAML 配置构建模型。
- `heads.py`：控制输出头。

### `src/vehicle_controller/training/`

训练和评估组件：

- `losses.py`：监督训练控制 loss。
- `trainer.py`：训练 epoch、梯度裁剪、batch loss 记录。
- `checkpoint.py`：模型保存和加载。
- `evaluator.py`、`metrics.py`：预测和误差指标。
- `offline_plots.py`：离线 expert / neural 控制对比图。
- `supervised_validation.py`：将监督 NPZ 的 22 维输入送入 checkpoint，和 2 维 target 做输出对比验证。
- `loss_plots.py`：loss 曲线和历史 CSV。
- `closed_loop_trainer.py`：可微分闭环 rollout、闭环 loss 和参考 batch 构造。

### `src/vehicle_controller/vehicle/`

车辆和执行器模型：

- `dynamics.py`：运动学自行车模型。
- `actuator_model.py`：执行器响应模型。
- `parameter_loader.py`：车辆参数和执行器限制的 YAML 加载。

### `src/vehicle_controller/control/`

在线控制 pipeline：

- `neural_policy.py`：封装神经网络推理和反归一化。
- `command_limiter.py`：方向盘和加速度命令限幅。
- `longitudinal_allocator.py`：有符号加速度到驱动扭矩 / 制动减速度的分配。
- `safety_supervisor.py`：输入、状态和命令安全检查。
- `fallback_controller.py`：异常或不安全时的降级控制。
- `controller_pipeline.py`：串联特征构建、网络、限幅、分配和安全监督。

### `src/vehicle_controller/simulation/`

闭环仿真与展示：

- `scenario.py`：仿真场景定义。
- `simulator.py`：闭环仿真器。
- `rollout.py`：按参考 profile rollout 并汇总指标。
- `showcase.py`：训练展示和闭环验证绘图。
- `disturbances.py`：扰动模型。

### `src/vehicle_controller/deployment/`

部署侧组件：

- `model_runtime.py`：运行时抽象接口。
- `torch_runtime.py`：TorchScript 后端。
- `onnx_runtime.py`：ONNXRuntime 后端。
- `realtime_controller.py`：部署实时控制封装。
- `health_monitor.py`：运行健康状态监控。
- `validation.py`：导出模型包验证。

### `src/vehicle_controller/adapters/`

中间件适配层：

- `base.py`：适配器基础接口。
- `replay_adapter.py`：离线回放适配。
- `ros2_adapter.py`：ROS2 集成占位。

### `src/vehicle_controller/utils/`

通用工具，包括 YAML 配置加载、随机种子设置和计时工具。

## 产物目录

常见运行产物：

```text
data/processed/
  simulated_controller_dataset.npz
  clean_ad_policy_sim_v1_aba9e399_imitation_dataset.npz

data/interim/
  clean_ad_policy_sim_v1_aba9e399_raw_data.pkl

artifacts/checkpoints/
  baseline.pt
  raw_data_imitation.pt
  closed_loop.pt

artifacts/reports/
  imitation_training/
  raw_data_imitation/
  training_showcase/
  offline_validation/
  supervised_validation/
  reference_trajectories/
  closed_loop/
  closed_loop_training/

artifacts/exported_models/controller/
  model.pt
  model.onnx
  metadata.json
  normalization.json
```

`data/`、`artifacts/`、`runs/` 通常是本地运行产物目录，不建议把大文件提交到 Git。

## 常见问题

### 为什么图里不再绘制 `speed-plan feedforward acceleration`？

训练和验证关心的是生成数据时 expert 控制器输出与当前神经网络控制器输出是否一致。`a_ref` 是参考速度规划中的前馈量，不等于专家纵向 PID 闭环输出。为了避免误读，控制对比图默认不再把 speed-plan feedforward acceleration 当作 expert acceleration 绘制。

### 为什么需要同时看 offline 和 closed-loop？

offline 指标只说明网络在数据集样本点上是否拟合 expert 标签；closed-loop 会把网络输出重新作用到车辆模型上，误差会随时间累积，更能暴露稳定性、限幅、安全监督和执行器响应问题。

### 导出模型时为什么优先使用 checkpoint 内部 config？

checkpoint 保存了训练时的模型结构和输出尺度。导出时默认读取 checkpoint 内部 config，可以避免训练结构和导出结构不一致。如果确实需要覆盖，可使用 `--model-config`。

### ONNX 不是必须的吗？

不是。TorchScript 是默认部署格式；ONNX 用于跨语言或非 PyTorch 运行时。使用 `--format both` 时会同时导出并验证两种格式。

# 神经网络车辆控制器实现工程目录结构与模块说明

## 1. 工程目标

本工程实现从数据准备、监督训练、闭环验证到在线控制部署的完整链路。基础控制器采用 21 维输入：

\[
\mathbf{x} =
[
x_1,y_1,\ldots,x_5,y_5,
\kappa,
e_{lat},e_v,e_s,
a_{ref},v_{ref},s_{ref},
v_x,v_y,a_x,a_y,r
]
\]

控制器对外输出：

- 方向盘转角 `steering_wheel_angle_cmd`；
- 驱动扭矩 `drive_torque_cmd`；
- 制动减速度 `brake_decel_cmd`。

内部推荐由神经网络预测方向盘转角和有符号纵向加速度需求，再由确定性的纵向控制分配模块生成驱动与制动指令。

---

## 2. 技术选型

| 类别 | 选型 | 用途 |
|---|---|---|
| 开发语言 | Python 3.10+ | 数据、训练、仿真及参考部署实现 |
| 神经网络 | PyTorch | 模型定义和训练 |
| 数值计算 | NumPy | 几何、车辆状态和控制计算 |
| 表格数据 | Pandas | CSV/Parquet 数据处理 |
| 配置 | YAML + dataclass | 可读配置与运行时类型约束 |
| 测试 | pytest | 单元、集成和回归测试 |
| 训练日志 | TensorBoard | 损失与指标可视化 |
| 模型交换 | TorchScript / ONNX | 实时部署与跨语言集成 |
| 代码质量 | Ruff + mypy（建议） | 静态检查和格式约束 |

实时车辆接口可能使用 ROS 2、Cyber RT、CAN 或自定义中间件。核心控制逻辑不依赖具体中间件，平台适配代码放在 `adapters/` 中。

---

## 3. 规划目录结构

```text
data_driven_Proxy_Model/
├── README.md
├── pyproject.toml
├── requirements.txt
├── neural_network_vehicle_controller_design.md
│
├── configs/
│   ├── default.yaml
│   ├── data/
│   │   ├── dataset.yaml
│   │   └── normalization.yaml
│   ├── model/
│   │   ├── mlp_controller.yaml
│   │   ├── direct_mlp_controller.yaml
│   │   └── gru_controller.yaml
│   ├── training/
│   │   ├── imitation.yaml
│   │   └── closed_loop.yaml
│   ├── vehicle/
│   │   ├── vehicle_params.yaml
│   │   └── actuator_limits.yaml
│   └── deployment/
│       ├── runtime.yaml
│       └── safety_limits.yaml
│
├── docs/
│   ├── project_structure.md
│   ├── data_contract.md
│   ├── training_guide.md
│   ├── deployment_guide.md
│   ├── safety_design.md
│   └── validation_plan.md
│
├── src/
│   └── vehicle_controller/
│       ├── __init__.py
│       ├── types.py
│       ├── constants.py
│       │
│       ├── geometry/
│       │   ├── __init__.py
│       │   ├── coordinate_transform.py
│       │   ├── trajectory_sampler.py
│       │   └── curvature.py
│       │
│       ├── features/
│       │   ├── __init__.py
│       │   ├── error_calculator.py
│       │   ├── feature_builder.py
│       │   ├── normalizer.py
│       │   └── validator.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── schema.py
│       │   ├── dataset.py
│       │   ├── sequence_dataset.py
│       │   ├── augmentation.py
│       │   ├── sampler.py
│       │   └── split.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── mlp_controller.py
│       │   ├── direct_mlp_controller.py
│       │   ├── gru_controller.py
│       │   ├── heads.py
│       │   └── model_factory.py
│       │
│       ├── training/
│       │   ├── __init__.py
│       │   ├── losses.py
│       │   ├── metrics.py
│       │   ├── trainer.py
│       │   ├── evaluator.py
│       │   ├── checkpoint.py
│       │   └── closed_loop_trainer.py
│       │
│       ├── vehicle/
│       │   ├── __init__.py
│       │   ├── dynamics.py
│       │   ├── actuator_model.py
│       │   └── parameter_loader.py
│       │
│       ├── control/
│       │   ├── __init__.py
│       │   ├── neural_policy.py
│       │   ├── longitudinal_allocator.py
│       │   ├── command_limiter.py
│       │   ├── fallback_controller.py
│       │   ├── safety_supervisor.py
│       │   └── controller_pipeline.py
│       │
│       ├── simulation/
│       │   ├── __init__.py
│       │   ├── scenario.py
│       │   ├── simulator.py
│       │   ├── rollout.py
│       │   └── disturbances.py
│       │
│       ├── deployment/
│       │   ├── __init__.py
│       │   ├── model_runtime.py
│       │   ├── torch_runtime.py
│       │   ├── onnx_runtime.py
│       │   ├── realtime_controller.py
│       │   └── health_monitor.py
│       │
│       ├── adapters/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── replay_adapter.py
│       │   └── ros2_adapter.py
│       │
│       └── utils/
│           ├── __init__.py
│           ├── config.py
│           ├── logging.py
│           ├── random.py
│           └── timing.py
│
├── scripts/
│   ├── prepare_dataset.py
│   ├── compute_normalization.py
│   ├── train_imitation.py
│   ├── train_closed_loop.py
│   ├── evaluate_offline.py
│   ├── evaluate_closed_loop.py
│   ├── export_model.py
│   ├── replay_controller.py
│   └── benchmark_runtime.py
│
├── tests/
│   ├── unit/
│   │   ├── test_coordinate_transform.py
│   │   ├── test_trajectory_sampler.py
│   │   ├── test_curvature.py
│   │   ├── test_feature_builder.py
│   │   ├── test_normalizer.py
│   │   ├── test_model_shapes.py
│   │   ├── test_longitudinal_allocator.py
│   │   └── test_command_limiter.py
│   ├── integration/
│   │   ├── test_training_smoke.py
│   │   ├── test_controller_pipeline.py
│   │   ├── test_model_export.py
│   │   └── test_fallback_switch.py
│   └── regression/
│       ├── test_reference_scenarios.py
│       └── reference_metrics.yaml
│
├── tools/
│   ├── inspect_dataset.py
│   ├── plot_trajectory.py
│   ├── plot_control_response.py
│   └── compare_checkpoints.py
│
├── examples/
│   ├── minimal_inference.py
│   ├── sample_input.json
│   └── sample_scenario.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── splits/
│
├── artifacts/
│   ├── checkpoints/
│   ├── exported_models/
│   ├── normalization/
│   └── reports/
│
└── runs/
    ├── training/
    ├── evaluation/
    └── simulation/
```

`data/`、`artifacts/` 和 `runs/` 中的大文件及运行产物不应提交到 Git，仅保留目录说明文件。

---

## 4. 顶层文件

### `README.md`

工程入口文档，提供项目目标、环境安装、快速开始、训练与推理命令、文档索引。

### `pyproject.toml`

定义 Python 包、依赖、测试和代码质量工具配置。源码采用 `src` 布局，避免开发目录意外覆盖已安装包。

### `requirements.txt`

提供简单的依赖安装入口。依赖的权威版本应与 `pyproject.toml` 保持一致。

### `neural_network_vehicle_controller_design.md`

保存控制器总体方案、数学定义、训练原则、安全要求和验证方法。

---

## 5. 配置目录 `configs/`

所有可能随车型、实验或部署环境变化的参数放入 YAML，不在代码中硬编码。

### `configs/default.yaml`

统一入口配置，引用数据、模型、训练、车辆和部署子配置，并设置随机种子、设备和输出目录。

### `configs/data/`

- `dataset.yaml`：数据路径、字段映射、采样周期、5 个轨迹点预瞄时间和数据划分；
- `normalization.yaml`：21 维输入顺序、均值、标准差、裁剪范围及版本号。

### `configs/model/`

- `mlp_controller.yaml`：三分支 MLP 的层宽、激活函数、输出缩放；
- `direct_mlp_controller.yaml`：21维输入直连 MLP，隐藏层为 `128→128→64`；
- `gru_controller.yaml`：时序长度、GRU 隐藏维度和输出头配置。

MLP 是第一阶段基线，GRU 作为后续时序增强模型。

### `configs/training/`

- `imitation.yaml`：监督训练优化器、学习率、Batch size 和损失权重；
- `closed_loop.yaml`：滚动步长、车辆模型、闭环损失和扰动范围。

### `configs/vehicle/`

- `vehicle_params.yaml`：质量、轴距、轮胎、传动比、车轮半径等车辆参数；
- `actuator_limits.yaml`：转角、转角速度、扭矩、制动减速度和 jerk 限制。
 
### `configs/deployment/`

- `runtime.yaml`：模型格式、推理设备、控制周期和超时阈值；
- `safety_limits.yaml`：输入范围、稳定性边界、OOD 阈值和降级条件。

---

## 6. 核心源码 `src/vehicle_controller/`

### 6.1 公共类型

#### `types.py`

定义跨模块使用的强类型数据对象：

- `Pose2D`
- `TrajectoryPoint`
- `ReferenceTrajectory`
- `TrackingErrors`
- `VehicleState`
- `ControllerFeatures`
- `NeuralPolicyOutput`
- `VehicleCommand`
- `SafetyDecision`

这些对象是模块间的数据契约，避免使用无名称的数组传递车辆状态。

#### `constants.py`

定义稳定不变的常量，例如 21 维输入特征名称和顺序：

```python
FEATURE_NAMES = (
    "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "x5", "y5",
    "kappa", "e_lat", "e_v", "e_s", "a_ref", "v_ref", "s_ref",
    "vx", "ax", "ay", "r",
)
```

训练、导出和推理都必须引用同一个定义。

### 6.2 几何模块 `geometry/`

#### `coordinate_transform.py`

将全局轨迹点转换到车辆坐标系，统一坐标轴和角度符号。

#### `trajectory_sampler.py`

根据 `0.1s~0.5s` 预瞄时间和参考速度规划换算的弧长偏移，从上游轨迹插值得到 5 个有稳定物理语义的轨迹点。

#### `curvature.py`

计算、加权和滤波参考曲率 `kappa`。首版支持直接使用上游曲率和根据轨迹点估算曲率两种模式。

### 6.3 特征模块 `features/`

#### `error_calculator.py`

计算横向轨迹跟踪误差 `e_lat`、纵向速度跟踪误差 `e_v=v_ref-vx` 和
纵向位置误差 `e_s`，并明确误差正方向。

#### `feature_builder.py`

按照固定顺序组装 21 维输入。该模块是在线推理和训练数据预处理的共同入口。

#### `normalizer.py`

加载归一化参数，执行标准化、数值裁剪和反归一化辅助操作。

#### `validator.py`

检查 `NaN`、无穷值、信号超时、轨迹点顺序和物理范围，返回结构化失败原因。

### 6.4 数据模块 `data/`

#### `schema.py`

定义原始样本、处理后样本、标签及元数据字段，校验文件列是否完整。

#### `dataset.py`

加载单帧样本，为 MLP 监督训练返回：

```text
features[21], steering_target[1], signed_accel_target[1]
```

#### `sequence_dataset.py`

加载连续时间窗口，供平滑损失、GRU 和闭环初始化使用。

#### `augmentation.py`

实现轨迹噪声、状态噪声、延迟和左右镜像。镜像操作同步反转 `y_i`、
`kappa`、`e_lat`、`ay`、`r` 和转向标签，`e_v` 不反转。

#### `sampler.py`

按曲率、速度、横向误差和纵向控制强度分桶采样，缓解直行数据占比过高的问题。

#### `split.py`

按完整场景划分训练、验证和测试集，防止连续帧泄漏。

### 6.5 模型模块 `models/`

#### `base.py`

定义控制器模型统一接口、输入维数和输出语义。

#### `mlp_controller.py`

实现设计文档中的三分支网络：

- 轨迹分支：10 维；
- 曲率、误差与前馈参考分支：7 维；
- 车辆状态分支：5 维；
- 拼接形成 128 维融合特征；
- 共享层保持原结构：`128→64`；
- 转向输出头；
- 有符号纵向需求输出头。

#### `direct_mlp_controller.py`

实现不带输入编码分支的纯 MLP：

```text
21维归一化输入 -> 128 -> 128 -> 64 -> 2维归一化输出
```

三个隐藏层使用 `SiLU`，第一层后使用 `LayerNorm`，最终输出使用 `tanh`。

#### `gru_controller.py`

实现可选的时序增强模型，输入最近若干帧的 21 维特征。

#### `heads.py`

封装转向和纵向输出头、`tanh` 输出缩放及可选不确定性输出。

#### `model_factory.py`

根据配置创建模型，训练和部署不直接依赖具体模型类。

### 6.6 训练模块 `training/`

#### `losses.py`

实现：

- 转向和纵向 Huber 损失；
- 控制一阶、二阶平滑损失；
- 可选曲率一致性损失；
- 闭环跟踪、稳定性和舒适性损失。

#### `metrics.py`

计算控制 MAE/RMSE、方向错误率、饱和比例、跟踪误差和 jerk 等指标。

#### `trainer.py`

监督训练主循环，包括自动混合精度、梯度裁剪、验证和早停。

#### `evaluator.py`

离线评估并按速度、曲率和场景分桶输出结果。

#### `checkpoint.py`

保存模型权重、优化器、配置、归一化版本、特征版本和训练元数据。

#### `closed_loop_trainer.py`

在可微车辆模型或仿真环境中进行多步滚动训练。

### 6.7 车辆模块 `vehicle/`

#### `dynamics.py`

实现基础运动学自行车模型，并预留动力学自行车模型接口，用于闭环仿真和训练。

#### `actuator_model.py`

模拟方向盘、驱动和制动执行器的限幅、迟滞、时延及一阶响应。

#### `parameter_loader.py`

加载车辆和执行器参数，执行单位及范围检查。

### 6.8 控制模块 `control/`

#### `neural_policy.py`

封装归一化、模型推理和输出反缩放，得到方向盘角及有符号纵向加速度需求。

#### `longitudinal_allocator.py`

将有符号纵向需求分配为驱动扭矩或制动减速度，实现死区、滞环和驱动制动互斥。

#### `command_limiter.py`

执行方向盘、扭矩和制动的幅值及变化率限制。

#### `fallback_controller.py`

实现基础备份控制器。首版可采用横向 Pure Pursuit 或 Stanley，纵向采用 PID。

#### `safety_supervisor.py`

结合输入有效性、输出范围、稳定性指标和模型健康状态，决定放行、限制或降级。

#### `controller_pipeline.py`

串联整个控制周期：

```text
轨迹预处理
  -> 误差计算
  -> 21维特征构建
  -> 输入校验与归一化
  -> 神经网络推理
  -> 纵向控制分配
  -> 指令限制
  -> 安全监督
  -> 最终车辆指令
```

### 6.9 仿真模块 `simulation/`

#### `scenario.py`

定义道路轨迹、参考速度、车辆初始状态、附着和扰动。

#### `simulator.py`

组合车辆动力学、执行器模型和控制器，推进离散仿真。

#### `rollout.py`

执行单场景或批量闭环运行，记录状态、误差和控制指令。

#### `disturbances.py`

提供传感器噪声、执行器延迟、侧风、质量变化和附着变化。

### 6.10 部署模块 `deployment/`

#### `model_runtime.py`

定义与模型后端无关的推理接口。

#### `torch_runtime.py`

加载 PyTorch 或 TorchScript 模型。

#### `onnx_runtime.py`

加载 ONNX 模型，并检查输入输出名称和维度。

#### `realtime_controller.py`

实现实时控制调用、上一周期指令维护、超时处理和运行统计。

#### `health_monitor.py`

监控模型推理时间、连续饱和、输入异常、输出异常和降级状态。

### 6.11 平台适配 `adapters/`

#### `base.py`

定义车辆状态输入、参考轨迹输入和车辆指令输出接口。

#### `replay_adapter.py`

从日志文件回放数据，用于不连接实车的端到端验证。

#### `ros2_adapter.py`

预留 ROS 2 消息适配。若最终平台不是 ROS 2，可替换为对应中间件适配器。

### 6.12 工具模块 `utils/`

提供配置加载、日志、随机种子控制和耗时统计，不包含具体控制业务逻辑。

---

## 7. 命令脚本 `scripts/`

脚本只负责解析参数并调用 `src/` 中的实现，不在脚本内堆积业务逻辑。

| 脚本 | 作用 |
|---|---|
| `prepare_dataset.py` | 原始日志转标准训练数据 |
| `compute_normalization.py` | 仅用训练集统计 21 维均值和标准差 |
| `train_imitation.py` | 运行监督模仿训练 |
| `train_closed_loop.py` | 运行闭环滚动训练 |
| `evaluate_offline.py` | 输出离线拟合指标和分桶结果 |
| `evaluate_closed_loop.py` | 批量运行闭环场景 |
| `export_model.py` | 导出 TorchScript 或 ONNX 模型及元数据 |
| `replay_controller.py` | 日志回放端到端控制链路 |
| `benchmark_runtime.py` | 测试平均、P99 和最大推理时间 |

预期命令形式：

```bash
python scripts/train_imitation.py --config configs/default.yaml
python scripts/evaluate_closed_loop.py --config configs/default.yaml \
  --checkpoint artifacts/checkpoints/baseline.pt
python scripts/export_model.py --checkpoint artifacts/checkpoints/beseline.pt \
  --format onnx
```

---

## 8. 测试目录 `tests/`

### `tests/unit/`

验证单模块确定性行为，重点覆盖：

- 坐标转换符号；
- 基于预瞄时间的轨迹点插值；
- 曲率左右转符号；
- 21 维特征顺序；
- 归一化与裁剪；
- 网络输入输出维度；
- 驱动制动互斥；
- 指令幅值和变化率限制。

### `tests/integration/`

验证模块组合：

- 小数据集能否完成一次训练；
- 控制流水线能否从参考轨迹产生有效指令；
- PyTorch 与 ONNX 输出是否一致；
- 输入异常时是否切换备份控制器。

### `tests/regression/`

保存固定场景和可接受指标边界，防止代码修改造成闭环性能退化。

---

## 9. 数据与产物目录

### `data/`

- `raw/`：未经修改的原始日志；
- `interim/`：清洗、同步后的中间数据；
- `processed/`：可直接训练的数据；
- `splits/`：按场景划分的训练、验证和测试清单。

原始数据只能追加，不应被训练脚本覆盖。

### `artifacts/`

- `checkpoints/`：训练检查点；
- `exported_models/`：TorchScript、ONNX 及模型元数据；
- `normalization/`：归一化参数快照；
- `reports/`：离线与闭环评估报告。

### `runs/`

保存每次训练、评估和仿真的日志。每个运行目录应包含：

- 完整配置快照；
- Git 提交号（仓库初始化后）；
- 随机种子；
- 指标；
- 日志；
- 模型或报告引用。

---

## 10. 关键数据契约

### 10.1 网络输入

网络输入形状：

```text
MLP: [batch_size, 21]
GRU: [batch_size, sequence_length, 21]
```

固定输入顺序：

```text
x1, y1, x2, y2, x3, y3, x4, y4, x5, y5,
kappa, e_lat, e_v, e_s, a_ref, v_ref, s_ref, vx, ax, ay, r
```

### 10.2 网络原始输出

```text
steering_normalized: [-1, 1]
signed_accel_normalized: [-1, 1]
```

经过输出缩放后：

```text
steering_des: deg
signed_accel_des: m/s^2
```

### 10.3 最终车辆指令

```text
steering_wheel_angle_cmd: rad
drive_torque_cmd: N*m, >= 0
brake_decel_cmd: m/s^2, >= 0
source: neural | limited_neural | fallback
reason: safety decision code
```

神经网络控制器的方向盘物理输出使用 deg；最终车辆指令、执行器限幅和动力学模型仍使用 rad，由控制 pipeline 做单位转换。

任何时刻必须满足：

\[
T_{drive,cmd}\cdot a_{brake,cmd}=0
\]

---

## 11. 模型包格式

导出的可部署模型目录建议为：

```text
exported_models/controller_v003/
├── model.onnx
├── metadata.json
├── normalization.json
├── vehicle_params.yaml
├── actuator_limits.yaml
└── validation_report.json
```

`metadata.json` 至少包含：

```json
{
  "model_version": "controller_v003",
  "feature_version": "features_v004",
  "feature_count": 21,
  "feature_names": [
    "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "x5", "y5",
    "kappa", "e_lat", "e_v", "e_s", "a_ref", "v_ref", "s_ref",
    "vx", "ax", "ay", "r"
  ],
  "output_names": ["steering_normalized", "signed_accel_normalized"],
  "trajectory_sampling_version": "sampling_v001",
  "curvature_definition": "lookahead_weighted_mean",
  "control_period_s": 0.01
}
```

在线加载时必须检查模型、特征、归一化和车辆标定版本是否兼容。

---

## 12. 模块依赖方向

为防止训练代码与实时控制代码相互缠绕，依赖方向规定为：

```text
types/constants
      |
geometry + features + vehicle
      |
models
      |
control
      |
simulation / deployment / adapters
```

`training/` 可以依赖 `models/`、`data/`、`vehicle/` 和 `simulation/`；核心 `control/` 不应依赖 Pandas、训练器或可视化工具。

---

## 13. 第一阶段最小可运行范围

首个可运行版本不需要一次实现全部规划文件，应优先完成：

1. 公共数据类型与 21 维特征契约；
2. 坐标转换、轨迹采样、误差计算和归一化；
3. 原三分支 MLP，并提供无输入编码的 Direct MLP 可选配置；
4. 数据集加载、监督损失和训练脚本；
5. 神经策略、纵向分配、指令限制和安全监督；
6. 运动学自行车仿真及基础备份控制器；
7. 单元测试、训练冒烟测试和闭环冒烟测试；
8. 模型导出与日志回放。

GRU、可微闭环训练、DAgger、ROS 2 和 ONNX Runtime 可在基线闭环稳定后增加。

---

## 14. 推荐实现顺序

1. 建立 `pyproject.toml`、配置加载和公共类型；
2. 实现几何与 21 维特征流水线，并完成单元测试；
3. 实现数据集、MLP、损失、训练和离线评估；
4. 实现纵向分配、限幅、安全监督和备份控制器；
5. 实现车辆模型与软件在环闭环验证；
6. 加入模型导出、部署运行时和性能基准；
7. 根据闭环结果增加数据增强、时序网络或闭环训练；
8. 最后接入目标车辆中间件和实车接口。

该顺序先锁定数据与控制接口，再训练模型，能够降低因特征顺序、符号或标定不一致导致的重复工作。

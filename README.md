# Data-Driven Proxy Model

本工程用于实现基于神经网络的车辆轨迹与速度联合控制器。控制器输入为 5 个局部参考轨迹点、参考曲率 `kappa`、跟踪误差和车辆状态，输出方向盘转角、驱动扭矩与制动减速度。

当前已提供第一阶段可运行基线，包括：

- 固定 22 维输入数据契约；
- 坐标转换、轨迹采样、曲率和误差处理；
- 原三分支 MLP、无输入编码的 `128→128→64` Direct MLP，以及可选 GRU；
- 监督训练、检查点和 TorchScript 导出；
- 纵向控制分配、执行器限幅和安全降级；
- 运动学自行车模型与软件在环仿真；
- 单元与集成测试。

## 文档索引

- [神经网络车辆控制器整体设计方案](neural_network_vehicle_controller_design.md)
- [实现工程目录结构与模块说明](docs/project_structure.md)
- [工程总体架构图](docs/diagrams/vehicle_controller_project_architecture.drawio)
- [神经网络结构图](docs/diagrams/vehicle_controller_neural_network_architecture.drawio)
- [架构图说明](docs/diagrams/README.md)

## 推荐技术栈

- Python 3.10+
- PyTorch
- NumPy、Pandas
- PyYAML
- pytest
- TensorBoard
- ONNX / ONNX Runtime（可选部署后端）

## 环境安装

```bash
python3 -m pip install -e .
```

开发与测试依赖：

```bash
python3 -m pip install -e ".[dev]"
```

## 快速验证

```bash
python3 -m pytest
python3 scripts/train_imitation.py --config configs/default.yaml --epochs 1
python3 scripts/evaluate_closed_loop.py \
  --config configs/default.yaml \
  --checkpoint artifacts/checkpoints/baseline.pt
python3 scripts/export_model.py artifacts/checkpoints/baseline.pt
```

闭环评估默认在 `artifacts/reports/closed_loop/` 生成：

- `trajectory_comparison.png`：参考轨迹与实际轨迹的全局、局部对比；
- `control_comparison.png`：网络原始需求、限幅结果和最终执行控制量；
- `tracking_stability.png`：跟踪误差、速度和稳定性曲线。

可通过 `--output-dir` 修改图片目录，或通过 `--no-plots` 禁用绘图。

仓库中的 `scripts/` 和 `examples/` 入口会自动加载 `src/`，因此无需额外设置
`PYTHONPATH`。其他 Python 程序使用本工程时，仍建议按上文安装为可编辑包。

使用无输入编码的 Direct MLP：

```bash
python3 scripts/train_imitation.py \
  --model-config configs/model/direct_mlp_controller.yaml \
  --output artifacts/checkpoints/direct_mlp.pt
```

## 核心接口

输入顺序固定为：

```text
x1, y1, x2, y2, x3, y3, x4, y4, x5, y5,
kappa, e_lat, e_v, e_s, a_ref, v_ref, s_ref, vx, vy, ax, ay, r
```

模型内部输出归一化方向盘转角和有符号纵向加速度需求。控制分配层再生成互斥的驱动扭矩与制动减速度。

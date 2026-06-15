# 工程架构图

## 独立图文件

- [工程总体架构图](vehicle_controller_project_architecture.drawio)
- [神经网络结构图](vehicle_controller_neural_network_architecture.drawio)
- [三分支 MLP 控制器结构图](mlp_controller_structure.drawio)
- [Direct MLP 控制器结构图](direct_mlp_controller_structure.drawio)
- [GRU 控制器结构图](gru_controller_structure.drawio)

所有文件均为独立的单页面 draw.io 图。其中三种控制器的独立结构图为：

1. `mlp_controller_structure.drawio`
   - 将 22 维输入切分为 10 维轨迹、7 维参考误差和 5 维车辆状态；
   - 三个分支分别编码为 64、32、32 维；
   - 拼接后经过 `128→64` 共享网络和两个有界输出头。
2. `direct_mlp_controller_structure.drawio`
   - 22 维输入不做分支切分；
   - 直接经过 `22→128→128→64→2` 前馈网络；
   - 最终通过 `Tanh` 输出两个归一化控制量。
3. `gru_controller_structure.drawio`
   - 接收形状为 `[batch, 10, 22]` 的时序输入；
   - 使用单层、64 维隐藏状态的 GRU 编码序列；
   - 取最后时间步表示，并通过两个有界输出头生成控制量。

原有综合图包括：

1. `vehicle_controller_project_architecture.drawio`
   - 离线数据准备、训练、评估和模型导出；
   - 22维在线特征构建；
   - 神经策略、纵向分配、执行器约束和安全监督；
   - 传统控制器降级；
   - 平台适配、车辆执行器、状态反馈和困难场景回灌。
2. `vehicle_controller_neural_network_architecture.drawio`
   - 10维轨迹几何分支；
   - 7维参考、曲率与误差分支；
   - 5维车辆状态分支；
   - 三分支编码网络及图中版本的 `128→128→64` 共享层；
   - 误差、归一化、前向传播、输出缩放、控制分配和训练损失公式；
   - 方向盘转角与有符号纵向加速度的物理量缩放；
   - 驱动扭矩和制动减速度的确定性分配。

## 打开方式

在 [diagrams.net](https://app.diagrams.net/) 中选择：

```text
File -> Open From -> Device
```

然后打开所需的 `.drawio` 文件。文件采用未压缩 XML，便于版本管理和人工审查。

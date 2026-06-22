# 训练 Loss 计算方法说明

本文档根据 `src/vehicle_controller/training/losses.py` 梳理当前控制器训练中已实现的 loss 计算方法，并说明它们在模仿学习和闭环训练中的使用位置。

## 1. 输出与标签约定

控制器输出统一使用最后一维长度为 2 的张量：

```text
[..., 0] = steering
[..., 1] = signed longitudinal acceleration
```

其中：

- `steering`：方向盘转角通道；
- `signed longitudinal acceleration`：带符号纵向加速度通道，正值表示加速，负值表示制动。

监督模仿学习中，模型直接输出归一化控制量，训练标签也来自数据集中的归一化 `targets`。实际部署时再通过 `NeuralPolicy` 乘以 `steering_limit_deg` 和 `accel_limit_mps2` 还原为物理单位。车辆执行器和动力学内部仍使用 rad，pipeline 会在写入车辆命令前完成单位转换。

## 2. 输入合法性检查

`losses.py` 中的 loss 在计算前会进行形状和参数检查：

- `_validate_control_pair(prediction, target)`：
  - 要求 `prediction` 与 `target` 形状相同；
  - 最后一维必须为 2；
  - 张量不能为空。
- `_validate_control_sequence(outputs)`：
  - 要求 `outputs` 形状为 `[batch, time, 2]`；
  - batch 维不能为空。
- `_validate_matching_non_empty(name, *values)`：
  - 要求多个误差或状态张量形状一致；
  - 张量不能为空。
- `_validate_time_step(time_step_s)`：
  - `time_step_s` 必须大于 0。
- 所有权重必须非负，Huber loss 的 `delta` 必须大于 0。

这些检查用于提前暴露训练数据、闭环 rollout 或配置错误。

## 3. 模仿学习监督 Loss

当前 `scripts/train_imitation.py` 实际使用的是 `ControllerLoss`。

### 3.1 单通道 Huber Loss

方向盘通道：

```python
steering_huber_loss(prediction, target, delta=1.0)
```

计算：

```text
L_steer = Huber(prediction[..., 0], target[..., 0])
```

纵向加速度通道：

```python
longitudinal_huber_loss(prediction, target, delta=1.0)
```

计算：

```text
L_accel = Huber(prediction[..., 1], target[..., 1])
```

PyTorch `F.huber_loss` 默认使用 mean reduction。Huber loss 对小误差近似二次惩罚，对大误差近似线性惩罚，因此比纯 MSE 更不容易被异常标签主导。

Huber loss 的单点形式为：

```text
Huber(e) =
  0.5 * e^2                         , |e| <= delta
  delta * (|e| - 0.5 * delta)        , |e| > delta
```

其中 `e = prediction - target`。

### 3.2 ControllerLoss

```python
ControllerLoss(
    steering_weight=1.0,
    acceleration_weight=1.0,
    steering_delta=1.0,
    acceleration_delta=1.0,
)
```

总 loss：

```text
L_controller =
    steering_weight     * L_steer
  + acceleration_weight * L_accel
```

默认配置下：

```text
L_controller = L_steer + L_accel
```

当前训练配置位于 `configs/training/imitation.yaml`：

```yaml
steering_loss_weight: 1.0
acceleration_loss_weight: 1.0
```

训练脚本中的调用路径：

```text
scripts/train_imitation.py
  -> Trainer.train_epoch(...)
  -> ControllerLoss.forward(prediction, target)
```

需要注意：`configs/training/imitation.yaml` 中存在 `smoothness_loss_weight`，但当前 `scripts/train_imitation.py` 并未将它接入 `ControllerLoss`。因此当前模仿学习训练的优化目标只包含方向盘与纵向加速度两个 Huber 监督项。

## 4. 时序平滑 Loss

时序平滑 loss 用于对控制序列的变化率或二阶变化进行惩罚，输入要求为：

```text
outputs: [batch, time, 2]
```

### 4.1 差分计算

内部函数：

```python
_control_difference(outputs, order, time_step_s)
```

一阶差分：

```text
d1[t] = (outputs[t + 1] - outputs[t]) / dt
```

二阶差分：

```text
d2[t] = (outputs[t + 2] - 2 * outputs[t + 1] + outputs[t]) / dt^2
```

如果序列长度不足以计算对应阶数差分，函数返回一个可反向传播的 0 loss，避免短序列训练崩溃。

### 4.2 一阶平滑项

```python
first_order_smoothness_loss(
    outputs,
    time_step_s=1.0,
    steering_weight=1.0,
    longitudinal_weight=1.0,
)
```

计算：

```text
L_first =
    steering_weight     * mean(d1[..., 0]^2)
  + longitudinal_weight * mean(d1[..., 1]^2)
```

物理含义：

- 方向盘通道：惩罚 steering rate；
- 纵向加速度通道：惩罚 acceleration rate，也就是 jerk。

### 4.3 二阶平滑项

```python
second_order_smoothness_loss(
    outputs,
    time_step_s=1.0,
    steering_weight=1.0,
    longitudinal_weight=1.0,
)
```

计算：

```text
L_second =
    steering_weight     * mean(d2[..., 0]^2)
  + longitudinal_weight * mean(d2[..., 1]^2)
```

物理含义：

- 方向盘通道：惩罚 steering acceleration；
- 纵向加速度通道：惩罚 jerk 的变化率，也可理解为 snap。

### 4.4 temporal_smoothness_loss

```python
temporal_smoothness_loss(outputs)
```

这是向后兼容接口，等价于：

```python
first_order_smoothness_loss(outputs)
```

即使用默认 `time_step_s=1.0` 和默认权重。

## 5. 闭环训练 Loss

`losses.py` 还实现了闭环 rollout 训练相关 loss。`scripts/train_closed_loop.py` 会加载模仿学习 checkpoint 作为初始网络参数，并通过 torch 版本的运动学自行车模型进行可微闭环 rollout 微调。以下 loss 不属于默认模仿学习目标，但会参与闭环微调。

### 5.1 跟踪误差 Loss

```python
closed_loop_tracking_loss(
    lateral_error,
    speed_error,
    longitudinal_error,
    lateral_weight=10.0,
    speed_weight=2.0,
    longitudinal_weight=1.0,
)
```

计算：

```text
L_tracking =
    lateral_weight      * mean(lateral_error^2)
  + speed_weight        * mean(speed_error^2)
  + longitudinal_weight * mean(longitudinal_error^2)
```

默认权重强调横向误差：

```text
lateral : speed : longitudinal = 10 : 2 : 1
```

### 5.2 稳定性 Loss

```python
closed_loop_stability_loss(
    yaw_rate,
    lateral_acceleration,
    yaw_rate_weight=0.2,
    lateral_acceleration_weight=0.2,
)
```

计算：

```text
L_stability =
    yaw_rate_weight             * mean(yaw_rate^2)
  + lateral_acceleration_weight * mean(lateral_acceleration^2)
```

物理含义：

- 限制过大的横摆角速度；
- 限制过大的横向加速度。

### 5.3 舒适性 Loss

```python
closed_loop_comfort_loss(
    outputs,
    time_step_s,
    steering_rate_weight=0.1,
    longitudinal_jerk_weight=0.1,
    steering_acceleration_weight=0.01,
    longitudinal_snap_weight=0.01,
)
```

计算：

```text
L_comfort = L_first + L_second
```

其中：

```text
L_first =
    steering_rate_weight      * mean(d1_steering^2)
  + longitudinal_jerk_weight  * mean(d1_accel^2)

L_second =
    steering_acceleration_weight * mean(d2_steering^2)
  + longitudinal_snap_weight     * mean(d2_accel^2)
```

一阶项主要约束控制变化率，二阶项主要约束控制变化率的变化。

### 5.4 ClosedLoopLoss

```python
ClosedLoopLoss(...)
```

组合目标：

```text
L_closed_loop = L_tracking + L_stability + L_comfort
```

默认权重：

| 项目 | 默认权重 |
|---|---:|
| lateral_error_weight | 10.0 |
| speed_error_weight | 2.0 |
| longitudinal_error_weight | 1.0 |
| yaw_rate_weight | 0.2 |
| lateral_acceleration_weight | 0.2 |
| steering_rate_weight | 0.1 |
| longitudinal_jerk_weight | 0.1 |
| steering_acceleration_weight | 0.01 |
| longitudinal_snap_weight | 0.01 |

`ClosedLoopLoss.forward(...)` 需要传入闭环 rollout 中记录的：

```text
lateral_error
speed_error
longitudinal_error
yaw_rate
lateral_acceleration
outputs
time_step_s
```

其中 `outputs` 是 `[batch, time, 2]` 的控制序列。

## 6. 训练过程中的实际计算链路

当前模仿学习训练每个 batch 的过程为：

```text
features, targets
  -> model(features)
  -> prediction
  -> ControllerLoss(prediction, targets)
  -> backward()
  -> gradient clipping
  -> optimizer.step()
```

对应 `Trainer.train_epoch(...)`：

1. 将 batch 的 `features` 和 `targets` 移动到训练设备；
2. 前向计算 `prediction = model(features)`；
3. 计算 `loss = ControllerLoss(prediction, targets)`；
4. 反向传播；
5. 使用 `clip_grad_norm_` 做梯度裁剪；
6. 更新参数；
7. 累积 batch loss，最终返回 epoch 平均 loss。

epoch 平均 loss 的计算方式是按样本数加权：

```text
epoch_loss = sum(batch_loss * batch_size) / sum(batch_size)
```

这避免了最后一个较小 batch 对 epoch loss 的权重过大。

## 7. 数值示例

以测试中的例子为参考：

```python
prediction = torch.tensor([[2.0, 0.5], [0.0, -2.0]])
target = torch.zeros_like(prediction)
```

在默认 `delta=1.0` 时：

```text
L_steer = 0.75
L_accel = 0.8125
```

若：

```python
ControllerLoss(steering_weight=2.0, acceleration_weight=3.0)
```

则：

```text
L_controller = 2.0 * 0.75 + 3.0 * 0.8125 = 3.9375
```

平滑项示例：

```python
outputs = torch.tensor([[[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]]])
```

当 `time_step_s=0.5` 时，一阶差分为：

```text
steering rate = 2
longitudinal jerk = 4
```

因此默认权重下：

```text
L_first = mean(2^2) + mean(4^2) = 20
```

## 8. 使用建议

- 当前模仿学习阶段优先关注 `ControllerLoss`，它直接监督网络拟合专家控制器输出。
- 如果要让控制输出更平滑，需要显式把 `first_order_smoothness_loss` 或 `second_order_smoothness_loss` 接入训练脚本；当前 `smoothness_loss_weight` 尚未生效。
- 闭环微调可通过 `scripts/train_closed_loop.py` 启动，它会使用 `ClosedLoopLoss`、可微 rollout、闭环误差、车辆稳定性指标和控制序列继续优化网络参数。
- 修改 loss 权重后，应同时查看：
  - 离线验证控制输出对比图；
  - `artifacts/reports/imitation_training/loss_curve.png`；
  - training showcase 中的轨迹、控制和稳定性图。

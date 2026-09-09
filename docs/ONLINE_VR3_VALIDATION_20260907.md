# VR-3 在线控制实测（2026-09-07）

结论：三点在线参考已进入真实 G1 Transformer/PhysX 控制链，基本生命周期验收通过。
**这是单站立种子、5 mm 目标扰动的功能验收，不是完整 BFM、搬运能力或通用任务成功率。**

## 环境与输入

- 沿用 Python 3.12.3 / 本机 IsaacLab 3.0–Sim 6，不安装/升级环境。
- RTX 5080；一个 G1，VR-3；`env.seed=42 agent.seed=42`。播放器没有 `--seed` 参数。
- 官方 `humanoid_transformer_m/model_22200.pt`，SHA-256：
  `88d5a79946c03ed25503f48b2af71d16290844ef066ca9b6c8fa8dc3837422e3`。
- 训练集站立种子 `ACCAD/Male2General_c3d/A1-_Stand_stageii`，190 帧 / 3.8 秒；NPZ SHA-256：
  `10e4d9fdb1387efd729149e598db16ce619aea8f85b4ea03f0feb02b0714d446`。
- 本次独立索引：`logs/online_control/20260907_gui_vr3/seed.yaml`；不改已有训练/验证索引。
- 无箱子场景。没有测试物体接触、抓取、导航或其他控制掩码。

## 已完成运行

| 运行 | 实际证据 | 退出 |
| --- | --- | --- |
| `normal-c.json` | 2108 个 PhysX 积分事件，10.5399998 秒；527 个控制步；两轮 Pause ≥2 秒真实状态冻结，Resume 后 ≥1 秒等待新 Apply，随后恢复 | 0 |
| `stale-a.json` | 先正常推进物理；取消真实 Kit 心跳订阅后约 0.511 秒进入 `PAUSED_FAULT`；≥2 秒冻结；在 pause 的 Kit pump 内再次原生 Play 仍冻结；心跳停止时原生 X 退出 | 0 |
| `normal-e.json`（最终 UI） | 再次完成 10.54 秒 / 527 步；在物理步轨迹中观察到 110 个不同接受序号；111 次目标提交前后观测检查全部确认 policy_task 变化；两轮越界 Apply 被拒绝且保持物理冻结，随后合法 Apply 恢复；内部窗口截图通过人工检查 | 0 |
| `setup-a.json`（最终 UI） | 真实环境及原生窗口创建后注入 marker 初始化异常；原始异常保留，面板及环境 close 均尝试并成功一次 | 1（预期失败） |

报告均位于 `logs/online_control/20260907_gui_vr3/`，包括参数、五个生产源文件指纹、
分阶段时间、参考/目标/实际轨迹，以及 articulation 14-body/root/joint 遥测。

上述正常/失联运行报告都确认：post-attach 的 env/scene/command/event reset、robot root/joint state-write
计数全部为零；观测组的 key/shape/dtype/device 保持原策略布局。没有动作/状态替身，
没有循环重置、禁用健康保护或保存新权重。

`normal-c` 最后接受序号为 111、拒绝计数为 0。序号不是独立接受数量：Pause/Resume 会丢弃
尚未消费的目标；最终 `normal-e` 另计不同接受序号，避免将序号上限误写成接受数量。

`stale-a` 故障时冻结在 1.17999997 秒物理时间，59 个策略求值 / env 步。
再次原生 Play 后这些计数均未增长，真实状态亦保持不变。

## 跟踪误差（有限范围）

下面来自 `normal-c` 每控制步记录的实际三点相对**限速 reference**误差（包含两次
Resume 参考重建附近的样本）。不是全语料、离线配对评估或全身平均指标。

| 点 | 平均位置误差 | 最大位置误差 |
| --- | --- | --- |
| 骨盆 | 3.32 cm | 3.70 cm |
| 左腕 | 1.24 cm | 2.04 cm |
| 右腕 | 4.07 cm | 4.41 cm |

目标仅沿 X 做 ±0.005 m 正弦变化，姿态保持初始值。这些数字不能外推到抬手、转身、
大范围位移或操作箱子；成功维持站立也不等于所有动作都学会。

## 生命周期与已知限制

每轮使用独立 transient user service，RuntimeMaxSec=180、TimeoutStopSec=15、
KillMode=control-group、Restart=no、`--collect`。进程退出后检查所属服务、PID 与 GPU。
正常/失联/最终 UI/异常注入运行 PID 分别为 490621 / 493656 / 507175 / 509515，均已结束。

以报告的 `exit_requested_monotonic_s` 到 journal 中服务结束后的 CPU accounting 事件计算，
`normal-c` / `stale-a` / `normal-e` 关闭完成的上界分别约 1.013 / 0.877 / 0.962 秒，低于 15 秒。进程退出码来自等待执行结果，
不是已被 collect 的空服务默认 `Result=success`。

初次诊断中 `normal-a` 因误用 `--seed` 在建环境前退出，`normal-b` 因测试启动变量命名
错误在 App 前退出，均未计为能力通过；纠正后才产生上述正式结果。

强化运行 `normal-d` 的 App 启动耗时约 81 秒，随后触发原来的 120 秒总测试期限，
在 7.56 秒物理时间退出 1；早期越界检查和截图已有，但该轮整体不计通过。
后续将测试内部总期限设为 165 秒，外层保持 180 秒并留 15 秒收尾；没有改变策略心跳、
健康保护或物理时钟。最终 `normal-e` 完整通过。机器上另有 `rl100_playground` 作业，
仅识别以区分进程归属，没有停止它；这些 GUI 运行的墙钟耗时不作为性能基准。

桌面锁屏时，桌面截图只抓到了锁屏，不作为 GUI 外观验收；没有解锁或修改系统安全设置。
改用 Kit 自身 swapchain 捕获后，人工检查 `normal-d.png` 发现巨大坐标轴和控件裁切，
修正为 600×820 面板、固定按钮/输入、独立详情滚动、0.08 比例坐标轴。最终截图
`logs/online_control/20260907_gui_vr3/normal-e.png` 显示全部 XYZ/旋转输入、五个按钮完整可见，
机器人不再被目标轴遮挡。Kit 仍打印 PointInstancer/Fabric 警告；暂未修改 SDK，
不能据此宣称底层渲染警告已消除。

最终 CPU 回归：`259 passed, 62 warnings`，后者为既有警告。输入 NPZ 和官方 checkpoint
运行前后 SHA-256 一致。初始化故障测试中的 `app.close` 会快速结束进程，报告主要记录
close 尝试，应用最终退出由外层进程与服务核验，而非依赖关闭后还能写出 Python 报告。

## 使用与下一步

启动/操作说明见 [在线控制说明](ONLINE_VR3_CONTROL.md)。仍需较大目标、不同初始动作、
更长窗口和更多种子验证，再开发目标点行走与双臂夹箱任务。不根据这两轮结果扩大安全阈值，
也不把官方权重替换成本地新权重。

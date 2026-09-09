# 三种在线稀疏控制模式实测（2026-09-07）

结论：Pelvis-1、UMI-2、VR-3 已接入同一个真实 Transformer/PhysX 控制链，
原生 Kit 面板切换、暂停和新目标恢复通过本轮功能验收。
**这是单个站立种子、小幅目标的有限窗口实测，不是完整 BFM、导航或搬运验收。**

## 本轮新增实现

- `online_modes.py` 统一模式名称、三行目标与 14 个参考连杆的映射。
- `LiveMotionCommand` 支持按初始模式接入，以及运行中修改模式掩码。
  Provider 仍提供完整 33 帧未来参考；未激活部分由 actor 的输入掩码屏蔽。
- Kit 面板增加三个模式按钮，仅允许编辑/显示当前激活目标。
  目标包包含 `mode_name` / `mode_epoch`，往返切回同名模式也不能复用旧包。
- 运行中切换依次经过 Pause、模式确认、Resume、fresh Apply。
  没有重置环境、写关节/根状态、改策略输出维数或训练三套新模型。
- 本机 IsaacLab API 核对确认：只刷新 task/mode 观测，并使用
  `compute_group(..., update_history=False)` 保留身体、动作与 critic 历史。
  原生 PhysX 步事件用于计时，不用 Kit timeline 冒充实际物理积分时间。

切换中途出现程序异常时采取暂停并关闭的策略，不尝试带着部分更新继续运行。
目前不是跨 Provider/UI/观测的可回滚事务；不把程序异常伪装成普通目标拒绝。

## 输入、协议与结果

沿用既有 IsaacLab 3.0 / Sim 6、Python 3.12、RTX 5080；单 G1 29 DoF，无箱子。
官方 `humanoid_transformer_m/model_22200.pt` 与站立种子
`ACCAD/Male2General_c3d/A1-_Stand_stageii`，`env.seed=42 agent.seed=42`。

先在初始 VR-3 中实际推进 0.22 秒，再依次切换 Pelvis-1 → UMI-2 → VR-3。
每段围绕切换时的真实姿态，向激活点输入 X 方向 ±0.02 m、欧拉 yaw ±2° 的
2 秒周期正弦目标；未激活目标行保持不变。不是从同一个 reset 启动的配对性能实验。

| 模式 | 激活连杆索引 | 实际物理时间 | 控制步 / 不同接受目标数 | 目标拒绝数 |
| --- | --- | --- | --- | --- |
| Pelvis-1 | 0 | 5.02 s | 251 / 251 | 0 |
| UMI-2 | 10、13 | 5.02 s | 251 / 251 | 0 |
| VR-3 | 0、10、13 | 5.02 s | 251 / 251 | 0 |

整个会话累计 **3056 个 PhysX 积分事件、15.28 秒物理时间、764 个真实策略/环境步**。
每段最后排队的目标可能在切换时被丢弃，因此报告统计实际不同接受序号，
不把最后序号 755 写成 755 个接受目标；三段实际共接受 753 个目标。

三次切换均验证：

- 从发出切换请求到确认 Pause，物理事件计数、实际 root/body/joint 状态、
  policy/action/critic 历史增量均为零，未漏检暂停事件处理期间的隐式步进。
- 模式切换后分别保持约 0.501 / 0.511 / 0.688 秒墙钟冻结，状态和历史最大差均为 0。
- Resume 后分别等待约 0.504 / 0.506 / 0.503 秒新 Apply，物理仍保持冻结；
  新的有效 Apply 才恢复。
- 14-bit 命令掩码、观测模式与 actor 输入中的模式一致。只在克隆观测上将未激活
  字段加 123，actor 的掩码后任务输入仍完全相同；没有替换真实策略或物理。
- 全部观测 key/shape/dtype/device 保持原布局；原生输入行的 enabled 状态正确。
- post-attach 的 env/scene/articulation reset、command resample、robot root/joint
  state-write 九个计数全部为 0；未触发健康或心跳故障。

## 激活目标误差：仅采样描述

以下来自每段约每 0.5 秒记录一次的 **11 个时间样本**，在当前模式激活点上平均。
不是全 251 步的完整统计，也不是全语料指标；不同模式不能据此排性能高低。

| 模式 | 相对 reference 平均位置误差 | 采样最大位置误差 | 平均姿态差 |
| --- | --- | --- | --- |
| Pelvis-1 | 3.08 cm | 3.78 cm | 0.80° |
| UMI-2 | 1.58 cm | 3.11 cm | 3.25° |
| VR-3 | 1.34 cm | 2.28 cm | 3.31° |

reference 是限速参考，actual 是真实 articulation 遥测。此次小幅低速输入中，
保存样本的 goal 与 reference 位置相同，因此两种位置误差恰好一致；一般情况下并不相同。
截图取自切换后的暂停/参考重建阶段，图中零误差不是运行时跟踪性能的证据。
本轮没有设置更大范围任务成功率门槛，`PASS` 表示上述功能协议通过。

## 可复查产物与进程清理

- 驱动：`ScaleTrack/tests/online_modes_gui_smoke.py`。
- 本地报告：`logs/online_control/20260907_sparse_modes/modes-a.json`，包含完整 argv、
  六个生产源文件指纹、各阶段时间、模式掩码、前后状态/历史、目标接受数和采样轨迹。
- 截图：同目录 `modes-a.png`。使用 Kit 自身 swapchain，未解锁桌面或更改安全设置。
  已目视检查三个模式按钮、五个生命周期按钮和全部输入可见，机器人未被大坐标轴遮挡。
- 服务：`bfm-online-modes-20260907-a.service`，PID `536603`，退出码 **0**。
  外层 RuntimeMaxSec=180、TimeoutStopSec=15、KillMode=control-group、Restart=no、collect；
  内部总期限 165 秒。退出后服务和 PID 均不存在，无本轮残留 GPU 进程。
  从报告 Exit 时间到 journal 的服务结束后 accounting 事件约 **0.805 秒**。

输入运行前后 SHA-256 不变：

- checkpoint：`88d5a79946c03ed25503f48b2af71d16290844ef066ca9b6c8fa8dc3837422e3`
- seed NPZ：`10e4d9fdb1387efd729149e598db16ce619aea8f85b4ea03f0feb02b0714d446`
- 本次 smoke 驱动：`5c0bf2735f21423ed05d213acd2ff0329a0184311741e703d89d6ecd2090a85d`

仍有既有 Kit PointInstancer/Fabric 警告，未修改 SDK，也不宣称渲染警告已经消除。
没有重装 IsaacLab、新增训练、覆盖权重、修改原始数据、操作真机或推送远端。

最终 CPU 回归：`294 passed, 62 warnings`（既有警告），覆盖 `ScaleTrack/tests`、
`tests/test_gui_launcher.py`、`tests/test_ci_workflow.py` 和 `tests/test_mask_report_compare.py`。
新增边界包括初次 Enable 前选模式、事件处理/推理中切换、旧 epoch 拒绝、故障禁止换模式、
中途 UI 确认异常时暂停关闭。`bash -n` 和 `git diff --check` 均通过。

## 使用和下一步

启动及切换说明见 [在线稀疏控制操作指南](ONLINE_VR3_CONTROL.md)。
本轮增加了三模式切换验收；此前 VR-3 失联、原生 Play 防绕过和初始化异常实测
保留在 [原安全验收记录](ONLINE_VR3_VALIDATION_20260907.md)，不冒充本轮重新测过的结果。

下一步仍需更长时间、更多种子与不同目标范围评估，再做目标点到达/停止的闭环，
随后开发真实接触下的双臂夹箱、抬升和搬运。当前没有手指、自动导航或搬运任务成功证据。

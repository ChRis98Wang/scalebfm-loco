# 三维目标点验收：2026-09-08

后续已完成真实 actor 输入诊断和水平/高度分项测试，见
[第二轮开发记录](ONLINE_WAYPOINT_DIAGNOSTICS_20260908.md)。下方保留 A/B 原始组合结果。

结论：**XYZ + yaw 任务入口、未来轨迹和安全闭环已接入；真实目标任务两次均 FAIL，
不能宣布目标点行走或完整 ScaleBFM 已实现。** 本次沿用现有 IsaacLab 和官方
Transformer 权重，没有训练、替换 checkpoint、重装环境、推送或操作真机。

## 新增与修复

- 原生 Kit 面板包含绝对环境局部 XYZ（Z = 骨盆高度）、yaw、Go/Stop 和任务状态。
  Pelvis-1 才能 Go；必须已 Enable，不能绕过 Resume 后的新 Apply 门控。
- 纯 NumPy 三维规划器限制总速度、竖直速度、前探距离和朝向变化。
  到达条件来自真实 body-link 位置与完整三维速度，且按实际控制步累计停稳时间。
- 无副作用的终点预检查；不合格新任务不替换旧任务。
  修复新测试中把非单位四元数乘 2 却期待接收的错误，没有放宽单位约束。
- 修正初始 SETTLING 的独立高度判据、未实测速度标记和未启动状态边界。
- 显式多帧 `preview → submit_trajectory`：当前帧 + 33 帧未来，严格 anchor、
  全帧 workspace/速率、序号和时钟检查，实际 command step 才消费下一帧。
- 共享 UI/internal 序号，先消费人工意图再提交内部轨迹；手动 Apply 可以取消路径。
  更新 task/mode 观测不推进 history，不修改机器人 root/joint state。

完整使用方法和工程限制见 [三维目标点控制](ONLINE_WAYPOINT_CONTROL.md)。

最终 CPU 回归 **447 passed，62 条既有 warnings，6.19 秒**，以及 `git diff --check`
通过。覆盖规划器、Provider、controller、UI、原训练/播放相关测试及启动器/CI 配置检查。
包含显式 trajectory 的逐帧消费、输入防别名、乱序/时钟拒绝、过期 anchor 保留旧计划、
手动接管、完整 XYZ/yaw 未来窗口、无额外 history 推进等；这些不替代真实策略能力验收。

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/sw/isaaclab_ws/env_isaaclab_sim6_newton/bin/python -m pytest \
  ScaleTrack/tests tests/test_gui_launcher.py tests/test_ci_workflow.py \
  tests/test_mask_report_compare.py -q --tb=short --disable-warnings
```

## 固定真实测试协议

`ScaleTrack/tests/waypoint_gui_smoke.py` 通过真实 `play.py`、原始策略和单机器人 PhysX
运行，无物体、Pelvis-1 精确 mask `[0]`，环境与策略 seed 均 42。
先 Enable 并运行约 1 秒，再按此时真实朝向指定前方 **0.20 m**、降低骨盆 **0.06 m**，
yaw 不变。两轮起点和目标一致：

```text
actual XYZ = [0.473057330, 0.478135943, 0.786284089] m
target XYZ = [0.479475074, 0.678032948, 0.726284089] m
target yaw = 88.161136114 deg
```

固定到达条件：三维距离 ≤0.05 m、高度差 ≤0.03 m、yaw 差 ≤5°、
三维线速度 ≤0.05 m/s、角速度模长 ≤0.10 rad/s，连续满足 ≥0.5 秒实际物理时间。
ARRIVED 后仍要继续策略/物理 2 秒并验收保持，再检查 Pause 的物理/状态/history 冻结和退出。
没有将速度降低、暂停物理或参考抵达当作真实到达。

规划默认值没有为通过测试调整：0.08 m/s 总速度、0.04 m/s 竖直速度、0.10 m 前探、
0.25 rad/s yaw 速度、15 秒超时，4 秒内未缩短距离至少 0.01 m 触发 STALLED。

## 两轮结果

| 指标 | A：单个下一步目标 | B：显式未来轨迹 |
| --- | --- | --- |
| 结果 | FAIL / STALLED | FAIL / STALLED |
| 任务实际运行时间 | 6.36 s | 6.22 s |
| 总 PhysX 时间（含初始站立） | 7.38 s | 7.24 s |
| 实际 policy/env step | 369 / 369 | 362 / 362 |
| PhysX 事件数 | 1476 | 1448 |
| 最终三维距离 | 0.153362 m | 0.153032 m |
| 最终水平距离 | 0.149715 m | 0.149440 m |
| 最终高度差 | 0.033246 m | 0.032962 m |
| 最终三维线速度 | 0.013392 m/s | 0.013417 m/s |
| 最终角速度模长 | 0.009511 rad/s | 0.009879 rad/s |
| 连续合格停稳时间 | 0 s | 0 s |

两轮均只前移约 5 cm，未触发跌倒/非有限状态保护，但因距离不再有效缩短而暂停。
**两轮都没有进入 ARRIVED，故后续 2 秒保持与成功路径 Pause/X 验收未执行。**
不能把旧在线模式安全测试的通过结果当成本次完整 waypoint 成功路径证据。

A 暴露出参考窗口缺陷：规划每步约 0.0016 m，普通 Provider 单帧可前进 0.004 m，
所以未来第 1 帧即变成停止姿态。B 改为显式逐帧轨迹，实际报告中 offsets 1、2、4、32
的 pelvis XYZ 确实不同，且包含 Z 变化；临近前探上限时预测尾部允许停止。
但 B 任务误差与 A 接近，**不能据此声称未来窗口修复解决了迈步问题**。
现有证据只表明当前权重与这组保守在线参考的组合未完成任务，不证明所有在线输入均失败。

B 的 310 条逐步记录中，123 条有非零未来位置跨度，最大 frame 0→32 为 0.0512 m；
约任务时间 2.46 秒后参考达到前探上限，随后未来保持是边界规则的结果。
源码复查未发现新的 anchor/cache/mask 接线错误：目标按真实骨盆系变换，
Pelvis-1 保留 body 0 的位置/姿态观测，Provider commit 递增版本且 task 观测随后刷新。
这仍是源码与 Provider 层证据，不是已捕获全部 masked actor 输入的数值一致性证明。
“当前策略对缓慢、无步态相位的骨盆合成轨迹倾斜而不换步”是待验证解释，不写成已证实根因。

## GUI 与物理安全证据

已查看 B 的 Kit 原生 swapchain 截图，G1、XYZ/yaw 编辑框、三模式、Go/Stop、Enable/
Pause/Resume/Exit 和 READY 状态可见。截图前后仍为 0 PhysX 时间/回调，再请求 Enable。
这是布局/启动证据，不是成功移动视频。

本机产物（位于被忽略的 logs，不自动公开数据/权重）：

- [A 原始失败报告](../logs/online_control/20260908_waypoint/xyz-a.json)
- [B 原始失败报告](../logs/online_control/20260908_waypoint/xyz-b.json)
- [B READY 截图](../logs/online_control/20260908_waypoint/xyz-b.ready.png)

报告保存失败时的真实状态、目标、误差和任务状态，早于 controller/env close，
避免清理阶段的 CANCELLED/CLOSED 掩盖根因。B 还逐步保存未来 pelvis XYZ 与线速度。
两轮均完整安装 11 项 reset/resample/state-write 计数器，全部为 0；B 的仪器错误列表为空。
实际观测沿用 CUDA float32，policy `[1,3,64]`、action `[1,3,29]`、
policy_task `[1,6,253]`，接受任务的边界检查未改变布局。

两轮采用独立 `bfm-waypoint-20260908-a/b.service`，内部 165 秒、外部 180 秒上限，
`KillMode=control-group`，退出状态均为 1（测试主动报 FAIL），不是 timeout 或成功。
B 主 PID 30200 已消失；报告请求退出到 journal 记录主进程退出约 0.51 秒。
最终核对无残留 `bfm-*` 服务，两轮 exec 会话均回收；未终止旁路 RL100 进程。

## 输入与可复查性

本机现有 Python：`/home/sw/isaaclab_ws/env_isaaclab_sim6_newton/bin/python`。
已有 SDK：`IsaacLab-3.0-sim6-newton`；本次按本地 `base_articulation_data.py`
核对 body-link 速度 `.torch` 展开为 `[env, body, 3]`、世界系、运行时 xyzw。
Provider 边界为环境局部坐标和 wxyz，仅做一次转换。

权重 `humanoid_transformer_m/model_22200.pt` SHA-256：
`88d5a79946c03ed25503f48b2af71d16290844ef066ca9b6c8fa8dc3837422e3`。
动作 `ACCAD/Male2General_c3d/A1-_Stand_stageii` NPZ SHA-256：
`10e4d9fdb1387efd729149e598db16ce619aea8f85b4ea03f0feb02b0714d446`。
输入哈希本轮已核对；原报告保存命令行、控制配置和 7 个生产源码 SHA-256。

## 下一步

先捕获并解码真正送入 actor 的 masked pelvis 观测，对比同一状态下静止/移动未来参考
的确定性动作响应；再核对与原动作参考的数值一致性，将“水平迈步”“单独高度调整”
拆成诊断子任务，定位策略执行范围与参考分布的差距。这些子任务即使通过也不能替换
本页的组合任务门槛。若需适配训练，使用独立实验与原八模式回归，不覆盖现有基线。
在任务完成、停稳和长期保持有实证前，不宣布导航、双臂夹箱或搬运已经实现。

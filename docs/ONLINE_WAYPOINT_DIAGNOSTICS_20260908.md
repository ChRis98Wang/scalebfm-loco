# 在线 XYZ 控制诊断与改进（2026-09-08，第二轮开发）

本记录接续 [A/B 组合任务失败](ONLINE_WAYPOINT_VALIDATION_20260908.md)。
本轮改进的是输入纯函数性和可核查诊断，不把调试工具通过写成行走成功。
既有 IsaacLab、官方 `model_22200.pt`、站立种子及 29 DoF G1 不变，未训练、替换权重或推送。

后续进展：[KIT 原始行走参考四组物理验收](KIT_GAIT_VALIDATION_20260908.md) 已通过，
补齐本文末尾候选片段此前只有运动学检查的状态。它没有改变本文在线前进/组合失败的结果，
也未将超出在线速度上限的原始轨迹接入 waypoint。

## 修复：掩码不再污染 observation

`ActorCriticHumanoidTransformer.get_actor_obs` 原先对 `policy_task` 做原地乘法，
会改变环境/调用者持有的 observation。现在使用非原地计算：训练掩码输出数值不变，
但重复诊断、不同 mask 对照及 autograd 不再依赖之前调用留下的零值。
普通 inference 的全量 task/全一 mode 行为保留；固定模式播放仍通过原 instance override
走真实 Pelvis-1 掩码路径。这个问题已修复，**不是已证实的停滞根因**。

## 新增可复查策略诊断

- `waypoint_diagnostics.py`：严格验证单一 Pelvis-1 schema，clone 全部 observation，
  仅把未来 pelvis pose 特征替换为 frame 0；保留时间 offset、当前帧、mask 和历史。
- `waypoint_policy_probe.py`：截取真正进入 `actor_task_embedder` 的 267 维输入，
  包括 253 维 task 和 14 维 mode；核对固定 mask、位置/相对位置/旋转/相对旋转。
- 独立 NumPy wxyz 变换按真实骨盆坐标系重算四个 pose 特征块，包括 Z；
  不只证明 Provider 中存在未来数据，而是检查最终 actor 输入数值。
- 同一状态依次推理 moving、flat、moving-repeat；**仅首次原目标 action 返回执行**。
  另外两次结果只写报告，不能驱动机器人。重复原输入 action 必须逐元素相同。
- 每次检查 CPU/CUDA RNG、模型参数/buffer、原 observation、机器人状态/history、
  PhysX 回调/时间、reference frame/version 和 waypoint elapsed 均不变；异常立即中止。
- 探针默认关闭，仅 `BFM_WAYPOINT_DIAGNOSTICS=1` 开启。它会增加同步计算时间，
  慢设备上可能触发原有 0.5 秒心跳保护；没有调宽心跳或增加隐式 Kit/物理步。

配置用 AST 测试锁定四个 pose term 顺序、future indices `[0,1,2,3,4,-1]` 和
mode mapping `[3,3,6,6] + time`，配置变化须同步诊断 schema，不能静默误解码。

## C：真实 actor 输入与动作敏感性

输入与 A/B 完全相同：真实朝向前方 0.20 m，骨盆降高 0.06 m，yaw 不变。
原始记录：[xyz-c.json](../logs/online_control/20260908_waypoint/xyz-c.json)。

8 个真实状态上的检查均通过：mask、time offsets 和 actor 输入布局正确；
位置变换最大残差小于 `5e-8 m`，旋转 tangent/normal 特征残差小于 `3.4e-7`。
eval 模式由 runner 移除 observation term noise，本次保留 `1e-5` 数值一致性门槛。
这些残差是**输入计算的一致性**，绝不是机器人目标跟踪误差。

| 任务时间（s） | moving 与 flat 动作最大分量差 |
| --- | --- |
| 0.0 | 0 |
| 0.5 | 0.022793 |
| 1.0 | 0.025003 |
| 1.5 | 0.028542 |
| 2.0 | 0.036209 |
| 2.5 | 0.001159 |
| 3.0 / 5.0 | 0 |

单位是原策略 action 单位，不是米、角度或已经过关节 scale 的弧度。
初始参考尚未推进以及达到前探上限后，moving/flat 一致，因此差为零。
其余状态输出不同，直接表明当前策略对未来目标有响应，不能再将停滞解释为“目标没传进去”。

**组合任务仍 FAIL / STALLED**：6.22 秒任务时间，7.24 秒总 PhysX，362 次实际 policy/env
step，1448 次 PhysX 回调，最终三维距离 0.153032 m、高度差 0.032962 m。
未到 ARRIVED，后续保持/成功路径 Pause 验收没有执行。

C 的 310 条逐步真实遥测与 B **全部逐元素相同**，终点和实际步数也完全相同；
说明非原地掩码修复与诊断未扰动这一输入下的真实运行。16 次额外 policy 调用均只用于诊断。
11 项 reset/resample/root/joint write 计数全为零，8 次副作用检查全通过。
独立 user cgroup `bfm-waypoint-20260908-c.service` 退出码 1（主动报告任务失败），
主 PID 126033 已退出，退出请求到 journal 记录主进程退出约 0.51 秒，exec 已回收。

## 分项测试与后续边界

组合门槛不变。另设 `horizontal_only`（前进 0.20 m、保持目标高度）与
`height_only`（保持目标 XY、降高 0.06 m）作为归因测试，不能替代 combined 的结果。
三项共用相同的实际到达、停稳、保持、暂停和清理断言，不开放任意降低距离的测试参数。

### D：仅高度目标通过现有容差，不是精确降高

记录：[height-d.json](../logs/online_control/20260908_waypoint/height-d.json)，
已查看 [到达并保持后的原生 GUI 截图](../logs/online_control/20260908_waypoint/height-d.png)。
目标保持初始真实 XY，Z 从 0.786284 m 降至 0.726284 m，yaw 不变。

- **PASS（height_only 分项）**：任务 1.80 秒时 ARRIVED；独立实际 PhysX 停稳累计约 0.50 秒。
- 到达时三维距离 0.029080 m、高度差 0.026690 m，符合原 0.05 / 0.03 m 判据。
- 随后继续真实 policy/physics 2.02 秒、101 个 env step，没有冻结仿真代替保持。
- 保持结束三维距离 0.029416 m、高度差 **0.026837 m**，线速度 0.014800 m/s、
  角速度模长 0.020344 rad/s，均在既有判据内。
- 实际最终 Z 为 0.753121 m，相对起点只降低 **0.033163 m（3.32 cm）**。
  不能把“6 cm 降高目标通过 3 cm 误差容限”改写成“精确降低了 6 cm”。
- Pause 请求到确认的物理步、状态/history 变化全为 0；确认后冻结 0.51 秒仍全为 0，
  最后触发 native panel X 退出。
- 总 PhysX 4.84 秒、968 回调、242 次 policy/env step，11 项写状态/重置计数全零。
  `bfm-waypoint-20260908-height.service` 正常退出 0，PID 130259 已退出；
  退出请求到 journal 结束记账约 1.01 秒，exec 已回收。

这是单个站立种子、单次下降输入、宽松工程容差内的短窗口证据；未验证不同高度/朝向、
上升目标、扰动恢复或毫米/厘米级精度。不能用它替代组合目标或导航验收。

### E：保持目标高度，前进仍失败

记录：[horizontal-e.json](../logs/online_control/20260908_waypoint/horizontal-e.json)。
目标前进 0.20 m、保持初始 Z = 0.786284 m，不再要求同时下降。
**FAIL / STALLED**，任务 6.66 秒时停止，最终水平距离 0.136490 m、
三维距离 0.136680 m、高度差 0.007195 m；实际向前约 6.4 cm。
所以即便高度已在容限内，水平到达依然不成立，不能把组合失败仅归因于下降目标。

总 PhysX 7.68 秒、1536 回调、384 次 policy/env step，11 项写状态/重置计数全零。
未到 ARRIVED，未执行成功后的保持与 Pause 冻结验收。
`bfm-waypoint-20260908-horizontal.service` 退出 1，PID 131593 已退出；
退出请求到 journal 主进程退出约 0.50 秒，exec 已回收。

## 回归、复查与清理

最终完整 CPU 回归 **511 passed / 62 条既有 warnings / 6.42 秒**，`git diff --check` 通过：

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/sw/isaaclab_ws/env_isaaclab_sim6_newton/bin/python -m pytest \
  ScaleTrack/tests tests/test_gui_launcher.py tests/test_ci_workflow.py \
  tests/test_mask_report_compare.py -q --tb=short --disable-warnings
```

真实 harness 使用原 `play.py --online_targets --motion_menu --mode_index 0 --num_envs 1 --viz kit`
参数，站立种子与模型见 A/B 记录。`BFM_WAYPOINT_SCENARIO` 默认 `combined`，仅另允许
`height_only` / `horizontal_only`；`BFM_WAYPOINT_SMOKE_REPORT` 必须指向独立报告路径。
测试继续使用内部 165 秒、外部 systemd user cgroup 180 秒上限，不直接留下无限运行的 App。
`BFM_WAYPOINT_DIAGNOSTICS=1` 只增加动作/输入诊断，不改变场景目标或通过门槛。

输入权重与 NPZ SHA-256 再次核对，与 A/B 一致；报告保存生产源码、测试驱动与诊断模块
指纹。所有 C/D/E user units 和 exec 会话已回收，最终核对无残留 `bfm-*` 服务。
旁路 RL100 作业不在本轮范围，未终止或修改。没有训练、模型保存、重装 SDK 或推送。

当前证据排除了本次运行的 mask/坐标/未来输入丢失；它还没有证明是哪一种训练分布
或步态条件造成执行不足。下一阶段应根据分项结果设计参考动作载体或独立任务适配训练，
继续保留原八模式回归、原组合目标和基线权重。尚不具备已验收的自主导航、夹箱或搬运。

只读检查了三个既有 NPZ，后续可优先单独验收两个 KIT 行走候选：

- `amass_kit_batch_v1_r016_processed/8/WalkingStraightForwards07_stageii.npz`：
  6.08 秒、50 Hz，骨盆净水平位移约 2.40 m、Z 范围约 4.3 cm。
- `amass_kit_batch_v1_r003_processed/167/walking_slow04_stageii.npz`：
  6.50 秒、50 Hz，骨盆净水平位移约 2.31 m、Z 范围约 5.7 cm。

路径均相对 `ScaleRetarget/retargeted_dataset/`，两者在 KIT train 与 full-v2 train 索引中。
它们含交替足部和全身关节序列，但这里只检查了运动学，**没有验证当前 checkpoint 的物理跟踪**，
也不意味着旧 checkpoint 已学习 KIT。原动作平均速度高于当前 waypoint 0.08 m/s 上限，
不能直接注入或随意慢放来宣称符合原约束；应先单独做原始行走参考的物理验收，再设计
与目标点/停止条件兼容的轨迹或适配实验。此次未改动任何动作数据和训练索引。

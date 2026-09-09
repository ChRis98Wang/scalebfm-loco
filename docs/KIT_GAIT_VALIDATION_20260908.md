# KIT 原始行走参考物理验收（2026-09-08）

结论：官方 Transformer 权重在两个本地 KIT 行走片段上，**Pelvis-1 和 WholeBody-14
共四组短片段行走指标通过**。仅给骨盆参考也能产生实际前进和双脚前后交替。
这补齐了上一轮只有 NPZ 运动学检查的证据，**没有修复或通过在线 XYZ waypoint**。
没有新训练、权重替换、数据重定向或训练索引改写，也不是完整 BFM 复现。

## 实测结果

单环境、seed 42、50 Hz 控制 / 200 Hz PhysX、29 DoF G1、同一官方
`humanoid_transformer_m/model_22200.pt`。真实机器人状态来自 articulation，
不是参考轨迹位置、GUI 标记或累计命令位移。

| KIT 片段 | 模式 | 实际前进（m） | 参考前进（m） | 骨盆三维误差均值 / 最大值（cm） | 双脚前后交换次数 | 结果 |
| --- | --- | --- | --- | --- | --- | --- |
| `8/WalkingStraightForwards07_stageii` | Pelvis-1 | 2.4153 | 2.3987 | 2.615 / 7.556 | 4 | PASS |
| `167/walking_slow04_stageii` | Pelvis-1 | 2.3361 | 2.3096 | 4.386 / 8.124 | 4 | PASS |
| `8/WalkingStraightForwards07_stageii` | WholeBody-14 | 2.3770 | 2.3987 | 4.085 / 9.945 | 3 | PASS |
| `167/walking_slow04_stageii` | WholeBody-14 | 2.2740 | 2.3096 | 4.476 / 9.468 | 4 | PASS |

“前进”是实际骨盆净 XY 位移在参考净 XY 方向上的有符号投影，不是路径长度。
JSON 中 `actual_path_m` / `reference_path_m` 是 **三维**路径长度。
两种模式均用相同骨盆/足部指标；WholeBody-14 表示激活 14 个参考点，
**此表没有统计或验收全部 14 点的逐体误差**，也不能据此判断哪种模式普遍更优。

四组最小骨盆高度均大于 0.722 m，最小骨盆竖直方向点积均大于 0.967。
没有触发预设高度/倾斜保护。足部交换是前后相对位置的符号变化，
不是完整步态周期数、触地次数、接触稳定性或无滑步证明。

原始本地报告（`logs/` 不随源码发布）：

- [Pelvis-1 两片段](../logs/gait_control/20260908/pelvis-b.json)
- [WholeBody-14 两片段](../logs/gait_control/20260908/wholebody-a.json)
- [隔离的两片段索引](../logs/gait_control/20260908/kit_walking.yaml)

## 验收协议与实现

新增 [真实物理驱动](../ScaleTrack/tests/gait_reference_smoke.py)、
[纯 NumPy 指标](../ScaleTrack/scripts/pretrain/rsl_rl/gait_metrics.py) 和
[禁止中途写状态的守卫](../ScaleTrack/scripts/pretrain/rsl_rl/gait_runtime_guard.py)，并配套 CPU 测试。
本轮按 `isaaclab-api-context` 技能核对本机 API：真实物理步、MotionCommand 的参考推进顺序、
环境原点及运行时四元数顺序；没有重装或升级 SDK。

1. 每片段开始时允许一次显式 frame-zero 初始化，包括参考初态对应的机器人状态。
   不是从任意站姿连续切换起步的验收。片段之间允许 reset，片段内部不允许。
2. 直接执行原 MotionCommand 的时序参考与原策略推理，不用在线 Provider，
   不改写关节轨迹代替策略输出。固定模式调用真实 actor 掩码路径；每步核对 mode。
3. 测试实例关闭 observation noise、reset disturbance、interval events 和自动终止项。
   这些是隔离诊断配置，不改变生产训练配置；不能当作扰动鲁棒性证据。
4. 对 N 帧源数据恰好执行 N−1 个 control step，不跨入片尾自动重采样边界。
   `WalkingStraightForwards07`：304 帧、303 步、6.06 秒、1212 次 PhysX 回调；
   `walking_slow04`：325 帧、324 步、6.48 秒、1296 次回调。两种模式完全一致。
5. 与原环境指标时序一致：每次 post-physics 实际状态配对 **pre-step 参考**，
   因为随后 command 才推进。另保留初始样本，所以指标中参考 frame 0 出现两次，
   最后配对 frame N−2；已推进到 N−1 的终态参考单独保存，不假称又执行了一步。
6. policy 调用不允许推进 PhysX；每次 env step 后检查物理累计时间、参考恰好推进一帧、
   动作有限且为 29 维、mode 未变、没有 done。四元数转为 wxyz 后检查有限性与单位模长。
7. 每组运行窗口内的 11 项 reset/resample/root/joint-state 写入计数全为零。
   守卫在原方法执行前抛错；窗口结束再断言计数为零，防止 SDK 吞掉异常造成假通过。
   正常关节 target actuation 不在禁止范围内。

测试前确定的固定工程门槛：

- 实际前进至少 0.50 m，且不少于参考净前进的 50%。
- 骨盆三维误差均值不超过 0.15 m、最大值不超过 0.30 m。
- 骨盆高度至少 0.35 m，骨盆局部 +Z 与世界 +Z 点积至少 0.5。
- 左右脚各自相对初始位置的最大 XY 偏移至少 0.10 m。
- 左右脚前后顺序至少交换两次，使用 0.03 m 死区过滤接近重合的符号抖动。
- 必须完成 N−1 步；安全提前中止不能 PASS。

门槛只用于这两个短行走片段。双脚都随身体平移会满足偏移量一项，因此必须同时检查
前后交换；即便全部通过，也不排除脚滑、非理想触地或更长时间失稳。
本轮没有新 GUI 截图或视频，也没有接触力验收。

## 输入溯源、失败尝试与清理

两个片段来自既有 `amass_kit_batch_v1_r016_processed/8/` 和
`amass_kit_batch_v1_r003_processed/167/`，均已在 KIT train 与 full-v2 train 索引中。
这里显式隔离索引做推理，不修改训练集/验证集。官方 checkpoint 的预训练数据重叠情况
未知，所以不称为“未见 KIT 的零样本泛化”，也不说明本地已经训练过全部 KIT。

SHA-256：

```text
checkpoint     88d5a79946c03ed25503f48b2af71d16290844ef066ca9b6c8fa8dc3837422e3
test YAML      a5c85719a607c31f68443548612dbf17f181a7b8cbc1884b5946ca83ac2e20a2
straight07 NPZ 01aae332be0eeea0b91fcd87cefe4b3df35ef28e47f0da1b601753f121db204b
slow04 NPZ     e6619054b54c66b63fc77c183912d3d80554398883b77b36d0d3b44f5e6c6618
```

checkpoint 路径是既有本地链接，manifest 保存其解析后的绝对路径。
两份成功报告的 **整个 input_manifest 完全相同**：checkpoint、索引、NPZ、
entrypoints/test harnesses 及 scaletrack/my_rsl_rl/isaaclab/isaaclab_rl/isaaclab_tasks
Python 源码指纹。两次结束都验证输入未变化；没有覆盖仿真二进制、全部资产和驱动，
因此不称为完全封闭可复现环境。

首次 [pelvis-a.json](../logs/gait_control/20260908/pelvis-a.json) 保留为 FAIL：第一片段
指标通过，但切换第二片段时，inference mode 内创建的 observation history 在外部被
reset，触发 PyTorch 禁止原地更新 inference tensor 的错误。修复仅在测试驱动中将
片段间初始化置于 `torch.inference_mode()`；增加覆盖两次切换的回归后整组重跑。
这不是策略跌倒，也不把失败尝试提升成两段通过证据。

所有 App 运行均为独立 systemd user cgroup：内部 165 秒 / 外部 180 秒期限、
KillMode=control-group、15 秒退出宽限、16 GiB 内存上限、无自动重启。
成功的 Pelvis B 与 WholeBody A 主 PID 分别为 217720、220658，均退出 0；
首次失败 PID 215072 退出 1。三个 exec 会话均已回收，最终无 `bfm-*` 残留服务或
`gait_reference_smoke.py` 进程。两次成功运行从退出请求到 journal 结束记账均约 0.24 秒。
未干预旁路 RL100 作业。

完整 CPU 回归 **542 passed / 62 条既有 warnings / 6.79 秒**，另通过 `git diff --check`：

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/sw/isaaclab_ws/env_isaaclab_sim6_newton/bin/python -m pytest \
  ScaleTrack/tests tests/test_gui_launcher.py tests/test_ci_workflow.py \
  tests/test_mask_report_compare.py -q --tb=short --disable-warnings
```

## 下一步：受限在线步态参考，不直接注入原始轨迹

[在线分项诊断](ONLINE_WAYPOINT_DIAGNOSTICS_20260908.md) 的结论仍成立：actor 收到了正确
XYZ/未来输入；仅高度目标通过原 3 cm 容限，但前进与组合目标仍 STALLED。
本次表明同一策略能执行合适的原始行走参考。**参考生成方式及在线约束的适配**值得优先
试验；尚未证明唯一根因，也没有证明缩放后的步态仍能成功。

对原 NPZ pelvis XYZ 做 50 Hz 相邻差分、32 步（0.64 秒）窗口位移的只读比较：

| 原始参考 | 3D 速度中位 / 最大（m/s） | 垂直速度绝对值最大（m/s） | 0.64 秒位移中位 / 最大（m） | 高度峰峰值（cm） |
| --- | --- | --- | --- | --- |
| straight07 | 0.321 / 1.073 | 0.166 | 0.319 / 0.557 | 4.29 |
| slow04 | 0.337 / 0.923 | 0.207 | 0.327 / 0.496 | 5.67 |

当前在线总速度上限 0.08 m/s、垂直上限 0.04 m/s、实际状态前探上限 0.10 m，
0.64 秒总路程预算只有 0.0512 m。原数据约 65–73% 运动增量超总速率、约 44%
超垂直速率；原片段还有独立的起点坐标。**不能直接塞进在线 Provider 或调大上限来报通过。**

建议的后续实现与验收（本轮未实现）：

1. 从已验证片段提取局部骨盆时序；Fresh Apply 时对齐当前 reference 的 XY/yaw/Z，
   保证 sample 0 连续。三维目标、高度与停止状态仍采用真实机器人反馈。
2. 在提交 Provider 前确定性生成受限轨迹：水平趋势与 Z/侧向周期分量分别处理，
   检查每个增量的总速度、垂直速度和转速，以及相对实际位置的未来前探距离。
   禁止事后逐点裁剪后假设原步态相位/导数仍连续。
3. 仅 after-step 推进相位；前探满时暂停参考，物理仍由策略运行。固定模式先做
   Pelvis-1，不能把未激活的足/腕坐标当作已跟踪目标。
4. 单片段缩放后若到不了指定目标，明确报告不可达；循环片段须先验收周期边界的
   位置/速度/姿态连续性，不能静默跳回开头。空间缩小、时间慢放都需重新物理验证。
5. 保留原 0.20 m 前进 / 0.06 m 降高 combined 场景与原到达、停稳、2 秒保持、
   Pause 冻结、无中途写状态和退出清理判据。先 CPU 契约测试，再隔离真实物理 A/B。

## 本机查看这两段动作

在本机桌面终端、仓库根目录运行以下命令；本轮只核对了 launcher dry-run，
没有为此另启动 GUI。它使用既有原生 Kit 浏览器，可在菜单切换两段动作：

```bash
cd /home/sw/bfm
BFM_LOAD_RUN=humanoid_transformer_m BFM_CHECKPOINT=model_22200.pt \
BFM_VALIDATION_INDEX=/home/sw/bfm/logs/gait_control/20260908/kit_walking.yaml \
BFM_TRAIN_INDEX=/home/sw/bfm/logs/gait_control/20260908/kit_walking.yaml \
BFM_INITIAL_MOTION=KIT/8/WalkingStraightForwards07_stageii BFM_MODE_INDEX=0 \
bash scripts/open_motion_browser.sh
```

改为 `BFM_MODE_INDEX=7` 可查看 WholeBody-14。这里**不要加 `--online-targets`**，
否则进入在线参考而非本次原始片段播放。GUI 循环/切换会在片段边界初始化，属于预览，
不是上述无中途重置的固定窗口验收。用 **Exit player** 退出；紧急停止：

```bash
systemctl --user stop bfm-motion-browser.service
```

launcher 仍有原 30 分钟自动期限，不自动重启。共享源码不代表可以重新分发 KIT/AMASS
或 checkpoint；没有本地授权数据和权重时，这些本机路径不可用。

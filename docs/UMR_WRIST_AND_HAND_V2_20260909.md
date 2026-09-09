# UMR 腕部迁移排查与 canonical-hand v2

本页是 v1 学习 A/B 完成后的后续开发记录，不修改
[已冻结学习协议](UMR_BEHAVIOR_AB_PROTOCOL_20260909.md) 或其未通过的质量结论。

## 为何先处理重定向目标

v1 同 origin 的双腕骨盆相对姿态差，17 train 平均 51.17°、10 development 56.49°。
这是参考之间的差，不是机器人执行误差。直接继续加 PPO 轮数，不能回答目标究竟改变了什么。

两套流程的实际约束不同：

| 内容 | 原 ScaleRetarget/GMR | 本批 UMR material-surface adapter |
|---|---|---|
| 腕监督 | 人体 world wrist rotation + 固定局部 offset，完整 FrameTask | 手部材料表面点位置、三角法线；无显式 wrist FrameTask |
| 左/右腕 offset，wxyz | `[1,0,0,0]` / `[0,0,0,-1]`，右乘 | 不能直接假定表面法线等于这些 wrist frame |
| 手指 | 人体 loader 手指置零，腕 pose_body 保留 | pose_hand 进入 moving SMPL-X mesh |
| canonical 手形 | 不依赖 UMR 表面对应 | flat_hand_mean=True，canonical 手指零，moving 手指固定 mean hand |
| 腕是否能动 | 六腕 DOF 每条动作均有变化 | 六腕 DOF 参与求解，posture cost 0.02 向零位弱正则 |

原 GMR 的 cost 10/10 和 offsets 有历史实际加载日志证据，不只是当前 YAML：
`logs/amass_preparation/accad_batch_v1_20260905.log:125`，以及本批相关 KIT r005/r007/r010/r011/r017。
CNRS/BML 的部分完整 IK 字典日志未保留，具有相同 retarget 指纹与历史 Hydra 路径；不扩大其直接证据强度。
旧 Mink 的精确历史版本也没有完整展开记录。

## 确认了什么，尚未确认什么

对 17 train + 10 development 原始源检查，27 条 `pose_hand` 都是全时段恒定的同一 90 维向量，
与加载 SMPL-X 模型的 hand mean 最大差约 2.97e-8 rad。它不是动态捕获的手指行为。
`flat_hand_mean=True` 保证不会在输入上重复加 mean，因此不能将问题描述为“双加 mean”。

真正的待验机制是：canonical 平手与 moving 弯手的表面差异，可能让没有独立手指的 G1 用腕/臂补偿。
27 套对应点的 hand 标签均在同侧 rubber hand/wrist，没有观察到左右交换；这不等于证明全部对应正确。

另确认上游 `configs/g1_29dof_rev_1_0.yaml` 中声明的 `max_velocity=12` 没有接到实际 constructor/limits。
因此 receipt 中有这个配置，不能当作满足速度限制的证明；本次不同时改它，以保持单变量实验。

## 新工具与数值定义

1. `scripts/analyze_umr_wrist_targets_v2.py`：完整 27 origin 的旧目标兼容性诊断，CPU 只读。
2. `scripts/umr_hand_reference_v2.py`：从 SHA 绑定的 v1 prepared source 派生 canonical-hand 一致变体，
   重建原 flat canonical 作为一致性核验，保持材料采样和所有 moving arrays 不变。
3. `scripts/run_umr_hand_pilot_v2.py`：固定四来源各一条、fresh control 和 matched-hand 各一组，
   默认只读计划，有界服务内 `--execute` 才运行，全部写新版本目录。

细节固定在 [v2 机制试验协议](UMR_HAND_REFERENCE_PROTOCOL_V2_20260909.md)。
prepared `joint_rotations` 已经包含源推导的固定 heading，不能再乘一次。
历史 GMR 目标为 `T_p = R_h,p C_p`、`T_w = R_h,w C_w`；pelvis offset
`C_p` 的 wxyz 为 `[0.5,-0.5,-0.5,-0.5]`，不是单位阵。
相对目标必须是 `C_pᵀ R_h,pᵀ R_h,w C_w`，不能遗漏 `C_pᵀ`。
同时报告 world wrist、world pelvis、pelvis-relative wrist，可区分全局骨盆误差的影响。

所有角是最短 SO(3) 角，不依赖 quaternion 正负号。与 GMR 目标更近不自动代表 UMR 更正确，
因为 UMR 优化的并非同一目标。手形对照还需看原 control 重现差、穿地、自碰撞和速度变化。

## 执行记录

### 27 origin 目标诊断：完成

真实报告：`logs/behavior_learning/umr_wrist_targets_v2_20260909a.json`，SHA
`7a084347abee90878a1ada2e20a3fe7c5b180394c817f416685050f0bde246cb`。
train17 共 3,151 packed 帧，dev10 共 1,969 帧；不遗漏几何拒绝的 development。
独立重算全部 origin 等权汇总，并重新核验 10,587 个文件 SHA 和 2,009 个路径别名，一致。

| origin 等权平均角度 ° | 17 train：旧 / UMR | 10 development：旧 / UMR |
|---|---:|---:|
| world 双腕 | 0.364 / 51.432 | 0.610 / 57.656 |
| pelvis-relative 双腕 | 1.290 / 50.875 | 2.050 / 56.111 |
| world pelvis | 1.157 / 6.842 | 1.782 / 8.980 |

说明差异不只是两侧机器人 pelvis 的变化：world wrist 在源 heading 中也明显不同。
但这些仍是**旧映射兼容性**，不是 UMR 自身 point/normal loss，更不是人体真实度的唯一标准。
旧数据也并非每帧完美：独立复核的 CNRS/288/12_L_1_stageii 有约 94.06° 左腕瞬时极值。
用独立 SciPy quaternion composition/magnitude 重算该 250 帧 origin，三类统计最大差 6.22e-7°。

旧 GMR 在接近 30 Hz 的端点保真时钟求解，再重采样到 50 Hz；本诊断用 AMASS→prepared 50 Hz
的人体旋转，重采样顺序不一样。因此不能称为精确重放旧 solver 每个优化帧的损失。
诊断服务正常结束，MainPID=0、空 cgroup；未启动模型推理或 PPO。

### 四来源 canonical-hand 对照

**完成，未晋升数据。** `local/umr_hand_v2_20260909a/status.json`，SHA
`7416751c1b83c20d46f4d375123e0cc311f417b8011fbe3f0b5d6b755d0b3f0b`。
协议 SHA `c2df165b9b6dfa295f585c1bafc9c570b03993d33da8f5bdf359fe1ab624cc2f`。
运行 271.82 秒（不含 controller 前置输入核验），8/8 正常完成、全部 solve_failures=0。
这些是 native inclusive 窗口，各 arm 671 帧，共 1,342 帧，不和上面的 packed half-open 数字混用。

| 固定 train origin 简称 | 帧数 | 原 UMR 双腕目标差 ° | matched-hand ° | 变化 |
|---|---:|---:|---:|---|
| ACCAD A2 Sway | 251 | 50.781 | 31.586 | 平均下降，但左腕 33.345→50.436°，右腕 68.217→12.736° |
| BMLmovi Subject22 F9 | 166 | 51.688 | 16.723 | 双腕下降；新增自碰撞最大 4.274 mm、5 帧超过 1 mm |
| BMLrub normal_jog4 | 82 | 34.800 | 44.372 | 双腕均上升 |
| KIT walking_run06 | 172 | 45.671 | 24.088 | 双腕下降 |
| 四 origin 等权平均 | — | 45.735 | 29.192 | 下降约 36.17%，不是全库/策略质量提升率 |

原版四条均无本模型检测到的自碰撞；新版除 BMLmovi 上述情况外也没有。
两侧全部无脚底超过 1 mm 穿地、无检测到的全身穿地、无关节位置限位违例。
这些只是当前碰撞模型的 FK 几何结果，不代表接触力、抓箱稳定性或实物标定。

速度问题也出现了实际证据：native 相邻输出帧的关节割线速度，原版 28 个 joint-interval、
新版 21 个超过 12 rad/s；两版最大均约 17.748 rad/s。新版 ACCAD 从 0 增为 2 个，
因此不能因总体数量变少而隐藏局部退步。这里按“关节×相邻帧”计数，不是动作数或独立异常事件数。
`max_velocity=12` 配置没有生效的实现缺口仍未修复；目前产物不声称受速度约束。

独立验收（重新读数组，而非仅相信 status）：

- 4/4 case 的全部 17 个 protected/moving/material arrays 逐 dtype、值一致；
  actor_height、后续 scale、ground_offset 全部完全相同，没有采样/缩放混杂。
- 4/4 fresh control qpos 与对应冻结原 UMR 逐元素完全相同，重现差为零。
- 8/8 Stage I 日志显示实际 CUDA，且 correspondence 的 `loss_history` 均为有限 `(2500,5)`，
  epoch 连续为 0..2499；没有把请求 CUDA 或退出码当作唯一执行证据。
- 全部 motion 数值数组、receipt、frame clock 一致；68/68 输出文件集合及 SHA 独立核验一致。
- controller 核验 10,628 个输入文件，前后不变；原 v1 core/数据/检查点未改。
- 有界服务 `bfm-umr-hand-20260909a` 已 inactive/dead、MainPID=0、空 cgroup，峰值内存约 5.9 GB。
  新实验目录约 128 MiB；最终没有运行中的 `bfm-*` unit 或 GPU 计算进程。

没有新增 PPO、没有替换 7,174 条主数据，默认官方模型不变。
实际 Stage I 会训练点云对应网络（每组 2,500 epochs），这与 BFM Transformer/PPO 策略训练不同。
本轮共 8×2,500=20,000 个 Stage I epochs；报告 `training_updates=0` 明确定义为策略 PPO 更新数。
没有把这一步称为已完成 behavior、夹箱搬运、接触标定或完整 BFM 复现。

### 结论与下一步

实验支持“canonical hand reference 会显著改变解”的机制；同 moving targets、同尺度、可精确重现 control
时，改变 canonical 手形确实改变了对应学习和后续机器人姿态。然而平均改善并不一致，也出现自碰撞，
所以**不把这个变体当作通用修复，不将这四条直接升级为训练数据**。

下一版重点是明确可追踪的腕姿态语义与实际输出帧间速度约束：

1. 在独立版本中试验显式 human→G1 wrist orientation 目标与 UMR 表面/碰撞目标的组合，保留位置和穿地审计。
2. 按真实相邻输出帧的 `dt=0.02s` 验证速度边界；不能只在若干内部 IK 迭代中加限速后就宣称帧间限速有效。
3. 先重新冻结新几何/目标对照，再扩到完整同源 pilot、八模式策略跟踪；有证据后才启动下一轮配对 PPO。

这些是下一段开发，不计入本轮已完成项；也不修改 v1 的失败门槛或按结果丢弃 BMLrub 等不利样本。

### 回归测试

三个新文件对应 **73 项新测试**：hand preparation 18、wrist target 40、controller 15。
与两个已有相关模块组合，146 项测试及 347 subtests 通过；不是新增 146 项。
之后完整重跑：现有 Isaac Python 下 **1,061 passed / 911 subtests / 65 warnings**，66.94 秒；
原依赖隔离的 Python 3.11 八模块另 **73 tests OK**，5.75 秒。合计 **1,134 项主测试**，不重复计子集。

- `logs/behavior_learning/umr_hand_v2_tests_20260909a.log`
- `logs/behavior_learning/umr_hand_v2_fulltests_20260909a.log`
- `logs/behavior_learning/umr_hand_v2_py311tests_20260909a.log`

未安装、升级 IsaacLab 或任何依赖；未声称新增测试已在远端 CI 执行。
